//! Float32 inference for the frozen v3.1-A update-500 checkpoint.

use crate::Result;
use std::collections::HashMap;
use tch::{Device, Kind, Reduction, Tensor};

const BATCH: i64 = 8;
const CONTEXT: i64 = 512;
const STEPS: i64 = CONTEXT - 1;
const WIDTH: i64 = 512;
const VOCAB: i64 = 8192;
const HEAD_DIM: i64 = 64;
const QUERY_HEADS: i64 = 8;
const KV_HEADS: i64 = 2;
const EXPERTS: i64 = 8;
const LAYERS: usize = 8;

type Weights = HashMap<String, Tensor>;

pub struct BatchResult {
    pub ordinary: Tensor,
    pub altered: Tensor,
    pub eligible: Tensor,
    pub substitute: Tensor,
}

pub struct Model {
    embedding: Tensor,
    dense_block: Block<SwiGlu>,
    routed_blocks: Vec<Block<Mixture>>,
    final_norm: Tensor,
    head: Tensor,
    positions: Positions,
}

fn take_weight(weights: &mut Weights, prefix: &str, shape: &[i64]) -> Result<Tensor> {
    let key = format!("{prefix}.weight");
    let weight = weights
        .remove(&key)
        .ok_or_else(|| format!("missing checkpoint tensor {key}"))?;
    if weight.size() != shape || weight.kind() != Kind::Float {
        return Err(format!("{key} must be a float32 tensor of shape {shape:?}").into());
    }
    Ok(weight)
}

fn linear(input: &Tensor, weight: &Tensor) -> Tensor {
    input.linear(weight, None::<&Tensor>)
}

fn rms_norm(input: &Tensor, weight: &Tensor) -> Tensor {
    let mean_square = input
        .pow_tensor_scalar(2.)
        .mean_dim(&[-1i64][..], true, Kind::Float);
    input * (mean_square + 1e-6).rsqrt() * weight
}

struct Positions {
    cosine: Tensor,
    sine: Tensor,
    full_mask: Tensor,
    sliding_mask: Tensor,
}

impl Positions {
    fn new(device: Device) -> Self {
        let cpu_float = (Kind::Float, Device::Cpu);
        let exponent = Tensor::arange_start_step(0, HEAD_DIM, 2, cpu_float) / HEAD_DIM as f64;
        let inverse_frequency = Tensor::pow_scalar(10000., &exponent).reciprocal();
        let angle = Tensor::arange(CONTEXT, cpu_float).outer(&inverse_frequency);
        let positions = Tensor::arange(STEPS, (Kind::Int64, Device::Cpu));
        let distance = positions.unsqueeze(1) - positions.unsqueeze(0);
        let causal = distance.ge(0);
        let sliding = causal.logical_and(&distance.lt(256));
        Self {
            cosine: angle.cos().narrow(0, 0, STEPS).to_device(device),
            sine: angle.sin().narrow(0, 0, STEPS).to_device(device),
            full_mask: causal.unsqueeze(0).unsqueeze(0).to_device(device),
            sliding_mask: sliding.unsqueeze(0).unsqueeze(0).to_device(device),
        }
    }

    fn rotate(&self, input: &Tensor) -> Tensor {
        let halves = input.chunk(2, -1);
        Tensor::cat(
            &[
                &halves[0] * &self.cosine - &halves[1] * &self.sine,
                &halves[0] * &self.sine + &halves[1] * &self.cosine,
            ],
            -1,
        )
    }
}

struct SwiGlu {
    gate: Tensor,
    value: Tensor,
    output: Tensor,
}

impl SwiGlu {
    fn load(weights: &mut Weights, prefix: &str, hidden: i64) -> Result<Self> {
        Ok(Self {
            gate: take_weight(weights, &format!("{prefix}.w1"), &[hidden, WIDTH])?,
            value: take_weight(weights, &format!("{prefix}.w2"), &[hidden, WIDTH])?,
            output: take_weight(weights, &format!("{prefix}.w3"), &[WIDTH, hidden])?,
        })
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden = linear(input, &self.gate).silu() * linear(input, &self.value);
        linear(&hidden, &self.output)
    }
}

struct Block<M> {
    attention_norm: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    full_attention: bool,
    mlp_norm: Tensor,
    mlp: M,
}

impl<M> Block<M> {
    fn load(weights: &mut Weights, index: usize, mlp: M) -> Result<Self> {
        let prefix = format!("blocks.{index}");
        let kv_width = KV_HEADS * HEAD_DIM;
        Ok(Self {
            attention_norm: take_weight(weights, &format!("{prefix}.ln1"), &[WIDTH])?,
            query: take_weight(weights, &format!("{prefix}.attn.q_proj"), &[WIDTH, WIDTH])?,
            key: take_weight(
                weights,
                &format!("{prefix}.attn.k_proj"),
                &[kv_width, WIDTH],
            )?,
            value: take_weight(
                weights,
                &format!("{prefix}.attn.v_proj"),
                &[kv_width, WIDTH],
            )?,
            output: take_weight(weights, &format!("{prefix}.attn.o_proj"), &[WIDTH, WIDTH])?,
            full_attention: (index + 1).is_multiple_of(4),
            mlp_norm: take_weight(weights, &format!("{prefix}.ln2"), &[WIDTH])?,
            mlp,
        })
    }

    fn prepare_mlp(&self, input: &Tensor, positions: &Positions) -> (Tensor, Tensor) {
        let normalized = rms_norm(input, &self.attention_norm);
        let project = |weight, heads| {
            linear(&normalized, weight)
                .view([BATCH, STEPS, heads, HEAD_DIM])
                .transpose(1, 2)
        };
        let query = positions.rotate(&project(&self.query, QUERY_HEADS));
        let key = positions
            .rotate(&project(&self.key, KV_HEADS))
            .repeat_interleave_self_int(QUERY_HEADS / KV_HEADS, 1, None);
        let value = project(&self.value, KV_HEADS).repeat_interleave_self_int(
            QUERY_HEADS / KV_HEADS,
            1,
            None,
        );
        let mask = if self.full_attention {
            &positions.full_mask
        } else {
            &positions.sliding_mask
        };
        let attended = Tensor::scaled_dot_product_attention(
            &query,
            &key,
            &value,
            Some(mask),
            0.,
            false,
            None,
            false,
        );
        let attended = attended.transpose(1, 2).reshape([BATCH, STEPS, WIDTH]);
        let residual = input + linear(&attended, &self.output);
        let mlp_input = rms_norm(&residual, &self.mlp_norm);
        (residual, mlp_input)
    }
}

struct Mixture {
    gate: Tensor,
    experts: Vec<SwiGlu>,
}

fn mixture_weights(scores: &Tensor, ids: &Tensor) -> Tensor {
    let assignments = ids.one_hot(EXPERTS).to_kind(Kind::Float);
    (assignments * scores.softmax(-1, Kind::Float).unsqueeze(-1)).sum_dim_intlist(
        &[1i64][..],
        false,
        Kind::Float,
    )
}

struct Intervention {
    ordinary_weights: Tensor,
    altered_weights: Tensor,
    eligible: Tensor,
    substitute: Tensor,
}

fn replace_expert3(logits: &Tensor, punctuation: &Tensor) -> Result<Intervention> {
    let (scores, ids) = logits.topk(2, -1, true, true);
    let ordinary_weights = mixture_weights(&scores, &ids);
    let eligible = punctuation
        .reshape([-1])
        .logical_and(&ids.select(1, 0).eq(3));
    let unused_scores = logits.scatter_value(1, &ids, f64::NEG_INFINITY);
    let substitute = unused_scores.argmax(1, false);
    let selected = Tensor::stack(&[substitute.shallow_clone(), ids.select(1, 1)], 1);
    let prescribed_logits = logits
        .full_like(f64::NEG_INFINITY)
        .scatter(1, &selected, &scores);
    let masked_logits = prescribed_logits.where_self(&eligible.unsqueeze(1), logits);
    let (new_scores, new_ids) = masked_logits.topk(2, -1, true, true);
    let altered_weights = mixture_weights(&new_scores, &new_ids);

    // Equal scores can reorder the two ranks. Check the effective weight on each expert.
    let prescribed_weights =
        logits
            .zeros_like()
            .scatter(1, &selected, &scores.softmax(-1, Kind::Float));
    if !altered_weights
        .equal(&prescribed_weights.where_self(&eligible.unsqueeze(1), &ordinary_weights))
    {
        return Err("expert 3 substitution changed the prescribed mixture weights".into());
    }
    Ok(Intervention {
        ordinary_weights,
        altered_weights,
        substitute: substitute.masked_select(&eligible),
        eligible: eligible.reshape_as(punctuation),
    })
}

impl Mixture {
    fn load(weights: &mut Weights, prefix: &str) -> Result<Self> {
        let gate = take_weight(weights, &format!("{prefix}.gate"), &[EXPERTS, WIDTH])?;
        let experts = (0..EXPERTS)
            .map(|i| SwiGlu::load(weights, &format!("{prefix}.experts.{i}"), 960))
            .collect::<Result<_>>()?;
        Ok(Self { gate, experts })
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let flat = input.reshape([-1, WIDTH]);
        let (scores, ids) = linear(&flat, &self.gate).topk(2, -1, true, true);
        let weights = mixture_weights(&scores, &ids);
        let mut output = flat.zeros_like();
        for (i, expert) in self.experts.iter().enumerate() {
            output += weights.narrow(1, i as i64, 1) * expert.forward(&flat);
        }
        output.view_as(input)
    }

    fn intervene(
        &self,
        input: &Tensor,
        punctuation: &Tensor,
    ) -> Result<(Tensor, Tensor, Intervention)> {
        let flat = input.reshape([-1, WIDTH]);
        let route = replace_expert3(&linear(&flat, &self.gate), punctuation)?;
        let mut ordinary = flat.zeros_like();
        let mut altered = flat.zeros_like();
        // The intervention changes only the two routing weights at this layer.
        // Evaluate each expert once and preserve the reference summation order in both arms.
        for (i, expert) in self.experts.iter().enumerate() {
            let value = expert.forward(&flat);
            ordinary += route.ordinary_weights.narrow(1, i as i64, 1) * &value;
            altered += route.altered_weights.narrow(1, i as i64, 1) * &value;
        }
        Ok((ordinary.view_as(input), altered.view_as(input), route))
    }
}

impl Model {
    pub fn new(mut weights: Weights, device: Device) -> Result<Self> {
        for tensor in weights.values_mut() {
            *tensor = tensor.f_to_device(device)?;
        }
        let dense = SwiGlu::load(&mut weights, "blocks.0.mlp", WIDTH * 8 / 3)?;
        let dense_block = Block::load(&mut weights, 0, dense)?;
        let mut routed_blocks = Vec::with_capacity(LAYERS - 1);
        for i in 1..LAYERS {
            let mlp = Mixture::load(&mut weights, &format!("blocks.{i}.mlp"))?;
            routed_blocks.push(Block::load(&mut weights, i, mlp)?);
        }
        let model = Self {
            embedding: take_weight(&mut weights, "tok_emb", &[VOCAB, WIDTH])?,
            dense_block,
            routed_blocks,
            final_norm: take_weight(&mut weights, "ln_f", &[WIDTH])?,
            head: take_weight(&mut weights, "head", &[VOCAB, WIDTH])?,
            positions: Positions::new(device),
        };
        if !weights.is_empty() {
            return Err(format!("{} unexpected checkpoint tensors", weights.len()).into());
        }
        Ok(model)
    }

    fn continuation_loss(&self, mut hidden: Tensor, targets: &Tensor) -> Tensor {
        for block in &self.routed_blocks[1..] {
            let (residual, input) = block.prepare_mlp(&hidden, &self.positions);
            hidden = residual + block.mlp.forward(&input);
        }
        let logits = linear(&rms_norm(&hidden, &self.final_norm), &self.head);
        logits
            .flatten(0, 1)
            .cross_entropy_loss(targets, None::<&Tensor>, Reduction::None, -100, 0.)
            .view([BATCH, STEPS])
    }

    pub fn evaluate_batch(&self, tokens: &Tensor, punctuation: &Tensor) -> Result<BatchResult> {
        let _no_grad = tch::no_grad_guard();
        if tokens.size() != [BATCH, CONTEXT]
            || tokens.kind() != Kind::Int64
            || tokens.device() != Device::Cpu
            || punctuation.size() != [BATCH, STEPS]
            || punctuation.kind() != Kind::Bool
            || punctuation.device() != Device::Cpu
        {
            return Err(
                "expected CPU int64 tokens [8,512] and bool punctuation mask [8,511]".into(),
            );
        }
        if tokens.min().int64_value(&[]) < 0 || tokens.max().int64_value(&[]) >= VOCAB {
            return Err("token ID outside the frozen model vocabulary".into());
        }
        let device = self.embedding.device();
        let tokens = tokens.to_device(device);
        let punctuation = punctuation.to_device(device);
        let inputs = tokens.narrow(1, 0, STEPS);
        let targets = tokens.narrow(1, 1, STEPS).reshape([-1]);
        let embedding = Tensor::embedding(&self.embedding, &inputs, -1, false, false);
        let (residual, input) = self.dense_block.prepare_mlp(&embedding, &self.positions);
        let hidden = residual + self.dense_block.mlp.forward(&input);
        let first_moe = &self.routed_blocks[0];
        let (residual, input) = first_moe.prepare_mlp(&hidden, &self.positions);
        let (ordinary, altered, route) = first_moe.mlp.intervene(&input, &punctuation)?;
        Ok(BatchResult {
            ordinary: self
                .continuation_loss(&residual + ordinary, &targets)
                .to_device(Device::Cpu),
            altered: self
                .continuation_loss(&residual + altered, &targets)
                .to_device(Device::Cpu),
            eligible: route.eligible.to_device(Device::Cpu),
            substitute: route.substitute.to_device(Device::Cpu),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_moves_the_original_top_weight_and_preserves_other_positions() {
        let logits = Tensor::from_slice(&[
            0f32, 1., 3., 5., 3., 0., 4., 0., 0., 1., 3., 5., 3., 0., 4., 0., 0., 7., 3., 5., 3.,
            0., 4., 0.,
        ])
        .view([3, 8]);
        let mask = Tensor::from_slice(&[true, false, true]).view([1, 3]);
        let result = replace_expert3(&logits, &mask).unwrap();
        assert_eq!(Vec::<i64>::try_from(&result.substitute).unwrap(), [2]);
        assert!(
            result
                .eligible
                .equal(&Tensor::from_slice(&[true, false, false]).view([1, 3]))
        );
        let expected = result.ordinary_weights.copy();
        expected.get(0).get(2).copy_(&expected.get(0).get(3));
        let _ = expected.get(0).get(3).fill_(0.);
        assert!(expected.equal(&result.altered_weights));
    }
}
