//! Single-block causal Transformer and a checkpointable, deterministic AdamW.
use crate::Result;
use std::{collections::BTreeMap, f64::consts::PI, path::Path};
use tch::{Device, Kind, Tensor, nn, nn::Module};

pub const VOCAB: i64 = 8192;
pub const WIDTH: i64 = 64;
pub const HEADS: i64 = 4;
pub const CONTEXT: i64 = 512;
pub const UPDATES: usize = 500;

pub fn learning_rate(step: usize) -> f64 {
    if step < 15 {
        0.0003 * (step + 1) as f64 / 15.
    } else {
        0.0003 * (0.1 + 0.9 * (1. + (PI * (step - 15) as f64 / 485.).cos()) / 2.)
    }
}

pub struct Dense {
    vs: nn::VarStore,
    tokens: nn::Embedding,
    positions: nn::Embedding,
    norm_att: nn::LayerNorm,
    qkv: nn::Linear,
    proj: nn::Linear,
    norm_ff: nn::LayerNorm,
    ff: nn::Linear,
    out: nn::Linear,
    norm_final: nn::LayerNorm,
    lm: nn::Linear,
    heads: i64,
    width: i64,
}

impl Dense {
    pub fn new(device: Device, vocab: i64, width: i64, heads: i64, context: i64) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();
        let init = nn::Init::Randn {
            mean: 0.,
            stdev: 0.02,
        };
        let embedding = nn::EmbeddingConfig {
            ws_init: init,
            ..Default::default()
        };
        let linear = nn::LinearConfig {
            ws_init: init,
            bias: false,
            bs_init: None,
        };
        let norm = |name| nn::layer_norm(&root / name, vec![width], Default::default());
        Self {
            tokens: nn::embedding(&root / "tokens", vocab, width, embedding),
            positions: nn::embedding(&root / "positions", context, width, embedding),
            norm_att: norm("norm_att"),
            qkv: nn::linear(&root / "qkv", width, 3 * width, linear),
            proj: nn::linear(&root / "proj", width, width, linear),
            norm_ff: norm("norm_ff"),
            ff: nn::linear(&root / "ff", width, 4 * width, linear),
            out: nn::linear(&root / "out", 4 * width, width, linear),
            norm_final: norm("norm_final"),
            lm: nn::linear(&root / "lm", width, vocab, linear),
            vs,
            heads,
            width,
        }
    }
    pub fn baseline(device: Device) -> Self {
        Self::new(device, VOCAB, WIDTH, HEADS, CONTEXT)
    }
    pub fn logits(&self, x: &Tensor) -> Tensor {
        let (b, t) = x.size2().expect("batch of token sequences");
        let pos = Tensor::arange(t, (Kind::Int64, x.device()));
        let h = self.tokens.forward(x) + self.positions.forward(&pos);
        let qkv = h.apply(&self.norm_att).apply(&self.qkv).chunk(3, -1);
        let heads = |z: &Tensor| {
            z.reshape([b, t, self.heads, self.width / self.heads])
                .transpose(1, 2)
        };
        let attended = Tensor::scaled_dot_product_attention(
            &heads(&qkv[0]),
            &heads(&qkv[1]),
            &heads(&qkv[2]),
            None::<&Tensor>,
            0.,
            true,
            None,
            false,
        )
        .transpose(1, 2)
        .reshape([b, t, self.width]);
        let h = h + attended.apply(&self.proj);
        let f = h
            .apply(&self.norm_ff)
            .apply(&self.ff)
            .gelu("none")
            .apply(&self.out);
        (h + f).apply(&self.norm_final).apply(&self.lm)
    }
    pub fn count(&self) -> usize {
        self.vs.variables().values().map(Tensor::numel).sum()
    }
    pub fn device(&self) -> Device {
        self.vs.device()
    }
}

// tch's optimizer wrapper omits serialization of the Adam moments. Keep them
// with their parameter names and restore them alongside the completed step.
struct Slot {
    name: String,
    param: Tensor,
    first: Tensor,
    second: Tensor,
}

pub struct Adam {
    slots: Vec<Slot>,
    pub step: usize,
}

impl Adam {
    pub fn new(model: &Dense) -> Self {
        let vars: BTreeMap<_, _> = model.vs.variables().into_iter().collect();
        let slots = vars
            .into_iter()
            .map(|(name, param)| Slot {
                first: param.zeros_like(),
                second: param.zeros_like(),
                name,
                param,
            })
            .collect();
        Self { slots, step: 0 }
    }
    pub fn update(&mut self) -> Result<f64> {
        let norms: Vec<_> = self.slots.iter().map(|s| s.param.grad().norm()).collect();
        let norm = Tensor::stack(&norms, 0).norm().double_value(&[]);
        if !norm.is_finite() {
            return Err("nonfinite gradient norm".into());
        }
        let clip = (1. / (norm + 1e-6)).min(1.);
        let lr = learning_rate(self.step);
        let t = (self.step + 1) as i32;
        tch::no_grad(|| {
            for slot in &mut self.slots {
                let grad = slot.param.grad() * clip;
                slot.first = &slot.first * 0.9 + &grad * 0.1;
                slot.second = &slot.second * 0.999 + grad.square() * 0.001;
                let denom = (&slot.second / (1. - 0.999f64.powi(t))).sqrt() + 1e-8;
                let decay = if slot.param.dim() >= 2 { 0.01 } else { 0. };
                let next = &slot.param * (1. - lr * decay)
                    - (&slot.first / (1. - 0.9f64.powi(t))) / denom * lr;
                slot.param.copy_(&next);
                slot.param.zero_grad();
            }
        });
        self.step += 1;
        Ok(norm)
    }
    pub fn save(&self, path: &Path, state: &[u8]) -> Result<()> {
        let mut fields = vec![
            ("metadata".to_owned(), Tensor::from_slice(state)),
            ("step".to_owned(), Tensor::from(self.step as i64)),
        ];
        for s in &self.slots {
            for (prefix, tensor) in [("weight", &s.param), ("m", &s.first), ("v", &s.second)] {
                fields.push((
                    format!("{prefix}.{}", s.name),
                    tensor.detach().to_device(Device::Cpu),
                ));
            }
        }
        Tensor::save_multi(&fields, path)?;
        Ok(())
    }
    pub fn load(&mut self, path: &Path) -> Result<Vec<u8>> {
        let mut fields: BTreeMap<_, _> = Tensor::load_multi_with_device(path, Device::Cpu)?
            .into_iter()
            .collect();
        let metadata = Vec::<u8>::try_from(fields.remove("metadata").ok_or("missing metadata")?)?;
        let step: usize = fields
            .remove("step")
            .ok_or("missing optimizer step")?
            .int64_value(&[])
            .try_into()?;
        if step > UPDATES {
            return Err("invalid optimizer step".into());
        }
        let _guard = tch::no_grad_guard();
        for slot in &mut self.slots {
            for (prefix, target) in [
                ("weight", &mut slot.param),
                ("m", &mut slot.first),
                ("v", &mut slot.second),
            ] {
                let key = format!("{prefix}.{}", slot.name);
                let value = fields
                    .remove(&key)
                    .ok_or_else(|| format!("missing {key}"))?;
                if value.kind() != target.kind() || value.size() != target.size() {
                    return Err(format!("checkpoint tensor mismatch: {key}").into());
                }
                target.copy_(&value);
            }
        }
        if !fields.is_empty() {
            return Err("unexpected checkpoint tensors".into());
        }
        self.step = step;
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_prefix_and_resumed_adam_step_match_uninterrupted_cpu() -> Result<()> {
        tch::set_num_threads(1);
        tch::manual_seed(0);
        let original = Dense::new(Device::Cpu, 8, 8, 2, 4);
        let x = Tensor::from_slice(&[1i64, 2, 3, 4]).reshape([1, 4]);
        let other = Tensor::from_slice(&[1i64, 2, 5, 6]).reshape([1, 4]);
        assert!(
            original
                .logits(&x)
                .narrow(1, 0, 2)
                .equal(&original.logits(&other).narrow(1, 0, 2))
        );
        let train = |model: &Dense, opt: &mut Adam| -> Result<()> {
            model
                .logits(&x)
                .flatten(0, 1)
                .cross_entropy_for_logits(&other.flatten(0, 1))
                .backward();
            opt.update()?;
            Ok(())
        };
        let mut opt = Adam::new(&original);
        train(&original, &mut opt)?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("state.ot");
        opt.save(&path, b"test")?;
        let resumed = Dense::new(Device::Cpu, 8, 8, 2, 4);
        let mut restored = Adam::new(&resumed);
        assert_eq!(restored.load(&path)?, b"test");
        assert_eq!(restored.step, 1);
        train(&original, &mut opt)?;
        train(&resumed, &mut restored)?;
        assert_eq!(restored.step, opt.step);
        for (a, b) in opt.slots.iter().zip(&restored.slots) {
            assert!(
                a.param.equal(&b.param) && a.first.equal(&b.first) && a.second.equal(&b.second)
            );
        }
        Ok(())
    }
}
