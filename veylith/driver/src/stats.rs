//! Paired, token-weighted expert 3 outcomes on the frozen validation samples.

use crate::Result;
use serde_json::{Map, Value, json};
use tch::{Device, Kind, Tensor};

const OUTCOMES: [&str; 5] = [
    "all",
    "sentence_end",
    "replaced",
    "first_replaced_in_sequence",
    "unaffected_prefix",
];
const SOURCES: usize = 3;
const PER_SOURCE: usize = 80;
const DRAWS: usize = 1_000;

#[derive(Default)]
struct Sequence {
    counts: [u64; OUTCOMES.len()],
    baseline: [f64; OUTCOMES.len()],
    delta: [f64; OUTCOMES.len()],
}

pub struct Stats {
    rows: Vec<Sequence>,
    substitutes: [u64; 8],
}

impl Stats {
    pub fn new() -> Self {
        Self {
            rows: Vec::with_capacity(SOURCES * PER_SOURCE),
            substitutes: [0; 8],
        }
    }

    pub fn add_batch(
        &mut self,
        ordinary: &Tensor,
        altered: &Tensor,
        punctuation: &Tensor,
        eligible: &Tensor,
        substitute: &Tensor,
    ) -> Result<()> {
        for (tensor, kind) in [
            (ordinary, Kind::Float),
            (altered, Kind::Float),
            (punctuation, Kind::Bool),
            (eligible, Kind::Bool),
        ] {
            if tensor.size() != [8, 511] || tensor.kind() != kind || tensor.device() != Device::Cpu
            {
                return Err("expected an 8 × 511 CPU batch of frozen-study targets".into());
            }
        }
        if substitute.dim() != 1
            || substitute.kind() != Kind::Int64
            || substitute.device() != Device::Cpu
        {
            return Err("expected one-dimensional substitute expert IDs".into());
        }
        if ordinary.isfinite().all().int64_value(&[]) == 0
            || altered.isfinite().all().int64_value(&[]) == 0
        {
            return Err("nonfinite next-token loss".into());
        }
        if eligible
            .logical_and(&punctuation.logical_not())
            .any()
            .int64_value(&[])
            != 0
        {
            return Err("expert 3 was replaced outside the punctuation group".into());
        }
        let seen = eligible.cumsum(1, Kind::Int64);
        let prefix = seen.eq(0);
        if !ordinary
            .masked_select(&prefix)
            .equal(&altered.masked_select(&prefix))
        {
            return Err("next-token loss changed before the first replacement".into());
        }
        if eligible.sum(Kind::Int64).int64_value(&[]) != substitute.numel() as i64 {
            return Err("substitute count differs from the eligible positions".into());
        }
        let mut substitutes = [0; 8];
        for expert in Vec::<i64>::try_from(substitute)? {
            if !(0..8).contains(&expert) || expert == 3 {
                return Err(format!("invalid substitute expert {expert}").into());
            }
            substitutes[expert as usize] += 1;
        }

        let mut batch: [Sequence; 8] = Default::default();
        let delta = altered.to_kind(Kind::Double) - ordinary.to_kind(Kind::Double);
        let masks = [
            Tensor::ones_like(eligible),
            punctuation.shallow_clone(),
            eligible.shallow_clone(),
            eligible.logical_and(&seen.eq(1)),
            prefix,
        ];
        for (i, mask) in masks.iter().enumerate() {
            // Match the reference's row-wise LibTorch reduction order before
            // accumulating across sequences and resampling entire rows.
            let dims = &[1_i64][..];
            let counts = Vec::<i64>::try_from(mask.sum_dim_intlist(dims, false, Kind::Int64))?;
            let baseline =
                Vec::<f64>::try_from((ordinary * mask).sum_dim_intlist(dims, false, Kind::Double))?;
            let deltas =
                Vec::<f64>::try_from((&delta * mask).sum_dim_intlist(dims, false, Kind::Double))?;
            for (j, row) in batch.iter_mut().enumerate() {
                row.counts[i] = counts[j] as u64;
                row.baseline[i] = baseline[j];
                row.delta[i] = deltas[j];
            }
        }
        self.rows.extend(batch);
        for (total, n) in self.substitutes.iter_mut().zip(substitutes) {
            *total += n;
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<Value> {
        if self.rows.len() != SOURCES * PER_SOURCE {
            return Err(format!("expected 240 sequences, received {}", self.rows.len()).into());
        }
        let mut rng = BootstrapRng::new();
        let mut estimates: [Vec<f64>; OUTCOMES.len()] =
            std::array::from_fn(|_| Vec::with_capacity(DRAWS));
        for _ in 0..DRAWS {
            let mut counts = [0u64; OUTCOMES.len()];
            let mut sums = [0.0; OUTCOMES.len()];
            for index in rng.draw() {
                let row = &self.rows[index];
                for i in 0..OUTCOMES.len() {
                    counts[i] += row.counts[i];
                    sums[i] += row.delta[i];
                }
            }
            for i in 0..OUTCOMES.len() {
                if counts[i] != 0 {
                    estimates[i].push(sums[i] / counts[i] as f64);
                }
            }
        }

        let mut outcomes = Map::new();
        for (i, name) in OUTCOMES.iter().enumerate() {
            let targets: u64 = self.rows.iter().map(|r| r.counts[i]).sum();
            let sequences = self.rows.iter().filter(|r| r.counts[i] != 0).count();
            let delta: f64 = self.rows.iter().map(|r| r.delta[i]).sum();
            let baseline: f64 = self.rows.iter().map(|r| r.baseline[i]).sum();
            let finite = &mut estimates[i];
            finite.sort_unstable_by(f64::total_cmp);
            let interval =
                (!finite.is_empty()).then(|| [quantile(finite, 0.025), quantile(finite, 0.975)]);
            outcomes.insert(
                (*name).to_owned(),
                json!({
                    "targets": targets,
                    "sequences": sequences,
                    "baseline_ce": (targets != 0).then(|| baseline / targets as f64),
                    "delta_ce": (targets != 0).then(|| delta / targets as f64),
                    "sequence_bootstrap_ci95": interval,
                    "bootstrap_nonempty": finite.len(),
                }),
            );
        }
        Ok(json!({
            "sequences": self.rows.len(),
            "substitute_counts": self.substitutes,
            "unaffected_prefix_ce_exact": true,
            "outcomes": outcomes,
        }))
    }
}

pub(crate) fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let at = probability * (sorted.len() - 1) as f64;
    let lower = sorted[at.floor() as usize];
    let upper = sorted[at.ceil() as usize];
    let fraction = at.fract();
    // NumPy's linear quantile interpolates from the nearer endpoint.
    if fraction >= 0.5 {
        upper - (upper - lower) * (1.0 - fraction)
    } else {
        lower + (upper - lower) * fraction
    }
}

pub(crate) struct BootstrapRng {
    state: u128,
    upper_word: Option<u32>,
}

impl BootstrapRng {
    pub(crate) fn new() -> Self {
        // Initial state of numpy.random.default_rng(20261118).bit_generator.
        // The frozen experiment uses a single seed across all 1,000 draws.
        Self {
            state: 0xff45d3e2f17d4793cea989619372f68a,
            upper_word: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(word) = self.upper_word.take() {
            return word;
        }
        // PCG64 XSL-RR 128/64 advances before output; NumPy consumes the
        // lower 32-bit half first for a bounded choice on an 80-element row.
        self.state = self
            .state
            .wrapping_mul(0x2360ed051fc65da44385df649fccf645)
            .wrapping_add(0xd6cb3f61133f1c4b81c8f8df51cda0eb);
        let mixed = ((self.state >> 64) ^ self.state) as u64;
        let value = mixed.rotate_right((self.state >> 122) as u32);
        self.upper_word = Some((value >> 32) as u32);
        value as u32
    }

    pub(crate) fn draw(&mut self) -> [usize; SOURCES * PER_SOURCE] {
        std::array::from_fn(|i| {
            // Lemire bounded sampling; reject the incomplete low interval.
            let width = PER_SOURCE as u32;
            let threshold = width.wrapping_neg() % width;
            loop {
                let product = u64::from(self.next_u32()) * u64::from(width);
                if product as u32 >= threshold {
                    break i / PER_SOURCE * PER_SOURCE + (product >> 32) as usize;
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn all_stratified_draws_match_the_frozen_numpy_generator() {
        // Hash of 1,000 repetitions of concatenate([rng.choice(rows, 80)
        // for rows in arange(240).reshape(3, 80)]).astype(uint8).tobytes().
        let mut rng = BootstrapRng::new();
        let mut hash = Sha256::new();
        for _ in 0..DRAWS {
            hash.update(rng.draw().map(|row| row as u8));
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            "e87210ac5b383577150f6bea59fbbbd27f919cdfd8f86a08ed849b56adf59d73"
        );
    }

    #[test]
    fn token_weighted_quantiles_match_numpy_and_empty_groups_have_no_interval() {
        let mut stats = Stats::new();
        for i in 0..240 {
            let mut row = Sequence::default();
            row.counts[0] = 2 + i % 7;
            row.delta[0] = (i as f64 % 11.0 - 5.0) / 16.0;
            stats.rows.push(row);
        }
        let result = stats.finish().unwrap();
        let actual = &result["outcomes"]["all"];
        assert_eq!(actual["targets"], 1195);
        assert_eq!(actual["delta_ce"], -0.0004707112970711297);
        assert_eq!(
            actual["sequence_bootstrap_ci95"],
            json!([-0.005391943209484492, 0.004509502391645039])
        );
        assert!(result["outcomes"]["replaced"]["sequence_bootstrap_ci95"].is_null());
        assert_eq!(result["outcomes"]["replaced"]["bootstrap_nonempty"], 0);
    }
}
