//! Outcome-blind matching and sequence-paired estimates for the VEYLITH-22 intervention.

use crate::Result;
use crate::stats::{BootstrapRng, quantile};
use serde_json::{Value, json};
use tch::{Device, Kind, Tensor};

const SOURCES: [&str; 3] = ["cosmopedia-v2", "fineweb-edu", "wikipedia"];
const PER_SOURCE: usize = 80;
const BATCH: usize = 8;
const STEPS: usize = 511;
const POSITION_CALIPER: usize = 32;
const CONFIDENCE_CALIPER: f64 = 0.05;
const MIN_TOTAL: usize = 120;
const MIN_PER_SOURCE: usize = 30;
const DRAWS: usize = 1_000;

#[derive(Clone, Copy)]
struct Point {
    pos: usize,
    confidence: f64,
}

#[derive(Clone, Copy)]
struct Pair {
    punctuation: Point,
    word: Point,
}

impl Pair {
    fn position_gap(self) -> usize {
        self.punctuation.pos.abs_diff(self.word.pos)
    }

    fn confidence_gap(self) -> f64 {
        (self.punctuation.confidence - self.word.confidence).abs()
    }

    fn distance(self) -> f64 {
        self.position_gap() as f64 / POSITION_CALIPER as f64
            + self.confidence_gap() / CONFIDENCE_CALIPER
    }

    fn order(self, other: Self) -> std::cmp::Ordering {
        self.distance()
            .total_cmp(&other.distance())
            .then_with(|| self.position_gap().cmp(&other.position_gap()))
            .then_with(|| self.punctuation.pos.cmp(&other.punctuation.pos))
            .then_with(|| self.word.pos.cmp(&other.word.pos))
    }
}

#[derive(Default, Clone, Copy)]
struct Pool {
    count: usize,
    position_sum: usize,
    confidence_sum: f64,
}

impl Pool {
    fn add(&mut self, point: Point) {
        self.count += 1;
        self.position_sum += point.pos;
        self.confidence_sum += point.confidence;
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.position_sum += other.position_sum;
        self.confidence_sum += other.confidence_sum;
    }

    fn means(self) -> Value {
        json!({
            "mean_position": mean(self.position_sum as f64, self.count),
            "mean_confidence": mean(self.confidence_sum, self.count),
        })
    }
}

fn mean(sum: f64, count: usize) -> Option<f64> {
    (count > 0).then(|| sum / count as f64)
}

#[derive(Default, Clone, Copy)]
struct Row {
    punctuation: Pool,
    word: Pool,
    within_calipers: usize,
    pair: Option<Pair>,
}

fn select_row(punctuation: &[Point], words: &[Point]) -> Row {
    let mut row = Row::default();
    for &p in punctuation {
        row.punctuation.add(p);
        for &w in words {
            let candidate = Pair {
                punctuation: p,
                word: w,
            };
            if candidate.position_gap() <= POSITION_CALIPER
                && candidate.confidence_gap() <= CONFIDENCE_CALIPER
            {
                row.within_calipers += 1;
                if row.pair.is_none_or(|old| candidate.order(old).is_lt()) {
                    row.pair = Some(candidate);
                }
            }
        }
    }
    for &w in words {
        row.word.add(w);
    }
    row
}

pub struct Selection {
    pub punctuation_mask: Tensor,
    pub word_mask: Tensor,
    rows: [Row; BATCH],
}

fn check_tensor(tensor: &Tensor, kind: Kind) -> Result<()> {
    if tensor.device() != Device::Cpu
        || tensor.kind() != kind
        || tensor.size() != [BATCH as i64, STEPS as i64]
    {
        return Err(
            format!("matched study requires a CPU {kind:?} tensor of shape [8, 511]").into(),
        );
    }
    Ok(())
}

pub fn select_pairs(
    punctuation: &Tensor,
    word: &Tensor,
    top1_ids: &Tensor,
    confidence: &Tensor,
) -> Result<Selection> {
    check_tensor(punctuation, Kind::Bool)?;
    check_tensor(word, Kind::Bool)?;
    check_tensor(top1_ids, Kind::Int64)?;
    check_tensor(confidence, Kind::Float)?;
    let punctuation = Vec::<bool>::try_from(punctuation.flatten(0, -1))?;
    let word = Vec::<bool>::try_from(word.flatten(0, -1))?;
    let ids = Vec::<i64>::try_from(top1_ids.flatten(0, -1))?;
    let weights = Vec::<f32>::try_from(confidence.flatten(0, -1))?;
    let mut punctuation_mask = vec![false; BATCH * STEPS];
    let mut word_mask = vec![false; BATCH * STEPS];
    let mut rows = [Row::default(); BATCH];
    for (seq, row) in rows.iter_mut().enumerate() {
        let offset = seq * STEPS;
        let mut punctuation_candidates = Vec::new();
        let mut word_candidates = Vec::new();
        for pos in 0..STEPS {
            let i = offset + pos;
            if !(0..8).contains(&ids[i])
                || !(0.5..=1.0).contains(&weights[i])
                || (punctuation[i] && word[i])
            {
                return Err(
                    "invalid ordinary top-1 route, confidence, or overlapping token classes".into(),
                );
            }
            if ids[i] == 3 {
                let point = Point {
                    pos,
                    confidence: f64::from(weights[i]),
                };
                if punctuation[i] {
                    punctuation_candidates.push(point);
                } else if word[i] {
                    word_candidates.push(point);
                }
            }
        }
        *row = select_row(&punctuation_candidates, &word_candidates);
        if let Some(pair) = row.pair {
            punctuation_mask[offset + pair.punctuation.pos] = true;
            word_mask[offset + pair.word.pos] = true;
        }
    }
    let as_mask = |v: &[bool]| Tensor::from_slice(v).view([BATCH as i64, STEPS as i64]);
    Ok(Selection {
        punctuation_mask: as_mask(&punctuation_mask),
        word_mask: as_mask(&word_mask),
        rows,
    })
}

fn sufficient(by_source: &[usize; 3]) -> bool {
    by_source.iter().sum::<usize>() >= MIN_TOTAL && by_source.iter().all(|&n| n >= MIN_PER_SOURCE)
}

pub fn coverage_adequate(selections: &[Selection]) -> Result<bool> {
    if selections.len() * BATCH != SOURCES.len() * PER_SOURCE {
        return Err("matched coverage requires 30 batches in frozen source order".into());
    }
    let mut counts = [0; 3];
    for (i, row) in selections.iter().flat_map(|s| s.rows.iter()).enumerate() {
        counts[i / PER_SOURCE] += usize::from(row.pair.is_some());
    }
    Ok(sufficient(&counts))
}

#[derive(Default)]
struct Coverage {
    sequences: usize,
    punctuation: Pool,
    word: Pool,
    sequences_with_punctuation: usize,
    sequences_with_word: usize,
    sequences_with_both: usize,
    within_calipers: usize,
    matched_punctuation: Pool,
    matched_word: Pool,
    absolute_position_gap: usize,
    absolute_confidence_gap: f64,
    max_position_gap: usize,
    max_confidence_gap: f64,
}

impl Coverage {
    fn add(&mut self, row: Row) {
        self.sequences += 1;
        self.punctuation.merge(row.punctuation);
        self.word.merge(row.word);
        self.sequences_with_punctuation += usize::from(row.punctuation.count > 0);
        self.sequences_with_word += usize::from(row.word.count > 0);
        self.sequences_with_both += usize::from(row.punctuation.count > 0 && row.word.count > 0);
        self.within_calipers += row.within_calipers;
        if let Some(pair) = row.pair {
            self.matched_punctuation.add(pair.punctuation);
            self.matched_word.add(pair.word);
            self.absolute_position_gap += pair.position_gap();
            self.max_position_gap = self.max_position_gap.max(pair.position_gap());
            self.absolute_confidence_gap += pair.confidence_gap();
            self.max_confidence_gap = self.max_confidence_gap.max(pair.confidence_gap());
        }
    }

    fn counts(&self) -> Value {
        let n = self.matched_punctuation.count;
        json!({
            "sequences": self.sequences, "matched_sequences": n,
            "unmatched_sequences": self.sequences - n,
            "eligible_punctuation_positions": self.punctuation.count, "eligible_word_positions": self.word.count,
            "matched_positions_per_class": n,
            "unmatched_eligible_punctuation_positions": self.punctuation.count - n,
            "unmatched_eligible_word_positions": self.word.count - n,
            "sequences_with_eligible_punctuation": self.sequences_with_punctuation,
            "sequences_with_eligible_words": self.sequences_with_word,
            "sequences_with_both_eligible_classes": self.sequences_with_both,
            "candidate_pairs_within_calipers": self.within_calipers,
        })
    }

    fn balance(&self) -> Value {
        let n = self.matched_punctuation.count;
        json!({
            "eligible_punctuation": self.punctuation.means(), "eligible_words": self.word.means(),
            "matched_punctuation": self.matched_punctuation.means(), "matched_words": self.matched_word.means(),
            "mean_absolute_position_gap": mean(self.absolute_position_gap as f64, n),
            "max_absolute_position_gap": (n > 0).then_some(self.max_position_gap),
            "mean_absolute_confidence_gap": mean(self.absolute_confidence_gap, n),
            "max_absolute_confidence_gap": (n > 0).then_some(self.max_confidence_gap),
        })
    }
}

#[derive(Default)]
pub struct MatchedStats {
    coverage: Coverage,
    sources: [Coverage; 3],
    effects: Vec<Option<[f64; 2]>>,
    ordinary_sum: f64,
}

impl MatchedStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_coverage(&mut self, selection: &Selection) -> Result<()> {
        if self.coverage.sequences + BATCH > PER_SOURCE * SOURCES.len() {
            return Err("too many sequences in matched study".into());
        }
        let source = self.coverage.sequences / PER_SOURCE;
        for &row in &selection.rows {
            self.coverage.add(row);
            self.sources[source].add(row);
        }
        Ok(())
    }

    pub fn add_batch(
        &mut self,
        selection: &Selection,
        ordinary: &Tensor,
        punctuation_altered: &Tensor,
        word_altered: &Tensor,
    ) -> Result<()> {
        let flat = |values: &Tensor| -> Result<Vec<f32>> {
            check_tensor(values, Kind::Float)?;
            let values = Vec::<f32>::try_from(values.flatten(0, -1))?;
            if values.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite next-token CE in matched study".into());
            }
            Ok(values)
        };
        let base = flat(ordinary)?;
        let punctuation = flat(punctuation_altered)?;
        let word = flat(word_altered)?;
        let mut effects = Vec::with_capacity(BATCH);
        for (i, row) in selection.rows.iter().enumerate() {
            let start = i * STEPS;
            let effect = |altered: &[f32], pos: Option<usize>| -> Result<f64> {
                let end = start + pos.unwrap_or(STEPS);
                if altered[start..end] != base[start..end] {
                    return Err(
                        "matched intervention altered CE before its sole replacement".into(),
                    );
                }
                Ok(pos.map_or(0.0, |p| {
                    f64::from(altered[start + p]) - f64::from(base[start + p])
                }))
            };
            let p = effect(&punctuation, row.pair.map(|m| m.punctuation.pos))?;
            let w = effect(&word, row.pair.map(|m| m.word.pos))?;
            effects.push(row.pair.map(|_| [p, w]));
        }
        let base_sums =
            Vec::<f64>::try_from(ordinary.sum_dim_intlist(&[1i64][..], false, Kind::Double))?;
        self.add_coverage(selection)?;
        self.effects.extend(effects);
        for sum in base_sums {
            self.ordinary_sum += sum;
        }
        Ok(())
    }

    fn coverage_summary(&self) -> Result<Value> {
        if self.coverage.sequences != SOURCES.len() * PER_SOURCE {
            return Err("matched statistics require the complete 240-sequence sample".into());
        }
        let adequate = sufficient(&self.sources.each_ref().map(|s| s.matched_punctuation.count));
        let by_source: serde_json::Map<_, _> = SOURCES
            .iter()
            .zip(&self.sources)
            .map(|(name, s)| {
                let summary = json!({
                    "counts": s.counts(),
                    "balance": s.balance(),
                    "source_minimum_met": s.matched_punctuation.count >= MIN_PER_SOURCE,
                });
                ((*name).into(), summary)
            })
            .collect();
        Ok(json!({
            "support_adequate": adequate, "minimum_matched_overall": MIN_TOTAL,
            "minimum_matched_per_source": MIN_PER_SOURCE,
            "counts": self.coverage.counts(), "by_source": by_source, "balance": self.coverage.balance(),
            "protocol": {
                "punctuation": "contains . ! or ? after excluding special pieces",
                "word": "one optional leading ▁ followed exclusively by one or more ASCII letters; special pieces excluded",
                "eligible_route": "ordinary block 1 top-1 expert ID 3",
                "gate_confidence": "ordinary rank-one top-two softmax weight",
                "position_caliper_inclusive": POSITION_CALIPER,
                "confidence_caliper_inclusive": CONFIDENCE_CALIPER,
                "selection": "within each sequence minimize abs(position gap)/32 + abs(confidence gap)/0.05",
                "tie_break": ["smaller absolute position gap","earlier punctuation index","earlier word index"],
                "maximum_pairs_per_sequence": 1,
                "minimum_matched_overall": MIN_TOTAL, "minimum_matched_per_source": MIN_PER_SOURCE,
                "primary": "matched mean punctuation CE increase minus word CE increase, both versus ordinary",
                "bootstrap": {"unit": "sequence", "strata": "source", "draws": DRAWS, "seed": 20261118,
                              "interval": "linear 2.5 and 97.5 percentiles"},
            },
        }))
    }

    pub fn finish(&self) -> Result<Value> {
        let mut result = self.coverage_summary()?;
        let adequate = result["support_adequate"] == true;
        let observed = self.effects.len();
        if (adequate && observed != self.coverage.sequences) || (!adequate && observed != 0) {
            return Err("coverage gate and number of scored sequences disagree".into());
        }
        let targets = observed * STEPS;
        result["baseline_all_targets"] = json!(targets);
        result["baseline_all_ce"] = json!(mean(self.ordinary_sum, targets));
        result["unaffected_prefix_ce_exact"] = json!(adequate.then_some(true));
        result["matched_effects"] = json!(adequate.then(|| paired_effects(&self.effects)));
        Ok(result)
    }
}

pub fn coverage_report(selections: &[Selection]) -> Result<Value> {
    let mut stats = MatchedStats::new();
    for selection in selections {
        stats.add_coverage(selection)?;
    }
    stats.coverage_summary()
}

#[derive(Default)]
struct EffectSums {
    n: usize,
    sums: [f64; 3],
}

impl EffectSums {
    fn add(&mut self, effect: Option<[f64; 2]>) {
        if let Some([p, w]) = effect {
            self.n += 1;
            self.sums[0] += p;
            self.sums[1] += w;
            self.sums[2] += p - w;
        }
    }
}

fn paired_effects(rows: &[Option<[f64; 2]>]) -> Value {
    let mut observed = EffectSums::default();
    for &row in rows {
        observed.add(row);
    }
    let mut rng = BootstrapRng::new();
    let mut estimates: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(DRAWS));
    let mut counts = Vec::with_capacity(DRAWS);
    for _ in 0..DRAWS {
        let mut draw = EffectSums::default();
        for i in rng.draw() {
            draw.add(rows[i]);
        }
        counts.push(draw.n);
        if draw.n > 0 {
            for (estimates, sum) in estimates.iter_mut().zip(draw.sums) {
                estimates.push(sum / draw.n as f64);
            }
        }
    }
    let nonempty = estimates[0].len();
    let mut results = serde_json::Map::new();
    for (i, name) in [
        "punctuation_increase",
        "word_increase",
        "punctuation_minus_word_increase",
    ]
    .iter()
    .enumerate()
    {
        estimates[i].sort_unstable_by(f64::total_cmp);
        let ci = (nonempty > 0).then(|| {
            [
                quantile(&estimates[i], 0.025),
                quantile(&estimates[i], 0.975),
            ]
        });
        results.insert(
            (*name).into(),
            json!({"mean": mean(observed.sums[i], observed.n), "sequence_bootstrap_ci95": ci}),
        );
    }
    json!({
        "matched_sequences": observed.n, "effects_nats": results,
        "bootstrap": {"unit": "sequence", "strata": "source", "draws": DRAWS, "seed": 20261118,
                      "nonempty_draws": nonempty, "min_matched": counts.iter().min(), "max_matched": counts.iter().max(),
                      "matched_sequences_by_draw": counts},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(pos: usize, confidence: f64) -> Point {
        Point { pos, confidence }
    }

    #[test]
    fn match_obeys_both_calipers_and_stable_tie_breaks() {
        let p = point(40, 0.75);
        let row = select_row(&[p], &[point(8, 0.75), point(73, 0.75), point(40, 0.8125)]);
        assert_eq!(row.within_calipers, 1);
        assert_eq!(row.pair.unwrap().word.pos, 8);
        // All three pairs below share the minimum distance. Inputs are in reverse order.
        let row = select_row(
            &[point(60, 0.75), p],
            &[point(59, 0.75), point(41, 0.75), point(39, 0.75)],
        );
        let pair = row.pair.unwrap();
        assert_eq!((pair.punctuation.pos, pair.word.pos), (40, 39));
        // A nearby word with worse confidence can lose to a more balanced pair.
        let row = select_row(&[p], &[point(39, 0.78), point(34, 0.75)]);
        assert_eq!(row.pair.unwrap().word.pos, 34);
    }

    #[test]
    fn support_requires_both_thresholds() {
        assert!(sufficient(&[30, 30, 60]));
        assert!(!sufficient(&[29, 80, 80]));
        assert!(!sufficient(&[39, 40, 40]));
    }

    #[test]
    fn outcome_blind_counts_survive_zero_support() {
        let zeros = || Tensor::zeros([8, 511], (Kind::Bool, Device::Cpu));
        let selection = Selection {
            rows: [select_row(&[point(0, 0.5)], &[point(100, 0.90)]); BATCH],
            punctuation_mask: zeros(),
            word_mask: zeros(),
        };
        let mut stats = MatchedStats::new();
        for _ in 0..30 {
            stats.add_coverage(&selection).unwrap();
        }
        let report = stats.finish().unwrap();
        assert_eq!(report["support_adequate"], false);
        assert_eq!(
            report["counts"]["unmatched_eligible_punctuation_positions"],
            240
        );
        assert_eq!(
            report["by_source"]["wikipedia"]["counts"]["sequences_with_both_eligible_classes"],
            80
        );
        assert_eq!(report["baseline_all_targets"], 0);
        assert!(report["matched_effects"].is_null());
        assert!(report["balance"]["max_absolute_confidence_gap"].is_null());
    }

    #[test]
    fn estimates_and_intervals_preserve_pairing_in_presence_of_unmatched_rows() {
        let rows: Vec<_> = (0..240)
            .map(|i| {
                let effect = ((i / 80) * 2) as f64;
                (i % 2 == 0).then_some([effect + 1.0, effect])
            })
            .collect();
        let result = paired_effects(&rows);
        let effects = &result["effects_nats"];
        assert_eq!(result["matched_sequences"], 120);
        assert_eq!(effects["punctuation_increase"]["mean"], 3.);
        assert_eq!(effects["word_increase"]["mean"], 2.);
        assert_eq!(
            effects["punctuation_minus_word_increase"]["sequence_bootstrap_ci95"],
            json!([1.0, 1.0])
        );
        assert_eq!(result["bootstrap"]["nonempty_draws"], 1000);
        assert!(result["bootstrap"]["min_matched"].as_u64().unwrap() < 120);
        assert!(result["bootstrap"]["max_matched"].as_u64().unwrap() > 120);
    }
}
