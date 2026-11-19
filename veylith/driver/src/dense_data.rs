//! Fixed input stream for the v3.1-L comparison. Training uses shuffled windows
//! of the concatenated train split. Validation packs each source separately.
use crate::Result;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tch::Tensor;

pub const SOURCES: [&str; 3] = ["cosmopedia-v2", "fineweb-edu", "wikipedia"];
const WIDTH: usize = 512;
const BATCH: usize = 8;
const ACCUMULATION: usize = 8;
const UPDATES: usize = 500;
const TRAIN_ROWS: usize = UPDATES * ACCUMULATION * BATCH;
const TRAIN_TOKENS: usize = 95_951_233;
const ROW_BYTES: usize = WIDTH * 2;
const PAD: i64 = 3;
// The manifest, tokenizer and complete shuffle digest are recorded in the v3.1-L
// summary and the v3.1-M analysis. The ordered stream hashes were reproduced
// separately with archive/python-research's loader.py and training/data.py.
const MANIFEST_SHA: &str = "4200d524075adf5773bb4634003bfe6296d735bd62118f740ad944bfe881e7b9";
const MANIFEST_DIGEST: &str = "7a6e7991dd83244ced4c9c13627028e1824f00b0f58735cc665323ff111bdcf0";
const TOKENIZER_SHA: &str = "411227a8e5289d6f4b450059b744ed278532a4624d3895f1a4979600c49ae034";
const EPOCH_ORDER_SHA: &str = "0ba1c027522451ef9518540dcb6e9c1a4cf63e0eb335687bc4827101890826c3";
const TRAIN_ROWS_SHA: &str = "aa66e1dc13514173e5218bab3f5c1217476c17f3f5c6daaee97a6a26049ae2ab";
const VAL_TARGETS: [usize; 3] = [310_531, 1_296_602, 511_097];
const VAL_STREAM_SHA: [&str; 3] = [
    "0694b704ed955e52413465d268102f919d8d61ed9e86a2396d3d7ac9eafc0265",
    "65f5411f06169d7f03189304a2f871729651bbc279b6b02ff8e6c13349cf1c62",
    "3f07030fa0b02230b8788871769a66533d957bf90aad2bf4d57bbfdffceda443",
];

#[derive(Deserialize)]
struct Manifest {
    digest: String,
    tokenization: Tokenization,
}
#[derive(Deserialize)]
struct Tokenization {
    tokenizer: Tokenizer,
    special_ids: HashMap<String, i64>,
    vocab_size: usize,
    splits: HashMap<String, Split>,
}
#[derive(Deserialize)]
struct Tokenizer {
    model: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
struct Split {
    tokens: usize,
    docs: usize,
    shards: Vec<Shard>,
}
#[derive(Deserialize)]
struct Shard {
    path: PathBuf,
    meta: PathBuf,
    sha256: String,
    tokens: usize,
    docs: usize,
}
#[derive(Deserialize)]
struct ShardMetadata {
    docs: Vec<Document>,
}
#[derive(Deserialize)]
struct Document {
    source: String,
    doc_key: String,
    text_sha256: String,
    start: usize,
    end: usize,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The frozen manifest pins each complete .npy file, including its header. Only
/// NumPy v1 little-endian u16 vectors written by the archived pipeline occur.
fn shard_data<'a>(shard: &Shard, bytes: &'a [u8]) -> Result<&'a [u8]> {
    if hash(bytes) != shard.sha256 || bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\x00" {
        return Err("token shard hash or NumPy format mismatch".into());
    }
    let start = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let data = bytes.get(start..).ok_or("truncated NumPy token shard")?;
    if shard.tokens.checked_mul(2) != Some(data.len()) {
        return Err("token shard count differs from the manifest".into());
    }
    Ok(data)
}

/// Visit every document span in file order and prove that the spans exactly
/// partition each shard. A concatenation can then cross shard boundaries.
fn visit_split(
    corpus: &Path,
    split: &Split,
    mut visit: impl FnMut(Document, &[u8]) -> Result<()>,
) -> Result<()> {
    let mut tokens = 0;
    let mut docs = 0;
    for shard in &split.shards {
        let bytes = fs::read(corpus.join(&shard.path))?;
        let data = shard_data(shard, &bytes)?;
        let meta: ShardMetadata = serde_json::from_slice(&fs::read(corpus.join(&shard.meta))?)?;
        if meta.docs.len() != shard.docs {
            return Err("token metadata document count mismatch".into());
        }
        let mut cursor = 0;
        for doc in meta.docs {
            if doc.start != cursor || doc.end <= doc.start || doc.end > shard.tokens {
                return Err("document spans do not partition the token shard".into());
            }
            let span = &data[doc.start * 2..doc.end * 2];
            cursor = doc.end;
            visit(doc, span)?;
        }
        if cursor != shard.tokens {
            return Err("incomplete token shard coverage".into());
        }
        docs += shard.docs;
        tokens += shard.tokens;
    }
    if docs != split.docs || tokens != split.tokens {
        return Err("split totals differ from the manifest".into());
    }
    Ok(())
}

// NumPy default_rng(SeedSequence([0, 0])) gives this reproducible PCG64 state.
// permutation(count) uses Fisher-Yates and rejects masked values above the bound.
fn epoch_zero_order(count: usize) -> Vec<usize> {
    let mut state: u128 = 35399562948360463058890781895381311971;
    let mut upper: Option<u32> = None;
    let mut next_u32 = || {
        if let Some(high) = upper.take() {
            return high;
        }
        state = state
            .wrapping_mul(0x2360ed051fc65da44385df649fccf645)
            .wrapping_add(87136372517582989555478159403783844777);
        let word = ((state >> 64) as u64 ^ state as u64).rotate_right((state >> 122) as u32);
        upper = Some((word >> 32) as u32);
        word as u32
    };
    let mut order: Vec<_> = (0..count).collect();
    for top in (1..count).rev() {
        let mask = u32::MAX >> (top as u32).leading_zeros();
        let index = loop {
            let value = (next_u32() & mask) as usize;
            if value <= top {
                break value;
            }
        };
        order.swap(top, index);
    }
    order
}

fn window_order_hash(order: &[usize]) -> String {
    let mut sha = Sha256::new();
    for &index in order {
        sha.update((index as i64).to_le_bytes());
    }
    format!("{:x}", sha.finalize())
}

pub struct Corpus {
    ordered_train: Vec<u8>,
    validation: [Vec<u8>; 3],
    evidence: Value,
}

impl Corpus {
    pub fn load(root: &Path) -> Result<Self> {
        let corpus = root.join("corpus");
        let bytes = fs::read(corpus.join("manifest.json"))?;
        if hash(&bytes) != MANIFEST_SHA {
            return Err("corpus manifest differs from v3.1-L".into());
        }
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        let tok = manifest.tokenization;
        if manifest.digest != MANIFEST_DIGEST
            || tok.vocab_size != 8192
            || tok.special_ids.get("pad") != Some(&PAD)
            || tok.tokenizer.sha256 != TOKENIZER_SHA
            || hash(&fs::read(corpus.join(tok.tokenizer.model))?) != TOKENIZER_SHA
        {
            return Err("tokenizer or corpus identity differs from v3.1-L".into());
        }
        let train = tok.splits.get("train").ok_or("missing train split")?;
        let val = tok
            .splits
            .get("validation")
            .ok_or("missing validation split")?;
        if train.tokens != TRAIN_TOKENS {
            return Err("training token count differs from v3.1-L".into());
        }
        let mut train_stream = Vec::with_capacity(TRAIN_TOKENS * 2);
        let mut train_keys = HashSet::with_capacity(train.docs);
        let mut train_texts = HashSet::with_capacity(train.docs);
        visit_split(&corpus, train, |doc, span| {
            train_keys.insert(doc.doc_key);
            train_texts.insert(doc.text_sha256);
            train_stream.extend_from_slice(span);
            Ok(())
        })?;

        let order = epoch_zero_order(TRAIN_TOKENS.div_ceil(WIDTH));
        let order_sha = window_order_hash(&order);
        if order_sha != EPOCH_ORDER_SHA {
            return Err("epoch-zero order differs from the recorded v3.1-M digest".into());
        }
        let mut ordered_train = Vec::with_capacity(TRAIN_ROWS * ROW_BYTES);
        for &index in &order[..TRAIN_ROWS] {
            let start = index * ROW_BYTES;
            let row = train_stream
                .get(start..start + ROW_BYTES)
                .ok_or("partial row inside the 500-update training run")?;
            ordered_train.extend_from_slice(row);
        }
        drop(train_stream);
        let training_sha = hash(&ordered_train);
        if training_sha != TRAIN_ROWS_SHA {
            return Err("selected training tokens differ from the archived loader".into());
        }

        let mut validation: [Vec<u8>; 3] = Default::default();
        visit_split(&corpus, val, |doc, span| {
            if train_keys.contains(&doc.doc_key) || train_texts.contains(&doc.text_sha256) {
                return Err("validation overlaps a training document".into());
            }
            let source = SOURCES
                .iter()
                .position(|s| *s == doc.source)
                .ok_or("unknown validation source")?;
            validation[source].extend_from_slice(span);
            Ok(())
        })?;
        let mut source_evidence = serde_json::Map::new();
        for (i, stream) in validation.iter().enumerate() {
            let tokens = stream.len() / 2;
            let rows = tokens.div_ceil(WIDTH);
            let targets = tokens - rows; // The first token in every row is input only.
            let sha = hash(stream);
            if targets != VAL_TARGETS[i] || sha != VAL_STREAM_SHA[i] {
                return Err(
                    "validation coverage or source order differs from the reference".into(),
                );
            }
            source_evidence.insert(
                SOURCES[i].into(),
                json!({"rows":rows,"targets":targets,"ordered_tokens_sha256":sha}),
            );
        }
        let evidence = json!({
            "manifest_sha256":MANIFEST_SHA,"manifest_digest":MANIFEST_DIGEST,"tokenizer_sha256":TOKENIZER_SHA,
            "train":{"split":"train","packing":"concat","seed":0,"epoch":0,"epoch_sequences":order.len(),
                     "split_tokens":TRAIN_TOKENS,"verified_shards":train.shards.len(),"epoch_order_sha256":order_sha,
                     "reference_500_updates":{"rows":TRAIN_ROWS,"positions":TRAIN_ROWS*WIDTH,"targets":TRAIN_ROWS*(WIDTH-1),
                     "ordered_tokens_sha256":training_sha}},
            "validation":{"split":"validation","verified_shards":val.shards.len(),"sources":source_evidence,
                          "targets":VAL_TARGETS.iter().sum::<usize>(),"overlapping_train_doc_keys":0,"overlapping_train_text_hashes":0}
        });
        Ok(Self {
            ordered_train,
            validation,
            evidence,
        })
    }

    /// Zero-based optimizer update and microbatch position in the historic 8×8 schedule.
    pub fn train_batch(&self, update: usize, microbatch: usize) -> Result<Tensor> {
        if update >= UPDATES || microbatch >= ACCUMULATION {
            return Err("training batch outside the v3.1-L horizon".into());
        }
        let from = (update * ACCUMULATION + microbatch) * BATCH * ROW_BYTES;
        let bytes = &self.ordered_train[from..from + BATCH * ROW_BYTES];
        let ids = decode_ids(bytes);
        Ok(Tensor::f_from_slice(&ids)?.f_view([BATCH as i64, WIDTH as i64])?)
    }

    /// Source-pure [rows, 512] IDs and [rows, 511] valid-target mask. Keep the
    /// final incomplete batch and its right-padded last sequence for each source.
    pub fn validation_batches(
        &self,
        source: usize,
    ) -> impl Iterator<Item = Result<(Tensor, Tensor)>> + '_ {
        let stream = self.validation.get(source);
        let count = stream.map_or(1, |s| s.len().div_ceil(BATCH * ROW_BYTES));
        (0..count).map(move |index| {
            let stream = stream.ok_or("unknown validation source index")?;
            let start = index * BATCH * ROW_BYTES;
            let bytes = &stream[start..(start + BATCH * ROW_BYTES).min(stream.len())];
            validation_batch(bytes)
        })
    }

    pub fn evidence(&self) -> Value {
        self.evidence.clone()
    }
}

fn decode_ids(bytes: &[u8]) -> Vec<i64> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| i64::from(u16::from_le_bytes(pair)))
        .collect()
}

fn validation_batch(bytes: &[u8]) -> Result<(Tensor, Tensor)> {
    let mut ids = decode_ids(bytes);
    let valid_tokens = ids.len();
    let rows = valid_tokens.div_ceil(WIDTH);
    ids.resize(rows * WIDTH, PAD);
    let mut mask = Vec::with_capacity(rows * (WIDTH - 1));
    for row in 0..rows {
        let length = (valid_tokens - row * WIDTH).min(WIDTH);
        mask.extend((1..WIDTH).map(|pos| pos < length));
    }
    Ok((
        Tensor::f_from_slice(&ids)?.f_view([rows as i64, WIDTH as i64])?,
        Tensor::f_from_slice(&mask)?.f_view([rows as i64, WIDTH as i64 - 1])?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reproduces_archived_epoch_digest_and_training_prefix() {
        let order = epoch_zero_order(187405);
        assert_eq!(window_order_hash(&order), EPOCH_ORDER_SHA);
        assert_eq!(
            window_order_hash(&order[..TRAIN_ROWS]),
            "24cbdaa46620e76701cf8bd652a803f652f8cd8c9b1e22f3072ab8bb7b180619"
        );
        assert!(!order[..TRAIN_ROWS].contains(&187404));
    }
    #[test]
    fn validation_tail_excludes_padding_and_row_boundary_target() {
        let bytes = vec![0u8; (WIDTH + 2) * 2];
        let (ids, mask) = validation_batch(&bytes).unwrap();
        assert_eq!(ids.size(), [2, WIDTH as i64]);
        assert_eq!(ids.int64_value(&[1, 2]), PAD);
        assert_eq!(mask.sum(tch::Kind::Int64).int64_value(&[]), 512);
        assert_eq!(mask.int64_value(&[1, 0]), 1);
        assert_eq!(mask.int64_value(&[1, 1]), 0);
    }
}
