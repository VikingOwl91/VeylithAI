use crate::Result;
use candle_core::pickle::read_pth_tensor_info;
use prost::Message;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tch::{Kind, Tensor};

const MODEL_SHA256: &str = "8f3c7d9847803b2a0feb316a1c01ed6065d3b15f2cf67bc1909a4a514615e570";
const ORIGINAL_SAMPLE_SHA256: &str =
    "70f2643cb4f3e685543b1610e6c28934bfb678891adacc0f98147de21c4c0369";
const SOURCES: [&str; 3] = ["cosmopedia-v2", "fineweb-edu", "wikipedia"];
const SEQUENCES_PER_SOURCE: usize = 80;
const SEQ_LEN: usize = 512;
const BATCH_SIZE: usize = 8;
const BYTES_PER_SEQUENCE: usize = SEQ_LEN * size_of::<u16>();

pub fn load_weights(checkpoint: &Path, scratch: &Path) -> Result<HashMap<String, Tensor>> {
    // The frozen torch.save ZIP has a central directory that Candle's ZIP reader
    // rejects. Repack the members without modifying the original or unpickling in Python.
    let zip = tempfile::NamedTempFile::new_in(scratch)?;
    let mut input = OsString::from("@");
    input.push(checkpoint.canonicalize()?);
    let status = Command::new("bsdtar")
        .arg("-cf")
        .arg(zip.path())
        .args(["--format", "zip", "--options", "zip:compression=store"])
        .arg(input)
        .stdin(Stdio::null())
        .status()
        .map_err(|err| format!("launching bsdtar to read the frozen checkpoint: {err}"))?;
    if !status.success() {
        return Err(format!("checkpoint archive normalization failed with {status}").into());
    }

    let mut parameters = read_pth_tensor_info(zip.path(), false, Some("model"))?;
    parameters.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    let mut archive = zip::ZipArchive::new(BufReader::new(File::open(zip.path())?))?;
    let mut weights = HashMap::new();
    let mut digest = Sha256::new();
    for param in parameters {
        if param.dtype != candle_core::DType::F32 || !param.layout.is_contiguous() {
            return Err("expected contiguous float32 checkpoint parameters".into());
        }
        let dims: Vec<i64> = param
            .layout
            .dims()
            .iter()
            .copied()
            .map(i64::try_from)
            .collect::<std::result::Result<_, _>>()?;
        let size = param
            .layout
            .dims()
            .iter()
            .try_fold(4usize, |n, &d| n.checked_mul(d))
            .ok_or("invalid parameter size")?;
        // Candle exposes the storage offset in bytes.
        let offset = param.layout.start_offset() as u64;
        let mut member = archive.by_name(&param.path)?;
        if offset
            .checked_add(size as u64)
            .is_none_or(|end| end > member.size())
        {
            return Err("parameter exceeds its checkpoint storage member".into());
        }
        if offset != 0 {
            io::copy(&mut member.by_ref().take(offset), &mut io::sink())?;
        }
        let mut bytes = vec![0; size];
        member.read_exact(&mut bytes)?;
        digest.update(param.name.as_bytes());
        digest.update([0]);
        digest.update(&bytes);
        #[cfg(target_endian = "big")]
        for word in bytes.chunks_exact_mut(4) {
            word.reverse();
        }
        weights.insert(
            param.name,
            Tensor::f_from_data_size(&bytes, &dims, Kind::Float)?,
        );
    }
    if weights.len() != 229 || format!("{:x}", digest.finalize()) != MODEL_SHA256 {
        return Err(
            "model parameter count or float32 payload digest differs from update 500".into(),
        );
    }
    Ok(weights)
}

#[derive(Clone, PartialEq, Message)]
struct Vocabulary {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<Piece>,
}

#[derive(Clone, PartialEq, Message)]
struct Piece {
    #[prost(string, tag = "1")]
    text: String,
}

#[derive(Deserialize)]
struct Manifest {
    tokenization: Tokenization,
}
#[derive(Deserialize)]
struct Tokenization {
    tokenizer: Tokenizer,
    splits: HashMap<String, Split>,
}
#[derive(Deserialize)]
struct Tokenizer {
    model: PathBuf,
}
#[derive(Deserialize)]
struct Split {
    shards: Vec<Shard>,
}
#[derive(Deserialize)]
struct Shard {
    path: PathBuf,
    meta: PathBuf,
    tokens: usize,
    sha256: String,
}
#[derive(Deserialize)]
struct ShardMetadata {
    docs: Vec<Document>,
}
#[derive(Deserialize)]
struct Document {
    source: String,
    start: usize,
    end: usize,
}

fn punctuation(piece: &str) -> bool {
    piece.contains(['.', '!', '?']) && !(piece.starts_with('<') && piece.ends_with('>'))
}

// These frozen shards are NumPy v1 vectors of little-endian unsigned 16-bit IDs.
fn npy_payload(file: &[u8], tokens: usize) -> Result<&[u8]> {
    if file.len() < 10 || &file[..8] != b"\x93NUMPY\x01\x00" {
        return Err("invalid NumPy v1 token shard".into());
    }
    let end = 10 + u16::from_le_bytes([file[8], file[9]]) as usize;
    let header =
        std::str::from_utf8(file.get(10..end).ok_or("truncated NumPy header")?)?.replace(' ', "");
    let data = &file[end..];
    if !header.contains("'descr':'<u2'")
        || !header.contains("'fortran_order':False")
        || !header.contains(&format!("'shape':({tokens},)"))
        || tokens.checked_mul(2) != Some(data.len())
    {
        return Err(
            "NumPy token shard dtype, shape or byte count differs from the manifest".into(),
        );
    }
    Ok(data)
}

pub fn load_batches(
    root: &Path,
    confirmation: bool,
    expected_sample_sha: &str,
) -> Result<Vec<(Tensor, Tensor)>> {
    let corpus = root.join("corpus");
    let manifest: Manifest = serde_json::from_reader(File::open(corpus.join("manifest.json"))?)?;
    let vocabulary = Vocabulary::decode(
        fs::read(corpus.join(manifest.tokenization.tokenizer.model))?.as_slice(),
    )?;
    let is_punctuation: Vec<bool> = vocabulary
        .pieces
        .iter()
        .map(|p| punctuation(&p.text))
        .collect();
    let split = manifest
        .tokenization
        .splits
        .get("validation")
        .ok_or("missing validation split")?;
    let span = SEQUENCES_PER_SOURCE * BYTES_PER_SEQUENCE;
    let limit = if confirmation { 2 * span } else { span };
    let mut streams: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::with_capacity(limit));
    for shard in &split.shards {
        let file = fs::read(corpus.join(&shard.path))?;
        if format!("{:x}", Sha256::digest(&file)) != shard.sha256 {
            return Err("validation shard SHA-256 mismatch".into());
        }
        let bytes = npy_payload(&file, shard.tokens)?;
        let meta: ShardMetadata = serde_json::from_reader(File::open(corpus.join(&shard.meta))?)?;
        for doc in meta.docs {
            let Some(index) = SOURCES.iter().position(|&s| s == doc.source) else {
                continue;
            };
            let stream = &mut streams[index];
            if stream.len() == limit {
                continue;
            }
            if doc.start > doc.end || doc.end > shard.tokens {
                return Err("document token offsets exceed their validation shard".into());
            }
            let available = &bytes[doc.start * 2..doc.end * 2];
            let take = available.len().min(limit - stream.len());
            stream.extend_from_slice(&available[..take]);
        }
        if streams.iter().all(|s| s.len() == limit) {
            break;
        }
    }
    if streams.iter().any(|s| s.len() != limit) {
        return Err("incomplete validation sequences in one or more sources".into());
    }
    let mut original_digest = Sha256::new();
    let mut selected_digest = Sha256::new();
    let mut original_rows = HashSet::new();
    for stream in &streams {
        original_digest.update(&stream[..span]);
        selected_digest.update(&stream[limit - span..]);
        if confirmation {
            original_rows.extend(stream[..span].chunks_exact(BYTES_PER_SEQUENCE));
        }
    }
    if format!("{:x}", original_digest.finalize()) != ORIGINAL_SAMPLE_SHA256
        || format!("{:x}", selected_digest.finalize()) != expected_sample_sha
    {
        return Err("ordered validation token digest differs from the frozen sample".into());
    }

    let mut batches = Vec::new();
    for stream in &streams {
        for batch in stream[limit - span..].chunks_exact(BATCH_SIZE * BYTES_PER_SEQUENCE) {
            let mut tokens = Vec::with_capacity(BATCH_SIZE * SEQ_LEN);
            let mut mask = Vec::with_capacity(BATCH_SIZE * (SEQ_LEN - 1));
            for row in batch.chunks_exact(BYTES_PER_SEQUENCE) {
                if confirmation && original_rows.contains(row) {
                    return Err("confirmation sample overlaps the original sample".into());
                }
                for (pos, word) in row.chunks_exact(2).enumerate() {
                    let id = u16::from_le_bytes([word[0], word[1]]);
                    let punct = *is_punctuation
                        .get(usize::from(id))
                        .ok_or("token ID outside vocabulary")?;
                    tokens.push(i64::from(id));
                    if pos + 1 < SEQ_LEN {
                        mask.push(punct);
                    }
                }
            }
            batches.push((
                Tensor::f_from_slice(&tokens)?.f_view([BATCH_SIZE as i64, SEQ_LEN as i64])?,
                Tensor::f_from_slice(&mask)?.f_view([BATCH_SIZE as i64, SEQ_LEN as i64 - 1])?,
            ));
        }
    }
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_matches_sentencepiece_inputs() {
        assert!(punctuation("▁Dr."));
        assert!(punctuation("!?"));
        assert!(!punctuation("<.>"));
        assert!(!punctuation("▁word"));
    }

    #[test]
    fn invalid_headers_are_errors() {
        assert!(npy_payload(b"", 0).is_err());
        assert!(npy_payload(b"\x93NUMPY\x01\x00\xff\xff", 1).is_err());
    }
}
