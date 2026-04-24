//! Neural-network player using a TorchScript model via tch-rs.
//!
//! Runs batched inference: all pending decisions are concatenated into
//! a single tensor, forwarded through the model, and sampled via
//! masked softmax — entirely in Rust, no Python / GIL.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use rand::Rng;
use tch::{no_grad, CModule, Device, Tensor};

use crate::player::*;

// -----------------------------------------------------------------------
// Global TorchScript model cache
//
// `CModule::load_on_device` deserializes the full model graph from disk
// on every call — this costs tens to hundreds of milliseconds and, for
// training pipelines that issue many `run_self_play` invocations per
// iteration (e.g. the duplicate-RL pairing adapter), dominates the wall
// clock. We keep a process-wide cache keyed by (canonical_path, device,
// mtime) so repeated loads of the same checkpoint are free. The mtime
// component invalidates the cache automatically when a checkpoint is
// re-exported (e.g. the learner's `_current.pt` at the end of each
// iteration), so callers do not need to invalidate manually.
// -----------------------------------------------------------------------

type ModelCacheKey = (PathBuf, String, Option<SystemTime>);

fn model_cache() -> &'static Mutex<HashMap<ModelCacheKey, Arc<CModule>>> {
    static CACHE: OnceLock<Mutex<HashMap<ModelCacheKey, Arc<CModule>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn device_key(device: Device) -> String {
    // `Device` doesn't implement Hash, so stringify. All training flows
    // currently use Cpu; listing others here is future-proofing.
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(i) => format!("cuda:{i}"),
        Device::Mps => "mps".to_string(),
        Device::Vulkan => "vulkan".to_string(),
    }
}

fn path_mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

/// Load a TorchScript model, reusing a cached `Arc<CModule>` when the
/// underlying file hasn't changed since the last load. Falls back to a
/// fresh load if the path cannot be canonicalised or metadata cannot be
/// read (cache is bypassed in that case).
fn load_cached(model_path: &str, device: Device) -> Arc<CModule> {
    let raw = Path::new(model_path);
    let canonical = fs::canonicalize(raw).unwrap_or_else(|_| raw.to_path_buf());
    let mtime = path_mtime(&canonical);
    let key: ModelCacheKey = (canonical.clone(), device_key(device), mtime);

    {
        let cache = model_cache().lock().expect("model cache poisoned");
        if let Some(existing) = cache.get(&key) {
            return existing.clone();
        }
    }

    let fresh = Arc::new(
        CModule::load_on_device(&canonical, device)
            .expect("Failed to load TorchScript model"),
    );

    let mut cache = model_cache().lock().expect("model cache poisoned");
    // Drop stale entries for this (path, device) combination so the
    // cache doesn't grow unbounded as checkpoints are re-exported.
    cache.retain(|(p, d, _), _| !(p == &canonical && d == &key.1));
    cache.insert(key, fresh.clone());
    fresh
}

#[doc(hidden)]
pub fn clear_model_cache_for_tests() {
    model_cache().lock().expect("model cache poisoned").clear()
}

// -----------------------------------------------------------------------
// NeuralNetPlayer
// -----------------------------------------------------------------------

pub struct NeuralNetPlayer {
    model: Arc<CModule>,
    device: Device,
    explore_rate: f64,
}

impl NeuralNetPlayer {
    pub fn new(model_path: &str, device: Device, explore_rate: f64) -> Self {
        let model = load_cached(model_path, device);
        NeuralNetPlayer {
            model,
            device,
            explore_rate,
        }
    }
}

impl BatchPlayer for NeuralNetPlayer {
    fn batch_decide(&self, contexts: &[DecisionContext<'_>]) -> Vec<DecisionResult> {
        if contexts.is_empty() {
            return Vec::new();
        }

        let mut rng = rand::rng();
        let batch_size = contexts.len();
        let state_size = contexts[0].state_encoding.len();

        // Build flat state tensor [B, state_size]
        let mut flat_states = vec![0f32; batch_size * state_size];
        for (i, ctx) in contexts.iter().enumerate() {
            flat_states[i * state_size..(i + 1) * state_size].copy_from_slice(&ctx.state_encoding);
        }

        // Forward pass — model returns (bid_logits, king_logits, talon_logits, card_logits, values)
        let outputs = no_grad(|| {
            let input = tch::IValue::Tensor(
                Tensor::from_slice(&flat_states)
                    .reshape([batch_size as i64, state_size as i64])
                    .to_device(self.device),
            );
            self.model
                .forward_is(&[input])
                .expect("model forward failed")
        });

        let tuple = match outputs {
            tch::IValue::Tuple(t) => t,
            _ => panic!("Expected tuple output from model"),
        };
        // Pull each head once into a contiguous f32 buffer on CPU. This
        // avoids the per-element `double_value(&[i,j])` indexing that
        // dominates readback cost for large batches.
        let head_to_vec = |idx: usize, action_size: usize| -> Vec<f32> {
            let t = match &tuple[idx] {
                tch::IValue::Tensor(t) => t.to_device(Device::Cpu).contiguous(),
                _ => panic!("bad"),
            };
            let total = batch_size * action_size;
            let mut buf = vec![0f32; total];
            t.copy_data(&mut buf, total);
            buf
        };
        let bid_buf = head_to_vec(0, DecisionType::Bid.action_size());
        let king_buf = head_to_vec(1, DecisionType::KingCall.action_size());
        let talon_buf = head_to_vec(2, DecisionType::TalonPick.action_size());
        let card_buf = head_to_vec(3, DecisionType::CardPlay.action_size());
        let head_bufs: [&[f32]; 4] = [&bid_buf, &king_buf, &talon_buf, &card_buf];
        let head_strides: [usize; 4] = [
            DecisionType::Bid.action_size(),
            DecisionType::KingCall.action_size(),
            DecisionType::TalonPick.action_size(),
            DecisionType::CardPlay.action_size(),
        ];

        let values_t = match &tuple[4] {
            tch::IValue::Tensor(t) => t.to_device(Device::Cpu).contiguous(),
            _ => panic!("bad"),
        };
        let mut values_buf = vec![0f32; batch_size];
        values_t.copy_data(&mut values_buf, batch_size);

        // Sample per context
        let mut results = Vec::with_capacity(batch_size);
        for (i, ctx) in contexts.iter().enumerate() {
            let head_idx = ctx.decision_type as usize;
            let action_size = ctx.decision_type.action_size();
            let value = values_buf[i];

            // Epsilon-greedy exploration
            if self.explore_rate > 0.0 && rng.random::<f64>() < self.explore_rate {
                let legal: Vec<usize> = ctx
                    .legal_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v > 0.5)
                    .map(|(j, _)| j)
                    .collect();
                if !legal.is_empty() {
                    let action = legal[rng.random_range(0..legal.len())];
                    results.push(DecisionResult {
                        action,
                        log_prob: 0.0,
                        value,
                    });
                    continue;
                }
            }

            // Masked softmax sampling — slice into the pre-copied head buffer.
            let stride = head_strides[head_idx];
            let row = &head_bufs[head_idx][i * stride..i * stride + action_size];
            let (action, log_prob) =
                sample_masked(row, &ctx.legal_mask[..action_size], &mut rng);
            results.push(DecisionResult {
                action,
                log_prob,
                value,
            });
        }

        results
    }

    fn name(&self) -> &str {
        "neural_net"
    }
}

// -----------------------------------------------------------------------
// Masked softmax + categorical sampling
// -----------------------------------------------------------------------

pub(crate) fn sample_masked(logits: &[f32], mask: &[f32], rng: &mut impl Rng) -> (usize, f32) {
    let n = logits.len().min(mask.len());

    // Apply mask
    let mut masked: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        if mask[i] > 0.5 {
            masked.push(logits[i]);
        } else {
            masked.push(f32::NEG_INFINITY);
        }
    }

    // Softmax
    let max_val = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = masked.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    } else {
        let legal_count = mask.iter().filter(|&&m| m > 0.5).count();
        let uniform = if legal_count > 0 {
            1.0 / legal_count as f32
        } else {
            0.0
        };
        for (i, p) in probs.iter_mut().enumerate() {
            *p = if mask[i] > 0.5 { uniform } else { 0.0 };
        }
    }

    // Sample
    let u: f32 = rng.random::<f32>();
    let mut cumsum = 0.0f32;
    let mut chosen = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            chosen = i;
            break;
        }
    }
    if chosen >= n {
        chosen = n - 1;
    }
    if mask.get(chosen).map_or(true, |&m| m < 0.5) {
        chosen = mask.iter().position(|&m| m > 0.5).unwrap_or(0);
    }

    let log_prob = probs[chosen].max(1e-10).ln();
    (chosen, log_prob)
}
