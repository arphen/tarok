//! Neural-network player using a TorchScript model via tch-rs.
//!
//! Runs batched inference: all pending decisions are concatenated into
//! a single tensor, forwarded through the model, and sampled via
//! masked softmax — entirely in Rust, no Python / GIL.

use rand::Rng;
use tch::{CModule, Device, Tensor, no_grad};

use crate::player::*;

// -----------------------------------------------------------------------
// NeuralNetPlayer
// -----------------------------------------------------------------------

pub struct NeuralNetPlayer {
    model: CModule,
    device: Device,
    explore_rate: f64,
}

impl NeuralNetPlayer {
    pub fn new(model_path: &str, device: Device, explore_rate: f64) -> Self {
        let model = CModule::load_on_device(model_path, device)
            .expect("Failed to load TorchScript model");
        NeuralNetPlayer { model, device, explore_rate }
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
            flat_states[i * state_size..(i + 1) * state_size]
                .copy_from_slice(&ctx.state_encoding);
        }

        // Forward pass — model returns (bid_logits, king_logits, talon_logits, card_logits, values)
        let outputs = no_grad(|| {
            let input = tch::IValue::Tensor(
                Tensor::from_slice(&flat_states)
                    .reshape([batch_size as i64, state_size as i64])
                    .to_device(self.device),
            );
            self.model.forward_is(&[input]).expect("model forward failed")
        });

        let tuple = match outputs {
            tch::IValue::Tuple(t) => t,
            _ => panic!("Expected tuple output from model"),
        };
        let logit_tensors: [Tensor; 4] = [
            match &tuple[0] { tch::IValue::Tensor(t) => t.to_device(Device::Cpu), _ => panic!("bad") },
            match &tuple[1] { tch::IValue::Tensor(t) => t.to_device(Device::Cpu), _ => panic!("bad") },
            match &tuple[2] { tch::IValue::Tensor(t) => t.to_device(Device::Cpu), _ => panic!("bad") },
            match &tuple[3] { tch::IValue::Tensor(t) => t.to_device(Device::Cpu), _ => panic!("bad") },
        ];
        let values_t = match &tuple[4] {
            tch::IValue::Tensor(t) => t.to_device(Device::Cpu),
            _ => panic!("bad"),
        };

        // Sample per context
        let mut results = Vec::with_capacity(batch_size);
        for (i, ctx) in contexts.iter().enumerate() {
            let head_idx = ctx.decision_type as usize;
            let action_size = ctx.decision_type.action_size();
            let value = values_t.double_value(&[i as i64]) as f32;

            // Epsilon-greedy exploration
            if self.explore_rate > 0.0 && rng.random::<f64>() < self.explore_rate {
                let legal: Vec<usize> = ctx.legal_mask.iter().enumerate()
                    .filter(|(_, &v)| v > 0.5)
                    .map(|(j, _)| j)
                    .collect();
                if !legal.is_empty() {
                    let action = legal[rng.random_range(0..legal.len())];
                    results.push(DecisionResult { action, log_prob: 0.0, value });
                    continue;
                }
            }

            // Masked softmax sampling
            let logits: Vec<f32> = (0..action_size)
                .map(|j| logit_tensors[head_idx].double_value(&[i as i64, j as i64]) as f32)
                .collect();
            let (action, log_prob) = sample_masked(&logits, &ctx.legal_mask[..action_size], &mut rng);
            results.push(DecisionResult { action, log_prob, value });
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
        let uniform = if legal_count > 0 { 1.0 / legal_count as f32 } else { 0.0 };
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
    if chosen >= n { chosen = n - 1; }
    if mask.get(chosen).map_or(true, |&m| m < 0.5) {
        chosen = mask.iter().position(|&m| m > 0.5).unwrap_or(0);
    }

    let log_prob = probs[chosen].max(1e-10).ln();
    (chosen, log_prob)
}
