# QwenVLA: Adapting Qwen 3 VL for Vision-Language-Action

## Project Overview

Adapt Qwen 3 VL (2B) to a Vision-Language-Action model using the SmolVLA approach, trained on custom Miniworld 3D environment recordings.

**Key Design Decisions:**
- Base model: Qwen 3 VL 2B (frozen, 4-bit quantized)
- Action space: Mouse + keyboard (discrete + continuous)
- Visual tokens: Start without aggressive pooling, use moderate resolution (384×384)
- Temporal context: 3 frames (t, t-2, t-4) for motion/velocity understanding
- Action chunks: 50 actions per inference
- Training objective: Flow matching
- Target hardware: RTX 3090 (24GB VRAM)

**Memory Budget (3090, 24GB):**
| Component | Estimated VRAM |
|-----------|----------------|
| Qwen 3 VL 2B (4-bit) | ~1.5 GB |
| VLM activations (3 frames) | ~3-4 GB |
| Action expert (~100M) | ~200 MB |
| Optimizer states | ~400 MB |
| Gradients + activations | ~2 GB |
| **Headroom** | ~16 GB |

With 4-bit quantization and frozen VLM, batch size 8-16 should be feasible.

---

## 1. Data Pipeline

### 1.1 Raw Capture Format (Observed)
- [x] Document episode directory layout:
  - `video.mp4` (screen recording at ~10 fps, 384×384 canvas)
  - `events.jsonl` (JSONL stream of input events with timestamps)
  - `frame_ts.bin` (int64 QPC timestamps per video frame)
  - `meta.json` (recording metadata: fps, canvas size, qpc_freq, offset_ms)
- [x] Capture event schema from `events.jsonl`:
  - Common fields: `ts` (QPC timestamp), `kind`
  - Mouse events: `mousemove` (`x`, `y`), `mouseclick` (`x`, `y`, `buttons` optional)
  - Keyboard events: `keydown`/`keyup` (`vk` virtual key code)
- [x] Specify timestamp alignment:
  - Convert QPC → ms using `qpc_freq` from `meta.json`
  - Apply `offset_ms` before aligning to frame timestamps
  - Note that `frame_ts.bin` may repeat timestamps (duplicate frames)
- [x] Define action space based on captured data:
  - Mouse position in pixel coordinates (0..canvas_w-1, 0..canvas_h-1)
  - Mouse buttons as bitmask or separate binary channels
  - Keyboard as pressed-key set derived from keydown/keyup

### 1.2 Dataset Conversion (Raw → Training)
- [ ] Parse `frame_ts.bin` as little-endian int64 timestamps, map to frame indices.
- [ ] Decode `video.mp4` and build a frame index → image mapping.
- [ ] Align events to frames:
  - For each frame interval `[ts_i, ts_{i+1})`, collect all events.
  - Derive per-frame action labels (mouse position, button changes, key state).
- [ ] Convert event streams into model-friendly actions:
  - Mouse: normalize position/delta to [-1, 1] relative to canvas size.
  - Buttons: map to discrete channels (left/right/middle) or bitmask.
  - Keyboard: map `vk` to fixed vocabulary; maintain pressed-state vector.
- [ ] Optionally resample to fixed fps or drop duplicate frame timestamps.
- [ ] Emit per-episode training samples:
  - Frame triplets (t, t-2, t-4) + action for frame t (or chunk).
  - Persist aligned `frame_idx`, `timestamp_ms`, and raw event list for audits.
- [ ] Generate train/val/test splits and save as HF dataset or shards.

### 1.3 PyTorch Dataset Class
- [ ] Create `MiniworldVLADataset` class that reads processed shards.
- [ ] Implement `__getitem__` returning:
  - `images`: Tensor [num_frames, C, H, W] (e.g., [3, 3, 384, 384])
  - `text_instruction`: str (task description, if available)
  - `states`: Tensor [state_dim] (optional: game state)
  - `action_chunk`: Tensor [chunk_size, action_dim]
- [ ] Implement temporal frame sampling:
  - [ ] Support configurable `num_frames` (default: 3)
  - [ ] Support configurable `frame_skip` (default: 2, meaning frames t, t-2, t-4)
  - [ ] Handle episode boundaries (pad with first frame if needed)
  - [ ] Store frame indices and timestamps for debugging
- [ ] Implement image preprocessing (resize to 384×384, normalize)
- [ ] Implement data augmentation (optional: color jitter, crops—apply consistently across temporal frames)
- [ ] Add caching for faster loading
- [ ] Write unit tests for dataset class

### 1.4 DataLoader Setup
- [ ] Create collate function for batching
- [ ] Handle variable-length text instructions
- [ ] Implement efficient image loading (memory mapping if needed)
- [ ] Test with small batch to verify shapes

---

## 2. Model Architecture

### 2.1 Qwen 3 VL Integration
- [x] Load Qwen 3 VL 2B with transformers library
- [x] Setup 4-bit quantization with bitsandbytes:
  ```python
  from transformers import BitsAndBytesConfig

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
  )
  model = Qwen3VL.from_pretrained(..., quantization_config=bnb_config)
  ```
- [x] Identify intermediate layer for feature extraction (e.g., layer 14 of 28)
- [x] Write hook or modify forward to extract hidden states at target layer
- [x] Freeze all VLM parameters (required for 4-bit)
- [x] Test forward pass with multi-frame input (3 images + text)
- [x] Document hidden dimension size and output shapes
  - Hidden size: 2048
  - Output shape: [batch, seq_len, 2048] where seq_len ≈ 156 per frame at 384×384
- [x] Verify Qwen handles multiple images correctly (native multi-image support)
  - Uses chat template with `<|vision_start|><|image_pad|><|vision_end|>` tokens

### 2.2 Temporal Feature Handling
- [x] Implement `TemporalFeatureExtractor` wrapper:
  ```python
  class TemporalFeatureExtractor:
      def __init__(self, qwen_vl, target_layer=14):
          self.qwen_vl = qwen_vl
          self.target_layer = target_layer

      def forward(self, frames_list, instruction):
          # frames_list: [frame_t_minus_4, frame_t_minus_2, frame_t]
          # Returns: features with temporal dimension preserved
          # Shape: [batch, num_frames * seq_len, hidden_dim]
  ```
- [x] Add temporal position embeddings (optional, to distinguish frame order)
- [x] Test that gradients don't flow back to frozen VLM (verified: VLM has 0 gradients)
- [x] Benchmark memory usage with 1, 2, 3 frames
  - 1 frame: 1.71 GB
  - 2 frames: 1.72 GB
  - 3 frames: 1.72 GB

### 2.2 Projector Modules
- [x] Implement `StateProjector` (Linear: state_dim → hidden_dim)
- [x] Implement `ActionProjector` (Linear: action_dim → expert_hidden_dim)
- [x] Implement `FeatureProjector` (Linear: vlm_hidden_dim → expert_hidden_dim)
- [x] Add LayerNorm where appropriate
- [x] Test projector shapes with dummy inputs

### 2.3 Flow Matching Action Expert
- [x] Implement `FlowMatchingExpert` transformer
  - Config: num_layers, hidden_dim, num_heads, mlp_ratio
- [x] Implement timestep embedding (sinusoidal or learned)
- [x] Implement interleaved attention blocks:
  - [x] Cross-attention layer (queries=actions, keys/values=VLM features)
  - [x] Causal self-attention layer (actions attend to past actions)
- [x] Implement feed-forward blocks (MLP with GELU)
- [x] Implement final projection to action_dim
- [x] Add positional embeddings for action sequence
- [x] Test forward pass: (noisy_actions, vlm_features, timestep) → velocity

### 2.4 Full QwenVLA Model
- [x] Create `QwenVLA` nn.Module combining all components
- [x] Implement forward method for training:
  ```python
  def forward(self, images, text, states, actions, timesteps):
      # Returns predicted velocity for flow matching loss
  ```
- [x] Implement `sample` method for inference:
  ```python
  def sample(self, images, text, states, num_steps=10):
      # Returns action chunk via ODE integration
  ```
- [x] Add config dataclass for all hyperparameters
- [x] Test full model forward pass
- [x] Print model parameter counts (frozen vs trainable)
  - Total: 1,248,126,088 parameters
  - Trainable: 26,563,720 (2.1%)
  - Frozen (VLM): 1,221,562,368 (97.9%)

### 2.5 Visual Token Handling (Optional Optimization)
- [x] Implement `TokenPooler` module (average pool or learned)
- [x] Add flag to enable/disable pooling
- [ ] Benchmark inference time with/without pooling
- [ ] Decide on final approach based on latency

---

## 3. Training Loop

### 3.1 Flow Matching Loss
- [x] Implement `flow_matching_loss` function
  ```python
  def flow_matching_loss(model, images, text, states, actions):
      # Sample timesteps from Beta distribution
      # Compute noisy actions: a_t = t * actions + (1-t) * noise
      # Predict velocity: v = model(images, text, states, a_t, t)
      # Target: u = noise - actions
      # Loss: MSE(v, u)
  ```
- [x] Add optional loss weighting by timestep
- [ ] Test loss computation with dummy data

### 3.2 Optimizer and Scheduler
- [x] Setup AdamW optimizer
  - Separate param groups for action expert vs projectors (if different LR)
- [x] Implement cosine learning rate schedule with warmup
- [x] Add gradient clipping
- [x] Add weight decay (exclude biases and LayerNorm)

### 3.3 Training Script
- [x] Parse config from YAML or command line
- [x] Initialize model, optimizer, scheduler
- [x] Setup wandb/tensorboard logging
- [x] Implement training loop:
  ```python
  for epoch in range(num_epochs):
      for batch in dataloader:
          loss = flow_matching_loss(model, **batch)
          loss.backward()
          optimizer.step()
          scheduler.step()
          log_metrics(loss, lr, ...)
  ```
- [x] Add checkpointing (save every N steps + best model)
- [x] Add gradient accumulation support
- [x] Add mixed precision training (bf16/fp16)
- [x] Add resume from checkpoint functionality

### 3.4 Validation Loop
- [ ] Implement validation loss computation
- [ ] Implement action prediction and comparison to ground truth
- [ ] Add metrics: MSE, action accuracy (for discrete), smoothness
- [ ] Run validation every N steps

### 3.5 Multi-GPU Support (Optional)
- [ ] Add DistributedDataParallel or FSDP support
- [ ] Use HuggingFace Accelerate for easy multi-GPU
- [ ] Test on 2+ GPUs

---

## 4. Inference Loop

### 4.1 ODE Solver for Sampling
- [ ] Implement Euler integration for flow matching:
  ```python
  def euler_sample(model, images, text, states, num_steps=10):
      actions = torch.randn(batch, chunk_size, action_dim)
      dt = 1.0 / num_steps
      for t in reversed(linspace(0, 1, num_steps)):
          velocity = model(images, text, states, actions, t)
          actions = actions - velocity * dt
      return actions
  ```
- [ ] Implement higher-order solver (optional: RK4, Heun)
- [ ] Add CFG (classifier-free guidance) support if using conditioning
- [ ] Benchmark sampling quality vs num_steps

### 4.2 Temporal Feature Caching
The VLM is the bottleneck. At inference time, cache features to avoid re-encoding old frames.

- [ ] Implement `FeatureCache` class:
  ```python
  class FeatureCache:
      def __init__(self, max_frames=3):
          self.cache = deque(maxlen=max_frames)
          self.frame_indices = deque(maxlen=max_frames)
      
      def add(self, frame_idx, features):
          """Add new frame features to cache"""
          self.cache.append(features)
          self.frame_indices.append(frame_idx)
      
      def get_temporal_features(self):
          """Return concatenated features for all cached frames"""
          return torch.cat(list(self.cache), dim=1)
      
      def clear(self):
          """Clear cache at episode boundary"""
          self.cache.clear()
          self.frame_indices.clear()
  ```
- [ ] Implement cached inference loop:
  ```python
  def inference_step(self, new_frame, instruction, cache):
      # Only encode the NEW frame (not all 3)
      new_features = self.vlm_encode(new_frame, instruction)
      cache.add(current_frame_idx, new_features)
      
      # Combine with cached features from previous frames
      temporal_features = cache.get_temporal_features()
      
      # Action expert sees all temporal context
      return self.action_expert.sample(temporal_features)
  ```
- [ ] Handle episode start (cache is empty, need to pad or wait)
- [ ] Add cache warmup strategy (run VLM on first 3 frames before acting)
- [ ] Benchmark: cached inference should be ~3x faster than full multi-frame forward

### 4.3 Action Execution Interface
- [ ] Define abstract `ActionExecutor` class
- [ ] Implement `MiniworldExecutor`:
  - [ ] Convert normalized actions to mouse/keyboard commands
  - [ ] Handle action queue (FIFO)
  - [ ] Implement `step(action)` method
  - [ ] Implement `get_observation()` method
- [ ] Add action smoothing/interpolation (optional)

### 4.4 Synchronous Inference Loop
- [ ] Implement basic inference loop with feature caching:
  ```python
  feature_cache = FeatureCache(max_frames=3)
  
  while not done:
      frame = env.get_observation()
      
      # Cache warmup at episode start
      if len(feature_cache) < 3:
          features = vlm_encode(frame, instruction)
          feature_cache.add(frame_idx, features)
          if len(feature_cache) < 3:
              continue  # Wait until we have enough temporal context
      
      # Normal inference with caching
      action_chunk = model.sample_with_cache(frame, instruction, feature_cache)
      
      for action in action_chunk:
          env.step(action)
          frame_idx += 1
  ```
- [ ] Add FPS limiting/timing control
- [ ] Add episode timeout handling
- [ ] Add success/failure detection (if available from env)
- [ ] Clear feature cache at episode boundaries

### 4.5 Asynchronous Inference Loop (Advanced)
- [ ] Implement `PolicyServer` class:
  - [ ] Load model on GPU
  - [ ] Maintain per-session feature caches
  - [ ] Accept observations via queue/gRPC
  - [ ] Return action chunks asynchronously
- [ ] Implement `RobotClient` class:
  - [ ] Maintain action queue
  - [ ] Send observations when queue below threshold
  - [ ] Execute actions from queue continuously
- [ ] Implement observation similarity filter (avoid redundant inference)
- [ ] Add threshold parameter `g` for queue refill trigger
- [ ] Test async vs sync performance comparison

### 4.6 Real-time Optimizations
- [ ] Add torch.compile() for action expert (VLM may not compile well with 4-bit)
- [ ] Verify 4-bit quantization is active at inference
- [ ] Profile inference breakdown:
  - [ ] VLM encoding time (should be ~100-200ms per frame)
  - [ ] Action expert time (should be ~20-50ms)
  - [ ] With caching, total should be ~150ms instead of ~400ms
- [ ] Implement KV cache reuse within VLM if supported
- [ ] Target latency: <200ms per inference step on RTX 3090

---

## 5. Evaluation

### 5.1 Offline Metrics
- [ ] Action prediction MSE on held-out data
- [ ] Action chunk smoothness (jerk/acceleration metrics)
- [ ] Per-dimension action accuracy

### 5.2 Online Evaluation
- [ ] Define evaluation tasks in Miniworld (e.g., navigate to goal, collect item)
- [ ] Implement success rate metric
- [ ] Implement episode length / time-to-completion metric
- [ ] Run N episodes per task, report mean and std

### 5.3 Ablation Studies (Optional)
- [ ] Compare with/without VLM pretraining (random init action expert)
- [ ] Compare different chunk sizes (10, 25, 50, 100)
- [ ] Compare layer extraction depths
- [ ] Compare pooling vs no pooling
- [ ] Compare temporal context:
  - [ ] 1 frame (no temporal)
  - [ ] 3 frames (default)
  - [ ] 5 frames (more context)
  - [ ] Different frame_skip values (1, 2, 4)
- [ ] Compare quantization impact:
  - [ ] 4-bit vs 8-bit vs bf16 (if memory allows)

---

## 6. Utilities and Infrastructure

### 6.1 Configuration Management
- [ ] Create `config.py` with dataclasses for all configs:
  - ModelConfig
  - TrainingConfig
  - DataConfig
  - InferenceConfig
- [ ] Add YAML loading/saving
- [ ] Add command-line override support (Hydra or argparse)
- [ ] Create default config for RTX 3090:
  ```yaml
  # configs/3090_default.yaml
  model:
    vlm_name: "Qwen/Qwen3-VL-2B"
    quantization: "4bit"
    vlm_layer: 14
    action_expert_layers: 8
    action_expert_dim: 512
    chunk_size: 50
  
  data:
    num_frames: 3
    frame_skip: 2
    image_size: 384
    
  training:
    batch_size: 8
    gradient_accumulation: 2
    learning_rate: 1e-4
    warmup_steps: 100
    max_steps: 100000
    mixed_precision: "bf16"
    
  inference:
    sampling_steps: 10
    use_feature_cache: true
    cache_max_frames: 3
  ```

### 6.2 Memory Management and Troubleshooting
- [ ] Add memory monitoring utilities:
  ```python
  def log_memory_usage(tag=""):
      allocated = torch.cuda.memory_allocated() / 1e9
      reserved = torch.cuda.memory_reserved() / 1e9
      print(f"[{tag}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
  ```
- [ ] Add gradient checkpointing for action expert (if needed)
- [ ] Create memory debugging script to test each component
- [ ] Document fallback options if OOM:
  - Reduce batch size
  - Reduce num_frames from 3 to 2
  - Reduce image_size from 384 to 256
  - Use more aggressive gradient accumulation
  - Pool visual tokens (last resort)

### 6.3 Logging and Visualization
- [ ] Setup wandb project
- [ ] Log training curves (loss, LR)
- [ ] Log sample action predictions vs ground truth
- [ ] Log videos of inference rollouts
- [ ] Log memory usage over training
- [ ] Add attention visualization (optional)

### 6.3 Model Export
- [ ] Save model in HuggingFace format
- [ ] Add model card with usage instructions
- [ ] Export to ONNX (optional, for deployment)

### 6.4 Testing
- [ ] Unit tests for dataset class
- [ ] Unit tests for model components
- [ ] Integration test: train for 10 steps, run inference
- [ ] CI pipeline with pytest

---

## 7. Project Structure

```
qwen-vla/
├── configs/
│   ├── model.yaml
│   ├── training.yaml
│   ├── inference.yaml
│   └── 3090_default.yaml       # Full config for RTX 3090
├── data/
│   ├── recordings/              # Raw recordings from Miniworld
│   ├── processed/               # Converted to training format
│   └── dataset.py               # PyTorch dataset class
├── models/
│   ├── qwen_vla.py              # Main model class
│   ├── action_expert.py         # Flow matching transformer
│   ├── projectors.py            # Linear projectors
│   ├── temporal_encoder.py      # Temporal feature extraction
│   ├── quantization.py          # 4-bit loading utilities
│   └── utils.py                 # Model utilities
├── training/
│   ├── train.py                 # Training script
│   ├── loss.py                  # Flow matching loss
│   └── scheduler.py             # LR scheduler
├── inference/
│   ├── sampler.py               # ODE sampling
│   ├── feature_cache.py         # Temporal feature caching
│   ├── sync_loop.py             # Synchronous inference
│   ├── async_loop.py            # Asynchronous inference
│   └── executor.py              # Action execution interface
├── evaluation/
│   ├── evaluate.py              # Evaluation script
│   └── metrics.py               # Metric computation
├── utils/
│   ├── config.py                # Configuration dataclasses
│   ├── logging.py               # Logging setup
│   ├── memory.py                # Memory monitoring utilities
│   └── visualization.py         # Plotting utilities
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_memory.py           # Memory usage tests
├── scripts/
│   ├── convert_recordings.py
│   ├── run_evaluation.py
│   └── memory_check.py          # Verify 3090 fits
├── requirements.txt
├── setup.py
└── README.md
```

---

## 8. Milestones

### Milestone 0: Environment Setup & Memory Verification ✓
- [ ] Install dependencies (transformers, bitsandbytes, torch)
- [ ] Verify CUDA and bitsandbytes 4-bit works
- [ ] Run memory check script:
  ```python
  # scripts/memory_check.py
  # Load Qwen 3 VL 2B in 4-bit
  # Forward pass with 3 frames at 384x384
  # Print memory usage
  # Verify headroom exists for training
  ```
- [ ] Confirm batch_size=8 fits in 24GB with 3 frames
- [ ] Document actual memory usage

### Milestone 1: Data Pipeline ✓
- [ ] Recording schema defined
- [ ] Converter script working
- [ ] Dataset class tested with DataLoader
- [ ] Temporal frame sampling verified

### Milestone 2: Model Forward Pass ✓
- [ ] Qwen VL loading and feature extraction
- [ ] Action expert forward pass
- [ ] Full model trainable params identified

### Milestone 3: Training Loop ✓
- [ ] Loss function implemented
- [ ] Single batch overfitting test passes
- [ ] Checkpointing working

### Milestone 4: Basic Inference ✓
- [ ] Sampling working
- [ ] Sync inference loop running in Miniworld
- [ ] Model produces coherent (not random) actions

### Milestone 5: Evaluation ✓
- [ ] Online evaluation on held-out tasks
- [ ] Success rate > random baseline

### Milestone 6: Optimization (Optional) ✓
- [ ] Async inference implemented
- [ ] Quantization applied
- [ ] Latency targets met

---

## Quick Start Commands (To Be Filled)

```bash
# 0. Verify memory fits on 3090
python scripts/memory_check.py --num_frames 3 --image_size 384 --batch_size 8

# 1. Convert recordings to dataset
python scripts/convert_recordings.py --input data/recordings --output data/processed

# 2. Train model
python training/train.py --config configs/3090_default.yaml

# 3. Run inference (with feature caching)
python inference/sync_loop.py \
    --checkpoint checkpoints/best.pt \
    --task "navigate to the red door" \
    --use_cache

# 4. Evaluate
python evaluation/evaluate.py --checkpoint checkpoints/best.pt --num_episodes 100
```

---

## Important Notes

### Training vs Inference Temporal Handling

**Training time:** Full multi-frame forward pass through VLM
- All 3 frames passed together to Qwen VL
- VLM attention can learn cross-frame relationships
- More expensive but that's okay for training

**Inference time:** Feature caching for efficiency
- Only encode 1 new frame per step
- Concatenate with cached features from previous frames
- ~3x faster than re-encoding all frames

This asymmetry is intentional—we want the model to learn from full temporal context, but deploy efficiently.

### Why Frozen VLM + 4-bit?

1. **Memory**: 4-bit cuts VLM from ~5GB to ~1.5GB
2. **Stability**: Frozen VLM means no risk of catastrophic forgetting
3. **Speed**: Fewer gradients to compute
4. **Sufficiency**: VLM already has good visual features; we just need to learn the action mapping

If performance is lacking, consider LoRA on attention layers (will need Unsloth or similar for memory efficiency).

---

## References

- [SmolVLA Paper](https://arxiv.org/abs/2506.01844)
- [Qwen 2.5 VL Docs](https://huggingface.co/Qwen/Qwen2.5-VL-2B-Instruct)
- [LeRobot](https://github.com/huggingface/lerobot)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Miniworld](https://github.com/Farama-Foundation/Miniworld)
