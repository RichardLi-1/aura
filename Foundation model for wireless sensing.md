# RF-FM: Foundation Model for Wireless Sensing

## Why This Exists

**The idea:** Wi-Fi and radar signals bounce off people. By measuring how the signal changes on its way to a receiver, a computer can tell whether someone is in a room, what they are doing, even how fast they are breathing. No camera needed, so it works in the dark and avoids visual privacy issues.

**The problem:** Models for this are trained from scratch for one room and one device. Move the router, change the room, or swap the person, and accuracy collapses, because the signal reflects the room more than it reflects the human. Every new deployment needs new labeled data and retraining.

**The foundation-model fix:** Do for radio signals what GPT did for text and CLIP did for images.
1. Pretrain one model on a large amount of unlabeled RF data from many rooms and devices, so it learns a general representation of what RF signals look like.
2. Adapt it to a specific task or room with only a few labels (few-shot) or none (zero-shot), instead of retraining from scratch.

**Success looks like:** one pretrained model that works in a new place with little or no new data.

---

## What Already Exists (Related Work)

This idea is not new. Early Wi-Fi sensing foundation models are already published (summarized from abstracts; read the full papers and verify before citing):

| Work | What it does |
|---|---|
| Scale What Counts, Mask What Matters (arXiv 2511.18792) | Masked-autoencoder pretraining on a large pool of Wi-Fi CSI; zero-shot cross-domain evaluation; scaling trends |
| AM-FM (arXiv 2602.11200) | Wi-Fi foundation model combining contrastive learning, masked reconstruction, and physics-informed autocorrelation prediction |
| 6G Radio Foundation Models (arXiv 2411.09996) | ViT pretrained on radio data; tested on CSI activity sensing and spectrogram segmentation |
| CSI-CLIP (arXiv 2502.11965) | Contrastive pretraining that pairs CSI with channel impulse response for MIMO perception tasks |

**Consequence:** masked pretraining on Wi-Fi CSI is a known baseline, not the contribution. This project reproduces it, then targets the gaps below.

## Contribution Targets (The Gaps)

| Gap | Addressed in |
|---|---|
| Generalization to unseen rooms, devices, and users is still weak | Phase 5 |
| RF grounded in text/video/pose so zero-shot activity recognition works | Phase 6 |
| One model across RF types (Wi-Fi, radar, UWB) and larger, more diverse data | Phase 7 |

---

## End-Goal Behavior

The finished system must:

1. **Take in** a short window of RF channel data (Wi-Fi CSI, radar, or UWB).
2. **Output** a fixed-size embedding that represents what is happening in the space.
3. **Predict** downstream tasks from that embedding: presence, activity, motion.
4. **Transfer** to a new room, device, or person using few labels (few-shot) or none (zero-shot).
5. **Run live** in a demo with minimal calibration.

Phases 0-5 are the solo-buildable mini version. Phases 6-8 are the company-level target.

---

## What the Model Knows and Doesn't Know

**Input it receives (and nothing else):**
- Channel measurements over a short time window: amplitude per subcarrier, per antenna, per time step
- Phase per subcarrier and antenna (added in Phase 5)

**Not given to it as input:**

| Missing info | Why it matters |
|---|---|
| Room shape, size, wall layout | Never provided. It only appears indirectly through multipath (see below). |
| Furniture and object positions | Same as above. |
| Router/transmitter and receiver positions | The model does not know the link geometry. |
| Device model and hardware quirks | Appears only as noise (phase offsets, gain changes). |
| Person's identity, height, or body size | Not provided; it can only be inferred from how the signal changes. |
| Absolute location of the person | Not provided; only the signal change is visible. |
| Labels (activity names) during pretraining | Pretraining is self-supervised, so no labels are used. |
| Video, pose, or text at test time | Used only in Phase 6 training to give embeddings meaning. At inference the model sees RF only. |

**The catch:** the model is never told the room shape, but the signal contains it. Walls and furniture create a fixed pattern of reflections (multipath) that dominates the measurement. A model can accidentally learn to recognize *the room* instead of *the human*. This is the main reason models fail in new rooms.

**Requirement:** the model must rely on what changes over time (the person) and not on what stays constant (the room).

**To do this:**
- [ ] Subtract each window's temporal mean (or high-pass filter) to remove the static room pattern
- [ ] Use augmentations that scramble room-specific signal features (amplitude scaling, subcarrier dropout)
- [ ] Hold out entire rooms in evaluation (Phase 5) to prove it is not memorizing them
- [ ] Do not feed room ID, device ID, or user ID as model input

---

## Phase 0: Setup + Literature

**Behavior:** Anyone can clone the repo and run code on a GPU with one command. You know what is already published before building.

**To do this:**
- [ ] Create a GitHub repo `rf-fm` with folders `data/`, `src/`, `notebooks/`, `results/`
- [ ] Add `requirements.txt` (torch, numpy, scipy, scikit-learn, matplotlib, wandb)
- [ ] Get GPU access (Colab, Kaggle, or school cluster)
- [ ] Set one global random seed used by every script
- [ ] Verify with `python -c "import torch; print(torch.cuda.is_available())"` (must print `True`)
- [ ] Read the four papers in the Related Work table
- [ ] Write 5 lines in `results/lit_notes.md`: what each paper does, and what it does not do

---

## Phase 1: Data + Supervised Baseline

**Behavior:** The system loads a CSI dataset and a from-scratch classifier reaches published-level accuracy. This is the number everything else is compared to.

**To do this:**
- [ ] Download UT-HAR or NTU-Fi (the SenseFi repo has loaders)
- [ ] Print the shape of one sample
- [ ] Plot one sample as a heatmap (time x subcarrier)
- [ ] Use the official train / val / test split
- [ ] Train a small CNN from scratch on 100% of labels
- [ ] Save test accuracy and F1 to `results/baseline.md`
- [ ] Compare to the SenseFi paper numbers (should be within a few points)

---

## Phase 2: Preprocessing + Tokenization

**Behavior:** Any raw CSI window is converted to a sequence of patch tokens with a fixed shape, with no NaNs, ready for a transformer.

**To do this:**
- [ ] Normalize amplitude per sample
- [ ] Start with amplitude only (add phase in Phase 5)
- [ ] Slice recordings into fixed-length windows
- [ ] Cut each window into patches (e.g. 10 time steps x 30 subcarriers)
- [ ] Project each patch to a `D`-dim embedding with a linear layer
- [ ] Add positional embeddings
- [ ] Write a unit test: input `[B, T, F]` returns tokens `[B, N, D]`

---

## Phase 3: Reproduce the Published Recipe (Masked Pretraining)

**Behavior:** Without using any labels, the model learns to fill in hidden parts of an RF window, matching the masked-autoencoder approach already published. Reconstructed patches must visibly match the originals. This model is your **published-recipe baseline** for all later phases.

**To do this:**
- [ ] Build a transformer encoder (4-6 layers, dim 128-256)
- [ ] Randomly mask 50-75% of patch tokens
- [ ] Add a light decoder that reconstructs the masked patches
- [ ] Use MSE loss on masked patches only
- [ ] Train on training data with labels ignored
- [ ] Log the loss curve and save checkpoints
- [ ] Plot 5 reconstructions next to the originals
- [ ] Save this checkpoint as `ckpt/mae_baseline.pt`

---

## Phase 4: Label-Efficient Transfer

**Behavior:** With a frozen pretrained encoder and only a linear head, the model must beat the from-scratch baseline when labels are scarce (1% and 10%).

**To do this:**
- [ ] Freeze the encoder
- [ ] Train a linear head on the embeddings (linear probe)
- [ ] Repeat at 1%, 10%, and 100% of labels
- [ ] Run the same label levels for the from-scratch baseline
- [ ] Run full fine-tuning at each label level
- [ ] Plot accuracy vs % labels for all three methods
- [ ] If pretrained does not win at low labels, debug Phase 3 before continuing

---

## Phase 5: Cross-Environment Generalization (Gap 1)

**Behavior:** The model must keep most of its accuracy on a room, device, or person it never saw in pretraining. The drop from in-domain to held-out must be smaller than the drop for both the from-scratch baseline and the Phase 3 masked-only model.

**To do this:**
- [ ] Pick a multi-domain dataset (Widar3.0 has multiple rooms and people)
- [ ] Split by domain: train on some, test on held-out
- [ ] Measure the in-domain vs held-out accuracy gap for the Phase 3 model
- [ ] Add a contrastive objective (two augmented views of one window should have similar embeddings)
- [ ] Use augmentations: noise, time shift, subcarrier dropout, amplitude scaling
- [ ] Add an invariance step: pull together embeddings of the same activity recorded in different rooms
- [ ] Compare masked only vs contrastive only vs both vs both + invariance
- [ ] Add phase features (phase difference between antennas) and re-compare
- [ ] Test few-shot adaptation: fine-tune on 5 samples per class from the new room
- [ ] Save a results table: variant x (in-domain, held-out)

---

## Phase 6: Multimodal Alignment (Gap 2)

**Behavior:** The model must recognize activities it was never given labels for, by matching RF embeddings to text descriptions. Zero-shot accuracy must be clearly above random chance.

**To do this:**
- [ ] Get a dataset with synchronized RF + video or pose (MM-Fi pairs Wi-Fi CSI with pose; also check XRF55; verify availability and license first)
- [ ] Extract pose keypoints from video with a pretrained pose model
- [ ] Add a pose (or video) encoder and a text encoder for activity names
- [ ] Train a CLIP-style loss: matched RF/pose/text pairs close, mismatched pairs far
- [ ] Classify RF windows by nearest text-label embedding (zero-shot)
- [ ] Compare zero-shot vs few-shot vs fully supervised
- [ ] Test zero-shot on a held-out room (combines Gap 1 and Gap 2)

---

## Phase 7: Multi-RF, Foundation-Scale Model (Gap 3)

**Behavior:** One pretrained model must handle multiple datasets, devices, RF types, and tasks, and match or beat task-specific models, especially on unseen environments.

**To do this:**
- [ ] Combine multiple datasets into one pretraining corpus
- [ ] Standardize input format (different antenna counts, subcarriers, sampling rates)
- [ ] Add at least one non-Wi-Fi RF source (mmWave radar or UWB; pick a public dataset and verify its license)
- [ ] Scale model size and data; log loss vs params and data
- [ ] Add a generative objective (autoregressive or diffusion) and compare
- [ ] Define a benchmark of 3+ tasks: presence, activity recognition, gesture/motion
- [ ] Evaluate each task with linear probe, fine-tune, few-shot, and zero-shot
- [ ] Compare to the published models in the Related Work table and from-scratch baselines
- [ ] Run ablations: objectives, patch size, model size, data amount

---

## Phase 8: Demo + Writeup

**Behavior:** The model runs live (or replayed) and shows presence/activity in a new room with minimal calibration. Anyone can reproduce the main results.

**To do this:**
- [ ] Stream CSI (or replay recorded data) into the model and display predictions live
- [ ] Demo it in a room that was not in training
- [ ] Write a paper-style report: intro, related work, method, experiments, limitations
- [ ] State clearly which gap(s) your results address and what the baseline comparisons show
- [ ] Include label-efficiency curves, generalization tables, ablations, and failure cases
- [ ] Add a one-command reproduce script for the main results table
- [ ] Release code and checkpoints

---

## Rules for Every Phase

- Change one thing at a time and log every run
- Always keep a from-scratch baseline and the Phase 3 masked-only baseline to compare against
- Never evaluate on data used in pretraining or tuning
- If a phase's behavior is not met, fix it before moving on
