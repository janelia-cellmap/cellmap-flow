# Handoff: nuc finetuning outputs all-0, no learning

## MOST LIKELY PRIMARY BUG (found last, read this first): double sigmoid, and there's ALREADY a detector for this that's producing a false negative
IMPORTANT CORRECTION to an earlier version of this section: I first thought
this codebase had no handling at all for "model already outputs
probabilities." That was wrong — `lora_trainer.py` already has a built-in
sigmoid detector (`_get_cached_model_has_sigmoid`/`_cache_model_has_sigmoid`/
`_apply_probability_output_mode`, around lines 424-571, committed
2026-04-13/2026-05-12, long predates this session). It's just not firing
correctly for this model. Read this whole section, not just the first
paragraph, before doing anything.

**Step 1 — confirmed the model's raw output really is already [0,1]:**
Verified directly, using the *exact* graph format used during actual LoRA
training (`torch.export`/`.pt2`, loaded via `torch.export.load(...).module()`
— confirmed to give IDENTICAL outputs to the `.ts` serving copy), on real
correction-patch raw data:
```
model.ts and model.pt2 give IDENTICAL values:
chunk_36_14: min=5.2e-09, max=0.99994, mean=0.514
chunk_36_15: min=5.1e-10, max=0.287,   mean=0.00087
chunk_36_16: min=2.9e-12, max=0.00226, mean=0.000021
chunk_36_17: min=6.4e-11, max=0.99783, mean=0.037
```
This model has its own final activation already baked into the exported
graph (sigmoid-like, [0,1] — not tanh's [-1,1]). It is NOT raw unbounded
logits.

**Step 2 — checked whether the existing detector caught this in the actual
training logs. It didn't; it never even logged anything about it:**
`grep -ni sigmoid` on both real nuc training logs from today
(`.../20260805_165802/runs/nuc_distance_32nm_20260805_170919/training_log.txt`
and `.../20260805_182542/runs/.../training_log.txt`) returns **zero
matches** — not the success message, not the cached-hit message, not even
the exception-handler's warning message.

**Step 3 — that silence is actually fully diagnostic given the code's own
logic** (see `train()` around line 527-571): there are exactly three
possible log outcomes — `"Detected built-in sigmoid..."` (probe ran, found
True), `"Using cached built-in sigmoid detection"` (cache hit, True), or
`"WARNING: Sigmoid probe failed (...)"` (exception). None of these appear.
The ONLY code path that produces total silence is: the probe ran once,
computed `model_has_sigmoid = False`, cached it (`_cache_model_has_sigmoid`
is called unconditionally after a successful probe, regardless of the
result) — and the code only logs when the cached value is `True`, so a
cached `False` is silent forever after, across every restart within that
same process (confirmed 4 restarts in the first log, all silent).

**Step 4 — CONFIRMED on real GPU hardware (not just hypothesized).** Ran
the exact probe logic on an actual A100 node via
`ssh submit` + `bsub -P cellmap -n 12 -gpu 'num=1' -q gpu_a100 -Is ...`
(per [[feedback_use_submit_host]] — login node has no LSF binaries, and no
working CUDA in the interactive dev shell used for the rest of this doc):
```
Exact probe replication (randn*100, fp16 autocast):
  min/max: nan nan | has NaN: True | min>=0 and max<=1 ? False   <- the false negative, reproduced exactly

Control (same extreme input, fp32, no autocast):
  min/max: 0.0 1.0 | has NaN: False | check -> True              <- confirms model DOES have a real built-in sigmoid

Control (realistic-scale input, fp16 autocast):
  min/max: 1.0 1.0 | has NaN: False | check -> True               <- confirms it's the extreme x100 input specifically, not fp16 itself
```
So: the probe (`lora_trainer.py:541-560`) feeds `torch.randn(shape)*100` —
extreme, wildly out-of-distribution noise — through the model under real
FP16 `autocast` on the GPU. This genuinely overflows to NaN before reaching
the final sigmoid (`sigmoid(NaN) = NaN`, `NaN >= 0` is `False` in IEEE754),
so `(min>=0) & (max<=1)` silently evaluates `False` instead of raising —
a real, reproduced false negative, NOT a sign the model lacks a sigmoid
(the fp32 control proves it has one) and NOT a general fp16 problem (the
realistic-scale fp16 control works fine). This is the *exact same class*
of fp16-overflow bug already found and fixed in `DiceLoss`/`MarginLoss`
earlier this session, just biting a different, older piece of code.

So: this makes the training *objective itself* wrong for this model from
the very first batch (real confident predictions get compressed by a
spurious second sigmoid into a narrow ~0.5-0.73 band), independent of and
probably more impactful than the rank/LR/margin-saturation/weight-decay/
fp16-in-losses issues found earlier (those are all still real and worth
keeping fixed — they make training *worse* — but this makes the objective
wrong regardless of those).

**FIXED and verified on real GPU hardware.** Hardened the existing probe in
`lora_trainer.py` (`train()`, the `else:` branch around line 537 that runs
when there's no cache hit): the probe forward pass now always runs in fp32
(`autocast('cuda', enabled=False)`, `probe_extreme.float()`) regardless of
`use_mixed_precision` — it's a one-off diagnostic call, cost doesn't
matter — and explicitly checks `torch.isfinite(probe_out).all()` before
trusting the bounded-output check. A non-finite result now logs a distinct
warning ("...inconclusive, assuming raw logits output.") and is NOT cached,
instead of silently caching a confident-looking wrong `False`. Verified via
`python3 -m py_compile` and by rerunning the exact fixed probe logic on a
real A100 node (`bsub -P cellmap -n 12 -gpu 'num=1' -q gpu_a100 -Is ...`):
```
probe_out dtype: torch.float32
is finite: True
min/max: 0.0 1.0
model_has_sigmoid (fixed probe): True
```
Correctly detects the built-in sigmoid now. This fix only changes behavior
for models whose probe previously produced a false-negative-via-NaN — it
doesn't touch the `except Exception` path or change caching behavior for
models that were already being detected correctly.

Not yet done: the trainer only re-probes once per process lifetime (result
cached on the model object, reused across restarts) — this fix prevents a
*bad* value from ever being cached, but the nuc job's *existing* cache
(from before this fix) may still hold a stale wrong `False` if the same
model object/process is reused; a genuinely fresh training run (new
process) will re-probe correctly. Also still worth doing: rerun the same
empirical check (base model raw output range, real vs. extreme-input probe
under fp16) against `mito_aff_unet_setup_16` to confirm it genuinely
outputs unbounded logits (not secretly [0,1] too) — that run trained
successfully with sigmoid applied, so if it turns out to have the same
false-negative problem, the "some models need sigmoid, some don't, and
detection usually works" theory needs rethinking. And separately: rerun
nuc finetuning from scratch with this fix in place to confirm it actually
learns something useful now that the double-sigmoid should be gone.

## Symptom
LoRA finetuning runs for `nuc_distance_32nm` (session
`corrections/script_test/20260805_165802`) don't learn — loss doesn't
meaningfully improve, and the served finetuned model outputs look like solid
0 (black), regardless of learning rate tried (0.01, 0.001, 0.0001 all show
the same behavior).

## Root cause (confirmed from training_log.txt diagnostics)

1. **Gradient dies after epoch 1.** The `[diag] gradient flow` lines in
   `lora_trainer.py`'s `_train_epoch` show `38/38 trainable params got
   nonzero grad` in epoch 1, then `0/38` (all dead) for every epoch after,
   in every restart. A learning rate can't fix a gradient that is exactly
   `0.000e+00` — that's why "regardless of LR" holds.

2. **Why gradient dies: `margin` loss + too much capacity + too high LR for
   this target.** `MarginLoss` (`lora_trainer.py`) intentionally gives zero
   gradient once a voxel's prediction is confidently on the correct side of
   the margin — that's by design, for scribble-style annotations, so you
   don't over-punish regions already predicted correctly. But this run used:
   - `output_type=binary` (single channel) — the nuc model only has one
     output channel, so there's only one binary target per voxel to satisfy.
   - `lora_r=64` — 4x the adapter capacity of the last working run.
   - LR up to `0.01` — large steps.
   With only 100 correction patches (confirmed real, all correctly tagged
   `nuc_distance_32nm` / `nucleus`, from today's Gemini annotation of
   `jrc_axolotl-heart-1`, volume `vol-5ddb9619-20260805-165802` — NOT the
   old relabeled mito data, that lives in a separate
   `nuc_distance_32nm_test/corrections` folder and was never used here),
   this combination is enough to trivially satisfy the margin on all 100
   patches within epoch 1. After that, margin loss reports (correctly)
   "nothing left to correct" — hence permanent zero gradient.

   **Comparison to a run that worked**: the last successful
   `mito_aff_unet_setup_16` run (`corrections/script_test/20260801_005012`)
   also used `margin` loss and the same 100-correction count, but stayed
   `38/38` alive for all 20 epochs (loss `0.377 → 0.017`) because it used
   `output_type=affinities` (3 independent x/y/z channels per voxel = 3x the
   signal), `lora_r=16` (1/4 the capacity), and `lr=0.0001` (10-100x
   smaller). None of those runs hit the "solved everything" wall within 20
   epochs; today's nuc config blew past it in epoch 1.

3. **`AdamW` has no `weight_decay` override anywhere in the pipeline**
   (`lora_trainer.py:276` — confirmed via grep across
   `lora_trainer.py`/`finetune_cli.py`/`training.py`), so PyTorch's default
   `weight_decay=0.01` (decoupled) applies unconditionally. This is a
   separate, always-present issue: once gradient is zero, weight decay is
   the *only* thing still touching the adapter, and it pulls weights toward
   0 every step regardless of gradient. This matches the diagnostic
   `param_delta` shrinking geometrically epoch-over-epoch even while
   `mean|grad|=0.000e+00` (`4.2e-3 → 9.3e-4 → 2.7e-4 → 8.0e-5 → 2.2e-5 →
   5.8e-6 ...`). It didn't visibly hurt the mito run (real gradient kept
   flowing and dominated), but for nuc it quietly erased epoch 1's progress
   over the following ~19 epochs — so whichever checkpoint the "best loss"
   logic picks (e.g. `Loaded best checkpoint (epoch 6, ...)`) is already
   partway back to a no-op adapter, i.e. ≈ the frozen base checkpoint
   (`salivary_20250806_nuc_mouse_distance_32nm_342000`, pretrained on mouse
   salivary gland, being run zero-shot on axolotl heart — plausible that it
   just doesn't fire confidently there). The live-serving postprocessor then
   clips raw output to `[0,1]` and multiplies by 255 for display, so a
   low-confidence/near-zero raw prediction shows as flat black — the "all
   0s" you're seeing.

## Update: switching to `combined` loss surfaced a second, separate bug
Retried with `loss_type=combined`, `lora_r=8`, `lr=1e-4` — still got
degenerate (~1e-20) outputs. Log shows why immediately:
```
[diag] ...lora_B...: mean|grad|=inf (over 13 batches), mean|param_delta|=4.519e-04 this epoch
Epoch 1/20 - Loss: 0.733674 - Best: inf
```
The gradient is literally `inf` in epoch 1. Root cause: `DiceLoss.forward()`
(`lora_trainer.py`, used by both `dice` and `combined`) summed `pred`/
`target`/`pred*target` over the flattened spatial dimension **inside the
`autocast` fp16 block**. A patch is 56x56x56 = 175,616 voxels — summing that
many values in fp16 (max representable ~65504) overflows straight to `inf`,
and backprop through that overflowed sum poisons the gradient. One `inf`
update blows the LoRA weights to extremes, driving the sigmoid output to a
corner (`sigmoid(-46) ≈ 1e-20` — matches what was observed), after which
gradient goes dead for the rest of training (confirmed in log: 2/38 alive
epoch 2, 0/38 from epoch 3 on) — regardless of `combined` vs `margin`, this
was never an LR or rank issue for this particular failure. The exact same
fp16-overflow class of bug was already known and fixed elsewhere in this
file (the distillation loss explicitly casts to float32 before its masked
sum, with a comment about avoiding this), but the fix was never applied to
`DiceLoss`.

## Update: margin loss had the same fp16-overflow bug too
Went back and checked `MarginLoss.forward` — it has the identical hazard,
actually worse: its `.sum()` calls (no `dim=` argument) reduce over the
*entire batch* at once (`batch_size(8) x 56^3 voxels ~= 1.4M elements` in
fp16, vs. Dice's per-sample-per-channel reduction). This isn't
hypothetical — the very first `margin`-loss attempt in this job hit exactly
this:
```
NaN/Inf in student pred!
NaN/Inf supervised_loss: nan
NaN/Inf loss at epoch 1, batch 8. Aborting epoch.
WARNING: NaN loss at epoch 1 under FP16 — falling back to FP32 and restarting training.
```
After that fallback, the rest of *that* run was numerically safe (fp16
disabled), so its epoch-2+ zero-gradient behavior is genuinely the
"hinge already satisfied on a small/high-capacity/high-LR setup" story
above. But every later restart (LR 0.01, then 0.0001, then r=8) spawned a
fresh process with `Mixed precision: True` again — so every restart was
freshly exposed to the same overflow risk in `MarginLoss`, whether or not
it happened to hard-crash visibly that time. So margin loss had two
compounding problems, not one: this fp16 sum-overflow bug, plus the
legitimate hinge-loss-saturates-a-small-dataset issue.

## Fixes applied (this session)
- **`DiceLoss.forward`** (`cellmap_flow/finetune/lora_trainer.py`): cast
  `pred`/`target` to `.float()` before the intersection/union sums, so the
  reduction happens in fp32 and can't silently overflow under autocast.
- **`MarginLoss.forward`** (same file): same fix — cast `pred`/`target`/
  `mask` to `.float()` before all the batch-wide `.sum()` calls, for the
  same reason.
- **`AdamW` weight decay**: added `weight_decay=0.0` to both `AdamW`
  constructions in `lora_trainer.py` (initial `__init__` and the
  `_reset_for_restart` path). PyTorch's default (`0.01`, decoupled) was
  previously unset and applied silently everywhere, actively eroding LoRA
  weights toward zero on any step with small/zero gradient — now explicit
  and off.
- Both changes verified with `python3 -m py_compile`. Not yet
  runtime-tested — next step is to rerun the nuc training job and confirm
  gradient stays finite/nonzero past epoch 1.

## Correction: the "one inf update corrupts everything" story was overclaimed
Reproduced the Dice/Margin reduction in isolation with a toy Conv3d +
realistic sparse mask, under real autocast+GradScaler: PyTorch's `.sum()` on
an fp16 input tensor already auto-promotes its *output* to float32 — it did
not overflow in isolation the way originally assumed. Also confirmed
`GradScaler` is specifically designed to detect an inf/nan gradient and
**skip** that optimizer step entirely (not apply a corrupting update) — so a
single bad batch shouldn't be able to blow up the weights the way described
above. The float32 casts added to `DiceLoss`/`MarginLoss` are still correct
and worth keeping (defensive, matches the existing pattern already used by
the distillation loss in this file), but they should not be presented as a
fully-confirmed fix for the exact `mean|grad|=inf` observed — the real
~818M-parameter UNet has far more compounding fp16 operations across its
many layers than a toy single-conv repro can capture. If the inf/degenerate
output recurs after these fixes, next step should be live instrumentation
(log raw pred min/max per batch, not just an `isfinite` check) rather than
further static analysis.

Swept the rest of `cellmap_flow/finetune/*.py` for the same class of bug
(unmasked large-tensor `.sum()`/`.mean()` under autocast without a float32
cast) — nothing else in this module has it.

## Retracted: the "sparse/imbalanced annotations" finding was a measurement artifact
Initially measured per-z-slice annotation density in the 100 correction
patches and concluded the data was extremely sparse (~1.5-1.8% of voxels
per patch) and that half the patches had zero foreground. **This was
wrong.** Traced the actual pipeline end to end (Gemini's dense mask ->
`write_ai_mask_to_minio` in `cellmap_flow/dashboard/routes/finetune/overlay.py`
-> `extract_correction_from_chunk` in `cellmap_flow/dashboard/finetune_utils.py`)
and verified directly: each click paints one full, dense 2D plane into the
3D correction cube, with no thinning/downsampling anywhere in that path.
The earlier density measurement was bugged — it sliced along the z-axis to
check density, but the annotated plane in the inspected chunk is actually
oriented along **x** (depends on which neuroglancer viewport the click was
made in — see `depth_axis_from_layout`,
`cellmap_flow/dashboard/routes/finetune/ai_annotate.py:136-152`). Slicing
across a dense plane on the wrong axis cuts through it edge-on, producing
a thin ~56-pixel line and giving a false impression of sparsity.

Correctly inspected, the annotated plane in each chunk is fully dense, and
aggregated across all 100 correction chunks it's **~30% foreground / ~70%
background** — a real, reasonably balanced, sufficient dataset. The
remaining ~98% of each 3D cube being "unannotated" is intentional (one
annotated plane per click, `mask_unannotated` correctly ignores the rest),
not a data quality problem. **This is not a contributing factor** to the
training issues found above — the real, still-standing causes are the
`margin`-loss gradient-death mechanism (rank/LR/single-channel combo), the
`AdamW` weight-decay erosion, and the fp16-precision hardening in
`DiceLoss`/`MarginLoss`, all described earlier in this doc.

## Investigated and resolved: apparent "inverted polarity" was domain shift, not a bug
User noticed the served output looked lower inside the nucleus than outside.
Tested the frozen base model (`salivary_20250806_nuc_mouse_distance_32nm_342000/model.ts`,
no LoRA) directly on 29 real correction patches, comparing its raw output at
foreground(nucleus)-labeled voxels vs. background-labeled voxels:
```
patches: 29, fg>bg in only 7/29
Overall mean raw output on FOREGROUND(nucleus) voxels = 0.0514
Overall mean raw output on BACKGROUND voxels          = 0.2404
```
This looked like the base model's native polarity might be inverted relative
to what the training loss assumes (`BinaryTargetTransform`/`MarginLoss`/
`DiceLoss`/BCE all push target=1(nucleus) toward `sigmoid(pred)->1`). **User
clarified this is not a design/code mismatch**: the intended/correct
convention genuinely is positive-inside/negative-outside (matching what the
training code already assumes) — the base model just performs poorly here
because it's out-of-domain (pretrained on mouse salivary gland, being run
zero-shot on axolotl heart tissue it's never seen). The `~1e-10`-`~1e-14`
values actually being observed come from the served/heavily-restarted
checkpoint (already degraded by the gradient-death/weight-decay issues
found earlier), a different regime entirely from the 0.05-0.24 raw
base-model measurement above. So: no polarity bug, no target-inversion fix
needed. The fg/bg gap measured is a real but expected symptom of domain
shift, consistent with the very first hypothesis in this doc (base model
mismatch with axolotl heart data) — not something to fix in
`lora_trainer.py`. **No `invert_target` code change was made** (correctly
stopped before implementing once this was clarified).

## New finding: most correction patches don't show the nucleus boundary at all
User hypothesized that distance-transform-style prediction needs enough
spatial context to be meaningful (the network needs to "see" how far the
boundary is). Checked the physical scale first: the raw input patch is
178 voxels x 32nm = ~5.7 um per side, while nucleus is documented as
5-15 um in diameter (see the AI-annotate prompt in
`vol-5ddb9619-20260805-165802.zarr`'s `.zattrs`) — so a nucleus can be
larger than the entire input patch, meaning a click deep inside a large
nucleus (or deep in background) may have no boundary visible anywhere in
the field of view.

Verified directly against the 100 real correction patches: for each,
checked whether the annotated 2D plane contains both labels (a visible
boundary) or only one:
```
patches with BOTH fg and bg in the annotated plane (boundary visible): 29
patches with only ONE label (no boundary visible in the annotated plane): 71
```
**71 of 100 patches show only one label** — the click landed either fully
inside a nucleus or fully in background, with no boundary in view.

Nuance: since this finetuning trains on a **binary** target (nucleus
yes/no via `BinaryTargetTransform`/`mask_unannotated`), not literal distance
regression, the network doesn't strictly need to see the boundary to learn
local nucleus-vs-background texture — so this isn't quite "can't learn at
all" the way true distance regression would be blocked by insufficient
context. But it does mean 71% of training examples are "easy," single-class
cases that get satisfied (sigmoid saturates to exact 0/1, confirmed earlier)
almost immediately, which is consistent with and likely a direct
contributor to the fast gradient death already measured. The 29
boundary-containing patches are the actually-informative ones (they teach
where the edge precisely is) and are a minority of the data. Likely
practical improvement: prioritize annotating boundary-adjacent regions
(clicks near the actual nucleus edge) rather than deep interior/background,
to get more informative signal per patch.

## Not yet done
- Dashboard defaults for `output_type=binary` (single-channel) finetuning
  still default to `margin` loss when sparse annotations are detected
  (`cellmap_flow/dashboard/routes/finetune/training.py:155-165`), and there
  are no rank/LR defaults tuned for this case. Even with the DiceLoss/AdamW
  fixes above, `margin` loss can still self-extinguish gradient early on a
  small (100-patch), high-capacity (`r=64`) adapter — see the earlier
  section of this doc. Worth revisiting whether to default to
  `combined`/`bce` + lower rank/LR for binary output types.
- Optional follow-up: instrument/inspect raw (pre-clip) prediction values
  from a served checkpoint to directly confirm the "clipped to black"
  display explanation from the original all-0s report.
- Optional follow-up: add live per-batch instrumentation (raw pred
  min/max/mean, not just isfinite) if the inf-gradient/degenerate-output
  issue recurs after the DiceLoss/MarginLoss/AdamW fixes, to pin down the
  exact mechanism in the real model rather than inferring from a toy repro.
- (Removed: "need more/denser annotations" — retracted, see above. The
  existing 100 patches are a properly dense, reasonably balanced dataset;
  this is not believed to be a limiting factor.)

## Also still outstanding from earlier in this session (unrelated to the above)
- `cellmap_flow/finetune/finetune_cli.py` still has the TorchScript-fallback
  try/except added early in this session (wraps `cellmap_model.train()`,
  falls back to `cellmap_model.ts_model` on failure). This fallback is
  known-broken for LoRA (TorchScript modules can't be PEFT-wrapped) and was
  never reverted or converted to fail loudly. Latent risk for other models
  that hit the same `torch.export` schema-version error.
- The "Update Prompt for Active Volume" fix (new
  `/api/finetune/ai-annotate/update-prompt` endpoint + dashboard button, so
  editing the Gemini prompt textbox after volume creation actually takes
  effect on the next Shift+G) is implemented and `py_compile`-verified but
  has not been runtime-tested in a live dashboard session yet.
