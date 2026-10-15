// A painted chunk's labels as training targets: the port of
// cellmap_flow/finetune/target_transforms.py (BinaryTargetTransform,
// BroadcastBinaryTargetTransform, AffinityTargetTransform) for the browser,
// checked against the Python classes by web/test/parity/targets.test.ts.
import type { ModelEntry } from "../contracts";
import type { TargetTransform } from "./types";

/** The transform a model's labels train it with: affinities for a model
 * whose channels are `*_aff_z/y/x`, the binary target on every channel otherwise. */
export function targetTransformFor(_entry: ModelEntry): TargetTransform {
  throw new Error("targetTransformFor is not implemented yet");
}
