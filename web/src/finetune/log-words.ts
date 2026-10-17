// The words the two finetune runs' logs share, the head's (run.ts) and the
// whole network's (full/run-full.ts): counts with thousands separators,
// plurals, and the line a run ends on.

/** `n` with thousands separators: "12,345". */
export const count = (n: number): string => n.toLocaleString("en-US");

/** `n` and `word`, plural unless `n` is 1: "1 tile", "12,345 voxels". */
export const plural = (n: number, word: string): string => `${count(n)} ${word}${n === 1 ? "" : "s"}`;

/** The line a run ends on, from its per-epoch `losses` and the `epochs` it
 * was to train: "Trained 20 epochs; final loss 0.011300" when it trained
 * them all, else "Stopped after 3 of 20 epochs; loss 0.012345" (without a
 * loss when no epoch finished). `extra` follows the loss (the head's
 * supervised term, when distillation adds another). */
export function epochSummary(losses: readonly number[], epochs: number, extra = ""): string {
  const finalLoss = losses.length ? losses[losses.length - 1] : null;
  return losses.length === epochs
    ? `Trained ${plural(losses.length, "epoch")}; final loss ${finalLoss!.toFixed(6)}${extra}`
    : `Stopped after ${losses.length} of ${epochs} epochs${finalLoss === null ? "" : `; loss ${finalLoss.toFixed(6)}${extra}`}`;
}
