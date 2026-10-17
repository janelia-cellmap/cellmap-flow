export const count = (n: number) => n.toLocaleString("en-US");
export const plural = (n: number, word: string) => `${count(n)} ${word}${n === 1 ? "" : "s"}`;

/** The line a run ends on; `extra` follows its loss. */
export function epochSummary(losses: readonly number[], epochs: number, extra = ""): string {
  const finalLoss = losses.at(-1) ?? null;
  return losses.length === epochs
    ? `Trained ${plural(losses.length, "epoch")}; final loss ${finalLoss!.toFixed(6)}${extra}`
    : `Stopped after ${losses.length} of ${epochs} epochs${finalLoss === null ? "" : `; loss ${finalLoss.toFixed(6)}${extra}`}`;
}
