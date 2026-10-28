// The learning rate's schedule, the server's (cellmap_flow/finetune/trainer/
// loop.py learning_rate_factor), which the browser's three trainers step at:
// the last layer's Adam (head-trainer.ts), and the whole network's in 2D
// (full/adam.ts, for run-full.ts) and in 3D (full/net3d.ts). A run starts
// at the form's rate and eases down along half a cosine to
// LEARNING_RATE_FLOOR of it at its last step: big steps while the model is
// far from the paint, small ones once it is close. A restart is a run of its
// own, and starts at its rate again.
import { PYTHON } from "../generated/python-data";

/** The share of the form's rate a run ends at (cellmap_flow/finetune/options.py LEARNING_RATE_FLOOR). */
export const LEARNING_RATE_FLOOR: number = PYTHON.finetune.learning_rate_floor;

/** The learning rate of optimizer step `step` (0 to `steps` − 1) of a run
 * of `steps` from `rate`: `rate` at the first, `rate` × LEARNING_RATE_FLOOR
 * at the last, half a cosine between; `rate` for a run of one step. A step
 * past the last stays at the floor. The server's twin is loop.py's
 * learning_rate_factor(step, steps), the share of the rate this is. */
export function learningRateAt(rate: number, step: number, steps: number): number {
  if (steps <= 1) return rate;
  const k = Math.min(step, steps - 1);
  return rate * (LEARNING_RATE_FLOOR + ((1 - LEARNING_RATE_FLOOR) * (1 + Math.cos((Math.PI * k) / (steps - 1)))) / 2);
}

/** A learning rate as the form's presets write it: 1e-4, 3e-5, 1.5e-3 (loop.py rate_words). */
export function rateWords(rate: number): string {
  const [mantissa, exponent] = rate.toExponential(2).split("e");
  return `${Number(mantissa)}e${Number(exponent)}`;
}

/** What a run's learning rate does, for its start line: "1e-2, cosine to
 * 1e-3 over 40 steps" (loop.py schedule_words, the server's line). */
export function scheduleWords(rate: number, steps: number): string {
  if (steps <= 1) return rateWords(rate);
  return `${rateWords(rate)}, cosine to ${rateWords(rate * LEARNING_RATE_FLOOR)} over ${steps} steps`;
}
