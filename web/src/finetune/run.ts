// One finetune, start to finish: the model's head found in its bytes, its
// features on every painted chunk, the head trained on the painted voxels
// and written into a copy of the model (see types.ts for why the head).
import type { RunFinetune } from "./types";

export const runFinetune: RunFinetune = async (_run) => {
  throw new Error("runFinetune is not implemented yet");
};
