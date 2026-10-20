// PUT /api/pipeline (routes/pipeline.py): set the chain, and redraw the
// viewer's prediction layers through it.
//
// The body is {input_norm: [step], postprocess: [step]}, each step
// {name, ...parameters} with the parameters as the form sends them
// (strings). Every step must be one the browser runs (src/infer/norms.ts and
// post.ts say which); a step it does not is a 400 {success: false, error}
// that changes nothing, as a step Python's registry does not know is. The
// answer is {success: true, pipeline: {input_norm, postprocess}, digest,
// layers}: the chain as kept, its digest (which names the layers' source),
// and the prediction layers drawn. Without a dataset the chain is kept and
// nothing is drawn (layers: []), as the Python route does without a viewer.
import type { Chain, Step } from "../../contracts";
import { SUPPORTED_NORMALIZERS } from "../../infer/norms";
import { SUPPORTED_POSTPROCESSORS } from "../../infer/post";
import type { Router } from "../router";
import { badRequest, json, objectBody } from "../router";
import { blobFor, setChain } from "../serving";

/** A chain from a request body, or the reason it is not one. The steps are
 * kept as given (values are not coerced, like PipelineSpec), only checked:
 * each is an object naming a step the browser can run. */
export function readChain(body: Record<string, unknown>): { chain: Chain } | { error: string } {
  const chains: Partial<Chain> = {};
  for (const [key, supported, kind] of [
    ["input_norm", SUPPORTED_NORMALIZERS, "normalizer"],
    ["postprocess", SUPPORTED_POSTPROCESSORS, "postprocessor"],
  ] as const) {
    const steps = body[key];
    if (!Array.isArray(steps)) return { error: `${key}: expected a list of steps` };
    for (const [i, step] of steps.entries()) {
      if (!step || typeof step !== "object" || Array.isArray(step)) return { error: `${key}[${i}]: a step is an object with a name` };
      const name = (step as Step).name;
      if (typeof name !== "string" || !name) return { error: `${key}[${i}]: a step needs a name` };
      if (!supported.includes(name)) {
        return {
          error: `${name} is not available in the browser. The ${kind}s it can run are ${supported.join(", ")}; the others need a server.`,
        };
      }
    }
    chains[key] = steps as Step[];
  }
  return { chain: { input_norm: chains.input_norm!, postprocess: chains.postprocess! } };
}

export function pipelineRoutes(router: Router): void {
  router.put("/api/pipeline", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const read = readChain(body);
    if ("error" in read) return badRequest(read.error);
    const chain = read.chain;
    const [served, { digest }] = await Promise.all([setChain(ctx, chain), blobFor(chain)]);
    return json(200, { success: true, pipeline: { input_norm: chain.input_norm, postprocess: chain.postprocess }, digest, layers: (served ?? []).map((s) => s.entry.name) });
  });
}
