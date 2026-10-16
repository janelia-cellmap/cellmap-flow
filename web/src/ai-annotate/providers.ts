// The providers the browser mode offers, by id, in the order the panel lists
// them: what config.py's `providers` map is on the server, fixed here since
// the browser has no config file. The routes (backend/routes/ai-annotate.ts)
// look a provider up by the id the panel sends and show each one's `info`.
import { createFakeProvider } from "./fake";
import { createGeminiProvider, type GeminiDeps } from "./gemini";
import type { Provider } from "./types";

/** What the providers are built over: the Gemini ones' deps (fetch, the
 * page's ImageCodec, how retries wait, a clock); the fake needs nothing. */
export type ProviderDeps = GeminiDeps;

/** The provider the panel starts on (config.py default_provider: the first). */
export const DEFAULT_PROVIDER = "gemini";

/** gemini (the Gemini API with a key), vertex (Vertex AI with a project and
 * an access token) and fake (runs here), keyed by id, in that order. */
export function createProviders(deps: ProviderDeps): Map<string, Provider> {
  const providers = [createGeminiProvider("gemini_api", deps), createGeminiProvider("vertex_gemini", deps), createFakeProvider()];
  return new Map(providers.map((provider) => [provider.info.id, provider]));
}
