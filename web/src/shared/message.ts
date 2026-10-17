// What a caught error says, in words: an Error's message, or anything else
// thrown (a string, an object) as String() spells it. For the text a
// `catch (e)` puts on a status line, in a log line or in an answer's `error`
// (backend/router.ts re-exports it for the routes).

/** An error's message: `e.message` for an Error, else `String(e)`. */
export function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
