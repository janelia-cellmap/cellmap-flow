export function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
