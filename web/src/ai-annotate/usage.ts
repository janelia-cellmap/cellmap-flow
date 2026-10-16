// A per-browser daily cap on hosted-model calls: the port of
// cellmap_flow/ai_annotate/usage.py. Each call to a hosted model costs
// money, and a stuck key or a runaway loop could make hundreds, so the
// calls one browser makes in a local calendar day are counted against
// DAILY_CALL_LIMIT (types.ts). The server keeps its count in the user's
// home so that a restart does not reset it; here it is this browser's
// localStorage, under USAGE_KEY as `{date: "YYYY-MM-DD", calls}`, which a
// page reload does not reset either. Where storage is blocked the count
// lives for the page's life, so the cap still holds within it.
//
// As in the Python, a call is counted before it is made (checkAndCount), so
// a failed call counts too, since it may still have been billed; one that
// never reached the model (a sign-in or config failure) is refunded.
import type { KeyValueStore } from "../models/listing-cache";
import { AIAnnotateError } from "./types";

/** The localStorage key the day's count lives under. */
export const USAGE_KEY = "cellmap-flow:ai-annotate:usage";

interface Count {
  date: string;
  calls: number;
}

export interface Usage {
  /** Calls counted so far today. */
  callsToday(): number;
  /** Count one call against today's `limit` and return the calls used, this
   * one included; AIAnnotateError("limit") without counting when today's
   * calls already reached it. */
  checkAndCount(limit: number): number;
  /** Take back one call counted today: one refused before it reached the model. Never below zero. */
  refund(): void;
}

/** The message a refused call carries: usage.py's first sentence, and in
 * place of its "raise daily_call_limit in the config", where the browser's
 * cap comes from. */
export function limitMessage(limit: number): string {
  return `The daily limit of ${limit} AI-annotate calls is used up; it resets at midnight. The cap is this browser's own (DAILY_CALL_LIMIT), not the provider's quota.`;
}

/** `date` as the local calendar day, "YYYY-MM-DD" (Python's date.today().isoformat()). */
export function localDay(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function createUsage(storage: KeyValueStore | null, now: () => Date = () => new Date()): Usage {
  // The count when storage has none for today, or cannot be used.
  let fallback: Count = { date: "", calls: 0 };

  function read(today: string): number {
    let text: string | null = null;
    if (storage) {
      try {
        text = storage.getItem(USAGE_KEY);
      } catch {
        text = null;
      }
    }
    if (text) {
      try {
        const data = JSON.parse(text) as Partial<Count>;
        if (data && data.date === today && typeof data.calls === "number" && Number.isInteger(data.calls) && data.calls >= 0) return data.calls;
        return 0; // another day's count, or a record that is not one
      } catch {
        // unreadable: today's count starts from 0, as the Python's does
      }
    }
    return fallback.date === today ? fallback.calls : 0;
  }

  function write(today: string, calls: number): void {
    fallback = { date: today, calls };
    if (!storage) return;
    try {
      storage.setItem(USAGE_KEY, JSON.stringify({ date: today, calls }));
    } catch {
      // storage refused the write: the page's own count stands in
    }
  }

  return {
    callsToday() {
      return read(localDay(now()));
    },
    checkAndCount(limit) {
      const today = localDay(now());
      const used = read(today);
      if (used >= limit) throw new AIAnnotateError("limit", limitMessage(limit));
      write(today, used + 1);
      return used + 1;
    },
    refund() {
      const today = localDay(now());
      const used = read(today);
      if (used > 0) write(today, used - 1);
    },
  };
}
