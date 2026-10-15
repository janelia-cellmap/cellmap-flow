import { describe, expect, it } from "vitest";

import { Semaphore } from "../../src/infer/semaphore";

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

describe("Semaphore", () => {
  it("lets one run at a time, in order", async () => {
    const lock = new Semaphore(1);
    const events: string[] = [];
    let release1: () => void = () => {};
    const first = lock.run(() => new Promise<void>((resolve) => { events.push("1 start"); release1 = () => { events.push("1 end"); resolve(); }; }));
    const second = lock.run(async () => { events.push("2"); });
    const third = lock.run(async () => { events.push("3"); });
    await tick();
    expect(events).toEqual(["1 start"]);
    expect(lock.inflight).toBe(1);
    expect(lock.waiting).toBe(2);
    release1();
    await Promise.all([first, second, third]);
    expect(events).toEqual(["1 start", "1 end", "2", "3"]);
    expect(lock.inflight).toBe(0);
    expect(lock.waiting).toBe(0);
  });

  it("releases on failure and ignores a double release", async () => {
    const lock = new Semaphore(1);
    await expect(lock.run(async () => { throw new Error("boom"); })).rejects.toThrow("boom");
    const release = await lock.acquire();
    release();
    release();
    expect(lock.inflight).toBe(0);
    expect(await lock.run(async () => 42)).toBe(42);
  });

  it("allows as many as it has slots", async () => {
    const lock = new Semaphore(2);
    const a = await lock.acquire();
    await lock.acquire();
    expect(lock.inflight).toBe(2);
    let got = false;
    const third = lock.acquire().then(() => { got = true; });
    await tick();
    expect(got).toBe(false);
    a();
    await third;
    expect(got).toBe(true);
  });

  it("needs at least one slot", () => {
    expect(() => new Semaphore(0)).toThrow(/at least one slot/);
  });
});
