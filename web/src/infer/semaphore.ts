// A counting semaphore: `slots` callers hold it at once, the rest wait their
// turn in order. One per ORT session, with one slot: an onnxruntime-web
// InferenceSession is not re-entrant, and Marwan found that two overlapping
// session.run() calls on the WebGPU provider never resolve (0 blocks done,
// the page hangs), so every forward pass of a session goes through its lock.
export class Semaphore {
  private held = 0;
  private waiters: (() => void)[] = [];

  constructor(private readonly slots = 1) {
    if (!(slots >= 1)) throw new Error(`a semaphore needs at least one slot, got ${slots}`);
  }

  /** Callers holding the semaphore now. */
  get inflight(): number { return this.held; }
  /** Callers waiting for it. */
  get waiting(): number { return this.waiters.length; }

  /** Resolves with the release function once a slot is free. Release it once. */
  acquire(): Promise<() => void> {
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      this.held--;
      this.waiters.shift()?.();
    };
    if (this.held < this.slots) {
      this.held++;
      return Promise.resolve(release);
    }
    return new Promise((resolve) => this.waiters.push(() => { this.held++; resolve(release); }));
  }

  /** `fn` run while holding a slot, released when it settles. */
  async run<T>(fn: () => Promise<T>): Promise<T> {
    const release = await this.acquire();
    try {
      return await fn();
    } finally {
      release();
    }
  }
}
