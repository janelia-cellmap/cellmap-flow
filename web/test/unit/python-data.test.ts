// What the browser runs against what Python says it runs: the ports in
// src/infer and the job states of backend/finetune-jobs.ts, against
// src/generated/python-data.json (web/build/export_data.py, kept fresh by
// tests/web/test_python_data_fresh.py).
import { describe, expect, it } from "vitest";
import type { JobStatus } from "../../src/backend/finetune-jobs";
import { PYTHON } from "../../src/generated/python-data";
import { SUPPORTED_NORMALIZERS } from "../../src/infer/norms";
import { SUPPORTED_POSTPROCESSORS } from "../../src/infer/post";

const browserOps = (kind: "input_norm" | "postprocess") => PYTHON.ops[kind].filter((op) => op.browser).map((op) => op.name).sort();

describe("the data shared with Python", () => {
  it("runs exactly the ops Python marks as the browser's, which are the ones the page offers", () => {
    expect([...SUPPORTED_NORMALIZERS].sort()).toEqual(browserOps("input_norm"));
    expect([...SUPPORTED_POSTPROCESSORS].sort()).toEqual(browserOps("postprocess"));
  });

  it("knows every job status Python's JobStatus has", () => {
    // Every key of the union, and only those, or this does not typecheck.
    const statuses: Record<JobStatus, true> = { PENDING: true, RUNNING: true, COMPLETED: true, FAILED: true, CANCELLED: true, WAITING_FOR_RESTART: true };
    expect(Object.keys(statuses).sort()).toEqual([...PYTHON.finetune.job_statuses].sort());
  });
});
