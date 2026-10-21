// The YAML emitter against PyYAML: the exported config's text (what
// yaml.dump(config, default_flow_style=False, sort_keys=False) writes, the
// reference generated with the project's Python), and which strings PyYAML
// writes plain and which it quotes.
import { describe, expect, it } from "vitest";

import { plainSafe, toYaml } from "../../src/backend/yaml";

describe("toYaml", () => {
  it("writes the exported config as PyYAML does (strings that look like numbers quoted)", () => {
    const config = {
      data_path: "https://example.org/em.zarr",
      resample: false,
      json_data: {
        input_norm: [
          { name: "MinMaxNormalizer", min_value: "0", max_value: "255" },
          { name: "LambdaNormalizer", expression: "2*x-1" },
        ],
        postprocess: [{ name: "SigmoidPostprocessor" }, { name: "ThresholdPostprocessor", threshold: 0.5 }],
      },
      models: [{ type: "huggingface", repo: "cellmap/onnx-web-models", name: "setup_54", folder: "setup_54", onnx: "setup_54_fp16.onnx" }],
    };
    // PyYAML's text, with its single quotes as the double quotes written here
    // (the same string either way).
    const expected = [
      "data_path: https://example.org/em.zarr",
      "resample: false",
      "json_data:",
      "  input_norm:",
      "  - name: MinMaxNormalizer",
      '    min_value: "0"',
      '    max_value: "255"',
      "  - name: LambdaNormalizer",
      "    expression: 2*x-1",
      "  postprocess:",
      "  - name: SigmoidPostprocessor",
      "  - name: ThresholdPostprocessor",
      "    threshold: 0.5",
      "models:",
      "- type: huggingface",
      "  repo: cellmap/onnx-web-models",
      "  name: setup_54",
      "  folder: setup_54",
      "  onnx: setup_54_fp16.onnx",
      "",
    ].join("\n");
    expect(toYaml(config)).toBe(expected);
  });
});

describe("plainSafe", () => {
  // Each string and whether PyYAML wrote it plain (true) or quoted (false),
  // from yaml.dump on the project's Python.
  const PYYAML: [string, boolean][] = [
    ["", false],
    ["yes", false],
    ["No", false],
    ["null", false],
    ["10", false],
    ["2026-10-14", false],
    ["a: b", false],
    ["a\nb", false],
    ["2*x-1", true],
    ["/nrs/cellmap/x.zarr", true],
    ["s3://bucket/key", true],
    ["y", true], // not a boolean to PyYAML (yes/no are)
  ];

  it.each(PYYAML)("%j is written plain: %s", (s, plain) => {
    expect(plainSafe(s)).toBe(plain);
  });

  it("quotes 1e3, which PyYAML (YAML 1.1) writes plain but a YAML 1.2 reader takes as a float", () => {
    expect(plainSafe("1e3")).toBe(false);
  });
});
