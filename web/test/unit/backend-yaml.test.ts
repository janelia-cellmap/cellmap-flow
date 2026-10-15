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

  it("writes nested lists and mappings, empty ones inline, as PyYAML does", () => {
    const value = {
      none: null,
      inf: Infinity,
      neg: -3,
      list_empty: [],
      map_empty: {},
      nested: { a: [1, { b: "c" }], d: { e: "f" } },
      lists: [["x", "y"], [1, 2]],
    };
    expect(toYaml(value)).toBe(
      ["none: null", "inf: .inf", "neg: -3", "list_empty: []", "map_empty: {}", "nested:", "  a:", "  - 1", "  - b: c", "  d:", "    e: f", "lists:", "- - x", "  - y", "- - 1", "  - 2", ""].join("\n"),
    );
    expect(toYaml([])).toBe("[]\n");
    expect(toYaml({})).toBe("{}\n");
    expect(toYaml("x")).toBe("x\n");
  });

  it("quotes keys by the same rule as values", () => {
    expect(toYaml({ yes: "yes", No: "No", plain: "p" })).toBe('"yes": "yes"\n"No": "No"\nplain: p\n');
  });
});

describe("plainSafe", () => {
  // Each string and whether PyYAML wrote it plain (true) or quoted (false),
  // from yaml.dump on the project's Python.
  const PYYAML: [string, boolean][] = [
    ["", false],
    [" x", false],
    ["yes", false],
    ["No", false],
    ["null", false],
    ["~", false],
    ["10", false],
    ["1.5", false],
    ["2026-10-14", false],
    ["a: b", false],
    ["a #b", false],
    ["- x", false],
    ["*x", false],
    ["a\nb", false],
    ["x:", false],
    ["0x1f", false],
    ["1:30", false],
    ["@x", false],
    ["`x", false],
    ["it's", true],
    ['say "hi"', true],
    ["2*x-1", true],
    ["/nrs/cellmap/x.zarr", true],
    ["s3://bucket/key", true],
    ["a:b", true],
    ["50%", true],
    ["a,b", true],
    ["a[b]", true],
    ["y", true], // not a boolean to PyYAML (yes/no are)
    ["n", true],
  ];

  it.each(PYYAML)("%j is written plain: %s", (s, plain) => {
    expect(plainSafe(s)).toBe(plain);
  });

  it("quotes 1e3, which PyYAML (YAML 1.1) writes plain but a YAML 1.2 reader takes as a float", () => {
    expect(plainSafe("1e3")).toBe(false);
  });
});
