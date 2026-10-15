// The safe expression evaluator behind the Lambda steps: Python semantics
// for the operators, the whitelisted numpy functions, and clear refusals.
import { describe, expect, it } from "vitest";

import { compileExpression } from "../../src/infer/expression";

const at = (expression: string, x: number) => compileExpression(expression)(x);

describe("compileExpression: arithmetic", () => {
  it("evaluates the models' normalization", () => {
    expect(at("2*x-1", 0.5)).toBe(0);
    expect(at("2*x-1", 1)).toBe(1);
    expect(at("x/255", 51)).toBeCloseTo(0.2, 12);
  });

  it("follows Python precedence and associativity", () => {
    expect(at("1+2*3", 0)).toBe(7);
    expect(at("(1+2)*3", 0)).toBe(9);
    expect(at("2**3**2", 0)).toBe(512);  // right associative
    expect(at("-x**2", 3)).toBe(-9);  // ** binds tighter than unary minus
    expect(at("2**-1", 0)).toBe(0.5);
    expect(at("10-2-3", 0)).toBe(5);
    expect(at("16/4/2", 0)).toBe(2);
  });

  it("floors // and gives % the divisor's sign, as numpy does", () => {
    expect(at("x//2", 7)).toBe(3);
    expect(at("x//2", -7)).toBe(-4);
    expect(at("x%3", -7)).toBe(2);
    expect(at("x%-3", 7)).toBe(-2);
    expect(at("x%2.5", 6)).toBeCloseTo(1, 12);
  });

  it("reads numbers in every form", () => {
    expect(at("1e3+.5+2.", 0)).toBe(1002.5);
    expect(at("1.5E-1", 0)).toBe(0.15);
  });

  it("makes comparisons 1 or 0", () => {
    expect(at("x>0.5", 0.7)).toBe(1);
    expect(at("x>0.5", 0.5)).toBe(0);
    expect(at("x>=0.5", 0.5)).toBe(1);
    expect(at("x==1", 1)).toBe(1);
    expect(at("x!=1", 1)).toBe(0);
    expect(at("(x>0)*x", -2)).toBe(-0);
  });
});

describe("compileExpression: numpy functions and methods", () => {
  it("applies elementwise functions", () => {
    expect(at("np.exp(x)", 0)).toBe(1);
    expect(at("np.log(x)", Math.E)).toBeCloseTo(1, 12);
    expect(at("np.sqrt(x)", 16)).toBe(4);
    expect(at("np.tanh(x)", 0)).toBe(0);
    expect(at("abs(x)", -3)).toBe(3);
    expect(at("np.abs(x)", -3)).toBe(3);
    expect(at("np.square(x)", 3)).toBe(9);
    expect(at("np.sign(x)", -0.1)).toBe(-1);
    expect(at("np.floor(x)", -1.5)).toBe(-2);
  });

  it("rounds half to even like numpy", () => {
    expect(at("np.rint(x)", 2.5)).toBe(2);
    expect(at("np.round(x)", 3.5)).toBe(4);
    expect(at("x.round()", -2.5)).toBe(-2);
  });

  it("clips, both as a function and as a method", () => {
    expect(at("np.clip(x, 0, 1)", 1.5)).toBe(1);
    expect(at("np.clip(x, 0, 1)", -1)).toBe(0);
    expect(at("x.clip(0, 1)", 0.3)).toBe(0.3);
  });

  it("takes two-argument functions and where", () => {
    expect(at("np.maximum(x, 0)", -1)).toBe(0);
    expect(at("np.minimum(x, 0)", -1)).toBe(-1);
    expect(at("np.power(x, 2)", 3)).toBe(9);
    expect(at("np.where(x > 0, 1, -1)", 2)).toBe(1);
    expect(at("np.where(x > 0, 1, -1)", -2)).toBe(-1);
  });

  it("casts with astype as a C cast does", () => {
    expect(at("x.astype('uint8')", 300.7)).toBe(44);
    expect(at('x.astype("uint8")', -1)).toBe(255);
    expect(at("x.astype('int8')", 200)).toBe(-56);
    expect(at("x.astype('float32')", 0.1)).toBe(0.1);
    expect(at("x.astype('bool')", 0.1)).toBe(1);
    expect(at("(x*255).astype(np.uint8)", 1)).toBe(255);
    expect(at("np.uint8(x)", 257)).toBe(1);
    expect(at("x.astype(np.bool_)", 2)).toBe(1);
  });

  it("knows the constants", () => {
    expect(at("np.pi", 0)).toBe(Math.PI);
    expect(at("np.e", 0)).toBe(Math.E);
    expect(at("np.inf", 0)).toBe(Infinity);
    expect(Number.isNaN(at("np.nan", 0))).toBe(true);
    expect(at("np.isnan(x)", NaN)).toBe(1);
    expect(at("np.isfinite(x)", Infinity)).toBe(0);
  });
});

describe("compileExpression: refusals", () => {
  it("names what is not allowed", () => {
    expect(() => compileExpression("y+1")).toThrow(/name "y"/);
    expect(() => compileExpression("np.mean(x)")).toThrow(/np\.mean/);
    expect(() => compileExpression("x[0]")).toThrow(/unexpected character "\["/);
    expect(() => compileExpression("eval('1')")).toThrow(/name "eval"/);
    expect(() => compileExpression("x.shape")).toThrow(/attribute \.shape/);
    expect(() => compileExpression("np.clip(x, 0)")).toThrow(/np\.clip takes 3 argument/);
    expect(() => compileExpression("np.clip(x, a_min=0, a_max=1)")).toThrow(/unexpected character "="/);
    expect(() => compileExpression("x.astype('U1000')")).toThrow(/not a numeric dtype/);
    expect(() => compileExpression("'float32'")).toThrow(/string may only be the dtype/);
    expect(() => compileExpression("x +")).toThrow(/unexpected the end/);
    expect(() => compileExpression("(x")).toThrow(/expected "\)"/);
  });

  it("rejects non-strings, empty and over-long expressions", () => {
    expect(() => compileExpression(5)).toThrow(/must be a string/);
    expect(() => compileExpression("  ")).toThrow(/empty/);
    expect(() => compileExpression(`x${"+1".repeat(300)}`)).toThrow(/longer than 500/);
  });
});
