// The expressions behind LambdaNormalizer and LambdaPostprocessor, evaluated
// per voxel: the browser side of cellmap_flow/norm/safe_expression.py.
//
// An expression such as "2*x-1" or "np.clip(x, 0, 1)" arrives inside a layer
// URL's args blob, which anyone who can hand the page a link controls, so it
// is never run as JavaScript (Marwan's prototype used new Function; a crafted
// expression would then run in the page). It is parsed once, here, against a
// whitelist close to the Python one, into a tree of closures that take the
// voxel value, and that tree is applied to every voxel.
//
// Scalar arithmetic only: Python operators with their numpy meaning (/ is true
// division, // floors, % takes the divisor's sign, ** is right associative and
// binds tighter than a unary minus), comparisons giving 1 or 0, the numpy
// functions and array methods listed below applied elementwise, and the
// constants np.e, np.pi, np.inf, np.nan. Reductions (np.mean, np.percentile),
// indexing and shape-changing functions are not here: a per-voxel closure
// cannot see the other voxels. Those expressions are rejected with a clear
// error, as the backend's chain validation reports it.
//
// Arithmetic is done in double precision and rounded to float32 when stored,
// where numpy rounds after every operation; the two agree to the last bit or
// one ulp.

export type Evaluate = (x: number) => number;

const MAX_LENGTH = 500;

/** Single-argument numpy functions and their elementwise meaning. */
const UNARY: Record<string, (a: number) => number> = {
  abs: Math.abs, absolute: Math.abs,
  exp: Math.exp, log: Math.log, log1p: Math.log1p, log2: Math.log2, log10: Math.log10,
  sqrt: Math.sqrt, square: (a) => a * a, tanh: Math.tanh, sin: Math.sin, cos: Math.cos,
  arctan: Math.atan, arcsin: Math.asin, arccos: Math.acos,
  floor: Math.floor, ceil: Math.ceil, rint: roundHalfEven, round: roundHalfEven,
  sign: (a) => (Number.isNaN(a) ? NaN : Math.sign(a)),
  isnan: (a) => (Number.isNaN(a) ? 1 : 0), isfinite: (a) => (Number.isFinite(a) ? 1 : 0),
  logical_not: (a) => (a ? 0 : 1),
};

/** Two-argument numpy functions. */
const BINARY: Record<string, (a: number, b: number) => number> = {
  maximum: Math.max, minimum: Math.min, power: Math.pow,
  logical_and: (a, b) => (a && b ? 1 : 0), logical_or: (a, b) => (a || b ? 1 : 0),
};

const CONSTANTS: Record<string, number> = { e: Math.E, pi: Math.PI, inf: Infinity, nan: NaN };

/** `x.astype(dtype)` of a scalar: numpy's C cast, truncation toward zero and
 * wrap-around for the integer types; a float type is left as it is (the
 * values end up in a Float32Array anyway). */
const CASTS: Record<string, (a: number) => number> = {
  float: (a) => a, float16: (a) => a, float32: (a) => a, float64: (a) => a,
  bool: (a) => (a ? 1 : 0), bool_: (a) => (a ? 1 : 0),
  uint8: (a) => Math.trunc(a) & 0xff, int8: (a) => ((Math.trunc(a) & 0xff) << 24) >> 24,
  uint16: (a) => Math.trunc(a) & 0xffff, int16: (a) => ((Math.trunc(a) & 0xffff) << 16) >> 16,
  uint32: (a) => Math.trunc(a) >>> 0, int32: (a) => Math.trunc(a) | 0,
  uint64: Math.trunc, int64: Math.trunc, int: Math.trunc,
};

/** numpy's round half to even, which Math.round is not (it rounds half up). */
function roundHalfEven(a: number): number {
  const r = Math.round(a);
  return Math.abs(a % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Python's floor division and modulo: the result takes the divisor's sign. */
const floorDiv = (a: number, b: number) => Math.floor(a / b);
const pyMod = (a: number, b: number) => a - b * Math.floor(a / b);

const COMPARISONS: Record<string, (p: number, q: number) => boolean> = {
  "<": (p, q) => p < q, "<=": (p, q) => p <= q, ">": (p, q) => p > q, ">=": (p, q) => p >= q,
  "==": (p, q) => p === q, "!=": (p, q) => p !== q,
};

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

type Token =
  | { kind: "number"; value: number }
  | { kind: "name"; value: string }
  | { kind: "string"; value: string }
  | { kind: "op"; value: string }
  | { kind: "end" };

const OPERATORS = ["**", "//", "<=", ">=", "==", "!=", "+", "-", "*", "/", "%", "<", ">", "(", ")", ",", "."];

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (/\s/.test(c)) { i++; continue; }
    const number = /^(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)/.exec(text.slice(i));
    if (number) { tokens.push({ kind: "number", value: Number(number[0]) }); i += number[0].length; continue; }
    const name = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.slice(i));
    if (name) { tokens.push({ kind: "name", value: name[0] }); i += name[0].length; continue; }
    if (c === '"' || c === "'") {
      const end = text.indexOf(c, i + 1);
      if (end < 0) throw new Error(`unterminated string at ${i}`);
      tokens.push({ kind: "string", value: text.slice(i + 1, end) });
      i = end + 1;
      continue;
    }
    const op = OPERATORS.find((o) => text.startsWith(o, i));
    if (!op) throw new Error(`unexpected character ${JSON.stringify(c)} at ${i}`);
    tokens.push({ kind: "op", value: op });
    i += op.length;
  }
  tokens.push({ kind: "end" });
  return tokens;
}

// ---------------------------------------------------------------------------
// A recursive-descent parser producing closures. Precedence, loosest first:
// comparison; + -; * / // %; unary + -; ** (right associative, and its right
// side may be a unary expression, as in Python: 2**-1); then calls,
// attributes and atoms.
// ---------------------------------------------------------------------------

/** A parsed value: a number closure, or (only as a call argument) a dtype name. */
type Node = { fn: Evaluate } | { dtype: string };

class Parser {
  private pos = 0;
  constructor(private tokens: Token[], private text: string) {}

  parse(): Evaluate {
    const node = this.comparison();
    if (this.peek().kind !== "end") this.fail(`unexpected ${describe(this.peek())}`);
    return this.number(node);
  }

  private peek(): Token { return this.tokens[this.pos]; }
  private next(): Token { return this.tokens[this.pos++]; }
  private isOp(...values: string[]): boolean {
    const t = this.peek();
    return t.kind === "op" && values.includes(t.value);
  }
  private expectOp(value: string) {
    if (!this.isOp(value)) this.fail(`expected "${value}", got ${describe(this.peek())}`);
    this.next();
  }
  private fail(what: string): never {
    throw new Error(`Expression ${JSON.stringify(this.text)} is not allowed: ${what}`);
  }
  private number(node: Node): Evaluate {
    if ("dtype" in node) this.fail("a string may only be the dtype argument of astype");
    return node.fn;
  }

  private comparison(): Node {
    let left = this.additive();
    while (this.isOp("<", "<=", ">", ">=", "==", "!=")) {
      const op = (this.next() as { value: string }).value;
      const a = this.number(left), b = this.number(this.additive());
      const f = COMPARISONS[op];
      left = { fn: (x) => (f(a(x), b(x)) ? 1 : 0) };
    }
    return left;
  }

  private additive(): Node {
    let left = this.term();
    while (this.isOp("+", "-")) {
      const op = (this.next() as { value: string }).value;
      const a = this.number(left), b = this.number(this.term());
      left = { fn: op === "+" ? (x) => a(x) + b(x) : (x) => a(x) - b(x) };
    }
    return left;
  }

  private term(): Node {
    let left = this.unary();
    while (this.isOp("*", "/", "//", "%")) {
      const op = (this.next() as { value: string }).value;
      const a = this.number(left), b = this.number(this.unary());
      const f = op === "*" ? (p: number, q: number) => p * q : op === "/" ? (p: number, q: number) => p / q : op === "//" ? floorDiv : pyMod;
      left = { fn: (x) => f(a(x), b(x)) };
    }
    return left;
  }

  private unary(): Node {
    if (this.isOp("-")) { this.next(); const a = this.number(this.unary()); return { fn: (x) => -a(x) }; }
    if (this.isOp("+")) { this.next(); return this.unary(); }
    return this.power();
  }

  private power(): Node {
    const base = this.postfix();
    if (this.isOp("**")) {
      this.next();
      const a = this.number(base), b = this.number(this.unary());  // right associative: unary() reaches power() again
      return { fn: (x) => Math.pow(a(x), b(x)) };
    }
    return base;
  }

  /** An atom followed by attribute accesses and calls: np.clip(x, 0, 1), x.astype("uint8"). */
  private postfix(): Node {
    let node = this.atom();
    while (this.isOp(".")) {
      this.next();
      const attr = this.next();
      if (attr.kind !== "name") this.fail(`expected an attribute name after "."`);
      if ("dtype" in node) this.fail(`attribute .${attr.value} of a string`);
      node = this.method(node.fn, attr.value);
    }
    return node;
  }

  private atom(): Node {
    const t = this.next();
    if (t.kind === "number") { const v = t.value; return { fn: () => v }; }
    if (t.kind === "string") return { dtype: t.value };
    if (t.kind === "op" && t.value === "(") {
      const inner = this.comparison();
      this.expectOp(")");
      return inner;
    }
    if (t.kind === "name") {
      if (t.value === "x") return { fn: (x) => x };
      if (t.value === "abs") return this.call("abs", [UNARY.abs]);
      if (t.value === "np") {
        this.expectOp(".");
        const member = this.next();
        if (member.kind !== "name") this.fail(`expected a name after "np."`);
        return this.numpy(member.value);
      }
      this.fail(`name "${t.value}" (only x, np and abs are available)`);
    }
    this.fail(`unexpected ${describe(t)}`);
  }

  /** np.<name>: a constant, a dtype (x.astype(np.uint8), or np.uint8(x) as a
   * cast), or a function applied to the arguments that follow. */
  private numpy(name: string): Node {
    if (name in CONSTANTS && !this.isOp("(")) { const v = CONSTANTS[name]; return { fn: () => v }; }
    if (name in CASTS) return this.isOp("(") ? this.call(`np.${name}`, [CASTS[name]]) : { dtype: name };
    if (name in UNARY) return this.call(`np.${name}`, [UNARY[name]]);
    if (name in BINARY) return this.call(`np.${name}`, [undefined, BINARY[name]]);
    if (name === "clip") return this.call("np.clip", [undefined, undefined, clip]);
    if (name === "where") return this.call("np.where", [undefined, undefined, where]);
    this.fail(`np.${name} (not an elementwise function the browser evaluates)`);
  }

  /** `<value>.<method>(...)`: the array methods that are elementwise. */
  private method(value: Evaluate, name: string): Node {
    if (name === "astype") {
      this.expectOp("(");
      const arg = this.comparison();
      this.expectOp(")");
      if (!("dtype" in arg)) this.fail("astype takes a dtype name as a string");
      const cast = CASTS[arg.dtype];
      if (!cast) this.fail(`astype(${JSON.stringify(arg.dtype)}): not a numeric dtype`);
      return { fn: (x) => cast(value(x)) };
    }
    if (name === "clip") return this.callWith("x.clip", value, [undefined, undefined, clip]);
    if (name === "round") return this.callWith("x.round", value, [roundHalfEven]);
    this.fail(`attribute .${name} (only astype, clip and round)`);
  }

  /** `(args)` after a function name; `byArity[n - 1]` is the function of n arguments. */
  private call(what: string, byArity: (((...a: number[]) => number) | undefined)[]): Node {
    this.expectOp("(");
    const args = this.arguments();
    const f = byArity[args.length - 1];
    if (!f) this.fail(`${what} takes ${byArity.flatMap((g, i) => (g ? [i + 1] : [])).join(" or ")} argument(s), got ${args.length}`);
    return { fn: apply(f, args) };
  }

  /** A method call: the receiver is the first argument. */
  private callWith(what: string, receiver: Evaluate, byArity: (((...a: number[]) => number) | undefined)[]): Node {
    this.expectOp("(");
    const args = [receiver, ...this.arguments()];
    const f = byArity[args.length - 1];
    if (!f) this.fail(`${what} takes ${byArity.flatMap((g, i) => (g ? [i] : [])).join(" or ")} argument(s), got ${args.length - 1}`);
    return { fn: apply(f, args) };
  }

  private arguments(): Evaluate[] {
    const args: Evaluate[] = [];
    // Keyword arguments (axis=0) never get here: "=" alone is not a token, so
    // the tokenizer already rejected them.
    while (!this.isOp(")")) {
      args.push(this.number(this.comparison()));
      if (this.isOp(",")) this.next();
      else if (!this.isOp(")")) this.fail(`expected "," or ")" in a call, got ${describe(this.peek())}`);
    }
    this.next();
    return args;
  }
}

const clip = (a: number, lo: number, hi: number) => Math.min(Math.max(a, lo), hi);
const where = (c: number, a: number, b: number) => (c ? a : b);

function apply(f: (...a: number[]) => number, args: Evaluate[]): Evaluate {
  if (args.length === 1) { const [a] = args; return (x) => f(a(x)); }
  if (args.length === 2) { const [a, b] = args; return (x) => f(a(x), b(x)); }
  const [a, b, c] = args;
  return (x) => f(a(x), b(x), c(x));
}

function describe(t: Token): string {
  if (t.kind === "end") return "the end of the expression";
  if (t.kind === "number") return `number ${t.value}`;
  if (t.kind === "string") return `string ${JSON.stringify(t.value)}`;
  return `"${t.value}"`;
}

/** The function of `x` an expression describes, or an Error naming what in it
 * is not allowed (compile_expression in Python). */
export function compileExpression(expression: unknown): Evaluate {
  if (typeof expression !== "string") throw new Error(`Expression must be a string, got ${typeof expression}`);
  if (expression.length > MAX_LENGTH) throw new Error(`Expression is longer than ${MAX_LENGTH} characters`);
  const text = expression.trim();
  if (!text) throw new Error("Expression is empty");
  let tokens: Token[];
  try {
    tokens = tokenize(text);
  } catch (e) {
    throw new Error(`Invalid expression ${JSON.stringify(expression)}: ${(e as Error).message}`);
  }
  return new Parser(tokens, text).parse();
}
