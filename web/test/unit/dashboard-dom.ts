// The DOM calls the dashboard's page scripts (cellmap_flow/dashboard/static/
// js/) make, on plain objects, for their unit tests: the tests run in node,
// which has no DOM. An attribute set on an element sets the property a
// browser derives from it (checked, value, type, hidden, disabled, id); a
// selector is tags, ids, classes, :checked and [name=value], joined by
// spaces; and an event bubbles, its listeners getting the element it
// happened to as its target.

// What dom.js h() tells a child node from a string by.
export class FakeNode {
  parentNode: FakeElement | null = null;
}

// A text node.
export class FakeText extends FakeNode {
  constructor(public textContent: string) {
    super();
  }
}

export class FakeElement extends FakeNode {
  childNodes: (FakeElement | FakeText)[] = [];
  attributes = new Map<string, string>();
  className = "";
  id = "";
  type = "";
  value = "";
  checked = false;
  disabled = false;
  hidden = false;
  open = false;
  innerHTML = "";
  onclick: (() => void) | null = null;
  dataset: Record<string, string> = {};
  style: Record<string, string> = {};
  listeners = new Map<string, ((event: unknown) => void)[]>();
  classList = {
    add: (...names: string[]) => names.forEach((name) => this.classList.toggle(name, true)),
    contains: (name: string) => this.className.split(/\s+/).includes(name),
    toggle: (name: string, on: boolean) => {
      const names = this.className.split(/\s+/).filter((n) => n && n !== name);
      this.className = (on ? [...names, name] : names).join(" ");
    },
  };

  constructor(public tagName: string) {
    super();
  }

  get children(): FakeElement[] {
    return this.childNodes.filter((node): node is FakeElement => node instanceof FakeElement);
  }

  get childElementCount(): number {
    return this.children.length;
  }

  get firstElementChild(): FakeElement | undefined {
    return this.children[0];
  }

  get textContent(): string {
    return this.childNodes.map((node) => node.textContent).join("");
  }

  set textContent(text: string) {
    this.replaceChildren(text);
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    if (name === "id" || name === "type" || name === "value") this[name] = value;
    if (name === "checked" || name === "hidden" || name === "disabled") this[name] = true;
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  append(...nodes: (FakeElement | FakeText | string)[]): void {
    for (const node of nodes) {
      const child = typeof node === "string" ? new FakeText(node) : node;
      child.parentNode?.removeChild(child);
      child.parentNode = this;
      this.childNodes.push(child);
    }
  }

  appendChild(child: FakeElement): FakeElement {
    this.append(child);
    return child;
  }

  replaceChildren(...nodes: (FakeElement | FakeText | string)[]): void {
    for (const child of this.childNodes) child.parentNode = null;
    this.childNodes = [];
    this.append(...nodes);
  }

  removeChild(child: FakeElement | FakeText): void {
    this.childNodes = this.childNodes.filter((node) => node !== child);
    child.parentNode = null;
  }

  remove(): void {
    this.parentNode?.removeChild(this);
  }

  addEventListener(type: string, listener: (event: unknown) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }

  dispatchEvent(event: { type: string; key?: string }): void {
    const passed = { type: event.type, key: event.key, target: this, preventDefault: () => {} };
    for (let el: FakeElement | null = this; el; el = el.parentNode) {
      for (const listener of el.listeners.get(event.type) ?? []) listener(passed);
    }
  }

  get descendants(): FakeElement[] {
    return this.children.flatMap((child) => [child, ...child.descendants]);
  }

  querySelectorAll(selector: string): FakeElement[] {
    const parts = selector.trim().split(/\s+/);
    return this.descendants.filter((el) => matches(el, parts, this));
  }

  querySelector(selector: string): FakeElement | null {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

function matchesOne(el: FakeElement, part: string): boolean {
  return (part.match(/^[a-z0-9]+|[#.:][\w-]+|\[\w+=\w+\]/gi) ?? []).every((token) => {
    if (token[0] === "#") return el.id === token.slice(1);
    if (token[0] === ".") return el.classList.contains(token.slice(1));
    if (token === ":checked") return el.checked;
    if (token[0] === "[") {
      const [name, value] = token.slice(1, -1).split("=");
      return String((el as unknown as Record<string, unknown>)[name]) === value;
    }
    return el.tagName === token.toLowerCase();
  });
}

// Whether `el` matches the last of `parts` and has ancestors, under `root`,
// matching the others in order.
function matches(el: FakeElement, parts: string[], root: FakeElement): boolean {
  if (!matchesOne(el, parts[parts.length - 1])) return false;
  let rest = parts.slice(0, -1);
  for (let up = el.parentNode; up && up !== root && rest.length; up = up.parentNode) {
    if (matchesOne(up, rest[rest.length - 1])) rest = rest.slice(0, -1);
  }
  return rest.length === 0;
}
