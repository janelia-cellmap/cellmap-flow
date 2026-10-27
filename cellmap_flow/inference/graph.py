"""The graph a torch model computes, for its output activation (``portable.activation``).

``module_graph(model)`` reads the graph of what the inference server runs,
as ``portable.activation.output_activation`` takes it: ``(nodes, output)``,
each node ``{"op", "inputs", "outputs"}`` (a clip's ``min`` and ``max``
where they are constants), ``output`` the name of the value the model
returns. The op names are the format's own; ``activation.canonical`` reads
them.

- A TorchScript module (a cellmap-models or Hugging Face export's model.ts,
  a fly .ts): its ``inlined_graph``, every submodule's call inlined. Its
  constants and attribute reads (the weights) are values no node writes; a
  list built for ``aten::cat`` is the cat's inputs.
- A torch.export program, or a module with an fx graph of its own (an
  ExportedProgram's ``module()``, ``torch.export.unflatten``'s): that graph,
  each submodule called with a graph of its own inlined.
- Any other module (a script's, a DaCapo run's, a fly checkpoint's network in
  a Sequential with its sigmoid): a torch.fx symbolic trace, torch.nn's own
  modules and TorchScript submodules (a fly .ts inside its Sequential) left
  whole and read as above. A forward that branches on its input's values
  cannot be traced: no graph.

An in-place op (``aten::sigmoid_``, ``x.sigmoid_()``) changes the value it
is given, and a traced graph goes on reading that value by its old name
(``x.sigmoid_(); return x`` returns the name the sigmoid was given): each
later read of it is read as the in-place op's output instead.

torch is imported when a graph is first read: the runner imports this module.
"""

import logging

logger = logging.getLogger(__name__)


def module_graph(model):
    """``(nodes, output)`` of ``model``, a torch module or torch.export program; None when it cannot be read."""
    import torch

    try:
        if isinstance(model, torch.jit.ScriptModule):
            return _torchscript(model.inlined_graph)
        if isinstance(model, torch.export.ExportedProgram):
            model = model.graph_module
        if isinstance(model, torch.nn.Module):
            return _fx(model)
    except Exception as e:
        logger.info(f"Could not read the graph of {type(model).__name__}: {e}")
    return None


class _Names:
    """Value names, prefixed by where a graph was inlined, and read past the in-place ops that changed them."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.newer = {}

    def __call__(self, name):
        name = self.prefix + name
        while name in self.newer:
            name = self.newer[name]
        return name

    def changed_in_place(self, op, inputs, outputs):
        """After ``op`` (its name): if it works in place (``sigmoid_``, not
        ``__add__``), later reads of its input read its output."""
        if op.endswith("_") and not op.endswith("__") and inputs and outputs:
            self.newer[inputs[0]] = outputs[0]


def _identity(source, name):
    """A node passing value ``source`` on as ``name``: a graph's input or output where it is inlined."""
    return {"op": "identity", "inputs": [source], "outputs": [name]}


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _is_clip(op) -> bool:
    from cellmap_flow.portable.activation import CLIP_OPS, canonical

    return canonical(op) in CLIP_OPS


def _clip(node, op, args, kwargs):
    """``node``, a clip's, with ``min`` and ``max`` set from its arguments after the tensor (``clamp(x, 0, 1)``,
    ``hardtanh(x, min_val=-1, max_val=1)``, whose defaults those are), where they are numbers."""
    from cellmap_flow.portable.activation import canonical

    defaults = (-1.0, 1.0) if canonical(op) == "hardtanh" else (None, None)
    for i, key in enumerate(("min", "max")):
        given = kwargs.get(key, kwargs.get(f"{key}_val", args[i] if len(args) > i else defaults[i]))
        if _number(given) is not None:
            node[key] = given
    return node


def _torchscript(graph, prefix="", bound=None):
    """``(nodes, output)`` of a TorchScript graph. ``bound``: the names its
    inputs after ``self`` take, where an fx graph calls it."""
    names = _Names(prefix)
    inputs = list(graph.inputs())[1:]  # self first
    nodes = [_identity(outer, names(value.debugName())) for value, outer in zip(inputs, bound or ())]
    lists = {}

    def constant(value):
        return value.node().kind() == "prim::Constant"

    for node in graph.nodes():
        kind = node.kind()
        if kind in ("prim::Constant", "prim::GetAttr"):
            continue  # a constant, a weight: values no node writes
        reads = []
        for value in node.inputs():
            if not constant(value):
                reads.extend(lists.get(names(value.debugName()), [names(value.debugName())]))
        writes = [names(value.debugName()) for value in node.outputs()]
        if kind == "prim::ListConstruct" and len(writes) == 1:
            lists[writes[0]] = reads
            continue
        entry = {"op": kind, "inputs": reads, "outputs": writes}
        if _is_clip(kind):
            _clip(entry, kind, [value.toIValue() if constant(value) else None for value in list(node.inputs())[1:]], {})
        nodes.append(entry)
        names.changed_in_place(kind, reads, writes)
    outputs = [names(value.debugName()) for value in graph.outputs()]
    return nodes, outputs[0] if len(outputs) == 1 else None


def _op_name(target) -> str:
    """An fx call_function's target by name: ``aten.sigmoid.default`` for a torch.export op, else its ``__name__``."""
    import torch

    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    return getattr(target, "__name__", str(target))


def _fx(module, prefix="", bound=None):
    """``(nodes, output)`` of ``module``'s fx graph, its own or a symbolic
    trace's. ``bound``: the names its inputs take, where it is inlined."""
    import torch
    import torch.fx

    class LeavesWhole(torch.fx.Tracer):
        def is_leaf_module(self, m, qualname):
            return isinstance(m, torch.jit.ScriptModule) or super().is_leaf_module(m, qualname)

    graph = getattr(module, "graph", None)
    if not isinstance(graph, torch.fx.Graph):
        graph = LeavesWhole().trace(module)
    names = _Names(prefix)
    nodes, output, placeholders = [], None, 0
    for node in graph.nodes:
        name = names(node.name)
        if node.op == "placeholder":
            if bound is not None and placeholders < len(bound):
                nodes.append(_identity(bound[placeholders], name))
            placeholders += 1
            continue
        if node.op == "get_attr":
            continue  # a weight or a constant
        reads = [names(n.name) for n in node.all_input_nodes]
        if node.op == "output":
            value = node.args[0]
            if isinstance(value, (tuple, list)) and len(value) == 1:
                value = value[0]  # a torch.export graph returns its outputs in a tuple
            output = names(value.name) if isinstance(value, torch.fx.Node) else None
            continue
        args = list(node.args[1:])
        if node.op == "call_module":
            child = module.get_submodule(node.target)
            if isinstance(child, torch.jit.ScriptModule) or isinstance(getattr(child, "graph", None), torch.fx.Graph):
                inner = f"{name}/"
                inside, returned = (_torchscript(child.inlined_graph, inner, reads)
                                    if isinstance(child, torch.jit.ScriptModule) else _fx(child, inner, reads))
                nodes.extend(inside)
                if returned is not None:
                    nodes.append(_identity(returned, name))
                continue
            op = type(child).__name__
            kwargs = {k: getattr(child, k) for k in ("min_val", "max_val") if hasattr(child, k)}
            args = []
        else:
            op = _op_name(node.target) if node.op == "call_function" else str(node.target)
            kwargs = dict(node.kwargs)
        entry = {"op": op, "inputs": reads, "outputs": [name]}
        nodes.append(_clip(entry, op, args, kwargs) if _is_clip(op) else entry)
        names.changed_in_place(op, reads, [name])
    return nodes, output
