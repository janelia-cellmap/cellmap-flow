"""A model published to Hugging Face: its model card, its manifest's
"published" block and the links that open it in the app, written once for
both modes.

The browser (web/src/backend/routes/publish.ts, through publish_calls.py)
and the server (dashboard/routes/finetune/publish.py) upload, in one commit,
what ``publication_files`` gives and the model's own files beside them:

    README.md                  the model card (model_card)
    <folder>/manifest.json     the model's manifest in the catalog's format
                               (web/src/models/manifest.ts parseManifest, the
                               server's models/resolve.py), with a
                               "published" block on top (published_manifest)
    <folder>/<file>.onnx       one per variant of the manifest: the one file
                               a browser trained, or each precision a
                               server's export wrote (fp32, fp16, int8w)

<folder> is the model's name. The manifest is the catalog's format, so the
folder is a model like any of cellmap/onnx-web-models': both modes load it
by its ref, ``<owner>/<repo>/<folder>``. The "published" block says what the
manifest alone would not: the model's own name (a manifest finetuned in the
browser keeps its base's task and model, after which the catalog would
name it), what it was finetuned from (a ref) and how, and when.

A publish, ``pub``, is plain JSON, which each mode builds from its own
records of the model:

    repo, folder, published_at   where it goes (``owner/name``, the folder in
                                 it) and when (ISO 8601)
    manifest                     the catalog-format manifest, whose variants
                                 are the model's files uploaded
    origin                       "trained" where it was published from, or
                                 weights that came there in a "link" or a "file"
    where                        "browser" or "server": where that was
    finetuned_from               the ref of the model it was finetuned from, or None
    finetune_mode                "head" (its last layer), "full" (its whole
                                 network), "lora" (adapters, merged into the
                                 network) or None
    epochs, final_loss,          the training's, or None where not known
    trained_at
    app_version                  cellmap-flow's release
    dataset                      the dataset's URL, or None: only when the
                                 user asked for it, since a dataset's URL can
                                 be internal

Nothing here reads or writes a file or the network: the callers upload.
"""

import json
import re
from typing import List, Optional
from urllib.parse import quote

# The app a published model's card links to: the public site's, wherever
# the model was published from (a development server's address means
# nothing to the card's readers).
PUBLIC_APP = "https://janelia-cellmap.github.io/cellmap-flow/app/"

# Hugging Face's site, where a repository's page is <HUB>/<owner>/<repo>.
HUB = "https://huggingface.co"

# Sign in with Hugging Face, in both modes (the browser's popup,
# web/src/publish/oauth.ts, and the server's device code). The client id is
# the URL of a client metadata document the public app serves (CIMD,
# https://huggingface.co/docs/hub/oauth#automated-oauth-app-creation), so no
# OAuth app is registered on anyone's account. The scopes are who the user
# is, and to create repositories and write to those made through this app
# (contribute-repos), which is all a publish does: not write-repos or
# manage-repos, with which a page could change or delete any of the user's
# repositories.
HF_OAUTH_CLIENT_ID = "https://janelia-cellmap.github.io/cellmap-flow/app/.well-known/oauth-cimd"
HF_OAUTH_SCOPES = ("openid", "profile", "contribute-repos")

# The query parameters of the app's links: web/src/link.ts reads them back
# under these names (exported to the browser by web/build/export_data.py),
# and web/src/share-link.ts writes the Share button's with them.
MODEL_PARAM = "model"
DATASET_PARAM = "dataset"

_CELLMAP_FLOW = "[cellmap-flow](https://github.com/janelia-cellmap/cellmap-flow)"

# What the paint trained, as the card says it, by the publish's
# finetune_mode; a mode not said is "its weights".
_TRAINED = {
    "head": "its last layer was trained",
    "full": "its whole network was trained",
    "lora": "LoRA adapters were trained",
}

# Where a model was trained and published from, as the card says it: the
# server's annotations are painted in its dashboard's viewer.
_PLACES = {
    "browser": {"at": "in the browser", "painted": "in the browser", "from": "from the browser", "there": "that browser"},
    "server": {"at": "on a server", "painted": "in the dashboard", "from": "from a server", "there": "that server"},
}


def ref_of(repo: str, folder: str) -> str:
    """The ref of ``folder`` in ``repo``, as links and *Add a model* take it."""
    return f"{repo}/{folder}"


def _query_value(value: str) -> str:
    """``value`` as a query value, as the Share button writes one
    (share-link.ts queryValue): kept readable, with only what would end the
    query (#), split it (&) or be read back as a space (+) escaped, and a
    space or a character outside ASCII as in any URL. The safe characters
    are JavaScript's encodeURI's but for those three."""
    return quote(value, safe="!*'();/?:@=$,")


def app_links(app: str, ref: str, dataset: Optional[str]) -> dict:
    """The links that open the model ``ref`` in the app at ``app`` (a page's
    own address, or PUBLIC_APP in a card), whatever query or hash ``app``
    has: ``try_link``, the model alone, which the app opens with the model
    chosen for whatever volume the user connects; and ``dataset_link``, the
    Share button's link of ``dataset`` with the model, None without a
    dataset."""
    base = re.split(r"[?#]", app, maxsplit=1)[0]
    model = f"{MODEL_PARAM}={_query_value(ref)}"
    return {
        "try_link": f"{base}?{model}",
        "dataset_link": f"{base}?{DATASET_PARAM}={_query_value(dataset)}&{model}" if dataset else None,
    }


def published_links(repo: str, folder: str, app: str, dataset: Optional[str]) -> dict:
    """Where a published model is, as both modes' publish routes answer once
    it is done: its ``ref``, its repository's page (``url``), and
    ``app_links`` on ``app`` (the page's own address) with ``dataset``."""
    ref = ref_of(repo, folder)
    return {"ref": ref, "url": f"{HUB}/{repo}", **app_links(app, ref, dataset)}


def published_manifest(pub: dict) -> dict:
    """The manifest uploaded as ``<folder>/manifest.json``: ``pub``'s, its
    variants as they are, and the "published" block in place of any it had
    (a model opened from a published file brings that one's). What is not
    known is left out of the block, and the dataset is in it only when
    ``pub`` names one."""
    block = {
        "name": pub["folder"],
        "finetuned_from": pub.get("finetuned_from"),
        "finetune_mode": pub.get("finetune_mode"),
        "epochs": pub.get("epochs"),
        "final_loss": pub.get("final_loss"),
        "trained_at": pub.get("trained_at"),
        "published_at": pub["published_at"],
        "app": f"cellmap-flow {pub['app_version']} ({pub['where']})",
        "dataset": pub.get("dataset"),
    }
    manifest = {k: v for k, v in pub["manifest"].items() if k != "published"}
    return {**manifest, "published": {k: v for k, v in block.items() if v is not None}}


def _voxels(size) -> str:
    """A voxel size in nm as the card gives it: one number when the axes
    agree (16), else each axis's (8x4x4), to four decimals."""
    parts = [f"{round(float(n), 4):g}" for n in size]
    return parts[0] if len(set(parts)) == 1 else "x".join(parts)


def _block(size) -> str:
    return "×".join(str(n) for n in size)


def _listed(items: List[str]) -> str:
    """``items`` as a sentence lists them: "a", "a and b", "a, b and c"."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _blocks(variants: List[dict]) -> str:
    """What the card's first paragraph says of the blocks: each block size
    in and out, with the variants exported at it (one precision for a
    browser's model, several for a server's export at one block)."""
    sizes = {}
    for v in variants:
        sizes.setdefault((_block(v["input_size"]), _block(v["output_size"])), []).append(v["variant"])
    return ", or ".join(f"{i} voxels in and {o} out ({', '.join(names)})" for (i, o), names in sizes.items())


def _provenance(pub: dict) -> str:
    """The card's paragraph on where the model came from: trained where it
    was published from (on what, how long and how), or weights that came
    there in a link or a file and were trained elsewhere."""
    place = _PLACES[pub["where"]]
    app = f"{_CELLMAP_FLOW} {pub['app_version']}"
    architecture = pub["manifest"].get("architecture")
    arch = f" ({architecture})" if architecture else ""
    base = pub.get("finetuned_from")
    dataset = pub.get("dataset")
    if pub["origin"] == "trained":
        mode = pub.get("finetune_mode")
        epochs = ""
        if pub.get("epochs"):
            loss = pub.get("final_loss")
            epochs = f" for {pub['epochs']} epoch(s)" + ("" if loss is None else f", to a final loss of {loss:.3g}")
        when = f" ({pub['trained_at'][:10]})" if pub.get("trained_at") else ""
        # A LoRA finetune's adapters are folded into the weights they adapt
        # before the export, so the graph is the base's, with no adapter in it.
        merged = ", and merged into the network" if mode == "lora" else ""
        return (
            f"Finetuned {place['at']} with {app}{f' from `{base}`' if base else ''}{arch}: "
            f"{_TRAINED.get(mode, 'its weights were trained')} on annotations painted {place['painted']}{epochs}{when}{merged}."
            + (f" The annotations were painted on {dataset}." if dataset else "")
        )
    came = "in a shared link" if pub["origin"] == "link" else "as a file"
    return (
        f"Published {place['from']} with {app}{arch}{f', finetuned from `{base}`' if base else ''}: "
        f"its weights came to {place['there']} {came}."
        + (f" It was published with {dataset} open." if dataset else "")
    )


def model_card(pub: dict) -> str:
    """The model card, README.md: Hugging Face's front matter (the library,
    and tags that find it), then what the model predicts, at what voxel
    size and in what blocks, what it was finetuned from and how, and how to
    open it."""
    m = pub["manifest"]
    folder, variants = pub["folder"], m["variants"]
    ref = ref_of(pub["repo"], folder)
    links = app_links(PUBLIC_APP, ref, pub.get("dataset"))
    task = re.sub(r"[^a-z0-9]+", "-", m["task"].lower()).strip("-")
    tags = [tag for tag in ("cellmap-flow", "onnx", "electron-microscopy", task) if tag]
    output = f" ({m['output']})" if m.get("output") else ""
    trained_on = "trained on" if pub["origin"] == "trained" else "published with"
    files = _listed([f"`{folder}/{v['file']}`" + (f" ({v['variant']})" if len(variants) > 1 else "") for v in variants])
    graph = "is the ONNX graph" if len(variants) == 1 else "are the ONNX graph, one file per variant"
    return "\n".join([
        "---",
        "library_name: onnx",
        "tags:",
        *(f"- {tag}" for tag in tags),
        "---",
        "",
        f"# {folder}",
        "",
        f"A {m['task']} model for electron microscopy: it predicts {', '.join(m['channels'])}{output}, "
        f"from {_voxels(m['input_voxel_size_nm'])} nm voxels to {_voxels(m['output_voxel_size_nm'])} nm voxels, "
        f"in blocks of {_blocks(variants)}.",
        "",
        _provenance(pub),
        "",
        "## Open it",
        "",
        f"- [Try it in the browser]({links['try_link']}): cellmap-flow opens with the model chosen; connect any volume to run it.",
        *([f"- [Open it on the dataset it was {trained_on}]({links['dataset_link']})."] if links["dataset_link"] else []),
        f"- Anywhere cellmap-flow takes a model (*Add a model* in the dashboard's Models tab, in the browser or on a server): `{ref}`.",
        "",
        f"`{folder}/manifest.json` says the model's voxel sizes, blocks, channels and input normalization in cellmap-flow's "
        f"format, and {files} {graph}.",
        "",
    ])


def publication_files(pub: dict) -> List[dict]:
    """The card and the manifest of the publish ``pub``, by their path in
    the repository, in the order they are uploaded: ``[{"path", "text"}]``.
    The model's files are the caller's to add after them, from its own
    bytes: ``<folder>/<file>`` for each variant of ``pub["manifest"]``."""
    manifest = json.dumps(published_manifest(pub), indent=2, ensure_ascii=False) + "\n"
    return [
        {"path": "README.md", "text": model_card(pub)},
        {"path": f"{pub['folder']}/manifest.json", "text": manifest},
    ]
