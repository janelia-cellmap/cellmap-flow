"""A model published to Hugging Face (cellmap_flow.portable.publish): the
card, the manifest and the links that both modes upload and answer, here
for a server's export of one model, its TorchScript and three precisions of
ONNX, or the TorchScript alone (the browser's one file is
web/test/unit/publish.test.ts's, through publish_calls)."""

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from cellmap_flow.portable import publish

REPO, FOLDER = "alice/mito-ft", "mito_16nm-ft1"
REF = f"{REPO}/{FOLDER}"


def _pub(**changes):
    # As onnx_web writes a cube's block: its side.
    variants = [{"variant": v, "input_size": 178, "output_size": 56, "file": f"mito_{v}.onnx"}
                for v in ("fp32", "fp16", "int8w")]
    manifest = {"model": FOLDER, "task": "mito", "architecture": "UNet", "channels": ["mito"],
                "input_voxel_size_nm": [16] * 3, "output_voxel_size_nm": [16] * 3, "output": "logits",
                "input_norm": [], "io": {"input": "input", "output": "output"}, "variants": variants,
                # A base's block, which the published one replaces.
                "published": {"name": "mito_16nm", "finetune_mode": "head"}}
    return {"repo": REPO, "folder": FOLDER, "published_at": "2026-10-26T12:00:00Z", "manifest": manifest,
            "origin": "trained", "where": "server", "finetuned_from": "cellmap/onnx-web-models/setup_54",
            "finetune_mode": "lora", "epochs": 12, "final_loss": 0.042, "trained_at": "2026-10-26T10:00:00Z",
            "app_version": "0.3.0", "dataset": None,
            "server": {"files": ["metadata.json", "model.ts", "model.pt2", "export_record.json"],
                       "input_size": [178] * 3, "output_size": [56] * 3, "no_browser": None}, **changes}


def test_a_server_publish_is_a_card_and_a_manifest_naming_every_file_it_uploads():
    readme, manifest_file = publish.publication_files(_pub())
    assert (readme["path"], manifest_file["path"]) == ("README.md", f"{FOLDER}/manifest.json")

    manifest = json.loads(manifest_file["text"])
    assert [v["file"] for v in manifest["variants"]] == ["mito_fp32.onnx", "mito_fp16.onnx", "mito_int8w.onnx"]
    assert manifest["published"] == {
        "name": FOLDER, "finetuned_from": "cellmap/onnx-web-models/setup_54", "finetune_mode": "lora",
        "epochs": 12, "final_loss": 0.042, "trained_at": "2026-10-26T10:00:00Z",
        "published_at": "2026-10-26T12:00:00Z", "app": "cellmap-flow 0.3.0 (server)",
    }, "the base's block replaced, and no dataset the user did not ask to name"

    card = readme["text"]
    assert all(f"{FOLDER}/mito_{v}.onnx" in card for v in ("fp32", "fp16", "int8w"))
    assert all(f"`{name}`" in card for name in ("metadata.json", "model.ts", "model.pt2", "export_record.json"))
    assert f"`{REPO}`" in card and "178×178×178" in card, "the server's ref, and the block in voxels"
    assert f"({publish.PUBLIC_APP}?model={REF})" in card
    assert "dataset=" not in card


def test_a_server_publish_with_no_browser_form_is_its_card_alone_and_opens_on_a_server_only():
    why = "The network has 812,345,678 parameters, 3.2 GB in float32"
    pub = _pub(dataset="s3://bucket/raw.zarr")
    pub["manifest"]["variants"] = []
    pub["server"]["no_browser"] = why

    [readme] = publish.publication_files(pub)
    assert readme["path"] == "README.md"
    card = readme["text"]
    assert f"`{REPO}`" in card and why in card
    assert publish.PUBLIC_APP not in card and f"`{REF}`" not in card, "nothing the browser would open"
    assert publish.published_links(REPO, None, publish.PUBLIC_APP, pub["dataset"]) == {
        "ref": None, "url": f"{publish.HUB}/{REPO}", "try_link": None, "dataset_link": None}


def test_the_links_carry_the_dataset_and_the_model_as_the_app_reads_them_back():
    dataset = "https://example.org/a b/c&d#e+f.zarr"
    links = publish.app_links("http://localhost:5173/app/?model=old#!{}", REF, dataset)
    assert links["try_link"] == f"http://localhost:5173/app/?model={REF}"
    assert parse_qs(urlsplit(links["dataset_link"]).query) == {"dataset": [dataset], "model": [REF]}
    assert publish.app_links(publish.PUBLIC_APP, REF, None)["dataset_link"] is None

    card = publish.publication_files(_pub(dataset=dataset))[0]["text"]
    assert publish.app_links(publish.PUBLIC_APP, REF, dataset)["dataset_link"] in card, "the card's links are the public app's"


def test_the_client_metadata_the_site_serves_names_this_app_and_its_sign_in_pages():
    # Hugging Face reads the document at the client id's URL (CIMD), which
    # the Pages workflow publishes from web/.well-known: a sign-in fails
    # unless it names that same id and lists the page the popup returns to.
    path = Path(__file__).parents[2] / "web" / ".well-known" / "oauth-cimd"
    document = json.loads(path.read_text())
    assert document["client_id"] == publish.HF_OAUTH_CLIENT_ID
    assert document["client_id"] == f"{publish.PUBLIC_APP}.well-known/oauth-cimd"
    assert f"{publish.PUBLIC_APP}oauth.html" in document["redirect_uris"]
    assert document["token_endpoint_auth_method"] == "none"
