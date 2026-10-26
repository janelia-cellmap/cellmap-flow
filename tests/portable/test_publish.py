"""A model published to Hugging Face (cellmap_flow.portable.publish): the
card, the manifest and the links that both modes upload and answer, here
for a server's export of one model in three precisions (the browser's one
file is web/test/unit/publish.test.ts's, through publish_calls)."""

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from cellmap_flow.portable import publish

REPO, FOLDER = "alice/mito-ft", "mito_16nm-ft1"
REF = f"{REPO}/{FOLDER}"


def _pub(**changes):
    variants = [{"variant": v, "input_size": [1, 556, 556], "output_size": [1, 468, 468], "file": f"mito_{v}.onnx"}
                for v in ("fp32", "fp16", "int8w")]
    manifest = {"model": FOLDER, "task": "mito", "architecture": "UNet", "channels": ["mito"],
                "input_voxel_size_nm": [16] * 3, "output_voxel_size_nm": [16] * 3, "output": "logits",
                "input_norm": [], "io": {"input": "input", "output": "output"}, "variants": variants,
                # A base's block, which the published one replaces.
                "published": {"name": "mito_16nm", "finetune_mode": "head"}}
    return {"repo": REPO, "folder": FOLDER, "published_at": "2026-10-26T12:00:00Z", "manifest": manifest,
            "origin": "trained", "where": "server", "finetuned_from": "cellmap/onnx-web-models/setup_54",
            "finetune_mode": "lora", "epochs": 12, "final_loss": 0.042, "trained_at": "2026-10-26T10:00:00Z",
            "app_version": "0.3.0", "dataset": None, **changes}


def test_a_server_publish_is_a_card_and_a_manifest_naming_every_variant_it_uploads():
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
    assert card.startswith("---\nlibrary_name: onnx\n")
    assert all(f"{FOLDER}/mito_{v}.onnx" in card for v in ("fp32", "fp16", "int8w"))
    assert f"({publish.PUBLIC_APP}?model={REF})" in card
    assert "dataset=" not in card


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
