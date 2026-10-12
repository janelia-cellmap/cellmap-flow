"""AI-assisted annotation: a hosted model proposes a mask for one plane, the user reviews it.

In the Finetune tab the user points at a structure and asks for it to be
annotated; the dashboard sends that plane of raw EM to a model with a prompt
asking it to paint the structure, turns the reply into a mask, and stages it
for the user to accept into the annotation volume or reject.

- ``config``: the server-side config that turns the feature on and lists
  the providers and models it may use. Off without it.
- ``errors``: ``AIAnnotateError``, a category and a message safe to show.
- ``secrets``: redacting secrets from logs, and reading API keys safely.
- ``usage``: the per-user daily call limit.
- ``images``: checking a model's image before decoding it; PNGs for the browser.
- ``organelles``, ``prompts``: what the model is told about each structure.
- ``mask_extraction``: EM plane to RGB, and the model's reply to a mask.
- ``backends``: the providers (Vertex Gemini, a local fake) behind one interface.
- ``geometry``, ``pipeline``, ``staging``, ``audit``: which plane to read and
  where its mask goes, staging the result for review, and the audit log.

Import the submodule you need; this package imports none of them, so the
SDKs a provider needs load only when a call is made.
"""
