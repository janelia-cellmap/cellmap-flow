"""What the browser calls to publish a model to Hugging Face: the functions
``web/src/backend/routes/publish.ts`` runs through chunkmirage's
``Engine.call("cellmap_flow.portable.publish_calls:<name>", args, arrays)``
(``calls.py`` says the shape). Each takes JSON and answers JSON; what they
answer is ``publish.py``'s, which the server's publish route calls in its
process, so a model published from either mode has the same card, the same
manifest and the same links.
"""

from cellmap_flow.portable import publish


def publication(args, arrays):
    """The model card and the manifest of the publish ``args["pub"]``
    (``publish.publication_files``): ``[{path, text}]``, in the order they
    are uploaded, before the model's own file."""
    return publish.publication_files(args["pub"])


def published_links(args, arrays):
    """``publish.published_links``: where the model ``folder`` of ``repo``
    is (its ref and its repository's page) and the links that open it in
    the app at ``app`` with ``dataset`` (or null)."""
    return publish.published_links(args["repo"], args["folder"], args["app"], args.get("dataset"))
