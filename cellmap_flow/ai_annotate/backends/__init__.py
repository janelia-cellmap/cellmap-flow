"""The providers a config may name, behind one interface (``base.Backend``).

``get_backend`` builds the provider a config entry describes. Adding a
provider type is a module here, a branch in ``get_backend``, and its
options in ``config.PROVIDER_OPTIONS``.
"""

from cellmap_flow.ai_annotate.config import DEFAULT_TIMEOUT_S, DEFAULT_VERTEX_LOCATION
from cellmap_flow.ai_annotate.errors import AIAnnotateError


def get_backend(provider):
    """The backend for a ``config.ProviderConfig``; a config error for an unknown type."""
    if provider.type == "vertex_gemini":
        from cellmap_flow.ai_annotate.backends.vertex_gemini import VertexGeminiBackend

        return VertexGeminiBackend(
            project=provider.options.get("project"),
            location=provider.options.get("location", DEFAULT_VERTEX_LOCATION),
            timeout_s=provider.options.get("timeout_s", DEFAULT_TIMEOUT_S),
            credentials_file=provider.options.get("credentials_file"),
        )
    if provider.type == "fake":
        from cellmap_flow.ai_annotate.backends.fake import FakeBackend

        return FakeBackend()
    raise AIAnnotateError("config", "The AI-annotate config names a provider type this dashboard does not know.")
