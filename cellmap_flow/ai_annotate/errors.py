"""The one error type AI-assisted annotation raises, and what the user is shown.

A hosted model's SDK raises exceptions whose text can hold request details,
response bodies or, for a key-based provider, the key itself. The dashboard
shows errors in the browser, so it must never show ``str(e)`` from an SDK.
Everything in this package raises ``AIAnnotateError`` instead: a category the
routes turn into an HTTP status, and a short ``user_message`` written here,
safe to show. The SDK's own detail goes to the server log only, where the
redacting filter (``secrets``) has already removed registered secrets.
"""

CATEGORIES = ("config", "auth", "quota", "timeout", "bad_response", "unavailable", "refused", "limit")

# What each category answers when the raiser does not say otherwise. 502 for
# auth and bad_response: the dashboard is fine, the upstream provider is not.
_DEFAULT_STATUS = {
    "config": 500,
    "auth": 502,
    "quota": 429,
    "timeout": 504,
    "bad_response": 502,
    "unavailable": 503,
    "refused": 403,
    "limit": 429,
}

INSTALL_HINT = (
    "The google-genai package is not installed in the dashboard's environment: "
    "run `pixi install -e default`, or `pip install cellmap-flow[ai-annotate]`."
)


class AIAnnotateError(Exception):
    """A failure the user can be told about, as a category and a safe message.

    ``str()`` of it is the user message too, so a caller that logs or returns
    ``str(e)`` by habit still shows nothing unvetted. ``http_status``
    overrides the category's default, e.g. 400 for a request naming a model
    the config does not allow.
    """

    CATEGORIES = CATEGORIES

    def __init__(self, category, user_message, *, http_status=None):
        if category not in CATEGORIES:
            raise ValueError(f"unknown AIAnnotateError category {category!r}")
        super().__init__(user_message)
        self.category = category
        self.user_message = user_message
        self.http_status = http_status if http_status is not None else _DEFAULT_STATUS[category]
