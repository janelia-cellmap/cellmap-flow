"""The blueprint every finetune route is on.

Each module of this package adds its own routes to ``finetune_bp`` when it
is imported, and the package's ``__init__`` imports them all before
dashboard.app registers the blueprint. So a module imports the blueprint
from here, never from the package, which is still being imported then.
"""

from flask import Blueprint

finetune_bp = Blueprint("finetune", __name__)
