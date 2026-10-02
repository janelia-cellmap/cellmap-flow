"""The finetune tab's routes, and the viewer-layer routes scripts use.

Importing the package puts every route on ``finetune_bp``, which
dashboard.app registers. Each module has its own:

- ``training``: submit, restart, the job list, a job's status and log (and
  its live stream), cancel and stop-early;
- ``annotation_core``: the models to annotate for, a new annotation volume,
  and the user's saved settings;
- ``annotation_sessions``: listing earlier sessions and resuming one;
- ``yaml_crops``: importing crops from a YAML, and reading a YAML file;
- ``overlay``: an annotation volume's layer, the annotated-regions boxes,
  and syncing the annotations from MinIO;
- ``good_regions``: marking the current view as a region the model gets right;
- ``view_labels``: labelling the view from the model's prediction, or all
  background;
- ``instance_correction``: instance-correction volumes;
- ``layers``: adding, removing and renaming viewer layers.
"""

from cellmap_flow.dashboard.routes.finetune import (  # noqa: F401  (each adds its routes)
    annotation_core,
    annotation_sessions,
    good_regions,
    instance_correction,
    layers,
    overlay,
    training,
    view_labels,
    yaml_crops,
)
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp

__all__ = ["finetune_bp"]
