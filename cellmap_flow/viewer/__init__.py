"""The neuroglancer viewer: making one, and the layers it shows.

- ``bootstrap``: ``new_viewer()``, the one place a viewer is made.
- ``layers``: a model's prediction layer and the raw data's layer, built the
  same way by every path that shows one.

Nothing here reads the dashboard's state or starts the dashboard; callers
pass in what a layer needs.
"""
