"""Starting and talking to inference servers.

The server itself is ``serving.chunkmirage_server.ChunkmirageServer``; the
model and its chains run inside it as chunkmirage ops
(``serving.chunkmirage_ops``).

- ``engine``: ``make_server``, which every place that starts a server
  builds it with, and ``check_server``.
- ``launch``: the command line that starts a server for a model.
- ``protocol``: what a layer URL carries to a server (its chain, and the
  dataset it shows), the markers a server prints its address between, and
  ``DATASET_CONTROL_PATH``, where a running server is asked to open another
  dataset and serve it, its model still loaded.
- ``client``: asking a running server about its model (``model_info``), and
  to switch datasets (``switch_dataset``).
- ``probe``: what a model's output range says about the postprocessing it
  needs.
- ``restart_token``: the secret a server's control routes take: a finetune
  job's restart and stop, and every server's dataset switch.
"""
