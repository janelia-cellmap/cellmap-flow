"""Starting and talking to inference servers.

The server itself is ``cellmap_flow.server.CellMapFlowServer``.

- ``launch``: the command line that starts a server for a model.
- ``protocol``: what a layer URL carries to a server, and the markers a
  server prints its address between.
- ``virtual_zarr``: the zarr metadata and chunk keys a server answers with.
- ``client``: asking a running server about its model (``model_info``).
- ``probe``: what a model's output range says about the postprocessing it
  needs.
- ``restart_token``: the secret a finetune job's server takes a restart with.
"""
