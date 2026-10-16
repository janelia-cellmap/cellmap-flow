In the browser
==============

`janelia-cellmap.github.io/cellmap-flow/app <https://janelia-cellmap.github.io/cellmap-flow/app/>`_
is cellmap-flow with nothing to install: the dashboard as a web page, with the
models running in your browser. Open a volume, tick a model, and look at its
predictions in Neuroglancer as you move around. It is the main place to try
cellmap-flow, and the first place to test a change: every push to ``main``
deploys it (GitHub Pages, from ``.github/workflows/pages.yml``).

`Open this example <https://janelia-cellmap.github.io/cellmap-flow/app/?dataset=https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8&model=setup_53>`_ to start from something running: an
OpenOrganelle mouse-liver volume with the mitochondria model ``setup_53`` on
it. A link's query names the volume and the models, so any such pair can be
shared::

    https://janelia-cellmap.github.io/cellmap-flow/app/?dataset=<url>&model=<name>&model=<name>

A model is named as the Models tab lists it (``setup_53``) or by its
repository path (``cellmap/onnx-web-models/setup_53``). *Share* in the header
copies such a link to what you have open, with the view (position, zoom,
each layer as shown) and the Input and Postprocess settings the models run
with, for someone else to open. The address bar itself stays bare, so a new
tab starts fresh.

How it works
------------

The page reads the raw volume straight from its URL, runs the model with
`onnxruntime-web <https://onnxruntime.ai/>`_ on your machine's GPU through
WebGPU (or on its CPU through WebAssembly when there is none), and serves the
predictions to Neuroglancer as a zarr array through a service worker, chunk by
chunk, as the viewer asks for them. The models are ONNX exports of cellmap's
networks from `cellmap/onnx-web-models <https://huggingface.co/cellmap/onnx-web-models>`_
on Hugging Face, cached in the browser after the first download.

It is the same dashboard as the server's: the same Models, Input and
Postprocess tabs, the same normalizers and postprocessing steps, a Finetune
tab that paints and trains in the browser, and a Run tab of its own that
saves the predictions over a region to a folder (both below). What is missing
is what a browser cannot do: models that are not ONNX exports (PyTorch,
TensorFlow, DaCapo, BioImage.io, Cellpose), a finetune of a whole network,
crops imported from disk, the Pipeline Builder and blockwise runs on the
cluster. Those are the server mode's (:doc:`cli`).

Opening a volume
----------------

**A URL.** Paste the URL of an OME-Zarr group or array served with CORS
(``Access-Control-Allow-Origin``). Public S3 and GCS buckets are, such as
`OpenOrganelle <https://openorganelle.janelia.org>`_'s; the connect panel's
placeholder is one of them.

**A folder on your machine.** In Chrome or Edge, *Browse local folder…* on the
connect panel opens a directory picker, and the panel takes an OME-Zarr, zarr
or N5 folder dropped on it from the file manager. Nothing is uploaded: the
page reads the files itself. The browser asks once to allow reading the
folder, and again after a reload. Firefox and Safari have no directory picker.

**A volume on your institution's network.** A host with a private address
(``10.x``, ``192.168.x``, a ``.local`` name) opens as it does in
Neuroglancer. To Chrome such a host is on your *local network*, which a
public site such as this one may use only with your permission; if Chrome
asks ("wants to look for and connect to devices on your local network"),
allow it. Until it is allowed, connecting fails with *Failed to fetch*, and
the connect panel says so; the setting is under *Local network access* in
the site-information panel beside the address bar (or at
``chrome://settings/content/localNetworkAccess``). Running the app locally
on ``http://localhost`` (below) is not gated.

Running the models over a region
--------------------------------

The viewer computes the chunks it shows, and keeps none of them. The *Run*
tab runs the models served on the Models tab over a region, or the whole
volume, on your GPU, and saves each one's predictions as an OME-Zarr in a
folder on your machine, to open in the app again or to copy to the cluster.

1. Tick the models (all are ticked at first). Each runs through the Input and
   Postprocess steps it is served with, so what is saved is what its layer
   shows.
2. *Draw a box* adds a ``run_region`` layer to the viewer. Hold ``Ctrl``
   and click two opposite corners of the box in the viewer, scrolling
   between the two clicks to give it depth (a flat box is one chunk
   deep). The latest box drawn fills the *Offset* and *Size* fields (nm, z y
   x), and typing in them moves the box in the viewer to match, once you
   pause. *Whole volume* fills them with the volume's
   extent instead. Under the fields, the tab says what each model would
   compute: the box in its voxels and in chunks (a chunk is one block the
   model computes at a time), with a warning for a run of more than 2,000
   chunks.
3. *Choose folder…* picks the folder to save into, or drop a folder on the
   *Output folder* block; the browser asks you to allow the page to edit
   files in it. Chrome refuses some folders themselves, such as Downloads
   and your home folder ("it contains system files"): make a new folder
   inside one (the picker has a *New Folder* button) and choose that. The
   folder is remembered, and after a reload the browser asks again when you
   click *Run*.
4. *Run* writes each model to ``<folder>/<model>.zarr`` with a progress bar
   (chunks, megabytes, time left) and *Cancel*, which keeps what was written.
   The run goes on while you move around the viewer or use the other tabs,
   and the viewer stays responsive: its chunks go first. If a
   ``<model>.zarr`` is in the folder already, the tab asks before replacing
   it (it is deleted first, so the old chunks do not mix with the new ones).
5. *Show the saved predictions in the viewer* adds each saved zarr as a
   ``<model> (saved)`` layer, read back from the folder, over the model's live
   layer.

Each zarr is zarr v2 with OME-Zarr 0.4 metadata, gzip-compressed (or not,
with *Compression* set to *None*: Fiji's zarr reader has no gzip), holding
exactly the region: an array of the region's voxels, not the whole volume's,
at the voxel size of the model's layer and placed where the layer is by its
translation (empty chunks are left out), so Neuroglancer, zarr-python and
the server mode read it as they read any OME-Zarr. Its attributes, under
``cellmap_flow``, say what made it: the model and its variant, the
normalizers and postprocessing, the volume's URL, the region in nm and the
box in voxels, and when. Saving to a folder needs Chrome or Edge (the File
System Access API); in Firefox and Safari a run of up to 256 MB of
predictions is kept in memory and offered as a zip of the ``.zarr`` folders.

Finetuning in the browser
-------------------------

With a model running, the Finetune tab's *New Volume* adds a paint layer to
the viewer, selected, on its Draw tab: paint background (value 1) and the
objects (2 and up) on a few slices with the brush (``A``) and flood fill
(``F``), as on the server (:doc:`finetuning`). On the tab's Training panel,
*Start Finetuning* trains in the tab: the model's last layer, its output
convolution, is fitted on the voxels you painted, from the model's own
weights, in seconds on any machine; the job card and the loss plot follow it.
With *Auto-serve* on, the finetuned model takes the layer over as soon as it
is trained. Paint where it is still wrong and train again (*Restart* goes on
from the trained layer, with the form's values as they are then). Once the
job waits for a restart, ``Shift+R`` over the viewer does the same without
asking first, so painting and training again need not leave the viewer; it
says in the tab's logs what it did, or why there was nothing to restart. A
restart's model replaces the one before it: the Models tab keeps one model
per job, the latest, and a new *Start Finetuning* makes a new one beside it.

The training panel's controls are the server's where they mean something for
one layer: the epochs, the learning rate, the loss (BCE by default, MSE or
margin; Dice scores whole patches and needs the server), label smoothing,
class balance, and *Distillation Weight* and *Scope*, which keep the layer
close to the model's own output where you painted nothing: the loss adds the
weight times the mean squared distance of the layer's logits from the
original's, on the unpainted voxels of the painted chunks (or all of their
voxels), up to 32,768 of them. 0.01, the default, is a light hold; 0.1 to 1
holds firmly. *Voxels per Step* is the browser's own batch: how many painted
voxels each step of the training averages over. *Mark as Good*, under *Patch
on screen*, marks the output block where the viewer looks as one the model
already gets right; it shows as the ``good_regions`` box layer and is kept
with the volume. Once there are good regions, distillation holds the model to
its own output inside them, wherever you have not painted, instead of over
the scope, as on the server; *clear* removes them.

What this can and cannot fix. The rest of the network stays as exported, so
the finetune learns what the model mislabels where its features already tell
the two apart (a membrane it calls mitochondrion, a texture it misses), not
new shapes; that is a finetune of the whole network, which the server mode
runs on the cluster, and whose result the browser can serve as any other
model.

The *Patch on screen* tools label the box centred where the viewer looks, as
on the server (:doc:`finetuning`): *Seed from Prediction* copies a model's
prediction into the unpainted voxels, an id per object and background 1, for
you to fix with the brush; *All Background* labels them background; *Split
Objects* gives each connected object an id of its own (paint a background wall
through a merge first); *Undo* takes back the last of these, or of an accepted
AI annotation, up to 10. *Prediction from* lists the models served on the
Models tab, the ones finetuned here among them (default: the volume model's
latest finetune), and the seed reads the prediction the page has already
computed for the viewer. Its methods are *Threshold + components* for any
model and *Mutex watershed* for an affinity model; the server's *Distance
watershed* and *Model's instances* fit outputs no browser model has.

AI-assisted annotation is in the tab too, under *Patch on screen*: hover over
a structure and press ``Shift+G`` (or click *Annotate at View Centre*), and
one plane of raw EM goes to a Gemini image model asked to paint it; the
review opens over the page, *Accept* writes the mask into the paint volume as
an object per piece, and the tools' *Undo* takes it back. The page calls Google itself, with
credentials you give it in the panel: a Gemini API key, or a Google Cloud
project and an access token for Vertex AI (``gcloud auth print-access-token``,
good for about an hour). They stay in this browser and go to Google alone,
with each plane, and are kept in the page's memory only, never in the
browser's storage, so a reload asks for them again. On the Gemini API's free tier Google may use what is
sent to improve its products, on its paid tier and on Vertex AI it does not;
the server mode uses Vertex AI for that reason. A *fake* provider paints in
the page, for trying the panel without a key. The server's guide,
`docs/ai_annotate.md <https://github.com/janelia-cellmap/cellmap-flow/blob/main/docs/ai_annotate.md>`_,
has the details under "In the browser".

The paint and the finetuned models stay in the browser (IndexedDB): *Resume
Existing Volume* lists the volumes painted here, newest first, and the
Models tab lists the finetuned models under the Hugging Face ones. *Download
paint* saves the volume as a zarr (zipped, the server's layout, so a server
session can train on it) and *Download finetuned model* the ``.onnx``, with
its manifest (voxel sizes, block, channels, input normalization) written into
the file's metadata, so that the server mode serves the file as it is.
Nothing leaves your machine unless you download and share it.

To run the same models on the cluster, *Export Config* (in the header) saves
a YAML that ``cellmap_flow yaml`` runs: each model a ``type: onnx`` entry
(:ref:`onnx`), a Hugging Face export by its repo, folder and the file this
browser ran, a finetuned one by ``path: <name>.onnx``, the file *Download
finetuned model* saves, which goes beside the YAML (a relative path is the
YAML's folder's), and the Input and Postprocess steps as its ``json_data``.
Add ``charge_group:`` and ``queue:`` for the cluster:

.. code-block:: yaml

    models:
    - type: onnx
      name: mito_aff_setup54_16nm
      repo: cellmap/onnx-web-models
      folder: setup_54
      file: setup_54_mito_affinities_16nm_in132_fp16.onnx
    - type: onnx
      name: mito_aff_setup54_16nm-ft1
      path: mito_aff_setup54_16nm-ft1.onnx  # the file Download finetuned model saves: put it beside this YAML

Browsers and devices
--------------------

- **WebGPU.** Chrome or Edge (any current version), or Safari 26 and later.
  Without WebGPU the page still works, on the CPU: tens of seconds per chunk
  instead of well under one.
- **Phones.** The viewer fills the screen and the *Dashboard* button in the
  header opens the controls over it.
- **Memory.** Model files are 11–43 MB each, kept in IndexedDB; the page
  picks a smaller variant (fp16, int8 weights) when the GPU's limits call for
  it.

Running it locally
------------------

The browser mode lives in the repository's ``web/`` directory; its
`README <https://github.com/janelia-cellmap/cellmap-flow/blob/main/web/README.md>`_
says how to build and run it (``npm run dev``, then ``http://localhost:5173/``:
the address must be ``localhost`` or ``https://`` for WebGPU) and how its
parts fit together. The in-browser engine is
`chunkmirage <https://github.com/JaneliaSciComp/chunkmirage>`_'s, and the
browser mode itself is a port of Marwan Zouinkhi's serverless prototype of
cellmap-flow.
