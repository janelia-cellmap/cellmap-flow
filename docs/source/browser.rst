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
Postprocess tabs, the same normalizers and postprocessing steps, and a
Finetune tab that paints and trains in the browser (below). What is missing
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
from the trained layer).

What this can and cannot fix. The rest of the network stays as exported, so
the finetune learns what the model mislabels where its features already tell
the two apart (a membrane it calls mitochondrion, a texture it misses), not
new shapes; that is a finetune of the whole network, which the server mode
runs on the cluster, and whose result the browser can serve as any other
model.

AI-assisted annotation is in the tab too, under *Patch on screen*: hover over
a structure and press ``Shift+G`` (or click *Annotate at View Centre*), and
one plane of raw EM goes to a Gemini image model asked to paint it; the
review opens over the page, *Accept* writes the mask into the paint volume as
an object per piece, *Undo* takes it back. The page calls Google itself, with
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
session can train on it) and *Download finetuned model* the ``.onnx``.
Nothing leaves your machine unless you download and share it.

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
