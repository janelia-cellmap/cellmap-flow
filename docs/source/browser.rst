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

A model is named as the Models tab lists it (``setup_53``), by its
repository path (``cellmap/onnx-web-models/setup_53``), or, for a model of
another Hugging Face repo, as ``<owner>/<repo>/<folder>``. A link may name
models and no volume (``?model=<owner>/<repo>/<folder>``): they run on the
first volume you open. *Share* in the header
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

Other models come in two ways, from the Models tab's *Add a model*. A
public Hugging Face model repo laid out as cellmap/onnx-web-models is (a
folder per model, holding its ``manifest.json`` and the ``.onnx`` files it
names; a model published from the app is one) is added by its ref,
``<owner>/<repo>/<folder>``, or ``<owner>/<repo>`` when it holds one model,
or its huggingface.co URL. Its models are then listed under Hugging Face
after cellmap's, on later visits too. A model sent as a file is opened with
*Open an .onnx file…*: an ``.onnx`` the app downloaded carries its manifest,
any other is picked with its ``manifest.json``. It is kept in the browser
like a finetuned model (below), and nothing is uploaded.

It is the same dashboard as the server's: the same Models, Input and
Postprocess tabs, the same normalizers and postprocessing steps, a Finetune
tab that paints and trains in the browser, and a Run tab of its own that
saves the predictions over a region to a folder (both below). What is missing
is what a browser cannot do: models that are not ONNX exports (PyTorch,
TensorFlow, DaCapo, BioImage.io, Cellpose), a finetune of a 3D model's whole
network (the 2D models' can train in the browser), crops imported from disk,
the Pipeline Builder and blockwise runs on the cluster. Those are the server
mode's (:doc:`cli`).

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
   and the viewer stays responsive: its chunks go first; reloading the page
   during a run asks first. If a
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
(``F``), as on the server
(`docs/finetuning.md <https://github.com/janelia-cellmap/cellmap-flow/blob/main/docs/finetuning.md>`__).
On the tab's Training panel,
*Start Finetuning* trains in the tab: the model's last layer, its output
convolution, is fitted on the voxels you painted, from the model's own
weights, in seconds on any machine; the job card and the loss plot follow it.
With *Auto-serve* on, the finetuned model takes the layer over as soon as it
is trained. Paint where it is still wrong and train again (*Restart* goes on
from the trained layer, with the form's values as they are then). Once the
job waits for a restart, ``Shift+R`` over the viewer does the same without
asking first, so painting and training again need not leave the viewer; it
says in the tab's logs what it did, or why there was nothing to restart. A
restart's model replaces the one before it, its layer included, and the one
before is deleted from the browser: a job keeps its latest model only, as the
server serves only a job's latest iteration, and a new *Start Finetuning*
makes a new one beside it.

You can also paint before any model is running. *New Volume* then shows a
*Voxel size (nm)* field that offers the dataset's levels and takes any size
you type (one number, or z, y, x); left blank, it uses the level the viewer
shows when you click. The volume lies on the raw's grid at that size, in
64-voxel chunks, with no model recorded, as on the server. *Seed from
Prediction* and *Fit to Paint* work on it with the model picked under
*Prediction from* (or the only one served). *Start Finetuning* refuses it,
because it is on no model's grid. Rank the models on
the paint under *Which model to start from* (below) and click *Use this
model*: that copies the paint onto the chosen model's grid, and you train on
the new volume.

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
model. The models can train their whole network in the browser too (below).

Which model to start from
~~~~~~~~~~~~~~~~~~~~~~~~~

A finetune has least to learn from the model that already gets most of your
paint right, and that is not always the one the volume was made for. The
Training panel's *Which model to start from* ranks the models the browser can
run, those finetuned in it among them, by how well each one's prediction
matches the volume's paint: open the card (it starts closed, and stays as you
leave it), tick the models, press *Rank models*, and each is scored in turn,
a few seconds to half a minute each, and listed best first.

- **Match**, which ranks them, is the share of the painted voxels the model
  gets right, objects and background counting equally, so that paint mostly
  of background does not reward a model that calls everything background:
  such a model matches 50%. Beside it are the share of painted objects it
  calls object and of painted background it calls background.
- Each model is scored at its own cut-off: the threshold on its output that
  fits the paint best. The row's *Threshold* gives it as a probability, as
  *Seed from Prediction* takes it (0.5 is the model's own boundary), with the
  match at 0.5 when that differs. A model that sees the objects but calls too
  much or too little at 0.5 is ranked as it would be used once its threshold
  is set.
- Raw that spans only a few grey levels (``jrc_jurkat-1``'s about 20 of
  0–255) looks almost flat to a model through its usual normalizer. Where
  the raw under the paint spans less than a quarter of the normalizer's
  range, the model is scored again with the normalizer stretched to the
  raw's 1st to 99th percentiles, and that is kept when it matches at least 3
  points better. The row's *Input* gives the grey levels the model read as 0
  to 1: the normalizer's own range, or the stretched one with the match
  unstretched beside it.
- A model is scored on what its own finetune would train on: a nucleus or
  mitochondria model on its channel (a model of several such channels on the
  one that matches best, which the row names), an affinity model on the pairs
  of painted neighbours, the same object or not. Its prediction is read on
  the paint's grid, each painted voxel taking the model's voxel its centre
  lies in, so a model at another voxel size is scored on the same paint.
- Rough strokes are forgiven: for a nucleus or mitochondria model, a painted
  voxel next to one painted the other class is not counted.
- A model finer than the paint (8 or 16 nm under 128 nm strokes) needs a
  block of its own for every few painted voxels, so it is scored under at
  most 32 of its blocks, spread over the paint, and its row says so.
- A model not loaded is loaded for its score alone, at its smallest block
  size, and released after: the Models tab, the viewer and the GPU are left
  as they were. *Show* loads one and adds its layer.

With paint on one object, scores a few points apart are as good as tied, and
the list says so; a little paint in two or three places ranks them more
clearly. On ``jrc_jurkat-1``, with a slice of one nucleus painted (4,430
voxels), ``setup_56`` came first at 62% and ``setup_55``, the volume's own
model, which sees none of the dataset's nuclei, matched 50%. The box above
the list filters it (``nuc``, ``16 nm``, ``affinities``, ``2d``), and only
the ticked models it shows are ranked.

*Use this model* makes a ranked model the one to finetune without painting
again: it is loaded with its layer and picked under *Select Model*, a volume
is made for it, the paint and the good regions are copied into that volume
on the model's own grid, and its paint layer takes the old one's place (the
old volume stays, under *Resume Existing Volume*). What its score chose comes
with it: a stretched raw becomes the session's input normalization (the raw
is the dataset's, so every model reads it so), and its cut-off becomes *Seed
from Prediction*'s threshold, marked *set by ranking* beside the field until
you change it; a line under *Label it*'s buttons says what was set. On the
same grid the paint
is copied voxel for voxel; on a finer one each painted voxel becomes the
voxels it covers; on a coarser one each voxel takes the label of the painted
voxels inside it, and is left unpainted where they disagree (object against
background in one voxel), which the status line counts. A stroke thinner
than the coarser voxel still lands, a voxel wide.

*Fit to Paint*, beside *Seed from Prediction*, does the same for the model
you already seed from (*Prediction from*), with no new volume and nothing
served: it scores that model on your paint as a ranking would, sets *Seed
from Prediction*'s threshold to the one that fits best, and stretches the
input normalization when that scored clearly better. Paint a few strokes of
object and background, fit, then seed: the seed is cut where your paint says.

Training the whole network
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each model's manifest on Hugging Face says how it can be trained, and the
Training panel offers just that. For the 2D students (``setup_53_2d`` and
``setup_54_2d``) and the 3D models (``setup_53``, ``setup_54``, ``setup_55``,
``setup_56``, ``setup_61`` and ``setup_67``) it offers both, so a *Training*
choice appears at the top of the panel: *Last layer*, as above, or *Whole
network*: every weight of the model trains on your paint, with
`TensorFlow.js <https://www.tensorflow.org/js>`_ on this machine's GPU
(WebGPU), from a training graph Marwan Zouinkhi exported beside each model.
The weights trained are written back into the model's ``.onnx``, which is
then served, kept, downloaded and restarted from as a last-layer finetune
is; switching between the two needs a new *Start Finetuning*. A model served
from an int8 file offers neither (its weights are quantized and cannot be
written): serve its fp32 or fp16 file to finetune it.

A 2D model's whole network trains on square tiles cut from the painted
slices, in the plane you painted in (x–y, or z–y and z–x where the raw data's
voxels are cubes, as the server's 2D training does), with Adam; a 3D model's
on cubes around the paint (below). Its controls are the
epochs, its own learning rates (1e-4 by default, a hundredth of the last
layer's), the server's *Batch Size*, in tiles per step (1 by default), *Tile
Size*, *Augmentation*, and the last layer's loss, distillation, smoothing and
balance fields, which it reads as the last layer does. Distillation holds the
model to the starting network's output on the voxels its scope picks in each
tile, worked out on the same tile it trains on. Good regions are not used.

*Augmentation*, off by default as on the server, gives each step's tiles a
random brightness (×0.8 to ×1.2) and noise (1% of the raw's range) before
normalization, then random flips and quarter turns of the image and the paint
together, the server's augmentation. The targets, and the starting network's
output for distillation, are made after the turn, from the turned paint and
image, so an affinity still means its own direction. The whole network sees
the same tiles every epoch, and augmentation varies them, so that it learns
what it is shown rather than how one tile looks; the price is several times
the steps, and with paint on only one or two objects it can stall (see the 3D
models below). The last layer trains on features computed once, and takes no
augmentation.

*Tile Size* is the tile's side in pixels: 172, 300 (the default), 428 or 556.
The model reads 88 pixels of context around what it predicts, so a tile's
output is 88 pixels smaller: half of a 300² tile is output, a quarter of a
172² one, 71% of a 556² one. A smaller tile trains a step faster and in less
memory, but more of each step is context. The hint under the field gives the
time and memory of one step at the batch chosen, as measured in Chrome on an
Apple laptop GPU (TF.js 4.22.0, batch 1) and scaled by area for the sides not
measured:

.. list-table::
   :header-rows: 1

   * - Model
     - 300², a step
     - 300², memory
     - 556², a step
     - 556², memory
   * - setup_53_2d
     - 294 ms
     - 320 MB
     - 1178 ms
     - 1307 MB
   * - setup_54_2d
     - 462 ms
     - 908 MB
     - 1919 ms
     - 3702 MB

300² is the default because it fits a laptop: four times faster than 556²
and a quarter of its memory, which for setup_54_2d is more than many laptop
GPUs give a tab. An epoch is one pass over the tiles, which cover each slice's
paint once. *Whole network* needs a GPU: where the browser finds none
through WebGPU, the option is shown disabled with why (on the CPU a step
would take minutes, and the page would answer nothing else meanwhile). The
log says how long a step takes.

A 3D model trains on cubes instead of tiles: a grid of cubes the size of the
model's output is laid over the painted voxels, and each cube that holds
paint is read with the raw around it, whatever planes you painted in, so
strokes on a few neighbouring slices teach the model along z too (an
affinity model's z channel learns from them). The field is then *Block
Size*, the cube's side, in the sides the model's files come in: 124³, 140³
and 172³ for the UNets (``setup_53``, ``setup_55``), 116³, 132³ and 148³ for
the ResUNets (``setup_54``, ``setup_56``, ``setup_61``, ``setup_67``), the
one it is served at by default. The run trains that side's file, at the
precision served, and the finetune is served from it, so it is served at
the block it trained on: a ResUNet's InstanceNorm normalizes each block by
its own statistics, and a network trained on one size of block and served
at another meets statistics it did not learn on. A *Restart* keeps the
block; another one is a new job, as the trained network is in its file. A
model finetuned here trains at its own block only. The model
reads 88 voxels of context on each axis, so a 124³ block's output is 36³,
2.4% of it, and a block costs its volume: 172³ takes about three times the
time and memory of 124³. A 3D model trains one block a step, so *Batch Size*
is held at 1. *Augmentation* flips a block along z, y and x and turns it
only in the x–y plane. The network is trained in less than a third of the
memory TensorFlow.js's own gradients would take (its activations recomputed
rather than kept), and the hint gives the time and memory of one step as
measured on an NVIDIA RTX 2080 Ti (11 GB), the time with distillation's
forward pass of the starting network:

.. list-table::
   :header-rows: 1

   * - Model
     - 124³, a step
     - 124³, memory
     - 172³, a step
     - 172³, memory
   * - setup_53, setup_55 (UNet)
     - 2.8 s
     - 2.2 GB
     - 8.2 s
     - 5.3 GB
   * - setup_54, setup_56, setup_61, setup_67 (ResUNet)
     - 4.0 s
     - 4.0 GB
     - 11.9 s
     - 9.9 GB

A block that does not fit the GPU's memory stops the run with a message
naming a smaller *Block Size* to try; where even 124³ does not fit, train the
last layer, which needs no more than inference.

A run is a number of steps, epochs × blocks (its first line gives the
blocks), and a 3D model's whole network wants 60 to 150 of them: *Number of
Epochs* goes up to 500 for paint that lies in one or two blocks. In a test on
``jrc_jurkat-1`` with ``setup_55``, which sees none of its nuclei to start
with, one painted slice of one nucleus (4 blocks) took it to 98% of that
nucleus and 62% of the ten it was not shown in 300 steps, scored against
OpenOrganelle's nucleus segmentation; painting the whole slice rather than a
ring of background around the nucleus cut the background it called nucleus
from 18% to 6.5%. With *Augmentation* on, a ring around the same nucleus held
the loss between 0.63 and 0.68, next to guessing, through 500 steps; turn it
on once several objects are painted, with 3 to 5 times the epochs. The epoch
lines' loss reads
against two marks: 0.69 for the default class-balanced BCE is guessing, and
about 0.20 with label smoothing of 0.1 is the paint fitted exactly.

The *Patch on screen* tools label the box centred where the viewer looks, as
on the server (`docs/finetuning.md <https://github.com/janelia-cellmap/cellmap-flow/blob/main/docs/finetuning.md>`__):
*Seed from Prediction* copies a model's
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
Models tab lists the finetuned models under the Hugging Face ones. The ✕
beside a finetuned model under *Loaded models* (see *Usage* on the
:doc:`front page <index>`) stops serving it and keeps it, to tick again.
*Download paint* saves the volume as a zarr (zipped, the server's layout, so a server
session can train on it) and *Download finetuned model* the ``.onnx``, with
its manifest (voxel sizes, block, channels, input normalization) written into
the file's metadata, so that the server mode serves the file as it is and
another browser opens it with *Open an .onnx file…*.
Nothing leaves your machine unless you download and share it. *Share* puts a
finetuned model in its link as its last layer's numbers, rebuilt from its
base where the link opens; one whose whole network was trained, or that was
opened from a file, goes in a link only once it has been published to
Hugging Face, where the link then names it.

To run the same models on the cluster, *Export Config* (in the header) saves
a YAML that ``cellmap_flow yaml`` runs, each model a ``type: onnx`` entry; a
finetuned model's ``.onnx``, from *Download finetuned model*, goes beside it.
See :ref:`onnx` for what the YAML holds and what to add for the cluster.

Publishing a finetuned model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Publish to Hugging Face*, beside *Download finetuned model*, puts the model
(its last layer or its whole network finetuned) in a repository of your own
Hugging Face account, uploaded from the page when you click *Publish*, where
anyone can run it: in this app, in the server
dashboard, or from the files. Sign in with Hugging Face, in a popup that
closes itself (the page keeps everything it has open), or paste an access
token with write access: at
`huggingface.co/settings/tokens <https://huggingface.co/settings/tokens>`__,
*Create new token* of type *Write* (or a fine-grained one that may write to
your repositories). Either is kept in the page's memory only and sent only to
Hugging Face; a reload asks again.

The repository is ``<you>/<model name>`` unless you type another, yours or one
of your organizations'. A new repository is **public** unless you tick
*Private*; one that exists already gets the files as a new commit, its
visibility as it was, so publishing a model again updates it. The upload is
the model card (``README.md``: what the model predicts, at what voxel sizes,
what it was finetuned from and how, and how to open it), the model's manifest
(``<model>/manifest.json``, as cellmap-flow reads a model's geometry, channels
and input normalization) and its ``.onnx``. The model card names the dataset
it was trained on only when you tick *Name the dataset's URL in the model
card*: a dataset's URL can be internal.

When it is done the panel shows the model's Hugging Face page, its ref
(``<you>/<repo>/<model>``, which *Add a model* takes in this app and on the
server) and two links, each with *Copy*: *Try it in the browser* opens the
app with the model chosen, for whatever volume one opens, and *Open on this
dataset* opens it on the dataset you have open. The model card carries the
first always, and the second only with the dataset named.

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
