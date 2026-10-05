# AI-Assisted Annotation

This guide covers the **AI annotate** panel in the dashboard's Finetune tab. It sends one 2D plane of your EM data to a hosted image model, which paints the structure you ask for. You review what it painted, and if you accept it, it is written into your annotation volume as labels. It is a quicker start than painting from nothing, in the same way as **Seed from Prediction** (see [the finetuning guide](finetuning.md#label-the-patch-on-screen-in-one-click)).

The feature is **off until you turn it on** with a config file (see [Turning it on](#turning-it-on)), because using it sends image data off the cluster to an outside company. Read [What data leaves the cluster](#what-data-leaves-the-cluster-and-where) before you turn it on.

## What it does

With an annotation volume open in the Finetune tab (see [Create or Resume an Annotation Volume](finetuning.md#2-create-or-resume-an-annotation-volume)):

1. In the **AI annotate** panel, pick the structure to label (mitochondria, ER, nucleus, ...) and the model. The first time for a dataset, tick the box that acknowledges where the data goes.
2. Hover over the structure in the viewer and press **Shift+G**, or click **Annotate at view centre** to use the centre of the view instead.
3. The dashboard reads one plane of raw EM around that point, in the plane you are looking at, and sends it to the model with a prompt asking it to paint the structure in a colour. It turns the colours the model painted into a mask. This takes from a few seconds to a minute or two.
4. The panel shows three images side by side: the **input** plane, the **model's output**, and the mask as an **overlay** on the input.
5. Then:
   - **Accept** writes the mask into the annotation volume (see [Accept: fill or overwrite](#accept-fill-or-overwrite)).
   - **Reject** throws it away. Nothing is written.
   - **Resend** asks the model again with the same plane, optionally with an edited prompt (see [Editing the prompt](#editing-the-prompt)). Image models answer differently each time, so a second try is often better.
6. **Undo** in the **Patch on screen** panel takes back an accepted mask, like any other fill there.

Only one request can be in flight or waiting for review at a time: accept or reject the one you have before asking for another.

### What gets written

The mask covers one plane, one annotation voxel thick, `crop_size_px` annotation voxels on a side (512 by default), centred on the point you chose and clipped to the volume. Within that plane, each connected piece of the mask becomes an object with its own id (2 and up) and every other voxel becomes background (1), as with **Seed from Prediction**. Check the result, then fix it with the brush as usual.

### Choosing the plane

The plane is the one the viewer's single panel is showing. cellmap-flow's viewer orders its axes z, y, x, and neuroglancer names its layouts after display slots rather than data axes, so the layout names do not match the planes:

| Viewer layout | Plane sent |
|---|---|
| **yz** (single panel) | XY, at the hovered z |
| **xz** (single panel) | XZ, at the hovered y |
| **xy** (single panel) | YZ, at the hovered x |
| **4panel**, **3d** or anything else | the top-left panel's plane (YZ) |

Neuroglancer keeps one layout for the whole viewer, so in the 4-panel layout the server cannot tell which panel the mouse is over: switch to a single-panel layout to choose the plane. The panel's status line names the plane that was sent (XY, XZ or YZ), so check it before accepting.

## Turning it on

The feature reads a config file on the machine the dashboard runs on. With no file, the panel says the feature is off and points here.

The file is `~/.cellmap_flow/ai_annotate.yaml`, or the path in the environment variable `CELLMAP_FLOW_AI_ANNOTATE_CONFIG` if that is set. An example with every option is in [`docs/examples/ai_annotate.yaml`](examples/ai_annotate.yaml). The shortest one that works:

```yaml
enabled: true
providers:
  vertex:
    type: vertex_gemini
    project: my-gcp-project
    location: global
    models: [gemini-3-pro-image]
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | — | `true` to turn the feature on. `false` turns it off without deleting the file. |
| `providers` | — | The models the dashboard may call, by an id you choose (`vertex` above). The browser can only pick among these; it cannot name another service or endpoint. |
| `default_provider` | the first provider | Which provider the panel starts on. |
| `daily_call_limit` | `200` | The most model calls you can make in a day (see [Limits](#limits)). |
| `crop_size_px` | `512` | The plane's size on a side, in annotation voxels. |
| `allowed_dataset_prefixes` | `[]` (any dataset) | If not empty, only datasets whose path starts with one of these can be sent. |

Each provider has:

| Key | Default | Meaning |
|---|---|---|
| `type` | — | `vertex_gemini` (Google Cloud Vertex AI) or `fake` (no network; for trying the panel out and for tests). |
| `models` | — | The model ids the panel may offer, e.g. `[gemini-3-pro-image]`. For `fake`: `[fake-threshold]`. |
| `project` | `$GOOGLE_CLOUD_PROJECT` | Vertex only: the Google Cloud project that is billed. |
| `location` | — | Vertex only: use `global` (see below). |
| `timeout_s` | `120` | How long to wait for one answer before giving up. |

The file must not contain any API key or password: a key written in it (`api_key:`) is refused as a config error. See [API keys](#api-keys).

The dashboard reads the file when the panel asks for it, so after editing it, reload the dashboard page.

### Trying it without sending anything

To see how the panel works before setting up Google Cloud, use the fake provider. It paints the darker pixels in a disc around the point you chose, entirely on the dashboard's machine:

```yaml
enabled: true
providers:
  fake:
    type: fake
    models: [fake-threshold]
```

## Vertex AI setup

Vertex AI is Google Cloud's service for its Gemini models. You need a Google Cloud project with billing enabled; ask whoever manages your lab's Google Cloud account which project to use.

1. **Install the Google Cloud CLI** (`gcloud`) if you do not have it: <https://cloud.google.com/sdk/docs/install>.
2. **Log in**, on the machine the dashboard runs on, as the user it runs as:

   ```bash
   gcloud auth application-default login
   ```

   This opens a browser to sign in with your Google account, and saves a credential file at `~/.config/gcloud/application_default_credentials.json`. The dashboard uses that file; there is no API key to copy anywhere. Treat it like a password: it lets anyone who can read it use your Google Cloud account, so keep your home directory private.
3. **Choose the project**: put it in the config as `project:`, or set the environment variable `GOOGLE_CLOUD_PROJECT` before starting the dashboard. If gcloud warns about a quota project, set it to the same one:

   ```bash
   gcloud auth application-default set-quota-project my-gcp-project
   ```

4. **Enable the Vertex AI API** in that project (once per project):

   ```bash
   gcloud services enable aiplatform.googleapis.com --project my-gcp-project
   ```

5. **Use `location: global`.** The gemini-3 image models are only served from Google's global endpoint; in a regional location such as `us-central1` the call fails with "404 not found". See below for what `global` means for where your data is processed.

The Python package the dashboard needs, `google-genai`, is in the pixi environments (`pixi install`). In a conda or pip install, add it with:

```bash
pip install -e ".[ai-annotate]"
```

Without it, the panel says the feature is unavailable and gives this command.

## What data leaves the cluster and where

Each call (each Shift+G, button click or Resend) sends:

- **one 2D plane of raw EM**: the region you chose, as a greyscale image of at most 1024 × 1024 pixels (a larger plane is shrunk before sending), and
- **the prompt text**: the instructions for the model, which name the structure, the colour to paint it, and the image's resolution in nm.

Nothing else is sent: not the rest of the dataset, not its path or name, not your annotations. The model sends back one image and sometimes a line of text.

**Where it goes.** With the `vertex_gemini` provider the plane goes to Google Cloud's Vertex AI, in the project from the config:

- Google's terms for Vertex AI say it does not use customer data to train its models. That is why this feature uses Vertex AI and not the free Gemini API from Google AI Studio, which is **not supported**: on its free tier, Google may use what you send to improve its products.
- `location: global` means Google may process the request in **any region** where it runs the model, not one country you choose. If your data may not leave a particular jurisdiction, do not use this provider.
- Your data still leaves Janelia's cluster and is handled by an outside company. If the dataset is unpublished, under a data use agreement, or from human tissue, check that sending it is allowed before you turn the feature on.

**Safeguards in the dashboard:**

- **Acknowledgement.** Before the first call for a dataset, the panel shows where the data goes (for example "Google Cloud Vertex AI, location global (Google may process data in any region)") and asks you to tick a box. The server refuses calls until you have, separately for each provider and each dataset, and again after the dashboard restarts.
- **`allowed_dataset_prefixes`.** To make sure only some data can ever be sent, list the path prefixes that are allowed. Any other dataset is refused, whatever is ticked:

  ```yaml
  allowed_dataset_prefixes:
    - /nrs/cellmap/data/jrc_mus-liver
    - s3://janelia-cosem-datasets/
  ```

- **`daily_call_limit`.** See [Limits](#limits).

## API keys

Vertex AI uses the login from `gcloud auth application-default login` and needs no API key. Other providers may be added later that do (an OpenAI-compatible service, for example). For those, the rules are:

- **Never put a key in the config file.** Config files get copied, shared and committed; a key in one leaks. A provider with `api_key:` in it is refused.
- Instead, either name an **environment variable** that holds the key:

  ```yaml
  api_key_env: MY_PROVIDER_API_KEY
  ```

  or a **file** that holds it, readable only by you:

  ```yaml
  api_key_file: ~/.cellmap_flow/my_provider.key
  ```

  ```bash
  chmod 600 ~/.cellmap_flow/my_provider.key
  ```

  The file must be owned by you and not readable by your group or anyone else, or the config is refused.
- The dashboard reads the key only when it makes a call, and never writes it to the staging folder, the audit log or anything it sends to the browser.
- **Keys are removed from the logs.** The dashboard streams its server log to the browser (the log panel), so every key the dashboard has read is replaced with `[REDACTED]` before a log line is written. Error messages shown in the panel are short summaries of what went wrong, never the service's raw reply.

## Accept: fill or overwrite

**Accept** has one option, **Overwrite existing labels**:

- **Off (the default): fill only unlabelled voxels.** The mask goes only into voxels you have not labelled yet (value 0). Everything you painted stays as it is.
- **On: overwrite.** The model's objects replace whatever is under them, your painting included. Its background still only fills unlabelled voxels, so an object you painted that the model missed is kept. Use it when the model outlines objects better than you had them.

Each object the model paints gets an id of its own. An object that overlaps one already labelled in the plane directly before or after keeps that object's id, so an object annotated plane by plane stays one object; a new object never reuses an id from those planes.

Both can be undone with **Undo** in the **Patch on screen** panel, which restores the plane as it was before the Accept. The panel reports how many voxels were labelled foreground and background, and, when overwriting, how many already-labelled voxels changed.

## Editing the prompt

The prompt is filled in for you from the structure you picked; each structure in the catalog has one that has been tried on EM. You never have to edit it. If the model keeps missing something (painting the wrong organelle, merging neighbours), you can:

- edit the prompt in the panel before pressing Shift+G, or
- edit it in the review panel and press **Resend**, which asks again about the same plane.

The dashboard keeps the sentences that state the image's resolution and ask for separate objects to be painted apart, so your edit only replaces the description of what to paint. A prompt is limited to 4000 characters.

## Cost

Vertex AI is billed to the Google Cloud project in the config. On `gemini-3-pro-image`, each input image costs a flat **~560 input tokens, whatever its size**, so sending a larger or sharper plane costs no more, and sending a smaller one saves nothing. The prompt text adds a few hundred tokens, and the image the model returns is billed as output. See Google's Vertex AI pricing page for current rates.

## Limits

- **Daily calls.** `daily_call_limit` (200 by default) caps how many calls you can make per day, counting every Shift+G, button click and Resend. The count is per user and per calendar day (local time), and survives dashboard restarts: it is kept in `~/.cellmap_flow/ai_annotate_usage.json`. Over the limit, the panel says so and refuses further calls until the next day. The panel shows how many you have used today.
- **Timeouts and retries.** Each call waits up to `timeout_s` (120 s by default). When the service is busy or over quota for a moment, the dashboard tries again up to 3 times, waiting a little longer each time, before reporting the error.

## The audit log and the staging folder

Both live in the session's `corrections` folder, `<Output Path>/<session>/corrections/`, next to the annotation volumes.

- **`ai_annotate_log.jsonl`** records every request, one JSON object per line: the time, your username, the event (`requested`, `staged`, `failed`, `resent`, `accepted`, `rejected`) and details such as the provider, model, prompt, plane and write box. It never holds keys or credentials. Use it to see what was sent, when, and what was accepted.
- **`.ai_annotate/<id>/`** holds a request waiting for review: the input plane sent (`input.png`), the model's image (`model.png`), the mask (`mask_write.npy`) and its details (`meta.json`). The folder is deleted when you Accept or Reject.

## Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| The panel says the feature is off | No config file, or `enabled: false` | Create `~/.cellmap_flow/ai_annotate.yaml` ([Turning it on](#turning-it-on)), then reload the page. |
| A config error naming a key | The file is malformed, has `api_key:`, or a key file is readable by others | Fix what the message names. For a key file: `chmod 600 <file>`. |
| The feature is unavailable, with an install command | `google-genai` is not installed in the dashboard's environment | `pixi install`, or `pip install -e ".[ai-annotate]"`. |
| Authentication error | Not logged in, or the login expired | Run `gcloud auth application-default login` again, as the user the dashboard runs as. |
| 403, "API has not been used in project ... or it is disabled" | The Vertex AI API is not enabled in the project | `gcloud services enable aiplatform.googleapis.com --project <project>`; wait a minute, then retry. |
| 403, permission denied | Your Google account has no Vertex AI access in that project | Ask the project's owner for the *Vertex AI User* role. |
| 404, model not found | `location` is not `global`, or the model id is wrong | Set `location: global`; check the model id. |
| 429, quota exceeded or resource exhausted | Google's per-minute quota for the project, after the dashboard's retries | Wait a minute and retry. If it keeps happening, ask for a higher quota in the Cloud console. |
| The daily limit is reached | You hit `daily_call_limit` | Wait until tomorrow, or raise the limit in the config. |
| Asked to acknowledge where the data goes | The box has not been ticked for this provider and dataset | Read the destination and tick the box. |
| The dataset is not allowed | `allowed_dataset_prefixes` does not include this dataset | Add its prefix, if sending it is allowed. |
| The model answered but nothing was painted | The model returned no image, or did not use the requested colour | The panel shows the model's reply, if any. Resend, or edit the prompt. |
| Shift+G does nothing | No settings chosen yet, or no annotation volume open | Pick a structure and model in the panel first; open a volume. |

The dashboard's server log (the log panel) has more detail about every failure, with keys removed.
