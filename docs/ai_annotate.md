# AI-Assisted Annotation

This guide covers the **AI annotate** panel in the dashboard's Finetune tab. It sends one 2D plane of your EM data to a hosted image model, which paints the structure you ask for. You review what it painted, and if you accept it, it is written into your annotation volume as labels. It is a quicker start than painting from nothing, in the same way as **Seed from Prediction** (see [the finetuning guide](finetuning.md#label-the-patch-on-screen-in-one-click)). The browser mode has the same panel, with your own credentials in place of a config file: see [In the browser](#in-the-browser).

The feature is **off until you turn it on** with a config file (see [Turning it on](#turning-it-on)), because using it sends image data off the cluster to an outside company. Read [What data leaves the cluster](#what-data-leaves-the-cluster-and-where) before you turn it on.

## What it does

With an annotation volume open in the Finetune tab (see [Create or Resume an Annotation Volume](finetuning.md#2-create-or-resume-an-annotation-volume)):

1. In the **AI annotate** panel, pick the structure to label (mitochondria, ER, nucleus, ...) and the model. The first time for a dataset, tick the box that acknowledges where the data goes. Untick it at any time to withdraw that: the next run asks again.
2. Hover over the structure in the viewer and press **Shift+G**, or click **Annotate at view centre** to use the centre of the view instead.
3. The dashboard reads one plane of raw EM around that point, in the plane you are looking at, and sends it to the model with a prompt asking it to paint the structure in a colour. It turns the colours the model painted into a mask. This takes from a few seconds to a minute or two.
4. A review opens over the page with three large images side by side: the **input** plane, the **model's output**, and the mask as an **overlay** on the input. Click an image to see it at full width, and click again to go back. **Close** puts the review away without deciding; **Review Result** in the panel opens it again.
5. Then:
   - **Accept** writes the mask into the annotation volume (see [Accept: fill or overwrite](#accept-fill-or-overwrite)).
   - **Reject** throws it away. Nothing is written.
   - **Resend** asks the model again with the same plane, optionally with an edited prompt (see [Editing the prompt](#editing-the-prompt)). Image models answer differently each time, so a second try is often better.
6. **Undo** in the **Patch on screen** panel takes back an accepted mask, like any other fill there.

Only one request can be in flight or waiting for review at a time: accept or reject the one you have before asking for another.

### What gets written

The mask covers one plane, one annotation voxel thick, `crop_size_px` annotation voxels on a side (512 by default), centred on the point you chose and clipped to the data (the volume can reach past the data's edge, rounded up to whole chunks of the model's output). The model is sent the plane as the panel you took it in shows it, and the review shows it the same way. Within that plane, each connected piece of the mask, as the model drew it, becomes an object with its own id (2 and up) and every other voxel becomes background (1), as with **Seed from Prediction**. Check the result, then fix it with the brush as usual.

### Choosing the plane

The plane is the one the viewer's single panel is showing. cellmap-flow's viewer orders its axes z, y, x, and neuroglancer names its layouts after display slots rather than data axes, so the layout names do not match the planes:

| Viewer layout | Plane sent |
|---|---|
| **yz** (single panel) | XY, at the hovered z |
| **xz** (single panel) | XZ, at the hovered y |
| **xy** (single panel) | YZ, at the hovered x |
| **4panel**, **3d** or anything else | the top-left panel's plane (YZ) |

Neuroglancer keeps one layout for the whole viewer, so in the 4-panel layout the server cannot tell which panel the mouse is over: switch to a single-panel layout to choose the plane. The panel's status line names the plane that was sent (XY, XZ or YZ), so check it before accepting.

## Quick start

Everything below is explained in the sections that follow. With Gemini on Vertex AI, on the Janelia cluster:

1. **Sign in to Google once** and keep a copy of the login on group storage, so it works on any node (details in [Vertex AI setup](#vertex-ai-setup)):

   ```bash
   gcloud auth application-default login
   mkdir -p /groups/<lab>/<you>/.gcloud_adc
   cp ~/.config/gcloud/application_default_credentials.json /groups/<lab>/<you>/.gcloud_adc/
   chmod 600 /groups/<lab>/<you>/.gcloud_adc/application_default_credentials.json
   ```

2. **Write the config once**, `~/.cellmap_flow/ai_annotate.yaml`:

   ```yaml
   enabled: true
   providers:
     vertex:
       type: vertex_gemini
       project: <your Google Cloud project>
       location: global
       credentials_file: /groups/<lab>/<you>/.gcloud_adc/application_default_credentials.json
       models: [gemini-3-pro-image]
     fake:                      # optional: runs locally, sends nothing
       type: fake
       models: [fake-threshold]
   ```

3. **Start the dashboard as usual** from the pixi environment, e.g. `pixi run cellmap_flow yaml <your.yaml>`. The first `pixi run` installs everything, `google-genai` included. Nothing needs exporting first.

4. **In the Finetune tab**, create or resume an annotation volume, open **AI-assisted annotation**, pick the provider, model and structure, and tick the box agreeing to where the planes are sent.

5. **Annotate**: switch the viewer to a single-panel layout (**yz** shows the XY plane, see [Choosing the plane](#choosing-the-plane)), hover over the structure and press **Shift+G**. The review opens over the page when the result is ready: **Accept** or **Reject** it there.

## Turning it on

The feature reads a config file on the machine the dashboard runs on. With no file, the panel says the feature is off and points here.

The file is `~/.cellmap_flow/ai_annotate.yaml` in the home directory of the user the dashboard runs as (on the cluster, your cluster home, e.g. `/groups/<group>/home/<you>`), or the path in the environment variable `CELLMAP_FLOW_AI_ANNOTATE_CONFIG` if that is set. An example with every option is in [`docs/examples/ai_annotate.yaml`](examples/ai_annotate.yaml). The shortest one that works:

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
| `credentials_file` | gcloud's file in your home directory | Vertex only: the path of the Google login to use, e.g. a copy on group storage for cluster nodes (see [Vertex AI setup](#vertex-ai-setup)). Only the path goes here; the file must be yours and `chmod 600`. |

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
   **On the cluster**, the dashboard may run on a node where your home directory is not where gcloud saved that file. Copy it to group storage that every node mounts, readable only by you, and name it in the config instead:

   ```bash
   mkdir -p /groups/<lab>/<you>/.gcloud_adc
   cp ~/.config/gcloud/application_default_credentials.json /groups/<lab>/<you>/.gcloud_adc/
   chmod 600 /groups/<lab>/<you>/.gcloud_adc/application_default_credentials.json
   ```

   ```yaml
   providers:
     vertex:
       type: vertex_gemini
       credentials_file: /groups/<lab>/<you>/.gcloud_adc/application_default_credentials.json
       # ...
   ```

   The config holds only the path. The dashboard refuses the file unless you own it and only you can read it, hands the credentials straight to Google's library, and never logs them or sends them to the browser. When the login expires, run `gcloud auth application-default login` again and copy the new file over the old one.
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

### Why the login expires, and a key that does not

Your personal login from `gcloud auth application-default login` stops working after a while: HHMI's Google Workspace makes everyone using Google Cloud tools sign in again periodically (its "Google Cloud session control"; at most a week, possibly less). The panel then says "Google asks you to sign in again". Run the login again and copy the file over your `credentials_file`; nothing needs restarting.

To stop that, sign in with a **service account key** instead. A service account is a Google Cloud identity for a program rather than a person, and its key does not expire:

1. In the [Google Cloud console](https://console.cloud.google.com/), open your project, then **IAM & Admin → Service Accounts → Create service account**.
2. Give it the role **Vertex AI User** (and nothing more).
3. On its **Keys** tab: **Add key → Create new key → JSON**. Your browser downloads a `.json` file.
4. Put it on group storage, readable only by you, and name it in the config:

   ```bash
   cp <downloaded file>.json /groups/<lab>/<you>/.gcloud_adc/vertex-sa.json
   chmod 600 /groups/<lab>/<you>/.gcloud_adc/vertex-sa.json
   ```

   ```yaml
       credentials_file: /groups/<lab>/<you>/.gcloud_adc/vertex-sa.json
   ```

The key is a long-lived secret: anyone who can read the file can spend from the project until the key is deleted. Keep it `chmod 600`, never copy it into a repository or a shared folder, and delete it in the console (**Keys** tab) if it ever leaks. Some organisations do not allow keys to be created; if the console refuses, use the personal login above.

## Running it from Fileglancer

The Fileglancer app runs the same dashboard as a job on the cluster, as you, from the same pixi environment, so the feature works there with no extra install. What it needs is the same as anywhere else, put where a cluster job can find it:

- **The config file goes in your cluster home directory**: `~/.cellmap_flow/ai_annotate.yaml` as the job sees it, which on the Janelia cluster is `/groups/<group>/home/<you>/.cellmap_flow/ai_annotate.yaml`, not your workstation's local home if that differs. If the panel says "no config file at ...", the path it names is where the job looked.
- **`credentials_file` on group storage**, as in [Vertex AI setup](#vertex-ai-setup), since the job runs on whichever node it lands on. A service account key is the better choice here: a Fileglancer session runs for hours unattended, and a personal login can expire during it.
- **Each user sets up their own.** The config and the credentials are per user, and a credentials file that others can read is refused, so a lab cannot share one login through a group-readable file. Each person who turns the feature on uses their own login or key, billed to the project it belongs to.
- **The job must reach Google**: the cluster node calls `aiplatform.googleapis.com` and `oauth2.googleapis.com` over HTTPS. If calls fail with "Could not reach Vertex AI", the node has no route out; ask Scientific Computing.

## In the browser

The browser mode ([janelia-cellmap.github.io/cellmap-flow/app](https://janelia-cellmap.github.io/cellmap-flow/app/); the docs' *In the browser* page) has the same panel, under **Patch on screen** in its Finetune tab, once an annotation volume is open. There is no server and no config file: the page itself reads the plane from the raw volume, sends it to the model, turns the answer into a mask, and writes an accepted mask into the paint volume the browser keeps. Hover and **Shift+G**, **Annotate at View Centre**, the review with its three images, **Resend**, **Accept** with **Overwrite existing labels**, **Reject** and the acknowledgement are as above, and so is **Undo** in the **Patch on screen** panel (up to 10 actions per volume, accepted annotations and the panel's seeds and fills alike, kept while the page is open).

Three providers, picked in the panel:

| Provider | Sends the plane to | Credentials |
|---|---|---|
| `gemini` | Google's Gemini API (`generativelanguage.googleapis.com`) | an API key, from [Google AI Studio](https://aistudio.google.com/apikey) |
| `vertex` | Google Cloud Vertex AI, location `global` (`aiplatform.googleapis.com`) | a Google Cloud project and an access token |
| `fake` | nowhere: it runs in the browser | none |

**Credentials.** The panel's **Credentials** block, above the acknowledgement, takes what the chosen provider needs, and uses it as soon as you paste it: there is nothing to save. The key or token goes from your browser straight to Google with each plane, and nowhere else: not to Janelia, not into the paint volume or its audit log, never into a log line, a share link or the session the page keeps across reloads. It is kept in the page's memory only, never in the browser's storage, where it would sit unencrypted for any script on the site to read: so after a reload, paste it again. **Clear** removes it at once. The block says *in use* or *paste a key* for the chosen provider, and a run is refused until there is one. To limit the harm if a key ever leaks, restrict it in the Google Cloud console to the Generative Language API and to your page's address (`http://localhost/*`, or the app's site).

**The Gemini API's data use.** The server mode refuses the Gemini API and uses Vertex AI because on the Gemini API's **free tier** Google may use what is sent (the plane and the prompt) to improve its products; on its **paid tier** (a key from a project with billing enabled) it does not, as on Vertex AI. The browser offers the key path because it is the one most people can set up in a minute, and says this beside the key field. Do not send unpublished data, or data under an agreement, on a free-tier key. Vertex AI's terms are the same in the browser as on the server, `location: global` included.

**Vertex AI from the browser.** A browser cannot run `gcloud`, so the Vertex provider takes a project and an access token: run `gcloud auth print-access-token` where you are signed in to Google Cloud (`gcloud auth login`; the project needs the Vertex AI API enabled and your account the *Vertex AI User* role, as in [Vertex AI setup](#vertex-ai-setup)) and paste its output. A token lasts about an hour; when a call fails asking you to sign in, paste a new one and **Save**. *Remember* keeps the project and the token, but the token still expires.

**Models.** When a Gemini API key is saved, the page lists the key's image models and the picker shows them; Vertex AI lists none, so the picker shows the image models Google's docs name (`gemini-3-pro-image`, `gemini-3.1-flash-image`, `gemini-2.5-flash-image`), and **Other…** takes any model id.

**Limits, retries and the log.** The daily cap is 200 calls per browser and calendar day, kept in the browser's local storage, counted before the call and given back when the call fails signing in or on a configuration error, as on the server; the panel says how many are used. Retries, the timeout and the error messages are the server's. The audit log, `ai_annotate_log.jsonl`, one JSON line per request as on the server with `browser` as the user, is written into the paint volume and travels with it in **Download paint**'s zip; a result waiting for review is held in memory, not in a staging folder. The fake provider is for trying the panel without a key: it paints the darker pixels in a disc around the point, in the page, as the server's does.

## What data leaves the cluster and where

Each call (each Shift+G, button click or Resend) sends:

- **one 2D plane of raw EM**: the region you chose, as a greyscale image of at most 1024 × 1024 pixels (a larger plane is shrunk before sending), and
- **the prompt text**: the instructions for the model, which name the structure, the colour to paint it, and the image's resolution in nm.

Nothing else is sent: not the rest of the dataset, not its path or name, not your annotations. The model sends back one image and sometimes a line of text.

**Where it goes.** With the `vertex_gemini` provider the plane goes to Google Cloud's Vertex AI, in the project from the config:

- Google's terms for Vertex AI say it does not use customer data to train its models. That is why the server uses Vertex AI and not the Gemini API from Google AI Studio, which it does **not support**: on its free tier, Google may use what you send to improve its products. (The browser mode offers the Gemini API with your own key and says this beside the key field; see [In the browser](#in-the-browser).)
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

Vertex AI uses the login from `gcloud auth application-default login` and needs no API key. That login file is held to the same rules as a key file below when the config names it (`credentials_file`): owned by you, `chmod 600`, read only when a call is made, and its secret fields blanked from the logs. Leave `credentials_file` out and Google's library finds the login itself; cellmap-flow never reads it then. Other providers may be added later that do (an OpenAI-compatible service, for example). For those, the rules are:

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
- **Keys are removed from the logs.** The dashboard streams its server log to the browser (the log panel), so every key and login secret the dashboard has read is replaced with `[REDACTED]` before a log line is written. Error messages shown in the panel are short summaries of what went wrong, never the service's raw reply.

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

- **Daily calls.** `daily_call_limit` (200 by default) caps how many calls you can make per day, counting every Shift+G, button click and Resend. It is your own cap against runaway cost, not a limit Google sets: raise or lower it in the config file. A call that fails before it reaches the model (signing in, a config error) costs nothing and is not counted. The count is per user and per calendar day (local time), and survives dashboard restarts: it is kept in `~/.cellmap_flow/ai_annotate_usage.json`. Over the limit, the panel says so and refuses further calls until the next day. The panel shows how many you have used today.
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
| "Google asks you to sign in again: the saved login has expired" | Your organisation makes Google logins sign in again every so often, and the saved one is past that | Run `gcloud auth application-default login` again, as the user the dashboard runs as, and copy `~/.config/gcloud/application_default_credentials.json` over your `credentials_file`, if the config names one. No restart needed. |
| "No usable Google Cloud credentials" | No login was found at all | Run `gcloud auth application-default login`, or name the file in `credentials_file`. |
| 403, "API has not been used in project ... or it is disabled" | The Vertex AI API is not enabled in the project | `gcloud services enable aiplatform.googleapis.com --project <project>`; wait a minute, then retry. |
| 403, permission denied | Your Google account has no Vertex AI access in that project | Ask the project's owner for the *Vertex AI User* role. |
| 404, model not found | `location` is not `global`, or the model id is wrong | Set `location: global`; check the model id. |
| 429, quota exceeded or resource exhausted | Google's per-minute quota for the project, after the dashboard's retries | Wait a minute and retry. If it keeps happening, ask for a higher quota in the Cloud console. |
| The daily limit is reached | You hit `daily_call_limit` | Wait until tomorrow, or raise the limit in the config. |
| Asked to acknowledge where the data goes | The box has not been ticked for this provider and dataset | Read the destination and tick the box. |
| The dataset is not allowed | `allowed_dataset_prefixes` does not include this dataset | Add its prefix, if sending it is allowed. |
| The model answered but nothing was painted | The model returned no image, or did not use the requested colour | The panel shows the model's reply, if any. Resend, or edit the prompt. |
| Shift+G does nothing | No settings chosen yet, or no annotation volume open | Pick a structure and model in the panel first; open a volume. |
| In the browser: "Paste ... first", or the Credentials block says *paste a key* | No credentials in the page: none pasted yet, or the page was reloaded, which forgets them | Paste the API key (or the project and an access token) into the **Credentials** block again. |
| In the browser, Vertex AI: a call fails asking you to sign in | The pasted access token has expired (they last about an hour) | Run `gcloud auth print-access-token` again, paste the new token and click **Save**. |

The dashboard's server log (the log panel) has more detail about every failure, with keys removed.
