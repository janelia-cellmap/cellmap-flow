// The Processing tab's Channel (templates/_processing_tab.html): which
// channel of a raw of several (light microscopy's, say) every model reads.
//
// The pick is the input list's ChannelSelector step, first in the chain, so
// the two always agree: the select shows the list's step (channel 0 when it
// has none), and a pick makes that step the list's first and applies the
// chain, as the Apply bar does, so every model's layer is drawn again on it.
// The select is shown only for a raw of more than one channel, as GET
// /api/raw/channels counts them, each option named by its label there or by
// its index.
import { getJSON } from "../lib/api.js";
import { h } from "../lib/dom.js";

const CHANNEL_STEP = "ChannelSelector";

// The channel `chain` ([{name, ...params}], the list's) picks, as text: its
// first ChannelSelector's, else 0, as the backend reads it
// (io/raw.py selected_channel).
function pickedChannel(chain) {
  const step = chain.find((s) => s.name === CHANNEL_STEP);
  return step ? String(step.channel) : "0";
}

// `input` is the input list (shared/op-chain.js), `apply` the Apply bar's.
export function initRawChannel({ input, apply }) {
  const row = document.getElementById("rawChannelRow");
  const select = document.getElementById("rawChannel");
  const showPick = () => { select.value = pickedChannel(input.getChain()); };

  select.addEventListener("change", () => {
    input.show([{ name: CHANNEL_STEP, channel: select.value },
      ...input.getChain().filter((step) => step.name !== CHANNEL_STEP)]);
    apply();
  });
  // A ChannelSelector ticked, typed in or dragged in the list, or the
  // backend's chain shown there, moves the select with it.
  input.onEdit(showPick);
  return getJSON("/api/raw/channels")
    .then((data) => {
      const labels = data.labels || [];
      select.replaceChildren(...Array.from({ length: data.channels || 1 },
        (_, k) => h("option", { value: String(k) }, labels[k] || `channel ${k}`)));
      showPick();
      row.hidden = !(data.channels > 1);
    })
    .catch((err) => console.warn("Could not read the raw's channels:", err));
}
