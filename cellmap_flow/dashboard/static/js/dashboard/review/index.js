// The dashboard's Review tab (templates/_review_tab.html): walk a review
// index (a review.sqlite, see cellmap_flow/review_index.py) one candidate
// instance at a time, and record a verdict on each. The server moves the
// viewer to every instance it hands out.
//
// Each module drives one part of the tab; this one wires them together:
// - current.js: the Current instance block, and the verdict buttons that act
//   on it;
// - progress.js: the Progress card and the queue list, refreshed every 5 s
//   while an index is open;
// - session.js: Open, the index already open when the tab is shown, and the
//   viewer's pick stream;
// - queue.js: Next, Go to ID, Bless, Edit, Erase and Undo.
// Nothing is asked of the server until the tab is shown or used.
import { createCurrentInstance } from "./current.js";
import { createProgress } from "./progress.js";
import { initQueue } from "./queue.js";
import { initSession } from "./session.js";

export function initReviewTab() {
  const current = createCurrentInstance();
  const progress = createProgress();
  initSession({ current, progress });
  initQueue({ current, progress });
}
