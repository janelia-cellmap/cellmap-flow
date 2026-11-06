// Durations and sizes as the dashboard writes them: a whole number of the
// small unit, or one decimal of the large one past a thousand.

export function formatMs(ms) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export function formatMb(mb) {
  return mb < 1000 ? `${Math.round(mb)} MB` : `${(mb / 1000).toFixed(1)} GB`;
}
