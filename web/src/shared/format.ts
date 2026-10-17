// Numbers, sizes, durations and times as the page writes them for a person:
// on a status line, in a log line, in a file name.

/** `n` bytes in a unit a person reads: B, kB, MB or GB, decimal (as file
 * managers count them), one decimal place below 10. */
export function formatBytes(n: number): string {
  const units = ["B", "kB", "MB", "GB", "TB"];
  let value = n, unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit++;
  }
  return `${unit === 0 || value >= 10 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

/** `ms` as a person reads a duration: seconds under two minutes, minutes
 * under two hours, hours and minutes past that. */
export function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 120) return `${s} s`;
  const m = Math.round(s / 60);
  if (m < 120) return `${m} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

/** `n` with its thousands grouped by commas ("12,345"), whatever the
 * browser's locale, so the page reads the same everywhere. */
export function count(n: number): string {
  return n.toLocaleString("en-US");
}

/** `date` as `YYYYmmdd-HHMMSS` in local time: how the server stamps its
 * volume ids and run directories, and the page the files it names after a
 * moment (a zip of predictions), so they sort by time. */
export function stamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}
