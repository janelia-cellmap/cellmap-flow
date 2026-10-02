// The Training Logs card's plot of the loss at each epoch (a <canvas>), and
// the summary line above it.

// An epoch's summary line in a training log, e.g. "Epoch 18/20 - Loss:
// 0.011371", as [line, epoch, total, loss]. Per-batch lines don't match.
export const EPOCH_LOSS = /Epoch\s+(\d+)\/(\d+)\s*-\s*Loss:\s*([0-9]*\.?[0-9]+)/i;

export function createLossPlot(canvas, summary) {
  const lossByEpoch = new Map();

  function render() {
    const width = Math.max(300, Math.floor(canvas.clientWidth || 600));
    const height = 180;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Read CSS design tokens for chart colors
    const styles = getComputedStyle(document.documentElement);
    const colorMuted = styles.getPropertyValue("--text-muted").trim() || "#8896a6";
    const colorBorder = styles.getPropertyValue("--border").trim() || "#1e293b";
    const colorBorderHover = styles.getPropertyValue("--border-hover").trim() || "#334155";
    const colorAccent = styles.getPropertyValue("--accent").trim() || "#10b981";

    const points = Array.from(lossByEpoch.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([epoch, loss]) => ({ epoch, loss }));

    if (points.length === 0) {
      summary.textContent = "No loss data yet";
      ctx.fillStyle = colorMuted;
      ctx.font = "12px monospace";
      ctx.fillText("Waiting for epoch loss...", 10, 20);
      return;
    }

    const minLoss = Math.min(...points.map(p => p.loss));
    const maxLoss = Math.max(...points.map(p => p.loss));
    const minEpoch = points[0].epoch;
    const maxEpoch = points[points.length - 1].epoch;
    const pad = 48;
    const chartW = Math.max(1, width - pad * 2);
    const chartH = Math.max(1, height - pad * 2);
    const lossSpan = Math.max(1e-9, maxLoss - minLoss);
    const epochSpan = Math.max(1, maxEpoch - minEpoch);

    const xFor = (e) => pad + ((e - minEpoch) / epochSpan) * chartW;
    const yFor = (l) => pad + (1 - ((l - minLoss) / lossSpan)) * chartH;

    // Axes
    ctx.strokeStyle = colorBorder;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, height - pad);
    ctx.lineTo(width - pad, height - pad);
    ctx.stroke();

    // Tick marks + labels
    ctx.fillStyle = colorMuted;
    ctx.font = "10px monospace";
    ctx.lineWidth = 1;
    ctx.strokeStyle = colorBorderHover;

    const xTickCount = Math.min(6, Math.max(2, points.length));
    for (let i = 0; i < xTickCount; i++) {
      const t = xTickCount === 1 ? 0 : i / (xTickCount - 1);
      const epochTick = Math.round(minEpoch + t * epochSpan);
      const x = xFor(epochTick);
      ctx.beginPath();
      ctx.moveTo(x, height - pad);
      ctx.lineTo(x, height - pad + 4);
      ctx.stroke();
      ctx.fillText(String(epochTick), x - 8, height - pad + 14);
    }

    const yTickCount = 5;
    for (let i = 0; i < yTickCount; i++) {
      const t = yTickCount === 1 ? 0 : i / (yTickCount - 1);
      const lossTick = maxLoss - t * lossSpan;
      const y = yFor(lossTick);
      ctx.beginPath();
      ctx.moveTo(pad - 4, y);
      ctx.lineTo(pad, y);
      ctx.stroke();
      const label = lossTick.toFixed(4);
      const labelW = ctx.measureText(label).width;
      ctx.fillText(label, pad - 6 - labelW, y + 3);
    }

    // Line
    ctx.strokeStyle = colorAccent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, idx) => {
      const x = xFor(p.epoch);
      const y = yFor(p.loss);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Last point marker
    const last = points[points.length - 1];
    const lx = xFor(last.epoch);
    const ly = yFor(last.loss);
    ctx.fillStyle = colorAccent;
    ctx.beginPath();
    ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
    ctx.fill();

    summary.textContent = `Epoch ${last.epoch}: ${last.loss.toFixed(6)} (min ${minLoss.toFixed(6)})`;
  }

  return {
    render,
    // Forget every point: a new job, or a restarted run.
    reset() {
      lossByEpoch.clear();
      render();
    },
    // One epoch's loss; a later report for the same epoch replaces it.
    add(epoch, loss) {
      if (!Number.isFinite(epoch) || epoch <= 0 || !Number.isFinite(loss)) {
        return;
      }
      lossByEpoch.set(epoch, loss);
      render();
    },
    // Every epoch summary line of a log, drawn at once (a restored job).
    addLog(text) {
      for (const line of text.split("\n")) {
        const match = line.match(EPOCH_LOSS);
        if (match) {
          const epoch = parseInt(match[1], 10);
          const loss = parseFloat(match[3]);
          if (Number.isFinite(epoch) && Number.isFinite(loss)) {
            lossByEpoch.set(epoch, loss);
          }
        }
      }
      render();
    },
  };
}
