#!/usr/bin/env python3
"""Serve a local web dashboard for live Benchmark Run progress.

Read-only: the dashboard polls the same per-worker SQLite stores as
scripts/check_progress.py and renders one auto-refreshing page at
http://127.0.0.1:<port>/.
"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from eleusis.analysis.live_progress import collect_live_progress

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eleusis Benchmark Dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0;
         background: #0f1115; color: #e6e8ec; }
  header { padding: 18px 24px 10px; border-bottom: 1px solid #232732; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .meta { font-size: 12px; color: #8b93a3; }
  .stale { color: #f0b452; }
  main { padding: 16px 24px 40px; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }
  .card { background: #171a21; border: 1px solid #232732; border-radius: 10px;
          padding: 10px 16px; min-width: 130px; }
  .card .value { font-size: 20px; font-weight: 600; }
  .card .label { font-size: 11px; color: #8b93a3; text-transform: uppercase;
                 letter-spacing: 0.06em; }
  .worker { background: #171a21; border: 1px solid #232732; border-radius: 10px;
            padding: 12px 16px; margin-bottom: 12px; }
  .worker.error { border-color: #a3392f; }
  .head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .name { font-weight: 600; font-size: 14px; }
  .state { font-size: 12px; color: #8b93a3; }
  .state.done { color: #5fc38a; }
  .bar { height: 8px; border-radius: 4px; background: #232732; margin: 10px 0;
         overflow: hidden; }
  .bar > div { height: 100%; background: #4f8fdb; border-radius: 4px; }
  .rounds { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .chip { font-size: 11px; padding: 3px 8px; border-radius: 999px;
          border: 1px solid #2c313d; color: #aab2c2; white-space: nowrap; }
  .chip.completed.win { border-color: #2f6b4a; color: #5fc38a; }
  .chip.completed.loss { border-color: #6b4a2f; color: #d9a35f; }
  .chip.active { border-color: #35507a; color: #7fa9e0; }
  .chip.scheduled { opacity: 0.55; }
</style>
</head>
<body>
<header>
  <h1>Eleusis Benchmark Dashboard</h1>
  <div class="meta" id="meta">loading…</div>
</header>
<main>
  <div class="cards" id="cards"></div>
  <div id="workers"></div>
</main>
<script>
const REFRESH_MS = 5000;

function fmt(n) { return n == null ? "—" : String(n); }
function h(seconds) {
  if (!seconds || !isFinite(seconds)) return "—";
  if (seconds < 90) return seconds.toFixed(0) + "s";
  return (seconds / 3600).toFixed(1) + "h";
}

function overall(workers) {
  let completed = 0, total = 0, successful = 0, score = 0;
  let reasoning = 0, answer = 0, prompt = 0, duration = 0;
  for (const w of workers) {
    completed += w.completed; total += w.total;
    successful += w.successful; score += w.score;
    duration += w.duration_seconds || 0;
    if (w.usage) {
      reasoning += w.usage.reasoning_tokens || 0;
      answer += w.usage.answer_tokens || 0;
      prompt += w.usage.prompt_tokens || 0;
    }
  }
  return {completed, total, successful, score, reasoning, answer, prompt, duration};
}

function card(value, label) {
  const div = document.createElement("div");
  div.className = "card";
  div.innerHTML = `<div class="value">${value}</div><div class="label">${label}</div>`;
  return div;
}

function render(payload) {
  const workers = payload.workers;
  const o = overall(workers);
  const cards = document.getElementById("cards");
  cards.replaceChildren(
    card(`${o.completed}/${o.total}`, "rounds"),
    card(o.completed ? (100 * o.successful / o.completed).toFixed(0) + "%"
                    : "—", "solved"),
    card(o.completed ? (o.score / o.completed).toFixed(1) : "—", "avg score"),
    card(o.reasoning.toLocaleString(), "reasoning tokens"),
    card(o.answer.toLocaleString(), "answer tokens"),
    card(h(o.duration), "provider time"),
  );

  const host = document.getElementById("workers");
  host.replaceChildren();
  for (const w of workers) {
    const box = document.createElement("div");
    box.className = "worker" + (w.error ? " error" : "");
    const meta = [];
    if (w.error) {
      meta.push(`error: ${w.error}`);
    } else {
      meta.push(`${w.successful} wins · score ${w.score}`);
      if (w.completed >= w.total && w.total > 0) meta.push("DONE");
      else if (w.completed > 0) {
        const left = (w.total - w.completed) * w.duration_seconds / w.completed;
        meta.push("~" + h(left) + " left");
      }
      if (w.active_round_number != null)
        meta.push(`Round ${w.active_round_number}` +
                  ` · turn ${fmt(w.committed_turns + 1)}`);
    }
    const pct = w.total ? (100 * w.completed / w.total).toFixed(0) : 0;
    box.innerHTML =
      `<div class="head"><span class="name">${w.name}</span>` +
      `<span class="state ${w.completed >= w.total && w.total > 0 ? "done" : ""}">` +
      `${meta.join(" · ")}</span></div>` +
      `<div class="bar"><div style="width:${pct}%"></div></div>`;
    const chips = document.createElement("div");
    chips.className = "rounds";
    for (const r of w.rounds) {
      const chip = document.createElement("span");
      chip.className = "chip " + r.status +
        (r.status === "completed"
          ? (r.terminal_kind === "correct_formal_guess" ? " win" : " loss") : "");
      let text = `#${r.round_number} ${r.rule_name}`;
      if (r.status === "completed")
        text += ` · ${r.score} pts · ${r.turn_count} turns`;
      else if (r.status === "active") text += ` · turn ${fmt((r.turn_count || 0) + 1)}`;
      else text += " · queued";
      chip.textContent = text;
      chips.append(chip);
    }
    box.append(chips);
    host.append(box);
  }

  const refreshed = new Date(payload.generated_at).toLocaleTimeString();
  document.getElementById("meta").textContent =
    `pattern: ${payload.pattern} · refreshed ${refreshed} · polling every 5s`;
}

async function tick() {
  try {
    const response = await fetch("/api/progress", {cache: "no-store"});
    render(await response.json());
    document.getElementById("meta").classList.remove("stale");
  } catch (error) {
    document.getElementById("meta").classList.add("stale");
    document.getElementById("meta").textContent = "refresh failed — retrying…";
  }
}

tick();
setInterval(tick, REFRESH_MS);
</script>
</body>
</html>
"""


def _progress_payload(pattern: str) -> dict[str, object]:
    """Build the /api/progress JSON document for one pattern."""
    workers = collect_live_progress(pattern)
    overall_totals = {
        "completed": sum(worker.completed for worker in workers),
        "total": sum(worker.total for worker in workers),
        "successful": sum(worker.successful for worker in workers),
        "score": sum(worker.score for worker in workers),
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "pattern": pattern,
        "workers": [asdict(worker) for worker in workers],
        "overall": overall_totals,
    }


def create_server(
    pattern: str, port: int, host: str = "0.0.0.0"
) -> ThreadingHTTPServer:
    """Create the read-only dashboard server bound to ``host``.

    The default binds every interface so the dashboard is reachable on the
    tailnet (for example ``http://desktop:8390/``); pass ``127.0.0.1`` to
    restrict it to localhost.
    """

    class DashboardHandler(BaseHTTPRequestHandler):
        """Serve the dashboard page and its live JSON endpoint."""

        def do_GET(self) -> None:
            """Answer GET / and GET /api/progress; anything else is a 404."""
            if self.path == "/":
                body = _DASHBOARD_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
            elif self.path == "/api/progress":
                body = json.dumps(_progress_payload(pattern)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
            else:
                body = b"not found"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the console quiet; the page shows its own status."""

    return ThreadingHTTPServer((host, port), DashboardHandler)


def main() -> None:
    """Serve the dashboard until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default="solo_evaluation_*",
        help="Glob pattern for results/ worker folders (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8390,
        help="Local port to serve on (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help=(
            "Interface to bind (default: all interfaces, so the dashboard is "
            "reachable on the tailnet; use 127.0.0.1 for localhost only)"
        ),
    )
    args = parser.parse_args()
    server = create_server(pattern=args.pattern, port=args.port, host=args.host)
    print(f"Eleusis Benchmark Dashboard: http://{args.host}:{args.port}/")
    print(f"pattern: {args.pattern} | Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
