"""Compiler tournament: compare rule-compiler candidates by real results and cost.

Entrants: 5 gpt-oss-120b endpoints + deepseek-v4-flash-0731 @ Cloudflare.
Efforts: low / medium / high. 26 rules x 3 reps per cell.

Uses the exact production compile path: prompt from get_rule_compile_prompt,
extraction via BaseLLMClient._extract_content_from_response, syntax check
identical to BaseLLMClient._validate_code_syntax, and behavioral equivalence
via RuleValidator.check_equivalence_by_simulation (100 sims x 40 turns,
seed 42, production settings).
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import os
import random as pyrandom
import sys
import threading
import time
from dataclasses import dataclass

import requests

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from eleusis.game.engine import Rule  # noqa: E402
from eleusis.game.validator import RuleValidator  # noqa: E402
from eleusis.llm.base import BaseLLMClient  # noqa: E402
from eleusis.prompts import get_rule_compile_prompt  # noqa: E402

KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
REPS = 3
EFFORTS = ("low", "medium", "high")
MAX_TOKENS = 4096  # production compiler default
NUM_SIMS, TURNS_PER_SIM, SIM_SEED = 100, 40, 42

# label, model, provider pin, temperature, price in/out per M tokens
ENTRANTS = [
    ("cerebras/gpt-oss-120b", "openai/gpt-oss-120b", "Cerebras", 0.7, 0.35, 0.75),
    ("nebius/gpt-oss-120b", "openai/gpt-oss-120b", "Nebius", 0.7, 0.15, 0.60),
    ("novita/gpt-oss-120b", "openai/gpt-oss-120b", "Novita", 0.7, 0.05, 0.25),
    ("akashml/gpt-oss-120b", "openai/gpt-oss-120b", "AkashML", 0.7, 0.037, 0.49),
    ("groq/gpt-oss-120b", "openai/gpt-oss-120b", "Groq", 0.7, 0.15, 0.60),
    (
        "cloudflare/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-flash-0731",
        "Cloudflare",
        1.0,
        0.44,
        1.32,
    ),
]

print_lock = threading.Lock()
_results: list[dict[str, object]] = []
_done = 0


class _ExtractionShim(BaseLLMClient):
    """Only the shared extraction logic; no provider behind it."""

    def __init__(self) -> None:  # noqa: D107
        pass

    @property
    def provider_name(self) -> str:  # noqa: D102
        return "shim"

    def _call_api(self, messages, is_continuation=False, **_):  # noqa: D102, ANN001
        raise NotImplementedError

    def _stream_api(self, messages, **_):  # noqa: D102, ANN001
        raise NotImplementedError


SHIM = _ExtractionShim()


def validate_syntax(code: str) -> bool:
    """Copy of BaseLLMClient._validate_code_syntax."""
    import textwrap

    full = f"def _validate(card, mainline):\n{textwrap.indent(code, '    ')}"
    try:
        compile(full, "<string>", "exec")
        return True
    except SyntaxError:
        return False


@dataclass(frozen=True)
class Job:
    label: str
    model: str
    provider: str
    temp: float
    effort: str
    rule_name: str
    rule_desc: str
    truth_code: str
    rep: int


def one_call(job: Job) -> dict[str, object]:
    body = {
        "model": job.model,
        "messages": [{"role": "user", "content": get_rule_compile_prompt(job.rule_desc)}],
        "temperature": job.temp,
        "reasoning_effort": job.effort,
        "max_tokens": MAX_TOKENS,
        "provider": {"order": [job.provider], "allow_fallbacks": False},
    }
    content, usage, err, http_err = "", None, None, None
    for attempt in range(4):
        try:
            r = requests.post(
                URL,
                headers={"Authorization": f"Bearer {KEY}"},
                json=body,
                timeout=240,
            )
            if r.status_code == 200:
                data = r.json()
                msg = (data.get("choices") or [{}])[0].get("message", {})
                content = msg.get("content") or ""
                usage = data.get("usage") or {}
                break
            http_err = f"HTTP{r.status_code}:{r.text[:150]}"
            if r.status_code not in (402, 403, 404, 500, 502, 503, 524, 429):
                break
        except Exception as exc:  # noqa: BLE001
            http_err = f"EXC:{type(exc).__name__}"
        time.sleep(min(2 * 2**attempt, 30) + pyrandom.random() * 2)

    out: dict[str, object] = {
        "label": job.label,
        "effort": job.effort,
        "rule": job.rule_name,
        "rep": job.rep,
    }
    if http_err and not content:
        out.update(ok=False, error=http_err, syntax_valid=False, equivalent=False)
        return out
    try:
        code = SHIM._extract_content_from_response(content, ["CODE"], try_code_blocks=True)
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, error=f"extract:{exc}", syntax_valid=False, equivalent=False)
        return out
    syntax_valid = validate_syntax(code)
    out.update(
        ok=True,
        syntax_valid=syntax_valid,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        completion_tokens=(usage or {}).get("completion_tokens"),
        reasoning_tokens=(
            ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
        ),
        code=code,
    )
    if not syntax_valid:
        out["equivalent"] = False
        return out
    try:
        validator = RuleValidator()
        equiv, reason, comps, mism = validator.check_equivalence_by_simulation(
            Rule(job.rule_desc, job.truth_code),
            job.rule_desc,
            [],
            num_simulations=NUM_SIMS,
            turns_per_simulation=TURNS_PER_SIM,
            preconverted_code=code,
            simulation_seed=SIM_SEED,
        )
        out.update(equivalent=equiv, comparisons=comps, mismatches=mism)
    except Exception as exc:  # noqa: BLE001
        out.update(equivalent=False, error=f"sim:{type(exc).__name__}:{exc}")
    return out


def main() -> None:
    rules = json.load(open("rules.json"))["rules"]
    jobs: list[Job] = []
    for label, model, provider, temp, _, _ in ENTRANTS:
        for effort in EFFORTS:
            for rule in rules:
                for rep in range(REPS):
                    jobs.append(
                        Job(
                            label,
                            model,
                            provider,
                            temp,
                            effort,
                            rule["name"],
                            rule["description"],
                            rule["code"],
                            rep,
                        )
                    )
    print(f"{len(jobs)} jobs across {len(ENTRANTS)} entrants x {len(EFFORTS)} efforts")

    def wrapped(job: Job) -> dict[str, object]:
        global _done
        res = one_call(job)
        with print_lock:
            _done += 1
            if _done % 50 == 0 or _done == len(jobs):
                print(f"  ...{_done}/{len(jobs)}", flush=True)
        return res

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        _results.extend(pool.map(wrapped, jobs))
    print(f"done in {(time.time() - t0) / 60:.1f} min")

    with open("/tmp/compiler_tournament_results.json", "w") as f:
        json.dump(_results, f, indent=1)

    price = {e[0]: (e[4], e[5]) for e in ENTRANTS}
    cells = collections.defaultdict(list)
    for r in _results:
        cells[(r["label"], r["effort"])].append(r)

    print(f"\n{'entrant':<32}{'effort':<8}{'equiv':>7}{'syntax':>7}{'err':>5}{'cost$':>8}")
    ranked = []
    for (label, effort), rows in cells.items():
        n = len(rows)
        eq = sum(1 for r in rows if r.get("equivalent"))
        sx = sum(1 for r in rows if r.get("syntax_valid"))
        er = sum(1 for r in rows if r.get("error"))
        pt = sum(r.get("prompt_tokens") or 0 for r in rows)
        ct = sum(r.get("completion_tokens") or 0 for r in rows)
        pin, pout = price[label]
        cost = (pt * pin + ct * pout) / 1e6
        ranked.append((eq / n, -cost, label, effort, sx, er, cost, n))
    for eqr, negcost, label, effort, sx, er, cost, n in sorted(ranked, reverse=True):
        print(
            f"{label:<32}{effort:<8}{eqr:>6.0%}{sx / n:>7.0%}{er:>5}{cost:>8.2f}"
        )


if __name__ == "__main__":
    main()
