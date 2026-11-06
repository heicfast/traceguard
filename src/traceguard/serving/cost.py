"""Run-level cost metering with a fail-closed budget cap.

Every API/server call is metered (provider, model, prompt/completion tokens,
unit price, USD estimate).  Prices are carried by an auditable
:class:`PriceTable` whose entries carry the source URL and the verification
date — prices drift, so an entry without a verified date is refused for
research stages.  The ledger writes an NDJSON event stream plus a summary
manifest; when cumulative spend exceeds the configured hard cap the ledger
REFUSES further metering (raises), which surfaces as a provider error and
stops the run instead of silently exceeding the grant.

AWS compute (instance-hours) is metered alongside API spend so a single
``costs/summary.json`` answers "what did this study cost".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PriceEntry:
    """USD per 1M tokens (prompt, completion) for one endpoint model."""

    prompt_usd_per_mtok: float
    completion_usd_per_mtok: float
    source_url: str
    verified_at: str  # ISO date; required — an unverifiable price is a config error

    def validate(self) -> None:
        if self.prompt_usd_per_mtok < 0 or self.completion_usd_per_mtok < 0:
            raise ValueError("prices must be non-negative")
        if not self.source_url.startswith("http"):
            raise ValueError("price entries must cite a source URL")
        try:
            datetime.fromisoformat(self.verified_at)
        except ValueError as exc:
            raise ValueError(f"verified_at must be an ISO date: {self.verified_at}") from exc

    def __post_init__(self) -> None:
        self.validate()


class PriceTable:
    """Auditable price registry keyed by (provider, model_id)."""

    # Verified 2026-11-06 (see docs/REFERENCES.md §Model-and-pricing sources):
    #  - OpenAI GPT-6.1 Sol / GPT-6 Luna per-1M list prices
    #    (developers.openai.com/api/docs/pricing, cross-checked nOps 2026 guide)
    #  - DeepSeek V4.1-Flash off-peak (api-docs.deepseek.com/quick_start/pricing;
    #    treat as drift-prone and re-verify at probe time)
    DEFAULT_ENTRIES: dict[tuple[str, str], PriceEntry] = {
        ("openai", "gpt-6.1-sol"): PriceEntry(2.00, 10.00,
            "https://developers.openai.com/api/docs/pricing", "2026-11-06"),
        ("openai", "gpt-6-luna"): PriceEntry(0.10, 0.50,
            "https://developers.openai.com/api/docs/pricing", "2026-11-06"),
        ("deepseek", "deepseek-chat"): PriceEntry(0.15, 0.60,
            "https://api-docs.deepseek.com/quick_start/pricing", "2026-11-06"),
        ("deepseek", "deepseek-reasoner"): PriceEntry(0.30, 1.20,
            "https://api-docs.deepseek.com/quick_start/pricing", "2026-11-06"),
    }

    def __init__(self, entries: dict[tuple[str, str], PriceEntry] | None = None):
        self.entries: dict[tuple[str, str], PriceEntry] = dict(
            entries or self.DEFAULT_ENTRIES
        )
        for entry in self.entries.values():
            entry.validate()

    def add(self, provider: str, model_id: str, entry: PriceEntry) -> None:
        entry.validate()
        self.entries[(provider, model_id)] = entry

    def lookup(self, provider: str, model_id: str) -> PriceEntry | None:
        return self.entries.get((provider, model_id))


def price_for(
    price_table: PriceTable | None,
    provider: str,
    model_id: str,
    fallback: tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
    """Resolve a unit price, preferring the audited table over ad-hoc values."""
    if price_table is not None:
        entry = price_table.lookup(provider, model_id)
        if entry is not None:
            return (entry.prompt_usd_per_mtok, entry.completion_usd_per_mtok)
    return fallback


class BudgetExceededError(RuntimeError):
    """Raised when metered spend crosses the configured hard cap (fail closed)."""


class CostLedger:
    """Append-only NDJSON cost ledger with a hard budget cap."""

    def __init__(self, directory: str | Path, *, hard_cap_usd: float | None = None,
                 price_table: PriceTable | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "cost_events.jsonl"
        self.hard_cap_usd = hard_cap_usd
        self.price_table = price_table
        self._total_usd = 0.0
        self._total_api_usd = 0.0
        self._total_compute_hours = 0.0
        self._calls = 0

    # -- metering ----------------------------------------------------------------

    def record(
        self,
        *,
        provider: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        price_per_mtok: tuple[float | None, float | None],
    ) -> dict[str, Any]:
        prompt_price, completion_price = price_per_mtok
        if prompt_price is None or completion_price is None:
            cost_usd = None
        else:
            cost_usd = (
                prompt_tokens * prompt_price / 1_000_000
                + completion_tokens * completion_price / 1_000_000
            )
        event = {
            "schema_version": "1.0.0",
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "kind": "api_call",
            "provider": provider,
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "price_usd_per_mtok": {"prompt": prompt_price, "completion": completion_price},
            "cost_usd": cost_usd,
        }
        if cost_usd is not None:
            self._total_usd += cost_usd
            self._total_api_usd += cost_usd
            if self.hard_cap_usd is not None and self._total_usd > self.hard_cap_usd:
                self._append(event)
                self._append(self._cap_event())
                raise BudgetExceededError(
                    f"hard budget cap exceeded: ${self._total_usd:.2f} > "
                    f"${self.hard_cap_usd:.2f}; run halted fail-closed "
                    "(verify the ledger at costs/cost_events.jsonl)"
                )
        self._calls += 1
        self._append(event)
        return event

    def record_compute(
        self,
        *,
        provider: str,
        instance_type: str,
        hours: float,
        usd_per_hour: float,
        source_url: str = "",
    ) -> dict[str, Any]:
        cost_usd = hours * usd_per_hour
        self._total_usd += cost_usd
        self._total_compute_hours += hours
        event = {
            "schema_version": "1.0.0",
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "kind": "compute_hours",
            "provider": provider,
            "instance_type": instance_type,
            "hours": hours,
            "usd_per_hour": usd_per_hour,
            "cost_usd": cost_usd,
            "source_url": source_url,
        }
        if self.hard_cap_usd is not None and self._total_usd > self.hard_cap_usd:
            self._append(event)
            self._append(self._cap_event())
            raise BudgetExceededError(
                f"hard budget cap exceeded: ${self._total_usd:.2f} > ${self.hard_cap_usd:.2f}"
            )
        self._append(event)
        return event

    # -- reporting ------------------------------------------------------------------

    def _cap_event(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "kind": "budget_cap_exceeded",
            "hard_cap_usd": self.hard_cap_usd,
            "total_usd_at_cap": round(self._total_usd, 4),
        }

    def _append(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "total_usd": round(self._total_usd, 4),
            "api_usd": round(self._total_api_usd, 4),
            "compute_hours": round(self._total_compute_hours, 3),
            "api_calls_metered": self._calls,
            "hard_cap_usd": self.hard_cap_usd,
            "cap_headroom_usd": (
                round(self.hard_cap_usd - self._total_usd, 4)
                if self.hard_cap_usd is not None else None
            ),
        }

    def write_summary(self) -> dict[str, Any]:
        summary = self.summary()
        target = self.directory / "summary.json"
        tmp = self.directory / ".summary.tmp"
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
        return summary


__all__ = [
    "PriceEntry",
    "PriceTable",
    "price_for",
    "CostLedger",
    "BudgetExceededError",
]
