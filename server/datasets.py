from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class UploadedLogEntry:
    ts_ms: int
    source: str
    raw: str
    fields: Dict[str, Any]


UPLOADED_LOGS: List[UploadedLogEntry] = []


def _to_entry(source: str, raw: str, fields: Dict[str, Any] | None = None) -> UploadedLogEntry:
    return UploadedLogEntry(
        ts_ms=int(time.time() * 1000),
        source=source,
        raw=raw.strip(),
        fields=fields or {},
    )


def add_logs_from_content(filename: str, content: bytes) -> int:
    lower = filename.lower()
    text = content.decode("utf-8", errors="ignore")
    before = len(UPLOADED_LOGS)

    if lower.endswith(".json") or lower.endswith(".jsonl"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    UPLOADED_LOGS.append(_to_entry(filename, json.dumps(payload), payload))
                else:
                    UPLOADED_LOGS.append(_to_entry(filename, str(payload)))
            except json.JSONDecodeError:
                UPLOADED_LOGS.append(_to_entry(filename, line))
    elif lower.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            raw = ", ".join(f"{k}={v}" for k, v in row.items())
            UPLOADED_LOGS.append(_to_entry(filename, raw, dict(row)))
    else:
        # .log/.txt and unknown plain text: one line = one log
        for line in text.splitlines():
            line = line.strip()
            if line:
                UPLOADED_LOGS.append(_to_entry(filename, line))

    return len(UPLOADED_LOGS) - before


def clear_uploaded_logs() -> None:
    UPLOADED_LOGS.clear()


def search_uploaded_logs(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    q = (query or "").lower().strip()
    if not q:
        return [asdict(entry) for entry in UPLOADED_LOGS[:max_results]]

    hits: List[Dict[str, Any]] = []
    for entry in UPLOADED_LOGS:
        blob = f"{entry.raw} {json.dumps(entry.fields, ensure_ascii=True)}".lower()
        if q in blob:
            hits.append(asdict(entry))
            if len(hits) >= max_results:
                break
    return hits


def uploaded_logs_summary() -> Dict[str, Any]:
    sources: Dict[str, int] = {}
    for item in UPLOADED_LOGS:
        sources[item.source] = sources.get(item.source, 0) + 1
    return {"total_logs": len(UPLOADED_LOGS), "sources": sources}
