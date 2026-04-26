"""Server-Sent Events helpers."""

import json
from typing import Any


def sse(event: str, data: Any) -> str:
    """Format a single SSE message. Each line is JSON-encoded."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
