"""Per-turn latency metrics for tuning LLM and TTS pipelines."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnLatency:
    turn_id: int
    t_start: float
    user_text_preview: str = ""
    t_llm_first_token: Optional[float] = None
    t_llm_end: Optional[float] = None
    t_tts_first_chunk: Optional[float] = None
    t_playback_end: Optional[float] = None
    cached: bool = False


class LatencyMetrics:
    """Tracks timestamps for one conversation turn; logs summary at end."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._turn_id = 0
        self._current: Optional[TurnLatency] = None

    def begin_turn(self, user_text: str) -> None:
        if not self._enabled:
            return
        self._turn_id += 1
        self._current = TurnLatency(
            turn_id=self._turn_id,
            t_start=time.perf_counter(),
            user_text_preview=(user_text[:60] + "…") if len(user_text) > 60 else user_text,
        )

    def mark_llm_first_token(self) -> None:
        if not self._enabled or self._current is None:
            return
        if self._current.t_llm_first_token is None:
            self._current.t_llm_first_token = time.perf_counter()

    def mark_llm_end(self) -> None:
        if not self._enabled or self._current is None:
            return
        self._current.t_llm_end = time.perf_counter()

    def mark_tts_first_chunk(self) -> None:
        if not self._enabled or self._current is None:
            return
        if self._current.t_tts_first_chunk is None:
            self._current.t_tts_first_chunk = time.perf_counter()

    def mark_playback_end(self) -> None:
        if not self._enabled or self._current is None:
            return
        self._current.t_playback_end = time.perf_counter()

    def mark_cached_turn(self) -> None:
        if not self._enabled or self._current is None:
            return
        self._current.cached = True

    def log_turn_summary(self) -> None:
        if not self._enabled or self._current is None:
            return
        t = self._current
        base = t.t_start

        def _ms(x: Optional[float]) -> str:
            if x is None:
                return "n/a"
            return f"{(x - base) * 1000.0:.0f}"

        llm1 = t.t_llm_first_token
        tts1 = t.t_tts_first_chunk
        llm1_ms = f"{(llm1 - base) * 1000.0:.0f}" if llm1 is not None else "n/a"
        tts_after_llm_ms = (
            f"{(tts1 - llm1) * 1000.0:.0f}" if (llm1 is not None and tts1 is not None) else "n/a"
        )

        logger.info(
            "METRICS turn=%d cached=%s user=%r | "
            "start->llm_first_token_ms=%s | llm_first->tts_first_ms=%s | "
            "start->tts_first_ms=%s | fields=%s",
            t.turn_id,
            t.cached,
            t.user_text_preview,
            llm1_ms,
            tts_after_llm_ms,
            _ms(t.t_tts_first_chunk),
            {
                "llm_end_ms": _ms(t.t_llm_end),
                "playback_end_ms": _ms(t.t_playback_end),
            },
        )
        self._current = None
