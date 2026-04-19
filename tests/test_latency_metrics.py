"""Tests for latency metrics helper."""

import time

import pytest

from src.latency_metrics import LatencyMetrics


def test_metrics_disabled_noop():
    m = LatencyMetrics(enabled=False)
    m.begin_turn("hello")
    m.mark_llm_first_token()
    m.log_turn_summary()  # should not raise


def test_metrics_turn_lifecycle():
    m = LatencyMetrics(enabled=True)
    m.begin_turn("test query")
    time.sleep(0.01)
    m.mark_llm_first_token()
    time.sleep(0.01)
    m.mark_llm_end()
    time.sleep(0.01)
    m.mark_tts_first_chunk()
    m.mark_playback_end()
    m.log_turn_summary()
