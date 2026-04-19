"""Push-to-talk and wake-word gating for noisy environments."""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None

try:
    import numpy as np
    try:
        from openwakeword.model import Model as OWWModel
    except ImportError:
        from openwakeword import Model as OWWModel  # type: ignore
except ImportError:
    np = None  # type: ignore
    OWWModel = None  # type: ignore


class PushToTalkController:
    """Hold key to open audio gate (pynput). Optional stdin Enter-to-arm for IDE terminals."""

    def __init__(
        self,
        key_name: str = "space",
        stdin_arm_seconds: float = 30.0,
        startup_open_seconds: float = 12.0,
    ) -> None:
        self._key_name = (key_name or "space").lower().strip()
        self._stdin_arm_seconds = max(0.0, float(stdin_arm_seconds))
        self._startup_open_seconds = max(0.0, float(startup_open_seconds))
        self._pressed = threading.Event()
        self._listener: Optional[object] = None
        self._arm_until = 0.0
        self._stop = threading.Event()
        self._stdin_thread: Optional[threading.Thread] = None

    @property
    def is_pressed(self) -> bool:
        if time.monotonic() < self._arm_until:
            return True
        return self._pressed.is_set()

    def extend_open_window(self, seconds: float) -> None:
        """Keep the mic gate open for at least `seconds` from now (does not shorten an existing window)."""
        if seconds <= 0:
            return
        until = time.monotonic() + seconds
        self._arm_until = max(self._arm_until, until)

    def _stdin_arm_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except (EOFError, OSError):
                break
            if self._stop.is_set():
                break
            # Closed / non-interactive stdin: do not arm or spin hot.
            if line == "":
                logger.debug("PTT stdin arm thread exiting (stdin EOF or empty read)")
                break
            self._arm_until = time.monotonic() + self._stdin_arm_seconds
            logger.info("PTT armed for ~%.0fs — speak now", self._stdin_arm_seconds)

    def start(self) -> None:
        # Do not gate on isatty(): Cursor/VS Code often report a pipe (not a TTY) even
        # though readline() still works for Enter-to-arm.
        use_stdin = self._stdin_arm_seconds > 0
        use_pynput = pynput_keyboard is not None

        if not use_stdin and not use_pynput:
            raise RuntimeError(
                "push_to_talk needs pynput (pip install pynput) or set "
                "push_to_talk_stdin_arm_seconds > 0 for Enter-to-arm in the terminal"
            )

        # Open gate briefly at startup so STT/VAD get real audio without Space/Enter
        # (otherwise gate stays closed, only silence flows, and INFO logs look "stuck").
        if self._startup_open_seconds > 0:
            self._arm_until = time.monotonic() + self._startup_open_seconds
            logger.info(
                "PTT: mic open for the first ~%.0fs — speak now. Then press Enter or hold Space.",
                self._startup_open_seconds,
            )

        if use_stdin:
            logger.info(
                "PTT: press Enter in this terminal to open the mic for ~%.0fs "
                "(IDE terminals often do not deliver Space to push-to-talk).",
                self._stdin_arm_seconds,
            )
            self._stdin_thread = threading.Thread(target=self._stdin_arm_loop, daemon=True)
            self._stdin_thread.start()

        if use_pynput:
            key_obj = self._resolve_key(self._key_name)

            def on_press(key):
                try:
                    if key == key_obj or (
                        hasattr(key, "char") and key.char and key.char.lower() == self._key_name
                    ):
                        self._pressed.set()
                except Exception:
                    pass

            def on_release(key):
                try:
                    if key == key_obj or (
                        hasattr(key, "char") and key.char and key.char.lower() == self._key_name
                    ):
                        self._pressed.clear()
                except Exception:
                    pass

            self._listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
            logger.info("Push-to-talk: hold '%s' to speak", self._key_name)
        elif self._stdin_arm_seconds > 0:
            logger.warning("pynput not installed; using stdin Enter-to-arm only for PTT")

    def stop(self) -> None:
        self._stop.set()
        self._arm_until = 0.0
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as e:
                logger.debug("pynput listener stop: %s", e)
            self._listener = None
        self._pressed.clear()

    @staticmethod
    def _resolve_key(name: str):
        k = pynput_keyboard.Key
        mapping = {
            "space": k.space,
            "ctrl": k.ctrl,
            "ctrl_l": k.ctrl_l,
            "shift": k.shift,
            "alt": k.alt,
            "tab": k.tab,
        }
        return mapping.get(name, k.space)


class WakeWordGate:
    """Open audio for a window after OpenWakeWord detects a phrase (16 kHz int16)."""

    # OpenWakeWord default frame size for many models
    _FRAME_SAMPLES = 1280

    def __init__(
        self,
        models: Optional[list[str]] = None,
        sensitivity: float = 0.5,
        open_seconds: float = 30.0,
        on_wake: Optional[Callable[[], None]] = None,
    ) -> None:
        self._models = models or ["alexa"]
        self._sensitivity = sensitivity
        self._open_seconds = open_seconds
        self._on_wake = on_wake
        self._open_until = 0.0
        self._buffer = np.array([], dtype=np.int16) if np is not None else None
        self._oww = None

        if OWWModel is not None and np is not None:
            try:
                self._oww = OWWModel(wakeword_models=self._models)
                logger.info("OpenWakeWord loaded models: %s", self._models)
            except Exception as e:
                logger.error("OpenWakeWord init failed: %s", e)
                self._oww = None

    @property
    def is_available(self) -> bool:
        return self._oww is not None and np is not None

    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    def extend_window(self) -> None:
        self._open_until = time.monotonic() + self._open_seconds

    def process_chunk(self, samples: "np.ndarray") -> None:
        """Append PCM int16; run prediction when enough samples."""
        if self._oww is None or self._buffer is None:
            return
        if np is None:
            return
        self._buffer = np.concatenate([self._buffer, samples.astype(np.int16)])
        while len(self._buffer) >= self._FRAME_SAMPLES:
            frame = self._buffer[: self._FRAME_SAMPLES]
            self._buffer = self._buffer[self._FRAME_SAMPLES :]
            try:
                prediction = self._oww.predict(frame)
            except Exception as e:
                logger.debug("oww predict: %s", e)
                continue
            if not prediction:
                continue
            # prediction maps model name -> score
            best = max(float(v) for v in prediction.values()) if prediction else 0.0
            if best >= self._sensitivity:
                logger.info("Wake word detected (score=%.3f), opening gate %.1fs", best, self._open_seconds)
                self.extend_window()
                if self._on_wake:
                    try:
                        self._on_wake()
                    except Exception:
                        pass

    def reset(self) -> None:
        if self._buffer is not None:
            self._buffer = np.array([], dtype=np.int16)
