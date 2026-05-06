"""
Keyboard controller for Spot Isaac Sim — stdin backend only.

Single implementation for both headed and headless modes:

    StdinKeyboardController  — raw stdin reader thread + optional carb NUMPAD_5 in GUI mode

Exposes:
    subscribe(callback: (key_name: str, pressed: bool) -> None)
    update()   # must be called each physics step from the main thread

Use the factory:
    from .keyboard_controller import create_keyboard_controller
    kb = create_keyboard_controller(headless=False)
    kb.subscribe(locomotion.on_key_event)
    kb.subscribe(arm_controller.on_key_event)

In headless mode:  WASD + n/m for locomotion, 5/0/1/space for arm (terminal keys).
In headed (GUI) mode: same terminal keys, plus physical NUMPAD_5 fires the arm toggle via carb.
"""

import atexit
import queue
import select
import sys
import termios
import threading
import tty
from typing import Callable

# ---------------------------------------------------------------------------
# Locomotion key map — logical key name → [dx_body, dy_body, dyaw] per step
# ---------------------------------------------------------------------------
_KB_POS_RATE = 0.04   # m per forward() call at 50 Hz
_KB_YAW_RATE = 0.04   # rad per forward() call at 50 Hz

KEY_POSE_MAP: dict[str, list] = {
    "UP":    [+_KB_POS_RATE,  0.0,           0.0          ],
    "DOWN":  [-_KB_POS_RATE,  0.0,           0.0          ],
    "LEFT":  [ 0.0,          +_KB_POS_RATE,  0.0          ],
    "RIGHT": [ 0.0,          -_KB_POS_RATE,  0.0          ],
    "Q":     [ 0.0,           0.0,          +_KB_YAW_RATE ],
    "E":     [ 0.0,           0.0,          -_KB_YAW_RATE ],
}

# Arm key logical names
ARM_KEY_TOGGLE   = "NUMPAD_5"
ARM_KEY_RESTING  = "NUMPAD_0"
ARM_KEY_STANDING = "NUMPAD_1"
ARM_KEY_GRIPPER  = "SPACE"

# Terminal char/sequence → logical key name
_SEQ_MAP: dict[str, str] = {
    "w": "UP",    "W": "UP",
    "s": "DOWN",  "S": "DOWN",
    "a": "LEFT",  "A": "LEFT",
    "d": "RIGHT", "D": "RIGHT",
    "q": "Q",     "Q": "Q",
    "e": "E",     "E": "E",
    "5": ARM_KEY_TOGGLE,
    "0": ARM_KEY_RESTING,
    "1": ARM_KEY_STANDING,
    " ": ARM_KEY_GRIPPER,
}


# ---------------------------------------------------------------------------
# Stdin keyboard controller
# ---------------------------------------------------------------------------

class StdinKeyboardController:
    """Keyboard controller — reads raw keypresses from stdin.

    Works in both headless and headed (GUI) modes. In headed mode, also
    subscribes a minimal carb listener so the physical NUMPAD_5 key fires
    the arm toggle without needing terminal focus.

    KEY_RELEASE is simulated: a per-key timer fires 150 ms after the last
    keypress. Terminal auto-repeat fires every ~50 ms, so any held key keeps
    resetting the timer; the release fires only once the key is actually lifted.
    """

    def __init__(self, headed: bool = False) -> None:
        self._callbacks: list[Callable[[str, bool], None]] = []
        self._timers: dict[str, threading.Timer] = {}
        self._timers_lock = threading.Lock()
        self._event_queue: queue.SimpleQueue = queue.SimpleQueue()

        if not sys.stdin.isatty():
            print("[KB] stdin is not a TTY — headless keyboard unavailable")
            return

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)

        # Set raw mode here on the main thread so atexit always restores it,
        # even if the reader thread dies or the process crashes.
        tty.setraw(self._fd)
        atexit.register(self._restore_terminal)

        t = threading.Thread(target=self._reader_loop, daemon=True, name="StdinKeyboard")
        t.start()

        print("[KB] Stdin keyboard active")
        print("[KB]   Locomotion : W/S (forward/back)  A/D (strafe)  N/M (yaw)")
        print("[KB]   Arm        : 5 (toggle tracking)  0 (resting)  1 (standing)  Space (gripper)")

        if headed:
            self._attach_carb_numpad5()

    def subscribe(self, callback: Callable[[str, bool], None]) -> None:
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Drain the event queue and call subscribers.

        Must be called from the main physics thread (SpotRobot.update).
        Background threads never call subscribers directly — they only
        put events here so Isaac Sim state is only touched from one thread.
        """
        while not self._event_queue.empty():
            try:
                key_name, pressed = self._event_queue.get_nowait()
            except queue.Empty:
                break
            for cb in self._callbacks:
                try:
                    cb(key_name, pressed)
                except Exception as exc:
                    print(f"[KB] Callback error for '{key_name}' pressed={pressed}: {exc}")

    # ------------------------------------------------------------------
    # Carb NUMPAD_5 shim — headed (GUI) mode only
    # ------------------------------------------------------------------

    def _attach_carb_numpad5(self) -> None:
        """Subscribe a minimal carb listener that forwards only NUMPAD_5."""
        import carb
        import omni.appwindow
        iface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()

        def _on_event(event) -> bool:
            key_name = event.input.name if hasattr(event.input, 'name') else str(event.input)
            if key_name == ARM_KEY_TOGGLE and event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self._enqueue_press(ARM_KEY_TOGGLE)
            return True

        self._carb_sub = iface.subscribe_to_keyboard_events(keyboard, _on_event)
        print("[KB] Carb NUMPAD_5 listener active (GUI arm-toggle)")

    # ------------------------------------------------------------------
    # Internal — called from background threads, only touch _event_queue
    # ------------------------------------------------------------------

    def _enqueue_press(self, key_name: str) -> None:
        with self._timers_lock:
            existing = self._timers.get(key_name)
            if existing is not None:
                existing.cancel()   # key still held — extend timer, no new press event
            else:
                self._event_queue.put((key_name, True))   # genuine first press
            t = threading.Timer(0.15, self._enqueue_release, args=(key_name,))
            t.daemon = True
            t.start()
            self._timers[key_name] = t

    def _enqueue_release(self, key_name: str) -> None:
        with self._timers_lock:
            self._timers.pop(key_name, None)
        self._event_queue.put((key_name, False))

    def _restore_terminal(self) -> None:
        if hasattr(self, "_old_settings"):
            try:
                # TCSANOW: restore immediately — TCSADRAIN can hang if the
                # process is crashing and stdout is no longer draining.
                termios.tcsetattr(self._fd, termios.TCSANOW, self._old_settings)
            except termios.error as exc:
                print(f"[KB] Failed to restore terminal settings: {exc}")

    def _reader_loop(self) -> None:
        # Raw mode is already set by __init__ — do not call tty.setraw here.
        while True:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (OSError, ValueError):
                # stdin fd closed or became invalid (sim crash) — exit cleanly
                break
            if not ready:
                continue

            try:
                ch = sys.stdin.read(1)
            except OSError:
                break

            if ch == "\x03":    # Ctrl+C — restore terminal then signal main thread
                import os, signal
                self._restore_terminal()
                os.kill(os.getpid(), signal.SIGINT)
                break

            if ch == "\x1b":
                # Arrow keys arrive as 3-byte ESC sequences: ESC [ A/B/C/D.
                # Read the two suffix bytes separately to avoid blocking if
                # only one byte is buffered.
                try:
                    r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r2:
                        b1 = sys.stdin.read(1)
                        r3, _, _ = select.select([sys.stdin], [], [], 0.02)
                        b2 = sys.stdin.read(1) if r3 else ""
                        seq = ch + b1 + b2
                    else:
                        seq = ch
                except OSError:
                    break
            else:
                seq = ch

            key = _SEQ_MAP.get(seq)
            if key:
                self._enqueue_press(key)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_keyboard_controller(headless: bool = False) -> StdinKeyboardController:
    """Return a StdinKeyboardController.

    Args:
        headless: Pass True when SimulationApp was launched with headless=True.
                  In headed mode, also attaches a carb subscriber for NUMPAD_5.
    """
    return StdinKeyboardController(headed=not headless)
