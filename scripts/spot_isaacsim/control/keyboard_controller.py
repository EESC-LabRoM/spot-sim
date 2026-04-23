"""
Keyboard controller backends for Spot Isaac Sim.

Two implementations behind a common interface:

    CarbKeyboardController  — headed mode (omni.appwindow + carb.input subscription)
    StdinKeyboardController — headless mode (stdin raw reader thread)

Both expose:
    subscribe(callback: (key_name: str, pressed: bool) -> None)

Use the factory to get the right one automatically:
    from .keyboard_controller import create_keyboard_controller
    kb = create_keyboard_controller()
    kb.subscribe(locomotion.on_key_event)
    kb.subscribe(arm_controller.on_key_event)

Key names emitted match the carb key name convention used by _KEY_POSE_MAP:
    UP, DOWN, LEFT, RIGHT, N, M
    NUMPAD_5, NUMPAD_0, NUMPAD_1
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
# Locomotion key map — carb key name → [dx_body, dy_body, dyaw] per step
# Shared between CarbKeyboardController and StdinKeyboardController so both
# modes use identical logical key names.
# ---------------------------------------------------------------------------
_KB_POS_RATE = 0.04   # m per forward() call at 50 Hz
_KB_YAW_RATE = 0.04   # rad per forward() call at 50 Hz

KEY_POSE_MAP: dict[str, list] = {
    "UP":       [+_KB_POS_RATE,  0.0,           0.0          ],
    "NUMPAD_8": [+_KB_POS_RATE,  0.0,           0.0          ],
    "DOWN":     [-_KB_POS_RATE,  0.0,           0.0          ],
    "NUMPAD_2": [-_KB_POS_RATE,  0.0,           0.0          ],
    "LEFT":     [ 0.0,          +_KB_POS_RATE,   0.0          ],
    "NUMPAD_4": [ 0.0,          +_KB_POS_RATE,   0.0          ],
    "RIGHT":    [ 0.0,          -_KB_POS_RATE,   0.0          ],
    "NUMPAD_6": [ 0.0,          -_KB_POS_RATE,   0.0          ],
    "N":        [ 0.0,           0.0,           +_KB_YAW_RATE ],
    "NUMPAD_7": [ 0.0,           0.0,           +_KB_YAW_RATE ],
    "M":        [ 0.0,           0.0,           -_KB_YAW_RATE ],
    "NUMPAD_9": [ 0.0,           0.0,           -_KB_YAW_RATE ],
}

# Arm keys — same carb key names used by both backends
ARM_KEY_TOGGLE  = "NUMPAD_5"
ARM_KEY_RESTING = "NUMPAD_0"
ARM_KEY_STANDING = "NUMPAD_1"

# ---------------------------------------------------------------------------
# Stdin key map — terminal chars/sequences → carb key names
# Maps WASD + numpad-style chars to the same logical names used by carb,
# so locomotion_controller and arm_controller need no headless-specific code.
# ---------------------------------------------------------------------------
_SEQ_MAP: dict[str, str] = {
    # Locomotion
    "w": "UP",       "W": "UP",
    "s": "DOWN",     "S": "DOWN",
    "a": "LEFT",     "A": "LEFT",
    "d": "RIGHT",    "D": "RIGHT",
    "n": "N",        "N": "N",
    "m": "M",        "M": "M",
    # Arm
    "5": ARM_KEY_TOGGLE,
    "0": ARM_KEY_RESTING,
    "1": ARM_KEY_STANDING,
}

# ---------------------------------------------------------------------------
# Carb (headed) backend
# ---------------------------------------------------------------------------

class CarbKeyboardController:
    """Keyboard controller for headed Isaac Sim — uses omni.appwindow + carb.input."""

    def __init__(self) -> None:
        import carb
        import omni.appwindow
        self._callbacks: list[Callable[[str, bool], None]] = []
        self._carb = carb
        appwindow = omni.appwindow.get_default_app_window()
        iface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        self._sub = iface.subscribe_to_keyboard_events(keyboard, self._on_event)
        print("[KB] Carb keyboard active (headed mode)")
        print("[KB]   Locomotion : UP/DOWN (forward/back)  LEFT/RIGHT (strafe)  N/M (yaw)")
        print("[KB]   Arm        : Numpad5 (toggle tracking)  Numpad0 (resting)  Numpad1 (standing)")

    def subscribe(self, callback: Callable[[str, bool], None]) -> None:
        self._callbacks.append(callback)

    def _on_event(self, event) -> bool:
        key_name = event.input.name if hasattr(event.input, 'name') else str(event.input)
        pressed  = event.type == self._carb.input.KeyboardEventType.KEY_PRESS
        released = event.type == self._carb.input.KeyboardEventType.KEY_RELEASE
        if pressed or released:
            for cb in self._callbacks:
                cb(key_name, pressed)
        return True


# ---------------------------------------------------------------------------
# Stdin (headless) backend
# ---------------------------------------------------------------------------

class StdinKeyboardController:
    """Keyboard controller for headless Isaac Sim — reads raw keypresses from stdin.

    KEY_RELEASE is simulated: a per-key timer fires 150 ms after the last
    keypress. Terminal auto-repeat fires every ~50 ms, so any held key keeps
    resetting the timer; the release fires only once the key is actually lifted.
    """

    def __init__(self) -> None:
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

        print("[KB] Stdin keyboard active (headless mode)")
        print("[KB]   Locomotion : UP/DOWN (forward/back)  LEFT/RIGHT (strafe)  N/M (yaw)")
        print("[KB]   Arm        : 5 (toggle tracking)     0 (resting)          1 (standing)")

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
            except Exception:
                pass

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

def create_keyboard_controller(headless: bool = False) -> CarbKeyboardController | StdinKeyboardController:
    """Return the appropriate keyboard controller.

    Args:
        headless: Pass True when SimulationApp was launched with headless=True.
                  carb creates a keyboard object even in headless mode, so the
                  get_keyboard() check is unreliable — the caller must be explicit.
    """
    if headless:
        return StdinKeyboardController()
    return CarbKeyboardController()
