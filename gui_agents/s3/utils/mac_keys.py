"""Reliable key combinations on macOS.

pyautogui.hotkey('command', 'v') sends the modifier and the key as separate,
very fast events; apps often see a plain "v" with no Command flag. Here each
key event carries the modifier flags itself (Quartz CGEvent), which is what
macOS apps check.
"""

import time

# ANSI US keyboard virtual key codes.
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
    "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38,
    "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46, ".": 47,
    "`": 50,
    "return": 36, "enter": 36, "tab": 48, "space": 49, "backspace": 51, "delete": 51,
    "escape": 53, "esc": 53, "forwarddelete": 117, "home": 115, "end": 119,
    "pageup": 116, "pagedown": 121, "left": 123, "right": 124, "down": 125, "up": 126,
}

# name -> (key code, Quartz flag attribute)
MODIFIERS = {
    "command": (55, "kCGEventFlagMaskCommand"),
    "cmd": (55, "kCGEventFlagMaskCommand"),
    "shift": (56, "kCGEventFlagMaskShift"),
    "option": (58, "kCGEventFlagMaskAlternate"),
    "alt": (58, "kCGEventFlagMaskAlternate"),
    "ctrl": (59, "kCGEventFlagMaskControl"),
    "control": (59, "kCGEventFlagMaskControl"),
}


def split_combo(keys):
    """(modifiers, other keys), lower-cased; raises KeyError for unknown keys."""
    keys = [str(k).lower() for k in keys]
    mods = [k for k in keys if k in MODIFIERS]
    others = [k for k in keys if k not in MODIFIERS]
    for k in others:
        KEYCODES[k]  # KeyError if we can't send it
    return mods, others


def press_combo(keys, hold=0.03):
    """Press keys together, e.g. press_combo(['command', 'v'])."""
    try:
        mods, others = split_combo(keys)
        import Quartz
    except (KeyError, ImportError):
        import pyautogui

        pyautogui.hotkey(*keys, interval=0.05)
        return

    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)

    def post(code, down, flags):
        event = Quartz.CGEventCreateKeyboardEvent(source, code, down)
        Quartz.CGEventSetFlags(event, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(hold)

    flags = 0
    for m in mods:
        code, mask = MODIFIERS[m]
        flags |= getattr(Quartz, mask)
        post(code, True, flags)
    for k in others:
        post(KEYCODES[k], True, flags)
        post(KEYCODES[k], False, flags)
    for m in reversed(mods):
        code, mask = MODIFIERS[m]
        flags &= ~getattr(Quartz, mask)
        post(code, False, flags)
