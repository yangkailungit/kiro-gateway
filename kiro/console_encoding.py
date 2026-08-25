# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Console encoding hardening for non-UTF-8 terminals.

Problem this module solves:
    The gateway prints Unicode characters (emoji in the startup banner, arrows
    and check marks in log messages). On Windows the console defaults to a
    legacy code page (cp936/GBK for Chinese locales, cp1251, cp932, ...).
    Writing a character that the code page cannot represent raises
    ``UnicodeEncodeError``, which kills the process during startup - the
    infamous "the .bat window just flashes and disappears" symptom.

Strategy (applied in order, each step degrades gracefully):
    1. Switch the Windows console output code page to UTF-8 (65001) so Unicode
       is rendered correctly instead of turning into mojibake. The previous
       code page is restored at interpreter exit.
    2. Reconfigure ``sys.stdout`` / ``sys.stderr`` to UTF-8 with the
       ``backslashreplace`` error handler. Reconfiguration is done in place, so
       sinks that already captured the stream object (loguru, uvicorn) keep
       working. The error handler guarantees that no print can ever raise again,
       even if step 1 and step 2 both fail on an exotic stream.
    3. Expose :func:`symbol` so presentation code can fall back to ASCII
       equivalents when the output stream still cannot encode Unicode.

Typical usage (as early as possible in the entry point)::

    from kiro.console_encoding import configure_console_encoding
    configure_console_encoding()
"""

import atexit
import sys
from typing import Any, Dict, Optional, TextIO, Tuple

# Windows code page identifier for UTF-8
WINDOWS_UTF8_CODE_PAGE = 65001

# Target encoding and error handler for standard streams.
# "backslashreplace" is preferred over "replace" because it keeps the output
# debuggable (\U0001f47b instead of an anonymous "?").
TARGET_ENCODING = "utf-8"
TARGET_ERROR_HANDLER = "backslashreplace"

# Characters used to probe whether a stream can render our output.
# Covers the three risky classes we actually emit: astral-plane emoji,
# arrows and box-drawing characters.
UNICODE_PROBE = "\U0001f47b\u2192\u2500"

# Unicode symbols with ASCII fallbacks, keyed by logical name.
# Extend this map instead of hardcoding glyphs in presentation code.
SYMBOLS: Dict[str, Tuple[str, str]] = {
    "ghost": ("\U0001f47b", "*"),          # 👻 banner logo
    "arrow": ("\u279c", "->"),             # ➜ pointer
    "speech": ("\U0001f4ac", "//"),        # 💬 feedback invitation
    "hline": ("\u2500", "-"),              # ─ horizontal rule
}

# Cached result of the last configure/detect cycle (None = not computed yet)
_unicode_output_supported: Optional[bool] = None


def _restore_console_code_page(code_page: int) -> None:
    """
    Restore the Windows console output code page.

    Registered with :mod:`atexit` so an interactive terminal is not left in
    UTF-8 mode after the gateway exits.

    Args:
        code_page: Code page identifier to restore
    """
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(code_page)  # type: ignore[attr-defined]
    except (AttributeError, OSError, ImportError, ValueError):
        # Nothing sensible to do while the interpreter is shutting down
        pass


def _enable_windows_utf8_code_page() -> bool:
    """
    Switch the attached Windows console to the UTF-8 output code page.

    Returns:
        True if the console output code page is UTF-8 after the call,
        False on non-Windows platforms, when no console is attached
        (output redirected to a file or pipe), or when the switch failed
    """
    if sys.platform != "win32":
        return False

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        current_code_page = int(kernel32.GetConsoleOutputCP())
    except (AttributeError, OSError, ImportError, ValueError):
        return False

    # 0 means "no console attached" (stdout redirected to a file or pipe)
    if current_code_page == 0:
        return False

    if current_code_page == WINDOWS_UTF8_CODE_PAGE:
        return True

    try:
        if not kernel32.SetConsoleOutputCP(WINDOWS_UTF8_CODE_PAGE):
            return False
    except (AttributeError, OSError, ValueError):
        return False

    atexit.register(_restore_console_code_page, current_code_page)
    return True


def _stream_can_encode(stream: Any, probe: str = UNICODE_PROBE) -> bool:
    """
    Check whether a stream's encoding can represent the given characters.

    Args:
        stream: Stream-like object (may be None or lack an ``encoding``)
        probe: Characters to test

    Returns:
        True if the stream reports an encoding capable of encoding ``probe``
    """
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False

    try:
        probe.encode(encoding)
    except (UnicodeEncodeError, LookupError, TypeError, ValueError):
        return False

    return True


def _reconfigure_stream(stream: Optional[TextIO]) -> bool:
    """
    Reconfigure a text stream to UTF-8 in place.

    In-place reconfiguration (as opposed to wrapping or replacing the stream)
    is required so that already-registered sinks - loguru writes to the
    ``sys.stderr`` object captured at import time - stay valid.

    Args:
        stream: Stream to reconfigure, may be None (``pythonw``) or a
            non-reconfigurable object (pytest capture, custom wrappers)

    Returns:
        True if the stream can encode Unicode after the call
    """
    if stream is None:
        return False

    if _stream_can_encode(stream):
        # Already UTF-8 (or another Unicode-capable codec): still make sure a
        # surprising character cannot abort the process.
        _harden_error_handler(stream)
        return True

    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return False

    try:
        reconfigure(encoding=TARGET_ENCODING, errors=TARGET_ERROR_HANDLER)
    except (AttributeError, OSError, ValueError, TypeError, LookupError):
        return False

    return _stream_can_encode(stream)


def _harden_error_handler(stream: Any) -> None:
    """
    Make encoding failures non-fatal for an already Unicode-capable stream.

    Args:
        stream: Stream to adjust; unsupported streams are left untouched
    """
    if getattr(stream, "errors", None) in ("backslashreplace", "replace", "ignore", "xmlcharrefreplace"):
        return

    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return

    try:
        reconfigure(errors=TARGET_ERROR_HANDLER)
    except (AttributeError, OSError, ValueError, TypeError, LookupError):
        pass


def configure_console_encoding() -> bool:
    """
    Make standard output safe for the Unicode characters the gateway emits.

    Idempotent: safe to call multiple times (for example from both the CLI
    entry point and an embedding application).

    Returns:
        True if both ``sys.stdout`` and ``sys.stderr`` can render Unicode
        after configuration, False if presentation code should fall back to
        ASCII symbols
    """
    global _unicode_output_supported

    _enable_windows_utf8_code_page()

    stdout_ok = _reconfigure_stream(sys.stdout)
    stderr_ok = _reconfigure_stream(sys.stderr)

    _unicode_output_supported = stdout_ok and stderr_ok
    return _unicode_output_supported


def unicode_output_supported() -> bool:
    """
    Report whether Unicode output is safe to print.

    Detects the current stream state on first use if
    :func:`configure_console_encoding` has not been called yet.

    Returns:
        True if both standard streams can encode Unicode
    """
    global _unicode_output_supported

    if _unicode_output_supported is None:
        _unicode_output_supported = _stream_can_encode(sys.stdout) and _stream_can_encode(sys.stderr)

    return _unicode_output_supported


def reset_encoding_state() -> None:
    """
    Drop the cached Unicode-support verdict.

    Intended for tests that swap ``sys.stdout`` / ``sys.stderr``.
    """
    global _unicode_output_supported
    _unicode_output_supported = None


def symbol(name: str) -> str:
    """
    Return a decorative symbol appropriate for the current console.

    Args:
        name: Logical symbol name, must be a key of :data:`SYMBOLS`

    Returns:
        The Unicode glyph when the console supports it, the ASCII fallback
        otherwise

    Raises:
        KeyError: If ``name`` is not a known symbol

    Examples:
        >>> symbol("arrow") in ("\u279c", "->")
        True
    """
    unicode_glyph, ascii_glyph = SYMBOLS[name]
    return unicode_glyph if unicode_output_supported() else ascii_glyph
