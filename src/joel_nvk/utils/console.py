"""Console encoding.

Windows consoles often run a legacy codepage (cp1252, cp932, ...) that cannot
encode the characters in our help text, log lines and result tables — printing
one raises ``UnicodeEncodeError`` and takes the process down. Every entry point
calls ``use_utf8_output`` first so output degrades to a replacement character
instead of a crash.
"""

import sys


def use_utf8_output() -> None:
    """Switch stdout/stderr to UTF-8, replacing anything the console cannot show."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Redirected to something that cannot be reconfigured; leave it be.
            pass
