"""logging_setup: the one log format, colored on a terminal and plain
anywhere else; and the dashboard's access-log filter."""

import io
import logging

import pytest

from cellmap_flow import logging_setup
from cellmap_flow.dashboard.app import _only_failed_requests, _startup_banner
from cellmap_flow.logging_setup import LOG_DATEFMT, LOG_FORMAT, ConsoleFormatter, wants_color


class _Tty(io.StringIO):
    def isatty(self):
        return True


def _record(level, message, name="cellmap_flow.jobs.lsf", exc_info=None):
    return logging.LogRecord(name, level, "lsf.py", 1, message, (), exc_info)


def test_a_terminal_gets_color_unless_asked_not_to(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert wants_color(_Tty())
    assert not wants_color(io.StringIO())  # a pipe, a file, pytest's capture
    monkeypatch.setenv("NO_COLOR", "1")
    assert not wants_color(_Tty())
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert not wants_color(_Tty())


def test_the_console_format_dims_the_frame_and_colors_the_level():
    line = ConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT).format(_record(logging.INFO, "Job 5150 submitted"))
    # Time dimmed, level green and padded, the package prefix gone from the
    # name, the message itself untouched.
    assert "\033[2m20" in line  # the dimmed year
    assert "\033[32mINFO   \033[0m" in line
    assert "\033[2mjobs.lsf:\033[0m Job 5150 submitted" in line
    assert "cellmap_flow." not in line


def test_a_warning_colors_its_message_too_and_an_exception_follows():
    try:
        raise ValueError("boom")
    except ValueError as e:
        exc_info = (type(e), e, e.__traceback__)
    line = ConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT).format(
        _record(logging.ERROR, "Could not submit", name="werkzeug", exc_info=exc_info))
    assert "\033[31mERROR  \033[0m" in line
    assert "\033[31mCould not submit\033[0m" in line
    assert "\033[2mwerkzeug:\033[0m" in line  # another package's name, as it is
    assert "ValueError: boom" in line.split("\n", 1)[1]


def test_a_url_in_a_message_is_set_apart_and_the_messages_color_resumes_after_it():
    formatter = ConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    line = formatter.format(_record(logging.INFO, "Found host: http://10.36.105.26:29235 (from its ready file)"))
    assert "Found host: \033[1;4;36mhttp://10.36.105.26:29235\033[0m (from its ready file)" in line

    line = formatter.format(_record(logging.WARNING, "No answer from http://10.36.105.26:29235 yet; waiting"))
    # The URL's own style, then back to the warning's yellow for the rest.
    assert "\033[33mNo answer from \033[1;4;36mhttp://10.36.105.26:29235\033[0m\033[33m yet; waiting\033[0m" in line


def test_the_startup_banner_puts_each_url_on_a_line_of_its_own():
    banner = _startup_banner("http://10.36.106.13:37521", "http://10.36.106.13:34691/v/689de0bd/")
    assert banner.splitlines() == [
        "Dashboard is up",
        "",
        "    Dashboard  http://10.36.106.13:37521",
        "    Viewer     http://10.36.106.13:34691/v/689de0bd/",
    ]
    assert banner.endswith("/\n")  # a blank line after, as before
    assert "Viewer" not in _startup_banner("http://10.36.106.13:37521")


def test_configure_logging_colors_only_a_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        plain = io.StringIO()
        monkeypatch.setattr("sys.stderr", plain)
        logging_setup.configure_logging()
        (handler,) = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert type(handler.formatter) is logging.Formatter

        monkeypatch.setattr("sys.stderr", _Tty())
        logging_setup.configure_logging()
        (handler,) = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert isinstance(handler.formatter, ConsoleFormatter)
    finally:
        logging.basicConfig(force=True)
        root.handlers[:] = saved


@pytest.mark.parametrize("status, kept", [
    ("200", False), ("304", False), ("302", False), ("101", False),
    ("404", True), ("400", True), ("500", True),
])
def test_the_dashboards_access_log_keeps_only_failed_requests(status, kept):
    record = logging.LogRecord(
        "werkzeug", logging.INFO, "serving.py", 1,
        '10.8.0.115 - - [02/Nov/2026 11:30:26] "%s" %s %s\n', ("GET /static/js/lib/dom.js HTTP/1.1", status, "-"), None,
    )
    assert _only_failed_requests(record) is kept


def test_a_werkzeug_record_without_a_status_is_kept():
    record = logging.LogRecord("werkzeug", logging.WARNING, "serving.py", 1, "WARNING: This is a development server.", (), None)
    assert _only_failed_requests(record) is True
