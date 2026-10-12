"""
Static checks on the review page (clipforge/server/static/index.html).

The behavioural checks (escaping, in-place updates, polling, toasts) need a
real browser and live in tests/test_ui_browser.py.
"""
import re
from html.parser import HTMLParser
from pathlib import Path

INDEX = Path(__file__).parents[1] / "clipforge" / "server" / "static" / "index.html"


def _source() -> str:
    return INDEX.read_text(encoding="utf-8")


class _Attrs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.attrs: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        self.attrs += [(tag, name) for name, _ in attrs]

    handle_startendtag = handle_starttag


def test_no_inline_event_handler_attributes():
    p = _Attrs()
    p.feed(_source())
    inline = [(t, a) for t, a in p.attrs if a.lower().startswith("on")]
    assert inline == []


def test_no_handler_attributes_built_in_js_strings():
    # catches onclick="..." inside JS template strings, which the HTML
    # parser cannot see because it is inside <script>
    hits = re.findall(r"""[\s'"`<]on[a-z]+\s*=\s*[\\"'`]""", _source(), re.I)
    assert hits == []


def test_no_browser_dialogs():
    src = _source()
    for call in ("alert", "confirm", "prompt"):
        assert not re.search(rf"(?<![\w.]){call}\s*\(", src), call
        assert not re.search(rf"window\.{call}\s*\(", src), call


def test_no_html_string_injection():
    # server-supplied text must reach the DOM via textContent/.value only
    src = _source()
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML",
                 "document.write"):
        assert sink not in src, sink


def test_no_other_markup_or_handler_injection_routes():
    src = _source()
    # event handlers set through the DOM API with a literal attribute name
    assert not re.search(r"""setAttribute(NS)?\(\s*["'`](\w+:)?on""", src, re.I)
    for api in ("createContextualFragment", "DOMParser", "srcdoc",
                "eval(", "new Function("):
        assert api not in src, api


def test_polling_is_not_a_fixed_interval():
    # polling is a timer re-armed by refresh() only while a job is running;
    # a setInterval would poll forever
    assert not re.search(r"(?<![\w.])setInterval\s*\(", _source())
    assert not re.search(r"window\.setInterval\s*\(", _source())
