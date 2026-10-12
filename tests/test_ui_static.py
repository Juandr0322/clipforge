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


# ---- F06: keyboard shortcuts --------------------------------------------

SHORTCUT_KEYS = {"a", "r", " ", "j", "ArrowDown", "k", "ArrowUp", "[", "]",
                 "?", "Escape"}


def _shortcut_table() -> str:
    m = re.search(r"const SHORTCUTS = \[(.*?)\n\];", _source(), re.S)
    assert m, "SHORTCUTS table not found"
    return m.group(1)


def test_shortcut_table_lists_the_documented_keys():
    # one entry per key; the key handler and the help panel both read this
    # table, so what the help says is what the page does
    entries = re.findall(r"keys: \[(.*?)\],\s*label: \"([^\"]+)\"", _shortcut_table())
    keys = [k for group, _ in entries
            for k in re.findall(r'"((?:[^"\\]|\\.)*)"', group)]
    assert sorted(keys) == sorted(SHORTCUT_KEYS)
    assert len(entries) == len(re.findall(r"\bdesc:", _shortcut_table()))


def test_help_panel_and_key_handler_are_built_from_the_table():
    src = _source()
    assert "SHORTCUTS.flatMap" in src          # help panel rows
    assert "KEYMAP.get(" in src                # key handler lookup
    # no second, hand-written key dispatch that could drift from the table
    assert not re.search(r"e\.key\s*===\s*[\"'](a|r|j|k|\?|\[|\])[\"']", src, re.I)


def test_view_prefs_storage_is_guarded():
    src = _source()
    assert src.count("localStorage") == 2
    for m in re.finditer(r"localStorage", src):
        before = src[:m.start()]
        assert before.rfind("try {") > before.rfind("} catch"), "unguarded localStorage"


# ---- F07: upload ----------------------------------------------------------

def test_client_extensions_match_the_servers():
    from clipforge.ingest.ingest import SUPPORTED_EXTENSIONS
    m = re.search(r"const ACCEPTED_EXT = \[(.*?)\];", _source(), re.S)
    assert m, "ACCEPTED_EXT not found"
    client = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert client == set(SUPPORTED_EXTENSIONS)


def test_upload_progress_comes_from_xhr_upload_onprogress():
    src = _source()
    assert "xhr.upload.onprogress" in src
    # fetch() cannot report upload progress, so POST /api/jobs must not use it
    assert "/api/jobs\", {method: \"POST\"" not in src
    assert not re.search(r"fetch\(\s*[\"']/api/jobs[\"']", src)


def test_drag_and_drop_is_wired_without_inline_handlers():
    src = _source()
    for ev in ("dragenter", "dragover", "dragleave", "drop"):
        assert f'"{ev}"' in src, ev
    assert 'id="file" multiple' in src


def test_phone_layout_rules_exist():
    src = _source()
    assert "@media (max-width:700px)" in src
    assert "flex-direction:column" in src.split("@media (max-width:700px)")[1]


def test_upload_has_a_timeout_so_the_queue_cannot_hang():
    src = _source()
    assert "xhr.timeout = UPLOAD_TIMEOUT_MS" in src
    assert "xhr.ontimeout" in src
    assert int(re.search(r"UPLOAD_TIMEOUT_MS = (\d+) \* 60 \* 1000", src).group(1)) >= 15
