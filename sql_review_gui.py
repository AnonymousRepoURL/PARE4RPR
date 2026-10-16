"""Interactively review and safely replace SQL strings stored in a JSON file.

The GUI deliberately uses the project's own MATCH_RECOGNIZE parser.  A query is
written back only after that parser accepts it with the schema from the JSON.


# Opens data/crimes_sql.json by default
python sql_review_gui.py

# Specify just the filename; it automatically looks in the data/ directory
python sql_review_gui.py nasdaq_sql.json

# You can also provide the full relative path
python sql_review_gui.py data/nasdaq_sql.json
"""

from __future__ import annotations

import argparse
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse
import webbrowser

from queryanalysis import RPRAbstractSyntaxTree


class ExternalFileChangedError(RuntimeError):
    """Raised instead of overwriting a JSON file changed by another program."""


class ReviewDocument:
    """JSON loading, parser validation, and guarded atomic persistence."""

    def __init__(self, path: Path, payload: dict[str, Any], digest: str) -> None:
        self.path = path
        self.payload = payload
        self._digest = digest
        self._backup_path: Path | None = None

    @classmethod
    def load(cls, path: Path | str) -> "ReviewDocument":
        path = Path(path).expanduser().resolve()
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object")
        if not isinstance(payload.get("table_schemas"), dict):
            raise ValueError("'table_schemas' must be an object")
        sqls = payload.get("sqls")
        if not isinstance(sqls, list) or not sqls:
            raise ValueError("'sqls' must be a non-empty array")
        for index, item in enumerate(sqls):
            if not isinstance(item, dict) or not isinstance(item.get("sql"), str):
                raise ValueError(
                    f"sqls[{index}] must be an object containing a string 'sql' field"
                )
        return cls(path, payload, cls._hash(raw))

    @staticmethod
    def _hash(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @property
    def sqls(self) -> list[dict[str, Any]]:
        return self.payload["sqls"]

    def validate(self, sql: str) -> None:
        if not sql.strip():
            raise ValueError("SQL must not be empty")
        RPRAbstractSyntaxTree().parse_sql(sql, self.payload["table_schemas"])

    def replace(self, index: int, sql: str) -> Path:
        """Validate and atomically persist one replacement; return backup path."""

        self.validate(sql)
        current_raw = self.path.read_bytes()
        if self._hash(current_raw) != self._digest:
            raise ExternalFileChangedError(
                "The file was changed by another program. Restart this tool to load "
                "the latest content instead of overwriting it."
            )

        # Serialize a copy so an I/O failure cannot leave the in-memory model
        # claiming that an unapplied edit was saved.
        updated = dict(self.payload)
        updated_sqls = [dict(item) for item in self.sqls]
        updated_sqls[index]["sql"] = sql
        updated["sqls"] = updated_sqls
        encoded = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )

        if self._backup_path is None:
            candidate = self.path.with_suffix(self.path.suffix + ".bak")
            suffix = 1
            while candidate.exists():
                candidate = self.path.with_suffix(self.path.suffix + f".bak.{suffix}")
                suffix += 1
            shutil.copy2(self.path, candidate)
            self._backup_path = candidate

        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as temporary:
                temp_name = temporary.name
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temp_name, self.path.stat().st_mode)
            os.replace(temp_name, self.path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

        self.payload = updated
        self._digest = self._hash(encoded)
        return self._backup_path


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SQL Query Reviewer</title>
<style>
:root { color-scheme: light; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background:#f5f7fa; color:#1e293b; }
main { height:100vh; max-width:1400px; margin:auto; padding:16px; display:flex; flex-direction:column; gap:10px; }
.bar,.source,.actions { display:flex; align-items:center; gap:8px; }
.bar .path { margin-left:auto; color:#64748b; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
button,input { font:inherit; }
button { border:1px solid #cbd5e1; border-radius:6px; padding:7px 12px; background:white; cursor:pointer; }
button:hover { background:#eef2ff; } button.primary { color:white; background:#2563eb; border-color:#2563eb; }
#jump { width:62px; padding:6px; } #source { flex:1; color:#2563eb; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
#editor { flex:1; width:100%; resize:none; border:1px solid #94a3b8; border-radius:8px; padding:14px;
  font:14px/1.55 Menlo,Monaco,Consolas,monospace; white-space:pre-wrap; overflow-wrap:anywhere; tab-size:2; }
#editor.no-wrap { white-space:pre; overflow-wrap:normal; overflow-x:auto; }
.hint { margin-left:auto; color:#64748b; } #message { padding:10px 12px; border-radius:7px; background:#e8f5eb; color:#245b31; white-space:pre-wrap; }
#message.error { background:#fdecec; color:#8b1a1a; } .dirty { color:#b45309; font-weight:600; }
@media(max-width:750px){ .bar,.actions{flex-wrap:wrap}.path,.hint{width:100%;margin-left:0}.source{font-size:13px} }
</style></head><body><main>
  <div class="bar">
    <button id="prev">◀ Previous</button><button id="next">Next ▶</button>
    <label><span id="jump-label">Go to</span> <input id="jump" type="number" min="1"></label><button id="go">Go</button>
    <span id="counter"></span><span class="path" id="path"></span>
  </div>
  <div class="source"><strong id="source-label">Source URL:</strong><a id="source" target="_blank" rel="noopener noreferrer"></a></div>
  <textarea id="editor" wrap="soft" spellcheck="false"></textarea>
  <div class="actions">
    <button id="validate">Validate only Ctrl+Enter</button>
    <button class="primary" id="replace">Validate &amp; replace Ctrl+S</button>
    <button id="restore">Discard changes</button>
    <button id="wrap">Disable line wrapping</button>
    <span class="hint" id="hint">Only “Validate &amp; replace” writes to the file</span>
  </div>
  <div id="message">Loading…</div>
</main><script>
const base = location.pathname.replace(/\/$/, '');
const editor=document.querySelector('#editor'), msg=document.querySelector('#message');
let index=0, count=0, saved='', dirty=false, reviewed=new Set();
function message(text,error=false){msg.textContent=text;msg.classList.toggle('error',error)}
function mayLeave(){return !dirty || confirm('These edits have not been written to the JSON file. Discard them?')}
async function request(path, options={}){
  const response=await fetch(base+path, options); const data=await response.json();
  if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data;
}
async function show(target){
  try { const data=await request('/api/query?index='+target); index=data.index; count=data.count;
    saved=data.sql; editor.value=data.sql; dirty=false; document.querySelector('#jump').value=index+1;
    document.querySelector('#jump').max=count; document.querySelector('#path').textContent=data.path;
    const source=document.querySelector('#source');source.textContent=data.url||'(none)';source.href=data.url||'#';
    counter();message('Ready to edit or validate');editor.focus();
  } catch(error){message(`Failed to load: ${error.message}`,true)}
}
function counter(){document.querySelector('#counter').textContent=`Query ${index+1} of ${count}`+
  (dirty?' · Unsaved changes':reviewed.has(index)?' · Replaced this session':'')}
function move(delta){const target=index+delta;if(target>=0&&target<count&&mayLeave())show(target)}
async function validate(){
  try {await request('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sql:editor.value})});
    message('Parsing succeeded. Click “Validate & replace” to write it back.');return true;
  } catch(error){message(`Parsing failed: ${error.message}`,true);return false}
}
async function replace(){
  if(!await validate())return;
  try {const data=await request('/api/replace',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index,sql:editor.value})});
    saved=editor.value;dirty=false;reviewed.add(index);counter();message(`Safely replaced. Original file backup: ${data.backup}`);
  } catch(error){message(`Not written: ${error.message}`,true)}
}
editor.addEventListener('input',()=>{dirty=editor.value!==saved;counter()});
document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);
document.querySelector('#go').onclick=()=>{const target=Number(document.querySelector('#jump').value)-1;if(target>=0&&target<count){if(mayLeave())show(target)}else message('Query number is out of range.',true)};
document.querySelector('#jump').onkeydown=e=>{if(e.key==='Enter')document.querySelector('#go').click()};
document.querySelector('#validate').onclick=validate;document.querySelector('#replace').onclick=replace;
document.querySelector('#restore').onclick=()=>{if(mayLeave()){editor.value=saved;dirty=false;counter();message('Changes discarded')}};
document.querySelector('#wrap').onclick=event=>{const noWrap=editor.classList.toggle('no-wrap');
  editor.wrap=noWrap?'off':'soft';event.currentTarget.textContent=noWrap?'Enable line wrapping':'Disable line wrapping';editor.focus()};
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();replace()}
  else if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();validate()}
  else if(e.altKey&&e.key==='ArrowLeft'){e.preventDefault();move(-1)}else if(e.altKey&&e.key==='ArrowRight'){e.preventDefault();move(1)}});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});show(0);
</script></body></html>"""


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _route(self) -> str | None:
        path = urlparse(self.path).path
        prefix = f"/{self.server.token}"
        if path != prefix and not path.startswith(prefix + "/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        return path[len(prefix) :] or "/"

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid request length") from exc
        if length > 2_000_000:
            raise ValueError("Request body is too large")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        route = self._route()
        if route is None:
            return
        if route == "/":
            encoded = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)
            return
        if route == "/api/query":
            try:
                values = parse_qs(urlparse(self.path).query)
                index = int(values.get("index", ["0"])[0])
                item = self.server.document.sqls[index]
                if index < 0:
                    raise IndexError
                self._send_json(
                    {
                        "index": index,
                        "count": len(self.server.document.sqls),
                        "sql": item["sql"],
                        "url": item.get("url", ""),
                        "path": str(self.server.document.path),
                    }
                )
            except (ValueError, IndexError):
                self._send_json(
                    {"error": "Query number is out of range", "code": "invalidIndex"},
                    HTTPStatus.BAD_REQUEST,
                )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        route = self._route()
        if route is None:
            return
        try:
            body = self._body()
            sql = body.get("sql")
            if not isinstance(sql, str):
                raise ValueError("Missing string field 'sql'")
            if route == "/api/validate":
                self.server.document.validate(sql)
                self._send_json({"valid": True})
            elif route == "/api/replace":
                index = body.get("index")
                if not isinstance(index, int) or not 0 <= index < len(
                    self.server.document.sqls
                ):
                    raise ValueError("Query number is out of range")
                backup = self.server.document.replace(index, sql)
                self._send_json({"saved": True, "backup": str(backup)})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except ExternalFileChangedError as exc:
            self._send_json(
                {"error": str(exc), "code": "externalChanged"}, HTTPStatus.CONFLICT
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, document: ReviewDocument, port: int = 0) -> None:
        super().__init__(("127.0.0.1", port), ReviewRequestHandler)
        self.document = document
        self.token = secrets.token_urlsafe(24)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/{self.token}"


def default_json_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "crimes_sql.json"


def resolve_json_path(argument: Path | None) -> Path:
    """Resolve a CLI path, also accepting bare filenames from ``data/``."""

    if argument is None:
        return default_json_path()
    candidate = argument.expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return Path(__file__).resolve().parent / "data" / candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and edit SQL queries in a JSON file"
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        help="Path to the JSON file; bare filenames are looked up in data/",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to listen on (automatically selected by default)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically",
    )
    args = parser.parse_args(argv)
    document = ReviewDocument.load(resolve_json_path(args.json_path))
    server = ReviewServer(document, args.port)
    print(f"SQL review webpage: {server.url}")
    print("Press `Ctrl+C` to stop....")
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped....")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
