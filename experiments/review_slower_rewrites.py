"""Compare original queries with their LLM rewrites in a browser UI.

The two required arguments are experiment directory names under ``experiments``
or explicit directory paths.  Each directory must contain ``query_times.json``.
The browser UI is read-only and shows the original and rewritten SQL side by
side. By default, it includes only queries whose rewritten average latency is
strictly greater. Use ``--query-scope all`` to review every query regardless of
latency.

Examples:

python experiments/review_slower_rewrites.py \
    oracle_crimes_gpt5.6_rewriting oracle_crimes_original

python experiments/review_slower_rewrites.py \
    oracle_crimes_our_rewriting oracle_crimes_original \
    --query-scope all

python experiments/review_slower_rewrites.py \
    oracle_crimes_deepseek_rewriting oracle_crimes_original \
    --rewritten-sql data/crimes_sql_oracle_deepseek_v1_thinking.txt

python experiments/review_slower_rewrites.py \
    oracle_crimes_gpt5.6_rewriting oracle_crimes_original \
    --rewritten-sql data/crimes_sql_oracle_gpt-5.6-sol_v1_thinking.txt

"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path, PureWindowsPath
import secrets
import sys
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse
import webbrowser

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ORIGINAL_SQL = DATA_DIR / "crimes_sql.json"


@dataclass(frozen=True)
class MetricsDocument:
    directory: Path
    payload: dict[str, Any]
    queries: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class QueryComparison:
    query_id: int
    url: str
    original_sql: str
    rewritten_sql: str
    original_seconds: float
    rewritten_seconds: float
    original_status: str
    rewritten_status: str

    @property
    def delta_seconds(self) -> float:
        return self.rewritten_seconds - self.original_seconds

    @property
    def percent_slower(self) -> float | None:
        if self.original_seconds == 0:
            return None
        return self.delta_seconds / self.original_seconds * 100


def resolve_experiment_directory(value: str | Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        return supplied.resolve()
    under_experiments = EXPERIMENTS_DIR / supplied
    if under_experiments.is_dir():
        return under_experiments.resolve()
    return supplied.resolve()


def load_metrics(directory: Path) -> MetricsDocument:
    if not directory.is_dir():
        raise ValueError(f"Experiment directory does not exist: {directory}")
    path = directory / "query_times.json"
    if not path.is_file():
        raise ValueError(f"Performance data not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError(f"Missing queries array in {path}")

    queries: dict[int, dict[str, Any]] = {}
    for item in payload["queries"]:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid query record in {path}")
        query_id = item.get("id")
        if not isinstance(query_id, int) or isinstance(query_id, bool) or query_id < 1:
            raise ValueError(f"Invalid query ID in {path}: {query_id!r}")
        if query_id in queries:
            raise ValueError(f"Duplicate query ID {query_id} in {path}")
        queries[query_id] = item
    if not queries:
        raise ValueError(f"No query records in {path}")
    return MetricsDocument(directory, payload, queries)


def average_latency(query: dict[str, Any], context: str) -> float:
    value = query.get("average_query_latency_seconds")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid average_query_latency_seconds for {context}")
    return float(value)


def status_summary(query: dict[str, Any]) -> str:
    runs = query.get("runs")
    if not isinstance(runs, list) or not runs:
        complete = query.get("output_complete")
        if isinstance(complete, bool):
            return "complete" if complete else "incomplete"
        return "unknown"
    counts = Counter(
        str(run.get("status", "unknown")) for run in runs if isinstance(run, dict)
    )
    summary = ", ".join(f"{name}×{count}" for name, count in sorted(counts.items()))
    if query.get("output_complete") is False:
        summary += " · incomplete"
    return summary or "unknown"


def split_sql_statements(text: str) -> list[str]:
    """Split SQL terminators while respecting quotes and SQL comments."""

    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    in_line_comment = False
    in_block_comment = False
    index = 0

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_line_comment:
            current.append(char)
            if char in "\r\n":
                in_line_comment = False
            index += 1
            continue
        if in_block_comment:
            current.append(char)
            if char == "*" and next_char == "/":
                current.append(next_char)
                index += 2
                in_block_comment = False
            else:
                index += 1
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                if next_char == quote:
                    current.append(next_char)
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "-" and next_char == "-":
            current.extend((char, next_char))
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            current.extend((char, next_char))
            in_block_comment = True
            index += 2
            continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue
        current.append(char)
        index += 1

    if quote is not None:
        raise ValueError("SQL file ends inside an unterminated quote")
    if in_block_comment:
        raise ValueError("SQL file ends inside an unterminated block comment")
    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def load_sql_file(path: Path) -> tuple[list[str], list[str]]:
    """Return SQL strings and optional URLs from JSON or semicolon text."""

    if not path.is_file():
        raise ValueError(f"SQL file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read SQL file {path}: {exc}") from exc

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse SQL JSON {path}: {exc}") from exc
        items = payload.get("sqls") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError(f"Missing sqls array in {path}")
        sqls: list[str] = []
        urls: list[str] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("sql"), str):
                raise ValueError(f"Invalid SQL entry {index} in {path}")
            sqls.append(item["sql"].strip().rstrip(";").rstrip())
            url = item.get("url")
            urls.append(url if isinstance(url, str) else "")
        return sqls, urls

    return split_sql_statements(text), []


def _metadata_filename(value: str) -> str:
    """Extract a basename from either a POSIX or Windows metrics path."""

    return PureWindowsPath(value).name or Path(value).name


def resolve_metadata_path(raw_value: str, experiment_directory: Path) -> Path | None:
    supplied = Path(raw_value).expanduser()
    filename = _metadata_filename(raw_value)
    candidates = [
        supplied,
        experiment_directory / filename,
        DATA_DIR / filename,
        PROJECT_ROOT / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def infer_rewritten_sql_path(metrics: MetricsDocument) -> tuple[Path, str | None]:
    raw_value = metrics.payload.get("query_file")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(
            f"{metrics.directory / 'query_times.json'} has no query_file; "
            "use --rewritten-sql to specify the rewritten SQL file"
        )
    resolved = resolve_metadata_path(raw_value, metrics.directory)
    if resolved is not None:
        return resolved, None

    filename = _metadata_filename(raw_value)
    stem = Path(filename).stem
    candidates = sorted(DATA_DIR.glob(f"{stem}_*.txt"))
    thinking = [
        path
        for path in candidates
        if path.name.endswith("_thinking.txt")
        and not path.name.endswith("_no_thinking.txt")
    ]
    if len(thinking) == 1:
        warning = (
            f"query_file points to missing file {filename}; using historical "
            f"thinking file {thinking[0]}. Override it with --rewritten-sql if needed."
        )
        return thinking[0].resolve(), warning
    if len(candidates) == 1:
        warning = (
            f"query_file points to missing file {filename}; using {candidates[0]}."
        )
        return candidates[0].resolve(), warning
    candidate_text = ", ".join(str(path) for path in candidates) or "none"
    raise ValueError(
        f"Could not locate {filename} from query_file. Candidates: "
        f"{candidate_text}. Specify the file with --rewritten-sql."
    )


def infer_original_sql_path(metrics: MetricsDocument) -> Path:
    raw_value = metrics.payload.get("query_file")
    if isinstance(raw_value, str) and raw_value.strip():
        resolved = resolve_metadata_path(raw_value, metrics.directory)
        if resolved is not None:
            return resolved
    if DEFAULT_ORIGINAL_SQL.is_file():
        return DEFAULT_ORIGINAL_SQL.resolve()
    raise ValueError("Could not locate the original SQL file; use --original-sql")


def checked_sql_by_id(
    path: Path, query_ids: set[int], label: str
) -> tuple[dict[int, str], dict[int, str]]:
    sqls, urls = load_sql_file(path)
    largest_id = max(query_ids)
    if len(sqls) < largest_id:
        raise ValueError(
            f"The {label} SQL file {path} has only {len(sqls)} statements, "
            f"but the performance data contains query ID {largest_id}"
        )
    sql_by_id = {query_id: sqls[query_id - 1] for query_id in query_ids}
    url_by_id = {
        query_id: urls[query_id - 1]
        for query_id in query_ids
        if len(urls) >= query_id and urls[query_id - 1]
    }
    return sql_by_id, url_by_id


def build_query_comparisons(
    rewritten_metrics: MetricsDocument,
    original_metrics: MetricsDocument,
    rewritten_sql_path: Path,
    original_sql_path: Path,
    query_scope: str = "slower",
) -> list[QueryComparison]:
    if query_scope not in {"slower", "all"}:
        raise ValueError(f"Unsupported query scope: {query_scope}")

    rewritten_ids = set(rewritten_metrics.queries)
    missing_original = sorted(rewritten_ids - set(original_metrics.queries))
    if missing_original:
        raise ValueError(
            "Original performance data is missing query IDs: "
            + ", ".join(map(str, missing_original))
        )

    rewritten_sqls, rewritten_urls = checked_sql_by_id(
        rewritten_sql_path, rewritten_ids, "rewritten"
    )
    original_sqls, original_urls = checked_sql_by_id(
        original_sql_path, rewritten_ids, "original"
    )

    records: list[QueryComparison] = []
    for query_id in sorted(rewritten_ids):
        rewritten_query = rewritten_metrics.queries[query_id]
        original_query = original_metrics.queries[query_id]
        rewritten_seconds = average_latency(
            rewritten_query, f"rewritten query {query_id}"
        )
        original_seconds = average_latency(original_query, f"original query {query_id}")
        if query_scope == "slower" and rewritten_seconds <= original_seconds:
            continue
        url = rewritten_query.get("url") or original_query.get("url")
        if not isinstance(url, str):
            url = rewritten_urls.get(query_id) or original_urls.get(query_id, "")
        records.append(
            QueryComparison(
                query_id=query_id,
                url=url,
                original_sql=original_sqls[query_id],
                rewritten_sql=rewritten_sqls[query_id],
                original_seconds=original_seconds,
                rewritten_seconds=rewritten_seconds,
                original_status=status_summary(original_query),
                rewritten_status=status_summary(rewritten_query),
            )
        )
    return records


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Query Rewrite Review</title>
<style>
:root { color-scheme:light; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
* { box-sizing:border-box; }
body { margin:0; background:#f4f7fb; color:#172033; }
main { height:100vh; max-width:1800px; margin:auto; padding:14px; display:flex; flex-direction:column; gap:10px; }
.toolbar,.source,.summary { display:flex; align-items:center; gap:8px; }
.toolbar { flex-wrap:wrap; }
button,input { font:inherit; }
button { border:1px solid #cbd5e1; border-radius:7px; padding:7px 12px; background:white; cursor:pointer; }
button:hover:not(:disabled) { background:#e8efff; }
button:disabled { opacity:.45; cursor:default; }
#jump { width:70px; padding:6px; }
#counter { font-weight:650; }
.paths { margin-left:auto; max-width:48%; color:#64748b; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.summary { padding:10px 12px; border:1px solid #cbd5e1; border-radius:8px; background:white; flex-wrap:wrap; }
.metric { padding:5px 9px; border-radius:6px; background:#eef2f7; }
.slower { color:#a33a16; background:#fff0e8; font-weight:700; }
.faster { color:#166534; background:#dcfce7; font-weight:700; }
.status { color:#475569; }
.source { min-height:24px; }
.source a { color:#2563eb; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.panels { min-height:0; flex:1; display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.panel { min-width:0; display:flex; flex-direction:column; gap:6px; }
.panel h2 { margin:0; font-size:16px; }
textarea { min-height:0; flex:1; width:100%; resize:none; border:1px solid #94a3b8; border-radius:8px; padding:13px;
  background:white; color:#111827; font:13.5px/1.55 Menlo,Monaco,Consolas,monospace; white-space:pre-wrap; overflow-wrap:anywhere; tab-size:2; }
textarea.no-wrap { white-space:pre; overflow-wrap:normal; overflow-x:auto; }
.footer { color:#64748b; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
#empty { margin:auto; padding:28px; border-radius:10px; background:#e8f5eb; color:#245b31; font-size:18px; }
@media(max-width:850px){ .panels{grid-template-columns:1fr;overflow:auto}.panel{min-height:45vh}.paths{width:100%;max-width:none;margin-left:0} }
</style></head><body><main>
  <div class="toolbar">
    <button id="prev">◀ Previous</button><button id="next">Next ▶</button>
    <label><span id="jump-label">Go to result</span> <input id="jump" type="number" min="1"></label>
    <button id="go">Go</button><span id="counter"></span>
    <button id="wrap">Disable line wrapping</button>
    <span class="paths" id="paths"></span>
  </div>
  <div id="content">
    <div class="summary">
      <span class="metric" id="original-time"></span><span class="metric" id="rewritten-time"></span>
      <span class="metric" id="difference"></span>
      <span class="status" id="statuses"></span>
    </div>
    <div class="source">
      <strong id="query-id-label">Query ID:</strong><span id="query-id"></span>
      <strong id="source-label">Source URL:</strong><a id="source" target="_blank" rel="noopener noreferrer"></a>
    </div>
  </div>
  <div class="panels" id="panels">
    <section class="panel"><h2 id="original-heading">Original query</h2><textarea id="original" readonly spellcheck="false"></textarea></section>
    <section class="panel"><h2 id="rewritten-heading">LLM-rewritten query</h2><textarea id="rewritten" readonly spellcheck="false"></textarea></section>
  </div>
  <div class="footer" id="footer"></div>
  <div id="empty" hidden>No matching queries found.</div>
</main><script>
const base=location.pathname.replace(/\/$/,'');
const original=document.querySelector('#original'), rewritten=document.querySelector('#rewritten');
let position=0,count=0,current=null,queryScope='slower';
function fmt(value){return Number(value).toFixed(6)+' seconds'}
async function request(path){const response=await fetch(base+path);const data=await response.json();
  if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data}
function render(){if(!current)return;const data=current;
  const resultLabel=queryScope==='slower'?'Slower result':'Result';
  document.querySelector('#counter').textContent=`${resultLabel} ${position+1} of ${count}`;
  document.querySelector('#query-id').textContent=data.query_id;
  document.querySelector('#original-time').textContent='Original: '+fmt(data.original_seconds);
  document.querySelector('#rewritten-time').textContent='Rewritten: '+fmt(data.rewritten_seconds);
  const difference=document.querySelector('#difference');
  const percent=data.percent_slower===null?'original latency is 0':Math.abs(Number(data.percent_slower)).toFixed(2)+'%';
  const direction=data.delta_seconds>0?'Increase':data.delta_seconds<0?'Decrease':'No change';
  difference.textContent=direction==='No change'?direction:`${direction} ${fmt(Math.abs(data.delta_seconds))} · ${percent}`;
  difference.classList.toggle('slower',data.delta_seconds>0);difference.classList.toggle('faster',data.delta_seconds<0);
  document.querySelector('#statuses').textContent=`Run status | Original: ${data.original_status}; Rewritten: ${data.rewritten_status}`;
  const source=document.querySelector('#source');source.textContent=data.url||'(none)';source.href=data.url||'#';
  original.value=data.original_sql;rewritten.value=data.rewritten_sql;
  document.querySelector('#paths').textContent=`${data.rewritten_name} vs ${data.original_name}`;
  document.querySelector('#footer').textContent=`Rewritten SQL: ${data.rewritten_sql_path} | Original SQL: ${data.original_sql_path}`;
  document.querySelector('#prev').disabled=position===0;document.querySelector('#next').disabled=position===count-1;
}
async function show(target){
  try {const data=await request('/api/query?position='+target);current=data;position=data.position;count=data.count;
    document.querySelector('#jump').value=position+1;document.querySelector('#jump').max=count;
    render();
  } catch(error){alert('Failed to load: '+error.message)}
}
function move(delta){const target=position+delta;if(target>=0&&target<count)show(target)}
document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);
document.querySelector('#go').onclick=()=>{const target=Number(document.querySelector('#jump').value)-1;
  if(Number.isInteger(target)&&target>=0&&target<count)show(target);else alert('Result number is out of range.')};
document.querySelector('#jump').onkeydown=e=>{if(e.key==='Enter')document.querySelector('#go').click()};
document.querySelector('#wrap').onclick=event=>{const noWrap=original.classList.toggle('no-wrap');
  rewritten.classList.toggle('no-wrap',noWrap);original.wrap=noWrap?'off':'soft';rewritten.wrap=noWrap?'off':'soft';
  event.currentTarget.textContent=noWrap?'Enable line wrapping':'Disable line wrapping'};
document.addEventListener('keydown',e=>{if(e.altKey&&e.key==='ArrowLeft'){e.preventDefault();move(-1)}
  else if(e.altKey&&e.key==='ArrowRight'){e.preventDefault();move(1)}});
async function start(){const info=await request('/api/info');count=info.count;queryScope=info.query_scope;
  document.querySelector('#jump-label').textContent=queryScope==='slower'?'Go to slower result':'Go to result';
  document.title=queryScope==='slower'?'Slower Rewrite Review':'Query Rewrite Review';
  if(!count){document.querySelector('#content').hidden=true;document.querySelector('#panels').hidden=true;
    document.querySelector('#empty').textContent=queryScope==='slower'?'No queries have a higher average latency after rewriting.':'No queries found.';
    document.querySelector('#empty').hidden=false;document.querySelectorAll('#prev,#next,#go,#jump,#wrap').forEach(x=>x.disabled=true);return}show(0)}
start().catch(error=>alert('Failed to initialize: '+error.message));
</script></body></html>"""


class ComparisonRequestHandler(BaseHTTPRequestHandler):
    server: "ComparisonServer"

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
        if route == "/api/info":
            self._send_json(
                {
                    "count": len(self.server.records),
                    "query_scope": self.server.query_scope,
                }
            )
            return
        if route == "/api/query":
            try:
                values = parse_qs(urlparse(self.path).query)
                position = int(values.get("position", ["0"])[0])
                if position < 0:
                    raise IndexError
                record = self.server.records[position]
                self._send_json(
                    {
                        "position": position,
                        "count": len(self.server.records),
                        "query_id": record.query_id,
                        "url": record.url,
                        "original_sql": record.original_sql,
                        "rewritten_sql": record.rewritten_sql,
                        "original_seconds": record.original_seconds,
                        "rewritten_seconds": record.rewritten_seconds,
                        "delta_seconds": record.delta_seconds,
                        "percent_slower": record.percent_slower,
                        "original_status": record.original_status,
                        "rewritten_status": record.rewritten_status,
                        "original_name": self.server.original_name,
                        "rewritten_name": self.server.rewritten_name,
                        "original_sql_path": str(self.server.original_sql_path),
                        "rewritten_sql_path": str(self.server.rewritten_sql_path),
                    }
                )
            except (ValueError, IndexError):
                self._send_json(
                    {"error": "Result number is out of range"},
                    HTTPStatus.BAD_REQUEST,
                )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class ComparisonServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        records: list[QueryComparison],
        rewritten_metrics: MetricsDocument,
        original_metrics: MetricsDocument,
        rewritten_sql_path: Path,
        original_sql_path: Path,
        query_scope: str,
        port: int = 0,
    ) -> None:
        super().__init__(("127.0.0.1", port), ComparisonRequestHandler)
        self.records = records
        self.rewritten_name = rewritten_metrics.directory.name
        self.original_name = original_metrics.directory.name
        self.rewritten_sql_path = rewritten_sql_path
        self.original_sql_path = original_sql_path
        self.query_scope = query_scope
        self.token = secrets.token_urlsafe(24)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/{self.token}"


def resolve_sql_override(value: Path) -> Path:
    supplied = value.expanduser()
    if supplied.is_absolute() or supplied.is_file():
        return supplied.resolve()
    under_root = PROJECT_ROOT / supplied
    return under_root.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rewritten_directory", help="LLM rewrite experiment directory")
    parser.add_argument(
        "original_directory", help="Original query experiment directory"
    )
    parser.add_argument(
        "--query-scope",
        choices=("slower", "all"),
        default="slower",
        help="queries to show: only slower rewrites (default) or all queries",
    )
    parser.add_argument(
        "--rewritten-sql",
        type=Path,
        help="rewritten SQL file; inferred from query_times.json by default",
    )
    parser.add_argument(
        "--original-sql",
        type=Path,
        help="original SQL JSON/TXT; defaults to query_file or data/crimes_sql.json",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="listening port (chosen automatically by default)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser automatically",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rewritten_metrics = load_metrics(
            resolve_experiment_directory(args.rewritten_directory)
        )
        original_metrics = load_metrics(
            resolve_experiment_directory(args.original_directory)
        )
        warning: str | None = None
        if args.rewritten_sql is not None:
            rewritten_sql_path = resolve_sql_override(args.rewritten_sql)
        else:
            rewritten_sql_path, warning = infer_rewritten_sql_path(rewritten_metrics)
        original_sql_path = (
            resolve_sql_override(args.original_sql)
            if args.original_sql is not None
            else infer_original_sql_path(original_metrics)
        )
        records = build_query_comparisons(
            rewritten_metrics,
            original_metrics,
            rewritten_sql_path,
            original_sql_path,
            args.query_scope,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    ids = ", ".join(str(record.query_id) for record in records) or "(none)"
    print(f"Rewritten SQL: {rewritten_sql_path}")
    print(f"Original SQL: {original_sql_path}")
    if args.query_scope == "slower":
        print(f"Found {len(records)} slower rewritten queries: {ids}")
    else:
        print(f"Found {len(records)} queries: {ids}")

    server = ComparisonServer(
        records,
        rewritten_metrics,
        original_metrics,
        rewritten_sql_path,
        original_sql_path,
        args.query_scope,
        args.port,
    )
    print(f"SQL comparison page: {server.url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
