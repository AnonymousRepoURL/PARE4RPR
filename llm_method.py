"""Rewrite MATCH_RECOGNIZE queries with an OpenAI-compatible LLM API.

The module supports OpenAI and DeepSeek. Rewritten SQL is kept in a plain
text file so that it remains directly executable, while request latency,
token usage, pricing parameters, and estimated cost are written to a JSON
sidecar file.


------------------- How to run? -------------------

python3 llm_method.py data/crimes_sql.json \
  --provider deepseek \
  --model deepseek-flash \
  --deepseek-thinking enabled \
  --max-output-tokens 65536 \
  --prompt-version v1

+++++++++++++++++++++++++++++++++++++++++++++++++++
  
python3 llm_method.py data/crimes_sql.json \
  --provider openai \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --max-output-tokens 65536 \
  --prompt-version v1

+++++++++++++++++++++++++++++++++++++++++++++++++++  

python3 llm_method.py data/crimes_sql.json \
  --provider openai \
  --model gpt-6-astra \
  --reasoning-effort high \
  --max-output-tokens 65536 \
  --input-price-per-million 10 \
  --cached-input-price-per-million 1 \
  --output-price-per-million 50 \
  --prompt-version v1
  
Prompt v2 is used by default. Add ``--prompt-version v1`` to reproduce the
original prompt.

---------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

DATA_DIR = Path(__file__).resolve().parent / "data"
MILLION = 1_000_000

DEFAULT_PROMPT_VERSION = "v2"
PROMPT_VERSIONS = ("v1", "v2")


BASE_SYSTEM_PROMPT_V1 = """You are an elite SQL Database Architect and an expert in Row Pattern Recognition (RPR) statement optimization. Your objective is to analyze the provided SQL query containing a MATCH_RECOGNIZE clause and rewrite it into a strictly equivalent, but highly optimized and more efficient query for the explicitly specified target database.

# Optimization Guidelines
1. Intent Analysis: Deeply understand the sequential pattern defined in the PATTERN clause, the conditions in the DEFINE clause, and the expressions in the MEASURES clause.
2. Dialect Fidelity: Generate SQL that is executable by the specified target database and version. Do not transfer syntax, functions, casts, identifier rules, or MATCH_RECOGNIZE features from another database dialect. If support for a proposed construct is uncertain, keep the original construct unchanged.
3. Schema Awareness: Use the supplied table schemas, including column types, nullability, primary keys, and join keys, when determining whether a rewrite is valid and useful. Never invent tables, columns, keys, constraints, indexes, or physical data properties that are not supplied.
4. Performance Tuning: Identify opportunities to improve execution speed and reduce resource consumption. This includes:
   - Pushing down filters or predicates before MATCH_RECOGNIZE evaluation when doing so preserves semantics.
   - Simplifying complex regular expressions in the PATTERN clause.
   - Pruning unnecessary columns, states, or redundant computations.
   - Replacing MATCH_RECOGNIZE with standard window functions such as LEAD, LAG, or SUM(...) OVER only if it yields a faster execution plan while maintaining exact logical equivalence.
5. Strict Equivalence: The optimized query must guarantee identical output results compared with the original query. Pay special attention to SQL three-valued logic, NULL handling, duplicates, match selection, greedy and reluctant quantifiers, AFTER MATCH SKIP behavior, row ordering, and ties in ORDER BY.
6. Conservative Fallback: If equivalence, dialect support, or a performance benefit cannot be established from the query and supplied metadata, return the exact original query.

# Output Rules
- Output only one raw, executable SQL query.
- Do not include explanations, greetings, comments, labels, or markdown fences.
- Preserve the specified target database dialect and version.
- If no semantics-preserving performance improvement can be made, output the exact original query.
"""


BASE_SYSTEM_PROMPT_V2 = """You optimize MATCH_RECOGNIZE queries for the specified database and version.

Return exactly one executable SELECT or WITH query. Do not output markdown, comments, labels, or explanations.

Apply these requirements in order:
1. Dialect validity: Use only syntax and functions supported by the specified database version.
2. Exact equivalence: Preserve results under SQL three-valued logic, including NULLs, duplicates, match selection, greedy or reluctant quantifiers, AFTER MATCH SKIP, rows-per-match behavior, ordering, and ORDER BY ties.
3. Optimization: Rewrite only when a performance benefit is justified by the SQL and supplied schema. Permitted approaches include safe predicate pushdown, removal of redundant computations, pattern simplification, and replacement with window functions when exact equivalence is established.

Use only supplied schema metadata. Do not invent tables, columns, constraints, indexes, ordering guarantees, or data properties.

If dialect validity, equivalence, or performance benefit is uncertain, return the original SQL verbatim.
"""


# Compatibility alias for callers that imported the original prompt directly.
BASE_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT_V1


@dataclass(frozen=True)
class DialectProfile:
    name: str
    display_name: str
    instructions: str
    documentation: str


DIALECTS = {
    "oracle": DialectProfile(
        name="oracle",
        display_name="Oracle Database",
        instructions="""Use Oracle Database SQL and Oracle MATCH_RECOGNIZE semantics. Keep Oracle-specific row-pattern syntax and functions valid for the requested Oracle version. Any replacement using analytic functions, subqueries, or joins must use Oracle syntax and preserve Oracle NULL, ordering, and type-conversion behavior. Do not emit Trino-only functions or Apache IoTDB RPR_* navigation functions.""",
        documentation="https://docs.oracle.com/en/database/oracle/oracle-database/26/dwhsg/sql-pattern-matching-data-warehouses.html",
    ),
    "trino": DialectProfile(
        name="trino",
        display_name="Trino",
        instructions="""Use Trino SQL and Trino MATCH_RECOGNIZE semantics. MATCH_RECOGNIZE is a relation in the FROM clause. Keep pattern navigation, aggregate, RUNNING/FINAL, rows-per-match, and after-match-skip syntax within Trino's documented feature set. Any replacement using window functions, arrays, subqueries, or joins must use Trino functions and casts. Do not emit Oracle-only functions or Apache IoTDB RPR_* navigation functions.""",
        documentation="https://trino.io/docs/current/sql/match-recognize.html",
    ),
    "iotdb": DialectProfile(
        name="iotdb",
        display_name="Apache IoTDB Table Model",
        instructions="""Use Apache IoTDB Table Model SQL; MATCH_RECOGNIZE is available in the Table Model starting with IoTDB V2.0.5. Do not generate Tree Model path syntax. Use IoTDB's documented row-pattern functions and types, including RPR_FIRST and RPR_LAST where IoTDB requires those navigation names, rather than assuming Oracle or Trino function names. Any replacement with ordinary window functions or subqueries must use constructs supported by the requested IoTDB version. Preserve the time column, TAG/ATTRIBUTE/FIELD semantics, timestamp arithmetic, and result ordering.""",
        documentation="https://iotdb.apache.org/UserGuide/latest-Table/User-Manual/Pattern-Query_apache.html",
    ),
}


DIALECT_INSTRUCTIONS_V2 = {
    "oracle": """Use Oracle Database SQL and Oracle MATCH_RECOGNIZE semantics for the requested version. Preserve Oracle NULL, ordering, and type-conversion behavior.""",
    "trino": """Use Trino SQL and MATCH_RECOGNIZE relation semantics. Keep navigation, aggregates, RUNNING/FINAL, rows-per-match, and after-match-skip syntax within the requested version's feature set.""",
    "iotdb": """Use Apache IoTDB Table Model SQL, not Tree Model syntax. Use IoTDB row-pattern functions such as RPR_FIRST and RPR_LAST where required, and preserve time, TAG, ATTRIBUTE, FIELD, timestamp-arithmetic, and ordering semantics.""",
}


def require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' must be a non-empty string")
    return value.strip()


def normalize_database(database: str) -> str:
    normalized = database.strip().lower()
    if normalized not in DIALECTS:
        supported = ", ".join(sorted(DIALECTS))
        raise ValueError(
            f"Unsupported database '{database}'; choose one of: {supported}"
        )
    return normalized


def normalize_prompt_version(prompt_version: str) -> str:
    normalized = require_non_empty_string(prompt_version, "prompt_version").lower()
    if normalized not in PROMPT_VERSIONS:
        supported = ", ".join(PROMPT_VERSIONS)
        raise ValueError(
            f"Unsupported prompt version '{prompt_version}'; choose one of: {supported}"
        )
    return normalized


def base_system_prompt(prompt_version: str) -> str:
    prompt_version = normalize_prompt_version(prompt_version)
    if prompt_version == "v1":
        return BASE_SYSTEM_PROMPT_V1
    return BASE_SYSTEM_PROMPT_V2


def build_system_prompt(
    database: str,
    database_version: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    database = normalize_database(database)
    profile = DIALECTS[database]
    version = require_non_empty_string(database_version, "database_version")
    prompt_version = normalize_prompt_version(prompt_version)
    if prompt_version == "v1":
        return f"""{BASE_SYSTEM_PROMPT_V1}

# Target Database
- Database: {profile.display_name}
- Version: {version}

# Database-Specific Requirements
{profile.instructions}
"""
    return f"""{BASE_SYSTEM_PROMPT_V2}

Target database: {profile.display_name}
Version: {version}
Database-specific requirements: {DIALECT_INSTRUCTIONS_V2[database]}
"""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key_env: str
    base_url: str | None
    default_model: str


PROVIDERS = {
    "openai": ProviderConfig(
        name="openai",
        api_key_env="OPENAI_API_KEY",  # OPENAI_API_KEY
        base_url=None,
        default_model="gpt-5.6-sol",
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-flash",
    ),
}


@dataclass(frozen=True)
class Pricing:
    """USD prices per one million tokens."""

    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    profile: str
    source: str


# Defaults are copied to every metrics file. Use the CLI price overrides when
# provider prices change. DeepSeek pricing varies by time, hence two profiles.
OPENAI_PRICING = {
    "gpt-5.6-sol": Pricing(
        input_per_million=4.0,
        cached_input_per_million=0.4,
        output_per_million=20.0,
        profile="standard",
        source="https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    ),
}
# DeepSeek V4.1 Flash is served through the ``deepseek-flash`` model ID.
DEEPSEEK_PRICING = {
    "deepseek-flash": {
        "off_peak": Pricing(
            0.15,
            0.003,
            0.60,
            "off_peak",
            "https://api-docs.deepseek.com/quick_start/pricing/",
        ),
        "peak": Pricing(
            0.30,
            0.006,
            1.20,
            "peak",
            "https://api-docs.deepseek.com/quick_start/pricing/",
        ),
    },
}


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class LLMRewriteResult:
    sql: str
    prompt_version: str
    provider: str
    requested_model: str
    response_model: str | None
    target_database: str
    database_version: str
    latency_seconds: float
    usage: TokenUsage
    estimated_cost_usd: float
    finish_reason: str | None
    response_id: str | None
    pricing: Pricing


class ModelResponseTruncatedError(ValueError):
    """Raised when the model stops because its output-token limit was reached."""


class ChatCompletions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class Chat(Protocol):
    completions: ChatCompletions


class LLMClient(Protocol):
    chat: Chat


def _value(obj: Any, name: str, default: Any = None) -> Any:
    """Read an SDK object's attribute or a test double's mapping key."""

    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def extract_usage(response: Any) -> TokenUsage:
    usage = _value(response, "usage", {})
    prompt_tokens = _non_negative_int(
        _value(usage, "prompt_tokens", _value(usage, "input_tokens", 0))
    )
    completion_tokens = _non_negative_int(
        _value(usage, "completion_tokens", _value(usage, "output_tokens", 0))
    )
    prompt_details = _value(
        usage,
        "prompt_tokens_details",
        _value(usage, "input_tokens_details", {}),
    )
    cached_tokens = _non_negative_int(
        _value(
            prompt_details,
            "cached_tokens",
            _value(usage, "prompt_cache_hit_tokens", 0),
        )
    )
    completion_details = _value(
        usage,
        "completion_tokens_details",
        _value(usage, "output_tokens_details", {}),
    )
    reasoning_tokens = _non_negative_int(
        _value(completion_details, "reasoning_tokens", 0)
    )
    total_tokens = _non_negative_int(
        _value(usage, "total_tokens", prompt_tokens + completion_tokens)
    )
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_prompt_tokens=min(cached_tokens, prompt_tokens),
        reasoning_tokens=min(reasoning_tokens, completion_tokens),
    )


def estimate_cost(usage: TokenUsage, pricing: Pricing) -> float:
    uncached_tokens = max(0, usage.prompt_tokens - usage.cached_prompt_tokens)
    cost = (
        uncached_tokens * pricing.input_per_million
        + usage.cached_prompt_tokens * pricing.cached_input_per_million
        + usage.completion_tokens * pricing.output_per_million
    ) / MILLION
    return round(cost, 12)


def is_deepseek_peak(when: datetime | None = None) -> bool:
    """Return whether ``when`` falls in DeepSeek's documented UTC peak band."""

    current = when or datetime.now(timezone.utc)
    current = current.astimezone(timezone.utc)
    if current.weekday() >= 5:
        return False
    return 1 <= current.hour < 4 or 6 <= current.hour < 10


def default_pricing(provider: str, model: str, when: datetime | None = None) -> Pricing:
    if provider == "openai" and model in OPENAI_PRICING:
        return OPENAI_PRICING[model]
    if provider == "deepseek" and model in DEEPSEEK_PRICING:
        profile = "peak" if is_deepseek_peak(when) else "off_peak"
        return DEEPSEEK_PRICING[model][profile]
    raise ValueError(
        f"No built-in pricing for {provider}/{model}; provide all three "
        "--*-price-per-million options"
    )


def resolve_pricing(
    provider: str,
    model: str,
    input_price: float | None,
    cached_input_price: float | None,
    output_price: float | None,
    when: datetime | None = None,
) -> Pricing:
    supplied = (input_price, cached_input_price, output_price)
    if all(value is None for value in supplied):
        return default_pricing(provider, model, when)
    if any(value is None for value in supplied):
        raise ValueError(
            "Custom pricing requires input, cached-input, and output prices"
        )
    if any(value < 0 for value in supplied if value is not None):
        raise ValueError("Pricing values must be non-negative")
    return Pricing(
        input_per_million=float(input_price),
        cached_input_per_million=float(cached_input_price),
        output_per_million=float(output_price),
        profile="custom",
        source="command_line",
    )


def build_user_prompt(
    sql: str,
    table_schemas: Mapping[str, Any],
    database: str,
    database_version: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    database = normalize_database(database)
    profile = DIALECTS[database]
    database_version = require_non_empty_string(database_version, "database_version")
    prompt_version = normalize_prompt_version(prompt_version)
    schema_json = json.dumps(
        table_schemas,
        ensure_ascii=False,
        sort_keys=True,
        **({"indent": 2} if prompt_version == "v1" else {"separators": (",", ":")}),
    )
    if prompt_version == "v1":
        return f"""Rewrite the SQL query below using the supplied schema metadata.

# Target Database
- Database: {profile.display_name}
- Version: {database_version}

# Table Schemas (JSON)
{schema_json}

# Original SQL
{sql}
"""
    return f"""SCHEMA:
{schema_json}

SQL:
{sql}
"""


_FENCED_SQL = re.compile(
    r"\A\s*```(?:sql)?\s*\n?(.*?)\n?```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_sql_response(content: str) -> str:
    """Normalize a model response while rejecting obvious non-SQL output."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an empty response")
    text = content.strip()
    fenced = _FENCED_SQL.fullmatch(text)
    if fenced:
        text = fenced.group(1).strip()
    if "\x00" in text:
        raise ValueError("The model response contains a NUL character")
    if not re.match(r"^(?:SELECT|WITH)\b", text, flags=re.IGNORECASE):
        raise ValueError("The model response is not a SELECT/WITH SQL query")
    return text.rstrip().rstrip(";").rstrip()


def _response_content(response: Any) -> tuple[str, str | None]:
    choices = _value(response, "choices", [])
    if not choices:
        raise ValueError("The model response has no choices")
    choice = choices[0]
    message = _value(choice, "message", {})
    content = _value(message, "content")
    if not isinstance(content, str):
        raise ValueError("The model response contains no text content")
    return content, _value(choice, "finish_reason")


def rewrite_sql_with_llm(
    sql: str,
    table_schemas: Mapping[str, Any],
    *,
    client: LLMClient,
    provider: str,
    model: str,
    pricing: Pricing,
    database: str,
    database_version: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    max_output_tokens: int = 16384,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    deepseek_thinking: str = "disabled",
) -> LLMRewriteResult:
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("sql must be a non-empty string")
    database = normalize_database(database)
    database_version = require_non_empty_string(database_version, "database_version")
    prompt_version = normalize_prompt_version(prompt_version)
    system_prompt = build_system_prompt(database, database_version, prompt_version)

    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_prompt(
                    sql,
                    table_schemas,
                    database,
                    database_version,
                    prompt_version,
                ),
            },
        ],
    }
    if provider == "openai":
        request["max_completion_tokens"] = max_output_tokens
    else:
        request["max_tokens"] = max_output_tokens
        request["extra_body"] = {"thinking": {"type": deepseek_thinking}}
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        request["temperature"] = temperature

    started = time.perf_counter()
    response = client.chat.completions.create(**request)
    latency = time.perf_counter() - started

    raw_content, finish_reason = _response_content(response)
    if finish_reason == "length":
        raise ModelResponseTruncatedError(
            "The model response was truncated; increase --max-output-tokens"
        )
    rewritten_sql = clean_sql_response(raw_content)
    usage = extract_usage(response)
    return LLMRewriteResult(
        sql=rewritten_sql,
        prompt_version=prompt_version,
        provider=provider,
        requested_model=model,
        response_model=_value(response, "model"),
        target_database=database,
        database_version=database_version,
        latency_seconds=round(latency, 6),
        usage=usage,
        estimated_cost_usd=estimate_cost(usage, pricing),
        finish_reason=finish_reason,
        response_id=_value(response, "id"),
        pricing=pricing,
    )


def create_client(
    provider: str,
    *,
    api_key_env: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> LLMClient:
    """Create the SDK client lazily so unit tests do not require openai."""

    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    config = PROVIDERS[provider]
    env_name = api_key_env or config.api_key_env
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(f"Missing API key: set the {env_name} environment variable")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required; run: pip install -r requirements.txt"
        ) from exc

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    effective_base_url = base_url if base_url is not None else config.base_url
    if effective_base_url:
        kwargs["base_url"] = effective_base_url
    return OpenAI(**kwargs)


@dataclass(frozen=True)
class BatchPayload:
    table_schemas: dict[str, Any]
    sqls: list[dict[str, Any]]
    database: str
    database_version: str


def load_payload(json_path: Path) -> BatchPayload:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    table_schemas = payload.get("table_schemas", {})
    sqls = payload.get("sqls", [])
    if not isinstance(table_schemas, dict):
        raise ValueError("'table_schemas' must be an object")
    if not isinstance(sqls, list):
        raise ValueError("'sqls' must be an array")
    database = normalize_database(
        require_non_empty_string(payload.get("database"), "database")
    )
    database_version = require_non_empty_string(
        payload.get("database_version"), "database_version"
    )
    for index, item in enumerate(sqls):
        if not isinstance(item, dict):
            raise ValueError(f"sqls[{index}] must be an object")
        if "database" in item or "database_version" in item:
            raise ValueError(
                f"sqls[{index}] must not override root-level database metadata"
            )
    return BatchPayload(table_schemas, sqls, database, database_version)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def rewrite_json(
    json_path: Path,
    *,
    client: LLMClient,
    provider: str,
    model: str,
    pricing: Pricing,
    output_path: Path | None = None,
    metrics_path: Path | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    max_output_tokens: int = 4096,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    deepseek_thinking: str = "disabled",
    timeout: float = 180.0,
    max_retries: int = 2,
) -> tuple[Path, Path]:
    payload = load_payload(json_path)
    prompt_version = normalize_prompt_version(prompt_version)
    suffix = f"{payload.database}_{_safe_name(model)}_{prompt_version}"
    output_path = output_path or json_path.parent / f"{json_path.stem}_{suffix}.txt"
    metrics_path = (
        metrics_path or json_path.parent / f"{json_path.stem}_{suffix}.metrics.json"
    )

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    output_blocks: list[str] = []
    query_metrics: list[dict[str, Any]] = []

    for index, item in enumerate(payload.sqls, start=1):
        label = item.get("url") or f"sqls[{index - 1}]"
        sql = item.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"{label}: 'sql' must be a non-empty string")

        query_max_output_tokens = max_output_tokens
        truncation_retries = 0

        def rewrite_current_query() -> LLMRewriteResult:
            return rewrite_sql_with_llm(
                sql,
                payload.table_schemas,
                client=client,
                provider=provider,
                model=model,
                pricing=pricing,
                database=payload.database,
                database_version=payload.database_version,
                prompt_version=prompt_version,
                max_output_tokens=query_max_output_tokens,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                deepseek_thinking=deepseek_thinking,
            )

        try:
            try:
                result = rewrite_current_query()
            except ModelResponseTruncatedError:
                truncation_retries = 1
                query_max_output_tokens *= 2
                print(
                    f"[{index}/{len(payload.sqls)}] ({payload.database}): "
                    "response truncated; retrying this query with "
                    f"{query_max_output_tokens} max output tokens",
                    file=sys.stderr,
                )
                result = rewrite_current_query()
        except Exception as exc:
            exc.add_note(f"while rewriting {label} (item {index})")
            raise

        output_blocks.append(f"{result.sql};")
        metric = asdict(result)
        metric.pop("sql")
        metric["index"] = index
        metric["label"] = label
        metric["max_output_tokens"] = query_max_output_tokens
        metric["truncation_retries"] = truncation_retries
        metric["input_sql_sha256"] = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        metric["output_sql_sha256"] = hashlib.sha256(
            result.sql.encode("utf-8")
        ).hexdigest()
        metric["changed"] = result.sql.rstrip().rstrip(";") != sql.rstrip().rstrip(";")
        system_prompt = build_system_prompt(
            payload.database,
            payload.database_version,
            prompt_version,
        )
        metric["system_prompt_sha256"] = hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest()
        metric["dialect_documentation"] = DIALECTS[payload.database].documentation
        query_metrics.append(metric)
        print(
            f"[{index}/{len(payload.sqls)}] ({payload.database}): "
            f"{result.latency_seconds:.3f}s, "
            f"{result.usage.total_tokens} tokens, "
            f"estimated ${result.estimated_cost_usd:.8f}",
            file=sys.stderr,
        )

    batch_latency = time.perf_counter() - started_clock
    total_usage = TokenUsage(
        prompt_tokens=sum(item["usage"]["prompt_tokens"] for item in query_metrics),
        completion_tokens=sum(
            item["usage"]["completion_tokens"] for item in query_metrics
        ),
        total_tokens=sum(item["usage"]["total_tokens"] for item in query_metrics),
        cached_prompt_tokens=sum(
            item["usage"]["cached_prompt_tokens"] for item in query_metrics
        ),
        reasoning_tokens=sum(
            item["usage"]["reasoning_tokens"] for item in query_metrics
        ),
    )
    report = {
        "source_json": str(json_path.resolve()),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "requested_model": model,
        "target_database": payload.database,
        "database_version": payload.database_version,
        "request_parameters": {
            "prompt_version": prompt_version,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "deepseek_thinking": deepseek_thinking if provider == "deepseek" else None,
            "timeout_seconds": timeout,
            "max_retries": max_retries,
        },
        "pricing_usd_per_million_tokens": asdict(pricing),
        "prompt": {
            "version": prompt_version,
            "base_system_prompt_sha256": hashlib.sha256(
                base_system_prompt(prompt_version).encode("utf-8")
            ).hexdigest(),
            "base_system_prompt_characters": len(base_system_prompt(prompt_version)),
            "system_prompt_sha256": hashlib.sha256(
                build_system_prompt(
                    payload.database,
                    payload.database_version,
                    prompt_version,
                ).encode("utf-8")
            ).hexdigest(),
            "dialect_documentation": DIALECTS[payload.database].documentation,
            "schema_json_characters": len(
                json.dumps(payload.table_schemas, ensure_ascii=False, sort_keys=True)
            ),
            "schema_sha256": hashlib.sha256(
                json.dumps(
                    payload.table_schemas,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "totals": {
            "query_count": len(query_metrics),
            "changed_query_count": sum(item["changed"] for item in query_metrics),
            "batch_latency_seconds": round(batch_latency, 6),
            "api_latency_seconds": round(
                sum(item["latency_seconds"] for item in query_metrics), 6
            ),
            "usage": asdict(total_usage),
            "estimated_cost_usd": round(
                sum(item["estimated_cost_usd"] for item in query_metrics), 12
            ),
        },
        "queries": query_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(output_blocks) + "\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, metrics_path


def resolve_json_path(argument: str | None) -> Path:
    if argument is None:
        return DATA_DIR / "test_sql.json"
    json_path = Path(argument).expanduser()
    if not json_path.is_absolute() and not json_path.exists():
        json_path = DATA_DIR / json_path
    return json_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite MATCH_RECOGNIZE SQL with OpenAI or DeepSeek"
    )
    parser.add_argument(
        "json_path", nargs="?", help="input JSON (default: data/test_sql.json)"
    )
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="openai")
    parser.add_argument("--model", help="provider model ID")
    parser.add_argument(
        "--api-key-env", help="environment variable containing the API key"
    )
    parser.add_argument("--base-url", help="override the provider base URL")
    parser.add_argument("--output", type=Path, help="rewritten SQL output path")
    parser.add_argument("--metrics-output", type=Path, help="JSON metrics output path")
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default=DEFAULT_PROMPT_VERSION,
        help=f"prompt template version (default: {DEFAULT_PROMPT_VERSION})",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--reasoning-effort", help="provider-supported reasoning effort"
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--deepseek-thinking",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--cached-input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")

    json_path = resolve_json_path(args.json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    provider_config = PROVIDERS[args.provider]
    model = args.model or provider_config.default_model
    pricing = resolve_pricing(
        args.provider,
        model,
        args.input_price_per_million,
        args.cached_input_price_per_million,
        args.output_price_per_million,
    )
    client = create_client(
        args.provider,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    output_path, metrics_path = rewrite_json(
        json_path,
        client=client,
        provider=args.provider,
        model=model,
        pricing=pricing,
        output_path=args.output,
        metrics_path=args.metrics_output,
        prompt_version=args.prompt_version,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        deepseek_thinking=args.deepseek_thinking,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    print(f"rewritten queries written to {output_path}")
    print(f"request metrics written to {metrics_path}")
    metrics_report = json.loads(metrics_path.read_text(encoding="utf-8"))
    total_cost = float(metrics_report["totals"]["estimated_cost_usd"])
    print(f"total estimated LLM cost: ${total_cost:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
