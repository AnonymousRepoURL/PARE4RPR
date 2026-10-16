"""how to run?

pip install --break-system-packages requests pandas duckdb
export ALPACA_API_KEY="your_API_KEY"
export ALPACA_SECRET_KEY="your_SECRET_KEY"

caffeinate -i python3 download_nasdaq_trades.py


caffeinate -i python3 download_nasdaq_trades.py \
    --start 2024-07-31 \
    --end 2026-07-31 \
    --target 20000000 \
    --max-per-symbol 1000000 \
    --feed sip \
    --csv nasdaq_trades.csv
"""

import os
import json
import time
import argparse
from collections import deque
from datetime import datetime, date, timedelta, timezone
from urllib.parse import quote

import requests
import pandas as pd
import duckdb

# ============================================================
# Alpaca API
# ============================================================

DATA_BASE_URL = "https://data.alpaca.markets/v2/stocks"

ASSET_BASE_URL = "https://paper-api.alpaca.markets/v2/assets"


# ============================================================
# High-liquidity NASDAQ stocks prioritized for download
#
# The program first checks via Alpaca /assets whether these stocks are currently listed on NASDAQ.
# Then it continues downloading other NASDAQ stocks.
# ============================================================

PRIORITY_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "GOOGL",
    "GOOG",
    "AVGO",
    "AMD",
    "INTC",
    "QCOM",
    "MU",
    "AMAT",
    "CSCO",
    "NFLX",
    "ADBE",
    "PLTR",
    "PYPL",
    "INTU",
    "TXN",
    "ADI",
    "PANW",
    "CRWD",
    "MRVL",
    "ARM",
    "ASML",
    "MELI",
    "BKNG",
    "COST",
    "SBUX",
    "PEP",
    "GILD",
    "VRTX",
    "REGN",
    "MDLZ",
    "MAR",
    "ABNB",
    "WDAY",
    "TEAM",
    "DDOG",
    "ZS",
    "FTNT",
    "LRCX",
    "KLAC",
    "SNPS",
    "CDNS",
    "MCHP",
    "NXPI",
    "ON",
]


# ============================================================
# Parameters
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="Download NASDAQ historical trades from Alpaca."
    )

    # The default date range is now:
    # 2024-07-31 ~ 2026-07-31

    parser.add_argument(
        "--start", default="2024-07-31", help="start date, e.g., 2024-07-31"
    )

    parser.add_argument(
        "--end", default="2026-07-31", help="end date, e.g., 2026-07-31"
    )

    parser.add_argument(
        "--target", type=int, default=20_000_000, help="maximum number of records"
    )

    parser.add_argument(
        "--max-per-symbol",
        type=int,
        default=1_000_000,
        help="A maximum of 1,000,000 records per stock will be retained",
    )

    parser.add_argument(
        "--feed",
        choices=["sip", "iex"],
        default="sip",
        help="sip=US consolidated market; iex=IEX only",
    )

    parser.add_argument(
        "--rpm", type=int, default=180, help="maximum API requests per minute"
    )

    # Use a new DuckDB file for the new date range
    parser.add_argument(
        "--db",
        default="nasdaq_trades_20240731_20260731.duckdb",
        help="DuckDB file used for resumable downloads",
    )

    parser.add_argument(
        "--csv", default="nasdaq_trades_20240731_20260731.csv", help="final CSV file"
    )

    return parser.parse_args()


# ============================================================
# Rate Limiter
# ============================================================


class RateLimiter:

    def __init__(self, rpm):
        self.rpm = rpm
        self.timestamps = deque()

    def wait(self):

        now = time.time()

        while self.timestamps and now - self.timestamps[0] >= 60:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.rpm:

            wait_seconds = 60 - (now - self.timestamps[0]) + 0.2

            if wait_seconds > 0:

                # Do not print for waits under 2 seconds to avoid spamming
                if wait_seconds >= 2:
                    print(f"[RATE LIMIT] " f"waiting {wait_seconds:.1f}s...")

                time.sleep(wait_seconds)

            now = time.time()

            while self.timestamps and now - self.timestamps[0] >= 60:
                self.timestamps.popleft()

        self.timestamps.append(time.time())


# ============================================================
# HTTP requests + automatic retry
# ============================================================


def request_json(session, url, headers, limiter, params=None, retries=8):

    for attempt in range(retries):

        limiter.wait()

        try:

            response = session.get(url, headers=headers, params=params, timeout=90)

        except requests.RequestException as e:

            if attempt == retries - 1:
                raise

            wait_seconds = min(2**attempt, 30)

            print(f"[NETWORK ERROR] {e}")

            print(f"retrying in {wait_seconds}s...")

            time.sleep(wait_seconds)

            continue

        # -------------------------
        # Success
        # -------------------------

        if response.status_code == 200:
            return response.json()

        # -------------------------
        # Rate Limit
        # -------------------------

        if response.status_code == 429:

            retry_after = response.headers.get("Retry-After")

            if retry_after:

                wait_seconds = float(retry_after)

            else:

                wait_seconds = min(5 + 2**attempt, 60)

            print(f"[429] Alpaca rate limited, " f"waiting {wait_seconds:.1f}s")

            time.sleep(wait_seconds)

            continue

        # -------------------------
        # Server Error
        # -------------------------

        if 500 <= response.status_code < 600:

            wait_seconds = min(2**attempt, 30)

            print(f"[SERVER ERROR] " f"{response.status_code}")

            print(f"retrying in {wait_seconds}s...")

            time.sleep(wait_seconds)

            continue

        # -------------------------
        # Forbidden
        # -------------------------

        if response.status_code == 403:

            raise RuntimeError(
                "\nAlpaca returned 403 Forbidden.\n\n"
                "If using --feed sip, please confirm:\n"
                "1. API Key is correct;\n"
                "2. The account has historical market data permissions;\n"
                "3. The requested data is not real-time restricted data.\n\n"
                "You can also test with:\n"
                "    --feed iex\n\n"
                f"Response:\n{response.text}"
            )

        # -------------------------
        # Other errors
        # -------------------------

        raise RuntimeError(f"\nHTTP " f"{response.status_code}\n" f"{response.text}")

    raise RuntimeError("reached maximum retry attempts")


# ============================================================
# Fetch NASDAQ symbol list
# ============================================================


def get_nasdaq_symbols(session, headers, limiter):

    print("\nFetching " "NASDAQ symbol list from Alpaca...")

    params = {
        "status": "active",
        "asset_class": "us_equity",
        "exchange": "NASDAQ",
    }

    assets = request_json(session, ASSET_BASE_URL, headers, limiter, params=params)

    symbols = sorted(
        {
            asset["symbol"]
            for asset in assets
            if (asset.get("status") == "active" and asset.get("exchange") == "NASDAQ")
        }
    )

    symbol_set = set(symbols)

    # Prioritize high-liquidity stocks
    priority = [symbol for symbol in PRIORITY_SYMBOLS if symbol in symbol_set]

    priority_set = set(priority)

    remaining = [symbol for symbol in symbols if symbol not in priority_set]

    result = priority + remaining

    print(f"Got {len(result):,} " f"active NASDAQ symbols")

    print("First 20 symbols:")

    print(", ".join(result[:20]))

    return result


# ============================================================
# Quick check:
#
# Does a symbol have any trades within the remaining date range?
#
# If so, find the date of the first trade directly.
#
# This avoids:
#
# AAAP:
# 2024-07-31 0 rows
# 2024-08-01 0 rows
# ...
# 2026-07-31 0 rows
#
# ============================================================


def find_first_trade_date(session, headers, limiter, symbol, start_day, end_day, feed):

    start_dt = datetime(
        start_day.year, start_day.month, start_day.day, 0, 0, 0, tzinfo=timezone.utc
    )

    end_dt = datetime(
        end_day.year,
        end_day.month,
        end_day.day,
        23,
        59,
        59,
        999999,
        tzinfo=timezone.utc,
    )

    start_str = start_dt.isoformat().replace("+00:00", "Z")

    end_str = end_dt.isoformat().replace("+00:00", "Z")

    encoded_symbol = quote(symbol, safe="")

    url = f"{DATA_BASE_URL}/" f"{encoded_symbol}/trades"

    params = {
        "start": start_str,
        "end": end_str,
        # Only need the first one
        "limit": 1,
        "feed": feed,
        # Oldest to newest
        "sort": "asc",
    }

    data = request_json(session, url, headers, limiter, params=params)

    trades = data.get("trades") or []

    # No records at all in the whole range
    if not trades:
        return None

    first_timestamp = pd.Timestamp(trades[0]["t"])

    return first_timestamp.date()


# ============================================================
# trade JSON -> DataFrame
# ============================================================


def normalize_trades(symbol, trades, feed):

    if not trades:
        return pd.DataFrame()

    raw_timestamps = [trade.get("t") for trade in trades]

    timestamps = pd.to_datetime(raw_timestamps, utc=True, errors="raise")

    timestamp_ns = timestamps.astype("int64")

    df = pd.DataFrame(
        {
            "symbol": [symbol] * len(trades),
            "timestamp": (raw_timestamps),
            "timestamp_ns": (timestamp_ns),
            "price": [trade.get("p") for trade in trades],
            "size": [trade.get("s") for trade in trades],
            "exchange": [trade.get("x") for trade in trades],
            "conditions": [
                ";".join(map(str, trade.get("c") or [])) for trade in trades
            ],
            "trade_id": [
                (None if trade.get("i") is None else str(trade.get("i")))
                for trade in trades
            ],
            "tape": [trade.get("z") for trade in trades],
            "feed": [feed] * len(trades),
        }
    )

    return df


# ============================================================
# DuckDB initialization
# ============================================================


def init_database(conn):

    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            symbol          VARCHAR,
            timestamp       VARCHAR,
            timestamp_ns    BIGINT,
            price           DOUBLE,
            size            BIGINT,
            exchange        VARCHAR,
            conditions      VARCHAR,
            trade_id        VARCHAR,
            tape            VARCHAR,
            feed            VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS symbols_order (
            seq INTEGER PRIMARY KEY,
            symbol VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS download_state (
            id INTEGER PRIMARY KEY,
            symbol_idx INTEGER,
            trade_date DATE,
            page_token VARCHAR,
            total_rows BIGINT,
            symbol_rows BIGINT,
            done BOOLEAN,
            config_json VARCHAR
        )
    """)


# ============================================================
# Save symbol order
# ============================================================


def save_symbols(conn, symbols):

    count = conn.execute("""
        SELECT COUNT(*)
        FROM symbols_order
    """).fetchone()[0]

    if count > 0:
        return

    df = pd.DataFrame({"seq": range(len(symbols)), "symbol": symbols})

    conn.register("_symbols_temp", df)

    conn.execute("""
        INSERT INTO symbols_order
        SELECT
            seq,
            symbol
        FROM _symbols_temp
    """)

    conn.unregister("_symbols_temp")


def get_symbols_from_db(conn):

    rows = conn.execute("""
        SELECT symbol
        FROM symbols_order
        ORDER BY seq
    """).fetchall()

    return [row[0] for row in rows]


# ============================================================
# Read checkpoint
# ============================================================


def get_state(conn):

    return conn.execute("""
        SELECT
            symbol_idx,
            trade_date,
            page_token,
            total_rows,
            symbol_rows,
            done,
            config_json
        FROM download_state
        WHERE id = 1
    """).fetchone()


# ============================================================
# Initialize checkpoint
# ============================================================


def initialize_state(conn, start_date, config_json):

    state = get_state(conn)

    if state is None:

        conn.execute(
            """
            INSERT INTO download_state
            VALUES (
                1,
                0,
                ?,
                NULL,
                0,
                0,
                FALSE,
                ?
            )
        """,
            [start_date, config_json],
        )

        return

    old_config = state[6]

    if old_config != config_json:

        raise RuntimeError(
            "\nFound an existing DuckDB checkpoint, "
            "but the download parameters differ from this run.\n\n"
            "Please do not reuse the previous data file.\n"
            "This version uses by default:\n\n"
            "nasdaq_trades_20240731_20260731.duckdb\n"
        )


# ============================================================
# Update checkpoint
# ============================================================


def update_state(
    conn, symbol_idx, trade_date, page_token, total_rows, symbol_rows, done
):

    conn.execute(
        """
        UPDATE download_state
        SET
            symbol_idx = ?,
            trade_date = ?,
            page_token = ?,
            total_rows = ?,
            symbol_rows = ?,
            done = ?
        WHERE id = 1
    """,
        [symbol_idx, trade_date, page_token, total_rows, symbol_rows, done],
    )


# ============================================================
# A full UTC day
# ============================================================


def utc_day_range(day):

    start_dt = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)

    end_dt = start_dt + timedelta(days=1) - timedelta(microseconds=1)

    start_str = start_dt.isoformat().replace("+00:00", "Z")

    end_str = end_dt.isoformat().replace("+00:00", "Z")

    return (start_str, end_str)


# ============================================================
# Export CSV
# ============================================================


def export_csv(conn, csv_filename):

    csv_path = csv_filename.replace("'", "''")

    print()

    print(f"Exporting CSV: " f"{csv_filename}")

    # No ORDER BY
    #
    # Preserve the original physical write order,
    # to avoid artificially creating a pre-sorted state for later database sorting experiments.

    conn.execute(f"""
        COPY (
            SELECT
                symbol,
                timestamp,
                timestamp_ns,
                price,
                size,
                exchange,
                conditions,
                trade_id,
                tape,
                feed
            FROM trades
        )
        TO '{csv_path}'
        (
            FORMAT CSV,
            HEADER TRUE,
            DELIMITER ','
        )
    """)

    print("CSV export complete.")


# ============================================================
# MAIN
# ============================================================


def main():

    args = parse_args()

    # ========================================================
    # API Key
    # ========================================================

    api_key = os.environ.get("ALPACA_API_KEY")

    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:

        raise RuntimeError(
            "\nAlpaca API Key not found.\n\n"
            "Please run:\n\n"
            "export ALPACA_API_KEY='your_API_KEY'\n"
            "export ALPACA_SECRET_KEY='your_SECRET_KEY'\n"
        )

    # ========================================================
    # Time
    # ========================================================

    start_date = date.fromisoformat(args.start)

    end_date = date.fromisoformat(args.end)

    if start_date > end_date:

        raise ValueError("start cannot be later than end")

    # ========================================================
    # Configuration
    # ========================================================

    config = {
        "start": args.start,
        "end": args.end,
        "target": args.target,
        "max_per_symbol": args.max_per_symbol,
        "feed": args.feed,
    }

    config_json = json.dumps(config, sort_keys=True)

    # ========================================================
    # HTTP
    # ========================================================

    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

    session = requests.Session()

    limiter = RateLimiter(args.rpm)

    # ========================================================
    # DuckDB
    # ========================================================

    conn = duckdb.connect(args.db)

    init_database(conn)

    # ========================================================
    # Fetch NASDAQ symbol list
    # ========================================================

    existing_symbols = conn.execute("""
        SELECT COUNT(*)
        FROM symbols_order
    """).fetchone()[0]

    if existing_symbols == 0:

        symbols = get_nasdaq_symbols(session, headers, limiter)

        save_symbols(conn, symbols)

    else:

        symbols = get_symbols_from_db(conn)

        print(f"\nRead {len(symbols):,} symbols " f"from checkpoint database")

    # ========================================================
    # Initialize checkpoint
    # ========================================================

    initialize_state(conn, start_date, config_json)

    state = get_state(conn)

    symbol_idx = state[0]
    trade_date = state[1]
    page_token = state[2]
    total_rows = state[3]
    symbol_rows = state[4]
    done = state[5]

    # ========================================================
    # Already complete
    # ========================================================

    if done:

        print("\nThis task has already finished downloading.")

        print(f"Current record count: " f"{total_rows:,}")

        export_csv(conn, args.csv)

        conn.close()

        return

    # ========================================================
    # Configuration info
    # ========================================================

    print()
    print("================================================")

    print("NASDAQ Historical Trades Downloader")

    print("================================================")

    print(f"Date range     : " f"{start_date} -> {end_date}")

    print(f"Target total   : " f"{args.target:,}")

    print(f"Max per symbol : " f"{args.max_per_symbol:,}")

    print(f"Estimated syms : " f"about " f"{args.target // args.max_per_symbol:,}" f"+")

    print(f"Feed           : " f"{args.feed}")

    print(f"API RPM        : " f"{args.rpm}")

    print(f"DuckDB         : " f"{args.db}")

    print(f"CSV            : " f"{args.csv}")

    print(f"Existing rows  : " f"{total_rows:,}")

    print("================================================")

    print()

    # Print progress every 250k rows
    next_report = (total_rows // 250_000 + 1) * 250_000

    # ========================================================
    # Download main loop
    # ========================================================

    while not done and total_rows < args.target and symbol_idx < len(symbols):

        symbol = symbols[symbol_idx]

        # ====================================================
        # When a symbol is just starting to download:
        #
        # First query the first trade in the entire remaining date range.
        #
        # Solves the day-by-day scanning problem for empty symbols like AAAP.
        # ====================================================

        if symbol_rows == 0 and page_token is None:

            first_trade_date = find_first_trade_date(
                session, headers, limiter, symbol, trade_date, end_date, args.feed
            )

            # ================================================
            # No data at all within the two years
            # ================================================

            if first_trade_date is None:

                print(f"[SKIP NO DATA] " f"{symbol}: " f"{trade_date} ~ " f"{end_date}")

                symbol_idx += 1

                trade_date = start_date

                page_token = None

                symbol_rows = 0

                update_state(
                    conn,
                    symbol_idx,
                    trade_date,
                    page_token,
                    total_rows,
                    symbol_rows,
                    False,
                )

                continue

            # ================================================
            # First trade date is later
            # Skip directly to it
            # ================================================

            if first_trade_date > trade_date:

                print(
                    f"[JUMP] " f"{symbol}: " f"{trade_date} -> " f"{first_trade_date}"
                )

                trade_date = first_trade_date

                update_state(
                    conn, symbol_idx, trade_date, None, total_rows, symbol_rows, False
                )

        # ====================================================
        # Symbol has reached the per-symbol limit
        # ====================================================

        if symbol_rows >= args.max_per_symbol:

            print(f"[DONE SYMBOL] " f"{symbol}: " f"{symbol_rows:,} rows")

            symbol_idx += 1

            trade_date = start_date

            page_token = None

            symbol_rows = 0

            update_state(
                conn, symbol_idx, trade_date, page_token, total_rows, symbol_rows, False
            )

            continue

        # ====================================================
        # Date range scan finished
        # ====================================================

        if trade_date > end_date:

            print(f"[DONE RANGE] " f"{symbol}: " f"{symbol_rows:,} rows")

            symbol_idx += 1

            trade_date = start_date

            page_token = None

            symbol_rows = 0

            update_state(
                conn, symbol_idx, trade_date, page_token, total_rows, symbol_rows, False
            )

            continue

        # ====================================================
        # Skip weekends directly
        # ====================================================

        if trade_date.weekday() >= 5:

            trade_date = trade_date + timedelta(days=1)

            page_token = None

            update_state(
                conn, symbol_idx, trade_date, page_token, total_rows, symbol_rows, False
            )

            continue

        # ====================================================
        # Build API request
        # ====================================================

        start_str, end_str = utc_day_range(trade_date)

        encoded_symbol = quote(symbol, safe="")

        url = f"{DATA_BASE_URL}/" f"{encoded_symbol}/trades"

        params = {
            "start": start_str,
            "end": end_str,
            # Official max per page is 10000
            "limit": 10_000,
            "feed": args.feed,
            # Ascending time order:
            # old -> new
            "sort": "asc",
        }

        if page_token:

            params["page_token"] = page_token

        # ====================================================
        # Request Alpaca
        # ====================================================

        data = request_json(session, url, headers, limiter, params=params)

        trades = data.get("trades") or []

        next_token = data.get("next_page_token")

        # ====================================================
        # Convert to DataFrame
        # ====================================================

        df = normalize_trades(symbol, trades, args.feed)

        # ====================================================
        # Limit how many rows this batch actually writes
        # ====================================================

        remaining_total = args.target - total_rows

        remaining_symbol = args.max_per_symbol - symbol_rows

        keep_rows = min(len(df), remaining_total, remaining_symbol)

        if keep_rows > 0:

            df = df.iloc[:keep_rows].copy()

        else:

            df = pd.DataFrame()

        inserted = len(df)

        new_total = total_rows + inserted

        new_symbol_rows = symbol_rows + inserted

        # ====================================================
        # DuckDB transaction
        #
        # Data write and page token update happen together.
        # ====================================================

        conn.execute("BEGIN TRANSACTION")

        try:

            # -----------------------------
            # Write trades
            # -----------------------------

            if inserted > 0:

                conn.register("_batch", df)

                conn.execute("""
                    INSERT INTO trades
                    SELECT
                        symbol,
                        timestamp,
                        timestamp_ns,
                        price,
                        size,
                        exchange,
                        conditions,
                        trade_id,
                        tape,
                        feed
                    FROM _batch
                """)

                conn.unregister("_batch")

            # -----------------------------
            # Total target reached
            # -----------------------------

            reached_target = new_total >= args.target

            # -----------------------------
            # Current symbol reached the per-symbol limit
            # -----------------------------

            reached_symbol_limit = new_symbol_rows >= args.max_per_symbol

            # ================================================
            # Target reached
            # ================================================

            if reached_target:

                done = True

                next_symbol_idx = symbol_idx

                next_date = trade_date

                next_page = next_token

                state_symbol_rows = new_symbol_rows

            # ================================================
            # Current symbol reached the per-symbol limit
            # ================================================

            elif reached_symbol_limit:

                next_symbol_idx = symbol_idx + 1

                next_date = start_date

                next_page = None

                state_symbol_rows = 0

            # ================================================
            # There is still a next page for this day
            # ================================================

            elif next_token:

                next_symbol_idx = symbol_idx

                next_date = trade_date

                next_page = next_token

                state_symbol_rows = new_symbol_rows

            # ================================================
            # End of day
            # ================================================

            else:

                next_symbol_idx = symbol_idx

                next_date = trade_date + timedelta(days=1)

                next_page = None

                state_symbol_rows = new_symbol_rows

            # -----------------------------
            # Save checkpoint
            # -----------------------------

            update_state(
                conn,
                next_symbol_idx,
                next_date,
                next_page,
                new_total,
                state_symbol_rows,
                done,
            )

            conn.execute("COMMIT")

        except Exception:

            conn.execute("ROLLBACK")

            raise

        # ====================================================
        # Update program state
        # ====================================================

        total_rows = new_total

        symbol_idx = next_symbol_idx

        trade_date = next_date

        page_token = next_page

        symbol_rows = state_symbol_rows

        # ====================================================
        # Progress
        # ====================================================

        if total_rows >= next_report or done:

            percent = total_rows / args.target * 100

            print()

            print(
                f"[PROGRESS] "
                f"{total_rows:,} / "
                f"{args.target:,} "
                f"({percent:.2f}%)"
            )

            print(f"           " f"last symbol = " f"{symbol}")

            print(f"           " f"current date = " f"{trade_date}")

            print(f"           " f"last page rows = " f"{inserted:,}")

            next_report = (total_rows // 250_000 + 1) * 250_000

    # ========================================================
    # Ran out of symbols but still did not reach the target
    # ========================================================

    if total_rows < args.target:

        print()

        print("================================================")

        print("Target record count was not reached.")

        print(f"Currently only: " f"{total_rows:,}")

        print("Try increasing the date range or " "raising --max-per-symbol.")

        print("================================================")

        export_csv(conn, args.csv)

        conn.close()

        return

    # ========================================================
    # Complete
    # ========================================================

    actual_count = conn.execute("""
        SELECT COUNT(*)
        FROM trades
    """).fetchone()[0]

    print()

    print("================================================")

    print("Download complete!")

    print(f"Records: " f"{actual_count:,}")

    print("================================================")

    # ========================================================
    # Export CSV
    # ========================================================

    export_csv(conn, args.csv)

    # ========================================================
    # Data statistics
    # ========================================================

    stats = conn.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT symbol)
                AS symbols,
            MIN(timestamp)
                AS min_timestamp,
            MAX(timestamp)
                AS max_timestamp
        FROM trades
    """).fetchone()

    print()

    print("================================================")

    print("Dataset Summary")

    print("================================================")

    print(f"Rows       : " f"{stats[0]:,}")

    print(f"Symbols    : " f"{stats[1]:,}")

    print(f"Min TS     : " f"{stats[2]}")

    print(f"Max TS     : " f"{stats[3]}")

    print(f"CSV        : " f"{args.csv}")

    print(f"DuckDB     : " f"{args.db}")

    print("================================================")

    # ========================================================
    # Actual row count per symbol
    # ========================================================

    print()

    print("Row counts for top 20 symbols:")

    symbol_stats = conn.execute("""
        SELECT
            symbol,
            COUNT(*) AS rows
        FROM trades
        GROUP BY symbol
        ORDER BY rows DESC
        LIMIT 20
    """).fetchall()

    for symbol, count in symbol_stats:

        print(f"{symbol:10s} " f"{count:>12,}")

    conn.close()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
