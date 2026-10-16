# Democratize MATCH_RECOGNIZE! (VLDB demo paper)

This paper translates a sql with match_recognize keyword into a
plain SQL query executable by any database system that supports
window functions and recursive common table expressions.

Code URL: https://db.cs.uni-tuebingen.de/matchrecognize/

## Key Comments:
- The query performance of the translated sql is poor.
- If the 'from' clause is a subquery, the translation will fail.
- The translation results in an extremely long SQL statement, making manual optimization difficult.

Despite the many inconveniences, we think this work has indeed provided excellent insights.

## Main results

| query id | execution time of original sql | execution time of rewriting sql |
| - | - | - |
| 1 | 0.6s | 8.5s | 
| 2 | 1.8s | 27.8s | 


Mini NASDAQ dataset (about 8.7M row) is downloaded [here](https://github.com/Josehokec/ACER4CER/tree/main/src/main/dataset).

### Query example 1 (query cost: 0.6s --> 8.5s)

Original query is:
```sql
SELECT * FROM nasdaq MATCH_RECOGNIZE (
  PARTITION BY ticker
  ORDER BY     timestamp
  ALL ROWS PER MATCH
  PATTERN (S E)
  DEFINE S AS 1 = 1, E AS E.open > S.open * 1.05
) AS mr
WHERE ticker = 'GOOG';
```

Input the below context into the text box, and click `add new` 
button to compile query. Since the translation query can only be 
executed on DuckDB, we need to use a large model to convert it 
into the corresponding database SQL.

```sql
DROP TABLE IF EXISTS nasdaq;

-- Note that we intentionally pass in a narrow table 
-- to reduce sorting overhead, and subsequently 
-- use a JOIN to retrieve the full query results.
CREATE TABLE nasdaq (
  ticker TEXT,
  timestamp  BIGINT‌,
  open  DOUBLE
);

INSERT INTO nasdaq VALUES
  ('GOOG', 1, 1.42), ('GOOG', 2, 1.45), ('GOOG', 3, 1.46), ('GOOG', 4, 1.43), ('GOOG', 5, 1.40), 
  ('GOOG', 6, 1.49), ('GOOG', 7, 1.47), ('GOOG', 8, 1.42), ('GOOG', 9, 1.43), ('GOOG', 10, 1.44),
  ('GOOG', 11, 1.45), ('GOOG', 12, 1.46), ('GOOG', 13, 1.45), ('GOOG', 14, 1.46),('GOOG', 15, 1.45);

--!PRE-RUN_END

SELECT *
FROM nasdaq MATCH_RECOGNIZE (
  PARTITION BY ticker
  ORDER BY     timestamp
  ALL ROWS PER MATCH
  PATTERN (S E)
  DEFINE  
      S AS 1 = 1,
      E AS E.open > S.open * 1.05
) AS mr
WHERE ticker = 'GOOG';
```

We use chatGPT for database dialect conversion.
```sql
WITH automaton (
    from_state, to_state, transition, penalty, 
    start_flag, finished_flag, all_rows_flag
) AS (
    SELECT 0, 1, 1, 0, 1, 0, 1 FROM dual
    UNION ALL
    SELECT 1, 2, 2, 1, 0, 1, 1 FROM dual
),
prep AS (
    SELECT 
        t.rowid AS rid, 
        t.*, 
        ROW_NUMBER() OVER (
            PARTITION BY t.ticker
            ORDER BY t.timestamp
        ) AS row_id
    FROM nasdaq t
    WHERE ticker = 'GOOG'
),
bfs (
    ticker, finished_flag, start_id, row_id, 
    cur, path, match_length, open_1, open_2
) AS (
    SELECT
        p.ticker, a.finished_flag, p.row_id, p.row_id,
        a.to_state, CAST(a.penalty AS NUMBER) AS path,
        1 AS match_length,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.open
        END AS open_1,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.open
        END AS open_2

    FROM prep p
    CROSS JOIN automaton a
    WHERE a.start_flag = 1
      AND (
            a.transition = 1
            OR
            (
                a.transition = 2
                AND
                CASE
                    WHEN BITAND(a.transition, 2) > 0
                    THEN p.open
                END
                >
                CASE
                    WHEN BITAND(a.transition, 1) > 0
                    THEN p.open
                END * 1.05
            )
          )

    UNION ALL
    SELECT
        p.ticker, a.finished_flag,
        b.start_id, p.row_id, a.to_state,
        b.path * 2 + a.penalty AS path,
        b.match_length + 1,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.open
            ELSE b.open_1
        END AS open_1,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.open
            ELSE b.open_2
        END AS open_2
    FROM bfs b
    JOIN prep p
      ON p.ticker = b.ticker
     AND p.row_id = b.row_id + 1
    JOIN automaton a
      ON a.from_state = b.cur
    WHERE
        (
            a.transition = 1
            OR
            (
                a.transition = 2
                AND
                CASE
                    WHEN BITAND(a.transition, 2) > 0
                    THEN p.open
                    ELSE b.open_1
                END
                >
                CASE
                    WHEN BITAND(a.transition, 1) > 0
                    THEN p.open
                    ELSE b.open_2
                END * 1.05
            )
        )
),
finished_paths AS (
    SELECT
        b.ticker, b.start_id, b.row_id,
        b.path, b.match_length,
        ROW_NUMBER() OVER (
            PARTITION BY b.ticker, b.start_id
            ORDER BY
                b.path * POWER(2, 128 - b.match_length)
        ) AS rn

    FROM bfs b
    WHERE b.finished_flag = 1
),
matches AS (
    SELECT
        ticker, start_id,
        row_id       AS end_id,
        path, match_length
    FROM finished_paths
    WHERE rn = 1
),
seed_values AS (
    SELECT DISTINCT ticker, 1 AS seed
    FROM matches
    UNION
    SELECT DISTINCT
        ticker,
        end_id + 1 AS seed
    FROM matches
),
next_match_ranked AS (
    SELECT
        s.ticker, s.seed, m.start_id,
        m.end_id, m.path, m.match_length,
        ROW_NUMBER() OVER (
            PARTITION BY s.ticker, s.seed
            ORDER BY m.start_id
        ) AS rn
    FROM seed_values s
    JOIN matches m
      ON m.ticker = s.ticker
     AND m.start_id >= s.seed
),
next_match AS (
    SELECT
        ticker, seed, start_id,
        end_id, path, match_length
    FROM next_match_ranked
    WHERE rn = 1
),
skipping (
    ticker, start_id, end_id,
    path, match_length, seed
) AS (
    SELECT DISTINCT
        m.ticker, 0, 0,
        CAST(0 AS NUMBER), 0, 1
    FROM matches m
    UNION ALL
    SELECT
        n.ticker, n.start_id, n.end_id,
        n.path, n.match_length, n.end_id + 1
    FROM skipping s
    JOIN next_match n
      ON n.ticker = s.ticker
     AND n.seed = s.seed
),
collect (
    ticker, start_id, row_id, path,
    match_length, cur, all_rows_flag
) AS (
    SELECT
        s.ticker,
        s.start_id,
        b.row_id,
        s.path,
        s.match_length,
        a.to_state,
        a.all_rows_flag
    FROM skipping s
    JOIN bfs b
      ON b.ticker = s.ticker
     AND b.start_id = s.start_id
     AND b.row_id = s.start_id
    JOIN automaton a
      ON a.start_flag = 1
     AND a.to_state = b.cur
    WHERE s.match_length >= b.match_length
      AND TRUNC(
            s.path /
            POWER(
                2,
                s.match_length - b.match_length
            )
          ) = b.path
      AND s.start_id <> 0
    UNION ALL
    SELECT
        c.ticker, c.start_id, b.row_id, c.path,
        c.match_length, a.to_state, a.all_rows_flag
    FROM collect c
    JOIN bfs b
      ON b.ticker = c.ticker
     AND b.start_id = c.start_id
     AND b.row_id = c.row_id + 1
    JOIN automaton a
      ON a.from_state = c.cur
     AND a.to_state = b.cur
    WHERE c.match_length >= b.match_length
      AND TRUNC(c.path / POWER(2, c.match_length - b.match_length)) = b.path
),
postprocessing AS (
    SELECT n.*
    FROM collect c
    JOIN prep p
      ON p.ticker = c.ticker
     AND p.row_id = c.row_id
    JOIN nasdaq n
      ON n.rowid = p.rid
    WHERE c.all_rows_flag = 1
)
SELECT *
FROM postprocessing mr
WHERE mr.ticker = 'GOOG';
```

Then, we optimized the original translated query.
```sql
WITH automaton (
    from_state, to_state, transition, penalty, 
    start_flag, finished_flag, all_rows_flag
) AS (
    SELECT 0, 1, 1, 0, 1, 0, 1 FROM dual
    UNION ALL
    SELECT 1, 2, 2, 1, 0, 1, 1 FROM dual
),
new_nasdaq AS(
    SELECT * 
    FROM nasdaq
    WHERE ticker = 'GOOG'
),
prep AS (
    SELECT 
        t.rowid AS rid, 
        t.*, 
        ROW_NUMBER() OVER (
            PARTITION BY t.ticker
            ORDER BY t.timestamp
        ) AS row_id
    FROM new_nasdaq t
),
bfs (
    ticker, finished_flag, start_id, row_id, 
    cur, path, match_length, open_1, open_2
) AS (
    SELECT
        p.ticker, a.finished_flag, p.row_id, p.row_id,
        a.to_state, CAST(a.penalty AS NUMBER) AS path,
        1 AS match_length,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.open
        END AS open_1,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.open
        END AS open_2
    FROM prep p
    CROSS JOIN automaton a
    WHERE a.start_flag = 1
      AND (
            a.transition = 1
            OR
            (
                a.transition = 2
                AND
                CASE
                    WHEN BITAND(a.transition, 2) > 0
                    THEN p.open
                END
                >
                CASE
                    WHEN BITAND(a.transition, 1) > 0
                    THEN p.open
                END * 1.05
            )
          )
    UNION ALL
    SELECT
        p.ticker, a.finished_flag,
        b.start_id, p.row_id, a.to_state,
        b.path * 2 + a.penalty AS path,
        b.match_length + 1,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.open
            ELSE b.open_1
        END AS open_1,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.open
            ELSE b.open_2
        END AS open_2
    FROM bfs b
    JOIN prep p
      ON p.ticker = b.ticker
     AND p.row_id = b.row_id + 1
    JOIN automaton a
      ON a.from_state = b.cur
    WHERE
        (
            a.transition = 1
            OR
            (
                a.transition = 2
                AND
                CASE
                    WHEN BITAND(a.transition, 2) > 0
                    THEN p.open
                    ELSE b.open_1
                END
                >
                CASE
                    WHEN BITAND(a.transition, 1) > 0
                    THEN p.open
                    ELSE b.open_2
                END * 1.05
            )
        )
),
finished_paths AS (
    SELECT
        b.ticker, b.start_id, b.row_id,
        b.path, b.match_length,
        ROW_NUMBER() OVER (
            PARTITION BY b.ticker, b.start_id
            ORDER BY
                b.path * POWER(2, 128 - b.match_length)
        ) AS rn
    FROM bfs b
    WHERE b.finished_flag = 1
),
matches AS (
    SELECT
        ticker, start_id,
        row_id       AS end_id,
        path, match_length
    FROM finished_paths
    WHERE rn = 1
),
seed_values AS (
    SELECT DISTINCT ticker, 1 AS seed
    FROM matches
    UNION
    SELECT DISTINCT
        ticker,
        end_id + 1 AS seed
    FROM matches
),
next_match_ranked AS (
    SELECT
        s.ticker, s.seed, m.start_id,
        m.end_id, m.path, m.match_length,
        ROW_NUMBER() OVER (
            PARTITION BY s.ticker, s.seed
            ORDER BY m.start_id
        ) AS rn
    FROM seed_values s
    JOIN matches m
      ON m.ticker = s.ticker
     AND m.start_id >= s.seed
),
next_match AS (
    SELECT
        ticker, seed, start_id,
        end_id, path, match_length
    FROM next_match_ranked
    WHERE rn = 1
),
skipping (
    ticker, start_id, end_id,
    path, match_length, seed
) AS (
    SELECT DISTINCT
        m.ticker, 0, 0,
        CAST(0 AS NUMBER), 0, 1
    FROM matches m
    UNION ALL
    SELECT
        n.ticker, n.start_id, n.end_id,
        n.path, n.match_length, n.end_id + 1
    FROM skipping s
    JOIN next_match n
      ON n.ticker = s.ticker
     AND n.seed = s.seed
),
collect (
    ticker, start_id, row_id, path,
    match_length, cur, all_rows_flag
) AS (
    SELECT
        s.ticker,
        s.start_id,
        b.row_id,
        s.path,
        s.match_length,
        a.to_state,
        a.all_rows_flag
    FROM skipping s
    JOIN bfs b
      ON b.ticker = s.ticker
     AND b.start_id = s.start_id
     AND b.row_id = s.start_id
    JOIN automaton a
      ON a.start_flag = 1
     AND a.to_state = b.cur
    WHERE s.match_length >= b.match_length
      AND TRUNC(
            s.path /
            POWER(
                2,
                s.match_length - b.match_length
            )
          ) = b.path
      AND s.start_id <> 0
    UNION ALL
    SELECT
        c.ticker, c.start_id, b.row_id, c.path,
        c.match_length, a.to_state, a.all_rows_flag
    FROM collect c
    JOIN bfs b
      ON b.ticker = c.ticker
     AND b.start_id = c.start_id
     AND b.row_id = c.row_id + 1
    JOIN automaton a
      ON a.from_state = c.cur
     AND a.to_state = b.cur
    WHERE c.match_length >= b.match_length
      AND TRUNC(c.path / POWER(2, c.match_length - b.match_length)) = b.path
),
postprocessing AS (
    SELECT n.*
    FROM collect c
    JOIN prep p
      ON p.ticker = c.ticker
     AND p.row_id = c.row_id
    JOIN new_nasdaq n
      ON n.rowid = p.rid
    WHERE c.all_rows_flag = 1
)
SELECT *
FROM postprocessing mr
WHERE mr.ticker = 'GOOG';
```

Output:
```text
TICKER    TIMESTAMP       OPEN       HIGH        LOW      CLOSE        VOL
------- ----------- ---------- ---------- ---------- ---------- ----------
GOOG    1556526420 1.217E+003 1.218E+003 1.217E+003 1.217E+003  8.25E+002
GOOG    1556526480 1.288E+003 1.288E+003 1.218E+003 1.218E+003 5.346E+003
GOOG    1584002880 1.103E+003 1.103E+003 1.103E+003 1.103E+003  1.39E+002
GOOG    1584061260 1.177E+003 1.177E+003 1.177E+003 1.177E+003   1.3E+002
GOOG    1612255980 1.928E+003 2.052E+003 1.928E+003 2.052E+003 3.916E+003
GOOG    1612256040 2.052E+003 2.052E+003 2.051E+003 2.052E+003   1.7E+003
GOOG    1667984340 8.737E+001 8.737E+001 8.737E+001 8.737E+001   2.0E+002
GOOG    1668042000 9.204E+001  9.21E+001   9.2E+001  9.21E+001 7.739E+003
```



### Query example 2 (1.8s --> 27.8s)
Original query is:
```sql
SELECT *
FROM nasdaq MATCH_RECOGNIZE (
  PARTITION BY ticker
  ORDER BY     timestamp
  ALL ROWS PER MATCH
  PATTERN (DOWN+ UP* HIGH)
  DEFINE  
      DOWN AS DOWN.open <= PREV(DOWN.open),
      UP   AS UP.open >= PREV(UP.open),
      HIGH AS HIGH.open >= UP.open AND HIGH.open >= 1.067 * FIRST(DOWN.open)
)
WHERE ticker = 'GOOG'
```

We use chatGPT for database dialect conversion.
```sql
WITH automaton (
    from_state, to_state, transition, penalty,
    start_flag, finished_flag, all_rows_flag
) AS (
    SELECT 0, 1, 1, 0, 1, 0, 1 FROM dual
    UNION ALL
    SELECT 1, 2, 2, 1, 0, 0, 1 FROM dual
    UNION ALL
    SELECT 1, 1, 1, 0, 0, 0, 1 FROM dual
    UNION ALL
    SELECT 2, 2, 2, 0, 0, 0, 1 FROM dual
    UNION ALL
    SELECT 1, 3, 4, 3, 0, 1, 1 FROM dual
    UNION ALL
    SELECT 2, 3, 4, 2, 0, 1, 1 FROM dual
),
new_nasdaq AS(
    SELECT * FROM nasdaq WHERE ticker = 'GOOG'
),
prep AS (
    SELECT
        t.rowid AS rid, t.*,
        ROW_NUMBER() OVER (
            PARTITION BY t.ticker
            ORDER BY t.timestamp
        ) AS row_id,
        LAG(t.open, 1) OVER (
            PARTITION BY t.ticker
            ORDER BY t.timestamp
        ) AS prev_prep_open_2,
        LAG(t.open, 1) OVER (
            PARTITION BY t.ticker
            ORDER BY t.timestamp
        ) AS prev_prep_open_5
    FROM new_nasdaq t
),
bfs (
    ticker, finished_flag, start_id, row_id,
    cur, path_digits, match_length,
    open_1, prev_rec_open_3, open_4,
    prev_rec_open_6, open_7, open_8, open_9,
    first_open_10, first_open_offset_11
) AS (
    SELECT
        p.ticker,
        a.finished_flag,
        p.row_id AS start_id,
        p.row_id,
        a.to_state AS cur,
        CAST(
            TO_CHAR(a.penalty)
            AS VARCHAR2(64)
        ) AS path_digits,
        1 AS match_length,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.open
        END AS open_1,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.prev_prep_open_2
        END AS prev_rec_open_3,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.open
        END AS open_4,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.prev_prep_open_5
        END AS prev_rec_open_6,
        CASE
            WHEN BITAND(a.transition, 4) > 0
            THEN p.open
        END AS open_7,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.open
        END AS open_8,
        CASE
            WHEN BITAND(a.transition, 4) > 0
            THEN p.open
        END AS open_9,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.open
        END AS first_open_10,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN -1
            ELSE 0
        END AS first_open_offset_11
    FROM prep p
    CROSS JOIN automaton a
    WHERE a.start_flag = 1
      AND
      (
          (
              a.transition = 1
              AND
              CASE
                  WHEN BITAND(a.transition, 1) > 0
                  THEN p.open
              END
              <=
              CASE
                  WHEN BITAND(a.transition, 1) > 0
                  THEN p.prev_prep_open_2
              END
          )
          OR
          (
              a.transition = 2
              AND
              CASE
                  WHEN BITAND(a.transition, 2) > 0
                  THEN p.open
              END
              >=
              CASE
                  WHEN BITAND(a.transition, 2) > 0
                  THEN p.prev_prep_open_5
              END
          )
          OR
          (
              a.transition = 4
              AND
              CASE
                  WHEN BITAND(a.transition, 4) > 0
                  THEN p.open
              END
              >=
              CASE
                  WHEN BITAND(a.transition, 2) > 0
                  THEN p.open
              END
              AND
              CASE
                  WHEN BITAND(a.transition, 4) > 0
                  THEN p.open
              END
              >=
              (
                  1.067 *
                  CASE
                      WHEN BITAND(a.transition, 1) > 0
                      THEN p.open
                  END
              )
          )
      )
    UNION ALL
    SELECT
        p.ticker, a.finished_flag, b.start_id,
        p.row_id, a.to_state AS cur,
        CAST(
            b.path_digits || TO_CHAR(a.penalty)
            AS VARCHAR2(64)
        ) AS path_digits,
        b.match_length + 1 AS match_length,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.open
            ELSE b.open_1
        END AS open_1,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN p.prev_prep_open_2
            ELSE b.prev_rec_open_3
        END AS prev_rec_open_3,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.open
            ELSE b.open_4
        END AS open_4,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.prev_prep_open_5
            ELSE b.prev_rec_open_6
        END AS prev_rec_open_6,
        CASE
            WHEN BITAND(a.transition, 4) > 0
            THEN p.open
            ELSE b.open_7
        END AS open_7,
        CASE
            WHEN BITAND(a.transition, 2) > 0
            THEN p.open
            ELSE b.open_8
        END AS open_8,
        CASE
            WHEN BITAND(a.transition, 4) > 0
            THEN p.open
            ELSE b.open_9
        END AS open_9,
        COALESCE(
            b.first_open_10,
            CASE
                WHEN BITAND(a.transition, 1) > 0
                     AND b.first_open_offset_11 = 0
                THEN p.open
            END
        ) AS first_open_10,
        CASE
            WHEN BITAND(a.transition, 1) > 0
            THEN b.first_open_offset_11 - 1
            ELSE b.first_open_offset_11
        END AS first_open_offset_11
    FROM bfs b
    JOIN prep p
      ON p.ticker = b.ticker
     AND p.row_id = b.row_id + 1
    JOIN automaton a
      ON a.from_state = b.cur
    WHERE
      (
          (
              a.transition = 1
              AND
              CASE
                  WHEN BITAND(a.transition, 1) > 0
                  THEN p.open
                  ELSE b.open_1
              END
              <=
              CASE
                  WHEN BITAND(a.transition, 1) > 0
                  THEN p.prev_prep_open_2
                  ELSE b.prev_rec_open_3
              END
          )
          OR
          (
              a.transition = 2
              AND
              CASE
                  WHEN BITAND(a.transition, 2) > 0
                  THEN p.open
                  ELSE b.open_4
              END
              >=
              CASE
                  WHEN BITAND(a.transition, 2) > 0
                  THEN p.prev_prep_open_5
                  ELSE b.prev_rec_open_6
              END
          )
          OR
          (
              a.transition = 4
              AND
              CASE
                  WHEN BITAND(a.transition, 4) > 0
                  THEN p.open
                  ELSE b.open_7
              END
              >=
              CASE
                  WHEN BITAND(a.transition, 2) > 0
                  THEN p.open
                  ELSE b.open_8
              END
              AND
              CASE
                  WHEN BITAND(a.transition, 4) > 0
                  THEN p.open
                  ELSE b.open_9
              END
              >=
              (
                  1.067 *
                  COALESCE(
                      b.first_open_10,
                      CASE
                          WHEN BITAND(a.transition, 1) > 0
                               AND b.first_open_offset_11 = 0
                          THEN p.open
                      END
                  )
              )
          )
      )
),
finished_paths AS (
    SELECT
        b.ticker,
        b.start_id,
        b.row_id,
        b.path_digits,
        b.match_length,
        ROW_NUMBER() OVER (
            PARTITION BY b.ticker, b.start_id

            ORDER BY NLSSORT(
                RPAD(b.path_digits, 64, '0'),
                'NLS_SORT=BINARY'
            )
        ) AS rn
    FROM bfs b
    WHERE b.finished_flag = 1
),
matches AS (
    SELECT
        ticker,
        start_id,
        row_id AS end_id,
        path_digits,
        match_length
    FROM finished_paths
    WHERE rn = 1
),
seed_values AS (
    SELECT DISTINCT
        ticker,
        1 AS seed
    FROM matches
    UNION
    SELECT DISTINCT
        ticker,
        end_id + 1 AS seed
    FROM matches
),
next_match_ranked AS (
    SELECT
        s.ticker, s.seed, m.start_id,
        m.end_id, m.path_digits, m.match_length,
        ROW_NUMBER() OVER (
            PARTITION BY s.ticker, s.seed
            ORDER BY m.start_id
        ) AS rn
    FROM seed_values s
    JOIN matches m
      ON m.ticker = s.ticker
     AND m.start_id >= s.seed
),
next_match AS (
    SELECT
        ticker, seed, start_id,
        end_id, path_digits, match_length
    FROM next_match_ranked
    WHERE rn = 1
),
skipping (
    ticker, start_id, end_id,
    path_digits, match_length, seed
) AS (
    SELECT DISTINCT
        m.ticker,
        0 AS start_id,
        0 AS end_id,
        CAST(NULL AS VARCHAR2(64)) AS path_digits,
        0 AS match_length,
        1 AS seed
    FROM matches m
    UNION ALL
    SELECT
        n.ticker, n.start_id, n.end_id,
        n.path_digits, n.match_length,
        n.end_id + 1 AS seed
    FROM skipping s
    JOIN next_match n
      ON n.ticker = s.ticker
     AND n.seed = s.seed
),
collect (
    ticker, start_id, row_id, path_digits,
    match_length, cur, all_rows_flag
) AS (
    SELECT
        s.ticker, s.start_id, b.row_id, s.path_digits,
        s.match_length, a.to_state AS cur, a.all_rows_flag
    FROM skipping s
    JOIN bfs b
      ON b.ticker = s.ticker
     AND b.start_id = s.start_id
     AND b.row_id = s.start_id
    JOIN automaton a
      ON a.start_flag = 1
     AND a.to_state = b.cur
    WHERE s.match_length >= b.match_length
      AND SUBSTR(
              s.path_digits,
              1,
              b.match_length
          ) = b.path_digits
      AND s.start_id <> 0
    UNION ALL
    SELECT
        c.ticker,
        c.start_id,
        b.row_id,
        c.path_digits,
        c.match_length,
        a.to_state AS cur,
        a.all_rows_flag
    FROM collect c
    JOIN bfs b
      ON b.ticker = c.ticker
     AND b.start_id = c.start_id
     AND b.row_id = c.row_id + 1
    JOIN automaton a
      ON a.from_state = c.cur
     AND a.to_state = b.cur
    WHERE c.match_length >= b.match_length
      AND SUBSTR(
              c.path_digits,
              1,
              b.match_length
          ) = b.path_digits
),
postprocessing AS (
    SELECT n.*
    FROM collect c
    JOIN prep p
      ON p.ticker = c.ticker
     AND p.row_id = c.row_id
    JOIN new_nasdaq n
      ON n.rowid = p.rid
    WHERE c.all_rows_flag = 1
)
SELECT *
FROM postprocessing
```

Output:

```
TICKER   TIMESTAMP       OPEN       HIGH        LOW      CLOSE        VOL
------- ---------- ---------- ---------- ---------- ---------- ----------
GOOG    1564041540 1.133E+003 1.136E+003 1.132E+003 1.136E+003 2.057E+004
GOOG    1564041600 1.134E+003  1.14E+003 1.132E+003  1.14E+003 1.474E+005
GOOG    1564041660 1.141E+003 1.175E+003 1.141E+003 1.147E+003 5.637E+003
GOOG    1564041720 1.154E+003 1.184E+003 1.147E+003 1.184E+003 4.622E+003
GOOG    1564041780 1.187E+003 1.198E+003 1.184E+003 1.188E+003 8.899E+003
GOOG    1564041840  1.19E+003   1.2E+003  1.19E+003 1.199E+003 1.341E+004
GOOG    1564041900 1.199E+003 1.199E+003 1.195E+003 1.199E+003 2.163E+003
GOOG    1564041960 1.199E+003 1.207E+003 1.199E+003 1.207E+003 7.891E+003
GOOG    1564042020 1.209E+003 1.212E+003 1.205E+003 1.206E+003 1.174E+004
```


