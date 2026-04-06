# Prediction Market Lead-Lag Alpha

**Exploiting the information speed advantage of prediction market makers to front-run macro event repricing in equities.**

## Thesis

Large systematic market makers (Susquehanna/SIG, DRW, Jump Trading) operate
on prediction markets like Polymarket and Kalshi.  These firms have superior
infrastructure for processing macro news (CPI, FOMC, NFP, GDP, PCE, PMI, ...).
Because prediction markets are thin (low liquidity, small book depth), their
trades reprice contracts almost instantly after a news release.  Equity and ETF
markets, by contrast, have orders-of-magnitude more liquidity and depth. Therefore,
repricing takes longer to propagate through the full order book.

**This creates a measurable lead-lag relationship:**  prediction market prices
move first, equity/ETF prices follow seconds to minutes later.  We are
effectively piggy-backing on the fastest participants infrastructure
by reading their footprint in the prediction market and acting in the
equity market before the broader repricing is complete.

## Strategy Overview

```
 Macro Event (e.g. CPI release at 8:30 ET)
        |
        v
 Prediction Market      Equity / ETF Market
 (Polymarket)            (NYSE, NASDAQ)
        |                       |
  [Instant reprice]       [Slower reprice]
  by SIG/DRW/Jump         due to depth
        |                       |
        v                       |
  DETECT price move  --------->  TRADE before
  (lead signal)                  full repricing
                                 (lag capture)
```

### Phase 1 -- Lead-Lag Around Scheduled Macro Events
- Ingest 1-min OHLCV bars from EODHD (US equities + ETFs) and
  Polymarket/Telonex (prediction market fills).
- Build event-window datasets: for each macro release, align prediction
  market and equity price series from T-72h to T+72h.
- **Path signature lead-lag detection**: form 2D paths
  (pred_mkt_probability, equity_return) around each event and compute
  truncated path signatures.  The level-2 cross-terms (signed area)
  directly measure who moved first and by how much -- capturing nonlinear
  lead-lag that cross-correlation misses.  Higher-order terms detect
  acceleration, reversal, and more complex temporal patterns.
- Baseline comparisons: cross-correlation, Granger causality, impulse
  response at 1-min resolution.
- Generate trading signals: when prediction market probability moves > X
  standard deviations within Y minutes of a scheduled event, take a
  directional position in the corresponding ETF/equity basket.

### Phase 2 -- Unscheduled Volatility Wake-Up
- Monitor prediction market volatility in real-time.  When implied
  probability swings exceed a threshold outside of scheduled events
  (geopolitical shock, surprise policy, war, etc.), "wake up" the
  algorithm and scan for equity positioning opportunities.
- **Topological data analysis (TDA)**: use persistent homology on the
  joint (pred_mkt, equity) point cloud to detect regime changes in the
  lead-lag structure.  Persistence diagrams that shift between calm and
  crisis periods reveal when the usual lead-lag breaks down -- or
  when new topological features (loops, holes) appear in the joint
  dynamics, signaling a structural change worth trading.

### Phase 3 -- Tick-Level Microstructure
- Move from 1-min bars to tick-by-tick data.
- Test whether the lead-lag relationship is tighter at tick resolution
  and disappears at longer windows (evidence of rapid information
  diffusion that only leaves alpha at the highest frequency).
- Analyse how the lead-lag decay curve behaves across event types and
  market regimes.
- Use path signatures at tick resolution to measure how quickly the
  signed area builds and dissipates -- this gives the alpha half-life.

## Methodology: Path Signatures & TDA

### Why path signatures for lead-lag?

Standard approaches (cross-correlation, Granger causality) assume linear
relationships and fixed lag lengths.  Macro events are neither -- CPI
might reprice prediction markets in 10 seconds but take 3 minutes to
propagate through SPY, while FOMC might show a 30-second lead.

Path signatures (from rough path theory) solve this by encoding the
**sequential structure** of a multivariate time series.  For a 2D path
`(pred_mkt_t, equity_t)`:

- **Level 1**: total displacement of each series (trivial).
- **Level 2**: the **signed area** between the two curves.  If pred_mkt
  consistently moves before equity, the integral `int pred d(equity)`
  differs from `int equity d(pred)` -- this asymmetry IS the lead-lag.
- **Level 3+**: captures acceleration, reversal patterns, and higher-order
  temporal structure that linear methods miss entirely.

Key advantages for this project:
- Handles **irregular sampling** (prediction markets are sparse).
- Robust to **variable lag lengths** across event types.
- Signature features can be fed into ML models for signal generation.
- Mathematically principled (universal nonlinearity of paths).

### Why TDA for regime detection?

Persistent homology applied to sliding windows of the joint price series
produces persistence diagrams that encode the "shape" of the dynamics.
When the lead-lag relationship changes (regime shift), the topology
changes -- new loops appear or existing ones collapse.  This is a
model-free way to detect when your signal is live vs. dead.

## Data Sources

| Source | What | Resolution | API Cost |
|--------|------|-----------|----------|
| **EODHD** | US equities & ETFs intraday OHLCV | 1-min bars | 5 credits/req |
| **EODHD** | US equities tick-by-tick trades | Tick (ms) | 10 credits/req |
| **EODHD** | Macro economic event calendar | Event-level | 1 credit/req |
| **Telonex** | Polymarket on-chain fills | Fill-level | API key required |

> **Note on Telonex API throughput:** Telonex serves fill data as one
> Parquet file per market per day, with no bulk or batch endpoint.  For a
> macro-filtered universe of ~3,600 markets over a 2-year window, this
> translates to ~265k individual HTTP requests even though the total
> payload compresses to a few hundred megabytes.  The SDK provides
> concurrency within a single market (parallel day downloads), but each
> `download()` call still blocks on its own async event loop, limiting
> cross-market parallelism.  We mitigate this with `ThreadPoolExecutor`
> (4 markets in parallel, each with 20 concurrent day-level fetches), but
> a full ingestion still takes several hours where a single bulk query
> would take seconds.  If Telonex adds a batch endpoint or date-range
> downloads in the future, ingestion time would drop by 1--2 orders of
> magnitude.

> **Kalshi** (CFTC-regulated macro prediction market) has structured contracts
> for Fed rates, CPI, GDP, unemployment, etc.  However, **Kalshi restricts API
> access to US-based users only**.  The client and ingestion code are included
> in `src/data/` for future use but are disabled in the default pipelines.
> Polymarket offers better liquidity on the same macro events and has no
> geo-restriction, so it serves as the primary prediction market source.

## Project Structure

```
pred_mkt_tick_event/
|-- README.md
|-- pyproject.toml              # Dependencies + project metadata
|-- config.py                   # Loads .env, exposes API keys & paths
|-- .env                        # API keys (git-ignored)
|-- .env.example                # Template for .env
|
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- eodhd_client.py     # EODHD REST API client (rate-limited, budgeted)
|   |   |-- telonex_client.py   # Telonex/Polymarket SDK wrapper
|   |   |-- kalshi_client.py    # Kalshi v2 API client (disabled -- US-only access)
|   |   |-- watchlists.py       # Tiered ticker lists (macro / full / custom)
|   |   |-- macro_calendar.py   # EODHD economic event calendar -> DuckDB
|   |   |-- ingest_eodhd.py     # 1-min bar pipeline for US equities
|   |   |-- ingest_telonex.py   # 1-min bar pipeline for Polymarket
|   |   |-- ingest_kalshi.py    # 1-min bar pipeline for Kalshi (disabled -- US-only)
|   |   |-- ingest_ticks.py     # Tick-by-tick pipeline (EODHD) -> DuckDB
|   |   +-- ingest_cli.py       # CLI entry point for all pipelines
|   +-- analysis/
|       |-- __init__.py
|       |-- lead_lag.py          # Path signature lead-lag + classical baselines
|       +-- tda.py               # Topological data analysis (persistent homology)
|
+-- data/
    +-- raw/                    # All downloaded data lands here (git-ignored)
        |-- ticks.duckdb        # Tick data + macro calendar (single DuckDB file)
        |-- eodhd/
        |   |-- universe.parquet
        |   +-- bars_1m/*.parquet
        +-- telonex/
            |-- telonex_markets.parquet
            +-- bars_1m/*.parquet
```

## Quick Start

### 1. Install

```bash
# Clone and enter the project
git clone <repo-url> && cd pred_mkt_tick_event

# Create virtualenv and install dependencies
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install -e ".[dev]"
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

| Key | Where to get it | Required? |
|-----|----------------|-----------|
| `TELONEX_API_KEY` | https://telonex.io | Yes |
| `EODHD_API_KEY` | https://eodhd.com/pricing (Fundamentals or All-in-One plan) | Yes |
| `KALSHI_API_KEY` | https://kalshi.com/account/api | No (US-only, disabled by default) |

### 3. Ingest Data

```bash
# Fetch the macro event calendar first (very cheap: ~24 API credits)
python -m src.data.ingest_cli calendar -v

# Ingest 1-min bars
python -m src.data.ingest_cli eodhd --years 2 -v
python -m src.data.ingest_cli telonex -v

# Or run both bar pipelines at once:
python -m src.data.ingest_cli all --years 2 -v

# Tick-by-tick: macro-sensitive names only (~12k EODHD credits)
python -m src.data.ingest_cli ticks --mode macro -v

# Tick-by-tick: only around macro events (cheapest, most signal-dense)
python -m src.data.ingest_cli event-ticks --category CPI --category FOMC -v

# Tick-by-tick: specific tickers
python -m src.data.ingest_cli ticks --ticker AAPL --ticker SPY -v
```

### Cost Estimates (EODHD)

| Pipeline | Tickers | API Credits | Disk | Runtime |
|----------|---------|------------|------|---------|
| Calendar | - | ~24 | <1 MB | <1 min |
| 1-min bars (universe) | ~500 | ~15k | ~5 GB | ~50 min |
| Ticks (macro mode) | ~70 | ~12k | ~10 GB | ~30 min |
| Ticks (full universe) | ~500 | ~170k | ~150 GB | ~2 days |
| Event-ticks (macro) | ~70 | ~3k | ~2 GB | ~10 min |

All pipelines are **incremental** -- re-running picks up where you left off.
Use `--budget N` to cap EODHD credits per run.

## Storage Design

- **1-min bars**: one Parquet file per ticker (fast random access, easy to extend).
- **Tick data**: single DuckDB database with `eodhd_ticks` table.
  DuckDB gives native append (no read-concat-rewrite), SQL across all tickers, and
  columnar compression comparable to Parquet.
- **Macro calendar**: `macro_events` table in the same DuckDB file.

## Key Design Decisions

- **Rate limiting + budget tracking**: all API clients enforce per-second and
  per-minute rate limits.  EODHD client tracks credit cost per endpoint type
  and raises `BudgetExhausted` when the cap is hit.
- **Incremental ingestion**: every pipeline checks what's already on disk/DB
  and only fetches missing ranges.  Safe to interrupt and resume.
- **Event-window mode**: instead of fetching ticks for every second of 2 years,
  only fetch around macro events (T-72h to T+72h).  Cheaper than continuous,
  and that's where the signal lives.
- **Kalshi deferred**: Kalshi client and pipeline code is included but not wired
  into the default CLI.  If you gain US-based API access, enable it by passing
  `--source kalshi` or `--source both` to the ticks/event-ticks commands, or
  run `python -m src.data.ingest_cli kalshi -v` directly for 1-min bars.

## License

See [LICENSE](LICENSE).
