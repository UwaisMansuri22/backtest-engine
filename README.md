# Backtest Engine

A vectorized backtesting engine for daily-bar equity strategies, built in Python.

## Goals

1. **Vectorized engine** — pandas/numpy-based daily-bar backtesting, no slow row-by-row loops.
2. **Two strategies** — SMA crossover (trend-following) and cross-sectional momentum (ranking-based).
3. **Proper metrics** — Sharpe ratio, max drawdown, CAGR, win rate, profit factor.
4. **Bug-free by design** — no lookahead bias, no survivorship bias, transaction costs included.
5. **Paper trading** — eventual deployment via AWS Lambda + EventBridge (future phase).
6. **Production-grade** — fully typed (mypy strict), tested (pytest), linted (ruff), modular.

## Folder Structure

```
backtest_engine/
  data/         # data loaders (yfinance, CSV)
  strategies/   # strategy implementations (SMA crossover, momentum)
  backtest/     # core engine: signal → position → returns pipeline
  metrics/      # performance metrics (Sharpe, drawdown, CAGR, etc.)
  utils/        # shared helpers (logging, date utils)
tests/          # pytest test suite
notebooks/      # Jupyter notebooks for exploration and visualization
data_cache/     # locally cached price data (gitignored)
results/        # backtest output: CSVs, charts (gitignored)
```

## Setup

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11+.

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run mypy backtest_engine
```

## Current Status

- [x] Project scaffold: folder structure, pyproject.toml, .gitignore
- [ ] Data layer: yfinance loader with local caching
- [ ] Metrics module: Sharpe, CAGR, max drawdown, win rate, profit factor
- [ ] Core engine: signal → position sizing → returns with transaction costs
- [ ] Strategy: SMA crossover
- [ ] Strategy: Cross-sectional momentum
- [ ] Visualization: equity curve, drawdown chart
- [ ] Paper trading: AWS Lambda + EventBridge deployment
