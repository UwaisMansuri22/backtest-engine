"""Streamlit dashboard: Live Account · Performance · Trade Log · Backtest Replay."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

st.set_page_config(page_title="Trading Bot", layout="wide", page_icon="📈")

# ── data loaders ──────────────────────────────────────────────────────────────


@st.cache_resource
def _client():
    try:
        from backtest_engine.live.alpaca_client import AlpacaClient

        return AlpacaClient()
    except Exception:
        return None


@st.cache_data(ttl=300)
def account_data() -> dict | None:
    c = _client()
    if c is None:
        return None
    try:
        acc, pos = c.get_account(), c.get_positions()
    except Exception:
        return None
    return {
        "equity": float(acc.equity),
        "last_equity": float(acc.last_equity or acc.equity),
        "positions": [
            {
                "Symbol": str(p.symbol),
                "Qty": float(p.qty),
                "Entry Price": float(p.avg_entry_price),
                "Current Price": float(p.current_price),
                "Market Value": float(p.market_value),
                "P&L $": float(p.unrealized_pl),
                "P&L %": float(p.unrealized_plpc) * 100,
            }
            for p in pos
        ],
    }


@st.cache_data(ttl=300)
def live_logs() -> list[dict]:
    d = _ROOT / "results"
    if not d.exists():
        return []
    out: list[dict] = []
    for f in sorted(d.glob("live_log_*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out


@st.cache_data(ttl=300)
def backtest_files() -> list[Path]:
    d = _ROOT / "results"
    if not d.exists():
        return []
    found = list(d.glob("backtest_*.parquet")) + list(d.glob("backtest_*.json"))
    found += [f for f in d.glob("*.parquet") if "live" not in f.stem]
    return sorted(set(found))


@st.cache_data(ttl=86_400)
def spy_returns(start: str, end: str) -> pd.Series:
    try:
        import yfinance as yf

        return (
            yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)[
                "Close"
            ]
            .squeeze()
            .pct_change()
            .dropna()
        )
    except Exception:
        return pd.Series(dtype=float)


def _eq_df(logs: list[dict]) -> pd.DataFrame | None:
    rows = [
        {"date": entry["timestamp"][:10], "equity": entry["account_equity"]}
        for entry in logs
        if "account_equity" in entry
    ]
    if len(rows) < 2:
        return None
    df = (
        pd.DataFrame(rows)
        .drop_duplicates("date")
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .sort_values("date")
        .set_index("date")
    )
    df["ret"] = df["equity"].pct_change()
    df["cum_ret"] = (1 + df["ret"]).cumprod() - 1
    df["drawdown"] = (df["equity"] - df["equity"].cummax()) / df["equity"].cummax()
    return df


def _sharpe(r: pd.Series, min_days: int = 30) -> str:
    r = r.dropna()
    if len(r) < min_days:
        return f"— (min {min_days} days)"
    return f"{r.mean() / r.std() * 252**0.5:.2f}" if r.std() > 0 else "—"


def _max_dd(eq: pd.Series) -> str:
    return f"{((eq - eq.cummax()) / eq.cummax()).min():.1%}" if len(eq) > 1 else "—"


# ── sidebar / auto-refresh ────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigate",
    ["📊 Live Account", "📈 Performance", "📋 Trade Log", "🔁 Backtest Replay"],
)

_now = datetime.now(tz=UTC)
if _now.weekday() < 5 and 13 <= _now.hour < 21:
    st.sidebar.success("🟢 Market open — auto-refresh 60 s")
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)
else:
    st.sidebar.info("🔴 Market closed")
st.sidebar.caption(f"Loaded {_now.strftime('%H:%M UTC')}")

# ══════════════════════════════════════════════════════════════════════════════
# 1 — LIVE ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Live Account":
    st.title("📊 Live Account")
    data, logs = account_data(), live_logs()

    if data is None:
        st.error("Alpaca API keys not configured.")
        st.markdown("""
**To connect your paper-trading account:**

1. Get free paper-trading keys at [alpaca.markets](https://alpaca.markets)
   *(Dashboard → Paper Trading → API Keys)*
2. **Streamlit Cloud:** go to **Settings → Secrets** and add:
   ```toml
   ALPACA_API_KEY = "your_paper_api_key_here"
   ALPACA_SECRET_KEY = "your_paper_secret_key_here"
   ```
3. **Locally:** copy `.env.example` → `.env` and fill in your keys, then
   restart: `streamlit run dashboard/app.py`
""")
        if logs:
            st.divider()
            st.subheader("Last log entry (no-key dry-run data)")
            last = logs[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Equity (last log)", f"${last.get('account_equity', 0):,.0f}")
            c2.metric("Status", last.get("status", "—").upper())
            c3.metric("Regime", "RISK-ON" if last.get("regime_risk_on") else "RISK-OFF")
            with st.expander("Full log entry"):
                st.json(last)
        st.stop()

    eq, last_eq = data["equity"], data["last_equity"]
    today_pnl = eq - last_eq
    first_eq = logs[0].get("account_equity", eq) if logs else eq
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${eq:,.2f}")
    c2.metric(
        "Today P&L",
        f"${today_pnl:+,.2f}",
        f"{today_pnl / last_eq * 100:+.2f}%" if last_eq else None,
    )
    c3.metric("All-time P&L", f"${eq - first_eq:+,.2f}")

    if data["positions"]:
        df_pos = pd.DataFrame(data["positions"])
        left, right = st.columns([1, 2])
        with left:
            fig = px.pie(
                df_pos, values="Market Value", names="Symbol",
                title="Allocation", hole=0.4,
            )
            fig.update_traces(texttemplate="%{label} — %{percent:.1%}", textposition="outside")
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Open Positions")
            d = df_pos.copy()
            for col in ("Entry Price", "Current Price"):
                d[col] = d[col].map("${:.2f}".format)
            d["Market Value"] = d["Market Value"].map("${:,.2f}".format)
            d["P&L $"] = d["P&L $"].map("${:+,.2f}".format)
            d["P&L %"] = d["P&L %"].map("{:+.2f}%".format)
            st.dataframe(d, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    if logs:
        with st.expander("Last run log"):
            st.json(logs[-1])

# ══════════════════════════════════════════════════════════════════════════════
# 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance":
    st.title("📈 Performance")
    eq_df = _eq_df(live_logs())
    if eq_df is None:
        st.info("Need at least 2 run logs to chart performance.")
        st.stop()

    st.plotly_chart(
        px.line(eq_df, y="equity", title="Equity Curve", labels={"equity": "USD"}),
        use_container_width=True,
    )

    n_days = len(eq_df)
    c1, c2 = st.columns(2)

    # BUG 1 — Drawdown: recompute from decimal returns so values are in [-1, 0].
    # eq_df["drawdown"] used equity prices directly which produced micro-values.
    with c1:
        if n_days < 5:
            st.info("Insufficient data for drawdown chart — check back after 5+ trading days")
        else:
            cum = (1 + eq_df["ret"].fillna(0)).cumprod()
            dd = (cum / cum.cummax()) - 1
            fig_dd = px.area(
                dd.rename("drawdown"), title="Drawdown",
                color_discrete_sequence=["#e74c3c"], labels={"drawdown": ""},
            )
            fig_dd.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_dd, use_container_width=True)

    # BUG 2 — Rolling Sharpe: guard against <30 days
    with c2:
        if n_days < 30:
            st.info(
                f"Rolling Sharpe requires 30 days of data — "
                f"{30 - n_days} days remaining"
            )
        else:
            rs = eq_df["ret"].rolling(30).mean() / eq_df["ret"].rolling(30).std() * 252**0.5
            fig_sh = px.line(rs.rename("Sharpe"), title="Rolling 30-day Sharpe")
            fig_sh.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_sh, use_container_width=True)

    monthly = eq_df["ret"].resample("ME").apply(lambda r: (1 + r).prod() - 1) * 100
    # BUG 3 — Heatmap year axis: use int year (not float from pandas DatetimeIndex)
    heat = pd.DataFrame(
        {"y": monthly.index.year.astype(int), "m": monthly.index.month, "v": monthly.values}
    )
    pivot = heat.pivot(index="y", columns="m", values="v")
    pivot.index = pivot.index.astype(str)   # str prevents Plotly treating years as continuous
    pivot.index.name, pivot.columns.name = "Year", ""
    mnames = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    pivot.columns = [mnames[m - 1] for m in pivot.columns]
    st.plotly_chart(
        px.imshow(
            pivot, color_continuous_scale="RdYlGn", text_auto=".1f",
            title="Monthly Returns (%)", color_continuous_midpoint=0, aspect="auto",
        ),
        use_container_width=True,
    )

    start, end = str(eq_df.index[0].date()), str(eq_df.index[-1].date())
    spy_ret = spy_returns(start, end).reindex(eq_df.index).fillna(0)
    spy_cum = (1 + spy_ret).cumprod() - 1
    st.plotly_chart(
        px.line(
            pd.DataFrame({"Strategy": eq_df["cum_ret"], "SPY": spy_cum}),
            title="Cumulative Return vs SPY",
        ),
        use_container_width=True,
    )
    st.dataframe(
        pd.DataFrame({
            "Metric": ["Total Return", "Sharpe (ann.)", "Max Drawdown"],
            "Strategy": [
                f"{eq_df['cum_ret'].iloc[-1]:.1%}",
                _sharpe(eq_df["ret"]),
                _max_dd(eq_df["equity"]),
            ],
            "SPY": [
                f"{spy_cum.iloc[-1]:.1%}" if len(spy_cum) else "—",
                _sharpe(spy_ret) if len(spy_ret) else "—",
                _max_dd((1 + spy_ret).cumprod()) if len(spy_ret) else "—",
            ],
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"*Note: metrics based on {n_days} trading days of live data. "
        "Sharpe and drawdown become meaningful after 30+ days.*"
    )

# ══════════════════════════════════════════════════════════════════════════════
# 3 — TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Trade Log":
    st.title("📋 Trade Log")
    rows = [
        {"date": entry["timestamp"][:10], **order, "run_status": entry.get("status")}
        for entry in live_logs()
        for order in entry.get("orders", [])
    ]
    if not rows:
        st.info("No trades yet — run the bot to build history.")
        st.stop()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    c1, c2, c3, c4 = st.columns(4)
    ticker = c1.selectbox("Ticker", ["All"] + sorted(df["symbol"].unique().tolist()))
    d_from = c2.date_input("From", df["date"].min().date())
    d_to = c3.date_input("To", df["date"].max().date())
    side = c4.selectbox("Side", ["All", "buy", "sell"])

    mask = (df["date"].dt.date >= d_from) & (df["date"].dt.date <= d_to)
    if ticker != "All":
        mask &= df["symbol"] == ticker
    if side != "All":
        mask &= df["side"] == side
    df_f = df[mask]

    st.caption(f"{len(df_f):,} trades")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV", df_f.to_csv(index=False), "trade_log.csv", "text/csv"
    )

# ══════════════════════════════════════════════════════════════════════════════
# 4 — BACKTEST REPLAY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔁 Backtest Replay":
    st.title("🔁 Backtest Replay")
    files = backtest_files()
    if not files:
        st.info(
            "No backtest files in `results/`. "
            "Save an equity curve as `results/backtest_<name>.parquet`."
        )
        st.stop()

    chosen = st.selectbox("Backtest", [f.name for f in files])
    fpath = next(f for f in files if f.name == chosen)
    try:
        df_bt = (
            pd.read_parquet(fpath)
            if fpath.suffix == ".parquet"
            else pd.DataFrame(json.loads(fpath.read_text()))
        )
    except Exception as e:
        st.error(f"Cannot load {chosen}: {e}")
        st.stop()

    try:
        df_bt.index = pd.to_datetime(df_bt.index)
    except Exception:
        pass

    eq_col = next(
        (c for c in ("equity", "portfolio_value", "value") if c in df_bt.columns), None
    )
    if eq_col:
        bt_ret = df_bt[eq_col].pct_change().dropna()
        bt_dd = (df_bt[eq_col] - df_bt[eq_col].cummax()) / df_bt[eq_col].cummax()
        total = df_bt[eq_col].iloc[-1] / df_bt[eq_col].iloc[0] - 1
        st.plotly_chart(
            px.line(df_bt, y=eq_col, title=f"Equity — {chosen}"),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.area(
                    bt_dd.rename("drawdown"), title="Drawdown",
                    color_discrete_sequence=["#e74c3c"],
                ),
                use_container_width=True,
            )
        with c2:
            st.dataframe(
                pd.DataFrame({
                    "Metric": ["Total Return", "Sharpe (ann.)", "Max Drawdown", "Rows"],
                    "Value": [
                        f"{total:.1%}", _sharpe(bt_ret),
                        f"{bt_dd.min():.1%}", f"{len(df_bt):,}",
                    ],
                }),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info(f"No equity column. Available: {list(df_bt.columns)}")

    st.subheader("Columns & parameters")
    st.write(list(df_bt.columns))
    with st.expander("Data preview"):
        st.dataframe(df_bt.head(30), use_container_width=True)
