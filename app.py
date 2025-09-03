import streamlit as st
import yfinance as yf
import pandas as pd

# RSI calculation function
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Streamlit app
st.title("📈 RSI Calculator (Last 1 Month)")

ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL):", "AAPL")

if st.button("Get Data"):
    try:
        # Download last 1 month daily data
        df = yf.download(ticker, period="1mo", interval="1d")

        if not df.empty:
            df["RSI"] = calculate_rsi(df["Close"])
            df = df[["Close", "RSI"]].dropna()

            st.subheader(f"RSI for {ticker} (Last 1 Month)")
            st.dataframe(df)

            # Plot RSI
            st.subheader("📊 RSI Chart")
            st.line_chart(df[["RSI"]])

            # Plot Closing Price
            st.subheader("📊 Closing Price Chart")
            st.line_chart(df[["Close"]])

        else:
            st.error("No data found. Please check the ticker symbol.")
    except Exception as e:
        st.error(f"Error: {e}")

















'''# streamlit_app.py
# ----------------------------
# Strike-style RRG — fixed benchmark handling
# ----------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# ------------------------
# Helpers
# ------------------------
def _to_list(s: str):
    if not s:
        return []
    return [t.strip() for t in s.split(',') if t.strip()]

def fetch_prices(tickers, start, end, interval):
    """Download adjusted close prices for tickers."""
    if yf is None:
        raise RuntimeError("yfinance is not available. Please `pip install yfinance`.")

    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False, group_by='ticker')

    if isinstance(data.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            try:
                closes[t] = data[t]['Close']
            except Exception:
                try:
                    closes[t] = data['Close'][t]
                except Exception:
                    pass
        df = pd.DataFrame(closes)
    else:
        if 'Close' in data:
            df = data['Close'].to_frame(name=tickers[0])
        else:
            df = data.to_frame(name=tickers[0])
    df = df.sort_index().dropna(how='all')
    return df

def resample_prices(df, mode):
    if mode == 'Daily':
        return df
    rule = {'Weekly': 'W-FRI', 'Monthly': 'M'}.get(mode, None)
    if rule is None:
        return df
    return df.resample(rule).last().dropna(how='all')

def compute_rrg(df_prices, benchmark, span_ratio=20, span_mom=5):
    if benchmark not in df_prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' not found in price columns.")
    prices = df_prices.ffill().dropna(how='all')
    bench = prices[benchmark]

    out = []
    for t in prices.columns:
        if t == benchmark:
            continue
        series = prices[t]
        if series.dropna().empty:
            continue
        rs = (series / bench).replace([np.inf, -np.inf], np.nan).dropna()
        if rs.empty:
            continue
        rs_ratio = 100.0 * (rs / rs.ewm(span=span_ratio, adjust=False).mean())
        rs_mom = 100.0 * (rs_ratio / rs_ratio.ewm(span=span_mom, adjust=False).mean())

        tmp = pd.DataFrame({
            'Date': rs.index,
            'Ticker': t,
            'RS': rs.values,
            'RS_Ratio': rs_ratio.values,
            'RS_Momentum': rs_mom.values,
            'Price': series.reindex(rs.index).values,
            'Benchmark': bench.reindex(rs.index).values,
        })
        out.append(tmp)

    if not out:
        return pd.DataFrame(columns=['Date','Ticker','RS','RS_Ratio','RS_Momentum','Price','Benchmark'])
    return pd.concat(out, axis=0, ignore_index=True)

def make_rrg_figure(df_long, tail=15, today_only=False):
    if df_long.empty:
        fig = go.Figure()
        fig.update_layout(title='RRG — No data to display')
        return fig

    last_dates = df_long.groupby('Ticker')['Date'].max()
    latest = df_long.merge(last_dates.rename('LastDate'), on='Ticker')
    latest = latest[latest['Date'] == latest['LastDate']]

    x0, y0 = 100, 100
    xpad, ypad = 20, 20
    fig = go.Figure()
    fig.add_shape(type='rect', x0=-1e6, y0=y0, x1=x0, y1=1e6, opacity=0.06, layer='below')
    fig.add_shape(type='rect', x0=x0, y0=-1e6, x1=1e6, y1=y0, opacity=0.06, layer='below')
    fig.add_shape(type='line', x0=x0, x1=x0, y0=-1e6, y1=1e6, line=dict(width=1))
    fig.add_shape(type='line', x0=-1e6, x1=1e6, y0=y0, y1=y0, line=dict(width=1))

    if not today_only:
        for tkr, g in df_long.groupby('Ticker'):
            g = g.sort_values('Date').tail(tail)
            fig.add_trace(go.Scatter(
                x=g['RS_Ratio'], y=g['RS_Momentum'], mode='lines+markers',
                name=f"{tkr} trail", legendgroup=tkr, showlegend=False,
                hovertemplate=(
                    "<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<br>"
                    "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                ),
                text=[tkr]*len(g),
                customdata=g['Date']
            ))

    for _, row in latest.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['RS_Ratio']], y=[row['RS_Momentum']], mode='markers+text',
            text=[row['Ticker']], textposition='top center', name=row['Ticker'], legendgroup=row['Ticker'],
            marker=dict(size=12, line=dict(width=1)),
            hovertemplate=(
                f"<b>{row['Ticker']}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<br>"
                f"Date: %{{customdata|%Y-%m-%d}}<extra></extra>"
            ),
            customdata=[row['Date']]
        ))

    fig.update_layout(
        title='Relative Rotation Graph (RRG)',
        xaxis_title='RS-Ratio (~100 Neutral)',
        yaxis_title='RS-Momentum (~100 Neutral)',
        xaxis=dict(range=[x0 - xpad, x0 + xpad], zeroline=False, showgrid=True),
        yaxis=dict(range=[y0 - ypad, y0 + ypad], zeroline=False, showgrid=True),
        legend_title='Symbols',
        template='plotly_white',
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="RRG — Strike-style", layout="wide")
st.title("📈 Relative Rotation Graph (RRG) — Strike-style (Python)")

with st.sidebar:
	st.header("⚙️ Controls")
	default_universe = "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS"
	tickers_str = st.text_area("Tickers (comma-separated)", value=default_universe, height=90)
	benchmark = st.text_input("Benchmark ticker", value="^BSESN", help="e.g., ^NSEI, ^NSEBANK, NIFTYBEES.NS")
	col_a, col_b = st.columns(2)
	with col_a:
		start_date = st.date_input("Start", value=pd.Timestamp.today() - pd.Timedelta(days=365*3))
	with col_b:
		end_date = st.date_input("End", value=pd.Timestamp.today())

	tf = st.selectbox("Timeframe", options=["Daily", "Weekly", "Monthly"], index=0)
	span_ratio = st.slider("RS-Ratio EMA span", min_value=5, max_value=60, value=20, step=1)
	span_mom = st.slider("RS-Momentum EMA span", min_value=3, max_value=30, value=5, step=1)
	tail = st.slider("Tail length (periods)", min_value=1, max_value=60, value=15, step=1)
	run = st.button("▶️ Run RRG")

# ------------------------
# Run logic
# ------------------------
if run:
    st.info("Downloading data…")
    tickers = _to_list(tickers_str)

    # Ensure benchmark is included
    if benchmark not in tickers:
        tickers.append(benchmark)

    try:
        prices = fetch_prices(tickers, str(start_date), str(end_date), "1d")
        prices = prices.dropna(how='all')
        prices_rs = resample_prices(prices, tf)
        valid_cols = [c for c in prices_rs.columns if prices_rs[c].dropna().shape[0] > 5]
        prices_rs = prices_rs[valid_cols]

        if benchmark not in prices_rs.columns:
            st.error(f"Benchmark '{benchmark}' not found in Yahoo data. Try '^NSEI', '^BSESN', 'NIFTYBEES.NS', '^NSEBANK'.")
            st.stop()

        df_rrg = compute_rrg(prices_rs, benchmark=benchmark, span_ratio=span_ratio, span_mom=span_mom)
        st.success("Done.")
        fig = make_rrg_figure(df_rrg, tail=tail, today_only=False)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.subheader("Data Preview")
        st.dataframe(df_rrg.sort_values(['Ticker','Date']).tail(200), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)'''






