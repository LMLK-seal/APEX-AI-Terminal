<div align="center">

<!-- LOGO / BANNER -->
<img src="https://img.shields.io/badge/APEX%20TERMINAL-AI%20Financial%20Workstation-ff7a1a?style=for-the-badge&logo=python&logoColor=white" alt="APEX Terminal"/>

# APEX Terminal

### AI-Powered Bloomberg-Style Financial Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/Qt-PyQt6-41CD52?style=flat-square&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![LM Studio](https://img.shields.io/badge/AI-LM%20Studio-8B5CF6?style=flat-square)](https://lmstudio.ai)
[![License](https://img.shields.io/badge/License-Private-red?style=flat-square)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-0078D4?style=flat-square)]()

> **Private. Local. Institutional-grade.**  
> Real-time candlestick charting · 16 technical indicators · 5-agent AI consensus · Options analytics · Probability engine · News intelligence · AI stock screener — all running on your own hardware.

</div>

---

## 📸 Overview

APEX Terminal is a self-hosted, AI-enhanced market analysis workstation designed for sophisticated traders and quantitative analysts. All AI inference routes through a locally running **LM Studio** instance — no cloud API costs, no third-party data leakage, complete privacy.

---

## ⚡ Quickstart (EXE Users)

### Step 1 — Install LM Studio

1. Download from **[https://lmstudio.ai](https://lmstudio.ai)**
2. Install and launch LM Studio
3. In the **Discover** tab, download a model:
   - `mistral-3-14b-instruct` — Great vision model - recommended
   - `llama-3.1-8b-instruct` — balanced quality. (Text only)
   - `Gemma-4-26b-a4b-it` — deep reasoning and Vision.
   - `mistralai/mistral-nemo-instruct-2407` — Text only, great quality.
   
4. Go to the **Local Server** tab → select your model → click **Start Server**
5. Confirm: `Server running at http://localhost:1234`

### Step 2 — Launch APEX Terminal

- The app auto-checks LM Studio on startup
- Status bar shows **model name in green** when connected
- If offline: open **SETTINGS** tab → verify URL → click **Test Connection**

### Step 3 — Load a Chart

1. Double-click any ticker in the **Watchlist** on the left
2. Or type a symbol in the watchlist input box and press `Enter`
3. The chart loads automatically with default indicators active

---

## 🖥️ Interface Layout

---

## 📊 CHART Tab

The primary analysis workspace. Bloomberg-style candlestick chart with real-time price updates and 16 configurable overlays.

### Timeframes

| Label | Period | Interval | Best Use |
|-------|--------|----------|----------|
| 1D | 1 day | 5 min | Intraday / scalping |
| 5D | 5 days | 15 min | Short-swing setups |
| 1M | 1 month | 1 hour | Swing trading |
| 3M | 3 months | Daily | Medium-term trend |
| 6M | 6 months | Daily | Position trading |
| 1Y | 1 year | Daily | **Default — full cycle** |
| 2Y | 2 years | Weekly | Multi-year structure |
| 5Y | 5 years | Monthly | Strategic view |

### Indicator Toggles

| Toggle | Indicator | Default | Purpose |
|--------|-----------|---------|---------|
| MA | Moving Averages (20/50/200) | ✅ ON | Trend direction & dynamic S/R |
| BB | Bollinger Bands | ✅ ON | Volatility envelope & squeeze detection |
| VOL | Volume Bars | ✅ ON | Participation confirmation |
| RSI | Relative Strength Index (14) | ✅ ON | Momentum oscillator, divergence |
| VWAP | Volume-Weighted Avg Price | ✅ ON | Institutional fair-value benchmark |
| MACD | MACD / Signal / Histogram | OFF | Crossover & acceleration signals |
| ATR | Average True Range (14) | OFF | Volatility & stop-loss sizing |
| OBV | On-Balance Volume | OFF | Accumulation/distribution flow |
| CMF | Chaikin Money Flow (20) | OFF | Buying vs selling pressure |
| RVI | Relative Vigor Index | OFF | Closing-strength momentum filter |
| SCALP | Scalp Signal System | OFF | Composite fast-trade alert |
| GBM | Monte Carlo path overlay | OFF | Probabilistic future range cloud |
| MAXP | Options Max Pain | OFF | Strike-pinning and expiry gravity |
| FVG | Fair Value Gaps | OFF | Institutional imbalance zones |
| Regime | Market Regime (HMM) | OFF | Behavior classification |
| VP | Volume Profile | OFF | POC / value-area levels |

---

## 🔬 ANALYSIS Tab

Structured equity-research report auto-generated for the active ticker.

- **Trend Analysis** — price vs MA stack, VWAP position, structural bias
- **Momentum** — RSI reading, MACD state, volatility Z-score
- **Confluence Map** — weight-of-evidence ranking across all active signals
- **Options Flow** — Put/Call Ratio (OI + Volume), PCR Velocity, Smart Money signal
- **SPY Correlation** — Beta, relative strength 30d/90d/1yr, beta-adjusted alpha
- **Earnings Proximity** — reliability flag when binary event is near
- **Insider Activity** — SEC Form 4 derived signal and ownership context

### 🤖 POD — 5-Agent AI Consensus Engine

The most comprehensive analysis mode. Five specialist AI agents run sequentially:

| Agent | Role |
|-------|------|
| **CHARTIST** | Trend, momentum, support/resistance, overextension |
| **FUNDAMENTALIST** | Valuation, DCF, growth, margins, balance sheet |
| **NEWS SENTRY** | Catalysts, options risk, sentiment alignment |
| **MACRO SENTINEL** | Risk-on/off regime, VIX, sector rotation, dollar/bonds |
| **ORCHESTRATOR** | Synthesises all four → Confidence Score + Action Rating + Kelly Sizing |

> Output includes: Signal Alignment · ML XGBoost Forecast · Regime Context · Trade Thesis · Confidence Score (0–100) · Position Sizing · **STRONG BUY / BUY / HOLD / SELL / STRONG SELL**

---

## 📁 DOCUMENTS Tab

Load any financial document and interrogate it with your local AI.

**Supported formats:** `.pdf` · `.txt` · `.md` · `.csv` · `.json` · `.log`

| Button | Action |
|--------|--------|
| **LOAD DOCUMENT** | Extract and display file text |
| **UNLOAD** | Clear document from memory and reset AI context |
| **Summarize** | Executive summary |
| **Extract Financials** | Revenue, EPS, margins, guidance |
| **Key Highlights** | Most material investor takeaways |
| **TRANSCRIPT NLP** | Sentiment, themes, beat/miss prediction |
| **Ask box** | Free-form Q&A against the document |

> 💡 A loaded document becomes live context for Probability tab narrations and AI Chat responses.

---

## 📈 PROBABILITY Tab

Six institutional-grade quantitative models for risk and return framing.

| Model | Output |
|-------|--------|
| **Monte Carlo Simulation** | Distribution of price outcomes using GBM |
| **Bayesian Inference** | Updated directional probability |
| **VaR / CVaR** | Downside at confidence level + tail-risk severity |
| **Scenario Analysis** | Bull / Base / Bear price targets with probability weights |
| **Kelly Criterion** | Optimal position size fraction |
| **Black-Scholes Probability** | Chance of exceeding a defined target price |

**Configuration:** Simulations · Horizon (days) · Confidence level

---

## 📰 NEWS Tab

Three-panel market intelligence workspace.

- **Global News** — broad RSS feed from multiple financial sources
- **Ticker News** — symbol-specific headline pull
- **Auto-Score Feed** — per-headline AI scoring: BULLISH/BEARISH/NEUTRAL + score (–10 to +10) + summary
- **Bulk Sentiment** — portfolio-level AI synthesis: composite score, themes, sector signals, key risk/opportunity
- **Live Ticker Bar** — scrolling headline strip with orange LIVE badge at the bottom of the window

---

## 🔍 SCREENER Tab

Natural-language AI stock screener returning ranked institutional-style candidates.

**Usage:** Type a query or click a preset → press Enter or click SCREEN

**Preset screens:**

| Preset | Criteria |
|--------|----------|
| Value | Low P/E, positive FCF, dividend-paying |
| Growth | Revenue growth 25%+, expanding margins, large TAM |
| Quality | ROE 20%+, consistent earnings, reasonable valuation |
| Momentum | Near 52W high, strong relative strength vs SPY |
| Dividends | 3%+ yield, 10+ years consecutive growth |
| Deep Value | Near 52W low, P/B <1, asset-rich, hidden catalyst |
| AI Plays | Pure-play AI/ML with strong revenue growth and moats |

**Output format:** Criteria summary → Top 10 picks (with thesis, risk, conviction) → Sector overview → Portfolio construction notes

---

## 🤖 AI CHAT Panel

Always-visible analyst on the right side of the window.

| Button | Delivers |
|--------|----------|
| **Full** | Complete multi-factor analysis |
| **Bull** | Bull case, catalysts, upside targets |
| **Bear** | Bear case, downside risks |
| **Risk** | Volatility, invalidation, tail-risk framing |
| **Valuation** | P/E, EV/EBITDA, DCF fair value context |
| **Chart AI** | Deep enriched mode — all indicators + options + SPY + insider + confluence in one bundle |

---

## 📡 AI TRADER Tab

Autonomous strategy monitoring workspace with live dashboard metrics.

| Metric | Meaning |
|--------|---------|
| Available Cash | Undeployed capital |
| Open Position | Active share count / position state |
| Live Price | Real-time market price |
| Total Equity | Cash + marked-to-market position |
| Net Return | Performance vs starting capital |

> ⚠️ Requires LM Studio to be connected before starting. Monitor terminal logs throughout.

---

## ⚙️ SETTINGS Tab

| Field | Purpose |
|-------|---------|
| Server URL | LM Studio API endpoint (default: `http://localhost:1234`) |
| Active Model | Currently loaded model name |
| Test Connection | Verify reachability |
| Apply Settings | Save connection parameters |
| Dependency Status | Live ✅/❌ for all optional packages with pip install commands |

---

## 📦 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Windows 11 / macOS 14+ |
| Python | 3.9+ | 3.10 or 3.11 |
| RAM | 8 GB | 16–32 GB |
| CPU | 4 cores | 8+ cores |
| GPU (AI) | Optional | NVIDIA RTX 3060+ for larger models |
| Internet | Required for data/news | Stable broadband |

---

## 🚀 EXE Distribution Notes

The `.exe` bundles the Python runtime and all core libraries.

- **LM Studio must be installed and running separately** — it is not bundled in the EXE
- Change the LM Studio URL under **SETTINGS** if you use a non-default port
- Optional packages (HMM, XGBoost, statsmodels) — if not bundled at compile time, those features will be reported as unavailable in the SETTINGS dependency panel
- On first launch, allow firewall access for market data and news feeds

---

## 🔧 LM Studio Troubleshooting

| Problem | Fix |
|---------|-----|
| `LM Studio OFFLINE` in status bar | Open LM Studio → Local Server → Start Server |
| `Failed — is LM Studio running?` | Ensure a model is loaded before starting |
| Very slow responses | Use a smaller 7B model or enable GPU layers in LM Studio |
| Port mismatch | Update Server URL in APEX SETTINGS to match your LM Studio port |
| No model name shown | Load a model in LM Studio Local Server tab first |

---

## 📋 Feature Matrix

| Feature | Requires |
|---------|---------|
| Charting + all indicators | `yfinance matplotlib pandas numpy` |
| AI analysis, POD, Chat | LM Studio running locally |
| PDF loading | `pdfplumber` |
| Live news feed | `feedparser` |
| Regime detection (HMM) | `hmmlearn` |
| ML prediction (XGBoost) | `xgboost scikit-learn` |
| Cointegration analysis | `statsmodels` |
| Options Max Pain / PCR | Active options chain (not all tickers) |

---

## 📄 License

Private / Proprietary. All rights reserved - SchwartZ.

---

<div align="center">

Built with Python · PyQt6 · Matplotlib · yfinance · LM Studio

**Local. Private. Professional.**

</div>
