#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   APEX AI TERMINAL  v1.0  ·  AI-Powered Bloomberg-Style Financial Platform     ║
║   Local AI via LM Studio  ·  Real-time Data via yfinance  ·  PyQt6         ║
╚══════════════════════════════════════════════════════════════════════════════╝

FEATURES:
  ● Bloomberg-style dark UI with professional typography
  ● Interactive candlestick charts: MA20/50/200, Bollinger Bands, RSI, MACD, Volume
  ● LM Studio AI integration with streaming responses
  ● Document loader: PDF, TXT, CSV, JSON with AI analysis
  ● Live watchlist with real-time price updates
  ● RSS market news feed with AI sentiment analysis
  ● AI stock screener with natural language queries
  ● Structured equity research report generator
  ● DCF model & comparable company analysis
  ● Scrolling news ticker bar

REQUIREMENTS:
  pip install PyQt6 yfinance matplotlib pandas numpy requests pdfplumber feedparser
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import sys, os, json, threading, time, datetime, re, traceback
import requests
import pandas as pd
import numpy as np

# Optional dependencies
try:    import yfinance as yf;   HAS_YF   = True
except: HAS_YF   = False; print("⚠ Install yfinance: pip install yfinance")

try:    import pdfplumber;       HAS_PDF  = True
except: HAS_PDF  = False

try:    import feedparser;       HAS_FEED = True
except: HAS_FEED = False

# HMM Regime Detection (optional — install with: pip install hmmlearn)
try:
    from hmmlearn import hmm as _hmm_lib
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    
# XGBoost Predictive Engine & Gaussian Processes
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared, WhiteKernel
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Statsmodels for Cointegration (pip install statsmodels)
try:
    from statsmodels.tsa.stattools import coint
    HAS_SM = True
except ImportError:
    HAS_SM = False

# VMD for Advanced Signal Decomposition
try:
    from vmdpy import VMD
    HAS_VMD = True
except ImportError:
    HAS_VMD = False    

# Force QT_API before matplotlib loads (must be before any matplotlib import)
import os
os.environ["QT_API"] = "pyqt6"

import matplotlib

# ── Auto-detect available Qt binding and set matplotlib backend accordingly ──
def _setup_qt():
    """Try PyQt6 → PyQt5 → PySide6 → PySide2 in order."""
    bindings = [
        ('PyQt6',   'qtagg'),
        ('PyQt5',   'qt5agg'),
        ('PySide6', 'qtagg'),
        ('PySide2', 'qt5agg'),
    ]
    for pkg, backend in bindings:
        try:
            __import__(pkg)
            matplotlib.use(backend)
            return pkg
        except ImportError:
            continue
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  ERROR: No Qt binding found.  Install one with:          ║")
    print("║   pip install PyQt6                                      ║")
    print("║   pip install PyQt5                                      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    sys.exit(1)

_QT_BINDING = _setup_qt()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

# Import FigureCanvas - backend_qtagg works for all Qt versions in matplotlib 3.5+
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Import Qt classes from whichever binding is available
if _QT_BINDING == 'PyQt6':
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    # PyQt6 enums are already fully namespaced throughout the file — no shim needed
elif _QT_BINDING == 'PyQt5':
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
elif _QT_BINDING == 'PySide6':
    from PySide6.QtWidgets import *
    from PySide6.QtCore import *
    from PySide6.QtGui import *
    # PySide6 uses Signal instead of pyqtSignal
    pyqtSignal = Signal
else:  # PySide2
    from PySide2.QtWidgets import *
    from PySide2.QtCore import *
    from PySide2.QtGui import *
    pyqtSignal = Signal

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
APP      = "APEX AI TERMINAL"
VER      = "v1.0"
LM_URL   = "http://localhost:1234"

WATCHLIST_DEFAULT = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA",
    "NFLX","SPY","QQQ","GLD","BTC-USD","TSM","ASML"
]

TIMEFRAMES = {
    "1D":  ("1d",  "5m"),
    "5D":  ("5d",  "15m"),
    "1M":  ("1mo", "1h"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
    "2Y":  ("2y",  "1wk"),
    "5Y":  ("5y",  "1mo"),
}

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]

# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
C = {
    # Backgrounds
    'bg':       '#080808',
    'bg2':      '#0f0f0f',
    'bg3':      '#161616',
    'bg4':      '#1c1c1c',
    # Borders
    'border':   '#242424',
    'border2':  '#333333',
    # Accents
    'accent':   '#ff6600',
    'accent2':  '#ffaa33',
    'accent3':  '#ff8833',
    # Signal colours
    'green':    '#00e676',
    'red':      '#ff3d57',
    'blue':     '#40a9ff',
    'purple':   '#b388ff',
    'yellow':   '#ffd740',
    'cyan':     '#18ffff',
    # Text
    'text':     '#e0e0e0',
    'text2':    '#888888',
    'text3':    '#444444',
    # Chart specific
    'cand_up':  '#00e676',
    'cand_dn':  '#ff3d57',
    'ma20':     '#ff9800',
    'ma50':     '#2196f3',
    'ma200':    '#9c27b0',
    'bb':       '#546e7a',
}

# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB STYLE
# ═══════════════════════════════════════════════════════════════════════════════
MPLRC = {
    'figure.facecolor':  C['bg'],
    'axes.facecolor':    C['bg'],
    'axes.edgecolor':    C['border'],
    'axes.labelcolor':   C['text2'],
    'axes.grid':         True,
    'grid.color':        '#1a1a1a',
    'grid.linestyle':    '-',
    'grid.linewidth':    0.4,
    'grid.alpha':        0.9,
    'xtick.color':       C['text2'],
    'ytick.color':       C['text2'],
    'text.color':        C['text'],
    'font.size':         9,
    'font.family':       'monospace',
    'figure.autolayout': False,
}
plt.rcParams.update(MPLRC)


# ═══════════════════════════════════════════════════════════════════════════════
# STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════
def build_qss():
    return f"""
/* ── Global ──────────────────────────────────────────────────────── */
* {{
    font-family: Consolas, 'Courier New', monospace;
    font-size: 12px;
    color: {C['text']};
}}
QMainWindow, QDialog {{ background: {C['bg']}; }}
QWidget {{ background: {C['bg']}; }}

/* ── Tabs ─────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {C['border']};
    background: {C['bg2']};
    margin-top: -1px;
}}
QTabBar {{ background: {C['bg']}; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: {C['bg3']};
    color: {C['text2']};
    padding: 8px 18px;
    border: 1px solid {C['border']};
    border-bottom: none;
    font-weight: bold;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    min-width: 80px;
}}
QTabBar::tab:selected {{
    background: {C['bg2']};
    color: {C['accent']};
    border-top: 2px solid {C['accent']};
}}
QTabBar::tab:hover:!selected {{ color: {C['text']}; background: {C['bg4']}; }}

/* ── Input fields ─────────────────────────────────────────────────── */
QLineEdit {{
    background: {C['bg3']};
    border: 1px solid {C['border']};
    border-radius: 0px;
    padding: 5px 10px;
    color: {C['text']};
}}
QLineEdit:focus {{ border: 1px solid {C['accent']}; }}

/* ── Buttons ──────────────────────────────────────────────────────── */
QPushButton {{
    background: {C['bg3']};
    border: 1px solid {C['border']};
    color: {C['text']};
    padding: 5px 14px;
    border-radius: 0px;
    font-weight: bold;
    font-size: 11px;
    letter-spacing: 0.3px;
}}
QPushButton:hover {{
    background: {C['accent']};
    color: white;
    border-color: {C['accent']};
}}
QPushButton:pressed {{ background: #cc4400; }}
QPushButton[accent="true"] {{
    background: {C['accent']};
    color: white;
    border-color: {C['accent']};
}}
QPushButton[accent="true"]:hover {{ background: {C['accent2']}; }}

/* ── Text Areas ───────────────────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background: {C['bg2']};
    border: 1px solid {C['border']};
    color: {C['text']};
    padding: 4px;
}}

/* ── Tables ───────────────────────────────────────────────────────── */
QTableWidget {{
    background: {C['bg2']};
    border: 1px solid {C['border']};
    gridline-color: {C['bg3']};
    alternate-background-color: {C['bg3']};
}}
QTableWidget::item {{ padding: 4px 8px; border: none; }}
QTableWidget::item:selected {{ background: #1a2535; color: white; }}
QHeaderView::section {{
    background: {C['bg3']};
    color: {C['accent']};
    padding: 5px 8px;
    border: none;
    border-bottom: 1px solid {C['border']};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* ── Scrollbars ───────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {C['bg3']};
    width: 6px;
    border: none;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C['border2']};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {C['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {C['bg3']};
    height: 6px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {C['border2']};
    border-radius: 3px;
}}
QScrollBar::handle:horizontal:hover {{ background: {C['accent']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Combo ────────────────────────────────────────────────────────── */
QComboBox {{
    background: {C['bg3']};
    border: 1px solid {C['border']};
    padding: 5px 10px;
    color: {C['text']};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {C['bg3']};
    border: 1px solid {C['accent']};
    selection-background-color: {C['accent']};
    color: {C['text']};
}}

/* ── Misc ─────────────────────────────────────────────────────────── */
QSplitter::handle {{ background: {C['border']}; }}
QLabel {{ background: transparent; }}
QStatusBar {{
    background: {C['bg3']};
    border-top: 1px solid {C['border']};
    color: {C['text2']};
    font-size: 10px;
}}
QGroupBox {{
    border: 1px solid {C['border']};
    margin-top: 14px;
    padding-top: 10px;
    color: {C['accent']};
    font-weight: bold;
    font-size: 10px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QCheckBox {{ color: {C['text2']}; spacing: 6px; font-size: 11px; }}
QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {C['border']};
    background: {C['bg3']};
}}
QCheckBox::indicator:checked {{
    background: {C['accent']};
    border-color: {C['accent']};
}}
QMenu {{
    background: {C['bg3']};
    border: 1px solid {C['border']};
    color: {C['text']};
}}
QMenu::item:selected {{ background: {C['accent']}; color: white; }}
QToolTip {{
    background: {C['bg3']};
    border: 1px solid {C['accent']};
    color: {C['text']};
    padding: 4px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SIGNAL BUS
# ═══════════════════════════════════════════════════════════════════════════════
class SignalBus(QObject):
    """Central event hub — keeps widgets decoupled."""
    ticker_changed  = pyqtSignal(str)          # new symbol selected
    tf_changed      = pyqtSignal(str)          # timeframe button pressed
    data_ready      = pyqtSignal(str, object)  # (ticker, DataFrame)
    ai_chunk        = pyqtSignal(str)          # streaming token
    ai_done         = pyqtSignal()
    price_update    = pyqtSignal(str, float, float)  # ticker, price, pct
    status_msg      = pyqtSignal(str)
    news_ready      = pyqtSignal(list)
    lm_status       = pyqtSignal(bool, str)    # connected, model_name
    doc_loaded      = pyqtSignal(str, str)     # filename, text

BUS = SignalBus()


# ═══════════════════════════════════════════════════════════════════════════════
# LM STUDIO CLIENT
# ═══════════════════════════════════════════════════════════════════════════════
class LMClient:
    """OpenAI-compatible interface to LM Studio local server."""

    def __init__(self, base=LM_URL):
        self.base      = base
        self.model     = None
        self.connected = False

    def check(self):
        try:
            r = requests.get(f"{self.base}/v1/models", timeout=4)
            if r.status_code == 200:
                data = r.json().get('data', [])
                self.model     = data[0]['id'] if data else 'local-model'
                self.connected = True
                BUS.lm_status.emit(True, self.model)
                return True
        except Exception:
            pass
        self.connected = False
        BUS.lm_status.emit(False, '')
        return False

    def stream(self, messages, system=None, temperature=0.3, max_tokens=2500):
        """Generator — yields text from LM Studio robustly."""
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full += messages
        try:
            r = requests.post(
                f"{self.base}/v1/chat/completions",
                json={
                    "model":       self.model or "local-model",
                    "messages":    full,
                    "temperature": temperature,
                    "max_tokens":  max_tokens,
                    "stream":      True,
                },
                stream=True, timeout=180,
            )
            r.encoding = 'utf-8' 
            
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except Exception:
                    continue
                # Surface any API-level errors
                if "error" in data:
                    err = data["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    yield f"\n\n[LM Studio error: {msg}]"
                    return
                choices = data.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                # Try delta.content first (streaming format)
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    content_val = delta.get("content", "")
                    if isinstance(content_val, str) and content_val:
                        yield content_val
                        continue
                    if isinstance(content_val, list):
                        text = "".join(
                            item.get("text", "") for item in content_val
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                        if text:
                            yield text
                            continue
                # Fallback: message.content (non-streaming or final chunk)
                message = choice.get("message", {})
                if isinstance(message, dict):
                    content_val = message.get("content", "")
                    if isinstance(content_val, str) and content_val:
                        yield content_val
                        continue
                    if isinstance(content_val, list):
                        text = "".join(
                            item.get("text", "") for item in content_val
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                        if text:
                            yield text
        except Exception as exc:
            yield f"\n\n[LM Studio error: {exc}]"

    def complete(self, prompt, system=None, temperature=0.3, max_tokens=1024):
        return ''.join(self.stream(
            [{"role": "user", "content": prompt}],
            system, temperature, max_tokens,
        ))

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT ORCHESTRATOR — Pod Architecture (3 Specialists + 1 Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════
class AgentOrchestrator:
    """
    Runs 4 sequential LM calls acting as specialist agents + a final orchestrator.
    Phase 1 (parallel in logic, sequential on single GPU):
        - The Chartist       → Technical analysis agent
        - The Fundamentalist → Value / DCF agent
        - The News Sentry    → Sentiment / risk agent
    Phase 2:
        - The Portfolio Manager → Synthesizes all 3 into a Confidence Score + Trade Thesis
    """

    CHARTIST_SYSTEM = (
        "You are a senior technical analyst with 20 years of experience at a hedge fund. "
        "You speak in precise, concise terms. You focus ONLY on price action, momentum, "
        "and chart structure. You never discuss fundamentals or news."
    )

    FUNDAMENTALIST_SYSTEM = (
        "You are a CFA-certified equity research analyst specialising in valuation. "
        "You focus ONLY on fundamental metrics: P/E, EV/EBITDA, DCF intrinsic value, "
        "revenue growth, and balance sheet health. You never discuss charts or news."
    )

    SENTRY_SYSTEM = (
        "You are a risk intelligence analyst. Your job is to cut through noise and identify "
        "the single most material risk factor from news and options flow data. "
        "You are terse, direct, and contrarian. You never discuss charts or valuations."
    )

    ORCHESTRATOR_SYSTEM = (
        "You are a Portfolio Manager at a top-tier fund. You receive briefings from three "
        "specialist analysts and synthesise them into a final investment decision. "
        "You weigh technical momentum, fundamental value, and sentiment risk equally. "
        "You always output a structured verdict with a Confidence Score (0-100)."
    )

    def run(self, ticker, tech_data, fund_data, sent_data,
            hmm_result=None, gate_result=None, kelly=None, xgb_result=None, # <--- ADDED xgb_result
            on_phase=None, on_chunk=None, on_done=None):
        """
        Execute the full 4-call pipeline.
        on_phase(phase_name: str)  — called when each phase starts
        on_chunk(token: str)       — called for every streamed token (orchestrator only)
        on_done(report: str)       — called with the final synthesised text
        """
        results = {}

        # ── PHASE 1A: The Chartist ───────────────────────────────────────────
        if on_phase: on_phase("🔬 CHARTIST AGENT — Technical Analysis")
        chartist_prompt = f"""Analyse the following technical data for {ticker} and give your verdict.

TECHNICAL DATA:
{tech_data}

Answer these questions concisely:
1. Is the trend UP, DOWN, or SIDEWAYS?
2. Is price overextended (RSI/BB position)?
3. Is momentum accelerating or decelerating (MACD/RVI)?
4. What is the key support and resistance level?
5. Your technical verdict: BULLISH / BEARISH / NEUTRAL with one-sentence reasoning.

Be concise. Max 200 words."""

        try:
            results['chartist'] = ''.join(LM.stream(
                [{"role": "user", "content": chartist_prompt}],
                system=self.CHARTIST_SYSTEM,
                temperature=0.2,
                max_tokens=400,
            ))
        except Exception as e:
            results['chartist'] = f"Technical agent error: {e}"

        # ── PHASE 1B: The Fundamentalist ─────────────────────────────────────
        if on_phase: on_phase("📊 FUNDAMENTALIST AGENT — Value Analysis")
        fund_prompt = f"""Analyse the following fundamental data for {ticker} and give your verdict.

FUNDAMENTAL DATA:
{fund_data}

Answer these questions concisely:
1. Is the stock cheap, fairly valued, or expensive vs peers?
2. Is the DCF intrinsic value above or below current price?
3. Are revenue growth and margins expanding or contracting?
4. Is the balance sheet a strength or a risk?
5. Your fundamental verdict: VALUE / FAIRLY VALUED / OVERVALUED with one-sentence reasoning.

Be concise. Max 200 words."""

        try:
            results['fundamentalist'] = ''.join(LM.stream(
                [{"role": "user", "content": fund_prompt}],
                system=self.FUNDAMENTALIST_SYSTEM,
                temperature=0.2,
                max_tokens=400,
            ))
        except Exception as e:
            results['fundamentalist'] = f"Fundamental agent error: {e}"

        # ── PHASE 1C: The News Sentry ─────────────────────────────────────────
        if on_phase: on_phase("🛰 NEWS SENTRY AGENT — Sentiment & Risk")
        sentry_prompt = f"""Filter the following sentiment and options flow data for {ticker}.

SENTIMENT & RISK DATA:
{sent_data}

Answer these questions concisely:
1. What is the ONE most material risk factor? (ignore noise)
2. Is options flow signalling institutional accumulation or distribution?
3. Is market sentiment aligned with or diverging from price action?
4. Is there an earnings / macro event that could invalidate technical signals?
5. Your risk verdict: HIGH RISK / MODERATE RISK / LOW RISK with one-sentence reasoning.

Be concise. Max 200 words."""

        try:
            results['sentry'] = ''.join(LM.stream(
                [{"role": "user", "content": sentry_prompt}],
                system=self.SENTRY_SYSTEM,
                temperature=0.2,
                max_tokens=400,
            ))
        except Exception as e:
            results['sentry'] = f"Sentry agent error: {e}"

                # ── PHASE 1D: The Macro Sentinel (5th pod) ───────────────────────
        if on_phase: on_phase("🌐 MACRO SENTINEL — Global Environment Scan")
        macro_env = MACRO.fetch_environment()
        sentinel_prompt = f"""You are the Macro Sentinel. Analyse the global macro environment
and determine whether it SUPPORTS or UNDERMINES a trade in {ticker}.

MACRO DATA:
{macro_env}

TICKER CONTEXT:
{tech_data[:300]}

Answer concisely:
1. Is the current macro regime RISK-ON or RISK-OFF?
2. Does VIX level validate or invalidate technical signals?
3. Is sector rotation flowing toward or away from this ticker's sector?
4. Is the dollar/bond move a tailwind or headwind?
5. MACRO VERDICT: SUPPORTIVE / NEUTRAL / HEADWIND with one-sentence reasoning.

Max 200 words."""

        try:
            results['sentinel'] = ''.join(LM.stream(
                [{"role": "user", "content": sentinel_prompt}],
                system=(
                    "You are a macro strategist at a global macro fund. You focus exclusively "
                    "on top-down factors: central banks, dollar, bonds, volatility, and sector "
                    "rotation. You never discuss individual stock technicals or fundamentals."
                ),
                temperature=0.2,
                max_tokens=400,
            ))
        except Exception as e:
            results['sentinel'] = f"Macro sentinel error: {e}"    

        # ── PHASE 2: The Orchestrator ─────────────────────────────────────────
        if on_phase: on_phase("🧠 ORCHESTRATOR — Synthesising Consensus")
        orch_prompt = f"""You are a Portfolio Manager receiving four specialist briefings on {ticker}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ticker} | 5-AGENT CONSENSUS REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[CHARTIST — Technical]:
{results.get('chartist', 'N/A')}

[FUNDAMENTALIST — Value]:
{results.get('fundamentalist', 'N/A')}

[NEWS SENTRY — Risk]:
{results.get('sentry', 'N/A')}

[MACRO SENTINEL — Global Environment]:
{results.get('sentinel', 'N/A')}

HMM REGIME: {hmm_result.get('regime','N/A')} (confidence: {hmm_result.get('confidence',0)*100:.0f}%)
REGIME GATE: {gate_result.get('regime_used','N/A')} | Suppressed signals: {gate_result.get('suppressed',[])}
KELLY SIZING: Full={kelly.get('kelly',0)}% | Half={kelly.get('half_kelly',0)}%
XGBOOST ML PREDICTION: {xgb_result.get('interpretation', 'N/A') if xgb_result else 'N/A'}
ML TOP DRIVERS: {', '.join([f"{k} ({v*100:.1f}%)" for k, v in xgb_result.get('top_features', [])]) if xgb_result and 'top_features' in xgb_result else 'N/A'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

As Portfolio Manager, synthesise into a final verdict:

[SIGNAL ALIGNMENT]
Do all four agents agree? Where is consensus vs conflict?

[XGBOOST FORECAST]
Restate the ML probability of a positive return and list the exact features driving it. Does this align with the human agents?

[REGIME CONTEXT]
Given the HMM regime and macro environment, are technical signals valid or gated?

[RISK-ADJUSTED TRADE THESIS]
3-4 sentences: what is the trade, why, and what invalidates it?

[CONFIDENCE SCORE]
Score: XX/100
Breakdown: Technical XX/100 | Fundamental XX/100 | Sentiment XX/100 | Macro XX/100 | Quantitative ML XX/100

[POSITION SIZING]
Based on Kelly Criterion: recommend full/half/quarter Kelly and exact % of portfolio.

[RECOMMENDED ACTION]
STRONG BUY / BUY / HOLD / SELL / STRONG SELL
*Note: If there is an ACTIVE PTT Algorithmic Trade running, explicitly state trade management advice (e.g., where to trail the stop loss or take profit).*
Entry zone: $XX–$XX | Stop loss: $XX | Target (30d): $XX"""

        full_orch = []
        try:
            for tok in LM.stream(
                [{"role": "user", "content": orch_prompt}],
                system=self.ORCHESTRATOR_SYSTEM,
                temperature=0.25,
                max_tokens=800,
            ):
                full_orch.append(tok)
                if on_chunk: on_chunk(tok)
        except Exception as e:
            err = f"\n\n[Orchestrator error: {e}]"
            full_orch.append(err)
            if on_chunk: on_chunk(err)

        results['orchestrator'] = ''.join(full_orch)
        if on_done: on_done(results)
        return results


class PodThread(QThread):
    """Runs AgentOrchestrator in a background thread with Qt signals."""
    phase   = pyqtSignal(str)       
    chunk   = pyqtSignal(str)       
    done    = pyqtSignal(dict)      
    error   = pyqtSignal(str)

    def __init__(self, ticker, tech_data, fund_data, sent_data,
                 hmm_result=None, gate_result=None, kelly=None, xgb_result=None): # <-- ADDED
        super().__init__()
        self.ticker     = ticker
        self.tech_data  = tech_data
        self.fund_data  = fund_data
        self.sent_data  = sent_data
        self.hmm_result = hmm_result or {}
        self.gate_result= gate_result or {}
        self.kelly      = kelly or {}
        self.xgb_result = xgb_result or {} # <-- ADDED

    def run(self):
        try:
            AgentOrchestrator().run(
                self.ticker, self.tech_data, self.fund_data, self.sent_data,
                hmm_result=self.hmm_result,
                gate_result=self.gate_result,
                kelly=self.kelly,
                xgb_result=self.xgb_result, # <-- ADDED
                on_phase=self.phase.emit,
                on_chunk=self.chunk.emit,
                on_done=self.done.emit,
            )
        except Exception as exc:
            self.error.emit(str(exc))


ORCHESTRATOR = AgentOrchestrator()

LM = LMClient()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class DataEngine:
    """yfinance wrapper with 5-minute result cache."""
    def __init__(self):
        self._cache = {}

    def fetch(self, ticker, period='5d', interval='5m'):
        key = f"{ticker}|{period}|{interval}"
        if key in self._cache:
            ts, df = self._cache[key]
            if time.time() - ts < 300:
                return df
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            self._cache[key] = (time.time(), df)
            return df
        except Exception as exc:
            print(f"DataEngine.fetch error [{ticker}]: {exc}")
            return None

    def info(self, ticker):
        try:
            return yf.Ticker(ticker).info
        except Exception:
            return {}

    def fast_quote(self, ticker):
        """Ultra-resilient price fetcher that bypasses Yahoo's Info blocks."""
        try:
            import yfinance as yf
            # Use history (chart data) because Yahoo does not block it.
            # 5 days ensures we get at least 2 bars even over long holiday weekends.
            df = yf.Ticker(ticker).history(period='5d', interval='1d')
            if df is not None and not df.empty and len(df) >= 1:
                curr = float(df['Close'].iloc[-1])
                if len(df) >= 2:
                    prev = float(df['Close'].iloc[-2])
                    pct = ((curr - prev) / prev * 100) if prev > 0 else 0.0
                    return curr, pct
                return curr, 0.0
        except Exception:
            pass
        return None, None

    def add_indicators(self, df):
        d = df.copy()
        c = d['Close']

        # Moving averages
        d['MA20']  = c.rolling(20).mean()
        d['MA50']  = c.rolling(50).mean()
        d['MA200'] = c.rolling(200).mean()

        # ── KALMAN FILTER (STATE-SPACE MODEL) ─────────────────────────
        try:
            prices = c.values
            kf_mean = np.zeros(len(prices))
            kf_cov = np.zeros(len(prices))
            
            # Initial state guess
            kf_mean[0] = prices[0]
            kf_cov[0] = 1.0
            
            # Tuning matrices: Q = process noise (trend flexibility), R = measurement noise (price chop)
            Q = 1e-4 
            R = 0.01 * np.var(prices) if np.var(prices) > 0 else 0.01
            
            for i in range(1, len(prices)):
                # Prediction step
                prior_mean = kf_mean[i-1]
                prior_cov = kf_cov[i-1] + Q
                
                # Update step (Calculate Kalman Gain)
                K = prior_cov / (prior_cov + R)
                kf_mean[i] = prior_mean + K * (prices[i] - prior_mean)
                kf_cov[i] = (1 - K) * prior_cov
                
            d['Kalman'] = kf_mean
            d['Kalman_Diff'] = (d['Close'] - d['Kalman']) / d['Kalman'] * 100
        except Exception as e:
            print(f"Kalman Error: {e}")
            d['Kalman'] = np.nan
            d['Kalman_Diff'] = 0.0

        # Bollinger Bands (20, 2σ)
        std = c.rolling(20).std()
        d['BB_U'] = d['MA20'] + 2 * std
        d['BB_L'] = d['MA20'] - 2 * std

        # RSI (14)
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        d['RSI'] = 100 - 100 / (1 + rs)

        # MACD (12, 26, 9)
        ema12       = c.ewm(span=12, adjust=False).mean()
        ema26       = c.ewm(span=26, adjust=False).mean()
        d['MACD']   = ema12 - ema26
        d['Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
        d['MHist']  = d['MACD'] - d['Signal']

        # VWAP (cumulative)
        try:
            d['VWAP'] = (c * df['Volume']).cumsum() / df['Volume'].cumsum()
        except Exception:
            d['VWAP'] = np.nan
        
        # ATR (14)
        try:
            hl = df['High'] - df['Low']
            hc = (df['High'] - c.shift()).abs()
            lc = (df['Low']  - c.shift()).abs()
            d['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        except Exception:
            d['ATR'] = np.nan  

        # OBV
        try:
            obv = [0]
            closes  = df['Close'].values
            volumes = df['Volume'].values
            for i in range(1, len(df)):
                if   closes[i] > closes[i-1]: obv.append(obv[-1] + volumes[i])
                elif closes[i] < closes[i-1]: obv.append(obv[-1] - volumes[i])
                else:                          obv.append(obv[-1])
            d['OBV'] = obv
        except Exception:
            d['OBV'] = np.nan

        # CMF (20) — Chaikin Money Flow
        try:
            mfm = ((c - df['Low']) - (df['High'] - c)) / (df['High'] - df['Low']).replace(0, np.nan)
            mfv = mfm * df['Volume']
            d['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()
        except Exception:
            d['CMF'] = np.nan

        # RVI (10) — Relative Vigor Index
        try:
            num = (c - df['Open']).rolling(10).mean()
            den = (df['High'] - df['Low']).rolling(10).mean().replace(0, np.nan)
            d['RVI']        = num / den
            d['RVI_Signal'] = d['RVI'].rolling(4).mean()
        except Exception:
            d['RVI']        = np.nan
            d['RVI_Signal'] = np.nan

        # RSI Divergence
        try:
            lb = 20
            d['RSI_Bear_Div'] = (c >= c.rolling(lb).max()*0.99) & (d['RSI'] < d['RSI'].rolling(lb).max()-3)
            d['RSI_Bull_Div'] = (c <= c.rolling(lb).min()*1.01) & (d['RSI'] > d['RSI'].rolling(lb).min()+3)
        except Exception:
            d['RSI_Bear_Div'] = False; d['RSI_Bull_Div'] = False

        # Market Regime
        try:
            bb_w  = (d['BB_U'] - d['BB_L']) / d['MA20'].replace(0, np.nan)
            bb_wp = bb_w.rank(pct=True); atr_p = d['ATR'].rank(pct=True)
            def _regime(bw, ap):
                bw = bw if not np.isnan(bw) else 0.5; ap = ap if not np.isnan(ap) else 0.5
                if bw < 0.20 and ap < 0.20: return 'VOLATILITY SQUEEZE'
                elif bw > 0.75 and ap > 0.75: return 'HIGH EXPANSION'
                elif bw > 0.55: return 'EXPANSION'
                elif bw < 0.35: return 'CONTRACTION'
                else: return 'MEAN REVERSION'
            d['Market_Regime'] = [_regime(float(bb_wp.iloc[i]),float(atr_p.iloc[i])) for i in range(len(d))]
        except Exception:
            d['Market_Regime'] = ['UNKNOWN']*len(d)

            # ── 1H SMART SCALPING INDICATOR ──────────────────────────────────────
        try:
            highs, lows = d['High'].values, d['Low'].values
            closes, opens = d['Close'].values, d['Open'].values
            vols, atr = d['Volume'].values, d['ATR'].fillna(0).values
            
            last_ph, last_pl = highs[0], lows[0]
            ph_arr, pl_arr = np.full(len(d), np.nan), np.full(len(d), np.nan)
            trend = np.zeros(len(d))
            
            cur_trend = 1
            
            for i in range(3, len(d)):
                # 1. Non-repainting pivot detection (confirmed 1 bar later)
                if highs[i-2] > highs[i-1] and highs[i-2] > highs[i-3] and highs[i-2] > highs[i]:
                    last_ph = highs[i-2]
                if lows[i-2] < lows[i-1] and lows[i-2] < lows[i-3] and lows[i-2] < lows[i]:
                    last_pl = lows[i-2]
                    
                ph_arr[i], pl_arr[i] = last_ph, last_pl
                
                # 2. Trend change
                if closes[i] > last_ph: cur_trend = 1
                elif closes[i] < last_pl: cur_trend = -1
                trend[i] = cur_trend
                
            d['Pivot_H'], d['Pivot_L'], d['Scalp_Trend'] = ph_arr, pl_arr, trend
            
            # 3. Strong Candle Filter (Daily version: 
            # Body must be > 40% of range and body > 0.4 * ATR)
            body = np.abs(closes - opens)
            tot_len = highs - lows + 1e-9
            strong_bull = (closes > opens) & (body / tot_len > 0.1) & (body > atr * 0.1)
            strong_bear = (closes < opens) & (body / tot_len > 0.1) & (body > atr * 0.1)
            
            # 4. Breakout Confirmation
            prev_closes = d['Close'].shift(1).values
            break_up = (closes > ph_arr) & (prev_closes <= ph_arr)
            break_dn = (closes < pl_arr) & (prev_closes >= pl_arr)
            
            # 5. Volume Filter
            vol_ma = d['Volume'].rolling(20).mean().values
            vol_ok = vols > vol_ma
            
            # 6. Final UP/DOWN Signals
            d['Scalp_Buy'] = (trend == 1) & strong_bull & break_up & vol_ok
            d['Scalp_Sell'] = (trend == -1) & strong_bear & break_dn & vol_ok
        except Exception as e:
            d['Scalp_Buy'] = False; d['Scalp_Sell'] = False; d['Scalp_Trend'] = 0
            d['Pivot_H'] = np.nan; d['Pivot_L'] = np.nan


            # ── PRO TREND TRADER (PTT) ENGINE ─────────────────────────────────
        try:
            ema9 = d['Close'].ewm(span=9, adjust=False).mean().values
            ema21 = d['Close'].ewm(span=21, adjust=False).mean().values
            macd_hist = d['MHist'].fillna(0).values
            rvol = (d['Volume'] / d['Volume'].rolling(20).mean()).fillna(0).values
            atr = d['ATR'].fillna(0).values
            highs, lows, closes = d['High'].values, d['Low'].values, d['Close'].values

            trades =[]
            curr_trade = None

            for i in range(21, len(d)):
                if curr_trade is None:
                    # ENTRY LOGIC (Tri-Layer Validation)
                    if ema9[i] > ema21[i] and macd_hist[i] > 0 and rvol[i] > 1.0 and closes[i] > ema9[i]:
                        curr_trade = {
                            'dir': 1, 'entry_idx': i, 'entry_p': closes[i], 'atr': atr[i],
                            'sl_p': closes[i] - (atr[i] * 1.5),
                            'tps': [closes[i] + (atr[i] * x) for x in[1, 2, 3, 4]],
                            'tp_hits':[], 'exit_idx': None
                        }
                    elif ema9[i] < ema21[i] and macd_hist[i] < 0 and rvol[i] > 1.0 and closes[i] < ema9[i]:
                        curr_trade = {
                            'dir': -1, 'entry_idx': i, 'entry_p': closes[i], 'atr': atr[i],
                            'sl_p': closes[i] + (atr[i] * 1.5),
                            'tps': [closes[i] - (atr[i] * x) for x in [1, 2, 3, 4]],
                            'tp_hits': [], 'exit_idx': None
                        }
                else:
                    # MANAGEMENT LOGIC
                    tdir = curr_trade['dir']
                    tps = curr_trade['tps']
                    tp_idx = len(curr_trade['tp_hits'])

                    # 1. Exit Logic: SL Hit or Momentum Exhaustion (MACD shift)
                    if (tdir == 1 and (lows[i] <= curr_trade['sl_p'] or macd_hist[i] < 0)) or \
                       (tdir == -1 and (highs[i] >= curr_trade['sl_p'] or macd_hist[i] > 0)):
                        curr_trade['exit_idx'] = i
                        trades.append(curr_trade)
                        curr_trade = None
                        continue

                    # 2. Sequential TP Scaling Logic
                    if tp_idx < len(tps):
                        if (tdir == 1 and highs[i] >= tps[tp_idx]) or (tdir == -1 and lows[i] <= tps[tp_idx]):
                            curr_trade['tp_hits'].append({'idx': i, 'p': tps[tp_idx], 'label': f'TP{tp_idx+1} HIT'})
                            # Dynamic Trailing SL
                            curr_trade['sl_p'] = curr_trade['entry_p'] if tp_idx == 0 else tps[tp_idx-1]

            # Append the active trade at the very end so it renders live
            if curr_trade is not None:
                curr_trade['exit_idx'] = len(d) - 1
                trades.append(curr_trade)

            d.attrs['ptt_trades'] = trades # Save securely to dataframe attributes
            d['EMA9'] = ema9
            d['EMA21'] = ema21
        except Exception as e:
            print(f"PTT Engine Error: {e}")

                # ── VWAP Deviation Bands ─────────────────────────────────────────
        try:
            typical = (df['High'] + df['Low'] + df['Close']) / 3
            cum_vol  = df['Volume'].cumsum()
            cum_tpv  = (typical * df['Volume']).cumsum()
            d['VWAP_Cum'] = cum_tpv / cum_vol.replace(0, np.nan)

            # Rolling 20-bar VWAP std for deviation bands
            roll_mean = typical.rolling(20).mean()
            roll_std  = typical.rolling(20).std()
            d['VWAP_Upper1'] = roll_mean + 1 * roll_std
            d['VWAP_Lower1'] = roll_mean - 1 * roll_std
            d['VWAP_Upper2'] = roll_mean + 2 * roll_std
            d['VWAP_Lower2'] = roll_mean - 2 * roll_std

            # VWAP position signal
            price_now = float(d['Close'].iloc[-1])
            vwap_now  = float(d['VWAP_Cum'].iloc[-1])
            vwap_u2   = float(d['VWAP_Upper2'].iloc[-1])
            vwap_l2   = float(d['VWAP_Lower2'].iloc[-1])
            vwap_dev  = (price_now - vwap_now) / vwap_now * 100 if vwap_now else 0

            if   price_now > vwap_u2:  vwap_sig = 'EXTREMELY EXTENDED ABOVE VWAP'
            elif price_now > d['VWAP_Upper1'].iloc[-1]: vwap_sig = 'EXTENDED ABOVE VWAP'
            elif price_now > vwap_now: vwap_sig = 'ABOVE VWAP (buyers in control)'
            elif price_now < vwap_l2:  vwap_sig = 'EXTREMELY EXTENDED BELOW VWAP'
            elif price_now < d['VWAP_Lower1'].iloc[-1]: vwap_sig = 'EXTENDED BELOW VWAP'
            else:                      vwap_sig = 'BELOW VWAP (sellers in control)'

            d['VWAP_Signal'] = vwap_sig
            d['VWAP_Dev_Pct'] = vwap_dev
        except Exception:
            d['VWAP_Cum'] = np.nan
            d['VWAP_Signal'] = 'N/A'
            d['VWAP_Dev_Pct'] = 0.0

        # ── Volume Profile (price level with highest traded volume) ───────
        try:
            bins    = 20
            lo, hi  = float(df['Close'].min()), float(df['Close'].max())
            if hi > lo:
                edges   = np.linspace(lo, hi, bins + 1)
                vol_bins = np.zeros(bins)
                for price_val, vol_val in zip(df['Close'], df['Volume']):
                    idx = min(int((float(price_val) - lo) / (hi - lo) * bins), bins - 1)
                    vol_bins[idx] += float(vol_val)
                poc_idx   = int(np.argmax(vol_bins))
                poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2
                # Value Area: 70% of total volume around POC
                total_vol = vol_bins.sum()
                va_vol    = 0.0
                va_lo_idx = va_hi_idx = poc_idx
                while va_vol < total_vol * 0.70:
                    expand_hi = va_hi_idx + 1 < bins
                    expand_lo = va_lo_idx - 1 >= 0
                    if expand_hi and (not expand_lo or vol_bins[va_hi_idx + 1] >= vol_bins[va_lo_idx - 1]):
                        va_hi_idx += 1; va_vol += vol_bins[va_hi_idx]
                    elif expand_lo:
                        va_lo_idx -= 1; va_vol += vol_bins[va_lo_idx]
                    else:
                        break
                d['VP_POC']    = poc_price
                d['VP_VAH']    = (edges[va_hi_idx] + edges[va_hi_idx + 1]) / 2
                d['VP_VAL']    = (edges[va_lo_idx] + edges[va_lo_idx + 1]) / 2
                d['VP_Signal'] = (
                    'ABOVE VALUE AREA' if float(d['Close'].iloc[-1]) > d['VP_VAH']
                    else 'BELOW VALUE AREA' if float(d['Close'].iloc[-1]) < d['VP_VAL']
                    else 'INSIDE VALUE AREA'
                )
            else:
                d['VP_POC'] = lo; d['VP_VAH'] = hi; d['VP_VAL'] = lo
                d['VP_Signal'] = 'N/A'
        except Exception:
            d['VP_POC'] = np.nan; d['VP_VAH'] = np.nan; d['VP_VAL'] = np.nan
            d['VP_Signal'] = 'N/A'
            
        # ── MATHEMATICAL CANDLESTICK PATTERN SCANNER ─────────────────
        try:
            op, hi = d['Open'].values, d['High'].values
            lo, cl = d['Low'].values, d['Close'].values
            i = len(d) - 1 # We analyze the most recently closed candle
            
            body = abs(cl[i] - op[i])
            rng = hi[i] - lo[i] + 1e-9
            upper_wick = hi[i] - max(op[i], cl[i])
            lower_wick = min(op[i], cl[i]) - lo[i]
            
            pattern = "No clear classic pattern."
            
            # 1. Doji / Spinning Top
            if body <= rng * 0.1:
                pattern = "Doji (Indecision: Wait for confirmation)"
            elif body <= rng * 0.3 and upper_wick > body and lower_wick > body:
                pattern = "Spinning Top (Indecision: Wait for confirmation)"
            # 2. Hammer / Shooting Star
            elif lower_wick >= 2 * body and upper_wick <= 0.1 * rng and cl[i] > op[i]:
                pattern = "Hammer (Bullish reversal: Price may rise)"
            elif upper_wick >= 2 * body and lower_wick <= 0.1 * rng and cl[i] < op[i]:
                pattern = "Shooting Star (Bearish reversal: Price may fall)"
            # 3. Engulfing
            elif cl[i-1] < op[i-1] and cl[i] > op[i] and cl[i] >= op[i-1] and op[i] <= cl[i-1]:
                pattern = "Bullish Engulfing (Bullish reversal: Strong upward shift)"
            elif cl[i-1] > op[i-1] and cl[i] < op[i] and cl[i] <= op[i-1] and op[i] >= cl[i-1]:
                pattern = "Bearish Engulfing (Bearish reversal: Strong downward shift)"
            # 4. Stars (3-Candle Patterns)
            elif i >= 2:
                if cl[i-2] < op[i-2] and abs(cl[i-1]-op[i-1]) <= (hi[i-1]-lo[i-1])*0.1 and cl[i] > op[i] and cl[i] > (op[i-2]+cl[i-2])/2:
                    pattern = "Morning Star (Bullish reversal: Uptrend may start)"
                elif cl[i-2] > op[i-2] and abs(cl[i-1]-op[i-1]) <= (hi[i-1]-lo[i-1])*0.1 and cl[i] < op[i] and cl[i] < (op[i-2]+cl[i-2])/2:
                    pattern = "Evening Star (Bearish reversal: Downward trend may start)"
            
            d.attrs['candle_pattern'] = pattern
        except:
            d.attrs['candle_pattern'] = "N/A"
            
        return d

    def statistical_confidence(self, df):
        """Backtest key signals on the ticker's own history. Returns win-rate dict."""
        results = {}
        try:
            c   = df['Close']
            rsi = df.get('RSI', pd.Series(dtype=float))
            mac = df.get('MACD', pd.Series(dtype=float))
            sig = df.get('Signal', pd.Series(dtype=float))
            obv = df.get('OBV', pd.Series(dtype=float))

            def _win_rate(mask_series, fwd=10):
                mask = mask_series.fillna(False).astype(bool)
                wins = total = 0
                for i in range(len(df) - fwd):
                    if mask.iloc[i]:
                        total += 1
                        if c.iloc[i + fwd] > c.iloc[i]:
                            wins += 1
                return wins, total

            # RSI oversold
            w, t = _win_rate(rsi < 35, fwd=10)
            if t >= 5:
                results['RSI_Oversold'] = (w, t,
                    f"In the last {t} times RSI < 35, price rose {w} times "
                    f"({w/t*100:.0f}%) over the next 10 days.")

            # RSI overbought → bearish
            wins2 = total2 = 0
            for i in range(len(df) - 10):
                v = rsi.iloc[i]
                if not pd.isna(v) and v > 65:
                    total2 += 1
                    if c.iloc[i + 10] < c.iloc[i]:
                        wins2 += 1
            if total2 >= 5:
                results['RSI_Overbought'] = (wins2, total2,
                    f"In the last {total2} times RSI > 65, price fell {wins2} times "
                    f"({wins2/total2*100:.0f}%) over the next 10 days.")

            # MACD bullish crossover
            if len(mac) > 1 and len(sig) > 1:
                cross_up = (mac > sig) & (mac.shift(1) <= sig.shift(1))
                w, t = _win_rate(cross_up, fwd=20)
                if t >= 3:
                    results['MACD_Bullish_Cross'] = (w, t,
                        f"In the last {t} MACD bullish crossovers, price was higher "
                        f"after 20 days {w} times ({w/t*100:.0f}%).")

            # OBV accumulation divergence
            if len(obv) > 20:
                obv_rising  = obv.diff(10) > 0
                price_flat  = c.diff(10).abs() / c < 0.02
                w, t = _win_rate(obv_rising & price_flat, fwd=15)
                if t >= 3:
                    results['OBV_Accumulation'] = (w, t,
                        f"In the last {t} OBV accumulation divergences, price rose "
                        f"{w} times ({w/t*100:.0f}%) over 15 days.")
        except Exception as exc:
            results['error'] = str(exc)
        return results

    # ── PCR history buffer for velocity tracking ──
    _pcr_history: dict = {}

    def put_call_ratio(self, ticker):
        """Fetch options chain, compute PCR + 5-day velocity signal."""
        import collections
        result = {
            'pcr': None, 'put_oi': 0, 'call_oi': 0,
            'interpretation': 'Options data unavailable',
            'smart_money_signal': 'NEUTRAL',
            'pcr_velocity': 'N/A', 'velocity_signal': 'NEUTRAL',
            'velocity_interp': 'Insufficient PCR history (need ≥2 readings)',
        }
        try:
            tk  = yf.Ticker(ticker)
            exp = tk.options
            if not exp:
                return result
            total_put_oi = total_call_oi = 0
            total_put_vol = total_call_vol = 0
            for expiry in exp[:2]:
                chain = tk.option_chain(expiry)
                total_call_oi  += int(chain.calls['openInterest'].fillna(0).sum())
                total_put_oi   += int(chain.puts['openInterest'].fillna(0).sum())
                total_call_vol += int(chain.calls['volume'].fillna(0).sum())
                total_put_vol  += int(chain.puts['volume'].fillna(0).sum())
            if total_call_oi > 0:
                pcr_oi  = total_put_oi / total_call_oi
                pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else None
                result['pcr']        = round(pcr_oi, 3)
                result['pcr_volume'] = round(pcr_vol, 3) if pcr_vol else None
                result['put_oi']  = total_put_oi;  result['call_oi']  = total_call_oi
                result['put_vol'] = total_put_vol; result['call_vol'] = total_call_vol
                if   pcr_oi > 1.3: signal, interp = "BEARISH",         f"PCR={pcr_oi:.2f} — heavy put buying (bearish)"
                elif pcr_oi > 0.9: signal, interp = "NEUTRAL",         f"PCR={pcr_oi:.2f} — balanced positioning"
                elif pcr_oi > 0.6: signal, interp = "NEUTRAL-BULLISH", f"PCR={pcr_oi:.2f} — slight call skew"
                else:              signal, interp = "BULLISH",          f"PCR={pcr_oi:.2f} — calls dominating"
                if pcr_oi > 1.5: interp += " — CONTRARIAN: extreme puts may signal bottom"
                if pcr_oi < 0.5: interp += " — CONTRARIAN: extreme calls may signal top"
                result['interpretation'] = interp; result['smart_money_signal'] = signal
                # ── PCR VELOCITY ──
                hist = DataEngine._pcr_history.setdefault(ticker, collections.deque(maxlen=5))
                hist.append(pcr_oi)
                if len(hist) >= 2:
                    velocity = pcr_oi - hist[0]
                    pct_chg  = velocity / hist[0] * 100 if hist[0] != 0 else 0
                    result['pcr_velocity'] = round(velocity, 4)
                    result['pcr_pct_chg']  = round(pct_chg, 1)
                    n = len(hist)
                    if   velocity >  0.25: v_sig = "FEAR SPIKE";        v_int = f"PCR surged +{velocity:.3f} (+{pct_chg:.1f}%) over {n} readings — rapid put accumulation. Contrarian: potential capitulation bottom."
                    elif velocity >  0.10: v_sig = "PUT ACCUMULATION";  v_int = f"PCR rising +{velocity:.3f} — growing bearish positioning."
                    elif velocity < -0.25: v_sig = "COMPLACENCY SPIKE"; v_int = f"PCR collapsed {velocity:.3f} ({pct_chg:.1f}%) — rapid call surge. Contrarian: potential distribution top."
                    elif velocity < -0.10: v_sig = "CALL ACCUMULATION"; v_int = f"PCR falling {velocity:.3f} — growing bullish positioning."
                    else:                  v_sig = "STABLE";             v_int = f"PCR change {velocity:+.3f} over {n} readings — positioning stable."
                    result['velocity_signal'] = v_sig; result['velocity_interp'] = v_int
        except Exception as exc:
            result['interpretation'] = f"Options fetch error: {exc}"
        return result
    
    def get_max_pain(self, ticker):
        """Calculates the Options Max Pain strike for the nearest expiration date."""
        key = f"maxpain_{ticker}"
        # Cache it for 1 hour so we don't spam Yahoo Finance and slow down the app
        if key in self._cache:
            ts, mp = self._cache[key]
            if time.time() - ts < 3600: 
                return mp
        try:
            tk = yf.Ticker(ticker)
            exp_dates = tk.options
            if not exp_dates: 
                return None
                
            closest_expiry = exp_dates[0]  # The weekly options driving the current pin
            chain = tk.option_chain(closest_expiry)
            
            # Combine all unique strike prices
            strikes = sorted(set(chain.calls['strike']).union(set(chain.puts['strike'])))
            min_pain = float('inf')
            mp_strike = 0
            
            for strike in strikes:
                # Intrinsic value of calls and puts at this strike
                call_pain = (chain.calls[chain.calls['strike'] < strike]['openInterest'].fillna(0) * (strike - chain.calls[chain.calls['strike'] < strike]['strike'])).sum()
                put_pain = (chain.puts[chain.puts['strike'] > strike]['openInterest'].fillna(0) * (chain.puts[chain.puts['strike'] > strike]['strike'] - strike)).sum()
                
                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    mp_strike = strike
                    
            result = (mp_strike, closest_expiry)
            self._cache[key] = (time.time(), result)
            return result
        except Exception as e:
            print(f"Max Pain calculation error: {e}")
            return None

    def advanced_signals(self, df, ticker, period='1y'):
        """Confluence map, volatility-adjusted momentum, beta-adjusted alpha."""
        result = {
            'confluence': [], 'momentum_z': 'N/A', 'atr_context': 'N/A',
            'beta_alpha': 'N/A', 'hidden_strength': 'NEUTRAL', 'interpretation': '',
        }
        try:
            c    = df['Close']
            last = df.iloc[-1]

            # ── 1. CONFLUENCE MAP ────────────────────────────────────────────
            sigs = []
            try:
                bb_u  = float(last.get('BB_U', np.nan)); bb_l = float(last.get('BB_L', np.nan))
                price = float(last['Close'])
                pct_b = (price - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5
                if   pct_b <= 0.05: sigs.append("Price at Lower Bollinger Band (oversold zone)")
                elif pct_b >= 0.95: sigs.append("Price at Upper Bollinger Band (overbought zone)")
                elif 0.45 <= pct_b <= 0.55: sigs.append("Price at BB midline (equilibrium)")
            except Exception: pass
            try:
                rsi = float(last.get('RSI', np.nan))
                if   rsi < 35: sigs.append(f"RSI Oversold ({rsi:.1f})")
                elif rsi > 65: sigs.append(f"RSI Overbought ({rsi:.1f})")
            except Exception: pass
            try:
                if bool(last.get('RSI_Bear_Div', False)): sigs.append("RSI Bearish Divergence (reversal warning)")
                if bool(last.get('RSI_Bull_Div', False)): sigs.append("RSI Bullish Divergence (recovery signal)")
            except Exception: pass
            try:
                macd = float(last.get('MACD', np.nan)); sig = float(last.get('Signal', np.nan))
                prev_macd = float(df['MACD'].iloc[-2]); prev_sig = float(df['Signal'].iloc[-2])
                if   macd > sig and prev_macd <= prev_sig: sigs.append("MACD Fresh Bullish Crossover")
                elif macd < sig and prev_macd >= prev_sig: sigs.append("MACD Fresh Bearish Crossover")
                elif macd > sig: sigs.append("MACD Above Signal (bullish momentum)")
                else:            sigs.append("MACD Below Signal (bearish momentum)")
            except Exception: pass
            try:
                cmf = float(last.get('CMF', np.nan))
                if   cmf >  0.1: sigs.append(f"CMF Positive ({cmf:.2f}) — accumulation")
                elif cmf < -0.1: sigs.append(f"CMF Negative ({cmf:.2f}) — distribution")
            except Exception: pass
            try:
                obv = df.get('OBV', pd.Series(dtype=float))
                if len(obv) >= 10:
                    if obv.iloc[-1] > obv.iloc[-10]: sigs.append("OBV Rising (buying pressure)")
                    else:                             sigs.append("OBV Falling (selling pressure)")
            except Exception: pass
            try:
                ma20 = float(last.get('MA20',np.nan)); ma50 = float(last.get('MA50',np.nan))
                ma200= float(last.get('MA200',np.nan)); price = float(last['Close'])
                if   price > ma20 > ma50 > ma200: sigs.append("Full Bullish MA Stack (Price>MA20>MA50>MA200)")
                elif price < ma20 < ma50 < ma200: sigs.append("Full Bearish MA Stack (Price<MA20<MA50<MA200)")
                elif price > ma200: sigs.append("Price Above MA200 (long-term uptrend)")
                else:               sigs.append("Price Below MA200 (long-term downtrend)")
            except Exception: pass
            try:
                vwap = float(last.get('VWAP', np.nan)); price = float(last['Close'])
                if price > vwap: sigs.append("Price Above VWAP (buyers in control)")
                else:            sigs.append("Price Below VWAP (sellers in control)")
            except Exception: pass
            try:
                mr_col = df.get('Market_Regime', None)
                if isinstance(mr_col, list) and mr_col: regime = mr_col[-1]
                elif hasattr(mr_col, 'iloc'):            regime = str(mr_col.iloc[-1])
                else:                                    regime = 'UNKNOWN'
                if regime != 'UNKNOWN': sigs.append(f"Market Regime: {regime}")
            except Exception: pass

            # ── SMART MONEY CONCEPTS (FVG) AI INTEGRATION ──
            try:
                highs, lows = df['High'].values, df['Low'].values
                active_bull, active_bear = [],[]
                for i in range(2, len(df)):
                    if lows[i] > highs[i-2]:
                        active_bull.append({'top': lows[i], 'bot': highs[i-2]})
                    elif highs[i] < lows[i-2]:
                        active_bear.append({'top': lows[i-2], 'bot': highs[i]})
                        
                    # Filter out ones that got filled
                    active_bull = [g for g in active_bull if lows[i] > g['bot']]
                    active_bear = [g for g in active_bear if highs[i] < g['top']]
                
                # Check if current price is near an active FVG (within 5%)
                curr_p = float(df['Close'].iloc[-1])
                for g in active_bull:
                    if curr_p > g['bot'] and (curr_p - g['top'])/g['top'] < 0.05:
                        sigs.append(f"Price approaching Bullish FVG Support (${g['bot']:.2f}-${g['top']:.2f}) - Institutional Buying Zone")
                for g in active_bear:
                    if curr_p < g['top'] and (g['bot'] - curr_p)/curr_p < 0.05:
                        sigs.append(f"Price approaching Bearish FVG Resistance (${g['bot']:.2f}-${g['top']:.2f}) - Institutional Selling Zone")
            except Exception: pass
            result['confluence'] = sigs

            # ── 2. VOLATILITY-ADJUSTED MOMENTUM ─────────────────────────────
            try:
                atr     = float(last.get('ATR', np.nan))
                ret_1   = float(c.pct_change().iloc[-1]) * 100
                ret_s   = c.pct_change().dropna() * 100
                atr_pct = (atr / float(c.iloc[-2])) * 100 if float(c.iloc[-2]) > 0 else 1
                z_raw   = ret_1 / atr_pct if atr_pct != 0 else 0
                roll_z  = (ret_1 - ret_s.rolling(20).mean().iloc[-1]) / (ret_s.rolling(20).std().iloc[-1] + 1e-9)
                ctx = ("EXTREME OUTLIER" if abs(z_raw) > 2.0 else
                       "ELEVATED"        if abs(z_raw) > 1.5 else
                       "SUBDUED"         if abs(z_raw) < 0.3 else "NORMAL")
                result['momentum_z']  = round(float(roll_z), 3)
                result['atr_context'] = f"Today {ret_1:+.2f}% = {z_raw:+.1f}x ATR → {ctx}  (20d Z: {roll_z:+.2f})"
            except Exception: pass

            # ── 3. BETA-ADJUSTED ALPHA ───────────────────────────────────────
            try:
                import yfinance as yf
                spy_df = yf.download('SPY', period=period, interval='1d', auto_adjust=True, progress=False)
                if spy_df is not None and len(spy_df) >= 20:
                    spy_r = spy_df['Close'].pct_change().dropna()
                    stk_r = c.pct_change().dropna()
                    common = spy_r.index.intersection(stk_r.index)
                    spy_r  = spy_r.loc[common]; stk_r = stk_r.loc[common]
                    if len(common) >= 20:
                        cov  = np.cov(stk_r.values, spy_r.values)[0][1]
                        var  = np.var(spy_r.values)
                        beta = cov / var if var != 0 else 1.0
                        for days, lbl in [(5,'5d'),(20,'20d'),(60,'60d')]:
                            if len(stk_r) >= days:
                                sc = float((1+stk_r.tail(days)).prod()-1)*100
                                pc = float((1+spy_r.tail(days)).prod()-1)*100
                                result[f'alpha_{lbl}']   = round(sc - beta*pc, 2)
                                result[f'stk_ret_{lbl}'] = round(sc, 2)
                                result[f'spy_ret_{lbl}'] = round(pc, 2)
                        a5  = result.get('alpha_5d', 0)
                        s5  = result.get('spy_ret_5d', 0)
                        result['hidden_strength'] = (
                            'STRONG — market down, stock resilient (institutional accumulation)' if s5 < -0.5 and a5 > 0.5 else
                            'WEAK — market up, stock lagging (distribution)'                     if s5 > 0.5  and a5 < -0.5 else
                            'OUTPERFORMING'                                                       if a5 > 1.0  else
                            'UNDERPERFORMING'                                                     if a5 < -1.0 else
                            'NEUTRAL'
                        )
                        result['beta_alpha']      = round(beta, 3)
                        result['interpretation']  = (
                            f"Beta={beta:.2f} | "
                            f"5d alpha={result.get('alpha_5d','N/A')}%  "
                            f"20d alpha={result.get('alpha_20d','N/A')}%  "
                            f"60d alpha={result.get('alpha_60d','N/A')}%"
                        )
            except Exception: pass
        except Exception: pass
        return result

    def earnings_proximity(self, ticker):
        """Return days until next earnings + blackout window flag."""
        result = {'days_to_earnings': None, 'next_earnings_date': 'Unknown',
                  'earnings_warning': False, 'blackout': False,
                  'signal_reliability': 'NORMAL', 'note': 'Earnings data unavailable'}
        try:
            tk  = yf.Ticker(ticker)
            cal = tk.calendar
            ed  = []
            if isinstance(cal, dict):
                raw = cal.get('Earnings Date', [])
                ed  = list(raw) if hasattr(raw, '__iter__') and not isinstance(raw, str) else ([raw] if raw else [])
            elif hasattr(cal, 'empty') and not cal.empty:
                try: ed = [v for v in cal.loc['Earnings Date'].values if pd.notna(v)]
                except Exception: pass
            if not ed:
                try:
                    fi = tk.fast_info
                    ep = getattr(fi, 'earnings_date', None) or getattr(fi, 'next_earnings_date', None)
                    if ep: ed = [ep]
                except Exception: pass
            now = pd.Timestamp.now(tz='UTC')
            future = []
            for d in ed:
                try:
                    ts = pd.Timestamp(d)
                    if ts.tzinfo is None: ts = ts.tz_localize('UTC')
                    if ts > now - pd.Timedelta(days=1): future.append(ts)
                except Exception: pass
            if future:
                next_dt  = min(future)
                days_rem = int((next_dt - now).days)
                result['days_to_earnings']  = days_rem
                result['next_earnings_date'] = next_dt.strftime('%Y-%m-%d')
                if days_rem <= 2:
                    result['blackout'] = True; result['earnings_warning'] = True
                    result['signal_reliability'] = 'SEVERELY DEGRADED'
                    result['note'] = (f"⚠ EARNINGS IN {days_rem}d ({next_dt.strftime('%b %d')}) — "
                                      f"BLACKOUT WINDOW: technical signals unreliable. DO NOT trade on TA alone.")
                elif days_rem <= 7:
                    result['earnings_warning'] = True; result['signal_reliability'] = 'REDUCED'
                    result['note'] = (f"⚠ Earnings in {days_rem}d ({next_dt.strftime('%b %d')}) — "
                                      f"REDUCED reliability. Tighten stops, reduce size.")
                elif days_rem <= 14:
                    result['signal_reliability'] = 'SLIGHTLY REDUCED'
                    result['note'] = f"Earnings in {days_rem}d ({next_dt.strftime('%b %d')}) — approaching event window."
                else:
                    result['note'] = f"Next earnings: {next_dt.strftime('%b %d')} ({days_rem}d). Signals at full reliability."
            else:
                result['note'] = "No upcoming earnings date found. Signals at full reliability."
        except Exception as exc:
            result['note'] = f"Earnings lookup error: {exc}"
        return result

    def spy_correlation(self, ticker, period='1y'):
        """Compute rolling correlation and relative strength vs SPY."""
        result = {
            'corr_full': None, 'corr_30d': None, 'corr_90d': None,
            'beta': None, 'rs_signal': 'NEUTRAL',
            'interpretation': 'Correlation data unavailable'
        }
        try:
            t_df   = self.fetch(ticker, period=period, interval='1d')
            spy_df = self.fetch('SPY',   period=period, interval='1d')
            if t_df is None or spy_df is None or t_df.empty or spy_df.empty:
                return result
            t_ret   = t_df['Close'].pct_change().dropna()
            spy_ret = spy_df['Close'].pct_change().dropna()
            aligned = pd.concat([t_ret, spy_ret], axis=1, join='inner').dropna()
            aligned.columns = ['ticker', 'spy']
            if len(aligned) < 20:
                return result
            corr_full = aligned['ticker'].corr(aligned['spy'])
            corr_30   = aligned.tail(30)['ticker'].corr(aligned.tail(30)['spy'])
            corr_90   = aligned.tail(90)['ticker'].corr(aligned.tail(90)['spy'])
            cov  = aligned['ticker'].cov(aligned['spy'])
            var  = aligned['spy'].var()
            beta = round(cov / var, 3) if var != 0 else None
            def _rs(n):
                t_c   = (t_df['Close'].iloc[-1] / t_df['Close'].iloc[-min(n,len(t_df))] - 1) * 100
                spy_c = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-min(n,len(spy_df))] - 1) * 100
                return round(t_c - spy_c, 2)
            rs_30, rs_90, rs_252 = _rs(30), _rs(90), _rs(252)
            if   rs_30 >  3 and rs_90 >  3: signal = "STRONG RELATIVE STRENGTH";   note = f"{ticker} outperforming SPY — bullish divergence, institutional accumulation likely."
            elif rs_30 < -3 and rs_90 < -3: signal = "RELATIVE WEAKNESS ⚠";        note = f"{ticker} underperforming SPY — price dropping while market rises = distribution/selling pressure."
            elif rs_30 >  3 and rs_90 < -3: signal = "RECENT RECOVERY";             note = f"{ticker} showing recent strength vs SPY after longer underperformance."
            else:                            signal = "IN-LINE WITH MARKET";         note = f"{ticker} moving in line with SPY — no significant divergence detected."
            result.update({
                'corr_full': round(corr_full,4), 'corr_30d': round(corr_30,4),
                'corr_90d': round(corr_90,4),    'beta': beta,
                'rs_30d': rs_30, 'rs_90d': rs_90, 'rs_252d': rs_252,
                'rs_signal': signal, 'interpretation': note,
            })
        except Exception as exc:
            result['interpretation'] = f"Correlation error: {exc}"
        return result

    def cointegration_analysis(self, ticker_a, ticker_b, period='2y'):
        """Calculates the Z-Score and Cointegration p-value between two assets."""
        result = {'error': None, 'z_score': None, 'coint_pvalue': None, 'signal': 'NEUTRAL'}
        if not HAS_SM:
            result['error'] = "statsmodels not installed. Run: pip install statsmodels"
            return result
        try:
            df_a = self.fetch(ticker_a, period=period, interval='1d')
            df_b = self.fetch(ticker_b, period=period, interval='1d')
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                result['error'] = "Failed to fetch data for one or both tickers."
                return result
            
            # Align dates perfectly
            close_a = df_a['Close']
            close_b = df_b['Close']
            data = pd.concat([close_a, close_b], axis=1, join='inner').dropna()
            data.columns =[ticker_a, ticker_b]
            
            if len(data) < 100:
                result['error'] = "Insufficient overlapping data (need ≥ 100 bars)."
                return result

            # 1. Cointegration Test (Are they actually anchored together?)
            score, pvalue, _ = coint(data[ticker_a], data[ticker_b])
            result['coint_pvalue'] = pvalue
            
            # 2. Price Ratio & Rolling Z-Score (60-day moving window)
            ratio = data[ticker_a] / data[ticker_b]
            rolling_mean = ratio.rolling(window=60).mean()
            rolling_std  = ratio.rolling(window=60).std()
            z_score = (ratio - rolling_mean) / rolling_std
            
            current_z = float(z_score.iloc[-1])
            current_ratio = float(ratio.iloc[-1])
            
            result['z_score'] = current_z
            result['spread_current'] = current_ratio
            
            if pvalue < 0.05:
                result['is_cointegrated'] = True
                if current_z < -2.0:
                    result['signal'] = f"BUY {ticker_a} / SELL {ticker_b}"
                    result['interpretation'] = f"Ratio is -{abs(current_z):.2f} standard deviations below mean. Expect {ticker_a} to outperform {ticker_b}."
                elif current_z > 2.0:
                    result['signal'] = f"SELL {ticker_a} / BUY {ticker_b}"
                    result['interpretation'] = f"Ratio is +{abs(current_z):.2f} standard deviations above mean. Expect {ticker_b} to outperform {ticker_a}."
                else:
                    result['signal'] = "NEUTRAL"
                    result['interpretation'] = f"Spread is within normal bounds (Z-score: {current_z:+.2f}). Wait for extreme divergence (>2 or <-2)."
            else:
                result['is_cointegrated'] = False
                result['signal'] = "NO STATISTICAL RELATIONSHIP"
                result['interpretation'] = f"Not cointegrated (p-value {pvalue:.3f} > 0.05). The spread is random. Pairs trading is highly risky here."

            return result
        except Exception as e:
            result['error'] = str(e)
            return result

    def seasonality_analysis(self, ticker, years=20):
        """Calculates win-rates and avg returns by month over N years."""
        import calendar
        result = {'error': None, 'data': {}}
        try:
            # Fetch up to 20 years of monthly data
            df = yf.Ticker(ticker).history(period=f"{years}y", interval="1mo")
            if df.empty or len(df) < 12:
                result['error'] = "Insufficient long-term data for seasonality analysis."
                return result
            
            # Calculate monthly returns
            df['Return'] = df['Close'].pct_change() * 100
            df['Month'] = df.index.month
            df = df.dropna()
            
            for m in range(1, 13):
                m_data = df[df['Month'] == m]
                if len(m_data) == 0: continue
                
                wins = (m_data['Return'] > 0).sum()
                total = len(m_data)
                win_rate = (wins / total) * 100
                avg_ret = m_data['Return'].mean()
                
                result['data'][calendar.month_abbr[m].upper()] = {
                    'win_rate': win_rate,
                    'avg_return': avg_ret,
                    'years': total
                }
            return result
        except Exception as e:
            result['error'] = str(e)
            return result
        
    def get_insider_activity(self, ticker):
        """Fetches SEC Form 4 insider trading data over the last 6 months."""
        result = {'signal': 'NEUTRAL', 'interpretation': 'Insider data unavailable.', 'net_shares': 0, 'insider_pct': 0.0}
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            
            # 1. Total percentage of the company owned by insiders
            pct = info.get('heldPercentInsiders', 0)
            if pct is not None:
                result['insider_pct'] = pct * 100
                
            # 2. Grab the latest 6-month summary of insider trades
            ip = tk.insider_purchases
            if ip is not None and not ip.empty:
                # yfinance column names can shift slightly, so we dynamically match them
                net_cols =[c for c in ip.columns if 'Net' in c]
                buy_cols =[c for c in ip.columns if 'Purchases' in c or 'Buy' in c]
                sell_cols =[c for c in ip.columns if 'Sales' in c or 'Sell' in c]
                
                if net_cols and len(ip) > 0:
                    raw_net = ip[net_cols[0]].iloc[0]
                    # Clean the number (remove commas if it's a string)
                    try: net_shares = int(float(str(raw_net).replace(',', '')))
                    except: net_shares = 0
                    
                    buys = ip[buy_cols[0]].iloc[0] if buy_cols else 0
                    sells = ip[sell_cols[0]].iloc[0] if sell_cols else 0
                    
                    result['net_shares'] = net_shares
                    
                    # 3. Create algorithmic signals based on net volume
                    if net_shares > 10000:
                        result['signal'] = "STRONG BULLISH 🟢"
                        result['interpretation'] = f"Net {net_shares:,} shares BOUGHT by insiders over 6M ({buys} buys vs {sells} sells). C-suite is accumulating."
                    elif net_shares < -10000:
                        result['signal'] = "BEARISH 🔴"
                        result['interpretation'] = f"Net {abs(net_shares):,} shares SOLD by insiders over 6M ({sells} sells vs {buys} buys). C-suite is taking profit."
                    else:
                        result['signal'] = "NEUTRAL ⚖"
                        result['interpretation'] = f"Insignificant insider activity ({net_shares:,} net shares)."
            else:
                result['interpretation'] = f"Insiders hold {result['insider_pct']:.2f}% of float. No recent notable SEC filings."
                
        except Exception as e:
            result['interpretation'] = f"Insider fetch error: {e}"
        return result
    
    def fft_extrapolation(self, df, horizon=30):
        """Uses Fast Fourier Transform (FFT) to extract dominant cycles and project them into the future."""
        try:
            y = df['Close'].values
            n = len(y)
            x = np.arange(n)
            
            # 1. Detrend the data so the FFT can focus purely on the cycles (not the macro slope)
            poly = np.polyfit(x, y, 1)
            trend = np.polyval(poly, x)
            detrended = y - trend

            # 2. Execute the Fast Fourier Transform
            fft_vals = np.fft.fft(detrended)
            fft_freqs = np.fft.fftfreq(n)

            # 3. Extract the Top 3 Dominant Frequencies (ignoring noise)
            pos_idx = np.where(fft_freqs > 0)[0]
            amps = np.abs(fft_vals[pos_idx])
            top_idx = pos_idx[np.argsort(amps)[-3:]]

            # 4. Extrapolate the cycles 30 days into the future
            future_x = np.arange(n, n + horizon)
            extrapolated = np.zeros(len(future_x))
            
            for i in top_idx:
                amp = np.abs(fft_vals[i]) / n
                phase = np.angle(fft_vals[i])
                freq = fft_freqs[i]
                extrapolated += 2 * amp * np.cos(2 * np.pi * freq * future_x + phase)

            # 5. Re-add the linear trend to the future projection
            future_trend = np.polyval(poly, future_x)
            future_path = extrapolated + future_trend

            # Find exact days of the projected peak and trough
            max_idx = np.argmax(future_path)
            min_idx = np.argmin(future_path)

            return {
                'future_path': future_path,
                'next_peak_days': max_idx + 1,
                'next_peak_price': future_path[max_idx],
                'next_trough_days': min_idx + 1,
                'next_trough_price': future_path[min_idx],
                'target_30d': future_path[-1]
            }
        except Exception as e:
            print(f"FFT Error: {e}")
            return None

    def vmd_extrapolation(self, df, horizon=30):
        """Uses Variational Mode Decomposition to strip noise and project the true trend/cycle."""
        if not HAS_VMD: return None
        try:
            # 1. Clean the data and ensure EVEN length (VMD algorithm glitches on odd-length arrays)
            y = df['Close'].dropna().values
            if len(y) % 2 != 0:
                y = y[1:]  # Drop the oldest bar to make length an even number!
                
            n = len(y)
            
            # VMD Parameters
            alpha = 2000       # moderate bandwidth constraint
            tau = 0.           # noise-tolerance
            K = 4              # Decompose into 4 modes (0=Trend, 1=Macro Cycle, 2=Micro Cycle, 3=Noise)
            DC = 0             # no DC part imposed
            init = 1           # initialize omegas uniformly
            tol = 1e-7

            # Run VMD
            u, u_hat, omega = VMD(y, alpha, tau, K, DC, init, tol)
            
            # 🚀 FIX: Use actual length returned by VMD to guarantee numpy arrays perfectly match
            n_vmd = u.shape[1]
            future_x = np.arange(n_vmd, n_vmd + horizon)
            
            # 1. Extrapolate the Trend (Mode 0)
            trend_imf = u[0, :]
            poly = np.polyfit(np.arange(n_vmd), trend_imf, 1)
            future_trend = np.polyval(poly, future_x)
            
            # 2. Extrapolate the Cycles (Modes 1 and 2), IGNORING the Noise (Mode 3)
            extrapolated_cycles = np.zeros(horizon)
            for i in range(1, K-1): 
                imf = u[i, :]
                fft_vals = np.fft.fft(imf)
                freqs = np.fft.fftfreq(n_vmd)
                
                pos_idx = np.where(freqs > 0)[0]
                if len(pos_idx) == 0: continue
                dom_idx = pos_idx[np.argmax(np.abs(fft_vals[pos_idx]))]
                
                amp = np.abs(fft_vals[dom_idx]) / n_vmd
                phase = np.angle(fft_vals[dom_idx])
                freq = freqs[dom_idx]
                
                extrapolated_cycles += 2 * amp * np.cos(2 * np.pi * freq * future_x + phase)
            
            # 3. Recombine Trend + Clean Cycles
            future_path = future_trend + extrapolated_cycles

            return {
                'future_path': future_path,
                'target_30d': future_path[-1]
            }
        except Exception as e:
            print(f"VMD Error: {e}")
            return None


DE = DataEngine()

# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY AGENT — JSON Persistence
# ═══════════════════════════════════════════════════════════════════════════════
import json, os

class MemoryAgent:
    """
    Persists per-ticker analyst memory to a local JSON file.
    Stores: last signal, confidence, thesis, kelly size, timestamp.
    """
    # 👇 FIX: Ensure the JSON file saves next to the .exe when compiled
    if getattr(sys, 'frozen', False):
        _base_dir = os.path.dirname(sys.executable)
    else:
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        
    PATH = os.path.join(_base_dir, 'apex_memory.json')

    def __init__(self):
        self._mem = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.PATH):
                with open(self.PATH, 'r') as f:
                    self._mem = json.load(f)
        except Exception:
            self._mem = {}

    def _save(self):
        try:
            with open(self.PATH, 'w') as f:
                json.dump(self._mem, f, indent=2, default=str)
        except Exception:
            pass

    def store(self, ticker: str, data: dict):
        """Save a dict of fields for a ticker."""
        if ticker not in self._mem:
            self._mem[ticker] = []
        entry = {'timestamp': str(datetime.datetime.now()), **data}
        self._mem[ticker].append(entry)
        self._mem[ticker] = self._mem[ticker][-20:]   # keep last 20 entries
        self._save()

    def recall(self, ticker: str, n: int = 5) -> list:
        """Return last n memory entries for a ticker."""
        return self._mem.get(ticker, [])[-n:]

    def recall_text(self, ticker: str, n: int = 5) -> str:
        """Return formatted string of last n entries."""
        entries = self.recall(ticker, n)
        if not entries:
            return f"No memory entries for {ticker}."
        lines = [f"── MEMORY: {ticker} (last {len(entries)} analyses) ──"]
        for e in entries:
            ts  = e.get('timestamp', '')[:16]
            sig = e.get('signal', 'N/A')
            con = e.get('confidence', 'N/A')
            the = e.get('thesis', '')[:120]
            kel = e.get('kelly_half', 'N/A')
            lines.append(f"  [{ts}] Signal={sig} | Confidence={con} | Kelly½={kel}%")
            if the:
                lines.append(f"           Thesis: {the}")
        return '\n'.join(lines)

    def all_tickers(self) -> list:
        return list(self._mem.keys())

    def clear(self, ticker: str = None):
        if ticker:
            self._mem.pop(ticker, None)
        else:
            self._mem = {}
        self._save()

MEMORY = MemoryAgent()


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME CONFLUENCE
# ═══════════════════════════════════════════════════════════════════════════════
class MultiTimeframeAnalyzer:
    """
    Fetches 3 timeframes (daily, weekly, monthly) and scores
    directional agreement. Returns a confluence score 0–100
    and a dominant bias (BULLISH / BEARISH / MIXED).
    """
    TIMEFRAMES = [
        ('Daily',   '6mo',  '1d'),
        ('Weekly',  '2y',   '1wk'),
        ('Monthly', '5y',   '1mo'),
    ]

    def analyze(self, ticker: str) -> dict:
        signals = {}
        for label, period, interval in self.TIMEFRAMES:
            try:
                df = DE.fetch(ticker, period=period, interval=interval)
                if df is None or len(df) < 30:
                    signals[label] = 'NEUTRAL'
                    continue
                df = DE.add_indicators(df)
                last = df.iloc[-1]
                score = 0
                # Trend
                c = float(last['Close'])
                ma20  = float(last.get('MA20',  c))
                ma50  = float(last.get('MA50',  c))
                ma200 = float(last.get('MA200', c))
                if c > ma20 > ma50:   score += 2
                elif c < ma20 < ma50: score -= 2
                if c > ma200: score += 1
                else:         score -= 1
                # Momentum
                rsi = float(last.get('RSI', 50))
                if rsi > 55:  score += 1
                elif rsi < 45: score -= 1
                # MACD
                macd = float(last.get('MACD', 0))
                sig  = float(last.get('Signal', 0))
                if macd > sig:  score += 1
                else:           score -= 1
                # Label
                if   score >= 3:  signals[label] = 'BULLISH'
                elif score <= -3: signals[label] = 'BEARISH'
                else:             signals[label] = 'NEUTRAL'
            except Exception:
                signals[label] = 'NEUTRAL'

        bull = sum(1 for v in signals.values() if v == 'BULLISH')
        bear = sum(1 for v in signals.values() if v == 'BEARISH')
        total = len(signals)
        confluence_score = round((bull / total) * 100) if total else 50

        if bull == total:    bias = 'STRONGLY BULLISH'
        elif bull > bear:    bias = 'BULLISH'
        elif bear == total:  bias = 'STRONGLY BEARISH'
        elif bear > bull:    bias = 'BEARISH'
        else:                bias = 'MIXED / NEUTRAL'

        lines = [f"  {tf:<8} → {sig}" for tf, sig in signals.items()]
        interpretation = (
            f"Multi-Timeframe Confluence: {bias} ({confluence_score}% bull)\n"
            + '\n'.join(lines)
        )
        return {
            'signals':           signals,
            'confluence_score':  confluence_score,
            'bias':              bias,
            'interpretation':    interpretation,
        }

MTF = MultiTimeframeAnalyzer()


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED OPTIMIZING BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════════
class VectorizedBacktester:
    """
    Uses fast pandas/numpy arrays to instantly test multiple indicator parameters.
    Automatically selects the optimal parameters for the specific ticker.
    """
    def run(self, df, fwd_bars=10):
        results = {}
        if df is None or len(df) < 100:
            return {'error': 'Insufficient data for vectorized backtest (need ≥100 bars)'}
        
        try:
            c = df['Close']
            n = len(df)
            
            # 1. Calculate future returns instantly for every single bar
            fwd_ret = (c.shift(-fwd_bars) - c) / c * 100
            
            summary = {}

            # Helper to calculate metrics from an array of trades
            def _calc_metrics(trade_returns, name_label):
                arr = trade_returns.values
                wins = int((arr > 0).sum())
                wr = wins / len(arr) * 100
                avg_r = float(arr.mean())
                std_r = float(arr.std()) if arr.std() > 0 else 1e-9
                sharpe = (avg_r / std_r) * np.sqrt(252 / fwd_bars)
                return {
                    'note': name_label,
                    'trades': len(arr), 'win_rate': round(wr, 1),
                    'avg_return': round(avg_r, 2), 'sharpe': round(sharpe, 2),
                    'best': round(float(arr.max()), 2), 'worst': round(float(arr.min()), 2)
                }

            # 2. RSI Optimization (Test 30, 35, 40 and find the best fit for THIS stock)
            if 'RSI' in df:
                best_rsi_wr = 0
                best_rsi_res = None
                for thresh in[30, 35, 40]:
                    mask = df['RSI'] < thresh
                    trades = fwd_ret[mask].dropna()
                    if len(trades) >= 3:
                        res = _calc_metrics(trades, f"RSI < {thresh}")
                        # Optimize for highest win rate with at least 3 trades
                        if res['win_rate'] > best_rsi_wr:
                            best_rsi_wr = res['win_rate']
                            best_rsi_res = res
                if best_rsi_res:
                    summary['RSI_Oversold'] = best_rsi_res
                else:
                    summary['RSI_Oversold'] = {'trades': 0, 'note': 'No reliable RSI signals'}

            # 3. MACD Crossover Vectorized
            if 'MACD' in df and 'Signal' in df:
                mac = df['MACD']
                sig = df['Signal']
                mask = (mac > sig) & (mac.shift(1) <= sig.shift(1))
                trades = fwd_ret[mask].dropna()
                if len(trades) >= 3:
                    summary['MACD_Cross'] = _calc_metrics(trades, "MACD Bull Cross")
                else:
                    summary['MACD_Cross'] = {'trades': 0, 'note': 'Insufficient MACD crosses'}

            # 4. Bollinger Band Bounce
            if 'BB_L' in df:
                mask = c <= df['BB_L'] * 1.01
                trades = fwd_ret[mask].dropna()
                if len(trades) >= 3:
                    summary['BB_Bounce'] = _calc_metrics(trades, "BB Lower Bounce")
                else:
                    summary['BB_Bounce'] = {'trades': 0, 'note': 'Insufficient BB touches'}
                    
            # 5. NEW: Test the Scalper strategy we just added!
            if 'Scalp_Buy' in df:
                mask = df['Scalp_Buy']
                trades = fwd_ret[mask].dropna()
                if len(trades) >= 3:
                    summary['Scalp_Engine'] = _calc_metrics(trades, "1D Smart Scalp BUY")

            results['fwd_bars']   = fwd_bars
            results['signals']    = summary
            results['total_bars'] = n
            
        except Exception as exc:
            results['error'] = str(exc)
        return results

    def format(self, result: dict, ticker: str) -> str:
        if 'error' in result:
            return f"Vectorized Backtest Error: {result['error']}"
        lines =[f"VECTORIZED ML BACKTEST — {ticker}",
                 f"  Forward window: {result['fwd_bars']} bars | Total bars tested: {result['total_bars']}",
                 ""]
        for sig, r in result.get('signals', {}).items():
            if r.get('trades', 0) == 0:
                lines.append(f"  {sig:<14} — {r.get('note', 'Insufficient signals')}")
            else:
                lines.append(
                    f"  {r['note']:<18} | Trades={r['trades']:>3} | "
                    f"WinRate={r['win_rate']:>5.1f}% | "
                    f"AvgRet={r['avg_return']:>+5.2f}% | "
                    f"Sharpe={r['sharpe']:>+4.2f}"
                )
        return '\n'.join(lines)

WF_BACKTEST = VectorizedBacktester()

# ═══════════════════════════════════════════════════════════════════════════════
# XGBOOST MACHINE LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class XGBoostPredictor:
    """
    Trains an XGBoost Classifier on the fly using the ticker's recent history.
    Predicts the mathematical probability of a positive return over the next N days.
    """
    def __init__(self, fwd_days=5):
        self.fwd_days = fwd_days

    def predict(self, df):
        if not HAS_XGB:
            return {'error': 'XGBoost not installed.', 'interpretation': 'ML Engine Offline (pip install xgboost scikit-learn)'}
        if df is None or len(df) < 100:
            return {'error': 'Not enough data.', 'interpretation': 'Insufficient data for ML.'}

        try:
            d = df.copy()

            # 1. Feature Engineering
            d['BB_Width']      = (d.get('BB_U', d['Close']) - d.get('BB_L', d['Close'])) / d.get('MA20', d['Close'])
            d['Price_vs_MA20'] = d['Close'] / d.get('MA20', d['Close']) - 1
            d['Price_vs_MA50'] = d['Close'] / d.get('MA50', d['Close']) - 1
            
            features =['RSI', 'MACD', 'MHist', 'ATR', 'CMF', 'RVI', 'BB_Width', 'Price_vs_MA20', 'Price_vs_MA50']
            
            # Fill missing features with 0 to prevent crashes
            for f in features:
                if f not in d.columns:
                    d[f] = 0.0

            # 2. Define Target (1 if price goes UP in 'fwd_days', else 0)
            target_price = d['Close'].shift(-self.fwd_days)
            d['Target'] = np.where(target_price.isna(), np.nan, (target_price > d['Close']).astype(int))

            # 3. Separate the latest row for our prediction (it has no target yet!)
            latest_features = d[features].iloc[-1:].copy()

            # 4. Clean training data
            train_df = d.dropna(subset=features + ['Target'])
            if len(train_df) < 50:
                return {'error': 'Too many NaNs.', 'interpretation': 'Data insufficient for ML after cleaning.'}

            X = train_df[features]
            y = train_df['Target']

            # 5. Train the XGBoost Model on the fly
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X, y)

            # 6. Predict the probability of an UP move for today
            prob_up = model.predict_proba(latest_features)[0][1]

            # 7. Extract Feature Importance
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-3:][::-1]
            top_features = [(features[i], float(importances[i])) for i in top_idx]

            # Determine confidence signal
            if prob_up > 0.65:   signal = "HIGH PROBABILITY UP 🟢"
            elif prob_up < 0.35: signal = "HIGH PROBABILITY DOWN 🔴"
            elif prob_up > 0.55: signal = "MODERATE PROBABILITY UP ↗"
            elif prob_up < 0.45: signal = "MODERATE PROBABILITY DOWN ↘"
            else:                signal = "UNCERTAIN / CHOPPY ⚖"

            return {
                'prob_up': prob_up,
                'fwd_days': self.fwd_days,
                'top_features': top_features,
                'signal': signal,
                'interpretation': f"XGBoost {self.fwd_days}-Day Forecast: {prob_up*100:.1f}% chance of positive return. Signal: {signal}"
            }
        except Exception as e:
            return {'error': str(e), 'interpretation': f"ML Error: {str(e)}"}

XGB_ENGINE = XGBoostPredictor(fwd_days=5)


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO DCF
# ═══════════════════════════════════════════════════════════════════════════════
class MonteCarloDCF:
    """
    Runs N Monte Carlo simulations of a DCF model by sampling
    revenue growth, margins, WACC, and terminal growth from
    triangular distributions centred on analyst estimates.
    Returns a price distribution with percentiles.
    """

    def run(self, info: dict, n_sims: int = 10_000) -> dict:
        result = {}
        try:
            # Base inputs from yfinance info
            rev       = float(info.get('totalRevenue') or info.get('revenue', 0) or 0)
            fcf       = float(info.get('freeCashflow') or 0)
            shares    = float(info.get('sharesOutstanding') or 1)
            net_debt  = float(info.get('totalDebt') or 0) - float(info.get('totalCash') or 0)
            price     = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)

            if rev <= 0 or shares <= 0:
                return {'error': 'Insufficient fundamental data for Monte Carlo DCF'}

            rng = np.random.default_rng(42)

            # Sample parameters — triangular(low, mode, high)
            rev_growth  = rng.triangular(0.02, 0.10, 0.25,  n_sims)
            fcf_margin  = rng.triangular(0.05, 0.15, 0.30,  n_sims)
            wacc        = rng.triangular(0.07, 0.10, 0.15,  n_sims)
            term_growth = rng.triangular(0.01, 0.03, 0.05,  n_sims)

            implied_prices = []
            for i in range(n_sims):
                wc = wacc[i]
                tg = min(term_growth[i], wc - 0.01)
                pv = 0.0
                base_rev = rev
                for yr in range(1, 6):
                    base_rev *= (1 + rev_growth[i])
                    yr_fcf = base_rev * fcf_margin[i]
                    pv += yr_fcf / ((1 + wc) ** yr)
                # Terminal value
                terminal_fcf = base_rev * fcf_margin[i] * (1 + tg)
                tv = terminal_fcf / (wc - tg)
                pv += tv / ((1 + wc) ** 5)
                eq_val = pv - net_debt
                ip = eq_val / shares
                implied_prices.append(ip)

            arr = np.array(implied_prices)
            arr = arr[(arr > 0) & (arr < np.percentile(arr, 99))]   # trim extremes

            pcts = {
                'p10': round(float(np.percentile(arr, 10)), 2),
                'p25': round(float(np.percentile(arr, 25)), 2),
                'p50': round(float(np.percentile(arr, 50)), 2),
                'p75': round(float(np.percentile(arr, 75)), 2),
                'p90': round(float(np.percentile(arr, 90)), 2),
            }
            upside = (pcts['p50'] - price) / price * 100 if price else 0

            result = {
                'n_sims':          n_sims,
                'current_price':   round(price, 2),
                'percentiles':     pcts,
                'mean':            round(float(arr.mean()), 2),
                'std':             round(float(arr.std()),  2),
                'upside_p50':      round(upside, 1),
                'margin_of_safety': round((pcts['p25'] - price) / price * 100, 1) if price else 0,
            }
        except Exception as exc:
            result = {'error': str(exc)}
        return result

    def format(self, result: dict, ticker: str) -> str:
        if 'error' in result:
            return f"Walk-Forward Error: {result['error']}"
        p = result['percentiles']
        return (
            f"MONTE CARLO DCF — {ticker} ({result['n_sims']:,} simulations)\n"
            f"  Current Price:    ${result['current_price']}\n"
            f"  ─────────────────────────────────────────\n"
            f"  P10  (bear case): ${p['p10']}\n"
            f"  P25  (downside):  ${p['p25']}\n"
            f"  P50  (base):      ${p['p50']}  ← Median DCF\n"
            f"  P75  (upside):    ${p['p75']}\n"
            f"  P90  (bull case): ${p['p90']}\n"
            f"  ─────────────────────────────────────────\n"
            f"  Upside to P50:    {result['upside_p50']:+.1f}%\n"
            f"  Margin of Safety: {result['margin_of_safety']:+.1f}% (P25 vs price)\n"
            f"  Std Dev:          ${result['std']}"
        )

MC_DCF = MonteCarloDCF()


# ═══════════════════════════════════════════════════════════════════════════════
# EARNINGS TRANSCRIPT NLP
# ═══════════════════════════════════════════════════════════════════════════════
class EarningsTranscriptNLP:
    """
    Analyses earnings call transcripts (or any pasted text) via LM Studio.
    Extracts: sentiment score, guidance tone, key topics, red flags, bull signals.
    Works with the DocumentPanel — reads self.doctext if loaded.
    """

    SYSTEM = (
        "You are a buy-side analyst specialising in earnings call analysis. "
        "You read transcripts and extract signal, not noise. "
        "You focus on: management tone shifts, guidance language (raised/lowered/maintained), "
        "margin commentary, demand signals, and any forward-looking language. "
        "You are direct, structured, and quantitative where possible."
    )

    def build_prompt(self, ticker: str, transcript: str) -> str:
        chunk = transcript[:8000]
        return f"""Analyse this earnings call transcript for {ticker}.

TRANSCRIPT:
{chunk}

Provide a structured NLP analysis:

## SENTIMENT SCORE
Rate overall management tone: Bullish / Neutral / Bearish (X/10 confidence)

## GUIDANCE LANGUAGE
- Revenue guidance: Raised / Maintained / Lowered / No guidance
- Margin guidance: Expanding / Stable / Compressing
- Key forward phrases (quote directly, max 3)

## BULL SIGNALS
List specific positive signals from the transcript (max 5 bullets)

## RED FLAGS
List specific concerns or hedging language (max 5 bullets)

## KEY TOPICS
Top 5 topics discussed with sentiment tag (Positive / Neutral / Negative)

## MANAGEMENT CONFIDENCE
Body language of language: Confident / Cautious / Defensive

## INVESTMENT IMPLICATION
One paragraph: what does this transcript mean for the stock?

## TRANSCRIPT SCORE
Overall transcript quality signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
Conviction: X/100"""

    def analyze(self, ticker: str, transcript: str,
                 on_chunk=None, on_done=None):
        """Stream the NLP analysis. on_chunk/on_done are optional callbacks."""
        prompt = self.build_prompt(ticker, transcript)
        full = []
        try:
            for tok in LM.stream(
                [{"role": "user", "content": prompt}],
                system=self.SYSTEM,
                temperature=0.2,
                max_tokens=1200,
            ):
                full.append(tok)
                if on_chunk:
                    on_chunk(tok)
        except Exception as exc:
            err = f"\n[Transcript NLP error: {exc}]"
            full.append(err)
            if on_chunk:
                on_chunk(err)
        result = ''.join(full)
        if on_done:
            on_done(result)
        return result

TRANSCRIPT_NLP = EarningsTranscriptNLP()

# ═══════════════════════════════════════════════════════════════════════════════
# HMM REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
class HMMRegimeDetector:
    """
    3-state Hidden Markov Model for market regime detection.
    States: 0=Bear, 1=Sideways, 2=Bull  (auto-labelled by mean return)
    Falls back to rule-based regime if hmmlearn is not installed.
    """
    STATE_LABELS = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
    STATE_COLORS = {0: '#ff3d57', 1: '#ffd740', 2: '#00e676'}

    def fit_predict(self, df, n_components=3):
        """
        Returns dict with:
          'regime'        — current regime label (BULL/BEAR/SIDEWAYS)
          'state_probs'   — [p_bear, p_sideways, p_bull] for latest bar
          'confidence'    — probability of the most likely state (0-1)
          'regime_series' — pd.Series of regime labels per bar
          'method'        — 'HMM' or 'RULE_BASED'
          'interpretation'— human-readable summary
        """
        result = {
            'regime': 'UNKNOWN', 'state_probs': [0.33, 0.34, 0.33],
            'confidence': 0.0, 'regime_series': pd.Series(dtype=str),
            'method': 'RULE_BASED', 'interpretation': 'HMM unavailable',
        }
        try:
            closes  = df['Close'].dropna()
            returns = closes.pct_change().dropna().values.reshape(-1, 1)
            vol     = pd.Series(returns.flatten()).rolling(5).std().fillna(0).values.reshape(-1, 1)
            features = np.hstack([returns, vol])

            if HAS_HMM and len(features) >= 60:
                model = _hmm_lib.GaussianHMM(
                    n_components=n_components, covariance_type='diag',
                    n_iter=100, random_state=42
                )
                model.fit(features)
                hidden_states = model.predict(features)
                state_probs   = model.predict_proba(features)

                # Auto-label states by mean return (lowest=Bear, highest=Bull)
                means = {s: features[hidden_states == s, 0].mean()
                         for s in range(n_components) if (hidden_states == s).sum() > 0}
                sorted_states = sorted(means.keys(), key=lambda s: means[s])
                label_map = {}
                label_map[sorted_states[0]]  = 'BEAR'
                label_map[sorted_states[-1]] = 'BULL'
                for s in sorted_states[1:-1]:
                    label_map[s] = 'SIDEWAYS'

                cur_state = int(hidden_states[-1])
                cur_probs = state_probs[-1].tolist()
                cur_label = label_map.get(cur_state, 'SIDEWAYS')
                cur_conf  = float(cur_probs[cur_state])

                # Map state_probs to [bear, sideways, bull]
                ordered = {'BEAR': 0.0, 'SIDEWAYS': 0.0, 'BULL': 0.0}
                for s, lbl in label_map.items():
                    if s < len(cur_probs):
                        ordered[lbl] = round(cur_probs[s], 3)

                regime_series = pd.Series(
                    [label_map.get(s, 'SIDEWAYS') for s in hidden_states],
                    index=df.index[-len(hidden_states):]
                )

                # Regime duration
                duration = 1
                for s in reversed(hidden_states[:-1]):
                    if s == cur_state: duration += 1
                    else: break

                interp = (
                    f"HMM ({n_components}-state): Current regime = **{cur_label}** "
                    f"({cur_conf*100:.1f}% confidence). "
                    f"Bear={ordered['BEAR']*100:.1f}% / "
                    f"Sideways={ordered['SIDEWAYS']*100:.1f}% / "
                    f"Bull={ordered['BULL']*100:.1f}%. "
                    f"Regime duration: {duration} bars."
                )
                result.update({
                    'regime': cur_label,
                    'state_probs': [ordered['BEAR'], ordered['SIDEWAYS'], ordered['BULL']],
                    'confidence': cur_conf,
                    'regime_series': regime_series,
                    'method': 'HMM',
                    'interpretation': interp,
                    'duration_bars': duration,
                })
            else:
                # Fallback: rule-based from existing Market_Regime column
                result.update(self._rule_based(df))
        except Exception as exc:
            result.update(self._rule_based(df))
            result['interpretation'] += f' (HMM error: {exc})'
        return result

    @staticmethod
    def _rule_based(df):
        try:
            regime_col = df.get('Market_Regime', pd.Series(['UNKNOWN'] * len(df)))
            cur = str(regime_col.iloc[-1]) if len(regime_col) else 'UNKNOWN'
            if 'EXPANSION' in cur or 'BULL' in cur:
                lbl, probs = 'BULL', [0.15, 0.25, 0.60]
            elif 'SQUEEZE' in cur or 'CONTRACTION' in cur:
                lbl, probs = 'SIDEWAYS', [0.20, 0.60, 0.20]
            else:
                lbl, probs = 'BEAR', [0.60, 0.25, 0.15]
            return {
                'regime': lbl, 'state_probs': probs,
                'confidence': max(probs), 'method': 'RULE_BASED',
                'interpretation': f'Rule-based fallback: {cur} → {lbl}',
            }
        except Exception:
            return {'regime': 'UNKNOWN', 'state_probs': [0.33, 0.34, 0.33],
                    'confidence': 0.0, 'method': 'RULE_BASED',
                    'interpretation': 'Regime detection unavailable'}


HMM_DETECTOR = HMMRegimeDetector()


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════
class MacroFetcher:
    """
    Fetches macro context: VIX, DXY, TLT (bonds), sector ETF.
    Used by the Macro Sentinel agent in the Pod Architecture.
    """
    MACRO_TICKERS = {
        'VIX':    '^VIX',
        'DXY':    'DX-Y.NYB',
        'TLT':    'TLT',
        'SPY':    'SPY',
        'QQQ':    'QQQ',
        'IWM':    'IWM',      # small-cap
        'HYG':    'HYG',      # high-yield credit
        'GLD':    'GLD',
        'OIL':    'USO',
    }
    SECTOR_MAP = {
        'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
        'XLE': 'Energy',     'XLC': 'Comms',       'XLI': 'Industrials',
        'XLY': 'ConsDisc',   'XLP': 'ConsStaples', 'XLB': 'Materials',
        'XLRE':'RealEstate', 'XLU': 'Utilities',
    }

    def fetch_environment(self):
        """Returns a formatted macro context string for the AI agent."""
        lines_out = []
        try:
            for name, sym in self.MACRO_TICKERS.items():
                try:
                    df = DE.fetch(sym, period='5d', interval='1d')
                    if df is None or df.empty: continue
                    cur  = float(df['Close'].iloc[-1])
                    prev = float(df['Close'].iloc[-2]) if len(df) >= 2 else cur
                    chg  = (cur - prev) / prev * 100 if prev else 0
                    arrow = '▲' if chg >= 0 else '▼'
                    lines_out.append(f"  {name:<6} {cur:>9.2f}  {arrow}{abs(chg):.2f}%")
                except Exception:
                    pass

            # 1-month returns for key assets
            perf = []
            for name, sym in [('SPY','SPY'),('QQQ','QQQ'),('GLD','GLD'),('TLT','TLT')]:
                try:
                    df = DE.fetch(sym, period='1mo', interval='1d')
                    if df is not None and len(df) >= 2:
                        r = (float(df['Close'].iloc[-1]) / float(df['Close'].iloc[0]) - 1) * 100
                        perf.append(f"{name} {r:+.1f}%")
                except Exception:
                    pass

            # Top 3 sector ETF performers (1M)
            sector_perf = []
            for sym, name in self.SECTOR_MAP.items():
                try:
                    df = DE.fetch(sym, period='1mo', interval='1d')
                    if df is not None and len(df) >= 2:
                        r = (float(df['Close'].iloc[-1]) / float(df['Close'].iloc[0]) - 1) * 100
                        sector_perf.append((r, name, sym))
                except Exception:
                    pass
            sector_perf.sort(reverse=True)

            # VIX regime classification
            vix_regime = 'UNKNOWN'
            try:
                vdf = DE.fetch('^VIX', period='5d', interval='1d')
                if vdf is not None and len(vdf):
                    v = float(vdf['Close'].iloc[-1])
                    if   v < 15:  vix_regime = f'COMPLACENT ({v:.1f}) — low fear'
                    elif v < 20:  vix_regime = f'CALM ({v:.1f}) — normal'
                    elif v < 30:  vix_regime = f'ELEVATED ({v:.1f}) — caution'
                    elif v < 40:  vix_regime = f'HIGH FEAR ({v:.1f}) — risk-off'
                    else:         vix_regime = f'EXTREME FEAR ({v:.1f}) — potential capitulation'
            except Exception:
                pass

            result = "MACRO ENVIRONMENT:\n"
            result += '\n'.join(lines_out) if lines_out else "  (data unavailable)"
            result += f"\n\n1-MONTH PERFORMANCE: {' | '.join(perf)}"
            if sector_perf:
                result += f"\nSECTOR ROTATION (1M):"
                result += f"\n  Best:  {sector_perf[0][1]} ({sector_perf[0][0]:+.1f}%)"
                if len(sector_perf) > 1:
                    result += f"  |  {sector_perf[1][1]} ({sector_perf[1][0]:+.1f}%)"
                result += f"\n  Worst: {sector_perf[-1][1]} ({sector_perf[-1][0]:+.1f}%)"
            result += f"\nVIX REGIME: {vix_regime}"
            return result
        except Exception as exc:
            return f"Macro fetch error: {exc}"


MACRO = MacroFetcher()

# ═══════════════════════════════════════════════════════════════════════════════
# REGIME-GATED SIGNAL FILTER
# ═══════════════════════════════════════════════════════════════════════════════
class RegimeGatedFilter:
    """
    Suppresses signals that are statistically invalid in the current market regime.
    Rules derived from empirical research on signal reliability by regime.
    """
    # Which signal types are VALID in each regime
    VALID_SIGNALS = {
        'BULL':      ['breakout', 'momentum', 'ma_cross', 'obv', 'rvi', 'macd'],
        'BEAR':      ['breakdown', 'mean_reversion', 'rsi_oversold', 'divergence'],
        'SIDEWAYS':  ['mean_reversion', 'rsi_oversold', 'rsi_overbought', 'bb_bounce', 'vwap_mean_rev'],
        # Rule-based sub-regimes
        'VOLATILITY SQUEEZE':  ['breakout_watch', 'vwap_mean_rev'],
        'HIGH EXPANSION':      ['breakout', 'momentum', 'trend_follow'],
        'EXPANSION':           ['breakout', 'momentum', 'ma_cross'],
        'CONTRACTION':         ['mean_reversion', 'vwap_mean_rev'],
        'MEAN REVERSION':      ['mean_reversion', 'bb_bounce', 'rsi_oversold', 'rsi_overbought'],
    }

    # Signal suppression messages
    SUPPRESSION_REASONS = {
        'VOLATILITY SQUEEZE': (
            "⚠ SQUEEZE ACTIVE: RSI and MACD signals suppressed — indicators are "
            "meaningless in compressed volatility. Watch for breakout confirmation first."
        ),
        'HIGH EXPANSION': (
            "⚠ HIGH EXPANSION: Mean-reversion signals (RSI oversold, BB lower touch) "
            "suppressed — trending markets rarely revert. Follow momentum only."
        ),
    }

    def filter(self, signals: list, regime: str, hmm_regime: str = None) -> dict:
        """
        signals: list of signal type strings
        regime:  Market_Regime from rule-based (e.g. 'VOLATILITY SQUEEZE')
        hmm_regime: HMM label ('BULL' / 'BEAR' / 'SIDEWAYS')

        Returns:
          'valid'     — list of signals that pass the gate
          'suppressed'— list of signals that were gated out
          'reason'    — suppression reason message
          'gate_active'— bool
          'regime_used'— which regime label was primary
        """
        valid_for_rule = set(self.VALID_SIGNALS.get(regime, []))
        valid_for_hmm  = set(self.VALID_SIGNALS.get(hmm_regime or 'SIDEWAYS', []))

        # Require agreement between HMM and rule-based for signal to pass
        combined_valid = valid_for_rule | valid_for_hmm

        # Hard suppression rules by regime
        suppress_types = set()
        if regime == 'VOLATILITY SQUEEZE':
            suppress_types = {'rsi_oversold', 'rsi_overbought', 'macd', 'momentum'}
        elif regime == 'HIGH EXPANSION':
            suppress_types = {'mean_reversion', 'rsi_oversold', 'bb_bounce', 'vwap_mean_rev'}

        passed    = [s for s in signals if s not in suppress_types]
        suppressed = [s for s in signals if s in suppress_types]

        gate_active = len(suppressed) > 0
        reason = self.SUPPRESSION_REASONS.get(regime, '') if gate_active else ''

        # Kelly adjustment factor based on regime confidence
        kelly_modifier = 1.0
        if regime == 'VOLATILITY SQUEEZE':  kelly_modifier = 0.25
        elif hmm_regime == 'BEAR':          kelly_modifier = 0.50
        elif hmm_regime == 'BULL':          kelly_modifier = 1.00
        else:                               kelly_modifier = 0.75

        return {
            'valid':        passed,
            'suppressed':   suppressed,
            'reason':       reason,
            'gate_active':  gate_active,
            'regime_used':  f'{hmm_regime or "?"} (HMM) / {regime} (rule)',
            'kelly_mod':    kelly_modifier,
        }


REGIME_FILTER = RegimeGatedFilter()


# ═══════════════════════════════════════════════════════════════════════════════
# KELLY CRITERION CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                    kelly_mod: float = 1.0) -> dict:
    """
    Full Kelly and Half-Kelly position sizing.
    win_rate : fraction 0-1 (e.g. 0.58 = 58% win rate)
    avg_win  : average winning trade return (e.g. 0.08 = 8%)
    avg_loss : average losing trade return, positive (e.g. 0.04 = 4%)
    kelly_mod: regime modifier from RegimeGatedFilter (0.25-1.0)
    """
    try:
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return {'kelly': 0, 'half_kelly': 0, 'quarter_kelly': 0,
                    'interpretation': 'Insufficient data for Kelly calculation'}
        R = avg_win / avg_loss          # reward-to-risk ratio
        K = win_rate - (1 - win_rate) / R   # full Kelly fraction
        K = max(0.0, min(K, 0.25))     # cap at 25% (never bet the farm)
        K_regime = K * kelly_mod
        half_k   = K_regime / 2
        quarter_k = K_regime / 4

        if K_regime > 0.15:
            qual = 'STRONG EDGE — but use Half-Kelly for risk management'
        elif K_regime > 0.08:
            qual = 'MODERATE EDGE — Half-Kelly recommended'
        elif K_regime > 0.02:
            qual = 'MARGINAL EDGE — Quarter-Kelly, small size only'
        else:
            qual = 'NO STATISTICAL EDGE — stay flat or paper trade'

        return {
            'kelly':          round(K_regime * 100, 1),
            'half_kelly':     round(half_k * 100, 1),
            'quarter_kelly':  round(quarter_k * 100, 1),
            'raw_kelly':      round(K * 100, 1),
            'regime_mod':     kelly_mod,
            'reward_risk':    round(R, 2),
            'interpretation': (
                f"Full Kelly: {K_regime*100:.1f}% of portfolio | "
                f"Half-Kelly (recommended): {half_k*100:.1f}% | "
                f"Regime modifier: {kelly_mod:.2f}x. {qual}."
            ),
        }
    except Exception as exc:
        return {'kelly': 0, 'half_kelly': 0, 'quarter_kelly': 0,
                'interpretation': f'Kelly error: {exc}'}

# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND WORKER THREADS
# ═══════════════════════════════════════════════════════════════════════════════
class FetchThread(QThread):
    result = pyqtSignal(str, object)

    def __init__(self, ticker, period, interval):
        super().__init__()
        self.ticker   = ticker
        self.period   = period
        self.interval = interval

    def run(self):
        df = DE.fetch(self.ticker, self.period, self.interval)
        self.result.emit(self.ticker, df)


class StreamThread(QThread):
    chunk = pyqtSignal(str)
    done  = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, messages, system=None, temperature=0.3, max_tokens=2500):
        super().__init__()
        self.messages    = messages
        self.system      = system
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def run(self):
        try:
            for tok in LM.stream(self.messages, self.system, self.temperature, self.max_tokens):
                self.chunk.emit(tok)
        except Exception as exc:
            self.error.emit(str(exc))
        self.done.emit()


class WatchlistThread(QThread):
    update = pyqtSignal(str, float, float)

    def __init__(self, tickers, refresh_sec=45):
        super().__init__()
        self.tickers     = tickers
        self.refresh_sec = refresh_sec
        self._stop       = False
        self.daemon = True

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            for t in list(self.tickers):
                if self._stop:
                    return
                price, pct = DE.fast_quote(t)
                if price is not None:
                    self.update.emit(t, price, pct)
                time.sleep(0.4)
            time.sleep(self.refresh_sec)


class NewsThread(QThread):
    ready = pyqtSignal(list)

    def __init__(self, feeds=None):
        super().__init__()
        self.feeds = feeds or RSS_FEEDS

    def run(self):
        items = []
        if not HAS_FEED:
            self.ready.emit(items)
            return
        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:12]:
                    items.append({
                        'title':   e.get('title', '').strip(),
                        'summary': e.get('summary', '').strip()[:400],
                        'link':    e.get('link', ''),
                        'time':    e.get('published', ''),
                        'source':  feed.feed.get('title', url)[:20],
                    })
            except Exception:
                pass
        self.ready.emit(items)


class LMPingThread(QThread):
    """Checks LM Studio connectivity every 10 s."""
    def run(self):
        while True:
            LM.check()
            time.sleep(10)


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════
class DocParser:
    @staticmethod
    def parse(path):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.pdf':
                if not HAS_PDF:
                    return "⚠ pdfplumber not installed.\n\nRun:  pip install pdfplumber"
                with pdfplumber.open(path) as pdf:
                    pages = [p.extract_text() or '' for p in pdf.pages]
                    return '\n\n'.join(pages)
            elif ext in ('.txt', '.md', '.log', '.rst'):
                with open(path, 'r', errors='ignore') as f:
                    return f.read()
            elif ext == '.csv':
                df = pd.read_csv(path)
                return f"CSV — {len(df)} rows × {len(df.columns)} cols\n\nColumns: {list(df.columns)}\n\n{df.to_string(max_rows=80)}"
            elif ext == '.json':
                with open(path) as f:
                    return json.dumps(json.load(f), indent=2)[:15000]
            else:
                return f"Unsupported file type: {ext}"
        except Exception as exc:
            return f"Error reading file:\n{traceback.format_exc()}"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: stream text into a QTextEdit
# ═══════════════════════════════════════════════════════════════════════════════
def _fix_encoding(tok):
    """LM Studio returns proper UTF-8, no need to force encode/decode."""
    return tok


def _clean_token(tok):
    """Encoding fix only — keep Markdown markers for final render."""
    return _fix_encoding(tok)


def md_to_html(text: str) -> str:
    """Convert Markdown / AI response text to styled HTML for QTextEdit."""
    import re as _re
    ac  = C['accent'];  ac2 = C['accent2']; tx  = C['text']
    tx3 = C['text3'];   bg2 = C['bg2'];     bg3 = C['bg3']
    bg4 = C['bg4'];     brd = C['border2']; yw  = C['yellow']

    lines = text.split('\n')
    out   = []
    i     = 0

    def esc(s):
        # Only escape if not already part of an HTML entity to avoid double-escaping
        s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return s.replace('&amp;gt;', '>').replace('&amp;lt;', '<')

    def inline(s):
        s = _re.sub(r'`([^`]+)`',
            lambda m: (f'<code style="background:{bg4};color:{ac2};padding:1px 4px;'
                       f'border-radius:3px;font-family:Consolas,monospace;">'
                       f'{esc(m.group(1))}</code>'), s)
        s = _re.sub(r'\*\*\*(.+?)\*\*\*',
            lambda m: f'<b><i>{m.group(1)}</i></b>', s)
        s = _re.sub(r'(\*\*|__)(.+?)\1',
            lambda m: f'<b style="color:{ac2};">{m.group(2)}</b>', s)
        s = _re.sub(r'(\*|_)(?!\s)(.+?)(?<!\s)\1',
            lambda m: f'<i>{m.group(2)}</i>', s)
        return s

    def latex_to_text(expr):
        expr = _re.sub(r'\\text\{([^}]+)\}',            r'\1',         expr)
        expr = _re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)',  expr)
        expr = _re.sub(r'\\left[\(\[]',  '(',  expr)
        expr = _re.sub(r'\\right[\)\]]', ')',  expr)
        expr = _re.sub(r'\\times',  '\u00d7', expr)
        expr = _re.sub(r'\\cdot',   '\u00b7', expr)
        expr = _re.sub(r'\\sum',    '\u03a3', expr)
        expr = _re.sub(r'\\sqrt\{([^}]+)\}', lambda m: f'\u221a({m.group(1)})', expr)
        for pat, rep in [
            (r'\\alpha', '\u03b1'), (r'\\beta',  '\u03b2'),
            (r'\\gamma', '\u03b3'), (r'\\sigma', '\u03c3'),
            (r'\\mu',    '\u03bc'), (r'\\pi',    '\u03c0'),
            (r'\\infty', '\u221e'),
        ]:
            expr = _re.sub(pat, rep, expr)
        expr = _re.sub(r'\\[a-zA-Z]+', '', expr)
        expr = _re.sub(r'[{}]',         '', expr)
        return expr.strip()

    def math_block(expr):
        return (f'<div style="background:{bg4};border-left:3px solid {yw};'
                f'padding:8px 14px;margin:6px 0;border-radius:4px;'
                f'font-family:Consolas,monospace;color:{yw};font-size:12px;">'
                f'{esc(latex_to_text(expr))}</div>')

    def inline_math(s):
        def _r(m):
            return (f'<span style="background:{bg4};color:{yw};padding:1px 4px;'
                    f'border-radius:3px;font-family:Consolas,monospace;">'
                    f'{esc(latex_to_text(m.group(1)))}</span>')
        return _re.sub(r'\$([^$\n]+?)\$', _r, s)

    def is_table(ln):
        return '|' in ln and ln.strip().startswith('|')

    while i < len(lines):
        raw = lines[i]
        s   = raw.strip()

        if not s:
            out.append('<div style="height:4px;"></div>'); i += 1; continue

        # fenced code block
        if s.startswith('```'):
            lang = s[3:].strip(); i += 1; code_lines = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(esc(lines[i])); i += 1
            i += 1
            lbl = f'<span style="color:{tx3};font-size:10px;">{lang} </span>' if lang else ''
            out.append(
                f'<div style="background:{bg4};border:1px solid {brd};border-radius:6px;'
                f'padding:10px 14px;margin:6px 0;">{lbl}'
                f'<pre style="margin:0;color:{ac2};font-family:Consolas,monospace;'
                f'font-size:11px;white-space:pre-wrap;">' + '\n'.join(code_lines) + '</pre></div>')
            continue

        # display math $$...$$
        if s.startswith('$$'):
            after = s[2:].rstrip('$').strip()
            if after and s.endswith('$$') and len(s) > 4:
                out.append(math_block(after)); i += 1; continue
            math_lines = [after] if after else []
            i += 1
            while i < len(lines):
                ml = lines[i].strip()
                if ml == '$$' or ml.endswith('$$'):
                    if ml != '$$': math_lines.append(ml.rstrip('$'))
                    i += 1; break
                math_lines.append(ml); i += 1
            out.append(math_block(' '.join(math_lines))); continue

        # horizontal rule
        if _re.match(r'^[-*_]{3,}$', s) or _re.match(r'^[\u2500\u2550\u2501]{4,}$', s):
            out.append(f'<hr style="border:none;border-top:1px solid {brd};margin:8px 0;">')
            i += 1; continue

        # ATX headings
        hm = _re.match(r'^(#{1,4})\s+(.+)$', s)
        if hm:
            lvl   = len(hm.group(1))
            title = inline(inline_math(esc(hm.group(2))))
            sz    = {1: '17px', 2: '15px', 3: '13px', 4: '12px'}.get(lvl, '12px')
            cl    = ac if lvl <= 2 else ac2
            mt    = '14px' if lvl <= 2 else '8px'
            out.append(f'<p style="color:{cl};font-size:{sz};font-weight:bold;'
                       f'margin-top:{mt};margin-bottom:3px;">{title}</p>')
            i += 1; continue

        # **ALL-CAPS** bold section header
        if _re.match(r'^\*\*[A-Z][A-Z &/\-]+\*\*$', s):
            out.append(f'<p style="color:{ac};font-size:13px;font-weight:bold;'
                       f'margin-top:12px;margin-bottom:3px;letter-spacing:1px;">'
                       f'{esc(s.strip("*"))}</p>')
            i += 1; continue

        # table
        if is_table(s):
            tbl_lines = []
            while i < len(lines) and is_table(lines[i].strip()):
                tbl_lines.append(lines[i]); i += 1
            rows = [[c.strip() for c in tl.strip().strip('|').split('|')]
                    for tl in tbl_lines]
            rows = [r for r in rows if not all(_re.match(r'^[:\-]+$', c) for c in r)]
            if rows:
                tbl = (f'<table style="width:100%;border-collapse:collapse;'
                       f'margin:8px 0;font-size:11px;">')
                for ri, row in enumerate(rows):
                    bg  = bg3 if ri == 0 else (bg2 if ri % 2 == 0 else bg4)
                    fw  = 'bold' if ri == 0 else 'normal'
                    tcl = ac2   if ri == 0 else tx
                    tbl += f'<tr style="background:{bg};">'
                    for cell in row:
                        tbl += (f'<td style="padding:5px 10px;border:1px solid {brd};'
                                f'color:{tcl};font-weight:{fw};">'
                                f'{inline(inline_math(esc(cell)))}</td>')
                    tbl += '</tr>'
                tbl += '</table>'
                out.append(tbl)
            continue

        # bullet list
        if _re.match(r'^[-*+\u25cf\u2022]\s', s):
            lst = '<ul style="margin:4px 0 4px 16px;padding:0;">'
            while i < len(lines) and _re.match(r'^[-*+\u25cf\u2022]\s', lines[i].strip()):
                item = lines[i].strip()[2:].strip()
                lst += (f'<li style="color:{tx};margin:2px 0;list-style-type:none;'
                        f'padding-left:4px;"><span style="color:{ac};">\u25b8</span> '
                        f'{inline(inline_math(esc(item)))}</li>')
                i += 1
            lst += '</ul>'; out.append(lst); continue

        # numbered list
        if _re.match(r'^\d+[.)]\s', s):
            lst = '<ol style="margin:4px 0 4px 16px;padding:0;list-style:none;">'; n = 1
            while i < len(lines) and _re.match(r'^\d+[.)]\s', lines[i].strip()):
                item = _re.sub(r'^\d+[.)]\s+', '', lines[i].strip())
                lst += (f'<li style="color:{tx};margin:2px 0;padding-left:4px;">'
                        f'<span style="color:{ac2};font-weight:bold;">{n}.</span> '
                        f'{inline(inline_math(esc(item)))}</li>')
                n += 1; i += 1
            lst += '</ol>'; out.append(lst); continue

        # KEY: value line
        kv = _re.match(r'^([A-Z][A-Z0-9 /&,()\-]{1,40}):\s*(.+)$', s)
        if kv:
            out.append(f'<p style="margin:2px 0;color:{tx};">'
                       f'<span style="color:{ac2};font-weight:bold;">{esc(kv.group(1))}:</span> '
                       f'{inline(inline_math(esc(kv.group(2))))}</p>')
            i += 1; continue

        # plain paragraph
        out.append(f'<p style="margin:3px 0;color:{tx};line-height:1.55;">'
                   f'{inline(inline_math(esc(s)))}</p>')
        i += 1

    return (f'<div style="font-family:Consolas,\'Courier New\',monospace;'
            f'font-size:12px;color:{tx};padding:2px 0;line-height:1.5;">'
            + ''.join(out) + '</div>')


def stream_into(widget, messages, system=None, temperature=0.3, max_tokens=2500,
                on_done=None, clear=True):
    """Stream tokens live, then render full Markdown HTML on completion."""
    if clear:
        widget.clear()
    t   = StreamThread(messages, system, temperature, max_tokens)
    buf = ['']

    def _chunk(tok):
        tok = _fix_encoding(tok)
        buf[0] += tok
        cur = widget.textCursor()
        cur.movePosition(cur.MoveOperation.End if hasattr(cur, 'MoveOperation') else cur.End)
        cur.insertText(tok)
        widget.setTextCursor(cur)
        widget.ensureCursorVisible()

    def _finished():
        if buf[0].strip():
            widget.clear()
            widget.setHtml(md_to_html(buf[0]))
            sb = widget.verticalScrollBar()
            sb.setValue(sb.maximum())
        if on_done:
            on_done()

    t.chunk.connect(_chunk)
    t.done.connect(_finished)
    t.start()
    return t


class HeaderBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(54)
        self.setStyleSheet(
            f"background: {C['bg2']}; border-bottom: 2px solid {C['accent']};"
        )
        self.current_tf = "1Y"

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(10)

        # ── Logo ──────────────────────────────────────────────────────
        logo = QLabel(f"◈  {APP}")
        logo.setStyleSheet(
            f"color:{C['accent']}; font-size:18px; font-weight:bold; letter-spacing:2.5px;"
        )
        lay.addWidget(logo)

        ver_lbl = QLabel(VER)
        ver_lbl.setStyleSheet(f"color:{C['text3']}; font-size:9px;")
        lay.addWidget(ver_lbl)

        self._div(lay)

        # ── Ticker input ──────────────────────────────────────────────
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("TICKER...")
        self.ticker_input.setFixedSize(115, 34)
        self.ticker_input.returnPressed.connect(self._emit_ticker)
        self.ticker_input.setStyleSheet(
            f"font-size:15px; font-weight:bold; color:{C['accent2']};"
            f"background:{C['bg3']}; border:1px solid {C['accent']};"
            f"padding:4px 10px; letter-spacing:1.5px;"
        )
        lay.addWidget(self.ticker_input)

        go_btn = self._accent_btn("GO", 42, 34)
        go_btn.clicked.connect(self._emit_ticker)
        lay.addWidget(go_btn)

        self._div(lay)

        # ── Timeframe selector ────────────────────────────────────────
        tf_lbl = QLabel("TF :")
        tf_lbl.setStyleSheet(f"color:{C['text3']}; font-size:10px; font-weight:bold;")
        lay.addWidget(tf_lbl)

        self.tf_btns = QButtonGroup(self)
        for tf in ["1D", "5D", "1M", "3M", "1Y", "5Y"]:
            btn = QPushButton(tf)
            btn.setCheckable(True)
            btn.setFixedSize(36, 28)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:{C['bg3']}; border:1px solid {C['border']};
                    color:{C['text2']}; font-size:10px; font-weight:bold; padding:0;
                }}
                QPushButton:checked {{
                    background:{C['accent']}; color:white; border-color:{C['accent']};
                }}
                QPushButton:hover:!checked {{ color:{C['text']}; border-color:{C['border2']}; }}
            """)
            if tf == "1Y":
                btn.setChecked(True)
            self.tf_btns.addButton(btn)
            lay.addWidget(btn)
            btn.clicked.connect(lambda _, t=tf: self._emit_tf(t))

        lay.addStretch()

        # ── Mini market bar ───────────────────────────────────────────
        self.mkt_lbl = QLabel("SPY: ─  |  QQQ: ─  |  VIX: ─")
        self.mkt_lbl.setStyleSheet(f"color:{C['text2']}; font-size:10px;")
        lay.addWidget(self.mkt_lbl)

        self._div(lay)

        # ── LM Studio indicator ───────────────────────────────────────
        self.lm_dot = QLabel("●")
        self.lm_dot.setStyleSheet(f"color:{C['text3']}; font-size:16px;")
        self.lm_lbl = QLabel("LM Studio…")
        self.lm_lbl.setStyleSheet(f"color:{C['text3']}; font-size:10px;")
        lay.addWidget(self.lm_dot)
        lay.addWidget(self.lm_lbl)

        self._div(lay)

        # ── Clock ─────────────────────────────────────────────────────
        self.clock_lbl = QLabel()
        self.clock_lbl.setStyleSheet(
            f"color:{C['accent2']}; font-size:12px; font-weight:bold; letter-spacing:1px;"
        )
        lay.addWidget(self.clock_lbl)

        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(1000)
        self._tick()

        BUS.lm_status.connect(self._on_lm)

        # Refresh mini market bar every 5 min
        self._mkt_timer = QTimer(self)
        self._mkt_timer.timeout.connect(self._refresh_mkt)
        self._mkt_timer.start(300_000)
        QTimer.singleShot(2000, self._refresh_mkt)

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _div(lay):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color:{C['border2']}; max-width:1px;")
        lay.addWidget(sep)

    @staticmethod
    def _accent_btn(text, w, h):
        btn = QPushButton(text)
        btn.setFixedSize(w, h)
        btn.setProperty("accent", "true")
        btn.setStyleSheet(f"""
            QPushButton {{
                background:{C['accent']}; border:1px solid {C['accent']};
                color:white; font-weight:bold; font-size:12px;
            }}
            QPushButton:hover {{ background:{C['accent2']}; }}
            QPushButton:pressed {{ background:#cc4400; }}
        """)
        return btn

    def _emit_ticker(self):
        t = self.ticker_input.text().strip().upper()
        if t:
            BUS.ticker_changed.emit(t)

    def _emit_tf(self, tf):
        self.current_tf = tf
        BUS.tf_changed.emit(tf)

    def _tick(self):
        now = datetime.datetime.now()
        self.clock_lbl.setText(now.strftime("%H:%M:%S   %d %b %Y"))

    def _on_lm(self, ok, model):
        if ok:
            name = model.split('/')[-1][:22] if '/' in model else model[:22]
            self.lm_dot.setStyleSheet(f"color:{C['green']}; font-size:16px;")
            self.lm_lbl.setStyleSheet(f"color:{C['green']}; font-size:10px;")
            self.lm_lbl.setText(f"LM ● {name}")
        else:
            self.lm_dot.setStyleSheet(f"color:{C['red']}; font-size:16px;")
            self.lm_lbl.setStyleSheet(f"color:{C['red']}; font-size:10px;")
            self.lm_lbl.setText("LM Studio: OFFLINE")

    def _refresh_mkt(self):
        class MktThread(QThread):
            done_sig = pyqtSignal(str)
            def run(self):
                parts =[]
                for sym, label in[("SPY","SPY"),("QQQ","QQQ"),("^VIX","VIX")]:
                    p, pct = DE.fast_quote(sym)
                    if p is not None:
                        sign = '+' if pct >= 0 else ''
                        col_code = C['green'] if pct >= 0 else C['red']
                        parts.append(
                            f"<span style='color:{C['text2']};'>{label}: </span>"
                            f"<span style='color:{col_code}; font-weight:bold;'>"
                            f"{p:,.2f} ({sign}{pct:.2f}%)</span>"
                        )
                    else:
                        parts.append(f"<span style='color:{C['text2']};'>{label}: ─</span>")
                self.done_sig.emit("  |  ".join(parts))

        # Start the Qt thread safely
        self._mkt_worker = MktThread()
        self._mkt_worker.done_sig.connect(self.mkt_lbl.setText)
        self._mkt_worker.start()


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHLIST PANEL
# ═══════════════════════════════════════════════════════════════════════════════
class WatchlistPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(200)
        self.setMaximumWidth(240)
        self.tickers = list(WATCHLIST_DEFAULT)
        self._setup_ui()
        self._start_thread()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QLabel("  ◈ WATCHLIST")
        hdr.setFixedHeight(32)
        hdr.setStyleSheet(
            f"background:{C['bg3']}; color:{C['accent']}; font-size:10px;"
            f"font-weight:bold; letter-spacing:1px; border-bottom:1px solid {C['border']};"
        )
        lay.addWidget(hdr)

        add_row = QWidget()
        add_row.setFixedHeight(36)
        add_row.setStyleSheet(f"background:{C['bg3']}; border-bottom:1px solid {C['border']};")
        ar = QHBoxLayout(add_row)
        ar.setContentsMargins(4, 4, 4, 4)
        ar.setSpacing(4)
        self.add_inp = QLineEdit()
        self.add_inp.setPlaceholderText("Add ticker…")
        self.add_inp.setFixedHeight(26)
        self.add_inp.returnPressed.connect(self._add)
        ar.addWidget(self.add_inp)
        ab = QPushButton("+")
        ab.setFixedSize(26, 26)
        ab.clicked.connect(self._add)
        ar.addWidget(ab)
        lay.addWidget(add_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["SYM", "PRICE", "%CHG"])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("QTableWidget { font-size:11px; }")
        self.table.itemDoubleClicked.connect(self._select)
        
        # 👇 NEW: Right-Click Context Menu logic
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        lay.addWidget(self.table)

        self._rebuild()

    def _show_context_menu(self, pos):
        """Displays a right-click menu to remove a ticker."""
        item = self.table.itemAt(pos)
        if not item: return
        
        row = item.row()
        ticker_item = self.table.item(row, 0)
        if not ticker_item: return
        
        ticker = ticker_item.text()
        
        # Build the sleek dark-mode menu
        menu = QMenu(self)
        remove_action = menu.addAction(f"✕  Remove {ticker}")
        
        # Show the menu exactly where the mouse clicked
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        
        if action == remove_action:
            self._remove(ticker)

    def _remove(self, ticker):
        """Safely removes the ticker from memory and the background thread."""
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            self._rebuild()
            self.wt.tickers = self.tickers  # Update the live-feed thread
            BUS.status_msg.emit(f"Removed {ticker} from watchlist")

    def _rebuild(self):
        self.table.setRowCount(len(self.tickers))
        for i, t in enumerate(self.tickers):
            self.table.setItem(i, 0, QTableWidgetItem(t))
            self.table.setItem(i, 1, QTableWidgetItem("─"))
            self.table.setItem(i, 2, QTableWidgetItem("─"))
            self.table.setRowHeight(i, 28)

    def _add(self):
        t = self.add_inp.text().strip().upper()
        if t and t not in self.tickers:
            self.tickers.append(t)
            self._rebuild()
            self.wt.tickers = self.tickers
        self.add_inp.clear()

    def _start_thread(self):
        self.wt = WatchlistThread(self.tickers)
        self.wt.update.connect(self._update)
        self.wt.start()

    def _update(self, ticker, price, pct):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == ticker:
                col = C['green'] if pct >= 0 else C['red']
                sign = '+' if pct >= 0 else ''

                pi = QTableWidgetItem(f"{price:,.2f}")
                pi.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                pi.setForeground(QColor(col))

                ci = QTableWidgetItem(f"{sign}{pct:.2f}%")
                ci.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                ci.setForeground(QColor(col))

                self.table.setItem(row, 1, pi)
                self.table.setItem(row, 2, ci)
                break

    def _select(self, item):
        row = item.row()
        t = self.table.item(row, 0)
        if t:
            BUS.ticker_changed.emit(t.text())


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL CHART WIDGET
# ═══════════════════════════════════════════════════════════════════════════════
class TechnicalChart(QWidget):
    """Full Bloomberg-style chart: candles, volume, RSI, MACD, BB, MAs."""

    def __init__(self):
        super().__init__()
        self.ticker   = None
        self.df       = None
        self.period   = "1y"
        self.interval = "1d"
        # Indicator toggles
        self.opt = dict(ma=True, bb=True, vol=True, rsi=True, macd=False,
                        vwap=True, atr=False, obv=False, cmf=False, rvi=False, scalp=False, gbm=False, maxp=False, fvg=False, kalman=False, gp=False, ptt=False, fft=False, vmd=False)
        self._threads = []
        self._setup_ui()
        BUS.ticker_changed.connect(self._on_ticker)
        BUS.tf_changed.connect(self._on_tf)

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────
        tb1 = QWidget()
        tb1.setFixedHeight(36)
        tb1.setStyleSheet(f"background:{C['bg3']}; border-bottom:1px solid {C['border2']};")
        tb1_lay = QHBoxLayout(tb1)
        tb1_lay.setContentsMargins(10, 0, 6, 0)
        tb1_lay.setSpacing(8)

        self.title_lbl = QLabel("─── SELECT A TICKER ───")
        self.title_lbl.setStyleSheet(
            f"color:{C['text3']}; font-size:15px; font-weight:bold; letter-spacing:1.5px;"
        )
        tb1_lay.addWidget(self.title_lbl)
        self.price_lbl = QLabel()
        self.price_lbl.setStyleSheet(f"color:{C['text']}; font-size:14px; font-weight:bold;")
        tb1_lay.addWidget(self.price_lbl)
        self.chg_lbl = QLabel()
        self.chg_lbl.setStyleSheet("font-size:13px; font-weight:bold;")
        tb1_lay.addWidget(self.chg_lbl)
        tb1_lay.addStretch()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(lambda: self._fetch() if self.ticker else None)
        self._auto_on = False

        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setToolTip("Refresh chart now")
        self._refresh_btn.setFixedSize(30, 26)
        self._refresh_btn.setStyleSheet(
            f"QPushButton{{background:{C['bg4']};border:1px solid {C['border']};"
            f"color:{C['accent2']};font-size:15px;font-weight:bold;padding:0;}}"
            f"QPushButton:hover{{background:{C['accent']};color:white;}}"
        )
        self._refresh_btn.clicked.connect(lambda: self._fetch() if self.ticker else None)
        tb1_lay.addWidget(self._refresh_btn)

        self._auto_btn = QPushButton("AUTO: OFF")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setFixedHeight(26)
        self._auto_btn.setStyleSheet(
            f"QPushButton{{background:{C['bg4']};border:1px solid {C['border']};"
            f"color:{C['text2']};font-size:10px;padding:0 8px;font-weight:bold;}}"
            f"QPushButton:checked{{background:{C['green']};color:#000;border-color:{C['green']};}}"
            f"QPushButton:hover:!checked{{background:{C['accent']};color:white;}}"
        )
        self._auto_btn.toggled.connect(self._toggle_auto)
        tb1_lay.addWidget(self._auto_btn)

        self._interval_cb = QComboBox()
        for lbl in ["30s", "1m", "5m", "15m"]:
            self._interval_cb.addItem(lbl)
        self._interval_cb.setCurrentIndex(0)
        self._interval_cb.setFixedHeight(26)
        self._interval_cb.setFixedWidth(56)
        self._interval_cb.setStyleSheet(
            f"QComboBox{{background:{C['bg4']};border:1px solid {C['border']};"
            f"color:{C['text2']};font-size:10px;padding:0 4px;}}"
            f"QComboBox::drop-down{{border:none;width:16px;}}"
        )
        self._interval_cb.currentTextChanged.connect(self._update_auto_interval)
        tb1_lay.addWidget(self._interval_cb)
        lay.addWidget(tb1)

        tb2 = QWidget()
        tb2.setFixedHeight(28)
        tb2.setStyleSheet(f"background:{C['bg3']}; border-bottom:1px solid {C['border']};")
        tb2_lay = QHBoxLayout(tb2)
        tb2_lay.setContentsMargins(10, 0, 10, 0)
        tb2_lay.setSpacing(12)
        for label, key in[("MA","ma"),("BB","bb"),("VOL","vol"),("RSI","rsi"),("MACD","macd"),
                   ("VWAP","vwap"),("ATR","atr"),("OBV","obv"),("CMF","cmf"),("RVI","rvi"), ("SCALP","scalp"), ("GBM","gbm"), ("MAXP","maxp"), ("FVG","fvg"), ("KALMAN","kalman"), ("GP FORECAST","gp"), ("PRO TREND","ptt"), ("FFT CYCLE","fft"), ("VMD TARGET","vmd")]:
            cb = QCheckBox(label)
            cb.setChecked(self.opt[key])
            cb.stateChanged.connect(lambda s, k=key: self._toggle(k, s))
            tb2_lay.addWidget(cb)
        tb2_lay.addStretch()
        lay.addWidget(tb2)

        # ── Figure ────────────────────────────────────────────────────
        self.fig    = Figure(facecolor=C['bg'])
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background:{C['bg']};")
        lay.addWidget(self.canvas)

        self._draw_placeholder()

    def _toggle_auto(self, checked):
        self._auto_on = checked
        self._auto_btn.setText("AUTO: ON" if checked else "AUTO: OFF")
        if checked:
            self._update_auto_interval(self._interval_cb.currentText())
        else:
            self._refresh_timer.stop()

    def _update_auto_interval(self, label):
        secs = {"30s": 30, "1m": 60, "5m": 300, "15m": 900}.get(label, 60)
        if self._auto_on:
            self._refresh_timer.start(secs * 1000)

    def _start_price_ticker(self):
        if hasattr(self, '_price_timer'):
            self._price_timer.stop()
        self._price_timer = QTimer(self)
        self._price_timer.timeout.connect(self._update_live_price)
        self._price_timer.start(15_000)

    def _update_live_price(self):
        if not self.ticker:
            return
        import threading
        ticker = self.ticker
        def _go():
            try:
                import yfinance as yf
                fi    = yf.Ticker(ticker).fast_info
                price = getattr(fi, 'last_price', None) or getattr(fi, 'regularMarketPrice', None)
                prev  = getattr(fi, 'previous_close', None) or getattr(fi, 'regularMarketPreviousClose', None)
                if price is None:
                    return
                change = (price - prev) if prev else 0.0
                pct    = (change / prev * 100) if prev else 0.0
                QMetaObject.invokeMethod(
                    self, '_apply_live_price',
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(float, float(price)),
                    Q_ARG(float, float(change)),
                    Q_ARG(float, float(pct)),
                )
            except Exception:
                pass
        threading.Thread(target=_go, daemon=True).start()

    @pyqtSlot(float, float, float)
    def _apply_live_price(self, price: float, change: float, pct: float):
        try:
            self.price_lbl.setText(f"{price:,.2f}")
            colour = C['green'] if change >= 0 else C['red']
            sign   = '+' if change >= 0 else ''
            self.chg_lbl.setText(f"  {sign}{change:,.2f}  ({sign}{pct:.2f}%)")
            self.chg_lbl.setStyleSheet(f"font-size:13px; font-weight:bold; color:{colour};")
            BUS.status_msg.emit(
                f"{self.ticker}  |  live {price:,.2f}  {sign}{change:,.2f} ({sign}{pct:.2f}%)"
            )
        except Exception:
            pass

    def _toggle(self, key, state):
        is_checked = bool(state)
        self.opt[key] = is_checked
        
        # 🐛 FIX: Guardrail! If there is no data loaded yet, stop here. 
        # The checkbox state is saved, so it will automatically apply when you DO load a ticker.
        if self.df is None:
            return
            
        if key == 'scalp' and is_checked:
            BUS.tf_changed.emit("1Y")
            self._fetch() 
            
        elif key == 'maxp' and is_checked:
            def worker():
                DE.get_max_pain(self.ticker)
                QTimer.singleShot(0, self._render)
            import threading
            threading.Thread(target=worker, daemon=True).start()
            BUS.status_msg.emit("Calculating Options Max Pain...")

        elif key == 'gp' and is_checked:
            # Tell the UI we are doing heavy matrix math
            BUS.status_msg.emit("Inverting Covariance Matrix (Gaussian Process)...")
            self._render() # Render will handle the background thread internally
            
        else:
            self._render()

    def _on_ticker(self, ticker):
        self.ticker = ticker.upper()
        self.title_lbl.setText(f"⟳  {self.ticker}  loading…")
        self.title_lbl.setStyleSheet(
            f"color:{C['text2']}; font-size:15px; font-weight:bold;"
        )
        BUS.status_msg.emit(f"Fetching {self.ticker} — period={self.period}  interval={self.interval}")
        self._fetch()

    def _on_tf(self, tf):
        if tf in TIMEFRAMES:
            self.period, self.interval = TIMEFRAMES[tf]
            if self.ticker:
                self._fetch()

    def _fetch(self):
        t = FetchThread(self.ticker, self.period, self.interval)
        t.result.connect(self._on_data)
        t.start()
        self._threads.append(t)

    def _on_data(self, ticker, df):
        if ticker != self.ticker:
            return
        if df is None or df.empty:
            self.title_lbl.setText(f"✗  {ticker}: no data")
            self.title_lbl.setStyleSheet(f"color:{C['red']}; font-size:15px; font-weight:bold;")
            BUS.status_msg.emit(f"No data for {ticker}")
            return
            
        self.df = DE.add_indicators(df)
        self._render()
        
        # 🚨 SCALPING ALERTS 🚨
        if self.opt.get('scalp', False) and len(self.df) >= 2:
            if self.df['Scalp_Buy'].iloc[-1] or self.df['Scalp_Buy'].iloc[-2]:
                BUS.status_msg.emit(f"🚨 SCALP ALERT: Strong UP Signal detected on {ticker}!")
            elif self.df['Scalp_Sell'].iloc[-1] or self.df['Scalp_Sell'].iloc[-2]:
                BUS.status_msg.emit(f"🚨 SCALP ALERT: Strong DOWN Signal detected on {ticker}!")
            else:
                BUS.status_msg.emit(f"{ticker}  |  {len(df)} bars  |  last {df['Close'].iloc[-1]:,.2f}")
        else:
            BUS.status_msg.emit(f"{ticker}  |  {len(df)} bars  |  last {df['Close'].iloc[-1]:,.2f}")
            
        self._start_price_ticker()

    def _draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(C['bg'])
        ax.text(0.5, 0.5, "Enter a ticker symbol above to load chart",
                ha='center', va='center', color=C['text3'],
                fontsize=13, transform=ax.transAxes)
        for sp in ax.spines.values():
            sp.set_edgecolor(C['border'])
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()

    def get_base64_image(self):
        """Captures the current chart as a Base64 PNG image for Vision AI."""
        import io
        import base64
        buf = io.BytesIO()
        # Save the figure to memory
        self.fig.savefig(buf, format='png', facecolor=self.fig.get_facecolor(), bbox_inches='tight', dpi=100)
        buf.seek(0)
        # Convert to base64 string
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64
    
    def _render(self):
        df  = self.df
        n   = len(df)
        idx = np.arange(n)
        
        # 🚀 FORCE Chart Expansion for GBM, GP, FFT, and VMD
        x_max_val = (n + 35) if (self.opt.get('gbm', False) or self.opt.get('gp', False) or self.opt.get('fft', False) or self.opt.get('vmd', False)) else ((n + 15) if self.opt.get('fvg', False) else (n + 2))

        # ── subplot layout ────────────────────────────────────────────
        ratios = [5]
        if self.opt['vol']:  ratios.append(1)
        if self.opt['rsi']:  ratios.append(1)
        if self.opt['macd']: ratios.append(1)
        if self.opt['obv']:  ratios.append(1)
        if self.opt['cmf']:  ratios.append(1)
        if self.opt['rvi']:  ratios.append(1)
        if self.opt['atr']:  ratios.append(1)
        n_sub = len(ratios)

        self.fig.clear()
        gspec = gridspec.GridSpec(n_sub, 1, figure=self.fig,
                                  height_ratios=ratios, hspace=0.03)
        axes  = [self.fig.add_subplot(gspec[i]) for i in range(n_sub)]
        ax    = axes[0]

        # ── Candlesticks ──────────────────────────────────────────────
        w = 0.55
        for i, row in enumerate(df.itertuples()):
            up  = row.Close >= row.Open
            clr = C['cand_up'] if up else C['cand_dn']
            # body
            ax.bar(i, abs(row.Close - row.Open), w,
                   bottom=min(row.Open, row.Close),
                   color=clr, linewidth=0, zorder=3)
            # wick
            ax.plot([i, i], [row.Low, row.High],
                    color=clr, linewidth=0.7, zorder=2)

        # ── Moving Averages ───────────────────────────────────────────
        if self.opt['ma']:
            for col, clr, lw in [('MA20', C['ma20'], 1.0),
                                   ('MA50', C['ma50'], 1.0),
                                   ('MA200', C['ma200'], 1.0)]:
                if col in df and df[col].notna().any():
                    ax.plot(idx, df[col], color=clr, linewidth=lw,
                            alpha=0.85, label=col, zorder=4)

        # ── KALMAN FILTER ─────────────────────────────────────────────
        if self.opt.get('kalman', False) and 'Kalman' in df:
            # Draw the true state line in a striking magenta
            ax.plot(idx, df['Kalman'], color='#ff00ff', linewidth=1.5, label='Kalman True State', zorder=5)
            
            # Highlight extreme noise deviations (when price disconnects from reality)
            dev = np.abs(df['Close'] - df['Kalman']) / df['Kalman']
            noise_idx = idx[dev > 0.03] # 3% deviation from true state
            if len(noise_idx) > 0:
                ax.scatter(noise_idx, df['Close'].iloc[noise_idx], color='#ff00ff', s=15, zorder=6, alpha=0.6)            

        # ── Bollinger Bands ───────────────────────────────────────────
        if self.opt['bb'] and 'BB_U' in df:
            ax.plot(idx, df['BB_U'], color=C['bb'], linewidth=0.7,
                    alpha=0.5, linestyle='--')
            ax.plot(idx, df['BB_L'], color=C['bb'], linewidth=0.7,
                    alpha=0.5, linestyle='--')
            ax.fill_between(idx, df['BB_L'], df['BB_U'],
                            alpha=0.04, color=C['bb'])


        # ── GBM PROBABILITY CONE (30-Day Forward Projection) ──────────
        if self.opt.get('gbm', False):
            try:
                horizon = 30
                # 1. Calculate log returns to find historical drift (mu) and volatility (sigma)
                log_ret = np.log(df['Close'].astype(float) / df['Close'].astype(float).shift(1)).dropna()
                mu = log_ret.mean()
                sigma = log_ret.std()
                s0 = float(df['Close'].iloc[-1])
                
                fwd_steps = np.arange(1, horizon + 1)
                fwd_idx = np.arange(n, n + horizon)
                
                # 2. Calculate Expected Path and Confidence Intervals (68% and 95%)
                exp_path = s0 * np.exp(mu * fwd_steps)
                up_68 = s0 * np.exp(mu * fwd_steps + 1.00 * sigma * np.sqrt(fwd_steps))
                dn_68 = s0 * np.exp(mu * fwd_steps - 1.00 * sigma * np.sqrt(fwd_steps))
                up_95 = s0 * np.exp(mu * fwd_steps + 1.96 * sigma * np.sqrt(fwd_steps))
                dn_95 = s0 * np.exp(mu * fwd_steps - 1.96 * sigma * np.sqrt(fwd_steps))
                
                # Connect the last actual price to the future cone
                full_idx = np.concatenate([[n-1], fwd_idx])
                
                # 3. Draw the Cones
                ax.plot(full_idx, np.concatenate([[s0], exp_path]), color=C['accent2'], linestyle=':', linewidth=1.5, zorder=3)
                
                ax.fill_between(full_idx, np.concatenate([[s0], dn_68]), np.concatenate([[s0], up_68]), 
                                color=C['blue'], alpha=0.15, zorder=2, edgecolor='none')
                ax.fill_between(full_idx, np.concatenate([[s0], dn_95]), np.concatenate([[s0], up_95]), 
                                color=C['blue'], alpha=0.06, zorder=1, edgecolor='none')
                
                # 4. Add Price Targets at the end of the cone
                ax.text(n + horizon, up_95[-1], f" +2σ (95%): {up_95[-1]:.2f}", color=C['blue'], fontsize=8, va='center')
                ax.text(n + horizon, dn_95[-1], f" -2σ (95%): {dn_95[-1]:.2f}", color=C['blue'], fontsize=8, va='center')
            except Exception as e:
                # Print giant red text if the math fails so we aren't left guessing!
                y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
                ax.text(n/2, y_mid, f"GBM Math Error: {str(e)}", color='red', fontsize=12, fontweight='bold', ha='center', backgroundcolor='black')

        # ── OPTIONS MAX PAIN ──────────────────────────────────────────
        if self.opt.get('maxp', False):
            # Because we fetched and cached it in the background, this returns instantly
            mp_data = DE.get_max_pain(self.ticker)
            if mp_data:
                mp_strike, mp_date = mp_data
                ax.axhline(mp_strike, color=C['yellow'], linewidth=1.5, linestyle='-.', zorder=5)
                
                # Draw a high-visibility badge on the left side of the chart
                ax.text(0, mp_strike, f" MAX PAIN ({mp_date}): ${mp_strike:.2f} ", 
                        color=C['bg'], backgroundcolor=C['yellow'], fontsize=8, fontweight='bold', va='center', zorder=6)

         # ── SMART MONEY CONCEPTS (FAIR VALUE GAPS) ────────────────────
        if self.opt.get('fvg', False):
            import matplotlib.patches as patches
            highs, lows, closes = df['High'].values, df['Low'].values, df['Close'].values
            active_bull_fvgs, active_bear_fvgs = [],[]
            
            for i in range(2, n):
                # 1. Size Filter: Only flag gaps > 0.15% of the stock price to remove retail noise
                min_gap = closes[i] * 0.0015 
                
                # Bullish FVG
                if lows[i] - highs[i-2] > min_gap:
                    active_bull_fvgs.append({'start': i-1, 'top': lows[i], 'bot': highs[i-2], 'active': True})
                # Bearish FVG
                elif lows[i-2] - highs[i] > min_gap:
                    active_bear_fvgs.append({'start': i-1, 'top': lows[i-2], 'bot': highs[i], 'active': True})
                    
                # 2. Invalidation: Mark as filled if price overlaps the gap zone
                for fvg in active_bull_fvgs:
                    if fvg['active'] and lows[i] < fvg['top']:
                        fvg['active'] = False
                for fvg in active_bear_fvgs:
                    if fvg['active'] and highs[i] > fvg['bot']:
                        fvg['active'] = False
                        
            # 3. Clean UI: Only draw the 3 most recent active gaps
            recent_bull = [f for f in active_bull_fvgs if f['active']][-3:]
            recent_bear = [f for f in active_bear_fvgs if f['active']][-3:]
            
            for fvg in recent_bull:
                rect = patches.Rectangle((fvg['start'], fvg['bot']), x_max_val - fvg['start'], fvg['top'] - fvg['bot'], 
                                         linewidth=0, facecolor=C['green'], alpha=0.18, zorder=1)
                ax.add_patch(rect)
                ax.text(x_max_val, (fvg['top']+fvg['bot'])/2, " FVG", color=C['green'], fontsize=8, fontweight='bold', va='center', zorder=2)
                
            for fvg in recent_bear:
                rect = patches.Rectangle((fvg['start'], fvg['bot']), x_max_val - fvg['start'], fvg['top'] - fvg['bot'], 
                                         linewidth=0, facecolor=C['red'], alpha=0.18, zorder=1)
                ax.add_patch(rect)
                ax.text(x_max_val, (fvg['top']+fvg['bot'])/2, " FVG", color=C['red'], fontsize=8, fontweight='bold', va='center', zorder=2)       
           
        # ── GAUSSIAN PROCESS (RBF + PERIODIC) FORECAST ────────────────
        if self.opt.get('gp', False):
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                # 👇 FIX: Removed "as C" so it doesn't overwrite your color dictionary!
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared, WhiteKernel
                
                # 1. Prepare data (limit to last 252 days to avoid matrix explosion)
                lookback = min(252, n)
                X_train = np.arange(n - lookback, n).reshape(-1, 1)
                y_train = df['Close'].iloc[-lookback:].values
                
                # Normalize Y (Critical for math stability)
                y_mean, y_std = y_train.mean(), y_train.std()
                y_train_norm = (y_train - y_mean) / (y_std + 1e-9)

                # 2. Bulletproof Kernel
                # Combines Macro Trend (RBF), Seasonality (ExpSine), and Market Chop (WhiteKernel)
                kernel = ConstantKernel(1.0) * RBF(length_scale=50.0) \
                         + ExpSineSquared(length_scale=1.0, periodicity=63.0) \
                         + WhiteKernel(noise_level=0.5)

                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=False)
                
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(X_train, y_train_norm)
                
                # 3. Predict the past and 30 days into the future
                X_pred = np.arange(n - lookback, n + 30).reshape(-1, 1)
                y_pred_norm, sigma_norm = gp.predict(X_pred, return_std=True)
                
                # Denormalize back to stock prices
                y_pred = y_pred_norm * y_std + y_mean
                sigma = sigma_norm * y_std
                
                # 4. Draw the beautiful curve
                ax.plot(X_pred.flatten(), y_pred, color='#00e676', linestyle='-', linewidth=2, zorder=4)
                ax.fill_between(X_pred.flatten(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, 
                                color='#00e676', alpha=0.15, zorder=3, edgecolor='none')
                
                # Target Label
                end_price = y_pred[-1]
                ax.text(n + 30, end_price, f" GP Target: {end_price:.2f}", color='#00e676', fontsize=8, fontweight='bold', va='center')
                BUS.status_msg.emit("GP Optimization Complete.")
                
            except Exception as e:
                # If the math fails, print the error in giant red letters on the chart so we can see it!
                y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
                ax.text(n/2, y_mid, f"GP Math Error: {str(e)}", color='red', fontsize=12, fontweight='bold', ha='center', backgroundcolor='black')

        # ── FAST FOURIER TRANSFORM (FFT) CYCLE PROJECTION ─────────────
        if self.opt.get('fft', False):
            fft_res = DE.fft_extrapolation(df, horizon=30)
            if fft_res:
                fut_idx = np.arange(n, n + 30)
                fut_path = fft_res['future_path']
                
                # Connect the current price to the future projection smoothly
                full_idx = np.concatenate([[n-1], fut_idx])
                full_path = np.concatenate([[df['Close'].iloc[-1]], fut_path])
                
                # Draw the oscillating cycle wave in vibrant Cyan
                ax.plot(full_idx, full_path, color=C['cyan'], linestyle='-', linewidth=2, label='FFT Cycle', zorder=5)
                
                # Mark the Peak and Trough
                peak_day = n + fft_res['next_peak_days'] - 1
                trough_day = n + fft_res['next_trough_days'] - 1
                
                ax.scatter(peak_day, fft_res['next_peak_price'], color=C['red'], marker='v', s=100, zorder=6)
                ax.text(peak_day, fft_res['next_peak_price'], " PEAK", color=C['red'], fontsize=8, fontweight='bold', va='bottom')
                
                ax.scatter(trough_day, fft_res['next_trough_price'], color=C['green'], marker='^', s=100, zorder=6)
                ax.text(trough_day, fft_res['next_trough_price'], " TROUGH", color=C['green'], fontsize=8, fontweight='bold', va='top')

        # ── VARIATIONAL MODE DECOMPOSITION (VMD) FORECAST ─────────────
        if self.opt.get('vmd', False):
            if not HAS_VMD:
                # If the library is missing, print giant red text!
                y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
                ax.text(n/2, y_mid, "VMD Error: vmdpy library missing! Run: pip install vmdpy", color='red', fontsize=12, fontweight='bold', ha='center', backgroundcolor='black', zorder=20)
            else:
                vmd_res = DE.vmd_extrapolation(df, horizon=30)
                if vmd_res:
                    fut_idx = np.arange(n, n + 30)
                    fut_path = vmd_res['future_path']
                    
                    full_idx = np.concatenate([[n-1], fut_idx])
                    full_path = np.concatenate([[df['Close'].iloc[-1]], fut_path])
                    
                    # Draw VMD in a striking stark white line to contrast with FFT/GP
                    ax.plot(full_idx, full_path, color='#ffffff', linestyle='-', linewidth=2.5, label='VMD Target', zorder=6)
                    
                    ax.scatter(n + 29, vmd_res['target_30d'], color='#ffffff', marker='*', s=150, zorder=7)
                    ax.text(n + 30, vmd_res['target_30d'], f" VMD Target: ${vmd_res['target_30d']:.2f}", color='#ffffff', fontsize=8, fontweight='bold', va='center')
                else:
                    # If the math fails, print giant red text!
                    y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
                    ax.text(n/2, y_mid, "VMD Math Error: Could not decompose signal.", color='red', fontsize=12, fontweight='bold', ha='center', backgroundcolor='black', zorder=20)
                

        # ── Price label + horizontal line ─────────────────────────────
        last = float(df['Close'].iloc[-1])
        ax.axhline(last, color=C['accent'], linewidth=0.7,
                   linestyle='--', alpha=0.6, zorder=5)
        ax.text(n + 0.3, last, f" {last:,.2f}",
                color=C['accent'], fontsize=8, va='center', zorder=6)
        
        # ── SCALPING OVERLAY ──────────────────────────────────────────
        if self.opt.get('scalp', False) and 'Scalp_Buy' in df.columns:
            # 1. Background Coloring
            y_min, y_max = ax.get_ylim()
            # We use alpha=0.08 to make it visible against the dark background
            ax.fill_between(idx, y_min, y_max, where=(df['Scalp_Trend'].values == 1), color=C['green'], alpha=0.08, zorder=0)
            ax.fill_between(idx, y_min, y_max, where=(df['Scalp_Trend'].values == -1), color=C['red'], alpha=0.08, zorder=0)
            
            # 2. Plot signals
            for i in range(n):
                if df['Scalp_Buy'].iloc[i]:
                    ax.scatter(i, df['Low'].iloc[i] - (df['Close'].iloc[i]*0.002), 
                               marker='^', color=C['green'], s=100, zorder=10)
                    ax.text(i, df['Low'].iloc[i] - (df['Close'].iloc[i]*0.005), "UP", 
                            color=C['green'], fontsize=8, ha='center', va='top', zorder=10)
                
                if df['Scalp_Sell'].iloc[i]:
                    ax.scatter(i, df['High'].iloc[i] + (df['Close'].iloc[i]*0.002), 
                               marker='v', color=C['red'], s=100, zorder=10)
                    ax.text(i, df['High'].iloc[i] + (df['Close'].iloc[i]*0.005), "DOWN", 
                            color=C['red'], fontsize=8, ha='center', va='bottom', zorder=10)
       

        # ── PRO TREND TRADER (PTT) RENDERER ───────────────────────────
        if self.opt.get('ptt', False) and 'ptt_trades' in df.attrs:
            # Draw Dynamic Trend Core EMAs
            ax.plot(idx, df['EMA9'], color=C['cyan'], linewidth=1.2, alpha=0.85, zorder=3)
            ax.plot(idx, df['EMA21'], color=C['yellow'], linewidth=1.2, alpha=0.85, zorder=3)

            for t in df.attrs['ptt_trades']:
                # Smart Persistence: Only render trades from the last 60 bars to prevent clutter
                if t['exit_idx'] < n - 60:
                    continue

                color = C['green'] if t['dir'] == 1 else C['red']
                sign = "BUY" if t['dir'] == 1 else "SELL"
                e_idx, e_p, atr_v = t['entry_idx'], t['entry_p'], t['atr']

                # 1. Entry Badge & Visual Precision Connector
                y_offset = -atr_v * 1.5 if t['dir'] == 1 else atr_v * 1.5
                badge_y = df['Low'].iloc[e_idx] + y_offset if t['dir'] == 1 else df['High'].iloc[e_idx] + y_offset

                ax.plot([e_idx, e_idx], [df['Close'].iloc[e_idx], badge_y], color=color, linestyle=':', linewidth=1.5, alpha=0.8, zorder=5)
                ax.text(e_idx, badge_y, sign, color='white', fontweight='bold', fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"), zorder=10)

                # 2. Sequential TP Hit Badges
                for hit in t['tp_hits']:
                    h_idx, h_p = hit['idx'], hit['p']
                    ax.plot([h_idx, h_idx], [df['Close'].iloc[h_idx], h_p], color=C['accent2'], linestyle=':', linewidth=1.5, alpha=0.8, zorder=5)
                    ax.text(h_idx, h_p, hit['label'], color='black', fontweight='bold', fontsize=7, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc=C['accent2'], ec="none"), zorder=10)

                # 3. Horizontal Management Lines (Entry, SL, Unhit TPs)
                end_idx = t['exit_idx']
                
                # Entry Line (Gray)
                ax.plot([e_idx, end_idx], [e_p, e_p], color=C['text2'], linestyle='-', linewidth=1, alpha=0.6, zorder=4)
                
                # Stop Loss Line (Red Dotted)
                sl_p = t['sl_p'] 
                ax.plot([e_idx, end_idx], [sl_p, sl_p], color=C['red'], linestyle=':', linewidth=1.2, alpha=0.6, zorder=4)
                
                # TP Lines (Green Dotted)
                for i, tp_p in enumerate(t['tps']):
                    tp_end = end_idx
                    for hit in t['tp_hits']:
                        if hit['p'] == tp_p:
                            tp_end = hit['idx']
                            break
                    ax.plot([e_idx, tp_end],[tp_p, tp_p], color=C['green'], linestyle=':', linewidth=1.2, alpha=0.4, zorder=4)

                # 4. Momentum Exhaustion Exit Badge
                if t['exit_idx'] < n - 1:
                    ex_idx = t['exit_idx']
                    ax.text(ex_idx, df['Close'].iloc[ex_idx], "EXIT", color='white', fontweight='bold', fontsize=7, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc=C['text3'], ec="none"), zorder=11)
                    
        
        # ── Axis styling ──────────────────────────────────────────────
        ax.set_facecolor(C['bg'])
        ax.tick_params(colors=C['text2'], labelsize=8)
        ax.yaxis.tick_right()
        ax.set_xticklabels([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C['border'])
        ax.set_xlim(-1, n + 2)
        ax.grid(True, color=C['border'], linewidth=0.35, alpha=0.7)

        if self.opt['ma']:
            leg = ax.legend(loc='upper left', fontsize=8,
                            facecolor=C['bg2'], edgecolor=C['border'],
                            labelcolor=C['text2'])

        # update header labels
        prev  = float(df['Close'].iloc[-2]) if n > 1 else last
        chg   = last - prev
        pct   = chg / prev * 100 if prev else 0
        sign  = '+' if pct >= 0 else ''
        clr   = C['green'] if pct >= 0 else C['red']
        self.title_lbl.setText(f"  {self.ticker}")
        self.title_lbl.setStyleSheet(
            f"color:{C['accent']}; font-size:15px; font-weight:bold; letter-spacing:1.5px;"
        )
        self.price_lbl.setText(f"   {last:,.2f}")
        self.chg_lbl.setText(f"  {sign}{chg:,.2f}  ({sign}{pct:.2f}%)")
        self.chg_lbl.setStyleSheet(f"color:{clr}; font-size:13px; font-weight:bold;")

        sub = 1

        # ── Volume ────────────────────────────────────────────────────
        if self.opt['vol']:
            axv = axes[sub]; sub += 1
            vcols = [C['cand_up'] if df['Close'].iloc[i] >= df['Open'].iloc[i]
                     else C['cand_dn'] for i in range(n)]
            axv.bar(idx, df['Volume'], color=vcols, alpha=0.65, linewidth=0)
            axv.set_facecolor(C['bg'])
            axv.tick_params(colors=C['text2'], labelsize=7)
            axv.yaxis.tick_right()
            axv.yaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: f"{x/1e9:.1f}B" if x >= 1e9
                    else f"{x/1e6:.0f}M" if x >= 1e6
                    else f"{x/1e3:.0f}K"
                )
            )
            for sp in axv.spines.values(): sp.set_edgecolor(C['border'])
            axv.set_xticklabels([])
            axv.set_xlim(-1, n + 2)
            axv.grid(True, color=C['border'], linewidth=0.3, alpha=0.6)
            axv.text(-0.5, axv.get_ylim()[1] * 0.85, "VOL",
                     color=C['text3'], fontsize=7)

        # ── RSI ───────────────────────────────────────────────────────
        if self.opt['rsi'] and 'RSI' in df:
            axr = axes[sub]; sub += 1
            axr.plot(idx, df['RSI'], color=C['purple'], linewidth=0.9)
            axr.axhline(70, color=C['red'],   linewidth=0.6, linestyle='--', alpha=0.5)
            axr.axhline(30, color=C['green'], linewidth=0.6, linestyle='--', alpha=0.5)
            axr.fill_between(idx, 70, df['RSI'].clip(upper=70),
                             alpha=0.1, color=C['red'])
            axr.fill_between(idx, 30, df['RSI'].clip(lower=30),
                             alpha=0.1, color=C['green'])
            axr.set_ylim(0, 100)
            axr.set_facecolor(C['bg'])
            axr.tick_params(colors=C['text2'], labelsize=7)
            axr.yaxis.tick_right()
            for sp in axr.spines.values(): sp.set_edgecolor(C['border'])
            axr.set_xticklabels([])
            axr.set_xlim(-1, n + 2)
            axr.grid(True, color=C['border'], linewidth=0.3, alpha=0.6)
            axr.text(-0.5, 85, "RSI", color=C['text3'], fontsize=7)
            last_rsi = df['RSI'].dropna()
            if not last_rsi.empty:
                r = float(last_rsi.iloc[-1])
                axr.text(n + 0.3, r, f"{r:.1f}", color=C['purple'], fontsize=7, va='center')

        # ── MACD ──────────────────────────────────────────────────────
        if self.opt['macd'] and 'MACD' in df:
            axm = axes[sub]
            axm.plot(idx, df['MACD'],   color=C['blue'],   linewidth=0.9, label='MACD')
            axm.plot(idx, df['Signal'], color=C['accent'], linewidth=0.9, label='Signal')
            hist  = df['MHist'].fillna(0)
            hcols = [C['green'] if h >= 0 else C['red'] for h in hist]
            axm.bar(idx, hist, color=hcols, alpha=0.5, linewidth=0)
            axm.axhline(0, color=C['border'], linewidth=0.5)
            axm.set_facecolor(C['bg'])
            axm.tick_params(colors=C['text2'], labelsize=7)
            axm.yaxis.tick_right()
            for sp in axm.spines.values(): sp.set_edgecolor(C['border'])
            axm.set_xlim(-1, n + 2)
            axm.grid(True, color=C['border'], linewidth=0.3, alpha=0.6)
            axm.text(-0.5, axm.get_ylim()[1] * 0.8, "MACD", color=C['text3'], fontsize=7)

                 # ── VWAP overlay on main chart ───────────────────────────────────────
        if self.opt['vwap'] and 'VWAP' in df and df['VWAP'].notna().any():
            ax.plot(idx, df['VWAP'], color=C['cyan'], linewidth=1.1,
                    alpha=0.85, linestyle='-.', label='VWAP', zorder=4)

        # ── OBV ──────────────────────────────────────────────────────────────
        if self.opt['obv'] and 'OBV' in df:
            axob = axes[sub]; sub += 1
            axob.plot(idx, df['OBV'], color=C['blue'], linewidth=0.9)
            axob.axhline(0, color=C['border'], linewidth=0.5)
            axob.set_facecolor(C['bg']); axob.tick_params(colors=C['text2'], labelsize=7)
            axob.yaxis.tick_right()
            axob.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x/1e6:.0f}M" if abs(x)>=1e6 else f"{x/1e3:.0f}K"))
            for sp in axob.spines.values(): sp.set_edgecolor(C['border'])
            axob.set_xticklabels([]); axob.set_xlim(-1, n+2)
            axob.grid(True, color=C['border'], linewidth=0.3, alpha=0.6)
            axob.text(-0.5, axob.get_ylim()[1]*0.8, "OBV", color=C['text3'], fontsize=7)

        # ── CMF ──────────────────────────────────────────────────────────────
        if self.opt['cmf'] and 'CMF' in df:
            axcm = axes[sub]; sub += 1
            cmf_vals = df['CMF'].fillna(0)
            cmf_cols = [C['green'] if v >= 0 else C['red'] for v in cmf_vals]
            axcm.bar(idx, cmf_vals, color=cmf_cols, alpha=0.65, linewidth=0)
            axcm.axhline(0,      color=C['border'], linewidth=0.7)
            axcm.axhline( 0.05,  color=C['green'],  linewidth=0.5, linestyle='--', alpha=0.5)
            axcm.axhline(-0.05,  color=C['red'],    linewidth=0.5, linestyle='--', alpha=0.5)
            axcm.set_facecolor(C['bg']); axcm.tick_params(colors=C['text2'], labelsize=7)
            axcm.yaxis.tick_right(); axcm.set_ylim(-1, 1)
            for sp in axcm.spines.values(): sp.set_edgecolor(C['border'])
            axcm.set_xticklabels([]); axcm.set_xlim(-1, n+2)
            axcm.grid(True, color=C['border'], linewidth=0.3, alpha=0.6)
            axcm.text(-0.5, 0.7, "CMF", color=C['text3'], fontsize=7)
            last_cmf = df['CMF'].dropna()
            if not last_cmf.empty:
                axcm.text(n+0.3, float(last_cmf.iloc[-1]),
                          f"{float(last_cmf.iloc[-1]):.3f}", color=C['blue'], fontsize=7, va='center')

        # ── RVI ──────────────────────────────────────────────────────────────
        if self.opt['rvi'] and 'RVI' in df:
            axrv = axes[sub]; sub += 1
            axrv.plot(idx, df['RVI'].fillna(0),        color=C['yellow'],  linewidth=0.9)
            axrv.plot(idx, df['RVI_Signal'].fillna(0), color=C['accent2'], linewidth=0.9, linestyle='--')
            axrv.axhline(0, color=C['border'], linewidth=0.5)
            axrv.set_facecolor(C['bg']); axrv.tick_params(colors=C['text2'], labelsize=7)
            axrv.yaxis.tick_right()
            for sp in axrv.spines.values(): sp.set_edgecolor(C['border'])
            axrv.set_xticklabels([]); axrv.set_xlim(-1, n+2)
            axrv.grid(True, color=C['border'], linewidth=0.3, alpha=0.6)
            axrv.text(-0.5, axrv.get_ylim()[1]*0.8, "RVI", color=C['text3'], fontsize=7)

        # ── ATR ──────────────────────────────────────────────────────────────
        if self.opt['atr'] and 'ATR' in df:
            axat = axes[sub]; sub += 1
            axat.plot(idx, df['ATR'].fillna(0), color=C['yellow'], linewidth=0.9)
            axat.set_facecolor(C['bg']); axat.tick_params(colors=C['text2'], labelsize=7)
            axat.yaxis.tick_right()
            for sp in axat.spines.values(): sp.set_edgecolor(C['border'])
            axat.set_xticklabels([]); axat.set_xlim(-1, n+2)
            axat.grid(True, color=C['border'], linewidth=0.3, alpha=0.6)
            axat.text(-0.5, axat.get_ylim()[1]*0.8, "ATR", color=C['text3'], fontsize=7)
            last_atr = df['ATR'].dropna()
            if not last_atr.empty:
                axat.text(n+0.3, float(last_atr.iloc[-1]),
                          f"{float(last_atr.iloc[-1]):.2f}", color=C['yellow'], fontsize=7, va='center')
   

        # ── X-axis dates on bottom subplot ────────────────────────────
        btm = axes[-1]
        step = max(1, n // 8)
        ticks = list(range(0, n, step))
        
        # Add future date markers if GBM, FVG, GP, FFT, or VMD is on
        if self.opt.get('gbm', False) or self.opt.get('gp', False) or self.opt.get('fft', False) or self.opt.get('vmd', False):
            ticks.extend([n + 10, n + 20, n + 30])
        elif self.opt.get('fvg', False):
            ticks.extend([n + 5, n + 10, n + 15])
            
        fmt = '%b %d' if self.period in ('1d', '5d') else '%b %y'
        btm.set_xticks(ticks)
        
        labels =[]
        last_dt = df.index[-1]
        for i in ticks:
            if i < n:
                labels.append(df.index[i].strftime(fmt))
            else:
                # Convert trading days to calendar days (approx x 1.45)
                fut_dt = last_dt + pd.Timedelta(days=int((i - n + 1) * 1.45))
                labels.append(fut_dt.strftime(fmt) + '*')
                
        btm.set_xticklabels(labels, rotation=0, fontsize=7, color=C['text2'])

        # 👇 THIS LOOP FORCES ALL CHARTS TO EXPAND 👇
        for a in axes:
            a.set_xlim(-1, x_max_val)

        self.fig.patch.set_facecolor(C['bg'])
        self.fig.subplots_adjust(left=0.01, right=0.93, top=0.97, bottom=0.04)
        self.canvas.draw()

# ═══════════════════════════════════════════════════════════════════════════════
# AI CHAT PANEL
# ═══════════════════════════════════════════════════════════════════════════════
class AIChatPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.history     = []
        self.cur_ticker  = None
        self._st         = None
        self._cur_resp   = ""
        self._resp_open  = False
        self._setup_ui()
        BUS.ticker_changed.connect(lambda t: setattr(self, 'cur_ticker', t))

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr_row = QWidget()
        hdr_row.setFixedHeight(32)
        hdr_row.setStyleSheet(f"background:{C['bg3']}; border-bottom:1px solid {C['border']};")
        hdr_hl = QHBoxLayout(hdr_row)
        hdr_hl.setContentsMargins(8,0,6,0); hdr_hl.setSpacing(0)
        hdr = QLabel("  ◈ AI ANALYST")
        hdr.setStyleSheet(f"color:{C['accent']}; font-size:10px; font-weight:bold; letter-spacing:1px; background:transparent;")
        hdr_hl.addWidget(hdr); hdr_hl.addStretch()
        ai_clr = QPushButton('✕ Clear')
        ai_clr.setFixedHeight(22)
        ai_clr.setStyleSheet(
            f"QPushButton{{background:{C['bg4']};border:1px solid {C['border']};"
            f"color:{C['text3']};font-size:10px;padding:0 6px;}}"
            f"QPushButton:hover{{background:{C['red']};color:white;border-color:{C['red']};}}"
        )
        ai_clr.clicked.connect(self._clear)
        hdr_hl.addWidget(ai_clr)
        lay.addWidget(hdr_row)

        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setStyleSheet(
            f"QTextEdit {{ background:{C['bg2']}; border:none; padding:10px; }}"
        )
        lay.addWidget(self.display)

        # Quick-action buttons
        qb = QWidget()
        qb.setFixedHeight(36)
        qb.setStyleSheet(f"background:{C['bg3']}; border-top:1px solid {C['border']};")
        qb_lay = QHBoxLayout(qb)
        qb_lay.setContentsMargins(6, 4, 6, 4)
        qb_lay.setSpacing(4)
        for label, key in[("📊 Full","full"),("🐂 Bull","bull"),("🐻 Bear","bear"),
                   ("⚠ Risk","risk"),("$ Valuation","val"),("📈 Chart AI","chart_ai"), ("👁️ Vision AI","vision_ai")]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet(f"""
                QPushButton {{ background:{C['bg4']}; border:1px solid {C['border']};
                               color:{C['text2']}; font-size:10px; }}
                QPushButton:hover {{ background:{C['accent']}; color:white; }}
            """)
            btn.clicked.connect(lambda _, k=key: self._vision_ai() if k == "vision_ai" else (self._chart_ai() if k == "chart_ai" else self._quick(k)))
            qb_lay.addWidget(btn)
        lay.addWidget(qb)

        # Input row
        inp_row = QWidget()
        inp_row.setFixedHeight(44)
        inp_row.setStyleSheet(f"background:{C['bg3']}; border-top:1px solid {C['border']};")
        ir = QHBoxLayout(inp_row)
        ir.setContentsMargins(6, 6, 6, 6)
        ir.setSpacing(6)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask the AI analyst anything about markets…")
        self.input.setFixedHeight(30)
        self.input.returnPressed.connect(self._send)
        ir.addWidget(self.input)

        send = QPushButton("▶")
        send.setFixedSize(30, 30)
        send.setProperty("accent", "true")
        send.setStyleSheet(f"""
            QPushButton {{ background:{C['accent']}; border:none; color:white;
                           font-size:14px; font-weight:bold; }}
            QPushButton:hover {{ background:{C['accent2']}; }}
        """)
        send.clicked.connect(self._send)
        ir.addWidget(send)

        clr = QPushButton("⌫")
        clr.setFixedSize(30, 30)
        clr.setToolTip("Clear chat history")
        clr.clicked.connect(self._clear)
        ir.addWidget(clr)

        lay.addWidget(inp_row)

        self._sys("APEX AI Analyst online. Select a ticker or type a question.")

    def _chart_ai(self):
        """Enriched Chart AI: confidence scores + options flow + SPY correlation + Chaos Math."""
        if not self.cur_ticker or not hasattr(self, '_chart_ref') or self._chart_ref.df is None:
            self._sys("⚠ Load a chart first by selecting a ticker symbol.")
            return

        self._sys("⟳ Computing statistical confidence, chaos math, options flow & correlation…")
        BUS.status_msg.emit("Gathering enhanced data matrix…")

        ticker = self.cur_ticker
        df     = self._chart_ref.df
        last   = df.iloc[-1]

        def fv(col, fmt=".4f"):
            try:
                v = float(last[col])
                return "N/A" if v != v else f"{v:{fmt}}"
            except Exception:
                return "N/A"

        # ── EXTREME QUANT MATH CALCULATION ──
        hurst = 0.5
        order_flow_imbalance = 0.0
        if len(df) > 30:
            closes = df['Close'].values
            # A. Hurst Exponent (Chaos Theory)
            try:
                lags = range(2, 20)
                tau =[np.sqrt(np.std(np.subtract(closes[lag:], closes[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = poly[0] * 2.0
            except: pass
            
            # B. Order Flow Imbalance Proxy (Dark Pool footprint)
            try:
                rng = (df['High'] - df['Low']) + 1e-9
                cvd = df['Volume'] * ((df['Close'] - df['Open']) / rng)
                order_flow_imbalance = cvd.tail(10).sum() # Net volume delta over last 10 bars
            except: pass
            
        chaos_block = (
            f"\n\n── EXTREME QUANTS (CHAOS THEORY & ORDER FLOW) ──\n"
            f"  Hurst Exponent: {hurst:.3f} (If <0.45: Mean-Reverting. If >0.55: Trending. ~0.50: Random Noise)\n"
            f"  Order Flow Imbalance (10-bar CVD): {order_flow_imbalance:,.0f} (Positive = Hidden Institutional Accumulation, Negative = Distribution)"
        )

        # ── PRO TREND TRADER (PTT) STATE FOR AI ──
        ptt_status = "No active PTT trade."
        if 'ptt_trades' in df.attrs and len(df.attrs['ptt_trades']) > 0:
            last_trade = df.attrs['ptt_trades'][-1]
            if last_trade['exit_idx'] == len(df) - 1:  # If the exit index is the current live bar, it's active!
                tdir = "LONG" if last_trade['dir'] == 1 else "SHORT"
                entry_p = last_trade['entry_p']
                sl_p = last_trade['sl_p']
                hits = len(last_trade['tp_hits'])
                next_tp = last_trade['tps'][hits] if hits < len(last_trade['tps']) else "All TPs Hit"
                if isinstance(next_tp, float): next_tp = f"${next_tp:.2f}"
                ptt_status = f"ACTIVE {tdir} | Entry: ${entry_p:.2f} | Current SL: ${sl_p:.2f} | Targets Hit: {hits}/4 | Next Target: {next_tp}"

        # Base indicators
        summary = (
            f"Ticker : {ticker}\n"
            f"Price  : {fv('Close','.2f')}  |  Volume: {fv('Volume',',.0f')}\n"
            f"MA20   : {fv('MA20','.2f')}  |  MA50 : {fv('MA50','.2f')}  |  MA200: {fv('MA200','.2f')}\n"
            f"BB_Upp : {fv('BB_U','.2f')}  |  BB_Low: {fv('BB_L','.2f')}\n"
            f"RSI(14): {fv('RSI','.1f')}  |  MACD : {fv('MACD','.4f')}  |  Signal: {fv('Signal','.4f')}\n"
            f"VWAP   : {fv('VWAP','.2f')}  |  ATR(14): {fv('ATR','.2f')}\n"
            f"OBV    : {fv('OBV',',.0f')}  |  CMF(20): {fv('CMF','.4f')}\n"
            f"RVI    : {fv('RVI','.4f')}   |  RVI Sig: {fv('RVI_Signal','.4f')}\n"
            f"Kalman True State: {fv('Kalman','.2f')}  |  Kalman Deviation: {fv('Kalman_Diff','.2f')}%\n"
            f"PTT Algorithmic State: {ptt_status}\n"
            f"52w Hi : {df['Close'].max():.2f}  |  52w Lo: {df['Close'].min():.2f}\n"
            f"Bars   : {len(df)}"
        )

        # Statistical Confidence Scores
        conf_scores = DE.statistical_confidence(df)
        conf_lines  =[]
        for key, val in conf_scores.items():
            if key == 'error': continue
            w, t, desc = val
            pct = w / t * 100 if t > 0 else 0
            conf_lines.append(f"  • {desc}  →  Historical_Success_Rate: {pct:.0f}%")
        conf_block = (
            "\n\n── STATISTICAL CONFIDENCE SCORES (backtested on this ticker's own history) ──\n"
            + ("\n".join(conf_lines) if conf_lines else "  Insufficient history for confidence scoring.")
        )

        # Options Flow / Put-Call Ratio
        pcr  = DE.put_call_ratio(ticker)
        earn = DE.earnings_proximity(ticker)
        insider = DE.get_insider_activity(ticker)
        pcr_block = (
            f"\n\n── OPTIONS FLOW / SMART MONEY POSITIONING ──\n"
            f"  Put/Call Ratio (OI) : {pcr.get('pcr','N/A')}\n"
            f"  Put/Call Ratio (Vol): {pcr.get('pcr_volume','N/A')}\n"
            f"  Call OI: {pcr.get('call_oi',0):,}  |  Put OI: {pcr.get('put_oi',0):,}\n"
            f"  Options Signal: {pcr.get('smart_money_signal','N/A')}\n"
            f"  Interpretation: {pcr.get('interpretation','N/A')}\n"
            f"\n  ▸ PCR VELOCITY:\n"
            f"    Velocity Signal: {pcr.get('velocity_signal','N/A')}\n"
            f"    {pcr.get('velocity_interp','N/A')}\n"
            f"\n  ▸ EARNINGS PROXIMITY:\n"
            f"    Next: {earn.get('next_earnings_date','N/A')} ({earn.get('days_to_earnings','?')}d away)\n"
            f"    Signal Reliability: {earn.get('signal_reliability','NORMAL')}\n"
            f"    {earn.get('note','')}"
        )

        # SPY Correlation & Relative Strength
        corr = DE.spy_correlation(ticker)
        corr_block = (
            f"\n\n── SPY CORRELATION & RELATIVE STRENGTH ──\n"
            f"  Correlation — Full: {corr.get('corr_full','N/A')}  "
            f"| 90d: {corr.get('corr_90d','N/A')}  | 30d: {corr.get('corr_30d','N/A')}\n"
            f"  Beta: {corr.get('beta','N/A')}\n"
            f"  Relative Strength vs SPY — 30d: {corr.get('rs_30d','N/A')}%  "
            f"| 90d: {corr.get('rs_90d','N/A')}%  | 1yr: {corr.get('rs_252d','N/A')}%\n"
            f"  RS Signal: {corr.get('rs_signal','N/A')}\n"
            f"  Interpretation: {corr.get('interpretation','N/A')}"
        )

        # Advanced Feature Engineering
        adv = DE.advanced_signals(df, ticker)
        _cfl = '\n'.join(f'  ✦ {s}' for s in adv['confluence']) or '  No confluent signals'
        adv_block = (
            f"\n\n── ADVANCED FEATURE ENGINEERING ──\n"
            f"\n▸ CONFLUENCE MAP:\n{_cfl}\n"
            f"\n▸ VOLATILITY-ADJUSTED MOMENTUM:\n"
            f"  {adv.get('atr_context','N/A')}\n"
            f"  20d Z-score: {adv.get('momentum_z','N/A')}  "
            f"(>+2=extreme rally, <-2=extreme sell-off)\n"
            f"\n▸ BETA-ADJUSTED ALPHA (Hidden Strength vs SPY):\n"
            f"  {adv.get('interpretation','N/A')}\n"
            f"  Hidden Strength Signal: {adv.get('hidden_strength','N/A')}"
        )

        insider_block = (
            f"\n\n── CORPORATE INSIDER ACTIVITY (SEC Form 4) ──\n"
            f"  Signal: {insider.get('signal', 'N/A')}\n"
            f"  Details: {insider.get('interpretation', 'N/A')}\n"
            f"  Insider Ownership: {insider.get('insider_pct', 0):.2f}% of float"
        )
        
        # ── GAUSSIAN PROCESS FORECAST CALCULATION FOR AI ──
        gp_target = "N/A"
        try:
            if HAS_XGB: # Just ensuring sklearn is loaded
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared, WhiteKernel
                lookback = min(252, len(df))
                y_t = df['Close'].iloc[-lookback:].values
                ym, ys = y_t.mean(), y_t.std()
                yn = (y_t - ym) / (ys + 1e-9)
                X_t = np.arange(lookback).reshape(-1, 1)
                k = ConstantKernel(1.0) * RBF(50.0) + ExpSineSquared(1.0, 63.0) + WhiteKernel(0.5)
                gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(X_t, yn)
                yp = gp.predict(np.array([[lookback + 30]]))
                gp_target = f"${(yp[0] * ys + ym):.2f}"
        except Exception:
            pass

        gp_block = f"\n\n── GAUSSIAN PROCESS FORECAST ──\n  30-Day Mathematical Target: {gp_target}"

        # ── FAST FOURIER TRANSFORM (FFT) CALCULATION FOR AI ──
        fft_res = DE.fft_extrapolation(df)
        fft_block = ""
        if fft_res:
            fft_block = (
                f"\n\n── FAST FOURIER TRANSFORM (CYCLE ANALYSIS) ──\n"
                f"  Projected Next Peak: In {fft_res['next_peak_days']} days at ${fft_res['next_peak_price']:.2f}\n"
                f"  Projected Next Trough: In {fft_res['next_trough_days']} days at ${fft_res['next_trough_price']:.2f}"
            )

        prompt = (
            f"You are a professional technical analyst. Below is a fully enriched "
            f"data snapshot for {ticker} — including backtested confidence scores, "
            f"── INDICATOR SNAPSHOT ──\n{summary}"
            f"{chaos_block}{conf_block}{pcr_block}{corr_block}{adv_block}{insider_block}{gp_block}{fft_block}\n\n"
            f"options flow, relative strength vs SPY, and SEC Insider Trading data.\n\n"
            f"── INDICATOR SNAPSHOT ──\n{summary}"
            f"{chaos_block}{conf_block}{pcr_block}{corr_block}{adv_block}{insider_block}{gp_block}\n\n"
            "Using ALL data above, provide a structured analysis:\n\n"
            "1. TREND ANALYSIS — MA alignment, price vs VWAP, and Price vs Kalman True State.\n"
            "2. MOMENTUM — RSI flags + Volatility Z-score\n"
            "3. CONFLUENCE — treat CONFLUENCE MAP as weight-of-evidence\n"
            "4. MARKET REGIME — Look at Volatility Squeeze / Expansion state\n"
            "5. EXTREME QUANTS — Analyze the Hurst Exponent (is the market trending or mean-reverting?) and check the Order Flow Imbalance for hidden dark pool accumulation/distribution inside the wicks.\n"
            "6. HIDDEN STRENGTH — Institutional accumulation explicitly\n"
            "7. VOLUME & MONEY FLOW — OBV trend, CMF, options PCR\n"
            "8. STATISTICAL EDGE — Cite Historical_Success_Rate % from backtests\n"
            "9. KEY LEVELS — MAs, BBs, 52w range\n"
            "10. INSIDER SENTIMENT — Does C-Suite accumulation/distribution confirm or contradict the chart?\n"
            "11. GAUSSIAN FORECAST — What is the 30-day mathematical GP target?\n"
            "12. SIGNAL — BULLISH / BEARISH / NEUTRAL with confidence %\n"
            "13. SHORT-TERM OUTLOOK — scenario tailored to regime\n"
            "14. PTT TRADE MANAGEMENT — If there is an active PTT trade, give exact instructions on managing the stop loss and take profits.\n"
        )
        self._send_msg(prompt)   

    def _vision_ai(self):
        """Captures the chart image and sends it to a Vision model in LM Studio."""
        if not self.cur_ticker or not hasattr(self, '_chart_ref') or self._chart_ref.df is None:
            self._sys("⚠ Load a chart first by selecting a ticker symbol.")
            return

        self._sys("📸 Taking a screenshot of the chart for the Vision Model...")
        BUS.status_msg.emit("Sending chart image to Vision AI...")

        ticker = self.cur_ticker
        df = self._chart_ref.df
        
        # Grab the Math Candlestick pattern
        math_pattern = df.attrs.get('candle_pattern', 'No pattern detected.')
        
        # Grab the image
        base64_image = self._chart_ref.get_base64_image()

        prompt_text = f"""Act as an elite quantitative technical analyst. I am providing you with a screenshot of the live chart for {ticker}. 

[MATHEMATICAL SCANNER RESULT]
The backend math algorithm confirms the most recent candlestick pattern is: {math_pattern}

[YOUR TASK]
Look closely at the provided chart image and answer the following:
1. **VISUAL STRUCTURE**: Do you see any classical chart patterns forming across the broader trend (e.g., Head and Shoulders, Double Bottoms, Trendline breakouts)?
2. **ALGORITHMIC OVERLAYS**: Analyze the indicators visible on the screen (Look at the FVG blocks, the Probability Cones, the MACD, or Volume). How do they interact with the current price?
3. **CANDLESTICK CONFIRMATION**: Look at the actual shape of the candles on the right edge of the chart. Do they visually confirm the Mathematical Scanner's result?
4. **VERDICT**: What is the highest probability move for the next 5 days?
"""
        
        # Build the exact OpenAI Vision payload format
        messages = [{
            "role": "user",
            "content":[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }]

        if self._st and self._st.isRunning():
            self._st.terminate()

        self._cur_resp = ""
        self._resp_open = False
        
        self.history.append({"role": "user", "content": f"[Sent chart screenshot to Vision AI for {ticker}]"})
        self._append_user(f"Analyze this chart image for {ticker}.")
        
        # Send the Multimodal payload to LM Studio
        self._st = StreamThread(messages, system="You are an expert financial vision AI. Be precise and reference what you visually see in the image.", temperature=0.2, max_tokens=1500)
        self._st.chunk.connect(self._on_chunk)
        self._st.done.connect(self._on_done)
        self._st.start()

    # ── actions ──────────────────────────────────────────────────────
    def _quick(self, key):
        t = self.cur_ticker or "the market"
        prompts = {
            "full": f"Provide a comprehensive structured equity analysis of {t}. Include: executive summary, business model, financial health, competitive position, growth catalysts, risks, and investment outlook.",
            "bull": f"What is the strongest bull case for {t}? Cover all key upside catalysts, growth drivers, and why a long investor would be excited.",
            "bear": f"What is the strongest bear case for {t}? Cover all key risks, headwinds, and why someone would short or avoid this stock.",
            "risk": f"What are the top 5 risk factors for {t}? For each: describe the risk, probability (H/M/L), potential impact, and any mitigants.",
            "val":  f"Provide a comprehensive valuation analysis for {t}. Cover current multiples vs history and peers, a simplified DCF framework, and an implied price target range.",
        }
        self._send_msg(prompts[key])

    def _send(self):
        msg = self.input.text().strip()
        if not msg:
            return
        self.input.clear()
        self._send_msg(msg)

    def _send_msg(self, msg):
        if not LM.connected:
            self._sys("⚠ LM Studio not connected. Start LM Studio and load a model.")
            return
        self._append_user(msg)
        self.history.append({"role": "user", "content": msg})

        system = (
            f"You are a senior Wall Street equity analyst covering {self.cur_ticker or 'all sectors'}. "
            "Provide professional, structured, data-driven analysis. "
            "Use ── for section headers and ● for bullets. Be specific and direct."
        )

        if self._st and self._st.isRunning():
            self._st.terminate()

        self._cur_resp    = ""
        self._resp_open   = False
        self._st = StreamThread(list(self.history), system, 0.3, 2500)
        self._st.chunk.connect(self._on_chunk)
        self._st.done.connect(self._on_done)
        self._st.start()
        BUS.status_msg.emit("AI thinking…")

    def _on_chunk(self, tok):
        tok = _fix_encoding(tok)
        if not self._resp_open:
            self._resp_open = True
            self.display.append(
                f'<p style="color:{C["accent"]}; font-weight:bold; margin-top:12px;">◈ APEX AI</p>'
                f'<p style="color:{C["text3"]}; font-size:10px; font-style:italic;">'
                f'⏳ Generating response...</p>'
            )
        self._cur_resp += tok
        # During streaming just show a plain-text live preview (fast, no flicker)
        cur = self.display.textCursor()
        cur.movePosition(cur.MoveOperation.End if hasattr(cur, "MoveOperation") else cur.End)
        cur.insertText(tok)
        self.display.setTextCursor(cur)
        self.display.ensureCursorVisible()
    def _on_done(self):
        self._resp_open = False
        sep = f'<p style="color:{C["border2"]}">{"─"*36}</p>'
        if self._cur_resp.strip():
            self.history.append({"role": "assistant", "content": self._cur_resp})
            rendered = md_to_html(self._cur_resp)
            header = (
                f'<p style="color:{C["accent"]}; font-weight:bold; margin-top:12px;">◈ APEX AI</p>'
            )
            full_html = self.display.toHtml()
            anchor = "◈ APEX AI"
            idx = full_html.rfind(anchor)
            if idx != -1:
                tag_start = full_html.rfind("<p", 0, idx)
                if tag_start == -1:
                    tag_start = idx
                close_body = full_html.rfind("</body>")
                if close_body != -1:
                    new_html = (full_html[:tag_start]
                                + header + rendered + sep
                                + full_html[close_body:])
                else:
                    new_html = (full_html[:tag_start]
                                + header + rendered + sep
                                + "</body></html>")
                self.display.setHtml(new_html)
            else:
                self.display.append(header)
                self.display.append(rendered)
                self.display.append(sep)
            self.display.verticalScrollBar().setValue(
                self.display.verticalScrollBar().maximum())
        else:
            self.display.append(
                f'<p style="color:{C["red"]}; font-size:11px;">'
                "⚠ No text returned by LM Studio.<br>"
                "Check: (1) a model is fully loaded, "
                "(2) LM Studio → Logs tab for errors, "
                "(3) try a shorter prompt.</p>"
            )
            self.display.append(sep)
        BUS.status_msg.emit("Analysis complete")
    # ── helpers ──────────────────────────────────────────────────────
    def _sys(self, text):
        self.display.append(
            f'<p style="color:{C["text3"]}; font-size:10px; font-style:italic;">◈ {text}</p>'
        )

    def _append_user(self, text):
        self.display.append(
            f'<p style="color:{C["accent2"]}; font-weight:bold; margin-top:12px;">▶ YOU</p>'
            f'<p style="color:{C["text"]}; margin-left:8px;">{text}</p>'
        )

    def _clear(self):
        self.history.clear()
        self.display.clear()
        self._sys("Chat cleared.")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PANEL  (structured research report)
# ═══════════════════════════════════════════════════════════════════════════════
class AnalysisPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.ticker = None
        self._st    = None
        self._setup_ui()
        BUS.ticker_changed.connect(self._on_ticker)

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # ── Toolbar ───────────────────────────────────────────────────
        tb = QHBoxLayout()
        self.ticker_lbl = QLabel("─── SELECT A TICKER TO BEGIN ───")
        self.ticker_lbl.setStyleSheet(
            f"color:{C['text3']}; font-size:13px; font-weight:bold; letter-spacing:1px;"
        )
        tb.addWidget(self.ticker_lbl)
        tb.addStretch()

        for label, slot in[
            ("◈  FULL REPORT",  self._run_full),
            ("$  DCF MODEL",    self._run_dcf),
            ("⊞  COMPS TABLE",  self._run_comps),
            ("▶  EARNINGS Q&A", self._run_earnings),
            ("⬡  POD ANALYSIS", self._run_pod),
            ("⚄  PAIRS TRADING", self._run_pairs),
            ("📅  SEASONALITY",  self._run_seasonality),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            if "FULL" in label:
                btn.setProperty("accent", "true")
                btn.setStyleSheet(f"""
                    QPushButton {{ background:{C['accent']}; border:1px solid {C['accent']};
                                   color:white; font-weight:bold; font-size:11px; }}
                    QPushButton:hover {{ background:{C['accent2']}; }}
                """)
            elif "POD" in label:
                btn.setStyleSheet(f"""
                    QPushButton {{ background:{C['bg3']}; border:1px solid {C['purple']};
                                   color:{C['purple']}; font-weight:bold; font-size:11px; }}
                    QPushButton:hover {{ background:{C['purple']}; color:white; }}
                """)
            btn.clicked.connect(slot)
            tb.addWidget(btn)
        clr_btn = QPushButton('✕ Clear')
        clr_btn.setFixedHeight(30)
        clr_btn.setStyleSheet(
            f"QPushButton{{background:{C['bg4']};border:1px solid {C['border']};"
            f"color:{C['text3']};font-size:10px;padding:0 8px;}}"
            f"QPushButton:hover{{background:{C['red']};color:white;border-color:{C['red']};}}"
        )
        clr_btn.clicked.connect(self._welcome)
        tb.addWidget(clr_btn)
        lay.addLayout(tb)

        # ── Key metrics strip ─────────────────────────────────────────
        mf = QFrame()
        mf.setStyleSheet(
            f"background:{C['bg2']}; border:1px solid {C['border']};"
        )
        mf_lay = QHBoxLayout(mf)
        mf_lay.setContentsMargins(14, 8, 14, 8)
        mf_lay.setSpacing(28)

        self._metrics = {}
        for lbl in ["PRICE", "MKT CAP", "P/E", "FWD P/E", "EV/EBITDA",
                    "REV GROWTH", "GROSS MGN", "NET MGN", "BETA", "52W POS"]:
            box = QWidget()
            bl  = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(2)
            l = QLabel(lbl)
            l.setStyleSheet(f"color:{C['text3']}; font-size:8px; font-weight:bold; letter-spacing:0.5px;")
            v = QLabel("─")
            v.setStyleSheet(f"color:{C['text']}; font-size:13px; font-weight:bold;")
            bl.addWidget(l); bl.addWidget(v)
            self._metrics[lbl] = v
            mf_lay.addWidget(box)
        mf_lay.addStretch()
        lay.addWidget(mf)

        # ── Output ────────────────────────────────────────────────────
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet(
            f"QTextEdit {{ background:{C['bg2']}; border:1px solid {C['border']}; padding:16px; font-size:12px; }}"
        )
        lay.addWidget(self.output)

        self.prog_lbl = QLabel("")
        self.prog_lbl.setStyleSheet(f"color:{C['text3']}; font-size:10px;")
        lay.addWidget(self.prog_lbl)

        self._welcome()

    def _welcome(self):
        self.output.setHtml(f"""
        <div style='font-family:Consolas,monospace; color:{C["text2"]}; padding:24px;'>
        <p style='color:{C["accent"]}; font-size:18px; font-weight:bold; letter-spacing:2px;'>◈ APEX ANALYSIS ENGINE</p>
        <br>
        <p style='color:{C["text"]}; font-size:13px;'>Select a ticker then choose an analysis type:</p>
        <br>
        <p>● <b style='color:{C["accent2"]};'>FULL REPORT</b> — Complete equity research note with rating &amp; price target</p>
        <p>● <b style='color:{C["accent2"]};'>DCF MODEL</b> — Discounted cash flow with sensitivity table</p>
        <p>● <b style='color:{C["accent2"]};'>COMPS TABLE</b> — Comparable company analysis vs peers</p>
        <p>● <b style='color:{C["accent2"]};'>EARNINGS Q&amp;A</b> — Simulate earnings call key questions &amp; answers</p>
        <br><br>
        <p style='color:{C["text3"]}; font-size:10px;'>Powered by LM Studio local AI · Data via yfinance</p>
        </div>
        """)

    # ── ticker / data ─────────────────────────────────────────────────
    def _on_ticker(self, ticker):
        self.ticker = ticker
        self.ticker_lbl.setText(f"─── {ticker}  ·  EQUITY ANALYSIS ───")
        self.ticker_lbl.setStyleSheet(
            f"color:{C['accent']}; font-size:13px; font-weight:bold; letter-spacing:1px;"
        )
        
        class MetricsThread(QThread):
            data_ready = pyqtSignal(dict)
            
            def __init__(self, t):
                super().__init__()
                self.t = t
                
            def run(self):
                info = DE.info(self.t)
                hist = DE.fetch(self.t, period="1y", interval="1d")
                
                price = info.get('currentPrice') or info.get('regularMarketPrice')
                hi52 = info.get('fiftyTwoWeekHigh')
                lo52 = info.get('fiftyTwoWeekLow')
                
                # If Yahoo blocked the fundamental data, calculate it from the chart!
                if hist is not None and not hist.empty:
                    if not price: price = float(hist['Close'].iloc[-1])
                    if not hi52:  hi52 = float(hist['High'].max())
                    if not lo52:  lo52 = float(hist['Low'].min())
                    
                pos52 = None
                if price and hi52 and lo52 and hi52 != lo52:
                    pos52 = (price - lo52) / (hi52 - lo52) * 100

                mkt_cap = info.get('marketCap')
                if not mkt_cap:
                    try:
                        import yfinance as yf
                        mkt_cap = yf.Ticker(self.t).fast_info.get('marketCap')
                    except Exception:
                        pass

                data = {
                    "PRICE":     AnalysisPanel._f(price),
                    "MKT CAP":   AnalysisPanel._fB(mkt_cap),
                    "P/E":       AnalysisPanel._f(info.get('trailingPE'), 1),
                    "FWD P/E":   AnalysisPanel._f(info.get('forwardPE'), 1),
                    "EV/EBITDA": AnalysisPanel._f(info.get('enterpriseToEbitda'), 1),
                    "REV GROWTH":AnalysisPanel._fp(info.get('revenueGrowth')),
                    "GROSS MGN": AnalysisPanel._fp(info.get('grossMargins')),
                    "NET MGN":   AnalysisPanel._fp(info.get('profitMargins')),
                    "BETA":      AnalysisPanel._f(info.get('beta'), 2),
                    "52W POS":   f"{pos52:.0f}%" if pos52 is not None else "─",
                }
                self.data_ready.emit(data)

        self._metrics_thread = MetricsThread(ticker)
        self._metrics_thread.data_ready.connect(self._set_metrics)
        self._metrics_thread.start()

    def _set_metrics(self, data):
        for k, v in data.items():
            if k in self._metrics:
                self._metrics[k].setText(v)

    @staticmethod
    def _f(v, d=2):
        try:    return f"{float(v):,.{d}f}"
        except: return "─"

    @staticmethod
    def _fB(v):
        try:
            v = float(v)
            if v >= 1e12: return f"{v/1e12:.2f}T"
            if v >= 1e9:  return f"{v/1e9:.1f}B"
            if v >= 1e6:  return f"{v/1e6:.0f}M"
            return f"{v:,.0f}"
        except: return "─"

    @staticmethod
    def _fp(v):
        try:    return f"{float(v)*100:.1f}%"
        except: return "─"

    # ── analysis runners ──────────────────────────────────────────────
    def _check(self):
        if not self.ticker:
            self.output.setPlainText("Select a ticker first.")
            return False
        if not LM.connected:
            self.output.setPlainText("⚠ LM Studio not connected.")
            return False
        return True

    def _run_full(self):
        if not self._check(): return
        t = self.ticker
        prompt = f"""Write a FULL professional equity research initiation report for {t}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{t} | EQUITY RESEARCH | INITIATION OF COVERAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[RATING & PRICE TARGET]
Rating: BUY / HOLD / SELL  (choose one)
Price Target (12-month): $XX
Implied Upside: XX%
Conviction: HIGH / MEDIUM / LOW

[EXECUTIVE SUMMARY]
3-5 sentence snapshot covering thesis and key message.

[BUSINESS OVERVIEW]
Business model, revenue mix, competitive moats, customer base.

[INVESTMENT THESIS — 3 PILLARS]
Pillar 1: (specific catalyst with numbers)
Pillar 2: (specific catalyst with numbers)
Pillar 3: (specific catalyst with numbers)

[FINANCIAL ANALYSIS]
Revenue trend, margins, FCF profile, balance sheet health.
Use specific numbers and YoY comparisons.

[BULL CASE  ($XX price target)]
Three concrete reasons for significant upside.

[BASE CASE  ($XX price target)]
Most likely scenario and key assumptions.

[BEAR CASE  ($XX price target)]
Three concrete reasons for significant downside.

[RISK MATRIX]
Risk | Probability | Impact | Mitigation
(5 rows, table format)

[VALUATION]
Current multiples vs historical & peers. Justified multiple range.

[NEAR-TERM CATALYSTS]
Upcoming events that could move the stock (dates if known).

[CONCLUSION]
Restate rating, price target, and conviction with one-line rationale.

Be specific. Use real numbers. No filler."""
        self._stream(prompt)

    def _run_dcf(self):
        if not self._check(): return
        self._stream(f"""Build a detailed DCF valuation model for {self.ticker}.

[MODEL ASSUMPTIONS]
Revenue base year, growth rates (Year 1-5, terminal), EBIT margins, D&A, capex, NWC.

[5-YEAR PROJECTIONS]  (table: Year | Revenue | EBITDA | EBIT | FCF)

[WACC CALCULATION]
Cost of equity (CAPM), cost of debt, capital structure, blended WACC.

[TERMINAL VALUE]
Gordon Growth Method: TV = FCF(n) × (1+g) / (WACC - g)

[ENTERPRISE VALUE → EQUITY VALUE]
EV, net debt adjustment, shares out, implied price per share.

[SENSITIVITY TABLE]
Grid: WACC (rows: -1%, base, +1%, +2%) vs Terminal Growth Rate (cols: 1%, 2%, 3%, 4%)
Show implied share prices in each cell.

[CONCLUSION]
Intrinsic value range (bear/base/bull), comparison to current price.""")

    def _run_comps(self):
        if not self._check(): return
        self._stream(f"""Create a full comparable company analysis for {self.ticker}.

[PEER SELECTION]
Name 6-8 closest peers and explain selection rationale.

[COMPS TABLE]  (format as aligned table)
Company | Ticker | Mkt Cap | EV/Rev | EV/EBITDA | P/E | P/FCF | Rev Growth | EBIT Margin

[STATISTICAL SUMMARY]
Median, Mean, 25th percentile, 75th percentile for each metric.

[{self.ticker} POSITIONING]
Where does {self.ticker} sit vs peers on each metric?
Premium/discount and why it is or is not justified.

[IMPLIED VALUATION]
Using median peer multiples, derive a valuation range for {self.ticker}.

[CONCLUSION]
Is {self.ticker} cheap or expensive vs peers? Final take.""")

    def _run_earnings(self):
        if not self._check(): return
        self._stream(f"""Simulate a professional earnings call Q&A for {self.ticker}.

Imagine the most recent or upcoming earnings call. Write:

[CFO PREPARED REMARKS SUMMARY]
Key financial highlights, guidance, and management message.

[TOP 10 ANALYST QUESTIONS & MANAGEMENT ANSWERS]
Q1: [question from sell-side analyst]
A1: [management answer with specific metrics]
(repeat for 10 questions covering: revenue, margins, guidance, competition, macro, capex, buybacks, etc.)

[BULL INTERPRETATION]
What bulls take away from the call.

[BEAR INTERPRETATION]
What bears take away from the call.

[LIKELY STOCK REACTION]
Expected next-day move and why.""")

    def _run_pairs(self):
        if not self._check(): return
        
        # 1. Ask the user what stock they want to pair against the current one
        peer, ok = QInputDialog.getText(
            self, "Statistical Arbitrage", 
            f"Enter the peer ticker to pair trade against {self.ticker}\n(e.g., if analyzing KO, enter CVX):"
        )
        if not ok or not peer.strip():
            return
            
        peer = peer.strip().upper()
        self.output.clear()
        self.prog_lbl.setText(f"⟳ Calculating Cointegration for {self.ticker} vs {peer}…")
        BUS.status_msg.emit(f"Running Pairs Analysis for {self.ticker} vs {peer}…")
        
        # 2. Run the heavy math in a proper PyQt QThread so it doesn't silently crash
        class PairsThread(QThread):
            done_sig = pyqtSignal(str, dict)
            
            def __init__(self, t, p):
                super().__init__()
                self.t = t
                self.p = p
                
            def run(self):
                res = DE.cointegration_analysis(self.t, self.p)
                self.done_sig.emit(self.p, res)
                
        self._pairs_thread = PairsThread(self.ticker, peer)
        self._pairs_thread.done_sig.connect(self._on_pairs_data)
        self._pairs_thread.start()
        
    def _on_pairs_data(self, peer, res):
        if res.get('error'):
            self.output.setHtml(f"<p style='color:{C['red']}; padding:20px;'>⚠ Error: {res['error']}</p>")
            self.prog_lbl.setText("✗ Analysis failed")
            return
            
        # 3. Feed the math to the AI to write the Arbitrage Report
        prompt = f"""Act as a quantitative arbitrage trader. Analyze the following pairs trading spread between {self.ticker} and {peer}.

[STATISTICAL DATA]
Asset A: {self.ticker}
Asset B: {peer}
Cointegration P-Value: {res['coint_pvalue']:.4f} (Statistically valid if < 0.05)
Current Price Ratio ({self.ticker}/{peer}): {res['spread_current']:.4f}
Rolling 60-day Z-Score: {res['z_score']:+.2f}  (Extreme divergence if > 2.0 or < -2.0)
Statistical Signal: {res['signal']}

[ALGORITHMIC INTERPRETATION]
{res['interpretation']}

Provide a professional pairs trading strategy report covering:
1. **STATISTICAL VALIDITY**: Explain the p-value. Do these two assets truly move together historically, or is this a spurious correlation?
2. **CURRENT DIVERGENCE**: Explain the Z-score. Is the spread currently stretched? Who is overvalued and who is undervalued?
3. **TRADE EXECUTION**: Exactly how to execute this (e.g., "Go long X dollars of {self.ticker} and short X dollars of {peer} to remain market neutral").
4. **RISK MANAGEMENT**: What fundamental business event could permanently break this cointegration (causing the trade to blow up)?
5. **VERDICT**: Is this trade actionable RIGHT NOW, or should we keep it on the watchlist until the Z-Score expands further?
"""
        self._stream(prompt)   

    def _run_seasonality(self):
        if not self._check(): return
        
        self.output.clear()
        self.prog_lbl.setText(f"⟳ Crunching 20-Year Seasonality Matrix for {self.ticker}…")
        BUS.status_msg.emit(f"Running Seasonality Analysis for {self.ticker}…")
        
        # 🐛 FIX: Use a proper PyQt QThread instead of standard threading
        class SeasonalityThread(QThread):
            done_sig = pyqtSignal(dict)
            def __init__(self, t):
                super().__init__()
                self.t = t
            def run(self):
                res = DE.seasonality_analysis(self.t)
                self.done_sig.emit(res)
                
        self._sea_thread = SeasonalityThread(self.ticker)
        self._sea_thread.done_sig.connect(self._on_seasonality_data)
        self._sea_thread.start()

    def _on_seasonality_data(self, res):
        if res.get('error'):
            self.output.setHtml(f"<p style='color:{C['red']}; padding:20px;'>⚠ Error: {res['error']}</p>")
            self.prog_lbl.setText("✗ Analysis failed")
            return
            
        data = res['data']
        if not data: return
        
        # 1. Build the HTML Heatmap
        html = f"<div style='font-family:Consolas,monospace; margin-bottom: 20px;'>"
        html += f"<h3 style='color:{C['accent']}; margin-top:0;'>📅 {self.ticker} — YEARS SEASONALITY HEATMAP</h3>"
        html += f"<table style='width:100%; border-collapse:collapse; text-align:center; font-size:11px;'>"
        
        # Table Headers (Months)
        html += f"<tr style='background:{C['bg3']}; color:{C['text2']}; font-weight:bold;'>"
        for m in data.keys(): html += f"<td style='padding:6px; border:1px solid {C['border']};'>{m}</td>"
        html += "</tr>"
        
        # Win Rate Row (Color Coded)
        html += "<tr>"
        for m, stats in data.items():
            wr = stats['win_rate']
            # Dynamic gradient colors
            if wr >= 70:   bg = "#004d00" # Strong Green
            elif wr >= 60: bg = "#002b00" # Mild Green
            elif wr <= 30: bg = "#4d0000" # Strong Red
            elif wr <= 40: bg = "#2b0000" # Mild Red
            else:          bg = C['bg2']  # Neutral
            
            html += f"<td style='padding:8px; border:1px solid {C['border']}; background:{bg}; color:{C['text']}; font-weight:bold;'>{wr:.0f}%</td>"
        html += "</tr>"
        
        # Avg Return Row
        html += "<tr>"
        for m, stats in data.items():
            ret = stats['avg_return']
            col = C['green'] if ret > 0 else C['red']
            html += f"<td style='padding:8px; border:1px solid {C['border']}; background:{C['bg4']}; color:{col};'>{ret:+.1f}%</td>"
        html += "</tr>"
        
        html += "</table></div>"
        
        # Inject the heatmap into the UI immediately
        self.output.setHtml(html)
        
        # 2. Feed the raw data to the AI to analyze the next 60 days
        raw_text = "\n".join([f"{m}: WinRate={s['win_rate']:.0f}%, AvgRet={s['avg_return']:+.2f}%" for m, s in data.items()])
        
        prompt = f"""Act as a quantitative seasonal analyst. Here is the strict historical monthly performance for {self.ticker} over the last years.

[SEASONAL DATA]
{raw_text}

Provide a concise, professional analysis covering:
1. **BEST & WORST PERIODS**: Identify the strongest and weakest multi-month stretches.
2. **CURRENT CONTEXT**: Looking at the current calendar date, what does the seasonality dictate for the NEXT 60 DAYS?
3. **BEHAVIORAL DRIVERS**: What fundamental or institutional factors usually drive this specific cycle?
4. **ACTIONABLE VERDICT**: How should a portfolio manager position themselves right now based strictly on this calendar edge?
"""
        # 🐛 FIX: Custom AI streaming loop to preserve the Heatmap HTML above it
        self.prog_lbl.setText("◌ Generating AI analysis…")
        self._ai_buf =[]
        
        def _chunk(tok):
            tok = _fix_encoding(tok)
            self._ai_buf.append(tok)
            cur = self.output.textCursor()
            cur.movePosition(cur.MoveOperation.End if hasattr(cur, 'MoveOperation') else cur.End)
            cur.insertText(tok)
            self.output.setTextCursor(cur)
            self.output.ensureCursorVisible()
            
        def _done():
            final_markdown = "".join(self._ai_buf)
            # Combine the Heatmap HTML with the rendered AI Markdown
            final_html = html + md_to_html(final_markdown)
            self.output.clear()
            self.output.setHtml(final_html)
            self.prog_lbl.setText("✓ Seasonality analysis complete")
            BUS.status_msg.emit("Seasonality analysis complete")
            
        self._st = StreamThread([{"role": "user", "content": prompt}],
            system="You are a quantitative seasonal analyst. Use bullet points and be highly specific.",
            max_tokens=1500
        )
        self._st.chunk.connect(_chunk)
        self._st.done.connect(_done)
        self._st.start()    

    def _run_pod(self):
        """Pod Architecture: 3 specialist agents + 1 orchestrator synthesiser."""
        if not self._check(): return
        t = self.ticker

        # ── Gather data for each agent ────────────────────────────────────
        self.output.clear()
        self.prog_lbl.setText("⟳ Fetching data for all agents…")
        BUS.status_msg.emit(f"Pod Analysis starting for {t}…")

        try:
            df   = DE.fetch(t, period='1y', interval='1d')
            df   = DE.add_indicators(df) if df is not None else None
            inf  = DE.info(t)

            # ── NEW: HMM + Kelly + Regime Gate ──────────────────────────
            hmm_result  = HMM_DETECTOR.fit_predict(df) if df is not None else {}

            sc       = DE.statistical_confidence(df) if df is not None else {}
            rsi_sig  = sc.get('RSI_Oversold', (0, 1, ''))
            win_rate = rsi_sig[0] / max(rsi_sig[1], 1)

            last_row    = df.iloc[-1] if df is not None and len(df) > 0 else {}
            rule_regime = str(last_row.get('Market_Regime', 'UNKNOWN')) \
                          if hasattr(last_row, 'get') else 'UNKNOWN'
            raw_signals = [
                'rsi_oversold', 'macd', 'momentum',
                'mean_reversion', 'breakout', 'obv',
            ]
            gate_result = REGIME_FILTER.filter(
                raw_signals, rule_regime, hmm_result.get('regime')
            )
            try:
                mtf_result = MTF_ANALYZER.analyze(t) if 'MTF_ANALYZER' in dir() else {}
            except Exception:
                mtf_result = {}

            try:
                wf_result = WALK_FORWARD.run(df) if 'WALK_FORWARD' in dir() and df is not None else {}
            except Exception:
                wf_result = {}

            try:
                mc_result = MC_DCF.run(t, inf) if 'MC_DCF' in dir() else {}
            except Exception:
                mc_result = {}

            mc_text = str(mc_result.get('percentiles', {}).get('p50', 'N/A'))
            fft_res = DE.fft_extrapolation(df) if df is not None else None
            kelly = kelly_criterion(
                win_rate  = win_rate,
                avg_win   = 0.07,
                avg_loss  = 0.04,
                kelly_mod = gate_result.get('kelly_mod', 1.0),
            )
            
            # ── NEW: HMM + Kelly + Regime Gate + XGBoost ────────────────
            hmm_result  = HMM_DETECTOR.fit_predict(df) if df is not None else {}
            
            # 🚀 RUN XGBOOST ML ENGINE
            xgb_result = XGB_ENGINE.predict(df) if df is not None else {}
            
            # ────────────────────────────────────────────────────────────

            pcr  = DE.put_call_ratio(t)
            earn = DE.earnings_proximity(t)
            insider = DE.get_insider_activity(t)
            
            # ── Persist to Memory ─────────────────────────────────────────
            try:
                MEMORY.store(t, {
                    'signal':    results.get('orchestrator', '')[:80] if 'results' in dir() else 'N/A',
                    'thesis':    mc_text[:120],
                    'kelly_half': kelly.get('half_kelly', 'N/A'),
                    'mtf_bias':  mtf_result.get('bias', 'N/A'),
                })
            except Exception:
                pass

        except Exception as e:
            self.output.setPlainText(f"Data fetch error: {e}")
            return
        last = df.iloc[-1] if df is not None and len(df) > 0 else {}

        def g(col, fmt='.2f'):
            try: return format(float(last.get(col, 'N/A')), fmt)
            except: return str(last.get(col, 'N/A'))
            
        # GBM Probability Calculation for the AI
        try:
            log_r = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            s0_gbm = float(df['Close'].iloc[-1])
            mu_d, sig_d = log_r.mean(), log_r.std()
            gbm_up = s0_gbm * np.exp(mu_d * 30 + 1.96 * sig_d * np.sqrt(30))
            gbm_dn = s0_gbm * np.exp(mu_d * 30 - 1.96 * sig_d * np.sqrt(30))
            gbm_str = f"${gbm_dn:.2f} to ${gbm_up:.2f}"
        except:
            gbm_str = "N/A"

        # ── PRO TREND TRADER (PTT) STATE FOR AI ──
        ptt_status = "No active PTT trade."
        if df is not None and 'ptt_trades' in df.attrs and len(df.attrs['ptt_trades']) > 0:
            last_trade = df.attrs['ptt_trades'][-1]
            if last_trade['exit_idx'] == len(df) - 1:  # If the exit index is the current live bar, it's active!
                tdir = "LONG" if last_trade['dir'] == 1 else "SHORT"
                entry_p = last_trade['entry_p']
                sl_p = last_trade['sl_p']
                hits = len(last_trade['tp_hits'])
                next_tp = last_trade['tps'][hits] if hits < len(last_trade['tps']) else "All TPs Hit"
                
                # 👇 FIX: Format the number ONLY if it is actually a number!
                if isinstance(next_tp, (float, int, np.floating, np.integer)):
                    next_tp = f"${next_tp:.2f}"
                    
                ptt_status = f"ACTIVE {tdir} | Entry: ${entry_p:.2f} | Current SL: ${sl_p:.2f} | Targets Hit: {hits}/4 | Next Target: {next_tp}"

        # Calculate VMD for the AI
        vmd_res = DE.vmd_extrapolation(df) if df is not None else None
        vmd_text = f"${vmd_res['target_30d']:.2f}" if vmd_res else "N/A"        
                
        # ── Tech data block ───────────────────────────────────────────────
        price = g('Close')
        tech_data = f"""Ticker: {t}  |  Price: ${price}
RSI (14):          {g('RSI')}
MACD:              {g('MACD')}  |  Signal: {g('Signal')}  |  Hist: {g('MHist')}
BB Upper:          {g('BB_U')}  |  BB Lower: {g('BB_L')}  |  MA20: {g('MA20')}
MA50:              {g('MA50')}  |  MA200: {g('MA200')}
VWAP:              {g('VWAP')}
ATR (14):          {g('ATR')}
OBV trend:         {'Rising' if df is not None and len(df)>10 and float(df['OBV'].iloc[-1]) > float(df['OBV'].iloc[-10]) else 'Falling'}
CMF (20):          {g('CMF')}
RVI:               {g('RVI')}  |  RVI Signal: {g('RVI_Signal')}
Market Regime:     {last.get('Market_Regime', 'N/A')}
RSI Bull Div:      {last.get('RSI_Bull_Div', False)}
RSI Bear Div:      {last.get('RSI_Bear_Div', False)}
VWAP Deviation:    {g('VWAP_Dev_Pct')}%  |  Signal: {last.get('VWAP_Signal', 'N/A')}
VWAP (Cumulative): {g('VWAP_Cum')}  |  Upper2: {g('VWAP_Upper2')}  |  Lower2: {g('VWAP_Lower2')}
Volume Profile:    POC: ${g('VP_POC')}  |  VAH: ${g('VP_VAH')}  |  VAL: ${g('VP_VAL')}
VP Position:       {last.get('VP_Signal', 'N/A')}
HMM Regime:        {hmm_result.get('regime', 'N/A')} ({hmm_result.get('confidence', 0)*100:.0f}% confidence)  |  Method: {hmm_result.get('method', 'N/A')}
HMM Probs:         Bear={hmm_result.get('state_probs', [0,0,0])[0]*100:.0f}%  Sideways={hmm_result.get('state_probs', [0,0,0])[1]*100:.0f}%  Bull={hmm_result.get('state_probs', [0,0,0])[2]*100:.0f}%
Regime Gate:       Active={gate_result.get('gate_active', False)}  |  {gate_result.get('reason', 'No suppression')}
Kelly (Half):      {kelly.get('half_kelly', 0)}% of portfolio  |  Full: {kelly.get('kelly', 0)}%  |  Regime mod: {kelly.get('regime_mod', 1.0):.2f}x
MTF Confluence:    {mtf_result.get('bias', 'N/A')} ({mtf_result.get('confluence_score', 0)}% bull)
MTF Signals:       D={mtf_result.get('signals', {}).get('Daily','?')} | W={mtf_result.get('signals', {}).get('Weekly','?')} | M={mtf_result.get('signals', {}).get('Monthly','?')}
Walk-Fwd WinRate:  RSI={wf_result.get('signals',{}).get('RSI_Oversold',{}).get('win_rate','N/A')}% | MACD={wf_result.get('signals',{}).get('MACD_Cross',{}).get('win_rate','N/A')}% | BB={wf_result.get('signals',{}).get('BB_Bounce',{}).get('win_rate','N/A')}%
MC DCF P50:        ${mc_result.get('percentiles',{}).get('p50','N/A')} (upside {mc_result.get('upside_p50','N/A')}%)
FFT Cycle Forecast: Next Peak in {fft_res['next_peak_days']} days. Next Trough in {fft_res['next_trough_days']} days.
HMM Probs:         Bear={hmm_result.get('state_probs', [0,0,0])[0]*100:.0f}%  Sideways={hmm_result.get('state_probs', [0,0,0])[1]*100:.0f}%  Bull={hmm_result.get('state_probs',[0,0,0])[2]*100:.0f}%
ML Prediction:     {xgb_result.get('interpretation', 'N/A')}
Top Drivers:       {', '.join([f"{k} ({v*100:.1f}%)" for k, v in xgb_result.get('top_features', [])]) if 'top_features' in xgb_result else 'N/A'}
Regime Gate:       Active={gate_result.get('gate_active', False)}  |  {gate_result.get('reason', 'No suppression')}
RVI:               {g('RVI')}  |  RVI Signal: {g('RVI_Signal')}
Kalman True State: ${g('Kalman')}
Kalman Deviation:  {g('Kalman_Diff')}% (High deviation = pure noise, expect mean reversion to True State)
Market Regime:     {last.get('Market_Regime', 'N/A')}
Kalman True State: ${g('Kalman')}
Kalman Deviation:  {g('Kalman_Diff')}% 
PTT Algorithmic State: {ptt_status}  
Market Regime:     {last.get('Market_Regime', 'N/A')}
VMD Noise-Free Target (30d): {vmd_text}"""

        # ── Fundamental data block ────────────────────────────────────────
        def fi(k): return inf.get(k, 'N/A')
        fund_data = f"""Ticker: {t}
Current Price:     ${price}
Market Cap:        {fi('marketCap')}
P/E (TTM):         {fi('trailingPE')}
Fwd P/E:           {fi('forwardPE')}
EV/EBITDA:         {fi('enterpriseToEbitda')}
Price/Book:        {fi('priceToBook')}
Price/Sales:       {fi('priceToSalesTrailing12Months')}
Rev Growth (YoY):  {fi('revenueGrowth')}
Gross Margin:      {fi('grossMargins')}
Operating Margin:  {fi('operatingMargins')}
Net Margin:        {fi('profitMargins')}
FCF:               {fi('freeCashflow')}
Total Debt:        {fi('totalDebt')}
Cash:              {fi('totalCash')}
ROE:               {fi('returnOnEquity')}
52W High:          {fi('fiftyTwoWeekHigh')}  |  52W Low: {fi('fiftyTwoWeekLow')}
Analyst Target:    {fi('targetMeanPrice')}
Recommendation:    {fi('recommendationKey')}
52W High:          {fi('fiftyTwoWeekHigh')}  |  52W Low: {fi('fiftyTwoWeekLow')}
Analyst Target:    {fi('targetMeanPrice')}
Recommendation:    {fi('recommendationKey')}

CORPORATE INSIDER ACTIVITY (Last 6 Months):
  Signal:            {insider.get('signal', 'N/A')}
  Insider Ownership: {insider.get('insider_pct', 0):.2f}%
  Activity Details:  {insider.get('interpretation', 'N/A')}"""

        # ── Sentiment / risk data block ───────────────────────────────────
        sent_data = f"""Ticker: {t}
OPTIONS FLOW:
  Put/Call Ratio (OI):   {pcr.get('pcr', 'N/A')}
  Put/Call Ratio (Vol):  {pcr.get('pcr_volume', 'N/A')}
  Options Signal:        {pcr.get('smart_money_signal', 'N/A')}
  Interpretation:        {pcr.get('interpretation', 'N/A')}
  PCR Velocity Signal:   {pcr.get('velocity_signal', 'N/A')}
  Velocity Detail:       {pcr.get('velocity_interp', 'N/A')}

EARNINGS PROXIMITY:
  Next Earnings:         {earn.get('next_earnings_date', 'N/A')} ({earn.get('days_to_earnings', '?')}d away)
  Signal Reliability:    {earn.get('signal_reliability', 'NORMAL')}
  Warning:               {earn.get('note', '')}

BETA / VOLATILITY:
  Beta:                  {fi('beta')}
  ATR%:                  {g('ATR')} (absolute)
  Short Interest:        {fi('shortPercentOfFloat')}
  Short Ratio:           {fi('shortRatio')}"""

        # ── Multi-Timeframe Confluence ───────────────────────────────────
        mtf_result = MTF.analyze(t)

        # ── Walk-Forward Backtest ────────────────────────────────────────
        wf_result  = WF_BACKTEST.run(df) if df is not None else {}
        wf_text    = WF_BACKTEST.format(wf_result, t)

        # ── Monte Carlo DCF ──────────────────────────────────────────────
        mc_result  = MC_DCF.run(inf)
        mc_text    = MC_DCF.format(mc_result, t)

        # ── Wire up PodThread ─────────────────────────────────────────────
        self._pod_buf = []
        self._pod_phase_html = []

        def on_phase(label):
            html = self.output.toHtml()
            phase_block = (
                f'<p style="color:{C["accent"]}; font-weight:bold; '
                f'font-size:12px; margin-top:14px; margin-bottom:2px; '
                f'letter-spacing:1px;">▸ {label}</p>'
                f'<p style="color:{C["text3"]}; font-size:10px; margin:0;">'
                f'Processing…</p>'
            )
            self.output.setHtml(html + phase_block)
            self.prog_lbl.setText(f"⟳ {label}")
            sb = self.output.verticalScrollBar()
            sb.setValue(sb.maximum())

        def on_chunk(tok):
            self._pod_buf.append(tok)   # 🐛 FIX: Was self.pod_buf
            def _insert():
                cur = self.output.textCursor()
                cur.movePosition(cur.MoveOperation.End if hasattr(cur, 'MoveOperation') else cur.End)
                cur.insertText(tok)
                self.output.setTextCursor(cur)
                self.output.ensureCursorVisible()
            QTimer.singleShot(0, _insert)

        def on_done(results):
            full_md = (
                f"## CHARTIST\n{results.get('chartist','')}\n\n"
                f"## FUNDAMENTALIST\n{results.get('fundamentalist','')}\n\n"
                f"## NEWS SENTRY\n{results.get('sentry','')}\n\n"
                f"## MACRO SENTINEL\n{results.get('sentinel','')}\n\n"
                f"## ORCHESTRATOR — FINAL VERDICT\n{results.get('orchestrator','')}"
            )
            def _update():
                self.output.clear()
                self.output.setHtml(md_to_html(full_md))
                sb = self.output.verticalScrollBar()
                sb.setValue(sb.maximum())
                self.prog_lbl.setText("Pod Analysis complete — consensus generated") # 🐛 FIX: Was self.proglbl
                BUS.status_msg.emit(f"Pod Analysis complete for {t}")
            QTimer.singleShot(0, _update)

        def on_error(msg):
            self.output.append(f"\n\n⚠ Pod error: {msg}")
            self.prog_lbl.setText("✗ Pod Analysis failed")

        self._pod_thread = PodThread(
            t, tech_data, fund_data, sent_data,
            hmm_result=hmm_result, 
            gate_result=gate_result,
            kelly=kelly,
            xgb_result=xgb_result
        )
        self._pod_thread.phase.connect(on_phase)
        self._pod_thread.chunk.connect(on_chunk)
        self._pod_thread.done.connect(on_done)
        self._pod_thread.error.connect(on_error)
        self._pod_thread.start()

    def _stream(self, prompt):
        self.output.clear()
        self.prog_lbl.setText("◌ Generating analysis…")
        system = "You are a senior equity research analyst at a bulge-bracket bank. Be specific, structured, and professional."
        self._st = stream_into(
            self.output,
            [{"role": "user", "content": prompt}],
            system=system,
            max_tokens=3000,
            on_done=lambda: self.prog_lbl.setText("✓ Analysis complete"),
        )
        BUS.status_msg.emit(f"Running analysis on {self.ticker}…")


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PANEL
# ═══════════════════════════════════════════════════════════════════════════════
class DocumentPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.doc_text = ""
        self._st      = None
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # ── Toolbar ───────────────────────────────────────────────────
        tb = QHBoxLayout()
        
        load_btn = QPushButton("📂  LOAD DOCUMENT")
        load_btn.setFixedHeight(30)
        load_btn.setProperty("accent", "true")
        load_btn.setStyleSheet(f"""
            QPushButton {{ background:{C['accent']}; border:none; color:white; font-weight:bold; }}
            QPushButton:hover {{ background:{C['accent2']}; }}
        """)
        load_btn.clicked.connect(self._load)
        tb.addWidget(load_btn)

        # 👇 ADD THE UNLOAD BUTTON HERE 👇
        unload_btn = QPushButton("✕  UNLOAD")
        unload_btn.setFixedHeight(30)
        unload_btn.setStyleSheet(f"""
            QPushButton {{ background:{C['bg4']}; border:1px solid {C['border']}; color:{C['text3']}; font-weight:bold; }}
            QPushButton:hover {{ background:{C['red']}; color:white; border-color:{C['red']}; }}
        """)
        unload_btn.clicked.connect(self._unload)
        tb.addWidget(unload_btn)
        # 👆 END OF NEW BUTTON 👆

        self.file_lbl = QLabel("No document loaded")
        self.file_lbl.setStyleSheet(f"color:{C['text2']}; font-size:10px; padding-left:6px;")
        tb.addWidget(self.file_lbl)
        tb.addStretch()

        for label, slot in [
            ("◈ Summarize",           self._summarize),
            ("$ Extract Financials",  self._extract),
            ("📋 Key Highlights",     self._highlights),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(slot)
            tb.addWidget(btn)
        lay.addLayout(tb)

        btn_nlp = QPushButton("📋 TRANSCRIPT NLP")
        btn_nlp.setFixedHeight(30)
        btn_nlp.setStyleSheet(f"""
            QPushButton {{ background: #1a1a2e; border: 1px solid {C['purple']};
                          color: {C['purple']}; font-weight: bold; font-size: 11px; }}
            QPushButton:hover {{ background: {C['purple']}; color: white; }}
        """)
        btn_nlp.clicked.connect(self._run_transcript_nlp)
        tb.addWidget(btn_nlp)

        # ── Main split ────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Document pane
        doc_grp = QGroupBox("DOCUMENT")
        dg_lay  = QVBoxLayout(doc_grp)
        dg_lay.setContentsMargins(4, 4, 4, 4)
        self.doc_view = QPlainTextEdit()
        self.doc_view.setReadOnly(True)
        self.doc_view.setPlaceholderText(
            "Load a document (PDF, TXT, CSV, JSON) to begin…\n\n"
            "Supports: earnings reports, 10-K/Q filings, analyst notes, articles."
        )
        dg_lay.addWidget(self.doc_view)
        splitter.addWidget(doc_grp)

        # AI output pane
        ai_grp = QGroupBox("AI ANALYSIS")
        ag_lay = QVBoxLayout(ai_grp)
        ag_lay.setContentsMargins(4, 4, 4, 4)
        self.ai_view = QTextEdit()
        self.ai_view.setReadOnly(True)
        ag_lay.addWidget(self.ai_view)

        qa_row = QHBoxLayout()
        self.qa_inp = QLineEdit()
        self.qa_inp.setPlaceholderText("Ask a question about the document…")
        self.qa_inp.setFixedHeight(28)
        self.qa_inp.returnPressed.connect(self._ask)
        qa_row.addWidget(self.qa_inp)
        ask_btn = QPushButton("Ask")
        ask_btn.setFixedHeight(28)
        ask_btn.clicked.connect(self._ask)
        qa_row.addWidget(ask_btn)
        ag_lay.addLayout(qa_row)
        splitter.addWidget(ai_grp)

        splitter.setSizes([550, 500])
        lay.addWidget(splitter)

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Document", "",
            "All (*.pdf *.txt *.csv *.json *.md *.log);;"
            "PDF (*.pdf);;Text (*.txt *.md);;CSV (*.csv);;JSON (*.json)"
        )
        if not path:
            return
        # Check for missing pdfplumber before even trying
        if path.lower().endswith('.pdf') and not HAS_PDF:
            self.ai_view.setPlainText(
                "⚠  pdfplumber is not installed.\n\n"
                "Run this command in your terminal:\n\n"
                "    pip install pdfplumber\n\n"
                "Then restart the app and try loading the document again."
            )
            self.file_lbl.setText("✗  pdfplumber missing — pip install pdfplumber")
            return
        self.file_lbl.setText(f"⟳  Loading {os.path.basename(path)}…")
        self.doc_text = ""  # reset while loading
        # Store path on self so the lambda captures self, not the local var
        self._loading_path = path
        def _work():
            try:
                text = DocParser.parse(self._loading_path)
            except Exception as exc:
                text = f"Error loading document:\n{exc}"
            # Use a stored reference to avoid lambda closure issues in PyQt6
            self._pending_text = text
            QTimer.singleShot(0, self._finish_load)
        threading.Thread(target=_work, daemon=True).start()

    def _finish_load(self):
        path = getattr(self, '_loading_path', '')
        text = getattr(self, '_pending_text', '')
        self._loaded(path, text)

    def _loaded(self, path, text):
        self.doc_text = text
        self.doc_view.setPlainText(text)
        self.file_lbl.setText(f"✓  {os.path.basename(path)}  ({len(text):,} chars)")
        BUS.status_msg.emit(f"Document loaded: {os.path.basename(path)}")
        
        # Broadcast the document text to the Probability Panel and the rest of the app!
        BUS.doc_loaded.emit(os.path.basename(path), text)

    # 👇 ADD THIS NEW METHOD 👇
    def _unload(self):
        """Clears the document from memory and frees up the AI context window."""
        self.doc_text = ""
        self.doc_view.clear()
        self.ai_view.clear()
        self.file_lbl.setText("No document loaded")
        
        # Broadcast an empty string to force ProbabilityPanel to drop the document from memory
        BUS.doc_loaded.emit("", "")
        BUS.status_msg.emit("Document cleared from memory.")

    def _check(self):
        if not self.doc_text:
            self.ai_view.setPlainText(
                "⚠ Load a document first.\n\n"
                "Click the orange  📂 LOAD DOCUMENT  button above."
            )
            return False
        if not LM.connected:
            self.ai_view.setPlainText(
                "⚠ LM Studio not connected.\n\n"
                "Start LM Studio, load a model, then try again."
            )
            return False
        return True

    def _summarize(self):
        if not self._check(): return
        chunk = self.doc_text[:9000]
        self._run(f"""Analyze this financial document and provide:

1. DOCUMENT TYPE & CONTEXT
2. KEY FINANCIAL METRICS (revenue, EPS, margins, FCF, guidance)
3. MANAGEMENT COMMENTARY (tone, key themes, forward-looking statements)
4. POSITIVE HIGHLIGHTS
5. CONCERNS & RED FLAGS
6. INVESTMENT IMPLICATIONS (BUY/SELL/HOLD signal and reasoning)

Document:
{chunk}""")

    def _extract(self):
        if not self._check(): return
        chunk = self.doc_text[:9000]
        self._run(f"""Extract ALL financial data from this document into structured tables:

● INCOME STATEMENT  (Revenue, Gross Profit, EBIT, EBITDA, Net Income, EPS)
● BALANCE SHEET  (Cash, Total Assets, Total Debt, Equity)
● CASH FLOW  (Operating CF, Capex, Free Cash Flow)
● KEY RATIOS & METRICS  (Margins, Growth rates, Return metrics)
● GUIDANCE / FORWARD ESTIMATES
● YoY / QoQ COMPARISONS

Document:
{chunk}""")

    def _run_transcript_nlp(self):
        if not self._check(): return
        ticker = getattr(self, 'ticker', None) or 'UNKNOWN'
        self.ai_view.clear()

        # 🐛 FIX: Use a proper QThread so we can safely pass signals back to the GUI
        class NLPThread(QThread):
            chunk = pyqtSignal(str)
            done_sig = pyqtSignal(str)
            
            def __init__(self, t, text):
                super().__init__()
                self.t = t
                self.text = text
                
            def run(self):
                TRANSCRIPT_NLP.analyze(
                    self.t, self.text,
                    on_chunk=self.chunk.emit,
                    on_done=self.done_sig.emit
                )

        self._nlp_thread = NLPThread(ticker, self.doc_text)

        def on_chunk(tok):
            cur = self.ai_view.textCursor()
            cur.movePosition(cur.MoveOperation.End if hasattr(cur, 'MoveOperation') else cur.End)
            cur.insertText(tok)
            self.ai_view.setTextCursor(cur)
            self.ai_view.ensureCursorVisible()

        def on_done(text):
            self.ai_view.clear()
            self.ai_view.setHtml(md_to_html(text))
            BUS.status_msg.emit(f"Transcript NLP complete for {ticker}")

        self._nlp_thread.chunk.connect(on_chunk)
        self._nlp_thread.done_sig.connect(on_done)
        self._nlp_thread.start()

        BUS.status_msg.emit(f"Running Transcript NLP for {ticker}…")

    def _highlights(self):
        if not self._check(): return
        chunk = self.doc_text[:9000]
        self._run(f"""From this document extract:

● TOP 10 KEY QUOTES (verbatim, with context)
● 5 MOST IMPORTANT NUMBERS (with explanation)
● 3 BIGGEST SURPRISES (positive or negative vs expectations)
● WHAT CHANGED vs PRIOR PERIOD
● ONE SENTENCE BOTTOM LINE

Document:
{chunk}""")

    def _ask(self):
        q = self.qa_inp.text().strip()
        if not q or not self.doc_text: return
        if not LM.connected: return
        self.qa_inp.clear()
        chunk = self.doc_text[:9000]
        self._run(f"""Based on the following document, answer this question:

QUESTION: {q}

Provide a direct, specific answer. Quote or reference the relevant passage if applicable.

Document:
{chunk}""")

    def _run(self, prompt):
        self.ai_view.clear()

        def _doc_done():
            txt = self.ai_view.toPlainText().strip()
            if not txt:
                self.ai_view.setPlainText(
                    "⚠ No text returned by LM Studio.\n\n"
                    "Check:\n"
                    "  1. A model is fully loaded in LM Studio\n"
                    "  2. LM Studio → Logs tab for errors\n"
                    "  3. Try a shorter document or lower max_tokens"
                )
            BUS.status_msg.emit("Analysis complete")

        self._st = stream_into(
            self.ai_view,
            [{"role": "user", "content": prompt}],
            system="You are a financial document analyst. Be precise and structured.",
            max_tokens=2000,
            on_done=_doc_done,
        )
        BUS.status_msg.emit("Analyzing document…")
# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY PANEL
# ═══════════════════════════════════════════════════════════════════════════════
class ProbabilityPanel(QWidget):
    MODELS =[
        ("🎲  Monte Carlo Simulation",   "monte_carlo"),
        ("📊  Bayesian Inference",        "bayesian"),
        ("📉  Value at Risk (VaR/CVaR)",  "var"),
        ("🔀  Scenario Analysis",         "scenario"),
        ("🧮  Kelly Criterion",           "kelly"),
        ("📐  Black-Scholes Probability", "bsm"),
        ("🎛️  Kalman State Estimation",   "kalman_prob"),
        ("🚀  Merton Jump-Diffusion",     "merton"), 
        ("🌪️  Heston Stochastic Vol.",    "heston"), 
    ]

    def __init__(self):
        super().__init__()
        self.doc_text = ""
        self.ticker   = ""
        self._st      = None
        self._setup_ui()
        BUS.ticker_changed.connect(lambda t: setattr(self, 'ticker', t))
        BUS.doc_loaded.connect(lambda fn, txt: setattr(self, 'doc_text', txt))

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        hdr = QLabel("⬡  ADVANCED PROBABILITY ENGINE")
        hdr.setStyleSheet(f"color:{C['accent']};font-size:11px;font-weight:bold;letter-spacing:1.2px;padding:4px 0;")
        lay.addWidget(hdr)

        info = QLabel("Load a document in the DOCUMENTS tab — the AI uses it as context for every calculation.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color:{C['text2']};font-size:10px;padding-bottom:4px;")
        lay.addWidget(info)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("MODEL:"))
        self.model_combo = QComboBox()
        for label, _ in self.MODELS:
            self.model_combo.addItem(label)
        self.model_combo.setFixedHeight(30)
        sel_row.addWidget(self.model_combo)
        sel_row.addSpacing(16)

        for lbl_text, attr, default in [
            ("Simulations:", "_n_sims", "10000"),
            ("Horizon (days):", "_horizon", "252"),
            ("Confidence %:", "_conf", "95"),
        ]:
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(f"color:{C['text3']};font-size:10px;")
            sel_row.addWidget(lbl)
            inp = QLineEdit(default)
            inp.setFixedSize(64, 28)
            inp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sel_row.addWidget(inp)
            setattr(self, attr, inp)

        sel_row.addStretch()
        run_btn = QPushButton("▶  RUN ANALYSIS")
        run_btn.setFixedHeight(30)
        run_btn.setStyleSheet(
            f"QPushButton{{background:{C['accent']};border:none;color:white;font-weight:bold;padding:0 16px;}}"
            f"QPushButton:hover{{background:{C['accent2']};}}"
        )
        run_btn.clicked.connect(self._run)
        sel_row.addWidget(run_btn)
        lay.addLayout(sel_row)

        preset_row = QHBoxLayout()
        for label, idx in [("Upside prob", 0), ("VaR / Downside", 2), ("Bull/Bear/Base", 3), ("Kelly sizing", 4)]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet(
                f"QPushButton{{background:{C['bg4']};border:1px solid {C['border']};color:{C['text2']};font-size:10px;}}"
                f"QPushButton:hover{{background:{C['accent']};color:white;}}"
            )
            btn.clicked.connect(lambda _, i=idx: self._quick(i))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        lay.addLayout(preset_row)

        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setStyleSheet(
            f"QTextEdit{{background:{C['bg2']};border:1px solid {C['border']};padding:8px;font-family:Consolas,monospace;font-size:12px;}}"
        )
        self.result_view.setHtml(
            f'<div style="color:{C["text3"]};padding:20px;">'
            f'<p style="color:{C["accent"]};font-size:13px;font-weight:bold;">⬡ Probability Engine Ready</p>'
            f'<p>Select a model and click <b style="color:{C["accent2"]};">▶ RUN ANALYSIS</b>.</p>'
            f'<p>All results include probability estimates expressed as <b>%</b>.</p></div>'
        )
        lay.addWidget(self.result_view, 1)

        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        lay.addWidget(self.status_lbl)

    def _quick(self, i):
        self.model_combo.setCurrentIndex(i)
        self._run()

    def _params(self):
        def _i(w, d):
            try: return int(w.text())
            except: return d
        return _i(self._n_sims, 10000), _i(self._horizon, 252), _i(self._conf, 95)

    def _build_prompt(self, key):
        n, h, c = self._params()
        ticker = self.ticker or "the asset"
        
        # Safe defaults for advanced math
        curr_price = 100.0
        lam = mu_j = sig_j = prob_up_mjd = p95_mjd = p05_mjd = 0.0
        v0 = theta = xi = prob_up_h = p95_h = p05_h = 0.0
        rho = -0.5
        
        # ── NEW: FETCH LIVE DATA TO ANCHOR THE AI'S MATH ──
        live_data_str = ""
        if self.ticker:
            try:
                df = DE.fetch(self.ticker, period='1y', interval='1d')
                if df is not None and not df.empty:
                    curr_price = float(df['Close'].iloc[-1])
                    
                    # Calculate real annualized volatility and return
                    log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
                    ann_vol = log_ret.std() * np.sqrt(252) * 100
                    ann_ret = (float(df['Close'].iloc[-1]) / float(df['Close'].iloc[0]) - 1) * 100
                    
                    # 🚀 1. Kalman Filter State & Covariance
                    prices = df['Close'].values
                    kf_mean, kf_cov = np.zeros(len(prices)), np.zeros(len(prices))
                    kf_mean[0], kf_cov[0] = prices[0], 1.0
                    Q_noise, R_noise = 1e-4, 0.01 * np.var(prices) if np.var(prices) > 0 else 0.01
                    for i in range(1, len(prices)):
                        prior_mean, prior_cov = kf_mean[i-1], kf_cov[i-1] + Q_noise
                        K = prior_cov / (prior_cov + R_noise)
                        kf_mean[i] = prior_mean + K * (prices[i] - prior_mean)
                        kf_cov[i] = (1 - K) * prior_cov
                        
                    curr_kf_state, curr_kf_cov = kf_mean[-1], kf_cov[-1]
                    state_diff = (curr_price - curr_kf_state) / curr_kf_state * 100
                    kf_drift = (kf_mean[-1] - kf_mean[-21]) / 20 if len(kf_mean) > 20 else 0.0
                    
                    # 🚀 2. Merton Jump-Diffusion Simulation (10,000 paths)
                    dt, steps = 1/252, h
                    std_daily = log_ret.std()
                    jumps = log_ret[np.abs(log_ret) > 2.5 * std_daily] # Identify fat tails (>2.5 sigma)
                    lam = len(jumps) / (len(log_ret) / 252) if len(jumps) > 0 else 0.01
                    mu_j = jumps.mean() if len(jumps) > 0 else 0.0
                    sig_j = jumps.std() if len(jumps) > 0 else 0.01
                    
                    S_mjd = np.full(10000, curr_price)
                    for _ in range(steps):
                        z = np.random.standard_normal(10000)
                        n_jumps = np.random.poisson(lam * dt, 10000)
                        jump_sizes = np.random.normal(mu_j, sig_j, 10000) * n_jumps
                        S_mjd = S_mjd * np.exp((ann_ret/100 - 0.5*std_daily**2)*dt + std_daily*np.sqrt(dt)*z + jump_sizes)
                        
                    prob_up_mjd = (S_mjd > curr_price).mean() * 100
                    p95_mjd, p05_mjd = np.percentile(S_mjd, 95), np.percentile(S_mjd, 5)

                    # 🚀 3. Heston Stochastic Volatility Simulation (10,000 paths)
                    v0 = log_ret.tail(20).var() * 252
                    theta = log_ret.var() * 252
                    kappa = 2.0 # Reversion speed
                    rolling_var = log_ret.rolling(20).var() * 252
                    xi = rolling_var.std() if not pd.isna(rolling_var.std()) else 0.1 # Vol of Vol
                    
                    # Correlation between returns and variance
                    ret_aligned = log_ret.loc[rolling_var.dropna().index]
                    var_diff = rolling_var.diff().dropna()
                    ret_aligned = ret_aligned.loc[var_diff.index]
                    rho = np.corrcoef(ret_aligned, var_diff)[0,1] if len(ret_aligned)>1 else -0.5
                    
                    S_h, v_t = np.full(10000, curr_price), np.full(10000, v0)
                    for _ in range(steps):
                        z1 = np.random.standard_normal(10000)
                        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(10000)
                        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t * dt) * z2) # Feller boundary
                        S_h = S_h * np.exp((ann_ret/100 - 0.5 * v_t) * dt + np.sqrt(v_t * dt) * z1)
                        
                    prob_up_h = (S_h > curr_price).mean() * 100
                    p95_h, p05_h = np.percentile(S_h, 95), np.percentile(S_h, 5)

                    live_data_str = (
                        f"\n\n[LIVE MARKET DATA FOR {ticker}]\n"
                        f"Current Price (S): ${curr_price:.2f}\n"
                        f"Historical Annual Volatility (σ): {ann_vol:.2f}%\n"
                        f"Historical 1-Year Return (μ): {ann_ret:.2f}%\n"
                        f"CRITICAL RULE: You MUST use these exact Live Market Data values as your baseline inputs for your mathematical formulas. Do not invent assumptions."
                    )
                else:
                    p, _ = DE.fast_quote(self.ticker)
                    if p:
                        curr_price = p
                        live_data_str = f"\n\n[LIVE MARKET DATA FOR {ticker}]\nCurrent Price: ${p:.2f}\nCRITICAL RULE: Use this exact price in your formulas."
            except Exception:
                pass

        doc_ctx = (f"\n\n[DOCUMENT CONTEXT]:\n{self.doc_text[:6000]}"
                   if self.doc_text else "\n\n(No document loaded — rely strictly on the Live Market Data provided above.)")
                   
        base = f"You are a quantitative analyst. Asset: {ticker}.{live_data_str}{doc_ctx}\n\nSimulations:{n:,} | Horizon:{h}d | Confidence:{c}%\n\n"

        templates = {
            "monte_carlo": base + f"""Run a Monte Carlo simulation for {ticker} ({n:,} paths, {h}-day horizon).

1. **PARAMETER ESTIMATES** (from document if available, else use LIVE MARKET DATA)
   - Expected annual return μ: X%
   - Annual volatility σ: X%

2. **PRICE DISTRIBUTION** — table with percentiles P5/P10/P25/P50/P75/P90/P95

3. **KEY PROBABILITIES**
   - Probability price ABOVE current in {h} days: **X%**
   - Probability price BELOW current: **X%**
   - Expected return over horizon: **X%**

4. **{c}% CONFIDENCE INTERVAL** — lower/upper bound as % return

5. **INVESTMENT IMPLICATION**""",

            "bayesian": base + f"""Perform Bayesian probability analysis for {ticker}.
1. **PRIOR** (P(bull), P(neutral), P(bear))
2. **LIKELIHOOD UPDATE** from document
3. **POSTERIOR PROBABILITIES** (Table format)
4. **{c}% CREDIBLE INTERVAL**
5. **CONCLUSION**""",

            "var": base + f"""Calculate Value at Risk for {ticker} ({c}% confidence, {h}-day horizon).
1. **VOLATILITY INPUTS**
2. **VALUE AT RISK TABLE** (Parametric, Historical, Monte Carlo)
3. **CVaR (Expected Shortfall)**
4. **STRESS SCENARIOS** (Base, Adverse, Severe)
5. **RISK/REWARD VERDICT**""",

            "scenario": base + f"""Build a scenario analysis for {ticker}.
1. **BULL CASE** — Probability / Target return / Timeline
2. **BASE CASE** — Probability / Target return / Timeline
3. **BEAR CASE** — Probability / Target return / Timeline
4. **TAIL RISK** — Probability / Max downside
5. **SUMMARY TABLE**
6. **PROBABILITY-WEIGHTED EXPECTED RETURN**
7. **RECOMMENDATION**""",

            "kelly": base + f"""Calculate Kelly Criterion position sizing for {ticker}.
1. **INPUTS** (Win probability p, Avg win, Avg loss)
2. **KELLY FRACTIONS** (Full, Half, Quarter)
3. **EXPECTED VALUE**
4. **RISK OF RUIN**
5. **RECOMMENDED POSITION SIZE**""",

            "bsm": base + f"""Apply Black-Scholes probability analysis for {ticker}.
1. **INPUTS**: Use LIVE MARKET DATA exactly.
2. **PROBABILITY TABLE** (-20%, Flat, +10%, +25%)
3. **KEY RESULTS** (P(above current), P(below current))
4. **{c}% EXPECTED MOVE** over {h} days
5. **CONCLUSION**""",

            "kalman_prob": base + f"""Run a Kalman Filter Probability Projection for {ticker} ({h}-day horizon).
1. **STATE VS NOISE ANALYSIS** 
   - Current Price: ${curr_price:.2f} | Kalman True State: ${curr_kf_state:.2f} | Deviation: {state_diff:+.2f}%
   - Is current price action fundamentally justified by the State, or is it measurement noise?
2. **FORWARD PROJECTION (HORIZON: {h} DAYS)**
   - Using the Kalman Daily Drift (${kf_drift:+.3f}/day), project the expected True State in {h} days.
3. **PROBABILITIES**
   - Probability that price mean-reverts to the True State.
4. **{c}% CONFIDENCE INTERVAL**
5. **VERDICT**""",

            "merton": base + f"""Run a Merton Jump-Diffusion Probability Analysis for {ticker} ({h}-day horizon).

[MERTON JUMP-DIFFUSION SIMULATION RESULTS]
- Detected Jump Frequency (λ): {lam:.2f} jumps per year
- Mean Jump Size (μ_j): {mu_j*100:.2f}%
- Jump Volatility (σ_j): {sig_j*100:.2f}%
- Simulated Probability of closing > ${curr_price:.2f} (Current Price): {prob_up_mjd:.1f}%
- Simulated P95 Upside Bound: ${p95_mjd:.2f}
- Simulated P05 Downside Bound: ${p05_mjd:.2f}

Provide a structured analysis:
1. **JUMP RISK DYNAMICS**: Explain the detected jump parameters (λ, μ_j). What real-world events (e.g., earnings, FDA approvals) usually create these gaps for this asset?
2. **MERTON vs BLACK-SCHOLES**: How does this jump risk distort traditional options pricing for {ticker}? (Analyze the fat tails).
3. **PROBABILITY FORECAST**: Interpret the mathematical probability of a positive return and the P05/P95 bounds over the next {h} days.
4. **HEDGE FUND STRATEGY**: What options strategy (e.g., strangles, iron condors) or tail-risk hedge is optimal given this specific historical jump profile?""",

            "heston": base + f"""Run a Heston Stochastic Volatility Probability Analysis for {ticker} ({h}-day horizon).[HESTON MODEL SIMULATION RESULTS]
- Long-term Variance (θ): {theta*100:.2f}%
- Current Variance (v0): {v0*100:.2f}%
- Volatility of Volatility (ξ): {xi:.4f}
- Price/Variance Correlation (ρ): {rho:.2f} (Leverage effect)
- Simulated Probability of closing > ${curr_price:.2f} (Current Price): {prob_up_h:.1f}%
- Simulated P95 Upside Bound: ${p95_h:.2f}
- Simulated P05 Downside Bound: ${p05_h:.2f}

Provide a structured analysis:
1. **VOLATILITY REGIME**: Analyze Current Variance (v0) vs Long-term Variance (θ). Is the market currently in a volatility spike or a volatility squeeze?
2. **THE LEVERAGE EFFECT**: Analyze the Correlation (ρ). Does volatility explode when {ticker} drops? What does this mean for downside risk?
3. **PROBABILITY FORECAST**: Interpret the mathematical probability of a positive return and the P05/P95 bounds over the next {h} days.
4. **OPTIONS PRICING IMPLICATION**: How does the 'Volatility of Volatility' (ξ) affect the volatility smile for this asset?
5. **RECOMMENDED ACTION**: How should a quantitative trader position themselves right now given the stochastic (randomly changing) nature of {ticker}'s volatility?"""
        }
        return templates.get(key, templates["monte_carlo"])

    def _run(self):
        if not LM.connected:
            self.result_view.setPlainText("⚠ LM Studio not connected.")
            return
        idx = self.model_combo.currentIndex()
        _, key = self.MODELS[idx]
        prompt = self._build_prompt(key)

        self.result_view.clear()
        self.result_view.append(
            f'<p style="color:{C["text3"]};font-size:10px;">⟳ Running {self.MODELS[idx][0].strip()}…</p>'
        )
        self.status_lbl.setText(f"Running {self.MODELS[idx][0].strip()}…")
        buf = ['']

        def _chunk(tok):
            tok = _fix_encoding(tok)
            buf[0] += tok
            cur = self.result_view.textCursor()
            cur.movePosition(cur.MoveOperation.End if hasattr(cur, 'MoveOperation') else cur.End)
            cur.insertText(tok)
            self.result_view.setTextCursor(cur)
            self.result_view.ensureCursorVisible()

        def _done():
            if buf[0].strip():
                self.result_view.clear()
                self.result_view.setHtml(md_to_html(buf[0]))
                self.result_view.verticalScrollBar().setValue(
                    self.result_view.verticalScrollBar().maximum()
                )
            self.status_lbl.setText("Complete.")
            BUS.status_msg.emit("Probability analysis complete")

        if self._st and self._st.isRunning():
            self._st.terminate()
        self._st = StreamThread(
            [{"role": "user", "content": prompt}],
            system=(
                "You are a quantitative finance expert. Express ALL probability estimates "
                "as exact percentages. Use structured Markdown with tables and headings. "
                "Every probability claim must include a % value and brief justification."
            ),
            temperature=0.2, max_tokens=3000,
        )
        self._st.chunk.connect(_chunk)
        self._st.done.connect(_done)
        self._st.start()

# ═══════════════════════════════════════════════════════════════════════════════
# TICKER NEWS FETCHER THREAD
# ═══════════════════════════════════════════════════════════════════════════════
class TickerNewsThread(QThread):
    ready = pyqtSignal(list)

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker

    def run(self):
        items =[]
        if not HAS_FEED:
            self.ready.emit(items)
            return
            
        # 1. Try Yahoo Finance RSS for the specific ticker
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.ticker}&region=US&lang=en-US"
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:15]:
                items.append({
                    'title':   e.get('title', '').strip(),
                    'summary': e.get('summary', '').strip()[:400],
                    'link':    e.get('link', ''),
                    'time':    e.get('published', ''),
                    'source':  feed.feed.get('title', url)[:20],
                })
        except Exception:
            pass
            
        # 2. Fallback to yfinance .news if RSS comes back empty
        if not items and HAS_YF:
            try:
                import yfinance as yf
                import datetime
                yf_news = yf.Ticker(self.ticker).news
                for n in yf_news[:15]:
                    pub_time = datetime.datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M:%S')
                    items.append({
                        'title': n.get('title', ''),
                        'summary': str(n.get('relatedTickers', '')),
                        'link': n.get('link', ''),
                        'time': pub_time,
                        'source': n.get('publisher', '')
                    })
            except Exception:
                pass
                
        self.ready.emit(items)


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS PANEL
# ═══════════════════════════════════════════════════════════════════════════════
class NewsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.articles =[]
        self.current_ticker = ""
        self._st = None
        self._setup_ui()
        # Automatically update the ticker search box when you change tickers globally
        BUS.ticker_changed.connect(self._on_global_ticker)

    def _on_global_ticker(self, t):
        self.current_ticker = t
        self.ticker_search.setText(t)

    def refresh(self, articles):
        """Called by MainWindow when news is ready."""
        self.articles = articles
        self.list_w.setRowCount(len(articles))
        for i, a in enumerate(articles):
            src = QTableWidgetItem(a.get('source', '')[:18])
            ttl = QTableWidgetItem(a.get('title', ''))
            tme = QTableWidgetItem(a.get('time', '')[:10])
            src.setForeground(QColor(C['accent2']))
            self.list_w.setItem(i, 0, src)
            self.list_w.setItem(i, 1, ttl)
            self.list_w.setItem(i, 2, tme)
            self.list_w.setRowHeight(i, 26)
        self.stat_lbl.setText(f"✓  {len(articles)} articles loaded")

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        tb = QHBoxLayout()
        lbl = QLabel("◈ NEWS FEED")
        lbl.setStyleSheet(f"color:{C['accent']}; font-weight:bold; font-size:12px; letter-spacing:1px; margin-right:10px;")
        tb.addWidget(lbl)
        
        # ── NEW: TICKER NEWS SEARCH ──
        self.ticker_search = QLineEdit()
        self.ticker_search.setPlaceholderText("Ticker...")
        self.ticker_search.setFixedWidth(70)
        self.ticker_search.setFixedHeight(26)
        self.ticker_search.returnPressed.connect(self._fetch_ticker)
        tb.addWidget(self.ticker_search)
        
        tick_btn = QPushButton("🔍 Fetch Stock News")
        tick_btn.setFixedHeight(26)
        tick_btn.clicked.connect(self._fetch_ticker)
        tb.addWidget(tick_btn)
        
        tick_ai_btn = QPushButton("🧠 AI Stock Analysis")
        tick_ai_btn.setFixedHeight(26)
        tick_ai_btn.setStyleSheet(f"""
            QPushButton {{ background:#1a0a2a; border:1px solid {C['purple']}; 
                           color:{C['purple']}; font-weight:bold; padding:0 12px; }}
            QPushButton:hover {{ background:{C['purple']}; color:white; }}
        """)
        tick_ai_btn.clicked.connect(self._analyze_ticker_news)
        tb.addWidget(tick_ai_btn)
        # ───────────────────────────────

        tb.addStretch()

        ref_btn = QPushButton("⟳ Global News")
        ref_btn.setFixedHeight(26)
        ref_btn.clicked.connect(self._refresh)
        tb.addWidget(ref_btn)

        score_btn = QPushButton("🎯 Auto-Score Feed")
        score_btn.setFixedHeight(26)
        score_btn.clicked.connect(self._run_scoring)
        score_btn.setStyleSheet(f"""
            QPushButton {{ background:#0a1a10; border:1px solid {C['green']}; 
                           color:{C['green']}; font-weight:bold; padding:0 12px; }}
            QPushButton:hover {{ background:{C['green']}; color:black; }}
        """)
        tb.addWidget(score_btn)

        sent_btn = QPushButton("◈ Bulk Sentiment")
        sent_btn.setFixedHeight(26)
        sent_btn.clicked.connect(self._sentiment)
        tb.addWidget(sent_btn)
        lay.addLayout(tb)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.list_w = QTableWidget(0, 3)
        self.list_w.setHorizontalHeaderLabels(["SOURCE", "HEADLINE", "DATE"])
        h = self.list_w.horizontalHeader()
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.list_w.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.list_w.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.list_w.verticalHeader().setVisible(False)
        self.list_w.setAlternatingRowColors(True)
        self.list_w.itemClicked.connect(self._show)
        splitter.addWidget(self.list_w)

        # Build Tab Widget for the right side
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {C['border']}; background: {C['bg2']}; }}
            QTabBar::tab {{ padding: 8px 16px; font-size: 11px; font-weight: bold; }}
        """)

        # Tab 1: Detailed Reading View
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.right_tabs.addTab(self.detail, "📰 Article")

        # Tab 2: AI Individual Item Scoring
        self.score_view = QTextEdit()
        self.score_view.setReadOnly(True)
        self.score_view.setPlaceholderText("Click '🎯 Auto-Score Feed' to classify headlines...")
        self.right_tabs.addTab(self.score_view, "🎯 AI Scoring")

        # Tab 3: Bulk Market Sentiment
        self.ai_out = QTextEdit()
        self.ai_out.setReadOnly(True)
        self.ai_out.setPlaceholderText("AI Analysis will appear here…")
        self.ai_out.setStyleSheet(f"background:{C['bg3']}; padding:8px;")
        self.right_tabs.addTab(self.ai_out, "🌐 Market Sentiment")

        splitter.addWidget(self.right_tabs)
        splitter.setSizes([620, 420])
        lay.addWidget(splitter)

        self.stat_lbl = QLabel("Loading news…")
        self.stat_lbl.setStyleSheet(f"color:{C['text3']}; font-size:10px;")
        lay.addWidget(self.stat_lbl)

    def _fetch_ticker(self):
        t = self.ticker_search.text().strip().upper()
        if not t: return
        self.current_ticker = t
        self.stat_lbl.setText(f"⟳ Fetching specific news for {t}…")
        self._nt = TickerNewsThread(t)
        self._nt.ready.connect(self.refresh)
        self._nt.start()

    def _analyze_ticker_news(self):
        """Dedicated AI analysis for a specific stock's news."""
        if not self.articles or not LM.connected:
            self.right_tabs.setCurrentWidget(self.ai_out)
            self.ai_out.setHtml(f"<p style='color:{C['red']}; padding:12px;'>⚠ Please fetch ticker news first and connect LM Studio.</p>")
            return
            
        ticker = self.current_ticker or "the selected stock"
        self.right_tabs.setCurrentWidget(self.ai_out)
        self.ai_out.setHtml(f"<p style='color:{C['text2']}; padding:12px;'>⟳ AI is analyzing the news impact on {ticker}...</p>")
        
        headlines = '\n'.join(f"  - {a['title']}" for a in self.articles[:15] if a.get('title'))
        
        prompt = f"""Act as a Tier-1 Hedge Fund News Analyst. 
Analyze the following recent headlines specifically for {ticker}.

[LATEST HEADLINES]
{headlines}

Provide a structured, professional assessment:
1. **CATALYST SUMMARY**: What is the primary news driving {ticker} right now?
2. **SENTIMENT SCORE**: Score the news from -10 (Extremely Bearish) to +10 (Extremely Bullish).
3. **FUNDAMENTAL IMPACT**: How does this news affect the company's earnings, margins, or growth trajectory?
4. **MARKET REACTION**: Is the market likely overreacting, underreacting, or correctly pricing this news?
5. **TRADE IMPLICATION**: One concise sentence on how to position (Buy/Sell/Hold) based *strictly* on these headlines.
"""
        self._st = stream_into(
            self.ai_out, [{"role": "user", "content": prompt}],
            system=f"You are a specialized equity news analyst covering {ticker}.",
            max_tokens=1000
        )
        BUS.status_msg.emit(f"Analyzing {ticker} news sentiment…")

    def _refresh(self):
        self.stat_lbl.setText("⟳ Fetching global news…")
        t = NewsThread()
        t.ready.connect(self.refresh)
        t.start()
        self._nt = t

    def _show(self, item):
        row = item.row()
        if row >= len(self.articles): return
        a = self.articles[row]
        self.right_tabs.setCurrentWidget(self.detail)
        self.detail.setHtml(f"""
        <div style='font-family:Consolas,monospace; padding:12px;'>
        <p style='color:{C["accent"]}; font-weight:bold; font-size:14px;'>{a.get("title","")}</p>
        <p style='color:{C["text3"]}; font-size:11px;'>{a.get("source","")} &nbsp;·&nbsp; {a.get("time","")}</p>
        <hr style='border:none; border-top:1px solid {C["border"]}; margin:8px 0;'>
        <p style='color:{C["text"]}; line-height:1.6; font-size:12px;'>{a.get("summary","")}</p>
        <p><a href='{a.get("link","")}' style='color:{C["blue"]};'>{a.get("link","")[:80]}</a></p>
        </div>
        """)

    def _run_scoring(self):
        """Feed top articles to the AI for individual sentiment scoring."""
        if not self.articles or not LM.connected:
            self.right_tabs.setCurrentWidget(self.score_view)
            self.score_view.setHtml(f"<p style='color:{C['red']}; padding:12px;'>⚠ Please load news and connect LM Studio.</p>")
            return

        self.right_tabs.setCurrentWidget(self.score_view)
        self.score_view.setHtml(f"<p style='color:{C['text2']}; padding:12px;'>⟳ AI is classifying and scoring the feed...</p>")

        feed_text = ""
        for i, a in enumerate(self.articles[:12]):
            feed_text += f"ARTICLE {i+1}:\nHEADLINE: {a.get('title', '')}\nSUMMARY: {a.get('summary', '')}\n\n"

        prompt = f"""You are a quantitative financial news scoring AI. Analyze the following articles.
For each article, extract the sentiment and assign a score from -10 (extremely bearish) to +10 (extremely bullish). 0 is neutral.

{feed_text}

Output EXACTLY this format for each article, separated by ---:
HEADLINE: <exact headline>
SENTIMENT: <BULLISH, BEARISH, or NEUTRAL>
SCORE: <number>
SUMMARY: <1-2 sentence concise explanation of why>
---
"""
        class ScoreThread(QThread):
            chunk = pyqtSignal(str)
            done_sig = pyqtSignal(str)
            def __init__(self, prompt):
                super().__init__()
                self.prompt = prompt
            def run(self):
                full = []
                try:
                    for tok in LM.stream([{"role": "user", "content": self.prompt}],
                        system="You are a strict financial analysis AI. Only output the exact requested format.",
                        temperature=0.1, max_tokens=2000
                    ):
                        full.append(tok)
                        self.chunk.emit(tok)
                except Exception as e:
                    full.append(f"\nError: {e}")
                self.done_sig.emit("".join(full))

        self._st_score = ScoreThread(prompt)
        self.score_view.clear()

        def on_chunk(tok):
            cur = self.score_view.textCursor()
            cur.movePosition(cur.MoveOperation.End if hasattr(cur, 'MoveOperation') else cur.End)
            cur.insertText(tok)
            self.score_view.setTextCursor(cur)
            self.score_view.ensureCursorVisible()

        def on_done(text):
            html = self._parse_and_format_scores(text)
            self.score_view.setHtml(html)
            BUS.status_msg.emit("News feed scoring complete")

        self._st_score.chunk.connect(on_chunk)
        self._st_score.done_sig.connect(on_done)
        self._st_score.start()
        BUS.status_msg.emit("Scoring news feed...")

    def _parse_and_format_scores(self, text):
        import re
        blocks = re.split(r'---+', text)
        if len(blocks) < 2:
            blocks = text.split('HEADLINE:')
            blocks =['HEADLINE:' + b for b in blocks if b.strip()]

        out_html = f"<div style='font-family:Consolas,monospace; padding:8px;'>"
        out_html += f"<h3 style='color:{C['accent']}; margin-top:0; margin-bottom:16px;'>🎯 AI Feed Scoring Results</h3>"

        for block in blocks:
            if not block.strip(): continue
            headline_match = re.search(r'HEADLINE:\s*(.+)', block, re.IGNORECASE)
            sentiment_match = re.search(r'SENTIMENT:\s*(BULLISH|BEARISH|NEUTRAL)', block, re.IGNORECASE)
            score_match = re.search(r'SCORE:\s*([+-]?\d+)', block, re.IGNORECASE)
            summary_match = re.search(r'SUMMARY:\s*(.+)', block, re.IGNORECASE | re.DOTALL)

            if not headline_match: continue
            headline = headline_match.group(1).strip()
            sentiment = sentiment_match.group(1).upper() if sentiment_match else "NEUTRAL"
            score = score_match.group(1).strip() if score_match else "0"
            summary = summary_match.group(1).strip() if summary_match else "No summary provided."
            summary = summary.split('HEADLINE:')[0].strip().replace('\n', ' ')

            if "BULLISH" in sentiment: color, bg = C['green'], "#0a1a10"
            elif "BEARISH" in sentiment: color, bg = C['red'], "#1a0a0a"
            else: color, bg = C['text2'], C['bg3']

            out_html += f"""
            <div style='background:{bg}; border:1px solid {color}; border-radius:6px; padding:12px; margin-bottom:14px;'>
                <p style='margin:0; font-size:14px; font-weight:bold; color:{C['text']};'>{headline}</p>
                <p style='margin:8px 0 10px 0; font-size:12px;'>
                    <span style='background:{color}; color:#000; padding:3px 8px; border-radius:4px; font-weight:bold;'>{sentiment}</span>
                    <span style='color:{color}; font-weight:bold; margin-left:12px; font-size:13px;'>Score: {score}/10</span>
                </p>
                <p style='margin:0; color:{C['text']}; font-size:12px; line-height:1.5;'>{summary}</p>
            </div>"""
        out_html += "</div>"
        return out_html

    def _sentiment(self):
        """Bulk market overview generation."""
        if not self.articles or not LM.connected:
            return
        self.right_tabs.setCurrentWidget(self.ai_out)
        headlines = '\n'.join(f"  - {a['title']}" for a in self.articles[:20] if a.get('title'))
        self.ai_out.clear()
        self._st = stream_into(
            self.ai_out,[{"role": "user", "content":
              f"Analyze market sentiment from these headlines:\n{headlines}\n\n"
              "Provide: 1) Sentiment score (-10 very bearish → +10 very bullish), "
              "2) Dominant themes, 3) Sector-specific signals, 4) Key risk/opportunity, "
              "5) One-line portfolio implication."}],
            system="You are a market intelligence analyst.",
            max_tokens=800,
        )
        BUS.status_msg.emit("Analyzing bulk news sentiment…")


# ═══════════════════════════════════════════════════════════════════════════════
# SCREENER PANEL
# ═══════════════════════════════════════════════════════════════════════════════
class ScreenerPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._st = None
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        hdr = QLabel("◈ AI-POWERED STOCK SCREENER")
        hdr.setStyleSheet(
            f"color:{C['accent']}; font-size:15px; font-weight:bold; letter-spacing:1.5px;"
        )
        lay.addWidget(hdr)

        # Preset buttons
        pre_row = QHBoxLayout()
        pre_row.addWidget(QLabel("Presets:"))
        presets = [
            ("⚡ Value",      "value picks: P/E < 15, FCF positive, dividend, stable"),
            ("🚀 Growth",     "high-growth tech: revenue >25%, expanding margins, large TAM"),
            ("💎 Quality",    "QARP: ROE >20%, consistent earnings growth, reasonable valuation"),
            ("📈 Momentum",   "price momentum: near 52w high, strong RS, earnings beats"),
            ("💰 Dividends",  "dividend aristocrats: 3%+ yield, 10+ years growth, low payout"),
            ("🔍 Deep Value", "contrarian: near 52w low, P/B <1, asset-rich, hidden catalyst"),
            ("🤖 AI Plays",   "pure-play AI/ML companies with strong revenue growth and moats"),
        ]
        for label, desc in presets:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet(f"""
                QPushButton {{ background:{C['bg3']}; border:1px solid {C['border']};
                               color:{C['text2']}; font-size:10px; padding:0 8px; }}
                QPushButton:hover {{ background:{C['accent']}; color:white; border-color:{C['accent']}; }}
            """)
            btn.clicked.connect(lambda _, d=desc: self._screen(d))
            pre_row.addWidget(btn)
        pre_row.addStretch()
        lay.addLayout(pre_row)

        # Custom query
        q_row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText(
            "Natural language screen: e.g. 'AI companies with 30%+ revenue growth, P/S < 10, profitable'"
        )
        self.query.setFixedHeight(34)
        self.query.returnPressed.connect(lambda: self._screen(self.query.text()))
        q_row.addWidget(self.query)

        run = QPushButton("◈  SCREEN  ▶")
        run.setFixedHeight(34)
        run.setStyleSheet(f"""
            QPushButton {{ background:{C['accent']}; border:none; color:white;
                           font-weight:bold; font-size:12px; padding:0 16px; }}
            QPushButton:hover {{ background:{C['accent2']}; }}
        """)
        run.clicked.connect(lambda: self._screen(self.query.text()))
        q_row.addWidget(run)
        lay.addLayout(q_row)

        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setStyleSheet(
            f"QTextEdit {{ background:{C['bg2']}; border:1px solid {C['border']}; "
            f"padding:16px; font-size:12px; }}"
        )
        lay.addWidget(self.results)

        self._intro()

    def _intro(self):
        self.results.setHtml(f"""
        <div style='font-family:Consolas,monospace; color:{C["text2"]}; padding:24px;'>
        <p style='color:{C["accent"]}; font-size:16px; font-weight:bold;'>◈ APEX AI SCREENER</p>
        <br>
        <p>Use <b>natural language</b> or click a preset to screen for stocks.</p>
        <br>
        <p style='color:{C["accent2"]};'>Examples:</p>
        <p>● "Cloud software companies with >20% FCF margins and &lt;30x EV/FCF"</p>
        <p>● "Semiconductor stocks with strong AI exposure and net cash balance sheets"</p>
        <p>● "Mid-cap biotech with Phase 3 catalysts in 2025, &lt;$5B market cap"</p>
        <p>● "Beaten-down retailers with improving margins and insider buying"</p>
        <br>
        <p style='color:{C["text3"]}; font-size:10px;'>
        The AI analyst will identify specific stocks, provide rationale, and flag key risks.</p>
        </div>
        """)

    def _screen(self, desc):
        if not desc.strip():
            return
        if not LM.connected:
            self.results.setPlainText("⚠ LM Studio not connected.")
            return
        prompt = f"""Act as a senior buy-side equity analyst. Screen for stocks matching: "{desc}"

Provide a professional screen with:

[SCREENING CRITERIA]
Translate the request into specific quantitative and qualitative filters.

[TOP 10 STOCK PICKS]
For each stock provide:
● Company name & ticker
● Market cap & sector
● Why it matches the screen (specific metrics)
● Key investment thesis (2 sentences)
● Main risk
● Conviction: HIGH / MEDIUM / LOW

[SECTOR & THEME OVERVIEW]
Current market dynamics relevant to this screen.

[PORTFOLIO CONSTRUCTION NOTES]
How to size and diversify a basket of these names.

Be specific with real tickers and numbers. No generic commentary."""
        self.results.clear()
        self._st = stream_into(
            self.results,
            [{"role": "user", "content": prompt}],
            system="You are a senior buy-side equity analyst with deep market knowledge. Be specific with tickers.",
            temperature=0.4,
            max_tokens=2800,
            on_done=lambda: BUS.status_msg.emit("Screen complete"),
        )
        BUS.status_msg.emit("Running AI stock screen…")


# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS PANEL
# ═══════════════════════════════════════════════════════════════════════════════
class SettingsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
        BUS.lm_status.connect(self._lm_update)

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        hdr = QLabel(f"◈  {APP}  ·  SETTINGS")
        hdr.setStyleSheet(
            f"color:{C['accent']}; font-size:18px; font-weight:bold; letter-spacing:2px;"
        )
        lay.addWidget(hdr)

        # LM Studio
        lm_grp = QGroupBox("LM STUDIO CONNECTION")
        lm_lay = QFormLayout(lm_grp)
        lm_lay.setSpacing(10)
        lm_lay.setContentsMargins(12, 16, 12, 12)

        self.url_inp = QLineEdit(LM_URL)
        lm_lay.addRow("Server URL:", self.url_inp)

        self.model_lbl = QLabel("─")
        self.model_lbl.setStyleSheet(f"color:{C['accent2']};")
        lm_lay.addRow("Active Model:", self.model_lbl)

        self.conn_lbl = QLabel("Checking…")
        self.conn_lbl.setStyleSheet(f"color:{C['text2']};")
        lm_lay.addRow("Status:", self.conn_lbl)

        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self._test)
        lm_lay.addRow("", test_btn)
        lay.addWidget(lm_grp)

        # Dependencies
        dep_grp = QGroupBox("DEPENDENCY STATUS")
        dep_lay = QVBoxLayout(dep_grp)
        dep_lay.setContentsMargins(12, 16, 12, 12)
        # We loop through all dependencies including xgboost and statsmodels
        for name, ok in[("yfinance", HAS_YF), ("pdfplumber", HAS_PDF), ("feedparser", HAS_FEED), ("xgboost", HAS_XGB), ("statsmodels", HAS_SM)]:
            icon = "✓" if ok else "✗"
            clr  = C['green'] if ok else C['red']
            dep_lay.addWidget(QLabel(f'<span style="color:{clr};">{icon}</span>  {name}'))
            
        if not (HAS_YF and HAS_PDF and HAS_FEED and HAS_XGB and HAS_SM):
            dep_lay.addWidget(QLabel(
                f'<span style="color:{C["text2"]}; font-size:10px;">'
                f'pip install yfinance pdfplumber feedparser xgboost scikit-learn statsmodels</span>'
            ))
        lay.addWidget(dep_grp)
        # 👇 Add XGBoost to this list 👇
        for name, ok in[("yfinance", HAS_YF), ("pdfplumber", HAS_PDF), ("feedparser", HAS_FEED), ("xgboost & sklearn", HAS_XGB)]:
            icon = "✓" if ok else "✗"
            clr  = C['green'] if ok else C['red']
            dep_lay.addWidget(QLabel(f'<span style="color:{clr};">{icon}</span>  {name}'))
        if not (HAS_YF and HAS_PDF and HAS_FEED):
            dep_lay.addWidget(QLabel(
                f'<span style="color:{C["text2"]}; font-size:10px;">'
                f'pip install yfinance pdfplumber feedparser</span>'
            ))
        lay.addWidget(dep_grp)

        # About
        ab_grp = QGroupBox("ABOUT")
        ab_lay = QVBoxLayout(ab_grp)
        ab_lay.setContentsMargins(12, 16, 12, 12)
        ab_lay.setSpacing(4)
        for line in [
            f"<b style='color:{C['accent']};'>{APP} {VER}</b>",
            "Bloomberg-Style AI Financial Analysis Platform",
            f"<span style='color:{C['text2']};'>UI: PyQt5 &nbsp;·&nbsp; Charts: Matplotlib &nbsp;·&nbsp; Data: yfinance &nbsp;·&nbsp; AI: LM Studio</span>",
        ]:
            lbl = QLabel(line)
            ab_lay.addWidget(lbl)
        lay.addWidget(ab_grp)

        lay.addStretch()

        save_btn = QPushButton("◈  APPLY SETTINGS")
        save_btn.setFixedHeight(36)
        save_btn.setStyleSheet(f"""
            QPushButton {{ background:{C['accent']}; border:none; color:white;
                           font-weight:bold; font-size:12px; letter-spacing:0.5px; }}
            QPushButton:hover {{ background:{C['accent2']}; }}
        """)
        save_btn.clicked.connect(self._save)
        lay.addWidget(save_btn)

    def _test(self):
        LM.base = self.url_inp.text().strip()
        ok = LM.check()
        if ok:
            self.conn_lbl.setText("✓ Connected")
            self.conn_lbl.setStyleSheet(f"color:{C['green']};")
            self.model_lbl.setText(LM.model)
        else:
            self.conn_lbl.setText("✗ Failed — is LM Studio running?")
            self.conn_lbl.setStyleSheet(f"color:{C['red']};")

    def _lm_update(self, ok, model):
        if ok:
            self.conn_lbl.setText("✓ Connected")
            self.conn_lbl.setStyleSheet(f"color:{C['green']};")
            self.model_lbl.setText(model)
        else:
            self.conn_lbl.setText("✗ Offline")
            self.conn_lbl.setStyleSheet(f"color:{C['red']};")

    def _save(self):
        BUS.status_msg.emit("Settings applied")


# ═══════════════════════════════════════════════════════════════════════════════
# SCROLLING NEWS TICKER
# ═══════════════════════════════════════════════════════════════════════════════
class NewsTicker(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(26)
        self.setStyleSheet(
            f"background:{C['bg3']}; border-top:1px solid {C['border']};"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        badge = QLabel("  LIVE  ")
        badge.setStyleSheet(
            f"background:{C['accent']}; color:white; font-weight:bold;"
            f"font-size:10px; padding:0 8px; letter-spacing:0.5px;"
        )
        badge.setFixedHeight(26)
        lay.addWidget(badge)

        self.scroll = QLabel("  Loading market news…")
        self.scroll.setStyleSheet(f"color:{C['text2']}; font-size:10px; padding-left:8px;")
        lay.addWidget(self.scroll)

        self._headlines = []
        self._idx = 0
        t = QTimer(self)
        t.timeout.connect(self._next)
        t.start(6000)
        BUS.news_ready.connect(self._on_news)

    def _on_news(self, articles):
        self._headlines = [a.get('title', '') for a in articles if a.get('title')]
        if not self._headlines:
            self._headlines = ["No news available"]
        self._next()

    def _next(self):
        if not self._headlines:
            return
        self._idx = (self._idx + 1) % len(self._headlines)
        self.scroll.setText(f"  {self._headlines[self._idx]}  ")
        
# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS AI TRADER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class AIBotThread(QThread):
    log_msg = pyqtSignal(str)
    update_dash = pyqtSignal(float, int, float, float) # cash, shares, price, equity
    
    def __init__(self, ticker, capital, timeframe, scan_interval_str, use_vision=False):
        super().__init__()
        self.ticker = ticker.upper()
        self.capital = capital
        self.cash = capital
        self.shares = 0
        self.timeframe = timeframe
        self.use_vision = use_vision # 👁️ NEW: Vision Toggle
        self.running = True
        self.last_feedback = "System just booted. No prior actions."
        
        if 'm' in scan_interval_str:
            self.sleep_sec = int(scan_interval_str.replace('m', '')) * 60
        else:
            self.sleep_sec = int(scan_interval_str.replace('s', ''))
        
        self.tf_map = {
            "1m": ("5d", "1m"), "5m": ("5d", "5m"),
            "15m": ("1mo", "15m"), "1h": ("1mo", "1h"), "1d": ("1y", "1d")
        }
        
    def stop(self):
        self.running = False
        
    def run(self):
        self.log_msg.emit(f"🟢 <b style='color:{C['green']}'>SYSTEM BOOT:</b> Autonomous Trading initialized for {self.ticker}.")
        self.log_msg.emit(f"Interval: {self.timeframe} | Scan Rate: Every {self.sleep_sec}s | Capital: ${self.capital:,.2f}")
        
        period, interval = self.tf_map.get(self.timeframe, ("1mo", "1h"))
        
        while self.running:
            try:
                # 1. Get Live Price
                price, _ = DE.fast_quote(self.ticker)
                if not price:
                    self.log_msg.emit("⚠ Failed to fetch live price. Retrying...")
                    self._sleep(15)
                    continue
                    
                equity = self.cash + (self.shares * price)
                self.update_dash.emit(self.cash, self.shares, price, equity)
                
                if self.cash >= price:
                    min_pct = (price / self.cash) * 100
                    min_pct_str = f"{min_pct:.1f}%"
                else:
                    min_pct = 101.0
                    min_pct_str = "INSUFFICIENT TOTAL CASH TO BUY 1 SHARE"
                
                # 2. Gather Institutional Intelligence
                self.log_msg.emit(f"⟳ [Cycle] Fetching live {interval} data, XGBoost, Kalman, and Chaos Math for {self.ticker}...")
                
                df = DE.fetch(self.ticker, period, interval) 
                if df is not None: df = DE.add_indicators(df)
                
                xgb_res = XGB_ENGINE.predict(df) if df is not None else {}
                hmm_res = HMM_DETECTOR.fit_predict(df) if df is not None else {}
                mp = DE.get_max_pain(self.ticker)
                
                hurst = 0.5
                order_flow_imbalance = 0.0
                if df is not None and len(df) > 30:
                    closes = df['Close'].values
                    try:
                        lags = range(2, 20)
                        tau =[np.sqrt(np.std(np.subtract(closes[lag:], closes[:-lag]))) for lag in lags]
                        poly = np.polyfit(np.log(lags), np.log(tau), 1)
                        hurst = poly[0] * 2.0
                    except: pass
                    try:
                        rng = (df['High'] - df['Low']) + 1e-9
                        cvd = df['Volume'] * ((df['Close'] - df['Open']) / rng)
                        order_flow_imbalance = cvd.tail(10).sum()
                    except: pass
                
                last = df.iloc[-1] if df is not None and not df.empty else {}
                def fv(val): return f"{float(val):.2f}" if pd.notna(val) else "N/A"
                
                ptt_status = "No active PTT trade."
                if df is not None and 'ptt_trades' in df.attrs and len(df.attrs['ptt_trades']) > 0:
                    last_trade = df.attrs['ptt_trades'][-1]
                    if last_trade['exit_idx'] == len(df) - 1:
                        tdir = "LONG" if last_trade['dir'] == 1 else "SHORT"
                        ptt_status = f"ACTIVE {tdir} | Entry: ${last_trade['entry_p']:.2f} | SL: ${last_trade['sl_p']:.2f}"

                # 👇 ENHANCED REAL-TIME NEWS FETCHER 👇
                self.log_msg.emit(f"📰 Scanning recent news and summaries for {self.ticker}...")
                news_str = "No recent news."
                news_list =[]
                try:
                    if HAS_FEED:
                        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.ticker}&region=US&lang=en-US"
                        feed = feedparser.parse(url)
                        for e in feed.entries[:3]: # Top 3 most recent articles
                            title = e.get('title', '').strip()
                            summary = e.get('summary', '').strip()
                            # Clean any messy HTML tags out of the summary
                            summary = re.sub('<[^<]+>', '', summary)[:250] 
                            news_list.append(f"Headline: {title}\nSummary: {summary}...")
                            
                    # Fallback to standard yfinance if RSS fails
                    if not news_list and HAS_YF:
                        tk = yf.Ticker(self.ticker)
                        for n in tk.news[:3]:
                            news_list.append(f"Headline: {n.get('title', '')}")
                            
                    if news_list:
                        news_str = "\n\n".join(news_list)
                except Exception:
                    pass
                # 👆 END NEWS FETCHER 👆

                # ── HEADLESS VISION RENDERING (Only if enabled!) ──
                bot_img_b64 = ""
                if self.use_vision:
                    self.log_msg.emit("📸 Generating invisible chart for Vision AI analysis...")
                    try:
                        from matplotlib.figure import Figure
                        from matplotlib.backends.backend_agg import FigureCanvasAgg
                        import io, base64
                        
                        fig = Figure(figsize=(6, 4), facecolor=C['bg'])
                        canvas = FigureCanvasAgg(fig)
                        ax = fig.add_subplot(111)
                        ax.set_facecolor(C['bg'])
                        
                        closes_arr, opens_arr = df['Close'].values, df['Open'].values
                        highs_arr, lows_arr = df['High'].values, df['Low'].values
                        for idx_bar in range(len(df)):
                            up = closes_arr[idx_bar] >= opens_arr[idx_bar]
                            clr = C['cand_up'] if up else C['cand_dn']
                            ax.bar(idx_bar, abs(closes_arr[idx_bar] - opens_arr[idx_bar]), 0.6, bottom=min(opens_arr[idx_bar], closes_arr[idx_bar]), color=clr)
                            ax.plot([idx_bar, idx_bar], [lows_arr[idx_bar], highs_arr[idx_bar]], color=clr, linewidth=1)
                            
                        if 'MA20' in df: ax.plot(df['MA20'].values, color=C['ma20'], linewidth=1)
                        if 'MA50' in df: ax.plot(df['MA50'].values, color=C['ma50'], linewidth=1)
                        
                        ax.axis('off')
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=72)
                        buf.seek(0)
                        bot_img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    except Exception as e:
                        self.log_msg.emit(f"⚠ Vision render failed: {e}")

                # 3. Build the Strict Autonomous Prompt
                prompt_text = f"""You are an autonomous quantitative trading bot managing a virtual portfolio.

[LIVE PORTFOLIO STATE]
Asset: {self.ticker}
Trading Interval: {interval}
Current Price: ${price:.2f}
Available Cash: ${self.cash:,.2f}
Current Position: {self.shares} shares
Minimum Allocation % needed to buy 1 share: {min_pct_str}

[LAST CYCLE FEEDBACK]
{self.last_feedback}

[LATEST REAL-TIME NEWS (CRITICAL)]
{news_str}

[EXTREME QUANTS]
Hurst Exponent: {hurst:.3f} (<0.45=Mean-Reverting, >0.55=Trending).
Order Flow Imbalance (10-bar): {order_flow_imbalance:,.0f} (Positive=Accumulation, Negative=Distribution)

[TECHNICALS]
RSI: {fv(last.get('RSI'))} | MACD: {fv(last.get('MACD'))}
VWAP: ${fv(last.get('VWAP'))} | PTT State: {ptt_status}

[MODELS]
XGBoost: {xgb_res.get('prob_up', 0)*100:.1f}% UP probability.
Kalman True State: ${fv(last.get('Kalman'))}
HMM Regime: {hmm_res.get('regime', 'UNKNOWN')}

[INSTRUCTIONS]
{"Analyze the image, market data, and LATEST NEWS to make a decision." if self.use_vision else "Analyze the market data and LATEST NEWS to make a decision."}
If "BUY", specify 'allocation_pct' (1-100) of Available Cash.
If "SELL", specify 'allocation_pct' (1-100) of Current Position shares.
If "HOLD", set allocation_pct to 0.

CRITICAL RULE: Respond ONLY with a raw JSON object. Do not include markdown formatting like ```json. 
The 'reason' field MUST be a single string. DO NOT use nested objects or dictionaries for the reason.
Format exactly like this:
{{"action": "BUY", "allocation_pct": 25, "reason": "Hurst is 0.62, XGBoost shows 75% up-prob, and recent news regarding earnings is highly bullish."}}
"""
                if self.use_vision and bot_img_b64:
                    self.log_msg.emit("🧠 Feeding Quants, Chart Image, and News Summaries to Vision AI...")
                    messages =[{"role": "user", "content":[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bot_img_b64}"}}
                    ]}]
                else:
                    self.log_msg.emit("🧠 Feeding Quants and News Summaries to Text AI...")
                    # Safe text-only payload for models like Llama-3
                    messages =[{"role": "user", "content": prompt_text}]
                
                # Run the LLM synchronously (Bumped to 800 tokens just in case)
                response = ''.join(LM.stream(messages, system="You are an autonomous AI trading server. Output valid JSON only. No nested objects.", temperature=0.1, max_tokens=800))
                
                # 👇 FIX: Aggressively scrub markdown backticks before parsing
                response_clean = response.replace("```json", "").replace("```", "").strip()
                
                # 4. Parse the LLM's JSON Response
                match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        action = data.get('action', 'HOLD').upper()
                        reason = data.get('reason', 'No reason provided.')
                        
                        raw_alloc = str(data.get('allocation_pct', 100)).replace('%', '')
                        alloc_pct = max(0.0, min(float(raw_alloc), 100.0)) 
                        
                        if action == 'BUY':
                            spend_amount = self.cash * (alloc_pct / 100.0)
                            qty = int(spend_amount // price)
                            
                            if qty > 0:
                                cost = qty * price
                                self.cash -= cost
                                self.shares += qty
                                self.log_msg.emit(f"📈 <b style='color:{C['green']}'>EXECUTED BUY:</b> {qty} shares @ ${price:.2f} (Using {alloc_pct:.0f}% of cash)")
                                self.log_msg.emit(f"<i>AI Rationale: {reason}</i>")
                                self.last_feedback = f"SUCCESS: Bought {qty} shares."
                            else:
                                self.log_msg.emit(f"⚖ <b style='color:{C['accent']}'>AI DECISION: BUY {alloc_pct}%</b> (Insufficient cash)")
                                self.log_msg.emit(f"<i>AI Rationale: {reason}</i>")
                                self.last_feedback = f"FAILED BUY: You allocated {alloc_pct}%, but need {min_pct_str} to buy a share."
                                
                        elif action == 'SELL':
                            sell_qty = int(self.shares * (alloc_pct / 100.0))
                            
                            if sell_qty > 0:
                                revenue = sell_qty * price
                                self.cash += revenue
                                self.shares -= sell_qty
                                self.log_msg.emit(f"📉 <b style='color:{C['red']}'>EXECUTED SELL:</b> {sell_qty} shares @ ${price:.2f} (Sold {alloc_pct:.0f}%)")
                                self.log_msg.emit(f"<i>AI Rationale: {reason}</i>")
                                self.last_feedback = f"SUCCESS: Sold {sell_qty} shares."
                            else:
                                self.log_msg.emit(f"⚖ <b style='color:{C['accent']}'>AI DECISION: SELL {alloc_pct}%</b> (No shares held)")
                                self.log_msg.emit(f"<i>AI Rationale: {reason}</i>")
                                self.last_feedback = f"FAILED SELL: Allocation {alloc_pct}% resulted in 0 shares."
                                
                        else: # HOLD
                            self.log_msg.emit(f"🛡 <b style='color:{C['text2']}'>AI DECISION: HOLD</b>")
                            self.log_msg.emit(f"<i>AI Rationale: {reason}</i>")
                            self.last_feedback = "SUCCESS: Held position."
                            
                    except json.JSONDecodeError:
                        self.log_msg.emit(f"⚠ <b style='color:{C['red']}'>JSON Error:</b> AI output malformed.")
                        self.log_msg.emit(f"<span style='color:{C['text2']}'>Raw Output: {response}</span>") # 👁️ SHOW RAW TEXT
                        self.last_feedback = "FAILED: Invalid JSON format."
                else:
                    self.log_msg.emit(f"⚠ <b style='color:{C['red']}'>Parse Error:</b> No JSON found in response.")
                    self.log_msg.emit(f"<span style='color:{C['text2']}'>Raw Output: {response}</span>") # 👁️ SHOW RAW TEXT
                    self.last_feedback = "FAILED: Could not find JSON braces in your output."
                    
                equity = self.cash + (self.shares * price)
                self.update_dash.emit(self.cash, self.shares, price, equity)
                
            except Exception as e:
                self.log_msg.emit(f"⚠ <b style='color:{C['red']}'>System Error:</b> {str(e)}")
                
            self.log_msg.emit(f"<span style='color:{C['text3']}'>[Sleeping for {self.sleep_sec}s until next sweep...]</span><br>")
            self._sleep(self.sleep_sec)
            
    def _sleep(self, seconds):
        for _ in range(seconds * 2):
            if not self.running: break
            time.sleep(0.5)

class AITraderPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.bot_thread = None
        self.initial_capital = 0.0
        self._setup_ui()
        BUS.ticker_changed.connect(self._on_global_ticker)

    def _on_global_ticker(self, t):
        if self.start_btn.isEnabled(): 
            self.tick_inp.setText(t)

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        hdr = QLabel("🤖  AUTONOMOUS AI TRADER")
        hdr.setStyleSheet(f"color:{C['accent']}; font-size:16px; font-weight:bold; letter-spacing:1.5px;")
        lay.addWidget(hdr)

        # ── Control Bar ──
        ctrl_lay = QHBoxLayout()
        
        ctrl_lay.addWidget(QLabel("TICKER:"))
        self.tick_inp = QLineEdit("NVDA")
        self.tick_inp.setFixedWidth(55)
        ctrl_lay.addWidget(self.tick_inp)
        
        ctrl_lay.addSpacing(10)
        
        ctrl_lay.addWidget(QLabel("TF:"))
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["1m", "5m", "15m", "1h", "1d"])
        self.tf_combo.setCurrentText("1h")
        self.tf_combo.setFixedWidth(50)
        ctrl_lay.addWidget(self.tf_combo)
        
        ctrl_lay.addSpacing(10)
        
        ctrl_lay.addWidget(QLabel("SCAN:"))
        self.scan_combo = QComboBox()
        self.scan_combo.addItems(["10s", "30s", "60s", "2m", "5m"])
        self.scan_combo.setCurrentText("60s")
        self.scan_combo.setFixedWidth(50)
        ctrl_lay.addWidget(self.scan_combo)
        
        ctrl_lay.addSpacing(10)

        # 👇 NEW: Vision Toggle Checkbox
        self.vision_cb = QCheckBox("USE VISION AI")
        self.vision_cb.setToolTip("Check this ONLY if you have a Vision model (like LLaVA) loaded in LM Studio.")
        self.vision_cb.setStyleSheet(f"color:{C['accent2']}; font-size:10px; font-weight:bold;")
        ctrl_lay.addWidget(self.vision_cb)
        
        ctrl_lay.addSpacing(10)
        
        ctrl_lay.addWidget(QLabel("BANKROLL ($):"))
        self.cap_inp = QLineEdit("10000")
        self.cap_inp.setFixedWidth(70)
        ctrl_lay.addWidget(self.cap_inp)
        
        ctrl_lay.addStretch()

        self.start_btn = QPushButton("▶  START BOT")
        self.start_btn.setStyleSheet(f"QPushButton {{ background:#004d00; color:#00ff00; border:1px solid #00ff00; font-weight:bold; padding:6px 16px; }}")
        self.start_btn.clicked.connect(self._start_bot)
        ctrl_lay.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  STOP")
        self.stop_btn.setStyleSheet(f"QPushButton {{ background:#4d0000; color:#ff0000; border:1px solid #ff0000; font-weight:bold; padding:6px 16px; }}")
        self.stop_btn.clicked.connect(self._stop_bot)
        self.stop_btn.setEnabled(False)
        ctrl_lay.addWidget(self.stop_btn)

        lay.addLayout(ctrl_lay)

        # ── Dashboard ──
        dash_frame = QFrame()
        dash_frame.setStyleSheet(f"background:{C['bg2']}; border:1px solid {C['border']};")
        dash_lay = QHBoxLayout(dash_frame)
        dash_lay.setContentsMargins(20, 20, 20, 20)
        
        self.val_cash = self._dash_metric(dash_lay, "AVAILABLE CASH", "$0.00")
        self.val_pos  = self._dash_metric(dash_lay, "OPEN POSITION", "0 Shares")
        self.val_px   = self._dash_metric(dash_lay, "LIVE PRICE", "$0.00")
        self.val_eq   = self._dash_metric(dash_lay, "TOTAL EQUITY", "$0.00", C['accent'])
        self.val_ret  = self._dash_metric(dash_lay, "NET RETURN", "0.00%", C['text'])
        
        lay.addWidget(dash_frame)

        # ── Terminal Logs ──
        lay.addWidget(QLabel("TERMINAL LOGS:"))
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setStyleSheet(f"background:{C['bg']}; border:1px solid {C['border']}; font-family:Consolas,monospace; font-size:12px; padding:10px;")
        lay.addWidget(self.logs)

    def _dash_metric(self, parent_lay, title, default_val, color=C['text']):
        v_lay = QVBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color:{C['text3']}; font-size:10px; font-weight:bold; letter-spacing:1px;")
        v_lay.addWidget(lbl)
        
        val = QLabel(default_val)
        val.setStyleSheet(f"color:{color}; font-size:18px; font-weight:bold;")
        v_lay.addWidget(val)
        parent_lay.addLayout(v_lay)
        return val

    def _start_bot(self):
        if not LM.connected:
            self.logs.append(f"<span style='color:{C['red']}'>⚠ Error: LM Studio is not connected. Start a model first.</span>")
            return
            
        ticker = self.tick_inp.text().strip().upper()
        timeframe = self.tf_combo.currentText()
        scan_interval = self.scan_combo.currentText()
        use_vision = self.vision_cb.isChecked() # Get the checkbox state
        
        try:
            self.initial_capital = float(self.cap_inp.text().replace(',', ''))
        except:
            self.logs.append(f"<span style='color:{C['red']}'>⚠ Error: Invalid capital amount.</span>")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tick_inp.setEnabled(False)
        self.tf_combo.setEnabled(False)
        self.scan_combo.setEnabled(False)
        self.vision_cb.setEnabled(False)
        self.cap_inp.setEnabled(False)
        
        self.logs.clear()
        
        self.bot_thread = AIBotThread(ticker, self.initial_capital, timeframe, scan_interval, use_vision)
        self.bot_thread.log_msg.connect(self._append_log)
        self.bot_thread.update_dash.connect(self._update_dash)
        self.bot_thread.start()

    def _stop_bot(self):
        if self.bot_thread:
            self.bot_thread.stop()
            self.bot_thread.wait()
            self._append_log(f"<span style='color:{C['red']}'>🛑 SYSTEM OFFLINE: Autonomous Trading Halted.</span>")
            
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.tick_inp.setEnabled(True)
        self.tf_combo.setEnabled(True)
        self.scan_combo.setEnabled(True)
        self.vision_cb.setEnabled(True)
        self.cap_inp.setEnabled(True)

    def _append_log(self, msg):
        self.logs.append(msg)
        self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    def _update_dash(self, cash, shares, price, equity):
        self.val_cash.setText(f"${cash:,.2f}")
        self.val_pos.setText(f"{shares} Shares")
        self.val_px.setText(f"${price:,.2f}")
        self.val_eq.setText(f"${equity:,.2f}")
        
        ret_pct = ((equity / self.initial_capital) - 1) * 100
        color = C['green'] if ret_pct >= 0 else C['red']
        sign = "+" if ret_pct >= 0 else ""
        self.val_ret.setText(f"{sign}{ret_pct:.2f}%")
        self.val_ret.setStyleSheet(f"color:{color}; font-size:18px; font-weight:bold;")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP}  {VER}")
        self.setMinimumSize(1380, 820)
        self.resize(1680, 980)
        self.setStyleSheet(build_qss())
        self._setup_ui()
        self._start_services()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────
        self.header = HeaderBar()
        root.addWidget(self.header)

        # ── Body ──────────────────────────────────────────────────────
        body = QSplitter(Qt.Orientation.Horizontal)
        body.setHandleWidth(2)

        # Left: watchlist
        self.watchlist = WatchlistPanel()
        body.addWidget(self.watchlist)

        # Centre: tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.chart    = TechnicalChart()
        self.analysis = AnalysisPanel()
        self.docs     = DocumentPanel()
        self.prob     = ProbabilityPanel()
        self.news_pnl = NewsPanel()
        self.screener = ScreenerPanel()
        self.settings = SettingsPanel()
        self.ai_trader = AITraderPanel()

        self.tabs.addTab(self.chart,    "◈  CHART")
        self.tabs.addTab(self.analysis, "⊞  ANALYSIS")
        self.tabs.addTab(self.docs,     "📄  DOCUMENTS")
        self.tabs.addTab(self.prob,     "⬡  PROBABILITY")
        self.tabs.addTab(self.news_pnl, "📰  NEWS")
        self.tabs.addTab(self.screener, "🔍  SCREENER")
        # 👇 2. ADD THE TAB HERE 👇
        self.tabs.addTab(self.ai_trader,"🤖  AI TRADER")
        self.tabs.addTab(self.settings, "⚙  SETTINGS")

        body.addWidget(self.tabs)


        # Right: AI chat
        self.ai_chat = AIChatPanel()
        self.ai_chat._chart_ref = self.chart
        body.addWidget(self.ai_chat)

        body.setSizes([200, 1000, 360])
        root.addWidget(body)

        # ── News ticker ───────────────────────────────────────────────
        self.ticker_bar = NewsTicker()
        root.addWidget(self.ticker_bar)

        # ── Status bar ────────────────────────────────────────────────
        sb = self.statusBar()
        sb.showMessage(f"◈  {APP}  {VER}  —  Ready")
        BUS.status_msg.connect(sb.showMessage)

        self._lm_sb = QLabel("LM Studio: checking…")
        self._lm_sb.setStyleSheet(f"color:{C['text3']}; padding:0 12px;")
        sb.addPermanentWidget(self._lm_sb)
        BUS.lm_status.connect(self._lm_sb_update)

    def _lm_sb_update(self, ok, model):
        name = model.split('/')[-1][:28] if '/' in model else model[:28]
        if ok:
            self._lm_sb.setText(f"● LM  {name}")
            self._lm_sb.setStyleSheet(f"color:{C['green']}; padding:0 12px;")
        else:
            self._lm_sb.setText("● LM Studio: OFFLINE")
            self._lm_sb.setStyleSheet(f"color:{C['red']}; padding:0 12px;")

    def _start_services(self):
        # LM Studio heartbeat
        self._ping = LMPingThread()
        self._ping.start()

        # Initial news fetch — feeds result to both news panel and ticker
        nt = NewsThread()
        nt.ready.connect(self._on_news)
        nt.start()
        self._news_thread = nt

        # Load default chart after a short delay (allow window to appear first)
        QTimer.singleShot(800, lambda: BUS.ticker_changed.emit("SPY"))

    def _on_news(self, articles):
        BUS.news_ready.emit(articles)
        self.news_pnl.refresh(articles)

    def closeEvent(self, event):
        self.watchlist.wt.stop()
        self.watchlist.wt.quit()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Suppress matplotlib font warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setApplicationVersion(VER)
    # AA_EnableHighDpiScaling / AA_UseHighDpiPixmaps removed in Qt6 (always enabled).
    # Guard for PyQt5 backwards compatibility.
    for _attr in ('AA_EnableHighDpiScaling', 'AA_UseHighDpiPixmaps'):
        if hasattr(Qt, _attr):
            app.setAttribute(getattr(Qt, _attr), True)

    win = MainWindow()
    win.show()

    BUS.status_msg.emit(
        f"◈  {APP} {VER} initialized  ·  "
        f"yfinance={'✓' if HAS_YF else '✗'}  "
        f"pdfplumber={'✓' if HAS_PDF else '✗'}  "
        f"feedparser={'✓' if HAS_FEED else '✗'}"
    )

    # exec() in PyQt6; exec_() in PyQt5 — support both
    _exec = getattr(app, 'exec', None) or app.exec_
    sys.exit(_exec())


if __name__ == '__main__':
    main()
