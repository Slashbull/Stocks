"""
Wave Detection Ultimate 4.0 - PRODUCTION BEAST EDITION
=======================================================
Professional Stock Ranking System with Institutional-Grade Analytics
Engineered for Performance, Intelligence, and Reliability

Version: 4.0.0-BEAST
Last Updated: December 2024
Status: PRODUCTION READY - FULLY OPTIMIZED
Architecture: Hedge Fund Grade Alpha Generator
"""

# ============================================
# IMPORTS AND SETUP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import time
from io import BytesIO
import warnings
import gc
import re
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict
import numba

# Suppress warnings for clean production output
warnings.filterwarnings('ignore')

# Set NumPy for optimal performance
np.seterr(all='ignore')
np.random.seed(42)

# ============================================
# LOGGING CONFIGURATION
# ============================================

log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class Config:
    """
    Centralized configuration for the Wave Detection Beast system.
    Frozen to prevent runtime modification, ensuring consistency.
    """
    
    # --- Data Source Settings ---
    CSV_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    DEFAULT_GID: str = "1823439984"
    
    # --- Performance & Caching ---
    CACHE_TTL: int = 3600
    STALE_DATA_HOURS: int = 24
    PERFORMANCE_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'data_processing': 2.0,
        'filtering': 0.2,
        'pattern_detection': 0.5,
        'export_generation': 1.0,
        'search': 0.05
    })
    
    # --- Core Algorithm Weights (Must sum to 1.0) ---
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # --- Critical Data Columns ---
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'ret_30d', 'from_low_pct', 'from_high_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d', 'sector', 'industry', 'rvol'
    ])
    
    # --- Percentage Columns ---
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_3d', 'ret_7d', 
        'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ])
    
    # --- Volume Ratio Columns ---
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    # --- Pattern Detection Thresholds ---
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'category_leader': 90,
        'hidden_gem': 85,
        'acceleration': 75,
        'institutional': 70,
        'volume_explosion': 3.0,  # Using vol_ratio instead of RVOL
        'breakout_ready': 80,
        'market_leader': 90,
        'momentum_wave': 75,
        'liquid_leader': 80,
        'long_strength': 80,
        'trend_quality': 80,
        'value_momentum': 15,  # PE threshold
        'earnings_rocket': 50,  # EPS change threshold
        'quality_leader': 25,  # PE upper bound
        'turnaround': 100,  # EPS change for turnaround
        'high_pe_warning': 100,
        '52w_high_approach': -5,
        '52w_low_bounce': 20,
        'golden_zone_low': 60,
        'golden_zone_high': -40,
        'volume_accumulation': 1.2,
        'momentum_divergence': 1.5,
        'range_compression': 50,
        'stealth_accumulation': 1.1
    })
    
    # --- Value Bounds & Validation ---
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000),
        'pe': (-1000, 1000),
        'returns': (-99.99, 9999.99),
        'volume': (0, 1e12),
        'vol_ratio': (0.01, 100.0)  # Normalized volume ratios
    })
    
    # --- Tier Classifications ---
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {
            "Loss": (-float('inf'), 0),
            "0-5": (0, 5),
            "5-10": (5, 10),
            "10-20": (10, 20),
            "20-50": (20, 50),
            "50-100": (50, 100),
            "100+": (100, float('inf'))
        },
        "pe": {
            "Negative/NA": (-float('inf'), 0),
            "0-10": (0, 10),
            "10-15": (10, 15),
            "15-20": (15, 20),
            "20-30": (20, 30),
            "30-50": (30, 50),
            "50+": (50, float('inf'))
        },
        "price": {
            "0-100": (0, 100),
            "100-250": (100, 250),
            "250-500": (250, 500),
            "500-1000": (500, 1000),
            "1000-2500": (1000, 2500),
            "2500-5000": (2500, 5000),
            "5000+": (5000, float('inf'))
        }
    })
    
    # --- Display Settings ---
    DEFAULT_TOP_N: int = 25
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 200, 500])
    
    # --- Market Regime Thresholds ---
    REGIME_THRESHOLDS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'bull': {'breadth': 0.6, 'category_spread': 10},
        'bear': {'breadth': 0.4, 'category_spread': -10},
        'volatile': {'avg_rvol': 1.5, 'breadth': 0.5}
    })
    
    # --- Metric Tooltips ---
    METRIC_TOOLTIPS: Dict[str, str] = field(default_factory=lambda: {
        'vmi': 'Volume Momentum Index: Weighted volume trend score',
        'position_tension': 'Range position stress: Distance from 52W boundaries',
        'momentum_harmony': 'Multi-timeframe alignment: 0-4 consistency score',
        'overall_wave_strength': 'Composite wave score: Combined momentum indicators',
        'money_flow_mm': 'Money Flow in millions: Price × Volume × Vol Ratio',
        'master_score': 'Overall ranking score (0-100) with intelligent adjustments',
        'acceleration_score': 'Rate of momentum change (0-100)',
        'breakout_score': 'Probability of price breakout (0-100)',
        'trend_quality': 'SMA alignment quality (0-100)',
        'liquidity_score': 'Trading liquidity measure (0-100)',
        'momentum_decay': 'Momentum acceleration/deceleration rate',
        'institutional_footprint': 'Smart money accumulation score',
        'wyckoff_phase': 'Market cycle phase detection',
        'earnings_anomaly': 'PE/EPS mismatch opportunity score',
        'stock_dna': 'Unique behavioral pattern of the stock'
    })
    
    def __post_init__(self):
        """Validates configuration upon instantiation"""
        total_weight = (self.POSITION_WEIGHT + self.VOLUME_WEIGHT +
                       self.MOMENTUM_WEIGHT + self.ACCELERATION_WEIGHT +
                       self.BREAKOUT_WEIGHT + self.RVOL_WEIGHT)
        if not np.isclose(total_weight, 1.0, rtol=1e-5):
            raise ValueError(f"Scoring weights must sum to 1.0, but got {total_weight}")

# Instantiate global configuration
CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """Track and report performance metrics with intelligent alerts"""
    
    @staticmethod
    def timer(target_time: Optional[float] = None):
        """Performance timing decorator with target comparison and optimization hints"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    
                    # Intelligent performance alerts
                    if target_time and elapsed > target_time * 1.5:
                        logger.warning(f"{func.__name__} took {elapsed:.2f}s (target: {target_time}s) - Consider optimization")
                    elif elapsed > 1.0:
                        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                    
                    # Store metrics
                    if 'performance_metrics' not in st.session_state:
                        st.session_state.performance_metrics = {}
                    st.session_state.performance_metrics[func.__name__] = elapsed
                    
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
                    raise
            return wrapper
        return decorator

# ============================================
# SESSION STATE MANAGEMENT (UNIFIED)
# ============================================

class SessionStateManager:
    """Unified session state management - no duplicates"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with defaults"""
        defaults = {
            # Core State
            'search_query': "",
            'last_refresh': datetime.now(timezone.utc),
            'data_source': "sheet",
            'sheet_id': '',
            'gid': CONFIG.DEFAULT_GID,
            'user_preferences': {
                'default_top_n': CONFIG.DEFAULT_TOP_N,
                'display_mode': 'Technical',
                'last_filters': {},
                'adaptive_scoring': True
            },
            'active_filter_count': 0,
            'quick_filter': None,
            'quick_filter_applied': False,
            'show_debug': False,
            'performance_metrics': {},
            'data_quality': {},
            'market_regime': 'NEUTRAL',
            
            # Filters
            'display_count': CONFIG.DEFAULT_TOP_N,
            'sort_by': 'Rank',
            'export_template': 'Full Analysis (All Data)',
            'category_filter': [],
            'sector_filter': [],
            'industry_filter': [],
            'min_score': 0,
            'patterns': [],
            'trend_filter': "All Trends",
            'eps_tier_filter': [],
            'pe_tier_filter': [],
            'price_tier_filter': [],
            'min_eps_change': "",
            'min_pe': "",
            'max_pe': "",
            'require_fundamental_data': False,
            
            # Wave Radar Filters
            'wave_states_filter': [],
            'wave_strength_range_slider': (0, 100),
            'show_sensitivity_details': False,
            'show_market_regime': True,
            'wave_timeframe_select': "All Waves",
            'wave_sensitivity': "Balanced",
            
            # Cache
            'pattern_cache': {},
            'last_pattern_calculation': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def safe_get(key: str, default: Any = None) -> Any:
        """Safely get a session state value"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def safe_set(key: str, value: Any) -> None:
        """Safely set a session state value"""
        st.session_state[key] = value
    
    @staticmethod
    def build_filter_dict() -> Dict[str, Any]:
        """Build comprehensive filter dictionary from session state"""
        filters = {}
        
        # Categorical filters
        for key, filter_name in [
            ('category_filter', 'categories'),
            ('sector_filter', 'sectors'),
            ('industry_filter', 'industries')
        ]:
            if st.session_state.get(key):
                filters[filter_name] = st.session_state[key]
        
        # Numeric filters
        if st.session_state.get('min_score', 0) > 0:
            filters['min_score'] = st.session_state['min_score']
        
        # Pattern filters
        if st.session_state.get('patterns'):
            filters['patterns'] = st.session_state['patterns']
        
        # Trend filters
        if st.session_state.get('trend_filter') != "All Trends":
            trend_options = {
                "🔥 Strong Uptrend (80+)": (80, 100),
                "✅ Good Uptrend (60-79)": (60, 79),
                "➡️ Neutral Trend (40-59)": (40, 59),
                "⚠️ Weak/Downtrend (<40)": (0, 39)
            }
            filters['trend_range'] = trend_options.get(st.session_state['trend_filter'], (0, 100))
        
        # Wave filters
        if st.session_state.get('wave_strength_range_slider') != (0, 100):
            filters['wave_strength_range'] = st.session_state['wave_strength_range_slider']
        if st.session_state.get('wave_states_filter'):
            filters['wave_states'] = st.session_state['wave_states_filter']
        
        # Fundamental filters
        if st.session_state.get('require_fundamental_data'):
            filters['require_fundamental_data'] = True
        
        # PE filters
        for key in ['min_pe', 'max_pe', 'min_eps_change']:
            value = st.session_state.get(key)
            if value and str(value).strip():
                try:
                    filters[key] = float(value)
                except ValueError:
                    pass
        
        return filters
    
    @staticmethod
    def clear_filters():
        """Reset all filters to defaults"""
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter',
            'min_score', 'patterns', 'trend_filter',
            'eps_tier_filter', 'pe_tier_filter', 'price_tier_filter',
            'min_eps_change', 'min_pe', 'max_pe',
            'require_fundamental_data', 'quick_filter', 'quick_filter_applied',
            'wave_states_filter', 'wave_strength_range_slider'
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = "All Trends"
                    else:
                        st.session_state[key] = ""
                elif isinstance(st.session_state[key], tuple):
                    st.session_state[key] = (0, 100)
                else:
                    st.session_state[key] = 0 if isinstance(st.session_state.get(key), (int, float)) else None
        
        st.session_state.active_filter_count = 0
        logger.info("All filters cleared")

# ============================================
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidator:
    """Comprehensive data validation with intelligent correction"""
    
    _clipping_counts = {}
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> Tuple[bool, str]:
        """Validate DataFrame structure and quality"""
        if df is None:
            return False, f"{context}: DataFrame is None"
        
        if df.empty:
            return False, f"{context}: DataFrame is empty"
        
        # Check critical columns
        missing_critical = [col for col in CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            return False, f"{context}: Missing critical columns: {missing_critical}"
        
        # Check duplicates
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"{context}: Found {duplicates} duplicate tickers")
        
        # Calculate data completeness
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        completeness = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Update session state
        SessionStateManager.safe_set('data_quality', {
            'completeness': completeness,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_tickers': duplicates,
            'context': context,
            'timestamp': datetime.now(timezone.utc)
        })
        
        logger.info(f"{context}: Validated {len(df)} rows, {completeness:.1f}% complete")
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Any, col_name: str, is_percentage: bool = False, 
                           bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """Clean and validate numeric values with intelligent bounds"""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        try:
            cleaned = str(value).strip()
            
            # Handle invalid strings
            if cleaned.upper() in ['', '-', 'N/A', 'NA', 'NAN', 'NONE', '#VALUE!', '#ERROR!', '#DIV/0!']:
                return np.nan
            
            # Remove symbols
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace('%', '')
            
            result = float(cleaned)
            
            # Apply intelligent bounds
            if bounds:
                min_val, max_val = bounds
                if result < min_val or result > max_val:
                    DataValidator._clipping_counts[col_name] = DataValidator._clipping_counts.get(col_name, 0) + 1
                    result = np.clip(result, min_val, max_val)
            
            if np.isnan(result) or np.isinf(result):
                return np.nan
            
            return result
            
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    @staticmethod
    def sanitize_string(value: Any, default: str = "Unknown") -> str:
        """Sanitize string values"""
        if pd.isna(value) or value is None:
            return default
        
        cleaned = str(value).strip()
        if cleaned.upper() in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']:
            return default
        
        return ' '.join(cleaned.split())

# ============================================
# INTELLIGENT METRICS ENGINE
# ============================================

class IntelligentMetrics:
    """Game-changing metrics from institutional strategies"""
    
    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_momentum_decay_vectorized(ret_7d: np.ndarray, ret_30d: np.ndarray) -> np.ndarray:
        """Vectorized momentum decay calculation using numba"""
        result = np.empty(len(ret_7d))
        for i in range(len(ret_7d)):
            if ret_30d[i] != 0 and not np.isnan(ret_7d[i]) and not np.isnan(ret_30d[i]):
                daily_7d = ret_7d[i] / 7
                daily_30d = ret_30d[i] / 30
                if daily_30d != 0:
                    result[i] = daily_7d / daily_30d
                else:
                    result[i] = 1.0
            else:
                result[i] = 1.0
        return result
    
    @staticmethod
    def calculate_momentum_decay(df: pd.DataFrame) -> pd.DataFrame:
        """Nobel Prize worthy insight - momentum has a half-life"""
        if 'ret_7d' not in df.columns or 'ret_30d' not in df.columns:
            df['momentum_decay'] = 1.0
            df['expected_next_7d'] = 0.0
            return df
        
        # Use vectorized numba function
        df['momentum_acceleration_factor'] = IntelligentMetrics._calculate_momentum_decay_vectorized(
            df['ret_7d'].fillna(0).values,
            df['ret_30d'].fillna(0).values
        )
        
        # Predict next 7 days
        df['expected_next_7d'] = df['ret_7d'] * df['momentum_acceleration_factor']
        
        # Classify momentum state
        df['momentum_state'] = 'STABLE'
        df.loc[df['momentum_acceleration_factor'] > 2, 'momentum_state'] = 'EXPLOSIVE'
        df.loc[df['momentum_acceleration_factor'] > 1.2, 'momentum_state'] = 'BUILDING'
        df.loc[df['momentum_acceleration_factor'] < 0.8, 'momentum_state'] = 'FADING'
        
        return df
    
    @staticmethod
    def detect_institutional_footprint(df: pd.DataFrame) -> pd.DataFrame:
        """Detect institutional vs retail behavior patterns"""
        if not all(col in df.columns for col in ['volume_30d', 'volume_180d', 'vol_ratio_1d_90d']):
            df['institutional_score'] = 0.0
            return df
        
        # Vectorized calculation
        historical_avg = df['volume_180d'] / 180
        recent_avg = df['volume_30d'] / 30
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            accumulation_ratio = np.where(recent_avg > 0, historical_avg / recent_avg, 1.0)
        
        # Institutional pattern: quiet accumulation then spike
        df['institutional_score'] = accumulation_ratio * df['vol_ratio_1d_90d'].fillna(1.0)
        
        # Flag institutional activity
        df['institutional_activity'] = 'NONE'
        df.loc[df['institutional_score'] > 3, 'institutional_activity'] = 'ACCUMULATION'
        df.loc[(df['institutional_score'] > 5) & (df['ret_30d'] > 10), 'institutional_activity'] = 'MARKUP'
        
        return df
    
    @staticmethod
    def detect_fibonacci_levels(df: pd.DataFrame) -> pd.DataFrame:
        """Hidden Fibonacci levels in 52-week range"""
        if not all(col in df.columns for col in ['high_52w', 'low_52w', 'price']):
            df['near_fibonacci'] = 0
            return df
        
        # Vectorized Fibonacci calculation
        range_52w = df['high_52w'] - df['low_52w']
        current_position = df['price'] - df['low_52w']
        
        with np.errstate(divide='ignore', invalid='ignore'):
            position_ratio = np.where(range_52w > 0, current_position / range_52w, 0.5)
        
        # Key Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        df['near_fibonacci'] = 0
        
        for level in fib_levels:
            mask = np.abs(position_ratio - level) < 0.03
            df.loc[mask, 'near_fibonacci'] = int(level * 100)
        
        # Golden ratio gets special treatment
        golden_mask = np.abs(position_ratio - 0.618) < 0.02
        df.loc[golden_mask, 'near_fibonacci'] = 618  # Golden ratio indicator
        
        return df
    
    @staticmethod
    def detect_volatility_squeeze(df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Band squeeze without calculating bands"""
        return_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m']
        available_cols = [col for col in return_cols if col in df.columns]
        
        if len(available_cols) < 4:
            df['volatility_compression'] = 1.0
            df['squeeze_state'] = 'NORMAL'
            return df
        
        # Calculate short-term vs long-term volatility
        short_cols = [c for c in ['ret_1d', 'ret_3d', 'ret_7d'] if c in df.columns]
        long_cols = [c for c in ['ret_30d', 'ret_3m', 'ret_6m'] if c in df.columns]
        
        if short_cols and long_cols:
            short_vol = df[short_cols].std(axis=1)
            long_vol = df[long_cols].std(axis=1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                df['volatility_compression'] = np.where(long_vol > 0, short_vol / long_vol, 1.0)
            
            # Classify squeeze state
            df['squeeze_state'] = 'NORMAL'
            df.loc[df['volatility_compression'] < 0.5, 'squeeze_state'] = 'COMPRESSED'
            df.loc[df['volatility_compression'] < 0.3, 'squeeze_state'] = 'EXTREME_SQUEEZE'
            df.loc[df['volatility_compression'] > 2, 'squeeze_state'] = 'EXPANDING'
        else:
            df['volatility_compression'] = 1.0
            df['squeeze_state'] = 'NORMAL'
        
        return df
    
    @staticmethod
    def detect_wyckoff_phase(df: pd.DataFrame) -> pd.DataFrame:
        """Richard Wyckoff's accumulation/distribution phases"""
        df['wyckoff_phase'] = 'NONE'
        
        if not all(col in df.columns for col in ['from_low_pct', 'vol_ratio_30d_90d', 'ret_30d']):
            return df
        
        # Accumulation: Near lows, volume building, price stable
        accumulation = (
            (df['from_low_pct'] < 20) &
            (df['vol_ratio_30d_90d'] > 1.2) &
            (np.abs(df['ret_30d']) < 10)
        )
        
        # Markup: Price and volume both increasing
        markup = (
            (df['ret_30d'] > 15) &
            (df.get('vol_ratio_7d_90d', 1) > df.get('vol_ratio_30d_90d', 1)) &
            (df.get('momentum_harmony', 0) >= 3)
        )
        
        # Distribution: High prices, high volume, momentum fading
        distribution = (
            (df.get('from_high_pct', -100) > -10) &
            (df.get('vol_ratio_1d_90d', 1) > 2) &
            (df.get('momentum_acceleration_factor', 1) < 0.8)
        )
        
        # Markdown: Price falling, volume may be high (panic)
        markdown = (
            (df['ret_30d'] < -15) &
            (df.get('momentum_harmony', 4) <= 1)
        )
        
        df.loc[accumulation, 'wyckoff_phase'] = 'ACCUMULATION'
        df.loc[markup, 'wyckoff_phase'] = 'MARKUP'
        df.loc[distribution, 'wyckoff_phase'] = 'DISTRIBUTION'
        df.loc[markdown, 'wyckoff_phase'] = 'MARKDOWN'
        
        return df
    
    @staticmethod
    def calculate_earnings_anomaly(df: pd.DataFrame) -> pd.DataFrame:
        """PE/EPS mismatch opportunities"""
        if not all(col in df.columns for col in ['pe', 'eps_change_pct']):
            df['earnings_anomaly_score'] = 0.0
            return df
        
        # Vectorized calculation
        valid_pe = (df['pe'] > 0) & (df['pe'] < 100)
        valid_eps = df['eps_change_pct'].notna()
        
        df['earnings_anomaly_score'] = 0.0
        
        # High growth, low PE = opportunity
        mask = valid_pe & valid_eps
        df.loc[mask, 'earnings_anomaly_score'] = df.loc[mask, 'eps_change_pct'] / (df.loc[mask, 'pe'] + 1)
        
        # Classify anomalies
        df['earnings_signal'] = 'NORMAL'
        df.loc[df['earnings_anomaly_score'] > 5, 'earnings_signal'] = 'UNDERVALUED'
        df.loc[df['earnings_anomaly_score'] > 10, 'earnings_signal'] = 'EXTREME_VALUE'
        df.loc[df['earnings_anomaly_score'] < -2, 'earnings_signal'] = 'OVERVALUED'
        
        return df
    
    @staticmethod
    def calculate_rrg_position(df: pd.DataFrame) -> pd.DataFrame:
        """Relative Rotation Graph positioning"""
        if 'ret_30d' not in df.columns or 'sector' not in df.columns:
            df['rrg_quadrant'] = 'UNKNOWN'
            return df
        
        # Calculate relative strength vs sector
        df['relative_strength'] = df['ret_30d'] - df.groupby('sector')['ret_30d'].transform('mean')
        
        # Calculate momentum of relative strength
        if 'ret_7d' in df.columns:
            df['rs_momentum'] = df['ret_7d'] - (df['ret_30d'] / 4)
        else:
            df['rs_momentum'] = 0
        
        # Classify into quadrants
        df['rrg_quadrant'] = 'LAGGING'
        df.loc[(df['relative_strength'] > 0) & (df['rs_momentum'] > 0), 'rrg_quadrant'] = 'LEADING'
        df.loc[(df['relative_strength'] < 0) & (df['rs_momentum'] > 0), 'rrg_quadrant'] = 'IMPROVING'
        df.loc[(df['relative_strength'] > 0) & (df['rs_momentum'] < 0), 'rrg_quadrant'] = 'WEAKENING'
        
        return df
    
    @staticmethod
    def calculate_stock_dna(df: pd.DataFrame) -> pd.DataFrame:
        """Every stock has a unique behavioral DNA"""
        df['stock_dna'] = 'NORMAL'
        
        # Analyze volatility profile
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            returns = df[['ret_1d', 'ret_7d', 'ret_30d']]
            volatility = returns.std(axis=1)
            
            # Classify DNA types
            high_vol = volatility > volatility.quantile(0.75)
            low_vol = volatility < volatility.quantile(0.25)
            
            # Momentum characteristics
            if 'momentum_acceleration_factor' in df.columns:
                explosive = df['momentum_acceleration_factor'] > 1.5
                steady = (df['momentum_acceleration_factor'] > 0.8) & (df['momentum_acceleration_factor'] < 1.2)
            else:
                explosive = pd.Series(False, index=df.index)
                steady = pd.Series(True, index=df.index)
            
            # Volume patterns
            if 'vol_ratio_30d_90d' in df.columns:
                accumulating = df['vol_ratio_30d_90d'] > 1.2
            else:
                accumulating = pd.Series(False, index=df.index)
            
            # Assign DNA profiles
            df.loc[high_vol & explosive, 'stock_dna'] = 'EXPLOSIVE_MOVER'
            df.loc[low_vol & steady, 'stock_dna'] = 'STEADY_GROWER'
            df.loc[accumulating & steady, 'stock_dna'] = 'ACCUMULATION_PLAY'
            df.loc[high_vol & ~explosive, 'stock_dna'] = 'VOLATILE_TRADER'
            df.loc[low_vol & explosive, 'stock_dna'] = 'BREAKOUT_CANDIDATE'
        
        return df

# ============================================
# DATA PROCESSING ENGINE
# ============================================

class DataProcessor:
    """Handles data processing pipeline with performance optimization"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Main data processing pipeline - fully vectorized"""
        df = df.copy()
        initial_count = len(df)
        
        # Process numeric columns - vectorized
        numeric_cols = [col for col in df.columns if col not in 
                       ['ticker', 'company_name', 'category', 'sector', 'industry', 'year', 'market_cap']]
        
        for col in numeric_cols:
            if col in df.columns:
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                
                # Determine bounds
                if 'volume' in col.lower():
                    bounds = CONFIG.VALUE_BOUNDS['volume']
                elif col == 'pe':
                    bounds = CONFIG.VALUE_BOUNDS['pe']
                elif is_pct:
                    bounds = CONFIG.VALUE_BOUNDS['returns']
                elif 'vol_ratio' in col or col == 'rvol':
                    bounds = CONFIG.VALUE_BOUNDS['vol_ratio']
                else:
                    bounds = CONFIG.VALUE_BOUNDS.get('price', None)
                
                # Vectorized cleaning
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if bounds:
                    df[col] = df[col].clip(bounds[0], bounds[1])
        
        # Process categorical columns
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
        # Fix RVOL calculation if needed
        if 'rvol' in df.columns and 'vol_ratio_1d_90d' not in df.columns:
            # Convert RVOL to normalized ratio
            df['vol_ratio_1d_90d'] = df['rvol'] * 90  # Approximate conversion
            logger.info("Converted RVOL to normalized volume ratio")
        
        # Handle volume ratios
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                # These are already normalized (1.0 = average)
                df[col] = df[col].clip(0.01, 100.0)
                df[col] = df[col].fillna(1.0)
        
        # Critical validation
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        
        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        # Fill missing values intelligently
        df = DataProcessor._fill_missing_values(df)
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        # Log processing results
        removed_count = initial_count - len(df)
        if removed_count > 0:
            metadata['warnings'].append(f"Removed {removed_count} invalid rows")
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows")
        
        return df
    
    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with intelligent defaults"""
        # Position metrics
        if 'from_low_pct' in df.columns:
            df['from_low_pct'] = df['from_low_pct'].fillna(50)
        if 'from_high_pct' in df.columns:
            df['from_high_pct'] = df['from_high_pct'].fillna(-50)
        
        # Volume ratios default to 1 (average)
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(1.0)
        
        # Returns default to 0
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols:
            df[col] = df[col].fillna(0)
        
        # Volume columns default to 0
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols:
            df[col] = df[col].fillna(0)
        
        # Categorical defaults
        for col in ['category', 'sector', 'industry']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = 'Unknown'
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications for filtering"""
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val <= value <= max_val:
                    return tier_name
                if min_val == -float('inf') and value <= max_val:
                    return tier_name
                if max_val == float('inf') and value > min_val:
                    return tier_name
            
            return "Unknown"
        
        # Add tier classifications
        if 'eps_current' in df.columns:
            df['eps_tier'] = df['eps_current'].apply(lambda x: classify_tier(x, CONFIG.TIERS['eps']))
        
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(lambda x: classify_tier(x, CONFIG.TIERS['pe']))
        
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(lambda x: classify_tier(x, CONFIG.TIERS['price']))
        
        return df

# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    """Calculate advanced metrics with vectorized operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics including intelligent ones"""
        if df.empty:
            return df
        
        # Money Flow (in millions) - using vol_ratio instead of rvol
        if all(col in df.columns for col in ['price', 'volume_1d', 'vol_ratio_1d_90d']):
            df['money_flow'] = df['price'] * df['volume_1d'] * df['vol_ratio_1d_90d'].fillna(1.0)
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
        else:
            df['money_flow_mm'] = 0.0
        
        # Volume Momentum Index (VMI) - vectorized
        vmi_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d']
        if all(col in df.columns for col in vmi_cols):
            df['vmi'] = (
                df['vol_ratio_1d_90d'].fillna(1.0) * 4 +
                df['vol_ratio_7d_90d'].fillna(1.0) * 3 +
                df['vol_ratio_30d_90d'].fillna(1.0) * 2 +
                df['vol_ratio_90d_180d'].fillna(1.0) * 1
            ) / 10
        else:
            df['vmi'] = 1.0
        
        # Position Tension - vectorized
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'].fillna(50) + np.abs(df['from_high_pct'].fillna(-50))
        else:
            df['position_tension'] = 100.0
        
        # Momentum Harmony - vectorized
        harmony_score = pd.Series(0, index=df.index, dtype=int)
        
        if 'ret_1d' in df.columns:
            harmony_score += (df['ret_1d'] > 0).astype(int)
        
        if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
            daily_7d = df['ret_7d'] / 7
            daily_30d = df['ret_30d'] / 30
            harmony_score += (daily_7d > daily_30d).astype(int)
        
        if 'ret_30d' in df.columns and 'ret_3m' in df.columns:
            daily_30d = df['ret_30d'] / 30
            daily_3m = df['ret_3m'] / 90
            harmony_score += (daily_30d > daily_3m).astype(int)
        
        if 'ret_3m' in df.columns:
            harmony_score += (df['ret_3m'] > 0).astype(int)
        
        df['momentum_harmony'] = harmony_score
        
        # Wave State - using vol_ratio instead of rvol
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)
        
        # Overall Wave Strength
        score_cols = ['momentum_score', 'acceleration_score', 'rvol_score', 'breakout_score']
        if all(col in df.columns for col in score_cols):
            df['overall_wave_strength'] = (
                df['momentum_score'].fillna(50) * 0.3 +
                df['acceleration_score'].fillna(50) * 0.3 +
                df['rvol_score'].fillna(50) * 0.2 +
                df['breakout_score'].fillna(50) * 0.2
            )
        else:
            df['overall_wave_strength'] = 50.0
        
        # Apply intelligent metrics
        df = IntelligentMetrics.calculate_momentum_decay(df)
        df = IntelligentMetrics.detect_institutional_footprint(df)
        df = IntelligentMetrics.detect_fibonacci_levels(df)
        df = IntelligentMetrics.detect_volatility_squeeze(df)
        df = IntelligentMetrics.detect_wyckoff_phase(df)
        df = IntelligentMetrics.calculate_earnings_anomaly(df)
        df = IntelligentMetrics.calculate_rrg_position(df)
        df = IntelligentMetrics.calculate_stock_dna(df)
        
        return df
    
    @staticmethod
    def _get_wave_state(row: pd.Series) -> str:
        """Determine wave state with enhanced logic"""
        signals = 0
        
        # Use vol_ratio instead of raw rvol
        if row.get('momentum_score', 0) > 70:
            signals += 1
        if row.get('volume_score', 0) > 70:
            signals += 1
        if row.get('acceleration_score', 0) > 70:
            signals += 1
        if row.get('vol_ratio_1d_90d', 1) > 2:  # Changed from rvol
            signals += 1
        
        # Enhanced wave states
        if signals >= 4 and row.get('from_high_pct', -100) > -3:
            return "🌊🌊🌊🔥 CRESTING+BREAKOUT"
        elif signals >= 4:
            return "🌊🌊🌊 CRESTING"
        elif signals >= 3:
            return "🌊🌊 BUILDING"
        elif signals >= 1:
            return "🌊 FORMING"
        else:
            return "💥 BREAKING"

# ============================================
# RANKING ENGINE - OPTIMIZED
# ============================================

class RankingEngine:
    """Core ranking calculations with intelligent adjustments"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all scores with market regime adaptation"""
        if df.empty:
            return df
        
        logger.info("Starting optimized ranking calculations...")
        
        # Calculate component scores - all vectorized
        df['position_score'] = RankingEngine._calculate_position_score_vectorized(df)
        df['volume_score'] = RankingEngine._calculate_volume_score_vectorized(df)
        df['momentum_score'] = RankingEngine._calculate_momentum_score_vectorized(df)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score_vectorized(df)
        df['breakout_score'] = RankingEngine._calculate_breakout_score_vectorized(df)
        df['rvol_score'] = RankingEngine._calculate_rvol_score_vectorized(df)
        
        # Calculate auxiliary scores
        df['trend_quality'] = RankingEngine._calculate_trend_quality_vectorized(df)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength_vectorized(df)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score_vectorized(df)
        
        # Calculate base master score
        scores_matrix = df[[
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score'
        ]].fillna(50).values
        
        weights = np.array([
            CONFIG.POSITION_WEIGHT, CONFIG.VOLUME_WEIGHT, CONFIG.MOMENTUM_WEIGHT,
            CONFIG.ACCELERATION_WEIGHT, CONFIG.BREAKOUT_WEIGHT, CONFIG.RVOL_WEIGHT
        ])
        
        df['master_score'] = np.dot(scores_matrix, weights)
        
        # Apply intelligent adjustments
        df = RankingEngine._apply_intelligent_adjustments(df)
        
        # Final clipping and ranking
        df['master_score'] = df['master_score'].clip(0, 100)
        
        # Calculate ranks
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom')
        df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
        
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        
        # Calculate category ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        logger.info(f"Ranking complete: {len(df)} stocks processed")
        
        return df
    
    @staticmethod
    def _calculate_position_score_vectorized(df: pd.DataFrame) -> pd.Series:
        """Vectorized position score calculation"""
        if 'from_low_pct' not in df.columns and 'from_high_pct' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        from_low = df.get('from_low_pct', pd.Series(50, index=df.index))
        from_high = df.get('from_high_pct', pd.Series(-50, index=df.index))
        
        # Vectorized percentile calculation
        from_low_pct = from_low.rank(pct=True) * 100
        from_high_pct = (100 - from_high.rank(pct=True) * 100)
        
        score = from_low_pct * 0.6 + from_high_pct * 0.4
        return score.fillna(50).clip(0, 100)
    
    @staticmethod
    def _calculate_volume_score_vectorized(df: pd.DataFrame) -> pd.Series:
        """Vectorized volume score calculation"""
        vol_cols = [
            ('vol_ratio_1d_90d', 0.20),
            ('vol_ratio_7d_90d', 0.20),
            ('vol_ratio_30d_90d', 0.20),
            ('vol_ratio_30d_180d', 0.15),
            ('vol_ratio_90d_180d', 0.25)
        ]
        
        score = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for col, weight in vol_cols:
            if col in df.columns:
                col_rank = df[col].rank(pct=True) * 100
                score += col_rank.fillna(50) * weight
                total_weight += weight
        
        if total_weight > 0:
            score = score / total_weight
        else:
            score = pd.Series(50.0, index=df.index)
        
        return score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score_vectorized(df: pd.DataFrame) -> pd.Series:
        """Vectorized momentum score calculation"""
        if 'ret_30d' in df.columns:
            base_score = df['ret_30d'].rank(pct=True) * 100
        elif 'ret_7d' in df.columns:
            base_score = df['ret_7d'].rank(pct=True) * 100
        else:
            return pd.Series(50.0, index=df.index)
        
        # Add consistency bonus
        bonus = pd.Series(0.0, index=df.index)
        
        if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            bonus[all_positive] = 5
            
            # Acceleration bonus
            daily_7d = df['ret_7d'] / 7
            daily_30d = df['ret_30d'] / 30
            accelerating = all_positive & (daily_7d > daily_30d)
            bonus[accelerating] = 10
        
        return (base_score + bonus).fillna(50).clip(0, 100)
    
    @staticmethod
    def _calculate_acceleration_score_vectorized(df: pd.DataFrame) -> pd.Series:
        """Vectorized acceleration score calculation"""
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        if sum(c in df.columns for c in req_cols) < 2:
            return pd.Series(50.0, index=df.index)
        
        score = pd.Series(50.0, index=df.index)
        
        if all(c in df.columns for c in req_cols):
            ret_1d = df['ret_1d'].fillna(0)
            ret_7d = df['ret_7d'].fillna(0)
            ret_30d = df['ret_30d'].fillna(0)
            
            avg_daily_1d = ret_1d
            avg_daily_7d = ret_7d / 7
            avg_daily_30d = ret_30d / 30
            
            # Vectorized conditions
            perfect = (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
            good = ~perfect & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0)
            moderate = ~perfect & ~good & (ret_1d > 0)
            slight_decel = (ret_1d <= 0) & (ret_7d > 0)
            strong_decel = (ret_1d <= 0) & (ret_7d <= 0)
            
            score[perfect] = 100
            score[good] = 80
            score[moderate] = 60
            score[slight_decel] = 40
            score[strong_decel] = 20
        
        return score.clip(0, 100)
    
    @staticmethod
    def _calculate_breakout_score_vectorized(df: pd.DataFrame) -> pd.Series:
        """Vectorized breakout score calculation"""
        score = pd.Series(50.0, index=df.index)
        
        # Distance from high
        if 'from_high_pct' in df.columns:
            distance_factor = (100 + df['from_high_pct']).clip(0, 100)
        else:
            distance_factor = pd.Series(50.0, index=df.index)
        
        # Volume surge
        if 'vol_ratio_7d_90d' in df.columns:
            volume_factor = ((df['vol_ratio_7d_90d'] - 1) * 50).clip(0, 100)
        else:
            volume_factor = pd.Series(50.0, index=df.index)
        
        # Trend alignment
        trend_factor = pd.Series(50.0, index=df.index)
        if 'price' in df.columns:
            sma_count = 0
            above_sma = pd.Series(0.0, index=df.index)
            
            for sma_col in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma_col in df.columns:
                    above_sma += (df['price'] > df[sma_col]).astype(float)
                    sma_count += 1
            
            if sma_count > 0:
                trend_factor = (above_sma / sma_count) * 100
        
        score = distance_factor * 0.4 + volume_factor * 0.4 + trend_factor * 0.2
        
        return score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score_vectorized(df: pd.DataFrame) -> pd.Series:
        """Calculate RVOL score using vol_ratio"""
        if 'vol_ratio_1d_90d' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        vol_ratio = df['vol_ratio_1d_90d'].fillna(1.0)
        score = pd.Series(50.0, index=df.index)
        
        # Adjusted thresholds for normalized volume ratios
        score[vol_ratio > 5] = 95
        score[(vol_ratio > 3) & (vol_ratio <= 5)] = 90
        score[(vol_ratio > 2) & (vol_ratio <= 3)] = 85
        score[(vol_ratio > 1.5) & (vol_ratio <= 2)] = 80
        score[(vol_ratio > 1.2) & (vol_ratio <= 1.5)] = 70
        score[(vol_ratio > 1.0) & (vol_ratio <= 1.2)] = 60
        score[(vol_ratio > 0.8) & (vol_ratio <= 1.0)] = 50
        score[(vol_ratio > 0.5) & (vol_ratio <= 0.8)] = 40
        score[vol_ratio <= 0.5] = 30
        
        return score.clip(0, 100)
    
    @staticmethod
    def _calculate_trend_quality_vectorized(df: pd.DataFrame) -> pd.Series:
        """Vectorized trend quality calculation"""
        if 'price' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        score = pd.Series(50.0, index=df.index)
        price = df['price']
        
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        available_smas = [col for col in sma_cols if col in df.columns]
        
        if len(available_smas) == 3:
            # Perfect trend: Price > 20 > 50 > 200
            perfect = (price > df['sma_20d']) & (df['sma_20d'] > df['sma_50d']) & (df['sma_50d'] > df['sma_200d'])
            score[perfect] = 100
            
            # Strong trend: Price above all
            strong = ~perfect & (price > df['sma_20d']) & (price > df['sma_50d']) & (price > df['sma_200d'])
            score[strong] = 85
            
            # Count how many SMAs price is above
            above_count = sum([(price > df[sma]).astype(int) for sma in available_smas])
            score[above_count == 2] = 70
            score[above_count == 1] = 40
            score[above_count == 0] = 20
        
        return score.clip(0, 100)
    
    @staticmethod
    def _calculate_long_term_strength_vectorized(df: pd.DataFrame) -> pd.Series:
        """Vectorized long-term strength calculation"""
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available = [col for col in lt_cols if col in df.columns]
        
        if not available:
            return pd.Series(50.0, index=df.index)
        
        avg_return = df[available].mean(axis=1)
        score = pd.Series(50.0, index=df.index)
        
        score[avg_return > 100] = 100
        score[(avg_return > 50) & (avg_return <= 100)] = 90
        score[(avg_return > 30) & (avg_return <= 50)] = 80
        score[(avg_return > 15) & (avg_return <= 30)] = 70
        score[(avg_return > 5) & (avg_return <= 15)] = 60
        score[(avg_return > 0) & (avg_return <= 5)] = 50
        score[(avg_return > -10) & (avg_return <= 0)] = 40
        score[(avg_return > -25) & (avg_return <= -10)] = 30
        score[avg_return <= -25] = 20
        
        return score.clip(0, 100)
    
    @staticmethod
    def _calculate_liquidity_score_vectorized(df: pd.DataFrame) -> pd.Series:
        """Vectorized liquidity score calculation"""
        if 'volume_30d' not in df.columns or 'price' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        dollar_volume = df['volume_30d'] * df['price']
        return (dollar_volume.rank(pct=True) * 100).fillna(50).clip(0, 100)
    
    @staticmethod
    def _apply_intelligent_adjustments(df: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent score adjustments based on patterns"""
        
        # Perfect harmony bonus
        if 'momentum_harmony' in df.columns and 'rvol_score' in df.columns:
            perfect_harmony = (df['momentum_harmony'] == 4) & (df['rvol_score'] > 80)
            df.loc[perfect_harmony, 'master_score'] *= 1.05
        
        # Breakout setup bonus
        if 'from_high_pct' in df.columns and 'ret_30d' in df.columns:
            breakout_setup = (df['from_high_pct'] > -5) & (df['ret_30d'] > 20)
            df.loc[breakout_setup, 'master_score'] *= 1.03
        
        # Institutional accumulation bonus
        if 'institutional_activity' in df.columns:
            inst_markup = df['institutional_activity'] == 'MARKUP'
            df.loc[inst_markup, 'master_score'] *= 1.04
        
        # Wyckoff markup bonus
        if 'wyckoff_phase' in df.columns:
            wyckoff_markup = df['wyckoff_phase'] == 'MARKUP'
            df.loc[wyckoff_markup, 'master_score'] *= 1.03
        
        # Earnings anomaly bonus
        if 'earnings_anomaly_score' in df.columns:
            extreme_value = df['earnings_anomaly_score'] > 10
            df.loc[extreme_value, 'master_score'] *= 1.05
        
        # Fibonacci support bonus
        if 'near_fibonacci' in df.columns:
            golden_ratio = df['near_fibonacci'] == 618
            df.loc[golden_ratio, 'master_score'] *= 1.02
        
        # Explosive DNA bonus
        if 'stock_dna' in df.columns:
            explosive = df['stock_dna'] == 'EXPLOSIVE_MOVER'
            high_vol = df.get('vol_ratio_1d_90d', pd.Series(1, index=df.index)) > 2
            df.loc[explosive & high_vol, 'master_score'] *= 1.03
        
        # Market regime adaptation
        market_regime = SessionStateManager.safe_get('market_regime', 'NEUTRAL')
        if market_regime == 'RISK_ON_BULL':
            # Favor small caps and momentum
            if 'category' in df.columns:
                small_caps = df['category'].isin(['Small Cap', 'Micro Cap'])
                df.loc[small_caps, 'master_score'] *= 1.02
        elif market_regime == 'RISK_OFF_DEFENSIVE':
            # Favor large caps and low PE
            if 'category' in df.columns and 'pe' in df.columns:
                large_caps = df['category'].isin(['Large Cap', 'Mega Cap'])
                low_pe = (df['pe'] > 0) & (df['pe'] < 20)
                df.loc[large_caps & low_pe, 'master_score'] *= 1.02
        
        return df
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ranks within categories"""
        if 'category' not in df.columns or 'master_score' not in df.columns:
            return df
        
        df['category_rank'] = df.groupby('category')['master_score'].rank(
            method='first', ascending=False, na_option='bottom'
        )
        
        df['category_percentile'] = df.groupby('category')['master_score'].rank(
            pct=True, ascending=True, na_option='bottom'
        ) * 100
        
        return df

# ============================================
# PATTERN DETECTION ENGINE - OPTIMIZED
# ============================================

class PatternDetector:
    """Advanced pattern detection with caching"""
    
    PATTERN_METADATA = {
        '🔥 CAT LEADER': {'importance_weight': 10},
        '💎 HIDDEN GEM': {'importance_weight': 10},
        '🚀 ACCELERATING': {'importance_weight': 10},
        '🏦 INSTITUTIONAL': {'importance_weight': 10},
        '⚡ VOL EXPLOSION': {'importance_weight': 15},
        '🎯 BREAKOUT': {'importance_weight': 10},
        '👑 MARKET LEADER': {'importance_weight': 15},
        '🌊 MOMENTUM WAVE': {'importance_weight': 10},
        '💰 LIQUID LEADER': {'importance_weight': 5},
        '💪 LONG STRENGTH': {'importance_weight': 5},
        '📈 QUALITY TREND': {'importance_weight': 10},
        '💎 VALUE MOMENTUM': {'importance_weight': 10},
        '📊 EARNINGS ROCKET': {'importance_weight': 10},
        '🏆 QUALITY LEADER': {'importance_weight': 10},
        '⚡ TURNAROUND': {'importance_weight': 10},
        '⚠️ HIGH PE': {'importance_weight': -5},
        '🎯 52W HIGH APPROACH': {'importance_weight': 10},
        '🔄 52W LOW BOUNCE': {'importance_weight': 10},
        '👑 GOLDEN ZONE': {'importance_weight': 5},
        '📊 VOL ACCUMULATION': {'importance_weight': 5},
        '🔀 MOMENTUM DIVERGE': {'importance_weight': 10},
        '🎯 RANGE COMPRESS': {'importance_weight': 5},
        '🤫 STEALTH': {'importance_weight': 10},
        '🧛 VAMPIRE': {'importance_weight': 10},
        '⛈️ PERFECT STORM': {'importance_weight': 20},
        '🏆 EXTREME OPPORTUNITY': {'importance_weight': 25}  # New pattern
    }
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.3)
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def detect_all_patterns_cached(df_hash: str, df_dict: dict) -> Tuple[List[str], List[float]]:
        """Cached pattern detection"""
        df = pd.DataFrame(df_dict)
        patterns, confidence = PatternDetector._detect_patterns_internal(df)
        return patterns.tolist(), confidence.tolist()
    
    @staticmethod
    def detect_all_patterns_optimized(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns with caching"""
        if df.empty:
            df['patterns'] = ''
            df['pattern_confidence'] = 0.0
            return df
        
        # Check cache
        df_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
        
        try:
            # Try to use cached results
            patterns_list, confidence_list = PatternDetector.detect_all_patterns_cached(
                df_hash, df.to_dict()
            )
            df['patterns'] = patterns_list
            df['pattern_confidence'] = confidence_list
        except:
            # Fallback to direct calculation
            patterns, confidence = PatternDetector._detect_patterns_internal(df)
            df['patterns'] = patterns
            df['pattern_confidence'] = confidence
        
        logger.info(f"Pattern detection completed for {len(df)} stocks")
        return df
    
    @staticmethod
    def _detect_patterns_internal(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Internal pattern detection logic - fully vectorized"""
        patterns_list = []
        
        # Technical Patterns
        patterns_list.append(('🔥 CAT LEADER', 
            df.get('category_percentile', pd.Series(0, index=df.index)) >= CONFIG.PATTERN_THRESHOLDS['category_leader']))
        
        patterns_list.append(('💎 HIDDEN GEM',
            (df.get('category_percentile', pd.Series(0, index=df.index)) >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) &
            (df.get('percentile', pd.Series(100, index=df.index)) < 70)))
        
        patterns_list.append(('🚀 ACCELERATING',
            df.get('acceleration_score', pd.Series(0, index=df.index)) >= CONFIG.PATTERN_THRESHOLDS['acceleration']))
        
        patterns_list.append(('🏦 INSTITUTIONAL',
            (df.get('volume_score', pd.Series(0, index=df.index)) >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
            (df.get('vol_ratio_90d_180d', pd.Series(1, index=df.index)) > 1.1)))
        
        patterns_list.append(('⚡ VOL EXPLOSION',
            df.get('vol_ratio_1d_90d', pd.Series(1, index=df.index)) > CONFIG.PATTERN_THRESHOLDS['volume_explosion']))
        
        patterns_list.append(('🎯 BREAKOUT',
            df.get('breakout_score', pd.Series(0, index=df.index)) >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']))
        
        patterns_list.append(('👑 MARKET LEADER',
            df.get('percentile', pd.Series(0, index=df.index)) >= CONFIG.PATTERN_THRESHOLDS['market_leader']))
        
        patterns_list.append(('🌊 MOMENTUM WAVE',
            (df.get('momentum_score', pd.Series(0, index=df.index)) >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
            (df.get('acceleration_score', pd.Series(0, index=df.index)) >= 70)))
        
        # Fundamental Patterns
        if 'pe' in df.columns:
            patterns_list.append(('💎 VALUE MOMENTUM',
                (df['pe'] > 0) & (df['pe'] < CONFIG.PATTERN_THRESHOLDS['value_momentum']) &
                (df.get('master_score', pd.Series(0, index=df.index)) >= 70)))
            
            patterns_list.append(('⚠️ HIGH PE',
                df['pe'] > CONFIG.PATTERN_THRESHOLDS['high_pe_warning']))
        
        if 'eps_change_pct' in df.columns:
            patterns_list.append(('📊 EARNINGS ROCKET',
                (df['eps_change_pct'] > CONFIG.PATTERN_THRESHOLDS['earnings_rocket']) &
                (df.get('acceleration_score', pd.Series(0, index=df.index)) >= 70)))
            
            patterns_list.append(('⚡ TURNAROUND',
                (df['eps_change_pct'] > CONFIG.PATTERN_THRESHOLDS['turnaround']) &
                (df.get('volume_score', pd.Series(0, index=df.index)) >= 60)))
        
        # Range Patterns
        patterns_list.append(('🎯 52W HIGH APPROACH',
            (df.get('from_high_pct', pd.Series(-100, index=df.index)) > CONFIG.PATTERN_THRESHOLDS['52w_high_approach']) &
            (df.get('volume_score', pd.Series(0, index=df.index)) >= 70)))
        
        patterns_list.append(('🔄 52W LOW BOUNCE',
            (df.get('from_low_pct', pd.Series(100, index=df.index)) < CONFIG.PATTERN_THRESHOLDS['52w_low_bounce']) &
            (df.get('acceleration_score', pd.Series(0, index=df.index)) >= 80)))
        
        # Special Patterns
        patterns_list.append(('⛈️ PERFECT STORM',
            (df.get('momentum_harmony', pd.Series(0, index=df.index)) == 4) &
            (df.get('master_score', pd.Series(0, index=df.index)) > 80)))
        
        # NEW: Extreme Opportunity Pattern
        patterns_list.append(('🏆 EXTREME OPPORTUNITY',
            (df.get('master_score', pd.Series(0, index=df.index)) > 85) &
            (df.get('vol_ratio_1d_90d', pd.Series(1, index=df.index)) > 3) &
            (df.get('momentum_harmony', pd.Series(0, index=df.index)) >= 3) &
            (df.get('from_high_pct', pd.Series(-100, index=df.index)) > -10)))
        
        # Combine patterns
        pattern_matrix = pd.DataFrame({name: mask for name, mask in patterns_list})
        
        # Create pattern strings
        patterns = pattern_matrix.apply(
            lambda row: ' | '.join(row.index[row].tolist()), axis=1
        )
        
        # Calculate confidence scores
        max_score = sum(abs(item['importance_weight']) for item in PatternDetector.PATTERN_METADATA.values())
        
        confidence = pattern_matrix.apply(
            lambda row: sum(
                PatternDetector.PATTERN_METADATA.get(p, {'importance_weight': 0})['importance_weight']
                for p in row.index[row]
            ) / max_score * 100 if max_score > 0 else 0,
            axis=1
        )
        
        return patterns, confidence.clip(0, 100)

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    """Advanced market analysis with regime detection"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def detect_market_regime(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Detect current market regime with intelligent analysis"""
        if df.empty:
            return "😴 NO DATA", {}
        
        metrics = {}
        
        # Category analysis
        if 'category' in df.columns and 'master_score' in df.columns:
            category_scores = df.groupby('category')['master_score'].mean()
            
            micro_small = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])]
            large_mega = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])]
            
            metrics['micro_small_avg'] = micro_small.mean() if len(micro_small) > 0 else 50
            metrics['large_mega_avg'] = large_mega.mean() if len(large_mega) > 0 else 50
            metrics['category_spread'] = metrics['micro_small_avg'] - metrics['large_mega_avg']
        
        # Market breadth
        if 'ret_30d' in df.columns:
            metrics['breadth'] = (df['ret_30d'] > 0).mean()
        else:
            metrics['breadth'] = 0.5
        
        # Average volume activity
        if 'vol_ratio_1d_90d' in df.columns:
            metrics['avg_vol_ratio'] = df['vol_ratio_1d_90d'].median()
        else:
            metrics['avg_vol_ratio'] = 1.0
        
        # Determine regime
        if metrics['micro_small_avg'] > metrics['large_mega_avg'] + 10 and metrics['breadth'] > 0.6:
            regime = "🔥 RISK-ON BULL"
        elif metrics['large_mega_avg'] > metrics['micro_small_avg'] + 10 and metrics['breadth'] < 0.4:
            regime = "🛡️ RISK-OFF DEFENSIVE"
        elif metrics['avg_vol_ratio'] > 1.5 and metrics['breadth'] > 0.5:
            regime = "⚡ VOLATILE OPPORTUNITY"
        else:
            regime = "😴 RANGE-BOUND"
        
        metrics['regime'] = regime
        
        # Store in session state for adaptive scoring
        SessionStateManager.safe_set('market_regime', regime)
        
        return regime, metrics
    
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Detect sector rotation with smart sampling"""
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        
        # Smart sampling for fair comparison
        sector_samples = []
        for sector, group in df.groupby('sector'):
            if sector != 'Unknown':
                # Dynamic sampling based on size
                n = len(group)
                if n <= 5:
                    sample = group
                elif n <= 20:
                    sample = group.nlargest(int(n * 0.8), 'master_score')
                elif n <= 50:
                    sample = group.nlargest(int(n * 0.6), 'master_score')
                else:
                    sample = group.nlargest(min(50, int(n * 0.25)), 'master_score')
                
                sector_samples.append(sample)
        
        if not sector_samples:
            return pd.DataFrame()
        
        normalized_df = pd.concat(sector_samples, ignore_index=True)
        
        # Calculate flow metrics
        sector_metrics = normalized_df.groupby('sector').agg({
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'vol_ratio_1d_90d': 'mean',
            'ret_30d': 'mean',
            'money_flow_mm': 'sum'
        }).round(2)
        
        # Flatten column names
        sector_metrics.columns = ['_'.join(col).strip() for col in sector_metrics.columns]
        
        # Calculate flow score
        sector_metrics['flow_score'] = (
            sector_metrics['master_score_mean'] * 0.3 +
            sector_metrics['master_score_median'] * 0.2 +
            sector_metrics['momentum_score_mean'] * 0.25 +
            sector_metrics['volume_score_mean'] * 0.25
        )
        
        # Add original counts
        original_counts = df.groupby('sector').size()
        sector_metrics['total_stocks'] = original_counts
        sector_metrics['analyzed_stocks'] = sector_metrics['master_score_count']
        sector_metrics['sampling_pct'] = (sector_metrics['analyzed_stocks'] / sector_metrics['total_stocks'] * 100).round(1)
        
        return sector_metrics.sort_values('flow_score', ascending=False)

# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    """High-performance filtering with vectorized operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters efficiently"""
        if df.empty:
            return df
        
        # Build mask list
        masks = []
        
        # Categorical filters
        for col, filter_key in [('category', 'categories'), ('sector', 'sectors'), ('industry', 'industries')]:
            if filter_key in filters and filters[filter_key]:
                masks.append(df[col].isin(filters[filter_key]))
        
        # Score filter
        if filters.get('min_score', 0) > 0:
            masks.append(df['master_score'] >= filters['min_score'])
        
        # Pattern filter
        if filters.get('patterns'):
            pattern_regex = '|'.join([re.escape(p) for p in filters['patterns']])
            masks.append(df['patterns'].str.contains(pattern_regex, case=False, na=False, regex=True))
        
        # Trend filter
        if filters.get('trend_range'):
            min_trend, max_trend = filters['trend_range']
            masks.append((df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend))
        
        # Wave filters
        if filters.get('wave_states'):
            masks.append(df['wave_state'].isin(filters['wave_states']))
        
        if filters.get('wave_strength_range') and filters['wave_strength_range'] != (0, 100):
            min_ws, max_ws = filters['wave_strength_range']
            masks.append((df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws))
        
        # Fundamental filters
        if filters.get('min_pe') is not None:
            masks.append((df['pe'] > 0) & (df['pe'] >= filters['min_pe']))
        
        if filters.get('max_pe') is not None:
            masks.append((df['pe'] > 0) & (df['pe'] <= filters['max_pe']))
        
        if filters.get('min_eps_change') is not None:
            masks.append(df['eps_change_pct'] >= filters['min_eps_change'])
        
        if filters.get('require_fundamental_data'):
            masks.append(df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna())
        
        # Apply all masks
        if masks:
            combined_mask = np.logical_and.reduce(masks)
            filtered_df = df[combined_mask].copy()
        else:
            filtered_df = df.copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df

# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """Fast search with intelligent ranking"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with tiered relevance"""
        if not query or df.empty:
            return pd.DataFrame()
        
        query = query.upper().strip()
        
        # Vectorized search
        ticker_exact = df[df['ticker'].str.upper() == query]
        if not ticker_exact.empty:
            return ticker_exact
        
        # Ticker contains
        ticker_mask = df['ticker'].str.upper().str.contains(query, na=False, regex=False)
        
        # Company contains
        company_mask = pd.Series(False, index=df.index)
        if 'company_name' in df.columns:
            company_mask = df['company_name'].str.upper().str.contains(query, na=False, regex=False)
        
        # Combine results
        all_matches = df[ticker_mask | company_mask].copy()
        
        if not all_matches.empty:
            # Calculate relevance scores
            all_matches['relevance'] = 0
            all_matches.loc[all_matches['ticker'].str.upper() == query, 'relevance'] = 100
            all_matches.loc[all_matches['ticker'].str.upper().str.startswith(query), 'relevance'] += 50
            if 'company_name' in all_matches.columns:
                all_matches.loc[all_matches['company_name'].str.upper().str.startswith(query), 'relevance'] += 30
            
            # Sort by relevance and score
            return all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False]).drop('relevance', axis=1)
        
        return pd.DataFrame()

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Professional data export with multiple formats"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=2.0)
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        """Create comprehensive Excel report"""
        if df.empty:
            raise ValueError("No data to export")
        
        output = BytesIO()
        
        templates = {
            'day_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'vol_ratio_1d_90d',
                           'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d',
                           'volume_score', 'vmi', 'wave_state', 'patterns', 'category', 'industry'],
                'focus': 'Intraday momentum and volume'
            },
            'swing_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score',
                           'breakout_score', 'position_score', 'position_tension',
                           'from_high_pct', 'from_low_pct', 'trend_quality',
                           'momentum_harmony', 'patterns', 'industry', 'wyckoff_phase'],
                'focus': 'Position and breakout setups'
            },
            'investor': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe',
                           'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y',
                           'long_term_strength', 'money_flow_mm', 'category', 'sector',
                           'industry', 'earnings_anomaly_score'],
                'focus': 'Fundamentals and long-term performance'
            },
            'full': {
                'columns': None,
                'focus': 'Complete analysis with all metrics'
            }
        }
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Header format
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#3498db',
                    'font_color': 'white',
                    'border': 1
                })
                
                # 1. Top 100 Stocks
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                
                if template in templates and templates[template]['columns']:
                    export_cols = [col for col in templates[template]['columns'] if col in top_100.columns]
                    top_100_export = top_100[export_cols]
                else:
                    top_100_export = top_100
                
                top_100_export.to_excel(writer, sheet_name='Top 100', index=False)
                
                # 2. Market Intelligence
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                intel_data = pd.DataFrame([
                    {'Metric': 'Market Regime', 'Value': regime},
                    {'Metric': 'Market Breadth', 'Value': f"{regime_metrics.get('breadth', 0):.1%}"},
                    {'Metric': 'Avg Vol Ratio', 'Value': f"{regime_metrics.get('avg_vol_ratio', 1):.2f}x"}
                ])
                intel_data.to_excel(writer, sheet_name='Market Intelligence', index=False)
                
                # 3. Sector Rotation
                sector_rotation = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rotation.empty:
                    sector_rotation.to_excel(writer, sheet_name='Sector Rotation')
                
                # 4. Pattern Analysis
                pattern_counts = {}
                for patterns in df['patterns'].dropna():
                    if patterns:
                        for p in patterns.split(' | '):
                            pattern_counts[p] = pattern_counts.get(p, 0) + 1
                
                if pattern_counts:
                    pattern_df = pd.DataFrame(
                        list(pattern_counts.items()),
                        columns=['Pattern', 'Count']
                    ).sort_values('Count', ascending=False)
                    pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
                
                # 5. Extreme Opportunities
                extreme = df[
                    (df['master_score'] >= 85) &
                    (df.get('vol_ratio_1d_90d', 1) > 2) &
                    (df.get('momentum_harmony', 0) >= 3)
                ].head(50)
                
                if len(extreme) > 0:
                    extreme[['ticker', 'company_name', 'master_score', 'momentum_score',
                            'acceleration_score', 'wave_state', 'patterns']].to_excel(
                        writer, sheet_name='Extreme Opportunities', index=False
                    )
                
                logger.info(f"Excel report created with {len(writer.sheets)} sheets")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create clean CSV export"""
        if df.empty:
            return ""
        
        # Select important columns
        export_cols = [
            'rank', 'ticker', 'company_name', 'master_score',
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score',
            'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
            'from_low_pct', 'from_high_pct',
            'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
            'vol_ratio_1d_90d', 'vmi', 'money_flow_mm',
            'position_tension', 'momentum_harmony', 'wave_state', 'patterns',
            'category', 'sector', 'industry',
            'wyckoff_phase', 'institutional_activity', 'stock_dna',
            'momentum_state', 'earnings_signal', 'rrg_quadrant'
        ]
        
        available_cols = [col for col in export_cols if col in df.columns]
        export_df = df[available_cols].copy()
        
        return export_df.to_csv(index=False)

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create professional visualizations"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        scores_to_plot = [
            ('position_score', 'Position', '#3498db'),
            ('volume_score', 'Volume', '#e74c3c'),
            ('momentum_score', 'Momentum', '#2ecc71'),
            ('acceleration_score', 'Acceleration', '#f39c12'),
            ('breakout_score', 'Breakout', '#9b59b6'),
            ('rvol_score', 'RVOL', '#e67e22')
        ]
        
        for score_col, label, color in scores_to_plot:
            if score_col in df.columns:
                fig.add_trace(go.Box(
                    y=df[score_col].dropna(),
                    name=label,
                    marker_color=color,
                    boxpoints='outliers'
                ))
        
        fig.update_layout(
            title="Score Component Distribution",
            yaxis_title="Score (0-100)",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, n: int = 10) -> go.Figure:
        """Create momentum acceleration profiles"""
        fig = go.Figure()
        
        plot_df = df.dropna(subset=['ret_1d', 'ret_7d', 'ret_30d'], how='any')
        
        if plot_df.empty:
            fig.add_annotation(
                text="No complete return data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        accel_df = plot_df.nlargest(min(n, len(plot_df)), 'acceleration_score')
        
        for _, stock in accel_df.iterrows():
            x_points = ['Start', '30D', '7D', 'Today']
            y_points = [0, stock['ret_30d'], stock['ret_7d'], stock['ret_1d']]
            
            accel_score = stock.get('acceleration_score', 0)
            if accel_score >= 85:
                line_style = dict(width=3, dash='solid')
                marker_style = dict(size=10, symbol='star')
            elif accel_score >= 70:
                line_style = dict(width=2, dash='solid')
                marker_style = dict(size=8)
            else:
                line_style = dict(width=2, dash='dot')
                marker_style = dict(size=6)
            
            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines+markers',
                name=f"{stock['ticker']} ({accel_score:.0f})",
                line=line_style,
                marker=marker_style
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f"Acceleration Profiles - Top {len(accel_df)} Momentum Builders",
            xaxis_title="Time Frame",
            yaxis_title="Return %",
            height=400,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None,
                          help_text: Optional[str] = None) -> None:
        """Render a styled metric card"""
        metric_key = label.lower().replace(' ', '_')
        if not help_text and metric_key in CONFIG.METRIC_TOOLTIPS:
            help_text = CONFIG.METRIC_TOOLTIPS[metric_key]
        
        if help_text:
            st.metric(label, value, delta, help=help_text)
        else:
            st.metric(label, value, delta)
    
    @staticmethod
    def render_summary_section(df: pd.DataFrame) -> None:
        """Render executive summary dashboard"""
        if df.empty:
            st.warning("No data available for summary")
            return
        
        # Market Pulse
        st.markdown("### 📊 Market Pulse")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'ret_1d' in df.columns:
                advancing = (df['ret_1d'] > 0).sum()
                declining = (df['ret_1d'] < 0).sum()
                ad_ratio = advancing / declining if declining > 0 else float('inf')
                
                if ad_ratio == float('inf'):
                    ad_display = "∞"
                    ad_emoji = "🔥🔥"
                elif ad_ratio > 2:
                    ad_display = f"{ad_ratio:.2f}"
                    ad_emoji = "🔥"
                elif ad_ratio > 1:
                    ad_display = f"{ad_ratio:.2f}"
                    ad_emoji = "📈"
                else:
                    ad_display = f"{ad_ratio:.2f}"
                    ad_emoji = "📉"
                
                UIComponents.render_metric_card(
                    "A/D Ratio",
                    f"{ad_emoji} {ad_display}",
                    f"{advancing}/{declining}"
                )
        
        with col2:
            if 'momentum_score' in df.columns:
                high_momentum = (df['momentum_score'] >= 70).sum()
                momentum_pct = (high_momentum / len(df) * 100)
                UIComponents.render_metric_card(
                    "Momentum Health",
                    f"{momentum_pct:.0f}%",
                    f"{high_momentum} strong"
                )
        
        with col3:
            if 'vol_ratio_1d_90d' in df.columns:
                avg_vol = df['vol_ratio_1d_90d'].median()
                high_vol = (df['vol_ratio_1d_90d'] > 2).sum()
                
                vol_emoji = "🌊" if avg_vol > 1.5 else "💧" if avg_vol > 1.2 else "🏜️"
                
                UIComponents.render_metric_card(
                    "Volume State",
                    f"{vol_emoji} {avg_vol:.1f}x",
                    f"{high_vol} surges"
                )
        
        with col4:
            # Risk assessment
            risk_factors = 0
            
            if 'from_high_pct' in df.columns:
                overextended = ((df['from_high_pct'] >= 0) & (df['momentum_score'] < 50)).sum()
                if overextended > 20:
                    risk_factors += 1
            
            if 'vol_ratio_1d_90d' in df.columns:
                pump_risk = ((df['vol_ratio_1d_90d'] > 5) & (df['master_score'] < 50)).sum()
                if pump_risk > 10:
                    risk_factors += 1
            
            if 'trend_quality' in df.columns:
                downtrends = (df['trend_quality'] < 40).sum()
                if downtrends > len(df) * 0.3:
                    risk_factors += 1
            
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]
            risk_level = risk_levels[min(risk_factors, 3)]
            
            UIComponents.render_metric_card(
                "Risk Level",
                risk_level,
                f"{risk_factors} factors"
            )
        
        # Today's Opportunities
        st.markdown("### 🎯 Today's Best Opportunities")
        
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        
        with opp_col1:
            # Ready to run stocks
            ready_to_run = df[
                (df.get('momentum_score', 0) >= 70) &
                (df.get('acceleration_score', 0) >= 70) &
                (df.get('vol_ratio_1d_90d', 1) >= 2)
            ].nlargest(5, 'master_score') if all(c in df.columns for c in ['momentum_score', 'acceleration_score']) else pd.DataFrame()
            
            st.markdown("**🚀 Ready to Run**")
            if len(ready_to_run) > 0:
                for _, stock in ready_to_run.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock.get('company_name', 'N/A')[:25]}")
                    st.caption(f"Score: {stock['master_score']:.1f} | Vol: {stock.get('vol_ratio_1d_90d', 1):.1f}x")
            else:
                st.info("No momentum leaders found")
        
        with opp_col2:
            # Hidden gems
            hidden_gems = df[df['patterns'].str.contains('HIDDEN GEM', na=False)].nlargest(5, 'master_score') if 'patterns' in df.columns else pd.DataFrame()
            
            st.markdown("**💎 Hidden Gems**")
            if len(hidden_gems) > 0:
                for _, stock in hidden_gems.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock.get('company_name', 'N/A')[:25]}")
                    st.caption(f"Cat %ile: {stock.get('category_percentile', 0):.0f}")
            else:
                st.info("No hidden gems today")
        
        with opp_col3:
            # Extreme opportunities
            extreme_opps = df[df['patterns'].str.contains('EXTREME OPPORTUNITY', na=False)].nlargest(5, 'master_score') if 'patterns' in df.columns else pd.DataFrame()
            
            st.markdown("**🏆 Extreme Opportunities**")
            if len(extreme_opps) > 0:
                for _, stock in extreme_opps.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock.get('company_name', 'N/A')[:25]}")
                    st.caption(f"Score: {stock['master_score']:.1f} | {stock.get('wave_state', 'N/A')}")
            else:
                st.info("No extreme opportunities detected")

# ============================================
# DATA LOADING AND CACHING
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL, persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None,
                         sheet_id: str = None, gid: str = None,
                         data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """Load and process data with intelligent caching"""
    start_time = time.perf_counter()
    metadata = {
        'source_type': source_type,
        'data_version': data_version,
        'processing_start': datetime.now(timezone.utc),
        'errors': [],
        'warnings': []
    }
    
    try:
        # Load data
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            try:
                df = pd.read_csv(file_data, low_memory=False)
                metadata['source'] = "User Upload"
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        file_data.seek(0)
                        df = pd.read_csv(file_data, low_memory=False, encoding=encoding)
                        metadata['warnings'].append(f"Used {encoding} encoding")
                        break
                    except:
                        continue
                else:
                    raise ValueError("Unable to decode CSV file")
        else:
            # Load from Google Sheets
            if not sheet_id:
                raise ValueError("Please enter a Google Sheets ID")
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            logger.info(f"Loading data from Google Sheets ID: {sheet_id}")
            
            df = pd.read_csv(csv_url, low_memory=False)
            metadata['source'] = "Google Sheets"
        
        # Validate initial data
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Process data through pipeline
        df = DataProcessor.process_dataframe(df, metadata)
        df = RankingEngine.calculate_all_scores(df)
        df = PatternDetector.detect_all_patterns_optimized(df)
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        # Final validation
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid:
            raise ValueError(validation_msg)
        
        timestamp = datetime.now(timezone.utc)
        
        # Store in session as backup
        st.session_state.last_good_data = (df.copy(), timestamp, metadata)
        
        # Performance metrics
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        # Garbage collection
        if processing_time > 2:
            gc.collect()
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        metadata['errors'].append(str(e))
        
        # Try to use cached data
        if 'last_good_data' in st.session_state:
            logger.info("Using cached data as fallback")
            df, timestamp, old_metadata = st.session_state.last_good_data
            metadata['warnings'].append("Using cached data due to load failure")
            metadata['cache_used'] = True
            return df, timestamp, metadata
        
        raise

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application - Wave Detection Ultimate Beast Edition"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 4.0 Beast",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    SessionStateManager.initialize()
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Beast Mode CSS */
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        overflow-wrap: break-word;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 5px;
    }
    div.stButton > button {
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    @media (max-width: 768px) {
        .stDataFrame {font-size: 12px;}
        div[data-testid="metric-container"] {padding: 3%;}
        .main {padding: 0rem 0.5rem;}
    }
    .stDataFrame > div {overflow-x: auto;}
    .stSpinner > div {
        border-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 4.0</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            BEAST EDITION • Institutional-Grade Alpha Generator • Peak Performance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now(timezone.utc)
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                gc.collect()
                st.success("Cache cleared!")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        data_source_col1, data_source_col2 = st.columns(2)
        
        with data_source_col1:
            if st.button("📊 Google Sheets",
                        type="primary" if st.session_state.data_source == "sheet" else "secondary",
                        use_container_width=True):
                st.session_state.data_source = "sheet"
                st.rerun()
        
        with data_source_col2:
            if st.button("📁 Upload CSV",
                        type="primary" if st.session_state.data_source == "upload" else "secondary",
                        use_container_width=True):
                st.session_state.data_source = "upload"
                st.rerun()
        
        uploaded_file = None
        sheet_id = None
        gid = None
        
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type="csv",
                help="Upload your stock data CSV file"
            )
            if uploaded_file is None:
                st.info("Please upload a CSV file to continue")
        else:
            st.markdown("#### 📊 Google Sheets Configuration")
            
            sheet_input = st.text_input(
                "Google Sheets ID or URL",
                value=st.session_state.get('sheet_id', ''),
                placeholder="Enter Sheet ID or full URL",
                help="Example: 1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM"
            )
            
            if sheet_input:
                # Extract ID from URL if needed
                sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_input)
                if sheet_id_match:
                    sheet_id = sheet_id_match.group(1)
                else:
                    sheet_id = sheet_input.strip()
                
                st.session_state.sheet_id = sheet_id
            
            gid = st.text_input(
                "Sheet Tab GID (Optional)",
                value=st.session_state.get('gid', CONFIG.DEFAULT_GID),
                placeholder=f"Default: {CONFIG.DEFAULT_GID}"
            )
            
            if not sheet_id:
                st.warning("Please enter a Google Sheets ID to continue")
        
        # Data quality indicator
        data_quality = st.session_state.get('data_quality', {})
        if data_quality:
            with st.expander("📊 Data Quality", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    completeness = data_quality.get('completeness', 0)
                    emoji = "🟢" if completeness > 80 else "🟡" if completeness > 60 else "🔴"
                    st.metric("Completeness", f"{emoji} {completeness:.1f}%")
                    st.metric("Total Stocks", f"{data_quality.get('total_rows', 0):,}")
                
                with col2:
                    if 'timestamp' in data_quality:
                        age = datetime.now(timezone.utc) - data_quality['timestamp']
                        hours = age.total_seconds() / 3600
                        freshness = "🟢 Fresh" if hours < 1 else "🟡 Recent" if hours < 24 else "🔴 Stale"
                        st.metric("Data Age", freshness)
                    
                    duplicates = data_quality.get('duplicate_tickers', 0)
                    if duplicates > 0:
                        st.metric("Duplicates", f"⚠️ {duplicates}")
        
        # Performance metrics
        perf_metrics = st.session_state.get('performance_metrics', {})
        if perf_metrics:
            with st.expander("⚡ Performance"):
                total_time = sum(perf_metrics.values())
                perf_emoji = "🟢" if total_time < 3 else "🟡" if total_time < 5 else "🔴"
                st.metric("Load Time", f"{perf_emoji} {total_time:.1f}s")
                
                # Show slowest operations
                if len(perf_metrics) > 0:
                    slowest = sorted(perf_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
                    for func_name, elapsed in slowest:
                        if elapsed > 0.1:
                            st.caption(f"{func_name}: {elapsed:.2f}s")
        
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        # Count active filters
        active_filter_count = sum([
            len(st.session_state.get('category_filter', [])) > 0,
            len(st.session_state.get('sector_filter', [])) > 0,
            len(st.session_state.get('industry_filter', [])) > 0,
            st.session_state.get('min_score', 0) > 0,
            len(st.session_state.get('patterns', [])) > 0,
            st.session_state.get('trend_filter', 'All Trends') != 'All Trends',
            st.session_state.get('quick_filter_applied', False),
            len(st.session_state.get('wave_states_filter', [])) > 0,
            st.session_state.get('wave_strength_range_slider', (0, 100)) != (0, 100)
        ])
        
        if active_filter_count > 0:
            st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count != 1 else ''} active**")
        
        if st.button("🗑️ Clear All Filters",
                    use_container_width=True,
                    type="primary" if active_filter_count > 0 else "secondary"):
            SessionStateManager.clear_filters()
            st.success("✅ All filters cleared!")
            st.rerun()
        
        st.markdown("---")
        show_debug = st.checkbox("🐛 Show Debug Info",
                                value=st.session_state.get('show_debug', False))
        st.session_state.show_debug = show_debug
    
    # Main content
    try:
        # Check data source
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        if st.session_state.data_source == "sheet" and not sheet_id:
            st.warning("Please enter a Google Sheets ID to continue")
            st.stop()
        
        # Load and process data
        with st.spinner("📥 Loading and processing data with BEAST algorithms..."):
            if st.session_state.data_source == "upload" and uploaded_file is not None:
                ranked_df, data_timestamp, metadata = load_and_process_data(
                    "upload", file_data=uploaded_file
                )
            else:
                ranked_df, data_timestamp, metadata = load_and_process_data(
                    "sheet", sheet_id=sheet_id, gid=gid
                )
            
            st.session_state.ranked_df = ranked_df
            st.session_state.data_timestamp = data_timestamp
            st.session_state.last_refresh = datetime.now(timezone.utc)
            
            # Show warnings/errors
            if metadata.get('warnings'):
                for warning in metadata['warnings']:
                    st.warning(warning)
            
            if metadata.get('errors'):
                for error in metadata['errors']:
                    st.error(error)
    
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        
        # Try cached data
        if 'last_good_data' in st.session_state:
            ranked_df, data_timestamp, metadata = st.session_state.last_good_data
            st.warning("Failed to load fresh data, using cached version")
        else:
            st.error(f"❌ Error: {str(e)}")
            st.stop()
    
    # Quick Action Buttons
    st.markdown("### ⚡ Quick Actions")
    qa_cols = st.columns(5)
    
    with qa_cols[0]:
        if st.button("📈 Top Gainers", use_container_width=True):
            st.session_state.quick_filter = 'top_gainers'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_cols[1]:
        if st.button("🔥 Volume Surges", use_container_width=True):
            st.session_state.quick_filter = 'volume_surges'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_cols[2]:
        if st.button("🎯 Breakout Ready", use_container_width=True):
            st.session_state.quick_filter = 'breakout_ready'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_cols[3]:
        if st.button("💎 Hidden Gems", use_container_width=True):
            st.session_state.quick_filter = 'hidden_gems'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_cols[4]:
        if st.button("🌊 Show All", use_container_width=True):
            st.session_state.quick_filter = None
            st.session_state.quick_filter_applied = False
            st.rerun()
    
    # Apply quick filters
    quick_filter = st.session_state.get('quick_filter')
    if quick_filter:
        if quick_filter == 'top_gainers':
            ranked_df_display = ranked_df[ranked_df['momentum_score'] >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ 80")
        elif quick_filter == 'volume_surges':
            ranked_df_display = ranked_df[ranked_df.get('vol_ratio_1d_90d', 1) >= 3]
            st.info(f"Showing {len(ranked_df_display)} stocks with volume ratio ≥ 3x")
        elif quick_filter == 'breakout_ready':
            ranked_df_display = ranked_df[ranked_df['breakout_score'] >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with breakout score ≥ 80")
        elif quick_filter == 'hidden_gems':
            ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('HIDDEN GEM', na=False)]
            st.info(f"Showing {len(ranked_df_display)} hidden gem stocks")
    else:
        ranked_df_display = ranked_df
    
    # Sidebar filters
    with st.sidebar:
        filters = SessionStateManager.build_filter_dict()
        
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if st.session_state.user_preferences['display_mode'] == 'Technical' else 1,
            help="Technical: Pure momentum | Hybrid: Adds PE & EPS data"
        )
        
        st.session_state.user_preferences['display_mode'] = display_mode
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        # Category filter
        categories = FilterEngine.get_filter_options(ranked_df_display, 'category', filters) if hasattr(FilterEngine, 'get_filter_options') else ranked_df_display['category'].dropna().unique().tolist()
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=categories,
            default=st.session_state.get('category_filter', []),
            key="category_filter"
        )
        
        # Sector filter
        sectors = ranked_df_display['sector'].dropna().unique().tolist() if 'sector' in ranked_df_display.columns else []
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=st.session_state.get('sector_filter', []),
            key="sector_filter"
        )
        
        # Industry filter
        industries = ranked_df_display['industry'].dropna().unique().tolist() if 'industry' in ranked_df_display.columns else []
        selected_industries = st.multiselect(
            "Industry",
            options=industries,
            default=st.session_state.get('industry_filter', []),
            key="industry_filter"
        )
        
        # Score filter
        min_score = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=st.session_state.get('min_score', 0),
            step=5,
            key="min_score"
        )
        
        # Pattern filter
        all_patterns = set()
        for patterns in ranked_df_display['patterns'].dropna():
            if patterns:
                all_patterns.update(patterns.split(' | '))
        
        if all_patterns:
            selected_patterns = st.multiselect(
                "Patterns",
                options=sorted(all_patterns),
                default=st.session_state.get('patterns', []),
                key="patterns"
            )
        
        # Trend filter
        st.markdown("#### 📈 Trend Strength")
        trend_filter = st.selectbox(
            "Trend Quality",
            options=[
                "All Trends",
                "🔥 Strong Uptrend (80+)",
                "✅ Good Uptrend (60-79)",
                "➡️ Neutral Trend (40-59)",
                "⚠️ Weak/Downtrend (<40)"
            ],
            index=0,
            key="trend_filter"
        )
        
        # Wave filters
        st.markdown("#### 🌊 Wave Filters")
        wave_states_options = ranked_df_display['wave_state'].dropna().unique().tolist() if 'wave_state' in ranked_df_display.columns else []
        wave_states_filter = st.multiselect(
            "Wave State",
            options=wave_states_options,
            default=st.session_state.get('wave_states_filter', []),
            key="wave_states_filter"
        )
        
        # Wave strength slider
        if 'overall_wave_strength' in ranked_df_display.columns:
            wave_strength_range = st.slider(
                "Overall Wave Strength",
                min_value=0,
                max_value=100,
                value=st.session_state.get('wave_strength_range_slider', (0, 100)),
                key="wave_strength_range_slider"
            )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_pe = st.text_input(
                        "Min PE Ratio",
                        value=st.session_state.get('min_pe', ""),
                        key="min_pe"
                    )
                
                with col2:
                    max_pe = st.text_input(
                        "Max PE Ratio",
                        value=st.session_state.get('max_pe', ""),
                        key="max_pe"
                    )
                
                require_fundamental_data = st.checkbox(
                    "Only show stocks with PE and EPS data",
                    value=st.session_state.get('require_fundamental_data', False),
                    key="require_fundamental_data"
                )
    
    # Apply filters
    filters = SessionStateManager.build_filter_dict()
    filtered_df = FilterEngine.apply_filters(ranked_df_display, filters)
    filtered_df = filtered_df.sort_values('rank')
    
    # Show debug info
    if show_debug:
        with st.sidebar.expander("🐛 Debug Info", expanded=True):
            st.write("**Active Filters:**")
            for key, value in filters.items():
                if value:
                    st.write(f"• {key}: {value}")
            
            st.write(f"\n**Filter Result:**")
            st.write(f"Before: {len(ranked_df)} stocks")
            st.write(f"After: {len(filtered_df)} stocks")
    
    # Filter status
    if active_filter_count > 0 or st.session_state.get('quick_filter_applied'):
        filter_status_col1, filter_status_col2 = st.columns([5, 1])
        with filter_status_col1:
            if quick_filter:
                quick_filter_names = {
                    'top_gainers': '📈 Top Gainers',
                    'volume_surges': '🔥 Volume Surges',
                    'breakout_ready': '🎯 Breakout Ready',
                    'hidden_gems': '💎 Hidden Gems'
                }
                filter_display = quick_filter_names.get(quick_filter, 'Filtered')
                st.info(f"**Viewing:** {filter_display} | **{len(filtered_df):,} stocks** shown")
        
        with filter_status_col2:
            if st.button("Clear Filters", type="secondary"):
                SessionStateManager.clear_filters()
                st.rerun()
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = len(filtered_df)
        total_original = len(ranked_df)
        pct_of_all = (total_stocks/total_original*100) if total_original > 0 else 0
        UIComponents.render_metric_card(
            "Total Stocks",
            f"{total_stocks:,}",
            f"{pct_of_all:.0f}% of {total_original:,}"
        )
    
    with col2:
        if 'master_score' in filtered_df.columns:
            avg_score = filtered_df['master_score'].mean()
            std_score = filtered_df['master_score'].std()
            UIComponents.render_metric_card(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={std_score:.1f}"
            )
    
    with col3:
        if 'vol_ratio_1d_90d' in filtered_df.columns:
            high_vol = (filtered_df['vol_ratio_1d_90d'] > 2).sum()
            UIComponents.render_metric_card("High Volume", f"{high_vol}")
    
    with col4:
        if 'momentum_state' in filtered_df.columns:
            explosive = (filtered_df['momentum_state'] == 'EXPLOSIVE').sum()
            UIComponents.render_metric_card("Explosive", f"{explosive}")
    
    with col5:
        if 'wyckoff_phase' in filtered_df.columns:
            markup = (filtered_df['wyckoff_phase'] == 'MARKUP').sum()
            UIComponents.render_metric_card("Markup Phase", f"{markup}")
    
    with col6:
        if 'patterns' in filtered_df.columns:
            with_patterns = (filtered_df['patterns'] != '').sum()
            UIComponents.render_metric_card("With Patterns", f"{with_patterns}")
    
    # Main tabs
    tabs = st.tabs([
        "📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis",
        "🔍 Search", "📥 Export", "ℹ️ About"
    ])
    
    # Tab 1: Summary
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        if not filtered_df.empty:
            UIComponents.render_summary_section(filtered_df)
            
            st.markdown("---")
            st.markdown("#### 💾 Quick Downloads")
            
            download_cols = st.columns(3)
            with download_cols[0]:
                csv_filtered = ExportEngine.create_csv_export(filtered_df)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_filtered,
                    file_name=f"wave_detection_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with download_cols[1]:
                top_100 = filtered_df.nlargest(100, 'master_score')
                csv_top100 = ExportEngine.create_csv_export(top_100)
                st.download_button(
                    label="📥 Download Top 100 (CSV)",
                    data=csv_top100,
                    file_name=f"wave_detection_top100_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with download_cols[2]:
                pattern_stocks = filtered_df[filtered_df['patterns'] != '']
                if len(pattern_stocks) > 0:
                    csv_patterns = ExportEngine.create_csv_export(pattern_stocks)
                    st.download_button(
                        label="📥 Download Pattern Stocks (CSV)",
                        data=csv_patterns,
                        file_name=f"wave_detection_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("No data available for summary")
    
    # Tab 2: Rankings
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        
        col1, col2 = st.columns([2, 4])
        with col1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n'])
            )
            st.session_state.user_preferences['default_top_n'] = display_count
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=['Rank', 'Master Score', 'Volume Ratio', 'Momentum', 'Money Flow']
            )
        
        # Get display data
        display_df = filtered_df.head(display_count).copy()
        
        # Apply sorting
        if sort_by == 'Master Score':
            display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'Volume Ratio':
            display_df = display_df.sort_values('vol_ratio_1d_90d', ascending=False)
        elif sort_by == 'Momentum':
            display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Money Flow':
            display_df = display_df.sort_values('money_flow_mm', ascending=False)
        
        if not display_df.empty:
            # Prepare display columns
            display_cols = {
                'rank': 'Rank',
                'ticker': 'Ticker',
                'company_name': 'Company',
                'master_score': 'Score',
                'wave_state': 'Wave',
                'price': 'Price',
                'from_low_pct': 'From Low',
                'ret_30d': '30D Ret',
                'vol_ratio_1d_90d': 'Vol Ratio',
                'vmi': 'VMI',
                'patterns': 'Patterns',
                'category': 'Category'
            }
            
            if show_fundamentals:
                if 'pe' in display_df.columns:
                    display_cols['pe'] = 'PE'
                if 'eps_change_pct' in display_df.columns:
                    display_cols['eps_change_pct'] = 'EPS Δ%'
            
            # Add intelligent columns
            if 'wyckoff_phase' in display_df.columns:
                display_cols['wyckoff_phase'] = 'Wyckoff'
            if 'momentum_state' in display_df.columns:
                display_cols['momentum_state'] = 'Momentum'
            if 'stock_dna' in display_df.columns:
                display_cols['stock_dna'] = 'DNA'
            
            # Format display
            available_display_cols = [c for c in display_cols.keys() if c in display_df.columns]
            display_formatted = display_df[available_display_cols].copy()
            
            # Format numeric columns
            if 'master_score' in display_formatted.columns:
                display_formatted['master_score'] = display_formatted['master_score'].apply(lambda x: f"{x:.1f}")
            if 'price' in display_formatted.columns:
                display_formatted['price'] = display_formatted['price'].apply(lambda x: f"₹{x:,.0f}")
            if 'from_low_pct' in display_formatted.columns:
                display_formatted['from_low_pct'] = display_formatted['from_low_pct'].apply(lambda x: f"{x:.0f}%")
            if 'ret_30d' in display_formatted.columns:
                display_formatted['ret_30d'] = display_formatted['ret_30d'].apply(lambda x: f"{x:+.1f}%")
            if 'vol_ratio_1d_90d' in display_formatted.columns:
                display_formatted['vol_ratio_1d_90d'] = display_formatted['vol_ratio_1d_90d'].apply(lambda x: f"{x:.1f}x")
            if 'vmi' in display_formatted.columns:
                display_formatted['vmi'] = display_formatted['vmi'].apply(lambda x: f"{x:.2f}")
            
            display_formatted.columns = [display_cols[c] for c in available_display_cols]
            
            st.dataframe(
                display_formatted,
                use_container_width=True,
                height=min(600, len(display_formatted) * 35 + 50),
                hide_index=True
            )
            
            # Quick statistics
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown("**Score Distribution**")
                    st.text(f"Max: {filtered_df['master_score'].max():.1f}")
                    st.text(f"Min: {filtered_df['master_score'].min():.1f}")
                    st.text(f"Mean: {filtered_df['master_score'].mean():.1f}")
                    st.text(f"Median: {filtered_df['master_score'].median():.1f}")
                
                with stat_cols[1]:
                    st.markdown("**Returns (30D)**")
                    if 'ret_30d' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['ret_30d'].max():.1f}%")
                        st.text(f"Min: {filtered_df['ret_30d'].min():.1f}%")
                        st.text(f"Positive: {(filtered_df['ret_30d'] > 0).sum()}")
                
                with stat_cols[2]:
                    st.markdown("**Volume Activity**")
                    if 'vol_ratio_1d_90d' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['vol_ratio_1d_90d'].max():.1f}x")
                        st.text(f"Avg: {filtered_df['vol_ratio_1d_90d'].mean():.1f}x")
                        st.text(f">2x: {(filtered_df['vol_ratio_1d_90d'] > 2).sum()}")
                
                with stat_cols[3]:
                    st.markdown("**Intelligent Metrics**")
                    if 'wyckoff_phase' in filtered_df.columns:
                        phases = filtered_df['wyckoff_phase'].value_counts()
                        for phase, count in phases.head(3).items():
                            st.text(f"{phase}: {count}")
        else:
            st.warning("No stocks match the selected filters")
    
    # Tab 3: Wave Radar
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection")
        
        # Wave Radar Controls
        radar_cols = st.columns(3)
        
        with radar_cols[0]:
            wave_timeframe = st.selectbox(
                "Detection Timeframe",
                options=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"],
                key="wave_timeframe_select"
            )
        
        with radar_cols[1]:
            sensitivity = st.select_slider(
                "Sensitivity",
                options=["Conservative", "Balanced", "Aggressive"],
                value="Balanced",
                key="wave_sensitivity"
            )
        
        with radar_cols[2]:
            show_market_regime = st.checkbox(
                "Show Market Regime",
                value=True,
                key="show_market_regime"
            )
        
        # Filter based on timeframe
        wave_filtered_df = filtered_df.copy()
        
        if wave_timeframe != "All Waves":
            if wave_timeframe == "Intraday Surge":
                wave_filtered_df = wave_filtered_df[
                    (wave_filtered_df.get('vol_ratio_1d_90d', 1) >= 2.5) &
                    (wave_filtered_df.get('ret_1d', 0) > 2)
                ]
            elif wave_timeframe == "3-Day Buildup":
                wave_filtered_df = wave_filtered_df[
                    (wave_filtered_df.get('ret_3d', 0) > 5) &
                    (wave_filtered_df.get('vol_ratio_7d_90d', 1) > 1.5)
                ]
            elif wave_timeframe == "Weekly Breakout":
                wave_filtered_df = wave_filtered_df[
                    (wave_filtered_df.get('ret_7d', 0) > 8) &
                    (wave_filtered_df.get('from_high_pct', -100) > -10)
                ]
            elif wave_timeframe == "Monthly Trend":
                wave_filtered_df = wave_filtered_df[
                    (wave_filtered_df.get('ret_30d', 0) > 15) &
                    (wave_filtered_df.get('from_low_pct', 100) > 30)
                ]
        
        if not wave_filtered_df.empty:
            # Momentum Shifts
            st.markdown("#### 🚀 Momentum Shifts")
            
            momentum_threshold = {'Conservative': 60, 'Balanced': 50, 'Aggressive': 40}[sensitivity]
            accel_threshold = {'Conservative': 70, 'Balanced': 60, 'Aggressive': 50}[sensitivity]
            
            momentum_shifts = wave_filtered_df[
                (wave_filtered_df['momentum_score'] >= momentum_threshold) &
                (wave_filtered_df['acceleration_score'] >= accel_threshold)
            ].head(20)
            
            if len(momentum_shifts) > 0:
                shift_display = momentum_shifts[['ticker', 'company_name', 'master_score',
                                                'momentum_score', 'acceleration_score',
                                                'vol_ratio_1d_90d', 'wave_state', 'category']].copy()
                
                shift_display.columns = ['Ticker', 'Company', 'Score', 'Momentum',
                                        'Acceleration', 'Vol Ratio', 'Wave', 'Category']
                
                st.dataframe(shift_display, use_container_width=True, hide_index=True)
            else:
                st.info("No momentum shifts detected")
            
            # Acceleration Profiles
            st.markdown("#### 📈 Acceleration Profiles")
            fig_accel = Visualizer.create_acceleration_profiles(wave_filtered_df.head(10))
            st.plotly_chart(fig_accel, use_container_width=True)
            
            # Market Regime
            if show_market_regime:
                st.markdown("#### 🎯 Market Regime Analysis")
                regime, metrics = MarketIntelligence.detect_market_regime(wave_filtered_df)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Sector rotation chart
                    sector_rotation = MarketIntelligence.detect_sector_rotation(wave_filtered_df)
                    if not sector_rotation.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=sector_rotation.index[:10],
                            y=sector_rotation['flow_score'][:10],
                            marker_color=['#2ecc71' if s > 60 else '#e74c3c' if s < 40 else '#f39c12'
                                        for s in sector_rotation['flow_score'][:10]]
                        ))
                        fig.update_layout(
                            title=f"Sector Flow - {regime}",
                            xaxis_title="Sector",
                            yaxis_title="Flow Score",
                            height=300,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"**{regime}**")
                    st.metric("Market Breadth", f"{metrics.get('breadth', 0.5):.1%}")
                    st.metric("Avg Vol Ratio", f"{metrics.get('avg_vol_ratio', 1):.2f}x")
                    st.metric("Category Spread", f"{metrics.get('category_spread', 0):.1f}")
        else:
            st.warning("No data available for Wave Radar analysis")
    
    # Tab 4: Analysis
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Pattern frequency
                pattern_counts = {}
                for patterns in filtered_df['patterns'].dropna():
                    if patterns:
                        for p in patterns.split(' | '):
                            pattern_counts[p] = pattern_counts.get(p, 0) + 1
                
                if pattern_counts:
                    pattern_df = pd.DataFrame(
                        list(pattern_counts.items()),
                        columns=['Pattern', 'Count']
                    ).sort_values('Count', ascending=True).tail(15)
                    
                    fig_patterns = go.Figure([
                        go.Bar(
                            x=pattern_df['Count'],
                            y=pattern_df['Pattern'],
                            orientation='h',
                            marker_color='#3498db'
                        )
                    ])
                    
                    fig_patterns.update_layout(
                        title="Pattern Frequency",
                        xaxis_title="Count",
                        yaxis_title="Pattern",
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Sector Performance
            st.markdown("---")
            st.markdown("#### 🏢 Sector Performance")
            
            sector_rotation = MarketIntelligence.detect_sector_rotation(filtered_df)
            if not sector_rotation.empty:
                st.dataframe(
                    sector_rotation[['flow_score', 'master_score_mean', 'momentum_score_mean',
                                    'volume_score_mean', 'analyzed_stocks', 'total_stocks']].head(15),
                    use_container_width=True
                )
            else:
                st.info("No sector data available")
        else:
            st.info("No data available for analysis")
    
    # Tab 5: Search
    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search stocks",
                placeholder="Enter ticker or company name...",
                key="search_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        
        if search_query or search_clicked:
            search_results = SearchEngine.search_stocks(filtered_df, search_query)
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                
                for idx, stock in search_results.iterrows():
                    with st.expander(
                        f"📊 {stock['ticker']} - {stock.get('company_name', 'N/A')} "
                        f"(Rank #{int(stock['rank'])})",
                        expanded=True
                    ):
                        # Display metrics
                        metric_cols = st.columns(6)
                        
                        with metric_cols[0]:
                            UIComponents.render_metric_card(
                                "Master Score",
                                f"{stock['master_score']:.1f}",
                                f"Rank #{int(stock['rank'])}"
                            )
                        
                        with metric_cols[1]:
                            UIComponents.render_metric_card(
                                "Price",
                                f"₹{stock['price']:,.0f}",
                                f"{stock.get('ret_1d', 0):+.1f}%"
                            )
                        
                        with metric_cols[2]:
                            UIComponents.render_metric_card(
                                "From Low",
                                f"{stock.get('from_low_pct', 0):.0f}%"
                            )
                        
                        with metric_cols[3]:
                            UIComponents.render_metric_card(
                                "30D Return",
                                f"{stock.get('ret_30d', 0):+.1f}%"
                            )
                        
                        with metric_cols[4]:
                            UIComponents.render_metric_card(
                                "Vol Ratio",
                                f"{stock.get('vol_ratio_1d_90d', 1):.1f}x"
                            )
                        
                        with metric_cols[5]:
                            UIComponents.render_metric_card(
                                "Wave State",
                                stock.get('wave_state', 'N/A')
                            )
                        
                        # Show patterns
                        if stock.get('patterns'):
                            st.markdown(f"**🎯 Patterns:** {stock['patterns']}")
                        
                        # Show intelligent metrics
                        if 'wyckoff_phase' in stock:
                            st.markdown(f"**Wyckoff Phase:** {stock['wyckoff_phase']}")
                        if 'stock_dna' in stock:
                            st.markdown(f"**Stock DNA:** {stock['stock_dna']}")
                        if 'momentum_state' in stock:
                            st.markdown(f"**Momentum State:** {stock['momentum_state']}")
            else:
                st.warning("No stocks found matching your search")
    
    # Tab 6: Export
    with tabs[5]:
        st.markdown("### 📥 Export Data")
        
        export_template = st.radio(
            "Choose export template:",
            options=["Full Analysis", "Day Trader Focus", "Swing Trader Focus", "Investor Focus"]
        )
        
        template_map = {
            "Full Analysis": "full",
            "Day Trader Focus": "day_trader",
            "Swing Trader Focus": "swing_trader",
            "Investor Focus": "investor"
        }
        
        selected_template = template_map[export_template]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Excel Report")
            if st.button("Generate Excel Report", type="primary", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export")
                else:
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_file = ExportEngine.create_excel_report(
                                filtered_df, template=selected_template
                            )
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_file,
                                file_name=f"wave_detection_beast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("Excel report generated!")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        with col2:
            st.markdown("#### 📄 CSV Export")
            if st.button("Generate CSV Export", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export")
                else:
                    csv_data = ExportEngine.create_csv_export(filtered_df)
                    
                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv_data,
                        file_name=f"wave_detection_beast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    st.success("CSV export generated!")
    
    # Tab 7: About
    with tabs[6]:
        st.markdown("""
        ### ℹ️ Wave Detection Ultimate 4.0 - BEAST Edition
        
        #### 🌊 The Most Advanced Stock Ranking System
        
        This is the **BEAST EDITION** - a professional-grade, institutional-quality stock ranking system
        that combines:
        
        **Core Features:**
        - **Master Score 3.0** with intelligent adjustments
        - **25+ Pattern Detection** including extreme opportunities
        - **Institutional-Grade Metrics** from hedge fund strategies
        - **Market Regime Adaptation** for different conditions
        - **Wyckoff Phase Detection** for cycle analysis
        - **Stock DNA Profiling** for behavioral patterns
        - **Momentum Decay Analysis** with predictions
        - **Fibonacci Level Detection** for support/resistance
        - **Volatility Squeeze Detection** for breakout timing
        - **Earnings Anomaly Scoring** for value opportunities
        
        **Performance:**
        - Sub-2 second processing for 1500+ stocks
        - Fully vectorized calculations with NumPy
        - Intelligent caching with versioning
        - Numba JIT compilation for critical paths
        - Memory-efficient data structures
        
        **Intelligence Features:**
        - Adaptive scoring based on market regime
        - Predictive momentum decay calculations
        - Institutional footprint detection
        - Relative rotation graph positioning
        - Smart money flow analysis
        - Dynamic sector rotation detection
        
        **Version:** 4.0.0-BEAST
        **Status:** PRODUCTION READY
        **Architecture:** Hedge Fund Grade
        
        ---
        
        **Created for professional traders who demand the best.**
        """)
        
        # System stats
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            UIComponents.render_metric_card(
                "Total Stocks",
                f"{len(ranked_df):,}" if 'ranked_df' in locals() else "0"
            )
        
        with stats_cols[1]:
            UIComponents.render_metric_card(
                "Filtered",
                f"{len(filtered_df):,}" if 'filtered_df' in locals() else "0"
            )
        
        with stats_cols[2]:
            data_quality = st.session_state.get('data_quality', {})
            completeness = data_quality.get('completeness', 0)
            UIComponents.render_metric_card(
                "Data Quality",
                f"{completeness:.1f}%"
            )
        
        with stats_cols[3]:
            cache_time = datetime.now(timezone.utc) - st.session_state.last_refresh
            minutes = int(cache_time.total_seconds() / 60)
            UIComponents.render_metric_card(
                "Cache Age",
                f"{minutes} min"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            🌊 Wave Detection Ultimate 4.0 - BEAST EDITION<br>
            <small>Institutional-Grade Stock Ranking • Peak Performance • Professional Engineering</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================
# APPLICATION ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}")
        logger.error(f"Application crashed: {str(e)}", exc_info=True)
        
        if st.button("🔄 Restart Application"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📧 Report Issue"):
            st.info("Please take a screenshot and report this error.")
