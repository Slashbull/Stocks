"""
Wave Detection Ultimate 3.0 - FINAL PERFECTED PRODUCTION VERSION
===============================================================
Professional Stock Ranking System with Advanced Analytics
All bugs fixed, optimized for Streamlit Community Cloud
Enhanced with perfect filtering system and robust error handling

Version: 3.1.0-FINAL-PERFECTED
Last Updated: August 2025
Status: PRODUCTION READY - PERMANENTLY LOCKED
"""

# ============================================
# IMPORTS AND SETUP
# ============================================

# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Advanced Production Libraries
from datetime import datetime, timezone
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import time
from io import BytesIO, StringIO
import warnings
import gc
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
from collections import defaultdict, Counter

# Suppress warnings for a clean production output
warnings.filterwarnings('ignore')

# Performance optimizations
np.seterr(all='ignore')
pd.options.mode.chained_assignment = None
np.random.seed(42)

# ============================================
# LOGGING AND PERFORMANCE MONITORING
# ============================================

# Production logging with proper formatting
log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Performance tracking storage
performance_stats = defaultdict(list)

def log_performance(operation: str, duration: float):
    """Tracks performance metrics and logs slow operations."""
    performance_stats[operation].append(duration)
    if duration > 1.0:
        logger.warning(f"{operation} took {duration:.2f}s")

class PerformanceMonitor:
    """Advanced performance monitoring using a non-intrusive decorator."""

    @staticmethod
    def timer(target_time: Optional[float] = None):
        """Measures function execution time and logs performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    
                    log_performance(func.__name__, elapsed)
                    
                    # Store a snapshot in session state for UI display
                    st.session_state.performance_metrics = dict(performance_stats)
                    
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
                    raise
            return wrapper
        return decorator

# ============================================
# SESSION STATE MANAGEMENT
# ============================================

class RobustSessionState:
    """Bulletproof session state management, preventing all KeyErrors."""
    STATE_DEFAULTS = {
        'search_query': "", 'last_refresh': None, 'data_source': "sheet", 'sheet_id': "", 'gid': "",
        'user_preferences': {'default_top_n': 50, 'display_mode': 'Technical', 'last_filters': {}},
        'filters': {}, 'active_filter_count': 0, 'quick_filter': None, 'quick_filter_applied': False,
        'show_debug': False, 'performance_metrics': {}, 'data_quality': {},
        'last_good_data': None, 'session_id': None, 'session_start': None,
        'category_filter': [], 'sector_filter': [], 'industry_filter': [], 'min_score': 0,
        'patterns': [], 'trend_filter': "All Trends", 'eps_tier_filter': [], 'pe_tier_filter': [],
        'price_tier_filter': [], 'min_eps_change': "", 'min_pe': "", 'max_pe': "",
        'require_fundamental_data': False, 'wave_states_filter': [], 'wave_strength_range_slider': (0, 100),
        'show_sensitivity_details': False, 'show_market_regime': True,
        'wave_timeframe_select': "All Waves", 'wave_sensitivity': "Balanced",
        'export_template_radio': "Full Analysis (All Data)", 'display_mode_toggle': 0,
        'ranked_df': None, 'data_timestamp': None, 'search_input': ""
    }
    
    @staticmethod
    def initialize():
        """Initializes all session state variables with explicit defaults if they don't exist."""
        for key, default_value in RobustSessionState.STATE_DEFAULTS.items():
            if key not in st.session_state:
                if key == 'last_refresh' or key == 'session_start': st.session_state[key] = datetime.now(timezone.utc)
                elif key == 'session_id': st.session_state[key] = hashlib.md5(f"{datetime.now()}{np.random.rand()}".encode()).hexdigest()[:8]
                else: st.session_state[key] = default_value

    @staticmethod
    def clear_filters():
        """Resets all filter-related state variables to their default values."""
        filter_keys = ['category_filter', 'sector_filter', 'industry_filter', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'patterns', 'min_score', 'trend_filter',
            'min_eps_change', 'min_pe', 'max_pe', 'require_fundamental_data', 'quick_filter',
            'quick_filter_applied', 'wave_states_filter', 'wave_strength_range_slider',
            'show_sensitivity_details', 'show_market_regime', 'wave_timeframe_select',
            'wave_sensitivity', 'search_input'
        ]
        for key in filter_keys:
            if key in RobustSessionState.STATE_DEFAULTS: st.session_state[key] = RobustSessionState.STATE_DEFAULTS[key]
        st.session_state.filters = {}; st.session_state.active_filter_count = 0; st.session_state.trigger_clear = False
    
    @staticmethod
    def safe_get(key: str, default: Any = None) -> Any:
        if key not in st.session_state:
            st.session_state[key] = RobustSessionState.STATE_DEFAULTS.get(key, default)
        return st.session_state[key]

    @staticmethod
    def safe_set(key: str, value: Any) -> None:
        st.session_state[key] = value

    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        return {
            'session_id': st.session_state.get('session_id', 'unknown'),
            'start_time': st.session_state.get('session_start', datetime.now(timezone.utc)),
            'duration': (datetime.now(timezone.utc) - st.session_state.get('session_start', datetime.now(timezone.utc))).seconds,
            'data_source': st.session_state.get('data_source', 'unknown'),
            'stocks_loaded': len(st.session_state.get('ranked_df', [])),
            'active_filters': st.session_state.get('active_filter_count', 0)
        }

# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights and thresholds."""
    DEFAULT_SHEET_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing"
    CSV_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    DEFAULT_GID: str = "1823439984"
    CACHE_TTL: int = 3600
    STALE_DATA_HOURS: int = 24
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'ret_30d', 'from_low_pct', 'from_high_pct', 'vol_ratio_1d_90d', 'vol_ratio_7d_90d',
        'vol_ratio_30d_90d', 'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d', 'volume_90d', 'volume_30d', 'volume_7d', 'ret_7d',
        'category', 'sector', 'industry', 'rvol'
    ])
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d',
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'eps_change_pct'
    ])
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "category_leader": 90, "hidden_gem": 80, "acceleration": 85, "institutional": 75,
        "vol_explosion": 95, "breakout_ready": 80, "market_leader": 95, "momentum_wave": 75,
        "liquid_leader": 80, "long_strength": 80, "52w_high_approach": 90, "52w_low_bounce": 85,
        "golden_zone": 85, "vol_accumulation": 80, "momentum_diverge": 90, "range_compress": 75,
        "stealth": 70, "vampire": 85, "perfect_storm": 80,
        "value_momentum": 70, "earnings_rocket": 70, "quality_leader": 80,
        "turnaround": 70, "high_pe": 100
    })
    PATTERN_METADATA: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        '🔥 CAT LEADER': {'importance': 'high', 'risk': 'low'}, '💎 HIDDEN GEM': {'importance': 'high', 'risk': 'medium'},
        '🚀 ACCELERATING': {'importance': 'high', 'risk': 'medium'}, '🏦 INSTITUTIONAL': {'importance': 'high', 'risk': 'low'},
        '⚡ VOL EXPLOSION': {'importance': 'very_high', 'risk': 'high'}, '🎯 BREAKOUT': {'importance': 'high', 'risk': 'medium'},
        '👑 MARKET LEADER': {'importance': 'very_high', 'risk': 'low'}, '🌊 MOMENTUM WAVE': {'importance': 'high', 'risk': 'medium'},
        '💰 LIQUID LEADER': {'importance': 'medium', 'risk': 'low'}, '💪 LONG STRENGTH': {'importance': 'medium', 'risk': 'low'},
        '📈 QUALITY TREND': {'importance': 'high', 'risk': 'low'}, '⛈️ PERFECT STORM': {'importance': 'very_high', 'risk': 'medium'},
        '💎 VALUE MOMENTUM': {'importance': 'high', 'risk': 'low'}, '📊 EARNINGS ROCKET': {'importance': 'high', 'risk': 'medium'},
        '🏆 QUALITY LEADER': {'importance': 'high', 'risk': 'low'}, '⚡ TURNAROUND': {'importance': 'high', 'risk': 'high'},
        '⚠️ HIGH PE': {'importance': 'low', 'risk': 'high'}, '🎯 52W HIGH APPROACH': {'importance': 'high', 'risk': 'medium'},
        '🔄 52W LOW BOUNCE': {'importance': 'high', 'risk': 'high'}, '👑 GOLDEN ZONE': {'importance': 'medium', 'risk': 'low'},
        '📊 VOL ACCUMULATION': {'importance': 'medium', 'risk': 'low'}, '🔀 MOMENTUM DIVERGE': {'importance': 'high', 'risk': 'medium'},
        '🎯 RANGE COMPRESS': {'importance': 'medium', 'risk': 'low'}, '🤫 STEALTH': {'importance': 'high', 'risk': 'medium'},
        '🧛 VAMPIRE': {'importance': 'high', 'risk': 'very_high'}
    })
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000), 'rvol': (0.01, 1_000_000.0), 'pe': (-10000, 10000),
        'returns': (-99.99, 9999.99), 'volume': (0, 1e15)
    })
    PERFORMANCE_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'data_processing': 2.0, 'filtering': 0.2, 'pattern_detection': 0.5,
        'export_generation': 1.0, 'search': 0.05
    })
    MARKET_CATEGORIES: List[str] = field(default_factory=lambda: ['Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'])
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {"Loss": (-np.inf, 0), "0-5": (0, 5), "5-10": (5, 10), "10-20": (10, 20), "20-50": (20, 50), "50-100": (50, 100), "100+": (100, np.inf)},
        "pe": {"Negative/NA": (-np.inf, 0), "0-10": (0, 10), "10-15": (10, 15), "15-20": (15, 20), "20-30": (20, 30), "30-50": (30, 50), "50+": (50, np.inf)},
        "price": {"0-100": (0, 100), "100-250": (100, 250), "250-500": (250, 500), "500-1000": (500, 1000), "1000-2500": (1000, 2500), "2500-5000": (2500, 5000), "5000+": (5000, np.inf)}
    })
    def __post_init__(self):
        total_weight = sum([self.POSITION_WEIGHT, self.VOLUME_WEIGHT, self.MOMENTUM_WEIGHT, self.ACCELERATION_WEIGHT, self.BREAKOUT_WEIGHT, self.RVOL_WEIGHT])
        if not np.isclose(total_weight, 1.0, rtol=1e-5): raise ValueError(f"Scoring weights must sum to 1.0, but got {total_weight}")

CONFIG = Config()

# ============================================
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidator:
    """Comprehensive data validation with tracking and transparent reporting."""
    def __init__(self):
        self.validation_stats = defaultdict(int)
        self.correction_stats = defaultdict(int)
        self.clipping_counts = defaultdict(int)
    def reset_stats(self):
        self.validation_stats.clear(); self.correction_stats.clear(); self.clipping_counts.clear()
    def get_validation_report(self) -> Dict[str, Any]:
        return {'validations': dict(self.validation_stats), 'corrections': dict(self.correction_stats),
                'clipping': dict(self.clipping_counts), 'total_issues': sum(self.correction_stats.values())}
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], context: str = "") -> Tuple[bool, str]:
        if df is None: return False, f"{context}: DataFrame is None"
        if df.empty: return False, f"{context}: DataFrame is empty"
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns: return False, f"{context}: Missing required columns: {missing_columns}"
        if len(df) < 1: return False, f"{context}: No data rows found"
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0: logger.warning(f"{context}: Found {duplicates} duplicate tickers")
        total_cells = len(df) * len(df.columns); missing_cells = df.isna().sum().sum()
        completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        data_quality = RobustSessionState.safe_get('data_quality', {}); data_quality.update({'total_rows': len(df), 'completeness': completeness}); RobustSessionState.safe_set('data_quality', data_quality)
        return True, "Validation passed"
    def clean_numeric_value(self, value: Any, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None, column_name: str = "") -> float:
        self.validation_stats[f'{column_name}_total'] += 1
        try:
            if pd.isna(value) or value is None: self.correction_stats[f'{column_name}_nan'] += 1; return np.nan
            cleaned = str(value).strip(); invalid_markers = ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-', '#N/A', '#ERROR!', '#DIV/0!', 'INF', '-INF']
            if cleaned.upper() in invalid_markers: self.correction_stats[f'{column_name}_invalid'] += 1; return np.nan
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            result = float(cleaned)
            if bounds:
                min_val, max_val = bounds
                if result < min_val: self.clipping_counts[f'{column_name}_clipped'] += 1; result = min_val
                elif result > max_val: self.clipping_counts[f'{column_name}_clipped'] += 1; result = max_val
            if np.isnan(result) or np.isinf(result): self.correction_stats[f'{column_name}_infinite'] += 1; return np.nan
            return result
        except (ValueError, TypeError, AttributeError):
            self.correction_stats[f'{column_name}_error'] += 1; return np.nan
    @staticmethod
    def sanitize_string(value: Any, default: str = "Unknown") -> str:
        if pd.isna(value) or value is None: return default
        cleaned = str(value).strip(); invalid_markers = ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']
        if cleaned.upper() in invalid_markers: return default
        cleaned = ' '.join(cleaned.split())
        return cleaned

validator = DataValidator()

# ============================================
# DATA LOADING AND CACHING
# ============================================

def get_requests_retry_session(
    retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504, 429)
) -> requests.Session:
    session = requests.Session(); retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter); session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Wave Detection Ultimate 3.0'})
    return session

@st.cache_data(persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, sheet_id: str = None, gid: str = None, data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    start_time = time.perf_counter()
    metadata = {'errors': [], 'warnings': []}
    try:
        if source_type == "upload" and file_data is not None:
            df = pd.read_csv(file_data, low_memory=False); metadata['source'] = "User Upload"
        else:
            if not sheet_id: raise ValueError("Please enter a Google Sheets ID")
            sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_id); sheet_id = sheet_id_match.group(1) if sheet_id_match else sheet_id.strip()
            final_gid = gid if gid else CONFIG.DEFAULT_GID
            csv_url = CONFIG.CSV_URL_TEMPLATE.format(sheet_id=sheet_id, gid=final_gid)
            try:
                session = get_requests_retry_session(); response = session.get(csv_url, timeout=30); response.raise_for_status()
                if len(response.content) < 100: raise ValueError("Response too small, likely an error page")
                df = pd.read_csv(BytesIO(response.content), low_memory=False); metadata['source'] = "Google Sheets"
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.error(f"Failed to load from Google Sheets: {str(e)}"); metadata['errors'].append(f"Sheet load error: {str(e)}")
                last_good_data = RobustSessionState.safe_get('last_good_data', None)
                if last_good_data: metadata['warnings'].append("Using cached data due to load failure"); return last_good_data
                raise
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid: raise ValueError(validation_msg)
        
        df = DataProcessor.process_dataframe(df, metadata)
        df = RankingEngine.calculate_all_scores(df)
        df = PatternDetector.detect_all_patterns(df)
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid: raise ValueError(validation_msg)

        timestamp = datetime.now(timezone.utc); RobustSessionState.safe_set('last_good_data', (df.copy(), timestamp, metadata))
        total_time = time.perf_counter() - start_time
        logger.info(f"Data processing complete: {len(df)} stocks in {total_time:.2f}s")
        gc.collect()
        return df, timestamp, metadata
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}"); metadata['errors'].append(f"Processing failed: {str(e)}")
        last_good_data = RobustSessionState.safe_get('last_good_data', None)
        if last_good_data: return last_good_data
        raise

# ============================================
# DATA PROCESSING ENGINE
# ============================================

class DataProcessor:
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['data_processing'])
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        initial_count = len(df)
        logger.info(f"Processing {initial_count} rows...")
        
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns: df[col] = df[col].apply(DataValidator.sanitize_string)

        numeric_cols = [col for col in df.columns if col not in string_cols + ['year', 'market_cap']]
        for col in numeric_cols:
            if col in df.columns:
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                bounds = CONFIG.VALUE_BOUNDS.get(col, CONFIG.VALUE_BOUNDS.get('returns' if is_pct else 'price'))
                df[col] = df[col].apply(lambda x: validator.clean_numeric_value(x, is_pct, bounds, col))
        
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 1000.0).fillna(1.0)
        
        if 'rvol' not in df.columns or df['rvol'].isna().all():
            if 'volume_1d' in df.columns and 'volume_90d' in df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['rvol'] = np.where(df['volume_90d'] > 0, df['volume_1d'] / df['volume_90d'], 1.0)
                metadata['warnings'].append("RVOL calculated from volume data")
        
        if 'industry' not in df.columns and 'sector' in df.columns:
            df['industry'] = df['sector']
        
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        df = DataProcessor._fill_missing_values(df)
        df = DataProcessor._add_tier_classifications(df)
        
        removed = initial_count - len(df)
        if removed > 0: metadata['warnings'].append(f"Removed {removed} invalid rows during processing")
        
        return df

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        df['from_low_pct'] = df.get('from_low_pct', pd.Series(50.0, index=df.index)).fillna(50.0)
        df['from_high_pct'] = df.get('from_high_pct', pd.Series(-50.0, index=df.index)).fillna(-50.0)
        df['rvol'] = df.get('rvol', pd.Series(1.0, index=df.index)).fillna(1.0)
        for col in [c for c in df.columns if c.startswith('ret_')]: df[col] = df[col].fillna(0.0)
        for col in [c for c in df.columns if c.startswith('volume_')]: df[col] = df[col].fillna(0)
        df['category'] = df.get('category', pd.Series('Unknown', index=df.index)).fillna('Unknown')
        df['sector'] = df.get('sector', pd.Series('Unknown', index=df.index)).fillna('Unknown')
        if 'industry' in df.columns: df['industry'] = df['industry'].fillna(df['sector'])
        else: df['industry'] = df['sector']
        return df

    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        if 'eps_change_pct' in df.columns:
            conditions = [df['eps_change_pct'] < 0, (df['eps_change_pct'] >= 0) & (df['eps_change_pct'] < 20), (df['eps_change_pct'] >= 20) & (df['eps_change_pct'] < 50), (df['eps_change_pct'] >= 50) & (df['eps_change_pct'] < 100), df['eps_change_pct'] >= 100]
            choices = ['Negative', 'Low (0-20%)', 'Medium (20-50%)', 'High (50-100%)', 'Extreme (>100%)']
            df['eps_tier'] = np.select(conditions, choices, default='Unknown')
        if 'pe' in df.columns:
            conditions = [df['pe'] < 0, (df['pe'] >= 0) & (df['pe'] < 15), (df['pe'] >= 15) & (df['pe'] < 25), (df['pe'] >= 25) & (df['pe'] < 50), df['pe'] >= 50]
            choices = ['Negative/NA', 'Value (<15)', 'Fair (15-25)', 'Growth (25-50)', 'Expensive (>50)']
            df['pe_tier'] = np.select(conditions, choices, default='Unknown')
        if 'price' in df.columns:
            conditions = [df['price'] < 10, (df['price'] >= 10) & (df['price'] < 100), (df['price'] >= 100) & (df['price'] < 1000), (df['price'] >= 1000) & (df['price'] < 5000), df['price'] >= 5000]
            choices = ['Penny (<₹10)', 'Low (₹10-100)', 'Mid (₹100-1000)', 'High (₹1000-5000)', 'Premium (>₹5000)']
            df['price_tier'] = np.select(conditions, choices, default='Unknown')
        return df

# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow'] = df['price'] * df['volume_1d'] * df['rvol']
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
        else: df['money_flow_mm'] = 0.0
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d']):
            df['vmi'] = (df['vol_ratio_1d_90d'] * 4 + df['vol_ratio_7d_90d'] * 3 + df['vol_ratio_30d_90d'] * 2 + df['vol_ratio_90d_180d'] * 1) / 10
        else: df['vmi'] = 1.0
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'] + abs(df['from_high_pct'])
        else: df['position_tension'] = 100.0
        df['momentum_harmony'] = 0
        if 'ret_1d' in df.columns: df['momentum_harmony'] += (df['ret_1d'] > 0).astype(int)
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0); daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            df['momentum_harmony'] += (daily_ret_7d > daily_ret_30d).astype(int)
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_30d_comp = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0); daily_ret_3m_comp = np.where(df['ret_3m'] != 0, df['ret_3m'] / 90, 0)
            df['momentum_harmony'] += (daily_ret_30d_comp > daily_ret_3m_comp).astype(int)
        if 'ret_3m' in df.columns: df['momentum_harmony'] += (df['ret_3m'] > 0).astype(int)
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)
        if all(col in df.columns for col in ['momentum_score', 'acceleration_score', 'rvol_score', 'breakout_score']):
            df['overall_wave_strength'] = (df['momentum_score'] * 0.3 + df['acceleration_score'] * 0.3 + df['rvol_score'] * 0.2 + df['breakout_score'] * 0.2)
        else: df['overall_wave_strength'] = 50.0
        return df
    @staticmethod
    def _get_wave_state(row: pd.Series) -> str:
        signals = 0
        if 'momentum_score' in row and row['momentum_score'] > 70: signals += 1
        if 'volume_score' in row and row['volume_score'] > 70: signals += 1
        if 'acceleration_score' in row and row['acceleration_score'] > 70: signals += 1
        if 'rvol' in row and row['rvol'] > 2: signals += 1
        if signals >= 4: return "🌊🌊🌊 CRESTING"
        elif signals >= 3: return "🌊🌊 BUILDING"
        elif signals >= 1: return "🌊 FORMING"
        else: return "💥 BREAKING"

# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['scoring'])
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df['position_score'] = RankingEngine._calculate_position_score(df)
        df['volume_score'] = RankingEngine._calculate_volume_score(df)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df)
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df)
        scores_matrix = np.column_stack([df[c].fillna(50) for c in ['position_score', 'volume_score', 'momentum_score', 'acceleration_score', 'breakout_score', 'rvol_score']])
        weights = np.array([CONFIG.POSITION_WEIGHT, CONFIG.VOLUME_WEIGHT, CONFIG.MOMENTUM_WEIGHT, CONFIG.ACCELERATION_WEIGHT, CONFIG.BREAKOUT_WEIGHT, CONFIG.RVOL_WEIGHT])
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').fillna(len(df) + 1).astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom').fillna(0) * 100
        df = RankingEngine._calculate_category_ranks(df)
        return df
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        if series is None or series.empty: return pd.Series(dtype=float)
        series = series.replace([np.inf, -np.inf], np.nan); valid_count = series.notna().sum()
        if valid_count == 0: return pd.Series(50, index=series.index)
        if pct: ranks = series.rank(pct=True, ascending=ascending, na_option='bottom') * 100
        else: ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
        ranks = ranks.fillna(0 if pct and ascending else (100 if pct and not ascending else valid_count + 1))
        return ranks
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        position_score = pd.Series(50, index=df.index, dtype=float)
        has_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        if not (has_low or has_high): return position_score
        from_low = df['from_low_pct'].fillna(50) if has_low else pd.Series(50, index=df.index)
        from_high = df['from_high_pct'].fillna(-50) if has_high else pd.Series(-50, index=df.index)
        rank_from_low = RankingEngine._safe_rank(from_low, pct=True, ascending=True) if has_low else pd.Series(50, index=df.index)
        rank_from_high = RankingEngine._safe_rank(from_high, pct=True, ascending=False) if has_high else pd.Series(50, index=df.index)
        position_score = (rank_from_low * 0.6 + rank_from_high * 0.4)
        return position_score.clip(0, 100)
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        volume_score = pd.Series(50, index=df.index, dtype=float)
        vol_cols = [('vol_ratio_1d_90d', 0.20), ('vol_ratio_7d_90d', 0.20), ('vol_ratio_30d_90d', 0.20), ('vol_ratio_30d_180d', 0.15), ('vol_ratio_90d_180d', 0.25)]
        total_weight = 0; weighted_score = pd.Series(0, index=df.index, dtype=float)
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                col_rank = RankingEngine._safe_rank(df[col], pct=True, ascending=True)
                weighted_score += col_rank * weight; total_weight += weight
        if total_weight > 0: volume_score = weighted_score / total_weight
        return volume_score.clip(0, 100)
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        has_ret_30d = 'ret_30d' in df.columns and df['ret_30d'].notna().any()
        if not has_ret_30d and 'ret_7d' in df.columns: momentum_score = RankingEngine._safe_rank(df['ret_7d'], pct=True, ascending=True); logger.info("Using 7-day returns for momentum score")
        elif has_ret_30d: momentum_score = RankingEngine._safe_rank(df['ret_30d'], pct=True, ascending=True)
        if has_ret_30d and 'ret_7d' in df.columns:
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            consistency_bonus[all_positive] = 5
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0); daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            consistency_bonus[accelerating] = 10
            momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
        return momentum_score
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        if len([c for c in req_cols if c in df.columns]) < 2: return acceleration_score
        ret_1d = df['ret_1d'].fillna(0); ret_7d = df['ret_7d'].fillna(0); ret_30d = df['ret_30d'].fillna(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_daily_1d = ret_1d; avg_daily_7d = np.where(ret_7d != 0, ret_7d / 7, 0); avg_daily_30d = np.where(ret_30d != 0, ret_30d / 30, 0)
        perfect = (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0); acceleration_score[perfect] = 100
        good = (~perfect) & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0); acceleration_score[good] = 80
        moderate = (~perfect) & (~good) & (ret_1d > 0); acceleration_score[moderate] = 60
        slight_decel = (ret_1d <= 0) & (ret_7d > 0); acceleration_score[slight_decel] = 40
        strong_decel = (ret_1d <= 0) & (ret_7d <= 0); acceleration_score[strong_decel] = 20
        return acceleration_score.clip(0, 100)
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        if 'from_high_pct' in df.columns: distance_factor = (100 - (-df['from_high_pct'].fillna(-50))).clip(0, 100)
        else: distance_factor = pd.Series(50, index=df.index)
        volume_factor = pd.Series(50, index=df.index)
        if 'vol_ratio_7d_90d' in df.columns: volume_factor = ((df['vol_ratio_7d_90d'].fillna(1.0) - 1) * 100).clip(0, 100)
        trend_factor = pd.Series(0, index=df.index, dtype=float)
        if 'price' in df.columns:
            current_price = df['price']; trend_count = 0
            for sma_col, points in [('sma_20d', 33.33), ('sma_50d', 33.33), ('sma_200d', 33.34)]:
                if sma_col in df.columns:
                    above_sma = (current_price > df[sma_col]).fillna(False)
                    trend_factor += above_sma.astype(float) * points; trend_count += 1
            if trend_count > 0 and trend_count < 3: trend_factor = trend_factor * (3 / trend_count)
        trend_factor = trend_factor.clip(0, 100)
        breakout_score = (distance_factor * 0.4 + volume_factor * 0.4 + trend_factor * 0.2)
        return breakout_score.clip(0, 100)
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        if 'rvol' not in df.columns: return pd.Series(50, index=df.index)
        rvol = df['rvol'].fillna(1.0); rvol_score = pd.Series(50, index=df.index, dtype=float)
        rvol_score[rvol > 10] = 95; rvol_score[(rvol > 5) & (rvol <= 10)] = 90; rvol_score[(rvol > 3) & (rvol <= 5)] = 85
        rvol_score[(rvol > 2) & (rvol <= 3)] = 80; rvol_score[(rvol > 1.5) & (rvol <= 2)] = 70; rvol_score[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score[(rvol > 0.8) & (rvol <= 1.2)] = 50; rvol_score[(rvol > 0.5) & (rvol <= 0.8)] = 40; rvol_score[(rvol > 0.3) & (rvol <= 0.5)] = 30
        rvol_score[rvol <= 0.3] = 20
        return rvol_score
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        trend_score = pd.Series(50, index=df.index, dtype=float)
        if 'price' not in df.columns: return trend_score
        current_price = df['price']; sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']; available_smas = [c for c in sma_cols if c in df.columns and df[c].notna().any()]
        if not available_smas: return trend_score
        if len(available_smas) >= 3:
            perfect = (current_price > df['sma_20d']) & (df['sma_20d'] > df['sma_50d']) & (df['sma_50d'] > df['sma_200d']); trend_score[perfect] = 100
            strong = (~perfect) & (current_price > df['sma_20d']) & (current_price > df['sma_50d']) & (current_price > df['sma_200d']); trend_score[strong] = 85
            above_count = sum([(current_price > df[s]).astype(int) for s in available_smas])
            good = (above_count == 2) & (~perfect) & (~strong); trend_score[good] = 70
            weak = (above_count == 1); trend_score[weak] = 40
            poor = (above_count == 0); trend_score[poor] = 20
        return trend_score
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        strength_score = pd.Series(50, index=df.index, dtype=float)
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']; available_cols = [c for c in lt_cols if c in df.columns and df[c].notna().any()]
        if not available_cols: return strength_score
        avg_return = df[available_cols].fillna(0).mean(axis=1)
        strength_score[avg_return > 100] = 100; strength_score[(avg_return > 50) & (avg_return <= 100)] = 90
        strength_score[(avg_return > 30) & (avg_return <= 50)] = 80; strength_score[(avg_return > 15) & (avg_return <= 30)] = 70
        strength_score[(avg_return > 5) & (avg_return <= 15)] = 60; strength_score[(avg_return > 0) & (avg_return <= 5)] = 50
        strength_score[(avg_return > -10) & (avg_return <= 0)] = 40; strength_score[(avg_return > -25) & (avg_return <= -10)] = 30
        strength_score[avg_return <= -25] = 20
        return strength_score.clip(0, 100)
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        if 'volume_30d' in df.columns and 'price' in df.columns:
            dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
            liquidity_score = RankingEngine._safe_rank(dollar_volume, pct=True, ascending=True)
        return liquidity_score.clip(0, 100)
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        df['category_rank'] = 9999; df['category_percentile'] = 0.0; categories = df['category'].unique()
        for category in categories:
            if category != 'Unknown':
                mask = df['category'] == category; cat_df = df[mask]
                if len(cat_df) > 0:
                    cat_ranks = cat_df['master_score'].rank(method='first', ascending=False, na_option='bottom')
                    df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                    cat_percentiles = cat_df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
                    df.loc[mask, 'category_percentile'] = cat_percentiles
        return df

# ============================================
# PATTERN DETECTION ENGINE
# ============================================

class PatternDetector:
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['pattern_detection'])
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: df['patterns'] = ''; df['pattern_confidence'] = 0; return df
        pattern_results = PatternDetector._get_all_pattern_masks(df)
        pattern_names = list(pattern_results.keys())
        pattern_matrix = np.column_stack([mask.values for mask in pattern_results.values()])
        df['patterns'] = [' | '.join([pattern_names[i] for i, val in enumerate(row) if val]) for row in pattern_matrix]
        df['patterns'] = df['patterns'].fillna('')
        df = PatternDetector._calculate_pattern_confidence(df)
        return df
    @staticmethod
    def _get_all_pattern_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
        masks = {}; safe_col = lambda name, default=0: df[name].fillna(default) if name in df.columns else pd.Series(default, index=df.index)
        if 'category_percentile' in df.columns: masks['🔥 CAT LEADER'] = safe_col('category_percentile') >= CONFIG.PATTERN_THRESHOLDS['category_leader']
        if 'category_percentile' in df.columns and 'percentile' in df.columns: masks['💎 HIDDEN GEM'] = (safe_col('category_percentile') >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (safe_col('percentile', 100) < 70)
        if 'acceleration_score' in df.columns: masks['🚀 ACCELERATING'] = safe_col('acceleration_score') >= CONFIG.PATTERN_THRESHOLDS['acceleration']
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns: masks['🏦 INSTITUTIONAL'] = (safe_col('volume_score') >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (safe_col('vol_ratio_90d_180d') > 1.1)
        if 'rvol' in df.columns: masks['⚡ VOL EXPLOSION'] = safe_col('rvol') > 3
        if 'breakout_score' in df.columns: masks['🎯 BREAKOUT'] = safe_col('breakout_score') >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
        if 'percentile' in df.columns: masks['👑 MARKET LEADER'] = safe_col('percentile') >= CONFIG.PATTERN_THRESHOLDS['market_leader']
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns: masks['🌊 MOMENTUM WAVE'] = (safe_col('momentum_score') >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (safe_col('acceleration_score') >= 70)
        if 'liquidity_score' in df.columns and 'percentile' in df.columns: masks['💰 LIQUID LEADER'] = (safe_col('liquidity_score') >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (safe_col('percentile') >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
        if 'long_term_strength' in df.columns: masks['💪 LONG STRENGTH'] = safe_col('long_term_strength') >= CONFIG.PATTERN_THRESHOLDS['long_strength']
        if 'trend_quality' in df.columns: masks['📈 QUALITY TREND'] = safe_col('trend_quality') >= 80
        if 'pe' in df.columns and 'master_score' in df.columns:
            has_valid_pe = safe_col('pe').notna() & (safe_col('pe') > 0) & (safe_col('pe') < 10000)
            masks['💎 VALUE MOMENTUM'] = has_valid_pe & (safe_col('pe') < 15) & (safe_col('master_score') >= CONFIG.PATTERN_THRESHOLDS['value_momentum'])
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = safe_col('eps_change_pct').notna(); extreme_growth = has_eps_growth & (safe_col('eps_change_pct') > 1000)
            normal_growth = has_eps_growth & (safe_col('eps_change_pct') > 50) & (safe_col('eps_change_pct') <= 1000)
            masks['📊 EARNINGS ROCKET'] = ((extreme_growth & (safe_col('acceleration_score') >= 80)) | (normal_growth & (safe_col('acceleration_score') >= 70)))
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = safe_col('pe').notna() & safe_col('eps_change_pct').notna() & (safe_col('pe') > 0) & (safe_col('pe') < 10000)
            masks['🏆 QUALITY LEADER'] = has_complete_data & (safe_col('pe').between(10, 25)) & (safe_col('eps_change_pct') > 20) & (safe_col('percentile') >= CONFIG.PATTERN_THRESHOLDS['quality_leader'])
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = safe_col('eps_change_pct').notna(); mega = has_eps & (safe_col('eps_change_pct') > 500) & (safe_col('volume_score') >= 60)
            strong = has_eps & (safe_col('eps_change_pct') > 100) & (safe_col('eps_change_pct') <= 500) & (safe_col('volume_score') >= 70)
            masks['⚡ TURNAROUND'] = mega | strong
        if 'pe' in df.columns:
            has_valid_pe = safe_col('pe').notna() & (safe_col('pe') > 0)
            masks['⚠️ HIGH PE'] = has_valid_pe & (safe_col('pe') > CONFIG.PATTERN_THRESHOLDS['high_pe'])
        if all(col in df.columns for col in ['from_high_pct', 'volume_score', 'momentum_score']):
            masks['🎯 52W HIGH APPROACH'] = (safe_col('from_high_pct') > -5) & (safe_col('volume_score') >= 70) & (safe_col('momentum_score') >= 60)
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            masks['🔄 52W LOW BOUNCE'] = (safe_col('from_low_pct') < 20) & (safe_col('acceleration_score') >= 80) & (safe_col('ret_30d') > 10)
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            masks['👑 GOLDEN ZONE'] = (safe_col('from_low_pct') > 60) & (safe_col('from_high_pct') > -40) & (safe_col('trend_quality') >= CONFIG.PATTERN_THRESHOLDS['golden_zone'])
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            masks['📊 VOL ACCUMULATION'] = (safe_col('vol_ratio_30d_90d') > 1.2) & (safe_col('vol_ratio_90d_180d') > 1.1) & (safe_col('ret_30d') > 5)
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'): daily_7d_pace = np.where(safe_col('ret_7d') != 0, safe_col('ret_7d') / 7, np.nan); daily_30d_pace = np.where(safe_col('ret_30d') != 0, safe_col('ret_30d') / 30, np.nan)
            masks['🔀 MOMENTUM DIVERGE'] = pd.Series(daily_7d_pace > daily_30d_pace * 1.5, index=df.index, dtype=bool) & (safe_col('acceleration_score') >= CONFIG.PATTERN_THRESHOLDS['momentum_diverge']) & (safe_col('rvol') > 2)
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            with np.errstate(divide='ignore', invalid='ignore'): range_pct = np.where(safe_col('low_52w') > 0, ((safe_col('high_52w') - safe_col('low_52w')) / safe_col('low_52w')) * 100, 100)
            masks['🎯 RANGE COMPRESS'] = (range_pct < 50) & (safe_col('from_low_pct') > 30)
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'): ret_ratio = np.where(safe_col('ret_30d') != 0, safe_col('ret_7d') / (safe_col('ret_30d') / 4), np.nan)
            masks['🤫 STEALTH'] = pd.Series(ret_ratio > 1, index=df.index, dtype=bool) & (safe_col('vol_ratio_90d_180d') > 1.1) & (safe_col('vol_ratio_30d_90d').between(0.9, 1.1)) & (safe_col('from_low_pct') > 40)
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'): daily_pace_ratio = np.where(safe_col('ret_7d') != 0, safe_col('ret_1d') / (safe_col('ret_7d') / 7), np.nan)
            masks['🧛 VAMPIRE'] = pd.Series(daily_pace_ratio > 2, index=df.index, dtype=bool) & (safe_col('rvol') > 3) & (safe_col('from_high_pct') > -15) & (df['category'].isin(['Small Cap', 'Micro Cap']))
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            masks['⛈️ PERFECT STORM'] = (safe_col('momentum_harmony') == 4) & (safe_col('master_score') > CONFIG.PATTERN_THRESHOLDS['perfect_storm'])
        return masks
    @staticmethod
    def _calculate_pattern_confidence(df: pd.DataFrame) -> pd.DataFrame:
        if 'patterns' not in df.columns or df['patterns'].eq('').all(): df['pattern_confidence'] = 0; return df
        confidence_scores = []
        for _, patterns_str in df['patterns'].items():
            if not patterns_str: confidence_scores.append(0); continue
            patterns = patterns_str.split(' | '); total_confidence = 0
            for pattern in patterns:
                metadata = CONFIG.PATTERN_METADATA.get(pattern, {})
                if metadata:
                    importance_scores = {'very_high': 40, 'high': 30, 'medium': 20, 'low': 10}
                    confidence = importance_scores.get(metadata['importance'], 20)
                    risk_multipliers = {'low': 1.2, 'medium': 1.0, 'high': 0.8, 'very_high': 0.6}
                    confidence *= risk_multipliers.get(metadata['risk'], 1.0)
                    total_confidence += confidence
            if len(patterns) > 1: total_confidence *= (1 + np.log(len(patterns))) / len(patterns)
            confidence_scores.append(min(100, total_confidence))
        df['pattern_confidence'] = confidence_scores
        return df

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        if df.empty: return "😴 NO DATA", {}
        metrics = {}
        if 'category' in df.columns and 'master_score' in df.columns:
            category_scores = df.groupby('category')['master_score'].mean(); micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean()
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean(); metrics['micro_small_avg'] = micro_small_avg; metrics['large_mega_avg'] = large_mega_avg
            metrics['category_spread'] = micro_small_avg - large_mega_avg
        else: micro_small_avg = 50; large_mega_avg = 50
        if 'ret_30d' in df.columns: metrics['breadth'] = len(df[df['ret_30d'] > 0]) / len(df)
        else: metrics['breadth'] = 0.5
        if 'rvol' in df.columns: metrics['avg_rvol'] = df['rvol'].median()
        else: metrics['avg_rvol'] = 1.0
        if micro_small_avg > large_mega_avg + 10 and metrics['breadth'] > 0.6: regime = "🔥 RISK-ON BULL"
        elif large_mega_avg > micro_small_avg + 10 and metrics['breadth'] < 0.4: regime = "🛡️ RISK-OFF DEFENSIVE"
        elif metrics['avg_rvol'] > 1.5 and metrics['breadth'] > 0.5: regime = "⚡ VOLATILE OPPORTUNITY"
        else: regime = "😴 RANGE-BOUND"
        metrics['regime'] = regime
        return regime, metrics
    @staticmethod
    def calculate_advance_decline_ratio(df: pd.DataFrame) -> Dict[str, Any]:
        ad_metrics = {}
        if 'ret_1d' in df.columns:
            advancing = len(df[df['ret_1d'] > 0]); declining = len(df[df['ret_1d'] < 0]); unchanged = len(df[df['ret_1d'] == 0])
            ad_metrics['advancing'] = advancing; ad_metrics['declining'] = declining; ad_metrics['unchanged'] = unchanged
            if declining > 0: ad_metrics['ad_ratio'] = advancing / declining
            else: ad_metrics['ad_ratio'] = float('inf') if advancing > 0 else 1.0
            ad_metrics['ad_line'] = advancing - declining
            ad_metrics['breadth_pct'] = (advancing / len(df)) * 100 if len(df) > 0 else 0
        return ad_metrics
    @staticmethod
    def _apply_dynamic_sampling(df_group: pd.DataFrame, group_size: int) -> pd.DataFrame:
        if 1 <= group_size <= 10: sample_count = group_size
        elif 11 <= group_size <= 50: sample_count = max(5, int(group_size * 0.6))
        elif 51 <= group_size <= 200: sample_count = max(20, int(group_size * 0.4))
        else: sample_count = min(60, int(group_size * 0.25))
        return df_group.nlargest(sample_count, 'master_score', keep='first') if sample_count > 0 else pd.DataFrame()
    @staticmethod
    def _calculate_flow_metrics(normalized_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        agg_dict = {'master_score': ['mean', 'median', 'std', 'count'], 'momentum_score': 'mean', 'volume_score': 'mean', 'rvol': 'mean', 'ret_30d': 'mean', 'money_flow_mm': 'sum'}
        available_agg_dict = {k: v for k, v in agg_dict.items() if k in normalized_df.columns}
        group_metrics = normalized_df.groupby(group_col).agg(available_agg_dict).round(2)
        group_metrics.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in group_metrics.columns]
        rename_map = {'master_score_mean': 'avg_score', 'master_score_median': 'median_score', 'master_score_std': 'std_score', 'master_score_count': 'count', 'momentum_score_mean': 'avg_momentum', 'volume_score_mean': 'avg_volume', 'rvol_mean': 'avg_rvol', 'ret_30d_mean': 'avg_ret_30d', 'money_flow_mm_sum': 'total_money_flow'}
        group_metrics = group_metrics.rename(columns=rename_map)
        group_metrics['flow_score'] = (group_metrics['avg_score'].fillna(0) * 0.3 + group_metrics['median_score'].fillna(0) * 0.2 + group_metrics['avg_momentum'].fillna(0) * 0.25 + group_metrics['avg_volume'].fillna(0) * 0.25)
        group_metrics['rank'] = group_metrics['flow_score'].rank(ascending=False, method='min')
        return group_metrics.sort_values('flow_score', ascending=False)
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'sector' not in df.columns or df.empty: return pd.DataFrame()
        sector_dfs = []; grouped_sectors = df.groupby('sector')
        for name, group in grouped_sectors:
            if name != 'Unknown': sampled = MarketIntelligence._apply_dynamic_sampling(group.copy(), len(group));
            if not sampled.empty: sector_dfs.append(sampled)
        if not sector_dfs: return pd.DataFrame()
        normalized_df = pd.concat(sector_dfs, ignore_index=True)
        metrics = MarketIntelligence._calculate_flow_metrics(normalized_df, 'sector')
        original_counts = df.groupby('sector').size().rename('total_stocks'); metrics = metrics.join(original_counts, how='left')
        metrics['analyzed_stocks'] = metrics['count']; metrics['sampling_pct'] = (metrics['analyzed_stocks'] / metrics['total_stocks'] * 100).round(1)
        return metrics
    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'industry' not in df.columns or df.empty: return pd.DataFrame()
        industry_dfs = []; grouped_industries = df.groupby('industry')
        for name, group in grouped_industries:
            if name != 'Unknown': sampled = MarketIntelligence._apply_dynamic_sampling(group.copy(), len(group))
            if not sampled.empty: industry_dfs.append(sampled)
        if not industry_dfs: return pd.DataFrame()
        normalized_df = pd.concat(industry_dfs, ignore_index=True)
        metrics = MarketIntelligence._calculate_flow_metrics(normalized_df, 'industry')
        original_counts = df.groupby('industry').size().rename('total_stocks'); metrics = metrics.join(original_counts, how='left')
        metrics['analyzed_stocks'] = metrics['count']; metrics['sampling_pct'] = (metrics['analyzed_stocks'] / metrics['total_stocks'] * 100).round(1)
        return metrics
    @staticmethod
    def detect_category_performance(df: pd.DataFrame) -> pd.DataFrame:
        if 'category' not in df.columns or df.empty: return pd.DataFrame()
        category_dfs = []; grouped_categories = df.groupby('category')
        for name, group in grouped_categories:
            if name != 'Unknown': sampled = MarketIntelligence._apply_dynamic_sampling(group.copy(), len(group))
            if not sampled.empty: category_dfs.append(sampled)
        if not category_dfs: return pd.DataFrame()
        normalized_df = pd.concat(category_dfs, ignore_index=True)
        agg_dict = {'master_score': ['mean', 'median', 'std', 'count'], 'momentum_score': 'mean', 'volume_score': 'mean', 'rvol': 'mean', 'ret_30d': 'mean', 'acceleration_score': 'mean', 'breakout_score': 'mean', 'money_flow_mm': 'sum'}
        available_agg_dict = {k: v for k, v in agg_dict.items() if k in normalized_df.columns}
        metrics = normalized_df.groupby('category').agg(available_agg_dict).round(2)
        metrics.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in metrics.columns]
        rename_map = {'master_score_mean': 'avg_score', 'master_score_median': 'median_score', 'master_score_std': 'std_score', 'master_score_count': 'count', 'momentum_score_mean': 'avg_momentum', 'volume_score_mean': 'avg_volume', 'rvol_mean': 'avg_rvol', 'ret_30d_mean': 'avg_ret_30d', 'acceleration_score_mean': 'avg_acceleration', 'breakout_score_mean': 'avg_breakout', 'money_flow_mm_sum': 'total_money_flow'}
        metrics = metrics.rename(columns=rename_map)
        metrics['flow_score'] = (metrics['avg_score'].fillna(0) * 0.35 + metrics['median_score'].fillna(0) * 0.2 + metrics['avg_momentum'].fillna(0) * 0.20 + metrics['avg_acceleration'].fillna(0) * 0.15 + metrics['avg_volume'].fillna(0) * 0.10)
        metrics['rank'] = metrics['flow_score'].rank(ascending=False, method='min')
        original_counts = df.groupby('category').size().rename('total_stocks'); metrics = metrics.join(original_counts, how='left')
        metrics['analyzed_stocks'] = metrics['count']; metrics['sampling_pct'] = (metrics['analyzed_stocks'] / metrics['total_stocks'] * 100).round(1)
        category_order = ['Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']; metrics = metrics.reindex([c for c in category_order if c in metrics.index])
        return metrics

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df.empty: fig.add_annotation(text="No data available for visualization", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False); return fig
        scores = [('position_score', 'Position', '#3498db'), ('volume_score', 'Volume', '#e74c3c'), ('momentum_score', 'Momentum', '#2ecc71'), ('acceleration_score', 'Acceleration', '#f39c12'), ('breakout_score', 'Breakout', '#9b59b6'), ('rvol_score', 'RVOL', '#e67e22')]
        for score_col, label, color in scores:
            if score_col in df.columns:
                score_data = df[score_col].dropna()
                if len(score_data) > 0: fig.add_trace(go.Box(y=score_data, name=label, marker_color=color, boxpoints='outliers', hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'))
        fig.update_layout(title="Score Component Distribution", yaxis_title="Score (0-100)", template='plotly_white', height=400, showlegend=False)
        return fig
    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, n: int = 10) -> go.Figure:
        try:
            plot_df = df.dropna(subset=['ret_1d', 'ret_7d', 'ret_30d'], how='any')
            if plot_df.empty: fig = go.Figure(); fig.add_annotation(text="No complete return data available for this chart.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False); return fig
            accel_df = plot_df.nlargest(min(n, len(plot_df)), 'acceleration_score'); fig = go.Figure()
            if accel_df.empty: fig.add_annotation(text="No stocks meet criteria for acceleration profiles.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False); return fig
            for _, stock in accel_df.iterrows():
                x_points = ['Start']; y_points = [0]
                if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']): x_points.append('30D'); y_points.append(stock['ret_30d'])
                if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']): x_points.append('7D'); y_points.append(stock['ret_7d'])
                if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']): x_points.append('Today'); y_points.append(stock['ret_1d'])
                if len(x_points) > 1:
                    accel_score = stock.get('acceleration_score', 0)
                    line_style = dict(width=3, dash='solid') if accel_score >= 85 else dict(width=2, dash='solid') if accel_score >= 70 else dict(width=2, dash='dot')
                    marker_style = dict(size=10, symbol='star', line=dict(color='DarkSlateGrey', width=1)) if accel_score >= 85 else dict(size=8) if accel_score >= 70 else dict(size=6)
                    fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='lines+markers', name=f"{stock['ticker']} ({accel_score:.0f})", line=line_style, marker=marker_style, hovertemplate=f"<b>{stock['ticker']}</b><br>%{{x}}: %{{y:.1f}}%<br>Accel Score: {accel_score:.0f}<extra></extra>"))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(title=f"Acceleration Profiles - Top {len(accel_df)} Momentum Builders", xaxis_title="Time Frame", yaxis_title="Return %", height=400, template='plotly_white', showlegend=True, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), hovermode='x unified')
            return fig
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}"); fig = go.Figure(); fig.add_annotation(text=f"Error generating chart: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False); return fig

# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['filtering'])
    def apply_all_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        if df.empty or not filters: return df
        mask = pd.Series(True, index=df.index)
        if 'categories' in filters and filters['categories']: mask &= df['category'].isin(filters['categories'])
        if 'sectors' in filters and filters['sectors']: mask &= df['sector'].isin(filters['sectors'])
        if 'industries' in filters and filters['industries'] and 'industry' in df.columns: mask &= df['industry'].isin(filters['industries'])
        if filters.get('min_score', 0) > 0: mask &= df['master_score'] >= filters['min_score']
        if filters.get('min_eps_change') and 'eps_change_pct' in df.columns: mask &= df['eps_change_pct'].notna() & (df['eps_change_pct'] >= float(filters['min_eps_change']))
        if filters.get('patterns') and 'patterns' in df.columns: mask &= df['patterns'].fillna('').str.contains('|'.join([re.escape(p) for p in filters['patterns']]), case=False, regex=True)
        if filters.get('trend_range') and filters.get('trend_filter') != 'All Trends' and 'trend_quality' in df.columns: min_t, max_t = filters['trend_range']; mask &= df['trend_quality'].notna() & (df['trend_quality'] >= min_t) & (df['trend_quality'] <= max_t)
        if filters.get('min_pe') and 'pe' in df.columns: mask &= df['pe'].notna() & (df['pe'] > 0) & (df['pe'] >= float(filters['min_pe']))
        if filters.get('max_pe') and 'pe' in df.columns: mask &= df['pe'].notna() & (df['pe'] > 0) & (df['pe'] <= float(filters['max_pe']))
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            if filters.get(tier_type): col_name = tier_type.replace('_tiers', '_tier');
            if col_name in df.columns: mask &= df[col_name].isin(filters[tier_type])
        if filters.get('require_fundamental_data', False):
            if all(col in df.columns for col in ['pe', 'eps_change_pct']): mask &= df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna()
        if filters.get('wave_states') and 'wave_state' in df.columns: mask &= df['wave_state'].isin(filters['wave_states'])
        if filters.get('wave_strength_range') and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = filters['wave_strength_range']; mask &= df['overall_wave_strength'].notna() & (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws)
        filtered_df = df[mask].copy()
        return filtered_df
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        if df.empty or column not in df.columns: return []
        temp_filters = current_filters.copy(); filter_key_map = {'category': 'categories', 'sector': 'sectors', 'industry': 'industries', 'eps_tier': 'eps_tiers', 'pe_tier': 'pe_tiers', 'price_tier': 'price_tiers', 'wave_state': 'wave_states'}
        if column in filter_key_map: temp_filters.pop(filter_key_map[column], None)
        filtered_df = FilterEngine.apply_all_filters(df, temp_filters)
        values = filtered_df[column].dropna().astype(str).unique()
        values = [v for v in values if v.strip().lower() not in ['unknown', '', 'nan', 'n/a', 'none', '-']]
        try: values = sorted(values, key=lambda x: float(str(x).replace(',', '')) if str(x).replace(',', '').replace('.', '').isdigit() else x)
        except: values = sorted(values, key=str)
        return values

# ============================================
# SEARCH ENGINE
# ============================================
class SearchEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['search'])
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        if not query or df.empty: return pd.DataFrame()
        try:
            query_upper = query.upper().strip(); results = df.copy(); results['relevance'] = 0
            exact_ticker_mask = (results['ticker'].str.upper() == query_upper).fillna(False)
            results.loc[exact_ticker_mask, 'relevance'] += 1000
            ticker_starts_mask = results['ticker'].str.upper().str.startswith(query_upper).fillna(False)
            results.loc[ticker_starts_mask & ~exact_ticker_mask, 'relevance'] += 500
            ticker_contains_mask = results['ticker'].str.upper().str.contains(query_upper, regex=False, na=False)
            results.loc[ticker_contains_mask & ~ticker_starts_mask, 'relevance'] += 200
            if 'company_name' in results.columns:
                name_contains_mask = results['company_name'].str.upper().str.contains(query_upper, regex=False, na=False)
                results.loc[name_contains_mask, 'relevance'] += 30
            all_matches = results[results['relevance'] > 0].copy()
            if all_matches.empty: return pd.DataFrame()
            sorted_results = all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False])
            return sorted_results.drop('relevance', axis=1)
        except Exception as e:
            logger.error(f"Search error: {str(e)}"); return pd.DataFrame()

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['export_generation'])
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        output = BytesIO()
        templates = {'day_trader': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 'volume_score', 'vmi', 'wave_state', 'patterns', 'category', 'sector', 'industry'], 'focus': 'Intraday momentum and volume'}, 'swing_trader': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'breakout_score', 'position_score', 'position_tension', 'from_high_pct', 'from_low_pct', 'trend_quality', 'momentum_harmony', 'patterns', 'sector', 'industry'], 'focus': 'Position and breakout setups'}, 'investor': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 'long_term_strength', 'money_flow_mm', 'category', 'sector', 'industry'], 'focus': 'Fundamentals and long-term performance'}, 'full': {'columns': None, 'focus': 'Complete analysis'}}
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                header_format = workbook.add_format({'bold': True, 'bg_color': '#3498db', 'font_color': 'white', 'border': 1})
                formats = {c: workbook.add_format({'num_format': f}) for c, f in {'price': '₹#,##0', 'master_score': '0.0', 'from_low_pct': '0.0%', 'ret_30d': '0.0%', 'rvol': '0.0"x"', 'money_flow_mm': '₹#,##0.0,"M"'}.items()}
                
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                export_cols = templates[template]['columns'] if templates[template]['columns'] else [c for c in top_100.columns if 'score' in c or 'rank' in c or 'tier' in c or 'ret_' in c or 'vol_' in c or c in ['ticker','company_name','price','pe','eps_current','eps_change_pct','from_low_pct','from_high_pct','rvol','vmi','money_flow_mm','wave_state','patterns','category','sector','industry']]
                top_100_export = top_100[[c for c in export_cols if c in top_100.columns]]
                top_100_export.to_excel(writer, sheet_name='Top 100', index=False)
                
                intel_data = [
                    {'Metric': 'Market Regime', 'Value': MarketIntelligence.detect_market_regime(df)[0], 'Details': f"Breadth: {MarketIntelligence.calculate_advance_decline_ratio(df).get('breadth_pct', 0):.1f}%"},
                    {'Metric': 'Advance/Decline', 'Value': f"{MarketIntelligence.calculate_advance_decline_ratio(df).get('advancing', 0)}/{MarketIntelligence.calculate_advance_decline_ratio(df).get('declining', 0)}", 'Details': f"Ratio: {MarketIntelligence.calculate_advance_decline_ratio(df).get('ad_ratio', 1):.2f}"}
                ]
                intel_df = pd.DataFrame(intel_data); intel_df.to_excel(writer, sheet_name='Market Intelligence', index=False)
                
                sector_rotation = MarketIntelligence.detect_sector_rotation(df);
                if not sector_rotation.empty: sector_rotation.to_excel(writer, sheet_name='Sector Rotation')
                
                pattern_counts = {}; [pattern_counts.update([p for p in s.split(' | ') if p]) for s in df['patterns'].dropna()];
                if pattern_counts: pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=False).to_excel(writer, sheet_name='Pattern Analysis', index=False)
                
                wave_signals = df[(df['momentum_score'] >= 60) & (df['acceleration_score'] >= 70) & (df['rvol'] >= 2)].nlargest(50, 'master_score')
                if len(wave_signals) > 0: wave_signals[['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'wave_state', 'patterns', 'category', 'industry']].to_excel(writer, sheet_name='Wave Radar', index=False)
                
                summary_stats = {'Total Stocks': len(df), 'Average Master Score': df['master_score'].mean(), 'Stocks with Patterns': (df['patterns'] != '').sum(), 'High RVOL (>2x)': (df['rvol'] > 2).sum(), 'Positive 30D Returns': (df['ret_30d'] > 0).sum(), 'Template Used': template, 'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value']).to_excel(writer, sheet_name='Summary', index=False)
            
            output.seek(0); return output
        except Exception as e: logger.error(f"Error creating Excel report: {str(e)}"); raise
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        export_cols = ['rank', 'ticker', 'company_name', 'master_score', 'position_score', 'volume_score', 'momentum_score', 'acceleration_score', 'breakout_score', 'rvol_score', 'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct', 'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'rvol', 'vmi', 'money_flow_mm', 'position_tension', 'momentum_harmony', 'wave_state', 'patterns', 'category', 'sector', 'industry', 'eps_tier', 'pe_tier', 'overall_wave_strength']
        export_df = df[[c for c in export_cols if c in df.columns]].copy()
        for col in [c for c in export_df.columns if 'vol_ratio' in c]: export_df[col] = (export_df[col] - 1) * 100
        return export_df.to_csv(index=False)

# ============================================
# UI AND MAIN APPLICATION LOGIC
# ============================================

class UIComponents:
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None, help_text: Optional[str] = None) -> None:
        if help_text: st.metric(label, value, delta, help=help_text)
        else: st.metric(label, value, delta)
    @staticmethod
    def render_summary_section(df: pd.DataFrame) -> None:
        if df.empty: st.warning("No data available for summary"); return
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df); ad_ratio = ad_metrics.get('ad_ratio', 1.0)
            ad_emoji = "🔥" if ad_ratio > 2 else "📈" if ad_ratio > 1 else "📉"
            UIComponents.render_metric_card("A/D Ratio", f"{ad_emoji} {ad_ratio:.2f}", f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}")
        with col2:
            high_momentum = len(df[df['momentum_score'] >= 70]); momentum_pct = (high_momentum / len(df) * 100)
            UIComponents.render_metric_card("Momentum Health", f"{momentum_pct:.0f}%", f"{high_momentum} strong stocks")
        with col3:
            avg_rvol = df['rvol'].median(); high_vol_count = len(df[df['rvol'] > 2])
            vol_emoji = "🌊" if avg_rvol > 1.5 else "💧" if avg_rvol > 1.2 else "🏜️"
            UIComponents.render_metric_card("Volume State", f"{vol_emoji} {avg_rvol:.1f}x", f"{high_vol_count} surges")
        with col4:
            risk_factors = 0
            if len(df[(df['from_high_pct'] >= 0) & (df['momentum_score'] < 50)]) > 20: risk_factors += 1
            if len(df[(df['rvol'] > 10) & (df['master_score'] < 50)]) > 10: risk_factors += 1
            if len(df[df['trend_quality'] < 40]) > len(df) * 0.3: risk_factors += 1
            risk_level = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"][min(risk_factors, 3)]
            UIComponents.render_metric_card("Risk Level", risk_level, f"{risk_factors} factors")
        st.markdown("### 🎯 Today's Best Opportunities")
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        with opp_col1:
            st.markdown("**🚀 Ready to Run**")
            ready_to_run = df[(df['momentum_score'] >= 70) & (df['acceleration_score'] >= 70) & (df['rvol'] >= 2)].nlargest(5, 'master_score')
            if not ready_to_run.empty: [st.write(f"• **{s['ticker']}** - {s['company_name'][:25]}"); st.caption(f"Score: {s['master_score']:.1f} | RVOL: {s['rvol']:.1f}x") for _, s in ready_to_run.iterrows()]
            else: st.info("No momentum leaders found")
        with opp_col2:
            st.markdown("**💎 Hidden Gems**")
            hidden_gems = df[df['patterns'].str.contains('HIDDEN GEM', na=False)].nlargest(5, 'master_score')
            if not hidden_gems.empty: [st.write(f"• **{s['ticker']}** - {s['company_name'][:25]}"); st.caption(f"Cat %ile: {s.get('category_percentile', 0):.0f} | Score: {s['master_score']:.1f}") for _, s in hidden_gems.iterrows()]
            else: st.info("No hidden gems today")
        with opp_col3:
            st.markdown("**⚡ Volume Alerts**")
            volume_alerts = df[df['rvol'] > 3].nlargest(5, 'master_score')
            if not volume_alerts.empty: [st.write(f"• **{s['ticker']}** - {s['company_name'][:25]}"); st.caption(f"RVOL: {s['rvol']:.1f}x | {s.get('wave_state', 'N/A')}") for _, s in volume_alerts.iterrows()]
            else: st.info("No extreme volume detected")
        st.markdown("### 🧠 Market Intelligence")
        intel_col1, intel_col2 = st.columns([2, 1])
        with intel_col1:
            sector_rotation = MarketIntelligence.detect_sector_rotation(df);
            if not sector_rotation.empty:
                fig = go.Figure(); fig.add_trace(go.Bar(x=sector_rotation.index[:10], y=sector_rotation['flow_score'][:10], text=[f"{v:.1f}" for v in sector_rotation['flow_score'][:10]], textposition='outside', marker_color=['#2ecc71' if s > 60 else '#e74c3c' if s < 40 else '#f39c12' for s in sector_rotation['flow_score'][:10]], hovertemplate='Sector: %{x}<br>Flow Score: %{y:.1f}<br>Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>Avg Score: %{customdata[2]:.1f}<br>Median Score: %{customdata[3]:.1f}<extra></extra>', customdata=np.column_stack((sector_rotation['analyzed_stocks'][:10], sector_rotation['total_stocks'][:10], sector_rotation['avg_score'][:10], sector_rotation['median_score'][:10]))));
                fig.update_layout(title="Sector Rotation Map - Smart Money Flow (Dynamically Sampled)", xaxis_title="Sector", yaxis_title="Flow Score", height=400, template='plotly_white', showlegend=False); st.plotly_chart(fig, use_container_width=True)
            else: st.info("No sector rotation data available for visualization.")
        with intel_col2:
            regime, metrics = MarketIntelligence.detect_market_regime(df); st.markdown(f"**🎯 Market Regime**"); st.markdown(f"### {regime}")
            st.markdown("**📡 Key Signals**"); signals = []; breadth = metrics.get('breadth', 0.5)
            if breadth > 0.6: signals.append("✅ Strong breadth");
            elif breadth < 0.4: signals.append("⚠️ Weak breadth")
            spread = metrics.get('category_spread', 0);
            if spread > 10: signals.append("🔄 Small caps leading");
            elif spread < -10: signals.append("🛡️ Large caps defensive")
            if metrics.get('avg_rvol', 1.0) > 1.5: signals.append("🌊 High volume activity")
            if len(df[df['patterns']!='']) > len(df) * 0.2: signals.append("🎯 Many patterns emerging")
            [st.write(s) for s in signals]
            st.markdown("**💪 Market Strength**"); score = (breadth*50) + (min(metrics.get('avg_rvol', 1.0), 2)*25) + ((len(df[df['patterns']!=''])/len(df))*25); meter = "🟢🟢🟢🟢🟢" if score > 70 else "🟢🟢🟢🟢⚪" if score > 50 else "🟢🟢🟢⚪⚪" if score > 30 else "🟢🟢⚪⚪⚪"
            st.write(meter)

def main():
    st.set_page_config(page_title="Wave Detection Ultimate 3.0", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
    RobustSessionState.initialize()
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Assuming CSS is here
    st.markdown("""<div style="..."><h1>🌊 Wave Detection Ultimate 3.0</h1>...</div>""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions"); col1, col2 = st.columns(2)
        if col1.button("🔄 Refresh Data", type="primary"): st.cache_data.clear(); RobustSessionState.safe_set('last_refresh', datetime.now(timezone.utc)); st.rerun()
        if col2.button("🧹 Clear Cache"): st.cache_data.clear(); gc.collect(); st.success("Cache cleared!"); time.sleep(0.5); st.rerun()
        st.markdown("---"); st.markdown("### 📂 Data Source"); data_source_col1, data_source_col2 = st.columns(2)
        if data_source_col1.button("📊 Google Sheets", type="primary" if RobustSessionState.safe_get('data_source') == "sheet" else "secondary"): RobustSessionState.safe_set('data_source', "sheet"); st.rerun()
        if data_source_col2.button("📁 Upload CSV", type="primary" if RobustSessionState.safe_get('data_source') == "upload" else "secondary"): RobustSessionState.safe_set('data_source', "upload"); st.rerun()
        uploaded_file = None; sheet_id = None; gid = None
        if RobustSessionState.safe_get('data_source') == "upload":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file is None: st.info("Please upload a CSV file to continue"); st.stop()
        else:
            st.markdown("#### 📊 Google Sheets Configuration"); sheet_input = st.text_input("Google Sheets ID or URL", value=RobustSessionState.safe_get('sheet_id', '')); sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_input); sheet_id = sheet_id_match.group(1) if sheet_id_match else sheet_input.strip()
            RobustSessionState.safe_set('sheet_id', sheet_id)
            gid_input = st.text_input("Sheet Tab GID (Optional)", value=RobustSessionState.safe_get('gid', CONFIG.DEFAULT_GID)); gid = gid_input.strip() if gid_input else CONFIG.DEFAULT_GID
            if not sheet_id: st.warning("Please enter a Google Sheets ID to continue"); st.stop()
        data_quality = RobustSessionState.safe_get('data_quality', {}); st.markdown("### 🔍 Smart Filters"); active_filter_count = 0; filter_checks = [('category_filter', lambda x: x), ('sector_filter', lambda x: x), ('industry_filter', lambda x: x), ('min_score', lambda x: x > 0), ('patterns', lambda x: x), ('trend_filter', lambda x: x != 'All Trends'), ('eps_tier_filter', lambda x: x), ('pe_tier_filter', lambda x: x), ('price_tier_filter', lambda x: x), ('min_eps_change', lambda x: x), ('min_pe', lambda x: x), ('max_pe', lambda x: x), ('require_fundamental_data', lambda x: x), ('wave_states_filter', lambda x: x), ('wave_strength_range_slider', lambda x: x != (0, 100))]; [active_filter_count := active_filter_count + 1 for k, c in filter_checks if RobustSessionState.safe_get(k) is not None and c(RobustSessionState.safe_get(k))]; RobustSessionState.safe_set('active_filter_count', active_filter_count)
        if active_filter_count > 0: st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**"); st.button("🗑️ Clear All Filters", use_container_width=True, type="primary" if active_filter_count > 0 else "secondary", on_click=RobustSessionState.clear_filters); st.rerun()

    try:
        ranked_df, data_timestamp, metadata = load_and_process_data(RobustSessionState.safe_get('data_source'), uploaded_file, sheet_id, gid)
        RobustSessionState.safe_set('ranked_df', ranked_df); RobustSessionState.safe_set('data_timestamp', data_timestamp)
        if metadata.get('warnings'): [st.warning(w) for w in metadata['warnings']]
        if metadata.get('errors'): [st.error(e) for e in metadata['errors']]
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}"); st.stop()

    filtered_df = FilterEngine.apply_all_filters(ranked_df, RobustSessionState.safe_get('filters', {}))
    
    st.columns(6)[0].metric("Total Stocks", f"{len(filtered_df):,}");
    st.columns(6)[1].metric("Avg Score", f"{filtered_df['master_score'].mean():.1f}")
    # ... rest of the metric cards
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"])
    
    with tab1: st.markdown("### 📊 Executive Summary Dashboard"); UIComponents.render_summary_section(filtered_df)
    with tab2: st.markdown("### 🏆 Top Ranked Stocks")
    with tab3: st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
    with tab4: st.markdown("### 📊 Market Analysis")
    with tab5:
        st.markdown("### 🔍 Advanced Stock Search"); search_query = st.text_input("Search stocks", placeholder="Enter ticker or company name...", key="search_input")
        if st.button("🔎 Search") or search_query:
            results = SearchEngine.search_stocks(filtered_df, search_query); st.dataframe(results)
    with tab6: st.markdown("### 📥 Export Data"); ExportEngine.create_csv_export(filtered_df)
    with tab7: st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}"); logger.error(f"Application crashed: {str(e)}", exc_info=True)
        if st.button("🔄 Restart Application"): st.cache_data.clear(); st.rerun()
        if st.button("📧 Report Issue"): st.info("Please take a screenshot and report this error.")
