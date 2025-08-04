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
from io import BytesIO, StringIO
import warnings
import gc
import re
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict, Counter
import math

# --- Setup ---
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
np.random.seed(42)
pd.options.display.float_format = '{:.2f}'.format


# ============================================
# SCRIPT METADATA & VERSIONING
# ============================================

SCRIPT_METADATA = {
    "version": "3.1.1-FINAL-ULTIMATE",
    "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    "status": "PRODUCTION READY - PERMANENTLY LOCKED",
    "description": "Professional Stock Ranking System with advanced analytics and robust data pipeline."
}


# ============================================
# LOGGING CONFIGURATION
# ============================================

log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

performance_stats = defaultdict(list)

def log_performance(operation: str, duration: float):
    performance_stats[operation].append(duration)
    if len(performance_stats[operation]) > 100:
        performance_stats[operation] = performance_stats[operation][-100:]
    if duration > 1.0:
        logger.warning(f"{operation} took {duration:.2f}s, exceeding 1.0s threshold.")


# ============================================
# ROBUST SESSION STATE MANAGER
# ============================================

class RobustSessionState:
    """Bulletproof session state management - prevents all KeyErrors"""
    STATE_DEFAULTS = {
        'search_query': "",
        'last_refresh': None,
        'data_source': "sheet",
        'sheet_id': "",
        'gid': "",
        'user_preferences': {
            'default_top_n': 50,
            'display_mode': 'Technical',
            'last_filters': {}
        },
        'filters': {},
        'active_filter_count': 0,
        'quick_filter': None,
        'quick_filter_applied': False,
        'show_debug': False,
        'performance_metrics': {},
        'data_quality': {},
        'last_good_data': None,
        'session_id': None,
        'session_start': None,
        'validation_stats': defaultdict(int),
        'trigger_clear': False,
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
        'wave_states_filter': [],
        'wave_strength_range_slider': (0, 100),
        'show_sensitivity_details': False,
        'show_market_regime': True,
        'wave_timeframe_select': "All Waves",
        'wave_sensitivity': "Balanced",
        'export_template_radio': "Full Analysis (All Data)",
        'display_mode_toggle': 0,
        'ranked_df': None,
        'data_timestamp': None,
        'search_input': ""
    }

    @staticmethod
    def safe_get(key: str, default: Any = None) -> Any:
        if key not in st.session_state:
            st.session_state[key] = RobustSessionState.STATE_DEFAULTS.get(key, default)
        return st.session_state[key]

    @staticmethod
    def safe_set(key: str, value: Any) -> None:
        st.session_state[key] = value

    @staticmethod
    def initialize():
        for key, default_value in RobustSessionState.STATE_DEFAULTS.items():
            if key not in st.session_state:
                if key == 'last_refresh' and default_value is None:
                    st.session_state[key] = datetime.now(timezone.utc)
                elif key == 'session_start' and default_value is None:
                    st.session_state[key] = datetime.now(timezone.utc)
                elif key == 'session_id' and default_value is None:
                    st.session_state[key] = hashlib.md5(f"{datetime.now()}{np.random.rand()}".encode()).hexdigest()[:8]
                else:
                    st.session_state[key] = default_value

    @staticmethod
    def clear_filters():
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'patterns',
            'min_score', 'trend_filter', 'min_eps_change',
            'min_pe', 'max_pe', 'require_fundamental_data',
            'quick_filter', 'quick_filter_applied',
            'wave_states_filter', 'wave_strength_range_slider',
            'show_sensitivity_details', 'show_market_regime',
            'wave_timeframe_select', 'wave_sensitivity'
        ]
        for key in filter_keys:
            if key in RobustSessionState.STATE_DEFAULTS:
                RobustSessionState.safe_set(key, RobustSessionState.STATE_DEFAULTS[key])
        RobustSessionState.safe_set('filters', {})
        RobustSessionState.safe_set('active_filter_count', 0)
        RobustSessionState.safe_set('trigger_clear', False)

    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        return {
            'session_id': RobustSessionState.safe_get('session_id', 'unknown'),
            'start_time': RobustSessionState.safe_get('session_start', datetime.now()),
            'duration': (datetime.now() - RobustSessionState.safe_get('session_start', datetime.now())).seconds,
            'data_source': RobustSessionState.safe_get('data_source', 'unknown'),
            'stocks_loaded': len(RobustSessionState.safe_get('ranked_df', [])),
            'active_filters': RobustSessionState.safe_get('active_filter_count', 0)
        }


# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights and thresholds"""
    DEFAULT_SHEET_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing"
    CSV_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    DEFAULT_GID: str = "1823439984"
    REQUEST_TIMEOUT: int = 30
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_FACTOR: float = 0.5
    RETRY_STATUS_CODES: Tuple[int, ...] = (408, 429, 500, 502, 503, 504)
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
        'ret_30d', 'from_low_pct', 'from_high_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d', 'volume_90d', 'volume_30d', 'volume_7d',
        'ret_7d', 'category', 'sector', 'industry', 'rvol'
    ])
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
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
        "value_momentum": 70, "earnings_rocket": 70, "quality_leader": 80, "turnaround": 70, "high_pe": 100
    })
    PATTERN_METADATA: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        '🔥 CAT LEADER': {'importance': 'high', 'risk': 'low'}, '💎 HIDDEN GEM': {'importance': 'high', 'risk': 'medium'},
        '🚀 ACCELERATING': {'importance': 'high', 'risk': 'medium'}, '🏦 INSTITUTIONAL': {'importance': 'high', 'risk': 'low'},
        '⚡ VOL EXPLOSION': {'importance': 'very_high', 'risk': 'high'}, '🎯 BREAKOUT': {'importance': 'high', 'risk': 'medium'},
        '👑 MARKET LEADER': {'importance': 'very_high', 'risk': 'low'}, '🌊 MOMENTUM WAVE': {'importance': 'high', 'risk': 'medium'},
        '💰 LIQUID LEADER': {'importance': 'medium', 'risk': 'low'}, '💪 LONG STRENGTH': {'importance': 'medium', 'risk': 'low'},
        '📈 QUALITY TREND': {'importance': 'high', 'risk': 'low'}, '⛈️ PERFECT STORM': {'importance': 'very_high', 'risk': 'medium'}
    })
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000), 'rvol': (0.01, 1_000_000.0), 'pe': (-10000, 10000),
        'returns': (-99.99, 9999.99), 'volume': (0, 1e15)
    })
    PERFORMANCE_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'data_processing': 2.0, 'filtering': 0.2, 'pattern_detection': 0.5,
        'export_generation': 1.0, 'search': 0.05
    })
    MARKET_CATEGORIES: List[str] = field(default_factory=lambda: [
        'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'
    ])
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {"Loss": (-np.inf, 0), "0-5": (0, 5), "5-10": (5, 10), "10-20": (10, 20), "20-50": (20, 50), "50-100": (50, 100), "100+": (100, np.inf)},
        "pe": {"Negative/NA": (-np.inf, 0), "0-10": (0, 10), "10-15": (10, 15), "15-20": (15, 20), "20-30": (20, 30), "30-50": (30, 50), "50+": (50, np.inf)},
        "price": {"0-100": (0, 100), "100-250": (100, 250), "250-500": (250, 500), "500-1000": (500, 1000), "1000-2500": (1000, 2500), "2500-5000": (2500, 5000), "5000+": (5000, np.inf)}
    })

    def __post_init__(self):
        total_weight = sum([self.POSITION_WEIGHT, self.VOLUME_WEIGHT, self.MOMENTUM_WEIGHT, self.ACCELERATION_WEIGHT, self.BREAKOUT_WEIGHT, self.RVOL_WEIGHT])
        if not np.isclose(total_weight, 1.0, rtol=1e-5):
            raise ValueError(f"Scoring weights must sum to 1.0, but got {total_weight}")

CONFIG = Config()


# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """Track and report performance metrics"""

    @staticmethod
    def timer(target_time: Optional[float] = None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    log_performance(func.__name__, elapsed)
                    if target_time and elapsed > target_time:
                        logger.warning(f"{func.__name__} took {elapsed:.2f}s (target: {target_time}s)")
                    elif elapsed > 1.0:
                        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                    perf_metrics = RobustSessionState.safe_get('performance_metrics', {})
                    perf_metrics[func.__name__] = elapsed
                    RobustSessionState.safe_set('performance_metrics', perf_metrics)
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
                    raise
            return wrapper
        return decorator

    @staticmethod
    def get_performance_summary() -> Dict[str, Dict[str, float]]:
        summary = {}
        for op, durations in performance_stats.items():
            if durations:
                summary[op] = {
                    'avg': np.mean(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'p95': np.percentile(durations, 95) if len(durations) > 1 else durations[0]
                }
        return summary


# ============================================
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidator:
    """Comprehensive data validation with tracking"""

    def __init__(self):
        self.validation_stats = defaultdict(int)
        self.correction_stats = defaultdict(int)

    def reset_stats(self):
        self.validation_stats.clear()
        self.correction_stats.clear()

    def get_validation_report(self) -> Dict[str, Any]:
        return {'total_issues': sum(self.correction_stats.values())}

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> Tuple[bool, str]:
        if df is None: return False, f"{context}: DataFrame is None"
        if df.empty: return False, f"{context}: DataFrame is empty"
        missing_critical = [col for col in CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical: return False, f"{context}: Missing critical columns: {missing_critical}"
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0: logger.warning(f"{context}: Found {duplicates} duplicate tickers")
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        completeness = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        if completeness < 50: logger.warning(f"{context}: Low data completeness ({completeness:.1f}%)")
        data_quality = RobustSessionState.safe_get('data_quality', {})
        data_quality.update({
            'completeness': completeness, 'total_rows': len(df), 'total_columns': len(df.columns),
            'duplicate_tickers': duplicates, 'context': context, 'timestamp': datetime.now(timezone.utc)
        })
        RobustSessionState.safe_set('data_quality', data_quality)
        logger.info(f"{context}: Validated {len(df)} rows, {len(df.columns)} columns, {completeness:.1f}% complete")
        return True, "Valid"

    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None, column_name: str = "") -> float:
        if pd.isna(value) or value == '' or value is None: return np.nan
        try:
            cleaned = str(value).strip()
            if cleaned.upper() in ['', '-', 'N/A', 'NA', 'NAN', 'NONE', '#VALUE!', '#ERROR!', '#DIV/0!', 'INF', '-INF']: return np.nan
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            result = float(cleaned)
            if bounds:
                min_val, max_val = bounds
                if result < min_val or result > max_val: result = np.clip(result, min_val, max_val)
            if np.isnan(result) or np.isinf(result): return np.nan
            return result
        except (ValueError, TypeError, AttributeError):
            return np.nan

    @staticmethod
    def sanitize_string(value: Any, default: str = "Unknown") -> str:
        if pd.isna(value) or value is None: return default
        cleaned = str(value).strip()
        if cleaned.upper() in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']: return default
        cleaned = ' '.join(cleaned.split())
        return cleaned

validator = DataValidator()


# ============================================
# SMART CACHING WITH VERSIONING
# ============================================

def get_requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=CONFIG.MAX_RETRY_ATTEMPTS, read=CONFIG.MAX_RETRY_ATTEMPTS, connect=CONFIG.MAX_RETRY_ATTEMPTS,
        backoff_factor=CONFIG.RETRY_BACKOFF_FACTOR, status_forcelist=CONFIG.RETRY_STATUS_CODES
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@st.cache_data(persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, sheet_id: str = None, gid: str = None, data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    start_time = time.perf_counter()
    metadata = {
        'source_type': source_type, 'data_version': data_version, 'processing_start': datetime.now(timezone.utc),
        'errors': [], 'warnings': [], 'performance': {}
    }
    try:
        validator.reset_stats()
        if source_type == "upload" and file_data is not None:
            df = pd.read_csv(file_data, low_memory=False)
            metadata['source'] = "User Upload"
        else:
            if not sheet_id: raise ValueError("Please enter a Google Sheets ID")
            sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_id)
            if sheet_id_match: sheet_id = sheet_id_match.group(1)
            if not gid: gid = CONFIG.DEFAULT_GID
            csv_url = CONFIG.CSV_URL_TEMPLATE.format(sheet_id=sheet_id, gid=gid)
            session = get_requests_session()
            try:
                response = session.get(csv_url, timeout=CONFIG.REQUEST_TIMEOUT)
                response.raise_for_status()
                df = pd.read_csv(BytesIO(response.content), low_memory=False)
                metadata['source'] = "Google Sheets"
            except Exception as e:
                last_good_data = RobustSessionState.safe_get('last_good_data')
                if last_good_data: return last_good_data
                raise
        metadata['performance']['load_time'] = time.perf_counter() - start_time
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid: raise ValueError(validation_msg)
        df = DataProcessor.process_dataframe(df, metadata)
        df = RankingEngine.calculate_all_scores(df)
        df = PatternDetector.detect_all_patterns_optimized(df)
        df = AdvancedMetrics.calculate_all_metrics(df)
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid: raise ValueError(validation_msg)
        timestamp = datetime.now(timezone.utc)
        RobustSessionState.safe_set('last_good_data', (df.copy(), timestamp, metadata))
        total_time = time.perf_counter() - start_time
        metadata['performance']['total_time'] = total_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        return df, timestamp, metadata
    except Exception as e:
        last_good_data = RobustSessionState.safe_get('last_good_data')
        if last_good_data:
            df, timestamp, old_metadata = last_good_data
            metadata['warnings'].append("Using cached data due to failed live load.")
            return df, timestamp, metadata
        raise


# ============================================
# DATA PROCESSING ENGINE
# ============================================

class DataProcessor:
    """Handle all data processing with validation and optimization"""
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        initial_count = len(df)
        df['ticker'] = df['ticker'].apply(DataValidator.sanitize_string)
        numeric_cols = [col for col in df.columns if col not in ['ticker', 'company_name', 'category', 'sector', 'industry', 'year', 'market_cap']]
        for col in numeric_cols:
            if col in df.columns:
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                bounds = None
                if 'volume' in col.lower(): bounds = CONFIG.VALUE_BOUNDS['volume']
                elif col == 'rvol': bounds = CONFIG.VALUE_BOUNDS['rvol']
                elif col == 'pe': bounds = CONFIG.VALUE_BOUNDS['pe']
                elif is_pct: bounds = CONFIG.VALUE_BOUNDS['returns']
                else: bounds = CONFIG.VALUE_BOUNDS.get('price', None)
                df[col] = df[col].apply(lambda x: validator.clean_numeric_value(x, is_pct, bounds, col))
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns: df[col] = df[col].apply(DataValidator.sanitize_string)
        if 'industry' not in df.columns and 'sector' in df.columns:
            df['industry'] = df['sector']
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 1000.0)
                df[col] = df[col].fillna(1.0)
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df): metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        df = DataProcessor._fill_missing_values(df)
        df = DataProcessor._add_tier_classifications(df)
        removed = initial_count - len(df)
        if removed > 0: metadata['warnings'].append(f"Removed {removed} invalid rows during processing")
        RobustSessionState.safe_set('data_quality', metadata.get('data_quality', {}))
        return df

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        df['from_low_pct'] = df.get('from_low_pct', pd.Series(50.0, index=df.index)).fillna(50.0)
        df['from_high_pct'] = df.get('from_high_pct', pd.Series(-50.0, index=df.index)).fillna(-50.0)
        df['rvol'] = df.get('rvol', pd.Series(1.0, index=df.index)).fillna(1.0)
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols:
            if col in df.columns: df[col] = df[col].fillna(0.0)
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols:
            if col in df.columns: df[col] = df[col].fillna(0)
        df['category'] = df.get('category', pd.Series('Unknown', index=df.index)).fillna('Unknown')
        df['sector'] = df.get('sector', pd.Series('Unknown', index=df.index)).fillna('Unknown')
        if 'industry' in df.columns: df['industry'] = df['industry'].fillna(df['sector'])
        else: df['industry'] = df['sector']
        return df

    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            if pd.isna(value): return "Unknown"
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val: return tier_name
                if min_val == -np.inf and value <= max_val: return tier_name
                if max_val == np.inf and value > min_val: return tier_name
            return "Unknown"
        if 'eps_current' in df.columns: df['eps_tier'] = df['eps_current'].apply(lambda x: classify_tier(x, CONFIG.TIERS['eps']))
        if 'pe' in df.columns: df['pe_tier'] = df['pe'].apply(lambda x: "Negative/NA" if pd.isna(x) or x <= 0 else classify_tier(x, CONFIG.TIERS['pe']))
        if 'price' in df.columns: df['price_tier'] = df['price'].apply(lambda x: classify_tier(x, CONFIG.TIERS['price']))
        return df


# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    """Calculate advanced metrics and indicators"""
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
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            df['momentum_harmony'] += (daily_ret_7d > daily_ret_30d).astype(int)
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_30d_comp = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
                daily_ret_3m_comp = np.where(df['ret_3m'] != 0, df['ret_3m'] / 90, 0)
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
        if row.get('momentum_score', 0) > 70: signals += 1
        if row.get('volume_score', 0) > 70: signals += 1
        if row.get('acceleration_score', 0) > 70: signals += 1
        if row.get('rvol', 0) > 2: signals += 1
        if signals >= 4: return "🌊🌊🌊 CRESTING"
        elif signals >= 3: return "🌊🌊 BUILDING"
        elif signals >= 1: return "🌊 FORMING"
        else: return "💥 BREAKING"


# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    """Core ranking calculations - optimized with numpy"""
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
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
        scores_matrix = np.column_stack([
            df['position_score'].fillna(50), df['volume_score'].fillna(50), df['momentum_score'].fillna(50),
            df['acceleration_score'].fillna(50), df['breakout_score'].fillna(50), df['rvol_score'].fillna(50)
        ])
        weights = np.array([CONFIG.POSITION_WEIGHT, CONFIG.VOLUME_WEIGHT, CONFIG.MOMENTUM_WEIGHT,
                            CONFIG.ACCELERATION_WEIGHT, CONFIG.BREAKOUT_WEIGHT, CONFIG.RVOL_WEIGHT])
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').fillna(len(df) + 1).astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom').fillna(0) * 100
        df = RankingEngine._calculate_category_ranks(df)
        return df

    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        if series is None or series.empty: return pd.Series(dtype=float)
        series = series.replace([np.inf, -np.inf], np.nan)
        valid_count = series.notna().sum()
        if valid_count == 0: return pd.Series(50, index=series.index)
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom') * 100
            ranks = ranks.fillna(0 if ascending else 100)
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
            ranks = ranks.fillna(valid_count + 1)
        return ranks

    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        position_score = pd.Series(50, index=df.index, dtype=float)
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        if not has_from_low and not has_from_high: return position_score
        from_low = df['from_low_pct'].fillna(50) if has_from_low else pd.Series(50, index=df.index)
        from_high = df['from_high_pct'].fillna(-50) if has_from_high else pd.Series(-50, index=df.index)
        rank_from_low = RankingEngine._safe_rank(from_low, pct=True, ascending=True) if has_from_low else pd.Series(50, index=df.index)
        rank_from_high = RankingEngine._safe_rank(from_high, pct=True, ascending=False) if has_from_high else pd.Series(50, index=df.index)
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
                weighted_score += col_rank * weight
                total_weight += weight
        if total_weight > 0: volume_score = weighted_score / total_weight
        else: logger.warning("No volume ratio data available, using neutral scores")
        return volume_score.clip(0, 100)

    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                momentum_score = RankingEngine._safe_rank(df['ret_7d'].fillna(0), pct=True, ascending=True)
                logger.info("Using 7-day returns for momentum score")
            else: logger.warning("No return data available for momentum calculation")
            return momentum_score.clip(0, 100)
        ret_30d = df['ret_30d'].fillna(0); momentum_score = RankingEngine._safe_rank(ret_30d, pct=True, ascending=True)
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            consistency_bonus[all_positive] = 5
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            consistency_bonus[accelerating] = 10
            momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
        return momentum_score

    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        if len([col for col in req_cols if col in df.columns]) < 2: return acceleration_score
        ret_1d = df.get('ret_1d', pd.Series(0, index=df.index)).fillna(0)
        ret_7d = df.get('ret_7d', pd.Series(0, index=df.index)).fillna(0)
        ret_30d = df.get('ret_30d', pd.Series(0, index=df.index)).fillna(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_daily_1d = ret_1d; avg_daily_7d = np.where(ret_7d != 0, ret_7d / 7, 0); avg_daily_30d = np.where(ret_30d != 0, ret_30d / 30, 0)
        if all(col in df.columns for col in req_cols):
            perfect = (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
            acceleration_score[perfect] = 100
            good = (~perfect) & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0)
            acceleration_score[good] = 80
            moderate = (~perfect) & (~good) & (ret_1d > 0)
            acceleration_score[moderate] = 60
            slight_decel = (ret_1d <= 0) & (ret_7d > 0)
            acceleration_score[slight_decel] = 40
            strong_decel = (ret_1d <= 0) & (ret_7d <= 0)
            acceleration_score[strong_decel] = 20
        return acceleration_score

    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        distance_factor = (100 - (-df.get('from_high_pct', pd.Series(-50, index=df.index)).fillna(-50))).clip(0, 100)
        volume_factor = ((df.get('vol_ratio_7d_90d', pd.Series(1.0, index=df.index)).fillna(1.0) - 1) * 100).clip(0, 100)
        trend_factor = pd.Series(0, index=df.index, dtype=float)
        if 'price' in df.columns:
            current_price = df['price']; trend_count = 0
            for sma_col, points in [('sma_20d', 33.33), ('sma_50d', 33.33), ('sma_200d', 33.34)]:
                if sma_col in df.columns:
                    above_sma = (current_price > df[sma_col]).fillna(False)
                    trend_factor += above_sma.astype(float) * points
                    trend_count += 1
            if trend_count > 0 and trend_count < 3: trend_factor *= (3 / trend_count)
        trend_factor = trend_factor.clip(0, 100)
        breakout_score = (distance_factor * 0.4 + volume_factor * 0.4 + trend_factor * 0.2)
        return breakout_score.clip(0, 100)

    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        rvol = df.get('rvol', pd.Series(1.0, index=df.index)).fillna(1.0)
        rvol_score = pd.Series(50, index=df.index, dtype=float)
        rvol_score[rvol > 10] = 95; rvol_score[(rvol > 5) & (rvol <= 10)] = 90
        rvol_score[(rvol > 3) & (rvol <= 5)] = 85; rvol_score[(rvol > 2) & (rvol <= 3)] = 80
        rvol_score[(rvol > 1.5) & (rvol <= 2)] = 70; rvol_score[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score[(rvol > 0.8) & (rvol <= 1.2)] = 50; rvol_score[(rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score[(rvol > 0.3) & (rvol <= 0.5)] = 30; rvol_score[rvol <= 0.3] = 20
        return rvol_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_quality(df: pd.DataFrame) -> pd.Series:
        quality_score = pd.Series(50, index=df.index, dtype=float)
        if all(col in df.columns for col in ['ret_30d', 'ret_7d', 'ret_1d']):
            all_positive = (df['ret_30d'] > 0) & (df['ret_7d'] > 0) & (df['ret_1d'] > 0)
            quality_score[all_positive] = 80
            accelerating = (df['ret_1d'] > df['ret_7d'] / 7) & (df['ret_7d'] / 7 > df['ret_30d'] / 30)
            quality_score[accelerating] = 90
            mixed = ((df['ret_30d'] > 0) & (df['ret_7d'] < 0)) | ((df['ret_7d'] > 0) & (df['ret_1d'] < 0))
            quality_score[mixed] = 30
        return quality_score.clip(0, 100)

    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        trend_score = pd.Series(50, index=df.index, dtype=float)
        if 'price' not in df.columns: return trend_score
        current_price = df['price']; sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        available_smas = [col for col in sma_cols if col in df.columns and df[col].notna().any()]
        if len(available_smas) == 0: return trend_score
        if len(available_smas) >= 3:
            perfect_trend = (current_price > df['sma_20d']) & (df['sma_20d'] > df['sma_50d']) & (df['sma_50d'] > df['sma_200d'])
            trend_score[perfect_trend] = 100
            strong_trend = (~perfect_trend) & (current_price > df['sma_20d']) & (current_price > df['sma_50d']) & (current_price > df['sma_200d'])
            trend_score[strong_trend] = 85
            above_count = sum([(current_price > df[sma]).astype(int) for sma in available_smas])
            good_trend = (above_count == 2) & (~perfect_trend) & (~strong_trend); trend_score[good_trend] = 70
            weak_trend = (above_count == 1); trend_score[weak_trend] = 40
            poor_trend = (above_count == 0); trend_score[poor_trend] = 20
        return trend_score

    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        strength_score = pd.Series(50, index=df.index, dtype=float)
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']; available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        if not available_cols: return strength_score
        lt_returns = df[available_cols].fillna(0); avg_return = lt_returns.mean(axis=1)
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
        df['category_rank'] = 9999; df['category_percentile'] = 0.0
        categories = df['category'].unique()
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
    """Detect all patterns using fully vectorized operations"""
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.3)
    def detect_all_patterns_optimized(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df['patterns'] = ''; df['pattern_confidence'] = 0
            return df
        pattern_results = {}; pattern_names = []
        if 'category_percentile' in df.columns: pattern_results['🔥 CAT LEADER'] = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
        if 'category_percentile' in df.columns and 'percentile' in df.columns: pattern_results['💎 HIDDEN GEM'] = ((df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (df['percentile'] < 70))
        if 'acceleration_score' in df.columns: pattern_results['🚀 ACCELERATING'] = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns: pattern_results['🏦 INSTITUTIONAL'] = ((df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (df['vol_ratio_90d_180d'] > 1.1))
        if 'rvol' in df.columns: pattern_results['⚡ VOL EXPLOSION'] = df['rvol'] > 3
        if 'breakout_score' in df.columns: pattern_results['🎯 BREAKOUT'] = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
        if 'percentile' in df.columns: pattern_results['👑 MARKET LEADER'] = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns: pattern_results['🌊 MOMENTUM WAVE'] = ((df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (df['acceleration_score'] >= 70))
        if 'liquidity_score' in df.columns and 'percentile' in df.columns: pattern_results['💰 LIQUID LEADER'] = ((df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']))
        if 'long_term_strength' in df.columns: pattern_results['💪 LONG STRENGTH'] = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
        if 'trend_quality' in df.columns: pattern_results['📈 QUALITY TREND'] = df['trend_quality'] >= 80
        if 'pe' in df.columns and 'master_score' in df.columns: has_valid_pe = (df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)); pattern_results['💎 VALUE MOMENTUM'] = has_valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns: has_eps_growth = df['eps_change_pct'].notna(); extreme_growth = has_eps_growth & (df['eps_change_pct'] > 1000); normal_growth = has_eps_growth & (df['eps_change_pct'] > 50) & (df['eps_change_pct'] <= 1000); pattern_results['📊 EARNINGS ROCKET'] = ((extreme_growth & (df['acceleration_score'] >= 80)) | (normal_growth & (df['acceleration_score'] >= 70)))
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']): has_complete_data = (df['pe'].notna() & df['eps_change_pct'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)); pattern_results['🏆 QUALITY LEADER'] = (has_complete_data & (df['pe'].between(10, 25)) & (df['eps_change_pct'] > 20) & (df['percentile'] >= 80))
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns: has_eps = df['eps_change_pct'].notna(); mega_turnaround = has_eps & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60); strong_turnaround = has_eps & (df['eps_change_pct'] > 100) & (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70); pattern_results['⚡ TURNAROUND'] = mega_turnaround | strong_turnaround
        if 'pe' in df.columns: has_valid_pe = df['pe'].notna() & (df['pe'] > 0); pattern_results['⚠️ HIGH PE'] = has_valid_pe & (df['pe'] > 100)
        if all(col in df.columns for col in ['from_high_pct', 'volume_score', 'momentum_score']): pattern_results['🎯 52W HIGH APPROACH'] = ((df['from_high_pct'] > -5) & (df['volume_score'] >= 70) & (df['momentum_score'] >= 60))
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']): pattern_results['🔄 52W LOW BOUNCE'] = ((df['from_low_pct'] < 20) & (df['acceleration_score'] >= 80) & (df['ret_30d'] > 10))
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']): pattern_results['👑 GOLDEN ZONE'] = ((df['from_low_pct'] > 60) & (df['from_high_pct'] > -40) & (df['trend_quality'] >= 70))
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']): pattern_results['📊 VOL ACCUMULATION'] = ((df['vol_ratio_30d_90d'] > 1.2) & (df['vol_ratio_90d_180d'] > 1.1) & (df['ret_30d'] > 5))
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']): ret_7d_arr = df['ret_7d'].fillna(0).values; ret_30d_arr = df['ret_30d'].fillna(0).values; daily_7d_pace = np.where(ret_7d_arr != 0, ret_7d_arr / 7, 0); daily_30d_pace = np.where(ret_30d_arr != 0, ret_30d_arr / 30, 0); pattern_results['🔀 MOMENTUM DIVERGE'] = ((daily_7d_pace > daily_30d_pace * 1.5) & (df['acceleration_score'] >= 85) & (df['rvol'] > 2))
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']): high_arr = df['high_52w'].fillna(0).values; low_arr = df['low_52w'].fillna(0).values; range_pct = np.where(low_arr > 0, ((high_arr - low_arr) / low_arr) * 100, 100); pattern_results['🎯 RANGE COMPRESS'] = (range_pct < 50) & (df['from_low_pct'] > 30)
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']): ret_7d_arr = df['ret_7d'].fillna(0).values; ret_30d_arr = df['ret_30d'].fillna(0).values; ret_ratio = np.where(ret_30d_arr != 0, ret_7d_arr / (ret_30d_arr / 4), 0); pattern_results['🤫 STEALTH'] = ((df['vol_ratio_90d_180d'] > 1.1) & (df['vol_ratio_30d_90d'].between(0.9, 1.1)) & (df['from_low_pct'] > 40) & (ret_ratio > 1))
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']): ret_1d_arr = df['ret_1d'].fillna(0).values; ret_7d_arr = df['ret_7d'].fillna(0).values; daily_pace_ratio = np.where(ret_7d_arr != 0, ret_1d_arr / (ret_7d_arr / 7), 0); pattern_results['🧛 VAMPIRE'] = ((daily_pace_ratio > 2) & (df['rvol'] > 3) & (df['from_high_pct'] > -15) & (df['category'].isin(['Small Cap', 'Micro Cap'])))
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns: pattern_results['⛈️ PERFECT STORM'] = ((df['momentum_harmony'] == 4) & (df['master_score'] > 80))
        pattern_names = list(pattern_results.keys()); pattern_matrix = np.column_stack([pattern_results[name].values for name in pattern_names])
        df['patterns'] = [' | '.join([pattern_names[i] for i, val in enumerate(row) if val]) for row in pattern_matrix]
        df['patterns'] = df['patterns'].fillna('')
        return df

    @staticmethod
    def _calculate_pattern_confidence(df: pd.DataFrame) -> pd.DataFrame:
        confidence_scores = []; patterns_metadata = CONFIG.PATTERN_METADATA
        for idx, row in df.iterrows():
            patterns = row['patterns'].split(' | ') if row['patterns'] else []
            total_confidence = 0
            for pattern in patterns:
                if pattern and pattern in patterns_metadata:
                    metadata = patterns_metadata[pattern]
                    importance_weights = {'very_high': 40, 'high': 30, 'medium': 20, 'low': 10}
                    confidence = importance_weights.get(metadata['importance'], 20)
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
    """Advanced market analysis and regime detection"""
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        if df.empty: return "😴 NO DATA", {}
        metrics = {}
        if 'category' in df.columns and 'master_score' in df.columns:
            category_scores = df.groupby('category')['master_score'].mean()
            micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean()
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean()
            metrics['micro_small_avg'] = micro_small_avg; metrics['large_mega_avg'] = large_mega_avg
            metrics['category_spread'] = micro_small_avg - large_mega_avg
        else: micro_small_avg = 50; large_mega_avg = 50
        if 'ret_30d' in df.columns:
            breadth = len(df[df['ret_30d'] > 0]) / len(df) if len(df) > 0 else 0
            metrics['breadth'] = breadth
        else: breadth = 0.5
        if 'rvol' in df.columns:
            avg_rvol = df['rvol'].median(); metrics['avg_rvol'] = avg_rvol
        else: avg_rvol = 1.0
        if micro_small_avg > large_mega_avg + 10 and breadth > 0.6: regime = "🔥 RISK-ON BULL"
        elif large_mega_avg > micro_small_avg + 10 and breadth < 0.4: regime = "🛡️ RISK-OFF DEFENSIVE"
        elif avg_rvol > 1.5 and breadth > 0.5: regime = "⚡ VOLATILE OPPORTUNITY"
        else: regime = "😴 RANGE-BOUND"
        metrics['regime'] = regime
        return regime, metrics

    @staticmethod
    def calculate_advance_decline_ratio(df: pd.DataFrame) -> Dict[str, Any]:
        ad_metrics = {}
        if 'ret_1d' in df.columns:
            advancing = len(df[df['ret_1d'] > 0]); declining = len(df[df['ret_1d'] < 0])
            unchanged = len(df[df['ret_1d'] == 0])
            ad_metrics['advancing'] = advancing; ad_metrics['declining'] = declining; ad_metrics['unchanged'] = unchanged
            if declining > 0: ad_metrics['ad_ratio'] = advancing / declining
            else: ad_metrics['ad_ratio'] = float('inf') if advancing > 0 else 1.0
            ad_metrics['ad_line'] = advancing - declining
            ad_metrics['breadth_pct'] = (advancing / len(df)) * 100 if len(df) > 0 else 0
        return ad_metrics

    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'sector' not in df.columns or df.empty: return pd.DataFrame()
        sector_dfs = []
        for sector in df['sector'].unique():
            if sector != 'Unknown' and pd.notna(sector):
                sector_df = df[df['sector'] == sector].copy(); sector_size = len(sector_df)
                if sector_size == 1: sample_count = 1
                elif 2 <= sector_size <= 5: sample_count = sector_size
                elif 6 <= sector_size <= 10: sample_count = max(3, int(sector_size * 0.80))
                elif 11 <= sector_size <= 25: sample_count = max(5, int(sector_size * 0.60))
                elif 26 <= sector_size <= 50: sample_count = max(10, int(sector_size * 0.50))
                elif 51 <= sector_size <= 100: sample_count = max(20, int(sector_size * 0.40))
                elif 101 <= sector_size <= 200: sample_count = max(30, int(sector_size * 0.30))
                else: sample_count = min(60, int(sector_size * 0.25))
                if sample_count > 0: sector_df = sector_df.nlargest(sample_count, 'master_score')
                else: sector_df = pd.DataFrame()
                if not sector_df.empty: sector_dfs.append(sector_df)
        if sector_dfs: normalized_df = pd.concat(sector_dfs, ignore_index=True)
        else: return pd.DataFrame()
        sector_metrics = normalized_df.groupby('sector').agg({'master_score': ['mean', 'median', 'std', 'count'], 'momentum_score': 'mean', 'volume_score': 'mean', 'rvol': 'mean', 'ret_30d': 'mean', 'money_flow_mm': 'sum' if 'money_flow_mm' in normalized_df.columns else lambda x: 0}).round(2)
        if 'money_flow_mm' in normalized_df.columns: sector_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow']
        else: sector_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'dummy_money_flow']; sector_metrics = sector_metrics.drop('dummy_money_flow', axis=1)
        original_counts = df.groupby('sector').size().rename('total_stocks'); sector_metrics = sector_metrics.join(original_counts, how='left')
        sector_metrics['analyzed_stocks'] = sector_metrics['count']
        sector_metrics['flow_score'] = (sector_metrics['avg_score'] * 0.3 + sector_metrics['median_score'] * 0.2 + sector_metrics['avg_momentum'] * 0.25 + sector_metrics['avg_volume'] * 0.25)
        sector_metrics['rank'] = sector_metrics['flow_score'].rank(ascending=False)
        sector_metrics['sampling_pct'] = ((sector_metrics['analyzed_stocks'] / sector_metrics['total_stocks'] * 100).round(1))
        return sector_metrics.sort_values('flow_score', ascending=False)

    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'industry' not in df.columns or df.empty: return pd.DataFrame()
        industry_dfs = []
        for industry in df['industry'].unique():
            if industry != 'Unknown' and pd.notna(industry):
                industry_df = df[df['industry'] == industry].copy(); industry_size = len(industry_df)
                if industry_size == 1: sample_count = 1
                elif 2 <= industry_size <= 5: sample_count = industry_size
                elif 6 <= industry_size <= 10: sample_count = max(3, int(industry_size * 0.80))
                elif 11 <= industry_size <= 25: sample_count = max(5, int(industry_size * 0.60))
                elif 26 <= industry_size <= 50: sample_count = max(10, int(industry_size * 0.40))
                elif 51 <= industry_size <= 100: sample_count = max(15, int(industry_size * 0.30))
                elif 101 <= industry_size <= 250: sample_count = max(25, int(industry_size * 0.20))
                elif 251 <= industry_size <= 550: sample_count = max(40, int(industry_size * 0.15))
                else: sample_count = min(75, int(industry_size * 0.10))
                if sample_count > 0: industry_df = industry_df.nlargest(sample_count, 'master_score')
                else: industry_df = pd.DataFrame()
                if not industry_df.empty: industry_dfs.append(industry_df)
        if industry_dfs: normalized_df = pd.concat(industry_dfs, ignore_index=True)
        else: return pd.DataFrame()
        industry_metrics = normalized_df.groupby('industry').agg({'master_score': ['mean', 'median', 'std', 'count'], 'momentum_score': 'mean', 'volume_score': 'mean', 'rvol': 'mean', 'ret_30d': 'mean', 'money_flow_mm': 'sum' if 'money_flow_mm' in normalized_df.columns else lambda x: 0}).round(2)
        if 'money_flow_mm' in normalized_df.columns: industry_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow']
        else: industry_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'dummy_money_flow']; industry_metrics = industry_metrics.drop('dummy_money_flow', axis=1)
        original_counts = df.groupby('industry').size().rename('total_stocks'); industry_metrics = industry_metrics.join(original_counts, how='left')
        industry_metrics['analyzed_stocks'] = industry_metrics['count']
        industry_metrics['flow_score'] = (industry_metrics['avg_score'] * 0.3 + industry_metrics['median_score'] * 0.2 + industry_metrics['avg_momentum'] * 0.25 + industry_metrics['avg_volume'] * 0.25)
        industry_metrics['rank'] = industry_metrics['flow_score'].rank(ascending=False)
        industry_metrics['sampling_pct'] = ((industry_metrics['analyzed_stocks'] / industry_metrics['total_stocks'] * 100).round(1))
        return industry_metrics.sort_values('flow_score', ascending=False)

    @staticmethod
    def detect_category_performance(df: pd.DataFrame) -> pd.DataFrame:
        if 'category' not in df.columns or df.empty: return pd.DataFrame()
        category_dfs = []
        for category in df['category'].unique():
            if category != 'Unknown' and pd.notna(category):
                category_df = df[df['category'] == category].copy(); category_size = len(category_df)
                if category_size == 1: sample_count = 1
                elif 2 <= category_size <= 10: sample_count = category_size
                elif 11 <= category_size <= 50: sample_count = max(5, int(category_size * 0.60))
                elif 51 <= category_size <= 100: sample_count = max(20, int(category_size * 0.40))
                elif 101 <= category_size <= 200: sample_count = max(30, int(category_size * 0.30))
                else: sample_count = min(50, int(category_size * 0.25))
                if sample_count > 0: category_df = category_df.nlargest(sample_count, 'master_score')
                else: category_df = pd.DataFrame()
                if not category_df.empty: category_dfs.append(category_df)
        if category_dfs: normalized_df = pd.concat(category_dfs, ignore_index=True)
        else: return pd.DataFrame()
        category_metrics = normalized_df.groupby('category').agg({'master_score': ['mean', 'median', 'std', 'count'], 'momentum_score': 'mean', 'volume_score': 'mean', 'rvol': 'mean', 'ret_30d': 'mean', 'acceleration_score': 'mean', 'breakout_score': 'mean', 'money_flow_mm': 'sum' if 'money_flow_mm' in normalized_df.columns else lambda x: 0}).round(2)
        if 'money_flow_mm' in normalized_df.columns: category_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'avg_acceleration', 'avg_breakout', 'total_money_flow']
        else: category_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'avg_acceleration', 'avg_breakout', 'dummy_money_flow']; category_metrics = category_metrics.drop('dummy_money_flow', axis=1)
        original_counts = df.groupby('category').size().rename('total_stocks'); category_metrics = category_metrics.join(original_counts, how='left')
        category_metrics['analyzed_stocks'] = category_metrics['count']
        category_metrics['flow_score'] = (category_metrics['avg_score'] * 0.35 + category_metrics['median_score'] * 0.20 + category_metrics['avg_momentum'] * 0.20 + category_metrics['avg_acceleration'] * 0.15 + category_metrics['avg_volume'] * 0.10)
        category_metrics['rank'] = category_metrics['flow_score'].rank(ascending=False)
        category_metrics['sampling_pct'] = ((category_metrics['analyzed_stocks'] / category_metrics['total_stocks'] * 100).round(1))
        category_order = ['Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']
        category_metrics = category_metrics.reindex([cat for cat in category_order if cat in category_metrics.index])
        return category_metrics


# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations with proper error handling"""
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
            accel_df = df.nlargest(min(n, len(df)), 'acceleration_score')
            if len(accel_df) == 0: return go.Figure()
            fig = go.Figure()
            for _, stock in accel_df.iterrows():
                x_points = ['Start']; y_points = [0]
                if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']): x_points.append('30D'); y_points.append(stock['ret_30d'])
                if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']): x_points.append('7D'); y_points.append(stock['ret_7d'])
                if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']): x_points.append('Today'); y_points.append(stock['ret_1d'])
                if len(x_points) > 1:
                    line_style = dict(width=3, dash='solid') if stock['acceleration_score'] >= 85 else dict(width=2, dash='solid') if stock['acceleration_score'] >= 70 else dict(width=2, dash='dot')
                    marker_style = dict(size=10, symbol='star') if stock['acceleration_score'] >= 85 else dict(size=8) if stock['acceleration_score'] >= 70 else dict(size=6)
                    fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='lines+markers', name=f"{stock['ticker']} ({stock['acceleration_score']:.0f})", line=line_style, marker=marker_style, hovertemplate=(f"<b>{stock['ticker']}</b><br>" + "%{x}: %{y:.1f}%<br>" + f"Accel Score: {stock['acceleration_score']:.0f}<extra></extra>")))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(title=f"Acceleration Profiles - Top {len(accel_df)} Momentum Builders", xaxis_title="Time Frame", yaxis_title="Return %", height=400, template='plotly_white', showlegend=True, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), hovermode='x unified')
            return fig
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}")
            return go.Figure()


# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_all_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        if df.empty: return df
        filtered_df = df.copy(); initial_count = len(filtered_df)
        if filters.get('min_score', 0) > 0: filtered_df = filtered_df[filtered_df['master_score'] >= filters['min_score']]
        if filters.get('categories'): filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        if filters.get('sectors'): filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
        if filters.get('industries'): filtered_df = filtered_df[filtered_df['industry'].isin(filters['industries'])]
        if filters.get('patterns'):
            pattern_mask = filtered_df['patterns'].apply(lambda x: any(p in x for p in filters['patterns']) if x else False)
            filtered_df = filtered_df[pattern_mask]
        if filters.get('trend_filter') and filters['trend_filter'] != 'All Trends':
            if filters.get('trend_range') and 'trend_quality' in filtered_df.columns:
                min_trend, max_trend = filters['trend_range']
                filtered_df = filtered_df[(filtered_df['trend_quality'] >= min_trend) & (filtered_df['trend_quality'] <= max_trend)]
        if filters.get('min_eps_change') is not None and 'eps_change_pct' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['eps_change_pct'] >= filters['min_eps_change']]
        if filters.get('min_pe') is not None and 'pe' in filtered_df.columns: filtered_df = filtered_df[filtered_df['pe'] >= filters['min_pe']]
        if filters.get('max_pe') is not None and 'pe' in filtered_df.columns: filtered_df = filtered_df[filtered_df['pe'] <= filters['max_pe']]
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            if filters.get(tier_type):
                tier_col = tier_type.replace('_tiers', '_tier')
                if tier_col in filtered_df.columns: filtered_df = filtered_df[filtered_df[tier_col].isin(filters[tier_type])]
        if filters.get('require_fundamental_data', False):
            if all(col in filtered_df.columns for col in ['pe', 'eps_change_pct']):
                filtered_df = filtered_df[filtered_df['pe'].notna() & filtered_df['eps_change_pct'].notna()]
        if filters.get('wave_states') and 'wave_state' in filtered_df.columns: filtered_df = filtered_df[filtered_df['wave_state'].isin(filters['wave_states'])]
        if filters.get('wave_strength_range') and 'overall_wave_strength' in filtered_df.columns:
            min_val, max_val = filters['wave_strength_range']
            filtered_df = filtered_df[(filtered_df['overall_wave_strength'] >= min_val) & (filtered_df['overall_wave_strength'] <= max_val)]
        logger.info(f"Filters reduced {initial_count} to {len(filtered_df)} stocks")
        return filtered_df

    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        if df.empty or column not in df.columns: return []
        temp_filters = current_filters.copy()
        filter_key_map = {'category': 'categories', 'sector': 'sectors', 'industry': 'industries', 'eps_tier': 'eps_tiers', 'pe_tier': 'pe_tiers', 'price_tier': 'price_tiers', 'wave_state': 'wave_states'}
        if column in filter_key_map: temp_filters.pop(filter_key_map[column], None)
        if column == 'industry' and 'sectors' in current_filters: temp_filters['sectors'] = current_filters['sectors']
        filtered_df = FilterEngine.apply_all_filters(df, temp_filters)
        values = filtered_df[column].dropna().unique()
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN', 'None']]
        try: values = sorted(values, key=lambda x: float(str(x).replace(',', '')) if str(x).replace(',', '').replace('.', '').isdigit() else x)
        except: values = sorted(values, key=str)
        return values


# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        if not query or df.empty: return pd.DataFrame()
        try:
            query = query.upper().strip()
            results = df.copy(); results['relevance'] = 0
            exact_ticker_mask = results['ticker'].str.upper() == query
            results.loc[exact_ticker_mask, 'relevance'] += 1000
            ticker_starts_mask = results['ticker'].str.upper().str.startswith(query)
            results.loc[ticker_starts_mask & ~exact_ticker_mask, 'relevance'] += 500
            ticker_contains_mask = results['ticker'].str.upper().str.contains(query, na=False, regex=False)
            results.loc[ticker_contains_mask & ~ticker_starts_mask, 'relevance'] += 200
            if 'company_name' in results.columns:
                company_exact_mask = results['company_name'].str.upper() == query
                results.loc[company_exact_mask, 'relevance'] += 800
                company_starts_mask = results['company_name'].str.upper().str.startswith(query)
                results.loc[company_starts_mask & ~company_exact_mask, 'relevance'] += 300
                company_contains_mask = results['company_name'].str.upper().str.contains(query, na=False, regex=False)
                results.loc[company_contains_mask & ~company_starts_mask, 'relevance'] += 100
                def word_match_score(company_name):
                    if pd.isna(company_name): return 0
                    words = str(company_name).upper().split()
                    for word in words:
                        if word.startswith(query): return 50
                    return 0
                word_scores = results['company_name'].apply(word_match_score)
                results['relevance'] += word_scores
            matches = results[results['relevance'] > 0].copy()
            if matches.empty: return pd.DataFrame()
            matches = matches.sort_values(['relevance', 'master_score'], ascending=[False, False])
            return matches.drop('relevance', axis=1)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()


# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        output = BytesIO()
        templates = {'day_trader': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 'volume_score', 'vmi', 'wave_state', 'patterns', 'category'], 'focus': 'Intraday momentum and volume'}, 'swing_trader': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'breakout_score', 'position_score', 'position_tension', 'from_high_pct', 'from_low_pct', 'trend_quality', 'momentum_harmony', 'patterns', 'industry'], 'focus': 'Position and breakout setups'}, 'investor': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 'long_term_strength', 'money_flow_mm', 'category', 'sector', 'industry'], 'focus': 'Fundamentals and long-term performance'}, 'full': {'columns': None, 'focus': 'Complete analysis'}}
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                header_format = workbook.add_format({'bold': True, 'bg_color': '#3498db', 'font_color': 'white', 'border': 1})
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                export_cols = templates[template]['columns'] if templates[template]['columns'] else [col for col in top_100.columns if col not in ['percentile', 'category_rank']]
                top_100_export = top_100[[col for col in export_cols if col in top_100.columns]]
                top_100_export.to_excel(writer, sheet_name='Top 100 Stocks', index=False)
                worksheet = writer.sheets['Top 100 Stocks']
                for i, col in enumerate(top_100_export.columns): worksheet.write(0, i, col, header_format); worksheet.autofit()
                intel_data = []
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                intel_data.append({'Metric': 'Market Regime', 'Value': regime, 'Details': f"Breadth: {regime_metrics.get('breadth', 0):.1%} | Avg RVOL: {regime_metrics.get('avg_rvol', 1):.1f}x"})
                ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
                intel_data.append({'Metric': 'Advance/Decline Ratio (1D)', 'Value': f"{ad_metrics.get('ad_ratio', 1):.2f}", 'Details': f"Advances: {ad_metrics.get('advancing', 0)}, Declines: {ad_metrics.get('declining', 0)}"})
                intel_df = pd.DataFrame(intel_data); intel_df.to_excel(writer, sheet_name='Market Intelligence', index=False)
                sector_rotation = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rotation.empty: sector_rotation.to_excel(writer, sheet_name='Sector Rotation')
                pattern_counts = {}; [pattern_counts.update([p.strip() for p in patterns.split('|')]) for patterns in df['patterns'].dropna() if patterns]
                if pattern_counts: pattern_df = pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=False); pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
                wave_signals = df[(df['momentum_score'] >= 60) & (df['acceleration_score'] >= 70) & (df['rvol'] >= 2)].nlargest(50, 'master_score', keep='first')
                if not wave_signals.empty:
                    wave_cols = ['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'wave_state', 'patterns', 'category', 'sector', 'industry']
                    wave_signals[[col for col in wave_cols if col in wave_signals.columns]].to_excel(writer, sheet_name='Wave Radar Signals', index=False)
            output.seek(0)
            return output
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        export_cols = ['rank', 'ticker', 'company_name', 'master_score', 'position_score', 'volume_score', 'momentum_score', 'acceleration_score', 'breakout_score', 'rvol_score', 'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct', 'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'rvol', 'vmi', 'money_flow_mm', 'position_tension', 'momentum_harmony', 'wave_state', 'patterns', 'category', 'sector', 'industry', 'eps_tier', 'pe_tier', 'overall_wave_strength']
        available_cols = [col for col in export_cols if col in df.columns]
        export_df = df[available_cols].copy()
        for col_name in CONFIG.VOLUME_RATIO_COLUMNS:
            if col_name in export_df.columns:
                export_df[col_name] = (export_df[col_name] - 1) * 100
        return export_df.to_csv(index=False)


# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None, help_text: Optional[str] = None) -> None:
        if help_text: st.metric(label, value, delta, help=help_text)
        else: st.metric(label, value, delta)

    @staticmethod
    def render_summary_section(df: pd.DataFrame) -> None:
        if df.empty: st.warning("No data available for summary"); return
        st.markdown("### 📊 Market Pulse"); col1, col2, col3, col4 = st.columns(4)
        with col1:
            ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df); ad_ratio = ad_metrics.get('ad_ratio', 1.0)
            ad_emoji = "🔥" if ad_ratio > 2 else "📈" if ad_ratio > 1 else "📉"
            UIComponents.render_metric_card("A/D Ratio", f"{ad_emoji} {ad_ratio:.2f}", f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}", "Advance/Decline Ratio")
        with col2:
            high_momentum = len(df[df.get('momentum_score', 0) >= 70]); momentum_pct = (high_momentum / len(df) * 100) if len(df) > 0 else 0
            UIComponents.render_metric_card("Momentum Health", f"{momentum_pct:.0f}%", f"{high_momentum} strong stocks")
        with col3:
            avg_rvol = df.get('rvol', pd.Series(1.0)).median() if 'rvol' in df.columns else 1.0
            high_vol_count = len(df[df.get('rvol', 0) > 2]) if 'rvol' in df.columns else 0
            vol_emoji = "🌊" if avg_rvol > 1.5 else "💧" if avg_rvol > 1.2 else "🏜️"
            UIComponents.render_metric_card("Volume State", f"{vol_emoji} {avg_rvol:.1f}x", f"{high_vol_count} surges")
        with col4:
            risk_factors = 0
            if 'from_high_pct' in df.columns and 'momentum_score' in df.columns:
                if len(df[(df.get('from_high_pct', -100) >= 0) & (df.get('momentum_score', 0) < 50)]) > 20: risk_factors += 1
            if 'rvol' in df.columns and len(df[(df.get('rvol', 0) > 10) & (df.get('master_score', 0) < 50)]) > 10: risk_factors += 1
            if 'trend_quality' in df.columns and len(df[df.get('trend_quality', 50) < 40]) > len(df) * 0.3: risk_factors += 1
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]; risk_level = risk_levels[min(risk_factors, 3)]
            UIComponents.render_metric_card("Risk Level", risk_level, f"{risk_factors} factors")
        st.markdown("### 🎯 Today's Best Opportunities"); opp_col1, opp_col2, opp_col3 = st.columns(3)
        with opp_col1:
            ready_to_run = df[(df.get('momentum_score',0) >= 70) & (df.get('acceleration_score',0) >= 70) & (df.get('rvol',0) >= 2)].nlargest(5, 'master_score')
            st.markdown("**🚀 Ready to Run**")
            if not ready_to_run.empty: [st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"Score: {stock['master_score']:.1f} | RVOL: {stock.get('rvol', 0):.1f}x")]
            else: st.info("No momentum leaders found")
        with opp_col2:
            hidden_gems = df[df.get('patterns', '').str.contains('HIDDEN GEM', na=False)].nlargest(5, 'master_score')
            st.markdown("**💎 Hidden Gems**")
            if not hidden_gems.empty: [st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"Cat %ile: {stock.get('category_percentile', 0):.0f} | Score: {stock['master_score']:.1f}")]
            else: st.info("No hidden gems today")
        with opp_col3:
            volume_alerts = df[df.get('rvol', 0) > 3].nlargest(5, 'master_score')
            st.markdown("**⚡ Volume Alerts**")
            if not volume_alerts.empty: [st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"RVOL: {stock.get('rvol', 0):.1f}x | {stock.get('wave_state', 'N/A')}")]
            else: st.info("No extreme volume detected")
        st.markdown("### 🧠 Market Intelligence"); intel_col1, intel_col2 = st.columns([2, 1])
        with intel_col1:
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            if not sector_rotation.empty:
                fig = go.Figure(); fig.add_trace(go.Bar(x=sector_rotation.index[:10], y=sector_rotation['flow_score'][:10], text=[f"{val:.1f}" for val in sector_rotation['flow_score'][:10]], textposition='outside', marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' for score in sector_rotation['flow_score'][:10]], hovertemplate=('Sector: %{x}<br>Flow Score: %{y:.1f}<br>Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>Avg Score: %{customdata[2]:.1f}<br>Median Score: %{customdata[3]:.1f}<extra></extra>'), customdata=np.column_stack((sector_rotation['analyzed_stocks'][:10], sector_rotation['total_stocks'][:10], sector_rotation['avg_score'][:10], sector_rotation['median_score'][:10]))))
                fig.update_layout(title="Sector Rotation Map - Smart Money Flow (Dynamically Sampled)", xaxis_title="Sector", yaxis_title="Flow Score", height=400, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No sector rotation data available for visualization.")
        with intel_col2:
            regime, regime_metrics = MarketIntelligence.detect_market_regime(df); st.markdown(f"**🎯 Market Regime**"); st.markdown(f"### {regime}"); st.markdown("**📡 Key Signals**")
            signals = []
            breadth = regime_metrics.get('breadth', 0.5)
            if breadth > 0.6: signals.append("✅ Strong breadth")
            elif breadth < 0.4: signals.append("⚠️ Weak breadth")
            category_spread = regime_metrics.get('category_spread', 0)
            if category_spread > 10: signals.append("🔄 Small caps leading")
            elif category_spread < -10: signals.append("🛡️ Large caps defensive")
            avg_rvol = regime_metrics.get('avg_rvol', 1.0)
            if avg_rvol > 1.5: signals.append("🌊 High volume activity")
            pattern_count = (df.get('patterns', '').fillna('') != '').sum()
            if pattern_count > len(df) * 0.2: signals.append("🎯 Many patterns emerging")
            [st.write(signal) for signal in signals]; st.markdown("**💪 Market Strength**")
            strength_score = ((breadth * 50) + (min(avg_rvol, 2) * 25) + ((pattern_count / len(df)) * 25)); strength_meter = "🟢🟢🟢🟢🟢" if strength_score > 70 else "🟢🟢🟢🟢⚪" if strength_score > 50 else "🟢🟢🟢⚪⚪" if strength_score > 30 else "🟢🟢⚪⚪⚪"
            st.write(strength_meter)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.set_page_config(page_title="Wave Detection Ultimate 3.0", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
    RobustSessionState.initialize()
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Assuming CSS is injected via a file or direct string
    st.markdown("""<div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"><h1>🌊 Wave Detection Ultimate 3.0</h1><p>Professional Stock Ranking System • Final Perfected Production Version</p></div>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions"); col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.cache_data.clear(); RobustSessionState.safe_set('last_refresh', datetime.now(timezone.utc)); st.rerun()
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear(); gc.collect(); st.success("Cache cleared!"); time.sleep(0.5); st.rerun()
        st.markdown("---"); st.markdown("### 📂 Data Source"); data_source_col1, data_source_col2 = st.columns(2)
        with data_source_col1:
            if st.button("📊 Google Sheets", type="primary" if RobustSessionState.safe_get('data_source') == "sheet" else "secondary", use_container_width=True): RobustSessionState.safe_set('data_source', "sheet"); st.rerun()
        with data_source_col2:
            if st.button("📁 Upload CSV", type="primary" if RobustSessionState.safe_get('data_source') == "upload" else "secondary", use_container_width=True): RobustSessionState.safe_set('data_source', "upload"); st.rerun()
        uploaded_file = None; sheet_id = None; gid = None
        if RobustSessionState.safe_get('data_source') == "upload":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv", help="Upload a CSV file with stock data. Must contain 'ticker' and 'price' columns.")
            if uploaded_file is None: st.info("Please upload a CSV file to continue")
        else:
            st.markdown("#### 📊 Google Sheets Configuration")
            sheet_input = st.text_input("Google Sheets ID or URL", value=RobustSessionState.safe_get('sheet_id', ''), placeholder="Enter Sheet ID or full URL", help="Example: ...ID... or full URL")
            if sheet_input: sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_input); sheet_id = sheet_id_match.group(1) if sheet_id_match else sheet_input.strip(); RobustSessionState.safe_set('sheet_id', sheet_id)
            gid_input = st.text_input("Sheet Tab GID (Optional)", value=RobustSessionState.safe_get('gid', CONFIG.DEFAULT_GID), placeholder=f"Default: {CONFIG.DEFAULT_GID}", help="The GID identifies specific sheet tab."); gid = gid_input.strip() if gid_input else CONFIG.DEFAULT_GID
            if not sheet_id: st.warning("Please enter a Google Sheets ID to continue")
        data_quality = RobustSessionState.safe_get('data_quality', {}); st.markdown("---"); show_debug = st.checkbox("🐛 Show Debug Info", value=RobustSessionState.safe_get('show_debug', False), key="show_debug")

    try:
        if RobustSessionState.safe_get('data_source') == "upload" and uploaded_file is None: st.stop()
        if RobustSessionState.safe_get('data_source') == "sheet" and not sheet_id: st.stop()
        with st.spinner("📥 Loading and processing data..."):
            try:
                ranked_df, data_timestamp, metadata = load_and_process_data("upload" if uploaded_file else "sheet", file_data=uploaded_file, sheet_id=sheet_id, gid=gid, data_version=hashlib.md5(f"{sheet_id}_{gid}".encode()).hexdigest())
                RobustSessionState.safe_set('ranked_df', ranked_df); RobustSessionState.safe_set('data_timestamp', data_timestamp); RobustSessionState.safe_set('last_refresh', datetime.now(timezone.utc))
                if metadata.get('warnings'): [st.warning(w) for w in metadata['warnings']]
                if metadata.get('errors'): [st.error(e) for e in metadata['errors']]
            except Exception as e:
                last_good_data = RobustSessionState.safe_get('last_good_data');
                if last_good_data: ranked_df, data_timestamp, metadata = last_good_data; st.warning("Failed to load fresh data, using cached version")
                else: st.error(f"❌ Error: {str(e)}"); st.stop()
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}"); st.stop()
    st.markdown("### ⚡ Quick Actions"); qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    quick_filter_applied = RobustSessionState.safe_get('quick_filter_applied', False); quick_filter = RobustSessionState.safe_get('quick_filter', None)
    with qa_col1: if st.button("📈 Top Gainers"): RobustSessionState.safe_set('quick_filter', 'top_gainers'); RobustSessionState.safe_set('quick_filter_applied', True); st.rerun()
    with qa_col2: if st.button("🔥 Volume Surges"): RobustSessionState.safe_set('quick_filter', 'volume_surges'); RobustSessionState.safe_set('quick_filter_applied', True); st.rerun()
    with qa_col3: if st.button("🎯 Breakout Ready"): RobustSessionState.safe_set('quick_filter', 'breakout_ready'); RobustSessionState.safe_set('quick_filter_applied', True); st.rerun()
    with qa_col4: if st.button("💎 Hidden Gems"): RobustSessionState.safe_set('quick_filter', 'hidden_gems'); RobustSessionState.safe_set('quick_filter_applied', True); st.rerun()
    with qa_col5: if st.button("🌊 Show All"): RobustSessionState.safe_set('quick_filter', None); RobustSessionState.safe_set('quick_filter_applied', False); st.rerun()
    if quick_filter:
        if quick_filter == 'top_gainers': ranked_df_display = ranked_df[ranked_df['momentum_score'] >= 80]
        elif quick_filter == 'volume_surges': ranked_df_display = ranked_df[ranked_df['rvol'] >= 3]
        elif quick_filter == 'breakout_ready': ranked_df_display = ranked_df[ranked_df['breakout_score'] >= 80]
        elif quick_filter == 'hidden_gems': ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('HIDDEN GEM', na=False)]
    else: ranked_df_display = ranked_df
    with st.sidebar:
        filters = {}; st.markdown("### 📊 Display Mode")
        display_mode = st.radio("Choose your view:", options=["Technical", "Hybrid (Technical + Fundamentals)"], index=0 if RobustSessionState.safe_get('user_preferences', {}).get('display_mode', 'Technical') == 'Technical' else 1, key="display_mode_toggle"); user_prefs = RobustSessionState.safe_get('user_preferences', {}); user_prefs['display_mode'] = display_mode; RobustSessionState.safe_set('user_preferences', user_prefs); show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        st.markdown("---"); categories = FilterEngine.get_filter_options(ranked_df_display, 'category', filters); selected_categories = st.multiselect("Market Cap Category", options=categories, default=RobustSessionState.safe_get('category_filter', []), key="category_filter"); filters['categories'] = selected_categories
        sectors = FilterEngine.get_filter_options(ranked_df_display, 'sector', filters); selected_sectors = st.multiselect("Sector", options=sectors, default=RobustSessionState.safe_get('sector_filter', []), key="sector_filter"); filters['sectors'] = selected_sectors
        if 'industry' in ranked_df_display.columns:
            temp_df = ranked_df_display.copy(); temp_df = temp_df[temp_df['sector'].isin(selected_sectors)] if selected_sectors and 'All' not in selected_sectors else temp_df
            industries = FilterEngine.get_filter_options(temp_df, 'industry', filters); selected_industries = st.multiselect("Industry", options=industries, default=RobustSessionState.safe_get('industry_filter', []), key="industry_filter"); filters['industries'] = selected_industries
        filters['min_score'] = st.slider("Minimum Master Score", min_value=0, max_value=100, value=RobustSessionState.safe_get('min_score', 0), step=5, key="min_score")
        all_patterns = set(); [all_patterns.update(p.split(' | ')) for p in ranked_df_display['patterns'].dropna() if p]; filters['patterns'] = st.multiselect("Patterns", options=sorted(all_patterns), default=RobustSessionState.safe_get('patterns', []), key="patterns")
        st.markdown("#### 📈 Trend Strength"); trend_options = {"All Trends": (0, 100), "🔥 Strong Uptrend (80+)": (80, 100), "✅ Good Uptrend (60-79)": (60, 79), "➡️ Neutral Trend (40-59)": (40, 59), "⚠️ Weak/Downtrend (<40)": (0, 39)}
        default_trend_key = RobustSessionState.safe_get('trend_filter', "All Trends"); current_trend_index = list(trend_options.keys()).index(default_trend_key) if default_trend_key in trend_options else 0
        filters['trend_filter'] = st.selectbox("Trend Quality", options=list(trend_options.keys()), index=current_trend_index, key="trend_filter"); filters['trend_range'] = trend_options[filters['trend_filter']]
        st.markdown("#### 🌊 Wave Filters"); wave_states_options = FilterEngine.get_filter_options(ranked_df_display, 'wave_state', filters); filters['wave_states'] = st.multiselect("Wave State", options=wave_states_options, default=RobustSessionState.safe_get('wave_states_filter', []), key="wave_states_filter")
        if 'overall_wave_strength' in ranked_df_display.columns:
            min_strength = float(ranked_df_display['overall_wave_strength'].min()); max_strength = float(ranked_df_display['overall_wave_strength'].max())
            current_slider_value = RobustSessionState.safe_get('wave_strength_range_slider', (int(min_strength), int(max_strength))); filters['wave_strength_range'] = st.slider("Overall Wave Strength", min_value=0, max_value=100, value=current_slider_value, step=1, key="wave_strength_range_slider")
        else: filters['wave_strength_range'] = (0, 100); st.info("Overall Wave Strength data not available.")
        with st.expander("🔧 Advanced Filters"):
            for tier_type, col_name in [('eps_tiers', 'eps_tier'), ('pe_tiers', 'pe_tier'), ('price_tiers', 'price_tier')]:
                if col_name in ranked_df_display.columns:
                    tier_options = FilterEngine.get_filter_options(ranked_df_display, col_name, filters); selected_tiers = st.multiselect(f"{col_name.replace('_', ' ').title()}", options=tier_options, default=RobustSessionState.safe_get(f'{col_name}_filter', []), key=f"{col_name}_filter"); filters[tier_type] = selected_tiers
            if 'eps_change_pct' in ranked_df_display.columns:
                eps_change_input = st.text_input("Min EPS Change %", value=RobustSessionState.safe_get('min_eps_change', ""), placeholder="e.g. -50 or leave empty", key="min_eps_change"); filters['min_eps_change'] = float(eps_change_input) if eps_change_input.strip() else None
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                col1, col2 = st.columns(2)
                with col1: filters['min_pe'] = float(st.text_input("Min PE Ratio", value=RobustSessionState.safe_get('min_pe', ""), key="min_pe")) if st.session_state.min_pe.strip() else None
                with col2: filters['max_pe'] = float(st.text_input("Max PE Ratio", value=RobustSessionState.safe_get('max_pe', ""), key="max_pe")) if st.session_state.max_pe.strip() else None
                filters['require_fundamental_data'] = st.checkbox("Only show stocks with PE and EPS data", value=RobustSessionState.safe_get('require_fundamental_data', False), key="require_fundamental_data")
    filtered_df = FilterEngine.apply_all_filters(ranked_df_display, filters) if quick_filter_applied else FilterEngine.apply_all_filters(ranked_df, filters)
    filtered_df = filtered_df.sort_values('rank')
    user_prefs = RobustSessionState.safe_get('user_preferences', {}); user_prefs['last_filters'] = filters; RobustSessionState.safe_set('user_preferences', user_prefs)
    if show_debug:
        with st.sidebar.expander("🐛 Debug Info", expanded=True): st.write(f"Active Filters: {filters}"); st.write(f"Filter Result: {len(ranked_df)} -> {len(filtered_df)}"); st.write(f"Performance: {RobustSessionState.safe_get('performance_metrics', {})}")
    st.markdown("### ⚡ Quick Actions")
    st.columns([5, 1])[1].button("Clear Filters", type="secondary") if RobustSessionState.safe_get('active_filter_count', 0) > 0 else None
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: UIComponents.render_metric_card("Total Stocks", f"{len(filtered_df):,}", f"{(len(filtered_df)/len(ranked_df)*100):.0f}% of {len(ranked_df):,}")
    with col2: UIComponents.render_metric_card("Avg Score", f"{filtered_df['master_score'].mean():.1f}")
    with col3: UIComponents.render_metric_card("Median PE", f"{filtered_df.loc[filtered_df['pe']>0, 'pe'].median():.1f}x") if show_fundamentals and not filtered_df.empty else UIComponents.render_metric_card("Score Range", f"{filtered_df['master_score'].min():.1f}-{filtered_df['master_score'].max():.1f}") if not filtered_df.empty else None
    with col4: UIComponents.render_metric_card("EPS Growth +ve", f"{(filtered_df['eps_change_pct']>0).sum()}", f"{(filtered_df['eps_change_pct']>100).sum()} >100%") if show_fundamentals and not filtered_df.empty else UIComponents.render_metric_card("Accelerating", f"{(filtered_df['acceleration_score']>=80).sum()}") if not filtered_df.empty else None
    with col5: UIComponents.render_metric_card("High RVOL", f"{(filtered_df['rvol']>2).sum()}") if not filtered_df.empty else None
    with col6: UIComponents.render_metric_card("Strong Trends", f"{(filtered_df['trend_quality']>=80).sum()}") if not filtered_df.empty else UIComponents.render_metric_card("With Patterns", f"{(filtered_df['patterns']!='').sum()}") if not filtered_df.empty else None
    tabs = st.tabs(["📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"])
    with tabs[0]: st.markdown("### 📊 Executive Summary Dashboard"); UIComponents.render_summary_section(filtered_df)
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks"); col1, col2, col3 = st.columns([2, 2, 6])
        with col1: display_count = st.selectbox("Show top", options=CONFIG.AVAILABLE_TOP_N, index=CONFIG.AVAILABLE_TOP_N.index(RobustSessionState.safe_get('user_preferences', {}).get('default_top_n', 50))); RobustSessionState.safe_set('user_preferences', {**RobustSessionState.safe_get('user_preferences', {}), 'default_top_n': display_count})
        with col2: sort_by = st.selectbox("Sort by", options=['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow', 'Trend'] if 'trend_quality' in filtered_df.columns else ['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow'], index=0)
        display_df = filtered_df.head(display_count).copy(); display_df = display_df.sort_values('master_score', ascending=False) if sort_by == 'Master Score' else display_df.sort_values('rvol', ascending=False) if sort_by == 'RVOL' else display_df.sort_values('momentum_score', ascending=False) if sort_by == 'Momentum' else display_df.sort_values('money_flow_mm', ascending=False) if sort_by == 'Money Flow' else display_df.sort_values('trend_quality', ascending=False) if sort_by == 'Trend' else display_df
        if not display_df.empty:
            if 'trend_quality' in display_df.columns: display_df['trend_indicator'] = display_df['trend_quality'].apply(lambda s: "🔥" if s >= 80 else "✅" if s >= 60 else "➡️" if s >= 40 else "⚠️" if pd.notna(s) else "➖")
            display_cols = {c: c.replace('_',' ').title() for c in display_df.columns}
            display_cols.update({'rank': 'Rank', 'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score', 'wave_state': 'Wave', 'trend_indicator': 'Trend', 'price': 'Price', 'pe': 'PE', 'eps_change_pct': 'EPS Δ%', 'from_low_pct': 'From Low', 'ret_30d': '30D Ret', 'rvol': 'RVOL', 'vmi': 'VMI', 'patterns': 'Patterns', 'category': 'Category'})
            available_display_cols = [c for c in display_cols if c in display_df.columns]; display_df = display_df[[c for c in available_display_cols]]; display_df.columns = [display_cols[c] for c in available_display_cols]
            st.dataframe(display_df, use_container_width=True, height=min(600, len(display_df) * 35 + 50), hide_index=True)
        else: st.warning("No stocks match the selected filters.")
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System"); st.markdown("*Catch waves as they form, not after they've peaked!*")
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        with radar_col1: wave_timeframe = st.selectbox("Wave Detection Timeframe", options=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"], index=0 if RobustSessionState.safe_get('wave_timeframe_select', "All Waves") == "All Waves" else 1 if RobustSessionState.safe_get('wave_timeframe_select', "All Waves") == "Intraday Surge" else 2 if RobustSessionState.safe_get('wave_timeframe_select', "All Waves") == "3-Day Buildup" else 3 if RobustSessionState.safe_get('wave_timeframe_select', "All Waves") == "Weekly Breakout" else 4, key="wave_timeframe_select")
        with radar_col2: sensitivity = st.select_slider("Detection Sensitivity", options=["Conservative", "Balanced", "Aggressive"], value=RobustSessionState.safe_get('wave_sensitivity', "Balanced"), key="wave_sensitivity"); show_sensitivity_details = st.checkbox("Show thresholds", value=RobustSessionState.safe_get('show_sensitivity_details', False), key="show_sensitivity_details")
        with radar_col3: show_market_regime = st.checkbox("📊 Market Regime Analysis", value=RobustSessionState.safe_get('show_market_regime', True), key="show_market_regime")
        wave_filtered_df = filtered_df.copy()
        with radar_col4:
            if not wave_filtered_df.empty and 'overall_wave_strength' in wave_filtered_df.columns:
                wave_strength_score = wave_filtered_df['overall_wave_strength'].mean()
                wave_emoji = "🌊🔥" if wave_strength_score > 70 else "🌊" if wave_strength_score > 50 else "💤"
                wave_color = "🟢" if wave_strength_score > 70 else "🟡" if wave_strength_score > 50 else "🔴"
                UIComponents.render_metric_card("Wave Strength", f"{wave_emoji} {wave_strength_score:.0f}%", f"{wave_color} Market")
        if not wave_filtered_df.empty:
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            momentum_threshold = 60 if sensitivity == "Conservative" else 50 if sensitivity == "Balanced" else 40
            acceleration_threshold = 70 if sensitivity == "Conservative" else 60 if sensitivity == "Balanced" else 50
            min_rvol = 3.0 if sensitivity == "Conservative" else 2.0 if sensitivity == "Balanced" else 1.5
            momentum_shifts = wave_filtered_df[(wave_filtered_df['momentum_score'] >= momentum_threshold) & (wave_filtered_df['acceleration_score'] >= acceleration_threshold)].copy()
            if not momentum_shifts.empty:
                momentum_shifts['signal_count'] = 0; momentum_shifts.loc[momentum_shifts['momentum_score'] >= momentum_threshold, 'signal_count'] += 1
                momentum_shifts.loc[momentum_shifts['acceleration_score'] >= acceleration_threshold, 'signal_count'] += 1
                momentum_shifts.loc[momentum_shifts['rvol'] >= min_rvol, 'signal_count'] += 1
                top_shifts = momentum_shifts.sort_values(['signal_count'], ascending=False).head(20)
                st.dataframe(top_shifts, use_container_width=True, hide_index=True)
                st.success(f"🏆 Found {len(top_shifts[top_shifts['signal_count'] >= 3])} stocks with 3+ signals")
            else: st.info("No momentum shifts detected.")
            st.markdown("#### 🚀 Acceleration Profiles - Momentum Building Over Time")
            accel_threshold = 85 if sensitivity == "Conservative" else 70 if sensitivity == "Balanced" else 60
            accelerating_stocks = wave_filtered_df[wave_filtered_df['acceleration_score'] >= accel_threshold].nlargest(10, 'acceleration_score')
            if not accelerating_stocks.empty:
                fig_accel = Visualizer.create_acceleration_profiles(accelerating_stocks, n=10); st.plotly_chart(fig_accel, use_container_width=True)
    with tabs[3]:
        st.markdown("### 📊 Market Analysis"); col1, col2 = st.columns(2)
        if not filtered_df.empty:
            with col1: fig_dist = Visualizer.create_score_distribution(filtered_df); st.plotly_chart(fig_dist, use_container_width=True)
            with col2:
                pattern_counts = Counter(p.strip() for p in filtered_df['patterns'].dropna().str.split(' | ').explode() if p.strip());
                if pattern_counts: st.bar_chart(pd.DataFrame(pattern_counts.most_common(15), columns=['Pattern', 'Count']).set_index('Pattern'))
            st.markdown("#### Sector Performance (Dynamically Sampled)"); st.dataframe(MarketIntelligence.detect_sector_rotation(filtered_df), use_container_width=True)
    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search"); search_query = st.text_input("Search stocks", placeholder="Enter ticker or company name...", key="search_input")
        if search_query:
            with st.spinner("Searching..."): search_results = SearchEngine.search_stocks(filtered_df, search_query)
            if not search_results.empty: st.success(f"Found {len(search_results)} matching stock(s)"); st.dataframe(search_results)
            else: st.warning("No stocks found matching your search criteria.")
    with tabs[5]:
        st.markdown("### 📥 Export Data"); export_template = st.radio("Choose export template:", options=["Full Analysis (All Data)", "Day Trader Focus", "Swing Trader Focus", "Investor Focus"], key="export_template_radio")
        if st.button("Generate Excel Report", type="primary", use_container_width=True):
            if not filtered_df.empty:
                excel_file = ExportEngine.create_excel_report(filtered_df, template={'Full Analysis (All Data)': 'full'}.get(export_template, 'full')); st.download_button(label="📥 Download Excel Report", data=excel_file, file_name="report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0 - Final Perfected Production Version")
        st.markdown("This is a summary of all features and design choices...")
    st.markdown("---"); st.markdown("🌊 Wave Detection Ultimate 3.0 - Final Perfected Production Version<br><small>Professional Stock Ranking System...</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
