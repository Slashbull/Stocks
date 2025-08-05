"""
Wave Detection Ultimate 3.0 - FINAL PERFECTED PRODUCTION VERSION
===============================================================
Professional Stock Ranking System with Advanced Analytics
All bugs fixed, optimized for Streamlit Community Cloud
Enhanced with perfect filtering system and robust error handling

Version: 3.1.0-FINAL-PERFECTED
Last Updated: December 2024
Status: PRODUCTION READY - PERMANENTLY LOCKED
"""

# ============================================
# IMPORTS AND SETUP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from functools import wraps
import time
from io import BytesIO
import warnings
import gc
import re
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from collections import defaultdict

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Performance optimizations
np.seterr(all='ignore')
pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:.2f}'.format

# Set random seed for reproducibility
np.random.seed(42)

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

# Performance tracking storage
performance_stats = defaultdict(list)

def log_performance(operation: str, duration: float):
    performance_stats[operation].append(duration)
    if len(performance_stats[operation]) > 100:
        performance_stats[operation] = performance_stats[operation][-100:]
    
    if duration > 1.0:
        logger.warning(f"{operation} took {duration:.2f}s")

# ============================================
# ROBUST SESSION STATE MANAGER
# ============================================

class RobustSessionState:
    """Bulletproof session state management - prevents all KeyErrors"""
    
    STATE_DEFAULTS = {
        'wd_search_query': "",
        'wd_last_refresh': None,
        'wd_data_source': "sheet",
        'wd_sheet_id': "",
        'wd_gid': "1823439984",
        'wd_user_preferences': {
            'default_top_n': 50,
            'display_mode': 'Technical',
            'last_filters': {}
        },
        'wd_filters': {},
        'wd_active_filter_count': 0,
        'wd_quick_filter': None,
        'wd_quick_filter_applied': False,
        'wd_show_debug': False,
        'wd_performance_metrics': {},
        'wd_data_quality': {},
        'wd_last_good_data': None,
        'wd_session_id': None,
        'wd_session_start': None,
        'wd_validation_stats': defaultdict(int),
        'wd_trigger_clear': False,
        
        'wd_category_filter': [],
        'wd_sector_filter': [],
        'wd_industry_filter': [],
        'wd_min_score': 0,
        'wd_patterns': [],
        'wd_trend_filter': "All Trends",
        'wd_eps_tier_filter': [],
        'wd_pe_tier_filter': [],
        'wd_price_tier_filter': [],
        'wd_min_eps_change': "",
        'wd_min_pe': "",
        'wd_max_pe': "",
        'wd_require_fundamental_data': False,
        'wd_wave_states_filter': [],
        'wd_wave_strength_range_slider': (0, 100),
        'wd_show_sensitivity_details': False,
        'wd_show_market_regime': True,
        'wd_wave_timeframe_select': "All Waves",
        'wd_wave_sensitivity': "Balanced",
        'wd_export_template_radio': "Full Analysis (All Data)",
        'wd_display_mode_toggle': "Technical",
        'wd_current_page_rankings': 0,
        
        'wd_ranked_df': None,
        'wd_data_timestamp': None,
        'wd_search_input': ""
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
                if key == 'wd_last_refresh' and default_value is None:
                    st.session_state[key] = datetime.now(timezone.utc)
                elif key == 'wd_session_start' and default_value is None:
                    st.session_state[key] = datetime.now(timezone.utc)
                elif key == 'wd_session_id' and default_value is None:
                    st.session_state[key] = hashlib.md5(
                        f"{datetime.now()}{np.random.rand()}".encode()
                    ).hexdigest()[:8]
                else:
                    st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        filter_keys = [
            'wd_category_filter', 'wd_sector_filter', 'wd_industry_filter', 'wd_eps_tier_filter',
            'wd_pe_tier_filter', 'wd_price_tier_filter', 'wd_patterns',
            'wd_min_score', 'wd_trend_filter', 'wd_min_eps_change',
            'wd_min_pe', 'wd_max_pe', 'wd_require_fundamental_data',
            'wd_quick_filter', 'wd_quick_filter_applied',
            'wd_wave_states_filter', 'wd_wave_strength_range_slider',
            'wd_show_sensitivity_details', 'wd_show_market_regime',
            'wd_wave_timeframe_select', 'wd_wave_sensitivity', 'wd_search_input',
            'wd_current_page_rankings'
        ]
        
        for key in filter_keys:
            if key in RobustSessionState.STATE_DEFAULTS:
                RobustSessionState.safe_set(key, RobustSessionState.STATE_DEFAULTS[key])
        
        RobustSessionState.safe_set('wd_filters', {})
        RobustSessionState.safe_set('wd_active_filter_count', 0)
        RobustSessionState.safe_set('wd_trigger_clear', False)

    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        return {
            'session_id': RobustSessionState.safe_get('wd_session_id', 'unknown'),
            'start_time': RobustSessionState.safe_get('wd_session_start', datetime.now()),
            'duration': (datetime.now() - RobustSessionState.safe_get('wd_session_start', datetime.now())).seconds,
            'data_source': RobustSessionState.safe_get('wd_data_source', 'unknown'),
            'stocks_loaded': len(RobustSessionState.safe_get('wd_ranked_df', [])),
            'active_filters': RobustSessionState.safe_get('wd_active_filter_count', 0)
        }

# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class Config:
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM/edit?usp=sharing"
    DEFAULT_GID: str = "1823439984"
    CSV_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    
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
        "category_leader": 90, "hidden_gem": 80, "acceleration": 85,
        "institutional": 75, "vol_explosion": 95, "breakout_ready": 80,
        "market_leader": 95, "momentum_wave": 75, "liquid_leader": 80,
        "long_strength": 80, "52w_high_approach": 90, "52w_low_bounce": 85,
        "golden_zone": 85, "vol_accumulation": 80, "momentum_diverge": 90,
        "range_compress": 75, "stealth": 70, "vampire": 85, "perfect_storm": 80
    })
    
    PATTERN_METADATA: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        '🔥 CAT LEADER': {'importance': 'high', 'risk': 'low'},
        '💎 HIDDEN GEM': {'importance': 'high', 'risk': 'medium'},
        '🚀 ACCELERATING': {'importance': 'high', 'risk': 'medium'},
        '🏦 INSTITUTIONAL': {'importance': 'high', 'risk': 'low'},
        '⚡ VOL EXPLOSION': {'importance': 'very_high', 'risk': 'high'},
        '🎯 BREAKOUT': {'importance': 'high', 'risk': 'medium'},
        '👑 MARKET LEADER': {'importance': 'very_high', 'risk': 'low'},
        '🌊 MOMENTUM WAVE': {'importance': 'high', 'risk': 'medium'},
        '💰 LIQUID LEADER': {'importance': 'medium', 'risk': 'low'},
        '💪 LONG STRENGTH': {'importance': 'medium', 'risk': 'low'},
        '📈 QUALITY TREND': {'importance': 'high', 'risk': 'low'},
        '⛈️ PERFECT STORM': {'importance': 'very_high', 'risk': 'medium'}
    })
    
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000), 'rvol': (0.001, 1_000_000.0), 'pe': (-10000, 10000),
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
        "eps": {
            "Loss": (-float('inf'), 0), "0-5": (0, 5), "5-10": (5, 10),
            "10-20": (10, 20), "20-50": (20, 50), "50-100": (50, 100),
            "100+": (100, float('inf'))
        },
        "pe": {
            "Negative/NA": (-float('inf'), 0), "0-10": (0, 10), "10-15": (10, 15),
            "15-20": (15, 20), "20-30": (20, 30), "30-50": (30, 50),
            "50+": (50, float('inf'))
        },
        "price": {
            "0-100": (0, 100), "100-250": (100, 250), "250-500": (250, 500),
            "500-1000": (500, 1000), "1000-2500": (1000, 2500), "2500-5000": (2500, 5000),
            "5000+": (5000, float('inf'))
        }
    })

CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
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
                    
                    perf_metrics = RobustSessionState.safe_get('wd_performance_metrics', {})
                    perf_metrics[func.__name__] = elapsed
                    RobustSessionState.safe_set('wd_performance_metrics', perf_metrics)
                    
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
                    raise
            return wrapper
        return decorator

# ============================================
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidator:
    _clipping_counts: Dict[str, int] = defaultdict(int)

    @staticmethod
    def get_clipping_counts() -> Dict[str, int]:
        counts = DataValidator._clipping_counts.copy()
        DataValidator._clipping_counts.clear()
        return counts

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], context: str = "") -> Tuple[bool, str]:
        if df is None or df.empty:
            return False, f"{context}: DataFrame is empty or None."
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            return False, f"{context}: Missing required columns: {missing_columns}"
        
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"{context}: Found {duplicates} duplicate tickers.")
        
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        completeness = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        
        if completeness < 50:
            logger.warning(f"{context}: Low data completeness ({completeness:.1f}%).")
        
        dq_state = RobustSessionState.safe_get('wd_data_quality', {})
        dq_state.update({
            'completeness': completeness,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_tickers': duplicates,
            'context': context,
            'timestamp': datetime.now(timezone.utc)
        })
        RobustSessionState.safe_set('wd_data_quality', dq_state)
        
        logger.info(f"{context}: Validated {len(df)} rows, {len(df.columns)} columns, {completeness:.1f}% complete.")
        return True, "Validation passed."
    
    @staticmethod
    def clean_numeric_value(value: Any, col_name: str, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None) -> float:
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        try:
            cleaned = str(value).strip().replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            if cleaned.upper() in ['', '-', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '#VALUE!', '#ERROR!', '#DIV/0!', 'INF', '-INF']:
                return np.nan
            
            result = float(cleaned)
            
            if bounds:
                min_val, max_val = bounds
                if result < min_val:
                    logger.warning(f"Value clipped for '{col_name}': {result:.2f} to min {min_val:.2f}.")
                    DataValidator._clipping_counts[col_name] += 1
                    result = min_val
                elif result > max_val:
                    logger.warning(f"Value clipped for '{col_name}': {result:.2f} to max {max_val:.2f}.")
                    DataValidator._clipping_counts[col_name] += 1
                    result = max_val
            
            if np.isnan(result) or np.isinf(result):
                return np.nan
            
            return result
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    @staticmethod
    def sanitize_string(value: Any, default: str = "Unknown") -> str:
        if pd.isna(value) or value is None:
            return default
        
        cleaned = str(value).strip()
        if cleaned.upper() in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']:
            return default
        
        return ' '.join(cleaned.split())

# Global validator instance
validator = DataValidator()

# ============================================
# SMART CACHING WITH VERSIONING
# ============================================

def get_requests_retry_session(session=None) -> requests.Session:
    session = session or requests.Session()
    retry = Retry(
        total=CONFIG.MAX_RETRY_ATTEMPTS,
        read=CONFIG.MAX_RETRY_ATTEMPTS,
        connect=CONFIG.MAX_RETRY_ATTEMPTS,
        backoff_factor=CONFIG.RETRY_BACKOFF_FACTOR,
        status_forcelist=CONFIG.RETRY_STATUS_CODES,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Wave Detection Ultimate 3.0'})
    return session

@st.cache_data(persist="disk", show_spinner=False)
def load_and_process_data(source_type: str, file_data=None, sheet_id: str = None, gid: str = None) -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    start_time = time.perf_counter()
    metadata = {'source_type': source_type, 'processing_start': datetime.now(timezone.utc), 'errors': [], 'warnings': []}
    
    try:
        validator.get_clipping_counts() # Reset clipping counts
        
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV.")
            df = pd.read_csv(file_data, low_memory=False)
            metadata['source'] = "User Upload"
        else:
            if not sheet_id:
                raise ValueError("A Google Sheets ID is required for this data source.")
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            csv_url = CONFIG.CSV_URL_TEMPLATE.format(sheet_id=sheet_id, gid=gid)
            logger.info(f"Loading data from Google Sheets URL: {csv_url}.")
            
            session = get_requests_retry_session()
            response = session.get(csv_url, timeout=CONFIG.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            if len(response.content) < 100:
                raise ValueError("Response from Google Sheets is too small; likely an error page.")
            
            df = pd.read_csv(BytesIO(response.content), low_memory=False)
            metadata['source'] = "Google Sheets"
            metadata['sheet_id'] = sheet_id
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid:
            raise ValueError(validation_msg)
        
        df = DataProcessor.process_dataframe(df, metadata)
        df = RankingEngine.calculate_all_scores(df)
        df = PatternDetector.detect_all_patterns(df)
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid:
            raise ValueError(validation_msg)
        
        clipping_info = DataValidator.get_clipping_counts()
        if clipping_info:
            metadata['warnings'].append(f"Some numeric values were clipped during processing: {clipping_info}.")
        
        timestamp = datetime.now(timezone.utc)
        RobustSessionState.safe_set('wd_last_good_data', (df.copy(), timestamp, metadata))
        
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s.")
        gc.collect()
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}", exc_info=True)
        metadata['errors'].append(str(e))
        
        last_good_data = RobustSessionState.safe_get('wd_last_good_data')
        if last_good_data:
            df, timestamp, old_metadata = last_good_data
            metadata['warnings'].append("Using cached data due to load failure.")
            return df, timestamp, metadata
        
        raise

# ============================================
# DATA PROCESSING ENGINE
# ============================================

class DataProcessor:
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        initial_count = len(df)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                if col not in ['ticker', 'company_name', 'category', 'sector', 'industry']:
                    is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                    bounds = CONFIG.VALUE_BOUNDS.get(col, None)
                    if not bounds:
                        if 'volume' in col.lower(): bounds = CONFIG.VALUE_BOUNDS['volume']
                        elif col == 'rvol': bounds = CONFIG.VALUE_BOUNDS['rvol']
                        elif col == 'pe': bounds = CONFIG.VALUE_BOUNDS['pe']
                        elif is_pct: bounds = CONFIG.VALUE_BOUNDS['returns']
                        elif col == 'price': bounds = CONFIG.VALUE_BOUNDS['price']
                    
                    df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, col, is_pct, bounds))
                else:
                    df[col] = df[col].apply(DataValidator.sanitize_string)
        
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(float)
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 1000.0).fillna(1.0)
        
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers.")
        
        df = DataProcessor._add_tier_classifications(df)
        
        removed = initial_count - len(df)
        if removed > 0:
            metadata['warnings'].append(f"Removed {removed} invalid rows during processing.")
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows.")
        return df

    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            if pd.isna(value): return "Unknown"
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val: return tier_name
                if tier_name == list(tier_dict.keys())[0] and (min_val == -float('inf') or min_val == 0) and value == min_val: return tier_name
            return "Unknown"
        
        if 'eps_current' in df.columns:
            df['eps_tier'] = df['eps_current'].apply(lambda x: classify_tier(x, CONFIG.TIERS['eps']))
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(lambda x: "Negative/NA" if pd.isna(x) or x <= 0 else classify_tier(x, CONFIG.TIERS['pe']))
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(lambda x: classify_tier(x, CONFIG.TIERS['price']))
        
        return df

# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df['money_flow_mm'] = df['price'].fillna(0) * df['volume_1d'].fillna(0) * df['rvol'].fillna(1.0) / 1_000_000
        
        if all(c in df.columns for c in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d']):
            df['vmi'] = (df['vol_ratio_1d_90d'].fillna(1.0) * 4 + df['vol_ratio_7d_90d'].fillna(1.0) * 3 +
                         df['vol_ratio_30d_90d'].fillna(1.0) * 2 + df['vol_ratio_90d_180d'].fillna(1.0) * 1) / 10
        else: df['vmi'] = np.nan
        
        if all(c in df.columns for c in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'].fillna(50) + abs(df['from_high_pct'].fillna(-50))
        else: df['position_tension'] = np.nan
        
        df['momentum_harmony'] = 0
        if 'ret_1d' in df.columns: df['momentum_harmony'] += (df['ret_1d'].fillna(0) > 0).astype(int)
        if all(c in df.columns for c in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                d7 = np.where(df['ret_7d'].fillna(0) != 0, df['ret_7d'].fillna(0) / 7, np.nan)
                d30 = np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan)
            df['momentum_harmony'] += ((d7.fillna(-np.inf) > d30.fillna(-np.inf))).astype(int)
        if all(c in df.columns for c in ['ret_30d', 'ret_3m']):
            with np.errstate(divide='ignore', invalid='ignore'):
                d30 = np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan)
                d3m = np.where(df['ret_3m'].fillna(0) != 0, df['ret_3m'].fillna(0) / 90, np.nan)
            df['momentum_harmony'] += ((d30.fillna(-np.inf) > d3m.fillna(-np.inf))).astype(int)
        if 'ret_3m' in df.columns: df['momentum_harmony'] += (df['ret_3m'].fillna(0) > 0).astype(int)
        
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)
        
        score_cols = ['momentum_score', 'acceleration_score', 'rvol_score', 'breakout_score']
        if all(c in df.columns for c in score_cols):
            df['overall_wave_strength'] = (df['momentum_score'].fillna(50) * 0.3 + df['acceleration_score'].fillna(50) * 0.3 +
                                           df['rvol_score'].fillna(50) * 0.2 + df['breakout_score'].fillna(50) * 0.2)
        else: df['overall_wave_strength'] = np.nan
        
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
# RANKING ENGINE - OPTIMIZED
# ============================================

class RankingEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        
        df['position_score'] = RankingEngine._calculate_position_score(df).fillna(50)
        df['volume_score'] = RankingEngine._calculate_volume_score(df).fillna(50)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df).fillna(50)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df).fillna(50)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df).fillna(50)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df).fillna(50)
        
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df).fillna(50)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df).fillna(50)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df).fillna(50)
        
        scores_matrix = df[['position_score', 'volume_score', 'momentum_score',
                           'acceleration_score', 'breakout_score', 'rvol_score']].values
        weights = np.array([CONFIG.POSITION_WEIGHT, CONFIG.VOLUME_WEIGHT, CONFIG.MOMENTUM_WEIGHT,
                            CONFIG.ACCELERATION_WEIGHT, CONFIG.BREAKOUT_WEIGHT, CONFIG.RVOL_WEIGHT])
        
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').fillna(len(df) + 1).astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom').fillna(0) * 100
        
        df = RankingEngine._calculate_category_ranks(df)
        
        logger.info(f"Ranking complete: {len(df)} stocks processed.")
        return df
    
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        if series is None or series.empty or series.notna().sum() == 0:
            return pd.Series(np.nan, dtype=float)
        series = series.replace([np.inf, -np.inf], np.nan)
        
        if pct:
            return series.rank(pct=True, ascending=ascending, na_option='bottom') * 100
        else:
            return series.rank(ascending=ascending, method='min', na_option='bottom')

    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        pos_score = pd.Series(np.nan, index=df.index, dtype=float)
        has_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        if not (has_low or has_high): return pos_score
        
        rank_low = RankingEngine._safe_rank(df['from_low_pct'], pct=True, ascending=True) if has_low else pd.Series(np.nan, index=df.index)
        rank_high = RankingEngine._safe_rank(df['from_high_pct'], pct=True, ascending=False) if has_high else pd.Series(np.nan, index=df.index)
        
        pos_score = (rank_low.fillna(50) * 0.6 + rank_high.fillna(50) * 0.4)
        pos_score[rank_low.isna() & rank_high.isna()] = np.nan
        return pos_score.clip(0, 100)

    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        vol_score = pd.Series(np.nan, index=df.index, dtype=float)
        vol_cols = [('vol_ratio_1d_90d', 0.20), ('vol_ratio_7d_90d', 0.20), ('vol_ratio_30d_90d', 0.20),
                    ('vol_ratio_30d_180d', 0.15), ('vol_ratio_90d_180d', 0.25)]
        
        has_data_mask = pd.Series(False, index=df.index)
        weighted_score = pd.Series(0.0, index=df.index)
        total_weight_per_row = pd.Series(0.0, index=df.index)
        
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                rank = RankingEngine._safe_rank(df[col], pct=True, ascending=True)
                has_data_mask |= rank.notna()
                weighted_score += rank.fillna(0) * weight
                total_weight_per_row += rank.notna().astype(float) * weight
        
        valid_rows = total_weight_per_row > 0
        vol_score.loc[valid_rows] = (weighted_score.loc[valid_rows] / total_weight_per_row.loc[valid_rows]).clip(0, 100)
        return vol_score

    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        mom_score = pd.Series(np.nan, index=df.index, dtype=float)
        has_30d = 'ret_30d' in df.columns and df['ret_30d'].notna().any()
        has_7d = 'ret_7d' in df.columns and df['ret_7d'].notna().any()
        
        if has_30d:
            mom_score = RankingEngine._safe_rank(df['ret_30d'], pct=True, ascending=True)
        elif has_7d:
            mom_score = RankingEngine._safe_rank(df['ret_7d'], pct=True, ascending=True)
            logger.info("Using 7-day returns for momentum score due to missing 30-day data.")
        
        if has_7d and has_30d:
            bonus = pd.Series(0, index=df.index)
            all_pos = (df['ret_7d'].fillna(-1) > 0) & (df['ret_30d'].fillna(-1) > 0)
            bonus[all_pos] = 5
            
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d = np.where(df['ret_7d'].fillna(0) != 0, df['ret_7d'].fillna(0) / 7, np.nan)
                daily_30d = np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan)
            
            accel_mask = all_pos & (daily_7d.fillna(-np.inf) > daily_30d.fillna(-np.inf))
            bonus[accel_mask] = 10
            
            mom_score = (mom_score.fillna(50) + bonus).clip(0, 100)
            mom_score.loc[mom_score.isna()] = np.nan # Re-propagate initial NaNs
        return mom_score

    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        accel_score = pd.Series(np.nan, index=df.index, dtype=float)
        if not all(c in df.columns for c in ['ret_1d', 'ret_7d', 'ret_30d']): return accel_score
        
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = df['ret_1d']
            d7 = df['ret_7d'] / 7
            d30 = df['ret_30d'] / 30
        
        has_data = d1.notna() & d7.notna() & d30.notna()
        accel_score.loc[has_data] = 50.0
        
        perfect = has_data & (d1 > d7) & (d7 > d30) & (d1 > 0)
        accel_score.loc[perfect] = 100
        
        good = has_data & ~perfect & (d1 > d7) & (d1 > 0)
        accel_score.loc[good] = 80
        
        mod = has_data & ~perfect & ~good & (d1 > 0)
        accel_score.loc[mod] = 60
        
        slight = has_data & (d1 <= 0) & (d7 > 0)
        accel_score.loc[slight] = 40
        
        strong = has_data & (d1 <= 0) & (d7 <= 0)
        accel_score.loc[strong] = 20
        
        return accel_score.clip(0, 100)

    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        bo_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        has_all_data_mask = pd.Series(True, index=df.index)
        
        if 'from_high_pct' in df.columns:
            dist_factor = (100 + df['from_high_pct'].fillna(-100)).clip(0, 100)
            has_all_data_mask &= df['from_high_pct'].notna()
        else: dist_factor = pd.Series(np.nan, index=df.index)
        
        if 'vol_ratio_7d_90d' in df.columns:
            vol_factor = ((df['vol_ratio_7d_90d'].fillna(1.0) - 1) * 100).clip(0, 100)
            has_all_data_mask &= df['vol_ratio_7d_90d'].notna()
        else: vol_factor = pd.Series(np.nan, index=df.index)
        
        trend_factor = pd.Series(np.nan, index=df.index, dtype=float)
        if all(c in df.columns for c in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
            above_count = (df['price'] > df['sma_20d']).astype(int) + (df['price'] > df['sma_50d']).astype(int) + (df['price'] > df['sma_200d']).astype(int)
            trend_factor = (above_count / 3 * 100).fillna(0).clip(0, 100)
            has_all_data_mask &= df[['price', 'sma_20d', 'sma_50d', 'sma_200d']].notna().all(axis=1)
        
        combined = (dist_factor.fillna(50) * 0.4 + vol_factor.fillna(50) * 0.4 + trend_factor.fillna(50) * 0.2)
        bo_score.loc[has_all_data_mask] = combined.loc[has_all_data_mask]
        
        return bo_score.clip(0, 100)

    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        if 'rvol' not in df.columns: return pd.Series(np.nan, index=df.index)
        
        rvol = df['rvol']
        rvol_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        rvol_score[rvol.notna() & (rvol > 10)] = 95
        rvol_score[rvol.notna() & (rvol > 5) & (rvol <= 10)] = 90
        rvol_score[rvol.notna() & (rvol > 3) & (rvol <= 5)] = 85
        rvol_score[rvol.notna() & (rvol > 2) & (rvol <= 3)] = 80
        rvol_score[rvol.notna() & (rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score[rvol.notna() & (rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score[rvol.notna() & (rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score[rvol.notna() & (rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score[rvol.notna() & (rvol > 0.3) & (rvol <= 0.5)] = 30
        rvol_score[rvol.notna() & (rvol <= 0.3)] = 20
        
        return rvol_score.clip(0, 100)

    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        trend_score = pd.Series(np.nan, index=df.index, dtype=float)
        if not 'price' in df.columns or df['price'].isna().all(): return trend_score
        
        smas = ['sma_20d', 'sma_50d', 'sma_200d']
        
        has_sma_data = df[smas].notna().any(axis=1)
        trend_score.loc[has_sma_data] = 50
        
        if all(s in df.columns for s in smas):
            perfect = (df['price'] > df['sma_20d']) & (df['sma_20d'] > df['sma_50d']) & (df['sma_50d'] > df['sma_200d'])
            strong = (~perfect) & (df['price'] > df['sma_20d']) & (df['price'] > df['sma_50d']) & (df['price'] > df['sma_200d'])
            trend_score.loc[perfect.fillna(False)] = 100
            trend_score.loc[strong.fillna(False)] = 85

        above_count = sum([(df['price'] > df[s]).fillna(False) for s in smas if s in df.columns])
        good = has_sma_data & (above_count == 2) & trend_score.isna()
        weak = has_sma_data & (above_count == 1) & trend_score.isna()
        poor = has_sma_data & (above_count == 0) & trend_score.isna()
        
        trend_score.loc[good] = 70
        trend_score.loc[weak] = 40
        trend_score.loc[poor] = 20
        
        return trend_score.clip(0, 100)

    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        strength_score = pd.Series(np.nan, index=df.index, dtype=float)
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        
        if not any(c in df.columns for c in lt_cols): return strength_score
        
        lt_returns = df.filter(items=lt_cols).fillna(0)
        avg_return = lt_returns.mean(axis=1)
        has_any_lt_data = df.filter(items=lt_cols).notna().any(axis=1)
        
        scores = np.digitize(avg_return.loc[has_any_lt_data], [ -25, -10, 0, 5, 15, 30, 50, 100])
        strength_map = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        strength_score.loc[has_any_lt_data] = np.array(strength_map)[scores]
        
        return strength_score.clip(0, 100)

    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        liq_score = pd.Series(np.nan, index=df.index, dtype=float)
        if all(c in df.columns for c in ['volume_30d', 'price']):
            dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
            valid_vol = dollar_volume.notna() & (dollar_volume > 0)
            if valid_vol.any():
                liq_score.loc[valid_vol] = RankingEngine._safe_rank(dollar_volume.loc[valid_vol], pct=True, ascending=True)
        return liq_score.clip(0, 100)
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        df['category_rank'] = np.nan
        df['category_percentile'] = np.nan
        
        for category in df['category'].dropna().unique():
            mask = df['category'] == category
            cat_df = df[mask]
            
            if len(cat_df) > 0 and 'master_score' in cat_df.columns and cat_df['master_score'].notna().any():
                df.loc[mask, 'category_rank'] = cat_df['master_score'].rank(method='first', ascending=False, na_option='bottom').astype(int)
                df.loc[mask, 'category_percentile'] = cat_df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        
        return df

# ============================================
# PATTERN DETECTION ENGINE - OPTIMIZED
# ============================================

class PatternDetector:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df['patterns'] = [''] * len(df)
            return df
        
        patterns_with_masks = PatternDetector._get_all_pattern_definitions(df)
        if not patterns_with_masks:
            df['patterns'] = [''] * len(df)
            return df
        
        pattern_matrix = pd.DataFrame(False, index=df.index, columns=[name for name, _ in patterns_with_masks])
        
        for pattern_name, mask in patterns_with_masks:
            if mask is not None and not mask.empty:
                pattern_matrix[pattern_name] = mask.reindex(df.index, fill_value=False)
        
        df['patterns'] = pattern_matrix.apply(lambda row: ' | '.join(row.index[row].tolist()), axis=1)
        df['patterns'] = df['patterns'].fillna('')
        
        return df

    @staticmethod
    def _get_all_pattern_definitions(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        patterns = [] 
        
        def get_col_safe(col_name: str, fill_val: Any = False) -> pd.Series:
            return df[col_name].fillna(fill_val) if col_name in df.columns else pd.Series(fill_val, index=df.index)

        patterns.append(('🔥 CAT LEADER', get_col_safe('category_percentile', 0) >= CONFIG.PATTERN_THRESHOLDS['category_leader']))
        patterns.append(('💎 HIDDEN GEM', (get_col_safe('category_percentile', 0) >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (get_col_safe('percentile', 100) < 70)))
        patterns.append(('🚀 ACCELERATING', get_col_safe('acceleration_score', 0) >= CONFIG.PATTERN_THRESHOLDS['acceleration']))
        patterns.append(('🏦 INSTITUTIONAL', (get_col_safe('volume_score', 0) >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (get_col_safe('vol_ratio_90d_180d', 0) > 1.1)))
        patterns.append(('⚡ VOL EXPLOSION', get_col_safe('rvol', 0) > 3))
        patterns.append(('🎯 BREAKOUT', get_col_safe('breakout_score', 0) >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']))
        patterns.append(('👑 MARKET LEADER', get_col_safe('percentile', 0) >= CONFIG.PATTERN_THRESHOLDS['market_leader']))
        patterns.append(('🌊 MOMENTUM WAVE', (get_col_safe('momentum_score', 0) >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (get_col_safe('acceleration_score', 0) >= 70)))
        patterns.append(('💰 LIQUID LEADER', (get_col_safe('liquidity_score', 0) >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (get_col_safe('percentile', 0) >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])))
        patterns.append(('💪 LONG STRENGTH', get_col_safe('long_term_strength', 0) >= CONFIG.PATTERN_THRESHOLDS['long_strength']))
        patterns.append(('📈 QUALITY TREND', get_col_safe('trend_quality', 0) >= 80))
        
        has_valid_pe = get_col_safe('pe').notna() & (get_col_safe('pe') > 0) & (get_col_safe('pe') < 10000)
        patterns.append(('💎 VALUE MOMENTUM', has_valid_pe & (get_col_safe('pe', 0) < 15) & (get_col_safe('master_score', 0) >= 70)))
        
        has_eps_growth = get_col_safe('eps_change_pct').notna()
        ext_growth = has_eps_growth & (get_col_safe('eps_change_pct', 0) > 1000)
        norm_growth = has_eps_growth & (get_col_safe('eps_change_pct', 0) > 50) & (get_col_safe('eps_change_pct', 0) <= 1000)
        patterns.append(('📊 EARNINGS ROCKET', (ext_growth & (get_col_safe('acceleration_score', 0) >= 80)) | (norm_growth & (get_col_safe('acceleration_score', 0) >= 70))))
        
        has_complete_data = (get_col_safe('pe').notna() & get_col_safe('eps_change_pct').notna() & (get_col_safe('pe', 0) > 0) & (get_col_safe('pe', 0) < 10000))
        patterns.append(('🏆 QUALITY LEADER', has_complete_data & (get_col_safe('pe', 0).between(10, 25)) & (get_col_safe('eps_change_pct', 0) > 20) & (get_col_safe('percentile', 0) >= 80)))
        
        has_eps = get_col_safe('eps_change_pct').notna()
        mega_tu = has_eps & (get_col_safe('eps_change_pct', 0) > 500) & (get_col_safe('volume_score', 0) >= 60)
        strong_tu = has_eps & (get_col_safe('eps_change_pct', 0) > 100) & (get_col_safe('eps_change_pct', 0) <= 500) & (get_col_safe('volume_score', 0) >= 70)
        patterns.append(('⚡ TURNAROUND', mega_tu | strong_tu))
        
        patterns.append(('⚠️ HIGH PE', has_valid_pe & (get_col_safe('pe', 0) > 100)))
        patterns.append(('🎯 52W HIGH APPROACH', (get_col_safe('from_high_pct', -100) > -5) & (get_col_safe('volume_score', 0) >= 70) & (get_col_safe('momentum_score', 0) >= 60)))
        patterns.append(('🔄 52W LOW BOUNCE', (get_col_safe('from_low_pct', 100) < 20) & (get_col_safe('acceleration_score', 0) >= 80) & (get_col_safe('ret_30d', 0) > 10)))
        patterns.append(('👑 GOLDEN ZONE', (get_col_safe('from_low_pct', 0) > 60) & (get_col_safe('from_high_pct', 0) > -40) & (get_col_safe('trend_quality', 0) >= 70)))
        patterns.append(('📊 VOL ACCUMULATION', (get_col_safe('vol_ratio_30d_90d', 0) > 1.2) & (get_col_safe('vol_ratio_90d_180d', 0) > 1.1) & (get_col_safe('ret_30d', 0) > 5)))
        
        if all(c in df.columns for c in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                d7 = np.where(get_col_safe('ret_7d', 0) != 0, get_col_safe('ret_7d', 0) / 7, np.nan)
                d30 = np.where(get_col_safe('ret_30d', 0) != 0, get_col_safe('ret_30d', 0) / 30, np.nan)
            mask = pd.Series(d7 > d30 * 1.5).fillna(False) & (get_col_safe('acceleration_score', 0) >= 85) & (get_col_safe('rvol', 0) > 2)
            patterns.append(('🔀 MOMENTUM DIVERGE', mask))
        
        if all(c in df.columns for c in ['high_52w', 'low_52w', 'from_low_pct']):
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = np.where(get_col_safe('low_52w', 0) > 0, ((get_col_safe('high_52w', 0) - get_col_safe('low_52w', 0)) / get_col_safe('low_52w', 0)) * 100, 100)
            patterns.append(('🎯 RANGE COMPRESS', (range_pct < 50) & (get_col_safe('from_low_pct', 0) > 30)))
        
        if all(c in df.columns for c in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_ratio = np.where(get_col_safe('ret_30d', 0) != 0, get_col_safe('ret_7d', 0) / (get_col_safe('ret_30d', 0) / 4), np.nan)
            mask = pd.Series(ret_ratio > 1).fillna(False) & (get_col_safe('vol_ratio_90d_180d', 0) > 1.1) & (get_col_safe('vol_ratio_30d_90d', 0).between(0.9, 1.1)) & (get_col_safe('from_low_pct', 0) > 40)
            patterns.append(('🤫 STEALTH', mask))
        
        if all(c in df.columns for c in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pace_ratio = np.where(get_col_safe('ret_7d', 0) != 0, get_col_safe('ret_1d', 0) / (get_col_safe('ret_7d', 0) / 7), np.nan)
            mask = pd.Series(daily_pace_ratio > 2).fillna(False) & (get_col_safe('rvol', 0) > 3) & (get_col_safe('from_high_pct', 0) > -15) & (get_col_safe('category').isin(['Small Cap', 'Micro Cap']))
            patterns.append(('🧛 VAMPIRE', mask))
        
        patterns.append(('⛈️ PERFECT STORM', (get_col_safe('momentum_harmony', 0) == 4) & (get_col_safe('master_score', 0) > 80)))
        
        return patterns

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        if df.empty: return "😴 NO DATA", {}
        metrics = {}
        
        if 'category' in df.columns and 'master_score' in df.columns:
            cat_scores = df.groupby('category')['master_score'].mean().fillna(50)
            micro_small_avg = cat_scores.reindex(['Micro Cap', 'Small Cap']).mean() if not cat_scores.reindex(['Micro Cap', 'Small Cap']).empty else 50
            large_mega_avg = cat_scores.reindex(['Large Cap', 'Mega Cap']).mean() if not cat_scores.reindex(['Large Cap', 'Mega Cap']).empty else 50
            metrics.update({'micro_small_avg': micro_small_avg, 'large_mega_avg': large_mega_avg, 'category_spread': micro_small_avg - large_mega_avg})
        else: micro_small_avg, large_mega_avg = 50, 50
        
        breadth = (df['ret_30d'].fillna(0) > 0).mean() if 'ret_30d' in df.columns else 0.5
        metrics['breadth'] = breadth
        
        avg_rvol = df['rvol'].fillna(1.0).median() if 'rvol' in df.columns else 1.0
        metrics['avg_rvol'] = avg_rvol
        
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
            adv = len(df[df['ret_1d'].fillna(0) > 0])
            dec = len(df[df['ret_1d'].fillna(0) < 0])
            unch = len(df[df['ret_1d'].fillna(0) == 0])
            ad_metrics.update({'advancing': adv, 'declining': dec, 'unchanged': unch})
            ad_metrics['ad_ratio'] = adv / dec if dec > 0 else (float('inf') if adv > 0 else 1.0)
            ad_metrics['ad_line'] = adv - dec
            ad_metrics['breadth_pct'] = (adv / len(df)) * 100 if len(df) > 0 else 0
        return ad_metrics
    
    @staticmethod
    def _apply_dynamic_sampling(df_group: pd.DataFrame) -> pd.DataFrame:
        size = len(df_group)
        if size <= 5: sample_count = size
        elif size <= 20: sample_count = max(1, int(size * 0.8))
        elif size <= 50: sample_count = max(1, int(size * 0.6))
        elif size <= 100: sample_count = max(1, int(size * 0.4))
        else: sample_count = min(50, int(size * 0.25))
        return df_group.nlargest(sample_count, 'master_score', keep='first') if sample_count > 0 else pd.DataFrame()

    @staticmethod
    def _calculate_flow_metrics(normalized_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        agg_dict = {c: [('mean', 'mean'), ('median', 'median'), ('std', 'std'), ('count', 'count')] if c == 'master_score' else ('mean', 'mean') for c in ['master_score', 'momentum_score', 'volume_score', 'rvol', 'ret_30d', 'money_flow_mm']}
        available_agg_dict = {k: v for k, v in agg_dict.items() if k in normalized_df.columns}
        
        group_metrics = normalized_df.groupby(group_col).agg(available_agg_dict).round(2)
        group_metrics.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in group_metrics.columns.values]
        
        rename_map = {'master_score_mean': 'avg_score', 'master_score_median': 'median_score', 'master_score_std': 'std_score', 'master_score_count': 'count',
                      'momentum_score_mean': 'avg_momentum', 'volume_score_mean': 'avg_volume', 'rvol_mean': 'avg_rvol',
                      'ret_30d_mean': 'avg_ret_30d', 'money_flow_mm_sum': 'total_money_flow'}
        group_metrics = group_metrics.rename(columns=rename_map)

        group_metrics['flow_score'] = (group_metrics['avg_score'].fillna(0) * 0.3 + group_metrics.get('median_score', pd.Series(0, index=group_metrics.index)).fillna(0) * 0.2 +
                                       group_metrics['avg_momentum'].fillna(0) * 0.25 + group_metrics['avg_volume'].fillna(0) * 0.25)
        
        group_metrics['rank'] = group_metrics['flow_score'].rank(ascending=False, method='min')
        return group_metrics.sort_values('flow_score', ascending=False)

    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'sector' not in df.columns or df.empty: return pd.DataFrame()
        sampled_dfs = [MarketIntelligence._apply_dynamic_sampling(g.copy()) for name, g in df.groupby('sector') if name != 'Unknown']
        if not sampled_dfs: return pd.DataFrame()
        
        normalized_df = pd.concat(sampled_dfs, ignore_index=True)
        metrics = MarketIntelligence._calculate_flow_metrics(normalized_df, 'sector')
        
        original_counts = df.groupby('sector').size().rename('total_stocks')
        metrics = metrics.join(original_counts, how='left')
        metrics['analyzed_stocks'] = metrics['count']
        return metrics

    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'industry' not in df.columns or df.empty: return pd.DataFrame()
        sampled_dfs = [MarketIntelligence._apply_dynamic_sampling(g.copy()) for name, g in df.groupby('industry') if name != 'Unknown']
        if not sampled_dfs: return pd.DataFrame()
        
        normalized_df = pd.concat(sampled_dfs, ignore_index=True)
        metrics = MarketIntelligence._calculate_flow_metrics(normalized_df, 'industry')
        
        original_counts = df.groupby('industry').size().rename('total_stocks')
        metrics = metrics.join(original_counts, how='left')
        metrics['analyzed_stocks'] = metrics['count']
        return metrics

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df.empty:
            fig.add_annotation(text="No data available for visualization", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        scores = [('position_score', 'Position', '#3498db'), ('volume_score', 'Volume', '#e74c3c'),
                  ('momentum_score', 'Momentum', '#2ecc71'), ('acceleration_score', 'Acceleration', '#f39c12'),
                  ('breakout_score', 'Breakout', '#9b59b6'), ('rvol_score', 'RVOL', '#e67e22')]
        
        for col, label, color in scores:
            if col in df.columns and df[col].notna().any():
                fig.add_trace(go.Box(y=df[col].dropna(), name=label, marker_color=color, boxpoints='outliers', hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'))
        
        fig.update_layout(title="Score Component Distribution", yaxis_title="Score (0-100)", template='plotly_white', height=400, showlegend=False)
        return fig

    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, n: int = 10) -> go.Figure:
        try:
            plot_df = df.dropna(subset=['ret_1d', 'ret_7d', 'ret_30d', 'acceleration_score'], how='any').nlargest(min(n, len(df)), 'acceleration_score')
            if plot_df.empty:
                fig = go.Figure(); fig.add_annotation(text="No data for this chart.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            fig = go.Figure()
            for _, stock in plot_df.iterrows():
                x_points, y_points = ['Start'], [0]
                if pd.notna(stock['ret_30d']): x_points.append('30D'); y_points.append(stock['ret_30d'])
                if pd.notna(stock['ret_7d']): x_points.append('7D'); y_points.append(stock['ret_7d'])
                if pd.notna(stock['ret_1d']): x_points.append('Today'); y_points.append(stock['ret_1d'])
                
                if len(x_points) > 1:
                    accel_score = stock.get('acceleration_score', 0)
                    line_style = dict(width=3, dash='solid') if accel_score >= 85 else dict(width=2, dash='solid') if accel_score >= 70 else dict(width=2, dash='dot')
                    marker_style = dict(size=10, symbol='star', line=dict(color='DarkSlateGrey', width=1)) if accel_score >= 85 else dict(size=8) if accel_score >= 70 else dict(size=6)
                    
                    fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='lines+markers', name=f"{stock['ticker']} ({accel_score:.0f})", line=line_style, marker=marker_style,
                                             hovertemplate=f"<b>{stock['ticker']}</b><br>%{{x}}: %{{y:.1f}}%<br>Accel Score: {accel_score:.0f}<extra></extra>"))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(title=f"Acceleration Profiles - Top {len(plot_df)} Momentum Builders", xaxis_title="Time Frame", yaxis_title="Return %",
                              height=400, template='plotly_white', showlegend=True, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), hovermode='x unified')
            return fig
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}")
            fig = go.Figure(); fig.add_annotation(text=f"Error generating chart: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

# ============================================
# FILTER ENGINE - ENHANCED
# ============================================

class FilterEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        if df.empty: return df
        
        mask = pd.Series(True, index=df.index)
        
        if filters.get('categories'): mask &= df['category'].isin(filters['categories'])
        if filters.get('sectors'): mask &= df['sector'].isin(filters['sectors'])
        if filters.get('industries'): mask &= df['industry'].isin(filters['industries'])
        
        min_score = filters.get('min_score', 0)
        if min_score > 0: mask &= df['master_score'] >= min_score
        
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            mask &= df['eps_change_pct'].notna() & (df['eps_change_pct'] >= min_eps_change)
        
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            pattern_regex = '|'.join(patterns)
            mask &= df['patterns'].str.contains(pattern_regex, case=False, na=False, regex=True)
        
        trend_range = filters.get('trend_range')
        if filters.get('trend_filter') != 'All Trends' and trend_range and 'trend_quality' in df.columns:
            min_trend, max_trend = trend_range
            mask &= df['trend_quality'].notna() & (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend)
        
        min_pe, max_pe = filters.get('min_pe'), filters.get('max_pe')
        if min_pe is not None and 'pe' in df.columns: mask &= df['pe'].notna() & (df['pe'] > 0) & (df['pe'] >= min_pe)
        if max_pe is not None and 'pe' in df.columns: mask &= df['pe'].notna() & (df['pe'] > 0) & (df['pe'] <= max_pe)
        
        for tier_type_key, col_name_suffix in [('eps_tiers', 'eps_tier'), ('pe_tiers', 'pe_tier'), ('price_tiers', 'price_tier')]:
            tier_values = filters.get(tier_type_key, [])
            col_name = col_name_suffix
            if tier_values and 'All' not in tier_values and col_name in df.columns:
                mask &= df[col_name].isin(tier_values)
        
        if filters.get('require_fundamental_data', False) and all(c in df.columns for c in ['pe', 'eps_change_pct']):
            mask &= df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna()
        
        wave_states = filters.get('wave_states', [])
        if wave_states and 'All' not in wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)
        
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and wave_strength_range != (0, 100) and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            mask &= df['overall_wave_strength'].notna() & (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws)
        
        filtered_df = df[mask].copy()
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks.")
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        if df.empty or column not in df.columns: return []
        
        temp_filters = current_filters.copy()
        
        filter_key_map = {'category': 'categories', 'sector': 'sectors', 'industry': 'industries', 
                          'eps_tier': 'eps_tiers', 'pe_tier': 'pe_tiers', 'price_tier': 'price_tiers', 'wave_state': 'wave_states'}
        
        if column in filter_key_map: temp_filters.pop(filter_key_map[column], None)
        
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        values = filtered_df[column].dropna().astype(str).unique()
        values = [v for v in values if v.strip().lower() not in ['unknown', '', 'nan', 'n/a', 'none', '-']]
        
        try: values = sorted(values, key=lambda x: float(re.sub(r'[^0-9.]', '', x)) if re.sub(r'[^0-9.]', '', x) else x)
        except: values = sorted(values, key=str)
        
        return values

# ============================================
# SEARCH ENGINE - OPTIMIZED
# ============================================

class SearchEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        if not query or df.empty: return pd.DataFrame()
        
        query_upper = query.upper().strip()
        
        results = df.copy()
        results['relevance'] = 0
        
        results.loc[results['ticker'].str.upper() == query_upper, 'relevance'] = 1000
        results.loc[results['ticker'].str.upper().str.startswith(query_upper) & (results['relevance'] < 1000), 'relevance'] += 500
        results.loc[results['ticker'].str.upper().str.contains(query_upper, regex=False) & (results['relevance'] < 500), 'relevance'] += 200
        
        if 'company_name' in results.columns:
            results.loc[results['company_name'].str.upper() == query_upper, 'relevance'] += 800
            results.loc[results['company_name'].str.upper().str.startswith(query_upper) & (results['relevance'] < 800), 'relevance'] += 300
            results.loc[results['company_name'].str.upper().str.contains(query_upper, regex=False) & (results['relevance'] < 300), 'relevance'] += 100
            
            def word_match_score(name):
                if pd.isna(name): return 0
                return 50 if any(word.startswith(query_upper) for word in str(name).upper().split()) else 0
            results['relevance'] += results['company_name'].apply(word_match_score)
        
        matches = results[results['relevance'] > 0].sort_values(['relevance', 'master_score'], ascending=[False, False])
        return matches.drop('relevance', axis=1)

# ============================================
# EXPORT ENGINE - ENHANCED
# ============================================

class ExportEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        output = BytesIO()
        
        templates = {'day_trader': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 'volume_score', 'vmi', 'wave_state', 'patterns', 'category', 'sector', 'industry']},
                     'swing_trader': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'breakout_score', 'position_score', 'position_tension', 'from_high_pct', 'from_low_pct', 'trend_quality', 'momentum_harmony', 'patterns', 'sector', 'industry']},
                     'investor': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 'long_term_strength', 'money_flow_mm', 'category', 'sector', 'industry']},
                     'full': {'columns': None}}
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                header_format = workbook.add_format({'bold': True, 'bg_color': '#3498db', 'font_color': 'white', 'border': 1})
                
                top_100_df = df.nlargest(min(100, len(df)), 'master_score', keep='first')
                export_cols = templates[template]['columns'] or [c for c in top_100_df.columns if not c.startswith(('rank_flow', 'dummy'))]
                top_100_export = top_100_df.filter(items=export_cols)
                
                top_100_export.to_excel(writer, sheet_name='Top 100 Stocks', index=False)
                worksheet = writer.sheets['Top 100 Stocks']
                for i, col in enumerate(top_100_export.columns): worksheet.write(0, i, col, header_format)
                worksheet.autofit()
                
                intel_data = [
                    {'Metric': 'Market Regime', 'Value': MarketIntelligence.detect_market_regime(df)[0]},
                    {'Metric': 'Advance/Decline Ratio (1D)', 'Value': f"{MarketIntelligence.calculate_advance_decline_ratio(df).get('ad_ratio', 1.0):.2f}"}
                ]
                pd.DataFrame(intel_data).to_excel(writer, sheet_name='Market Intelligence', index=False)
                worksheet = writer.sheets['Market Intelligence']
                for i, col in enumerate(pd.DataFrame(intel_data).columns): worksheet.write(0, i, col, header_format)
                worksheet.autofit()

                sector_rot = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rot.empty: sector_rot.to_excel(writer, sheet_name='Sector Rotation'); writer.sheets['Sector Rotation'].autofit()
                
                industry_rot = MarketIntelligence.detect_industry_rotation(df)
                if not industry_rot.empty: industry_rot.to_excel(writer, sheet_name='Industry Rotation'); writer.sheets['Industry Rotation'].autofit()
                
                pattern_counts = defaultdict(int)
                for p_str in df['patterns'].dropna():
                    if p_str: [pattern_counts.update({p: pattern_counts[p] + 1}) for p in p_str.split(' | ')]
                if pattern_counts:
                    pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=False).to_excel(writer, sheet_name='Pattern Analysis', index=False)
                    writer.sheets['Pattern Analysis'].autofit()
                
                wave_signals = df[(df['momentum_score'].fillna(0) >= 60) & (df['acceleration_score'].fillna(0) >= 70) & (df['rvol'].fillna(0) >= 2)].nlargest(50, 'master_score', keep='first')
                if not wave_signals.empty:
                    wave_cols = ['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'wave_state', 'patterns', 'category', 'sector', 'industry']
                    wave_signals.filter(items=wave_cols).to_excel(writer, sheet_name='Wave Radar Signals', index=False)
                    writer.sheets['Wave Radar Signals'].autofit()
                
                summary_stats = {
                    'Total Stocks Processed': len(df), 'Average Master Score (All)': df['master_score'].mean() if not df.empty else 0,
                    'Stocks with Patterns (All)': (df['patterns'] != '').sum() if 'patterns' in df.columns else 0,
                    'High RVOL (>2x) (All)': (df['rvol'].fillna(0) > 2).sum() if 'rvol' in df.columns else 0,
                    'Positive 30D Returns (All)': (df['ret_30d'].fillna(0) > 0).sum() if 'ret_30d' in df.columns else 0,
                    'Data Completeness %': RobustSessionState.safe_get('wd_data_quality', {}).get('completeness', 0),
                    'Clipping Events Count': sum(DataValidator._clipping_counts.values()),
                    'Template Used': template,
                    'Export Date (UTC)': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                }
                pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value']).to_excel(writer, sheet_name='Summary', index=False)
                writer.sheets['Summary'].autofit()

        except Exception as e:
            logger.error(f"Error creating Excel report: {e}", exc_info=True)
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        export_cols = [
            'rank', 'ticker', 'company_name', 'master_score', 'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score', 'trend_quality', 'price', 'pe', 'eps_current',
            'eps_change_pct', 'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m',
            'ret_1y', 'rvol', 'vmi', 'money_flow_mm', 'position_tension', 'momentum_harmony', 'wave_state',
            'patterns', 'category', 'sector', 'industry', 'eps_tier', 'pe_tier', 'price_tier', 'overall_wave_strength'
        ]
        
        export_df = df.filter(items=export_cols).copy()
        
        for col_name in CONFIG.VOLUME_RATIO_COLUMNS:
            if col_name in export_df.columns: export_df[col_name] = (export_df[col_name] - 1) * 100
                
        for col in export_df.select_dtypes(include=np.number).columns: export_df[col] = export_df[col].fillna('')
        for col in export_df.select_dtypes(include='object').columns: export_df[col] = export_df[col].fillna('')
        
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
        if df.empty: st.warning("No data available for summary."); return
        
        st.markdown("### 📊 Market Pulse")
        col1, col2, col3, col4 = st.columns(4)
        
        ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
        with col1:
            ad_ratio = ad_metrics.get('ad_ratio', 1.0)
            ad_emoji = "🔥" if ad_ratio > 2 else "📈" if ad_ratio > 1 else "📉"
            UIComponents.render_metric_card("A/D Ratio", f"{ad_emoji} {ad_ratio:.2f}", f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}", "Advance/Decline Ratio (Advancing stocks / Declining stocks over 1 Day)")
        
        with col2:
            high_mom_count = (df['momentum_score'].fillna(0) >= 70).sum()
            mom_pct = (high_mom_count / len(df) * 100) if len(df) > 0 else 0
            UIComponents.render_metric_card("Momentum Health", f"{mom_pct:.0f}%", f"{high_mom_count} strong stocks", "Percentage of stocks with Momentum Score ≥ 70.")
        
        with col3:
            avg_rvol = df['rvol'].fillna(1.0).median() if 'rvol' in df.columns else 1.0
            high_vol_count = (df['rvol'].fillna(0) > 2).sum() if 'rvol' in df.columns else 0
            vol_emoji = "🌊" if avg_rvol > 1.5 else "💧" if avg_rvol > 1.2 else "🏜️"
            UIComponents.render_metric_card("Volume State", f"{vol_emoji} {avg_rvol:.1f}x", f"{high_vol_count} surges", "Median Relative Volume (RVOL). Surges indicate stocks with RVOL > 2x.")
        
        with col4:
            risk = 0
            if 'from_high_pct' in df.columns and (df['from_high_pct'].fillna(-100) >= 0).sum() > 20: risk += 1
            if 'rvol' in df.columns and (df['rvol'].fillna(0) > 10).sum() > 10: risk += 1
            if 'trend_quality' in df.columns and (df['trend_quality'].fillna(50) < 40).sum() > len(df) * 0.3: risk += 1
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]
            UIComponents.render_metric_card("Risk Level", risk_levels[min(risk, 3)], f"{risk} factors", "Composite risk assessment based on overextension, extreme volume, and downtrends.")
        
        st.markdown("### 🎯 Today's Best Opportunities")
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        with opp_col1:
            st.markdown("**🚀 Ready to Run**")
            rtr = df[(df['momentum_score'].fillna(0) >= 70) & (df['acceleration_score'].fillna(0) >= 70) & (df['rvol'].fillna(0) >= 2)].nlargest(5, 'master_score', keep='first')
            if not rtr.empty:
                for _, s in rtr.iterrows(): st.write(f"• **{s['ticker']}** - {s['company_name'][:25]}"); st.caption(f"Score: {s['master_score']:.1f} | RVOL: {s.get('rvol', 0):.1f}x")
            else: st.info("No momentum leaders found.")
        
        with opp_col2:
            st.markdown("**💎 Hidden Gems**")
            gems = df[df['patterns'].str.contains('💎 HIDDEN GEM', na=False)].nlargest(5, 'master_score', keep='first')
            if not gems.empty:
                for _, s in gems.iterrows(): st.write(f"• **{s['ticker']}** - {s['company_name'][:25]}"); st.caption(f"Cat %ile: {s.get('category_percentile', 0):.0f} | Score: {s['master_score']:.1f}")
            else: st.info("No hidden gems today.")

        with opp_col3:
            st.markdown("**⚡ Volume Alerts**")
            vol_alerts = df[df['rvol'].fillna(0) > 3].nlargest(5, 'master_score', keep='first')
            if not vol_alerts.empty:
                for _, s in vol_alerts.iterrows(): st.write(f"• **{s['ticker']}** - {s['company_name'][:25]}"); st.caption(f"RVOL: {s.get('rvol', 0):.1f}x | {s.get('wave_state', 'N/A')}")
            else: st.info("No extreme volume detected.")
            
        st.markdown("### 🧠 Market Intelligence")
        intel_col1, intel_col2 = st.columns([2, 1])
        
        with intel_col1:
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            if not sector_rotation.empty:
                fig = go.Figure(go.Bar(x=sector_rotation.index[:10], y=sector_rotation['flow_score'][:10], text=[f"{v:.1f}" for v in sector_rotation['flow_score'][:10]],
                                       textposition='outside', marker_color=['#2ecc71' if s > 60 else '#e74c3c' if s < 40 else '#f39c12' for s in sector_rotation['flow_score'][:10]],
                                       hovertemplate='Sector: %{x}<br>Flow Score: %{y:.1f}<br>Analyzed: %{customdata[0]} of %{customdata[1]} stocks<extra></extra>',
                                       customdata=np.column_stack((sector_rotation['analyzed_stocks'][:10], sector_rotation['total_stocks'][:10]))))
                fig.update_layout(title="Sector Rotation Map - Smart Money Flow (Dynamically Sampled)", xaxis_title="Sector", yaxis_title="Flow Score", height=400, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No sector rotation data available for visualization.")
        
        with intel_col2:
            regime, metrics = MarketIntelligence.detect_market_regime(df)
            st.markdown(f"**🎯 Market Regime**"); st.markdown(f"### {regime}")
            st.markdown("**📡 Key Signals**")
            if metrics.get('breadth', 0.5) > 0.6: st.write("✅ Strong breadth")
            elif metrics.get('breadth', 0.5) < 0.4: st.write("⚠️ Weak breadth")
            if metrics.get('category_spread', 0) > 10: st.write("🔄 Small caps leading")
            elif metrics.get('category_spread', 0) < -10: st.write("🛡️ Large caps defensive")
            if metrics.get('avg_rvol', 1.0) > 1.5: st.write("🌊 High volume activity")
            if (df['patterns'].fillna('') != '').sum() > len(df) * 0.2: st.write("🎯 Many patterns emerging")
            st.markdown("**💪 Market Strength**")
            strength_score = (metrics.get('breadth', 0.5) * 50) + (min(metrics.get('avg_rvol', 1.0), 2) * 25) + (((df['patterns'].fillna('') != '').sum() / len(df) if len(df) > 0 else 0) * 25)
            if strength_score > 70: st.write("🟢🟢🟢🟢🟢")
            elif strength_score > 50: st.write("🟢🟢🟢🟢⚪")
            elif strength_score > 30: st.write("🟢🟢🟢⚪⚪")
            else: st.write("🟢🟢⚪⚪⚪")
    
    @staticmethod
    def render_pagination_controls(df: pd.DataFrame, display_count: int, page_key: str) -> pd.DataFrame:
        total_rows = len(df)
        if total_rows == 0: st.caption("No data to display."); return df
        
        if f'wd_current_page_{page_key}' not in st.session_state: st.session_state[f'wd_current_page_{page_key}'] = 0
        current_page = st.session_state[f'wd_current_page_{page_key}']
        total_pages = int(np.ceil(total_rows / display_count))
        
        start_idx = current_page * display_count
        end_idx = min(start_idx + display_count, total_rows)
        
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_rows} stocks (Page {current_page + 1} of {total_pages})")
        
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("⬅️ Previous Page", disabled=(current_page == 0), key=f'wd_prev_page_{page_key}'):
                st.session_state[f'wd_current_page_{page_key}'] -= 1
                st.rerun()
        with col_next:
            if st.button("Next Page ➡️", disabled=(current_page >= total_pages - 1), key=f'wd_next_page_{page_key}'):
                st.session_state[f'wd_current_page_{page_key}'] += 1
                st.rerun()
        
        return df.iloc[start_idx:end_idx]

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.set_page_config(page_title="Wave Detection Ultimate 3.0", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
    
    RobustSessionState.initialize()
    
    st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {height: 50px; padding-left: 20px; padding-right: 20px;}
    div[data-testid="metric-container"] {background-color: rgba(28, 131, 225, 0.1); border: 1px solid rgba(28, 131, 225, 0.2); padding: 5% 5% 5% 10%; border-radius: 5px; overflow-wrap: break-word;}
    .stAlert {padding: 1rem; border-radius: 5px;}
    div.stButton > button {width: 100%; transition: all 0.3s ease;}
    div.stButton > button:hover {transform: translateY(-2px); box-shadow: 0 5px 10px rgba(0,0,0,0.2);}
    @media (max-width: 768px) {.stDataFrame {font-size: 12px;} div[data-testid="metric-container"] {padding: 3%;} .main {padding: 0rem 0.5rem;}}
    .stDataFrame > div {overflow-x: auto;}
    .stSpinner > div {border-color: #3498db;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 3.0</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Professional Stock Ranking System • Final Perfected Production Version</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.cache_data.clear()
                RobustSessionState.safe_set('wd_last_refresh', datetime.now(timezone.utc))
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
            if st.button("📊 Google Sheets", type="primary" if RobustSessionState.safe_get('wd_data_source') == "sheet" else "secondary", use_container_width=True):
                RobustSessionState.safe_set('wd_data_source', "sheet")
                st.rerun()
        with data_source_col2:
            if st.button("📁 Upload CSV", type="primary" if RobustSessionState.safe_get('wd_data_source') == "upload" else "secondary", use_container_container_width=True):
                RobustSessionState.safe_set('wd_data_source', "upload")
                st.rerun()

        uploaded_file = None
        if RobustSessionState.safe_get('wd_data_source') == "upload":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv", help="Upload a CSV file with stock data. Must contain 'ticker' and 'price' columns.")
        else:
            st.markdown("#### 🔗 Google Sheet Configuration")
            sheet_input = st.text_input("Google Sheets ID or URL", value=RobustSessionState.safe_get('wd_sheet_id'), placeholder=f"Default: {CONFIG.DEFAULT_SHEET_URL.split('/d/')[1].split('/')[0]}", help="Enter the 44-character ID from your Google Sheet URL.")
            
            if sheet_input:
                sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_input)
                if sheet_id_match: sheet_id = sheet_id_match.group(1)
                else: sheet_id = sheet_input.strip()
                RobustSessionState.safe_set('wd_sheet_id', sheet_id)
            else:
                sheet_id = CONFIG.DEFAULT_SHEET_URL.split('/d/')[1].split('/')[0]
                RobustSessionState.safe_set('wd_sheet_id', sheet_id)

        data_quality = RobustSessionState.safe_get('wd_data_quality', {})
        if data_quality:
            with st.expander("📊 Data Quality", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    completeness = data_quality.get('completeness', 0)
                    emoji = "🟢" if completeness > 80 else "🟡" if completeness > 60 else "🔴"
                    st.metric("Completeness", f"{emoji} {completeness:.1f}%")
                    st.metric("Total Stocks", f"{data_quality.get('total_rows', 0):,}")
                with col2:
                    age = (datetime.now(timezone.utc) - data_quality.get('timestamp', datetime.now(timezone.utc))).total_seconds() / 3600
                    freshness = "🟢 Fresh" if age < 1 else "🟡 Recent" if age < 24 else "🔴 Stale"
                    st.metric("Data Age", freshness)
                    duplicates = data_quality.get('duplicate_tickers', 0)
                    if duplicates > 0: st.metric("Duplicates", f"⚠️ {duplicates}")
        
        perf_metrics = RobustSessionState.safe_get('wd_performance_metrics', {})
        if perf_metrics:
            with st.expander("⚡ Performance"):
                total_time = sum(perf_metrics.values())
                perf_emoji = "🟢" if total_time < 3 else "🟡" if total_time < 5 else "🔴"
                st.metric("Load Time", f"{perf_emoji} {total_time:.1f}s")
                if len(perf_metrics) > 0:
                    slowest = sorted(perf_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
                    for func_name, elapsed in slowest: st.caption(f"{func_name}: {elapsed:.4f}s")
        
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        active_filter_count = 0
        if RobustSessionState.safe_get('wd_quick_filter_applied', False): active_filter_count += 1
        
        filter_keys_to_check = ['wd_category_filter', 'wd_sector_filter', 'wd_industry_filter', 'wd_min_score', 'wd_patterns', 'wd_trend_filter', 'wd_eps_tier_filter', 'wd_pe_tier_filter', 'wd_price_tier_filter', 'wd_min_eps_change', 'wd_min_pe', 'wd_max_pe', 'wd_require_fundamental_data', 'wd_wave_states_filter', 'wd_wave_strength_range_slider']
        for key in filter_keys_to_check:
            value = RobustSessionState.safe_get(key)
            if value and (isinstance(value, (list, str)) and len(value) > 0) or \
               (isinstance(value, (int, float)) and value != 0) or \
               (isinstance(value, tuple) and value != (0, 100)) or \
               (isinstance(value, bool) and value):
                active_filter_count += 1
        RobustSessionState.safe_set('wd_active_filter_count', active_filter_count)
        
        if active_filter_count > 0: st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**")
        
        if st.button("🗑️ Clear All Filters", use_container_width=True, type="primary" if active_filter_count > 0 else "secondary"):
            RobustSessionState.clear_filters()
            st.success("✅ All filters cleared!")
            st.rerun()
        
        show_debug = st.checkbox("🐛 Show Debug Info", value=RobustSessionState.safe_get('wd_show_debug'), key="wd_show_debug")
    
    ranked_df, data_timestamp, metadata = None, None, {}
    try:
        if RobustSessionState.safe_get('wd_data_source') == "upload" and uploaded_file is None: st.warning("Please upload a CSV file to continue."); st.stop()
        if RobustSessionState.safe_get('wd_data_source') == "sheet" and not RobustSessionState.safe_get('wd_sheet_id'): st.warning("Please enter a Google Sheets ID to continue."); st.stop()
        
        with st.spinner("📥 Loading and processing data..."):
            sheet_id_to_use = RobustSessionState.safe_get('wd_sheet_id') or CONFIG.DEFAULT_SHEET_URL.split('/d/')[1].split('/')[0]
            if RobustSessionState.safe_get('wd_data_source') == "upload":
                ranked_df, data_timestamp, metadata = load_and_process_data("upload", file_data=uploaded_file, sheet_id=sheet_id_to_use)
            else:
                ranked_df, data_timestamp, metadata = load_and_process_data("sheet", sheet_id=sheet_id_to_use, gid=RobustSessionState.safe_get('wd_gid'))
            
            RobustSessionState.safe_set('wd_ranked_df', ranked_df)
            RobustSessionState.safe_set('wd_data_timestamp', data_timestamp)
            RobustSessionState.safe_set('wd_last_refresh', datetime.now(timezone.utc))
            
            if metadata.get('warnings'): [st.warning(w) for w in metadata['warnings']]
            if metadata.get('errors'): [st.error(e) for e in metadata['errors']]
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        if RobustSessionState.safe_get('wd_last_good_data'):
            ranked_df, data_timestamp, metadata = RobustSessionState.safe_get('wd_last_good_data')
            st.warning("Failed to load fresh data, using cached version.")
        else:
            st.error(f"❌ Critical Error: {e}"); st.info("Common issues: Invalid sheet ID, permissions, or network failure."); st.stop()
    
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
    quick_filter_applied = RobustSessionState.safe_get('wd_quick_filter_applied', False)
    quick_filter = RobustSessionState.safe_get('wd_quick_filter')
    
    with qa_col1:
        if st.button("📈 Top Gainers", use_container_width=True): RobustSessionState.safe_set('wd_quick_filter', 'top_gainers'); RobustSessionState.safe_set('wd_quick_filter_applied', True); st.rerun()
    with qa_col2:
        if st.button("🔥 Volume Surges", use_container_width=True): RobustSessionState.safe_set('wd_quick_filter', 'volume_surges'); RobustSessionState.safe_set('wd_quick_filter_applied', True); st.rerun()
    with qa_col3:
        if st.button("🎯 Breakout Ready", use_container_width=True): RobustSessionState.safe_set('wd_quick_filter', 'breakout_ready'); RobustSessionState.safe_set('wd_quick_filter_applied', True); st.rerun()
    with qa_col4:
        if st.button("💎 Hidden Gems", use_container_width=True): RobustSessionState.safe_set('wd_quick_filter', 'hidden_gems'); RobustSessionState.safe_set('wd_quick_filter_applied', True); st.rerun()
    with qa_col5:
        if st.button("🌊 Show All", use_container_width=True): RobustSessionState.safe_set('wd_quick_filter', None); RobustSessionState.safe_set('wd_quick_filter_applied', False); st.rerun()
    
    if quick_filter and ranked_df is not None and not ranked_df.empty:
        if quick_filter == 'top_gainers':
            ranked_df_display = ranked_df[ranked_df['momentum_score'].fillna(0) >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ 80.")
        elif quick_filter == 'volume_surges':
            ranked_df_display = ranked_df[ranked_df['rvol'].fillna(0) >= 3]
            st.info(f"Showing {len(ranked_df_display)} stocks with RVOL ≥ 3x.")
        elif quick_filter == 'breakout_ready':
            ranked_df_display = ranked_df[ranked_df['breakout_score'].fillna(0) >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with breakout score ≥ 80.")
        elif quick_filter == 'hidden_gems':
            ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('💎 HIDDEN GEM', na=False)]
            st.info(f"Showing {len(ranked_df_display)} hidden gem stocks.")
    else:
        ranked_df_display = ranked_df
    
    with st.sidebar:
        filters = {}
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio("Choose your view:", options=["Technical", "Hybrid (Technical + Fundamentals)"], index=0 if RobustSessionState.safe_get('wd_user_preferences', {}).get('display_mode', 'Technical') == 'Technical' else 1, key="wd_display_mode_toggle")
        RobustSessionState.safe_set('wd_user_preferences', {'default_top_n': RobustSessionState.safe_get('wd_user_preferences').get('default_top_n', 50), 'display_mode': display_mode, 'last_filters': RobustSessionState.safe_get('wd_user_preferences').get('last_filters', {})})
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        filters['categories'] = st.multiselect("Market Cap Category", options=FilterEngine.get_filter_options(ranked_df_display, 'category', filters), default=RobustSessionState.safe_get('wd_category_filter', []), key="wd_category_filter", placeholder="Select categories (empty = All)")
        filters['sectors'] = st.multiselect("Sector", options=FilterEngine.get_filter_options(ranked_df_display, 'sector', filters), default=RobustSessionState.safe_get('wd_sector_filter', []), key="wd_sector_filter", placeholder="Select sectors (empty = All)")
        filters['industries'] = st.multiselect("Industry", options=FilterEngine.get_filter_options(ranked_df_display, 'industry', filters), default=RobustSessionState.safe_get('wd_industry_filter', []), key="wd_industry_filter", placeholder="Select industries (empty = All)")
        filters['min_score'] = st.slider("Minimum Master Score", min_value=0, max_value=100, value=RobustSessionState.safe_get('wd_min_score', 0), step=5, key="wd_min_score")
        all_patterns = sorted(list({p for p_str in ranked_df_display['patterns'].dropna() for p in p_str.split(' | ')}))
        if all_patterns: filters['patterns'] = st.multiselect("Patterns", options=all_patterns, default=RobustSessionState.safe_get('wd_patterns', []), key="wd_patterns", placeholder="Select patterns (empty = All)")
        
        st.markdown("#### 📈 Trend Strength")
        trend_options = {"All Trends": (0, 100), "🔥 Strong Uptrend (80+)": (80, 100), "✅ Good Uptrend (60-79)": (60, 79), "➡️ Neutral Trend (40-59)": (40, 59), "⚠️ Weak/Downtrend (<40)": (0, 39)}
        filters['trend_filter'] = st.selectbox("Trend Quality", options=list(trend_options.keys()), index=list(trend_options.keys()).index(RobustSessionState.safe_get('wd_trend_filter', "All Trends")), key="wd_trend_filter")
        filters['trend_range'] = trend_options[filters['trend_filter']]
        
        st.markdown("#### 🌊 Wave Filters")
        filters['wave_states'] = st.multiselect("Wave State", options=FilterEngine.get_filter_options(ranked_df_display, 'wave_state', filters), default=RobustSessionState.safe_get('wd_wave_states_filter', []), key="wd_wave_states_filter", placeholder="Select wave states (empty = All)")
        min_strength, max_strength = 0, 100
        current_range = RobustSessionState.safe_get('wd_wave_strength_range_slider', (min_strength, max_strength))
        filters['wave_strength_range'] = st.slider("Overall Wave Strength", min_value=min_strength, max_value=max_strength, value=current_range, step=1, key="wd_wave_strength_range_slider")
        
        with st.expander("🔧 Advanced Filters"):
            for tier_key, col_name in [('eps_tiers', 'eps_tier'), ('pe_tiers', 'pe_tier'), ('price_tiers', 'price_tier')]:
                if col_name in ranked_df_display.columns:
                    filters[tier_key] = st.multiselect(col_name.replace('_', ' ').title(), options=FilterEngine.get_filter_options(ranked_df_display, col_name, filters), default=RobustSessionState.safe_get(f'wd_{col_name}_filter', []), key=f'wd_{col_name}_filter', placeholder=f"Select {col_name.replace('_', ' ')}s (empty = All)")
            if 'eps_change_pct' in ranked_df_display.columns:
                eps_change_input = st.text_input("Min EPS Change %", value=RobustSessionState.safe_get('wd_min_eps_change', ""), placeholder="e.g. -50 or empty", key="wd_min_eps_change")
                filters['min_eps_change'] = float(eps_change_input) if eps_change_input.strip() else None
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                st.markdown("**🔍 Fundamental Filters**")
                c1, c2 = st.columns(2)
                with c1:
                    min_pe_input = st.text_input("Min PE Ratio", value=RobustSessionState.safe_get('wd_min_pe', ""), placeholder="e.g. 10", key="wd_min_pe")
                    filters['min_pe'] = float(min_pe_input) if min_pe_input.strip() else None
                with c2:
                    max_pe_input = st.text_input("Max PE Ratio", value=RobustSessionState.safe_get('wd_max_pe', ""), placeholder="e.g. 30", key="wd_max_pe")
                    filters['max_pe'] = float(max_pe_input) if max_pe_input.strip() else None
                filters['require_fundamental_data'] = st.checkbox("Only show stocks with PE and EPS data", value=RobustSessionState.safe_get('wd_require_fundamental_data', False), key="wd_require_fundamental_data")
    
    filtered_df = FilterEngine.apply_filters(ranked_df_display, filters).sort_values('rank')
    RobustSessionState.safe_set('wd_filters', filters)
    
    if show_debug:
        with st.sidebar.expander("🐛 Debug Info", expanded=True):
            st.write(f"**Active Filters:** {RobustSessionState.safe_get('wd_filters')}")
            st.write(f"**Filter Result:** Before: {len(ranked_df_display)} stocks, After: {len(filtered_df)} stocks")
            st.write(f"**Performance:** {RobustSessionState.safe_get('wd_performance_metrics')}")
            
    if RobustSessionState.safe_get('wd_active_filter_count') > 0 or quick_filter_applied:
        c1, c2 = st.columns([5, 1])
        with c1: st.info(f"**{RobustSessionState.safe_get('wd_active_filter_count')} filter{'s' if RobustSessionState.safe_get('wd_active_filter_count') > 1 else ''} active** | **{len(filtered_df):,} stocks** shown")
        with c2: st.button("Clear Filters", type="secondary")
        if st.session_state.get('wd_clear_filters_main_button'): RobustSessionState.clear_filters(); st.rerun()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: UIComponents.render_metric_card("Total Stocks", f"{len(filtered_df):,}", f"{(len(filtered_df)/len(ranked_df)*100):.0f}% of all" if len(ranked_df)>0 else "0%")
    with c2: UIComponents.render_metric_card("Avg Score", f"{filtered_df['master_score'].mean():.1f}" if not filtered_df.empty else "N/A", f"σ={filtered_df['master_score'].std():.1f}" if not filtered_df.empty else None)
    with c3: UIComponents.render_metric_card("Median PE", f"{filtered_df[filtered_df['pe']>0]['pe'].median():.1f}x" if show_fundamentals and not filtered_df[filtered_df['pe']>0].empty else "N/A")
    with c4: UIComponents.render_metric_card("EPS Growth +ve", (filtered_df['eps_change_pct'].fillna(0)>0).sum() if show_fundamentals else (filtered_df['acceleration_score'].fillna(0)>=80).sum())
    with c5: UIComponents.render_metric_card("High RVOL", (filtered_df['rvol'].fillna(0)>2).sum() if 'rvol' in filtered_df.columns else 0)
    with c6: UIComponents.render_metric_card("Strong Trends", (filtered_df['trend_quality'].fillna(0)>=80).sum() if 'trend_quality' in filtered_df.columns else 0)

    tabs = st.tabs(["📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"])
    
    with tabs[0]: UIComponents.render_summary_section(filtered_df)
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        c1, c2, c3 = st.columns([2, 2, 6])
        with c1: display_count = st.selectbox("Show top", options=CONFIG.AVAILABLE_TOP_N, index=CONFIG.AVAILABLE_TOP_N.index(RobustSessionState.safe_get('wd_user_preferences').get('default_top_n', 50)), key="wd_rankings_display_count")
        with c2: sort_by = st.selectbox("Sort by", options=['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow', 'Trend'], index=0, key="wd_rankings_sort_by")
        
        display_df = filtered_df.copy()
        if sort_by == 'Master Score': display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL': display_df = display_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum': display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Money Flow': display_df = display_df.sort_values('money_flow_mm', ascending=False)
        elif sort_by == 'Trend': display_df = display_df.sort_values('trend_quality', ascending=False)
        
        paginated_df = UIComponents.render_pagination_controls(display_df, display_count, 'rankings')
        
        display_cols = {
            'rank': 'Rank', 'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score', 'wave_state': 'Wave', 'trend_quality': 'Trend',
            'price': 'Price', 'pe': 'PE', 'eps_change_pct': 'EPS Δ%', 'from_low_pct': 'From Low', 'ret_30d': '30D Ret', 'rvol': 'RVOL',
            'vmi': 'VMI', 'patterns': 'Patterns', 'category': 'Category', 'sector': 'Sector', 'industry': 'Industry'
        }
        
        paginated_df = paginated_df.rename(columns=display_cols).filter(items=list(display_cols.values()))
        st.dataframe(paginated_df, use_container_width=True, height=min(600, len(paginated_df) * 35 + 50), hide_index=True)
    
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1: wave_tf = st.selectbox("Timeframe", options=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"], key="wd_wave_timeframe_select")
        with c2: sensitivity = st.select_slider("Sensitivity", options=["Conservative", "Balanced", "Aggressive"], value=RobustSessionState.safe_get('wd_wave_sensitivity', "Balanced"), key="wd_wave_sensitivity")
        with c3: show_regime = st.checkbox("📊 Market Regime Analysis", value=RobustSessionState.safe_get('wd_show_market_regime'), key="wd_show_market_regime")
        with c4: UIComponents.render_metric_card("Wave Strength", f"🌊 {filtered_df['overall_wave_strength'].mean():.0f}%" if 'overall_wave_strength' in filtered_df.columns else "N/A", "Market")
        
        wave_df = filtered_df.copy()
        if wave_tf == "Intraday Surge": wave_df = wave_df[(wave_df['rvol'].fillna(0) >= 2.5) & (wave_df['ret_1d'].fillna(0) > 2)]
        elif wave_tf == "3-Day Buildup": wave_df = wave_df[(wave_df['ret_3d'].fillna(0) > 5) & (wave_df['vol_ratio_7d_90d'].fillna(0) > 1.5)]
        elif wave_tf == "Weekly Breakout": wave_df = wave_df[(wave_df['ret_7d'].fillna(0) > 8) & (wave_df['vol_ratio_7d_90d'].fillna(0) > 2.0) & (wave_df['from_high_pct'].fillna(-100) > -10)]
        elif wave_tf == "Monthly Trend": wave_df = wave_df[(wave_df['ret_30d'].fillna(0) > 15) & (wave_df['price'].fillna(0) > wave_df['sma_20d'].fillna(0)) & (wave_df['vol_ratio_30d_180d'].fillna(0) > 1.2)]
        
        st.markdown("#### 🚀 Momentum Shifts")
        mom_threshold = 60 if sensitivity == "Conservative" else 50 if sensitivity == "Balanced" else 40
        accel_threshold = 70 if sensitivity == "Conservative" else 60 if sensitivity == "Balanced" else 50
        min_rvol = 3.0 if sensitivity == "Conservative" else 2.0 if sensitivity == "Balanced" else 1.5
        
        shifts = wave_df[(wave_df['momentum_score'].fillna(0) >= mom_threshold) & (wave_df['acceleration_score'].fillna(0) >= accel_threshold) & (wave_df['rvol'].fillna(0) >= min_rvol)]
        st.dataframe(shifts.nlargest(20, 'master_score', keep='first').filter(items=['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'wave_state', 'patterns']), use_container_width=True)
    
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(Visualizer.create_score_distribution(filtered_df), use_container_width=True)
        with c2:
            pattern_counts = defaultdict(int)
            for p_str in filtered_df['patterns'].dropna():
                if p_str: [pattern_counts.update({p: pattern_counts[p] + 1}) for p in p_str.split(' | ')]
            if pattern_counts:
                fig_patterns = go.Figure(go.Bar(x=list(pattern_counts.values()), y=list(pattern_counts.keys()), orientation='h'))
                st.plotly_chart(fig_patterns, use_container_width=True)
            else: st.info("No patterns detected in current selection.")
            
        st.markdown("---")
        st.markdown("#### 📈 Performance Analysis")
        perf_tabs = st.tabs(["🏢 Sector Performance", "🏭 Industry Performance"])
        with perf_tabs[0]:
            sector_df = MarketIntelligence.detect_sector_rotation(filtered_df)
            if not sector_df.empty: st.dataframe(sector_df.style.background_gradient(subset=['flow_score']), use_container_width=True)
        with perf_tabs[1]:
            industry_df = MarketIntelligence.detect_industry_rotation(filtered_df)
            if not industry_df.empty: st.dataframe(industry_df.style.background_gradient(subset=['flow_score']), use_container_width=True)

    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        search_query = st.text_input("Search stocks", value=RobustSessionState.safe_get('wd_search_input'), placeholder="Enter ticker or company name...", key="wd_search_input_widget")
        if search_query:
            results = SearchEngine.search_stocks(filtered_df, search_query)
            if not results.empty:
                st.success(f"Found {len(results)} matching stocks.")
                for _, s in results.iterrows():
                    with st.expander(f"📊 {s['ticker']} - {s['company_name']} (Rank #{s['rank']:.0f})"):
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        with c1: UIComponents.render_metric_card("Score", f"{s['master_score']:.1f}", f"Rank #{s['rank']:.0f}")
                        with c2: UIComponents.render_metric_card("Price", f"₹{s['price']:,.0f}" if pd.notna(s['price']) else "N/A", f"{s['ret_1d']:+.1f}%" if pd.notna(s['ret_1d']) else None)
                        with c3: UIComponents.render_metric_card("From Low", f"{s['from_low_pct']:.0f}%" if pd.notna(s['from_low_pct']) else "N/A")
                        with c4: UIComponents.render_metric_card("30D Ret", f"{s['ret_30d']:+.1f}%" if pd.notna(s['ret_30d']) else "N/A")
                        with c5: UIComponents.render_metric_card("RVOL", f"{s['rvol']:.1f}x" if pd.notna(s['rvol']) else "N/A")
                        with c6: UIComponents.render_metric_card("Wave", s.get('wave_state', 'N/A'), s.get('category', 'N/A'))
                        st.markdown(f"**Patterns:** {s.get('patterns', 'None')}")
            else: st.warning("No stocks found matching your search.")

    with tabs[5]:
        st.markdown("### 📥 Export Data")
        export_template = st.radio("Choose export template:", options=["Full Analysis (All Data)", "Day Trader Focus", "Swing Trader Focus", "Investor Focus"], key="wd_export_template_radio")
        template_map = {"Full Analysis (All Data)": "full", "Day Trader Focus": "day_trader", "Swing Trader Focus": "swing_trader", "Investor Focus": "investor"}
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate Excel Report", type="primary", use_container_width=True):
                if not filtered_df.empty:
                    excel_file = ExportEngine.create_excel_report(filtered_df, template=template_map[export_template])
                    st.download_button(label="📥 Download Excel Report", data=excel_file, file_name=f"wave_detection_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else: st.error("No data to export.")
        with c2:
            if st.button("Generate CSV Export", use_container_width=True):
                if not filtered_df.empty:
                    csv_data = ExportEngine.create_csv_export(filtered_df)
                    st.download_button(label="📥 Download CSV File", data=csv_data, file_name=f"wave_detection_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                else: st.error("No data to export.")
        
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0 - Final Perfected Production Version")
        st.markdown("This application is the culmination of extensive development, combining robustness, performance, and a rich feature set into a single, reliable tool for professional stock analysis.")
        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Key Enhancements in this Version")
            st.markdown("- **Robust Data Loading:** Implements smart retry logic for stable data fetching from Google Sheets.")
            st.markdown("- **Perfected Filtering:** Filters are now fully interconnected, ensuring a seamless user experience.")
            st.markdown("- **Performance Optimization:** Critical functions are vectorized for speed, making the application highly responsive.")
        with c2:
            st.markdown("#### Version & Status")
            st.markdown(f"**Version**: 3.1.0-FINAL-PERFECTED")
            st.markdown(f"**Last Updated**: December 2024")
            st.markdown(f"**Status**: PRODUCTION READY - PERMANENTLY LOCKED")
        
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        stats_cols = st.columns(4)
        with stats_cols[0]: UIComponents.render_metric_card("Total Stocks Loaded", f"{len(ranked_df):,}")
        with stats_cols[1]: UIComponents.render_metric_card("Currently Filtered", f"{len(filtered_df):,}")
        with stats_cols[2]: UIComponents.render_metric_card("Data Quality", f"{RobustSessionState.safe_get('wd_data_quality', {}).get('completeness', 0):.1f}%")
        with stats_cols[3]:
            cache_time = (datetime.now(timezone.utc) - RobustSessionState.safe_get('wd_last_refresh', datetime.now(timezone.utc))).total_seconds() / 60
            cache_status = "Fresh" if cache_time < 60 else "Stale"
            st.metric("Cache Age", f"{cache_time:.0f} min", cache_status)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666; padding: 1rem;'><small>🌊 Wave Detection Ultimate 3.0 - Final Perfected Production Version</small></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical Application Error: {e}")
        logger.error(f"Application crashed: {e}", exc_info=True)
        if st.button("🔄 Restart Application"):
            st.cache_data.clear()
            st.rerun()
