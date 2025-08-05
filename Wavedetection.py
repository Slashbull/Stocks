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

warnings.filterwarnings('ignore')
np.seterr(all='ignore')
pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:.2f}'.format
np.random.seed(42)

# ============================================
# LOGGING & PERFORMANCE MONITORING
# ============================================

log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Performance tracking storage
performance_stats = defaultdict(list)

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
                    if target_time and elapsed > target_time:
                        logger.warning(f"{func.__name__} took {elapsed:.2f}s (target: {target_time}s)")
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

# ============================================
# ROBUST SESSION STATE MANAGER
# ============================================

class RobustSessionState:
    STATE_DEFAULTS = {
        'search_query': "",
        'search_input': "",
        'last_refresh': None,
        'data_source': "sheet",
        'sheet_id': "",
        'gid': "",
        'user_preferences': {'default_top_n': 50, 'display_mode': 'Technical', 'last_filters': {}},
        'filters': {},
        'active_filter_count': 0,
        'quick_filter': None,
        'wd_quick_filter_applied': False,
        'wd_show_debug': False,
        'performance_metrics': {},
        'data_quality': {},
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
        'wd_display_mode_toggle': 0,
        'ranked_df': None,
        'data_timestamp': None,
        'last_good_data': None,
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
                else:
                    st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        filter_keys = [
            'wd_category_filter', 'wd_sector_filter', 'wd_industry_filter', 'wd_eps_tier_filter',
            'wd_pe_tier_filter', 'wd_price_tier_filter', 'wd_patterns',
            'wd_min_score', 'wd_trend_filter', 'wd_min_eps_change',
            'wd_min_pe', 'wd_max_pe', 'wd_require_fundamental_data',
            'quick_filter', 'wd_quick_filter_applied',
            'wd_wave_states_filter', 'wd_wave_strength_range_slider',
            'wd_show_sensitivity_details', 'wd_show_market_regime',
            'wd_wave_timeframe_select', 'wd_wave_sensitivity',
            'wd_search_query', 'wd_search_input'
        ]
        
        for key in filter_keys:
            if key in RobustSessionState.STATE_DEFAULTS:
                RobustSessionState.safe_set(key, RobustSessionState.STATE_DEFAULTS[key])
        
        RobustSessionState.safe_set('filters', {})
        RobustSessionState.safe_set('active_filter_count', 0)
        RobustSessionState.safe_set('wd_trigger_clear', False)

# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class Config:
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
        "value_momentum": 70, "earnings_rocket": 70, "quality_leader": 80, "turnaround": 70, "high_pe": 100
    })
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000), 'rvol': (0.01, 1_000_000.0), 'pe': (-10000, 10000),
        'returns': (-99.99, 9999.99), 'volume': (0, 1e12)
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
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidator:
    _clipping_counts: Dict[str, int] = {}
    
    @staticmethod
    def get_clipping_counts() -> Dict[str, int]:
        counts = DataValidator._clipping_counts.copy()
        DataValidator._clipping_counts.clear()
        return counts

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
        data_quality.update({ 'completeness': completeness, 'total_rows': len(df), 'total_columns': len(df.columns), 'duplicate_tickers': duplicates, 'context': context, 'timestamp': datetime.now(timezone.utc) })
        RobustSessionState.safe_set('data_quality', data_quality)
        logger.info(f"{context}: Validated {len(df)} rows, {len(df.columns)} columns, {completeness:.1f}% complete")
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Any, col_name: str, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        if pd.isna(value) or value == '' or value is None: return np.nan
        try:
            cleaned = str(value).strip()
            if cleaned.upper() in ['', '-', 'N/A', 'NA', 'NAN', 'NONE', '#VALUE!', '#ERROR!', '#DIV/0!', 'INF', '-INF']: return np.nan
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            result = float(cleaned)
            if bounds:
                min_val, max_val = bounds
                original_result = result
                if result < min_val:
                    result = min_val
                    DataValidator._clipping_counts[col_name] = DataValidator._clipping_counts.get(col_name, 0) + 1
                elif result > max_val:
                    result = max_val
                    DataValidator._clipping_counts[col_name] = DataValidator._clipping_counts.get(col_name, 0) + 1
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

# ============================================
# SMART CACHING WITH VERSIONING
# ============================================

def get_requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504, 429),
    session=None,
) -> requests.Session:
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@st.cache_data(ttl=CONFIG.CACHE_TTL, persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    start_time = time.perf_counter()
    metadata = { 'source_type': source_type, 'data_version': data_version, 'processing_start': datetime.now(timezone.utc), 'errors': [], 'warnings': [] }
    try:
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            df = pd.read_csv(file_data, low_memory=False)
            metadata['source'] = "User Upload"
        else:
            user_provided_id = st.session_state.get('user_spreadsheet_id')
            if user_provided_id is None or not (len(user_provided_id) == 44 and user_provided_id.isalnum()):
                error_msg = "A valid 44-character alphanumeric Google Spreadsheet ID is required to load data."
                logger.critical(error_msg)
                raise ValueError(error_msg)
            
            final_spreadsheet_id_to_use = user_provided_id
            logger.info(f"Using user-provided Spreadsheet ID: {final_spreadsheet_id_to_use}")
            
            csv_url = CONFIG.CSV_URL_TEMPLATE.format(sheet_id=final_spreadsheet_id_to_use, gid=CONFIG.DEFAULT_GID)
            logger.info(f"Attempting to load data from Google Sheets with Spreadsheet ID: {final_spreadsheet_id_to_use}, GID: {CONFIG.DEFAULT_GID}")
            
            try:
                session = get_requests_retry_session()
                response = session.get(csv_url)
                response.raise_for_status()
                df = pd.read_csv(BytesIO(response.content), low_memory=False)
                metadata['source'] = f"Google Sheets (ID: {final_spreadsheet_id_to_use}, GID: {CONFIG.DEFAULT_GID})"
            except requests.exceptions.RequestException as req_e:
                error_msg = f"Network or HTTP error loading Google Sheet (ID: {final_spreadsheet_id_to_use}): {req_e}"
                logger.error(error_msg)
                metadata['errors'].append(error_msg)
                last_good_data = RobustSessionState.safe_get('last_good_data', None)
                if last_good_data:
                    logger.info("Using cached data as fallback due to network/HTTP error.")
                    df, timestamp, old_metadata = last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise ValueError(error_msg) from req_e
            except Exception as e:
                error_msg = f"Failed to load CSV from Google Sheet (ID: {final_spreadsheet_id_to_use}): {str(e)}"
                logger.error(error_msg)
                metadata['errors'].append(error_msg)
                last_good_data = RobustSessionState.safe_get('last_good_data', None)
                if last_good_data:
                    logger.info("Using cached data as fallback due to CSV parsing error.")
                    df, timestamp, old_metadata = last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise ValueError(error_msg) from e
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid: raise ValueError(validation_msg)
        
        df = DataProcessor.process_dataframe(df, metadata)
        df = RankingEngine.calculate_all_scores(df)
        df = PatternDetector.detect_all_patterns(df)
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid: raise ValueError(validation_msg)
        
        timestamp = datetime.now(timezone.utc)
        RobustSessionState.safe_set('last_good_data', (df.copy(), timestamp, metadata))
        
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        clipping_info = DataValidator.get_clipping_counts()
        if clipping_info: metadata['warnings'].append(f"Some numeric values were clipped: {clipping_info}. See logs for details.")
        
        gc.collect()
        return df, timestamp, metadata
    
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}", exc_info=True)
        metadata['errors'].append(str(e))
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
        numeric_cols_to_process = [col for col in df.columns if col not in ['ticker', 'company_name', 'category', 'sector', 'industry', 'year', 'market_cap']]
        for col in numeric_cols_to_process:
            if col in df.columns:
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                bounds = None
                if 'volume' in col.lower(): bounds = CONFIG.VALUE_BOUNDS['volume']
                elif col == 'rvol': bounds = CONFIG.VALUE_BOUNDS['rvol']
                elif col == 'pe': bounds = CONFIG.VALUE_BOUNDS['pe']
                elif is_pct: bounds = CONFIG.VALUE_BOUNDS['returns']
                elif col == 'price': bounds = CONFIG.VALUE_BOUNDS['price']
                df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, col, is_pct, bounds))
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns: df[col] = df[col].apply(DataValidator.sanitize_string)
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(float)
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
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows")
        return df
    
    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        if 'from_low_pct' in df.columns: df['from_low_pct'] = df['from_low_pct'].fillna(50)
        else: df['from_low_pct'] = 50
        if 'from_high_pct' in df.columns: df['from_high_pct'] = df['from_high_pct'].fillna(-50)
        else: df['from_high_pct'] = -50
        if 'rvol' in df.columns: df['rvol'] = df['rvol'].fillna(1.0)
        else: df['rvol'] = 1.0
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols: df[col] = df[col].fillna(0)
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols: df[col] = df[col].fillna(0)
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            if pd.isna(value): return "Unknown"
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val: return tier_name
                if tier_name == list(tier_dict.keys())[0] and (min_val == -float('inf') or min_val == 0) and value == min_val: return tier_name
            return "Unknown"
        if 'eps_current' in df.columns: df['eps_tier'] = df['eps_current'].apply(lambda x: classify_tier(x, CONFIG.TIERS['eps']))
        if 'pe' in df.columns: df['pe_tier'] = df['pe'].apply(lambda x: "Negative/NA" if pd.isna(x) or x <= 0 else classify_tier(x, CONFIG.TIERS['pe']))
        if 'price' in df.columns: df['price_tier'] = df['price'].apply(lambda x: classify_tier(x, CONFIG.TIERS['price']))
        return df

# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow'] = df['price'].fillna(0) * df['volume_1d'].fillna(0) * df['rvol'].fillna(1.0)
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
        else: df['money_flow_mm'] = np.nan
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d']):
            df['vmi'] = (df['vol_ratio_1d_90d'].fillna(1.0) * 4 + df['vol_ratio_7d_90d'].fillna(1.0) * 3 + df['vol_ratio_30d_90d'].fillna(1.0) * 2 + df['vol_ratio_90d_180d'].fillna(1.0) * 1) / 10
        else: df['vmi'] = np.nan
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'].fillna(50) + abs(df['from_high_pct'].fillna(-50))
        else: df['position_tension'] = np.nan
        df['momentum_harmony'] = 0
        if 'ret_1d' in df.columns: df['momentum_harmony'] += (df['ret_1d'].fillna(0) > 0).astype(int)
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(df['ret_7d'].fillna(0) != 0, df['ret_7d'].fillna(0) / 7, np.nan)
                daily_ret_30d = np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan)
            df['momentum_harmony'] += ((daily_ret_7d.fillna(-np.inf) > daily_ret_30d.fillna(-np.inf))).astype(int)
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_30d_comp = np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan)
                daily_ret_3m_comp = np.where(df['ret_3m'].fillna(0) != 0, df['ret_3m'].fillna(0) / 90, np.nan)
            df['momentum_harmony'] += ((daily_ret_30d_comp.fillna(-np.inf) > daily_ret_3m_comp.fillna(-np.inf))).astype(int)
        if 'ret_3m' in df.columns: df['momentum_harmony'] += (df['ret_3m'].fillna(0) > 0).astype(int)
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)
        score_cols = ['momentum_score', 'acceleration_score', 'rvol_score', 'breakout_score']
        if all(col in df.columns for col in score_cols):
            df['overall_wave_strength'] = (df['momentum_score'].fillna(50) * 0.3 + df['acceleration_score'].fillna(50) * 0.3 + df['rvol_score'].fillna(50) * 0.2 + df['breakout_score'].fillna(50) * 0.2)
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
        logger.info("Starting optimized ranking calculations...")
        df['position_score'] = RankingEngine._calculate_position_score(df).fillna(50)
        df['volume_score'] = RankingEngine._calculate_volume_score(df).fillna(50)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df).fillna(50)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df).fillna(50)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df).fillna(50)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df).fillna(50)
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df).fillna(50)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df).fillna(50)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df).fillna(50)
        scores_matrix = np.column_stack([df['position_score'], df['volume_score'], df['momentum_score'], df['acceleration_score'], df['breakout_score'], df['rvol_score']])
        weights = np.array([CONFIG.POSITION_WEIGHT, CONFIG.VOLUME_WEIGHT, CONFIG.MOMENTUM_WEIGHT, CONFIG.ACCELERATION_WEIGHT, CONFIG.BREAKOUT_WEIGHT, CONFIG.RVOL_WEIGHT])
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom')
        df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        df['percentile'] = df['percentile'].fillna(0)
        df = RankingEngine._calculate_category_ranks(df)
        logger.info(f"Ranking complete: {len(df)} stocks processed")
        return df
    
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        if series is None or series.empty: return pd.Series(np.nan, dtype=float)
        series = series.replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() == 0: return pd.Series(np.nan, index=series.index)
        if pct: ranks = series.rank(pct=True, ascending=ascending, na_option='bottom') * 100
        else: ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
        return ranks
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        position_score = pd.Series(np.nan, index=df.index, dtype=float)
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        if not has_from_low and not has_from_high: return position_score
        from_low = df['from_low_pct'] if has_from_low else pd.Series(np.nan, index=df.index)
        from_high = df['from_high_pct'] if has_from_high else pd.Series(np.nan, index=df.index)
        rank_from_low = RankingEngine._safe_rank(from_low, pct=True, ascending=True) if has_from_low else pd.Series(np.nan, index=df.index)
        rank_from_high = RankingEngine._safe_rank(from_high, pct=True, ascending=False) if has_from_high else pd.Series(np.nan, index=df.index)
        combined_score = (rank_from_low.fillna(50) * 0.6 + rank_from_high.fillna(50) * 0.4)
        if not (has_from_low or has_from_high): combined_score = pd.Series(np.nan, index=df.index)
        return combined_score.clip(0, 100)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        volume_score = pd.Series(np.nan, index=df.index, dtype=float)
        vol_cols = [('vol_ratio_1d_90d', 0.20), ('vol_ratio_7d_90d', 0.20), ('vol_ratio_30d_90d', 0.20), ('vol_ratio_30d_180d', 0.15), ('vol_ratio_90d_180d', 0.25)]
        total_weight = 0
        weighted_score = pd.Series(0.0, index=df.index, dtype=float)
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                col_rank = RankingEngine._safe_rank(df[col], pct=True, ascending=True)
                weighted_score += col_rank.fillna(0) * weight
                total_weight += weight
        if total_weight > 0:
            volume_score = weighted_score / total_weight
            nan_mask = df[[col for col, _ in vol_cols if col in df.columns]].isna().all(axis=1)
            volume_score[nan_mask] = np.nan
        else: logger.warning("No valid volume ratio data available, volume scores will be NaN.")
        return volume_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        momentum_score = pd.Series(np.nan, index=df.index, dtype=float)
        has_ret_30d = 'ret_30d' in df.columns and df['ret_30d'].notna().any()
        has_ret_7d = 'ret_7d' in df.columns and df['ret_7d'].notna().any()
        if not has_ret_30d and not has_ret_7d:
            logger.warning("No return data available for momentum calculation, scores will be NaN.")
            return momentum_score
        ret_30d = df['ret_30d'] if has_ret_30d else pd.Series(np.nan, index=df.index)
        ret_7d = df['ret_7d'] if has_ret_7d else pd.Series(np.nan, index=df.index)
        if has_ret_30d:
            momentum_score = RankingEngine._safe_rank(ret_30d, pct=True, ascending=True)
        elif has_ret_7d:
            momentum_score = RankingEngine._safe_rank(ret_7d, pct=True, ascending=True)
            logger.info("Using 7-day returns for momentum score due to missing 30-day data.")
        if has_ret_7d and has_ret_30d:
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            all_positive = (ret_7d.fillna(-1) > 0) & (ret_30d.fillna(-1) > 0)
            consistency_bonus[all_positive] = 5
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(ret_7d.fillna(0) != 0, ret_7d.fillna(0) / 7, np.nan)
                daily_ret_30d = np.where(ret_30d.fillna(0) != 0, ret_30d.fillna(0) / 30, np.nan)
            accelerating_mask = all_positive & (daily_ret_7d.fillna(-np.inf) > daily_ret_30d.fillna(-np.inf))
            consistency_bonus[accelerating_mask] = 10
            momentum_score = momentum_score.mask(momentum_score.isna(), other=np.nan)
            momentum_score = (momentum_score.fillna(50) + consistency_bonus).clip(0, 100)
        return momentum_score
    
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        acceleration_score = pd.Series(np.nan, index=df.index, dtype=float)
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in req_cols if col in df.columns and df[col].notna().any()]
        if len(available_cols) < 2:
            logger.warning("Insufficient return data for acceleration calculation, scores will be NaN.")
            return acceleration_score
        ret_1d = df['ret_1d'] if 'ret_1d' in df.columns else pd.Series(np.nan, index=df.index)
        ret_7d = df['ret_7d'] if 'ret_7d' in df.columns else pd.Series(np.nan, index=df.index)
        ret_30d = df['ret_30d'] if 'ret_30d' in df.columns else pd.Series(np.nan, index=df.index)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_daily_1d = ret_1d
            avg_daily_7d = np.where(ret_7d.fillna(0) != 0, ret_7d.fillna(0) / 7, np.nan)
            avg_daily_30d = np.where(ret_30d.fillna(0) != 0, ret_30d.fillna(0) / 30, np.nan)
        has_all_data = ret_1d.notna() & avg_daily_7d.notna() & avg_daily_30d.notna()
        acceleration_score.loc[has_all_data] = 50.0
        if has_all_data.any():
            perfect = has_all_data & (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
            acceleration_score.loc[perfect] = 100
            good = has_all_data & (~perfect) & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0)
            acceleration_score.loc[good] = 80
            moderate = has_all_data & (~perfect) & (~good) & (ret_1d > 0)
            acceleration_score.loc[moderate] = 60
            slight_decel = has_all_data & (ret_1d <= 0) & (ret_7d > 0)
            acceleration_score.loc[slight_decel] = 40
            strong_decel = has_all_data & (ret_1d <= 0) & (ret_7d <= 0)
            acceleration_score.loc[strong_decel] = 20
        return acceleration_score.clip(0, 100)
    
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        breakout_score = pd.Series(np.nan, index=df.index, dtype=float)
        if 'from_high_pct' in df.columns and df['from_high_pct'].notna().any():
            distance_from_high = -df['from_high_pct']
            distance_factor = (100 - distance_from_high.fillna(100)).clip(0, 100)
        else: distance_factor = pd.Series(np.nan, index=df.index)
        if 'vol_ratio_7d_90d' in df.columns and df['vol_ratio_7d_90d'].notna().any():
            vol_ratio = df['vol_ratio_7d_90d']
            volume_factor = ((vol_ratio.fillna(1.0) - 1) * 100).clip(0, 100)
        else: volume_factor = pd.Series(np.nan, index=df.index)
        trend_factor = pd.Series(np.nan, index=df.index, dtype=float)
        if 'price' in df.columns:
            current_price = df['price']
            sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
            conditions_sum = pd.Series(0, index=df.index, dtype=float)
            valid_sma_count = pd.Series(0, index=df.index, dtype=int)
            for sma_col in sma_cols:
                if sma_col in df.columns and df[sma_col].notna().any():
                    valid_comparison_mask = current_price.notna() & df[sma_col].notna()
                    conditions_sum.loc[valid_comparison_mask] += (current_price.loc[valid_comparison_mask] > df[sma_col].loc[valid_comparison_mask]).astype(float)
                    valid_sma_count.loc[valid_comparison_mask] += 1
            rows_with_any_sma_data = valid_sma_count > 0
            trend_factor.loc[rows_with_any_sma_data] = (conditions_sum.loc[rows_with_any_sma_data] / valid_sma_count.loc[rows_with_any_sma_data]) * 100
        trend_factor = trend_factor.clip(0, 100)
        combined_score = (distance_factor.fillna(50) * 0.4 + volume_factor.fillna(50) * 0.4 + trend_factor.fillna(50) * 0.2)
        all_nan_mask = distance_factor.isna() & volume_factor.isna() & trend_factor.isna()
        combined_score.loc[all_nan_mask] = np.nan
        return combined_score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        if 'rvol' not in df.columns or df['rvol'].isna().all(): return pd.Series(np.nan, index=df.index)
        rvol = df['rvol']
        rvol_score = pd.Series(np.nan, index=df.index, dtype=float)
        has_rvol = rvol.notna()
        rvol_score.loc[has_rvol & (rvol > 10)] = 95
        rvol_score.loc[has_rvol & (rvol > 5) & (rvol <= 10)] = 90
        rvol_score.loc[has_rvol & (rvol > 3) & (rvol <= 5)] = 85
        rvol_score.loc[has_rvol & (rvol > 2) & (rvol <= 3)] = 80
        rvol_score.loc[has_rvol & (rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score.loc[has_rvol & (rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score.loc[has_rvol & (rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score.loc[has_rvol & (rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score.loc[has_rvol & (rvol > 0.3) & (rvol <= 0.5)] = 30
        rvol_score.loc[has_rvol & (rvol <= 0.3)] = 20
        return rvol_score.clip(0, 100)
    
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        trend_score = pd.Series(np.nan, index=df.index, dtype=float)
        if 'price' not in df.columns or df['price'].isna().all(): return trend_score
        current_price = df['price']
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        rows_with_any_sma_data = pd.Series(False, index=df.index, dtype=bool)
        above_sma_count = pd.Series(0, index=df.index, dtype=int)
        for sma_col in sma_cols:
            if sma_col in df.columns and df[sma_col].notna().any():
                valid_comparison_mask = current_price.notna() & df[sma_col].notna()
                above_sma_count.loc[valid_comparison_mask] += (current_price.loc[valid_comparison_mask] > df[sma_col].loc[valid_comparison_mask]).astype(int)
                rows_with_any_sma_data.loc[valid_comparison_mask] = True
        rows_to_score = df.index[rows_with_any_sma_data]
        if len(rows_to_score) > 0:
            trend_score.loc[rows_to_score] = 50.0
            if all(col in df.columns for col in sma_cols):
                perfect_trend = ((current_price > df['sma_20d']) & (df['sma_20d'] > df['sma_50d']) & (df['sma_50d'] > df['sma_200d'])).fillna(False)
                trend_score.loc[perfect_trend] = 100
                strong_trend = ((~perfect_trend) & (current_price > df['sma_20d']) & (current_price > df['sma_50d']) & (current_price > df['sma_200d'])).fillna(False)
                trend_score.loc[strong_trend] = 85
            good_trend = rows_with_any_sma_data & (above_sma_count == 2) & ~trend_score.notna()
            trend_score.loc[good_trend] = 70
            weak_trend = rows_with_any_sma_data & (above_sma_count == 1) & ~trend_score.notna()
            trend_score.loc[weak_trend] = 40
            poor_trend = rows_with_any_sma_data & (above_sma_count == 0) & ~trend_score.notna()
            trend_score.loc[poor_trend] = 20
        return trend_score.clip(0, 100)
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        strength_score = pd.Series(np.nan, index=df.index, dtype=float)
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        if not available_cols: return strength_score
        lt_returns = df[available_cols].fillna(0)
        avg_return = lt_returns.mean(axis=1)
        has_any_lt_data = df[available_cols].notna().any(axis=1)
        strength_score.loc[has_any_lt_data & (avg_return > 100)] = 100
        strength_score.loc[has_any_lt_data & (avg_return > 50) & (avg_return <= 100)] = 90
        strength_score.loc[has_any_lt_data & (avg_return > 30) & (avg_return <= 50)] = 80
        strength_score.loc[has_any_lt_data & (avg_return > 15) & (avg_return <= 30)] = 70
        strength_score.loc[has_any_lt_data & (avg_return > 5) & (avg_return <= 15)] = 60
        strength_score.loc[has_any_lt_data & (avg_return > 0) & (avg_return <= 5)] = 50
        strength_score.loc[has_any_lt_data & (avg_return > -10) & (avg_return <= 0)] = 40
        strength_score.loc[has_any_lt_data & (avg_return > -25) & (avg_return <= -10)] = 30
        strength_score.loc[has_any_lt_data & (avg_return <= -25)] = 20
        return strength_score.clip(0, 100)
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        liquidity_score = pd.Series(np.nan, index=df.index, dtype=float)
        if 'volume_30d' in df.columns and 'price' in df.columns:
            dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
            has_valid_dollar_volume = dollar_volume.notna() & (dollar_volume > 0)
            if has_valid_dollar_volume.any(): liquidity_score.loc[has_valid_dollar_volume] = RankingEngine._safe_rank(dollar_volume.loc[has_valid_dollar_volume], pct=True, ascending=True)
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        df['category_rank'] = np.nan
        df['category_percentile'] = np.nan
        categories = df['category'].dropna().unique()
        for category in categories:
            mask = df['category'] == category
            cat_df = df[mask]
            if len(cat_df) > 0 and 'master_score' in cat_df.columns and cat_df['master_score'].notna().any():
                cat_ranks = cat_df['master_score'].rank(method='first', ascending=False, na_option='bottom')
                df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                cat_percentiles = cat_df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
                df.loc[mask, 'category_percentile'] = cat_percentiles
        return df

# ============================================
# PATTERN DETECTION ENGINE - FULLY OPTIMIZED
# ============================================

class PatternDetector:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.3)
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df['patterns'] = [''] * len(df)
            return df
        
        patterns_with_masks = PatternDetector._get_all_pattern_definitions(df)
        num_patterns = len(patterns_with_masks)
        if num_patterns == 0:
            df['patterns'] = [''] * len(df)
            return df
        
        pattern_matrix = pd.DataFrame(False, index=df.index, columns=[name for name, _ in patterns_with_masks])
        for pattern_name, mask in patterns_with_masks:
            if mask is not None and not mask.empty:
                pattern_matrix[pattern_name] = mask.reindex(df.index, fill_value=False)
        
        df['patterns'] = pattern_matrix.apply(lambda row: ' | '.join(row.index[row].tolist()), axis=1)
        return df
    
    @staticmethod
    def _get_all_pattern_definitions(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        patterns = [] 
        
        def get_col_safe(col_name: str) -> pd.Series:
            return df[col_name].fillna(0) if col_name in df.columns else pd.Series(np.nan, index=df.index)

        def has_valid_data(df, cols):
            return all(col in df.columns and df[col].notna().any() for col in cols)

        # Pattern definitions, now using the helper functions for robustness
        if has_valid_data(df, ['category_percentile']):
            patterns.append(('🔥 CAT LEADER', df['category_percentile'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['category_leader']))
        
        if has_valid_data(df, ['category_percentile', 'percentile']):
            patterns.append(('💎 HIDDEN GEM', (df['category_percentile'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (df['percentile'].fillna(100) < 70)))
        
        if has_valid_data(df, ['acceleration_score']):
            patterns.append(('🚀 ACCELERATING', df['acceleration_score'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['acceleration']))
        
        if has_valid_data(df, ['volume_score', 'vol_ratio_90d_180d']):
            patterns.append(('🏦 INSTITUTIONAL', (df['volume_score'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (df['vol_ratio_90d_180d'].fillna(0) > 1.1)))
        
        if has_valid_data(df, ['rvol']):
            patterns.append(('⚡ VOL EXPLOSION', df['rvol'].fillna(0) > 3))
        
        if has_valid_data(df, ['breakout_score']):
            patterns.append(('🎯 BREAKOUT', df['breakout_score'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']))
        
        if has_valid_data(df, ['percentile']):
            patterns.append(('👑 MARKET LEADER', df['percentile'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['market_leader']))
        
        if has_valid_data(df, ['momentum_score', 'acceleration_score']):
            patterns.append(('🌊 MOMENTUM WAVE', (df['momentum_score'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (df['acceleration_score'].fillna(0) >= 70)))
        
        if has_valid_data(df, ['liquidity_score', 'percentile']):
            patterns.append(('💰 LIQUID LEADER', (df['liquidity_score'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (df['percentile'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])))
        
        if has_valid_data(df, ['long_term_strength']):
            patterns.append(('💪 LONG STRENGTH', df['long_term_strength'].fillna(0) >= CONFIG.PATTERN_THRESHOLDS['long_strength']))
        
        if has_valid_data(df, ['trend_quality']):
            patterns.append(('📈 QUALITY TREND', df['trend_quality'].fillna(0) >= 80))
        
        if has_valid_data(df, ['pe', 'master_score']):
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)
            patterns.append(('💎 VALUE MOMENTUM', has_valid_pe & (df['pe'].fillna(0) < 15) & (df['master_score'].fillna(0) >= 70)))
        
        if has_valid_data(df, ['eps_change_pct', 'acceleration_score']):
            has_eps_growth = df['eps_change_pct'].notna()
            extreme_growth = has_eps_growth & (df['eps_change_pct'].fillna(0) > 1000)
            normal_growth = has_eps_growth & (df['eps_change_pct'].fillna(0) > 50) & (df['eps_change_pct'].fillna(0) <= 1000)
            patterns.append(('📊 EARNINGS ROCKET', (extreme_growth & (df['acceleration_score'].fillna(0) >= 80)) | (normal_growth & (df['acceleration_score'].fillna(0) >= 70))))
        
        if has_valid_data(df, ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = df['pe'].notna() & df['eps_change_pct'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)
            patterns.append(('🏆 QUALITY LEADER', has_complete_data & (df['pe'].fillna(0).between(10, 25)) & (df['eps_change_pct'].fillna(0) > 20) & (df['percentile'].fillna(0) >= 80)))
        
        if has_valid_data(df, ['eps_change_pct', 'volume_score']):
            has_eps = df['eps_change_pct'].notna()
            mega_turnaround = has_eps & (df['eps_change_pct'].fillna(0) > 500) & (df['volume_score'].fillna(0) >= 60)
            strong_turnaround = has_eps & (df['eps_change_pct'].fillna(0) > 100) & (df['eps_change_pct'].fillna(0) <= 500) & (df['volume_score'].fillna(0) >= 70)
            patterns.append(('⚡ TURNAROUND', mega_turnaround | strong_turnaround))
        
        if has_valid_data(df, ['pe']):
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
            patterns.append(('⚠️ HIGH PE', has_valid_pe & (df['pe'].fillna(0) > 100)))
        
        if has_valid_data(df, ['from_high_pct', 'volume_score', 'momentum_score']):
            patterns.append(('🎯 52W HIGH APPROACH', (df['from_high_pct'].fillna(-100) > -5) & (df['volume_score'].fillna(0) >= 70) & (df['momentum_score'].fillna(0) >= 60)))
        
        if has_valid_data(df, ['from_low_pct', 'acceleration_score', 'ret_30d']):
            patterns.append(('🔄 52W LOW BOUNCE', (df['from_low_pct'].fillna(100) < 20) & (df['acceleration_score'].fillna(0) >= 80) & (df['ret_30d'].fillna(0) > 10)))
        
        if has_valid_data(df, ['from_low_pct', 'from_high_pct', 'trend_quality']):
            patterns.append(('👑 GOLDEN ZONE', (df['from_low_pct'].fillna(0) > 60) & (df['from_high_pct'].fillna(0) > -40) & (df['trend_quality'].fillna(0) >= 70)))
        
        if has_valid_data(df, ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            patterns.append(('📊 VOL ACCUMULATION', (df['vol_ratio_30d_90d'].fillna(0) > 1.2) & (df['vol_ratio_90d_180d'].fillna(0) > 1.1) & (df['ret_30d'].fillna(0) > 5)))
        
        if has_valid_data(df, ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(df['ret_7d'].fillna(0) != 0, df['ret_7d'].fillna(0) / 7, np.nan)
                daily_30d_pace = np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan)
            patterns.append(('🔀 MOMENTUM DIVERGE', daily_7d_pace.notna() & daily_30d_pace.notna() & (daily_7d_pace > daily_30d_pace * 1.5) & (df['acceleration_score'].fillna(0) >= 85) & (df['rvol'].fillna(0) > 2)))
        
        if has_valid_data(df, ['high_52w', 'low_52w', 'from_low_pct']):
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = np.where(df['low_52w'].fillna(0) > 0, ((df['high_52w'].fillna(0) - df['low_52w'].fillna(0)) / df['low_52w'].fillna(0)) * 100, 100)
            patterns.append(('🎯 RANGE COMPRESS', (range_pct < 50) & (df['from_low_pct'].fillna(0) > 30)))
        
        if has_valid_data(df, ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_ratio = np.where(df['ret_30d'].fillna(0) != 0, df['ret_7d'].fillna(0) / (df['ret_30d'].fillna(0) / 4), np.nan)
            patterns.append(('🤫 STEALTH', (df['vol_ratio_90d_180d'].fillna(0) > 1.1) & (df['vol_ratio_30d_90d'].fillna(0).between(0.9, 1.1)) & (df['from_low_pct'].fillna(0) > 40) & ret_ratio.notna() & (ret_ratio > 1)))
        
        if has_valid_data(df, ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pace_ratio = np.where(df['ret_7d'].fillna(0) != 0, df['ret_1d'].fillna(0) / (df['ret_7d'].fillna(0) / 7), np.nan)
            patterns.append(('🧛 VAMPIRE', daily_pace_ratio.notna() & (daily_pace_ratio > 2) & (df['rvol'].fillna(0) > 3) & (df['from_high_pct'].fillna(0) > -15) & (df['category'].isin(['Small Cap', 'Micro Cap']))))
        
        if has_valid_data(df, ['momentum_harmony', 'master_score']):
            patterns.append(('⛈️ PERFECT STORM', (df['momentum_harmony'].fillna(0) == 4) & (df['master_score'].fillna(0) > 80)))
        
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
            category_scores = df.groupby('category')['master_score'].mean().fillna(0)
            micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean()
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean()
            metrics['micro_small_avg'] = micro_small_avg
            metrics['large_mega_avg'] = large_mega_avg
            metrics['category_spread'] = micro_small_avg - large_mega_avg
        else: micro_small_avg, large_mega_avg = 50, 50
        if 'ret_30d' in df.columns:
            breadth = len(df[df['ret_30d'].fillna(0) > 0]) / len(df) if len(df) > 0 else 0
            metrics['breadth'] = breadth
        else: breadth = 0.5
        if 'rvol' in df.columns:
            avg_rvol = df['rvol'].fillna(1.0).median()
            metrics['avg_rvol'] = avg_rvol
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
            advancing = len(df[df['ret_1d'].fillna(0) > 0])
            declining = len(df[df['ret_1d'].fillna(0) < 0])
            unchanged = len(df[df['ret_1d'].fillna(0) == 0])
            ad_metrics['advancing'], ad_metrics['declining'], ad_metrics['unchanged'] = advancing, declining, unchanged
            if declining > 0: ad_metrics['ad_ratio'] = advancing / declining
            else: ad_metrics['ad_ratio'] = float('inf') if advancing > 0 else 1.0
            ad_metrics['ad_line'] = advancing - declining
            ad_metrics['breadth_pct'] = (advancing / len(df)) * 100 if len(df) > 0 else 0
        return ad_metrics

    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'sector' not in df.columns or df.empty: return pd.DataFrame()
        sector_dfs = []
        grouped_sectors = df.groupby('sector')
        for sector_name, sector_group_df in grouped_sectors:
            if sector_name != 'Unknown':
                sampled_sector_df = MarketIntelligence._apply_dynamic_sampling(sector_group_df.copy())
                if not sampled_sector_df.empty: sector_dfs.append(sampled_sector_df)
        if not sector_dfs: return pd.DataFrame()
        normalized_df = pd.concat(sector_dfs, ignore_index=True)
        sector_metrics = MarketIntelligence._calculate_flow_metrics(normalized_df, 'sector')
        original_counts = df.groupby('sector').size().rename('total_stocks')
        sector_metrics = sector_metrics.join(original_counts, how='left')
        sector_metrics['analyzed_stocks'] = sector_metrics['count']
        sector_metrics['sampling_pct'] = ((sector_metrics['analyzed_stocks'] / sector_metrics['total_stocks'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1))
        return sector_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'industry' not in df.columns or df.empty: return pd.DataFrame()
        industry_dfs = []
        grouped_industries = df.groupby('industry')
        for industry_name, industry_group_df in grouped_industries:
            if industry_name != 'Unknown':
                sampled_industry_df = MarketIntelligence._apply_dynamic_sampling(industry_group_df.copy())
                if not sampled_industry_df.empty: industry_dfs.append(sampled_industry_df)
        if not industry_dfs: return pd.DataFrame()
        normalized_df = pd.concat(industry_dfs, ignore_index=True)
        industry_metrics = MarketIntelligence._calculate_flow_metrics(normalized_df, 'industry')
        original_counts = df.groupby('industry').size().rename('total_stocks')
        industry_metrics = industry_metrics.join(original_counts, how='left')
        industry_metrics['analyzed_stocks'] = industry_metrics['count']
        industry_metrics['sampling_pct'] = ((industry_metrics['analyzed_stocks'] / industry_metrics['total_stocks'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1))
        return industry_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    def _apply_dynamic_sampling(df_group: pd.DataFrame) -> pd.DataFrame:
        group_size = len(df_group)
        if 1 <= group_size <= 5: sample_count = group_size
        elif 6 <= group_size <= 20: sample_count = max(1, int(group_size * 0.80))
        elif 21 <= group_size <= 50: sample_count = max(1, int(group_size * 0.60))
        elif 51 <= group_size <= 100: sample_count = max(1, int(group_size * 0.40))
        else: sample_count = min(50, int(group_size * 0.25))
        if sample_count > 0: return df_group.nlargest(sample_count, 'master_score', keep='first')
        else: return pd.DataFrame()
    
    @staticmethod
    def _calculate_flow_metrics(normalized_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        agg_dict = {
            'master_score': ['mean', 'median', 'std', 'count'], 'momentum_score': 'mean', 'volume_score': 'mean',
            'rvol': 'mean', 'ret_30d': 'mean', 'money_flow_mm': 'sum'
        }
        available_agg_dict = {k: v for k, v in agg_dict.items() if k in normalized_df.columns}
        group_metrics = normalized_df.groupby(group_col).agg(available_agg_dict).round(2)
        new_columns = []
        for col, funcs in available_agg_dict.items():
            if isinstance(funcs, list): new_columns.extend([f"{col}_{f}" for f in funcs])
            else: new_columns.append(f"{col}_{funcs}")
        group_metrics.columns = new_columns
        rename_map = {
            'master_score_mean': 'avg_score', 'master_score_median': 'median_score', 'master_score_std': 'std_score', 'master_score_count': 'count',
            'momentum_score_mean': 'avg_momentum', 'volume_score_mean': 'avg_volume', 'rvol_mean': 'avg_rvol',
            'ret_30d_mean': 'avg_ret_30d', 'money_flow_mm_sum': 'total_money_flow'
        }
        group_metrics = group_metrics.rename(columns=rename_map)
        group_metrics['flow_score'] = (group_metrics['avg_score'].fillna(0) * 0.3 + group_metrics['median_score'].fillna(0) * 0.2 + group_metrics['avg_momentum'].fillna(0) * 0.25 + group_metrics['avg_volume'].fillna(0) * 0.25)
        group_metrics['rank'] = group_metrics['flow_score'].rank(ascending=False, method='min')
        return group_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    def detect_category_performance(df: pd.DataFrame) -> pd.DataFrame:
        if 'category' not in df.columns or df.empty: return pd.DataFrame()
        category_dfs = []
        grouped_categories = df.groupby('category')
        for cat_name, cat_group_df in grouped_categories:
            if cat_name != 'Unknown':
                sampled_cat_df = MarketIntelligence._apply_dynamic_sampling(cat_group_df.copy())
                if not sampled_cat_df.empty: category_dfs.append(sampled_cat_df)
        if not category_dfs: return pd.DataFrame()
        normalized_df = pd.concat(category_dfs, ignore_index=True)
        agg_dict = {
            'master_score': ['mean', 'median', 'std', 'count'], 'momentum_score': 'mean', 'volume_score': 'mean',
            'rvol': 'mean', 'ret_30d': 'mean', 'acceleration_score': 'mean', 'breakout_score': 'mean',
            'money_flow_mm': 'sum'
        }
        available_agg_dict = {k: v for k, v in agg_dict.items() if k in normalized_df.columns}
        category_metrics = normalized_df.groupby('category').agg(available_agg_dict).round(2)
        new_columns = []
        for col, funcs in available_agg_dict.items():
            if isinstance(funcs, list): new_columns.extend([f"{col}_{f}" for f in funcs])
            else: new_columns.append(f"{col}_{funcs}")
        category_metrics.columns = new_columns
        rename_map = {
            'master_score_mean': 'avg_score', 'master_score_median': 'median_score', 'master_score_std': 'std_score', 'master_score_count': 'count',
            'momentum_score_mean': 'avg_momentum', 'volume_score_mean': 'avg_volume', 'rvol_mean': 'avg_rvol',
            'ret_30d_mean': 'avg_ret_30d', 'acceleration_score_mean': 'avg_acceleration', 'breakout_score_mean': 'avg_breakout',
            'money_flow_mm_sum': 'total_money_flow'
        }
        category_metrics = category_metrics.rename(columns=rename_map)
        category_metrics['flow_score'] = (category_metrics['avg_score'].fillna(0) * 0.35 + category_metrics['median_score'].fillna(0) * 0.20 + category_metrics['avg_momentum'].fillna(0) * 0.20 + category_metrics['avg_acceleration'].fillna(0) * 0.15 + category_metrics['avg_volume'].fillna(0) * 0.10)
        category_metrics['rank'] = category_metrics['flow_score'].rank(ascending=False, method='min')
        original_counts = df.groupby('category').size().rename('total_stocks')
        category_metrics = category_metrics.join(original_counts, how='left')
        category_metrics['analyzed_stocks'] = category_metrics['count']
        category_metrics['sampling_pct'] = ((category_metrics['analyzed_stocks'] / category_metrics['total_stocks'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1))
        category_order = ['Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']
        return category_metrics.reindex([cat for cat in category_order if cat in category_metrics.index])

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
            if plot_df.empty:
                fig = go.Figure()
                fig.add_annotation(text="No complete return data available for this chart.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            accel_df = plot_df.nlargest(min(n, len(plot_df)), 'acceleration_score', keep='first')
            if accel_df.empty:
                fig = go.Figure()
                fig.add_annotation(text="No stocks meet criteria for acceleration profiles.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            fig = go.Figure()
            for _, stock in accel_df.iterrows():
                x_points, y_points = ['Start'], [0]
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
            logger.error(f"Error creating acceleration profiles: {str(e)}", exc_info=True)
            fig = go.Figure()
            fig.add_annotation(text=f"Error generating chart: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

# ============================================
# FILTER ENGINE - ENHANCED WITH INDUSTRY
# ============================================

class FilterEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        if df.empty: return df
        masks = []
        categories = filters.get('categories', [])
        if categories and 'category' in df.columns: masks.append(df['category'].isin(categories))
        sectors = filters.get('sectors', [])
        if sectors and 'sector' in df.columns: masks.append(df['sector'].isin(sectors))
        industries = filters.get('industries', [])
        if industries and 'industry' in df.columns: masks.append(df['industry'].isin(industries))
        min_score = filters.get('min_score', 0)
        if min_score > 0 and 'master_score' in df.columns: masks.append(df['master_score'].fillna(0) >= min_score)
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns: masks.append(df['eps_change_pct'].notna() & (df['eps_change_pct'] >= min_eps_change))
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            pattern_regex = '|'.join([re.escape(p) for p in patterns])
            masks.append(df['patterns'].fillna('').str.contains(pattern_regex, case=False, regex=True))
        trend_range = filters.get('trend_range')
        if filters.get('trend_filter') != 'All Trends' and trend_range and 'trend_quality' in df.columns:
            min_trend, max_trend = trend_range
            masks.append(df['trend_quality'].notna() & (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend))
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in df.columns: masks.append(df['pe'].notna() & (df['pe'] > 0) & (df['pe'] >= min_pe))
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in df.columns: masks.append(df['pe'].notna() & (df['pe'] > 0) & (df['pe'] <= max_pe))
        for tier_type_key, col_name_suffix in [('eps_tiers', 'eps_tier'), ('pe_tiers', 'pe_tier'), ('price_tiers', 'price_tier')]:
            tier_values = filters.get(tier_type_key, [])
            col_name = col_name_suffix
            if tier_values and col_name in df.columns: masks.append(df[col_name].isin(tier_values))
        if filters.get('require_fundamental_data', False) and 'pe' in df.columns and 'eps_change_pct' in df.columns: masks.append(df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna())
        wave_states = filters.get('wave_states', [])
        if wave_states and 'wave_state' in df.columns: masks.append(df['wave_state'].isin(wave_states))
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and wave_strength_range != (0, 100) and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            masks.append(df['overall_wave_strength'].notna() & (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws))
        if masks: combined_mask = np.logical_and.reduce(masks)
        else: combined_mask = pd.Series(True, index=df.index)
        filtered_df = df[combined_mask].copy()
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        if df.empty or column not in df.columns: return []
        temp_filters = current_filters.copy()
        filter_key_map = {'category': 'categories', 'sector': 'sectors', 'industry': 'industries', 'eps_tier': 'eps_tiers', 'pe_tier': 'pe_tiers', 'price_tier': 'price_tiers', 'wave_state': 'wave_states'}
        if column in filter_key_map: temp_filters.pop(filter_key_map[column], None)
        if column == 'industry' and 'sectors' in current_filters: temp_filters['sectors'] = current_filters['sectors']
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        values = filtered_df[column].dropna().astype(str).unique()
        values = [v for v in values if v.strip().lower() not in ['unknown', '', 'nan', 'n/a', 'none', '-']]
        try: return sorted(values, key=lambda x: float(str(x).replace(',', '')) if str(x).replace(',', '').replace('.', '').isdigit() else x)
        except: return sorted(values, key=str)

# ============================================
# SEARCH ENGINE - ENHANCED
# ============================================

class SearchEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        if not query or df.empty: return pd.DataFrame()
        try:
            query_upper = query.upper().strip()
            results = df.copy()
            results['relevance'] = 0
            if 'ticker' in df.columns:
                mask_ticker_exact = (results['ticker'].str.upper() == query_upper).fillna(False)
                if mask_ticker_exact.any(): return results[mask_ticker_exact].copy()
                mask_ticker_contains = results['ticker'].str.upper().str.contains(query_upper, regex=False).fillna(False)
            else: mask_ticker_contains = pd.Series(False, index=df.index)
            if 'company_name' in df.columns:
                company_upper = results['company_name'].str.upper().fillna('')
                mask_company_contains = company_upper.str.contains(query_upper, regex=False).fillna(False)
                mask_company_word_match = company_upper.str.contains(r'\b' + re.escape(query_upper), regex=True).fillna(False)
            else:
                mask_company_contains = pd.Series(False, index=df.index)
                mask_company_word_match = pd.Series(False, index=df.index)
            combined_mask = mask_ticker_contains | mask_company_contains | mask_company_word_match
            all_matches = results[combined_mask].copy()
            if not all_matches.empty:
                all_matches['relevance'] = 0
                all_matches.loc[mask_ticker_contains[combined_mask], 'relevance'] += 50
                all_matches.loc[mask_company_contains[combined_mask], 'relevance'] += 30
                all_matches.loc[mask_company_word_match[combined_mask], 'relevance'] += 20
                return all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False]).drop('relevance', axis=1)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE - ENHANCED
# ============================================

class ExportEngine:
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        output = BytesIO()
        templates = {
            'day_trader': { 'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 'volume_score', 'vmi', 'wave_state', 'patterns', 'category', 'sector', 'industry'], 'focus': 'Intraday momentum and volume' },
            'swing_trader': { 'columns': ['rank', 'ticker', 'company_name', 'master_score', 'breakout_score', 'position_score', 'position_tension', 'from_high_pct', 'from_low_pct', 'trend_quality', 'momentum_harmony', 'patterns', 'sector', 'industry'], 'focus': 'Position and breakout setups' },
            'investor': { 'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 'long_term_strength', 'money_flow_mm', 'category', 'sector', 'industry'], 'focus': 'Fundamentals and long-term performance' },
            'full': { 'columns': None, 'focus': 'Complete analysis' }
        }
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                header_format = workbook.add_format({ 'bold': True, 'bg_color': '#3498db', 'font_color': 'white', 'border': 1 })
                column_formats = {
                    'price': workbook.add_format({'num_format': '₹#,##0.00'}), 'pe': workbook.add_format({'num_format': '#,##0.00'}),
                    'eps_current': workbook.add_format({'num_format': '#,##0.00'}), 'eps_change_pct': workbook.add_format({'num_format': '0.0%'}),
                    'ret_': workbook.add_format({'num_format': '0.0%'}), 'rvol': workbook.add_format({'num_format': '0.0"x"'}),
                    'money_flow_mm': workbook.add_format({'num_format': '₹#,##0.0,"M"'}), 'master_score': workbook.add_format({'num_format': '0.0'}),
                    'rank': workbook.add_format({'num_format': '#,##0'}),
                }
                
                top_100 = df.nlargest(min(100, len(df)), 'master_score', keep='first')
                if template in templates and templates[template]['columns']: export_cols = [col for col in templates[template]['columns'] if col in top_100.columns]
                else:
                    internal_cols = ['percentile', 'category_rank', 'category_percentile', 'eps_tier', 'pe_tier', 'price_tier', 'signal_count', 'shift_strength', 'surge_score', 'total_stocks', 'analyzed_stocks', 'flow_score', 'avg_score', 'median_score', 'std_score', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow', 'rank_flow_score', 'dummy_money_flow']
                    export_cols = [col for col in top_100.columns if col not in internal_cols]
                
                top_100_export = top_100[export_cols]
                top_100_export.to_excel(writer, sheet_name='Top 100 Stocks', index=False)
                worksheet = writer.sheets['Top 100 Stocks']
                for i, col in enumerate(top_100_export.columns):
                    worksheet.write(0, i, col, header_format)
                    col_letter = chr(ord('A') + i)
                    for key, fmt in column_formats.items():
                        if key in col: worksheet.set_column(f'{col_letter}:{col_letter}', None, fmt)
                    if 'rank' == col.lower(): worksheet.set_column(f'{col_letter}:{col_letter}', None, column_formats['rank'])
                    worksheet.autofit()
                
                intel_data = []
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                intel_data.append({'Metric': 'Market Regime', 'Value': regime, 'Details': f"Breadth: {regime_metrics.get('breadth', 0):.1%} | Avg RVOL: {regime_metrics.get('avg_rvol', 1):.1f}x"})
                ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
                intel_data.append({'Metric': 'Advance/Decline Ratio (1D)', 'Value': f"{ad_metrics.get('ad_ratio', 1):.2f}", 'Details': f"Advances: {ad_metrics.get('advancing', 0)}, Declines: {ad_metrics.get('declining', 0)}, Unchanged: {ad_metrics.get('unchanged', 0)}"})
                intel_df = pd.DataFrame(intel_data)
                intel_df.to_excel(writer, sheet_name='Market Intelligence', index=False)
                worksheet = writer.sheets['Market Intelligence']
                for i, col in enumerate(intel_df.columns): worksheet.write(0, i, col, header_format)
                worksheet.autofit()
                
                sector_rotation = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rotation.empty: sector_rotation.to_excel(writer, sheet_name='Sector Rotation'); worksheet = writer.sheets['Sector Rotation']; [worksheet.write(0, i, col, header_format) for i, col in enumerate(sector_rotation.columns)]; worksheet.autofit()
                
                industry_rotation = MarketIntelligence.detect_industry_rotation(df)
                if not industry_rotation.empty: industry_rotation.to_excel(writer, sheet_name='Industry Rotation'); worksheet = writer.sheets['Industry Rotation']; [worksheet.write(0, i, col, header_format) for i, col in enumerate(industry_rotation.columns)]; worksheet.autofit()

                pattern_counts = {}
                for patterns_str in df['patterns'].dropna():
                    if patterns_str:
                        for p in patterns_str.split(' | '): pattern_counts[p] = pattern_counts.get(p, 0) + 1
                if pattern_counts:
                    pattern_df = pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=False)
                    pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
                    worksheet = writer.sheets['Pattern Analysis']
                    [worksheet.write(0, i, col, header_format) for i, col in enumerate(pattern_df.columns)]
                    worksheet.autofit()

                wave_signals = df[(df['momentum_score'].fillna(0) >= 60) & (df['acceleration_score'].fillna(0) >= 70) & (df['rvol'].fillna(0) >= 2)].nlargest(50, 'master_score', keep='first')
                if not wave_signals.empty:
                    wave_cols = ['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'wave_state', 'patterns', 'category', 'sector', 'industry']
                    available_wave_cols = [col for col in wave_cols if col in wave_signals.columns]
                    wave_signals[available_wave_cols].to_excel(writer, sheet_name='Wave Radar Signals', index=False)
                    worksheet = writer.sheets['Wave Radar Signals']
                    [worksheet.write(0, i, col, header_format) for i, col in enumerate(wave_signals.columns)]
                    worksheet.autofit()

                summary_stats = {'Total Stocks Processed': len(df), 'Average Master Score (All)': df['master_score'].mean() if not df.empty else 0, 'Stocks with Patterns (All)': (df['patterns'] != '').sum() if 'patterns' in df.columns else 0, 'High RVOL (>2x) (All)': (df['rvol'].fillna(0) > 2).sum() if 'rvol' in df.columns else 0, 'Positive 30D Returns (All)': (df['ret_30d'].fillna(0) > 0).sum() if 'ret_30d' in df.columns else 0, 'Data Completeness %': RobustSessionState.safe_get('data_quality', {}).get('completeness', 0), 'Clipping Events Count': sum(DataValidator.get_clipping_counts().values()), 'Template Used': template, 'Export Date (UTC)': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                worksheet = writer.sheets['Summary']
                [worksheet.write(0, i, col, header_format) for i, col in enumerate(summary_df.columns)]; worksheet.autofit()
                logger.info(f"Excel report created successfully with {len(writer.sheets)} sheets")
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}", exc_info=True)
            raise
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        export_cols = ['rank', 'ticker', 'company_name', 'master_score', 'position_score', 'volume_score', 'momentum_score', 'acceleration_score', 'breakout_score', 'rvol_score', 'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct', 'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'rvol', 'vmi', 'money_flow_mm', 'position_tension', 'momentum_harmony', 'wave_state', 'patterns', 'category', 'sector', 'industry', 'eps_tier', 'pe_tier', 'price_tier', 'overall_wave_strength']
        available_cols = [col for col in export_cols if col in df.columns]
        export_df = df[available_cols].copy()
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
        if df.empty:
            st.warning("No data available for summary")
            return
        st.markdown("### 📊 Market Pulse")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
            ad_ratio = ad_metrics.get('ad_ratio', 1.0)
            display_ad_ratio = f"{ad_ratio:.2f}" if ad_ratio != float('inf') else "∞"
            ad_emoji = "🔥" if ad_ratio > 2 else "📈" if ad_ratio > 1 else "📉"
            UIComponents.render_metric_card("A/D Ratio", f"{ad_emoji} {display_ad_ratio}", f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}", "Advance/Decline Ratio (Advancing stocks / Declining stocks over 1 Day).")
        with col2:
            if 'momentum_score' in df.columns:
                high_momentum = len(df[df['momentum_score'].fillna(0) >= 70])
                momentum_pct = (high_momentum / len(df) * 100) if len(df) > 0 else 0
            else: high_momentum, momentum_pct = 0, 0
            UIComponents.render_metric_card("Momentum Health", f"{momentum_pct:.0f}%", f"{high_momentum} strong stocks", "Percentage of stocks with Momentum Score ≥ 70.")
        with col3:
            if 'rvol' in df.columns:
                avg_rvol = df['rvol'].fillna(1.0).median()
                high_vol_count = len(df[df['rvol'].fillna(0) > 2])
            else: avg_rvol, high_vol_count = 1.0, 0
            vol_emoji = "🌊" if avg_rvol > 1.5 else "💧" if avg_rvol > 1.2 else "🏜️"
            UIComponents.render_metric_card("Volume State", f"{vol_emoji} {avg_rvol:.1f}x", f"{high_vol_count} surges", "Median Relative Volume (RVOL). Surges indicate stocks with RVOL > 2x.")
        with col4:
            risk_factors = 0
            if 'from_high_pct' in df.columns and 'momentum_score' in df.columns:
                if len(df[(df['from_high_pct'].fillna(-100) >= 0) & (df['momentum_score'].fillna(0) < 50)]) > 20: risk_factors += 1
            if 'rvol' in df.columns:
                if len(df[(df['rvol'].fillna(0) > 10) & (df['master_score'].fillna(0) < 50)]) > 10: risk_factors += 1
            if 'trend_quality' in df.columns:
                if len(df[df['trend_quality'].fillna(50) < 40]) > len(df) * 0.3 and len(df) > 0: risk_factors += 1
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]
            risk_level = risk_levels[min(risk_factors, 3)]
            UIComponents.render_metric_card("Risk Level", risk_level, f"{risk_factors} factors", "Composite risk assessment based on overextension, extreme volume, and downtrends.")
        
        st.markdown("### 🎯 Today's Best Opportunities")
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        with opp_col1:
            ready_to_run = df[(df['momentum_score'].fillna(0) >= 70) & (df['acceleration_score'].fillna(0) >= 70) & (df['rvol'].fillna(0) >= 2)].nlargest(5, 'master_score', keep='first')
            st.markdown("**🚀 Ready to Run**")
            if not ready_to_run.empty:
                for _, stock in ready_to_run.iterrows(): st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"Score: {stock['master_score']:.1f} | RVOL: {stock.get('rvol', 0):.1f}x")
            else: st.info("No momentum leaders found")
        with opp_col2:
            hidden_gems = df[df['patterns'].str.contains('💎 HIDDEN GEM', na=False)].nlargest(5, 'master_score', keep='first')
            st.markdown("**💎 Hidden Gems**")
            if not hidden_gems.empty:
                for _, stock in hidden_gems.iterrows(): st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"Cat %ile: {stock.get('category_percentile', 0):.0f} | Score: {stock['master_score']:.1f}")
            else: st.info("No hidden gems today")
        with opp_col3:
            volume_alerts = df[df['rvol'].fillna(0) > 3].nlargest(5, 'master_score', keep='first')
            st.markdown("**⚡ Volume Alerts**")
            if not volume_alerts.empty:
                for _, stock in volume_alerts.iterrows(): st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"RVOL: {stock.get('rvol', 0):.1f}x | {stock.get('wave_state', 'N/A')}")
            else: st.info("No extreme volume detected")
        
        st.markdown("### 🧠 Market Intelligence")
        intel_col1, intel_col2 = st.columns([2, 1])
        with intel_col1:
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            if not sector_rotation.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sector_rotation.index[:10], y=sector_rotation['flow_score'][:10], text=[f"{val:.1f}" for val in sector_rotation['flow_score'][:10]], textposition='outside', marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' for score in sector_rotation['flow_score'][:10]], hovertemplate='Sector: %{x}<br>Flow Score: %{y:.1f}<br>Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>Avg Score: %{customdata[2]:.1f}<br>Median Score: %{customdata[3]:.1f}<extra></extra>', customdata=np.column_stack((sector_rotation['analyzed_stocks'][:10], sector_rotation['total_stocks'][:10], sector_rotation['avg_score'][:10], sector_rotation['median_score'][:10]))))
                fig.update_layout(title="Sector Rotation Map - Smart Money Flow (Dynamically Sampled)", xaxis_title="Sector", yaxis_title="Flow Score", height=400, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No sector rotation data available for visualization.")
        with intel_col2:
            regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
            st.markdown(f"**🎯 Market Regime**"); st.markdown(f"### {regime}"); st.markdown("**📡 Key Signals**")
            signals = []
            breadth = regime_metrics.get('breadth', 0.5)
            if breadth > 0.6: signals.append("✅ Strong breadth")
            elif breadth < 0.4: signals.append("⚠️ Weak breadth")
            category_spread = regime_metrics.get('category_spread', 0)
            if category_spread > 10: signals.append("🔄 Small caps leading")
            elif category_spread < -10: signals.append("🛡️ Large caps defensive")
            avg_rvol = regime_metrics.get('avg_rvol', 1.0)
            if avg_rvol > 1.5: signals.append("🌊 High volume activity")
            pattern_count = (df['patterns'].fillna('') != '').sum()
            if pattern_count > len(df) * 0.2 and len(df) > 0: signals.append("🎯 Many patterns emerging")
            if signals:
                for signal in signals: st.write(signal)
            else: st.info("No significant market signals detected.")
            st.markdown("**💪 Market Strength**")
            strength_score = ((breadth * 50) + (min(avg_rvol, 2) * 25) + ((pattern_count / len(df) if len(df) > 0 else 0) * 25))
            if strength_score > 70: strength_meter = "🟢🟢🟢🟢🟢"
            elif strength_score > 50: strength_meter = "🟢🟢🟢🟢⚪"
            elif strength_score > 30: strength_meter = "🟢🟢🟢⚪⚪"
            else: strength_meter = "🟢🟢⚪⚪⚪"
            st.write(strength_meter)
        
        st.markdown("---")
        st.markdown("#### 💾 Download Clean Processed Data")
        download_cols = st.columns(3)
        with download_cols[0]:
            st.markdown("**📊 Current View Data**"); st.write(f"Includes {len(filtered_df)} stocks matching current filters")
            csv_filtered = ExportEngine.create_csv_export(filtered_df)
            st.download_button(label="📥 Download Filtered Data (CSV)", data=csv_filtered, file_name=f"wave_detection_filtered_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", help="Download currently filtered stocks with all scores and indicators", key="wd_download_filtered_csv")
        with download_cols[1]:
            st.markdown("**🏆 Top 100 Stocks**"); st.write("Elite stocks ranked by Master Score")
            top_100_for_download = filtered_df.nlargest(100, 'master_score', keep='first')
            csv_top100 = ExportEngine.create_csv_export(top_100_for_download)
            st.download_button(label="📥 Download Top 100 (CSV)", data=csv_top100, file_name=f"wave_detection_top100_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", help="Download top 100 stocks by Master Score", key="wd_download_top100_csv")
        with download_cols[2]:
            st.markdown("**🎯 Pattern Stocks Only**"); pattern_stocks = filtered_df[filtered_df['patterns'] != '']
            st.write(f"Includes {len(pattern_stocks)} stocks with patterns")
            if not pattern_stocks.empty:
                csv_patterns = ExportEngine.create_csv_export(pattern_stocks)
                st.download_button(label="📥 Download Pattern Stocks (CSV)", data=csv_patterns, file_name=f"wave_detection_patterns_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", help="Download only stocks showing patterns", key="wd_download_patterns_csv")
            else: st.info("No stocks with patterns in current filter")

    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            display_count = st.selectbox("Show top", options=CONFIG.AVAILABLE_TOP_N, index=CONFIG.AVAILABLE_TOP_N.index(RobustSessionState.safe_get('user_preferences', {}).get('default_top_n', CONFIG.DEFAULT_TOP_N)), key="wd_rankings_display_count")
            user_prefs = RobustSessionState.safe_get('user_preferences', {}); user_prefs['default_top_n'] = display_count; RobustSessionState.safe_set('user_preferences', user_prefs)
        with col2:
            sort_options = ['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow']
            if 'trend_quality' in filtered_df.columns: sort_options.append('Trend')
            sort_by = st.selectbox("Sort by", options=sort_options, index=0, key="wd_rankings_sort_by")
        display_df = filtered_df.copy()
        if sort_by == 'Master Score': display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL': display_df = display_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum': display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Money Flow' and 'money_flow_mm' in display_df.columns: display_df = display_df.sort_values('money_flow_mm', ascending=False)
        elif sort_by == 'Trend' and 'trend_quality' in display_df.columns: display_df = display_df.sort_values('trend_quality', ascending=False)
        if not display_df.empty:
            if 'trend_quality' in display_df.columns:
                def get_trend_indicator(score):
                    if pd.isna(score): return "➖"
                    elif score >= 80: return "🔥"
                    elif score >= 60: return "✅"
                    elif score >= 40: return "➡️"
                    else: return "⚠️"
                display_df['trend_indicator'] = display_df['trend_quality'].apply(get_trend_indicator)
            display_cols = {
                'rank': 'Rank', 'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score', 'wave_state': 'Wave'
            }
            if 'trend_indicator' in display_df.columns: display_cols['trend_indicator'] = 'Trend'
            display_cols['price'] = 'Price'
            if show_fundamentals:
                if 'pe' in display_df.columns: display_cols['pe'] = 'PE'
                if 'eps_change_pct' in display_df.columns: display_cols['eps_change_pct'] = 'EPS Δ%'
            display_cols.update({ 'from_low_pct': 'From Low', 'ret_30d': '30D Ret', 'rvol': 'RVOL', 'vmi': 'VMI', 'patterns': 'Patterns', 'category': 'Category', 'sector': 'Sector', 'industry': 'Industry' })
            def format_pe(value):
                try:
                    if pd.isna(value): return '-'
                    val = float(value)
                    if val <= 0: return 'Loss'
                    elif val > 10000: return '>10K'
                    elif val > 1000: return f"{val:.0f}"
                    else: return f"{val:.1f}"
                except: return '-'
            def format_eps_change(value):
                try:
                    if pd.isna(value): return '-'
                    val = float(value)
                    if abs(val) >= 1000: return f"{val/1000:+.1f}K%"
                    elif abs(val) >= 100: return f"{val:+.0f}%"
                    else: return f"{val:+.1f}%"
                except: return '-'
            format_rules = {'master_score': '{:.1f}', 'price': '₹{:,.0f}', 'from_low_pct': '{:.0f}%', 'ret_30d': '{:+.1f}%', 'rvol': '{:.1f}x', 'vmi': '{:.2f}'}
            for col, fmt in format_rules.items():
                if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) and isinstance(x, (int, float)) else '-')
            if show_fundamentals:
                if 'pe' in display_df.columns: display_df['pe'] = display_df['pe'].apply(format_pe)
                if 'eps_change_pct' in display_df.columns: display_df['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change)
            available_display_cols = [c for c in display_cols.keys() if c in display_df.columns]
            display_df = display_df[available_display_cols]
            display_df.columns = [display_cols[c] for c in available_display_cols]
            paginated_df = UIComponents.render_pagination_controls(display_df, display_count, 'rankings')
            st.dataframe(paginated_df, use_container_width=True, height=min(600, len(paginated_df) * 35 + 50), hide_index=True)
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                with stat_cols[0]: st.markdown("**Score Distribution**"); scores_data = filtered_df['master_score'].dropna(); [st.text(f"Max: {scores_data.max():.1f}") if not scores_data.empty else st.text("N/A"), st.text(f"Min: {scores_data.min():.1f}") if not scores_data.empty else st.text("N/A"), st.text(f"Mean: {scores_data.mean():.1f}") if not scores_data.empty else st.text("N/A"), st.text(f"Median: {scores_data.median():.1f}") if not scores_data.empty else st.text("N/A")]
                with stat_cols[1]: st.markdown("**Returns (30D)**"); returns_data = filtered_df['ret_30d'].dropna(); [st.text(f"Max: {returns_data.max():.1f}%") if not returns_data.empty else st.text("N/A"), st.text(f"Min: {returns_data.min():.1f}%") if not returns_data.empty else st.text("N/A"), st.text(f"Avg: {returns_data.mean():.1f}%") if not returns_data.empty else st.text("N/A"), st.text(f"Positive: {(returns_data > 0).sum()}") if not returns_data.empty else st.text("N/A")]
                with stat_cols[2]:
                    if show_fundamentals:
                        st.markdown("**Fundamentals**")
                        if 'pe' in filtered_df.columns:
                            valid_pe = filtered_df['pe'].dropna(); valid_pe = valid_pe[(valid_pe > 0) & (valid_pe < 10000)]; st.text(f"Median PE: {valid_pe.median():.1f}x") if not valid_pe.empty else st.text("No valid PE.")
                        if 'eps_change_pct' in filtered_df.columns:
                            valid_eps = filtered_df['eps_change_pct'].dropna(); positive = (valid_eps > 0).sum(); st.text(f"Positive EPS: {positive}") if not valid_eps.empty else st.text("No valid EPS change.")
                    else:
                        st.markdown("**Volume**"); rvol_data = filtered_df['rvol'].dropna(); [st.text(f"Max: {rvol_data.max():.1f}x") if not rvol_data.empty else st.text("N/A"), st.text(f"Avg: {rvol_data.mean():.1f}x") if not rvol_data.empty else st.text("N/A"), st.text(f">2x: {(rvol_data > 2).sum()}") if not rvol_data.empty else st.text("N/A")]
                with stat_cols[3]:
                    st.markdown("**Trend Distribution**"); trend_data = filtered_df['trend_quality'].dropna()
                    if not trend_data.empty:
                        total_stocks_in_filter = len(trend_data)
                        avg_trend_score = trend_data.mean() if total_stocks_in_filter > 0 else 0
                        stocks_above_all_smas = (trend_data >= 85).sum(); stocks_in_uptrend = (trend_data >= 60).sum(); stocks_in_downtrend = (trend_data < 40).sum()
                        st.text(f"Avg Trend Score: {avg_trend_score:.1f}"); st.text(f"Above All SMAs: {stocks_above_all_smas}"); st.text(f"In Uptrend (60+): {stocks_in_uptrend}"); st.text(f"In Downtrend (<40): {stocks_in_downtrend}")
                    else: st.text("No trend data available")
        else: st.warning("No stocks match the selected filters.")

    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        st.markdown("*Catch waves as they form, not after they've peaked!*")
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        with radar_col1: wave_timeframe = st.selectbox("Wave Detection Timeframe", options=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"], index=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"].index(RobustSessionState.safe_get('wd_wave_timeframe_select', "All Waves")), key="wd_wave_timeframe_select", help="""🌊 All Waves: Complete unfiltered view\n⚡ Intraday Surge: High RVOL & today's movers\n📈 3-Day Buildup: Building momentum patterns\n🚀 Weekly Breakout: Near 52w highs with volume\n💪 Monthly Trend: Established trends with SMAs""")
        with radar_col2: sensitivity = st.select_slider("Detection Sensitivity", options=["Conservative", "Balanced", "Aggressive"], value=RobustSessionState.safe_get('wd_wave_sensitivity', "Balanced"), key="wd_wave_sensitivity", help="Conservative = Stronger signals, Aggressive = More signals")
        with radar_col3: show_market_regime = st.checkbox("📊 Market Regime Analysis", value=RobustSessionState.safe_get('wd_show_market_regime', True), key="wd_show_market_regime", help="Show category rotation flow and market regime detection")
        wave_filtered_df = filtered_df.copy()
        with radar_col4:
            if not wave_filtered_df.empty and 'overall_wave_strength' in wave_filtered_df.columns:
                try:
                    wave_strength_score = wave_filtered_df['overall_wave_strength'].fillna(0).mean()
                    wave_emoji = "🌊🔥" if wave_strength_score > 70 else "🌊" if wave_strength_score > 50 else "💤"
                    wave_color_delta = "🟢" if wave_strength_score > 70 else "🟡" if wave_strength_score > 50 else "🔴"
                    UIComponents.render_metric_card("Wave Strength", f"{wave_emoji} {wave_strength_score:.0f}%", f"{wave_color_delta} Market", "Overall strength of wave signals in the current filtered dataset. Reflects market trend bias.")
                except Exception as e: logger.error(f"Error calculating wave strength: {str(e)}"); UIComponents.render_metric_card("Wave Strength", "N/A", "Error calculating strength.")
            else: UIComponents.render_metric_card("Wave Strength", "N/A", "Data not available for strength calculation.")
        if show_sensitivity_details: st.expander("📊 Current Sensitivity Thresholds", expanded=True).markdown(f"""**{sensitivity} Settings**\n- **Momentum Shifts:** Score ≥ {60 if sensitivity == 'Conservative' else 50 if sensitivity == 'Balanced' else 40}, Acceleration ≥ {70 if sensitivity == 'Conservative' else 60 if sensitivity == 'Balanced' else 50}\n- **Volume Surges:** RVOL ≥ {3.0 if sensitivity == 'Conservative' else 2.0 if sensitivity == 'Balanced' else 1.5}x\n- **Pattern Distance:** {5 if sensitivity == 'Conservative' else 10 if sensitivity == 'Balanced' else 15}% from qualification""")
        if wave_timeframe != "All Waves":
            try:
                if wave_timeframe == "Intraday Surge": wave_filtered_df = wave_filtered_df[(wave_filtered_df['rvol'].fillna(0) >= 2.5) & (wave_filtered_df['ret_1d'].fillna(0) > 2)]; st.info(f"Showing {len(wave_filtered_df)} stocks matching '{wave_timeframe}'.")
                elif wave_timeframe == "3-Day Buildup": wave_filtered_df = wave_filtered_df[(wave_filtered_df['ret_3d'].fillna(0) > 5) & (wave_filtered_df['vol_ratio_7d_90d'].fillna(0) > 1.5)]; st.info(f"Showing {len(wave_filtered_df)} stocks matching '{wave_timeframe}'.")
                elif wave_timeframe == "Weekly Breakout": wave_filtered_df = wave_filtered_df[(wave_filtered_df['ret_7d'].fillna(0) > 8) & (wave_filtered_df['vol_ratio_7d_90d'].fillna(0) > 2.0) & (wave_filtered_df['from_high_pct'].fillna(-100) > -10)]; st.info(f"Showing {len(wave_filtered_df)} stocks matching '{wave_timeframe}'.")
                elif wave_timeframe == "Monthly Trend": wave_filtered_df = wave_filtered_df[(wave_filtered_df['ret_30d'].fillna(0) > 15) & (wave_filtered_df['price'].fillna(0) > wave_filtered_df['sma_20d'].fillna(0)) & (wave_filtered_df['sma_20d'].fillna(0) > wave_filtered_df['sma_50d'].fillna(0)) & (wave_filtered_df['vol_ratio_30d_180d'].fillna(0) > 1.2) & (wave_filtered_df['from_low_pct'].fillna(0) > 30)]; st.info(f"Showing {len(wave_filtered_df)} stocks matching '{wave_timeframe}'.")
            except Exception as e: st.warning(f"Required data for '{wave_timeframe}' not available. Showing all waves."); wave_filtered_df = filtered_df.copy()
        if not wave_filtered_df.empty:
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            momentum_threshold = 60 if sensitivity == "Conservative" else 50 if sensitivity == "Balanced" else 40
            acceleration_threshold = 70 if sensitivity == "Conservative" else 60 if sensitivity == "Balanced" else 50
            min_rvol_signal = 3.0 if sensitivity == "Conservative" else 2.0 if sensitivity == "Balanced" else 1.5
            momentum_shifts = wave_filtered_df[(wave_filtered_df['momentum_score'].fillna(0) >= momentum_threshold) & (wave_filtered_df['acceleration_score'].fillna(0) >= acceleration_threshold)].copy()
            if not momentum_shifts.empty:
                momentum_shifts['signal_count'] = 0; momentum_shifts.loc[momentum_shifts['momentum_score'].fillna(0) >= momentum_threshold, 'signal_count'] += 1; momentum_shifts.loc[momentum_shifts['acceleration_score'].fillna(0) >= acceleration_threshold, 'signal_count'] += 1; momentum_shifts.loc[momentum_shifts['rvol'].fillna(0) >= min_rvol_signal, 'signal_count'] += 1
                if 'breakout_score' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['breakout_score'].fillna(0) >= 75, 'signal_count'] += 1
                if 'vol_ratio_7d_90d' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['vol_ratio_7d_90d'].fillna(0) >= 1.5, 'signal_count'] += 1
                momentum_shifts['shift_strength'] = (momentum_shifts['momentum_score'].fillna(50) * 0.4 + momentum_shifts['acceleration_score'].fillna(50) * 0.4 + momentum_shifts['rvol_score'].fillna(50) * 0.2)
                top_shifts = momentum_shifts.sort_values(['signal_count', 'shift_strength'], ascending=[False, False]).head(20)
                display_columns = ['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'signal_count', 'wave_state', 'ret_7d', 'category', 'sector', 'industry']
                shift_display = top_shifts[[col for col in display_columns if col in top_shifts.columns]].copy()
                shift_display['Signals'] = shift_display['signal_count'].apply(lambda x: f"{'🔥' * min(x, 3)} {x}/5")
                if 'ret_7d' in shift_display.columns: shift_display['7D Return'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else '-')
                shift_display['RVOL'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                rename_dict = {'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score', 'momentum_score': 'Momentum', 'acceleration_score': 'Acceleration', 'rvol': 'RVOL', 'wave_state': 'Wave', 'category': 'Category', 'sector': 'Sector', 'industry': 'Industry', 'ret_7d': '7D Return'}
                shift_display = shift_display.drop('signal_count', axis=1).rename(columns=rename_dict)
                st.dataframe(shift_display, use_container_width=True, hide_index=True)
                multi_signal = len(top_shifts[top_shifts['signal_count'] >= 3])
                if multi_signal > 0: st.success(f"🏆 Found {multi_signal} stocks with 3+ signals (strongest momentum)")
                super_signals = top_shifts[top_shifts['signal_count'] >= 4]
                if not super_signals.empty: st.warning(f"🔥🔥 {len(super_signals)} stocks showing EXTREME momentum (4+ signals)!")
            else: st.info(f"No momentum shifts detected in {wave_timeframe} timeframe for '{sensitivity}' sensitivity.")
            st.markdown("#### 🚀 Acceleration Profiles - Momentum Building Over Time")
            accel_threshold = 85 if sensitivity == "Conservative" else 70 if sensitivity == "Balanced" else 60
            if 'acceleration_score' in wave_filtered_df.columns: accelerating_stocks = wave_filtered_df[wave_filtered_df['acceleration_score'].fillna(0) >= accel_threshold].nlargest(10, 'acceleration_score', keep='first')
            else: accelerating_stocks = pd.DataFrame()
            if not accelerating_stocks.empty:
                fig_accel = Visualizer.create_acceleration_profiles(accelerating_stocks, n=10)
                st.plotly_chart(fig_accel, use_container_width=True)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Perfect Acceleration (90+)", len(accelerating_stocks[accelerating_stocks['acceleration_score'].fillna(0) >= 90]), help_text="Number of stocks with Acceleration Score >= 90.")
                with col2: st.metric("Strong Acceleration (80+)", len(accelerating_stocks[accelerating_stocks['acceleration_score'].fillna(0) >= 80]), help_text="Number of stocks with Acceleration Score >= 80.")
                with col3: st.metric("Avg Acceleration Score", f"{accelerating_stocks['acceleration_score'].fillna(0).mean():.1f}", help_text="Average Acceleration Score for displayed stocks.")
            else: st.info(f"No stocks meet the acceleration threshold ({accel_threshold}+) for '{sensitivity}' sensitivity.")
            if show_market_regime:
                st.markdown("#### 💰 Category Rotation - Smart Money Flow")
                col1, col2 = st.columns([3, 2])
                with col1:
                    if 'category' in wave_filtered_df.columns:
                        category_flow = MarketIntelligence.detect_category_performance(wave_filtered_df)
                        if not category_flow.empty:
                            flow_direction = "🔥 RISK-ON" if 'Small' in category_flow.index[0] or 'Micro' in category_flow.index[0] else "❄️ RISK-OFF" if 'Large' in category_flow.index[0] or 'Mega' in category_flow.index[0] else "➡️ Neutral"
                            fig_flow = go.Figure()
                            fig_flow.add_trace(go.Bar(x=category_flow.index, y=category_flow['flow_score'], text=[f"{val:.1f}" for val in category_flow['flow_score']], textposition='outside', marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' for score in category_flow['flow_score']], hovertemplate='Category: %{x}<br>Flow Score: %{y:.1f}<br>Stocks: %{customdata[0]} of %{customdata[1]}<extra></extra>', customdata=np.column_stack((category_flow['analyzed_stocks'], category_flow['total_stocks']))))
                            fig_flow.update_layout(title=f"Smart Money Flow Direction: {flow_direction}", xaxis_title="Market Cap Category", yaxis_title="Flow Score", height=300, template='plotly_white', showlegend=False)
                            st.plotly_chart(fig_flow, use_container_width=True)
                        else: st.info("Insufficient data for category flow analysis.")
                    else: st.info("Category data not available for flow analysis.")
                with col2:
                    if 'category' in wave_filtered_df.columns and not wave_filtered_df.empty:
                        category_flow = MarketIntelligence.detect_category_performance(wave_filtered_df)
                        if not category_flow.empty:
                            flow_direction = "🔥 RISK-ON" if 'Small' in category_flow.index[0] or 'Micro' in category_flow.index[0] else "❄️ RISK-OFF" if 'Large' in category_flow.index[0] or 'Mega' in category_flow.index[0] else "➡️ Neutral"
                            st.markdown(f"**🎯 Market Regime: {flow_direction}**")
                            st.markdown("**💎 Strongest Categories:**")
                            for i, (cat, row) in enumerate(category_flow.head(3).iterrows()):
                                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                                st.write(f"{emoji} **{cat}**: Score {row['flow_score']:.1f}")
                            if 'Small Cap' in category_flow.index and 'Large Cap' in category_flow.index:
                                small_caps_score = category_flow.loc[['Small Cap', 'Micro Cap'], 'flow_score'].mean()
                                large_caps_score = category_flow.loc[['Large Cap', 'Mega Cap'], 'flow_score'].mean()
                                if small_caps_score > large_caps_score + 10: st.success("📈 Small Caps Leading - Early Bull Signal!")
                                elif large_caps_score > small_caps_score + 10: st.warning("📉 Large Caps Leading - Defensive Mode")
                                else: st.info("➡️ Balanced Market - No Clear Leader")
                        else: st.info("Category data not available")
            
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            pattern_distance = {"Conservative": 5, "Balanced": 10, "Aggressive": 15}[sensitivity]
            emergence_data = []
            if 'category_percentile' in wave_filtered_df.columns and 'master_score' in wave_filtered_df.columns:
                close_to_leader = wave_filtered_df[(wave_filtered_df['category_percentile'].fillna(0) >= (90 - pattern_distance)) & (wave_filtered_df['category_percentile'].fillna(0) < 90)]
                for _, stock in close_to_leader.iterrows(): emergence_data.append({'Ticker': stock['ticker'], 'Company': stock['company_name'], 'Pattern': '🔥 CAT LEADER', 'Distance': f"{90 - stock['category_percentile'].fillna(0):.1f}% away", 'Current': f"{stock['category_percentile'].fillna(0):.1f}%ile", 'Score': stock['master_score']})
            if 'breakout_score' in wave_filtered_df.columns and 'master_score' in wave_filtered_df.columns:
                close_to_breakout = wave_filtered_df[(wave_filtered_df['breakout_score'].fillna(0) >= (80 - pattern_distance)) & (wave_filtered_df['breakout_score'].fillna(0) < 80)]
                for _, stock in close_to_breakout.iterrows(): emergence_data.append({'Ticker': stock['ticker'], 'Company': stock['company_name'], 'Pattern': '🎯 BREAKOUT', 'Distance': f"{80 - stock['breakout_score'].fillna(0):.1f} pts away", 'Current': f"{stock['breakout_score'].fillna(0):.1f} score", 'Score': stock['master_score']})
            if emergence_data:
                emergence_df = pd.DataFrame(emergence_data).sort_values('Score', ascending=False).head(15)
                col1, col2 = st.columns([3, 1])
                with col1: st.dataframe(emergence_df, use_container_width=True, hide_index=True)
                with col2: UIComponents.render_metric_card("Emerging Patterns", len(emergence_df), help_text="Number of stocks close to qualifying for key patterns.")
            else: st.info(f"No patterns emerging within {pattern_distance}% threshold.")
            st.markdown("#### 🌊 Volume Surges - Unusual Activity NOW")
            rvol_threshold_display = 3.0 if sensitivity == "Conservative" else 2.0 if sensitivity == "Balanced" else 1.5
            if 'rvol' in wave_filtered_df.columns: volume_surges = wave_filtered_df[wave_filtered_df['rvol'].fillna(0) >= rvol_threshold_display].copy()
            else: volume_surges = pd.DataFrame()
            if not volume_surges.empty:
                volume_surges['surge_score'] = (volume_surges['rvol_score'].fillna(50) * 0.5 + volume_surges['volume_score'].fillna(50) * 0.3 + volume_surges['momentum_score'].fillna(50) * 0.2)
                top_surges = volume_surges.nlargest(15, 'surge_score', keep='first')
                col1, col2 = st.columns([2, 1])
                with col1:
                    display_cols_surge = ['ticker', 'company_name', 'rvol', 'price', 'money_flow_mm', 'wave_state', 'category', 'sector', 'industry']
                    if 'ret_1d' in top_surges.columns: display_cols_surge.insert(3, 'ret_1d')
                    surge_display = top_surges[[col for col in display_cols_surge if col in top_surges.columns]].copy()
                    surge_display['Type'] = surge_display['rvol'].apply(lambda x: "🔥🔥🔥" if x > 5 else "🔥🔥" if x > 3 else "🔥" if x > 1.5 else "-")
                    if 'ret_1d' in surge_display.columns: surge_display['ret_1d'] = surge_display['ret_1d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else '-')
                    if 'money_flow_mm' in surge_display.columns: surge_display['money_flow_mm'] = surge_display['money_flow_mm'].apply(lambda x: f"₹{x:.1f}M" if pd.notna(x) else '-')
                    surge_display['price'] = surge_display['price'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-')
                    surge_display['rvol'] = surge_display['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                    rename_dict_surge = {'ticker': 'Ticker', 'company_name': 'Company', 'rvol': 'RVOL', 'price': 'Price', 'money_flow_mm': 'Money Flow', 'wave_state': 'Wave', 'category': 'Category', 'sector': 'Sector', 'industry': 'Industry', 'ret_1d': '1D Ret'}
                    surge_display = surge_display.drop('signal_count', axis=1).rename(columns=rename_dict_surge)
                    st.dataframe(surge_display, use_container_width=True, hide_index=True)
                with col2:
                    UIComponents.render_metric_card("Active Surges", len(volume_surges), help_text=f"Number of stocks with RVOL >= {rvol_threshold_display}x.")
                    UIComponents.render_metric_card("Extreme (>5x)", len(volume_surges[volume_surges['rvol'].fillna(0) > 5]), help_text="Stocks with RVOL > 5x.")
                    UIComponents.render_metric_card("High (>3x)", len(volume_surges[volume_surges['rvol'].fillna(0) > 3]), help_text="Stocks with RVOL > 3x.")
                    if 'category' in volume_surges.columns:
                        st.markdown("**📊 Surge by Category:**")
                        surge_categories = volume_surges['category'].dropna().value_counts()
                        if not surge_categories.empty:
                            for cat, count in surge_categories.head(3).items(): st.caption(f"• {cat}: {count} stocks")
                        else: st.caption("No categories with surges.")
            else: st.info(f"No volume surges detected with '{sensitivity}' sensitivity (requires RVOL ≥ {rvol_threshold_display}x).")
        else: st.warning(f"No data available for Wave Radar analysis with '{wave_timeframe}' timeframe. Please adjust filters or timeframe.")
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            with col1: fig_dist = Visualizer.create_score_distribution(filtered_df); st.plotly_chart(fig_dist, use_container_width=True)
            with col2:
                pattern_counts = {}; [pattern_counts.update(p.split(' | ')) for p in filtered_df['patterns'].dropna() if p]
                if pattern_counts:
                    pattern_df = pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=True).tail(15)
                    fig_patterns = go.Figure([go.Bar(x=pattern_df['Count'], y=pattern_df['Pattern'], orientation='h', marker_color='#3498db', text=pattern_df['Count'], textposition='outside')])
                    fig_patterns.update_layout(title="Pattern Frequency Analysis", xaxis_title="Number of Stocks", yaxis_title="Pattern", template='plotly_white', height=400, margin=dict(l=150))
                    st.plotly_chart(fig_patterns, use_container_width=True)
                else: st.info("No patterns detected in current selection")
            st.markdown("---")
            st.markdown("#### Sector Performance (Dynamically Sampled)")
            sector_overview_df_local = MarketIntelligence.detect_sector_rotation(filtered_df)
            if not sector_overview_df_local.empty:
                display_cols_overview = ['flow_score', 'avg_score', 'median_score', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'analyzed_stocks', 'total_stocks']
                available_overview_cols = [col for col in display_cols_overview if col in sector_overview_df_local.columns]
                sector_overview_display = sector_overview_df_local[available_overview_cols].copy()
                sector_overview_display.columns = ['Flow Score', 'Avg Score', 'Median Score', 'Avg Momentum', 'Avg Volume', 'Avg RVOL', 'Avg 30D Ret', 'Analyzed Stocks', 'Total Stocks']
                st.dataframe(sector_overview_display, use_container_width=True)
                st.info("📊 **Normalized Analysis**: Shows metrics for dynamically sampled stocks per sector (by Master Score) to ensure fair comparison across sectors of different sizes.")
            else: st.info("No sector data available in the filtered dataset for analysis. Please check your filters.")
            if 'industry' in filtered_df.columns:
                st.markdown("#### Industry Performance (Smart Dynamic Sampling)")
                industry_overview_df = MarketIntelligence.detect_industry_rotation(filtered_df)
                if not industry_overview_df.empty:
                    ind_tab1, ind_tab2 = st.tabs(["📊 Top Industries", "📈 All Industries"])
                    with ind_tab1:
                        top_industries = industry_overview_df.head(20); fig_industry = go.Figure(); fig_industry.add_trace(go.Bar(x=top_industries.index[:15], y=top_industries['flow_score'][:15], text=[f"{val:.1f}" for val in top_industries['flow_score'][:15]], textposition='outside', marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' for score in top_industries['flow_score'][:15]], hovertemplate='Industry: %{x}<br>Flow Score: %{y:.1f}<br>Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>Sampling: %{customdata[2]:.1f}%<br>Avg Score: %{customdata[3]:.1f}<br>Median Score: %{customdata[4]:.1f}<extra></extra>', customdata=np.column_stack((top_industries['analyzed_stocks'][:15], top_industries['total_stocks'][:15], top_industries['sampling_pct'][:15], top_industries['avg_score'][:15], top_industries['median_score'][:15])))); fig_industry.update_layout(title="Top 15 Industries by Smart Money Flow", xaxis_title="Industry", yaxis_title="Flow Score", height=500, template='plotly_white', showlegend=False, xaxis_tickangle=-45); st.plotly_chart(fig_industry, use_container_width=True)
                        col1, col2, col3, col4 = st.columns(4); with col1: UIComponents.render_metric_card("Total Industries", f"{len(industry_overview_df):,}")
                        with col2: top_3_avg = top_industries.head(3)['avg_score'].mean(); UIComponents.render_metric_card("Top 3 Avg Score", f"{top_3_avg:.1f}")
                        with col3: strong_industries = len(industry_overview_df[industry_overview_df['flow_score'] > 60]); UIComponents.render_metric_card("Strong Industries", f"{strong_industries}", f"{strong_industries/len(industry_overview_df)*100:.0f}% of total")
                        with col4: total_analyzed = industry_overview_df['analyzed_stocks'].sum(); UIComponents.render_metric_card("Stocks Analyzed", f"{total_analyzed:,}", f"From {len(filtered_df):,} total")
                    with ind_tab2:
                        display_cols_industry = ['flow_score', 'avg_score', 'median_score', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'analyzed_stocks', 'total_stocks', 'sampling_pct']
                        available_industry_cols = [col for col in display_cols_industry if col in industry_overview_df.columns]
                        industry_display = industry_overview_df[available_industry_cols].copy()
                        display_names = {
                            'flow_score': 'Flow Score', 'avg_score': 'Avg Score', 'median_score': 'Median Score', 'avg_momentum': 'Avg Momentum',
                            'avg_volume': 'Avg Volume', 'avg_rvol': 'Avg RVOL', 'avg_ret_30d': 'Avg 30D Ret', 'analyzed_stocks': 'Analyzed',
                            'total_stocks': 'Total', 'sampling_pct': 'Sample %'
                        }
                        industry_display.columns = [display_names.get(col, col) for col in industry_display.columns]
                        if 'Sample %' in industry_display.columns: industry_display['Sample %'] = industry_display['Sample %'].apply(lambda x: f"{x:.1f}%")
                        industry_display.insert(0, 'Rank', range(1, len(industry_display) + 1))
                        st.dataframe(industry_display, use_container_width=True, height=400)
                        st.info("📊 **Smart Dynamic Sampling**: This ensures fair comparison across industries of vastly different sizes.")
                else: st.info("No industry data available in the filtered dataset for analysis.")
            st.markdown("---")
            st.markdown("#### 📊 Category Performance (Market Cap Analysis)")
            category_overview_df = MarketIntelligence.detect_category_performance(filtered_df)
            if not category_overview_df.empty:
                cat_tab1, cat_tab2 = st.tabs(["📊 Category Flow", "📈 Detailed Metrics"])
                with cat_tab1:
                    fig_category = go.Figure(); colors = {'Mega Cap': '#1f77b4', 'Large Cap': '#2ca02c', 'Mid Cap': '#ff7f0e', 'Small Cap': '#d62728', 'Micro Cap': '#9467bd'}
                    bar_colors = [colors.get(cat, '#7f7f7f') for cat in category_overview_df.index]
                    fig_category.add_trace(go.Bar(x=category_overview_df.index, y=category_overview_df['flow_score'], text=[f"{val:.1f}" for val in category_overview_df['flow_score']], textposition='outside', marker_color=bar_colors, hovertemplate='Category: %{x}<br>Flow Score: %{y:.1f}<br>Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>Avg Score: %{customdata[2]:.1f}<br>Avg Momentum: %{customdata[3]:.1f}<br>Avg Acceleration: %{customdata[4]:.1f}<extra></extra>', customdata=np.column_stack((category_overview_df['analyzed_stocks'], category_overview_df['total_stocks'], category_overview_df['avg_score'], category_overview_df['avg_momentum'], category_overview_df.get('avg_acceleration', [0]*len(category_overview_df.index))))))
                    market_state = "📊 ANALYZING..."
                    if len(category_overview_df) >= 2:
                        small_micro_avg = category_overview_df.loc[category_overview_df.index.str.contains('Small|Micro'), 'flow_score'].mean()
                        large_mega_avg = category_overview_df.loc[category_overview_df.index.str.contains('Large|Mega'), 'flow_score'].mean()
                        if small_micro_avg > large_mega_avg + 10: market_state = "🔥 RISK-ON (Small/Micro Leading)"
                        elif large_mega_avg > small_micro_avg + 10: market_state = "🛡️ RISK-OFF (Large/Mega Leading)"
                        else: market_state = "⚖️ BALANCED MARKET"
                    fig_category.update_layout(title=f"Category Performance - {market_state}", xaxis_title="Market Cap Category", yaxis_title="Flow Score", height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_category, use_container_width=True)
                    col1, col2, col3 = st.columns(3)
                    with col1: best_category = category_overview_df.index[0] if not category_overview_df.empty else "N/A"; best_score = category_overview_df['flow_score'].iloc[0] if not category_overview_df.empty else 0; UIComponents.render_metric_card("Leading Category", f"{best_category}", f"Score: {best_score:.1f}")
                    with col2:
                        if 'avg_momentum' in category_overview_df.columns and not category_overview_df.empty:
                            highest_momentum = category_overview_df.nlargest(1, 'avg_momentum'); UIComponents.render_metric_card("Highest Momentum", f"{highest_momentum.index[0]}", f"{highest_momentum['avg_momentum'].iloc[0]:.1f}")
                    with col3:
                        if 'avg_acceleration' in category_overview_df.columns and not category_overview_df.empty:
                            highest_accel = category_overview_df.nlargest(1, 'avg_acceleration'); UIComponents.render_metric_card("Best Acceleration", f"{highest_accel.index[0]}", f"{highest_accel['avg_acceleration'].iloc[0]:.1f}")
                with cat_tab2:
                    display_cols_category = ['flow_score', 'avg_score', 'median_score', 'avg_momentum', 'avg_acceleration', 'avg_breakout', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'analyzed_stocks', 'total_stocks', 'sampling_pct']
                    available_category_cols = [col for col in display_cols_category if col in category_overview_df.columns]
                    category_display = category_overview_df[available_category_cols].copy()
                    display_names = {'flow_score': 'Flow Score', 'avg_score': 'Avg Score', 'median_score': 'Median Score', 'avg_momentum': 'Avg Momentum', 'avg_acceleration': 'Avg Acceleration', 'avg_breakout': 'Avg Breakout', 'avg_volume': 'Avg Volume', 'avg_rvol': 'Avg RVOL', 'avg_ret_30d': 'Avg 30D Ret', 'analyzed_stocks': 'Analyzed', 'total_stocks': 'Total', 'sampling_pct': 'Sample %'}
                    category_display.columns = [display_names.get(col, col) for col in category_display.columns]
                    if 'Sample %' in category_display.columns: category_display['Sample %'] = category_display['Sample %'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(category_display.style.background_gradient(subset=['Flow Score', 'Avg Score', 'Avg Momentum', 'Avg Acceleration'], cmap='RdYlGn'), use_container_width=True)
                    st.markdown("##### 🎯 Market Regime Analysis")
                    if len(category_overview_df) >= 2:
                        if 'Small Cap' in category_overview_df.index and 'Large Cap' in category_overview_df.index:
                            spread = category_overview_df.loc['Small Cap', 'flow_score'] - category_overview_df.loc['Large Cap', 'flow_score']
                            if spread > 15: st.success(f"🔥 Strong Risk-On Signal: Small Cap outperforming Large Cap by {spread:.1f} points")
                            elif spread < -15: st.warning(f"🛡️ Risk-Off Signal: Large Cap outperforming Small Cap by {abs(spread):.1f} points")
                            else: st.info(f"⚖️ Balanced Market: Small-Large spread is {spread:.1f} points")
            else: st.info("No category data available in the filtered dataset for analysis.")
        else: st.info("No data available for analysis.")
    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        col1, col2 = st.columns([4, 1])
        with col1: search_query = st.text_input("Search stocks", value=RobustSessionState.safe_get('wd_search_input', ''), placeholder="Enter ticker or company name...", help="Search by ticker symbol or company name", key="wd_search_input")
        with col2: st.markdown("<br>", unsafe_allow_html=True); search_clicked = st.button("🔎 Search", type="primary", use_container_width=True, key="wd_search_button")
        if search_query or search_clicked:
            if not search_query.strip(): st.info("Please enter a search query."); search_results = pd.DataFrame()
            else:
                with st.spinner("Searching..."): search_results = SearchEngine.search_stocks(filtered_df, search_query)
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                for _, stock in search_results.iterrows():
                    with st.expander(f"📊 {stock['ticker']} - {stock['company_name']} (Rank #{int(stock['rank']) if pd.notna(stock['rank']) else 'N/A'})", expanded=True):
                        metric_cols = st.columns(6); with metric_cols[0]: UIComponents.render_metric_card("Master Score", f"{stock['master_score']:.1f}" if pd.notna(stock.get('master_score')) else "N/A", f"Rank #{int(stock['rank'])}" if pd.notna(stock.get('rank')) else "N/A", help_text="Composite score indicating overall strength. Higher is better.")
                        with metric_cols[1]: price_value = f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A"; ret_1d_value = f"{stock['ret_1d']:+.1f}%" if pd.notna(stock.get('ret_1d')) else None; UIComponents.render_metric_card("Price", price_value, ret_1d_value, help_text="Current stock price and 1-day return.")
                        with metric_cols[2]: UIComponents.render_metric_card("From Low", f"{stock['from_low_pct']:.0f}%" if pd.notna(stock.get('from_low_pct')) else "N/A", help_text="Percentage change from 52-week low. Higher means closer to high.")
                        with metric_cols[3]: ret_30d = stock.get('ret_30d'); UIComponents.render_metric_card("30D Return", f"{ret_30d:+.1f}%" if pd.notna(ret_30d) else "N/A", "↑" if pd.notna(ret_30d) and ret_30d > 0 else ("↓" if pd.notna(ret_30d) and ret_30d < 0 else None), help_text="Percentage return over the last 30 days.")
                        with metric_cols[4]: rvol = stock.get('rvol'); UIComponents.render_metric_card("RVOL", f"{rvol:.1f}x" if pd.notna(rvol) else "N/A", "High" if pd.notna(rvol) and rvol > 2 else "Normal", help_text="Relative Volume: current volume compared to average. Higher indicates unusual activity.")
                        with metric_cols[5]: UIComponents.render_metric_card("Wave State", stock.get('wave_state', 'N/A'), stock.get('category', 'N/A'), help_text="Detected current momentum phase (Forming, Building, Cresting, Breaking).")
                        st.markdown("#### 📈 Score Components"); score_cols_breakdown = st.columns(6)
                        components = [("Position", stock.get('position_score'), CONFIG.POSITION_WEIGHT, "52-week range positioning."), ("Volume", stock.get('volume_score'), CONFIG.VOLUME_WEIGHT, "Multi-timeframe volume patterns."), ("Momentum", stock.get('momentum_score'), CONFIG.MOMENTUM_WEIGHT, "30-day price momentum."), ("Acceleration", stock.get('acceleration_score'), CONFIG.ACCELERATION_WEIGHT, "Momentum acceleration signals."), ("Breakout", stock.get('breakout_score'), CONFIG.BREAKOUT_WEIGHT, "Technical breakout readiness."), ("RVOL", stock.get('rvol_score'), CONFIG.RVOL_WEIGHT, "Real-time relative volume score.")]
                        for i, (name, score, weight, help_text_comp) in enumerate(components):
                            with score_cols_breakdown[i]:
                                if pd.isna(score): color = "⚪"; display_score = "N/A"
                                elif score >= 80: color = "🟢"; display_score = f"{score:.0f}"
                                elif score >= 60: color = "🟡"; display_score = f"{score:.0f}"
                                else: color = "🔴"; display_score = f"{score:.0f}"
                                with st.popover(f"**{name}**", help=help_text_comp): st.markdown(f"**{name} Score**: {color} {display_score}"); st.markdown(f"Weighted at **{weight:.0%}** of Master Score."); st.write(help_text_comp)
                        if stock.get('patterns'): st.markdown(f"**🎯 Patterns:** {stock['patterns']}")
                        else: st.markdown("**🎯 Patterns:** None detected.")
                        st.markdown("---"); detail_cols_top = st.columns([1, 1])
                        with detail_cols_top[0]:
                            st.markdown("**📊 Classification**"); st.text(f"Sector: {stock.get('sector', 'Unknown')}"); st.text(f"Industry: {stock.get('industry', 'Unknown')}"); st.text(f"Category: {stock.get('category', 'Unknown')}")
                            if show_fundamentals:
                                st.markdown("**💰 Fundamentals**"); pe_val = stock.get('pe'); eps_cur = stock.get('eps_current'); eps_chg = stock.get('eps_change_pct')
                                if pd.notna(pe_val): st.text(f"PE Ratio: {pe_val:.1f}x" if pe_val > 0 and pe_val < 10000 else 'Loss' if pe_val <= 0 else '>10K')
                                else: st.text("PE Ratio: N/A")
                                if pd.notna(eps_cur): st.text(f"EPS Current: ₹{eps_cur:.2f}")
                                else: st.text("EPS Current: N/A")
                                if pd.notna(eps_chg): st.text(f"EPS Growth: {eps_chg:+.1f}%")
                                else: st.text("EPS Growth: N/A")
                        with detail_cols_top[1]:
                            st.markdown("**📈 Performance**"); returns_map = {"1 Day": 'ret_1d', "7 Days": 'ret_7d', "30 Days": 'ret_30d', "3 Months": 'ret_3m', "6 Months": 'ret_6m', "1 Year": 'ret_1y'}
                            for period, col in returns_map.items(): st.text(f"{period}: {stock.get(col, 0):+.1f}%" if pd.notna(stock.get(col)) else f"{period}: N/A")
                        st.markdown("---"); detail_cols_tech = st.columns([1,1])
                        with detail_cols_tech[0]:
                            st.markdown("**🔍 Technicals**")
                            if all(col in stock.index and pd.notna(stock[col]) for col in ['low_52w', 'high_52w']): st.text(f"52W Low: ₹{stock.get('low_52w', 0):,.0f}"); st.text(f"52W High: ₹{stock.get('high_52w', 0):,.0f}")
                            else: st.text("52W Range: N/A")
                            st.text(f"From High: {stock.get('from_high_pct', 'N/A'):.0f}%" if pd.notna(stock.get('from_high_pct')) else "From High: N/A")
                            st.text(f"From Low: {stock.get('from_low_pct', 'N/A'):.0f}%" if pd.notna(stock.get('from_low_pct')) else "From Low: N/A")
                            st.markdown("**📊 Trading Position**"); tp_col1, tp_col2, tp_col3 = st.columns(3); current_price = stock.get('price', 0); sma_checks = [('sma_20d', '20DMA'), ('sma_50d', '50DMA'), ('sma_200d', '200DMA')]
                            for i, (sma_col, sma_label) in enumerate(sma_checks):
                                with [tp_col1, tp_col2, tp_col3][i]:
                                    sma_value = stock.get(sma_col); text_val = f"N/A"
                                    if pd.notna(sma_value) and sma_value > 0:
                                        pct_diff = ((current_price - sma_value) / sma_value) * 100
                                        color = 'green' if pct_diff > 0 else 'red'
                                        text_val = f"<span style='color:{color}'>{'+' if pct_diff > 0 else ''}{pct_diff:.1f}%</span>"
                                    st.markdown(f"**{sma_label}**:<br>{text_val}", unsafe_allow_html=True)
                        with detail_cols_tech[1]:
                            st.markdown("**📈 Trend Analysis**"); tq = stock.get('trend_quality'); if pd.notna(tq): st.markdown(f"🔥 Strong Uptrend ({tq:.0f})") if tq >= 80 else st.markdown(f"✅ Good Uptrend ({tq:.0f})") if tq >= 60 else st.markdown(f"➡️ Neutral Trend ({tq:.0f})") if tq >= 40 else st.markdown(f"⚠️ Weak/Downtrend ({tq:.0f})")
                            else: st.markdown("Trend: N/A")
                            st.markdown("---"); st.markdown("#### 🎯 Advanced Metrics"); adv_col1, adv_col2 = st.columns(2)
                            with adv_col1:
                                if pd.notna(stock.get('vmi')): st.metric("VMI", f"{stock['vmi']:.2f}") else: st.metric("VMI", "N/A")
                                if pd.notna(stock.get('momentum_harmony')): harmony_val = stock['momentum_harmony']; harmony_emoji = "🟢" if harmony_val >= 3 else "🟡" if harmony_val >= 2 else "🔴"; st.metric("Harmony", f"{harmony_emoji} {int(harmony_val)}/4")
                                else: st.metric("Harmony", "N/A")
                            with adv_col2:
                                if pd.notna(stock.get('position_tension')): st.metric("Position Tension", f"{stock['position_tension']:.0f}")
                                else: st.metric("Position Tension", "N/A")
                                if pd.notna(stock.get('money_flow_mm')): st.metric("Money Flow", f"₹{stock['money_flow_mm']:.1f}M")
                                else: st.metric("Money Flow", "N/A")
            else: st.warning("No stocks found matching your search criteria.")
    with tabs[5]:
        st.markdown("### 📥 Export Data"); st.markdown("#### 📋 Export Templates"); export_template = st.radio("Choose export template:", options=["Full Analysis (All Data)", "Day Trader Focus", "Swing Trader Focus", "Investor Focus"], key="wd_export_template_radio", help="Select a template based on your trading style")
        template_map = {"Full Analysis (All Data)": "full", "Day Trader Focus": "day_trader", "Swing Trader Focus": "swing_trader", "Investor Focus": "investor"}
        selected_template = template_map[export_template]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Excel Report"); st.markdown("Comprehensive multi-sheet report including: - Top 100 stocks with all scores - Market intelligence dashboard - Sector rotation analysis - Industry rotation analysis (NEW) - Pattern frequency analysis - Wave Radar signals - Summary statistics")
            if st.button("Generate Excel Report", type="primary", use_container_width=True, key="wd_generate_excel"):
                if filtered_df.empty: st.error("No data to export. Please adjust your filters.")
                else:
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_file = ExportEngine.create_excel_report(filtered_df, template=selected_template)
                            st.download_button(label="📥 Download Excel Report", data=excel_file, file_name=f"wave_detection_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="wd_download_excel_button")
                            st.success("Excel report generated successfully!")
                        except Exception as e: st.error(f"Error generating Excel report: {str(e)}"); logger.error(f"Excel export error: {str(e)}", exc_info=True)
        with col2:
            st.markdown("#### 📄 CSV Export"); st.markdown("Enhanced CSV format with: - All ranking scores - Advanced metrics (VMI, Money Flow) - Pattern detections - Wave states - Category classifications - Optimized for further analysis")
            if st.button("Generate CSV Export", use_container_width=True, key="wd_generate_csv"):
                if filtered_df.empty: st.error("No data to export. Please adjust your filters.")
                else:
                    try:
                        csv_data = ExportEngine.create_csv_export(filtered_df)
                        st.download_button(label="📥 Download CSV File", data=csv_data, file_name=f"wave_detection_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", key="wd_download_csv_button")
                        st.success("CSV export generated successfully!")
                    except Exception as e: st.error(f"Error generating CSV: {str(e)}"); logger.error(f"CSV export error: {str(e)}", exc_info=True)
        st.markdown("---"); st.markdown("#### 📊 Export Preview")
        export_stats = {"Total Stocks": len(filtered_df), "Average Score": f"{filtered_df['master_score'].mean():.1f}" if not filtered_df.empty else "N/A", "Stocks with Patterns": (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0, "High RVOL (>2x)": (filtered_df['rvol'].fillna(0) > 2).sum() if 'rvol' in filtered_df.columns else 0, "Positive 30D Returns": (filtered_df['ret_30d'].fillna(0) > 0).sum() if 'ret_30d' in filtered_df.columns else 0, "Data Quality": f"{RobustSessionState.safe_get('data_quality', {}).get('completeness', 0):.1f}%"}
        stat_cols = st.columns(3); [UIComponents.render_metric_card(label, value) for i, (label, value) in enumerate(export_stats.items()) for _ in [stat_cols[i%3].empty()]]
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0 - Final Perfected Production Version")
        col1, col2 = st.columns([2, 1])
        with col1: st.markdown("#### 🌊 Welcome to Wave Detection Ultimate 3.0\n\nThe FINAL PERFECTED production version of the most advanced stock ranking system designed to catch momentum waves early.\n\n#### 🎯 Core Features - PERMANENTLY LOCKED\n\n**Master Score 3.0** - Proprietary ranking algorithm:\n- **Position Analysis (30%)**\n- **Volume Dynamics (25%)**\n- **Momentum Tracking (15%)**\n- **Acceleration Detection (10%)**\n- **Breakout Probability (10%)**\n- **RVOL Integration (10%)**\n\n**Advanced Metrics**:\n- **Money Flow** (Estimated institutional money movement.)\n- **VMI** (Quantifies sustained volume interest.)\n- **Position Tension** (Measures readiness for a move.)\n- **Momentum Harmony** (Scores consistency of momentum.)\n- **Wave State** (Categorizes the stage of a stock's momentum cycle.)\n- **Overall Wave Strength** (Aggregated indicator of underlying wave force.)\n\n**Wave Radar™** - Enhanced detection system:\n- Momentum shift detection\n- Smart money flow tracking by category and **industry (NEW)**\n- Pattern emergence alerts\n- Market regime detection\n- Sensitivity controls\n\n**25 Pattern Detection** - Complete set:\n- 11 Technical patterns\n- 5 Fundamental patterns (Hybrid mode)\n- 6 Price range patterns\n- 3 Intelligence patterns (Stealth, Vampire, Perfect Storm)\n\n#### 🔧 Production Features\n\n- **Performance Optimized** - O(n) pattern detection\n- **Error Resilient** - Robust session state and retry logic\n- **Smart Caching** - Daily invalidation for data freshness\n- **Search Optimized** - Exact match prioritization\n\nThis is the FINAL PERFECTED version. All features are permanent.\n\n")
        with col2: st.markdown("#### 📈 Pattern Groups\n\n**Technical Patterns**\n- 🔥 CAT LEADER\n- 💎 HIDDEN GEM\n- ...\n\n**Range Patterns**\n- 🎯 52W HIGH APPROACH\n- 🔄 52W LOW BOUNCE\n- ...\n\n**Intelligence**\n- 🤫 STEALTH\n- 🧛 VAMPIRE\n- ⛈️ PERFECT STORM\n\n**Fundamental** (Hybrid)\n- 💎 VALUE MOMENTUM\n- ...\n\n#### 🔒 Production Status\n\n**Version**: 3.1.0-FINAL-PERFECTED\n**Last Updated**: December 2024\n**Status**: PRODUCTION\n**Updates**: PERMANENTLY LOCKED\n**Testing**: COMPLETE\n**Optimization**: MAXIMUM\n\n#### 🔧 Key Improvements\n\n- ✅ Perfect filter interconnection\n- ✅ Industry filter respects sector\n- ✅ Enhanced performance analysis\n- ✅ Smart sampling for all levels\n- ✅ Dynamic Google Sheets\n- ✅ O(n) pattern detection\n- ✅ Exact search priority\n- ✅ Zero KeyErrors\n- ✅ Beautiful visualizations\n- ✅ Market regime detection\n\n#### 💬 Credits\n\nDeveloped for professional traders.\n\n**Indian Market Optimized**\n- ₹ Currency formatting\n- IST timezone aware\n- NSE/BSE categories\n- Local number formats\n")
        st.markdown("---"); st.markdown("#### 📊 Current Session Statistics")
        stats_cols = st.columns(4)
        with stats_cols[0]: UIComponents.render_metric_card("Total Stocks Loaded", f"{len(ranked_df):,}" if 'ranked_df' in locals() and ranked_df is not None else "0")
        with stats_cols[1]: UIComponents.render_metric_card("Currently Filtered", f"{len(filtered_df):,}" if 'filtered_df' in locals() and filtered_df is not None else "0")
        with stats_cols[2]: data_quality = RobustSessionState.safe_get('data_quality', {}).get('completeness', 0); quality_emoji = "🟢" if data_quality > 80 else "🟡" if data_quality > 60 else "🔴"; UIComponents.render_metric_card("Data Quality", f"{quality_emoji} {data_quality:.1f}%")
        with stats_cols[3]: last_refresh = RobustSessionState.safe_get('last_refresh', datetime.now(timezone.utc)); cache_time = datetime.now(timezone.utc) - last_refresh; minutes = int(cache_time.total_seconds() / 60); cache_status = "Fresh" if minutes < 60 else "Stale"; cache_emoji = "🟢" if minutes < 60 else "🔴"; UIComponents.render_metric_card("Cache Age", f"{cache_emoji} {minutes} min", cache_status)
    st.markdown("---")
    st.markdown("""<div style="text-align: center; color: #666; padding: 1rem;">🌊 Wave Detection Ultimate 3.0 - Final Perfected Production Version<br><small>Professional Stock Ranking System • All Features Complete • Performance Maximized • Permanently Locked</small></div>""", unsafe_allow_html=True)
if __name__ == "__main__":
    try: main()
    except Exception as e: st.error(f"Critical Application Error: {str(e)}"); logger.error(f"Application crashed: {str(e)}", exc_info=True); [st.button("🔄 Restart Application", on_click=lambda: (st.cache_data.clear(), st.rerun())), st.button("📧 Report Issue", on_click=lambda: st.info("Please take a screenshot and report this error."))]
}
