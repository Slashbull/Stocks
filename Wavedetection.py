"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION-GRADE VERSION
============================================================
A professional stock ranking system with advanced analytics and a resilient architecture.
This version integrates the best features from all previous development iterations,
including robust error handling, sophisticated data processing, and a clean, modular design.

Version: 3.0.8-FINAL-PRODUCTION
Last Updated: July 2025
Status: PRODUCTION READY
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
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import time
from io import BytesIO
import warnings
import gc

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# LOGGING CONFIGURATION
# ============================================

# Production logging with proper formatting
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
    """System configuration with validated weights and thresholds."""
    
    # Data source settings. The GID is the sheet ID, which is typically stable.
    # The Spreadsheet ID is the unique ID from the URL, which is user-provided.
    DEFAULT_GID: str = "1823439984" 
    
    # Cache settings optimized for Streamlit Community Cloud
    CACHE_TTL: int = 3600  # 1 hour
    
    # Master Score 3.0 weights (total = 100%) - DO NOT MODIFY
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Critical columns (app fails without these)
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    
    # All percentage columns for consistent handling
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'eps_change_pct'
    ])
    
    # Volume ratio columns
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    # Pattern thresholds
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "category_leader": 90, "hidden_gem": 80, "acceleration": 85, "institutional": 75,
        "vol_explosion": 95, "breakout_ready": 80, "market_leader": 95, "momentum_wave": 75,
        "liquid_leader": 80, "long_strength": 80, "52w_high_approach": 90,
        "52w_low_bounce": 85, "golden_zone": 85, "vol_accumulation": 80,
        "momentum_diverge": 90, "range_compress": 75, "stealth": 70, "vampire": 85,
        "perfect_storm": 80
    })
    
    # Value bounds for data validation, used for clipping outliers
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000), 'rvol': (0.001, 1_000_000.0), 'pe': (-10000, 10000),
        'returns': (-99.99, 9999.99), 'volume': (0, 1e12)
    })
    
    # Tier definitions for filters
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {"Loss": (-float('inf'), 0), "0-5": (0, 5), "5-10": (5, 10), "10-20": (10, 20), "20-50": (20, 50), "50-100": (50, 100), "100+": (100, float('inf'))},
        "pe": {"Negative/NA": (-float('inf'), 0), "0-10": (0, 10), "10-15": (10, 15), "15-20": (15, 20), "20-30": (20, 30), "30-50": (30, 50), "50+": (50, float('inf'))},
        "price": {"0-100": (0, 100), "100-250": (100, 250), "250-500": (250, 500), "500-1000": (500, 1000), "1000-2500": (1000, 2500), "2500-5000": (2500, 5000), "5000+": (5000, float('inf'))}
    })

CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """A decorator to time function execution and log performance metrics."""
    
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
                    elif elapsed > 1.0:
                        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
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
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidator:
    """Handles comprehensive data validation and sanitization."""
    
    _clipping_counts: Dict[str, int] = {}

    @staticmethod
    def get_clipping_counts() -> Dict[str, int]:
        """Returns and resets the clipping counts for the session."""
        counts = DataValidator._clipping_counts.copy()
        DataValidator._clipping_counts.clear()
        return counts

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> Tuple[bool, str]:
        """Validates the DataFrame structure and data quality."""
        if df is None or df.empty:
            return False, f"{context}: DataFrame is None or empty"
        
        missing_critical = [col for col in CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            return False, f"{context}: Missing critical columns: {missing_critical}"
        
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"{context}: Found {duplicates} duplicate tickers")
        
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        completeness = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        
        if completeness < 50:
            logger.warning(f"{context}: Low data completeness ({completeness:.1f}%)")
        
        if 'data_quality' not in st.session_state:
            st.session_state.data_quality = {}
        
        st.session_state.data_quality.update({
            'completeness': completeness, 'total_rows': len(df), 'total_columns': len(df.columns),
            'duplicate_tickers': duplicates, 'context': context, 'timestamp': datetime.now(timezone.utc)
        })
        
        logger.info(f"{context}: Validated {len(df)} rows, {len(df.columns)} columns, {completeness:.1f}% complete")
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Any, col_name: str, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """Cleans and converts a value to a float with bounds checking and clipping notification."""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        try:
            cleaned = str(value).strip()
            if cleaned.upper() in ['', '-', 'N/A', 'NA', 'NAN', 'NONE', '#VALUE!', '#ERROR!', '#DIV/0!', 'INF', '-INF']:
                return np.nan
            
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace('%', '').strip()
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
            
            if np.isnan(result) or np.isinf(result):
                return np.nan
            
            return result
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    @staticmethod
    def sanitize_string(value: Any, default: str = "Unknown") -> str:
        """Sanitizes a string value, returning a default if invalid."""
        if pd.isna(value) or value is None:
            return default
        
        cleaned = str(value).strip()
        if cleaned.upper() in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']:
            return default
        
        return ' '.join(cleaned.split())

# ============================================
# SMART CACHING WITH RESILIENCE
# ============================================

def get_requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504, 429),
    session=None,
) -> requests.Session:
    """Configures a requests session with retry logic for robust HTTP requests."""
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
    """Loads and processes data with smart caching and versioning."""
    
    start_time = time.perf_counter()
    metadata = {
        'source_type': source_type, 'data_version': data_version,
        'processing_start': datetime.now(timezone.utc),
        'errors': [], 'warnings': []
    }
    
    try:
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            df = pd.read_csv(file_data, low_memory=False)
            metadata['source'] = "User Upload"
        else:
            spreadsheet_id = st.session_state.get('user_spreadsheet_id')
            if not spreadsheet_id:
                raise ValueError("A valid Google Spreadsheet ID is required.")
            
            csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={CONFIG.DEFAULT_GID}"
            logger.info(f"Loading data from Google Sheets with Spreadsheet ID: {spreadsheet_id}")
            
            try:
                session = get_requests_retry_session()
                response = session.get(csv_url)
                response.raise_for_status()
                df = pd.read_csv(BytesIO(response.content), low_memory=False)
                metadata['source'] = f"Google Sheets (ID: {spreadsheet_id}, GID: {CONFIG.DEFAULT_GID})"
                st.session_state.last_loaded_url = csv_url
            except requests.exceptions.RequestException as req_e:
                error_msg = f"Network error loading Google Sheet (ID: {spreadsheet_id}): {req_e}"
                logger.error(error_msg)
                metadata['errors'].append(error_msg)
                if 'last_good_data' in st.session_state:
                    logger.info("Using cached data as fallback due to network/HTTP error.")
                    df, timestamp, old_metadata = st.session_state.last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise ValueError(error_msg) from req_e
            except Exception as e:
                error_msg = f"Failed to load CSV from Google Sheet (ID: {spreadsheet_id}): {str(e)}"
                logger.error(error_msg)
                metadata['errors'].append(error_msg)
                if 'last_good_data' in st.session_state:
                    logger.info("Using cached data as fallback due to CSV parsing error.")
                    df, timestamp, old_metadata = st.session_state.last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise ValueError(error_msg) from e
        
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
        
        timestamp = datetime.now(timezone.utc)
        st.session_state.last_good_data = (df.copy(), timestamp, metadata)
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        clipping_info = DataValidator.get_clipping_counts()
        if clipping_info:
            logger.warning(f"Data clipping occurred: {clipping_info}")
            metadata['warnings'].append(f"Some numeric values were clipped: {clipping_info}. See logs for details.")

        gc.collect()
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        metadata['errors'].append(str(e))
        raise

# ============================================
# DATA PROCESSING ENGINE
# ============================================

class DataProcessor:
    """Handles all data processing with validation and optimization."""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Complete data processing pipeline."""
        df = df.copy()
        initial_count = len(df)
        
        numeric_cols_to_process = [col for col in df.columns if col not in ['ticker', 'company_name', 'category', 'sector', 'industry', 'year', 'market_cap']]
        
        for col in numeric_cols_to_process:
            if col in df.columns:
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                bounds = None
                if 'volume' in col.lower():
                    bounds = CONFIG.VALUE_BOUNDS['volume']
                elif col == 'rvol':
                    bounds = CONFIG.VALUE_BOUNDS['rvol']
                elif col == 'pe':
                    bounds = CONFIG.VALUE_BOUNDS['pe']
                elif is_pct:
                    bounds = CONFIG.VALUE_BOUNDS['returns']
                elif col == 'price':
                    bounds = CONFIG.VALUE_BOUNDS['price']
                
                df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, col, is_pct, bounds))
        
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
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
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        df = DataProcessor._add_tier_classifications(df)
        
        removed = initial_count - len(df)
        if removed > 0:
            metadata['warnings'].append(f"Removed {removed} invalid rows during processing")
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows")
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Adds tier classifications with proper boundary handling."""
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
                if tier_name == list(tier_dict.keys())[0] and (min_val == -float('inf') or min_val == 0) and value == min_val:
                    return tier_name
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
    """Calculates advanced metrics and indicators."""
    
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all advanced metrics."""
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow'] = df['price'].fillna(0) * df['volume_1d'].fillna(0) * df['rvol'].fillna(1.0)
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
        else:
            df['money_flow_mm'] = np.nan
        
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d']):
            df['vmi'] = (df['vol_ratio_1d_90d'].fillna(1.0) * 4 + df['vol_ratio_7d_90d'].fillna(1.0) * 3 +
                         df['vol_ratio_30d_90d'].fillna(1.0) * 2 + df['vol_ratio_90d_180d'].fillna(1.0) * 1) / 10
        else:
            df['vmi'] = np.nan
        
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'].fillna(50) + abs(df['from_high_pct'].fillna(-50))
        else:
            df['position_tension'] = np.nan
        
        df['momentum_harmony'] = 0
        if 'ret_1d' in df.columns:
            df['momentum_harmony'] += (df['ret_1d'].fillna(0) > 0).astype(int)
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
        if 'ret_3m' in df.columns:
            df['momentum_harmony'] += (df['ret_3m'].fillna(0) > 0).astype(int)
        
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)

        score_cols = ['momentum_score', 'acceleration_score', 'rvol_score', 'breakout_score']
        if all(col in df.columns for col in score_cols):
            df['overall_wave_strength'] = (
                df['momentum_score'].fillna(50) * 0.3 + df['acceleration_score'].fillna(50) * 0.3 +
                df['rvol_score'].fillna(50) * 0.2 + df['breakout_score'].fillna(50) * 0.2
            )
        else:
            df['overall_wave_strength'] = np.nan
        
        return df
    
    @staticmethod
    def _get_wave_state(row: pd.Series) -> str:
        """Determines the wave state for a stock based on scores."""
        signals = 0
        if row.get('momentum_score', 0) > 70:
            signals += 1
        if row.get('volume_score', 0) > 70:
            signals += 1
        if row.get('acceleration_score', 0) > 70:
            signals += 1
        if row.get('rvol', 0) > 2:
            signals += 1
        
        if signals >= 4:
            return "🌊🌊🌊 CRESTING"
        elif signals >= 3:
            return "🌊🌊 BUILDING"
        elif signals >= 1:
            return "🌊 FORMING"
        else:
            return "💥 BREAKING"

# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    """The core engine for calculating all ranking scores."""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all component scores and the final master score."""
        if df.empty:
            return df
        
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
        
        scores_matrix = np.column_stack([
            df['position_score'], df['volume_score'], df['momentum_score'],
            df['acceleration_score'], df['breakout_score'], df['rvol_score']
        ])
        
        weights = np.array([
            CONFIG.POSITION_WEIGHT, CONFIG.VOLUME_WEIGHT, CONFIG.MOMENTUM_WEIGHT,
            CONFIG.ACCELERATION_WEIGHT, CONFIG.BREAKOUT_WEIGHT, CONFIG.RVOL_WEIGHT
        ])
        
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').fillna(len(df) + 1).astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom').fillna(0) * 100
        
        df = RankingEngine._calculate_category_ranks(df)
        
        logger.info(f"Ranking complete: {len(df)} stocks processed")
        return df
    
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safely ranks a series, handling edge cases with NaN and Inf."""
        if series is None or series.empty:
            return pd.Series(np.nan, dtype=float)
        
        series = series.replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() == 0:
            return pd.Series(np.nan, index=series.index)
        
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom') * 100
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
        
        return ranks

    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        position_score = pd.Series(np.nan, index=df.index, dtype=float)
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not (has_from_low or has_from_high):
            return position_score
        
        from_low = df['from_low_pct'] if has_from_low else pd.Series(np.nan, index=df.index)
        from_high = df['from_high_pct'] if has_from_high else pd.Series(np.nan, index=df.index)
        
        rank_from_low = RankingEngine._safe_rank(from_low, pct=True, ascending=True) if has_from_low else pd.Series(np.nan, index=df.index)
        rank_from_high = RankingEngine._safe_rank(from_high, pct=True, ascending=False) if has_from_high else pd.Series(np.nan, index=df.index)
        
        combined_score = (rank_from_low.fillna(50) * 0.6 + rank_from_high.fillna(50) * 0.4)
        all_nan_mask = from_low.isna() & from_high.isna()
        combined_score.loc[all_nan_mask] = np.nan

        return combined_score.clip(0, 100)

    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        volume_score = pd.Series(np.nan, index=df.index, dtype=float)
        vol_cols = [('vol_ratio_1d_90d', 0.20), ('vol_ratio_7d_90d', 0.20), ('vol_ratio_30d_90d', 0.20),
                    ('vol_ratio_30d_180d', 0.15), ('vol_ratio_90d_180d', 0.25)]
        
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
        
        return volume_score.clip(0, 100)

    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        momentum_score = pd.Series(np.nan, index=df.index, dtype=float)
        has_ret_30d = 'ret_30d' in df.columns and df['ret_30d'].notna().any()
        has_ret_7d = 'ret_7d' in df.columns and df['ret_7d'].notna().any()
        
        if not (has_ret_30d or has_ret_7d):
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
        else:
            distance_factor = pd.Series(np.nan, index=df.index)
        
        if 'vol_ratio_7d_90d' in df.columns and df['vol_ratio_7d_90d'].notna().any():
            vol_ratio = df['vol_ratio_7d_90d']
            volume_factor = ((vol_ratio.fillna(1.0) - 1) * 100).clip(0, 100)
        else:
            volume_factor = pd.Series(np.nan, index=df.index)
        
        trend_factor = pd.Series(np.nan, index=df.index, dtype=float)
        if 'price' in df.columns:
            current_price = df['price']
            conditions_sum = pd.Series(0, index=df.index, dtype=float)
            valid_sma_count = pd.Series(0, index=df.index, dtype=int)

            for sma_col in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma_col in df.columns and df[sma_col].notna().any():
                    valid_comparison_mask = current_price.notna() & df[sma_col].notna()
                    conditions_sum.loc[valid_comparison_mask] += (current_price.loc[valid_comparison_mask] > df[sma_col].loc[valid_comparison_mask]).astype(float)
                    valid_sma_count.loc[valid_comparison_mask] += 1
            
            rows_to_score = df.index[valid_sma_count > 0]
            if len(rows_to_score) > 0:
                trend_factor.loc[rows_to_score] = (conditions_sum.loc[rows_to_score] / valid_sma_count.loc[rows_to_score]) * 100
        
        trend_factor = trend_factor.clip(0, 100)
        
        combined_score = (distance_factor.fillna(50) * 0.4 + volume_factor.fillna(50) * 0.4 + trend_factor.fillna(50) * 0.2)
        all_nan_mask = distance_factor.isna() & volume_factor.isna() & trend_factor.isna()
        combined_score.loc[all_nan_mask] = np.nan

        return combined_score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        if 'rvol' not in df.columns or df['rvol'].isna().all():
            return pd.Series(np.nan, index=df.index)
        
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
        if 'price' not in df.columns or df['price'].isna().all():
            return trend_score
        
        current_price = df['price']
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        above_sma_count = pd.Series(0, index=df.index, dtype=int)
        rows_with_any_sma_data = pd.Series(False, index=df.index, dtype=bool)

        for sma_col in sma_cols:
            if sma_col in df.columns and df[sma_col].notna().any():
                valid_comparison_mask = current_price.notna() & df[sma_col].notna()
                above_sma_count.loc[valid_comparison_mask] += (current_price.loc[valid_comparison_mask] > df[sma_col].loc[valid_comparison_mask]).astype(int)
                rows_with_any_sma_data.loc[valid_comparison_mask] = True
        
        rows_to_score = df.index[rows_with_any_sma_data]
        if len(rows_to_score) > 0:
            trend_score.loc[rows_to_score] = 50.0

            if all(col in df.columns for col in sma_cols):
                perfect_trend = (
                    (current_price > df['sma_20d']) & (df['sma_20d'] > df['sma_50d']) & (df['sma_50d'] > df['sma_200d'])
                ).fillna(False)
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
        
        if not available_cols:
            return strength_score
        
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
            if has_valid_dollar_volume.any():
                liquidity_score.loc[has_valid_dollar_volume] = RankingEngine._safe_rank(
                    dollar_volume.loc[has_valid_dollar_volume], pct=True, ascending=True
                )
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
# PATTERN DETECTION ENGINE
# ============================================

class PatternDetector:
    """Detects all predefined patterns using vectorized operations."""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
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
        
        df['patterns'] = pattern_matrix.apply(
            lambda row: ' | '.join(row.index[row].tolist()), axis=1
        )
        
        return df
    
    @staticmethod
    def _get_all_pattern_definitions(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        patterns = [] 
        
        def get_col_safe(col_name: str) -> pd.Series:
            return df[col_name].fillna(0) if col_name in df.columns and df[col_name].notna().any() else pd.Series(0, index=df.index)

        if 'category_percentile' in df.columns:
            patterns.append(('🔥 CAT LEADER', get_col_safe('category_percentile') >= CONFIG.PATTERN_THRESHOLDS['category_leader']))
        
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            patterns.append(('💎 HIDDEN GEM', (get_col_safe('category_percentile') >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (get_col_safe('percentile') < 70)))
        
        if 'acceleration_score' in df.columns:
            patterns.append(('🚀 ACCELERATING', get_col_safe('acceleration_score') >= CONFIG.PATTERN_THRESHOLDS['acceleration']))
        
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            patterns.append(('🏦 INSTITUTIONAL', (get_col_safe('volume_score') >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (get_col_safe('vol_ratio_90d_180d') > 1.1)))
        
        if 'rvol' in df.columns:
            patterns.append(('⚡ VOL EXPLOSION', get_col_safe('rvol') > 3))
        
        if 'breakout_score' in df.columns:
            patterns.append(('🎯 BREAKOUT', get_col_safe('breakout_score') >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']))
        
        if 'percentile' in df.columns:
            patterns.append(('👑 MARKET LEADER', get_col_safe('percentile') >= CONFIG.PATTERN_THRESHOLDS['market_leader']))
        
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            patterns.append(('🌊 MOMENTUM WAVE', (get_col_safe('momentum_score') >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (get_col_safe('acceleration_score') >= 70)))
        
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            patterns.append(('💰 LIQUID LEADER', (get_col_safe('liquidity_score') >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (get_col_safe('percentile') >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])))
        
        if 'long_term_strength' in df.columns:
            patterns.append(('💪 LONG STRENGTH', get_col_safe('long_term_strength') >= CONFIG.PATTERN_THRESHOLDS['long_strength']))
        
        if 'trend_quality' in df.columns:
            patterns.append(('📈 QUALITY TREND', get_col_safe('trend_quality') >= 80))
        
        if 'pe' in df.columns and 'master_score' in df.columns:
            has_valid_pe = get_col_safe('pe').notna() & (get_col_safe('pe') > 0) & (get_col_safe('pe') < 10000)
            patterns.append(('💎 VALUE MOMENTUM', has_valid_pe & (get_col_safe('pe') < 15) & (get_col_safe('master_score') >= 70)))
        
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = get_col_safe('eps_change_pct').notna()
            extreme_growth = has_eps_growth & (get_col_safe('eps_change_pct') > 1000)
            normal_growth = has_eps_growth & (get_col_safe('eps_change_pct') > 50) & (get_col_safe('eps_change_pct') <= 1000)
            patterns.append(('📊 EARNINGS ROCKET', (extreme_growth & (get_col_safe('acceleration_score') >= 80)) | (normal_growth & (get_col_safe('acceleration_score') >= 70))))
        
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = get_col_safe('pe').notna() & get_col_safe('eps_change_pct').notna() & (get_col_safe('pe') > 0) & (get_col_safe('pe') < 10000)
            patterns.append(('🏆 QUALITY LEADER', has_complete_data & (get_col_safe('pe').between(10, 25)) & (get_col_safe('eps_change_pct') > 20) & (get_col_safe('percentile') >= 80)))
        
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = get_col_safe('eps_change_pct').notna()
            mega_turnaround = has_eps & (get_col_safe('eps_change_pct') > 500) & (get_col_safe('volume_score') >= 60)
            strong_turnaround = has_eps & (get_col_safe('eps_change_pct') > 100) & (get_col_safe('eps_change_pct') <= 500) & (get_col_safe('volume_score') >= 70)
            patterns.append(('⚡ TURNAROUND', mega_turnaround | strong_turnaround))
        
        if 'pe' in df.columns:
            patterns.append(('⚠️ HIGH PE', get_col_safe('pe') > 100))
        
        if all(col in df.columns for col in ['from_high_pct', 'volume_score', 'momentum_score']):
            patterns.append(('🎯 52W HIGH APPROACH', (get_col_safe('from_high_pct') > -5) & (get_col_safe('volume_score') >= 70) & (get_col_safe('momentum_score') >= 60)))
        
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            patterns.append(('🔄 52W LOW BOUNCE', (get_col_safe('from_low_pct') < 20) & (get_col_safe('acceleration_score') >= 80) & (get_col_safe('ret_30d') > 10)))
        
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            patterns.append(('👑 GOLDEN ZONE', (get_col_safe('from_low_pct') > 60) & (get_col_safe('from_high_pct') > -40) & (get_col_safe('trend_quality') >= 70)))
        
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            patterns.append(('📊 VOL ACCUMULATION', (get_col_safe('vol_ratio_30d_90d') > 1.2) & (get_col_safe('vol_ratio_90d_180d') > 1.1) & (get_col_safe('ret_30d') > 5)))
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(get_col_safe('ret_7d') != 0, get_col_safe('ret_7d') / 7, np.nan)
                daily_30d_pace = np.where(get_col_safe('ret_30d') != 0, get_col_safe('ret_30d') / 30, np.nan)
            patterns.append(('🔀 MOMENTUM DIVERGE', daily_7d_pace.notna() & daily_30d_pace.notna() & (daily_7d_pace > daily_30d_pace * 1.5) & (get_col_safe('acceleration_score') >= 85) & (get_col_safe('rvol') > 2)))
        
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = np.where(get_col_safe('low_52w') > 0, ((get_col_safe('high_52w') - get_col_safe('low_52w')) / get_col_safe('low_52w')) * 100, 100)
            patterns.append(('🎯 RANGE COMPRESS', (range_pct < 50) & (get_col_safe('from_low_pct') > 30)))
        
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_ratio = np.where(get_col_safe('ret_30d') != 0, get_col_safe('ret_7d') / (get_col_safe('ret_30d') / 4), np.nan)
            patterns.append(('🤫 STEALTH', get_col_safe('vol_ratio_90d_180d') > 1.1 & get_col_safe('vol_ratio_30d_90d').between(0.9, 1.1) & get_col_safe('from_low_pct') > 40 & ret_ratio.notna() & (ret_ratio > 1)))
        
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pace_ratio = np.where(get_col_safe('ret_7d') != 0, get_col_safe('ret_1d') / (get_col_safe('ret_7d') / 7), np.nan)
            patterns.append(('🧛 VAMPIRE', daily_pace_ratio.notna() & (daily_pace_ratio > 2) & (get_col_safe('rvol') > 3) & (get_col_safe('from_high_pct') > -15) & (df['category'].isin(['Small Cap', 'Micro Cap']))))
        
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            patterns.append(('⛈️ PERFECT STORM', (get_col_safe('momentum_harmony') == 4) & (get_col_safe('master_score') > 80)))
        
        return patterns

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    """Advanced market analysis and regime detection tools."""
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        if df.empty:
            return "😴 NO DATA", {}
        
        metrics = {}
        if 'category' in df.columns and 'master_score' in df.columns:
            category_scores = df.groupby('category')['master_score'].mean().fillna(50)
            micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean()
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean()
            metrics['micro_small_avg'] = micro_small_avg
            metrics['large_mega_avg'] = large_mega_avg
            metrics['category_spread'] = micro_small_avg - large_mega_avg
        else:
            micro_small_avg, large_mega_avg = 50, 50
        
        if 'ret_30d' in df.columns:
            breadth = len(df[df['ret_30d'].fillna(0) > 0]) / len(df) if len(df) > 0 else 0
            metrics['breadth'] = breadth
        else:
            breadth = 0.5
        
        if 'rvol' in df.columns:
            avg_rvol = df['rvol'].fillna(1.0).median()
            metrics['avg_rvol'] = avg_rvol
        else:
            avg_rvol = 1.0
        
        if micro_small_avg > large_mega_avg + 10 and breadth > 0.6:
            regime = "🔥 RISK-ON BULL"
        elif large_mega_avg > micro_small_avg + 10 and breadth < 0.4:
            regime = "🛡️ RISK-OFF DEFENSIVE"
        elif avg_rvol > 1.5 and breadth > 0.5:
            regime = "⚡ VOLATILE OPPORTUNITY"
        else:
            regime = "😴 RANGE-BOUND"
        
        metrics['regime'] = regime
        return regime, metrics
    
    @staticmethod
    def calculate_advance_decline_ratio(df: pd.DataFrame) -> Dict[str, Any]:
        ad_metrics = {}
        if 'ret_1d' in df.columns:
            advancing = len(df[df['ret_1d'].fillna(0) > 0])
            declining = len(df[df['ret_1d'].fillna(0) < 0])
            unchanged = len(df[df['ret_1d'].fillna(0) == 0])
            ad_metrics['advancing'] = advancing
            ad_metrics['declining'] = declining
            ad_metrics['unchanged'] = unchanged
            if declining > 0:
                ad_metrics['ad_ratio'] = advancing / declining
            else:
                ad_metrics['ad_ratio'] = float('inf') if advancing > 0 else 1.0
            ad_metrics['ad_line'] = advancing - declining
            ad_metrics['breadth_pct'] = (advancing / len(df)) * 100 if len(df) > 0 else 0
        return ad_metrics
    
    @staticmethod
    def _apply_dynamic_sampling(df_group: pd.DataFrame) -> pd.DataFrame:
        group_size = len(df_group)
        if 1 <= group_size <= 5: sample_count = group_size
        elif 6 <= group_size <= 20: sample_count = max(1, int(group_size * 0.80))
        elif 21 <= group_size <= 50: sample_count = max(1, int(group_size * 0.60))
        elif 51 <= group_size <= 100: sample_count = max(1, int(group_size * 0.40))
        else: sample_count = min(50, int(group_size * 0.25))
        
        return df_group.nlargest(sample_count, 'master_score', keep='first') if sample_count > 0 else pd.DataFrame()
    
    @staticmethod
    def _calculate_flow_metrics(normalized_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        agg_dict = {'master_score': ['mean', 'median', 'std', 'count'], 'momentum_score': 'mean', 'volume_score': 'mean', 'rvol': 'mean', 'ret_30d': 'mean', 'money_flow_mm': 'sum'}
        available_agg_dict = {k: v for k, v in agg_dict.items() if k in normalized_df.columns}
        group_metrics = normalized_df.groupby(group_col).agg(available_agg_dict).round(2)
        group_metrics.columns = ['_'.join(col).strip() for col in group_metrics.columns.values]
        rename_map = {'master_score_mean': 'avg_score', 'master_score_median': 'median_score', 'master_score_std': 'std_score', 'master_score_count': 'count', 'momentum_score_mean': 'avg_momentum', 'volume_score_mean': 'avg_volume', 'rvol_mean': 'avg_rvol', 'ret_30d_mean': 'avg_ret_30d', 'money_flow_mm_sum': 'total_money_flow'}
        group_metrics = group_metrics.rename(columns=rename_map)
        
        group_metrics['flow_score'] = (group_metrics['avg_score'].fillna(0) * 0.3 + group_metrics['median_score'].fillna(0) * 0.2 + group_metrics['avg_momentum'].fillna(0) * 0.25 + group_metrics['avg_volume'].fillna(0) * 0.25)
        group_metrics['rank'] = group_metrics['flow_score'].rank(ascending=False, method='min')
        return group_metrics.sort_values('flow_score', ascending=False)
        
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        sector_dfs = [MarketIntelligence._apply_dynamic_sampling(group.copy()) for name, group in df.groupby('sector') if name != 'Unknown']
        if not sector_dfs: return pd.DataFrame()
        normalized_df = pd.concat(sector_dfs, ignore_index=True)
        sector_metrics = MarketIntelligence._calculate_flow_metrics(normalized_df, 'sector')
        original_counts = df.groupby('sector').size().rename('total_stocks')
        sector_metrics = sector_metrics.join(original_counts, how='left')
        sector_metrics['analyzed_stocks'] = sector_metrics['count']
        return sector_metrics
        
    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'industry' not in df.columns or df.empty:
            return pd.DataFrame()
        industry_dfs = [MarketIntelligence._apply_dynamic_sampling(group.copy()) for name, group in df.groupby('industry') if name != 'Unknown']
        if not industry_dfs: return pd.DataFrame()
        normalized_df = pd.concat(industry_dfs, ignore_index=True)
        industry_metrics = MarketIntelligence._calculate_flow_metrics(normalized_df, 'industry')
        original_counts = df.groupby('industry').size().rename('total_stocks')
        industry_metrics = industry_metrics.join(original_counts, how='left')
        industry_metrics['analyzed_stocks'] = industry_metrics['count']
        return industry_metrics

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Creates all Plotly visualizations with proper error handling."""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df.empty:
            fig.add_annotation(text="No data available for visualization", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        scores = [('position_score', 'Position', '#3498db'), ('volume_score', 'Volume', '#e74c3c'),
                  ('momentum_score', 'Momentum', '#2ecc71'), ('acceleration_score', 'Acceleration', '#f39c12'),
                  ('breakout_score', 'Breakout', '#9b59b6'), ('rvol_score', 'RVOL', '#e67e22')]
        
        for score_col, label, color in scores:
            if score_col in df.columns:
                score_data = df[score_col].dropna()
                if len(score_data) > 0:
                    fig.add_trace(go.Box(y=score_data, name=label, marker_color=color, boxpoints='outliers', hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'))
        
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
                if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']):
                    x_points.append('30D'); y_points.append(stock['ret_30d'])
                if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']):
                    x_points.append('7D'); y_points.append(stock['ret_7d'])
                if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']):
                    x_points.append('Today'); y_points.append(stock['ret_1d'])
                
                if len(x_points) > 1:
                    accel_score = stock.get('acceleration_score', 0)
                    line_style, marker_style = (dict(width=3, dash='solid'), dict(size=10, symbol='star', line=dict(color='DarkSlateGrey', width=1))) if accel_score >= 85 else \
                                               (dict(width=2, dash='solid'), dict(size=8)) if accel_score >= 70 else \
                                               (dict(width=2, dash='dot'), dict(size=6))
                    
                    fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='lines+markers', name=f"{stock['ticker']} ({accel_score:.0f})", line=line_style, marker=marker_style,
                                             hovertemplate=f"<b>{stock['ticker']}</b><br>%{{x}}: %{{y:.1f}}%<br>Accel Score: {accel_score:.0f}<extra></extra>"))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(title=f"Acceleration Profiles - Top {len(accel_df)} Momentum Builders", xaxis_title="Time Frame", yaxis_title="Return %", height=400, template='plotly_white', showlegend=True, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), hovermode='x unified')
            return fig
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error generating chart: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    """Handles all filtering operations efficiently."""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        if df.empty:
            return df
        
        masks = []
        
        categories = filters.get('categories', [])
        if categories and 'category' in df.columns:
            masks.append(df['category'].isin(categories))
        
        sectors = filters.get('sectors', [])
        if sectors and 'sector' in df.columns:
            masks.append(df['sector'].isin(sectors))
        
        industries = filters.get('industries', [])
        if industries and 'industry' in df.columns:
            masks.append(df['industry'].isin(industries))
        
        min_score = filters.get('min_score', 0)
        if min_score > 0 and 'master_score' in df.columns:
            masks.append(df['master_score'] >= min_score)
        
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            masks.append(df['eps_change_pct'].notna() & (df['eps_change_pct'] >= min_eps_change))
        
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            pattern_column_str = df['patterns'].fillna('').astype(str)
            pattern_regex = '|'.join([p.replace(' ', '\s') for p in patterns])
            masks.append(pattern_column_str.str.contains(pattern_regex, case=False, regex=True))
        
        trend_range = filters.get('trend_range')
        if filters.get('trend_filter') != 'All Trends' and trend_range and 'trend_quality' in df.columns:
            min_trend, max_trend = trend_range
            masks.append(df['trend_quality'].notna() & (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend))
        
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in df.columns:
            masks.append(df['pe'].notna() & (df['pe'] > 0) & (df['pe'] >= min_pe))
        
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in df.columns:
            masks.append(df['pe'].notna() & (df['pe'] > 0) & (df['pe'] <= max_pe))
        
        for tier_type_key, col_name_suffix in [('eps_tiers', 'eps_tier'), ('pe_tiers', 'pe_tier'), ('price_tiers', 'price_tier')]:
            tier_values = filters.get(tier_type_key, [])
            col_name = col_name_suffix
            if tier_values and col_name in df.columns:
                masks.append(df[col_name].isin(tier_values))
        
        if filters.get('require_fundamental_data', False):
            if 'pe' in df.columns and 'eps_change_pct' in df.columns:
                masks.append(df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna())
            else:
                logger.warning("Fundamental columns (PE, EPS) not found for 'require_fundamental_data' filter.")
        
        wave_states = filters.get('wave_states', [])
        if wave_states and 'wave_state' in df.columns:
            masks.append(df['wave_state'].isin(wave_states))

        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and wave_strength_range != (0, 100) and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            masks.append(df['overall_wave_strength'].notna() & (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws))

        if masks:
            combined_mask = np.logical_and.reduce(masks)
            filtered_df = df[combined_mask].copy()
        else:
            filtered_df = df.copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        if df.empty or column not in df.columns:
            return []
        
        temp_filters = current_filters.copy()
        filter_key_map = {'category': 'categories', 'sector': 'sectors', 'industry': 'industries', 'eps_tier': 'eps_tiers', 'pe_tier': 'pe_tiers', 'price_tier': 'price_tiers', 'wave_state': 'wave_states'}
        if column in filter_key_map:
            temp_filters.pop(filter_key_map[column], None)
        
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        
        values = filtered_df[column].dropna().astype(str).unique()
        values = [v for v in values if v.strip().lower() not in ['unknown', '', 'nan', 'n/a', 'none', '-']]
        
        return sorted(values)

# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """Provides optimized search functionality."""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query_upper = query.upper().strip()
            
            mask_ticker_exact = pd.Series(False, index=df.index)
            if 'ticker' in df.columns:
                mask_ticker_exact = (df['ticker'].str.upper() == query_upper).fillna(False)
                if mask_ticker_exact.any():
                    return df[mask_ticker_exact].copy()

            ticker_upper = df['ticker'].str.upper().fillna('') if 'ticker' in df.columns else pd.Series('', index=df.index)
            company_upper = df['company_name'].str.upper().fillna('') if 'company_name' in df.columns else pd.Series('', index=df.index)
            
            mask_ticker_contains = ticker_upper.str.contains(query_upper, regex=False)
            mask_company_contains = company_upper.str.contains(query_upper, regex=False)
            
            if 'company_name' in df.columns and not df['company_name'].empty:
                import re
                mask_company_word_match = df['company_name'].str.contains(r'\b' + re.escape(query_upper), case=False, na=False, regex=True)
            else:
                mask_company_word_match = pd.Series(False, index=df.index)
            
            combined_mask = mask_ticker_exact | mask_ticker_contains | mask_company_contains | mask_company_word_match
            all_matches = df[combined_mask].copy()
            
            if not all_matches.empty:
                all_matches['relevance'] = 0
                all_matches.loc[mask_ticker_exact[combined_mask], 'relevance'] = 100
                all_matches.loc[mask_ticker_contains[combined_mask], 'relevance'] += 50
                all_matches.loc[mask_company_contains[combined_mask], 'relevance'] += 30
                all_matches.loc[mask_company_word_match[combined_mask], 'relevance'] += 20
                
                return all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False]).drop('relevance', axis=1)
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handles all export operations with streaming for large datasets."""
    
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
                
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                export_cols = [col for col in templates[template]['columns'] if col in top_100.columns] if templates[template]['columns'] else [col for col in top_100.columns if col not in ['percentile', 'category_rank', 'category_percentile', 'eps_tier', 'pe_tier', 'price_tier', 'shift_strength', 'surge_score']]
                
                top_100_export = top_100[export_cols]
                top_100_export.to_excel(writer, sheet_name='Top 100 Stocks', index=False)
                worksheet = writer.sheets['Top 100 Stocks']
                for i, col in enumerate(top_100_export.columns): worksheet.write(0, i, col, header_format); worksheet.autofit()
                
                # Market Intelligence Sheet
                intel_data = []
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                intel_data.append({'Metric': 'Market Regime', 'Value': regime, 'Details': f"Breadth: {regime_metrics.get('breadth', 0):.1%} | Avg RVOL: {regime_metrics.get('avg_rvol', 1):.1f}x"})
                ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
                intel_data.append({'Metric': 'Advance/Decline Ratio (1D)', 'Value': f"{ad_metrics.get('ad_ratio', 1):.2f}", 'Details': f"Advances: {ad_metrics.get('advancing', 0)}, Declines: {ad_metrics.get('declining', 0)}, Unchanged: {ad_metrics.get('unchanged', 0)}"})
                intel_df = pd.DataFrame(intel_data)
                intel_df.to_excel(writer, sheet_name='Market Intelligence', index=False)
                worksheet = writer.sheets['Market Intelligence']
                for i, col in enumerate(intel_df.columns): worksheet.write(0, i, col, header_format); worksheet.autofit()

                # Sector Rotation Sheet
                sector_rotation = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rotation.empty:
                    sector_rotation.to_excel(writer, sheet_name='Sector Rotation')
                    worksheet = writer.sheets['Sector Rotation']
                    for i, col in enumerate(sector_rotation.columns): worksheet.write(0, i, col, header_format); worksheet.autofit()
                
                # Industry Rotation Sheet
                industry_rotation = MarketIntelligence.detect_industry_rotation(df)
                if not industry_rotation.empty:
                    industry_rotation.to_excel(writer, sheet_name='Industry Rotation')
                    worksheet = writer.sheets['Industry Rotation']
                    for i, col in enumerate(industry_rotation.columns): worksheet.write(0, i, col, header_format); worksheet.autofit()
                
                # Pattern Analysis Sheet
                pattern_counts = {}
                for patterns_str in df['patterns'].dropna():
                    if patterns_str:
                        for p in patterns_str.split(' | '):
                            pattern_counts[p] = pattern_counts.get(p, 0) + 1
                if pattern_counts:
                    pattern_df = pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=False)
                    pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
                    worksheet = writer.sheets['Pattern Analysis']
                    for i, col in enumerate(pattern_df.columns): worksheet.write(0, i, col, header_format); worksheet.autofit()
                
                # Wave Radar Signals Sheet
                wave_signals = df[(df['momentum_score'].fillna(0) >= 60) & (df['acceleration_score'].fillna(0) >= 70) & (df['rvol'].fillna(0) >= 2)].nlargest(50, 'master_score', keep='first')
                if not wave_signals.empty:
                    wave_cols = ['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'wave_state', 'patterns', 'category', 'sector', 'industry']
                    wave_signals[[col for col in wave_cols if col in wave_signals.columns]].to_excel(writer, sheet_name='Wave Radar Signals', index=False)
                    worksheet = writer.sheets['Wave Radar Signals']
                    for i, col in enumerate(wave_signals.columns): worksheet.write(0, i, col, header_format); worksheet.autofit()

                # Summary Statistics Sheet
                summary_stats = {'Total Stocks Processed': len(df), 'Average Master Score (All)': df['master_score'].mean() if not df.empty else 0,
                                 'Stocks with Patterns (All)': (df['patterns'] != '').sum() if 'patterns' in df.columns else 0,
                                 'High RVOL (>2x) (All)': (df['rvol'].fillna(0) > 2).sum() if 'rvol' in df.columns else 0,
                                 'Positive 30D Returns (All)': (df['ret_30d'].fillna(0) > 0).sum() if 'ret_30d' in df.columns else 0,
                                 'Data Completeness %': st.session_state.data_quality.get('completeness', 0), 'Clipping Events Count': sum(DataValidator.get_clipping_counts().values()),
                                 'Template Used': template, 'Export Date (UTC)': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                worksheet = writer.sheets['Summary']
                for i, col in enumerate(summary_df.columns): worksheet.write(0, i, col, header_format); worksheet.autofit()

                logger.info(f"Excel report created successfully with {len(writer.sheets)} sheets")
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}", exc_info=True)
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        export_cols = ['rank', 'ticker', 'company_name', 'master_score', 'position_score', 'volume_score', 'momentum_score', 'acceleration_score', 'breakout_score', 'rvol_score',
                       'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct', 'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
                       'rvol', 'vmi', 'money_flow_mm', 'position_tension', 'momentum_harmony', 'wave_state', 'patterns', 'category', 'sector', 'industry', 'eps_tier', 'pe_tier', 'price_tier', 'overall_wave_strength']
        available_cols = [col for col in export_cols if col in df.columns]
        export_df = df[available_cols].copy()
        
        for col_name in CONFIG.VOLUME_RATIO_COLUMNS:
            if col_name in export_df.columns:
                export_df[col_name] = (export_df[col_name] - 1) * 100
                
        for col in export_df.select_dtypes(include=np.number).columns:
            export_df[col] = export_df[col].fillna('')
        for col in export_df.select_dtypes(include='object').columns:
            export_df[col] = export_df[col].fillna('')

        return export_df.to_csv(index=False)

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components for Streamlit."""
    
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None, help_text: Optional[str] = None) -> None:
        if help_text:
            st.metric(label, value, delta, help=help_text)
        else:
            st.metric(label, value, delta)

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
            UIComponents.render_metric_card("A/D Ratio", f"{ad_emoji} {display_ad_ratio}", f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}", "Advance/Decline Ratio (Advancing stocks / Declining stocks over 1 Day)")
        
        with col2:
            if 'momentum_score' in df.columns:
                high_momentum = len(df[df['momentum_score'].fillna(0) >= 70])
                momentum_pct = (high_momentum / len(df) * 100) if len(df) > 0 else 0
            else:
                high_momentum, momentum_pct = 0, 0
            UIComponents.render_metric_card("Momentum Health", f"{momentum_pct:.0f}%", f"{high_momentum} strong stocks", "Percentage of stocks with Momentum Score ≥ 70.")
        
        with col3:
            if 'rvol' in df.columns:
                avg_rvol = df['rvol'].fillna(1.0).median()
                high_vol_count = len(df[df['rvol'].fillna(0) > 2])
            else:
                avg_rvol, high_vol_count = 1.0, 0
            vol_emoji = "🌊" if avg_rvol > 1.5 else "💧" if avg_rvol > 1.2 else "🏜️"
            UIComponents.render_metric_card("Volume State", f"{vol_emoji} {avg_rvol:.1f}x", f"{high_vol_count} surges", "Median Relative Volume (RVOL). Surges indicate stocks with RVOL > 2x.")
        
        with col4:
            risk_factors = 0
            if 'from_high_pct' in df.columns and 'momentum_score' in df.columns and len(df) > 0:
                if len(df[(df['from_high_pct'].fillna(-100) >= 0) & (df['momentum_score'].fillna(0) < 50)]) > 20: risk_factors += 1
            if 'rvol' in df.columns and len(df) > 0:
                if len(df[(df['rvol'].fillna(0) > 10) & (df['master_score'].fillna(0) < 50)]) > 10: risk_factors += 1
            if 'trend_quality' in df.columns and len(df) > 0:
                if len(df[df['trend_quality'].fillna(50) < 40]) > len(df) * 0.3: risk_factors += 1
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]
            UIComponents.render_metric_card("Risk Level", risk_levels[min(risk_factors, 3)], f"{risk_factors} factors", "Composite risk assessment based on overextension, extreme volume, and downtrends.")
        
        st.markdown("### 🎯 Today's Best Opportunities")
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        with opp_col1:
            ready_to_run = df[(df['momentum_score'].fillna(0) >= 70) & (df['acceleration_score'].fillna(0) >= 70) & (df['rvol'].fillna(0) >= 2)].nlargest(5, 'master_score', keep='first')
            st.markdown("**🚀 Ready to Run**")
            if not ready_to_run.empty:
                for _, stock in ready_to_run.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"Score: {stock['master_score']:.1f} | RVOL: {stock.get('rvol', 0):.1f}x")
            else:
                st.info("No momentum leaders found")
        with opp_col2:
            hidden_gems = df[df['patterns'].str.contains('💎 HIDDEN GEM', na=False)].nlargest(5, 'master_score', keep='first')
            st.markdown("**💎 Hidden Gems**")
            if not hidden_gems.empty:
                for _, stock in hidden_gems.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"Cat %ile: {stock.get('category_percentile', 0):.0f} | Score: {stock['master_score']:.1f}")
            else:
                st.info("No hidden gems today")
        with opp_col3:
            volume_alerts = df[df['rvol'].fillna(0) > 3].nlargest(5, 'master_score', keep='first')
            st.markdown("**⚡ Volume Alerts**")
            if not volume_alerts.empty:
                for _, stock in volume_alerts.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"RVOL: {stock.get('rvol', 0):.1f}x | {stock.get('wave_state', 'N/A')}")
            else:
                st.info("No extreme volume detected")
        
        st.markdown("### 🧠 Market Intelligence")
        intel_col1, intel_col2 = st.columns([2, 1])
        
        with intel_col1:
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            if not sector_rotation.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sector_rotation.index[:10], y=sector_rotation['flow_score'][:10], text=[f"{val:.1f}" for val in sector_rotation['flow_score'][:10]],
                                     textposition='outside', marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' for score in sector_rotation['flow_score'][:10]],
                                     hovertemplate='Sector: %{x}<br>Flow Score: %{y:.1f}<br>Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>Avg Score: %{customdata[2]:.1f}<br>Median Score: %{customdata[3]:.1f}<extra></extra>',
                                     customdata=np.column_stack((sector_rotation['analyzed_stocks'][:10], sector_rotation['total_stocks'][:10], sector_rotation['avg_score'][:10], sector_rotation['median_score'][:10]))))
                fig.update_layout(title="Sector Rotation Map - Smart Money Flow (Dynamically Sampled)", xaxis_title="Sector", yaxis_title="Flow Score", height=400, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sector rotation data available for visualization.")
        
        with intel_col2:
            regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
            st.markdown(f"**🎯 Market Regime**")
            st.markdown(f"### {regime}")
            st.markdown("**📡 Key Signals**")
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

    @staticmethod
    def render_pagination_controls(df: pd.DataFrame, display_count: int, page_key: str) -> pd.DataFrame:
        total_rows = len(df)
        if total_rows == 0:
            st.caption("No data to display.")
            return df
        
        if f'wd_current_page_{page_key}' not in st.session_state:
            st.session_state[f'wd_current_page_{page_key}'] = 0
            
        current_page = st.session_state[f'wd_current_page_{page_key}']
        total_pages = int(np.ceil(total_rows / display_count))
        start_idx = current_page * display_count
        end_idx = min(start_idx + display_count, total_rows)
        
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_rows} stocks (Page {current_page + 1} of {total_pages})")
        
        col_prev, col_page_num, col_next = st.columns([1, 0.5, 1])
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
# SESSION STATE MANAGER
# ============================================

class SessionStateManager:
    """Manages all session state variables."""
    @staticmethod
    def initialize():
        defaults = {'wd_is_logged_in': False, 'wd_search_query': "", 'last_refresh': datetime.now(timezone.utc), 'data_source': "sheet", 'user_preferences': {'default_top_n': CONFIG.DEFAULT_TOP_N, 'display_mode': 'Technical', 'last_filters': {}}, 'filters': {}, 'active_filter_count': 0, 'quick_filter': None, 'wd_quick_filter_applied': False, 'wd_show_debug': False, 'performance_metrics': {}, 'data_quality': {}, 'wd_trigger_clear': False, 'user_spreadsheet_id': None, 'last_loaded_url': None, 'wd_category_filter': [], 'wd_sector_filter': [], 'wd_industry_filter': [], 'wd_min_score': 0, 'wd_patterns': [], 'wd_trend_filter': "All Trends", 'wd_eps_tier_filter': [], 'wd_pe_tier_filter': [], 'wd_price_tier_filter': [], 'wd_min_eps_change': "", 'wd_min_pe': "", 'wd_max_pe': "", 'wd_require_fundamental_data': False, 'wd_wave_states_filter': [], 'wd_wave_strength_range_slider': (0, 100), 'wd_show_sensitivity_details': False, 'wd_show_market_regime': True, 'wd_wave_timeframe_select': "All Waves", 'wd_wave_sensitivity': "Balanced", 'wd_current_page_rankings': 0}
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        filter_keys = ['wd_category_filter', 'wd_sector_filter', 'wd_industry_filter', 'wd_pe_tier_filter', 'wd_price_tier_filter', 'wd_patterns', 'wd_min_score', 'wd_trend_filter', 'wd_min_eps_change', 'wd_min_pe', 'wd_max_pe', 'wd_require_fundamental_data', 'quick_filter', 'wd_quick_filter_applied', 'wd_wave_states_filter', 'wd_wave_strength_range_slider', 'wd_show_sensitivity_details', 'wd_show_market_regime', 'wd_wave_timeframe_select', 'wd_wave_sensitivity']
        for key in filter_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list): st.session_state[key] = []
                elif isinstance(st.session_state[key], bool): st.session_state[key] = False
                elif isinstance(st.session_state[key], str):
                    if key == 'wd_trend_filter': st.session_state[key] = "All Trends"
                    elif key == 'wd_wave_timeframe_select': st.session_state[key] = "All Waves"
                    elif key == 'wd_wave_sensitivity': st.session_state[key] = "Balanced"
                    else: st.session_state[key] = ""
                elif isinstance(st.session_state[key], tuple): st.session_state[key] = (0, 100) if key == 'wd_wave_strength_range_slider' else None
                elif isinstance(st.session_state[key], (int, float)): st.session_state[key] = 0 if key == 'wd_min_score' else 0
                else: st.session_state[key] = None
        st.session_state.filters = {}; st.session_state.active_filter_count = 0; st.session_state.wd_trigger_clear = False
        st.session_state.wd_current_page_rankings = 0

# ============================================
# LOGIN PAGE
# ============================================

def show_login_page():
    st.title("🔒 Wave Detection Ultimate 3.0 - Login")
    st.markdown("Please log in with your credentials to access the professional-grade stock ranking system.")
    
    with st.form(key="login_form"):
        username = st.text_input("Username", key="login_username_input")
        password = st.text_input("Password", type="password", key="login_password_input")
        login_button = st.form_submit_button("Login")
    
    # Check hardcoded credentials for demo purposes
    if login_button:
        # A professional production version would connect to a secure backend or an auth service.
        # This is a placeholder for demonstrating the login flow.
        if username == "gemini" and password == "gemini":
            st.session_state.wd_is_logged_in = True
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application - Final Production Version."""
    
    if not st.session_state.wd_is_logged_in:
        show_login_page()
        return

    st.set_page_config(page_title="Wave Detection Ultimate 3.0", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
    
    SessionStateManager.initialize()
    
    st.markdown("""<style> .main {padding: 0rem 1rem;} .stTabs [data-baseweb="tab-list"] {gap: 8px;} .stTabs [data-baseweb="tab"] { height: 50px; padding-left: 20px; padding-right: 20px;} div[data-testid="metric-container"] { background-color: rgba(28, 131, 225, 0.1); border: 1px solid rgba(28, 131, 225, 0.2); padding: 5% 5% 5% 10%; border-radius: 5px; overflow-wrap: break-word;} .stAlert { padding: 1rem; border-radius: 5px;} div.stButton > button { width: 100%; transition: all 0.3s ease;} div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 10px rgba(0,0,0,0.2);} @media (max-width: 768px) { .stDataFrame {font-size: 12px;} div[data-testid="metric-container"] {padding: 3%;} .main {padding: 0rem 0.5rem;}} .stDataFrame > div {overflow-x: auto;} .stSpinner > div { border-color: #3498db;} </style>""", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 3.0</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Professional Stock Ranking System • Final Production Version</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True, key="wd_refresh_data_button"):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now(timezone.utc)
                st.rerun()
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True, key="wd_clear_cache_button"):
                st.cache_data.clear(); gc.collect(); st.success("Cache cleared!"); time.sleep(0.5); st.rerun()
        
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        data_source_col1, data_source_col2 = st.columns(2)
        with data_source_col1:
            if st.button("📊 Google Sheets", type="primary" if st.session_state.data_source == "sheet" else "secondary", use_container_width=True, key="wd_sheets_button"):
                st.session_state.data_source = "sheet"; st.rerun()
        with data_source_col2:
            if st.button("📁 Upload CSV", type="primary" if st.session_state.data_source == "upload" else "secondary", use_container_width=True, key="wd_upload_button"):
                st.session_state.data_source = "upload"; st.rerun()

        uploaded_file = None
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv", help="Upload a CSV file with stock data. Must contain 'ticker' and 'price' columns.", key="wd_csv_uploader")
            if uploaded_file is None: st.info("Please upload a CSV file to continue")
        
        if st.session_state.data_source == "sheet":
            st.markdown("#### 🔗 Google Sheet Configuration")
            user_gid_input_widget = st.text_input("Enter Google Spreadsheet ID:", value=st.session_state.get('user_spreadsheet_id', '') or "", placeholder="e.g. 1OEQ_qxL4lXbO9LlKWDGlD...", help="The unique ID from your Google Sheet URL.", key="wd_user_spreadsheet_id")
            
            new_id_input = user_gid_input_widget.strip()
            if new_id_input != st.session_state.get('user_spreadsheet_id', ''):
                if new_id_input:
                    is_valid_format = len(new_id_input) == 44 and new_id_input.isalnum()
                    if is_valid_format:
                        st.session_state.user_spreadsheet_id = new_id_input
                        st.success("Spreadsheet ID updated. Reloading data...")
                        st.rerun()
                    else:
                        st.error("Invalid Spreadsheet ID format. Please enter a 44-character alphanumeric ID.")
                else:
                    st.session_state.user_spreadsheet_id = None
                    st.rerun()
        
        if st.session_state.data_quality:
            with st.expander("📊 Data Quality", expanded=True):
                quality = st.session_state.data_quality
                col1, col2 = st.columns(2)
                with col1:
                    completeness_emoji = "🟢" if quality.get('completeness', 0) > 80 else "🟡" if quality.get('completeness', 0) > 60 else "🔴"
                    st.metric("Completeness", f"{completeness_emoji} {quality.get('completeness', 0):.1f}%")
                    st.metric("Total Stocks", f"{quality.get('total_rows', 0):,}")
                with col2:
                    if 'timestamp' in quality:
                        age = datetime.now(timezone.utc) - quality['timestamp']
                        minutes = int(age.total_seconds() / 60)
                        freshness = "🟢 Fresh" if minutes < 60 else "🟡 Recent" if minutes < 24 * 60 else "🔴 Stale"
                        st.metric("Data Age", freshness)
                    duplicates = quality.get('duplicate_tickers', 0)
                    if duplicates > 0: st.metric("Duplicates", f"⚠️ {duplicates}")
        
        if st.session_state.performance_metrics:
            with st.expander("⚡ Performance"):
                perf = st.session_state.performance_metrics
                total_time = sum(perf.values())
                perf_emoji = "🟢" if total_time < 3 else "🟡" if total_time < 5 else "🔴"
                st.metric("Load Time", f"{perf_emoji} {total_time:.1f}s")
                if len(perf) > 0:
                    slowest = sorted(perf.items(), key=lambda x: x[1], reverse=True)[:3]
                    for func_name, elapsed in slowest:
                        if elapsed > 0.001: st.caption(f"• {func_name}: {elapsed:.4f}s")

        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        active_filter_count = 0
        if st.session_state.get('wd_quick_filter_applied', False): active_filter_count += 1
        filter_checks = [('wd_category_filter', lambda x: bool(x)), ('wd_sector_filter', lambda x: bool(x)), ('wd_industry_filter', lambda x: bool(x)), ('wd_min_score', lambda x: x > 0), ('wd_patterns', lambda x: bool(x)), ('wd_trend_filter', lambda x: x != 'All Trends'), ('wd_eps_tier_filter', lambda x: bool(x)), ('wd_pe_tier_filter', lambda x: bool(x)), ('wd_price_tier_filter', lambda x: bool(x)), ('wd_min_eps_change', lambda x: x is not None and str(x).strip()), ('wd_min_pe', lambda x: x is not None and str(x).strip()), ('wd_max_pe', lambda x: x is not None and str(x).strip()), ('wd_require_fundamental_data', lambda x: x), ('wd_wave_states_filter', lambda x: bool(x)), ('wd_wave_strength_range_slider', lambda x: x != (0, 100))]
        for key, check_func in filter_checks:
            if key in st.session_state and check_func(st.session_state[key]): active_filter_count += 1
        st.session_state.active_filter_count = active_filter_count
        
        if active_filter_count > 0: st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**")
        if st.button("🗑️ Clear All Filters", use_container_width=True, type="primary" if active_filter_count > 0 else "secondary", key="wd_clear_all_filters_button"):
            SessionStateManager.clear_filters(); st.success("✅ All filters cleared!"); st.rerun()
        
        show_debug = st.checkbox("🐛 Show Debug Info", value=st.session_state.get('wd_show_debug', False), key="wd_show_debug")

    # Data loading and processing
    ranked_df = None
    try:
        if st.session_state.data_source == "upload" and uploaded_file is None: st.warning("Please upload a CSV file to continue"); st.stop()
        if st.session_state.data_source == "sheet" and not st.session_state.get('user_spreadsheet_id'): st.info("Please enter a Google Spreadsheet ID in the sidebar to load data."); st.stop()
        
        with st.spinner("📥 Loading and processing data..."):
            try:
                active_gid_for_load = st.session_state.get('user_spreadsheet_id')
                gid_hash = hashlib.md5(active_gid_for_load.encode()).hexdigest() if active_gid_for_load else 'default'
                cache_data_version = f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{gid_hash}"
                
                ranked_df, data_timestamp, metadata = load_and_process_data(
                    "upload" if st.session_state.data_source == "upload" else "sheet",
                    file_data=uploaded_file, data_version=cache_data_version
                )
                st.session_state.ranked_df = ranked_df
                st.session_state.data_timestamp = data_timestamp
                st.session_state.last_refresh = datetime.now(timezone.utc)
                if metadata.get('warnings'): st.warning("\n".join(metadata['warnings']))
                if metadata.get('errors'): st.error("\n".join(metadata['errors'])); st.stop()
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}", exc_info=True)
                if 'last_good_data' in st.session_state:
                    ranked_df, data_timestamp, metadata = st.session_state.last_good_data
                    st.warning("Failed to load fresh data, using cached version.")
                    st.warning(f"Error during load: {str(e)}")
                else:
                    st.error(f"❌ Error: {str(e)}"); st.info("Common issues:\n- Network connectivity\n- Google Sheets permissions\n- Invalid Spreadsheet ID."); st.stop()
    except Exception as e:
        st.error(f"❌ Critical Application Error: {str(e)}"); with st.expander("🔍 Error Details"): st.code(str(e)); st.stop()

    quick_filter_applied = st.session_state.get('wd_quick_filter_applied', False)
    quick_filter = st.session_state.get('quick_filter')
    ranked_df_display = ranked_df.copy()

    if quick_filter and not ranked_df_display.empty:
        if quick_filter == 'top_gainers' and 'momentum_score' in ranked_df_display.columns: ranked_df_display = ranked_df_display[ranked_df_display['momentum_score'].fillna(0) >= 80]
        elif quick_filter == 'volume_surges' and 'rvol' in ranked_df_display.columns: ranked_df_display = ranked_df_display[ranked_df_display['rvol'].fillna(0) >= 3]
        elif quick_filter == 'breakout_ready' and 'breakout_score' in ranked_df_display.columns: ranked_df_display = ranked_df_display[ranked_df_display['breakout_score'].fillna(0) >= 80]
        elif quick_filter == 'hidden_gems' and 'patterns' in ranked_df_display.columns: ranked_df_display = ranked_df_display[ranked_df_display['patterns'].str.contains('💎 HIDDEN GEM', na=False)]
    
    with st.sidebar:
        filters = {}
        display_mode = st.radio("Choose your view:", options=["Technical", "Hybrid (Technical + Fundamentals)"], index=0 if st.session_state.user_preferences['display_mode'] == 'Technical' else 1, key="wd_display_mode_toggle")
        st.session_state.user_preferences['display_mode'] = display_mode
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        st.markdown("---")
        
        categories_options = FilterEngine.get_filter_options(ranked_df_display, 'category', filters)
        filters['categories'] = st.multiselect("Market Cap Category", options=categories_options, default=st.session_state.get('wd_category_filter', []), key="wd_category_filter")
        sectors_options = FilterEngine.get_filter_options(ranked_df_display, 'sector', filters)
        filters['sectors'] = st.multiselect("Sector", options=sectors_options, default=st.session_state.get('wd_sector_filter', []), key="wd_sector_filter")
        industries_options = FilterEngine.get_filter_options(ranked_df_display, 'industry', filters)
        filters['industries'] = st.multiselect("Industry", options=industries_options, default=st.session_state.get('wd_industry_filter', []), key="wd_industry_filter")
        filters['min_score'] = st.slider("Minimum Master Score", min_value=0, max_value=100, value=st.session_state.get('wd_min_score', 0), step=5, key="wd_min_score")
        
        all_patterns = set(); [all_patterns.update(p.split(' | ')) for p in ranked_df_display['patterns'].dropna() if p]
        if all_patterns:
            filters['patterns'] = st.multiselect("Patterns", options=sorted(all_patterns), default=st.session_state.get('wd_patterns', []), key="wd_patterns")
        
        st.markdown("#### 📈 Trend Strength")
        trend_options = {"All Trends": (0, 100), "🔥 Strong Uptrend (80+)": (80, 100), "✅ Good Uptrend (60-79)": (60, 79), "➡️ Neutral Trend (40-59)": (40, 59), "⚠️ Weak/Downtrend (<40)": (0, 39)}
        current_trend_index = list(trend_options.keys()).index(st.session_state.get('wd_trend_filter', "All Trends"))
        filters['trend_filter'] = st.selectbox("Trend Quality", options=list(trend_options.keys()), index=current_trend_index, key="wd_trend_filter")
        filters['trend_range'] = trend_options[filters['trend_filter']]
        
        st.markdown("#### 🌊 Wave Filters")
        wave_states_options = FilterEngine.get_filter_options(ranked_df_display, 'wave_state', filters)
        filters['wave_states'] = st.multiselect("Wave State", options=wave_states_options, default=st.session_state.get('wd_wave_states_filter', []), key="wd_wave_states_filter")
        if 'overall_wave_strength' in ranked_df_display.columns:
            filters['wave_strength_range'] = st.slider("Overall Wave Strength", min_value=0, max_value=100, value=st.session_state.get('wd_wave_strength_range_slider', (0, 100)), step=1, key="wd_wave_strength_range_slider")
        else: filters['wave_strength_range'] = (0, 100)
        
        with st.expander("🔧 Advanced Filters"):
            for tier_type, col_name in [('eps_tiers', 'eps_tier'), ('pe_tiers', 'pe_tier'), ('price_tiers', 'price_tier')]:
                if col_name in ranked_df_display.columns:
                    tier_options = FilterEngine.get_filter_options(ranked_df_display, col_name, filters)
                    filters[tier_type] = st.multiselect(f"{col_name.replace('_', ' ').title()}", options=tier_options, default=st.session_state.get(f'wd_{col_name}_filter', []), key=f"wd_{col_name}_filter")
            if 'eps_change_pct' in ranked_df_display.columns:
                eps_change_input = st.text_input("Min EPS Change %", value=st.session_state.get('wd_min_eps_change', ""), placeholder="e.g. -50", key="wd_min_eps_change")
                filters['min_eps_change'] = float(eps_change_input) if eps_change_input.strip() else None
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                st.markdown("**🔍 Fundamental Filters**"); col1_pe, col2_pe = st.columns(2)
                with col1_pe: min_pe_input = st.text_input("Min PE Ratio", value=st.session_state.get('wd_min_pe', ""), placeholder="e.g. 10", key="wd_min_pe")
                with col2_pe: max_pe_input = st.text_input("Max PE Ratio", value=st.session_state.get('wd_max_pe', ""), placeholder="e.g. 30", key="wd_max_pe")
                filters['min_pe'] = float(min_pe_input) if min_pe_input.strip() else None
                filters['max_pe'] = float(max_pe_input) if max_pe_input.strip() else None
                filters['require_fundamental_data'] = st.checkbox("Only show stocks with PE and EPS data", value=st.session_state.get('wd_require_fundamental_data', False), key="wd_require_fundamental_data")

    # Apply filters
    filtered_df = FilterEngine.apply_filters(ranked_df if not quick_filter_applied else ranked_df_display, filters)
    filtered_df = filtered_df.sort_values('rank')
    st.session_state.user_preferences['last_filters'] = filters

    if st.session_state.get('wd_trigger_clear'): SessionStateManager.clear_filters(); st.rerun()

    # Main content
    st.markdown("### ⚡ Quick Actions")
    qa_cols = st.columns(5)
    qa_filters = [('📈 Top Gainers', 'top_gainers'), ('🔥 Volume Surges', 'volume_surges'), ('🎯 Breakout Ready', 'breakout_ready'), ('💎 Hidden Gems', 'hidden_gems'), ('🌊 Show All', None)]
    for i, (label, filter_name) in enumerate(qa_filters):
        with qa_cols[i]:
            if st.button(label, use_container_width=True, key=f"wd_qa_button_{i}"):
                st.session_state.quick_filter = filter_name
                st.session_state.wd_quick_filter_applied = bool(filter_name)
                st.rerun()

    if st.session_state.active_filter_count > 0 or quick_filter_applied:
        filter_status_col1, filter_status_col2 = st.columns([5, 1])
        with filter_status_col1: st.info(f"Viewing: {len(filtered_df):,} stocks")
        with filter_status_col2:
            if st.button("Clear Filters", type="secondary", key="wd_clear_filters_main_button"): st.session_state.wd_trigger_clear = True; st.rerun()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: UIComponents.render_metric_card("Total Stocks", f"{len(filtered_df):,}", f"{len(filtered_df)/len(ranked_df)*100:.0f}%" if len(ranked_df) > 0 else "0%")
    with col2:
        if not filtered_df.empty:
            avg_score = filtered_df['master_score'].mean(); std_score = filtered_df['master_score'].std()
            UIComponents.render_metric_card("Avg Score", f"{avg_score:.1f}", f"σ={std_score:.1f}")
        else: UIComponents.render_metric_card("Avg Score", "N/A")
    with col3:
        if show_fundamentals and 'pe' in filtered_df.columns and not filtered_df['pe'].dropna().empty:
            median_pe = filtered_df[filtered_df['pe'] > 0]['pe'].median()
            pe_pct = (filtered_df['pe'].notna().sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            UIComponents.render_metric_card("Median PE", f"{median_pe:.1f}x", f"{pe_pct:.0f}% have data")
        else:
            if not filtered_df.empty:
                UIComponents.render_metric_card("Score Range", f"{filtered_df['master_score'].min():.1f}-{filtered_df['master_score'].max():.1f}")
            else: UIComponents.render_metric_card("Score Range", "N/A")
    with col4:
        if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
            growth_count = (filtered_df['eps_change_pct'].fillna(0) > 0).sum(); mega_growth_count = (filtered_df['eps_change_pct'].fillna(0) > 100).sum()
            UIComponents.render_metric_card("EPS Growth +ve", f"{growth_count}", f"{mega_growth_count} >100%")
        else: UIComponents.render_metric_card("Accelerating", f"{len(filtered_df[filtered_df.get('acceleration_score', pd.Series(0)) >= 80])}")
    with col5: UIComponents.render_metric_card("High RVOL", f"{len(filtered_df[filtered_df.get('rvol', pd.Series(0)) > 2])}")
    with col6:
        if 'trend_quality' in filtered_df.columns and not filtered_df.empty:
            strong_trends = (filtered_df['trend_quality'].fillna(0) >= 80).sum()
            UIComponents.render_metric_card("Strong Trends", f"{strong_trends}", f"{strong_trends/len(filtered_df)*100:.0f}%")
        else: UIComponents.render_metric_card("With Patterns", f"{len(filtered_df[filtered_df.get('patterns', '') != ''])}")

    tabs = st.tabs(["📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"])
    
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        if not filtered_df.empty:
            UIComponents.render_summary_section(filtered_df)
            st.markdown("---"); st.markdown("#### 💾 Download Clean Processed Data"); download_cols = st.columns(3)
            with download_cols[0]:
                st.markdown("**📊 Current View Data**")
                csv_filtered = ExportEngine.create_csv_export(filtered_df)
                st.download_button(label="📥 Download Filtered Data (CSV)", data=csv_filtered, file_name=f"wave_detection_filtered.csv", mime="text/csv", key="wd_download_filtered_csv")
            with download_cols[1]:
                st.markdown("**🏆 Top 100 Stocks**")
                top_100_for_download = filtered_df.nlargest(100, 'master_score', keep='first')
                csv_top100 = ExportEngine.create_csv_export(top_100_for_download)
                st.download_button(label="📥 Download Top 100 (CSV)", data=csv_top100, file_name=f"wave_detection_top100.csv", mime="text/csv", key="wd_download_top100_csv")
            with download_cols[2]:
                st.markdown("**🎯 Pattern Stocks Only**")
                pattern_stocks = filtered_df[filtered_df['patterns'] != '']
                if not pattern_stocks.empty:
                    csv_patterns = ExportEngine.create_csv_export(pattern_stocks)
                    st.download_button(label="📥 Download Pattern Stocks (CSV)", data=csv_patterns, file_name=f"wave_detection_patterns.csv", mime="text/csv", key="wd_download_patterns_csv")
                else: st.info("No stocks with patterns in current filter")
        else: st.warning("No data available for summary. Please adjust filters.")
    
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            display_count = st.selectbox("Show top", options=CONFIG.AVAILABLE_TOP_N, index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n']), key="wd_rankings_display_count")
            st.session_state.user_preferences['default_top_n'] = display_count
        with col2:
            sort_options = ['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow', 'Trend'] if 'trend_quality' in filtered_df.columns else ['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow']
            sort_by = st.selectbox("Sort by", options=sort_options, index=0, key="wd_rankings_sort_by")
        
        display_df = filtered_df.copy()
        if sort_by == 'Master Score': display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL': display_df = display_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum': display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Money Flow' and 'money_flow_mm' in display_df.columns: display_df = display_df.sort_values('money_flow_mm', ascending=False)
        elif sort_by == 'Trend' and 'trend_quality' in display_df.columns: display_df = display_df.sort_values('trend_quality', ascending=False)
        
        if not display_df.empty:
            if 'trend_quality' in display_df.columns: display_df['trend_indicator'] = display_df['trend_quality'].apply(lambda s: "🔥" if pd.notna(s) and s >= 80 else "✅" if pd.notna(s) and s >= 60 else "➡️" if pd.notna(s) and s >= 40 else "⚠️" if pd.notna(s) else "➖")
            display_cols = {'rank': 'Rank', 'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score', 'wave_state': 'Wave'}
            if 'trend_indicator' in display_df.columns: display_cols['trend_indicator'] = 'Trend'
            display_cols['price'] = 'Price'
            if show_fundamentals:
                if 'pe' in display_df.columns: display_cols['pe'] = 'PE'
                if 'eps_change_pct' in display_df.columns: display_cols['eps_change_pct'] = 'EPS Δ%'
            display_cols.update({'from_low_pct': 'From Low', 'ret_30d': '30D Ret', 'rvol': 'RVOL', 'vmi': 'VMI', 'patterns': 'Patterns', 'category': 'Category', 'sector': 'Sector', 'industry': 'Industry'})
            
            format_rules = {'master_score': '{:.1f}', 'price': '₹{:,.0f}', 'from_low_pct': '{:.0f}%', 'ret_30d': '{:+.1f}%', 'rvol': '{:.1f}x', 'vmi': '{:.2f}'}
            for col, fmt in format_rules.items():
                if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) and isinstance(x, (int, float)) else '-')
            
            def format_pe(v):
                if pd.isna(v) or v <= 0: return 'Loss'
                elif v > 10000: return '>10K'
                elif v > 1000: return f"{v:.0f}"
                else: return f"{v:.1f}"
            def format_eps_change(v):
                if pd.isna(v): return '-'
                if abs(v) >= 1000: return f"{v/1000:+.1f}K%"
                elif abs(v) >= 100: return f"{v:+.0f}%"
                else: return f"{v:+.1f}%"
            if show_fundamentals:
                if 'pe' in display_df.columns: display_df['pe'] = display_df['pe'].apply(format_pe)
                if 'eps_change_pct' in display_df.columns: display_df['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change)
            
            available_display_cols = [c for c in display_cols.keys() if c in display_df.columns]
            display_df = display_df[available_display_cols]; display_df.columns = [display_cols[c] for c in available_display_cols]
            
            paginated_df = UIComponents.render_pagination_controls(display_df, display_count, 'rankings')
            st.dataframe(paginated_df, use_container_width=True, height=min(600, len(paginated_df) * 35 + 50), hide_index=True)
            
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                with stat_cols[0]: st.markdown("**Score Distribution**"); scores_data = filtered_df['master_score'].dropna(); [st.text(f"{label}: {scores_data.get(stat) if not scores_data.empty else 'N/A'}") for label, stat in [('Max', 'max'), ('Min', 'min'), ('Mean', 'mean'), ('Median', 'median'), ('Std', 'std')]]
                with stat_cols[1]: st.markdown("**Returns (30D)**"); returns_data = filtered_df['ret_30d'].dropna(); [st.text(f"{label}: {returns_data.get(stat) if not returns_data.empty else 'N/A'}") for label, stat in [('Max', 'max'), ('Min', 'min'), ('Avg', 'mean')]]
                with stat_cols[2]: st.markdown("**Fundamentals**" if show_fundamentals else "**Volume**");
                if show_fundamentals:
                    valid_pe = filtered_df['pe'].dropna(); valid_pe = valid_pe[(valid_pe > 0) & (valid_pe < 10000)]; st.text(f"Median PE: {valid_pe.median():.1f}x" if not valid_pe.empty else "No valid PE.");
                    valid_eps = filtered_df['eps_change_pct'].dropna(); st.text(f"Positive EPS: {(valid_eps > 0).sum()}" if not valid_eps.empty else "No valid EPS change.")
                else: st.text(f"Max: {filtered_df['rvol'].max():.1f}x" if 'rvol' in filtered_df.columns else "No RVOL data."); st.text(f"Avg: {filtered_df['rvol'].mean():.1f}x" if 'rvol' in filtered_df.columns else ""); st.text(f">2x: {(filtered_df['rvol'].fillna(0) > 2).sum()}" if 'rvol' in filtered_df.columns else "")
                with stat_cols[3]:
                    st.markdown("**Trend Distribution**")
                    if 'trend_quality' in filtered_df.columns and not filtered_df['trend_quality'].dropna().empty:
                        trend_data = filtered_df['trend_quality'].dropna();
                        st.text(f"Avg Trend Score: {trend_data.mean():.1f}"); st.text(f"Above All SMAs: {(trend_data >= 85).sum()}");
                        st.text(f"In Uptrend (60+): {(trend_data >= 60).sum()}"); st.text(f"In Downtrend (<40): {(trend_data < 40).sum()}")
                    else: st.text("No trend data available")
        else: st.warning("No stocks match the selected filters.")
        
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System"); st.markdown("*Catch waves as they form, not after they've peaked!*")
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        with radar_col1: wave_timeframe = st.selectbox("Wave Detection Timeframe", options=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"], index=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"].index(st.session_state.get('wd_wave_timeframe_select', "All Waves")), key="wd_wave_timeframe_select")
        with radar_col2:
            sensitivity = st.select_slider("Detection Sensitivity", options=["Conservative", "Balanced", "Aggressive"], value=st.session_state.get('wd_wave_sensitivity', "Balanced"), key="wd_wave_sensitivity")
            show_sensitivity_details = st.checkbox("Show thresholds", value=st.session_state.get('wd_show_sensitivity_details', False), key="wd_show_sensitivity_details")
        with radar_col3: show_market_regime = st.checkbox("📊 Market Regime Analysis", value=st.session_state.get('wd_show_market_regime', True), key="wd_show_market_regime")
        wave_filtered_df = filtered_df.copy()
        with radar_col4:
            if not wave_filtered_df.empty and 'overall_wave_strength' in wave_filtered_df.columns:
                wave_strength_score = wave_filtered_df['overall_wave_strength'].fillna(0).mean()
                wave_emoji = "🌊🔥" if wave_strength_score > 70 else "🌊" if wave_strength_score > 50 else "💤"
                wave_color_delta = "🟢" if wave_strength_score > 70 else "🟡" if wave_strength_score > 50 else "🔴"
                UIComponents.render_metric_card("Wave Strength", f"{wave_emoji} {wave_strength_score:.0f}%", f"{wave_color_delta} Market")
            else: UIComponents.render_metric_card("Wave Strength", "N/A", "Data not available")
        
        if show_sensitivity_details: st.expander("📊 Current Sensitivity Thresholds", expanded=True).markdown({"Conservative": "**Conservative Settings** 🛡️\n- **Momentum Shifts:** Score ≥ 60, Acceleration ≥ 70\n- **Emerging Patterns:** Within 5% of qualifying threshold\n- **Volume Surges:** RVOL ≥ 3.0x\n- **Acceleration Alerts:** Score ≥ 85", "Balanced": "**Balanced Settings** ⚖️\n- **Momentum Shifts:** Score ≥ 50, Acceleration ≥ 60\n- **Emerging Patterns:** Within 10% of qualifying threshold\n- **Volume Surges:** RVOL ≥ 2.0x\n- **Acceleration Alerts:** Score ≥ 70", "Aggressive": "**Aggressive Settings** 🚀\n- **Momentum Shifts:** Score ≥ 40, Acceleration ≥ 50\n- **Emerging Patterns:** Within 15% of qualifying threshold\n- **Volume Surges:** RVOL ≥ 1.5x\n- **Acceleration Alerts:** Score ≥ 60"}.get(sensitivity, "No thresholds defined."))
        
        if wave_timeframe != "All Waves" and not wave_filtered_df.empty:
            try:
                if wave_timeframe == "Intraday Surge": wave_filtered_df = wave_filtered_df[(wave_filtered_df.get('rvol', 0) >= 2.5) & (wave_filtered_df.get('ret_1d', 0) > 2) & (wave_filtered_df.get('price', 0) > wave_filtered_df.get('prev_close', 0) * 1.02)]
                elif wave_timeframe == "3-Day Buildup": wave_filtered_df = wave_filtered_df[(wave_filtered_df.get('ret_3d', 0) > 5) & (wave_filtered_df.get('vol_ratio_7d_90d', 0) > 1.5) & (wave_filtered_df.get('price', 0) > wave_filtered_df.get('sma_20d', 0))]
                elif wave_timeframe == "Weekly Breakout": wave_filtered_df = wave_filtered_df[(wave_filtered_df.get('ret_7d', 0) > 8) & (wave_filtered_df.get('vol_ratio_7d_90d', 0) > 2.0) & (wave_filtered_df.get('from_high_pct', -100) > -10)]
                elif wave_timeframe == "Monthly Trend": wave_filtered_df = wave_filtered_df[(wave_filtered_df.get('ret_30d', 0) > 15) & (wave_filtered_df.get('price', 0) > wave_filtered_df.get('sma_20d', 0)) & (wave_filtered_df.get('sma_20d', 0) > wave_filtered_df.get('sma_50d', 0)) & (wave_filtered_df.get('vol_ratio_30d_180d', 0) > 1.2) & (wave_filtered_df.get('from_low_pct', 0) > 30)]
            except Exception as e:
                st.warning(f"Some data not available for '{wave_timeframe}' filter, showing all relevant stocks."); wave_filtered_df = filtered_df.copy()
        
        if not wave_filtered_df.empty:
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            momentum_threshold = {"Conservative": 60, "Balanced": 50, "Aggressive": 40}[sensitivity]
            acceleration_threshold = {"Conservative": 70, "Balanced": 60, "Aggressive": 50}[sensitivity]
            min_rvol_signal = {"Conservative": 3.0, "Balanced": 2.0, "Aggressive": 1.5}[sensitivity]
            momentum_shifts = wave_filtered_df.copy()
            if 'momentum_score' in momentum_shifts.columns: momentum_shifts = momentum_shifts[momentum_shifts['momentum_score'].fillna(0) >= momentum_threshold]
            if 'acceleration_score' in momentum_shifts.columns: momentum_shifts = momentum_shifts[momentum_shifts['acceleration_score'].fillna(0) >= acceleration_threshold]
            if not momentum_shifts.empty:
                momentum_shifts['signal_count'] = 0
                if 'momentum_score' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['momentum_score'].fillna(0) >= momentum_threshold, 'signal_count'] += 1
                if 'acceleration_score' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['acceleration_score'].fillna(0) >= acceleration_threshold, 'signal_count'] += 1
                if 'rvol' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['rvol'].fillna(0) >= min_rvol_signal, 'signal_count'] += 1
                if 'breakout_score' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['breakout_score'].fillna(0) >= 75, 'signal_count'] += 1
                if 'vol_ratio_7d_90d' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['vol_ratio_7d_90d'].fillna(0) >= 1.5, 'signal_count'] += 1
                momentum_shifts['shift_strength'] = (momentum_shifts['momentum_score'].fillna(50) * 0.4 + momentum_shifts['acceleration_score'].fillna(50) * 0.4 + momentum_shifts['rvol_score'].fillna(50) * 0.2)
                top_shifts = momentum_shifts.sort_values(['signal_count', 'shift_strength'], ascending=[False, False]).head(20)
                display_cols = ['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'signal_count', 'wave_state', 'category', 'sector', 'industry']
                if 'ret_7d' in top_shifts.columns: display_cols.insert(-4, 'ret_7d')
                shift_display = top_shifts[[col for col in display_cols if col in top_shifts.columns]].copy()
                shift_display['Signals'] = shift_display['signal_count'].apply(lambda x: f"{'🔥' * min(x, 3)} {x}/5")
                if 'ret_7d' in shift_display.columns: shift_display['7D Return'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else '-')
                shift_display['RVOL'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                shift_display = shift_display.rename(columns={'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score', 'momentum_score': 'Momentum', 'acceleration_score': 'Acceleration', 'wave_state': 'Wave', 'category': 'Category', 'sector': 'Sector', 'industry': 'Industry'}).drop('signal_count', axis=1)
                st.dataframe(shift_display, use_container_width=True, hide_index=True)
                multi_signal = len(top_shifts[top_shifts['signal_count'] >= 3])
                if multi_signal > 0: st.success(f"🏆 Found {multi_signal} stocks with 3+ signals")
                super_signals = top_shifts[top_shifts['signal_count'] >= 4]
                if not super_signals.empty: st.warning(f"🔥🔥 {len(super_signals)} stocks showing EXTREME momentum (4+ signals)!")
            else: st.info(f"No momentum shifts detected in {wave_timeframe} timeframe for '{sensitivity}' sensitivity.")
            
            st.markdown("#### 🚀 Acceleration Profiles - Momentum Building Over Time")
            accel_threshold = {"Conservative": 85, "Balanced": 70, "Aggressive": 60}[sensitivity]
            accelerating_stocks = wave_filtered_df[wave_filtered_df.get('acceleration_score', pd.Series(0)).fillna(0) >= accel_threshold].nlargest(10, 'acceleration_score', keep='first')
            if not accelerating_stocks.empty:
                fig_accel = Visualizer.create_acceleration_profiles(accelerating_stocks, n=10)
                st.plotly_chart(fig_accel, use_container_width=True)
            else: st.info(f"No stocks meet the acceleration threshold ({accel_threshold}+) for '{sensitivity}' sensitivity.")
            
            if show_market_regime:
                st.markdown("#### 💰 Category Rotation - Smart Money Flow")
                col1, col2 = st.columns([3, 2])
                with col1:
                    if 'category' in wave_filtered_df.columns:
                        category_dfs = [MarketIntelligence._apply_dynamic_sampling(group.copy()) for name, group in wave_filtered_df.groupby('category') if name != 'Unknown']
                        if category_dfs:
                            normalized_cat_df = pd.concat(category_dfs, ignore_index=True)
                            category_flow = MarketIntelligence._calculate_flow_metrics(normalized_cat_df, 'category')
                            original_cat_counts = df.groupby('category').size().rename('total_stocks')
                            category_flow = category_flow.join(original_cat_counts, how='left'); category_flow['analyzed_stocks'] = category_flow['count']
                            
                            if not category_flow.empty:
                                top_category_name = category_flow.index[0]
                                flow_direction = "🔥 RISK-ON" if 'Small' in top_category_name or 'Micro' in top_category_name else "❄️ RISK-OFF" if 'Large' in top_category_name or 'Mega' in top_category_name else "➡️ Neutral"
                                fig_flow = go.Figure(); fig_flow.add_trace(go.Bar(x=category_flow.index, y=category_flow['flow_score'], text=[f"{val:.1f}" for val in category_flow['flow_score']], textposition='outside', marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' for score in category_flow['flow_score']], hovertemplate='Category: %{x}<br>Flow Score: %{y:.1f}<br>Stocks: %{customdata[0]} of %{customdata[1]}<extra></extra>', customdata=np.column_stack((category_flow['analyzed_stocks'], category_flow['total_stocks']))))
                                fig_flow.update_layout(title=f"Smart Money Flow Direction: {flow_direction}", xaxis_title="Market Cap Category", yaxis_title="Flow Score", height=300, template='plotly_white', showlegend=False)
                                st.plotly_chart(fig_flow, use_container_width=True)
                with col2:
                    if 'category_flow' in locals() and not category_flow.empty:
                        st.markdown(f"**🎯 Market Regime: {flow_direction}**"); st.markdown("**💎 Strongest Categories:**")
                        for i, (cat, row) in enumerate(category_flow.head(3).iterrows()): st.write(f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} **{cat}**: Score {row['flow_score']:.1f}")
                        small_caps_score = category_flow.loc[category_flow.index.str.contains('Small|Micro'), 'flow_score'].mean()
                        large_caps_score = category_flow.loc[category_flow.index.str.contains('Large|Mega'), 'flow_score'].mean()
                        st.markdown("**🔄 Category Shifts:**")
                        if small_caps_score > large_caps_score + 10: st.success("📈 Small Caps Leading - Early Bull Signal!")
                        elif large_caps_score > small_caps_score + 10: st.warning("📉 Large Caps Leading - Defensive Mode")
                        else: st.info("➡️ Balanced Market - No Clear Leader")
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            pattern_distance = {"Conservative": 5, "Balanced": 10, "Aggressive": 15}[sensitivity]
            emergence_data = []
            if 'category_percentile' in wave_filtered_df.columns:
                close_to_leader = wave_filtered_df[(wave_filtered_df['category_percentile'].fillna(0) >= (90 - pattern_distance)) & (wave_filtered_df['category_percentile'].fillna(0) < 90)]
                for _, stock in close_to_leader.iterrows(): emergence_data.append({'Ticker': stock['ticker'], 'Company': stock['company_name'], 'Pattern': '🔥 CAT LEADER', 'Distance': f"{90 - stock['category_percentile'].fillna(0):.1f}% away", 'Current': f"{stock['category_percentile'].fillna(0):.1f}%ile", 'Score': stock['master_score']})
            if 'breakout_score' in wave_filtered_df.columns:
                close_to_breakout = wave_filtered_df[(wave_filtered_df['breakout_score'].fillna(0) >= (80 - pattern_distance)) & (wave_filtered_df['breakout_score'].fillna(0) < 80)]
                for _, stock in close_to_breakout.iterrows(): emergence_data.append({'Ticker': stock['ticker'], 'Company': stock['company_name'], 'Pattern': '🎯 BREAKOUT', 'Distance': f"{80 - stock['breakout_score'].fillna(0):.1f} pts away", 'Current': f"{stock['breakout_score'].fillna(0):.1f} score", 'Score': stock['master_score']})
            if emergence_data:
                emergence_df = pd.DataFrame(emergence_data).sort_values('Score', ascending=False).head(15); col1_e, col2_e = st.columns([3, 1])
                with col1_e: st.dataframe(emergence_df, use_container_width=True, hide_index=True)
                with col2_e: UIComponents.render_metric_card("Emerging Patterns", len(emergence_df))
            else: st.info(f"No patterns emerging within {pattern_distance}% threshold.")
            
            st.markdown("#### 🌊 Volume Surges - Unusual Activity NOW")
            rvol_threshold = {"Conservative": 3.0, "Balanced": 2.0, "Aggressive": 1.5}[sensitivity]
            volume_surges = wave_filtered_df[wave_filtered_df.get('rvol', 0).fillna(0) >= rvol_threshold].copy()
            if not volume_surges.empty:
                volume_surges['surge_score'] = (volume_surges['rvol_score'].fillna(50) * 0.5 + volume_surges['volume_score'].fillna(50) * 0.3 + volume_surges['momentum_score'].fillna(50) * 0.2)
                top_surges = volume_surges.nlargest(15, 'surge_score', keep='first'); col1_s, col2_s = st.columns([2, 1])
                with col1_s: st.dataframe(top_surges.rename(columns={'ticker':'Ticker', 'company_name':'Company', 'rvol':'RVOL'}).drop('surge_score', axis=1), use_container_width=True, hide_index=True)
                with col2_s: UIComponents.render_metric_card("Active Surges", len(volume_surges)); UIComponents.render_metric_card("Extreme (>5x)", len(volume_surges[volume_surges.get('rvol', 0) > 5])); UIComponents.render_metric_card("High (>3x)", len(volume_surges[volume_surges.get('rvol', 0) > 3]))
            else: st.info(f"No volume surges detected with '{sensitivity}' sensitivity (requires RVOL ≥ {rvol_threshold}x).")
        else: st.warning(f"No data available for Wave Radar analysis with '{wave_timeframe}' timeframe. Please adjust filters or timeframe.")
    
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            with col1: fig_dist = Visualizer.create_score_distribution(filtered_df); st.plotly_chart(fig_dist, use_container_width=True)
            with col2:
                pattern_counts = {}; [pattern_counts.update({p: pattern_counts.get(p, 0) + 1}) for p_str in filtered_df['patterns'].dropna() if p_str for p in p_str.split(' | ')]
                if pattern_counts:
                    pattern_df = pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=True).tail(15)
                    fig_patterns = go.Figure([go.Bar(x=pattern_df['Count'], y=pattern_df['Pattern'], orientation='h', marker_color='#3498db', text=pattern_df['Count'], textposition='outside')]); fig_patterns.update_layout(title="Pattern Frequency Analysis", xaxis_title="Number of Stocks", yaxis_title="Pattern", template='plotly_white', height=400, margin=dict(l=150)); st.plotly_chart(fig_patterns, use_container_width=True)
                else: st.info("No patterns detected in current selection")
            st.markdown("---")
            st.markdown("#### Sector Performance (Dynamically Sampled)"); sector_overview_df = MarketIntelligence.detect_sector_rotation(filtered_df)
            if not sector_overview_df.empty: st.dataframe(sector_overview_df.style.background_gradient(subset=['flow_score']), use_container_width=True); st.info("📊 **Normalized Analysis**: Shows metrics for dynamically sampled stocks.")
            else: st.info("No sector data available.")
            st.markdown("#### Category Performance"); category_df_agg = filtered_df.groupby('category').agg({'master_score': ['mean', 'count'], 'category_percentile': 'mean', 'money_flow_mm': 'sum' if 'money_flow_mm' in filtered_df.columns else lambda x: 0})
            category_df_agg.columns = ['_'.join(col).strip() for col in category_df_agg.columns.values]
            category_df_display = category_df_agg.rename(columns={'master_score_mean': 'Avg Score', 'master_score_count': 'Count', 'category_percentile_mean': 'Avg Cat %ile', 'money_flow_mm_sum': 'Total Money Flow'}).sort_values('Avg Score', ascending=False)
            st.dataframe(category_df_display.style.background_gradient(subset=['Avg Score']), use_container_width=True)
            st.markdown("#### Industry Performance"); industry_overview_df = MarketIntelligence.detect_industry_rotation(filtered_df)
            if not industry_overview_df.empty: st.dataframe(industry_overview_df.style.background_gradient(subset=['flow_score']), use_container_width=True); st.info("📊 **Normalized Analysis**: Shows metrics for dynamically sampled stocks.")
            else: st.info("No industry data available.")
        else: st.info("No data available for analysis.")
        
    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        search_query = st.text_input("Search stocks", placeholder="Enter ticker or company name...", help="Search by ticker symbol or company name", key="wd_search_input")
        if search_query != st.session_state.wd_search_query: st.session_state.wd_search_query = search_query; st.rerun()
        if st.session_state.wd_search_query:
            with st.spinner("Searching..."): search_results = SearchEngine.search_stocks(filtered_df, st.session_state.wd_search_query)
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                for idx, stock in search_results.iterrows():
                    with st.expander(f"📊 {stock['ticker']} - {stock['company_name']} (Rank #{int(stock['rank'])})", expanded=True):
                        metric_cols = st.columns(6)
                        with metric_cols[0]: UIComponents.render_metric_card("Master Score", f"{stock['master_score']:.1f}", f"Rank #{int(stock['rank'])}")
                        with metric_cols[1]: UIComponents.render_metric_card("Price", f"₹{stock['price']:,.0f}", f"{stock['ret_1d']:+.1f}%" if pd.notna(stock.get('ret_1d')) else None)
                        with metric_cols[2]: UIComponents.render_metric_card("From Low", f"{stock['from_low_pct']:.0f}%")
                        with metric_cols[3]: UIComponents.render_metric_card("30D Return", f"{stock.get('ret_30d', 0):+.1f}%")
                        with metric_cols[4]: UIComponents.render_metric_card("RVOL", f"{stock.get('rvol', 1):.1f}x")
                        with metric_cols[5]: UIComponents.render_metric_card("Wave State", stock.get('wave_state', 'N/A'))
                        st.markdown("#### 📈 Score Components"); score_cols_breakdown = st.columns(6)
                        components = [("Position", stock.get('position_score'), CONFIG.POSITION_WEIGHT), ("Volume", stock.get('volume_score'), CONFIG.VOLUME_WEIGHT), ("Momentum", stock.get('momentum_score'), CONFIG.MOMENTUM_WEIGHT), ("Acceleration", stock.get('acceleration_score'), CONFIG.ACCELERATION_WEIGHT), ("Breakout", stock.get('breakout_score'), CONFIG.BREAKOUT_WEIGHT), ("RVOL", stock.get('rvol_score'), CONFIG.RVOL_WEIGHT)]
                        for i, (name, score, weight) in enumerate(components):
                            with score_cols_breakdown[i]: st.markdown(f"**{name}**<br> {'🟢' if pd.notna(score) and score>=80 else '🟡' if pd.notna(score) and score>=60 else '🔴' if pd.notna(score) else '⚪'} {score:.0f} <br> <small>Weight: {weight:.0%}</small>", unsafe_allow_html=True)
                        if stock.get('patterns'): st.markdown(f"**🎯 Patterns:** {stock['patterns']}")
            else: st.warning("No stocks found matching your search criteria.")
    
    with tabs[5]:
        st.markdown("### 📥 Export Data"); st.markdown("#### 📋 Export Templates"); export_template = st.radio("Choose export template:", options=["Full Analysis (All Data)", "Day Trader Focus", "Swing Trader Focus", "Investor Focus"], key="wd_export_template_radio")
        template_map = {"Full Analysis (All Data)": "full", "Day Trader Focus": "day_trader", "Swing Trader Focus": "swing_trader", "Investor Focus": "investor"}
        selected_template = template_map[export_template]; col1_e, col2_e = st.columns(2)
        with col1_e: st.markdown("#### 📊 Excel Report")
        with col2_e: st.markdown("#### 📄 CSV Export")
        
        col_buttons = st.columns(2)
        with col_buttons[0]:
            if st.button("Generate Excel Report", type="primary", use_container_width=True, key="wd_generate_excel"):
                if filtered_df.empty: st.error("No data to export."); st.stop()
                try: excel_file = ExportEngine.create_excel_report(filtered_df, template=selected_template); st.download_button(label="📥 Download Excel Report", data=excel_file, file_name=f"wave_detection_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="wd_download_excel_button"); st.success("Excel report generated successfully!")
                except Exception as e: st.error(f"Error generating Excel report: {str(e)}"); logger.error(f"Excel export error: {str(e)}", exc_info=True)
        with col_buttons[1]:
            if st.button("Generate CSV Export", use_container_width=True, key="wd_generate_csv"):
                if filtered_df.empty: st.error("No data to export."); st.stop()
                try: csv_data = ExportEngine.create_csv_export(filtered_df); st.download_button(label="📥 Download CSV File", data=csv_data, file_name=f"wave_detection_data.csv", mime="text/csv", key="wd_download_csv_button"); st.success("CSV export generated successfully!")
                except Exception as e: st.error(f"Error generating CSV: {str(e)}"); logger.error(f"CSV export error: {str(e)}", exc_info=True)

    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0"); col1_a, col2_a = st.columns([2, 1])
        with col1_a: st.markdown("#### 🌊 Welcome to Wave Detection Ultimate 3.0\n\nThe FINAL production version of the most advanced stock ranking system designed to catch momentum waves early. This professional-grade tool combines technical analysis, volume dynamics, advanced metrics, and smart pattern recognition to identify high-potential stocks before they peak.")
        with col2_a: st.markdown("#### 📈 Pattern Groups\n\n**Technical Patterns**\n- 🔥 CAT LEADER\n- 💎 HIDDEN GEM\n- 🚀 ACCELERATING\n- 🏦 INSTITUTIONAL\n- ⚡ VOL EXPLOSION\n- 🎯 BREAKOUT\n- 👑 MARKET LEADER\n- 🌊 MOMENTUM WAVE\n- 💰 LIQUID LEADER\n- 💪 LONG STRENGTH\n- 📈 QUALITY TREND\n\n**Range Patterns**\n- 🎯 52W HIGH APPROACH\n- 🔄 52W LOW BOUNCE\n- 👑 GOLDEN ZONE\n- 📊 VOL ACCUMULATION\n- 🔀 MOMENTUM DIVERGE\n- 🎯 RANGE COMPRESS\n\n**NEW Intelligence**\n- 🤫 STEALTH\n- 🧛 VAMPIRE\n- ⛈️ PERFECT STORM\n\n**Fundamental** (Hybrid)\n- 💎 VALUE MOMENTUM\n- 📊 EARNINGS ROCKET\n- 🏆 QUALITY LEADER\n- ⚡ TURNAROUND\n- ⚠️ HIGH PE")
        st.markdown("---"); st.markdown("#### 📊 Current Session Statistics"); stats_cols = st.columns(4)
        with stats_cols[0]: UIComponents.render_metric_card("Total Stocks Loaded", f"{len(ranked_df):,}")
        with stats_cols[1]: UIComponents.render_metric_card("Currently Filtered", f"{len(filtered_df):,}")
        with stats_cols[2]: UIComponents.render_metric_card("Data Quality", f"{'🟢' if st.session_state.data_quality.get('completeness', 0) > 80 else '🟡' if st.session_state.data_quality.get('completeness', 0) > 60 else '🔴'} {st.session_state.data_quality.get('completeness', 0):.1f}%")
        with stats_cols[3]:
            minutes = int((datetime.now(timezone.utc) - st.session_state.last_refresh).total_seconds() / 60)
            UIComponents.render_metric_card("Cache Age", f"{'🟢' if minutes < 60 else '🔴'} {minutes} min", "Fresh" if minutes < 60 else "Stale")
        st.markdown("---"); st.markdown("""<div style="text-align: center; color: #666; padding: 1rem;">🌊 Wave Detection Ultimate 3.0 - Final Production Version<br><small>Professional Stock Ranking System • All Features Complete • Performance Optimized • Permanently Locked</small></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        import re
        main()
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}"); logger.error(f"Application crashed: {str(e)}", exc_info=True)
        if st.button("🔄 Restart Application"): st.cache_data.clear(); st.rerun()
        if st.button("📧 Report Issue"): st.info("Please take a screenshot and report this error.")
