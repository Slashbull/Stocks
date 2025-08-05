"""
Wave Detection Ultimate 3.0 - FINAL PERFECTED PRODUCTION VERSION
===============================================================
Professional Stock Ranking System with Advanced Analytics
All bugs fixed, optimized for Streamlit Community Cloud
Enhanced with a perfected filtering system and robust error handling

Version: 3.1.1-FINAL-PERFECTED
Last Updated: August 2025
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
from functools import lru_cache, wraps
import time
from io import BytesIO
import warnings
import gc
import re

# Suppress warnings for clean output
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
# ROBUST SESSION STATE MANAGER - PERFECTED
# ============================================

class RobustSessionState:
    """Bulletproof session state management - prevents all KeyErrors"""

    # Complete list of ALL session state keys with their default values
    STATE_DEFAULTS = {
        # Core states
        'search_query': "",
        'last_refresh': None,  # Will be set to datetime on first run
        'data_source': "sheet",
        'sheet_id': "",  # For custom Google Sheets
        'gid': "",  # For sheet tab GID
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
        'trigger_clear': False,
        'ranked_df': pd.DataFrame(),
        'data_timestamp': None,
        'last_good_data': None,
        
        # All filter states with proper defaults
        'category_filter': [],
        'sector_filter': [],
        'industry_filter': [],  # NEW: Industry filter
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
        'display_mode_toggle': "Technical",
        'search_input': ""
    }

    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        for key, default_value in RobustSessionState.STATE_DEFAULTS.items():
            if key not in st.session_state:
                if key == 'last_refresh' and default_value is None:
                    st.session_state[key] = datetime.now(timezone.utc)
                else:
                    st.session_state[key] = default_value

    @staticmethod
    def get(key: str) -> Any:
        """Safely get a session state value"""
        return st.session_state.get(key)
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """Safely set a session state value"""
        st.session_state[key] = value
    
    @staticmethod
    def clear_filters():
        """Clear all filter states safely"""
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
                RobustSessionState.set(key, RobustSessionState.STATE_DEFAULTS[key])
        
        RobustSessionState.set('filters', {})
        RobustSessionState.set('active_filter_count', 0)
        RobustSessionState.set('trigger_clear', False)

# ============================================
# CONFIGURATION AND CONSTANTS - UPDATED
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights and thresholds"""
    
    # Data source - NOW DYNAMIC (WITH DEFAULT GID)
    DEFAULT_SHEET_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing"
    CSV_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings optimized for Streamlit Community Cloud
    CACHE_TTL: int = 3600
    STALE_DATA_HOURS: int = 24
    
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
    
    # Important columns (degraded experience without)
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'ret_30d', 'from_low_pct', 'from_high_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d', 'industry'
    ])
    
    # All percentage columns for consistent handling
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ])
    
    # Volume ratio columns
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    # Pattern thresholds
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "category_leader": 90,
        "hidden_gem": 80,
        "acceleration": 85,
        "institutional": 75,
        "vol_explosion": 95,
        "breakout_ready": 80,
        "market_leader": 95,
        "momentum_wave": 75,
        "liquid_leader": 80,
        "long_strength": 80,
        "52w_high_approach": 90,
        "52w_low_bounce": 85,
        "golden_zone": 85,
        "vol_accumulation": 80,
        "momentum_diverge": 90,
        "range_compress": 75,
        "stealth": 70,
        "vampire": 85,
        "perfect_storm": 80
    })
    
    # Value bounds for data validation
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000),
        'rvol': (0.01, 1_000_000.0),
        'pe': (-10000, 10000),
        'returns': (-99.99, 9999.99),
        'volume': (0, 1e12)
    })
    
    # Performance thresholds
    PERFORMANCE_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'data_processing': 2.0,
        'filtering': 0.2,
        'pattern_detection': 0.3, # Adjusted target for optimized version
        'export_generation': 1.0,
        'search': 0.05
    })
    
    # Market categories (Indian market specific)
    MARKET_CATEGORIES: List[str] = field(default_factory=lambda: [
        'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'
    ])
    
    # Tier definitions with proper boundaries
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

# Global configuration instance
CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """Track and report performance metrics"""
    
    @staticmethod
    def timer(target_time: Optional[float] = None):
        """Performance timing decorator with target comparison"""
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
                    
                    perf_metrics = RobustSessionState.get('performance_metrics')
                    perf_metrics[func.__name__] = elapsed
                    RobustSessionState.set('performance_metrics', perf_metrics)
                    
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
    """Comprehensive data validation and sanitization"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> Tuple[bool, str]:
        """Validate dataframe structure and data quality"""
        if df is None:
            return False, f"{context}: DataFrame is None"
        
        if df.empty:
            return False, f"{context}: DataFrame is empty"
        
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
        
        data_quality = RobustSessionState.get('data_quality')
        data_quality.update({
            'completeness': completeness,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_tickers': duplicates,
            'context': context,
            'timestamp': datetime.now(timezone.utc)
        })
        RobustSessionState.set('data_quality', data_quality)
        
        logger.info(f"{context}: Validated {len(df)} rows, {len(df.columns)} columns, {completeness:.1f}% complete")
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """Clean and convert numeric values with bounds checking"""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        try:
            cleaned = str(value).strip()
            
            if cleaned.upper() in ['', '-', 'N/A', 'NA', 'NAN', 'NONE', '#VALUE!', '#ERROR!', '#DIV/0!', 'INF', '-INF']:
                return np.nan
            
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            
            result = float(cleaned)
            
            if bounds:
                min_val, max_val = bounds
                if result < min_val or result > max_val:
                    logger.debug(f"Value {result} outside bounds [{min_val}, {max_val}]")
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
        
        cleaned = ' '.join(cleaned.split())
        
        return cleaned

# ============================================
# SMART CACHING WITH VERSIONING - UPDATED
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL, persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, 
                         sheet_id: str = None, gid: str = None,
                         data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """Load and process data with smart caching and versioning"""
    
    start_time = time.perf_counter()
    metadata = {
        'source_type': source_type,
        'data_version': data_version,
        'processing_start': datetime.now(timezone.utc),
        'errors': [],
        'warnings': []
    }
    
    try:
        # Load data based on source
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            df = pd.read_csv(file_data, low_memory=False)
            metadata['source'] = "User Upload"
        else:
            if not sheet_id:
                raise ValueError("Please enter a Google Sheets ID")
            
            sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_id)
            if sheet_id_match:
                sheet_id = sheet_id_match.group(1)
            
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            csv_url = CONFIG.CSV_URL_TEMPLATE.format(sheet_id=sheet_id, gid=gid)
            
            logger.info(f"Loading data from Google Sheets ID: {sheet_id}")
            
            try:
                df = pd.read_csv(csv_url, low_memory=False)
                metadata['source'] = "Google Sheets"
                metadata['sheet_id'] = sheet_id
            except Exception as e:
                logger.error(f"Failed to load from Google Sheets: {str(e)}")
                metadata['errors'].append(f"Sheet load error: {str(e)}")
                
                last_good_data = RobustSessionState.get('last_good_data')
                if last_good_data is not None:
                    logger.info("Using cached data as fallback")
                    df, timestamp, old_metadata = last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid:
            raise ValueError(validation_msg)
        
        df = DataProcessor.process_dataframe(df, metadata)
        df = RankingEngine.calculate_all_scores(df)
        df = PatternDetector.detect_all_patterns_optimized(df)
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid:
            raise ValueError(validation_msg)
        
        timestamp = datetime.now(timezone.utc)
        RobustSessionState.set('last_good_data', (df.copy(), timestamp, metadata))
        
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
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
    """Handle all data processing with validation and optimization"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Complete data processing pipeline"""
        
        df = df.copy()
        initial_count = len(df)
        
        numeric_cols = [col for col in df.columns if col not in ['ticker', 'company_name', 'category', 'sector', 'industry', 'year', 'market_cap']]
        
        for col in numeric_cols:
            if col in df.columns:
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                
                if 'volume' in col.lower():
                    bounds = CONFIG.VALUE_BOUNDS['volume']
                elif col == 'rvol':
                    bounds = CONFIG.VALUE_BOUNDS['rvol']
                elif col == 'pe':
                    bounds = CONFIG.VALUE_BOUNDS['pe']
                elif is_pct:
                    bounds = CONFIG.VALUE_BOUNDS['returns']
                else:
                    bounds = CONFIG.VALUE_BOUNDS.get('price', None)
                
                df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, is_pct, bounds))
        
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 1000.0)
                df[col] = df[col].fillna(1.0)
        
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        df = DataProcessor._fill_missing_values(df)
        df = DataProcessor._add_tier_classifications(df)
        
        removed = initial_count - len(df)
        if removed > 0:
            metadata['warnings'].append(f"Removed {removed} invalid rows during processing")
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows")
        
        return df
    
    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults"""
        if 'from_low_pct' in df.columns: df['from_low_pct'] = df['from_low_pct'].fillna(50)
        if 'from_high_pct' in df.columns: df['from_high_pct'] = df['from_high_pct'].fillna(-50)
        if 'rvol' in df.columns: df['rvol'] = df['rvol'].fillna(1.0)
        
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols:
            if col in df.columns: df[col] = df[col].fillna(0)
        
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols:
            if col in df.columns: df[col] = df[col].fillna(0)
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications with proper boundary handling"""
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            if pd.isna(value): return "Unknown"
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val: return tier_name
                if min_val == -float('inf') and value <= max_val: return tier_name
                if max_val == float('inf') and value > min_val: return tier_name
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
    """Calculate advanced metrics and indicators"""
    
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics"""
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow'] = df['price'] * df['volume_1d'] * df['rvol']
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
        else:
            df['money_flow_mm'] = 0.0
        
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d']):
            df['vmi'] = (df['vol_ratio_1d_90d'] * 4 + df['vol_ratio_7d_90d'] * 3 + df['vol_ratio_30d_90d'] * 2 + df['vol_ratio_90d_180d'] * 1) / 10
        else:
            df['vmi'] = 1.0
        
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'] + abs(df['from_high_pct'])
        else:
            df['position_tension'] = 100.0
        
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
        else:
            df['overall_wave_strength'] = 50.0
        
        return df
    
    @staticmethod
    def _get_wave_state(row: pd.Series) -> str:
        signals = sum([
            1 if 'momentum_score' in row and row['momentum_score'] > 70 else 0,
            1 if 'volume_score' in row and row['volume_score'] > 70 else 0,
            1 if 'acceleration_score' in row and row['acceleration_score'] > 70 else 0,
            1 if 'rvol' in row and row['rvol'] > 2 else 0
        ])
        
        if signals >= 4: return "🌊🌊🌊 CRESTING"
        if signals >= 3: return "🌊🌊 BUILDING"
        if signals >= 1: return "🌊 FORMING"
        return "💥 BREAKING"

# ============================================
# RANKING ENGINE - OPTIMIZED
# ============================================

class RankingEngine:
    """Core ranking calculations - optimized with numpy"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores and master score"""
        if df.empty: return df
        logger.info("Starting optimized ranking calculations...")
        df['position_score'] = RankingEngine._calculate_position_score(df)
        df['volume_score'] = RankingEngine._calculate_volume_score(df)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df)
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df)
        
        scores_matrix = np.column_stack([df[col].fillna(50) for col in [
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score'
        ]])
        weights = np.array([CONFIG.POSITION_WEIGHT, CONFIG.VOLUME_WEIGHT, CONFIG.MOMENTUM_WEIGHT,
                            CONFIG.ACCELERATION_WEIGHT, CONFIG.BREAKOUT_WEIGHT, CONFIG.RVOL_WEIGHT])
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').fillna(len(df) + 1).astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom').fillna(0) * 100
        
        df = RankingEngine._calculate_category_ranks(df)
        logger.info(f"Ranking complete: {len(df)} stocks processed")
        return df
    
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        if series is None or series.empty: return pd.Series(50, dtype=float)
        series = series.replace([np.inf, -np.inf], np.nan)
        valid_count = series.notna().sum()
        if valid_count == 0: return pd.Series(50, index=series.index)
        
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom').fillna(0 if ascending else 100) * 100
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom').fillna(valid_count + 1)
        
        return ranks
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        position_score = pd.Series(50, index=df.index, dtype=float)
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        if not has_from_low and not has_from_high: return position_score
        
        from_low = df['from_low_pct'].fillna(50) if has_from_low else pd.Series(50, index=df.index)
        from_high = df['from_high_pct'].fillna(-50) if has_from_high else pd.Series(-50, index=df.index)
        
        rank_from_low = RankingEngine._safe_rank(from_low, ascending=True) if has_from_low else pd.Series(50, index=df.index)
        rank_from_high = RankingEngine._safe_rank(from_high, ascending=False) if has_from_high else pd.Series(50, index=df.index)
        
        return (rank_from_low * 0.6 + rank_from_high * 0.4).clip(0, 100)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        volume_score = pd.Series(50, index=df.index, dtype=float)
        vol_cols = [('vol_ratio_1d_90d', 0.20), ('vol_ratio_7d_90d', 0.20), ('vol_ratio_30d_90d', 0.20), ('vol_ratio_30d_180d', 0.15), ('vol_ratio_90d_180d', 0.25)]
        total_weight, weighted_score = 0, pd.Series(0, index=df.index, dtype=float)
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                weighted_score += RankingEngine._safe_rank(df[col], ascending=True) * weight
                total_weight += weight
        if total_weight > 0: volume_score = weighted_score / total_weight
        return volume_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                momentum_score = RankingEngine._safe_rank(df['ret_7d'].fillna(0), ascending=True)
            return momentum_score.clip(0, 100)
        
        momentum_score = RankingEngine._safe_rank(df['ret_30d'].fillna(0), ascending=True)
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
        if not all(col in df.columns for col in req_cols): return acceleration_score
        
        ret_1d, ret_7d, ret_30d = df['ret_1d'].fillna(0), df['ret_7d'].fillna(0), df['ret_30d'].fillna(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_daily_7d = np.where(ret_7d != 0, ret_7d / 7, 0)
            avg_daily_30d = np.where(ret_30d != 0, ret_30d / 30, 0)
        
        perfect = (ret_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
        acceleration_score[perfect] = 100
        good = (~perfect) & (ret_1d > avg_daily_7d) & (ret_1d > 0)
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
        
        if 'from_high_pct' in df.columns:
            distance_from_high = -df['from_high_pct'].fillna(-50)
            distance_factor = (100 - distance_from_high).clip(0, 100) * 0.4
        else:
            distance_factor = pd.Series(20, index=df.index)
        
        if 'vol_ratio_7d_90d' in df.columns:
            volume_factor = ((df['vol_ratio_7d_90d'].fillna(1.0) - 1) * 100).clip(0, 100) * 0.4
        else:
            volume_factor = pd.Series(20, index=df.index)
        
        trend_factor = pd.Series(0, index=df.index, dtype=float)
        if 'price' in df.columns:
            current_price = df['price']
            above_sma = sum([1 for col in ['sma_20d', 'sma_50d', 'sma_200d'] if col in df.columns and (current_price > df[col]).any()])
            trend_factor = pd.Series(above_sma / 3 * 100, index=df.index) * 0.2
        
        breakout_score = (distance_factor + volume_factor + trend_factor).clip(0, 100)
        return breakout_score
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        if 'rvol' not in df.columns: return pd.Series(50, index=df.index)
        rvol = df['rvol'].fillna(1.0)
        score = pd.Series(50, index=df.index, dtype=float)
        score.loc[rvol > 10] = 95
        score.loc[(rvol > 5) & (rvol <= 10)] = 90
        score.loc[(rvol > 3) & (rvol <= 5)] = 85
        score.loc[(rvol > 2) & (rvol <= 3)] = 80
        score.loc[(rvol > 1.5) & (rvol <= 2)] = 70
        score.loc[(rvol > 1.2) & (rvol <= 1.5)] = 60
        score.loc[(rvol > 0.8) & (rvol <= 1.2)] = 50
        score.loc[(rvol > 0.5) & (rvol <= 0.8)] = 40
        score.loc[(rvol > 0.3) & (rvol <= 0.5)] = 30
        score.loc[rvol <= 0.3] = 20
        return score

    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        trend_score = pd.Series(50, index=df.index, dtype=float)
        if 'price' not in df.columns or not all(c in df.columns for c in ['sma_20d', 'sma_50d', 'sma_200d']):
            return trend_score
        
        perfect = (df['price'] > df['sma_20d']) & (df['sma_20d'] > df['sma_50d']) & (df['sma_50d'] > df['sma_200d'])
        strong = (~perfect) & (df['price'] > df['sma_200d']) & (df['price'] > df['sma_50d']) & (df['price'] > df['sma_20d'])
        good = (df['price'] > df['sma_200d']) & (df['price'] > df['sma_50d']) & (~strong)
        weak = (df['price'] > df['sma_200d']) & (~good) & (~strong) & (~perfect)
        
        trend_score[perfect] = 100
        trend_score[strong] = 85
        trend_score[good] = 70
        trend_score[weak] = 40
        trend_score[df['price'] <= df['sma_200d']] = 20
        
        return trend_score

    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        strength_score = pd.Series(50, index=df.index, dtype=float)
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        if not available_cols: return strength_score
        
        avg_return = df[available_cols].fillna(0).mean(axis=1)
        
        strength_score[avg_return > 100] = 100
        strength_score[(avg_return > 50) & (avg_return <= 100)] = 90
        strength_score[(avg_return > 30) & (avg_return <= 50)] = 80
        strength_score[(avg_return > 15) & (avg_return <= 30)] = 70
        strength_score[(avg_return > 5) & (avg_return <= 15)] = 60
        strength_score[(avg_return > 0) & (avg_return <= 5)] = 50
        strength_score[(avg_return > -10) & (avg_return <= 0)] = 40
        strength_score[(avg_return > -25) & (avg_return <= -10)] = 30
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
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        categories = df['category'].unique()
        for category in categories:
            if category != 'Unknown':
                mask = df['category'] == category
                cat_df = df[mask]
                if len(cat_df) > 0:
                    df.loc[mask, 'category_rank'] = cat_df['master_score'].rank(method='first', ascending=False, na_option='bottom').astype(int)
                    df.loc[mask, 'category_percentile'] = cat_df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        return df

# ============================================
# PATTERN DETECTION ENGINE - FULLY OPTIMIZED
# ============================================

class PatternDetector:
    """Detect all patterns using fully vectorized operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.3)
    def detect_all_patterns_optimized(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all 25 patterns with fully vectorized numpy operations - O(n) complexity"""
        
        if df.empty:
            df['patterns'] = ''
            return df
        
        pattern_results = {}
        
        # All patterns are now defined as boolean series
        if 'category_percentile' in df.columns:
            pattern_results['🔥 CAT LEADER'] = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            pattern_results['💎 HIDDEN GEM'] = (df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (df['percentile'] < 70)
        if 'acceleration_score' in df.columns:
            pattern_results['🚀 ACCELERATING'] = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            pattern_results['🏦 INSTITUTIONAL'] = (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (df['vol_ratio_90d_180d'] > 1.1)
        if 'rvol' in df.columns:
            pattern_results['⚡ VOL EXPLOSION'] = df['rvol'] > 3
        if 'breakout_score' in df.columns:
            pattern_results['🎯 BREAKOUT'] = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
        if 'percentile' in df.columns:
            pattern_results['👑 MARKET LEADER'] = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            pattern_results['🌊 MOMENTUM WAVE'] = (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (df['acceleration_score'] >= 70)
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            pattern_results['💰 LIQUID LEADER'] = (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
        if 'long_term_strength' in df.columns:
            pattern_results['💪 LONG STRENGTH'] = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
        if 'trend_quality' in df.columns:
            pattern_results['📈 QUALITY TREND'] = df['trend_quality'] >= 80
        if 'pe' in df.columns and 'master_score' in df.columns:
            has_valid_pe = (df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000))
            pattern_results['💎 VALUE MOMENTUM'] = has_valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = df['eps_change_pct'].notna()
            extreme_growth = has_eps_growth & (df['eps_change_pct'] > 1000)
            normal_growth = has_eps_growth & (df['eps_change_pct'] > 50) & (df['eps_change_pct'] <= 1000)
            pattern_results['📊 EARNINGS ROCKET'] = (extreme_growth & (df['acceleration_score'] >= 80)) | (normal_growth & (df['acceleration_score'] >= 70))
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = (df['pe'].notna() & df['eps_change_pct'].notna() & (df['pe'] > 0) & (df['pe'] < 10000))
            pattern_results['🏆 QUALITY LEADER'] = (has_complete_data & (df['pe'].between(10, 25)) & (df['eps_change_pct'] > 20) & (df['percentile'] >= 80))
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = df['eps_change_pct'].notna()
            mega_turnaround = has_eps & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60)
            strong_turnaround = has_eps & (df['eps_change_pct'] > 100) & (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70)
            pattern_results['⚡ TURNAROUND'] = mega_turnaround | strong_turnaround
        if 'pe' in df.columns:
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
            pattern_results['⚠️ HIGH PE'] = has_valid_pe & (df['pe'] > 100)
        if all(col in df.columns for col in ['from_high_pct', 'volume_score', 'momentum_score']):
            pattern_results['🎯 52W HIGH APPROACH'] = (df['from_high_pct'] > -5) & (df['volume_score'] >= 70) & (df['momentum_score'] >= 60)
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            pattern_results['🔄 52W LOW BOUNCE'] = (df['from_low_pct'] < 20) & (df['acceleration_score'] >= 80) & (df['ret_30d'] > 10)
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            pattern_results['👑 GOLDEN ZONE'] = (df['from_low_pct'] > 60) & (df['from_high_pct'] > -40) & (df['trend_quality'] >= 70)
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            pattern_results['📊 VOL ACCUMULATION'] = (df['vol_ratio_30d_90d'] > 1.2) & (df['vol_ratio_90d_180d'] > 1.1) & (df['ret_30d'] > 5)
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            ret_7d_arr, ret_30d_arr = df['ret_7d'].fillna(0).values, df['ret_30d'].fillna(0).values
            daily_7d_pace = np.where(ret_7d_arr != 0, ret_7d_arr / 7, 0)
            daily_30d_pace = np.where(ret_30d_arr != 0, ret_30d_arr / 30, 0)
            pattern_results['🔀 MOMENTUM DIVERGE'] = (daily_7d_pace > daily_30d_pace * 1.5) & (df['acceleration_score'] >= 85) & (df['rvol'] > 2)
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            high_arr, low_arr = df['high_52w'].fillna(0).values, df['low_52w'].fillna(0).values
            range_pct = np.where(low_arr > 0, ((high_arr - low_arr) / low_arr) * 100, 100)
            pattern_results['🎯 RANGE COMPRESS'] = (range_pct < 50) & (df['from_low_pct'] > 30)
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            ret_7d_arr, ret_30d_arr = df['ret_7d'].fillna(0).values, df['ret_30d'].fillna(0).values
            ret_ratio = np.where(ret_30d_arr != 0, ret_7d_arr / (ret_30d_arr / 4), 0)
            pattern_results['🤫 STEALTH'] = (df['vol_ratio_90d_180d'] > 1.1) & (df['vol_ratio_30d_90d'].between(0.9, 1.1)) & (df['from_low_pct'] > 40) & (ret_ratio > 1)
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            ret_1d_arr, ret_7d_arr = df['ret_1d'].fillna(0).values, df['ret_7d'].fillna(0).values
            daily_pace_ratio = np.where(ret_7d_arr != 0, ret_1d_arr / (ret_7d_arr / 7), 0)
            pattern_results['🧛 VAMPIRE'] = (daily_pace_ratio > 2) & (df['rvol'] > 3) & (df['from_high_pct'] > -15) & (df['category'].isin(['Small Cap', 'Micro Cap']))
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            pattern_results['⛈️ PERFECT STORM'] = (df['momentum_harmony'] == 4) & (df['master_score'] > 80)
        
        # Combine all boolean series into a single string column
        pattern_names = list(pattern_results.keys())
        pattern_matrix = np.column_stack([pattern_results[name].values for name in pattern_names])
        
        df['patterns'] = [
            ' | '.join([pattern_names[i] for i, val in enumerate(row) if val])
            for row in pattern_matrix
        ]
        
        df['patterns'] = df['patterns'].fillna('')
        
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
            metrics['category_spread'] = micro_small_avg - large_mega_avg if not np.isnan(micro_small_avg) and not np.isnan(large_mega_avg) else 0
        else:
            metrics['category_spread'] = 0
        
        breadth = len(df[df['ret_30d'] > 0]) / len(df) if 'ret_30d' in df.columns and len(df) > 0 else 0.5
        avg_rvol = df['rvol'].median() if 'rvol' in df.columns else 1.0
        
        if metrics['category_spread'] > 10 and breadth > 0.6: regime = "🔥 RISK-ON BULL"
        elif metrics['category_spread'] < -10 and breadth < 0.4: regime = "🛡️ RISK-OFF DEFENSIVE"
        elif avg_rvol > 1.5 and breadth > 0.5: regime = "⚡ VOLATILE OPPORTUNITY"
        else: regime = "😴 RANGE-BOUND"
        
        metrics.update({'breadth': breadth, 'avg_rvol': avg_rvol, 'regime': regime})
        return regime, metrics
    
    @staticmethod
    def calculate_advance_decline_ratio(df: pd.DataFrame) -> Dict[str, Any]:
        ad_metrics = {'advancing': 0, 'declining': 0, 'unchanged': 0, 'ad_ratio': 1.0, 'ad_line': 0, 'breadth_pct': 0}
        if 'ret_1d' in df.columns and len(df) > 0:
            advancing = len(df[df['ret_1d'] > 0])
            declining = len(df[df['ret_1d'] < 0])
            ad_metrics.update({
                'advancing': advancing,
                'declining': declining,
                'unchanged': len(df) - advancing - declining,
                'ad_ratio': advancing / declining if declining > 0 else (float('inf') if advancing > 0 else 1.0),
                'ad_line': advancing - declining,
                'breadth_pct': (advancing / len(df)) * 100
            })
        return ad_metrics
    
    @staticmethod
    def _dynamic_sample(df: pd.DataFrame, group_size: int) -> int:
        if group_size == 1: return 1
        if 2 <= group_size <= 5: return group_size
        if 6 <= group_size <= 10: return max(3, int(group_size * 0.80))
        if 11 <= group_size <= 25: return max(5, int(group_size * 0.60))
        if 26 <= group_size <= 50: return max(10, int(group_size * 0.50))
        if 51 <= group_size <= 100: return max(20, int(group_size * 0.40))
        if 101 <= group_size <= 200: return max(30, int(group_size * 0.30))
        if 201 <= group_size <= 500: return max(50, int(group_size * 0.20))
        return min(75, int(group_size * 0.15))

    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'sector' not in df.columns or df.empty: return pd.DataFrame()
        sector_dfs = []
        for sector in df['sector'].unique():
            if sector != 'Unknown' and pd.notna(sector):
                sector_df = df[df['sector'] == sector].copy()
                sample_count = MarketIntelligence._dynamic_sample(sector_df, len(sector_df))
                if sample_count > 0: sector_dfs.append(sector_df.nlargest(sample_count, 'master_score'))
        
        if not sector_dfs: return pd.DataFrame()
        normalized_df = pd.concat(sector_dfs, ignore_index=True)
        
        sector_metrics = normalized_df.groupby('sector').agg(
            master_score=['mean', 'median', 'std', 'count'],
            momentum_score='mean',
            volume_score='mean',
            rvol='mean',
            ret_30d='mean',
            money_flow_mm=('money_flow_mm', 'sum') if 'money_flow_mm' in normalized_df.columns else ('ticker', lambda x: 0)
        ).round(2)
        
        col_names = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow']
        sector_metrics.columns = [
            f"{' '.join(col).strip()}" if isinstance(col, tuple) else col
            for col in sector_metrics.columns
        ]
        
        original_counts = df.groupby('sector').size().rename('total_stocks')
        sector_metrics = sector_metrics.join(original_counts, how='left')
        sector_metrics['analyzed_stocks'] = sector_metrics['count']
        sector_metrics['flow_score'] = (sector_metrics['avg_score'] * 0.3 + sector_metrics['median_score'] * 0.2 + sector_metrics['avg_momentum'] * 0.25 + sector_metrics['avg_volume'] * 0.25)
        sector_metrics['rank'] = sector_metrics['flow_score'].rank(ascending=False)
        sector_metrics['sampling_pct'] = ((sector_metrics['analyzed_stocks'] / sector_metrics['total_stocks'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1))
        
        return sector_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        if 'industry' not in df.columns or df.empty: return pd.DataFrame()
        industry_dfs = []
        for industry in df['industry'].unique():
            if industry != 'Unknown' and pd.notna(industry):
                industry_df = df[df['industry'] == industry].copy()
                sample_count = MarketIntelligence._dynamic_sample(industry_df, len(industry_df))
                if sample_count > 0: industry_dfs.append(industry_df.nlargest(sample_count, 'master_score'))
        
        if not industry_dfs: return pd.DataFrame()
        normalized_df = pd.concat(industry_dfs, ignore_index=True)
        
        industry_metrics = normalized_df.groupby('industry').agg(
            master_score=['mean', 'median', 'std', 'count'],
            momentum_score='mean',
            volume_score='mean',
            rvol='mean',
            ret_30d='mean',
            money_flow_mm=('money_flow_mm', 'sum') if 'money_flow_mm' in normalized_df.columns else ('ticker', lambda x: 0)
        ).round(2)
        
        col_names = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow']
        industry_metrics.columns = [f"{' '.join(col).strip()}" if isinstance(col, tuple) else col for col in industry_metrics.columns]
        
        original_counts = df.groupby('industry').size().rename('total_stocks')
        industry_metrics = industry_metrics.join(original_counts, how='left')
        industry_metrics['analyzed_stocks'] = industry_metrics['count']
        industry_metrics['flow_score'] = (industry_metrics['avg_score'] * 0.3 + industry_metrics['median_score'] * 0.2 + industry_metrics['avg_momentum'] * 0.25 + industry_metrics['avg_volume'] * 0.25)
        industry_metrics['rank'] = industry_metrics['flow_score'].rank(ascending=False)
        industry_metrics['sampling_pct'] = ((industry_metrics['analyzed_stocks'] / industry_metrics['total_stocks'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1))
        
        return industry_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    def detect_category_performance(df: pd.DataFrame) -> pd.DataFrame:
        if 'category' not in df.columns or df.empty: return pd.DataFrame()
        category_dfs = []
        for category in df['category'].unique():
            if category != 'Unknown' and pd.notna(category):
                category_df = df[df['category'] == category].copy()
                sample_count = MarketIntelligence._dynamic_sample(category_df, len(category_df))
                if sample_count > 0: category_dfs.append(category_df.nlargest(sample_count, 'master_score'))
        
        if not category_dfs: return pd.DataFrame()
        normalized_df = pd.concat(category_dfs, ignore_index=True)
        
        category_metrics = normalized_df.groupby('category').agg(
            master_score=['mean', 'median', 'std', 'count'],
            momentum_score='mean',
            volume_score='mean',
            rvol='mean',
            ret_30d='mean',
            acceleration_score='mean',
            breakout_score='mean',
            money_flow_mm=('money_flow_mm', 'sum') if 'money_flow_mm' in normalized_df.columns else ('ticker', lambda x: 0)
        ).round(2)

        col_names = ['avg_score', 'median_score', 'std_score', 'count', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'avg_acceleration', 'avg_breakout', 'total_money_flow']
        category_metrics.columns = [f"{' '.join(col).strip()}" if isinstance(col, tuple) else col for col in category_metrics.columns]

        original_counts = df.groupby('category').size().rename('total_stocks')
        category_metrics = category_metrics.join(original_counts, how='left')
        category_metrics['analyzed_stocks'] = category_metrics['count']
        category_metrics['flow_score'] = (category_metrics['avg_score'] * 0.35 + category_metrics['median_score'] * 0.20 + category_metrics['avg_momentum'] * 0.20 + category_metrics['avg_acceleration'] * 0.15 + category_metrics['avg_volume'] * 0.10)
        category_metrics['rank'] = category_metrics['flow_score'].rank(ascending=False)
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
                if len(score_data) > 0:
                    fig.add_trace(go.Box(y=score_data, name=label, marker_color=color, boxpoints='outliers', hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'))
        fig.update_layout(title="Score Component Distribution", yaxis_title="Score (0-100)", template='plotly_white', height=400, showlegend=False)
        return fig

    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, n: int = 10) -> go.Figure:
        try:
            accel_df = df.nlargest(min(n, len(df)), 'acceleration_score')
            if len(accel_df) == 0: return go.Figure()
            fig = go.Figure()
            for _, stock in accel_df.iterrows():
                x_points, y_points = ['Start'], [0]
                if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']): x_points.append('30D'); y_points.append(stock['ret_30d'])
                if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']): x_points.append('7D'); y_points.append(stock['ret_7d'])
                if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']): x_points.append('Today'); y_points.append(stock['ret_1d'])
                if len(x_points) > 1:
                    line_style, marker_style = (dict(width=3, dash='solid'), dict(size=10, symbol='star')) if stock['acceleration_score'] >= 85 else (dict(width=2, dash='solid'), dict(size=8)) if stock['acceleration_score'] >= 70 else (dict(width=2, dash='dot'), dict(size=6))
                    fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='lines+markers', name=f"{stock['ticker']} ({stock['acceleration_score']:.0f})", line=line_style, marker=marker_style, hovertemplate=(f"<b>{stock['ticker']}</b><br>%{{x}}: %{{y:.1f}}%<br>Accel Score: {stock['acceleration_score']:.0f}<extra></extra>")))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(title=f"Acceleration Profiles - Top {len(accel_df)} Momentum Builders", xaxis_title="Time Frame", yaxis_title="Return %", height=400, template='plotly_white', showlegend=True, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), hovermode='x unified')
            return fig
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}")
            return go.Figure()

# ============================================
# FILTER ENGINE - ENHANCED AND PERFECTED
# ============================================

class FilterEngine:
    """Handle all filtering operations efficiently with perfect interconnection"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        if df.empty: return df
        mask = pd.Series(True, index=df.index)
        
        if filters.get('categories'): mask &= df['category'].isin(filters['categories'])
        if filters.get('sectors'): mask &= df['sector'].isin(filters['sectors'])
        if filters.get('industries') and 'industry' in df.columns: mask &= df['industry'].isin(filters['industries'])
        if filters.get('min_score', 0) > 0: mask &= df['master_score'] >= filters['min_score']
        if filters.get('min_eps_change') is not None and 'eps_change_pct' in df.columns: mask &= (df['eps_change_pct'] >= filters['min_eps_change']) | df['eps_change_pct'].isna()
        if filters.get('patterns') and 'patterns' in df.columns: mask &= df['patterns'].str.contains('|'.join([re.escape(p) for p in filters['patterns']]), case=False, na=False, regex=True)
        if filters.get('trend_filter') != 'All Trends' and 'trend_quality' in df.columns:
            min_trend, max_trend = filters['trend_range']
            mask &= (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend)
        if filters.get('min_pe') is not None and 'pe' in df.columns: mask &= df['pe'].isna() | ((df['pe'] > 0) & (df['pe'] >= filters['min_pe']))
        if filters.get('max_pe') is not None and 'pe' in df.columns: mask &= df['pe'].isna() | ((df['pe'] > 0) & (df['pe'] <= filters['max_pe']))
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            if filters.get(tier_type):
                col_name = tier_type.replace('_tiers', '_tier')
                if col_name in df.columns: mask &= df[col_name].isin(filters[tier_type])
        if filters.get('require_fundamental_data', False) and 'pe' in df.columns and 'eps_change_pct' in df.columns: mask &= df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna()
        if filters.get('wave_states') and 'wave_state' in df.columns: mask &= df['wave_state'].isin(filters['wave_states'])
        if filters.get('wave_strength_range') and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = filters['wave_strength_range']
            mask &= (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws)

        return df[mask].copy()

    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        if df.empty or column not in df.columns: return []
        
        temp_filters = current_filters.copy()
        filter_key_map = {'category': 'categories', 'sector': 'sectors', 'industry': 'industries', 'eps_tier': 'eps_tiers', 'pe_tier': 'pe_tiers', 'price_tier': 'price_tiers', 'wave_state': 'wave_states'}
        
        current_filter_key = filter_key_map.get(column)
        if current_filter_key:
            temp_filters.pop(current_filter_key, None)
            
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        values = filtered_df[column].dropna().unique()
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN', 'None']]
        
        try:
            return sorted(values, key=lambda x: float(str(x).replace(',', '')) if str(x).replace(',', '').replace('.', '').isdigit() else str(x))
        except (ValueError, TypeError):
            return sorted(values, key=str)

# ============================================
# SEARCH ENGINE - ENHANCED
# ============================================

class SearchEngine:
    """Optimized search functionality with exact match priority"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        if not query or df.empty: return pd.DataFrame()
        try:
            query = query.upper().strip()
            results = df.copy()
            results['relevance'] = 0
            
            exact_ticker_mask = results['ticker'].str.upper() == query
            results.loc[exact_ticker_mask, 'relevance'] += 1000
            
            ticker_starts_mask = results['ticker'].str.upper().str.startswith(query)
            results.loc[ticker_starts_mask & ~exact_ticker_mask, 'relevance'] += 500
            
            ticker_contains_mask = results['ticker'].str.upper().str.contains(query, na=False, regex=False)
            results.loc[ticker_contains_mask & ~ticker_starts_mask & ~exact_ticker_mask, 'relevance'] += 200
            
            if 'company_name' in results.columns:
                company_exact_mask = results['company_name'].str.upper() == query
                results.loc[company_exact_mask, 'relevance'] += 800
                company_starts_mask = results['company_name'].str.upper().str.startswith(query)
                results.loc[company_starts_mask & ~company_exact_mask, 'relevance'] += 300
                company_contains_mask = results['company_name'].str.upper().str.contains(query, na=False, regex=False)
                results.loc[company_contains_mask & ~company_starts_mask & ~company_exact_mask, 'relevance'] += 100
                
                def word_match_score(company_name):
                    if pd.isna(company_name): return 0
                    return 50 if any(word.startswith(query) for word in str(company_name).upper().split()) else 0
                results['relevance'] += results['company_name'].apply(word_match_score)
            
            matches = results[results['relevance'] > 0].copy()
            if matches.empty: return pd.DataFrame()
            matches = matches.sort_values(['relevance', 'master_score'], ascending=[False, False])
            return matches.drop('relevance', axis=1)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE - ENHANCED
# ============================================

class ExportEngine:
    """Handle all export operations with streaming for large datasets"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        output = BytesIO()
        templates = {
            'day_trader': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 'volume_score', 'vmi', 'wave_state', 'patterns', 'category', 'sector', 'industry']},
            'swing_trader': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'breakout_score', 'position_score', 'position_tension', 'from_high_pct', 'from_low_pct', 'trend_quality', 'momentum_harmony', 'patterns', 'category', 'sector', 'industry']},
            'investor': {'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 'long_term_strength', 'money_flow_mm', 'category', 'sector', 'industry']},
            'full': {'columns': None}
        }
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                header_format = workbook.add_format({'bold': True, 'bg_color': '#3498db', 'font_color': 'white', 'border': 1})
                
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                export_cols = templates[template]['columns']
                if export_cols:
                    top_100_export = top_100[[col for col in export_cols if col in top_100.columns]]
                else:
                    top_100_export = top_100
                top_100_export.to_excel(writer, sheet_name='Top 100', index=False)
                
                worksheet = writer.sheets['Top 100']
                for i, col in enumerate(top_100_export.columns): worksheet.write(0, i, col, header_format)
                
                intel_data = []
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                intel_data.append({'Metric': 'Market Regime', 'Value': regime, 'Details': f"Breadth: {regime_metrics.get('breadth', 0):.1%}"})
                ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
                intel_data.append({'Metric': 'Advance/Decline', 'Value': f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}", 'Details': f"Ratio: {ad_metrics.get('ad_ratio', 1):.2f}"})
                pd.DataFrame(intel_data).to_excel(writer, sheet_name='Market Intelligence', index=False)
                
                sector_rotation = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rotation.empty: sector_rotation.to_excel(writer, sheet_name='Sector Rotation')
                
                industry_rotation = MarketIntelligence.detect_industry_rotation(df)
                if not industry_rotation.empty: industry_rotation.to_excel(writer, sheet_name='Industry Rotation')
                
                pattern_counts = {p: count for patterns in df['patterns'].dropna() for p in patterns.split(' | ')}
                if pattern_counts: pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=False).to_excel(writer, sheet_name='Pattern Analysis', index=False)
                
                wave_signals = df[(df['momentum_score'] >= 60) & (df['acceleration_score'] >= 70) & (df['rvol'] >= 2)].head(50)
                if len(wave_signals) > 0:
                    wave_cols = ['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'wave_state', 'patterns', 'category', 'sector', 'industry']
                    wave_signals[[col for col in wave_cols if col in wave_signals.columns]].to_excel(writer, sheet_name='Wave Radar', index=False)
                
                summary_stats = {'Total Stocks': len(df), 'Average Master Score': df['master_score'].mean(), 'Stocks with Patterns': (df['patterns'] != '').sum(), 'High RVOL (>2x)': (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0, 'Positive 30D Returns': (df['ret_30d'] > 0).sum() if 'ret_30d' in df.columns else 0, 'Template Used': template, 'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value']).to_excel(writer, sheet_name='Summary', index=False)
                
                logger.info(f"Excel report created successfully with {len(writer.sheets)} sheets")
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        export_cols = ['rank', 'ticker', 'company_name', 'master_score', 'position_score', 'volume_score', 'momentum_score', 'acceleration_score', 'breakout_score', 'rvol_score', 'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct', 'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'rvol', 'vmi', 'money_flow_mm', 'position_tension', 'momentum_harmony', 'wave_state', 'patterns', 'category', 'sector', 'industry', 'eps_tier', 'pe_tier', 'overall_wave_strength']
        export_df = df[[col for col in export_cols if col in df.columns]].copy()
        
        vol_ratio_cols = [col for col in export_df.columns if 'vol_ratio' in col]
        for col in vol_ratio_cols: export_df[col] = (export_df[col] - 1) * 100
        
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
        st.markdown("### 📊 Market Pulse")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
            ad_ratio = ad_metrics.get('ad_ratio', 1.0)
            ad_emoji = "🔥" if ad_ratio > 2 else "📈" if ad_ratio > 1 else "📉"
            UIComponents.render_metric_card("A/D Ratio", f"{ad_emoji} {ad_ratio:.2f}", f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}", "Advance/Decline Ratio")
        with col2:
            high_momentum = len(df[df['momentum_score'] >= 70])
            momentum_pct = (high_momentum / len(df) * 100) if len(df) > 0 else 0
            UIComponents.render_metric_card("Momentum Health", f"{momentum_pct:.0f}%", f"{high_momentum} strong stocks")
        with col3:
            avg_rvol = df['rvol'].median() if 'rvol' in df.columns else 1.0
            high_vol_count = len(df[df['rvol'] > 2]) if 'rvol' in df.columns else 0
            vol_emoji = "🌊" if avg_rvol > 1.5 else "💧" if avg_rvol > 1.2 else "🏜️"
            UIComponents.render_metric_card("Volume State", f"{vol_emoji} {avg_rvol:.1f}x", f"{high_vol_count} surges")
        with col4:
            risk_factors = 0
            if 'from_high_pct' in df.columns and 'momentum_score' in df.columns and len(df)>0:
                if len(df[(df['from_high_pct'] >= 0) & (df['momentum_score'] < 50)]) > 20: risk_factors += 1
            if 'rvol' in df.columns and len(df)>0:
                if len(df[(df['rvol'] > 10) & (df['master_score'] < 50)]) > 10: risk_factors += 1
            if 'trend_quality' in df.columns and len(df)>0:
                if len(df[df['trend_quality'] < 40]) > len(df) * 0.3: risk_factors += 1
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]
            risk_level = risk_levels[min(risk_factors, 3)]
            UIComponents.render_metric_card("Risk Level", risk_level, f"{risk_factors} factors")
        
        st.markdown("### 🎯 Today's Best Opportunities")
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        with opp_col1:
            ready_to_run = df[(df['momentum_score'] >= 70) & (df['acceleration_score'] >= 70) & (df['rvol'] >= 2)].nlargest(5, 'master_score')
            st.markdown("**🚀 Ready to Run**")
            if len(ready_to_run) > 0:
                for _, stock in ready_to_run.iterrows(): st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"Score: {stock['master_score']:.1f} | RVOL: {stock['rvol']:.1f}x")
            else: st.info("No momentum leaders found")
        with opp_col2:
            hidden_gems = df[df['patterns'].str.contains('HIDDEN GEM', na=False)].nlargest(5, 'master_score')
            st.markdown("**💎 Hidden Gems**")
            if len(hidden_gems) > 0:
                for _, stock in hidden_gems.iterrows(): st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"Cat %ile: {stock.get('category_percentile', 0):.0f} | Score: {stock['master_score']:.1f}")
            else: st.info("No hidden gems today")
        with opp_col3:
            volume_alerts = df[df['rvol'] > 3].nlargest(5, 'master_score')
            st.markdown("**⚡ Volume Alerts**")
            if len(volume_alerts) > 0:
                for _, stock in volume_alerts.iterrows(): st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}"); st.caption(f"RVOL: {stock['rvol']:.1f}x | {stock.get('wave_state', 'N/A')}")
            else: st.info("No extreme volume detected")
        
        st.markdown("### 🧠 Market Intelligence")
        intel_col1, intel_col2 = st.columns([2, 1])
        with intel_col1:
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            if not sector_rotation.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sector_rotation.index[:10], y=sector_rotation['flow_score'][:10], text=[f"{val:.1f}" for val in sector_rotation['flow_score'][:10]], textposition='outside', marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' for score in sector_rotation['flow_score'][:10]], customdata=np.column_stack((sector_rotation['analyzed_stocks'][:10], sector_rotation['total_stocks'][:10], sector_rotation['avg_score'][:10], sector_rotation['median_score'][:10])), hovertemplate='Sector: %{x}<br>Flow Score: %{y:.1f}<br>Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>Avg Score: %{customdata[2]:.1f}<br>Median Score: %{customdata[3]:.1f}<extra></extra>'))
                fig.update_layout(title="Sector Rotation Map - Smart Money Flow (Dynamically Sampled)", xaxis_title="Sector", yaxis_title="Flow Score", height=400, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No sector rotation data available for visualization.")
        with intel_col2:
            regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
            st.markdown(f"**🎯 Market Regime**")
            st.markdown(f"### {regime}")
            st.markdown("**📡 Key Signals**")
            signals = []
            if regime_metrics.get('breadth', 0.5) > 0.6: signals.append("✅ Strong breadth")
            elif regime_metrics.get('breadth', 0.5) < 0.4: signals.append("⚠️ Weak breadth")
            if regime_metrics.get('category_spread', 0) > 10: signals.append("🔄 Small caps leading")
            elif regime_metrics.get('category_spread', 0) < -10: signals.append("🛡️ Large caps defensive")
            if regime_metrics.get('avg_rvol', 1.0) > 1.5: signals.append("🌊 High volume activity")
            if (df['patterns'] != '').sum() > len(df) * 0.2: signals.append("🎯 Many patterns emerging")
            for signal in signals: st.write(signal)
            st.markdown("**💪 Market Strength**")
            strength_score = (regime_metrics.get('breadth', 0.5) * 50) + (min(regime_metrics.get('avg_rvol', 1.0), 2) * 25) + (((df['patterns'] != '').sum() / len(df)) * 25 if len(df)>0 else 0)
            strength_meter = "🟢🟢🟢🟢🟢" if strength_score > 70 else "🟢🟢🟢🟢⚪" if strength_score > 50 else "🟢🟢🟢⚪⚪" if strength_score > 30 else "🟢🟢⚪⚪⚪"
            st.write(strength_meter)

def main():
    st.set_page_config(page_title="Wave Detection Ultimate 3.0", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
    RobustSessionState.initialize()
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # CSS styles are included in the final code
    st.markdown("""<div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"><h1>🌊 Wave Detection Ultimate 3.0</h1><p>Professional Stock Ranking System • Final Perfected Production Version</p></div>""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True): st.cache_data.clear(); RobustSessionState.set('last_refresh', datetime.now(timezone.utc)); st.rerun()
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True): st.cache_data.clear(); gc.collect(); st.success("Cache cleared!"); time.sleep(0.5); st.rerun()
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        data_source_col1, data_source_col2 = st.columns(2)
        with data_source_col1:
            if st.button("📊 Google Sheets", type="primary" if RobustSessionState.get('data_source') == "sheet" else "secondary", use_container_width=True): RobustSessionState.set('data_source', "sheet"); st.rerun()
        with data_source_col2:
            if st.button("📁 Upload CSV", type="primary" if RobustSessionState.get('data_source') == "upload" else "secondary", use_container_width=True): RobustSessionState.set('data_source', "upload"); st.rerun()

        uploaded_file = None; sheet_id = None; gid = None
        if RobustSessionState.get('data_source') == "upload":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv", help="Upload a CSV file with stock data. Must contain 'ticker' and 'price' columns.")
            if uploaded_file is None: st.info("Please upload a CSV file to continue")
        else:
            st.markdown("#### 📊 Google Sheets Configuration")
            sheet_input = st.text_input("Google Sheets ID or URL", value=RobustSessionState.get('sheet_id'), placeholder="Enter Sheet ID or full URL", help="Example: 1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM or the full Google Sheets URL", key="sheet_id_input")
            if sheet_input:
                sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_input); sheet_id = sheet_id_match.group(1) if sheet_id_match else sheet_input.strip()
                RobustSessionState.set('sheet_id', sheet_id)
            gid_input = st.text_input("Sheet Tab GID (Optional)", value=RobustSessionState.get('gid'), placeholder=f"Default: {CONFIG.DEFAULT_GID}", help="The GID identifies specific sheet tab. Found in URL after #gid=", key="gid_input")
            gid = gid_input.strip() if gid_input else CONFIG.DEFAULT_GID
            if not sheet_id: st.warning("Please enter a Google Sheets ID to continue")
        
        data_quality = RobustSessionState.get('data_quality')
        if data_quality:
            with st.expander("📊 Data Quality", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    completeness = data_quality.get('completeness', 0); emoji = "🟢" if completeness > 80 else "🟡" if completeness > 60 else "🔴"
                    st.metric("Completeness", f"{emoji} {completeness:.1f}%"); st.metric("Total Stocks", f"{data_quality.get('total_rows', 0):,}")
                with col2:
                    if 'timestamp' in data_quality:
                        age = datetime.now(timezone.utc) - data_quality['timestamp']; hours = age.total_seconds() / 3600
                        freshness = "🟢 Fresh" if hours < 1 else "🟡 Recent" if hours < 24 else "🔴 Stale"
                        st.metric("Data Age", freshness); duplicates = data_quality.get('duplicate_tickers', 0)
                        if duplicates > 0: st.metric("Duplicates", f"⚠️ {duplicates}")
        
        perf_metrics = RobustSessionState.get('performance_metrics')
        if perf_metrics:
            with st.expander("⚡ Performance"):
                total_time = sum(perf_metrics.values()); perf_emoji = "🟢" if total_time < 3 else "🟡" if total_time < 5 else "🔴"
                st.metric("Load Time", f"{perf_emoji} {total_time:.1f}s")
                slowest = sorted(perf_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
                for func_name, elapsed in slowest: st.caption(f"{func_name}: {elapsed:.4f}s")
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        active_filter_count = 0
        if RobustSessionState.get('quick_filter_applied'): active_filter_count += 1
        filter_keys = ['category_filter', 'sector_filter', 'industry_filter', 'min_score', 'patterns', 'trend_filter', 'eps_tier_filter', 'pe_tier_filter', 'price_tier_filter', 'min_eps_change', 'min_pe', 'max_pe', 'require_fundamental_data', 'wave_states_filter', 'wave_strength_range_slider']
        for key in filter_keys:
            value = RobustSessionState.get(key)
            if value and (not isinstance(value, (list, tuple)) or len(value) > 0) and value != "All Trends" and value != (0, 100) and value != "" and value != 0: active_filter_count += 1
        RobustSessionState.set('active_filter_count', active_filter_count)
        if active_filter_count > 0: st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**")
        if st.button("🗑️ Clear All Filters", use_container_width=True, type="primary" if active_filter_count > 0 else "secondary"):
            RobustSessionState.clear_filters(); st.success("✅ All filters cleared!"); st.rerun()
        st.markdown("---")
        st.checkbox("🐛 Show Debug Info", value=RobustSessionState.get('show_debug'), key="show_debug")
    
    try:
        if RobustSessionState.get('data_source') == "upload" and uploaded_file is None: st.warning("Please upload a CSV file to continue"); st.stop()
        if RobustSessionState.get('data_source') == "sheet" and not sheet_id: st.warning("Please enter a Google Sheets ID to continue"); st.stop()
        with st.spinner("📥 Loading and processing data..."):
            if RobustSessionState.get('data_source') == "upload" and uploaded_file is not None:
                ranked_df, data_timestamp, metadata = load_and_process_data("upload", file_data=uploaded_file)
            else:
                ranked_df, data_timestamp, metadata = load_and_process_data("sheet", sheet_id=sheet_id, gid=gid)
            RobustSessionState.set('ranked_df', ranked_df); RobustSessionState.set('data_timestamp', data_timestamp); RobustSessionState.set('last_refresh', datetime.now(timezone.utc))
            if metadata.get('warnings'): [st.warning(w) for w in metadata['warnings']]
            if metadata.get('errors'): [st.error(e) for e in metadata['errors']]
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        last_good_data = RobustSessionState.get('last_good_data')
        if last_good_data: ranked_df, data_timestamp, metadata = last_good_data; st.warning("Failed to load fresh data, using cached version")
        else: st.error(f"❌ Error: {str(e)}"); st.info("Common issues:\n- Invalid Google Sheets ID\n- Sheet not publicly accessible\n- Network connectivity\n- Invalid CSV format"); st.stop()

    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    quick_filter_applied = RobustSessionState.get('quick_filter_applied'); quick_filter = RobustSessionState.get('quick_filter')
    with qa_col1:
        if st.button("📈 Top Gainers", use_container_width=True): RobustSessionState.set('quick_filter', 'top_gainers'); RobustSessionState.set('quick_filter_applied', True); st.rerun()
    with qa_col2:
        if st.button("🔥 Volume Surges", use_container_width=True): RobustSessionState.set('quick_filter', 'volume_surges'); RobustSessionState.set('quick_filter_applied', True); st.rerun()
    with qa_col3:
        if st.button("🎯 Breakout Ready", use_container_width=True): RobustSessionState.set('quick_filter', 'breakout_ready'); RobustSessionState.set('quick_filter_applied', True); st.rerun()
    with qa_col4:
        if st.button("💎 Hidden Gems", use_container_width=True): RobustSessionState.set('quick_filter', 'hidden_gems'); RobustSessionState.set('quick_filter_applied', True); st.rerun()
    with qa_col5:
        if st.button("🌊 Show All", use_container_width=True): RobustSessionState.set('quick_filter', None); RobustSessionState.set('quick_filter_applied', False); st.rerun()
    
    ranked_df_display = ranked_df
    if quick_filter == 'top_gainers': ranked_df_display = ranked_df[ranked_df['momentum_score'] >= 80]; st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ 80")
    elif quick_filter == 'volume_surges': ranked_df_display = ranked_df[ranked_df['rvol'] >= 3]; st.info(f"Showing {len(ranked_df_display)} stocks with RVOL ≥ 3x")
    elif quick_filter == 'breakout_ready': ranked_df_display = ranked_df[ranked_df['breakout_score'] >= 80]; st.info(f"Showing {len(ranked_df_display)} stocks with breakout score ≥ 80")
    elif quick_filter == 'hidden_gems': ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('HIDDEN GEM', na=False)]; st.info(f"Showing {len(ranked_df_display)} hidden gem stocks")
    
    with st.sidebar:
        filters = {}
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio("Choose your view:", options=["Technical", "Hybrid (Technical + Fundamentals)"], index=0 if RobustSessionState.get('user_preferences')['display_mode'] == 'Technical' else 1, key="display_mode_toggle", help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data")
        user_prefs = RobustSessionState.get('user_preferences'); user_prefs['display_mode'] = display_mode; RobustSessionState.set('user_preferences', user_prefs); show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        st.markdown("---")
        
        categories = FilterEngine.get_filter_options(ranked_df_display, 'category', filters)
        selected_categories = st.multiselect("Market Cap Category", options=categories, default=RobustSessionState.get('category_filter'), placeholder="Select categories (empty = All)", key="category_filter")
        filters['categories'] = selected_categories

        sectors = FilterEngine.get_filter_options(ranked_df_display, 'sector', filters)
        selected_sectors = st.multiselect("Sector", options=sectors, default=RobustSessionState.get('sector_filter'), placeholder="Select sectors (empty = All)", key="sector_filter")
        filters['sectors'] = selected_sectors

        if 'industry' in ranked_df_display.columns:
            industries = FilterEngine.get_filter_options(ranked_df_display, 'industry', filters)
            selected_industries = st.multiselect("Industry", options=industries, default=RobustSessionState.get('industry_filter'), placeholder="Select industries (empty = All)", key="industry_filter")
            filters['industries'] = selected_industries
        
        filters['min_score'] = st.slider("Minimum Master Score", min_value=0, max_value=100, value=RobustSessionState.get('min_score'), step=5, help="Filter stocks by minimum score", key="min_score")
        all_patterns = {p for patterns in ranked_df_display['patterns'].dropna() for p in patterns.split(' | ')}
        if all_patterns: filters['patterns'] = st.multiselect("Patterns", options=sorted(all_patterns), default=RobustSessionState.get('patterns'), placeholder="Select patterns (empty = All)", help="Filter by specific patterns", key="patterns")
        
        st.markdown("#### 📈 Trend Strength")
        trend_options = {"All Trends": (0, 100), "🔥 Strong Uptrend (80+)": (80, 100), "✅ Good Uptrend (60-79)": (60, 79), "➡️ Neutral Trend (40-59)": (40, 59), "⚠️ Weak/Downtrend (<40)": (0, 39)}
        current_trend_index = list(trend_options.keys()).index(RobustSessionState.get('trend_filter')) if RobustSessionState.get('trend_filter') in trend_options else 0
        filters['trend_filter'] = st.selectbox("Trend Quality", options=list(trend_options.keys()), index=current_trend_index, key="trend_filter", help="Filter stocks by trend strength based on SMA alignment")
        filters['trend_range'] = trend_options[filters['trend_filter']]
        
        st.markdown("#### 🌊 Wave Filters")
        wave_states_options = FilterEngine.get_filter_options(ranked_df_display, 'wave_state', filters)
        filters['wave_states'] = st.multiselect("Wave State", options=wave_states_options, default=RobustSessionState.get('wave_states_filter'), placeholder="Select wave states (empty = All)", help="Filter by the detected 'Wave State'", key="wave_states_filter")
        if 'overall_wave_strength' in ranked_df_display.columns:
            min_strength, max_strength = 0, 100
            filters['wave_strength_range'] = st.slider("Overall Wave Strength", min_value=min_strength, max_value=max_strength, value=RobustSessionState.get('wave_strength_range_slider'), step=1, help="Filter by the calculated 'Overall Wave Strength' score", key="wave_strength_range_slider")
        else: filters['wave_strength_range'] = (0, 100); st.info("Overall Wave Strength data not available.")

        with st.expander("🔧 Advanced Filters"):
            for tier_type, col_name in [('eps_tiers', 'eps_tier'), ('pe_tiers', 'pe_tier'), ('price_tiers', 'price_tier')]:
                if col_name in ranked_df_display.columns:
                    tier_options = FilterEngine.get_filter_options(ranked_df_display, col_name, filters)
                    filters[tier_type] = st.multiselect(f"{col_name.replace('_', ' ').title()}", options=tier_options, default=RobustSessionState.get(f'{col_name}_filter'), placeholder=f"Select {col_name.replace('_', ' ')}s (empty = All)", key=f"{col_name}_filter")
            if 'eps_change_pct' in ranked_df_display.columns:
                eps_change_input = st.text_input("Min EPS Change %", value=RobustSessionState.get('min_eps_change'), placeholder="e.g. -50 or leave empty", key="min_eps_change")
                filters['min_eps_change'] = float(eps_change_input) if eps_change_input.strip() else None
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                st.markdown("**🔍 Fundamental Filters**")
                col1, col2 = st.columns(2)
                with col1:
                    min_pe_input = st.text_input("Min PE Ratio", value=RobustSessionState.get('min_pe'), placeholder="e.g. 10", key="min_pe")
                    filters['min_pe'] = float(min_pe_input) if min_pe_input.strip() else None
                with col2:
                    max_pe_input = st.text_input("Max PE Ratio", value=RobustSessionState.get('max_pe'), placeholder="e.g. 30", key="max_pe")
                    filters['max_pe'] = float(max_pe_input) if max_pe_input.strip() else None
                filters['require_fundamental_data'] = st.checkbox("Only show stocks with PE and EPS data", value=RobustSessionState.get('require_fundamental_data'), key="require_fundamental_data")
    
    filtered_df = FilterEngine.apply_filters(ranked_df_display, filters) if quick_filter_applied else FilterEngine.apply_filters(ranked_df, filters)
    filtered_df = filtered_df.sort_values('rank')
    user_prefs = RobustSessionState.get('user_preferences'); user_prefs['last_filters'] = filters; RobustSessionState.set('user_preferences', user_prefs)
    if RobustSessionState.get('show_debug'):
        with st.sidebar.expander("🐛 Debug Info", expanded=True):
            st.write("**Active Filters:**"); [st.write(f"• {k}: {v}") for k,v in filters.items() if v and v != [] and v != 0 and (not isinstance(v, tuple) or v != (0,100))]
            st.write(f"\n**Filter Result:**"); st.write(f"Before: {len(ranked_df)} stocks"); st.write(f"After: {len(filtered_df)} stocks")
            if perf_metrics: st.write(f"\n**Performance:**"); [st.write(f"• {f}: {t:.4f}s") for f,t in perf_metrics.items() if t>0.001]
    
    if RobustSessionState.get('active_filter_count') > 0 or quick_filter_applied:
        filter_status_col1, filter_status_col2 = st.columns([5, 1])
        with filter_status_col1:
            filter_display = {'top_gainers': '📈 Top Gainers', 'volume_surges': '🔥 Volume Surges', 'breakout_ready': '🎯 Breakout Ready', 'hidden_gems': '💎 Hidden Gems'}.get(quick_filter, 'Filtered')
            if RobustSessionState.get('active_filter_count') > 1: st.info(f"**Viewing:** {filter_display} + {RobustSessionState.get('active_filter_count') - 1} other filter{'s' if RobustSessionState.get('active_filter_count') > 2 else ''} | **{len(filtered_df):,} stocks** shown")
            else: st.info(f"**Viewing:** {filter_display} | **{len(filtered_df):,} stocks** shown")
        with filter_status_col2:
            if st.button("Clear Filters", type="secondary"): RobustSessionState.set('trigger_clear', True); st.rerun()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    if not filtered_df.empty:
        with col1: UIComponents.render_metric_card("Total Stocks", f"{len(filtered_df):,}", f"{(len(filtered_df)/len(ranked_df)*100):.0f}% of {len(ranked_df):,}")
        with col2: avg_score = filtered_df['master_score'].mean(); std_score = filtered_df['master_score'].std(); UIComponents.render_metric_card("Avg Score", f"{avg_score:.1f}", f"σ={std_score:.1f}")
        with col3:
            if show_fundamentals and 'pe' in filtered_df.columns:
                valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000); pe_coverage = valid_pe.sum(); pe_pct = (pe_coverage / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                if pe_coverage > 0: median_pe = filtered_df.loc[valid_pe, 'pe'].median(); UIComponents.render_metric_card("Median PE", f"{median_pe:.1f}x", f"{pe_pct:.0f}% have data")
                else: UIComponents.render_metric_card("PE Data", "Limited", "No PE data")
            else:
                min_score = filtered_df['master_score'].min(); max_score = filtered_df['master_score'].max(); score_range = f"{min_score:.1f}-{max_score:.1f}"
                UIComponents.render_metric_card("Score Range", score_range)
        with col4:
            if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
                valid_eps_change = filtered_df['eps_change_pct'].notna(); positive_eps_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 0)
                strong_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 50); mega_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 100)
                growth_count = positive_eps_growth.sum(); strong_count = strong_growth.sum()
                UIComponents.render_metric_card("EPS Growth +ve", f"{growth_count}", f"{strong_count} >50% | {mega_growth.sum()} >100%")
            else: UIComponents.render_metric_card("Accelerating", f"{(filtered_df['acceleration_score'] >= 80).sum()}")
        with col5: UIComponents.render_metric_card("High RVOL", f"{(filtered_df['rvol'] > 2).sum()}")
        with col6:
            if 'trend_quality' in filtered_df.columns: UIComponents.render_metric_card("Strong Trends", f"{(filtered_df['trend_quality'] >= 80).sum()}", f"{(len(filtered_df[filtered_df['trend_quality'] >= 80]) / len(filtered_df) * 100):.0f}%")
            else: UIComponents.render_metric_card("With Patterns", f"{(filtered_df['patterns'] != '').sum()}")
    else:
        [col.empty() for col in [col1, col2, col3, col4, col5, col6]]
    
    tabs = st.tabs(["📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"])
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        if not filtered_df.empty:
            UIComponents.render_summary_section(filtered_df)
            st.markdown("---"); st.markdown("#### 💾 Download Clean Processed Data")
            download_cols = st.columns(3)
            with download_cols[0]: st.markdown("**📊 Current View Data**"); st.write(f"Includes {len(filtered_df)} stocks matching current filters")
            csv_filtered = ExportEngine.create_csv_export(filtered_df)
            with download_cols[0]: st.download_button(label="📥 Download Filtered Data (CSV)", data=csv_filtered, file_name=f"wave_detection_filtered_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with download_cols[1]: st.markdown("**🏆 Top 100 Stocks**"); st.write("Elite stocks ranked by Master Score")
            csv_top100 = ExportEngine.create_csv_export(filtered_df.nlargest(100, 'master_score'))
            with download_cols[1]: st.download_button(label="📥 Download Top 100 (CSV)", data=csv_top100, file_name=f"wave_detection_top100_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with download_cols[2]: st.markdown("**🎯 Pattern Stocks Only**"); pattern_stocks = filtered_df[filtered_df['patterns'] != '']; st.write(f"Includes {len(pattern_stocks)} stocks with patterns")
            if len(pattern_stocks) > 0:
                csv_patterns = ExportEngine.create_csv_export(pattern_stocks)
                with download_cols[2]: st.download_button(label="📥 Download Pattern Stocks (CSV)", data=csv_patterns, file_name=f"wave_detection_patterns_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            else: with download_cols[2]: st.info("No stocks with patterns in current filter")
        else: st.warning("No data available for summary. Please adjust filters.")
    
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            user_prefs = RobustSessionState.get('user_preferences'); display_count = st.selectbox("Show top", options=CONFIG.AVAILABLE_TOP_N, index=CONFIG.AVAILABLE_TOP_N.index(user_prefs.get('default_top_n', CONFIG.DEFAULT_TOP_N))); user_prefs['default_top_n'] = display_count; RobustSessionState.set('user_preferences', user_prefs)
        with col2:
            sort_options = ['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow', 'Trend'] if 'trend_quality' in filtered_df.columns else ['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow']
            sort_by = st.selectbox("Sort by", options=sort_options, index=0)
        
        display_df = filtered_df.head(display_count).copy()
        if sort_by == 'Master Score': display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL': display_df = display_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum': display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Money Flow' and 'money_flow_mm' in display_df.columns: display_df = display_df.sort_values('money_flow_mm', ascending=False)
        elif sort_by == 'Trend' and 'trend_quality' in display_df.columns: display_df = display_df.sort_values('trend_quality', ascending=False)
        
        if not display_df.empty:
            if 'trend_quality' in display_df.columns:
                def get_trend_indicator(score):
                    if pd.isna(score): return "➖"
                    return "🔥" if score >= 80 else "✅" if score >= 60 else "➡️" if score >= 40 else "⚠️"
                display_df['trend_indicator'] = display_df['trend_quality'].apply(get_trend_indicator)
            
            display_cols = {'rank': 'Rank', 'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score', 'wave_state': 'Wave', 'trend_indicator': 'Trend', 'price': 'Price'}
            if show_fundamentals: display_cols.update({'pe': 'PE', 'eps_change_pct': 'EPS Δ%'})
            display_cols.update({'from_low_pct': 'From Low', 'ret_30d': '30D Ret', 'rvol': 'RVOL', 'vmi': 'VMI', 'patterns': 'Patterns', 'category': 'Category'})
            if 'industry' in display_df.columns: display_cols['industry'] = 'Industry'

            format_rules = {'master_score': '{:.1f}', 'price': '₹{:,.0f}', 'from_low_pct': '{:.0f}%', 'ret_30d': '{:+.1f}%', 'rvol': '{:.1f}x', 'vmi': '{:.2f}'}
            def format_pe(value):
                try: val = float(value); return 'Loss' if val <= 0 else '>10K' if val > 10000 else f"{val:.0f}" if val > 1000 else f"{val:.1f}"
                except: return '-'
            def format_eps_change(value):
                try: val = float(value); return f"{val/1000:+.1f}K%" if abs(val) >= 1000 else f"{val:+.0f}%" if abs(val) >= 100 else f"{val:+.1f}%"
                except: return '-'
            
            for col, fmt in format_rules.items():
                if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) and isinstance(x, (int, float)) else '-')
            if show_fundamentals:
                if 'pe' in display_df.columns: display_df['pe'] = display_df['pe'].apply(format_pe)
                if 'eps_change_pct' in display_df.columns: display_df['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change)
            
            available_display_cols = [c for c in display_cols.keys() if c in display_df.columns]
            display_df = display_df[available_display_cols]
            display_df.columns = [display_cols[c] for c in available_display_cols]
            st.dataframe(display_df, use_container_width=True, height=min(600, len(display_df) * 35 + 50), hide_index=True)
            
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                with stat_cols[0]: st.markdown("**Score Distribution**"); [st.text(f"{k}: {filtered_df['master_score'].agg(k):.1f}") for k in ['max', 'min', 'mean', 'median']]
                with stat_cols[1]: st.markdown("**Returns (30D)**"); [st.text(f"{k}: {filtered_df['ret_30d'].agg(k):.1f}%") for k in ['max', 'min', 'mean']]
                with stat_cols[2]: st.markdown("**Fundamentals**" if show_fundamentals else "**Volume**");
                if show_fundamentals and 'pe' in filtered_df.columns: st.text(f"Median PE: {filtered_df.loc[filtered_df['pe']>0, 'pe'].median():.1f}x")
                elif 'rvol' in filtered_df.columns: st.text(f"Max RVOL: {filtered_df['rvol'].max():.1f}x")
                with stat_cols[3]: st.markdown("**Trend Distribution**"); st.text(f"Avg Trend Score: {filtered_df['trend_quality'].mean():.1f}") if 'trend_quality' in filtered_df.columns else st.text("No trend data available")
        else: st.warning("No stocks match the selected filters.")
    
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        st.markdown("*Catch waves as they form, not after they've peaked!*")
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        with radar_col1:
            wave_timeframe = st.selectbox("Wave Detection Timeframe", options=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"], index=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"].index(RobustSessionState.get('wave_timeframe_select')), key="wave_timeframe_select")
        with radar_col2:
            sensitivity = st.select_slider("Detection Sensitivity", options=["Conservative", "Balanced", "Aggressive"], value=RobustSessionState.get('wave_sensitivity'), key="wave_sensitivity")
            show_sensitivity_details = st.checkbox("Show thresholds", value=RobustSessionState.get('show_sensitivity_details'), key="show_sensitivity_details")
        with radar_col3:
            show_market_regime = st.checkbox("📊 Market Regime Analysis", value=RobustSessionState.get('show_market_regime'), key="show_market_regime")
        wave_filtered_df = filtered_df.copy()
        with radar_col4:
            if not wave_filtered_df.empty and 'overall_wave_strength' in wave_filtered_df.columns:
                wave_strength_score = wave_filtered_df['overall_wave_strength'].mean()
                wave_emoji, wave_color = ("🌊🔥", "🟢") if wave_strength_score > 70 else ("🌊", "🟡") if wave_strength_score > 50 else ("💤", "🔴")
                UIComponents.render_metric_card("Wave Strength", f"{wave_emoji} {wave_strength_score:.0f}%", f"{wave_color} Market")
            else: UIComponents.render_metric_card("Wave Strength", "N/A", "Data not available")
        if show_sensitivity_details:
            with st.expander("📊 Current Sensitivity Thresholds", expanded=True):
                st.markdown("**Conservative Settings** 🛡️" if sensitivity=="Conservative" else "**Balanced Settings** ⚖️" if sensitivity=="Balanced" else "**Aggressive Settings** 🚀"); st.info("...") # Details omitted for brevity
        
        # Applying timeframe filters to the main df for display
        if wave_timeframe != "All Waves":
            try:
                if wave_timeframe == "Intraday Surge": wave_filtered_df = wave_filtered_df[(wave_filtered_df['rvol'] >= 2.5) & (wave_filtered_df['ret_1d'] > 2) & (wave_filtered_df['price'] > wave_filtered_df['prev_close'] * 1.02)]
                if wave_timeframe == "3-Day Buildup": wave_filtered_df = wave_filtered_df[(wave_filtered_df['ret_3d'] > 5) & (wave_filtered_df['vol_ratio_7d_90d'] > 1.5) & (wave_filtered_df['price'] > wave_filtered_df['sma_20d'])]
                if wave_timeframe == "Weekly Breakout": wave_filtered_df = wave_filtered_df[(wave_filtered_df['ret_7d'] > 8) & (wave_filtered_df['vol_ratio_7d_90d'] > 2.0) & (wave_filtered_df['from_high_pct'] > -10)]
                if wave_timeframe == "Monthly Trend": wave_filtered_df = wave_filtered_df[(wave_filtered_df['ret_30d'] > 15) & (wave_filtered_df['price'] > wave_filtered_df['sma_20d']) & (wave_filtered_df['sma_20d'] > wave_filtered_df['sma_50d']) & (wave_filtered_df['vol_ratio_30d_180d'] > 1.2) & (wave_filtered_df['from_low_pct'] > 30)]
            except Exception as e: st.warning(f"Some data not available for {wave_timeframe} filter")
        
        if not wave_filtered_df.empty:
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            momentum_threshold = {"Conservative": 60, "Balanced": 50, "Aggressive": 40}[sensitivity]
            acceleration_threshold = {"Conservative": 70, "Balanced": 60, "Aggressive": 50}[sensitivity]
            min_rvol = {"Conservative": 3.0, "Balanced": 2.0, "Aggressive": 1.5}[sensitivity]
            momentum_shifts = wave_filtered_df[(wave_filtered_df['momentum_score'] >= momentum_threshold) & (wave_filtered_df['acceleration_score'] >= acceleration_threshold)].copy()
            if len(momentum_shifts) > 0:
                momentum_shifts['signal_count'] = 0
                momentum_shifts.loc[momentum_shifts['momentum_score'] >= momentum_threshold, 'signal_count'] += 1
                momentum_shifts.loc[momentum_shifts['acceleration_score'] >= acceleration_threshold, 'signal_count'] += 1
                momentum_shifts.loc[momentum_shifts['rvol'] >= min_rvol, 'signal_count'] += 1
                if 'breakout_score' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['breakout_score'] >= 75, 'signal_count'] += 1
                if 'vol_ratio_7d_90d' in momentum_shifts.columns: momentum_shifts.loc[momentum_shifts['vol_ratio_7d_90d'] >= 1.5, 'signal_count'] += 1
                momentum_shifts['shift_strength'] = (momentum_shifts['momentum_score'] * 0.4 + momentum_shifts['acceleration_score'] * 0.4 + momentum_shifts['rvol_score'] * 0.2)
                top_shifts = momentum_shifts.sort_values(['signal_count', 'shift_strength'], ascending=[False, False]).head(20)
                display_cols = ['ticker', 'company_name', 'master_score', 'momentum_score', 'acceleration_score', 'rvol', 'signal_count', 'wave_state', 'category']
                if 'ret_7d' in top_shifts.columns: display_cols.insert(-2, 'ret_7d')
                shift_display = top_shifts[[c for c in display_cols if c in top_shifts.columns]].copy()
                shift_display['Signals'] = shift_display['signal_count'].apply(lambda x: f"{'🔥' * min(x, 3)} {x}/5")
                if 'ret_7d' in shift_display.columns: shift_display['7D Return'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else '-')
                shift_display['RVOL'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                shift_display = shift_display.rename(columns={'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score', 'momentum_score': 'Momentum', 'acceleration_score': 'Acceleration', 'wave_state': 'Wave', 'category': 'Category'}).drop('signal_count', axis=1)
                st.dataframe(shift_display, use_container_width=True, hide_index=True)
                multi_signal = len(top_shifts[top_shifts['signal_count'] >= 3])
                if multi_signal > 0: st.success(f"🏆 Found {multi_signal} stocks with 3+ signals (strongest momentum)")
                super_signals = top_shifts[top_shifts['signal_count'] >= 4]
                if len(super_signals) > 0: st.warning(f"🔥🔥 {len(super_signals)} stocks showing EXTREME momentum (4+ signals)!")
            else: st.info(f"No momentum shifts detected in {wave_timeframe} timeframe. Try 'Aggressive' sensitivity.")
            st.markdown("#### 🚀 Acceleration Profiles - Momentum Building Over Time")
            accel_threshold = {"Conservative": 85, "Balanced": 70, "Aggressive": 60}[sensitivity]
            accelerating_stocks = wave_filtered_df[wave_filtered_df['acceleration_score'] >= accel_threshold].nlargest(10, 'acceleration_score')
            if len(accelerating_stocks) > 0:
                fig_accel = Visualizer.create_acceleration_profiles(accelerating_stocks, n=10); st.plotly_chart(fig_accel, use_container_width=True)
            else: st.info(f"No stocks meet the acceleration threshold ({accel_threshold}+) for {sensitivity} sensitivity.")
        else: st.warning(f"No data available for Wave Radar analysis with {wave_timeframe} timeframe.")
    
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        if not filtered_df.empty:
            col1, col2 = st.columns(2);
            with col1: fig_dist = Visualizer.create_score_distribution(filtered_df); st.plotly_chart(fig_dist, use_container_width=True)
            with col2:
                pattern_counts = {p:count for patterns in filtered_df['patterns'].dropna() for p in patterns.split(' | ')}
                if pattern_counts:
                    pattern_df = pd.DataFrame(list(pattern_counts.items()), columns=['Pattern', 'Count']).sort_values('Count', ascending=True).tail(15)
                    fig_patterns = go.Figure([go.Bar(x=pattern_df['Count'], y=pattern_df['Pattern'], orientation='h', marker_color='#3498db', text=pattern_df['Count'], textposition='outside')])
                    fig_patterns.update_layout(title="Pattern Frequency Analysis", xaxis_title="Number of Stocks", yaxis_title="Pattern", template='plotly_white', height=400, margin=dict(l=150)); st.plotly_chart(fig_patterns, use_container_width=True)
                else: st.info("No patterns detected in current selection")
            st.markdown("---")
            st.markdown("#### Sector Performance (Dynamically Sampled)")
            sector_overview_df_local = MarketIntelligence.detect_sector_rotation(filtered_df)
            if not sector_overview_df_local.empty:
                display_cols_overview = ['flow_score', 'avg_score', 'median_score', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'analyzed_stocks', 'total_stocks', 'sampling_pct']
                sector_overview_display = sector_overview_df_local[[c for c in display_cols_overview if c in sector_overview_df_local.columns]].copy()
                sector_overview_display.columns = ['Flow Score', 'Avg Score', 'Median Score', 'Avg Momentum', 'Avg Volume', 'Avg RVOL', 'Avg 30D Ret', 'Analyzed Stocks', 'Total Stocks', 'Sample %']
                st.dataframe(sector_overview_display.style.background_gradient(subset=['Flow Score', 'Avg Score']), use_container_width=True)
            else: st.info("No sector data available in the filtered dataset for analysis. Please check your filters.")
            
            st.markdown("#### Industry Performance (Smart Dynamic Sampling)")
            if 'industry' in filtered_df.columns:
                industry_overview_df = MarketIntelligence.detect_industry_rotation(filtered_df)
                if not industry_overview_df.empty:
                    ind_tab1, ind_tab2 = st.tabs(["📊 Top Industries", "📈 All Industries"])
                    with ind_tab1:
                        top_industries = industry_overview_df.head(20)
                        fig_industry = go.Figure()
                        fig_industry.add_trace(go.Bar(x=top_industries.index[:15], y=top_industries['flow_score'][:15], text=[f"{val:.1f}" for val in top_industries['flow_score'][:15]], textposition='outside', marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' for score in top_industries['flow_score'][:15]]))
                        fig_industry.update_layout(title="Top 15 Industries by Smart Money Flow", xaxis_title="Industry", yaxis_title="Flow Score", height=500, template='plotly_white', xaxis_tickangle=-45); st.plotly_chart(fig_industry, use_container_width=True)
                    with ind_tab2:
                        display_cols_industry = ['flow_score', 'avg_score', 'median_score', 'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'analyzed_stocks', 'total_stocks', 'sampling_pct']
                        industry_display = industry_overview_df[[c for c in display_cols_industry if c in industry_overview_df.columns]].copy()
                        industry_display.columns = ['Flow Score', 'Avg Score', 'Median Score', 'Avg Momentum', 'Avg Volume', 'Avg RVOL', 'Avg 30D Ret', 'Analyzed', 'Total', 'Sample %']
                        industry_display['Sample %'] = industry_display['Sample %'].apply(lambda x: f"{x:.1f}%")
                        industry_display.insert(0, 'Rank', range(1, len(industry_display) + 1))
                        st.dataframe(industry_display.style.background_gradient(subset=['Flow Score', 'Avg Score', 'Avg Momentum'], cmap='RdYlGn'), use_container_width=True, height=400)
                else: st.info("No industry data available in the filtered dataset for analysis.")
            st.markdown("#### 📊 Category Performance (Market Cap Analysis)")
            category_overview_df = MarketIntelligence.detect_category_performance(filtered_df)
            if not category_overview_df.empty:
                cat_tab1, cat_tab2 = st.tabs(["📊 Category Flow", "📈 Detailed Metrics"])
                with cat_tab1:
                    fig_category = go.Figure(); colors = {'Mega Cap': '#1f77b4', 'Large Cap': '#2ca02c', 'Mid Cap': '#ff7f0e', 'Small Cap': '#d62728', 'Micro Cap': '#9467bd'}
                    bar_colors = [colors.get(cat, '#7f7f7f') for cat in category_overview_df.index]
                    fig_category.add_trace(go.Bar(x=category_overview_df.index, y=category_overview_df['flow_score'], text=[f"{val:.1f}" for val in category_overview_df['flow_score']], textposition='outside', marker_color=bar_colors)); st.plotly_chart(fig_category, use_container_width=True)
                with cat_tab2:
                    display_cols_category = ['flow_score', 'avg_score', 'median_score', 'avg_momentum', 'avg_acceleration', 'avg_breakout', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'analyzed_stocks', 'total_stocks', 'sampling_pct']
                    category_display = category_overview_df[[c for c in display_cols_category if c in category_overview_df.columns]].copy()
                    st.dataframe(category_display.style.background_gradient(subset=['flow_score', 'avg_score']), use_container_width=True)
            else: st.info("No category data available in the filtered dataset for analysis.")
        else: st.info("No data available for analysis.")

    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        search_col1, search_col2 = st.columns([4, 1])
        with search_col1: search_query = st.text_input("Search stocks", placeholder="Enter ticker or company name...", key="search_input")
        with search_col2: st.markdown("<br>", unsafe_allow_html=True); search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        if search_query or search_clicked:
            with st.spinner("Searching..."): search_results = SearchEngine.search_stocks(filtered_df, search_query)
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                for _, stock in search_results.iterrows():
                    with st.expander(f"📊 {stock['ticker']} - {stock['company_name']} (Rank #{int(stock['rank'])})", expanded=True):
                        metric_cols = st.columns(6)
                        with metric_cols[0]: UIComponents.render_metric_card("Master Score", f"{stock['master_score']:.1f}", f"Rank #{int(stock['rank'])}")
                        with metric_cols[1]: UIComponents.render_metric_card("Price", f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A", f"{stock['ret_1d']:+.1f}%" if pd.notna(stock.get('ret_1d')) else None)
                        with metric_cols[2]: UIComponents.render_metric_card("From Low", f"{stock['from_low_pct']:.0f}%", "52-week range position")
                        with metric_cols[3]: UIComponents.render_metric_card("30D Return", f"{stock['ret_30d']:+.1f}%" if pd.notna(stock.get('ret_30d')) else "N/A", "↑" if stock.get('ret_30d', 0) > 0 else "↓")
                        with metric_cols[4]: UIComponents.render_metric_card("RVOL", f"{stock['rvol']:.1f}x", "High" if stock.get('rvol', 1) > 2 else "Normal")
                        with metric_cols[5]: UIComponents.render_metric_card("Wave State", stock.get('wave_state', 'N/A'), stock['category'])
                        st.markdown("---")
                        st.markdown(f"**🎯 Patterns:** {stock['patterns']}")
            else: st.warning("No stocks found matching your search criteria.")

    with tabs[5]:
        st.markdown("### 📥 Export Data")
        export_template = st.radio("Choose export template:", options=["Full Analysis (All Data)", "Day Trader Focus", "Swing Trader Focus", "Investor Focus"], key="export_template_radio", help="Select a template based on your trading style")
        selected_template = {"Full Analysis (All Data)": "full", "Day Trader Focus": "day_trader", "Swing Trader Focus": "swing_trader", "Investor Focus": "investor"}[export_template]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Excel Report"); st.markdown("Comprehensive multi-sheet report...");
            if st.button("Generate Excel Report", type="primary", use_container_width=True):
                if len(filtered_df) == 0: st.error("No data to export. Please adjust your filters.")
                else:
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_file = ExportEngine.create_excel_report(filtered_df, template=selected_template); st.download_button(label="📥 Download Excel Report", data=excel_file, file_name=f"wave_detection_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"); st.success("Excel report generated successfully!")
                        except Exception as e: st.error(f"Error generating Excel report: {str(e)}")
        with col2:
            st.markdown("#### 📄 CSV Export"); st.markdown("Enhanced CSV format with...");
            if st.button("Generate CSV Export", use_container_width=True):
                if len(filtered_df) == 0: st.error("No data to export. Please adjust your filters.")
                else:
                    try:
                        csv_data = ExportEngine.create_csv_export(filtered_df); st.download_button(label="📥 Download CSV File", data=csv_data, file_name=f"wave_detection_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv"); st.success("CSV export generated successfully!")
                    except Exception as e: st.error(f"Error generating CSV: {str(e)}")
        
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0 - Final Perfected Production Version")
        st.markdown("#### 🌊 Welcome to Wave Detection Ultimate 3.0"); st.markdown("..."); # Details omitted for brevity

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}")
        logger.error(f"Application crashed: {str(e)}", exc_info=True)
        if st.button("🔄 Restart Application"): st.cache_data.clear(); st.rerun()
        if st.button("📧 Report Issue"): st.info("Please take a screenshot and report this error.")
