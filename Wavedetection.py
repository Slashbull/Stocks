"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION VERSION
=====================================================
Professional Stock Ranking System with Advanced Analytics
All features complete, fully optimized, production-ready
Handles 2000+ stocks with 41 data columns

Version: 3.0-FINAL-PERMANENT
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
from functools import lru_cache, wraps
import time
from io import BytesIO
import warnings
import gc
from decimal import Decimal, ROUND_HALF_UP

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
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights and thresholds"""
    
    # Data source - HARDCODED for production (DO NOT CHANGE)
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM/edit?usp=sharing"
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings optimized for Streamlit Community Cloud
    CACHE_TTL: int = 3600  # 1 hour
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
        'vol_ratio_90d_180d'
    ])
    
    # All percentage columns for consistent handling
    # Data is stored as percentage values directly (e.g., -56.61 means -56.61%)
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
        'rvol': (0.01, 100.0),  # Cap at 100x
        'pe': (-10000, 10000),
        'returns': (-99.99, 9999.99),  # Percentage bounds
        'volume': (0, 1e12)
    })
    
    # Performance thresholds
    PERFORMANCE_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'data_processing': 2.0,  # seconds
        'filtering': 0.2,
        'pattern_detection': 0.5,
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
    
    # Alert thresholds
    ALERT_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'extreme_rvol': 5.0,
        'sector_rotation_std': 15.0,
        'regime_shift_spread': 10.0,
        'pattern_emergence': 10,
        'unusual_volume': 3.0
    })
    
    # Mobile breakpoints
    MOBILE_BREAKPOINT: int = 768
    TABLET_BREAKPOINT: int = 1024
    
    # Performance targets
    MAX_LOAD_TIME: float = 3.0
    MAX_FILTER_TIME: float = 0.3

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
                    
                    # Log if exceeds target
                    if target_time and elapsed > target_time:
                        logger.warning(f"{func.__name__} took {elapsed:.2f}s (target: {target_time}s)")
                    elif elapsed > 1.0:
                        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                    
                    # Store timing
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
    """Comprehensive data validation and sanitization"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> Tuple[bool, str]:
        """Validate dataframe structure and data quality"""
        if df is None:
            return False, f"{context}: DataFrame is None"
        
        if df.empty:
            return False, f"{context}: DataFrame is empty"
        
        # Check critical columns
        missing_critical = [col for col in CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            return False, f"{context}: Missing critical columns: {missing_critical}"
        
        # Check for duplicate tickers
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"{context}: Found {duplicates} duplicate tickers")
        
        # Calculate data quality metrics
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        completeness = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        
        if completeness < 50:
            logger.warning(f"{context}: Low data completeness ({completeness:.1f}%)")
        
        # Store quality metrics
        if 'data_quality' not in st.session_state:
            st.session_state.data_quality = {}
        
        st.session_state.data_quality.update({
            'completeness': completeness,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_tickers': duplicates,
            'context': context,
            'timestamp': datetime.now(timezone.utc)
        })
        
        logger.info(f"{context}: Validated {len(df)} rows, {len(df.columns)} columns, {completeness:.1f}% complete")
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """Clean and convert numeric values with bounds checking"""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        try:
            # Convert to string for cleaning
            cleaned = str(value).strip()
            
            # Check for invalid values
            if cleaned.upper() in ['', '-', 'N/A', 'NA', 'NAN', 'NONE', '#VALUE!', '#ERROR!', '#DIV/0!', 'INF', '-INF']:
                return np.nan
            
            # Remove common symbols and spaces
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            
            # Convert to float
            result = float(cleaned)
            
            # Apply bounds if specified
            if bounds:
                min_val, max_val = bounds
                if result < min_val or result > max_val:
                    logger.debug(f"Value {result} outside bounds [{min_val}, {max_val}]")
                    result = np.clip(result, min_val, max_val)
            
            # Check for unreasonable values
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
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    @staticmethod
    def validate_critical_calculations(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate all calculations completed successfully"""
        errors = []
        
        # Check for NaN in critical columns
        critical_cols = ['master_score', 'rank', 'percentile']
        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    errors.append(f"{col} has {nan_count} NaN values")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                errors.append(f"{col} has {inf_count} infinite values")
        
        # Check rank uniqueness
        if 'rank' in df.columns:
            unique_ranks = df['rank'].nunique()
            if unique_ranks < len(df):
                errors.append(f"Non-unique ranks: {unique_ranks} unique out of {len(df)}")
        
        return len(errors) == 0, errors

# ============================================
# SMART CACHING WITH VERSIONING
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL, persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, 
                         sheet_url: str = None, gid: str = None,
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
    
    # Progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Load data based on source
        status_placeholder.text('📥 Loading data...')
        progress_placeholder.progress(10)
        
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            df = pd.read_csv(file_data, low_memory=False)
            metadata['source'] = "User Upload"
        else:
            # Use defaults if not provided
            if not sheet_url:
                sheet_url = CONFIG.DEFAULT_SHEET_URL
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            # Construct CSV URL
            base_url = sheet_url.split('/edit')[0]
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from Google Sheets")
            
            try:
                df = pd.read_csv(csv_url, low_memory=False)
                metadata['source'] = "Google Sheets"
            except Exception as e:
                logger.error(f"Failed to load from Google Sheets: {str(e)}")
                metadata['errors'].append(f"Sheet load error: {str(e)}")
                
                # Try to use cached data as fallback
                if 'last_good_data' in st.session_state:
                    logger.info("Using cached data as fallback")
                    df, timestamp, old_metadata = st.session_state.last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    return df, timestamp, metadata
                raise
        
        # Validate loaded data
        status_placeholder.text('🔍 Validating data...')
        progress_placeholder.progress(20)
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Process the data
        status_placeholder.text('⚙️ Processing data...')
        progress_placeholder.progress(30)
        df = DataProcessor.process_dataframe(df, metadata)
        
        # Calculate all scores and rankings
        status_placeholder.text('📊 Calculating rankings...')
        progress_placeholder.progress(50)
        df = RankingEngine.calculate_all_scores(df)
        
        # Detect patterns
        status_placeholder.text('🎯 Detecting patterns...')
        progress_placeholder.progress(70)
        df = PatternDetector.detect_all_patterns(df)
        
        # Add advanced metrics
        status_placeholder.text('📈 Calculating advanced metrics...')
        progress_placeholder.progress(85)
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        # Final validation
        status_placeholder.text('✅ Finalizing...')
        progress_placeholder.progress(95)
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Validate calculations
        calc_valid, calc_errors = DataValidator.validate_critical_calculations(df)
        if not calc_valid:
            metadata['warnings'].extend(calc_errors)
        
        # Store as last good data
        timestamp = datetime.now(timezone.utc)
        st.session_state.last_good_data = (df.copy(), timestamp, metadata)
        
        # Record processing time
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        # Clean up progress indicators
        progress_placeholder.progress(100)
        time.sleep(0.5)
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Clean up memory
        gc.collect()
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        metadata['errors'].append(str(e))
        progress_placeholder.empty()
        status_placeholder.empty()
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
        
        # Create copy to avoid modifying original
        df = df.copy()
        initial_count = len(df)
        
        # Process numeric columns with vectorization
        numeric_cols = [col for col in df.columns if col not in ['ticker', 'company_name', 'category', 'sector', 'year', 'market_cap']]
        
        for col in numeric_cols:
            if col in df.columns:
                # Determine if percentage column
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                
                # Determine bounds
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
                
                # Vectorized cleaning
                df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, is_pct, bounds))
        
        # Process categorical columns
        string_cols = ['ticker', 'company_name', 'category', 'sector']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
        # Fix volume ratios (vectorized)
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                # Convert percentage change to ratio: (100 + change%) / 100
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 10.0)  # Reasonable bounds
                df[col] = df[col].fillna(1.0)
        
        # Validate critical data
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]  # Minimum valid price
        
        # Remove duplicates (keep first)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        # Fill missing values with sensible defaults
        df = DataProcessor._fill_missing_values(df)
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        # Data quality check
        removed = initial_count - len(df)
        if removed > 0:
            metadata['warnings'].append(f"Removed {removed} invalid rows during processing")
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows")
        
        return df
    
    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults"""
        
        # Position data defaults
        if 'from_low_pct' in df.columns:
            df['from_low_pct'] = df['from_low_pct'].fillna(50)
        else:
            df['from_low_pct'] = 50
        
        if 'from_high_pct' in df.columns:
            df['from_high_pct'] = df['from_high_pct'].fillna(-50)
        else:
            df['from_high_pct'] = -50
        
        # RVOL default
        if 'rvol' in df.columns:
            df['rvol'] = df['rvol'].fillna(1.0)
        else:
            df['rvol'] = 1.0
        
        # Return defaults (0 for missing returns)
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols:
            df[col] = df[col].fillna(0)
        
        # Volume defaults
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols:
            df[col] = df[col].fillna(0)
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications with proper boundary handling"""
        
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Classify value into tier with fixed boundary logic"""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                # Use < for upper bound, <= for lower bound to avoid gaps
                if min_val < value <= max_val:
                    return tier_name
                # Handle edge cases for first tier (includes min_val)
                if min_val == -float('inf') and value <= max_val:
                    return tier_name
                # Handle edge cases for last tier (includes max_val)
                if max_val == float('inf') and value > min_val:
                    return tier_name
                # Special case for zero boundaries
                if min_val == 0 and max_val > 0 and value == 0:
                    # Zero belongs to the tier that starts at 0
                    continue  # Let the next tier catch it
            
            return "Unknown"
        
        # Add tier columns
        if 'eps_current' in df.columns:
            df['eps_tier'] = df['eps_current'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['eps'])
            )
        
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: "Negative/NA" if pd.isna(x) or x <= 0 
                else classify_tier(x, CONFIG.TIERS['pe'])
            )
        
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['price'])
            )
        
        return df

# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    """Calculate advanced metrics and indicators"""
    
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics"""
        
        # Money Flow (in millions)
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow'] = df['price'] * df['volume_1d'] * df['rvol']
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
        
        # Volume Momentum Index (VMI)
        if all(col in df.columns for col in CONFIG.VOLUME_RATIO_COLUMNS[:4]):
            df['vmi'] = (
                df['vol_ratio_1d_90d'] * 4 +
                df['vol_ratio_7d_90d'] * 3 +
                df['vol_ratio_30d_90d'] * 2 +
                df['vol_ratio_90d_180d'] * 1
            ) / 10
        
        # Position Tension
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'] + abs(df['from_high_pct'])
        
        # Momentum Harmony
        df['momentum_harmony'] = 0
        
        if 'ret_1d' in df.columns:
            df['momentum_harmony'] += (df['ret_1d'] > 0).astype(int)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                df['momentum_harmony'] += (df['ret_7d'] / 7 > df['ret_30d'] / 30).astype(int)
        
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            with np.errstate(divide='ignore', invalid='ignore'):
                df['momentum_harmony'] += (df['ret_30d'] / 30 > df['ret_3m'] / 90).astype(int)
        
        if 'ret_3m' in df.columns:
            df['momentum_harmony'] += (df['ret_3m'] > 0).astype(int)
        
        # Wave State
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)
        
        # Wave Strength (for filtering)
        df['wave_strength'] = AdvancedMetrics._calculate_wave_strength(df)
        
        return df
    
    @staticmethod
    def _get_wave_state(row: pd.Series) -> str:
        """Determine wave state for a stock"""
        signals = 0
        
        if 'momentum_score' in row and row['momentum_score'] > 70:
            signals += 1
        if 'volume_score' in row and row['volume_score'] > 70:
            signals += 1
        if 'acceleration_score' in row and row['acceleration_score'] > 70:
            signals += 1
        if 'rvol' in row and row['rvol'] > 2:
            signals += 1
        
        if signals >= 4:
            return "🌊🌊🌊 CRESTING"
        elif signals >= 3:
            return "🌊🌊 BUILDING"
        elif signals >= 1:
            return "🌊 FORMING"
        else:
            return "💥 BREAKING"
    
    @staticmethod
    def _calculate_wave_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate wave strength percentage"""
        momentum_count = (df['momentum_score'] >= 60).astype(int) if 'momentum_score' in df.columns else 0
        accel_count = (df['acceleration_score'] >= 70).astype(int) if 'acceleration_score' in df.columns else 0
        rvol_count = (df['rvol'] >= 2).astype(int) if 'rvol' in df.columns else 0
        
        wave_strength = (
            momentum_count * 30 +
            accel_count * 30 +
            rvol_count * 40
        )
        
        return wave_strength

# ============================================
# RANKING ENGINE - OPTIMIZED
# ============================================

class RankingEngine:
    """Core ranking calculations - optimized with numpy"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores and master score"""
        
        if df.empty:
            return df
        
        logger.info("Starting optimized ranking calculations...")
        
        # Calculate component scores
        df['position_score'] = RankingEngine._calculate_position_score(df)
        df['volume_score'] = RankingEngine._calculate_volume_score(df)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df)
        
        # Calculate auxiliary scores
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df)
        
        # Calculate master score using numpy (DO NOT MODIFY FORMULA)
        scores_matrix = np.column_stack([
            df['position_score'].fillna(50),
            df['volume_score'].fillna(50),
            df['momentum_score'].fillna(50),
            df['acceleration_score'].fillna(50),
            df['breakout_score'].fillna(50),
            df['rvol_score'].fillna(50)
        ])
        
        weights = np.array([
            CONFIG.POSITION_WEIGHT,
            CONFIG.VOLUME_WEIGHT,
            CONFIG.MOMENTUM_WEIGHT,
            CONFIG.ACCELERATION_WEIGHT,
            CONFIG.BREAKOUT_WEIGHT,
            CONFIG.RVOL_WEIGHT
        ])
        
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        
        # Calculate ranks
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom')
        df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
        
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        df['percentile'] = df['percentile'].fillna(0)
        
        # Calculate category-specific ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        logger.info(f"Ranking complete: {len(df)} stocks processed")
        
        return df
    
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safely rank a series with proper edge case handling"""
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        # Replace inf values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Count valid values
        valid_count = series.notna().sum()
        if valid_count == 0:
            return pd.Series(50, index=series.index)
        
        # Rank with proper parameters
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom') * 100
            ranks = ranks.fillna(0 if ascending else 100)
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
            ranks = ranks.fillna(valid_count + 1)
        
        return ranks
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score from 52-week range (DO NOT MODIFY LOGIC)"""
        # Initialize with neutral score
        position_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check required columns
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
            logger.warning("No position data available, using neutral position scores")
            return position_score
        
        # Get data with defaults
        from_low = df['from_low_pct'].fillna(50) if has_from_low else pd.Series(50, index=df.index)
        from_high = df['from_high_pct'].fillna(-50) if has_from_high else pd.Series(-50, index=df.index)
        
        # Rank components
        if has_from_low:
            rank_from_low = RankingEngine._safe_rank(from_low, pct=True, ascending=True)
        else:
            rank_from_low = pd.Series(50, index=df.index)
        
        if has_from_high:
            # from_high is negative, less negative = closer to high = better
            rank_from_high = RankingEngine._safe_rank(from_high, pct=True, ascending=False)
        else:
            rank_from_high = pd.Series(50, index=df.index)
        
        # Combined position score (DO NOT MODIFY WEIGHTS)
        position_score = (rank_from_low * 0.6 + rank_from_high * 0.4)
        
        return position_score.clip(0, 100)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive volume score"""
        volume_score = pd.Series(50, index=df.index, dtype=float)
        
        # Volume ratio columns with weights
        vol_cols = [
            ('vol_ratio_1d_90d', 0.20),
            ('vol_ratio_7d_90d', 0.20),
            ('vol_ratio_30d_90d', 0.20),
            ('vol_ratio_30d_180d', 0.15),
            ('vol_ratio_90d_180d', 0.25)
        ]
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                col_rank = RankingEngine._safe_rank(df[col], pct=True, ascending=True)
                weighted_score += col_rank * weight
                total_weight += weight
        
        if total_weight > 0:
            volume_score = weighted_score / total_weight
        else:
            logger.warning("No volume ratio data available, using neutral scores")
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns"""
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            # Fallback to 7-day returns
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                ret_7d = df['ret_7d'].fillna(0)
                momentum_score = RankingEngine._safe_rank(ret_7d, pct=True, ascending=True)
                logger.info("Using 7-day returns for momentum score")
            else:
                logger.warning("No return data available for momentum calculation")
            
            return momentum_score.clip(0, 100)
        
        # Primary: 30-day returns
        ret_30d = df['ret_30d'].fillna(0)
        momentum_score = RankingEngine._safe_rank(ret_30d, pct=True, ascending=True)
        
        # Add consistency bonus
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            
            # Both positive
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            consistency_bonus[all_positive] = 5
            
            # Accelerating returns
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            consistency_bonus[accelerating] = 10
            
            momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
        
        return momentum_score
    
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating with proper division handling"""
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in req_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient return data for acceleration calculation")
            return acceleration_score
        
        # Get return data with defaults
        ret_1d = df['ret_1d'].fillna(0) if 'ret_1d' in df.columns else 0
        ret_7d = df['ret_7d'].fillna(0) if 'ret_7d' in df.columns else 0
        ret_30d = df['ret_30d'].fillna(0) if 'ret_30d' in df.columns else 0
        
        # Calculate daily averages with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_daily_1d = ret_1d  # Already daily
            avg_daily_7d = np.where(ret_7d != 0, ret_7d / 7, 0)
            avg_daily_30d = np.where(ret_30d != 0, ret_30d / 30, 0)
        
        if all(col in df.columns for col in req_cols):
            # Perfect acceleration
            perfect = (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
            acceleration_score[perfect] = 100
            
            # Good acceleration
            good = (~perfect) & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0)
            acceleration_score[good] = 80
            
            # Moderate
            moderate = (~perfect) & (~good) & (ret_1d > 0)
            acceleration_score[moderate] = 60
            
            # Deceleration
            slight_decel = (ret_1d <= 0) & (ret_7d > 0)
            acceleration_score[slight_decel] = 40
            
            strong_decel = (ret_1d <= 0) & (ret_7d <= 0)
            acceleration_score[strong_decel] = 20
        
        return acceleration_score
    
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout probability"""
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        
        # Factor 1: Distance from high (40% weight)
        if 'from_high_pct' in df.columns:
            # from_high_pct is negative, closer to 0 = closer to high
            distance_from_high = -df['from_high_pct'].fillna(-50)
            distance_factor = (100 - distance_from_high).clip(0, 100)
        else:
            distance_factor = pd.Series(50, index=df.index)
        
        # Factor 2: Volume surge (40% weight)
        volume_factor = pd.Series(50, index=df.index)
        if 'vol_ratio_7d_90d' in df.columns:
            vol_ratio = df['vol_ratio_7d_90d'].fillna(1.0)
            volume_factor = ((vol_ratio - 1) * 100).clip(0, 100)
        
        # Factor 3: Trend support (20% weight)
        trend_factor = pd.Series(0, index=df.index, dtype=float)
        
        if 'price' in df.columns:
            current_price = df['price']
            trend_count = 0
            
            for sma_col, points in [('sma_20d', 33.33), ('sma_50d', 33.33), ('sma_200d', 33.34)]:
                if sma_col in df.columns:
                    above_sma = (current_price > df[sma_col]).fillna(False)
                    trend_factor += above_sma.astype(float) * points
                    trend_count += 1
            
            if trend_count > 0 and trend_count < 3:
                trend_factor = trend_factor * (3 / trend_count)
        
        trend_factor = trend_factor.clip(0, 100)
        
        # Combine factors
        breakout_score = (
            distance_factor * 0.4 +
            volume_factor * 0.4 +
            trend_factor * 0.2
        )
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate RVOL-based score"""
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        rvol = df['rvol'].fillna(1.0)
        rvol_score = pd.Series(50, index=df.index, dtype=float)
        
        # Score based on RVOL ranges
        rvol_score[rvol > 10] = 95
        rvol_score[(rvol > 5) & (rvol <= 10)] = 90
        rvol_score[(rvol > 3) & (rvol <= 5)] = 85
        rvol_score[(rvol > 2) & (rvol <= 3)] = 80
        rvol_score[(rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score[(rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score[(rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score[(rvol > 0.3) & (rvol <= 0.5)] = 30
        rvol_score[rvol <= 0.3] = 20
        
        return rvol_score
    
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality score based on SMA alignment"""
        trend_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'price' not in df.columns:
            return trend_score
        
        current_price = df['price']
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        available_smas = [col for col in sma_cols if col in df.columns and df[col].notna().any()]
        
        if len(available_smas) == 0:
            return trend_score
        
        if len(available_smas) >= 3:
            # Perfect trend alignment
            perfect_trend = (
                (current_price > df['sma_20d']) & 
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            )
            trend_score[perfect_trend] = 100
            
            # Strong trend
            strong_trend = (
                (~perfect_trend) &
                (current_price > df['sma_20d']) & 
                (current_price > df['sma_50d']) & 
                (current_price > df['sma_200d'])
            )
            trend_score[strong_trend] = 85
            
            # Count SMAs price is above
            above_count = sum([(current_price > df[sma]).astype(int) for sma in available_smas])
            
            # Good trend
            good_trend = (above_count == 2) & (~perfect_trend) & (~strong_trend)
            trend_score[good_trend] = 70
            
            # Weak trend
            weak_trend = (above_count == 1)
            trend_score[weak_trend] = 40
            
            # Poor trend
            poor_trend = (above_count == 0)
            trend_score[poor_trend] = 20
        
        return trend_score
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score"""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        
        if not available_cols:
            return strength_score
        
        # Calculate average long-term return
        lt_returns = df[available_cols].fillna(0)
        avg_return = lt_returns.mean(axis=1)
        
        # Categorize based on average return
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
        """Calculate liquidity score based on trading volume"""
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'volume_30d' in df.columns and 'price' in df.columns:
            # Calculate dollar volume
            dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
            
            # Rank based on dollar volume
            liquidity_score = RankingEngine._safe_rank(dollar_volume, pct=True, ascending=True)
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks within each category"""
        # Initialize columns
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        # Get unique categories
        categories = df['category'].unique()
        
        # Rank within each category
        for category in categories:
            if category != 'Unknown':
                mask = df['category'] == category
                cat_df = df[mask]
                
                if len(cat_df) > 0:
                    # Calculate ranks
                    cat_ranks = cat_df['master_score'].rank(method='first', ascending=False, na_option='bottom')
                    df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                    
                    # Calculate percentiles
                    cat_percentiles = cat_df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
                    df.loc[mask, 'category_percentile'] = cat_percentiles
        
        return df

# ============================================
# PATTERN DETECTION ENGINE - OPTIMIZED
# ============================================

class PatternDetector:
    """Detect all patterns using vectorized operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized pattern detection using numpy broadcasting"""
        
        n = len(df)
        pattern_masks = np.zeros((n, 25), dtype=bool)  # Pre-allocate
        pattern_names = []
        idx = 0
        
        # 1. Category Leader
        if 'category_percentile' in df.columns:
            pattern_masks[:, idx] = df['category_percentile'].values >= CONFIG.PATTERN_THRESHOLDS['category_leader']
            pattern_names.append('🔥 CAT LEADER')
            idx += 1
        
        # 2. Hidden Gem
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            pattern_masks[:, idx] = (
                (df['category_percentile'].values >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
                (df['percentile'].values < 70)
            )
            pattern_names.append('💎 HIDDEN GEM')
            idx += 1
        
        # 3. Accelerating
        if 'acceleration_score' in df.columns:
            pattern_masks[:, idx] = df['acceleration_score'].values >= CONFIG.PATTERN_THRESHOLDS['acceleration']
            pattern_names.append('🚀 ACCELERATING')
            idx += 1
        
        # 4. Institutional
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            pattern_masks[:, idx] = (
                (df['volume_score'].values >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['vol_ratio_90d_180d'].values > 1.1)
            )
            pattern_names.append('🏦 INSTITUTIONAL')
            idx += 1
        
        # 5. Volume Explosion
        if 'rvol' in df.columns:
            pattern_masks[:, idx] = df['rvol'].values > 3
            pattern_names.append('⚡ VOL EXPLOSION')
            idx += 1
        
        # 6. Breakout Ready
        if 'breakout_score' in df.columns:
            pattern_masks[:, idx] = df['breakout_score'].values >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
            pattern_names.append('🎯 BREAKOUT')
            idx += 1
        
        # 7. Market Leader
        if 'percentile' in df.columns:
            pattern_masks[:, idx] = df['percentile'].values >= CONFIG.PATTERN_THRESHOLDS['market_leader']
            pattern_names.append('👑 MARKET LEADER')
            idx += 1
        
        # 8. Momentum Wave
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            pattern_masks[:, idx] = (
                (df['momentum_score'].values >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                (df['acceleration_score'].values >= 70)
            )
            pattern_names.append('🌊 MOMENTUM WAVE')
            idx += 1
        
        # 9. Liquid Leader
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            pattern_masks[:, idx] = (
                (df['liquidity_score'].values >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
                (df['percentile'].values >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
            )
            pattern_names.append('💰 LIQUID LEADER')
            idx += 1
        
        # 10. Long-term Strength
        if 'long_term_strength' in df.columns:
            pattern_masks[:, idx] = df['long_term_strength'].values >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            pattern_names.append('💪 LONG STRENGTH')
            idx += 1
        
        # 11. Quality Trend
        if 'trend_quality' in df.columns:
            pattern_masks[:, idx] = df['trend_quality'].values >= 80
            pattern_names.append('📈 QUALITY TREND')
            idx += 1
        
        # 12. Value Momentum (Fundamental)
        if 'pe' in df.columns and 'percentile' in df.columns:
            has_valid_pe = (df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)).values
            pattern_masks[:, idx] = has_valid_pe & (df['pe'].values < 15) & (df['master_score'].values >= 70)
            pattern_names.append('💎 VALUE MOMENTUM')
            idx += 1
        
        # 13. Earnings Rocket
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = df['eps_change_pct'].notna().values
            extreme_growth = has_eps_growth & (df['eps_change_pct'].values > 1000)
            normal_growth = has_eps_growth & (df['eps_change_pct'].values > 50) & (df['eps_change_pct'].values <= 1000)
            
            pattern_masks[:, idx] = (
                (extreme_growth & (df['acceleration_score'].values >= 80)) |
                (normal_growth & (df['acceleration_score'].values >= 70))
            )
            pattern_names.append('📊 EARNINGS ROCKET')
            idx += 1
        
        # 14. Quality Leader
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = (
                df['pe'].notna() & 
                df['eps_change_pct'].notna() & 
                (df['pe'] > 0) &
                (df['pe'] < 10000)
            ).values
            pattern_masks[:, idx] = (
                has_complete_data &
                (df['pe'].values >= 10) & (df['pe'].values <= 25) &
                (df['eps_change_pct'].values > 20) &
                (df['percentile'].values >= 80)
            )
            pattern_names.append('🏆 QUALITY LEADER')
            idx += 1
        
        # 15. Turnaround Play
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = df['eps_change_pct'].notna().values
            mega_turnaround = has_eps & (df['eps_change_pct'].values > 500) & (df['volume_score'].values >= 60)
            strong_turnaround = has_eps & (df['eps_change_pct'].values > 100) & (df['eps_change_pct'].values <= 500) & (df['volume_score'].values >= 70)
            
            pattern_masks[:, idx] = mega_turnaround | strong_turnaround
            pattern_names.append('⚡ TURNAROUND')
            idx += 1
        
        # 16. High PE Warning
        if 'pe' in df.columns:
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
            pattern_masks[:, idx] = has_valid_pe.values & (df['pe'].values > 100)
            pattern_names.append('⚠️ HIGH PE')
            idx += 1
        
        # 17. 52W High Approach
        if 'from_high_pct' in df.columns and 'volume_score' in df.columns and 'momentum_score' in df.columns:
            pattern_masks[:, idx] = (
                (df['from_high_pct'].values > -5) & 
                (df['volume_score'].values >= 70) & 
                (df['momentum_score'].values >= 60)
            )
            pattern_names.append('🎯 52W HIGH APPROACH')
            idx += 1
        
        # 18. 52W Low Bounce
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            pattern_masks[:, idx] = (
                (df['from_low_pct'].values < 20) & 
                (df['acceleration_score'].values >= 80) & 
                (df['ret_30d'].values > 10)
            )
            pattern_names.append('🔄 52W LOW BOUNCE')
            idx += 1
        
        # 19. Golden Zone
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            pattern_masks[:, idx] = (
                (df['from_low_pct'].values > 60) & 
                (df['from_high_pct'].values > -40) & 
                (df['trend_quality'].values >= 70)
            )
            pattern_names.append('👑 GOLDEN ZONE')
            idx += 1
        
        # 20. Volume Accumulation
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            pattern_masks[:, idx] = (
                (df['vol_ratio_30d_90d'].values > 1.2) & 
                (df['vol_ratio_90d_180d'].values > 1.1) & 
                (df['ret_30d'].values > 5)
            )
            pattern_names.append('📊 VOL ACCUMULATION')
            idx += 1
        
        # 21. Momentum Divergence
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(df['ret_7d'].values != 0, df['ret_7d'].values / 7, 0)
                daily_30d_pace = np.where(df['ret_30d'].values != 0, df['ret_30d'].values / 30, 0)
            
            pattern_masks[:, idx] = (
                (daily_7d_pace > daily_30d_pace * 1.5) & 
                (df['acceleration_score'].values >= 85) & 
                (df['rvol'].values > 2)
            )
            pattern_names.append('🔀 MOMENTUM DIVERGE')
            idx += 1
        
        # 22. Range Compression
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = np.where(
                    df['low_52w'].values > 0,
                    ((df['high_52w'].values - df['low_52w'].values) / df['low_52w'].values) * 100,
                    100
                )
            
            pattern_masks[:, idx] = (range_pct < 50) & (df['from_low_pct'].values > 30)
            pattern_names.append('🎯 RANGE COMPRESS')
            idx += 1
        
        # 23. Stealth Accumulator
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_ratio = np.where(df['ret_30d'].values != 0, df['ret_7d'].values / (df['ret_30d'].values / 4), 0)
            
            pattern_masks[:, idx] = (
                (df['vol_ratio_90d_180d'].values > 1.1) &
                (df['vol_ratio_30d_90d'].values >= 0.9) & (df['vol_ratio_30d_90d'].values <= 1.1) &
                (df['from_low_pct'].values > 40) &
                (ret_ratio > 1)
            )
            pattern_names.append('🤫 STEALTH')
            idx += 1
        
        # 24. Momentum Vampire
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pace_ratio = np.where(df['ret_7d'].values != 0, df['ret_1d'].values / (df['ret_7d'].values / 7), 0)
            
            is_small_micro = df['category'].isin(['Small Cap', 'Micro Cap']).values
            
            pattern_masks[:, idx] = (
                (daily_pace_ratio > 2) &
                (df['rvol'].values > 3) &
                (df['from_high_pct'].values > -15) &
                is_small_micro
            )
            pattern_names.append('🧛 VAMPIRE')
            idx += 1
        
        # 25. Perfect Storm
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            pattern_masks[:, idx] = (
                (df['momentum_harmony'].values == 4) &
                (df['master_score'].values > 80)
            )
            pattern_names.append('⛈️ PERFECT STORM')
            idx += 1
        
        # Single-pass pattern string creation
        pattern_strings = []
        for i in range(n):
            active_patterns = [pattern_names[j] for j in range(len(pattern_names)) if pattern_masks[i, j]]
            pattern_strings.append(' | '.join(active_patterns))
        
        df['patterns'] = pattern_strings
        
        return df

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    """Advanced market analysis and regime detection"""
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Detect current market regime with supporting data"""
        
        if df.empty:
            return "😴 NO DATA", {}
        
        # Calculate key metrics
        metrics = {}
        
        # Category performance
        if 'category' in df.columns and 'master_score' in df.columns:
            category_scores = df.groupby('category')['master_score'].mean()
            
            micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean()
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean()
            
            metrics['micro_small_avg'] = micro_small_avg
            metrics['large_mega_avg'] = large_mega_avg
            metrics['category_spread'] = micro_small_avg - large_mega_avg
        else:
            micro_small_avg = 50
            large_mega_avg = 50
        
        # Market breadth
        if 'ret_30d' in df.columns:
            breadth = len(df[df['ret_30d'] > 0]) / len(df)
            metrics['breadth'] = breadth
        else:
            breadth = 0.5
        
        # Average RVOL
        if 'rvol' in df.columns:
            avg_rvol = df['rvol'].median()
            metrics['avg_rvol'] = avg_rvol
        else:
            avg_rvol = 1.0
        
        # Determine regime
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
        """Calculate advance/decline ratio and related metrics"""
        
        ad_metrics = {}
        
        if 'ret_1d' in df.columns:
            advancing = len(df[df['ret_1d'] > 0])
            declining = len(df[df['ret_1d'] < 0])
            unchanged = len(df[df['ret_1d'] == 0])
            
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
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """PERCENTILE-BASED fair sector comparison"""
        
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        
        # Calculate market-wide benchmarks
        market_avg_score = df['master_score'].mean()
        market_median_score = df['master_score'].median()
        
        sector_metrics = []
        
        for sector in df['sector'].unique():
            if sector != 'Unknown':
                sector_df = df[df['sector'] == sector]
                
                # Size category
                size = len(sector_df)
                if size >= 200:
                    size_cat = "🐋 MEGA"
                elif size >= 100:
                    size_cat = "🦈 LARGE"
                elif size >= 50:
                    size_cat = "🐟 MEDIUM"
                elif size >= 20:
                    size_cat = "🦐 SMALL"
                else:
                    size_cat = "🐠 MICRO"
                
                # Calculate percentile-based metrics
                market_leaders_pct = (sector_df['percentile'] >= 80).mean() * 100
                avg_market_percentile = sector_df['percentile'].mean()
                beat_market_pct = (sector_df['master_score'] > market_avg_score).mean() * 100
                
                # Elite performance (top 10% within sector)
                sector_top10_threshold = sector_df['master_score'].quantile(0.9)
                elite_avg = sector_df[sector_df['master_score'] >= sector_top10_threshold]['master_score'].mean()
                
                # Smart money flow (money flow to top 30% stocks)
                if 'money_flow_mm' in sector_df.columns:
                    sector_top30_threshold = sector_df['master_score'].quantile(0.7)
                    top30_money_flow = sector_df[sector_df['master_score'] >= sector_top30_threshold]['money_flow_mm'].sum()
                    total_money_flow = sector_df['money_flow_mm'].sum()
                    smart_money_pct = (top30_money_flow / total_money_flow * 100) if total_money_flow > 0 else 0
                else:
                    smart_money_pct = 0
                
                # Additional metrics
                momentum_leaders = (sector_df['momentum_score'] >= 70).mean() * 100 if 'momentum_score' in sector_df.columns else 0
                weak_stocks_pct = (sector_df['percentile'] < 30).mean() * 100
                
                sector_metrics.append({
                    'sector': sector,
                    'size': size,
                    'size_category': size_cat,
                    'market_leaders_pct': market_leaders_pct,
                    'avg_market_percentile': avg_market_percentile,
                    'beat_market_pct': beat_market_pct,
                    'elite_avg': elite_avg,
                    'smart_money_pct': smart_money_pct,
                    'momentum_leaders': momentum_leaders,
                    'weak_stocks_pct': weak_stocks_pct,
                    'avg_score': sector_df['master_score'].mean(),
                    'median_score': sector_df['master_score'].median(),
                    'avg_rvol': sector_df['rvol'].mean() if 'rvol' in sector_df.columns else 1.0
                })
        
        # Create DataFrame and sort by avg_market_percentile
        sector_rotation_df = pd.DataFrame(sector_metrics)
        sector_rotation_df = sector_rotation_df.sort_values('avg_market_percentile', ascending=False)
        
        return sector_rotation_df
    
    @staticmethod
    def detect_category_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Category analysis using same percentile approach"""
        
        if 'category' not in df.columns or df.empty:
            return pd.DataFrame()
        
        # Calculate market-wide benchmarks
        market_avg_score = df['master_score'].mean()
        
        category_metrics = []
        
        for category in df['category'].unique():
            if category != 'Unknown':
                cat_df = df[df['category'] == category]
                
                # Calculate percentile-based metrics
                market_leaders_pct = (cat_df['percentile'] >= 80).mean() * 100
                avg_market_percentile = cat_df['percentile'].mean()
                beat_market_pct = (cat_df['master_score'] > market_avg_score).mean() * 100
                
                # Elite performance
                cat_top10_threshold = cat_df['master_score'].quantile(0.9)
                elite_avg = cat_df[cat_df['master_score'] >= cat_top10_threshold]['master_score'].mean()
                
                # Additional metrics
                momentum_leaders = (cat_df['momentum_score'] >= 70).mean() * 100 if 'momentum_score' in cat_df.columns else 0
                weak_stocks_pct = (cat_df['percentile'] < 30).mean() * 100
                
                category_metrics.append({
                    'category': category,
                    'size': len(cat_df),
                    'market_leaders_pct': market_leaders_pct,
                    'avg_market_percentile': avg_market_percentile,
                    'beat_market_pct': beat_market_pct,
                    'elite_avg': elite_avg,
                    'momentum_leaders': momentum_leaders,
                    'weak_stocks_pct': weak_stocks_pct,
                    'avg_score': cat_df['master_score'].mean(),
                    'median_score': cat_df['master_score'].median()
                })
        
        # Create DataFrame and sort
        category_rotation_df = pd.DataFrame(category_metrics)
        category_rotation_df = category_rotation_df.sort_values('avg_market_percentile', ascending=False)
        
        return category_rotation_df

# ============================================
# ALERT SYSTEM
# ============================================

class AlertSystem:
    """Generate intelligent market alerts"""
    
    @staticmethod
    def check_market_alerts(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for important market conditions"""
        alerts = []
        
        # 1. Extreme volume alert
        if 'rvol' in df.columns:
            extreme_rvol = df[df['rvol'] > CONFIG.ALERT_THRESHOLDS['extreme_rvol']]
            if len(extreme_rvol) > 5:
                top_stocks = extreme_rvol.nlargest(5, 'master_score')['ticker'].tolist()
                alerts.append({
                    'type': 'warning',
                    'icon': '⚡',
                    'title': 'Extreme Volume Detected',
                    'message': f"{len(extreme_rvol)} stocks showing extreme volume (>5x)",
                    'stocks': top_stocks,
                    'action': 'Check for news or unusual activity'
                })
        
        # 2. Sector rotation alert
        sector_rotation = MarketIntelligence.detect_sector_rotation(df)
        if not sector_rotation.empty and len(sector_rotation) > 1:
            # Check for significant rotation
            percentile_std = sector_rotation['avg_market_percentile'].std()
            if percentile_std > CONFIG.ALERT_THRESHOLDS['sector_rotation_std']:
                top_sector = sector_rotation.iloc[0]['sector']
                bottom_sector = sector_rotation.iloc[-1]['sector']
                alerts.append({
                    'type': 'info',
                    'icon': '🔄',
                    'title': 'Sector Rotation Detected',
                    'message': f"Strong rotation from {bottom_sector} to {top_sector}",
                    'stocks': [],
                    'action': 'Consider rebalancing sector exposure'
                })
        
        # 3. Market regime shift alert
        regime, metrics = MarketIntelligence.detect_market_regime(df)
        category_spread = metrics.get('category_spread', 0)
        if abs(category_spread) > CONFIG.ALERT_THRESHOLDS['regime_shift_spread']:
            alerts.append({
                'type': 'success' if category_spread > 0 else 'warning',
                'icon': '🎯',
                'title': 'Market Regime Shift',
                'message': f"Market entering {regime} mode",
                'stocks': [],
                'action': 'Adjust portfolio risk accordingly'
            })
        
        # 4. Pattern emergence alert
        if 'patterns' in df.columns:
            pattern_counts = {}
            for patterns in df['patterns'].dropna():
                if patterns:
                    for p in patterns.split(' | '):
                        pattern_counts[p] = pattern_counts.get(p, 0) + 1
            
            # Check for sudden pattern emergence
            for pattern, count in pattern_counts.items():
                if count > CONFIG.ALERT_THRESHOLDS['pattern_emergence']:
                    pattern_stocks = df[df['patterns'].str.contains(pattern, na=False)].nlargest(5, 'master_score')
                    alerts.append({
                        'type': 'info',
                        'icon': '🎯',
                        'title': f'Pattern Surge: {pattern}',
                        'message': f"{count} stocks showing {pattern} pattern",
                        'stocks': pattern_stocks['ticker'].tolist(),
                        'action': 'Review stocks with this pattern'
                    })
        
        # 5. Unusual volume concentration
        if 'money_flow_mm' in df.columns:
            top10_money_flow = df.nlargest(10, 'money_flow_mm')['money_flow_mm'].sum()
            total_money_flow = df['money_flow_mm'].sum()
            concentration = (top10_money_flow / total_money_flow * 100) if total_money_flow > 0 else 0
            
            if concentration > 50:
                top_stocks = df.nlargest(3, 'money_flow_mm')['ticker'].tolist()
                alerts.append({
                    'type': 'warning',
                    'icon': '💰',
                    'title': 'Volume Concentration',
                    'message': f"Top 10 stocks control {concentration:.0f}% of money flow",
                    'stocks': top_stocks,
                    'action': 'Monitor for potential manipulation'
                })
        
        return alerts

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations with proper error handling"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Score components to visualize
        scores = [
            ('position_score', 'Position', '#3498db'),
            ('volume_score', 'Volume', '#e74c3c'),
            ('momentum_score', 'Momentum', '#2ecc71'),
            ('acceleration_score', 'Acceleration', '#f39c12'),
            ('breakout_score', 'Breakout', '#9b59b6'),
            ('rvol_score', 'RVOL', '#e67e22')
        ]
        
        for score_col, label, color in scores:
            if score_col in df.columns:
                score_data = df[score_col].dropna()
                if len(score_data) > 0:
                    fig.add_trace(go.Box(
                        y=score_data,
                        name=label,
                        marker_color=color,
                        boxpoints='outliers',
                        hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'
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
    def create_master_score_breakdown(df: pd.DataFrame, n: int = 20) -> go.Figure:
        """Create master score breakdown chart showing component contributions"""
        # Get top stocks
        top_df = df.nlargest(min(n, len(df)), 'master_score').copy()
        
        if len(top_df) == 0:
            return go.Figure()
        
        # Calculate weighted contributions
        components = [
            ('Position', 'position_score', CONFIG.POSITION_WEIGHT, '#3498db'),
            ('Volume', 'volume_score', CONFIG.VOLUME_WEIGHT, '#e74c3c'),
            ('Momentum', 'momentum_score', CONFIG.MOMENTUM_WEIGHT, '#2ecc71'),
            ('Acceleration', 'acceleration_score', CONFIG.ACCELERATION_WEIGHT, '#f39c12'),
            ('Breakout', 'breakout_score', CONFIG.BREAKOUT_WEIGHT, '#9b59b6'),
            ('RVOL', 'rvol_score', CONFIG.RVOL_WEIGHT, '#e67e22')
        ]
        
        fig = go.Figure()
        
        # Add bars for each component
        for name, score_col, weight, color in components:
            if score_col in top_df.columns:
                # Calculate weighted contribution
                weighted_contrib = top_df[score_col] * weight
                
                fig.add_trace(go.Bar(
                    name=f'{name} ({weight:.0%})',
                    y=top_df['ticker'],
                    x=weighted_contrib,
                    orientation='h',
                    marker_color=color,
                    text=[f"{val:.1f}" for val in top_df[score_col]],
                    textposition='inside',
                    hovertemplate=(
                        f'{name}<br>'
                        'Raw Score: %{text}<br>'
                        'Weight: ' + f'{weight:.0%}' + '<br>'
                        'Contribution: %{x:.1f}<extra></extra>'
                    )
                ))
        
        # Add master score annotations
        for i, (idx, row) in enumerate(top_df.iterrows()):
            fig.add_annotation(
                x=row['master_score'] + 1,
                y=i,
                text=f"<b>{row['master_score']:.1f}</b>",
                showarrow=False,
                xanchor='left',
                font=dict(size=12, color='black'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        
        fig.update_layout(
            title=f"Top {len(top_df)} Stocks - Master Score 3.0 Component Breakdown",
            xaxis_title="Weighted Score Contribution",
            xaxis_range=[0, 110],
            barmode='stack',
            template='plotly_white',
            height=max(400, len(top_df) * 35),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=100, r=100, t=100, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance_scatter(df: pd.DataFrame) -> go.Figure:
        """Create sector performance scatter plot with bubble size"""
        try:
            # Use percentile-based sector rotation data
            sector_stats = MarketIntelligence.detect_sector_rotation(df)
            
            if sector_stats.empty:
                return go.Figure()
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add scatter trace
            fig.add_trace(go.Scatter(
                x=sector_stats['avg_market_percentile'],
                y=sector_stats['avg_score'],
                mode='markers+text',
                text=sector_stats['sector'],
                textposition='top center',
                marker=dict(
                    size=np.sqrt(sector_stats['size']) * 5,  # Scale bubble size
                    sizemin=10,
                    sizemode='diameter',
                    sizeref=2,
                    color=sector_stats['avg_rvol'],
                    colorscale='Viridis',
                    colorbar=dict(title="Avg RVOL"),
                    line=dict(width=2, color='white'),
                    showscale=True
                ),
                customdata=np.column_stack((
                    sector_stats['size'],
                    sector_stats['size_category'],
                    sector_stats['market_leaders_pct'],
                    sector_stats['beat_market_pct'],
                    sector_stats['momentum_leaders']
                )),
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    'Avg Score: %{y:.1f}<br>' +
                    'Avg Market Percentile: %{x:.1f}<br>' +
                    'Stocks: %{customdata[0]} %{customdata[1]}<br>' +
                    'Market Leaders: %{customdata[2]:.1f}%<br>' +
                    'Beat Market: %{customdata[3]:.1f}%<br>' +
                    'Momentum Leaders: %{customdata[4]:.1f}%<extra></extra>'
                )
            ))
            
            # Add quadrant lines
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant labels
            fig.add_annotation(x=75, y=75, text="Leaders", showarrow=False, 
                             font=dict(size=14, color="green"), opacity=0.7)
            fig.add_annotation(x=25, y=75, text="Hidden Gems", showarrow=False,
                             font=dict(size=14, color="blue"), opacity=0.7)
            fig.add_annotation(x=75, y=25, text="Overvalued", showarrow=False,
                             font=dict(size=14, color="orange"), opacity=0.7)
            fig.add_annotation(x=25, y=25, text="Laggards", showarrow=False,
                             font=dict(size=14, color="red"), opacity=0.7)
            
            fig.update_layout(
                title='Sector Performance Analysis - Percentile Based Fair Comparison',
                xaxis_title='Average Market Percentile',
                yaxis_title='Average Master Score',
                template='plotly_white',
                height=500,
                xaxis=dict(range=[0, 100]),
                yaxis=dict(range=[0, 100])
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sector scatter: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, n: int = 10) -> go.Figure:
        """Create acceleration profiles showing momentum over time"""
        try:
            # Get top accelerating stocks
            accel_df = df.nlargest(min(n, len(df)), 'acceleration_score')
            
            if len(accel_df) == 0:
                return go.Figure()
            
            fig = go.Figure()
            
            # Create lines for each stock
            for _, stock in accel_df.iterrows():
                # Build timeline data
                x_points = []
                y_points = []
                
                # Start point
                x_points.append('Start')
                y_points.append(0)
                
                # Add available return data points
                if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']):
                    x_points.append('30D')
                    y_points.append(stock['ret_30d'])
                
                if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']):
                    x_points.append('7D')
                    y_points.append(stock['ret_7d'])
                
                if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']):
                    x_points.append('Today')
                    y_points.append(stock['ret_1d'])
                
                if len(x_points) > 1:  # Only plot if we have data
                    # Determine line style based on acceleration
                    if stock['acceleration_score'] >= 85:
                        line_style = dict(width=3, dash='solid')
                        marker_style = dict(size=10, symbol='star')
                    elif stock['acceleration_score'] >= 70:
                        line_style = dict(width=2, dash='solid')
                        marker_style = dict(size=8)
                    else:
                        line_style = dict(width=2, dash='dot')
                        marker_style = dict(size=6)
                    
                    fig.add_trace(go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode='lines+markers',
                        name=f"{stock['ticker']} ({stock['acceleration_score']:.0f})",
                        line=line_style,
                        marker=marker_style,
                        hovertemplate=(
                            f"<b>{stock['ticker']}</b><br>" +
                            "%{x}: %{y:.1f}%<br>" +
                            f"Accel Score: {stock['acceleration_score']:.0f}<extra></extra>"
                        )
                    ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title=f"Acceleration Profiles - Top {len(accel_df)} Momentum Builders",
                xaxis_title="Time Frame",
                yaxis_title="Return %",
                height=400,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}")
            return go.Figure()

# ============================================
# FILTER ENGINE - OPTIMIZED
# ============================================

class FilterEngine:
    """Handle all filtering operations efficiently"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with optimized performance"""
        
        if df.empty:
            return df
        
        # Start with boolean index for all rows
        mask = pd.Series(True, index=df.index)
        
        # Category filter
        categories = filters.get('categories', [])
        if categories and 'All' not in categories:
            mask &= df['category'].isin(categories)
        
        # Sector filter
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors:
            mask &= df['sector'].isin(sectors)
        
        # Score filter
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            mask &= df['master_score'] >= min_score
        
        # EPS change filter
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            mask &= (df['eps_change_pct'] >= min_eps_change) | df['eps_change_pct'].isna()
        
        # Pattern filter
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            pattern_regex = '|'.join(patterns)
            mask &= df['patterns'].str.contains(pattern_regex, case=False, na=False, regex=True)
        
        # Trend filter
        if filters.get('trend_range') and filters.get('trend_filter') != 'All Trends':
            min_trend, max_trend = filters['trend_range']
            if 'trend_quality' in df.columns:
                mask &= (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend)
        
        # PE filters
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in df.columns:
            mask &= df['pe'].isna() | ((df['pe'] > 0) & (df['pe'] >= min_pe))
        
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in df.columns:
            mask &= df['pe'].isna() | ((df['pe'] > 0) & (df['pe'] <= max_pe))
        
        # Apply tier filters
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            tier_values = filters.get(tier_type, [])
            if tier_values and 'All' not in tier_values:
                col_name = tier_type.replace('_tiers', '_tier')
                if col_name in df.columns:
                    mask &= df[col_name].isin(tier_values)
        
        # Wave state filter
        wave_states = filters.get('wave_states', [])
        if wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)
        
        # Wave strength filter
        min_wave, max_wave = filters.get('wave_strength_range', (0, 100))
        if (min_wave > 0 or max_wave < 100) and 'wave_strength' in df.columns:
            mask &= (df['wave_strength'] >= min_wave) & (df['wave_strength'] <= max_wave)
        
        # Data completeness filter
        if filters.get('require_fundamental_data', False):
            if 'pe' in df.columns and 'eps_change_pct' in df.columns:
                mask &= df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna()
        
        # Apply mask efficiently
        filtered_df = df[mask].copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        """Get available filter options with smart interconnection"""
        
        if df.empty or column not in df.columns:
            return []
        
        # Apply other filters first for interconnected filtering
        temp_filters = current_filters.copy()
        
        # Remove the current column's filter to see all its options
        filter_key_map = {
            'category': 'categories',
            'sector': 'sectors',
            'eps_tier': 'eps_tiers',
            'pe_tier': 'pe_tiers',
            'price_tier': 'price_tiers'
        }
        
        if column in filter_key_map:
            temp_filters.pop(filter_key_map[column], None)
        
        # Apply remaining filters
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        
        # Get unique values
        values = filtered_df[column].dropna().unique()
        
        # Exclude Unknown and empty values
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN']]
        
        return sorted(values)

# ============================================
# SEARCH ENGINE - ENHANCED
# ============================================

class SearchEngine:
    """Enhanced search functionality with word matching"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Enhanced search with word matching and relevance scoring"""
        
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query = query.upper().strip()
            
            # Method 1: Direct ticker match (exact)
            ticker_exact = df[df['ticker'].str.upper() == query]
            if not ticker_exact.empty:
                return ticker_exact
            
            # Method 2: Ticker contains query
            ticker_contains = df[df['ticker'].str.upper().str.contains(query, na=False, regex=False)]
            
            # Method 3: Company name contains query (case insensitive)
            company_contains = df[df['company_name'].str.upper().str.contains(query, na=False, regex=False)]
            
            # Method 4: Word starts with (from OLD version)
            def word_starts_with(company_name):
                if pd.isna(company_name):
                    return False
                words = str(company_name).upper().split()
                return any(word.startswith(query) for word in words)
            
            company_word_match = df[df['company_name'].apply(word_starts_with)]
            
            # Combine all results and remove duplicates
            all_matches = pd.concat([
                ticker_contains,
                company_contains,
                company_word_match
            ]).drop_duplicates()
            
            # Sort by relevance: exact ticker match first, then by master score
            if not all_matches.empty:
                # Add relevance score
                all_matches['relevance'] = 0
                all_matches.loc[all_matches['ticker'].str.upper() == query, 'relevance'] = 100
                all_matches.loc[all_matches['ticker'].str.upper().str.startswith(query), 'relevance'] += 50
                all_matches.loc[all_matches['company_name'].str.upper().str.startswith(query), 'relevance'] += 30
                
                # Sort by relevance then master score
                all_matches = all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False])
                
                # Drop the relevance column before returning
                return all_matches.drop('relevance', axis=1)
            
            return pd.DataFrame()
            
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
        """Create comprehensive Excel report with smart templates"""
        
        output = BytesIO()
        
        # Define export templates
        templates = {
            'day_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 
                           'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 
                           'volume_score', 'vmi', 'wave_state', 'patterns', 'category'],
                'focus': 'Intraday momentum and volume'
            },
            'swing_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 
                           'breakout_score', 'position_score', 'position_tension',
                           'from_high_pct', 'from_low_pct', 'trend_quality', 
                           'momentum_harmony', 'patterns'],
                'focus': 'Position and breakout setups'
            },
            'investor': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 
                           'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 
                           'long_term_strength', 'money_flow_mm', 'category', 'sector'],
                'focus': 'Fundamentals and long-term performance'
            },
            'full': {
                'columns': None,  # Use all columns
                'focus': 'Complete analysis'
            }
        }
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#3498db',
                    'font_color': 'white',
                    'border': 1
                })
                
                number_format = workbook.add_format({'num_format': '#,##0.00'})
                percent_format = workbook.add_format({'num_format': '0.0%'})
                currency_format = workbook.add_format({'num_format': '₹#,##0'})
                
                # 1. Top 100 Stocks
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                
                # Select columns based on template
                if template in templates and templates[template]['columns']:
                    export_cols = [col for col in templates[template]['columns'] if col in top_100.columns]
                else:
                    export_cols = None  # Use all columns
                
                if export_cols:
                    top_100_export = top_100[export_cols]
                else:
                    top_100_export = top_100
                
                top_100_export.to_excel(writer, sheet_name='Top 100', index=False)
                
                # Format the sheet
                worksheet = writer.sheets['Top 100']
                for i, col in enumerate(top_100_export.columns):
                    worksheet.write(0, i, col, header_format)
                
                # 2. Market Intelligence
                intel_data = []
                
                # Market regime
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                intel_data.append({
                    'Metric': 'Market Regime',
                    'Value': regime,
                    'Details': f"Breadth: {regime_metrics.get('breadth', 0):.1%}"
                })
                
                # A/D Ratio
                ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
                intel_data.append({
                    'Metric': 'Advance/Decline',
                    'Value': f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}",
                    'Details': f"Ratio: {ad_metrics.get('ad_ratio', 1):.2f}"
                })
                
                intel_df = pd.DataFrame(intel_data)
                intel_df.to_excel(writer, sheet_name='Market Intelligence', index=False)
                
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
                
                # 5. Wave Radar Signals
                wave_signals = df[
                    (df['momentum_score'] >= 60) & 
                    (df['acceleration_score'] >= 70) &
                    (df['rvol'] >= 2)
                ].head(50)
                
                if len(wave_signals) > 0:
                    wave_cols = ['ticker', 'company_name', 'master_score', 
                                'momentum_score', 'acceleration_score', 'rvol',
                                'wave_state', 'patterns', 'category']
                    available_wave_cols = [col for col in wave_cols if col in wave_signals.columns]
                    
                    wave_signals[available_wave_cols].to_excel(
                        writer, sheet_name='Wave Radar', index=False
                    )
                
                # 6. Summary Statistics
                summary_stats = {
                    'Total Stocks': len(df),
                    'Average Master Score': df['master_score'].mean(),
                    'Stocks with Patterns': (df['patterns'] != '').sum(),
                    'High RVOL (>2x)': (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0,
                    'Positive 30D Returns': (df['ret_30d'] > 0).sum() if 'ret_30d' in df.columns else 0,
                    'Template Used': template,
                    'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                logger.info(f"Excel report created successfully with {len(writer.sheets)} sheets")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export efficiently"""
        
        # Select important columns for CSV
        export_cols = [
            'rank', 'ticker', 'company_name', 'master_score',
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score',
            'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
            'from_low_pct', 'from_high_pct',
            'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
            'rvol', 'vmi', 'money_flow_mm', 'position_tension',
            'momentum_harmony', 'wave_state', 'wave_strength', 'patterns', 
            'category', 'sector', 'eps_tier', 'pe_tier'
        ]
        
        available_cols = [col for col in export_cols if col in df.columns]
        
        # Create export dataframe
        export_df = df[available_cols].copy()
        
        # Convert volume ratios back to percentage for display
        vol_ratio_cols = [col for col in export_df.columns if 'vol_ratio' in col]
        for col in vol_ratio_cols:
            export_df[col] = (export_df[col] - 1) * 100
        
        return export_df.to_csv(index=False)

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None, 
                          help_text: Optional[str] = None) -> None:
        """Render a styled metric card"""
        if help_text:
            st.metric(label, value, delta, help=help_text)
        else:
            st.metric(label, value, delta)
    
    @staticmethod
    def render_summary_section(df: pd.DataFrame) -> None:
        """Render enhanced summary dashboard"""
        
        if df.empty:
            st.warning("No data available for summary")
            return
        
        # 1. MARKET PULSE
        st.markdown("### 📊 Market Pulse")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # A/D Ratio
            ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
            ad_ratio = ad_metrics.get('ad_ratio', 1.0)
            
            if ad_ratio > 2:
                ad_emoji = "🔥"
                ad_color = "inverse"
            elif ad_ratio > 1:
                ad_emoji = "📈"
                ad_color = "normal"
            else:
                ad_emoji = "📉"
                ad_color = "off"
            
            UIComponents.render_metric_card(
                "A/D Ratio",
                f"{ad_emoji} {ad_ratio:.2f}",
                f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}",
                "Advance/Decline Ratio"
            )
        
        with col2:
            # Momentum Health
            high_momentum = len(df[df['momentum_score'] >= 70])
            momentum_pct = (high_momentum / len(df) * 100)
            
            UIComponents.render_metric_card(
                "Momentum Health",
                f"{momentum_pct:.0f}%",
                f"{high_momentum} strong stocks"
            )
        
        with col3:
            # Volume State
            avg_rvol = df['rvol'].median() if 'rvol' in df.columns else 1.0
            high_vol_count = len(df[df['rvol'] > 2]) if 'rvol' in df.columns else 0
            
            if avg_rvol > 1.5:
                vol_emoji = "🌊"
            elif avg_rvol > 1.2:
                vol_emoji = "💧"
            else:
                vol_emoji = "🏜️"
            
            UIComponents.render_metric_card(
                "Volume State",
                f"{vol_emoji} {avg_rvol:.1f}x",
                f"{high_vol_count} surges"
            )
        
        with col4:
            # Risk Level
            risk_factors = 0
            
            # Check overextended stocks
            if 'from_high_pct' in df.columns and 'momentum_score' in df.columns:
                overextended = len(df[(df['from_high_pct'] >= 0) & (df['momentum_score'] < 50)])
                if overextended > 20:
                    risk_factors += 1
            
            # Check extreme RVOL
            if 'rvol' in df.columns:
                pump_risk = len(df[(df['rvol'] > 10) & (df['master_score'] < 50)])
                if pump_risk > 10:
                    risk_factors += 1
            
            # Check downtrends
            if 'trend_quality' in df.columns:
                downtrends = len(df[df['trend_quality'] < 40])
                if downtrends > len(df) * 0.3:
                    risk_factors += 1
            
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]
            risk_level = risk_levels[min(risk_factors, 3)]
            
            UIComponents.render_metric_card(
                "Risk Level",
                risk_level,
                f"{risk_factors} factors"
            )
        
        # 2. MARKET ALERTS
        alerts = AlertSystem.check_market_alerts(df)
        if alerts:
            st.markdown("### 🚨 Market Alerts")
            for alert in alerts[:3]:  # Show top 3 alerts
                alert_type = alert['type']
                if alert_type == 'warning':
                    st.warning(f"{alert['icon']} **{alert['title']}**: {alert['message']}")
                elif alert_type == 'success':
                    st.success(f"{alert['icon']} **{alert['title']}**: {alert['message']}")
                else:
                    st.info(f"{alert['icon']} **{alert['title']}**: {alert['message']}")
                
                if alert['stocks']:
                    st.caption(f"Top stocks: {', '.join(alert['stocks'][:3])}")
        
        # 3. TODAY'S OPPORTUNITIES
        st.markdown("### 🎯 Today's Best Opportunities")
        
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        
        with opp_col1:
            # Ready to Run
            ready_to_run = df[
                (df['momentum_score'] >= 70) & 
                (df['acceleration_score'] >= 70) &
                (df['rvol'] >= 2)
            ].nlargest(5, 'master_score')
            
            st.markdown("**🚀 Ready to Run**")
            if len(ready_to_run) > 0:
                for _, stock in ready_to_run.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"Score: {stock['master_score']:.1f} | RVOL: {stock['rvol']:.1f}x")
            else:
                st.info("No momentum leaders found")
        
        with opp_col2:
            # Hidden Gems
            hidden_gems = df[df['patterns'].str.contains('HIDDEN GEM', na=False)].nlargest(5, 'master_score')
            
            st.markdown("**💎 Hidden Gems**")
            if len(hidden_gems) > 0:
                for _, stock in hidden_gems.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"Cat %ile: {stock.get('category_percentile', 0):.0f} | Score: {stock['master_score']:.1f}")
            else:
                st.info("No hidden gems today")
        
        with opp_col3:
            # Volume Alerts
            volume_alerts = df[df['rvol'] > 3].nlargest(5, 'master_score')
            
            st.markdown("**⚡ Volume Alerts**")
            if len(volume_alerts) > 0:
                for _, stock in volume_alerts.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"RVOL: {stock['rvol']:.1f}x | {stock.get('wave_state', 'N/A')}")
            else:
                st.info("No extreme volume detected")
        
        # 4. MARKET INTELLIGENCE
        st.markdown("### 🧠 Market Intelligence")
        
        intel_col1, intel_col2 = st.columns([2, 1])
        
        with intel_col1:
            # Sector Rotation Map
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            
            if not sector_rotation.empty:
                # Create rotation visualization
                fig = go.Figure()
                
                # Add bar chart
                fig.add_trace(go.Bar(
                    x=sector_rotation['sector'][:10],  # Top 10 sectors
                    y=sector_rotation['avg_market_percentile'][:10],
                    text=[f"{val:.1f}" for val in sector_rotation['avg_market_percentile'][:10]],
                    textposition='outside',
                    marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' 
                                 for score in sector_rotation['avg_market_percentile'][:10]],
                    hovertemplate=(
                        'Sector: %{x}<br>'
                        'Avg Market Percentile: %{y:.1f}<br>'
                        'Size: %{customdata[0]} %{customdata[1]}<br>'
                        'Market Leaders: %{customdata[2]:.1f}%<br>'
                        'Beat Market: %{customdata[3]:.1f}%<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        sector_rotation['size'][:10],
                        sector_rotation['size_category'][:10],
                        sector_rotation['market_leaders_pct'][:10],
                        sector_rotation['beat_market_pct'][:10]
                    ))
                ))
                
                fig.update_layout(
                    title="Sector Rotation Map - Percentile Based Fair Comparison",
                    xaxis_title="Sector",
                    yaxis_title="Average Market Percentile",
                    height=400,
                    template='plotly_white',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with intel_col2:
            # Market Regime
            regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
            
            st.markdown(f"**🎯 Market Regime**")
            st.markdown(f"### {regime}")
            
            # Key signals
            st.markdown("**📡 Key Signals**")
            
            signals = []
            
            # Breadth signal
            breadth = regime_metrics.get('breadth', 0.5)
            if breadth > 0.6:
                signals.append("✅ Strong breadth")
            elif breadth < 0.4:
                signals.append("⚠️ Weak breadth")
            
            # Category spread signal
            cat_spread = regime_metrics.get('category_spread', 0)
            if cat_spread > 10:
                signals.append("🔥 Small caps leading")
            elif cat_spread < -10:
                signals.append("🛡️ Large caps defensive")
            
            # RVOL signal
            avg_rvol = regime_metrics.get('avg_rvol', 1.0)
            if avg_rvol > 1.5:
                signals.append("🌊 High volume activity")
            
            for signal in signals[:3]:
                st.caption(signal)
            
            # Quick Statistics (from old version)
            with st.expander("📊 Quick Statistics", expanded=False):
                if 'master_score' in df.columns:
                    q1 = df['master_score'].quantile(0.25)
                    median = df['master_score'].quantile(0.50)
                    q3 = df['master_score'].quantile(0.75)
                    
                    st.write(f"**Master Score Distribution**")
                    st.write(f"Q1 (25%): {q1:.1f}")
                    st.write(f"Median: {median:.1f}")
                    st.write(f"Q3 (75%): {q3:.1f}")
                    st.write(f"Spread: {q3 - q1:.1f}")
    
    @staticmethod
    def render_stock_card(stock: pd.Series, show_detailed: bool = False) -> None:
        """Render a comprehensive stock card with all details"""
        
        # Header with gradient background
        header_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">
                #{int(stock['rank'])} {stock['ticker']} - {stock['company_name']}
            </h2>
            <p style="color: #f0f0f0; margin: 5px 0;">
                {stock.get('category', 'Unknown')} | {stock.get('sector', 'Unknown')}
            </p>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
        
        # Score breakdown with progress bars
        st.markdown("### 📊 Score Components")
        
        score_col1, score_col2 = st.columns(2)
        
        with score_col1:
            # Master Score with visual gauge
            master_score = stock['master_score']
            color = '#2ecc71' if master_score >= 70 else '#f39c12' if master_score >= 50 else '#e74c3c'
            
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 48px; font-weight: bold; color: {color};">
                    {master_score:.1f}
                </div>
                <div style="color: #666;">Master Score 3.0</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Component scores with horizontal bars
            components = [
                ('Position', stock.get('position_score', 50), CONFIG.POSITION_WEIGHT),
                ('Volume', stock.get('volume_score', 50), CONFIG.VOLUME_WEIGHT),
                ('Momentum', stock.get('momentum_score', 50), CONFIG.MOMENTUM_WEIGHT),
                ('Acceleration', stock.get('acceleration_score', 50), CONFIG.ACCELERATION_WEIGHT),
                ('Breakout', stock.get('breakout_score', 50), CONFIG.BREAKOUT_WEIGHT),
                ('RVOL', stock.get('rvol_score', 50), CONFIG.RVOL_WEIGHT)
            ]
            
            for name, score, weight in components:
                st.progress(score / 100, text=f"{name}: {score:.0f} (×{weight:.0%})")
        
        with score_col2:
            # Key metrics
            st.markdown("### 🎯 Key Metrics")
            
            metrics_data = []
            
            # Price position
            if 'from_low_pct' in stock and 'from_high_pct' in stock:
                position_text = UIComponents.get_sma_position(stock)
                metrics_data.append(("📍 Position", position_text))
            
            # RVOL
            if 'rvol' in stock:
                rvol_text = f"{stock['rvol']:.1f}x"
                if stock['rvol'] > 3:
                    rvol_text = f"🔥 {rvol_text}"
                elif stock['rvol'] > 2:
                    rvol_text = f"⚡ {rvol_text}"
                metrics_data.append(("📊 RVOL", rvol_text))
            
            # Wave State
            if 'wave_state' in stock:
                metrics_data.append(("🌊 Wave", stock['wave_state']))
            
            # Recent performance
            if 'ret_30d' in stock:
                ret_color = '#2ecc71' if stock['ret_30d'] > 0 else '#e74c3c'
                metrics_data.append(("📈 30D Return", f"{stock['ret_30d']:.1f}%"))
            
            for label, value in metrics_data:
                st.markdown(f"**{label}**: {value}")
        
        # Patterns section
        if 'patterns' in stock and stock['patterns']:
            st.markdown("### 🎯 Active Patterns")
            patterns = stock['patterns'].split(' | ')
            
            # Group patterns by category
            momentum_patterns = []
            volume_patterns = []
            position_patterns = []
            fundamental_patterns = []
            warning_patterns = []
            
            for pattern in patterns:
                if any(keyword in pattern for keyword in ['MOMENTUM', 'ACCELERATING', 'WAVE']):
                    momentum_patterns.append(pattern)
                elif any(keyword in pattern for keyword in ['VOL', 'LIQUID', 'INSTITUTIONAL']):
                    volume_patterns.append(pattern)
                elif any(keyword in pattern for keyword in ['BREAKOUT', '52W', 'GOLDEN', 'RANGE']):
                    position_patterns.append(pattern)
                elif any(keyword in pattern for keyword in ['VALUE', 'EARNINGS', 'QUALITY', 'PE']):
                    fundamental_patterns.append(pattern)
                elif 'WARNING' in pattern or 'HIGH PE' in pattern:
                    warning_patterns.append(pattern)
                else:
                    momentum_patterns.append(pattern)  # Default category
            
            # Display grouped patterns
            pattern_groups = [
                ("Momentum", momentum_patterns, "#2ecc71"),
                ("Volume", volume_patterns, "#3498db"),
                ("Position", position_patterns, "#f39c12"),
                ("Fundamental", fundamental_patterns, "#9b59b6"),
                ("Warnings", warning_patterns, "#e74c3c")
            ]
            
            for group_name, group_patterns, color in pattern_groups:
                if group_patterns:
                    st.markdown(f"**{group_name}:**")
                    pattern_html = " ".join([
                        f'<span style="background-color: {color}; color: white; padding: 2px 8px; '
                        f'border-radius: 12px; margin: 2px; display: inline-block; font-size: 12px;">'
                        f'{p}</span>'
                        for p in group_patterns
                    ])
                    st.markdown(pattern_html, unsafe_allow_html=True)
        
        if show_detailed:
            # Additional detailed tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Scores", "💰 Fundamentals", "📊 Performance", 
                "🔍 Technicals", "🎯 Patterns", "🌊 Advanced"
            ])
            
            with tab1:
                # Detailed score breakdown
                score_df = pd.DataFrame({
                    'Component': ['Position', 'Volume', 'Momentum', 'Acceleration', 'Breakout', 'RVOL'],
                    'Score': [
                        stock.get('position_score', 50),
                        stock.get('volume_score', 50),
                        stock.get('momentum_score', 50),
                        stock.get('acceleration_score', 50),
                        stock.get('breakout_score', 50),
                        stock.get('rvol_score', 50)
                    ],
                    'Weight': [
                        CONFIG.POSITION_WEIGHT,
                        CONFIG.VOLUME_WEIGHT,
                        CONFIG.MOMENTUM_WEIGHT,
                        CONFIG.ACCELERATION_WEIGHT,
                        CONFIG.BREAKOUT_WEIGHT,
                        CONFIG.RVOL_WEIGHT
                    ]
                })
                score_df['Contribution'] = score_df['Score'] * score_df['Weight']
                
                fig = px.bar(score_df, x='Score', y='Component', orientation='h',
                            title="Score Component Breakdown",
                            color='Score', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Fundamental data
                fund_col1, fund_col2 = st.columns(2)
                
                with fund_col1:
                    if 'pe' in stock and pd.notna(stock['pe']):
                        st.metric("P/E Ratio", f"{stock['pe']:.1f}")
                    if 'eps_current' in stock and pd.notna(stock['eps_current']):
                        st.metric("EPS", f"₹{stock['eps_current']:.2f}")
                
                with fund_col2:
                    if 'eps_change_pct' in stock and pd.notna(stock['eps_change_pct']):
                        st.metric("EPS Growth", f"{stock['eps_change_pct']:.1f}%")
                    if 'price' in stock:
                        st.metric("Price", f"₹{stock['price']:.2f}")
            
            with tab3:
                # Performance timeline
                perf_data = []
                timeframes = [
                    ('1D', 'ret_1d'),
                    ('3D', 'ret_3d'),
                    ('7D', 'ret_7d'),
                    ('30D', 'ret_30d'),
                    ('3M', 'ret_3m'),
                    ('6M', 'ret_6m'),
                    ('1Y', 'ret_1y')
                ]
                
                for label, col in timeframes:
                    if col in stock and pd.notna(stock[col]):
                        perf_data.append({
                            'Period': label,
                            'Return': stock[col],
                            'Color': 'green' if stock[col] > 0 else 'red'
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    fig = px.bar(perf_df, x='Period', y='Return', color='Color',
                                color_discrete_map={'green': '#2ecc71', 'red': '#e74c3c'},
                                title="Performance Timeline")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Technical indicators
                tech_col1, tech_col2 = st.columns(2)
                
                with tech_col1:
                    # 52-week range position
                    if all(x in stock for x in ['from_low_pct', 'from_high_pct']):
                        st.markdown("**52-Week Range Position**")
                        
                        # Visual range slider
                        position_pct = stock['from_low_pct'] / (stock['from_low_pct'] + abs(stock['from_high_pct'])) * 100
                        
                        st.slider(
                            "Position in Range",
                            0, 100, int(position_pct),
                            disabled=True,
                            help=f"Low: {stock['from_low_pct']:.0f}% | High: {stock['from_high_pct']:.0f}%"
                        )
                
                with tech_col2:
                    # SMA positions
                    if 'price' in stock:
                        st.markdown("**SMA Analysis**")
                        sma_status = UIComponents.get_detailed_sma_status(stock)
                        st.markdown(sma_status)
            
            with tab5:
                # Pattern details
                if 'patterns' in stock and stock['patterns']:
                    pattern_list = stock['patterns'].split(' | ')
                    st.markdown(f"**Total Patterns: {len(pattern_list)}**")
                    
                    # Pattern explanations
                    pattern_info = {
                        '🔥 CAT LEADER': "Top 10% performer within its category",
                        '💎 HIDDEN GEM': "Strong category performer with lower market rank",
                        '🚀 ACCELERATING': "Momentum is rapidly increasing",
                        '⚡ VOL EXPLOSION': "Trading volume surge > 3x normal",
                        '🌊 MOMENTUM WAVE': "Strong sustained momentum trend",
                        '👑 MARKET LEADER': "Top 5% overall market performer"
                    }
                    
                    for pattern in pattern_list:
                        if pattern in pattern_info:
                            st.info(f"{pattern}: {pattern_info[pattern]}")
                        else:
                            st.info(pattern)
            
            with tab6:
                # Advanced metrics
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    if 'vmi' in stock:
                        st.metric("Volume Momentum Index", f"{stock['vmi']:.2f}")
                    if 'position_tension' in stock:
                        st.metric("Position Tension", f"{stock['position_tension']:.0f}")
                    if 'trend_quality' in stock:
                        quality_gauge = "🟢" if stock['trend_quality'] >= 70 else "🟡" if stock['trend_quality'] >= 40 else "🔴"
                        st.metric("Trend Quality", f"{quality_gauge} {stock['trend_quality']:.0f}")
                
                with adv_col2:
                    if 'momentum_harmony' in stock:
                        harmony_level = ["🔴", "🟠", "🟡", "🟢", "💚"][int(stock['momentum_harmony'])]
                        st.metric("Momentum Harmony", f"{harmony_level} {stock['momentum_harmony']}/4")
                    if 'money_flow_mm' in stock:
                        st.metric("Money Flow", f"₹{stock['money_flow_mm']:.1f}M")
                    if 'liquidity_score' in stock:
                        st.metric("Liquidity Score", f"{stock['liquidity_score']:.0f}")
    
    @staticmethod
    def get_sma_position(stock: pd.Series) -> str:
        """Get formatted SMA position string (from old version)"""
        if 'from_low_pct' in stock and 'from_high_pct' in stock:
            return f"Low+{stock['from_low_pct']:.0f}% | High{stock['from_high_pct']:.0f}%"
        return "N/A"
    
    @staticmethod
    def get_detailed_sma_status(stock: pd.Series) -> str:
        """Get detailed SMA status with percentage above/below"""
        if 'price' not in stock:
            return "No price data"
        
        price = stock['price']
        status_parts = []
        
        for sma_col, sma_label in [('sma_20d', '20D'), ('sma_50d', '50D'), ('sma_200d', '200D')]:
            if sma_col in stock and pd.notna(stock[sma_col]):
                sma_value = stock[sma_col]
                pct_diff = ((price - sma_value) / sma_value) * 100
                
                if price > sma_value:
                    status_parts.append(f"✅ {sma_label} +{pct_diff:.1f}%")
                else:
                    status_parts.append(f"❌ {sma_label} {pct_diff:.1f}%")
        
        return " | ".join(status_parts) if status_parts else "No SMA data"
    
    @staticmethod
    def render_search_results(results: pd.DataFrame) -> None:
        """Render beautiful search results display (from paste-2.txt)"""
        
        if results.empty:
            st.info("No stocks found matching your search.")
            return
        
        st.markdown(f"### 🔍 Search Results ({len(results)} found)")
        
        for idx, stock in results.iterrows():
            with st.container():
                # Use the complete search results UI from paste-2.txt
                UIComponents.render_stock_card(stock, show_detailed=True)
                st.markdown("---")

# ============================================
# SESSION STATE MANAGEMENT
# ============================================

class SessionStateManager:
    """Manage session state for the application"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        
        # Core state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.data_source = 'sheet'
            st.session_state.selected_tab = 0
            st.session_state.search_query = ''
            st.session_state.export_template = 'full'
            st.session_state.last_update = None
            st.session_state.quick_filter = None
            st.session_state.quick_filter_applied = False
            
            # Filter states
            st.session_state.filters = {
                'categories': [],
                'sectors': [],
                'min_score': 0,
                'patterns': [],
                'wave_states': [],
                'wave_strength_range': (0, 100)
            }
            
            # Display preferences
            st.session_state.show_patterns = True
            st.session_state.show_fundamentals = False
            st.session_state.mobile_view = False
            
            # Performance tracking
            st.session_state.performance_metrics = {}
            
            # Cache management
            st.session_state.last_good_data = None
            st.session_state.data_quality = {}
    
    @staticmethod
    def reset_filters():
        """Reset all filters to default"""
        st.session_state.filters = {
            'categories': [],
            'sectors': [],
            'min_score': 0,
            'patterns': [],
            'wave_states': [],
            'wave_strength_range': (0, 100)
        }
        st.session_state.quick_filter = None
        st.session_state.quick_filter_applied = False
    
    @staticmethod
    def save_session():
        """Save current session configuration"""
        config = {
            'filters': st.session_state.filters,
            'data_source': st.session_state.data_source,
            'export_template': st.session_state.export_template,
            'show_patterns': st.session_state.show_patterns,
            'show_fundamentals': st.session_state.show_fundamentals,
            'timestamp': datetime.now().isoformat()
        }
        return config
    
    @staticmethod
    def load_session(config: Dict[str, Any]):
        """Load session configuration"""
        if 'filters' in config:
            st.session_state.filters = config['filters']
        if 'data_source' in config:
            st.session_state.data_source = config['data_source']
        if 'export_template' in config:
            st.session_state.export_template = config['export_template']
        if 'show_patterns' in config:
            st.session_state.show_patterns = config['show_patterns']
        if 'show_fundamentals' in config:
            st.session_state.show_fundamentals = config['show_fundamentals']

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    /* Main content styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Metric card styling */
    [data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    
    /* Expander styling */
    .streamlit-expander {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main > div {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    SessionStateManager.initialize()
    
    # Sidebar
    with st.sidebar:
        st.title("🌊 Wave Detection Ultimate 3.0")
        st.markdown("**Professional Stock Ranking System**")
        st.markdown("---")
        
        # Data Source Selection (Enhanced from requirement #2)
        st.markdown("### 📂 Select Data Source")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Google Sheets", 
                        use_container_width=True,
                        type="primary" if st.session_state.data_source == "sheet" else "secondary"):
                st.session_state.data_source = "sheet"
                st.rerun()
        with col2:
            if st.button("📁 Upload CSV", 
                        use_container_width=True,
                        type="primary" if st.session_state.data_source == "upload" else "secondary"):
                st.session_state.data_source = "upload"
                st.rerun()
        
        uploaded_file = None
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload your watchlist CSV file"
            )
            
            if uploaded_file is None:
                st.warning("Please upload a CSV file to continue")
        else:
            # Google Sheets configuration
            with st.expander("📊 Sheet Settings", expanded=False):
                sheet_url = st.text_input(
                    "Google Sheet URL",
                    value=CONFIG.DEFAULT_SHEET_URL,
                    help="Enter your Google Sheets URL"
                )
                gid = st.text_input(
                    "Sheet GID",
                    value=CONFIG.DEFAULT_GID,
                    help="Sheet ID from the URL"
                )
        
        st.markdown("---")
        
        # Display Options
        st.markdown("### 🎨 Display Options")
        
        st.session_state.show_patterns = st.checkbox(
            "Show Pattern Detection",
            value=st.session_state.show_patterns,
            help="Display pattern badges in rankings"
        )
        
        st.session_state.show_fundamentals = st.checkbox(
            "Show Fundamental Data",
            value=st.session_state.show_fundamentals,
            help="Include P/E and EPS data"
        )
        
        st.session_state.mobile_view = st.checkbox(
            "Mobile View",
            value=st.session_state.mobile_view,
            help="Simplified view for mobile devices"
        )
        
        st.markdown("---")
        
        # Export Options
        st.markdown("### 💾 Export Template")
        st.session_state.export_template = st.selectbox(
            "Select Export Template",
            options=['full', 'day_trader', 'swing_trader', 'investor'],
            format_func=lambda x: {
                'full': '📊 Full Analysis',
                'day_trader': '⚡ Day Trader Focus',
                'swing_trader': '🎯 Swing Trader Focus',
                'investor': '💰 Investor Focus'
            }.get(x, x),
            index=['full', 'day_trader', 'swing_trader', 'investor'].index(st.session_state.export_template)
        )
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **Wave Detection Ultimate 3.0**
            
            Professional stock ranking system with:
            - Master Score 3.0 algorithm
            - 25 pattern detections
            - Real-time market intelligence
            - Advanced filtering system
            - Wave state analysis
            
            **Version**: 3.0-FINAL
            **Status**: Production Ready
            """)
    
    # Main content area
    try:
        # Load and process data
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.info("👆 Please upload a CSV file in the sidebar to begin analysis")
            return
        
        # Load data with caching
        with st.spinner("Loading and processing data..."):
            df, timestamp, metadata = load_and_process_data(
                source_type=st.session_state.data_source,
                file_data=uploaded_file,
                sheet_url=sheet_url if st.session_state.data_source == "sheet" else None,
                gid=gid if st.session_state.data_source == "sheet" else None,
                data_version="3.0"
            )
        
        # Display data quality info
        if metadata.get('warnings'):
            with st.expander("⚠️ Data Quality Warnings", expanded=False):
                for warning in metadata['warnings']:
                    st.warning(warning)
        
        # Update timestamp
        st.session_state.last_update = timestamp
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Rankings", "🔍 Search", "📈 Analytics", 
            "🏢 Sectors", "📁 Categories", "🌊 Wave Radar", "💾 Export"
        ])
        
        # Tab 1: Rankings
        with tab1:
            # Quick Actions Bar
            st.markdown("### ⚡ Quick Actions")
            qa_cols = st.columns(6)
            
            with qa_cols[0]:
                if st.button("🔥 Top Momentum", use_container_width=True):
                    st.session_state['quick_filter'] = 'top_momentum'
                    st.session_state['quick_filter_applied'] = True
                    st.rerun()
            
            with qa_cols[1]:
                if st.button("💎 Hidden Gems", use_container_width=True):
                    st.session_state['quick_filter'] = 'hidden_gems'
                    st.session_state['quick_filter_applied'] = True
                    st.rerun()
            
            with qa_cols[2]:
                if st.button("⚡ Volume Surges", use_container_width=True):
                    st.session_state['quick_filter'] = 'volume_surges'
                    st.session_state['quick_filter_applied'] = True
                    st.rerun()
            
            with qa_cols[3]:
                if st.button("🎯 Ready to Break", use_container_width=True):
                    st.session_state['quick_filter'] = 'breakout_ready'
                    st.session_state['quick_filter_applied'] = True
                    st.rerun()
            
            with qa_cols[4]:
                if st.button("👑 Market Leaders", use_container_width=True):
                    st.session_state['quick_filter'] = 'market_leaders'
                    st.session_state['quick_filter_applied'] = True
                    st.rerun()
            
            with qa_cols[5]:
                if st.button("🌊 Cresting Waves", use_container_width=True):
                    st.session_state['quick_filter'] = 'cresting_waves'
                    st.session_state['quick_filter_applied'] = True
                    st.rerun()
            
            # Apply quick filters
            ranked_df = df.copy()
            quick_filter = st.session_state.get('quick_filter')
            
            if st.session_state.get('quick_filter_applied') and quick_filter:
                if quick_filter == 'top_momentum':
                    ranked_df = ranked_df[ranked_df['momentum_score'] >= 80]
                    st.info(f"Showing {len(ranked_df)} stocks with momentum score ≥ 80")
                elif quick_filter == 'hidden_gems':
                    ranked_df = ranked_df[ranked_df['patterns'].str.contains('HIDDEN GEM', na=False)]
                    st.info(f"Showing {len(ranked_df)} hidden gem stocks")
                elif quick_filter == 'volume_surges':
                    ranked_df = ranked_df[ranked_df['rvol'] >= 2.5]
                    st.info(f"Showing {len(ranked_df)} stocks with RVOL ≥ 2.5x")
                elif quick_filter == 'breakout_ready':
                    ranked_df = ranked_df[ranked_df['breakout_score'] >= 80]
                    st.info(f"Showing {len(ranked_df)} breakout candidates")
                elif quick_filter == 'market_leaders':
                    ranked_df = ranked_df[ranked_df['percentile'] >= 90]
                    st.info(f"Showing {len(ranked_df)} market leaders (top 10%)")
                elif quick_filter == 'cresting_waves':
                    ranked_df = ranked_df[ranked_df['wave_state'] == '🌊🌊🌊 CRESTING']
                    st.info(f"Showing {len(ranked_df)} stocks in CRESTING wave state")
                
                # Clear quick filter button
                if st.button("❌ Clear Quick Filter"):
                    st.session_state['quick_filter'] = None
                    st.session_state['quick_filter_applied'] = False
                    st.rerun()
            
            # Advanced Filters Section
            with st.expander("🔧 Advanced Filters", expanded=False):
                filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                
                with filter_col1:
                    # Category filter
                    available_categories = FilterEngine.get_filter_options(
                        df, 'category', st.session_state.filters
                    )
                    if available_categories:
                        selected_categories = st.multiselect(
                            "Categories",
                            options=['All'] + available_categories,
                            default=st.session_state.filters.get('categories', []),
                            key='category_filter'
                        )
                        if 'All' not in selected_categories and selected_categories:
                            st.session_state.filters['categories'] = selected_categories
                        else:
                            st.session_state.filters['categories'] = []
                    
                    # Sector filter
                    available_sectors = FilterEngine.get_filter_options(
                        df, 'sector', st.session_state.filters
                    )
                    if available_sectors:
                        selected_sectors = st.multiselect(
                            "Sectors",
                            options=['All'] + available_sectors,
                            default=st.session_state.filters.get('sectors', []),
                            key='sector_filter'
                        )
                        if 'All' not in selected_sectors and selected_sectors:
                            st.session_state.filters['sectors'] = selected_sectors
                        else:
                            st.session_state.filters['sectors'] = []
                
                with filter_col2:
                    # Score filter
                    min_score = st.slider(
                        "Minimum Master Score",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filters.get('min_score', 0),
                        step=5,
                        key='score_filter'
                    )
                    st.session_state.filters['min_score'] = min_score
                    
                    # Pattern filter
                    if 'patterns' in df.columns:
                        all_patterns = set()
                        for patterns in df['patterns'].dropna():
                            if patterns:
                                all_patterns.update(patterns.split(' | '))
                        
                        if all_patterns:
                            selected_patterns = st.multiselect(
                                "Filter by Patterns",
                                options=sorted(all_patterns),
                                default=st.session_state.filters.get('patterns', []),
                                key='pattern_filter'
                            )
                            st.session_state.filters['patterns'] = selected_patterns
                
                with filter_col3:
                    # Wave state filter (from requirement #11)
                    if 'wave_state' in df.columns:
                        wave_states = df['wave_state'].unique().tolist()
                        
                        wave_state_info = {
                            '🌊🌊🌊 CRESTING': 'Strongest momentum (4+ signals)',
                            '🌊🌊 BUILDING': 'Strong momentum (3 signals)',
                            '🌊 FORMING': 'Emerging momentum (1-2 signals)',
                            '💥 BREAKING': 'No momentum signals'
                        }
                        
                        selected_wave_states = st.multiselect(
                            "Wave State Filter",
                            options=wave_states,
                            default=st.session_state.filters.get('wave_states', []),
                            format_func=lambda x: f"{x} - {wave_state_info.get(x, '')}" if x in wave_state_info else x,
                            placeholder="Select wave states",
                            key="wave_state_filter"
                        )
                        st.session_state.filters['wave_states'] = selected_wave_states
                    
                    # Wave strength filter
                    wave_strength_range = st.slider(
                        "Wave Strength Range",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filters.get('wave_strength_range', (0, 100)),
                        step=5,
                        key="wave_strength_filter"
                    )
                    st.session_state.filters['wave_strength_range'] = wave_strength_range
                
                with filter_col4:
                    # Fundamental filters
                    if st.session_state.show_fundamentals:
                        # PE filter
                        if 'pe' in df.columns:
                            col1, col2 = st.columns(2)
                            with col1:
                                min_pe = st.number_input(
                                    "Min P/E",
                                    min_value=0.0,
                                    value=st.session_state.filters.get('min_pe', 0.0),
                                    step=1.0,
                                    key='min_pe_filter'
                                )
                                st.session_state.filters['min_pe'] = min_pe if min_pe > 0 else None
                            
                            with col2:
                                max_pe = st.number_input(
                                    "Max P/E",
                                    min_value=0.0,
                                    value=st.session_state.filters.get('max_pe', 100.0),
                                    step=1.0,
                                    key='max_pe_filter'
                                )
                                st.session_state.filters['max_pe'] = max_pe if max_pe > 0 else None
                        
                        # EPS change filter
                        if 'eps_change_pct' in df.columns:
                            min_eps_change = st.number_input(
                                "Min EPS Growth %",
                                value=st.session_state.filters.get('min_eps_change', 0.0),
                                step=10.0,
                                key='eps_change_filter'
                            )
                            st.session_state.filters['min_eps_change'] = min_eps_change
                
                # Reset filters button
                col1, col2, col3, col4 = st.columns(4)
                with col4:
                    if st.button("🔄 Reset All Filters", use_container_width=True):
                        SessionStateManager.reset_filters()
                        st.rerun()
            
            # Apply all filters
            if not st.session_state.get('quick_filter_applied'):
                ranked_df = FilterEngine.apply_filters(ranked_df, st.session_state.filters)
            
            # Filter status display (from old version)
            active_filters = []
            if st.session_state.filters.get('categories'):
                active_filters.append(f"Categories: {len(st.session_state.filters['categories'])}")
            if st.session_state.filters.get('sectors'):
                active_filters.append(f"Sectors: {len(st.session_state.filters['sectors'])}")
            if st.session_state.filters.get('min_score', 0) > 0:
                active_filters.append(f"Score ≥ {st.session_state.filters['min_score']}")
            if st.session_state.filters.get('patterns'):
                active_filters.append(f"Patterns: {len(st.session_state.filters['patterns'])}")
            if st.session_state.filters.get('wave_states'):
                active_filters.append(f"Wave States: {len(st.session_state.filters['wave_states'])}")
            
            if active_filters:
                st.info(f"📌 Active filters: {' | '.join(active_filters)} | Showing {len(ranked_df)} of {len(df)} stocks")
            
            # Summary Dashboard
            UIComponents.render_summary_section(ranked_df)
            
            # Rankings Table
            st.markdown("### 📊 Master Rankings")
            
            # Select columns to display
            display_cols = ['rank', 'ticker', 'company_name', 'master_score']
            
            # Add score components
            display_cols.extend(['position_score', 'volume_score', 'momentum_score', 
                               'acceleration_score', 'breakout_score', 'rvol_score'])
            
            # Add key metrics
            if 'rvol' in ranked_df.columns:
                display_cols.append('rvol')
            if 'ret_30d' in ranked_df.columns:
                display_cols.append('ret_30d')
            if 'wave_state' in ranked_df.columns:
                display_cols.append('wave_state')
            
            # Add patterns if enabled
            if st.session_state.show_patterns and 'patterns' in ranked_df.columns:
                display_cols.append('patterns')
            
            # Add fundamentals if enabled
            if st.session_state.show_fundamentals:
                if 'pe' in ranked_df.columns:
                    display_cols.append('pe')
                if 'eps_change_pct' in ranked_df.columns:
                    display_cols.append('eps_change_pct')
            
            # Add category/sector
            display_cols.extend(['category', 'sector'])
            
            # Filter to available columns
            available_display_cols = [col for col in display_cols if col in ranked_df.columns]
            
            # Number of stocks to display
            n_display = st.selectbox(
                "Number of stocks to display",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(CONFIG.DEFAULT_TOP_N),
                key='n_display'
            )
            
            # Sort and display
            ranked_df_display = ranked_df.nlargest(min(n_display, len(ranked_df)), 'master_score')
            
            # Format the dataframe for display
            display_df = ranked_df_display[available_display_cols].copy()
            
            # Format numeric columns
            for col in display_df.select_dtypes(include=[np.number]).columns:
                if col == 'rank':
                    display_df[col] = display_df[col].astype(int)
                elif col in ['pe', 'eps_current']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
                elif col in CONFIG.PERCENTAGE_COLUMNS or 'ret_' in col or '_pct' in col:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
                elif col == 'rvol':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "")
                else:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
            
            # Display the table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600 if not st.session_state.mobile_view else 400
            )
            
            # Quick statistics (from old version)
            with st.expander("📊 Quick Statistics", expanded=False):
                if 'master_score' in ranked_df_display.columns:
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("Q1 (25%)", f"{ranked_df_display['master_score'].quantile(0.25):.1f}")
                    with stat_col2:
                        st.metric("Median", f"{ranked_df_display['master_score'].quantile(0.50):.1f}")
                    with stat_col3:
                        st.metric("Q3 (75%)", f"{ranked_df_display['master_score'].quantile(0.75):.1f}")
                    with stat_col4:
                        st.metric("Spread", f"{ranked_df_display['master_score'].quantile(0.75) - ranked_df_display['master_score'].quantile(0.25):.1f}")
            
            # Visualizations
            st.markdown("### 📈 Score Analysis")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Score distribution
                fig1 = Visualizer.create_score_distribution(ranked_df_display)
                st.plotly_chart(fig1, use_container_width=True)
            
            with viz_col2:
                # Master score breakdown
                fig2 = Visualizer.create_master_score_breakdown(ranked_df_display, n=20)
                st.plotly_chart(fig2, use_container_width=True)
        
        # Tab 2: Search (Enhanced with requirement #1)
        with tab2:
            st.markdown("### 🔍 Stock Search")
            
            search_col1, search_col2 = st.columns([3, 1])
            
            with search_col1:
                search_query = st.text_input(
                    "Search by ticker or company name",
                    placeholder="Enter ticker symbol or company name (e.g., RELIANCE or Infosys)",
                    key='search_input'
                )
            
            with search_col2:
                search_button = st.button("🔍 Search", use_container_width=True)
            
            if search_query and (search_button or st.session_state.search_query != search_query):
                st.session_state.search_query = search_query
                
                # Perform enhanced search
                search_results = SearchEngine.search_stocks(df, search_query)
                
                if not search_results.empty:
                    st.markdown(f"### Found {len(search_results)} results for '{search_query}'")
                    
                    # Display search results with enhanced UI (requirement #5)
                    UIComponents.render_search_results(search_results)
                else:
                    st.warning(f"No stocks found matching '{search_query}'")
                    
                    # Suggest alternatives
                    st.info("💡 **Search Tips:**")
                    st.write("- Try partial names (e.g., 'TATA' for Tata companies)")
                    st.write("- Use ticker symbols without spaces")
                    st.write("- Search works on both ticker and company name")
        
        # Tab 3: Analytics
        with tab3:
            st.markdown("### 📈 Market Analytics")
            
            # Market overview
            analytics_col1, analytics_col2 = st.columns([2, 1])
            
            with analytics_col1:
                # Sector performance scatter
                fig = Visualizer.create_sector_performance_scatter(df)
                st.plotly_chart(fig, use_container_width=True)
            
            with analytics_col2:
                # Market statistics
                st.markdown("### 📊 Market Statistics")
                
                # Score distribution stats
                score_stats = df['master_score'].describe()
                
                st.metric("Average Score", f"{score_stats['mean']:.1f}")
                st.metric("Median Score", f"{score_stats['50%']:.1f}")
                st.metric("Score Std Dev", f"{score_stats['std']:.1f}")
                
                # Market concentration
                top_10_pct = len(df[df['percentile'] >= 90]) / len(df) * 100
                st.metric("Top 10% Stocks", f"{top_10_pct:.1f}%")
                
                # Pattern emergence
                if 'patterns' in df.columns:
                    stocks_with_patterns = (df['patterns'] != '').sum()
                    pattern_pct = stocks_with_patterns / len(df) * 100
                    st.metric("Stocks with Patterns", f"{pattern_pct:.0f}%")
            
            # Acceleration profiles
            st.markdown("### 🚀 Momentum Acceleration Profiles")
            accel_fig = Visualizer.create_acceleration_profiles(df, n=15)
            st.plotly_chart(accel_fig, use_container_width=True)
        
        # Tab 4: Sectors (Enhanced with requirement #3)
        with tab4:
            st.markdown("### 🏢 Sector Analysis - Fair Percentile-Based Comparison")
            
            # Get sector rotation data
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            
            if sector_rotation.empty:
                st.warning("No sector data available")
            else:
                # Display sector metrics
                sector_display_cols = [
                    'sector', 'size_category', 'size', 'avg_market_percentile',
                    'market_leaders_pct', 'beat_market_pct', 'momentum_leaders',
                    'smart_money_pct', 'weak_stocks_pct', 'avg_rvol'
                ]
                
                # Format the dataframe
                sector_display = sector_rotation[sector_display_cols].copy()
                
                # Format percentages
                pct_cols = ['market_leaders_pct', 'beat_market_pct', 'momentum_leaders', 
                           'smart_money_pct', 'weak_stocks_pct']
                for col in pct_cols:
                    sector_display[col] = sector_display[col].apply(lambda x: f"{x:.1f}%")
                
                sector_display['avg_market_percentile'] = sector_display['avg_market_percentile'].apply(lambda x: f"{x:.1f}")
                sector_display['avg_rvol'] = sector_display['avg_rvol'].apply(lambda x: f"{x:.2f}x")
                
                # Rename columns for display
                sector_display.columns = [
                    'Sector', 'Size Cat', 'Stocks', 'Avg %ile',
                    'Leaders %', 'Beat Mkt %', 'Momentum %',
                    'Smart $ %', 'Weak %', 'Avg RVOL'
                ]
                
                st.dataframe(sector_display, use_container_width=True, height=500)
                
                # Sector deep dive
                st.markdown("### 🔍 Sector Deep Dive")
                
                selected_sector = st.selectbox(
                    "Select sector for detailed analysis",
                    options=sector_rotation['sector'].tolist()
                )
                
                if selected_sector:
                    sector_stocks = df[df['sector'] == selected_sector]
                    
                    # Sector summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Stocks", len(sector_stocks))
                    with col2:
                        st.metric("Avg Score", f"{sector_stocks['master_score'].mean():.1f}")
                    with col3:
                        top_10_count = len(sector_stocks[sector_stocks['percentile'] >= 90])
                        st.metric("Top 10% Stocks", top_10_count)
                    with col4:
                        avg_ret_30d = sector_stocks['ret_30d'].mean() if 'ret_30d' in sector_stocks.columns else 0
                        st.metric("Avg 30D Return", f"{avg_ret_30d:.1f}%")
                    
                    # Top performers in sector
                    st.markdown(f"### 🏆 Top 10 Stocks in {selected_sector}")
                    
                    sector_top_10 = sector_stocks.nlargest(10, 'master_score')
                    
                    display_cols = ['rank', 'ticker', 'company_name', 'master_score', 
                                   'momentum_score', 'rvol', 'patterns']
                    available_cols = [col for col in display_cols if col in sector_top_10.columns]
                    
                    st.dataframe(
                        sector_top_10[available_cols],
                        use_container_width=True,
                        height=400
                    )
        
        # Tab 5: Categories (Enhanced with requirement #3)
        with tab5:
            st.markdown("### 📁 Category Analysis - Fair Percentile-Based Comparison")
            
            # Get category rotation data
            category_rotation = MarketIntelligence.detect_category_rotation(df)
            
            if category_rotation.empty:
                st.warning("No category data available")
            else:
                # Display category metrics
                cat_display_cols = [
                    'category', 'size', 'avg_market_percentile',
                    'market_leaders_pct', 'beat_market_pct', 
                    'momentum_leaders', 'weak_stocks_pct'
                ]
                
                # Format the dataframe
                cat_display = category_rotation[cat_display_cols].copy()
                
                # Format percentages
                pct_cols = ['market_leaders_pct', 'beat_market_pct', 'momentum_leaders', 'weak_stocks_pct']
                for col in pct_cols:
                    cat_display[col] = cat_display[col].apply(lambda x: f"{x:.1f}%")
                
                cat_display['avg_market_percentile'] = cat_display['avg_market_percentile'].apply(lambda x: f"{x:.1f}")
                
                # Rename columns
                cat_display.columns = [
                    'Category', 'Stocks', 'Avg %ile',
                    'Leaders %', 'Beat Mkt %', 'Momentum %', 'Weak %'
                ]
                
                st.dataframe(cat_display, use_container_width=True)
                
                # Category comparison chart
                st.markdown("### 📊 Category Performance Comparison")
                
                fig = go.Figure()
                
                # Add bars for each metric
                metrics_to_plot = ['avg_market_percentile', 'market_leaders_pct', 'momentum_leaders']
                colors = ['#3498db', '#2ecc71', '#f39c12']
                
                for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
                    fig.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=category_rotation['category'],
                        y=category_rotation[metric],
                        marker_color=color
                    ))
                
                fig.update_layout(
                    title="Category Performance Metrics",
                    xaxis_title="Category",
                    yaxis_title="Percentage",
                    barmode='group',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 6: Wave Radar
        with tab6:
            st.markdown("### 🌊 Wave Radar - Momentum Detection System")
            
            # Wave signals
            wave_signals = df[
                ((df['momentum_score'] >= 60) | 
                 (df['acceleration_score'] >= 70) | 
                 (df['rvol'] >= 2))
            ].copy()
            
            if len(wave_signals) == 0:
                st.info("No active wave signals detected")
            else:
                # Calculate wave strength
                wave_signals['signal_count'] = (
                    (wave_signals['momentum_score'] >= 60).astype(int) +
                    (wave_signals['acceleration_score'] >= 70).astype(int) +
                    (wave_signals['rvol'] >= 2).astype(int) +
                    (wave_signals['breakout_score'] >= 70).astype(int)
                )
                
                # Sort by signal strength and master score
                wave_signals = wave_signals.sort_values(
                    ['signal_count', 'master_score'], 
                    ascending=[False, False]
                )
                
                # Display wave summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cresting = len(wave_signals[wave_signals['wave_state'] == '🌊🌊🌊 CRESTING'])
                    st.metric("🌊🌊🌊 Cresting", cresting)
                
                with col2:
                    building = len(wave_signals[wave_signals['wave_state'] == '🌊🌊 BUILDING'])
                    st.metric("🌊🌊 Building", building)
                
                with col3:
                    forming = len(wave_signals[wave_signals['wave_state'] == '🌊 FORMING'])
                    st.metric("🌊 Forming", forming)
                
                with col4:
                    st.metric("Total Signals", len(wave_signals))
                
                # Wave filters
                st.markdown("### 🎯 Wave Filters")
                
                wave_col1, wave_col2, wave_col3 = st.columns(3)
                
                with wave_col1:
                    min_signals = st.slider(
                        "Minimum Signal Count",
                        min_value=1,
                        max_value=4,
                        value=2,
                        key='wave_min_signals'
                    )
                
                with wave_col2:
                    wave_categories = st.multiselect(
                        "Categories",
                        options=['All'] + wave_signals['category'].unique().tolist(),
                        default=['All'],
                        key='wave_categories'
                    )
                
                with wave_col3:
                    show_only_cresting = st.checkbox(
                        "Show only CRESTING waves",
                        value=False,
                        key='wave_cresting_only'
                    )
                
                # Apply wave filters
                filtered_waves = wave_signals[wave_signals['signal_count'] >= min_signals]
                
                if 'All' not in wave_categories:
                    filtered_waves = filtered_waves[filtered_waves['category'].isin(wave_categories)]
                
                if show_only_cresting:
                    filtered_waves = filtered_waves[filtered_waves['wave_state'] == '🌊🌊🌊 CRESTING']
                
                # Display wave signals
                st.markdown(f"### 📡 Active Wave Signals ({len(filtered_waves)} stocks)")
                
                if len(filtered_waves) > 0:
                    # Prepare display columns
                    wave_display_cols = [
                        'ticker', 'company_name', 'wave_state', 'signal_count',
                        'master_score', 'momentum_score', 'acceleration_score',
                        'rvol', 'ret_30d', 'patterns', 'category'
                    ]
                    
                    available_wave_cols = [col for col in wave_display_cols if col in filtered_waves.columns]
                    
                    # Format for display
                    wave_display_df = filtered_waves[available_wave_cols].head(50).copy()
                    
                    # Apply formatting
                    if 'rvol' in wave_display_df.columns:
                        wave_display_df['rvol'] = wave_display_df['rvol'].apply(lambda x: f"{x:.1f}x")
                    if 'ret_30d' in wave_display_df.columns:
                        wave_display_df['ret_30d'] = wave_display_df['ret_30d'].apply(lambda x: f"{x:.1f}%")
                    
                    for score_col in ['master_score', 'momentum_score', 'acceleration_score']:
                        if score_col in wave_display_df.columns:
                            wave_display_df[score_col] = wave_display_df[score_col].apply(lambda x: f"{x:.0f}")
                    
                    st.dataframe(
                        wave_display_df,
                        use_container_width=True,
                        height=600
                    )
                    
                    # Wave visualization
                    st.markdown("### 🌊 Wave Strength Distribution")
                    
                    # Create wave strength chart
                    fig = go.Figure()
                    
                    # Add histogram
                    fig.add_trace(go.Histogram(
                        x=filtered_waves['wave_strength'],
                        nbinsx=20,
                        marker_color='#3498db',
                        name='Wave Strength'
                    ))
                    
                    fig.update_layout(
                        title="Distribution of Wave Strength",
                        xaxis_title="Wave Strength (%)",
                        yaxis_title="Number of Stocks",
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 7: Export
        with tab7:
            st.markdown("### 💾 Export Data")
            
            # Export options
            export_col1, export_col2 = st.columns([2, 1])
            
            with export_col1:
                st.markdown(f"""
                **Selected Template:** {st.session_state.export_template}
                
                Templates optimize the export for different trading styles:
                - **Full Analysis**: Complete data for comprehensive analysis
                - **Day Trader**: Focus on momentum, volume, and intraday signals
                - **Swing Trader**: Emphasis on position, breakouts, and patterns
                - **Investor**: Fundamental data and long-term performance
                """)
            
            with export_col2:
                # Data preview
                st.metric("Stocks to Export", len(ranked_df))
                st.metric("Data Completeness", f"{st.session_state.data_quality.get('completeness', 0):.1f}%")
            
            # Export buttons
            st.markdown("### 📥 Download Options")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                # Excel export
                if st.button("📊 Generate Excel Report", use_container_width=True):
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_buffer = ExportEngine.create_excel_report(
                                ranked_df, 
                                template=st.session_state.export_template
                            )
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_buffer,
                                file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            
                            st.success("✅ Excel report generated successfully!")
                        except Exception as e:
                            st.error(f"Error creating Excel report: {str(e)}")
            
            with download_col2:
                # CSV export
                if st.button("📄 Generate CSV Export", use_container_width=True):
                    try:
                        csv_data = ExportEngine.create_csv_export(ranked_df)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.success("✅ CSV export generated successfully!")
                    except Exception as e:
                        st.error(f"Error creating CSV: {str(e)}")
            
            with download_col3:
                # Session config export
                if st.button("💾 Save Session Config", use_container_width=True):
                    config = SessionStateManager.save_session()
                    config_json = json.dumps(config, indent=2)
                    
                    st.download_button(
                        label="📥 Download Config",
                        data=config_json,
                        file_name=f"wave_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    st.success("✅ Session configuration saved!")
            
            # Export preview
            with st.expander("👁️ Preview Export Data", expanded=False):
                preview_cols = ['rank', 'ticker', 'company_name', 'master_score', 
                              'momentum_score', 'rvol', 'patterns']
                available_preview_cols = [col for col in preview_cols if col in ranked_df.columns]
                
                st.dataframe(
                    ranked_df[available_preview_cols].head(20),
                    use_container_width=True
                )
        
        # Footer
        st.markdown("---")
        
        # Performance metrics
        if st.session_state.performance_metrics:
            with st.expander("⚡ Performance Metrics", expanded=False):
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                metrics = st.session_state.performance_metrics
                
                with perf_col1:
                    if 'process_dataframe' in metrics:
                        st.metric("Data Processing", f"{metrics['process_dataframe']:.2f}s")
                
                with perf_col2:
                    if 'calculate_all_scores' in metrics:
                        st.metric("Score Calculation", f"{metrics['calculate_all_scores']:.2f}s")
                
                with perf_col3:
                    if 'detect_all_patterns' in metrics:
                        st.metric("Pattern Detection", f"{metrics['detect_all_patterns']:.2f}s")
                
                with perf_col4:
                    if 'apply_filters' in metrics:
                        st.metric("Filter Application", f"{metrics.get('apply_filters', 0):.2f}s")
        
        # Data freshness indicator
        if st.session_state.last_update:
            time_since_update = (datetime.now(timezone.utc) - st.session_state.last_update).total_seconds() / 60
            
            if time_since_update < 5:
                freshness = "🟢 Just Updated"
            elif time_since_update < 60:
                freshness = f"🟢 Updated {int(time_since_update)} min ago"
            elif time_since_update < 1440:
                freshness = f"🟡 Updated {int(time_since_update/60)} hours ago"
            else:
                freshness = f"🔴 Updated {int(time_since_update/1440)} days ago"
            
            st.caption(f"Data Status: {freshness} | Source: {metadata.get('source', 'Unknown')}")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        
        # Provide helpful error messages (from old version)
        st.info("💡 **Common Issues:**")
        st.write("- **Google Sheets**: Ensure the sheet is publicly accessible")
        st.write("- **CSV Upload**: Check that all required columns are present")
        st.write("- **Data Format**: Numeric columns should not contain text")
        st.write("- **Memory**: Large datasets may require more processing time")
        
        if st.button("🔄 Refresh Application"):
            st.rerun()

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
