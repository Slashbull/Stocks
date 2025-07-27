"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION VERSION
=====================================================
Professional Stock Ranking System with Advanced Analytics
All bugs fixed, all features complete, production-ready
Handles 1791+ stocks with 41 data columns

Version: 3.0-FINAL-LOCKED
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
    
    try:
        # Load data based on source
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
                    return df, timestamp, metadata
                raise
        
        # Validate loaded data
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Process the data
        df = DataProcessor.process_dataframe(df, metadata)
        
        # Calculate all scores and rankings
        df = RankingEngine.calculate_all_scores(df)
        
        # Detect patterns
        df = PatternDetector.detect_all_patterns(df)
        
        # Add advanced metrics
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        # Final validation
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Store as last good data
        timestamp = datetime.now(timezone.utc)
        st.session_state.last_good_data = (df.copy(), timestamp, metadata)
        
        # Record processing time
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        # Clean up memory
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
        # Data is stored as percentage change (e.g., -56.61 means 56.61% decrease)
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
        """Optimized vectorized pattern detection for all 25 patterns"""
        
        n_stocks = len(df)
        # Pre-allocate boolean array for all patterns
        pattern_masks = np.zeros((n_stocks, 25), dtype=bool)
        pattern_names = []
        
        # Get all pattern definitions with vectorized masks
        pattern_idx = 0
        
        # 1. Category Leader
        if 'category_percentile' in df.columns:
            pattern_masks[:, pattern_idx] = df['category_percentile'].values >= CONFIG.PATTERN_THRESHOLDS['category_leader']
            pattern_names.append('🔥 CAT LEADER')
            pattern_idx += 1
        
        # 2. Hidden Gem
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            pattern_masks[:, pattern_idx] = (
                (df['category_percentile'].values >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
                (df['percentile'].values < 70)
            )
            pattern_names.append('💎 HIDDEN GEM')
            pattern_idx += 1
        
        # 3. Accelerating
        if 'acceleration_score' in df.columns:
            pattern_masks[:, pattern_idx] = df['acceleration_score'].values >= CONFIG.PATTERN_THRESHOLDS['acceleration']
            pattern_names.append('🚀 ACCELERATING')
            pattern_idx += 1
        
        # 4. Institutional
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            pattern_masks[:, pattern_idx] = (
                (df['volume_score'].values >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['vol_ratio_90d_180d'].values > 1.1)
            )
            pattern_names.append('🏦 INSTITUTIONAL')
            pattern_idx += 1
        
        # 5. Volume Explosion
        if 'rvol' in df.columns:
            pattern_masks[:, pattern_idx] = df['rvol'].values > 3
            pattern_names.append('⚡ VOL EXPLOSION')
            pattern_idx += 1
        
        # 6. Breakout Ready
        if 'breakout_score' in df.columns:
            pattern_masks[:, pattern_idx] = df['breakout_score'].values >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
            pattern_names.append('🎯 BREAKOUT')
            pattern_idx += 1
        
        # 7. Market Leader
        if 'percentile' in df.columns:
            pattern_masks[:, pattern_idx] = df['percentile'].values >= CONFIG.PATTERN_THRESHOLDS['market_leader']
            pattern_names.append('👑 MARKET LEADER')
            pattern_idx += 1
        
        # 8. Momentum Wave
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            pattern_masks[:, pattern_idx] = (
                (df['momentum_score'].values >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                (df['acceleration_score'].values >= 70)
            )
            pattern_names.append('🌊 MOMENTUM WAVE')
            pattern_idx += 1
        
        # 9. Liquid Leader
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            pattern_masks[:, pattern_idx] = (
                (df['liquidity_score'].values >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
                (df['percentile'].values >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
            )
            pattern_names.append('💰 LIQUID LEADER')
            pattern_idx += 1
        
        # 10. Long-term Strength
        if 'long_term_strength' in df.columns:
            pattern_masks[:, pattern_idx] = df['long_term_strength'].values >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            pattern_names.append('💪 LONG STRENGTH')
            pattern_idx += 1
        
        # 11. Quality Trend
        if 'trend_quality' in df.columns:
            pattern_masks[:, pattern_idx] = df['trend_quality'].values >= 80
            pattern_names.append('📈 QUALITY TREND')
            pattern_idx += 1
        
        # 12. Value Momentum (Fundamental)
        if 'pe' in df.columns and 'percentile' in df.columns:
            has_valid_pe = (df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)).values
            pattern_masks[:, pattern_idx] = has_valid_pe & (df['pe'].values < 15) & (df['master_score'].values >= 70)
            pattern_names.append('💎 VALUE MOMENTUM')
            pattern_idx += 1
        
        # 13. Earnings Rocket
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = df['eps_change_pct'].notna().values
            extreme_growth = has_eps_growth & (df['eps_change_pct'].values > 1000)
            normal_growth = has_eps_growth & (df['eps_change_pct'].values > 50) & (df['eps_change_pct'].values <= 1000)
            
            pattern_masks[:, pattern_idx] = (
                (extreme_growth & (df['acceleration_score'].values >= 80)) |
                (normal_growth & (df['acceleration_score'].values >= 70))
            )
            pattern_names.append('📊 EARNINGS ROCKET')
            pattern_idx += 1
        
        # 14. Quality Leader
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = (
                df['pe'].notna() & 
                df['eps_change_pct'].notna() & 
                (df['pe'] > 0) &
                (df['pe'] < 10000)
            ).values
            
            pattern_masks[:, pattern_idx] = (
                has_complete_data &
                (df['pe'].values >= 10) & (df['pe'].values <= 25) &
                (df['eps_change_pct'].values > 20) &
                (df['percentile'].values >= 80)
            )
            pattern_names.append('🏆 QUALITY LEADER')
            pattern_idx += 1
        
        # 15. Turnaround Play
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = df['eps_change_pct'].notna().values
            mega_turnaround = has_eps & (df['eps_change_pct'].values > 500) & (df['volume_score'].values >= 60)
            strong_turnaround = has_eps & (df['eps_change_pct'].values > 100) & (df['eps_change_pct'].values <= 500) & (df['volume_score'].values >= 70)
            
            pattern_masks[:, pattern_idx] = mega_turnaround | strong_turnaround
            pattern_names.append('⚡ TURNAROUND')
            pattern_idx += 1
        
        # 16. High PE Warning
        if 'pe' in df.columns:
            has_valid_pe = (df['pe'].notna() & (df['pe'] > 0)).values
            pattern_masks[:, pattern_idx] = has_valid_pe & (df['pe'].values > 100)
            pattern_names.append('⚠️ HIGH PE')
            pattern_idx += 1
        
        # 17. 52W High Approach
        if 'from_high_pct' in df.columns and 'volume_score' in df.columns and 'momentum_score' in df.columns:
            pattern_masks[:, pattern_idx] = (
                (df['from_high_pct'].values > -5) & 
                (df['volume_score'].values >= 70) & 
                (df['momentum_score'].values >= 60)
            )
            pattern_names.append('🎯 52W HIGH APPROACH')
            pattern_idx += 1
        
        # 18. 52W Low Bounce
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            pattern_masks[:, pattern_idx] = (
                (df['from_low_pct'].values < 20) & 
                (df['acceleration_score'].values >= 80) & 
                (df['ret_30d'].values > 10)
            )
            pattern_names.append('🔄 52W LOW BOUNCE')
            pattern_idx += 1
        
        # 19. Golden Zone
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            pattern_masks[:, pattern_idx] = (
                (df['from_low_pct'].values > 60) & 
                (df['from_high_pct'].values > -40) & 
                (df['trend_quality'].values >= 70)
            )
            pattern_names.append('👑 GOLDEN ZONE')
            pattern_idx += 1
        
        # 20. Volume Accumulation
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            pattern_masks[:, pattern_idx] = (
                (df['vol_ratio_30d_90d'].values > 1.2) & 
                (df['vol_ratio_90d_180d'].values > 1.1) & 
                (df['ret_30d'].values > 5)
            )
            pattern_names.append('📊 VOL ACCUMULATION')
            pattern_idx += 1
        
        # 21. Momentum Divergence
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(df['ret_7d'].values != 0, df['ret_7d'].values / 7, 0)
                daily_30d_pace = np.where(df['ret_30d'].values != 0, df['ret_30d'].values / 30, 0)
            
            pattern_masks[:, pattern_idx] = (
                (daily_7d_pace > daily_30d_pace * 1.5) & 
                (df['acceleration_score'].values >= 85) & 
                (df['rvol'].values > 2)
            )
            pattern_names.append('🔀 MOMENTUM DIVERGE')
            pattern_idx += 1
        
        # 22. Range Compression
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = np.where(
                    df['low_52w'].values > 0,
                    ((df['high_52w'].values - df['low_52w'].values) / df['low_52w'].values) * 100,
                    100
                )
            
            pattern_masks[:, pattern_idx] = (range_pct < 50) & (df['from_low_pct'].values > 30)
            pattern_names.append('🎯 RANGE COMPRESS')
            pattern_idx += 1
        
        # 23. Stealth Accumulator
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_ratio = np.where(df['ret_30d'].values != 0, df['ret_7d'].values / (df['ret_30d'].values / 4), 0)
            
            pattern_masks[:, pattern_idx] = (
                (df['vol_ratio_90d_180d'].values > 1.1) &
                (df['vol_ratio_30d_90d'].values >= 0.9) & (df['vol_ratio_30d_90d'].values <= 1.1) &
                (df['from_low_pct'].values > 40) &
                (ret_ratio > 1)
            )
            pattern_names.append('🤫 STEALTH')
            pattern_idx += 1
        
        # 24. Momentum Vampire
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pace_ratio = np.where(df['ret_7d'].values != 0, df['ret_1d'].values / (df['ret_7d'].values / 7), 0)
            
            pattern_masks[:, pattern_idx] = (
                (daily_pace_ratio > 2) &
                (df['rvol'].values > 3) &
                (df['from_high_pct'].values > -15) &
                df['category'].isin(['Small Cap', 'Micro Cap']).values
            )
            pattern_names.append('🧛 VAMPIRE')
            pattern_idx += 1
        
        # 25. Perfect Storm
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            pattern_masks[:, pattern_idx] = (
                (df['momentum_harmony'].values == 4) &
                (df['master_score'].values > 80)
            )
            pattern_names.append('⛈️ PERFECT STORM')
            pattern_idx += 1
        
        # Single-pass string creation
        df['patterns'] = ''
        pattern_masks_used = pattern_masks[:, :pattern_idx]  # Use only filled patterns
        
        for i in range(n_stocks):
            stock_patterns = [pattern_names[j] for j in range(pattern_idx) if pattern_masks_used[i, j]]
            if stock_patterns:
                df.loc[i, 'patterns'] = ' | '.join(stock_patterns)
        
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
        """Fair sector analysis using dynamic sampling based on sector size"""
        
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        
        # First, apply dynamic sampling based on sector size
        sector_dfs = []
        sampling_info = {}
        
        for sector in df['sector'].unique():
            if sector != 'Unknown':
                sector_df = df[df['sector'] == sector].copy()
                sector_size = len(sector_df)
                
                # Dynamic sampling logic based on sector size
                if sector_size <= 5:
                    # 1-5 stocks: Use ALL (100%)
                    sample_size = sector_size
                    sample_pct = 100
                elif sector_size <= 20:
                    # 6-20 stocks: Use 80%
                    sample_size = max(5, int(sector_size * 0.8))
                    sample_pct = 80
                elif sector_size <= 50:
                    # 21-50 stocks: Use 60%
                    sample_size = int(sector_size * 0.6)
                    sample_pct = 60
                elif sector_size <= 100:
                    # 51-100 stocks: Use 40%
                    sample_size = int(sector_size * 0.4)
                    sample_pct = 40
                else:
                    # 100+ stocks: Use 25% (max 50 stocks)
                    sample_size = min(50, int(sector_size * 0.25))
                    sample_pct = 25
                
                # Sample the top stocks by master score
                if sector_size > sample_size:
                    sector_df = sector_df.nlargest(sample_size, 'master_score')
                
                sector_dfs.append(sector_df)
                sampling_info[sector] = f"Top {sample_size}/{sector_size} ({sample_pct}%)"
        
        # Combine sampled sector data
        if sector_dfs:
            sampled_df = pd.concat(sector_dfs, ignore_index=True)
        else:
            sampled_df = df
        
        # Calculate sector metrics on sampled data
        sector_metrics = sampled_df.groupby('sector').agg({
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean',
            'money_flow_mm': 'sum' if 'money_flow_mm' in sampled_df.columns else 'count'
        }).round(2)
        
        # Flatten column names
        if 'money_flow_mm' in sampled_df.columns:
            sector_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                     'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow']
        else:
            sector_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                     'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'count2']
            sector_metrics = sector_metrics.drop('count2', axis=1)
        
        # Add original sector size and sampling info
        original_counts = df.groupby('sector').size()
        sector_metrics['total_stocks'] = original_counts
        sector_metrics['analyzed_stocks'] = sector_metrics['count']
        sector_metrics['sampling_method'] = pd.Series(sampling_info)
        
        # Calculate flow score with median for robustness
        sector_metrics['flow_score'] = (
            sector_metrics['avg_score'] * 0.3 +
            sector_metrics['median_score'] * 0.2 +
            sector_metrics['avg_momentum'] * 0.25 +
            sector_metrics['avg_volume'] * 0.25
        )
        
        # Rank sectors
        sector_metrics['rank'] = sector_metrics['flow_score'].rank(ascending=False)
        
        return sector_metrics.sort_values('flow_score', ascending=False)

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
    def create_pattern_distribution(df: pd.DataFrame) -> go.Figure:
        """Create pattern frequency distribution chart"""
        pattern_counts = {}
        for patterns in df['patterns'].dropna():
            if patterns:
                for p in patterns.split(' | '):
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
        
        if not pattern_counts:
            fig = go.Figure()
            fig.add_annotation(
                text="No patterns detected in current selection",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        pattern_df = pd.DataFrame(
            list(pattern_counts.items()),
            columns=['Pattern', 'Count']
        ).sort_values('Count', ascending=True).tail(15)
        
        fig = go.Figure([
            go.Bar(
                x=pattern_df['Count'],
                y=pattern_df['Pattern'],
                orientation='h',
                marker_color='#3498db',
                text=pattern_df['Count'],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Pattern Frequency Analysis",
            xaxis_title="Number of Stocks",
            yaxis_title="Pattern",
            template='plotly_white',
            height=400,
            margin=dict(l=150)
        )
        
        return fig

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
        
        # Wave State filter - NEW
        wave_states = filters.get('wave_states', [])
        if wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)
        
        # Wave Strength filter - NEW
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and len(wave_strength_range) == 2:
            min_wave, max_wave = wave_strength_range
            # Calculate wave strength on-the-fly if not in df
            if 'wave_strength' not in df.columns:
                wave_strength = pd.Series(0, index=df.index, dtype=float)
                if 'momentum_score' in df.columns:
                    wave_strength += (df['momentum_score'] >= 60).astype(int) * 25
                if 'acceleration_score' in df.columns:
                    wave_strength += (df['acceleration_score'] >= 70).astype(int) * 25
                if 'rvol' in df.columns:
                    wave_strength += (df['rvol'] >= 2).astype(int) * 25
                if 'volume_score' in df.columns:
                    wave_strength += (df['volume_score'] >= 70).astype(int) * 25
            else:
                wave_strength = df['wave_strength']
            
            mask &= (wave_strength >= min_wave) & (wave_strength <= max_wave)
        
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
# SEARCH ENGINE - OPTIMIZED WITH PARTIAL MATCHING
# ============================================

class SearchEngine:
    """Enhanced search functionality with partial matching and relevance scoring"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Enhanced search with relevance scoring and partial matching"""
        
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            # Sanitize input
            query = query.upper().strip()
            
            # Method 1: Direct ticker match (exact) - return immediately if found
            ticker_exact = df[df['ticker'].str.upper() == query]
            if not ticker_exact.empty:
                return ticker_exact
            
            # Method 2: Ticker contains query
            ticker_contains = df[df['ticker'].str.upper().str.contains(query, na=False, regex=False)]
            
            # Method 3: Company name contains query (case insensitive)
            company_contains = df[df['company_name'].str.upper().str.contains(query, na=False, regex=False)]
            
            # Method 4: Partial match at start of words in company name
            def word_starts_with(company_name):
                """Check if any word in company name starts with query"""
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
                all_matches = all_matches.copy()
                all_matches['relevance'] = 0
                
                # Ticker starts with query: +50 points
                ticker_starts = all_matches['ticker'].str.upper().str.startswith(query)
                all_matches.loc[ticker_starts, 'relevance'] += 50
                
                # Company name word starts with query: +30 points
                company_word_starts = all_matches['company_name'].apply(word_starts_with)
                all_matches.loc[company_word_starts, 'relevance'] += 30
                
                # Sort by relevance then master score
                return all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False]).drop('relevance', axis=1)
            
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
            'momentum_harmony', 'wave_state', 'patterns', 
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
        
        # 2. TODAY'S OPPORTUNITIES
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
        
        # 3. MARKET INTELLIGENCE
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
                    x=sector_rotation.index[:10],  # Top 10 sectors
                    y=sector_rotation['flow_score'][:10],
                    text=[f"{val:.1f}" for val in sector_rotation['flow_score'][:10]],
                    textposition='outside',
                    marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' 
                                 for score in sector_rotation['flow_score'][:10]],
                    hovertemplate=(
                        'Sector: %{x}<br>'
                        'Flow Score: %{y:.1f}<br>'
                        'Sampling: %{customdata}<extra></extra>'
                    ),
                    customdata=sector_rotation['sampling_method'][:10]
                ))
                
                fig.update_layout(
                    title="Sector Rotation Map - Smart Money Flow (Dynamic Sampling)",
                    xaxis_title="Sector",
                    yaxis_title="Flow Score",
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
            
            # Category rotation
            category_spread = regime_metrics.get('category_spread', 0)
            if category_spread > 10:
                signals.append("🔄 Small caps leading")
            elif category_spread < -10:
                signals.append("🛡️ Large caps defensive")
            
            # Volume signal
            avg_rvol = regime_metrics.get('avg_rvol', 1.0)
            if avg_rvol > 1.5:
                signals.append("🌊 High volume activity")
            
            # Pattern emergence
            pattern_count = (df['patterns'] != '').sum()
            if pattern_count > len(df) * 0.2:
                signals.append("🎯 Many patterns emerging")
            
            for signal in signals:
                st.write(signal)
            
            # Market strength meter
            st.markdown("**💪 Market Strength**")
            
            strength_score = (
                (breadth * 50) +
                (min(avg_rvol, 2) * 25) +
                ((pattern_count / len(df)) * 25)
            )
            
            if strength_score > 70:
                strength_meter = "🟢🟢🟢🟢🟢"
            elif strength_score > 50:
                strength_meter = "🟢🟢🟢🟢⚪"
            elif strength_score > 30:
                strength_meter = "🟢🟢🟢⚪⚪"
            else:
                strength_meter = "🟢🟢⚪⚪⚪"
            
            st.write(strength_meter)

# ============================================
# SESSION STATE MANAGER
# ============================================

class SessionStateManager:
    """Manage session state properly"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        
        defaults = {
            'search_query': "",
            'last_refresh': datetime.now(timezone.utc),
            'data_source': "sheet",
            'user_preferences': {
                'default_top_n': CONFIG.DEFAULT_TOP_N,
                'display_mode': 'Technical',
                'last_filters': {}
            },
            'filters': {},
            'active_filter_count': 0,
            'quick_filter': None,
            'quick_filter_applied': False,
            'show_debug': False,
            'performance_metrics': {},
            'data_quality': {}
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        """Clear all filter states properly"""
        
        # Reset all filter-related session state
        filter_keys = [
            'category_filter', 'sector_filter', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'patterns',
            'min_score', 'trend_filter', 'min_eps_change',
            'min_pe', 'max_pe', 'require_fundamental_data',
            'quick_filter', 'quick_filter_applied',
            'wave_states', 'wave_strength_range'  # NEW wave filters
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], str):
                    st.session_state[key] = ""
                elif isinstance(st.session_state[key], (int, float)):
                    st.session_state[key] = 0
                else:
                    st.session_state[key] = None
        
        # Reset filter dictionaries
        st.session_state.filters = {}
        st.session_state.active_filter_count = 0

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application - Final Production Version"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    SessionStateManager.initialize()
    
    # Custom CSS for production UI
    st.markdown("""
    <style>
    /* Production-ready CSS */
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
    /* Button styling */
    div.stButton > button {
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    /* Mobile responsive */
    @media (max-width: 768px) {
        .stDataFrame {font-size: 12px;}
        div[data-testid="metric-container"] {padding: 3%;}
        .main {padding: 0rem 0.5rem;}
    }
    /* Table optimization */
    .stDataFrame > div {overflow-x: auto;}
    /* Loading animation */
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
        <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 3.0</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Professional Stock Ranking System • Final Production Version
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now(timezone.utc)
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                gc.collect()  # Force garbage collection
                st.success("Cache cleared!")
                time.sleep(0.5)
                st.rerun()
        
        # Data source selection - ENHANCED WITH TOGGLE BUTTONS
        st.markdown("---")
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
        
        # Show upload widget immediately when CSV selected
        uploaded_file = None
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="Upload a CSV file with stock data. Must contain 'ticker' and 'price' columns."
            )
            if uploaded_file is None:
                st.info("Please upload a CSV file to continue")
        
        # Data quality indicator
        if st.session_state.data_quality:
            with st.expander("📊 Data Quality", expanded=True):
                quality = st.session_state.data_quality
                
                col1, col2 = st.columns(2)
                with col1:
                    completeness = quality.get('completeness', 0)
                    if completeness > 80:
                        emoji = "🟢"
                    elif completeness > 60:
                        emoji = "🟡"
                    else:
                        emoji = "🔴"
                    
                    st.metric("Completeness", f"{emoji} {completeness:.1f}%")
                    st.metric("Total Stocks", f"{quality.get('total_rows', 0):,}")
                
                with col2:
                    if 'timestamp' in quality:
                        age = datetime.now(timezone.utc) - quality['timestamp']
                        hours = age.total_seconds() / 3600
                        
                        if hours < 1:
                            freshness = "🟢 Fresh"
                        elif hours < 24:
                            freshness = "🟡 Recent"
                        else:
                            freshness = "🔴 Stale"
                        
                        st.metric("Data Age", freshness)
                    
                    duplicates = quality.get('duplicate_tickers', 0)
                    if duplicates > 0:
                        st.metric("Duplicates", f"⚠️ {duplicates}")
        
        # Performance metrics
        if st.session_state.performance_metrics:
            with st.expander("⚡ Performance"):
                perf = st.session_state.performance_metrics
                
                total_time = sum(perf.values())
                if total_time < 3:
                    perf_emoji = "🟢"
                elif total_time < 5:
                    perf_emoji = "🟡"
                else:
                    perf_emoji = "🔴"
                
                st.metric("Load Time", f"{perf_emoji} {total_time:.1f}s")
                
                # Show slowest operations
                if len(perf) > 0:
                    slowest = sorted(perf.items(), key=lambda x: x[1], reverse=True)[:3]
                    for func_name, elapsed in slowest:
                        if elapsed > 0.5:
                            st.caption(f"{func_name}: {elapsed:.2f}s")
        
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        # Count active filters
        active_filter_count = 0
        
        if st.session_state.get('quick_filter_applied', False):
            active_filter_count += 1
        
        # Check all filter states
        filter_checks = [
            ('category_filter', lambda x: x and len(x) > 0),
            ('sector_filter', lambda x: x and len(x) > 0),
            ('min_score', lambda x: x > 0),
            ('patterns', lambda x: x and len(x) > 0),
            ('trend_filter', lambda x: x != 'All Trends'),
            ('wave_states', lambda x: x and len(x) > 0),  # NEW
            ('wave_strength_range', lambda x: x is not None),  # NEW
            ('eps_tier_filter', lambda x: x and len(x) > 0),
            ('pe_tier_filter', lambda x: x and len(x) > 0),
            ('price_tier_filter', lambda x: x and len(x) > 0),
            ('min_eps_change', lambda x: x and x.strip() != ''),
            ('min_pe', lambda x: x and x.strip() != ''),
            ('max_pe', lambda x: x and x.strip() != ''),
            ('require_fundamental_data', lambda x: x),
        ]
        
        for key, check_func in filter_checks:
            if key in st.session_state and check_func(st.session_state[key]):
                active_filter_count += 1
        
        st.session_state.active_filter_count = active_filter_count
        
        # Show active filter count
        if active_filter_count > 0:
            st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**")
        
        # Clear filters button
        if st.button("🗑️ Clear All Filters", 
                    use_container_width=True, 
                    type="primary" if active_filter_count > 0 else "secondary"):
            SessionStateManager.clear_filters()
            st.success("✅ All filters cleared!")
            st.rerun()
        
        # Debug mode
        st.markdown("---")
        show_debug = st.checkbox("🐛 Show Debug Info", 
                               value=st.session_state.get('show_debug', False),
                               key="show_debug")
    
    # Data loading and processing
    try:
        # Check if we need to load data
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        # Load and process data with progress indicator
        with st.spinner("📥 Loading and processing data..."):
            progress_bar = st.progress(0)
            progress_bar.progress(10, "Loading data...")
            
            try:
                if st.session_state.data_source == "upload" and uploaded_file is not None:
                    ranked_df, data_timestamp, metadata = load_and_process_data(
                        "upload", file_data=uploaded_file
                    )
                else:
                    ranked_df, data_timestamp, metadata = load_and_process_data(
                        "sheet", 
                        sheet_url=CONFIG.DEFAULT_SHEET_URL, 
                        gid=CONFIG.DEFAULT_GID
                    )
                
                progress_bar.progress(50, "Processing data...")
                
                st.session_state.ranked_df = ranked_df
                st.session_state.data_timestamp = data_timestamp
                st.session_state.last_refresh = datetime.now(timezone.utc)
                
                progress_bar.progress(90, "Calculating metrics...")
                
                # Show any warnings or errors
                if metadata.get('warnings'):
                    for warning in metadata['warnings']:
                        st.warning(warning)
                
                if metadata.get('errors'):
                    for error in metadata['errors']:
                        st.error(error)
                
                progress_bar.progress(100, "Complete!")
                time.sleep(0.5)
                progress_bar.empty()
                
            except Exception as e:
                progress_bar.empty()
                logger.error(f"Failed to load data: {str(e)}")
                st.error(f"Failed to load data: {str(e)}")
                
                # Show common issues
                with st.expander("❓ Common Issues and Solutions"):
                    st.markdown("""
                    **Google Sheets Issues:**
                    - Check if the sheet is publicly accessible
                    - Verify the URL and GID are correct
                    - Try refreshing the page
                    
                    **CSV Upload Issues:**
                    - Ensure the file contains 'ticker' and 'price' columns
                    - Check for special characters in the CSV
                    - Verify the file isn't corrupted
                    
                    **Performance Issues:**
                    - Large datasets may take longer to process
                    - Try filtering to reduce data size
                    - Clear cache if experiencing issues
                    """)
                st.stop()
        
        # Get data from session state
        if 'ranked_df' not in st.session_state:
            st.error("No data loaded. Please refresh.")
            st.stop()
        
        ranked_df = st.session_state.ranked_df
        data_timestamp = st.session_state.data_timestamp
        
        # Apply filters - build filter dict from session state
        filters = {
            'categories': st.session_state.get('category_filter', []),
            'sectors': st.session_state.get('sector_filter', []),
            'min_score': st.session_state.get('min_score', 0),
            'patterns': st.session_state.get('patterns', []),
            'trend_filter': st.session_state.get('trend_filter', 'All Trends'),
            'trend_range': st.session_state.get('trend_range', [0, 100]),
            'wave_states': st.session_state.get('wave_states', []),  # NEW
            'wave_strength_range': st.session_state.get('wave_strength_range', None),  # NEW
            'eps_tiers': st.session_state.get('eps_tier_filter', []),
            'pe_tiers': st.session_state.get('pe_tier_filter', []),
            'price_tiers': st.session_state.get('price_tier_filter', []),
            'min_eps_change': st.session_state.get('min_eps_change'),
            'min_pe': st.session_state.get('min_pe'),
            'max_pe': st.session_state.get('max_pe'),
            'require_fundamental_data': st.session_state.get('require_fundamental_data', False),
            'quick_filter': st.session_state.get('quick_filter', None)
        }
        
        # Apply quick filter if active
        if st.session_state.get('quick_filter_applied', False) and filters['quick_filter']:
            if filters['quick_filter'] == 'momentum_surge':
                filters['min_score'] = max(filters['min_score'], 70)
                filtered_df = ranked_df[
                    (ranked_df['master_score'] >= 70) &
                    (ranked_df['momentum_score'] >= 70) &
                    (ranked_df['acceleration_score'] >= 70)
                ]
            elif filters['quick_filter'] == 'volume_explosion':
                filtered_df = ranked_df[ranked_df['rvol'] > 3]
            elif filters['quick_filter'] == 'category_leaders':
                filtered_df = ranked_df[ranked_df['category_percentile'] >= 90]
            elif filters['quick_filter'] == 'breakout_ready':
                filtered_df = ranked_df[ranked_df['breakout_score'] >= 80]
            elif filters['quick_filter'] == 'cresting_waves':  # NEW
                filtered_df = ranked_df[ranked_df['wave_state'] == '🌊🌊🌊 CRESTING']
            else:
                filtered_df = FilterEngine.apply_filters(ranked_df, filters)
        else:
            filtered_df = FilterEngine.apply_filters(ranked_df, filters)
        
        # Quick action buttons - INCLUDING NEW WAVE BUTTON
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            if st.button("🚀 Momentum Surge", use_container_width=True):
                st.session_state.quick_filter = 'momentum_surge'
                st.session_state.quick_filter_applied = True
                st.rerun()
        
        with col2:
            if st.button("⚡ Volume Explosions", use_container_width=True):
                st.session_state.quick_filter = 'volume_explosion'
                st.session_state.quick_filter_applied = True
                st.rerun()
        
        with col3:
            if st.button("👑 Category Leaders", use_container_width=True):
                st.session_state.quick_filter = 'category_leaders'
                st.session_state.quick_filter_applied = True
                st.rerun()
        
        with col4:
            if st.button("🎯 Breakout Ready", use_container_width=True):
                st.session_state.quick_filter = 'breakout_ready'
                st.session_state.quick_filter_applied = True
                st.rerun()
        
        with col5:
            if st.button("🌊 Cresting Waves", use_container_width=True):  # NEW
                st.session_state.quick_filter = 'cresting_waves'
                st.session_state.quick_filter_applied = True
                st.rerun()
        
        with col6:
            if st.button("📊 Top 100", use_container_width=True):
                st.session_state.quick_filter = None
                st.session_state.quick_filter_applied = False
                SessionStateManager.clear_filters()
                st.rerun()
        
        # Summary section
        UIComponents.render_summary_section(filtered_df)
        
        # Search bar
        st.markdown("---")
        st.markdown("### 🔍 Search Stocks")
        
        col1, col2 = st.columns([5, 1])
        with col1:
            search_query = st.text_input(
                "Search by ticker or company name",
                value=st.session_state.get('search_query', ''),
                placeholder="e.g., RELIANCE or Infosys",
                key="search_input"
            )
        
        with col2:
            if st.button("🔍 Search", type="primary", use_container_width=True):
                st.session_state.search_query = search_query
        
        # Display search results
        if search_query:
            search_results = SearchEngine.search_stocks(filtered_df, search_query)
            
            if not search_results.empty:
                st.markdown(f"### 🎯 Search Results ({len(search_results)} found)")
                
                for idx, stock in search_results.iterrows():
                    with st.container():
                        # Stock header
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"### {stock['ticker']} - {stock['company_name']}")
                        
                        with col2:
                            score_color = "🟢" if stock['master_score'] >= 70 else "🟡" if stock['master_score'] >= 50 else "🔴"
                            st.metric("Master Score", f"{score_color} {stock['master_score']:.1f}")
                        
                        with col3:
                            st.metric("Rank", f"#{int(stock['rank'])}")
                        
                        with col4:
                            st.metric("Percentile", f"{stock['percentile']:.0f}%")
                        
                        # Details sections
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**📊 Fundamentals**")
                            st.write(f"Price: ₹{stock['price']:,.2f}")
                            st.write(f"Category: {stock['category']}")
                            st.write(f"Sector: {stock.get('sector', 'N/A')}")
                            # Add PE and EPS - NEW
                            if 'pe' in stock and pd.notna(stock['pe']) and stock['pe'] > 0:
                                st.write(f"P/E Ratio: {stock['pe']:.1f}")
                            else:
                                st.write("P/E Ratio: N/A")
                            
                            if 'eps_current' in stock and pd.notna(stock['eps_current']):
                                st.write(f"EPS: ₹{stock['eps_current']:.2f}")
                            else:
                                st.write("EPS: N/A")
                        
                        with col2:
                            st.markdown("**📈 Performance**")
                            if 'ret_1d' in stock:
                                ret_1d_color = "🟢" if stock['ret_1d'] > 0 else "🔴"
                                st.write(f"1D: {ret_1d_color} {stock['ret_1d']:.1f}%")
                            if 'ret_30d' in stock:
                                ret_30d_color = "🟢" if stock['ret_30d'] > 0 else "🔴"
                                st.write(f"30D: {ret_30d_color} {stock['ret_30d']:.1f}%")
                            if 'ret_1y' in stock:
                                ret_1y_color = "🟢" if stock['ret_1y'] > 0 else "🔴"
                                st.write(f"1Y: {ret_1y_color} {stock['ret_1y']:.1f}%")
                            if 'rvol' in stock:
                                rvol_emoji = "🌊" if stock['rvol'] > 2 else "💧" if stock['rvol'] > 1.5 else "🏜️"
                                st.write(f"RVOL: {rvol_emoji} {stock['rvol']:.1f}x")
                        
                        with col3:
                            st.markdown("**🎯 Patterns**")
                            if pd.notna(stock.get('patterns', '')) and stock['patterns']:
                                patterns = stock['patterns'].split(' | ')
                                for pattern in patterns[:5]:  # Show max 5 patterns
                                    st.write(pattern)
                            else:
                                st.write("No patterns detected")
                        
                        # Technical analysis row - FIXED ALIGNMENT
                        st.markdown("---")
                        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                        
                        with tech_col1:
                            st.markdown("**📊 Trading Position**")
                            if 'from_high_pct' in stock:
                                st.write(f"From 52W High: {stock['from_high_pct']:.1f}%")
                            if 'from_low_pct' in stock:
                                st.write(f"From 52W Low: {stock['from_low_pct']:.1f}%")
                        
                        with tech_col2:
                            st.markdown("**📈 Trend Analysis**")
                            if 'trend_quality' in stock:
                                trend_emoji = "🟢" if stock['trend_quality'] >= 70 else "🟡" if stock['trend_quality'] >= 40 else "🔴"
                                st.write(f"Trend Quality: {trend_emoji} {stock['trend_quality']:.0f}")
                            if 'wave_state' in stock:
                                st.write(f"Wave State: {stock['wave_state']}")
                        
                        with tech_col3:
                            st.markdown("**🚀 Momentum**")
                            if 'momentum_score' in stock:
                                st.write(f"Momentum: {stock['momentum_score']:.0f}")
                            if 'acceleration_score' in stock:
                                st.write(f"Acceleration: {stock['acceleration_score']:.0f}")
                        
                        with tech_col4:
                            st.markdown("**🎯 Advanced Metrics**")
                            if 'breakout_score' in stock:
                                st.write(f"Breakout: {stock['breakout_score']:.0f}")
                            if 'momentum_harmony' in stock:
                                harmony_emoji = "🎵" if stock['momentum_harmony'] >= 3 else "🎶" if stock['momentum_harmony'] >= 2 else "🔇"
                                st.write(f"Harmony: {harmony_emoji} {stock['momentum_harmony']}/4")
                        
                        st.markdown("---")
            else:
                st.info(f"No stocks found matching '{search_query}'")
        
        # Main content tabs
        st.markdown("---")
        
        tabs = st.tabs([
            "📊 Rankings", 
            "📈 Analysis", 
            "🌊 Wave Radar", 
            "🎯 Patterns",
            "💾 Export",
            "🏭 Sector Analysis"  # NEW TAB
        ])
        
        # Tab 1: Rankings
        with tabs[0]:
            st.header("📊 Stock Rankings")
            
            # Display controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                top_n = st.selectbox(
                    "Number of stocks to display",
                    options=CONFIG.AVAILABLE_TOP_N,
                    index=CONFIG.AVAILABLE_TOP_N.index(CONFIG.DEFAULT_TOP_N),
                    key="top_n_display"
                )
            
            with col2:
                display_mode = st.selectbox(
                    "Display mode",
                    options=['Technical', 'Fundamental', 'Complete'],
                    index=0,
                    key="display_mode"
                )
            
            with col3:
                show_stats = st.checkbox("Show Statistics", value=True, key="show_stats")
            
            # Get display data
            display_df = filtered_df.head(top_n)
            
            if display_df.empty:
                st.warning("No stocks match the current filters")
            else:
                # Select columns based on display mode
                if display_mode == 'Technical':
                    display_cols = [
                        'rank', 'ticker', 'company_name', 'master_score',
                        'momentum_score', 'acceleration_score', 'volume_score',
                        'breakout_score', 'rvol', 'ret_1d', 'ret_30d',
                        'wave_state', 'patterns', 'category'
                    ]
                elif display_mode == 'Fundamental':
                    display_cols = [
                        'rank', 'ticker', 'company_name', 'master_score',
                        'price', 'pe', 'eps_current', 'eps_change_pct',
                        'ret_1y', 'long_term_strength', 'money_flow_mm',
                        'category', 'sector'
                    ]
                else:  # Complete
                    display_cols = [col for col in display_df.columns if col != 'relevance']
                
                # Filter to available columns
                available_cols = [col for col in display_cols if col in display_df.columns]
                
                # Format display
                format_dict = {}
                for col in available_cols:
                    if col in ['master_score', 'momentum_score', 'acceleration_score', 
                               'volume_score', 'breakout_score', 'position_score',
                               'rvol_score', 'trend_quality', 'long_term_strength',
                               'liquidity_score', 'percentile', 'category_percentile']:
                        format_dict[col] = '{:.1f}'
                    elif col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 
                                'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 
                                'from_low_pct', 'from_high_pct', 'eps_change_pct']:
                        format_dict[col] = '{:.1f}%'
                    elif col == 'rvol':
                        format_dict[col] = '{:.2f}x'
                    elif col == 'price':
                        format_dict[col] = '₹{:,.2f}'
                    elif col == 'money_flow_mm':
                        format_dict[col] = '₹{:,.1f}M'
                    elif col in ['pe', 'eps_current']:
                        format_dict[col] = '{:.2f}'
                
                # Display table
                st.dataframe(
                    display_df[available_cols].style.format(format_dict),
                    use_container_width=True,
                    height=600
                )
                
                # Statistics - ENHANCED FROM OLD VERSION
                if show_stats:
                    with st.expander("📊 Quick Statistics", expanded=True):
                        stat_cols = st.columns(4)
                        
                        with stat_cols[0]:
                            st.metric("Total Stocks", f"{len(filtered_df):,}")
                            st.metric("Avg Master Score", f"{filtered_df['master_score'].mean():.1f}")
                        
                        with stat_cols[1]:
                            high_momentum = len(filtered_df[filtered_df['momentum_score'] >= 70])
                            st.metric("High Momentum", f"{high_momentum:,}")
                            
                            if 'rvol' in filtered_df.columns:
                                high_vol = len(filtered_df[filtered_df['rvol'] > 2])
                                st.metric("High Volume (>2x)", f"{high_vol:,}")
                        
                        with stat_cols[2]:
                            if 'ret_30d' in filtered_df.columns:
                                positive_30d = len(filtered_df[filtered_df['ret_30d'] > 0])
                                st.metric("Positive 30D", f"{positive_30d:,}")
                            
                            pattern_count = (filtered_df['patterns'] != '').sum()
                            st.metric("With Patterns", f"{pattern_count:,}")
                        
                        with stat_cols[3]:
                            # Quartile stats
                            st.markdown("**Score Distribution:**")
                            q1 = filtered_df['master_score'].quantile(0.25)
                            median = filtered_df['master_score'].quantile(0.50)
                            q3 = filtered_df['master_score'].quantile(0.75)
                            st.write(f"Q1: {q1:.1f} | Med: {median:.1f} | Q3: {q3:.1f}")
        
        # Tab 2: Analysis
        with tabs[1]:
            st.header("📈 Visual Analysis")
            
            # Analysis controls
            viz_col1, viz_col2 = st.columns([3, 1])
            
            with viz_col1:
                analysis_type = st.selectbox(
                    "Select analysis type",
                    ["Score Distribution", "Pattern Frequency", "Acceleration Profiles"],
                    key="analysis_type"
                )
            
            with viz_col2:
                show_top_n = st.number_input(
                    "Top N for profiles",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=5,
                    key="profile_top_n"
                )
            
            # Create visualizations
            if analysis_type == "Score Distribution":
                fig = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional insights
                st.markdown("### 📊 Component Analysis")
                
                comp_cols = st.columns(6)
                components = [
                    ('position_score', 'Position', comp_cols[0]),
                    ('volume_score', 'Volume', comp_cols[1]),
                    ('momentum_score', 'Momentum', comp_cols[2]),
                    ('acceleration_score', 'Acceleration', comp_cols[3]),
                    ('breakout_score', 'Breakout', comp_cols[4]),
                    ('rvol_score', 'RVOL', comp_cols[5])
                ]
                
                for score_col, label, col in components:
                    if score_col in filtered_df.columns:
                        avg_score = filtered_df[score_col].mean()
                        
                        with col:
                            if avg_score >= 70:
                                emoji = "🟢"
                            elif avg_score >= 50:
                                emoji = "🟡"
                            else:
                                emoji = "🔴"
                            
                            st.metric(label, f"{emoji} {avg_score:.1f}")
            
            elif analysis_type == "Pattern Frequency":
                fig = Visualizer.create_pattern_distribution(filtered_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Pattern insights
                st.markdown("### 🎯 Pattern Insights")
                
                total_patterns = 0
                pattern_stocks = 0
                
                for patterns in filtered_df['patterns']:
                    if patterns:
                        pattern_stocks += 1
                        total_patterns += len(patterns.split(' | '))
                
                insight_cols = st.columns(3)
                
                with insight_cols[0]:
                    st.metric("Stocks with Patterns", f"{pattern_stocks:,}")
                
                with insight_cols[1]:
                    st.metric("Total Pattern Signals", f"{total_patterns:,}")
                
                with insight_cols[2]:
                    avg_patterns = total_patterns / pattern_stocks if pattern_stocks > 0 else 0
                    st.metric("Avg Patterns/Stock", f"{avg_patterns:.1f}")
            
            else:  # Acceleration Profiles
                fig = Visualizer.create_acceleration_profiles(filtered_df, show_top_n)
                st.plotly_chart(fig, use_container_width=True)
                
                # Acceleration insights
                st.markdown("### 🚀 Momentum Analysis")
                
                accel_cols = st.columns(4)
                
                with accel_cols[0]:
                    perfect_accel = len(filtered_df[filtered_df['acceleration_score'] >= 85])
                    st.metric("Perfect Acceleration", f"{perfect_accel:,}")
                
                with accel_cols[1]:
                    building = len(filtered_df[
                        (filtered_df['acceleration_score'] >= 70) &
                        (filtered_df['momentum_score'] >= 60)
                    ])
                    st.metric("Building Momentum", f"{building:,}")
                
                with accel_cols[2]:
                    if 'momentum_harmony' in filtered_df.columns:
                        harmony = len(filtered_df[filtered_df['momentum_harmony'] >= 3])
                        st.metric("High Harmony (3+)", f"{harmony:,}")
                
                with accel_cols[3]:
                    if all(col in filtered_df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
                        consistent = len(filtered_df[
                            (filtered_df['ret_1d'] > 0) &
                            (filtered_df['ret_7d'] > 0) &
                            (filtered_df['ret_30d'] > 0)
                        ])
                        st.metric("All Green", f"{consistent:,}")
        
        # Tab 3: Wave Radar
        with tabs[2]:
            st.header("🌊 Wave Radar - Momentum Detection")
            
            # Wave controls
            wave_col1, wave_col2, wave_col3 = st.columns([2, 1, 1])
            
            with wave_col1:
                wave_threshold = st.slider(
                    "Minimum wave signals",
                    min_value=1,
                    max_value=4,
                    value=2,
                    key="wave_threshold"
                )
            
            with wave_col2:
                show_wave_count = st.number_input(
                    "Display top",
                    min_value=10,
                    max_value=100,
                    value=25,
                    step=5,
                    key="wave_count"
                )
            
            with wave_col3:
                wave_sort = st.selectbox(
                    "Sort by",
                    ["Master Score", "RVOL", "Momentum"],
                    key="wave_sort"
                )
            
            # Calculate wave signals
            wave_df = filtered_df.copy()
            
            # Count signals for each stock
            wave_df['wave_signals'] = 0
            
            if 'momentum_score' in wave_df.columns:
                wave_df['wave_signals'] += (wave_df['momentum_score'] >= 60).astype(int)
            
            if 'acceleration_score' in wave_df.columns:
                wave_df['wave_signals'] += (wave_df['acceleration_score'] >= 70).astype(int)
            
            if 'rvol' in wave_df.columns:
                wave_df['wave_signals'] += (wave_df['rvol'] >= 2).astype(int)
            
            if 'volume_score' in wave_df.columns:
                wave_df['wave_signals'] += (wave_df['volume_score'] >= 70).astype(int)
            
            # Filter by threshold
            wave_df = wave_df[wave_df['wave_signals'] >= wave_threshold]
            
            # Sort
            if wave_sort == "RVOL":
                wave_df = wave_df.sort_values('rvol', ascending=False)
            elif wave_sort == "Momentum":
                wave_df = wave_df.sort_values('momentum_score', ascending=False)
            else:
                wave_df = wave_df.sort_values('master_score', ascending=False)
            
            # Display results
            if wave_df.empty:
                st.info(f"No stocks with {wave_threshold}+ wave signals")
            else:
                # Wave summary
                wave_summary_cols = st.columns(4)
                
                with wave_summary_cols[0]:
                    st.metric("Wave Stocks", f"{len(wave_df):,}")
                
                with wave_summary_cols[1]:
                    perfect_waves = len(wave_df[wave_df['wave_signals'] == 4])
                    st.metric("Perfect Waves (4/4)", f"{perfect_waves:,}")
                
                with wave_summary_cols[2]:
                    avg_rvol = wave_df['rvol'].mean() if 'rvol' in wave_df.columns else 0
                    st.metric("Avg RVOL", f"{avg_rvol:.1f}x")
                
                with wave_summary_cols[3]:
                    if 'money_flow_mm' in wave_df.columns:
                        total_flow = wave_df['money_flow_mm'].sum()
                        st.metric("Total Flow", f"₹{total_flow:,.0f}M")
                
                # Display wave stocks
                st.markdown("### 🌊 Active Wave Stocks")
                
                display_wave_cols = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'wave_signals', 'wave_state', 'momentum_score',
                    'acceleration_score', 'rvol', 'volume_score',
                    'ret_1d', 'ret_30d', 'patterns', 'category'
                ]
                
                available_wave_cols = [col for col in display_wave_cols if col in wave_df.columns]
                
                wave_format = {
                    'master_score': '{:.1f}',
                    'momentum_score': '{:.0f}',
                    'acceleration_score': '{:.0f}',
                    'volume_score': '{:.0f}',
                    'rvol': '{:.2f}x',
                    'ret_1d': '{:.1f}%',
                    'ret_30d': '{:.1f}%'
                }
                
                st.dataframe(
                    wave_df.head(show_wave_count)[available_wave_cols].style.format(wave_format),
                    use_container_width=True,
                    height=500
                )
                
                # Wave distribution chart
                st.markdown("### 📊 Wave Signal Distribution")
                
                wave_dist = wave_df['wave_signals'].value_counts().sort_index()
                
                fig = go.Figure([
                    go.Bar(
                        x=['1 Signal', '2 Signals', '3 Signals', '4 Signals'],
                        y=[wave_dist.get(i, 0) for i in range(1, 5)],
                        marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                        text=[wave_dist.get(i, 0) for i in range(1, 5)],
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Distribution of Wave Signals",
                    xaxis_title="Number of Signals",
                    yaxis_title="Stock Count",
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Patterns
        with tabs[3]:
            st.header("🎯 Pattern Analysis")
            
            # Pattern filter
            pattern_col1, pattern_col2 = st.columns([3, 1])
            
            with pattern_col1:
                # Get all unique patterns
                all_patterns = set()
                for patterns in filtered_df['patterns'].dropna():
                    if patterns:
                        all_patterns.update(patterns.split(' | '))
                
                selected_pattern = st.selectbox(
                    "Select pattern to analyze",
                    options=['All Patterns'] + sorted(list(all_patterns)),
                    key="pattern_select"
                )
            
            with pattern_col2:
                min_patterns = st.number_input(
                    "Min patterns/stock",
                    min_value=1,
                    max_value=10,
                    value=1,
                    key="min_patterns"
                )
            
            # Filter by pattern
            if selected_pattern == 'All Patterns':
                pattern_df = filtered_df[filtered_df['patterns'] != '']
            else:
                pattern_df = filtered_df[filtered_df['patterns'].str.contains(selected_pattern, na=False)]
            
            # Further filter by minimum pattern count
            if min_patterns > 1:
                pattern_df = pattern_df[
                    pattern_df['patterns'].str.count('\|') + 1 >= min_patterns
                ]
            
            if pattern_df.empty:
                st.info("No stocks match the pattern criteria")
            else:
                # Pattern summary
                st.markdown("### 📊 Pattern Summary")
                
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    st.metric("Stocks with Pattern", f"{len(pattern_df):,}")
                
                with summary_cols[1]:
                    avg_score = pattern_df['master_score'].mean()
                    st.metric("Avg Master Score", f"{avg_score:.1f}")
                
                with summary_cols[2]:
                    if 'ret_30d' in pattern_df.columns:
                        avg_return = pattern_df['ret_30d'].mean()
                        st.metric("Avg 30D Return", f"{avg_return:.1f}%")
                
                with summary_cols[3]:
                    # Most common category
                    top_category = pattern_df['category'].value_counts().index[0]
                    st.metric("Top Category", top_category)
                
                # Pattern performance analysis
                st.markdown("### 📈 Pattern Performance")
                
                # Calculate pattern statistics
                pattern_stats = []
                
                for pattern in all_patterns:
                    stocks_with_pattern = filtered_df[
                        filtered_df['patterns'].str.contains(pattern, na=False, regex=False)
                    ]
                    
                    if len(stocks_with_pattern) > 0:
                        stats = {
                            'Pattern': pattern,
                            'Count': len(stocks_with_pattern),
                            'Avg Score': stocks_with_pattern['master_score'].mean(),
                            'Avg 30D': stocks_with_pattern['ret_30d'].mean() if 'ret_30d' in stocks_with_pattern.columns else 0,
                            'Top Stock': stocks_with_pattern.nlargest(1, 'master_score')['ticker'].values[0]
                        }
                        pattern_stats.append(stats)
                
                if pattern_stats:
                    pattern_stats_df = pd.DataFrame(pattern_stats).sort_values('Count', ascending=False)
                    
                    # Format for display
                    pattern_format = {
                        'Avg Score': '{:.1f}',
                        'Avg 30D': '{:.1f}%'
                    }
                    
                    st.dataframe(
                        pattern_stats_df.style.format(pattern_format),
                        use_container_width=True,
                        height=400
                    )
                
                # Display stocks with selected pattern
                st.markdown(f"### 📋 Stocks with {selected_pattern}")
                
                display_pattern_cols = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'patterns', 'momentum_score', 'acceleration_score',
                    'ret_30d', 'rvol', 'category', 'sector'
                ]
                
                available_pattern_cols = [col for col in display_pattern_cols if col in pattern_df.columns]
                
                st.dataframe(
                    pattern_df.head(50)[available_pattern_cols],
                    use_container_width=True,
                    height=500
                )
        
        # Tab 5: Export
        with tabs[4]:
            st.header("💾 Export Data")
            
            export_col1, export_col2 = st.columns([2, 1])
            
            with export_col1:
                st.markdown("""
                ### 📊 Export Options
                
                Choose your export format and template based on your trading style:
                
                - **Excel Report**: Comprehensive multi-sheet analysis
                - **CSV Export**: Simple data for further analysis
                - **Day Trader**: Focus on momentum and volume
                - **Swing Trader**: Emphasis on position and breakouts
                - **Investor**: Fundamentals and long-term metrics
                """)
            
            with export_col2:
                export_template = st.selectbox(
                    "Select template",
                    ['Full Analysis', 'Day Trader', 'Swing Trader', 'Investor'],
                    key="export_template"
                )
                
                template_map = {
                    'Full Analysis': 'full',
                    'Day Trader': 'day_trader',
                    'Swing Trader': 'swing_trader',
                    'Investor': 'investor'
                }
            
            # Export buttons
            st.markdown("### 📥 Download Options")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_buffer = ExportEngine.create_excel_report(
                                filtered_df,
                                template_map[export_template]
                            )
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_buffer,
                                file_name=f"wave_detection_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            
                            st.success("✅ Excel report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error creating Excel report: {str(e)}")
                            logger.error(f"Excel export error: {str(e)}")
            
            with download_col2:
                if st.button("📄 Generate CSV Export", use_container_width=True):
                    try:
                        csv_data = ExportEngine.create_csv_export(filtered_df)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"wave_detection_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.success("✅ CSV export ready!")
                        
                    except Exception as e:
                        st.error(f"Error creating CSV: {str(e)}")
                        logger.error(f"CSV export error: {str(e)}")
            
            with download_col3:
                # Quick copy for top stocks
                if st.button("📋 Copy Top 10 Tickers", use_container_width=True):
                    top_tickers = filtered_df.head(10)['ticker'].tolist()
                    ticker_string = ', '.join(top_tickers)
                    
                    st.code(ticker_string, language=None)
                    st.info("📋 Tickers ready to copy!")
            
            # Export preview
            st.markdown("### 👁️ Export Preview")
            
            preview_df = filtered_df.head(10)
            
            if export_template == 'Day Trader':
                preview_cols = ['rank', 'ticker', 'master_score', 'rvol', 
                               'momentum_score', 'acceleration_score', 'wave_state']
            elif export_template == 'Swing Trader':
                preview_cols = ['rank', 'ticker', 'master_score', 'breakout_score',
                               'from_high_pct', 'trend_quality', 'patterns']
            elif export_template == 'Investor':
                preview_cols = ['rank', 'ticker', 'master_score', 'pe', 
                               'eps_change_pct', 'ret_1y', 'sector']
            else:
                preview_cols = ['rank', 'ticker', 'master_score', 'momentum_score',
                               'volume_score', 'patterns', 'category']
            
            available_preview_cols = [col for col in preview_cols if col in preview_df.columns]
            
            st.dataframe(
                preview_df[available_preview_cols],
                use_container_width=True,
                height=300
            )
        
        # Tab 6: Sector Analysis - NEW
        with tabs[5]:
            st.header("🏭 Sector Analysis")
            
            # Get sector rotation data
            sector_rotation = MarketIntelligence.detect_sector_rotation(filtered_df)
            
            if sector_rotation.empty:
                st.warning("No sector data available")
            else:
                # 1. Sector Performance Summary
                st.markdown("### 📊 Sector Performance Summary")
                
                # Format sector metrics for display
                sector_display = sector_rotation[['avg_score', 'median_score', 'avg_momentum', 
                                                 'avg_volume', 'avg_rvol', 'avg_ret_30d',
                                                 'total_stocks', 'analyzed_stocks', 'sampling_method']].copy()
                
                sector_format = {
                    'avg_score': '{:.1f}',
                    'median_score': '{:.1f}',
                    'avg_momentum': '{:.1f}',
                    'avg_volume': '{:.1f}',
                    'avg_rvol': '{:.2f}x',
                    'avg_ret_30d': '{:.1f}%'
                }
                
                st.dataframe(
                    sector_display.style.format(sector_format),
                    use_container_width=True,
                    height=400
                )
                
                # 2. Sector Rotation Indicators
                st.markdown("### 🔄 Sector Rotation Indicators")
                
                rot_col1, rot_col2, rot_col3 = st.columns(3)
                
                with rot_col1:
                    st.markdown("**🚀 Leading Sectors**")
                    top_sectors = sector_rotation.head(3)
                    for sector, row in top_sectors.iterrows():
                        st.write(f"• **{sector}**: {row['flow_score']:.1f}")
                        st.caption(f"  {row['sampling_method']}")
                
                with rot_col2:
                    st.markdown("**📉 Lagging Sectors**")
                    bottom_sectors = sector_rotation.tail(3)
                    for sector, row in bottom_sectors.iterrows():
                        st.write(f"• **{sector}**: {row['flow_score']:.1f}")
                        st.caption(f"  {row['sampling_method']}")
                
                with rot_col3:
                    st.markdown("**🌊 High Momentum**")
                    high_momentum = sector_rotation.nlargest(3, 'avg_momentum')
                    for sector, row in high_momentum.iterrows():
                        st.write(f"• **{sector}**: {row['avg_momentum']:.1f}")
                        st.caption(f"  30D: {row['avg_ret_30d']:.1f}%")
                
                # 3. Sector Deep Dive
                st.markdown("### 🔍 Sector Deep Dive")
                
                selected_sector = st.selectbox(
                    "Select sector to analyze",
                    options=sector_rotation.index.tolist(),
                    key="sector_deep_dive"
                )
                
                if selected_sector:
                    sector_stocks = filtered_df[filtered_df['sector'] == selected_sector].copy()
                    
                    # Sector overview
                    st.markdown(f"#### 📊 {selected_sector} Overview")
                    
                    overview_col1, overview_col2 = st.columns([3, 1])
                    
                    with overview_col1:
                        # Display logic based on sector size
                        total_in_sector = len(sector_stocks)
                        
                        if total_in_sector <= 25:
                            # Show all stocks if 25 or less
                            display_sector_df = sector_stocks
                            st.info(f"Showing all {total_in_sector} stocks in {selected_sector}")
                        else:
                            # Show top 25 with option to show all
                            display_sector_df = sector_stocks.nlargest(25, 'master_score')
                            
                            if st.checkbox(f"Show all {total_in_sector} stocks", key=f"show_all_{selected_sector}"):
                                display_sector_df = sector_stocks
                            else:
                                st.info(f"Showing top 25 of {total_in_sector} stocks. Check box above to see all.")
                        
                        # Stock table
                        sector_cols = ['rank', 'ticker', 'company_name', 'master_score',
                                     'momentum_score', 'volume_score', 'rvol',
                                     'ret_30d', 'wave_state', 'patterns', 'category']
                        
                        available_sector_cols = [col for col in sector_cols if col in display_sector_df.columns]
                        
                        st.dataframe(
                            display_sector_df[available_sector_cols],
                            use_container_width=True,
                            height=500
                        )
                    
                    with overview_col2:
                        # Sector statistics panel
                        st.markdown("**📈 Sector Stats**")
                        
                        st.metric("Total Stocks", f"{total_in_sector:,}")
                        st.metric("Avg Score", f"{sector_stocks['master_score'].mean():.1f}")
                        
                        if 'ret_30d' in sector_stocks.columns:
                            positive_30d = len(sector_stocks[sector_stocks['ret_30d'] > 0])
                            st.metric("Positive 30D", f"{positive_30d}/{total_in_sector}")
                        
                        # Category breakdown
                        st.markdown("**📊 Categories**")
                        cat_dist = sector_stocks['category'].value_counts()
                        for cat, count in cat_dist.items():
                            st.write(f"{cat}: {count}")
                        
                        # Wave state distribution
                        if 'wave_state' in sector_stocks.columns:
                            st.markdown("**🌊 Wave States**")
                            wave_dist = sector_stocks['wave_state'].value_counts()
                            for state, count in wave_dist.head(4).items():
                                st.write(f"{state}: {count}")
                    
                    # Common patterns in sector
                    st.markdown(f"#### 🎯 Common Patterns in {selected_sector}")
                    
                    sector_patterns = {}
                    for patterns in sector_stocks['patterns'].dropna():
                        if patterns:
                            for p in patterns.split(' | '):
                                sector_patterns[p] = sector_patterns.get(p, 0) + 1
                    
                    if sector_patterns:
                        pattern_df = pd.DataFrame(
                            list(sector_patterns.items()),
                            columns=['Pattern', 'Count']
                        ).sort_values('Count', ascending=False).head(10)
                        
                        fig = go.Figure([
                            go.Bar(
                                x=pattern_df['Count'],
                                y=pattern_df['Pattern'],
                                orientation='h',
                                marker_color='#9b59b6',
                                text=pattern_df['Count'],
                                textposition='outside'
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"Top Patterns in {selected_sector}",
                            xaxis_title="Frequency",
                            template='plotly_white',
                            height=300,
                            margin=dict(l=150)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Export sector data
                    if st.button(f"📥 Export {selected_sector} Data", use_container_width=True):
                        csv_data = sector_stocks.to_csv(index=False)
                        st.download_button(
                            label="Download Sector CSV",
                            data=csv_data,
                            file_name=f"{selected_sector.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
        
        # Debug information
        if show_debug:
            with st.expander("🐛 Debug Information"):
                st.markdown("### Session State")
                debug_state = {
                    'data_source': st.session_state.get('data_source'),
                    'last_refresh': st.session_state.get('last_refresh'),
                    'active_filters': st.session_state.get('active_filter_count'),
                    'total_stocks': len(ranked_df),
                    'filtered_stocks': len(filtered_df),
                    'data_timestamp': data_timestamp,
                    'data_quality': st.session_state.get('data_quality'),
                    'performance_metrics': st.session_state.get('performance_metrics')
                }
                st.json(debug_state)
                
                st.markdown("### Data Columns")
                st.write(f"Total columns: {len(filtered_df.columns)}")
                st.write(f"Columns: {', '.join(filtered_df.columns.tolist())}")
                
                st.markdown("### Memory Usage")
                memory_usage = filtered_df.memory_usage(deep=True).sum() / 1024 / 1024
                st.write(f"DataFrame memory: {memory_usage:.2f} MB")
        
        # Advanced filters in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎛️ Advanced Filters")
        
        # Category filter
        if 'category' in ranked_df.columns:
            available_categories = FilterEngine.get_filter_options(
                ranked_df, 'category', filters
            )
            
            category_filter = st.sidebar.multiselect(
                "Market Category",
                options=['All'] + available_categories,
                default=st.session_state.get('category_filter', []),
                key="category_filter"
            )
        
        # Sector filter
        if 'sector' in ranked_df.columns:
            available_sectors = FilterEngine.get_filter_options(
                ranked_df, 'sector', filters
            )
            
            sector_filter = st.sidebar.multiselect(
                "Sector",
                options=['All'] + available_sectors,
                default=st.session_state.get('sector_filter', []),
                key="sector_filter"
            )
        
        # Score filter
        min_score = st.sidebar.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=st.session_state.get('min_score', 0),
            step=5,
            key="min_score"
        )
        
        # Pattern filter
        if 'patterns' in ranked_df.columns:
            all_patterns = set()
            for patterns in ranked_df['patterns'].dropna():
                if patterns:
                    all_patterns.update(patterns.split(' | '))
            
            pattern_filter = st.sidebar.multiselect(
                "Patterns",
                options=sorted(list(all_patterns)),
                default=st.session_state.get('patterns', []),
                key="patterns"
            )
        
        # Trend strength filter
        st.sidebar.markdown("#### 🔰 Trend Strength Filter")
        
        trend_filter = st.sidebar.selectbox(
            "Trend Quality",
            options=['All Trends', 'Strong (70+)', 'Moderate (40-70)', 'Weak (<40)'],
            index=0,
            key="trend_filter"
        )
        
        if trend_filter != 'All Trends':
            if trend_filter == 'Strong (70+)':
                trend_range = [70, 100]
            elif trend_filter == 'Moderate (40-70)':
                trend_range = [40, 70]
            else:
                trend_range = [0, 40]
            
            st.session_state.trend_range = trend_range
        else:
            st.session_state.trend_range = [0, 100]
        
        # Wave State Filter - NEW
        st.sidebar.markdown("#### 🌊 Wave State Filter")
        
        wave_state_info = {
            '🌊🌊🌊 CRESTING': 'Strongest momentum (4+ signals)',
            '🌊🌊 BUILDING': 'Strong momentum (3 signals)',
            '🌊 FORMING': 'Emerging momentum (1-2 signals)',
            '💥 BREAKING': 'No momentum signals'
        }
        
        wave_states = st.sidebar.multiselect(
            "Select wave states",
            options=list(wave_state_info.keys()),
            default=st.session_state.get('wave_states', []),
            format_func=lambda x: f"{x} - {wave_state_info[x]}",
            key="wave_states"
        )
        
        # Wave strength range slider
        wave_strength_range = st.sidebar.slider(
            "Wave Strength Range",
            min_value=0,
            max_value=100,
            value=st.session_state.get('wave_strength_range', (0, 100)),
            step=25,
            key="wave_strength_range"
        )
        
        # Quick wave preset buttons
        wave_col1, wave_col2 = st.sidebar.columns(2)
        with wave_col1:
            if st.button("🌊 Strong Waves", use_container_width=True):
                st.session_state.wave_states = ['🌊🌊🌊 CRESTING', '🌊🌊 BUILDING']
                st.session_state.wave_strength_range = (50, 100)
                st.rerun()
        
        with wave_col2:
            if st.button("🚀 Cresting Only", use_container_width=True):
                st.session_state.wave_states = ['🌊🌊🌊 CRESTING']
                st.session_state.wave_strength_range = (75, 100)
                st.rerun()
        
        # Fundamental filters
        st.sidebar.markdown("#### 📊 Fundamental Filters")
        
        # EPS Change filter
        min_eps_change = st.sidebar.text_input(
            "Min EPS Change %",
            value=st.session_state.get('min_eps_change', ''),
            placeholder="e.g., 50",
            key="min_eps_change"
        )
        
        if min_eps_change:
            try:
                st.session_state.min_eps_change = float(min_eps_change)
            except ValueError:
                st.sidebar.error("Please enter a valid number")
        
        # PE filters
        pe_col1, pe_col2 = st.sidebar.columns(2)
        
        with pe_col1:
            min_pe = st.sidebar.text_input(
                "Min P/E",
                value=st.session_state.get('min_pe', ''),
                placeholder="e.g., 10",
                key="min_pe"
            )
            
            if min_pe:
                try:
                    st.session_state.min_pe = float(min_pe)
                except ValueError:
                    st.sidebar.error("Invalid number")
        
        with pe_col2:
            max_pe = st.sidebar.text_input(
                "Max P/E",
                value=st.session_state.get('max_pe', ''),
                placeholder="e.g., 30",
                key="max_pe"
            )
            
            if max_pe:
                try:
                    st.session_state.max_pe = float(max_pe)
                except ValueError:
                    st.sidebar.error("Invalid number")
        
        # Tier filters
        st.sidebar.markdown("#### 🏷️ Tier Filters")
        
        # EPS Tier
        if 'eps_tier' in ranked_df.columns:
            available_eps_tiers = FilterEngine.get_filter_options(
                ranked_df, 'eps_tier', filters
            )
            
            eps_tier_filter = st.sidebar.multiselect(
                "EPS Tier",
                options=['All'] + available_eps_tiers,
                default=st.session_state.get('eps_tier_filter', []),
                key="eps_tier_filter"
            )
        
        # PE Tier
        if 'pe_tier' in ranked_df.columns:
            available_pe_tiers = FilterEngine.get_filter_options(
                ranked_df, 'pe_tier', filters
            )
            
            pe_tier_filter = st.sidebar.multiselect(
                "P/E Tier",
                options=['All'] + available_pe_tiers,
                default=st.session_state.get('pe_tier_filter', []),
                key="pe_tier_filter"
            )
        
        # Price Tier
        if 'price_tier' in ranked_df.columns:
            available_price_tiers = FilterEngine.get_filter_options(
                ranked_df, 'price_tier', filters
            )
            
            price_tier_filter = st.sidebar.multiselect(
                "Price Tier",
                options=['All'] + available_price_tiers,
                default=st.session_state.get('price_tier_filter', []),
                key="price_tier_filter"
            )
        
        # Data completeness
        require_fundamental = st.sidebar.checkbox(
            "Require fundamental data",
            value=st.session_state.get('require_fundamental_data', False),
            key="require_fundamental_data"
        )
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; padding: 1rem;">
                <p>Wave Detection Ultimate 3.0 - Final Production Version</p>
                <p>Built with ❤️ for professional traders</p>
                <p>Last refresh: """ + st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S UTC') + """</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        
        # Error recovery suggestions
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("""
        Try these steps:
        1. Click **🔄 Refresh Data** in the sidebar
        2. Click **🧹 Clear Cache** to reset
        3. Check your data source settings
        4. Ensure CSV has required columns if uploading
        """)
        
        # Show debug info on error
        if st.checkbox("Show error details"):
            st.exception(e)


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
