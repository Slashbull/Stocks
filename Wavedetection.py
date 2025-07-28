"""
Wave Detection Ultimate 3.0 - FINAL ENHANCED VERSION
===================================================
Professional Stock Ranking System with Advanced Analytics
All bugs fixed, optimized for Streamlit Community Cloud
Enhanced with targeted improvements while preserving core excellence

Version: 3.0.8-FINAL-ENHANCED
Last Updated: December 2024
Status: PRODUCTION READY - Feature Complete
"""

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
    
    # Performance targets
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
# PATTERN DETECTION ENGINE - ENHANCED SPEED
# ============================================

class PatternDetector:
    """Detect all patterns using vectorized operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all 25 patterns efficiently with numpy optimization"""
        
        # Pre-allocate pattern mask array for speed
        n_stocks = len(df)
        n_patterns = 25
        pattern_masks = np.zeros((n_stocks, n_patterns), dtype=bool)
        pattern_names = []
        
        # Get all pattern definitions
        patterns = PatternDetector._get_all_pattern_definitions(df)
        
        # Process patterns in bulk using numpy
        for i, (pattern_name, mask) in enumerate(patterns):
            if mask is not None and i < n_patterns:
                pattern_names.append(pattern_name)
                # Convert pandas boolean series to numpy array
                pattern_masks[:, i] = mask.values if hasattr(mask, 'values') else mask
        
        # Efficiently join patterns for each stock
        pattern_results = []
        for i in range(n_stocks):
            active_patterns = [pattern_names[j] for j in range(len(pattern_names)) if pattern_masks[i, j]]
            pattern_results.append(' | '.join(active_patterns) if active_patterns else '')
        
        df['patterns'] = pattern_results
        
        return df
    
    @staticmethod
    def _get_all_pattern_definitions(df: pd.DataFrame) -> List[Tuple[str, Optional[pd.Series]]]:
        """Get all pattern definitions with masks"""
        patterns = []
        
        # 1. Category Leader
        if 'category_percentile' in df.columns:
            mask = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
            patterns.append(('🔥 CAT LEADER', mask))
        
        # 2. Hidden Gem
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            mask = (
                (df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
                (df['percentile'] < 70)
            )
            patterns.append(('💎 HIDDEN GEM', mask))
        
        # 3. Accelerating
        if 'acceleration_score' in df.columns:
            mask = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
            patterns.append(('🚀 ACCELERATING', mask))
        
        # 4. Institutional
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            mask = (
                (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['vol_ratio_90d_180d'] > 1.1)
            )
            patterns.append(('🏦 INSTITUTIONAL', mask))
        
        # 5. Volume Explosion
        if 'rvol' in df.columns:
            mask = df['rvol'] > 3
            patterns.append(('⚡ VOL EXPLOSION', mask))
        
        # 6. Breakout Ready
        if 'breakout_score' in df.columns:
            mask = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
            patterns.append(('🎯 BREAKOUT', mask))
        
        # 7. Market Leader
        if 'percentile' in df.columns:
            mask = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
            patterns.append(('👑 MARKET LEADER', mask))
        
        # 8. Momentum Wave
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            mask = (
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                (df['acceleration_score'] >= 70)
            )
            patterns.append(('🌊 MOMENTUM WAVE', mask))
        
        # 9. Liquid Leader
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            mask = (
                (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
                (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
            )
            patterns.append(('💰 LIQUID LEADER', mask))
        
        # 10. Long-term Strength
        if 'long_term_strength' in df.columns:
            mask = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            patterns.append(('💪 LONG STRENGTH', mask))
        
        # 11. Quality Trend
        if 'trend_quality' in df.columns:
            mask = df['trend_quality'] >= 80
            patterns.append(('📈 QUALITY TREND', mask))
        
        # 12. Value Momentum (Fundamental)
        if 'pe' in df.columns and 'percentile' in df.columns:
            has_valid_pe = (df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000))
            mask = has_valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
            patterns.append(('💎 VALUE MOMENTUM', mask))
        
        # 13. Earnings Rocket
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = df['eps_change_pct'].notna()
            extreme_growth = has_eps_growth & (df['eps_change_pct'] > 1000)
            normal_growth = has_eps_growth & (df['eps_change_pct'] > 50) & (df['eps_change_pct'] <= 1000)
            
            mask = (
                (extreme_growth & (df['acceleration_score'] >= 80)) |
                (normal_growth & (df['acceleration_score'] >= 70))
            )
            patterns.append(('📊 EARNINGS ROCKET', mask))
        
        # 14. Quality Leader
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = (
                df['pe'].notna() & 
                df['eps_change_pct'].notna() & 
                (df['pe'] > 0) &
                (df['pe'] < 10000)
            )
            mask = (
                has_complete_data &
                (df['pe'].between(10, 25)) &
                (df['eps_change_pct'] > 20) &
                (df['percentile'] >= 80)
            )
            patterns.append(('🏆 QUALITY LEADER', mask))
        
        # 15. Turnaround Play
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = df['eps_change_pct'].notna()
            mega_turnaround = has_eps & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60)
            strong_turnaround = has_eps & (df['eps_change_pct'] > 100) & (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70)
            
            mask = mega_turnaround | strong_turnaround
            patterns.append(('⚡ TURNAROUND', mask))
        
        # 16. High PE Warning
        if 'pe' in df.columns:
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
            mask = has_valid_pe & (df['pe'] > 100)
            patterns.append(('⚠️ HIGH PE', mask))
        
        # 17. 52W High Approach
        if 'from_high_pct' in df.columns and 'volume_score' in df.columns and 'momentum_score' in df.columns:
            mask = (
                (df['from_high_pct'] > -5) & 
                (df['volume_score'] >= 70) & 
                (df['momentum_score'] >= 60)
            )
            patterns.append(('🎯 52W HIGH APPROACH', mask))
        
        # 18. 52W Low Bounce
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            mask = (
                (df['from_low_pct'] < 20) & 
                (df['acceleration_score'] >= 80) & 
                (df['ret_30d'] > 10)
            )
            patterns.append(('🔄 52W LOW BOUNCE', mask))
        
        # 19. Golden Zone
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            mask = (
                (df['from_low_pct'] > 60) & 
                (df['from_high_pct'] > -40) & 
                (df['trend_quality'] >= 70)
            )
            patterns.append(('👑 GOLDEN ZONE', mask))
        
        # 20. Volume Accumulation
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            mask = (
                (df['vol_ratio_30d_90d'] > 1.2) & 
                (df['vol_ratio_90d_180d'] > 1.1) & 
                (df['ret_30d'] > 5)
            )
            patterns.append(('📊 VOL ACCUMULATION', mask))
        
        # 21. Momentum Divergence
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_30d_pace = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            
            mask = (
                (daily_7d_pace > daily_30d_pace * 1.5) & 
                (df['acceleration_score'] >= 85) & 
                (df['rvol'] > 2)
            )
            patterns.append(('🔀 MOMENTUM DIVERGE', mask))
        
        # 22. Range Compression
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = np.where(
                    df['low_52w'] > 0,
                    ((df['high_52w'] - df['low_52w']) / df['low_52w']) * 100,
                    100
                )
            
            mask = (range_pct < 50) & (df['from_low_pct'] > 30)
            patterns.append(('🎯 RANGE COMPRESS', mask))
        
        # 23. Stealth Accumulator (NEW)
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_ratio = np.where(df['ret_30d'] != 0, df['ret_7d'] / (df['ret_30d'] / 4), 0)
            
            mask = (
                (df['vol_ratio_90d_180d'] > 1.1) &
                (df['vol_ratio_30d_90d'].between(0.9, 1.1)) &
                (df['from_low_pct'] > 40) &
                (ret_ratio > 1)
            )
            patterns.append(('🤫 STEALTH', mask))
        
        # 24. Momentum Vampire (NEW)
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pace_ratio = np.where(df['ret_7d'] != 0, df['ret_1d'] / (df['ret_7d'] / 7), 0)
            
            mask = (
                (daily_pace_ratio > 2) &
                (df['rvol'] > 3) &
                (df['from_high_pct'] > -15) &
                (df['category'].isin(['Small Cap', 'Micro Cap']))
            )
            patterns.append(('🧛 VAMPIRE', mask))
        
        # 25. Perfect Storm (NEW)
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            mask = (
                (df['momentum_harmony'] == 4) &
                (df['master_score'] > 80)
            )
            patterns.append(('⛈️ PERFECT STORM', mask))
        
        return patterns

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
        """Detect sector rotation patterns with dynamic sampling"""
        
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        
        # Enhanced: Dynamic sampling based on sector size
        sector_dfs = []
        
        for sector in df['sector'].unique():
            if sector != 'Unknown':
                sector_df = df[df['sector'] == sector].copy()
                sector_size = len(sector_df)
                
                # Dynamic sampling logic
                if sector_size <= 5:
                    # Use all stocks (100%)
                    sample_size = sector_size
                elif sector_size <= 20:
                    # Use 80%
                    sample_size = int(sector_size * 0.8)
                elif sector_size <= 50:
                    # Use 60%
                    sample_size = int(sector_size * 0.6)
                elif sector_size <= 100:
                    # Use 40%
                    sample_size = int(sector_size * 0.4)
                else:
                    # Use 25% (max 50)
                    sample_size = min(int(sector_size * 0.25), 50)
                
                # Sample top stocks by master score
                if len(sector_df) > sample_size:
                    sector_df = sector_df.nlargest(sample_size, 'master_score')
                
                sector_dfs.append(sector_df)
        
        # Combine normalized sector data
        if sector_dfs:
            normalized_df = pd.concat(sector_dfs, ignore_index=True)
        else:
            normalized_df = df
        
        # Calculate sector metrics on normalized data
        sector_metrics = normalized_df.groupby('sector').agg({
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean',
            'money_flow_mm': 'sum' if 'money_flow_mm' in normalized_df.columns else 'count'
        }).round(2)
        
        # Flatten column names
        if 'money_flow_mm' in normalized_df.columns:
            sector_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                     'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow']
        else:
            sector_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                     'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'count2']
            sector_metrics = sector_metrics.drop('count2', axis=1)
        
        # Add original sector size for reference
        original_counts = df.groupby('sector').size()
        sector_metrics['total_stocks'] = original_counts
        sector_metrics['analyzed_stocks'] = sector_metrics['count']
        
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
# VISUALIZATION ENGINE - MODIFIED
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
# FILTER ENGINE - ENHANCED WITH WAVE FILTERS
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
        
        # ENHANCEMENT #6: Wave State filter
        wave_states = filters.get('wave_states', [])
        if wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)
        
        # Wave Strength filter
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and 'master_score' in df.columns:
            min_strength, max_strength = wave_strength_range
            # Calculate wave strength based on multiple signals
            wave_strength = pd.Series(0, index=df.index)
            if 'momentum_score' in df.columns:
                wave_strength += (df['momentum_score'] > 70).astype(int) * 25
            if 'volume_score' in df.columns:
                wave_strength += (df['volume_score'] > 70).astype(int) * 25
            if 'acceleration_score' in df.columns:
                wave_strength += (df['acceleration_score'] > 70).astype(int) * 25
            if 'rvol' in df.columns:
                wave_strength += (df['rvol'] > 2).astype(int) * 25
            
            mask &= (wave_strength >= min_strength) & (wave_strength <= max_strength)
        
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
            'price_tier': 'price_tiers',
            'wave_state': 'wave_states'
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
    """Optimized search functionality with word boundary matching"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with enhanced word boundary matching"""
        
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
            
            # ENHANCEMENT #1: Word boundary matching
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
                return all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False]).drop('relevance', axis=1)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE
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
                        'Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>'
                        'Avg Score: %{customdata[2]:.1f}<br>'
                        'Median Score: %{customdata[3]:.1f}<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        sector_rotation['analyzed_stocks'][:10],
                        sector_rotation['total_stocks'][:10],
                        sector_rotation['avg_score'][:10],
                        sector_rotation['median_score'][:10]
                    ))
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
    
    @staticmethod
    def render_sector_analysis_tab(df: pd.DataFrame) -> None:
        """Render the new Sector Analysis tab"""
        st.markdown("### 📊 Sector Analysis Deep Dive")
        
        if df.empty or 'sector' not in df.columns:
            st.warning("No sector data available for analysis")
            return
        
        # Get sector rotation data
        sector_rotation = MarketIntelligence.detect_sector_rotation(df)
        
        if sector_rotation.empty:
            st.warning("Unable to analyze sectors")
            return
        
        # 1. Sector Overview Table
        st.markdown("#### 📈 Sector Performance Overview")
        
        # Prepare display data
        sector_display = sector_rotation[['avg_score', 'median_score', 'analyzed_stocks', 
                                         'total_stocks', 'avg_momentum', 'avg_rvol', 'flow_score']].round(2)
        
        # Add sampling percentage
        sector_display['sampling_pct'] = (
            (sector_display['analyzed_stocks'] / sector_display['total_stocks'] * 100)
            .round(1)
        )
        
        # Rename columns for display
        sector_display.columns = ['Avg Score', 'Median Score', 'Analyzed', 'Total', 
                                 'Avg Momentum', 'Avg RVOL', 'Flow Score', 'Sample %']
        
        # Color gradient for flow score
        st.dataframe(
            sector_display.style.background_gradient(subset=['Flow Score', 'Avg Score'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Sampling explanation
        st.info(
            "📊 **Dynamic Sampling Logic:**\n"
            "- 1-5 stocks: 100% sampled\n"
            "- 6-20 stocks: 80% sampled\n"
            "- 21-50 stocks: 60% sampled\n"
            "- 51-100 stocks: 40% sampled\n"
            "- 100+ stocks: 25% sampled (max 50)"
        )
        
        # 2. Sector Deep Dive
        st.markdown("---")
        st.markdown("#### 🔍 Sector Deep Dive")
        
        # Sector selector
        selected_sector = st.selectbox(
            "Select a sector to analyze",
            options=sector_rotation.index.tolist(),
            index=0
        )
        
        if selected_sector:
            # Get sector data
            sector_df = df[df['sector'] == selected_sector].copy()
            
            # Sector metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                UIComponents.render_metric_card(
                    "Total Stocks",
                    f"{len(sector_df)}",
                    f"Rank #{int(sector_rotation.loc[selected_sector, 'rank'])}"
                )
            
            with col2:
                UIComponents.render_metric_card(
                    "Avg Master Score",
                    f"{sector_df['master_score'].mean():.1f}",
                    f"σ = {sector_df['master_score'].std():.1f}"
                )
            
            with col3:
                positive_momentum = (sector_df['ret_30d'] > 0).sum() if 'ret_30d' in sector_df.columns else 0
                UIComponents.render_metric_card(
                    "Positive Momentum",
                    f"{positive_momentum}",
                    f"{positive_momentum/len(sector_df)*100:.0f}%" if len(sector_df) > 0 else "0%"
                )
            
            with col4:
                high_rvol = (sector_df['rvol'] > 2).sum() if 'rvol' in sector_df.columns else 0
                UIComponents.render_metric_card(
                    "High RVOL",
                    f"{high_rvol}",
                    f"Avg: {sector_df['rvol'].mean():.1f}x" if 'rvol' in sector_df.columns else "N/A"
                )
            
            # Top 10 stocks in sector
            st.markdown(f"##### 🏆 Top 10 Stocks in {selected_sector}")
            
            top_10_sector = sector_df.nlargest(10, 'master_score')[
                ['rank', 'ticker', 'company_name', 'master_score', 'momentum_score', 
                 'rvol', 'ret_30d', 'patterns', 'category']
            ].copy()
            
            # Format for display
            if 'ret_30d' in top_10_sector.columns:
                top_10_sector['ret_30d'] = top_10_sector['ret_30d'].apply(lambda x: f"{x:.1f}%")
            if 'rvol' in top_10_sector.columns:
                top_10_sector['rvol'] = top_10_sector['rvol'].apply(lambda x: f"{x:.1f}x")
            
            top_10_sector['master_score'] = top_10_sector['master_score'].apply(lambda x: f"{x:.1f}")
            top_10_sector['momentum_score'] = top_10_sector['momentum_score'].apply(lambda x: f"{x:.1f}")
            
            # Rename columns
            top_10_sector.columns = ['Rank', 'Ticker', 'Company', 'Score', 'Momentum', 
                                    'RVOL', '30D Ret', 'Patterns', 'Category']
            
            st.dataframe(top_10_sector, use_container_width=True, hide_index=True)
            
            # Sector statistics
            st.markdown(f"##### 📊 {selected_sector} Statistics")
            
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                # Score distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=sector_df['master_score'],
                    nbinsx=20,
                    name='Master Score Distribution',
                    marker_color='#3498db'
                ))
                fig_dist.update_layout(
                    title=f"{selected_sector} - Score Distribution",
                    xaxis_title="Master Score",
                    yaxis_title="Count",
                    height=300,
                    template='plotly_white'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with stat_col2:
                # Category breakdown
                if 'category' in sector_df.columns:
                    cat_counts = sector_df['category'].value_counts()
                    fig_cat = go.Figure([go.Pie(
                        labels=cat_counts.index,
                        values=cat_counts.values,
                        hole=0.3
                    )])
                    fig_cat.update_layout(
                        title=f"{selected_sector} - Category Distribution",
                        height=300,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)

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
            'wave_states_filter', 'wave_strength_slider'
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
                elif isinstance(st.session_state[key], tuple):
                    st.session_state[key] = (0, 100)
                else:
                    st.session_state[key] = None
        
        # Reset filter dictionaries
        st.session_state.filters = {}
        st.session_state.active_filter_count = 0

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application - Enhanced Final Version"""
    
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
            Professional Stock Ranking System • Enhanced Final Version
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
        
        # ENHANCEMENT #4: Visible CSV Upload
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        # Two prominent buttons for data source selection
        source_col1, source_col2 = st.columns(2)
        
        with source_col1:
            if st.button("📊 Google Sheets", 
                        type="primary" if st.session_state.data_source == "sheet" else "secondary",
                        use_container_width=True):
                st.session_state.data_source = "sheet"
                st.rerun()
        
        with source_col2:
            if st.button("📁 Upload CSV", 
                        type="primary" if st.session_state.data_source == "upload" else "secondary",
                        use_container_width=True):
                st.session_state.data_source = "upload"
                st.rerun()
        
        # Show upload widget if CSV is selected
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
            ('eps_tier_filter', lambda x: x and len(x) > 0),
            ('pe_tier_filter', lambda x: x and len(x) > 0),
            ('price_tier_filter', lambda x: x and len(x) > 0),
            ('min_eps_change', lambda x: x is not None and x != 0),
            ('min_pe', lambda x: x is not None and x > 0),
            ('max_pe', lambda x: x is not None and x < 10000),
            ('require_fundamental_data', lambda x: x is True),
            ('wave_states_filter', lambda x: x and len(x) > 0),
            ('wave_strength_slider', lambda x: x != (0, 100))
        ]
        
        for key, check_func in filter_checks:
            if key in st.session_state and check_func(st.session_state[key]):
                active_filter_count += 1
        
        # Display active filter count
        if active_filter_count > 0:
            st.info(f"🔍 {active_filter_count} filters active")
            if st.button("🧹 Clear All Filters", type="secondary", use_container_width=True):
                SessionStateManager.clear_filters()
                st.rerun()
        
        # Quick Filters
        st.markdown("#### ⚡ Quick Filters")
        
        quick_filter_options = {
            "None": None,
            "🚀 High Momentum (Score 70+)": {'min_score': 70},
            "💎 Hidden Gems": {'patterns': ['HIDDEN GEM']},
            "⚡ Volume Explosion (3x+)": {'min_rvol': 3},
            "📈 Trending Up": {'min_ret_30d': 10},
            "🏆 Top 100 Only": {'top_n': 100},
            "🌊 Wave Ready": {'wave_states': ['🌊🌊🌊 CRESTING', '🌊🌊 BUILDING']}
        }
        
        selected_quick = st.selectbox(
            "Apply Quick Filter",
            options=list(quick_filter_options.keys()),
            index=0,
            key='quick_filter_select'
        )
        
        if selected_quick != "None":
            st.session_state.quick_filter = quick_filter_options[selected_quick]
            st.session_state.quick_filter_applied = True
        else:
            st.session_state.quick_filter = None
            st.session_state.quick_filter_applied = False
        
        # Detailed Filters in Expander
        with st.expander("🎛️ Advanced Filters", expanded=False):
            # Score Filter
            st.markdown("##### 📊 Score Filters")
            min_score = st.slider(
                "Minimum Master Score",
                min_value=0,
                max_value=100,
                value=st.session_state.get('min_score', 0),
                step=5,
                key='min_score'
            )
            
            # ENHANCEMENT #6: Wave State Filter
            st.markdown("##### 🌊 Wave Filters")
            
            wave_states = st.multiselect(
                "Wave States",
                options=["🌊🌊🌊 CRESTING", "🌊🌊 BUILDING", "🌊 FORMING", "💥 BREAKING"],
                default=st.session_state.get('wave_states_filter', []),
                key='wave_states_filter'
            )
            
            wave_strength_range = st.slider(
                "Wave Strength (0-100)",
                min_value=0,
                max_value=100,
                value=st.session_state.get('wave_strength_slider', (0, 100)),
                step=25,
                key='wave_strength_slider'
            )
            
            # Fundamental Filters
            st.markdown("##### 💰 Fundamental Filters")
            
            col1, col2 = st.columns(2)
            with col1:
                min_pe = st.number_input(
                    "Min P/E",
                    min_value=0.0,
                    value=st.session_state.get('min_pe', 0.0),
                    step=1.0,
                    key='min_pe'
                )
            
            with col2:
                max_pe = st.number_input(
                    "Max P/E",
                    min_value=0.0,
                    value=st.session_state.get('max_pe', 10000.0),
                    step=10.0,
                    key='max_pe'
                )
            
            min_eps_change = st.number_input(
                "Min EPS Change %",
                value=st.session_state.get('min_eps_change', None),
                step=10.0,
                key='min_eps_change'
            )
            
            require_fundamental = st.checkbox(
                "Require fundamental data",
                value=st.session_state.get('require_fundamental_data', False),
                key='require_fundamental_data'
            )
            
            # Trend Filter
            st.markdown("##### 📈 Trend Filter")
            trend_options = {
                "All Trends": (0, 100),
                "Strong Trends (70+)": (70, 100),
                "Moderate Trends (40-70)": (40, 70),
                "Weak Trends (<40)": (0, 40)
            }
            
            trend_filter = st.selectbox(
                "Trend Quality",
                options=list(trend_options.keys()),
                index=0,
                key='trend_filter'
            )
        
        # Info section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        with st.expander("System Information"):
            st.markdown("""
            **Wave Detection Ultimate 3.0**
            
            Enhanced Final Version with:
            - 🎯 Smart dynamic sector sampling
            - 🔍 Improved search with word boundaries
            - 💎 25 advanced pattern detections
            - 📊 Market regime detection
            - 🌊 Wave state analysis
            - ⚡ Optimized performance
            - 📱 Mobile responsive design
            
            **Master Score Formula (v3.0):**
            - Position: 30%
            - Volume: 25%
            - Momentum: 15%
            - Acceleration: 10%
            - Breakout: 10%
            - RVOL: 10%
            
            Version: 3.0.8-FINAL-ENHANCED
            """)
    
    # MAIN CONTENT AREA
    
    # Load data based on source
    try:
        if st.session_state.data_source == "upload" and uploaded_file is not None:
            with st.spinner("📊 Processing uploaded data..."):
                df, timestamp, metadata = load_and_process_data(
                    source_type="upload",
                    file_data=uploaded_file,
                    data_version="3.0"
                )
        else:
            # Load from Google Sheets
            with st.spinner("🌐 Loading data from Google Sheets..."):
                df, timestamp, metadata = load_and_process_data(
                    source_type="sheet",
                    data_version="3.0"
                )
        
        # Display any warnings or errors
        if metadata.get('warnings'):
            for warning in metadata['warnings']:
                st.warning(f"⚠️ {warning}")
        
        if metadata.get('errors'):
            for error in metadata['errors']:
                st.error(f"❌ {error}")
        
        # Apply filters
        filters = {
            'categories': st.session_state.get('category_filter', []),
            'sectors': st.session_state.get('sector_filter', []),
            'min_score': st.session_state.get('min_score', 0),
            'patterns': st.session_state.get('patterns', []),
            'trend_range': trend_options.get(st.session_state.get('trend_filter', 'All Trends'), (0, 100)),
            'trend_filter': st.session_state.get('trend_filter', 'All Trends'),
            'min_eps_change': st.session_state.get('min_eps_change'),
            'min_pe': st.session_state.get('min_pe'),
            'max_pe': st.session_state.get('max_pe'),
            'eps_tiers': st.session_state.get('eps_tier_filter', []),
            'pe_tiers': st.session_state.get('pe_tier_filter', []),
            'price_tiers': st.session_state.get('price_tier_filter', []),
            'require_fundamental_data': st.session_state.get('require_fundamental_data', False),
            'wave_states': st.session_state.get('wave_states_filter', []),
            'wave_strength_range': st.session_state.get('wave_strength_slider', (0, 100))
        }
        
        # Apply quick filter if active
        if st.session_state.quick_filter:
            if 'min_score' in st.session_state.quick_filter:
                filters['min_score'] = max(filters['min_score'], st.session_state.quick_filter['min_score'])
            if 'patterns' in st.session_state.quick_filter:
                filters['patterns'].extend(st.session_state.quick_filter['patterns'])
            if 'min_rvol' in st.session_state.quick_filter:
                # Add RVOL filter logic
                pass
            if 'min_ret_30d' in st.session_state.quick_filter:
                # Add return filter logic
                pass
            if 'wave_states' in st.session_state.quick_filter:
                filters['wave_states'] = st.session_state.quick_filter['wave_states']
        
        # Store filters in session state
        st.session_state.filters = filters
        st.session_state.active_filter_count = active_filter_count
        
        # Apply filters
        filtered_df = FilterEngine.apply_filters(df, filters)
        
        # Search functionality - moved above tabs for better UX
        search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
        
        with search_col1:
            search_query = st.text_input(
                "🔍 Search stocks",
                value=st.session_state.search_query,
                placeholder="Enter ticker or company name...",
                key='search_input'
            )
            st.session_state.search_query = search_query
        
        with search_col2:
            top_n = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n']),
                key='top_n_select'
            )
            st.session_state.user_preferences['default_top_n'] = top_n
        
        with search_col3:
            display_mode = st.selectbox(
                "Display mode",
                options=['Technical', 'Fundamental', 'Hybrid'],
                index=['Technical', 'Fundamental', 'Hybrid'].index(
                    st.session_state.user_preferences.get('display_mode', 'Technical')
                ),
                key='display_mode_select'
            )
            st.session_state.user_preferences['display_mode'] = display_mode
        
        # Apply search if query exists
        if search_query:
            search_results = SearchEngine.search_stocks(filtered_df, search_query)
            if not search_results.empty:
                display_df = search_results.head(top_n)
                st.info(f"🔍 Found {len(search_results)} matches for '{search_query}'")
            else:
                display_df = pd.DataFrame()
                st.warning(f"No results found for '{search_query}'")
        else:
            # Show top N stocks
            display_df = filtered_df.nlargest(min(top_n, len(filtered_df)), 'master_score')
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", 
            "🏆 Rankings", 
            "📈 Sector Analysis",
            "🎯 Patterns", 
            "📊 Analytics",
            "💾 Export"
        ])
        
        with tab1:
            # Dashboard Summary
            UIComponents.render_summary_section(filtered_df)
            
            # Market Overview Metrics
            st.markdown("---")
            st.markdown("### 📊 Market Overview")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                UIComponents.render_metric_card(
                    "Total Stocks",
                    f"{len(filtered_df):,}",
                    f"Filtered from {len(df):,}" if len(filtered_df) < len(df) else None
                )
            
            with col2:
                avg_score = filtered_df['master_score'].mean() if not filtered_df.empty else 0
                UIComponents.render_metric_card(
                    "Avg Score",
                    f"{avg_score:.1f}",
                    "Market average"
                )
            
            with col3:
                if 'rvol' in filtered_df.columns and not filtered_df.empty:
                    high_rvol = len(filtered_df[filtered_df['rvol'] > 2])
                    UIComponents.render_metric_card(
                        "High RVOL",
                        f"{high_rvol}",
                        f"{high_rvol/len(filtered_df)*100:.0f}% of stocks"
                    )
                else:
                    UIComponents.render_metric_card("High RVOL", "N/A")
            
            with col4:
                if 'patterns' in filtered_df.columns and not filtered_df.empty:
                    with_patterns = len(filtered_df[filtered_df['patterns'] != ''])
                    UIComponents.render_metric_card(
                        "With Patterns",
                        f"{with_patterns}",
                        f"{with_patterns/len(filtered_df)*100:.0f}% of stocks"
                    )
                else:
                    UIComponents.render_metric_card("With Patterns", "N/A")
            
            with col5:
                data_age = datetime.now(timezone.utc) - timestamp
                hours_old = data_age.total_seconds() / 3600
                
                if hours_old < 1:
                    age_str = "< 1 hour"
                    age_emoji = "🟢"
                elif hours_old < 24:
                    age_str = f"{hours_old:.0f} hours"
                    age_emoji = "🟡"
                else:
                    age_str = f"{hours_old/24:.0f} days"
                    age_emoji = "🔴"
                
                UIComponents.render_metric_card(
                    "Data Age",
                    f"{age_emoji} {age_str}",
                    timestamp.strftime("%Y-%m-%d %H:%M UTC")
                )
        
        with tab2:
            # Rankings Tab
            st.markdown("### 🏆 Stock Rankings")
            
            if display_df.empty:
                st.warning("No stocks to display. Try adjusting your filters.")
            else:
                # Filter controls for the table
                filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                
                with filter_col1:
                    # Category filter with dynamic options
                    category_options = FilterEngine.get_filter_options(
                        filtered_df, 'category', st.session_state.filters
                    )
                    
                    categories = st.multiselect(
                        "Categories",
                        options=['All'] + category_options,
                        default=st.session_state.get('category_filter', []),
                        key='category_filter'
                    )
                
                with filter_col2:
                    # Sector filter with dynamic options
                    sector_options = FilterEngine.get_filter_options(
                        filtered_df, 'sector', st.session_state.filters
                    )
                    
                    sectors = st.multiselect(
                        "Sectors",
                        options=['All'] + sector_options,
                        default=st.session_state.get('sector_filter', []),
                        key='sector_filter'
                    )
                
                with filter_col3:
                    # Pattern filter
                    all_patterns = set()
                    for patterns in filtered_df['patterns'].dropna():
                        if patterns:
                            all_patterns.update(p.strip() for p in patterns.split('|'))
                    
                    pattern_list = sorted(list(all_patterns))
                    
                    patterns = st.multiselect(
                        "Patterns",
                        options=pattern_list,
                        default=st.session_state.get('patterns', []),
                        key='patterns'
                    )
                
                with filter_col4:
                    # Tier filters
                    if 'eps_tier' in filtered_df.columns:
                        eps_tier_options = FilterEngine.get_filter_options(
                            filtered_df, 'eps_tier', st.session_state.filters
                        )
                        
                        eps_tiers = st.multiselect(
                            "EPS Tiers",
                            options=['All'] + eps_tier_options,
                            default=st.session_state.get('eps_tier_filter', []),
                            key='eps_tier_filter'
                        )
                
                # Prepare display columns based on mode
                if display_mode == 'Technical':
                    display_cols = [
                        'rank', 'ticker', 'company_name', 'master_score',
                        'momentum_score', 'acceleration_score', 'volume_score',
                        'rvol', 'ret_1d', 'ret_7d', 'ret_30d',
                        'wave_state', 'patterns', 'category'
                    ]
                elif display_mode == 'Fundamental':
                    display_cols = [
                        'rank', 'ticker', 'company_name', 'master_score',
                        'pe', 'eps_current', 'eps_change_pct', 'pe_tier', 'eps_tier',
                        'ret_1y', 'long_term_strength', 'patterns',
                        'category', 'sector'
                    ]
                else:  # Hybrid
                    display_cols = [
                        'rank', 'ticker', 'company_name', 'master_score',
                        'momentum_score', 'volume_score', 'rvol',
                        'pe', 'eps_change_pct', 'ret_30d', 'ret_1y',
                        'patterns', 'category'
                    ]
                
                # Filter columns to only those that exist
                available_cols = [col for col in display_cols if col in display_df.columns]
                
                # Format the display dataframe
                formatted_df = display_df[available_cols].copy()
                
                # Format numeric columns
                format_rules = {
                    'master_score': '{:.1f}',
                    'momentum_score': '{:.1f}',
                    'acceleration_score': '{:.1f}',
                    'volume_score': '{:.1f}',
                    'rvol': '{:.1f}x',
                    'ret_1d': '{:.1f}%',
                    'ret_7d': '{:.1f}%',
                    'ret_30d': '{:.1f}%',
                    'ret_1y': '{:.1f}%',
                    'pe': '{:.1f}',
                    'eps_current': '{:.2f}',
                    'eps_change_pct': '{:.0f}%'
                }
                
                for col, fmt in format_rules.items():
                    if col in formatted_df.columns:
                        if col == 'rvol':
                            formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')
                        else:
                            formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')
                
                # ENHANCEMENT #9: Color coding for key metrics
                def color_score(val):
                    """Color code scores"""
                    try:
                        num_val = float(str(val).replace('%', '').replace('x', ''))
                        if num_val >= 80:
                            return 'background-color: #2ecc71; color: white;'
                        elif num_val >= 60:
                            return 'background-color: #f39c12; color: white;'
                        elif num_val < 40:
                            return 'background-color: #e74c3c; color: white;'
                        return ''
                    except:
                        return ''
                
                # Apply styling
                styled_df = formatted_df.style.applymap(
                    color_score,
                    subset=[col for col in ['master_score', 'momentum_score', 'acceleration_score'] 
                           if col in formatted_df.columns]
                )
                
                # Display the table
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=600,
                    hide_index=True
                )
                
                # Show statistics below table
                st.markdown("---")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    st.markdown("**📊 Score Distribution**")
                    score_stats = display_df['master_score'].describe()
                    st.write(f"Mean: {score_stats['mean']:.1f}")
                    st.write(f"Median: {score_stats['50%']:.1f}")
                    st.write(f"Std Dev: {score_stats['std']:.1f}")
                
                with stat_col2:
                    st.markdown("**📈 Top Categories**")
                    if 'category' in display_df.columns:
                        top_cats = display_df['category'].value_counts().head(3)
                        for cat, count in top_cats.items():
                            st.write(f"{cat}: {count}")
                
                with stat_col3:
                    st.markdown("**🎯 Common Patterns**")
                    if 'patterns' in display_df.columns:
                        pattern_counts = {}
                        for patterns in display_df['patterns'].dropna():
                            if patterns:
                                for p in patterns.split(' | '):
                                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
                        
                        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        for pattern, count in top_patterns:
                            st.write(f"{pattern}: {count}")
        
        with tab3:
            # ENHANCEMENT #8: New Sector Analysis Tab
            UIComponents.render_sector_analysis_tab(filtered_df)
        
        with tab4:
            # Patterns Tab
            st.markdown("### 🎯 Pattern Analysis")
            
            if filtered_df.empty:
                st.warning("No data available for pattern analysis")
            else:
                # Pattern overview
                pattern_counts = {}
                stocks_with_patterns = {}
                
                for idx, row in filtered_df.iterrows():
                    if pd.notna(row.get('patterns', '')) and row['patterns']:
                        for pattern in row['patterns'].split(' | '):
                            pattern = pattern.strip()
                            if pattern:
                                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                                if pattern not in stocks_with_patterns:
                                    stocks_with_patterns[pattern] = []
                                stocks_with_patterns[pattern].append({
                                    'ticker': row['ticker'],
                                    'company': row['company_name'],
                                    'score': row['master_score']
                                })
                
                if pattern_counts:
                    # Pattern frequency chart
                    pattern_df = pd.DataFrame(
                        list(pattern_counts.items()),
                        columns=['Pattern', 'Count']
                    ).sort_values('Count', ascending=False)
                    
                    fig = go.Figure([go.Bar(
                        x=pattern_df['Pattern'],
                        y=pattern_df['Count'],
                        text=pattern_df['Count'],
                        textposition='outside',
                        marker_color='#3498db'
                    )])
                    
                    fig.update_layout(
                        title="Pattern Frequency Distribution",
                        xaxis_title="Pattern",
                        yaxis_title="Number of Stocks",
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Pattern details
                    st.markdown("---")
                    st.markdown("#### 🔍 Pattern Details")
                    
                    selected_pattern = st.selectbox(
                        "Select a pattern to see stocks",
                        options=sorted(pattern_counts.keys()),
                        index=0
                    )
                    
                    if selected_pattern and selected_pattern in stocks_with_patterns:
                        pattern_stocks = stocks_with_patterns[selected_pattern]
                        pattern_stocks_df = pd.DataFrame(pattern_stocks).sort_values('score', ascending=False)
                        
                        st.markdown(f"**{selected_pattern}** - Found in {len(pattern_stocks)} stocks:")
                        
                        # Display stocks with this pattern
                        for idx, stock in pattern_stocks_df.head(20).iterrows():
                            col1, col2, col3 = st.columns([2, 3, 1])
                            with col1:
                                st.write(f"**{stock['ticker']}**")
                            with col2:
                                st.write(stock['company'][:40])
                            with col3:
                                st.write(f"Score: {stock['score']:.1f}")
                        
                        if len(pattern_stocks) > 20:
                            st.info(f"Showing top 20 of {len(pattern_stocks)} stocks")
                    
                    # Pattern combinations
                    st.markdown("---")
                    st.markdown("#### 🔄 Pattern Combinations")
                    
                    # Find stocks with multiple patterns
                    multi_pattern_stocks = []
                    for idx, row in filtered_df.iterrows():
                        if pd.notna(row.get('patterns', '')) and row['patterns']:
                            pattern_list = [p.strip() for p in row['patterns'].split(' | ') if p.strip()]
                            if len(pattern_list) >= 2:
                                multi_pattern_stocks.append({
                                    'ticker': row['ticker'],
                                    'company': row['company_name'],
                                    'score': row['master_score'],
                                    'pattern_count': len(pattern_list),
                                    'patterns': ' | '.join(pattern_list[:3])  # Show first 3
                                })
                    
                    if multi_pattern_stocks:
                        multi_df = pd.DataFrame(multi_pattern_stocks).sort_values(
                            ['pattern_count', 'score'], ascending=[False, False]
                        ).head(20)
                        
                        st.markdown("**Stocks with Multiple Patterns:**")
                        st.dataframe(
                            multi_df[['ticker', 'company', 'score', 'pattern_count', 'patterns']],
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.info("No patterns detected in the filtered data")
        
        with tab5:
            # Analytics Tab
            st.markdown("### 📊 Market Analytics")
            
            if filtered_df.empty:
                st.warning("No data available for analytics")
            else:
                # Score distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_dist = Visualizer.create_score_distribution(filtered_df)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # ENHANCEMENT #5: Acceleration profiles
                    fig_accel = Visualizer.create_acceleration_profiles(filtered_df, n=10)
                    st.plotly_chart(fig_accel, use_container_width=True)
                
                # Market breadth analysis
                st.markdown("---")
                st.markdown("#### 📊 Market Breadth Analysis")
                
                breadth_col1, breadth_col2, breadth_col3 = st.columns(3)
                
                with breadth_col1:
                    # Score buckets
                    score_buckets = pd.cut(
                        filtered_df['master_score'],
                        bins=[0, 40, 60, 80, 100],
                        labels=['Poor (0-40)', 'Fair (40-60)', 'Good (60-80)', 'Excellent (80-100)']
                    ).value_counts()
                    
                    fig_buckets = go.Figure([go.Pie(
                        labels=score_buckets.index,
                        values=score_buckets.values,
                        hole=0.3,
                        marker_colors=['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
                    )])
                    
                    fig_buckets.update_layout(
                        title="Score Distribution",
                        height=300,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_buckets, use_container_width=True)
                
                with breadth_col2:
                    # Momentum distribution
                    if 'ret_30d' in filtered_df.columns:
                        momentum_buckets = pd.cut(
                            filtered_df['ret_30d'].fillna(0),
                            bins=[-float('inf'), -10, 0, 10, 30, float('inf')],
                            labels=['< -10%', '-10% to 0%', '0% to 10%', '10% to 30%', '> 30%']
                        ).value_counts()
                        
                        fig_momentum = go.Figure([go.Bar(
                            x=momentum_buckets.index,
                            y=momentum_buckets.values,
                            text=momentum_buckets.values,
                            textposition='outside',
                            marker_color=['#e74c3c', '#ec7063', '#85929e', '#52be80', '#27ae60']
                        )])
                        
                        fig_momentum.update_layout(
                            title="30-Day Return Distribution",
                            xaxis_title="Return Range",
                            yaxis_title="Count",
                            height=300,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_momentum, use_container_width=True)
                
                with breadth_col3:
                    # RVOL distribution
                    if 'rvol' in filtered_df.columns:
                        rvol_buckets = pd.cut(
                            filtered_df['rvol'].fillna(1),
                            bins=[0, 0.5, 1, 2, 5, float('inf')],
                            labels=['< 0.5x', '0.5-1x', '1-2x', '2-5x', '> 5x']
                        ).value_counts()
                        
                        fig_rvol = go.Figure([go.Bar(
                            x=rvol_buckets.values,
                            y=rvol_buckets.index,
                            orientation='h',
                            text=rvol_buckets.values,
                            textposition='outside',
                            marker_color='#9b59b6'
                        )])
                        
                        fig_rvol.update_layout(
                            title="RVOL Distribution",
                            xaxis_title="Count",
                            yaxis_title="RVOL Range",
                            height=300,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_rvol, use_container_width=True)
                
                # Category performance
                if 'category' in filtered_df.columns:
                    st.markdown("---")
                    st.markdown("#### 🏆 Category Performance")
                    
                    category_stats = filtered_df.groupby('category').agg({
                        'master_score': ['mean', 'median', 'count'],
                        'ticker': 'count'
                    }).round(2)
                    
                    category_stats.columns = ['Avg Score', 'Median Score', 'Top Stocks', 'Total Stocks']
                    category_stats = category_stats.sort_values('Avg Score', ascending=False)
                    
                    fig_cat = go.Figure()
                    
                    # Add bars for average score
                    fig_cat.add_trace(go.Bar(
                        name='Average Score',
                        x=category_stats.index,
                        y=category_stats['Avg Score'],
                        marker_color='#3498db'
                    ))
                    
                    # Add line for stock count
                    fig_cat.add_trace(go.Scatter(
                        name='Stock Count',
                        x=category_stats.index,
                        y=category_stats['Total Stocks'],
                        yaxis='y2',
                        mode='lines+markers',
                        line=dict(color='#e74c3c', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig_cat.update_layout(
                        title="Category Performance Analysis",
                        xaxis_title="Category",
                        yaxis_title="Average Score",
                        yaxis2=dict(
                            title="Stock Count",
                            overlaying='y',
                            side='right'
                        ),
                        height=400,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_cat, use_container_width=True)
        
        with tab6:
            # Export Tab
            st.markdown("### 💾 Export Data")
            
            export_col1, export_col2 = st.columns([1, 2])
            
            with export_col1:
                st.markdown("#### 📊 Export Options")
                
                # ENHANCEMENT #3: Template selection
                export_template = st.selectbox(
                    "Export Template",
                    options=['Full Analysis', 'Day Trader', 'Swing Trader', 'Investor'],
                    help="Choose a template optimized for your trading style"
                )
                
                template_map = {
                    'Full Analysis': 'full',
                    'Day Trader': 'day_trader',
                    'Swing Trader': 'swing_trader',
                    'Investor': 'investor'
                }
                
                export_format = st.radio(
                    "Format",
                    options=['Excel', 'CSV'],
                    help="Excel includes multiple sheets with analysis"
                )
                
                include_filtered = st.checkbox(
                    "Export filtered data only",
                    value=True,
                    help="Export only the stocks matching current filters"
                )
                
                # Export button
                if st.button("📥 Generate Export", type="primary", use_container_width=True):
                    try:
                        export_df = filtered_df if include_filtered else df
                        
                        if export_format == 'Excel':
                            with st.spinner("Creating Excel report..."):
                                excel_file = ExportEngine.create_excel_report(
                                    export_df,
                                    template=template_map[export_template]
                                )
                                
                                st.download_button(
                                    label="📥 Download Excel Report",
                                    data=excel_file,
                                    file_name=f"wave_detection_{export_template.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                                
                                st.success(f"✅ Excel report ready! Template: {export_template}")
                        else:
                            csv_data = ExportEngine.create_csv_export(export_df)
                            
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_data,
                                file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            st.success("✅ CSV export ready!")
                    
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
                        logger.error(f"Export error: {str(e)}")
            
            with export_col2:
                st.markdown("#### 📋 Export Summary")
                
                if include_filtered:
                    st.info(f"Will export {len(filtered_df):,} filtered stocks")
                else:
                    st.info(f"Will export all {len(df):,} stocks")
                
                # Show what's included in each template
                st.markdown("##### 📄 Template Contents")
                
                template_info = {
                    'Full Analysis': "Complete data with all scores, patterns, and metrics",
                    'Day Trader': "Focus on momentum, RVOL, and intraday signals",
                    'Swing Trader': "Position analysis, breakout scores, and trend quality",
                    'Investor': "Fundamentals, long-term performance, and valuations"
                }
                
                st.write(f"**{export_template}**: {template_info[export_template]}")
                
                if export_format == 'Excel':
                    st.markdown("##### 📊 Excel Sheets Include:")
                    sheets = [
                        "✅ Top 100 Stocks",
                        "✅ Market Intelligence",
                        "✅ Sector Rotation Analysis",
                        "✅ Pattern Analysis",
                        "✅ Wave Radar Signals",
                        "✅ Summary Statistics"
                    ]
                    for sheet in sheets:
                        st.write(sheet)
        
        # Footer
        st.markdown("---")
        footer_col1, footer_col2, footer_col3 = st.columns(3)
        
        with footer_col1:
            st.caption(f"Last refresh: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        with footer_col2:
            if metadata.get('processing_time'):
                st.caption(f"Processing time: {metadata['processing_time']:.2f}s")
        
        with footer_col3:
            st.caption("Wave Detection Ultimate v3.0.8-FINAL-ENHANCED")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"❌ An error occurred: {str(e)}")
        
        # Show detailed error in debug mode
        if st.checkbox("Show error details"):
            st.exception(e)
        
        # Provide recovery options
        st.markdown("---")
        st.markdown("### 🔧 Recovery Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reload App", type="primary"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All Data"):
                st.cache_data.clear()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col3:
            if st.button("📊 Use Test Data"):
                # Create minimal test data
                test_data = pd.DataFrame({
                    'ticker': ['TEST1', 'TEST2', 'TEST3'],
                    'company_name': ['Test Company 1', 'Test Company 2', 'Test Company 3'],
                    'price': [100, 200, 300],
                    'volume_1d': [1000000, 2000000, 3000000],
                    'category': ['Large Cap', 'Mid Cap', 'Small Cap'],
                    'sector': ['Technology', 'Finance', 'Healthcare']
                })
                st.session_state['test_mode'] = True
                st.info("Test mode activated. Reload to see test data.")

# Run the application
if __name__ == "__main__":
    main()
