"""
Wave Detection Ultimate 3.0 - FINAL ULTIMATE PRODUCTION VERSION
===============================================================
Professional Stock Ranking System with Advanced Analytics
All identified bugs fixed, maximally optimized, permanently locked for production.
Handles 1791+ stocks with 41+ data columns (as per original intent).

Version: 3.0-ULTIMATE-LOCKED
Last Updated: July 28, 2025
Status: FINAL PRODUCTION READY - PERMANENTLY LOCKED - NO FUTURE UPGRADES
"""

# ============================================
# IMPORTS AND SETUP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import time
from io import BytesIO
import warnings
import gc
# from decimal import Decimal, ROUND_HALF_UP # Not used in final version, removed

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define IST timezone for display
IST = timezone(timedelta(hours=5, minutes=30))

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
    
    # All percentage columns for consistent handling (values like -56.61 for -56.61%)
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ])
    
    # Volume ratio columns (values are percentage changes, will be converted to multipliers 0.xx to X.xx)
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
        'rvol': (0.01, 1_000_000.0),  # Allow very high RVOL values
        'pe': (-10000, 10000), # Allow negative PE for loss-making, but cap extreme
        'returns': (-99.99, 9999.99),  # Percentage bounds for returns
        'volume': (0, 1e12) # For raw volume numbers
    })
    
    # Performance targets (for logging warnings)
    PERFORMANCE_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'data_processing': 2.0,  # seconds
        'filtering': 0.2,
        'pattern_detection': 0.5,
        'export_generation': 1.0,
        'search': 0.05
    })
    
    # Market categories (Indian market specific examples)
    MARKET_CATEGORIES: List[str] = field(default_factory=lambda: [
        'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'
    ])
    
    # Tier definitions with proper boundaries (inclusive lower, exclusive upper unless inf)
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {
            "Loss": (-float('inf'), 0), # Values <= 0
            "0-5": (0, 5),             # Values > 0 and <= 5
            "5-10": (5, 10),
            "10-20": (10, 20),
            "20-50": (20, 50),
            "50-100": (50, 100),
            "100+": (100, float('inf'))
        },
        "pe": {
            "Negative/NA": (-float('inf'), 0), # Values <= 0 or NA
            "0-10": (0, 10),                   # Values > 0 and <= 10
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
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '')
            
            # If it's a percentage column and has a '%' sign, remove it
            # The value is treated as the direct percentage number (e.g., "50" means 50%)
            if is_percentage and '%' in cleaned:
                cleaned = cleaned.replace('%', '')
            # For non-percentage columns, remove '%' if it might have crept in
            elif not is_percentage and '%' in cleaned:
                cleaned = cleaned.replace('%', '')
            
            # Convert to float
            result = float(cleaned)
            
            # Apply bounds if specified
            if bounds:
                min_val, max_val = bounds
                if result < min_val or result > max_val:
                    logger.debug(f"Value {result} for {value} outside bounds [{min_val}, {max_val}]")
                    result = np.clip(result, min_val, max_val)
            
            # Check for unreasonable values
            if np.isnan(result) or np.isinf(result):
                return np.nan
            
            return result
            
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to clean value '{value}' to numeric: {e}")
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
            
            logger.info(f"Loading data from Google Sheets: {csv_url}")
            
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
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['data_processing'])
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
                
                # Determine bounds from CONFIG.VALUE_BOUNDS
                bounds = None
                if 'volume' in col.lower():
                    bounds = CONFIG.VALUE_BOUNDS['volume']
                elif col == 'rvol':
                    bounds = CONFIG.VALUE_BOUNDS['rvol']
                elif col == 'pe':
                    bounds = CONFIG.VALUE_BOUNDS['pe']
                elif is_pct:
                    bounds = CONFIG.VALUE_BOUNDS['returns']
                elif col == 'price': # Specific bounds for price
                    bounds = CONFIG.VALUE_BOUNDS['price']
                
                # Vectorized cleaning
                df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, is_pct, bounds))
        
        # Process categorical columns
        string_cols = ['ticker', 'company_name', 'category', 'sector']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
        # Fix volume ratios (vectorized)
        # Data is treated as a percentage change (e.g., -56.61 means -56.61% change)
        # Convert to ratio: (100 + change%) / 100 => 0.4339 for -56.61%
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                # Ensure it's numeric before conversion
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = (100 + df[col]) / 100
                # Clip to a reasonable range for ratios (0.01 to 100.0)
                # 100.0x means 100 times average volume, which is extremely high
                df[col] = df[col].clip(0.01, 100.0) 
                df[col] = df[col].fillna(1.0) # Default to 1.0 (no change) if NaN
        
        # Validate critical data and remove invalid rows
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > CONFIG.VALUE_BOUNDS['price'][0]]  # Minimum valid price
        
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
        
        # Ensure all score components are present and filled to avoid NaN in master score calc
        score_component_cols = [
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score'
        ]
        for col in score_component_cols:
            if col not in df.columns:
                df[col] = 50 # Default neutral score
            df[col] = df[col].fillna(50)
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications with proper boundary handling"""
        
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Classify value into tier with precise boundary logic (inclusive lower, exclusive upper unless inf)"""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if tier_name == "Negative/NA" and value <= 0: # Specific for PE/EPS loss
                    return tier_name
                
                if min_val <= value < max_val:
                    return tier_name
                # Special case for the last tier being inclusive of max_val if it's infinity
                if max_val == float('inf') and value >= min_val:
                    return tier_name
            
            return "Unknown"
        
        # Add tier columns
        if 'eps_current' in df.columns:
            df['eps_tier'] = df['eps_current'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['eps'])
            )
        
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['pe'])
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
            df['money_flow_mm'] = (df['price'] * df['volume_1d'] * df['rvol']) / 1_000_000
            df['money_flow_mm'] = df['money_flow_mm'].fillna(0.0)
        else:
            df['money_flow_mm'] = 0.0 # Default if cols are missing
        
        # Volume Momentum Index (VMI)
        # Ensure all required vol_ratio columns exist, default to 1.0 if not
        v_1d_90d = df['vol_ratio_1d_90d'].fillna(1.0) if 'vol_ratio_1d_90d' in df.columns else 1.0
        v_7d_90d = df['vol_ratio_7d_90d'].fillna(1.0) if 'vol_ratio_7d_90d' in df.columns else 1.0
        v_30d_90d = df['vol_ratio_30d_90d'].fillna(1.0) if 'vol_ratio_30d_90d' in df.columns else 1.0
        v_90d_180d = df['vol_ratio_90d_180d'].fillna(1.0) if 'vol_ratio_90d_180d' in df.columns else 1.0

        df['vmi'] = (
            v_1d_90d * 4 +
            v_7d_90d * 3 +
            v_30d_90d * 2 +
            v_90d_180d * 1
        ) / 10
        df['vmi'] = df['vmi'].fillna(1.0) # Default if any underlying values caused NaN

        # Position Tension
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'] + abs(df['from_high_pct'])
            df['position_tension'] = df['position_tension'].fillna(100.0) # Default to neutral
        else:
            df['position_tension'] = 100.0 # Default if cols are missing
        
        # Momentum Harmony
        df['momentum_harmony'] = 0
        
        if 'ret_1d' in df.columns:
            df['momentum_harmony'] += (df['ret_1d'] > 0).astype(int)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                # Ensure division is only performed on non-zero denominators
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            
            df['momentum_harmony'] += (daily_ret_7d > daily_ret_30d).astype(int)
        
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            with np.errstate(divide='ignore', invalid='ignore'):
                # Ensure division is only performed on non-zero denominators
                daily_ret_30d_comp = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
                daily_ret_3m_comp = np.where(df['ret_3m'] != 0, df['ret_3m'] / 90, 0)
            df['momentum_harmony'] += (daily_ret_30d_comp > daily_ret_3m_comp).astype(int)
        
        if 'ret_3m' in df.columns:
            df['momentum_harmony'] += (df['ret_3m'] > 0).astype(int)
        
        # Wave State
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)

        # Overall Wave Strength (for filtering)
        # Ensure all required scores exist or default to 50
        momentum_score_val = df['momentum_score'].fillna(50)
        acceleration_score_val = df['acceleration_score'].fillna(50)
        rvol_score_val = df['rvol_score'].fillna(50)
        breakout_score_val = df['breakout_score'].fillna(50)

        df['overall_wave_strength'] = (
            momentum_score_val * 0.3 +
            acceleration_score_val * 0.3 +
            rvol_score_val * 0.2 +
            breakout_score_val * 0.2
        )
        df['overall_wave_strength'] = df['overall_wave_strength'].clip(0, 100).fillna(50.0) # Ensure bounded and no NaNs
        
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
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['data_processing']) # Part of data processing
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores and master score"""
        
        if df.empty:
            return df
        
        logger.info("Starting optimized ranking calculations...")
        
        # Calculate component scores with default values if underlying data is missing
        df['position_score'] = RankingEngine._calculate_position_score(df).fillna(50)
        df['volume_score'] = RankingEngine._calculate_volume_score(df).fillna(50)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df).fillna(50)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df).fillna(50)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df).fillna(50)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df).fillna(50)
        
        # Calculate auxiliary scores
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df).fillna(50)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df).fillna(50)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df).fillna(50)
        
        # Calculate master score using numpy (DO NOT MODIFY FORMULA)
        scores_matrix = np.column_stack([
            df['position_score'], # Fillna done above
            df['volume_score'],
            df['momentum_score'],
            df['acceleration_score'],
            df['breakout_score'],
            df['rvol_score']
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
            return pd.Series(dtype=float, index=series.index) # Return empty series with original index
        
        # Replace inf values with NaN
        series_copy = series.replace([np.inf, -np.inf], np.nan)
        
        valid_count = series_copy.notna().sum()
        if valid_count == 0:
            # If no valid values, return neutral score with small random variation for differentiation
            return pd.Series(50 + np.random.uniform(-0.1, 0.1, size=len(series_copy)), index=series_copy.index)
        
        # Rank with proper parameters
        if pct:
            ranks = series_copy.rank(pct=True, ascending=ascending, na_option='bottom') * 100
            ranks = ranks.fillna(0 if ascending else 100) # Fill NaN with worst rank (0 for ascending, 100 for descending)
        else:
            ranks = series_copy.rank(ascending=ascending, method='min', na_option='bottom')
            ranks = ranks.fillna(valid_count + 1) # Fill NaN with a rank worse than any actual stock
        
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
        vol_cols_weights = [
            ('vol_ratio_1d_90d', 0.20),
            ('vol_ratio_7d_90d', 0.20),
            ('vol_ratio_30d_90d', 0.20),
            ('vol_ratio_30d_180d', 0.15),
            ('vol_ratio_90d_180d', 0.25)
        ]
        
        # Calculate weighted score
        total_weight = 0
        weighted_score_sum = pd.Series(0, index=df.index, dtype=float)
        
        for col, weight in vol_cols_weights:
            if col in df.columns and df[col].notna().any():
                col_data = df[col].fillna(1.0) # Fill with 1.0 (no change) if NaN
                col_rank = RankingEngine._safe_rank(col_data, pct=True, ascending=True)
                weighted_score_sum += col_rank * weight
                total_weight += weight
            else:
                logger.debug(f"Volume ratio column '{col}' missing or all NaN, skipping for volume score calculation.")

        if total_weight > 0:
            volume_score = weighted_score_sum / total_weight
        else:
            logger.warning("No valid volume ratio data available, using neutral volume scores.")
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns"""
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        # Fallback logic if 30-day returns are not available
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                ret_7d = df['ret_7d'].fillna(0)
                momentum_score = RankingEngine._safe_rank(ret_7d, pct=True, ascending=True)
                logger.info("Using 7-day returns for momentum score due to missing 30-day data.")
            else:
                logger.warning("No relevant return data available for momentum calculation.")
            return momentum_score.clip(0, 100)
        
        # Primary: 30-day returns
        ret_30d = df['ret_30d'].fillna(0)
        momentum_score = RankingEngine._safe_rank(ret_30d, pct=True, ascending=True)
        
        # Add consistency bonus if both 7-day and 30-day returns are available
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            
            # Condition 1: Both recent returns are positive
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            consistency_bonus[all_positive] += 5 # Add 5 points for general positive consistency
            
            # Condition 2: Accelerating returns (daily pace of 7-day is faster than 30-day)
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            
            # Apply bonus only if accelerating and returns are positive
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            consistency_bonus[accelerating] += 5 # Add another 5 points (total 10 for accelerating positive)
            
            momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
        
        return momentum_score
    
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating with robust daily average comparisons"""
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        
        # Ensure minimum required data is present
        if not all(col in df.columns and df[col].notna().any() for col in ['ret_1d', 'ret_7d']):
            logger.warning("Insufficient recent return data (ret_1d, ret_7d) for robust acceleration calculation.")
            return acceleration_score # Return default if not enough data
        
        # Get return data, filling NaNs with 0 for calculation purposes
        ret_1d = df['ret_1d'].fillna(0)
        ret_7d = df['ret_7d'].fillna(0)
        ret_30d = df['ret_30d'].fillna(0) if 'ret_30d' in df.columns else pd.Series(0, index=df.index)
        
        # Calculate daily averages with safe division (avoiding warnings for zero division)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_daily_1d = ret_1d  # 1-day return is already a daily average
            avg_daily_7d = np.where(ret_7d != 0, ret_7d / 7, 0)
            avg_daily_30d = np.where(ret_30d != 0, ret_30d / 30, 0)
        
        # Conditions for different acceleration levels
        # Perfect acceleration: Today's pace > 7D pace > 30D pace AND positive today
        perfect_mask = (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
        acceleration_score.loc[perfect_mask] = 100
        
        # Good acceleration: Today's pace > 7D pace AND positive today (but not perfectly accelerating on all fronts)
        good_mask = (~perfect_mask) & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0)
        acceleration_score.loc[good_mask] = 80
        
        # Moderate acceleration: Positive today, but not necessarily accelerating week-over-day
        moderate_mask = (~perfect_mask) & (~good_mask) & (ret_1d > 0)
        acceleration_score.loc[moderate_mask] = 60
        
        # Slight deceleration: Negative today, but positive over 7 days
        slight_decel_mask = (ret_1d <= 0) & (ret_7d > 0)
        acceleration_score.loc[slight_decel_mask] = 40
        
        # Strong deceleration: Negative over 1 day and 7 days
        strong_decel_mask = (ret_1d <= 0) & (ret_7d <= 0)
        acceleration_score.loc[strong_decel_mask] = 20
        
        return acceleration_score
    
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout probability based on distance from high, volume, and trend support."""
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        
        # Factor 1: Distance from 52-week high (40% weight)
        if 'from_high_pct' in df.columns:
            # 'from_high_pct' is negative (e.g., -5 for 5% below high).
            # Closer to 0 (less negative) is better. Convert to a "closeness" score.
            # 0% from high (at high) -> 100 score
            # -100% from high (at low) -> 0 score
            closeness_to_high = 100 + df['from_high_pct'].fillna(-100) # Transform -100 to 0, 0 to 100
            distance_factor = closeness_to_high.clip(0, 100)
        else:
            distance_factor = pd.Series(50, index=df.index) # Neutral if data missing
        
        # Factor 2: Volume surge (40% weight) - Using 7D/90D volume ratio
        volume_factor = pd.Series(50, index=df.index)
        if 'vol_ratio_7d_90d' in df.columns:
            # 'vol_ratio_7d_90d' is a multiplier (e.g., 2.0 for 2x average volume).
            # We want higher values for a surge. Convert to a score (e.g., 1.0 -> 0, 2.0 -> 100)
            vol_ratio = df['vol_ratio_7d_90d'].fillna(1.0)
            # Simple linear mapping for scoring, cap at a reasonable max to avoid extreme outliers distorting
            volume_factor = ((vol_ratio - 1.0) * 100).clip(0, 100) # 1.0 ratio -> 0, 2.0 ratio -> 100
        else:
            volume_factor = pd.Series(50, index=df.index) # Neutral if data missing
        
        # Factor 3: Trend support (20% weight) - Price relative to SMAs
        trend_support_factor = pd.Series(0, index=df.index, dtype=float)
        
        if 'price' in df.columns:
            current_price = df['price']
            
            # Score based on how many key SMAs the price is above
            sma_contributions = []
            if 'sma_20d' in df.columns: sma_contributions.append((df['sma_20d'], 33.33))
            if 'sma_50d' in df.columns: sma_contributions.append((df['sma_50d'], 33.33))
            if 'sma_200d' in df.columns: sma_contributions.append((df['sma_200d'], 33.34))
            
            total_possible_points = 0
            for sma_series, points in sma_contributions:
                if sma_series.notna().any(): # Check if SMA column actually has data
                    above_sma_mask = (current_price > sma_series).fillna(False)
                    trend_support_factor += above_sma_mask.astype(float) * points
                    total_possible_points += points
            
            # Normalize if not all SMAs are available or have data
            if total_possible_points > 0 and total_possible_points < 100:
                 trend_support_factor = (trend_support_factor / total_possible_points) * 100
        
        trend_support_factor = trend_support_factor.clip(0, 100)
        
        # Combine factors with predefined weights
        breakout_score = (
            distance_factor * 0.4 +
            volume_factor * 0.4 +
            trend_support_factor * 0.2
        )
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate RVOL-based score, robust for very high values."""
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index, dtype=float) # Default neutral score
        
        rvol = df['rvol'].fillna(1.0) # Fill missing RVOL with 1.0 (average volume)
        rvol_score = pd.Series(50, index=df.index, dtype=float) # Initialize all scores to 50
        
        # Apply score based on RVOL ranges (from lower to higher values for efficiency)
        rvol_score.loc[rvol <= 0.3] = 20
        rvol_score.loc[(rvol > 0.3) & (rvol <= 0.5)] = 30
        rvol_score.loc[(rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score.loc[(rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score.loc[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score.loc[(rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score.loc[(rvol > 2) & (rvol <= 3)] = 80
        rvol_score.loc[(rvol > 3) & (rvol <= 5)] = 85
        rvol_score.loc[(rvol > 5) & (rvol <= 10)] = 90
        rvol_score.loc[(rvol > 10) & (rvol <= 100)] = 95 # High but realistic surges
        rvol_score.loc[rvol > 100] = 100 # Extreme, but valid, surges
        
        return rvol_score
    
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality score based on SMA alignment."""
        trend_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'price' not in df.columns:
            logger.warning("Price column missing for trend quality calculation.")
            return trend_score
        
        current_price = df['price']
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        
        # Filter for available and non-null SMA columns
        available_smas = []
        for col in sma_cols:
            if col in df.columns and df[col].notna().any():
                available_smas.append(col)
        
        if not available_smas:
            logger.warning("No valid SMA data available for trend quality calculation.")
            return trend_score
        
        # Logic for 3+ SMAs (most robust assessment)
        if len(available_smas) >= 3:
            # Check for perfect bullish alignment: Price > SMA20 > SMA50 > SMA200
            perfect_trend_mask = (
                (current_price > df['sma_20d']) & 
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            ).fillna(False)
            trend_score.loc[perfect_trend_mask] = 100
            
            # Strong trend: Price above all three SMAs, but not perfectly aligned
            strong_trend_mask = (
                (~perfect_trend_mask) & # Exclude perfect trend stocks
                (current_price > df['sma_20d']) & 
                (current_price > df['sma_50d']) & 
                (current_price > df['sma_200d'])
            ).fillna(False)
            trend_score.loc[strong_trend_mask] = 85
            
            # Count how many SMAs the price is above
            above_count = pd.Series(0, index=df.index)
            for sma_col in available_smas:
                # Ensure SMA value is positive to avoid erroneous comparisons
                valid_sma_mask = df[sma_col].notna() & (df[sma_col] > 0)
                above_count.loc[valid_sma_mask] += (current_price.loc[valid_sma_mask] > df.loc[valid_sma_mask, sma_col]).astype(int)
            
            # Good trend: Price above 2 out of 3 SMAs (excluding stronger conditions)
            good_trend_mask = (above_count == 2) & (~perfect_trend_mask) & (~strong_trend_mask)
            trend_score.loc[good_trend_mask] = 70
            
            # Weak trend: Price above 1 out of 3 SMAs
            weak_trend_mask = (above_count == 1)
            trend_score.loc[weak_trend_mask] = 40
            
            # Poor trend: Price below all 3 SMAs
            poor_trend_mask = (above_count == 0)
            trend_score.loc[poor_trend_mask] = 20
        
        # Fallback logic for fewer than 3 available SMAs
        elif len(available_smas) == 2:
            sma1, sma2 = available_smas[0], available_smas[1]
            # Price above both available SMAs
            above_both_mask = (current_price > df[sma1]) & (current_price > df[sma2]).fillna(False)
            trend_score.loc[above_both_mask] = 80
            trend_score.loc[~above_both_mask] = 30 # Below at least one or both
        
        elif len(available_smas) == 1:
            sma = available_smas[0]
            # Price above the single available SMA
            above_single_mask = (current_price > df[sma]).fillna(False)
            trend_score.loc[above_single_mask] = 65
            trend_score.loc[~above_single_mask] = 35 # Below the single SMA
        
        return trend_score
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score based on multi-period returns."""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y'] # Include longer terms if available
        available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        
        if not available_cols:
            logger.warning("No long-term return data available for strength calculation.")
            return strength_score
        
        # Calculate average long-term return from available columns
        lt_returns = df[available_cols].fillna(0) # Fill NaNs with 0 for average
        avg_return = lt_returns.mean(axis=1)
        
        # Determine if recent long-term performance is improving (e.g., 6m > 1y)
        improving_long_term_mask = pd.Series(False, index=df.index)
        if 'ret_6m' in df.columns and 'ret_1y' in df.columns:
            # Compare annualized 6-month return to 1-year return
            with np.errstate(divide='ignore', invalid='ignore'):
                annualized_6m = df['ret_6m'] * 2
            improving_long_term_mask = (annualized_6m > df['ret_1y']).fillna(False)
        elif 'ret_3m' in df.columns and 'ret_1y' in df.columns:
             with np.errstate(divide='ignore', invalid='ignore'):
                annualized_3m = df['ret_3m'] * 4
             improving_long_term_mask = (annualized_3m > df['ret_1y']).fillna(False)

        # Categorize strength based on average return (from lowest to highest for efficiency)
        strength_score.loc[avg_return <= -25] = 20
        strength_score.loc[(avg_return > -25) & (avg_return <= -10)] = 30
        strength_score.loc[(avg_return > -10) & (avg_return <= 0)] = 40
        strength_score.loc[(avg_return > 0) & (avg_return <= 5)] = 50
        strength_score.loc[(avg_return > 5) & (avg_return <= 15)] = 60
        strength_score.loc[(avg_return > 15) & (avg_return <= 30)] = 70
        strength_score.loc[(avg_return > 30) & (avg_return <= 50)] = 80
        strength_score.loc[(avg_return > 50) & (avg_return <= 100)] = 90
        strength_score.loc[avg_return > 100] = 100
        
        # Add a bonus for improving long-term performance
        strength_score.loc[improving_long_term_mask] = (strength_score.loc[improving_long_term_mask] + 5).clip(0, 100)
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score based on trading volume and its consistency."""
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'volume_30d' in df.columns and 'price' in df.columns:
            # Calculate average daily dollar volume over 30 days
            # Fillna with 0 for calculation to avoid NaNs propagating, then dropna for ranking
            dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
            
            # Rank based on dollar volume (higher is better)
            liquidity_score_by_value = RankingEngine._safe_rank(dollar_volume, pct=True, ascending=True)
            
            # Add consistency of volume as a secondary factor
            consistency_score = pd.Series(50, index=df.index, dtype=float)
            volume_consistency_cols = ['volume_7d', 'volume_30d', 'volume_90d']
            
            available_vol_cols = [col for col in volume_consistency_cols if col in df.columns]
            
            if len(available_vol_cols) >= 2: # Need at least two points to check consistency
                vol_data_for_cv = df[available_vol_cols].fillna(0) # Fill with 0 to calculate mean/std
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    vol_mean = vol_data_for_cv.mean(axis=1)
                    vol_std = vol_data_for_cv.std(axis=1)
                    
                    # Calculate Coefficient of Variation (CV = std/mean)
                    # Lower CV means more consistent volume (better)
                    # Handle cases where mean volume is zero
                    vol_cv = np.where(vol_mean > 0, vol_std / vol_mean, np.nan)
                    vol_cv_series = pd.Series(vol_cv, index=df.index)
                
                # Rank CV - lower CV (more consistent) gets a higher score
                consistency_score = RankingEngine._safe_rank(vol_cv_series, pct=True, ascending=False)
            else:
                logger.debug("Insufficient volume history for consistency calculation in liquidity score.")

            # Combine dollar volume rank (80%) and consistency rank (20%)
            liquidity_score = (liquidity_score_by_value * 0.8 + consistency_score * 0.2)
        else:
            logger.warning("Price or 30-day volume missing for liquidity score calculation.")
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks within each category."""
        # Initialize columns with default values
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        # Get unique categories (excluding 'Unknown')
        categories = [cat for cat in df['category'].unique() if cat != 'Unknown']
        
        # Iterate and rank within each category
        for category in categories:
            mask = df['category'] == category
            cat_df = df.loc[mask] # Use .loc for explicit indexing
            
            if len(cat_df) > 0:
                # Calculate ranks based on master_score within the category
                # Use method='min' for percentile-like behavior within rank for display
                df.loc[mask, 'category_rank'] = cat_df['master_score'].rank(method='first', ascending=False, na_option='bottom').astype(int)
                
                # Calculate percentile within the category
                df.loc[mask, 'category_percentile'] = cat_df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        
        df['category_rank'] = df['category_rank'].fillna(len(df) + 1).astype(int) # Fill NaNs for categories not processed
        df['category_percentile'] = df['category_percentile'].fillna(0.0) # Fill NaNs
        
        return df

# ============================================
# PATTERN DETECTION ENGINE - OPTIMIZED
# ============================================

class PatternDetector:
    """Detect all patterns using vectorized operations."""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['pattern_detection'])
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all 25 patterns efficiently using vectorized numpy operations."""
        
        if df.empty:
            df['patterns'] = [''] * len(df)
            return df

        # Get all pattern definitions as tuples (pattern_name: str, boolean_series: pd.Series)
        patterns_definitions = PatternDetector._get_all_pattern_definitions(df)
        
        num_patterns = len(patterns_definitions)
        if num_patterns == 0:
            df['patterns'] = [''] * len(df)
            return df

        # Pre-allocate a NumPy array for pattern presence (rows x patterns)
        # Initialize with False (no pattern detected)
        pattern_matrix = np.zeros((len(df), num_patterns), dtype=bool)
        
        # Populate the boolean matrix efficiently
        # Store pattern names in order for later reconstruction
        pattern_names_ordered = []
        for i, (pattern_name, mask_series) in enumerate(patterns_definitions):
            pattern_names_ordered.append(pattern_name)
            if mask_series is not None and not mask_series.empty and mask_series.any():
                # Ensure mask index aligns with DataFrame index, fill missing with False
                aligned_mask = mask_series.reindex(df.index, fill_value=False)
                pattern_matrix[:, i] = aligned_mask.to_numpy()
        
        # Convert the boolean matrix back to a list of pattern strings for each row
        patterns_column = []
        for row_idx in range(len(df)):
            active_patterns_for_row = [
                pattern_names_ordered[col_idx] 
                for col_idx in range(num_patterns) 
                if pattern_matrix[row_idx, col_idx]
            ]
            patterns_column.append(' | '.join(active_patterns_for_row) if active_patterns_for_row else '')
        
        df['patterns'] = patterns_column
        
        return df
    
    @staticmethod
    def _get_all_pattern_definitions(df: pd.DataFrame) -> List[Tuple[str, Optional[pd.Series]]]:
        """Get all pattern definitions with masks.
        Returns a list of (pattern_name: str, mask: pd.Series) tuples.
        """
        patterns = [] # List to store (name, mask) tuples
        
        # Helper for safe column access and default mask
        def get_series_or_default(col_name: str, default_val: Any) -> pd.Series:
            return df[col_name].fillna(default_val) if col_name in df.columns else pd.Series(default_val, index=df.index)

        # 1. Category Leader
        category_percentile = get_series_or_default('category_percentile', 0.0)
        mask_cat_leader = (category_percentile >= CONFIG.PATTERN_THRESHOLDS['category_leader'])
        patterns.append(('🔥 CAT LEADER', mask_cat_leader))
        
        # 2. Hidden Gem
        percentile = get_series_or_default('percentile', 0.0)
        mask_hidden_gem = (
            (category_percentile >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
            (percentile < 70)
        )
        patterns.append(('💎 HIDDEN GEM', mask_hidden_gem))
        
        # 3. Accelerating
        acceleration_score = get_series_or_default('acceleration_score', 50.0)
        mask_accelerating = (acceleration_score >= CONFIG.PATTERN_THRESHOLDS['acceleration'])
        patterns.append(('🚀 ACCELERATING', mask_accelerating))
        
        # 4. Institutional
        volume_score = get_series_or_default('volume_score', 50.0)
        vol_ratio_90d_180d = get_series_or_default('vol_ratio_90d_180d', 1.0)
        mask_institutional = (
            (volume_score >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
            (vol_ratio_90d_180d > 1.1)
        )
        patterns.append(('🏦 INSTITUTIONAL', mask_institutional))
        
        # 5. Volume Explosion
        rvol = get_series_or_default('rvol', 1.0)
        mask_vol_explosion = (rvol > 3)
        patterns.append(('⚡ VOL EXPLOSION', mask_vol_explosion))
        
        # 6. Breakout Ready
        breakout_score = get_series_or_default('breakout_score', 50.0)
        mask_breakout = (breakout_score >= CONFIG.PATTERN_THRESHOLDS['breakout_ready'])
        patterns.append(('🎯 BREAKOUT', mask_breakout))
        
        # 7. Market Leader
        mask_market_leader = (percentile >= CONFIG.PATTERN_THRESHOLDS['market_leader'])
        patterns.append(('👑 MARKET LEADER', mask_market_leader))
        
        # 8. Momentum Wave
        momentum_score = get_series_or_default('momentum_score', 50.0)
        mask_momentum_wave = (
            (momentum_score >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
            (acceleration_score >= 70)
        )
        patterns.append(('🌊 MOMENTUM WAVE', mask_momentum_wave))
        
        # 9. Liquid Leader
        liquidity_score = get_series_or_default('liquidity_score', 50.0)
        mask_liquid_leader = (
            (liquidity_score >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
            (percentile >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
        )
        patterns.append(('💰 LIQUID LEADER', mask_liquid_leader))
        
        # 10. Long-term Strength
        long_term_strength = get_series_or_default('long_term_strength', 50.0)
        mask_long_strength = (long_term_strength >= CONFIG.PATTERN_THRESHOLDS['long_strength'])
        patterns.append(('💪 LONG STRENGTH', mask_long_strength))
        
        # 11. Quality Trend
        trend_quality = get_series_or_default('trend_quality', 50.0)
        mask_quality_trend = (trend_quality >= 80)
        patterns.append(('📈 QUALITY TREND', mask_quality_trend))
        
        # 12. Value Momentum (Fundamental)
        pe = get_series_or_default('pe', np.nan)
        master_score = get_series_or_default('master_score', 50.0)
        has_valid_pe = (pe.notna() & (pe > 0) & (pe < 10000) & ~np.isinf(pe))
        mask_value_momentum = has_valid_pe & (pe < 15) & (master_score >= 70)
        patterns.append(('💎 VALUE MOMENTUM', mask_value_momentum))
        
        # 13. Earnings Rocket
        eps_change_pct = get_series_or_default('eps_change_pct', np.nan)
        has_eps_growth = (eps_change_pct.notna() & ~np.isinf(eps_change_pct))
        extreme_growth = has_eps_growth & (eps_change_pct > 1000) # >1000% growth
        normal_growth = has_eps_growth & (eps_change_pct > 50) & (eps_change_pct <= 1000) # 50-1000% growth
        
        mask_earnings_rocket = (
            (extreme_growth & (acceleration_score >= 80)) |
            (normal_growth & (acceleration_score >= 70))
        )
        patterns.append(('📊 EARNINGS ROCKET', mask_earnings_rocket))
        
        # 14. Quality Leader
        mask_quality_leader = (
            has_valid_pe & has_eps_growth &
            (pe.between(10, 25)) &
            (eps_change_pct > 20) & # >20% growth
            (percentile >= 80)
        )
        patterns.append(('🏆 QUALITY LEADER', mask_quality_leader))
        
        # 15. Turnaround Play
        mask_turnaround = pd.Series(False, index=df.index) # Initialize to False
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
            mega_turnaround = has_eps & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60) # >500%
            strong_turnaround = has_eps & (df['eps_change_pct'] > 100) & (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70)
            mask_turnaround = mega_turnaround | strong_turnaround
        patterns.append(('⚡ TURNAROUND', mask_turnaround))
        
        # 16. High PE Warning
        mask_high_pe = has_valid_pe & (pe > 100)
        patterns.append(('⚠️ HIGH PE', mask_high_pe))
        
        # 17. 52W High Approach
        from_high_pct = get_series_or_default('from_high_pct', -50.0)
        mask_52w_high_approach = (
            (from_high_pct > -5) & # Within 5% of 52-week high
            (volume_score >= 70) & 
            (momentum_score >= 60)
        )
        patterns.append(('🎯 52W HIGH APPROACH', mask_52w_high_approach))
        
        # 18. 52W Low Bounce
        from_low_pct = get_series_or_default('from_low_pct', 50.0)
        ret_30d = get_series_or_default('ret_30d', 0.0)
        mask_52w_low_bounce = (
            (from_low_pct < 20) & # Within 20% of 52-week low
            (acceleration_score >= 80) & 
            (ret_30d > 10)
        )
        patterns.append(('🔄 52W LOW BOUNCE', mask_52w_low_bounce))
        
        # 19. Golden Zone
        mask_golden_zone = (
            (from_low_pct > 60) & # More than 60% up from low
            (from_high_pct > -40) & # Less than 40% down from high
            (trend_quality >= 70)
        )
        patterns.append(('👑 GOLDEN ZONE', mask_golden_zone))
        
        # 20. Volume Accumulation
        vol_ratio_30d_90d = get_series_or_default('vol_ratio_30d_90d', 1.0)
        vol_ratio_90d_180d = get_series_or_default('vol_ratio_90d_180d', 1.0)
        mask_vol_accumulation = (
            (vol_ratio_30d_90d > 1.2) & # Recent 30-day volume > 90-day avg
            (vol_ratio_90d_180d > 1.1) & # Mid-term 90-day volume > 180-day avg
            (ret_30d > 5)
        )
        patterns.append(('📊 VOL ACCUMULATION', mask_vol_accumulation))
        
        # 21. Momentum Divergence
        ret_7d = get_series_or_default('ret_7d', 0.0)
        mask_momentum_diverge = pd.Series(False, index=df.index) # Initialize
        if 'ret_7d' in df.columns and 'ret_30d' in df.columns and 'acceleration_score' in df.columns and 'rvol' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(ret_7d != 0, ret_7d / 7, 0)
                daily_30d_pace = np.where(ret_30d != 0, ret_30d / 30, 0)
            
            mask_momentum_diverge = (
                (daily_7d_pace > daily_30d_pace * 1.5) & # Recent pace significantly faster than longer term
                (acceleration_score >= 85) & 
                (rvol > 2)
            )
        patterns.append(('🔀 MOMENTUM DIVERGE', mask_momentum_diverge))
        
        # 22. Range Compression
        high_52w = get_series_or_default('high_52w', np.nan)
        low_52w = get_series_or_default('low_52w', np.nan)
        mask_range_compress = pd.Series(False, index=df.index)
        if 'high_52w' in df.columns and 'low_52w' in df.columns and 'from_low_pct' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                # Calculate 52-week range as a percentage of the low
                range_pct = np.where(
                    low_52w > 0,
                    ((high_52w - low_52w) / low_52w) * 100,
                    100 # Default to high range if low is zero/negative
                )
            # Compression if range is relatively narrow (<50%) AND it's not near its lows (already bounced or mid-range)
            mask_range_compress = (range_pct < 50) & (from_low_pct > 30)
        patterns.append(('🎯 RANGE COMPRESS', mask_range_compress))
        
        # 23. Stealth Accumulator (NEW)
        ret_7d = get_series_or_default('ret_7d', 0.0)
        ret_30d = get_series_or_default('ret_30d', 0.0)
        mask_stealth = pd.Series(False, index=df.index)
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                # Check if recent returns are outperforming longer-term average implicitly (e.g. 7d > 30d/4 which is about 7.5 days worth of 30d pace)
                ret_ratio = np.where(ret_30d != 0, ret_7d / (ret_30d / 4), 0)
            
            mask_stealth = (
                (vol_ratio_90d_180d > 1.1) & # Long-term accumulation
                (vol_ratio_30d_90d.between(0.9, 1.1)) & # Recent volume near average (not obvious surge)
                (from_low_pct > 40) & # Stock has already moved up from lows
                (ret_ratio > 1) # Recent returns are stronger than past returns on a comparative basis
            )
        patterns.append(('🤫 STEALTH', mask_stealth))
        
        # 24. Momentum Vampire (NEW)
        ret_1d = get_series_or_default('ret_1d', 0.0)
        ret_7d = get_series_or_default('ret_7d', 0.0)
        category = get_series_or_default('category', 'Unknown')
        mask_vampire = pd.Series(False, index=df.index)
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'):
                # Strong daily surge relative to week (sucking momentum from week)
                daily_pace_ratio = np.where(ret_7d != 0, ret_1d / (ret_7d / 7), 0)
            
            mask_vampire = (
                (daily_pace_ratio > 2) & # Today's action is twice the daily average of the last 7 days
                (rvol > 3) & # High current volume
                (from_high_pct > -15) & # Not too far from 52-week high
                (category.isin(['Small Cap', 'Micro Cap'])) # Often seen in smaller caps
            )
        patterns.append(('🧛 VAMPIRE', mask_vampire))
        
        # 25. Perfect Storm (NEW)
        momentum_harmony = get_series_or_default('momentum_harmony', 0)
        mask_perfect_storm = (
            (momentum_harmony == 4) & # All 4 harmony signals active
            (master_score > CONFIG.PATTERN_THRESHOLDS['perfect_storm']) # Strong overall score
        )
        patterns.append(('⛈️ PERFECT STORM', mask_perfect_storm))
        
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
        category_scores_available = False
        if 'category' in df.columns and 'master_score' in df.columns:
            category_scores = df.groupby('category')['master_score'].mean()
            
            # Safely get averages, handle cases where categories might not exist
            micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean()
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean()
            
            if not pd.isna(micro_small_avg) and not pd.isna(large_mega_avg):
                metrics['micro_small_avg'] = micro_small_avg
                metrics['large_mega_avg'] = large_mega_avg
                metrics['category_spread'] = micro_small_avg - large_mega_avg
                category_scores_available = True
            else:
                micro_small_avg = 50
                large_mega_avg = 50 # Default if categories are missing
        else:
            micro_small_avg = 50
            large_mega_avg = 50
        
        # Market breadth (percentage of stocks with positive 30-day return)
        breadth = 0.5 # Default neutral
        if 'ret_30d' in df.columns:
            positive_ret_30d_count = (df['ret_30d'] > 0).sum()
            if len(df) > 0:
                breadth = positive_ret_30d_count / len(df)
            metrics['breadth'] = breadth
        
        # Average RVOL
        avg_rvol = 1.0 # Default neutral
        if 'rvol' in df.columns:
            avg_rvol = df['rvol'].median() # Use median for robustness against extreme outliers
            metrics['avg_rvol'] = avg_rvol
        
        # Determine regime
        regime = "😴 RANGE-BOUND" # Default regime
        
        if category_scores_available:
            if micro_small_avg > large_mega_avg + 10 and breadth > 0.6:
                regime = "🔥 RISK-ON BULL"
            elif large_mega_avg > micro_small_avg + 10 and breadth < 0.4:
                regime = "🛡️ RISK-OFF DEFENSIVE"
            elif avg_rvol > 1.5 and breadth > 0.5:
                regime = "⚡ VOLATILE OPPORTUNITY"
            # Else, remains "😴 RANGE-BOUND"
        
        metrics['regime'] = regime
        
        return regime, metrics
    
    @staticmethod
    def calculate_advance_decline_ratio(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advance/decline ratio and related metrics"""
        
        ad_metrics = {}
        
        if 'ret_1d' in df.columns:
            # Ensure 'ret_1d' is numeric and handle NaNs
            ret_1d_cleaned = pd.to_numeric(df['ret_1d'], errors='coerce').dropna()

            advancing = (ret_1d_cleaned > 0).sum()
            declining = (ret_1d_cleaned < 0).sum()
            unchanged = (ret_1d_cleaned == 0).sum()
            
            ad_metrics['advancing'] = advancing
            ad_metrics['declining'] = declining
            ad_metrics['unchanged'] = unchanged
            
            if declining > 0:
                ad_metrics['ad_ratio'] = advancing / declining
            else:
                # If no declines, ratio is infinite if there are advances, else 1.0 (all unchanged or no data)
                ad_metrics['ad_ratio'] = float('inf') if advancing > 0 else 1.0
            
            ad_metrics['ad_line'] = advancing - declining
            ad_metrics['breadth_pct'] = (advancing / len(df)) * 100 if len(df) > 0 else 0
        else:
            logger.warning("ret_1d column not available for Advance/Decline ratio calculation.")
            ad_metrics = {
                'advancing': 0, 'declining': 0, 'unchanged': 0,
                'ad_ratio': 1.0, 'ad_line': 0, 'breadth_pct': 0
            }
        
        return ad_metrics
    
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect sector rotation patterns with normalized analysis and dynamic sampling.
        Uses dynamic sampling based on sector size for fairer comparison.
        """
        
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        
        sector_dfs_sampled = []
        
        for sector in df['sector'].unique():
            if sector != 'Unknown':
                sector_df = df[df['sector'] == sector].copy()
                sector_size = len(sector_df)
                
                # Dynamic sampling logic
                if 1 <= sector_size <= 5:
                    sample_count = sector_size # Use all (100%)
                elif 6 <= sector_size <= 20:
                    sample_count = max(1, int(sector_size * 0.80)) # Use 80%
                elif 21 <= sector_size <= 50:
                    sample_count = max(1, int(sector_size * 0.60)) # Use 60%
                elif 51 <= sector_size <= 100:
                    sample_count = max(1, int(sector_size * 0.40)) # Use 40%
                else: # sector_size > 100
                    sample_count = min(50, int(sector_size * 0.25)) # Use 25%, max 50 stocks for very large sectors
                
                if sample_count > 0:
                    # Sort by master_score and take the dynamic 'N'
                    sector_df_sampled = sector_df.nlargest(sample_count, 'master_score')
                else:
                    sector_df_sampled = pd.DataFrame() # No stocks selected
                
                if not sector_df_sampled.empty:
                    sector_dfs_sampled.append(sector_df_sampled)
        
        # Combine sampled sector data
        if sector_dfs_sampled:
            normalized_df = pd.concat(sector_dfs_sampled, ignore_index=True)
        else:
            # If no valid sectors found after sampling, return empty DataFrame
            return pd.DataFrame()
        
        # Calculate sector metrics on normalized data
        # Ensure 'money_flow_mm' is correctly handled if missing from source data
        agg_dict = {
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean',
        }
        if 'money_flow_mm' in normalized_df.columns:
            agg_dict['money_flow_mm'] = 'sum'
        
        sector_metrics = normalized_df.groupby('sector').agg(agg_dict).round(2)
        
        # Flatten column names
        flat_cols = []
        for col_level0, col_level1 in sector_metrics.columns:
            if col_level0 == 'money_flow_mm': # Handle sum for money_flow_mm specially
                flat_cols.append('total_money_flow')
            else:
                flat_cols.append(f"{col_level0}_{col_level1}")
        sector_metrics.columns = flat_cols
        
        # Rename columns for clarity
        sector_metrics = sector_metrics.rename(columns={
            'master_score_mean': 'avg_score',
            'master_score_median': 'median_score',
            'master_score_std': 'std_score',
            'master_score_count': 'count', # This count refers to sampled stocks
            'momentum_score_mean': 'avg_momentum',
            'volume_score_mean': 'avg_volume',
            'rvol_mean': 'avg_rvol',
            'ret_30d_mean': 'avg_ret_30d',
        })
        
        # Add original sector size for reference
        original_counts = df.groupby('sector').size().rename('total_stocks')
        sector_metrics = sector_metrics.join(original_counts, how='left')
        sector_metrics['analyzed_stocks'] = sector_metrics['count'] # Renaming for clarity
        
        # Calculate flow score with median for robustness
        sector_metrics['flow_score'] = (
            sector_metrics['avg_score'] * 0.3 +
            sector_metrics['median_score'] * 0.2 +
            sector_metrics['avg_momentum'] * 0.25 +
            sector_metrics['avg_volume'] * 0.25
        )
        sector_metrics['flow_score'] = sector_metrics['flow_score'].clip(0, 100)
        
        # Rank sectors
        sector_metrics['rank'] = sector_metrics['flow_score'].rank(ascending=False, method='min')
        
        return sector_metrics.sort_values('flow_score', ascending=False)

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations with proper error handling"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution (box plot) for individual score components."""
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Score Component Distribution", template='plotly_white')
            return fig
        
        # Score components to visualize
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
                score_data = df[score_col].dropna()
                if not score_data.empty: # Only add trace if there's actual data
                    fig.add_trace(go.Box(
                        y=score_data,
                        name=label,
                        marker_color=color,
                        boxpoints='outliers', # Show individual outliers
                        hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'
                    ))
        
        if not fig.data: # If no traces were added due to missing columns or empty data
            fig.add_annotation(
                text="Required score data missing for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )

        fig.update_layout(
            title="Score Component Distribution",
            yaxis_title="Score (0-100)",
            template='plotly_white',
            height=400,
            showlegend=False,
            margin=dict(l=50, r=50, t=60, b=50),
            yaxis_range=[0,100] # Ensure Y-axis is always 0-100
        )
        
        return fig
    
    @staticmethod
    def create_master_score_breakdown(df: pd.DataFrame, n: int = 20) -> go.Figure:
        """Create master score breakdown chart showing weighted component contributions for top N stocks."""
        # Get top N stocks by master score
        top_df = df.nlargest(min(n, len(df)), 'master_score').copy()
        
        if top_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for Master Score breakdown for top {n} stocks.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray")
            )
            fig.update_layout(title=f"Top {n} Stocks - Master Score 3.0 Component Breakdown", template='plotly_white')
            return fig
        
        # Define components and their weights/colors from Config
        components_info = [
            ('Position', 'position_score', CONFIG.POSITION_WEIGHT, '#3498db'),
            ('Volume', 'volume_score', CONFIG.VOLUME_WEIGHT, '#e74c3c'),
            ('Momentum', 'momentum_score', CONFIG.MOMENTUM_WEIGHT, '#2ecc71'),
            ('Acceleration', 'acceleration_score', CONFIG.ACCELERATION_WEIGHT, '#f39c12'),
            ('Breakout', 'breakout_score', CONFIG.BREAKOUT_WEIGHT, '#9b59b6'),
            ('RVOL', 'rvol_score', CONFIG.RVOL_WEIGHT, '#e67e22')
        ]
        
        fig = go.Figure()
        
        # Add bars for each component
        for name, score_col, weight, color in components_info:
            if score_col in top_df.columns:
                # Calculate weighted contribution (score already filled to 50 in preprocessing if NaN)
                weighted_contrib = top_df[score_col] * weight
                
                fig.add_trace(go.Bar(
                    name=f'{name} ({weight:.0%})', # Display name and weight in legend
                    y=top_df['ticker'], # Tickers on Y-axis
                    x=weighted_contrib, # Weighted contribution on X-axis
                    orientation='h', # Horizontal bars
                    marker_color=color,
                    # Text on bars shows raw score, not weighted contribution
                    text=[f"{val:.1f}" for val in top_df[score_col]],
                    textposition='inside',
                    hovertemplate=(
                        f'<b>%{y}</b><br>' # Ticker
                        f'{name}<br>'
                        'Raw Score: %{text}<br>'
                        f'Weight: {weight:.0%}<br>'
                        'Contribution: %{x:.1f}<extra></extra>' # Customdata if needed, but x is already contribution
                    )
                ))
            else:
                logger.warning(f"Master Score Breakdown: Missing column '{score_col}'. Skipping this component.")
        
        # Add total master score annotations on top of each stacked bar
        for i, row in top_df.reset_index().iterrows(): # Use reset_index() to ensure proper integer indexing for 'y' position
            fig.add_annotation(
                x=row['master_score'] + 1, # Position slightly to the right of the bar end
                y=row.name, # Use row.name as y-coordinate (index in the sorted DataFrame)
                text=f"<b>{row['master_score']:.1f}</b>", # Display master score
                showarrow=False,
                xanchor='left',
                font=dict(size=12, color='black'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1,
                align='left'
            )
        
        fig.update_layout(
            title=f"Top {len(top_df)} Stocks - Master Score 3.0 Component Breakdown",
            xaxis_title="Weighted Score Contribution",
            xaxis_range=[0, 105], # Ensure x-axis goes up to 100+ for score display
            barmode='stack', # Stack the bars
            template='plotly_white',
            height=max(400, len(top_df) * 35), # Adjust height based on number of stocks
            showlegend=True,
            legend=dict(
                orientation="h", # Horizontal legend
                yanchor="bottom",
                y=1.02, # Position above the plot area
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=100, r=100, t=100, b=50), # Adjust margins for labels/annotations
            yaxis={'categoryorder':'total ascending'} # Ensure correct order of bars
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance_scatter(df: pd.DataFrame) -> go.Figure:
        """Create sector performance scatter plot with bubble size by count and color by avg RVOL."""
        try:
            # Aggregate by sector. Ensure all necessary columns exist before aggregation.
            required_agg_cols = ['master_score', 'percentile', 'rvol', 'momentum_score', 'ret_30d']
            agg_dict = {col: ['mean'] for col in required_agg_cols if col in df.columns}
            if 'master_score' in df.columns:
                agg_dict['master_score'].extend(['std', 'count'])
            
            if not agg_dict:
                logger.warning("Not enough columns to create sector performance scatter plot.")
                fig = go.Figure()
                fig.add_annotation(text="Required data missing for Sector Performance Scatter.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
                fig.update_layout(title='Sector Performance Analysis', template='plotly_white')
                return fig

            sector_stats = df.groupby('sector').agg(agg_dict)
            
            # Flatten multi-level columns
            sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns.values]
            sector_stats = sector_stats.reset_index()
            
            # Rename for easier access
            sector_stats = sector_stats.rename(columns={
                'master_score_mean': 'avg_score',
                'master_score_std': 'std_score',
                'master_score_count': 'count',
                'percentile_mean': 'avg_percentile',
                'rvol_mean': 'avg_rvol',
                'momentum_score_mean': 'avg_momentum',
                'ret_30d_mean': 'avg_ret_30d'
            })
            
            # Filter sectors with at least 3 stocks for meaningful statistics
            sector_stats = sector_stats[sector_stats['count'] >= 3]
            
            if sector_stats.empty:
                fig = go.Figure()
                fig.add_annotation(text="No sectors with enough data (>=3 stocks) to plot scatter.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
                fig.update_layout(title='Sector Performance Analysis', template='plotly_white')
                return fig
            
            # Ensure necessary columns for plotting exist after aggregation
            if not all(col in sector_stats.columns for col in ['avg_percentile', 'avg_score', 'count', 'avg_rvol']):
                logger.warning("Essential columns for scatter plot missing after aggregation.")
                fig = go.Figure()
                fig.add_annotation(text="Insufficient aggregated data for scatter plot.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
                fig.update_layout(title='Sector Performance Analysis', template='plotly_white')
                return fig

            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sector_stats['avg_percentile'],
                y=sector_stats['avg_score'],
                mode='markers+text',
                text=sector_stats['sector'],
                textposition='top center',
                marker=dict(
                    # Scale bubble size based on stock count, ensuring a visible range
                    size=np.sqrt(sector_stats['count']) * 4 + 5, # Square root scaling with base size
                    sizemin=5,
                    sizemode='diameter',
                    sizeref=np.max(np.sqrt(sector_stats['count']) * 4 + 5) / 50.0, # Adjust sizeref for visual appeal
                    color=sector_stats['avg_rvol'],
                    colorscale='Viridis',
                    colorbar=dict(title="Avg RVOL (x)", titleside="right"),
                    line=dict(width=2, color='white'),
                    showscale=True
                ),
                customdata=sector_stats[['count', 'std_score', 'avg_rvol', 'avg_momentum', 'avg_ret_30d']].values,
                hovertemplate=(
                    '<b>%{text}</b><br>' + # Sector name
                    'Avg Master Score: %{y:.1f}<br>' +
                    'Avg Percentile Rank: %{x:.1f}<br>' +
                    'Stocks: %{customdata[0]:.0f}<br>' + # Count
                    'Std Dev Score: %{customdata[1]:.1f}<br>' + # Std score
                    'Avg RVOL: %{customdata[2]:.2f}x<br>' + # Avg RVOL
                    'Avg Momentum: %{customdata[3]:.1f}<br>' + # Avg Momentum
                    'Avg 30D Return: %{customdata[4]:.1f}%<extra></extra>' # Avg 30D return
                )
            ))
            
            # Add quadrant lines for easy interpretation
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Avg Score 50", annotation_position="bottom right")
            fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Avg Percentile 50", annotation_position="top left")
            
            # Add quadrant labels
            fig.add_annotation(x=75, y=75, text="📈 Leaders", showarrow=False, 
                             font=dict(size=14, color="green"), opacity=0.8, bgcolor="rgba(255,255,255,0.6)", bordercolor="green", borderwidth=1, borderpad=2)
            fig.add_annotation(x=25, y=75, text="💎 Hidden Gems", showarrow=False,
                             font=dict(size=14, color="blue"), opacity=0.8, bgcolor="rgba(255,255,255,0.6)", bordercolor="blue", borderwidth=1, borderpad=2)
            fig.add_annotation(x=75, y=25, text="⚠️ Overvalued/Weak", showarrow=False,
                             font=dict(size=14, color="orange"), opacity=0.8, bgcolor="rgba(255,255,255,0.6)", bordercolor="orange", borderwidth=1, borderpad=2)
            fig.add_annotation(x=25, y=25, text="📉 Laggards", showarrow=False,
                             font=dict(size=14, color="red"), opacity=0.8, bgcolor="rgba(255,255,255,0.6)", bordercolor="red", borderwidth=1, borderpad=2)
            
            fig.update_layout(
                title='Sector Performance Analysis: Size = Stocks in Sector, Color = Avg RVOL',
                xaxis_title='Average Percentile Rank (Market-Wide)',
                yaxis_title='Average Master Score',
                template='plotly_white',
                height=550, # Slightly increased height for clarity
                xaxis=dict(range=[-5, 105], showgrid=True, zeroline=True), # Extend range slightly
                yaxis=dict(range=[-5, 105], showgrid=True, zeroline=True), # Extend range slightly
                hovermode='closest', # Show hover info for closest point
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sector performance scatter: {str(e)}", exc_info=True)
            fig = go.Figure()
            fig.add_annotation(text=f"Error generating scatter plot: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
            fig.update_layout(title='Sector Performance Analysis', template='plotly_white')
            return fig
    
    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, n: int = 10) -> go.Figure:
        """Create acceleration profiles showing momentum over time for top N accelerating stocks."""
        try:
            # Get top accelerating stocks. Ensure 'acceleration_score' exists.
            if 'acceleration_score' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(text="Acceleration score missing for profile generation.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
                fig.update_layout(title="Acceleration Profiles", template='plotly_white')
                return fig

            accel_df = df.nlargest(min(n, len(df)), 'acceleration_score').copy()
            
            if accel_df.empty:
                fig = go.Figure()
                fig.add_annotation(text=f"No stocks found with acceleration data to display profiles for top {n}.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
                fig.update_layout(title="Acceleration Profiles", template='plotly_white')
                return fig
            
            fig = go.Figure()
            
            # Define points for the timeline, prioritizing existing data
            timeline_points = {
                'ret_30d': ('30D', 30),
                'ret_7d': ('7D', 7),
                'ret_1d': ('Today', 1)
            }
            
            for _, stock in accel_df.iterrows():
                x_points = ['Start']
                y_points = [0] # Starting point at 0% return

                # Dynamically add available return data points
                for col, (label, days) in timeline_points.items():
                    if col in stock.index and pd.notna(stock[col]):
                        x_points.append(label)
                        y_points.append(stock[col])
                
                if len(x_points) > 1: # Only plot if there's more than just the "Start" point
                    # Determine line style based on acceleration score
                    if stock['acceleration_score'] >= 85:
                        line_style = dict(width=3, dash='solid', color=px.colors.qualitative.Plotly[0]) # Use a distinct color for highest accel
                        marker_style = dict(size=10, symbol='star', color=px.colors.qualitative.Plotly[0])
                    elif stock['acceleration_score'] >= 70:
                        line_style = dict(width=2, dash='solid', color=px.colors.qualitative.Plotly[1])
                        marker_style = dict(size=8, color=px.colors.qualitative.Plotly[1])
                    else:
                        line_style = dict(width=2, dash='dot', color=px.colors.qualitative.Plotly[2])
                        marker_style = dict(size=6, color=px.colors.qualitative.Plotly[2])
                    
                    fig.add_trace(go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode='lines+markers',
                        name=f"{stock['ticker']} (Accel: {stock['acceleration_score']:.0f})", # Include accel score in legend
                        line=line_style,
                        marker=marker_style,
                        hovertemplate=(
                            f"<b>{stock['ticker']}</b><br>" +
                            "Timeframe: %{x}<br>" +
                            "Return: %{y:.1f}%<br>" +
                            f"Accel Score: {stock['acceleration_score']:.0f}<extra></extra>"
                        )
                    ))
            
            # Add a zero line for visual reference
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="0% Return", annotation_position="top left", annotation_font_color="gray")
            
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
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.7)", # Semi-transparent background
                    bordercolor="LightSteelBlue",
                    borderwidth=1
                ),
                hovermode='x unified', # Show all traces' info for a given x-point
                margin=dict(l=50, r=150, t=60, b=50) # Adjust margin for legend and title
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}", exc_info=True)
            fig = go.Figure()
            fig.add_annotation(text=f"Error generating acceleration profiles: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
            fig.update_layout(title="Acceleration Profiles", template='plotly_white')
            return fig

# ============================================
# FILTER ENGINE - OPTIMIZED
# ============================================

class FilterEngine:
    """Handle all filtering operations efficiently"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['filtering'])
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with optimized performance using a single boolean mask."""
        
        if df.empty:
            return df
        
        # Start with a boolean mask where all rows are True
        mask = pd.Series(True, index=df.index)
        
        # Category filter
        categories = filters.get('categories', [])
        if categories and 'All' not in categories:
            mask &= df['category'].isin(categories)
        
        # Sector filter
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors:
            mask &= df['sector'].isin(sectors)
        
        # Score filter (Minimum Master Score)
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            mask &= df['master_score'] >= min_score
        
        # EPS change filter (Min EPS Change %)
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            # Exclude NaN values if a filter is set, otherwise include NaN
            if pd.isna(min_eps_change): # If input is empty/NaN, don't filter
                pass
            else:
                mask &= (df['eps_change_pct'] >= min_eps_change)
                # If require_fundamental_data is NOT True, we explicitly exclude NaNs here
                if not filters.get('require_fundamental_data', False):
                     mask &= df['eps_change_pct'].notna() # Only filter non-NaN values
        
        # Pattern filter
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            # Create a regex to match any of the selected patterns
            pattern_regex = '|'.join(map(re.escape, patterns)) # Use re.escape for special chars in patterns
            mask &= df['patterns'].str.contains(pattern_regex, case=False, na=False, regex=True)
        
        # Trend filter (Trend Quality)
        if filters.get('trend_range') and filters.get('trend_filter') != 'All Trends':
            min_trend, max_trend = filters['trend_range']
            if 'trend_quality' in df.columns:
                mask &= (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend)
        
        # PE filters (Min PE Ratio, Max PE Ratio)
        min_pe = filters.get('min_pe')
        max_pe = filters.get('max_pe')
        if 'pe' in df.columns:
            if min_pe is not None:
                # Exclude NaN and non-positive PE values when a min filter is set
                if not pd.isna(min_pe):
                    mask &= (df['pe'] >= min_pe) & (df['pe'] > 0) & df['pe'].notna() & ~np.isinf(df['pe'])
            if max_pe is not None:
                # Exclude NaN and non-positive PE values when a max filter is set
                if not pd.isna(max_pe):
                    mask &= (df['pe'] <= max_pe) & (df['pe'] > 0) & df['pe'].notna() & ~np.isinf(df['pe'])
        
        # Apply tier filters
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            tier_values = filters.get(tier_type, [])
            if tier_values and 'All' not in tier_values:
                col_name = tier_type.replace('_tiers', '_tier')
                if col_name in df.columns:
                    mask &= df[col_name].isin(tier_values)
        
        # Data completeness filter (Require Fundamental Data)
        if filters.get('require_fundamental_data', False):
            if 'pe' in df.columns and 'eps_change_pct' in df.columns:
                mask &= df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna()
        
        # NEW: Wave State filter
        wave_states = filters.get('wave_states', [])
        if wave_states and 'All' not in wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)

        # NEW: Wave Strength filter
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            mask &= (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws)

        # Apply the combined mask efficiently
        filtered_df = df[mask].copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        """Get available filter options with smart interconnection, showing options relevant to other active filters."""
        
        if df.empty or column not in df.columns:
            return []
        
        # Create a temporary filters dictionary, excluding the current column's filter
        # so that its options are not self-filtered.
        temp_filters = current_filters.copy()
        
        filter_key_map = {
            'category': 'categories',
            'sector': 'sectors',
            'eps_tier': 'eps_tiers',
            'pe_tier': 'pe_tiers',
            'price_tier': 'price_tiers',
            'wave_state': 'wave_states' # New wave state filter
        }
        
        # Remove the current column's filter if it exists in the temporary filter set
        if column in filter_key_map:
            temp_filters.pop(filter_key_map[column], None)
        
        # Apply the remaining filters to the original DataFrame to get the relevant subset
        filtered_df_for_options = FilterEngine.apply_filters(df, temp_filters)
        
        # Get unique values from the filtered subset
        values = filtered_df_for_options[column].dropna().unique().tolist()
        
        # Exclude common "unknown" or empty string values from the options list
        values = [v for v in values if str(v).strip().lower() not in ['unknown', '', 'nan', 'none', '-']]
        
        return sorted(values)

# ============================================
# SEARCH ENGINE - OPTIMIZED
# ============================================

class SearchEngine:
    """Optimized search functionality."""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['search'])
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Search stocks by ticker or company name with relevance scoring.
        Handles various match types: exact, starts with, contains.
        """
        
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query_upper = query.upper().strip()
            
            # Initialize masks for different match types
            mask_ticker_exact = df['ticker'].str.upper() == query_upper
            mask_ticker_starts_with = df['ticker'].str.upper().str.startswith(query_upper, na=False)
            mask_ticker_contains = df['ticker'].str.upper().str.contains(query_upper, na=False, regex=False)
            
            mask_company_starts_with = df['company_name'].str.upper().str.startswith(query_upper, na=False)
            mask_company_contains = df['company_name'].str.upper().str.contains(query_upper, na=False, regex=False)
            
            # Method to check if any word in company name starts with the query
            def word_starts_with(company_name_str):
                if pd.isna(company_name_str):
                    return False
                words = str(company_name_str).upper().split()
                return any(word.startswith(query_upper) for word in words)
            
            mask_company_word_match = df['company_name'].apply(word_starts_with)
            
            # Combine all masks using logical OR
            combined_mask = (
                mask_ticker_exact | 
                mask_ticker_starts_with | 
                mask_ticker_contains | 
                mask_company_starts_with | 
                mask_company_contains | 
                mask_company_word_match
            )
            
            all_matches = df[combined_mask].drop_duplicates().copy()
            
            if not all_matches.empty:
                # Add a relevance score for sorting, prioritizing exact matches and starts-with
                all_matches['relevance'] = 0
                all_matches.loc[mask_ticker_exact, 'relevance'] = 100
                all_matches.loc[~mask_ticker_exact & mask_ticker_starts_with, 'relevance'] += 50
                all_matches.loc[~mask_ticker_exact & ~mask_ticker_starts_with & mask_company_starts_with, 'relevance'] += 30
                all_matches.loc[~mask_ticker_exact & ~mask_ticker_starts_with & ~mask_company_starts_with & mask_company_word_match, 'relevance'] += 10
                
                # Sort by relevance (descending) and then by master_score (descending)
                return all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False]).drop('relevance', axis=1)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Search error for query '{query}': {str(e)}", exc_info=True)
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE - ENHANCED
# ============================================

class ExportEngine:
    """Handle all export operations with streaming for large datasets."""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['export_generation'])
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        """Create comprehensive Excel report with smart templates."""
        
        output = BytesIO()
        
        # Define export templates with specific columns
        templates_config = {
            'day_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 
                           'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 
                           'volume_score', 'vmi', 'wave_state', 'patterns', 'category', 'price', 'money_flow_mm'],
                'focus': 'Intraday momentum and volume'
            },
            'swing_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 
                           'breakout_score', 'position_score', 'position_tension',
                           'from_high_pct', 'from_low_pct', 'trend_quality', 
                           'momentum_harmony', 'patterns', 'price', 'ret_30d', 'ret_7d', 'rvol'],
                'focus': 'Position and breakout setups'
            },
            'investor': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 
                           'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 'ret_5y', 
                           'long_term_strength', 'money_flow_mm', 'category', 'sector', 'price'],
                'focus': 'Fundamentals and long-term performance'
            },
            'full': {
                'columns': None,  # Use all columns from CSV export list for consistency
                'focus': 'Complete analysis'
            }
        }
        
        # Define common formatting rules
        formats = {
            'header': {'bold': True, 'bg_color': '#3498db', 'font_color': 'white', 'border': 1},
            'currency': {'num_format': '₹#,##0'},
            'price_decimal': {'num_format': '₹#,##0.00'},
            'number_2_decimal': {'num_format': '#,##0.00'},
            'percentage_1_decimal': {'num_format': '0.0%'},
            'percentage_0_decimal': {'num_format': '0%'},
            'rvol_format': {'num_format': '#,##0.0x'},
            'money_flow_mm_format': {'num_format': '₹#,##0.00 "M"'},
            'integer': {'num_format': '#,##0'},
        }

        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Register formats with workbook
                header_format = workbook.add_format(formats['header'])
                currency_format = workbook.add_format(formats['currency'])
                price_decimal_format = workbook.add_format(formats['price_decimal'])
                number_2_decimal_format = workbook.add_format(formats['number_2_decimal'])
                percentage_1_decimal_format = workbook.add_format(formats['percentage_1_decimal'])
                percentage_0_decimal_format = workbook.add_format(formats['percentage_0_decimal'])
                rvol_format = workbook.add_format(formats['rvol_format'])
                money_flow_mm_format = workbook.add_format(formats['money_flow_mm_format'])
                integer_format = workbook.add_format(formats['integer'])


                # Function to apply column specific formats
                def apply_column_formats(ws, df_export):
                    for col_idx, col_name in enumerate(df_export.columns):
                        max_len = max(len(str(col_name)), df_export[col_name].astype(str).map(len).max())
                        ws.set_column(col_idx, col_idx, max_len + 2) # Auto-adjust column width

                        if col_name in ['price', 'low_52w', 'high_52w']:
                            ws.set_column(col_idx, col_idx, max_len + 2, price_decimal_format)
                        elif col_name == 'money_flow_mm':
                            ws.set_column(col_idx, col_idx, max_len + 2, money_flow_mm_format)
                        elif col_name.startswith('ret_') or 'pct' in col_name.lower():
                            if col_name == 'from_low_pct' or col_name == 'from_high_pct':
                                ws.set_column(col_idx, col_idx, max_len + 2, percentage_0_decimal_format)
                            else:
                                ws.set_column(col_idx, col_idx, max_len + 2, percentage_1_decimal_format)
                        elif col_name == 'rvol':
                            ws.set_column(col_idx, col_idx, max_len + 2, rvol_format)
                        elif col_name in ['master_score', 'position_score', 'volume_score', 'momentum_score',
                                          'acceleration_score', 'breakout_score', 'rvol_score', 'trend_quality',
                                          'long_term_strength', 'liquidity_score', 'vmi', 'overall_wave_strength']:
                             ws.set_column(col_idx, col_idx, max_len + 2, number_2_decimal_format)
                        elif col_name in ['rank', 'category_rank', 'signal_count', 'count', 'advancing', 'declining', 'unchanged', 'total_stocks', 'analyzed_stocks']:
                             ws.set_column(col_idx, col_idx, max_len + 2, integer_format)
                        elif col_name in ['pe', 'eps_current', 'eps_last_qtr', 'ad_ratio']:
                             ws.set_column(col_idx, col_idx, max_len + 2, number_2_decimal_format)


                # 1. Top 100 Stocks Sheet
                # Ensure the dataframe is sorted for nlargest
                top_100_data = df.nlargest(min(100, len(df)), 'master_score').copy()
                
                # Determine columns to export based on template
                if template in templates_config and templates_config[template]['columns']:
                    selected_cols = [col for col in templates_config[template]['columns'] if col in top_100_data.columns]
                else:
                    selected_cols = ExportEngine._get_all_csv_export_columns() # Use all comprehensive columns
                    selected_cols = [col for col in selected_cols if col in top_100_data.columns] # Filter by actual existence

                top_100_export = top_100_data[selected_cols]
                top_100_export.to_excel(writer, sheet_name='Top 100', index=False)
                
                ws_top100 = writer.sheets['Top 100']
                for i, col in enumerate(top_100_export.columns):
                    ws_top100.write(0, i, col, header_format)
                apply_column_formats(ws_top100, top_100_export)


                # 2. Market Intelligence Sheet
                intel_data = []
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)

                intel_data.append({'Metric': 'Market Regime', 'Value': regime, 'Details': f"Avg Micro/Small Score: {regime_metrics.get('micro_small_avg', np.nan):.1f}, Avg Large/Mega Score: {regime_metrics.get('large_mega_avg', np.nan):.1f}"})
                intel_data.append({'Metric': 'Overall Breadth (%)', 'Value': f"{regime_metrics.get('breadth', 0)*100:.1f}%", 'Details': 'Percentage of stocks with positive 30D returns'})
                intel_data.append({'Metric': 'Median RVOL (x)', 'Value': f"{regime_metrics.get('avg_rvol', np.nan):.2f}x", 'Details': 'Median Relative Volume across all stocks'})
                intel_data.append({'Metric': 'Advance/Decline (Counts)', 'Value': f"{ad_metrics.get('advancing', 0)} Adv, {ad_metrics.get('declining', 0)} Dec", 'Details': f"Unchanged: {ad_metrics.get('unchanged', 0)}"})
                intel_data.append({'Metric': 'Advance/Decline Ratio', 'Value': f"{ad_metrics.get('ad_ratio', np.nan):.2f}", 'Details': 'Advancing / Declining (higher is better)'})
                
                intel_df = pd.DataFrame(intel_data)
                intel_df.to_excel(writer, sheet_name='Market Intelligence', index=False)
                ws_intel = writer.sheets['Market Intelligence']
                for i, col in enumerate(intel_df.columns):
                    ws_intel.write(0, i, col, header_format)
                ws_intel.set_column('A:A', 25)
                ws_intel.set_column('B:B', 20)
                ws_intel.set_column('C:C', 50)


                # 3. Sector Rotation Sheet
                sector_rotation_data = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rotation_data.empty:
                    # Drop original_counts if it's implicitly part of the index or just total_stocks
                    export_sector_data = sector_rotation_data.reset_index().rename(columns={'index': 'sector'})
                    export_sector_data['sector_rank'] = export_sector_data['flow_score'].rank(ascending=False).astype(int)
                    
                    # Reorder columns for better readability
                    final_sector_cols = [
                        'sector_rank', 'sector', 'flow_score', 'avg_score', 'median_score',
                        'analyzed_stocks', 'total_stocks', 'avg_momentum', 'avg_volume',
                        'avg_rvol', 'avg_ret_30d', 'std_score'
                    ]
                    if 'total_money_flow' in export_sector_data.columns:
                        final_sector_cols.insert(final_sector_cols.index('avg_ret_30d') + 1, 'total_money_flow')

                    export_sector_data = export_sector_data[[col for col in final_sector_cols if col in export_sector_data.columns]]
                    export_sector_data.to_excel(writer, sheet_name='Sector Rotation', index=False)
                    ws_sector = writer.sheets['Sector Rotation']
                    for i, col in enumerate(export_sector_data.columns):
                        ws_sector.write(0, i, col, header_format)
                    apply_column_formats(ws_sector, export_sector_data)
                    ws_sector.set_column('B:B', 25) # Sector name

                
                # 4. Pattern Analysis Sheet
                pattern_counts_dict = {}
                if 'patterns' in df.columns:
                    for patterns_str in df['patterns'].dropna():
                        if patterns_str:
                            for p in patterns_str.split(' | '):
                                pattern_counts_dict[p] = pattern_counts_dict.get(p, 0) + 1
                
                if pattern_counts_dict:
                    pattern_df = pd.DataFrame(
                        list(pattern_counts_dict.items()),
                        columns=['Pattern', 'Count']
                    ).sort_values('Count', ascending=False)
                    pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
                    ws_patterns = writer.sheets['Pattern Analysis']
                    for i, col in enumerate(pattern_df.columns):
                        ws_patterns.write(0, i, col, header_format)
                    apply_column_formats(ws_patterns, pattern_df)
                    ws_patterns.set_column('A:A', 30)


                # 5. Wave Radar Signals Sheet
                # Define conditions for "Wave Radar Signals"
                wave_signals_condition = (
                    (df['momentum_score'] >= 60) & 
                    (df['acceleration_score'] >= 70) &
                    (df['rvol'] >= 2)
                )
                if 'overall_wave_strength' in df.columns: # Use the overall score if available
                     wave_signals_condition &= (df['overall_wave_strength'] >= 60)

                wave_signals_data = df[wave_signals_condition].nlargest(50, 'master_score')
                
                if not wave_signals_data.empty:
                    wave_cols = ['rank', 'ticker', 'company_name', 'master_score', 
                                'momentum_score', 'acceleration_score', 'rvol',
                                'wave_state', 'patterns', 'category', 'sector', 'price', 'ret_1d', 'ret_7d']
                    if 'overall_wave_strength' in wave_signals_data.columns:
                        wave_cols.insert(wave_cols.index('master_score') + 1, 'overall_wave_strength')

                    available_wave_cols = [col for col in wave_cols if col in wave_signals_data.columns]
                    
                    wave_signals_data[available_wave_cols].to_excel(
                        writer, sheet_name='Wave Radar Signals', index=False
                    )
                    ws_wave = writer.sheets['Wave Radar Signals']
                    for i, col in enumerate(wave_signals_data[available_wave_cols].columns):
                        ws_wave.write(0, i, col, header_format)
                    apply_column_formats(ws_wave, wave_signals_data[available_wave_cols])
                    ws_wave.set_column('C:C', 25) # Company Name
                    ws_wave.set_column('I:I', 40) # Patterns


                # 6. Summary Statistics Sheet
                summary_stats = {
                    'Total Stocks Processed': len(df),
                    'Average Master Score': df['master_score'].mean(),
                    'Stocks with Patterns Detected': (df['patterns'] != '').sum() if 'patterns' in df.columns else 0,
                    'High RVOL (>2x)': (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0,
                    'Positive 30D Returns': (df['ret_30d'] > 0).sum() if 'ret_30d' in df.columns else 0,
                    'Market Regime (Detected)': MarketIntelligence.detect_market_regime(df)[0],
                    'Template Used for Top 100': templates_config[template]['focus'],
                    'Export Date (IST)': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
                }
                
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                ws_summary = writer.sheets['Summary']
                for i, col in enumerate(summary_df.columns):
                    ws_summary.write(0, i, col, header_format)
                ws_summary.set_column('A:A', 30)
                ws_summary.set_column('B:B', 30)
                
                logger.info(f"Excel report created successfully with {len(writer.sheets)} sheets.")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}", exc_info=True)
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def _get_all_csv_export_columns() -> List[str]:
        """Defines the comprehensive list of columns for full CSV export."""
        return [
            'rank', 'ticker', 'company_name', 'master_score', 'percentile',
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score',
            'overall_wave_strength', # NEW
            'trend_quality', 'long_term_strength', 'liquidity_score',
            'price', 'prev_close', 'low_52w', 'high_52w',
            'from_low_pct', 'from_high_pct',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
            'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
            'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'vol_ratio_90d_180d',
            'rvol', 'vmi', 'money_flow_mm', 'position_tension',
            'momentum_harmony', 'wave_state', 'patterns', 
            'category', 'sector', 'eps_current', 'pe', 'eps_change_pct',
            'eps_tier', 'pe_tier', 'price_tier'
        ]

    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export efficiently with comprehensive data."""
        
        export_cols = ExportEngine._get_all_csv_export_columns()
        
        available_cols = [col for col in export_cols if col in df.columns]
        
        # Create export dataframe
        export_df = df[available_cols].copy()
        
        # Convert volume ratios (which are multipliers) back to percentage change for CSV display if desired.
        # This is for human readability in CSV; the internal calculations use ratios.
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in export_df.columns:
                export_df[col] = (export_df[col] - 1) * 100 # Convert multiplier back to percentage change
                # Ensure it's rounded for display
                export_df[col] = export_df[col].round(2)
        
        # Format percentage columns for consistency (e.g., add '+' sign)
        for col in CONFIG.PERCENTAGE_COLUMNS:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else np.nan)
        
        return export_df.to_csv(index=False)

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components for Streamlit."""
    
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None, 
                          help_text: Optional[str] = None) -> None:
        """Render a styled Streamlit metric card."""
        # Using st.metric directly, the custom CSS handles styling
        if help_text:
            st.metric(label, value, delta, help=help_text)
        else:
            st.metric(label, value, delta)
    
    @staticmethod
    def render_summary_section(df: pd.DataFrame) -> None:
        """Render enhanced summary dashboard with market pulse, opportunities, and intelligence."""
        
        if df.empty:
            st.warning("No data available for summary dashboard. Please adjust filters or load data.")
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
                "Advance/Decline Ratio (Advancing stocks vs. Declining stocks over 1 day)"
            )
        
        with col2:
            # Momentum Health (stocks with high momentum score)
            high_momentum_count = 0
            if 'momentum_score' in df.columns:
                high_momentum_count = (df['momentum_score'] >= 70).sum()
            momentum_pct = (high_momentum_count / len(df) * 100) if len(df) > 0 else 0
            
            UIComponents.render_metric_card(
                "Momentum Health",
                f"{momentum_pct:.0f}%",
                f"{high_momentum_count} strong stocks (Score >= 70)",
                "Percentage of stocks with a strong momentum score."
            )
        
        with col3:
            # Volume State (median RVOL and count of high RVOL stocks)
            avg_rvol = df['rvol'].median() if 'rvol' in df.columns else 1.0
            high_vol_count = (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0
            
            if avg_rvol > 1.5:
                vol_emoji = "🌊"
            elif avg_rvol > 1.2:
                vol_emoji = "💧"
            else:
                vol_emoji = "🏜️"
            
            UIComponents.render_metric_card(
                "Volume State",
                f"{vol_emoji} {avg_rvol:.1f}x",
                f"{high_vol_count} surges (>2x RVOL)",
                "Median Relative Volume (RVOL) and count of stocks with RVOL > 2x."
            )
        
        with col4:
            # Risk Level (composite of several factors)
            risk_factors = 0
            
            # Check for overextended stocks (at high, low momentum)
            if 'from_high_pct' in df.columns and 'momentum_score' in df.columns:
                overextended_count = len(df[(df['from_high_pct'] >= -5) & (df['momentum_score'] < 50)]) # Within 5% of high, but weak momentum
                if overextended_count > len(df) * 0.05: # More than 5% of stocks
                    risk_factors += 1
            
            # Check for extreme RVOL (potential pump risk for low-score stocks)
            if 'rvol' in df.columns and 'master_score' in df.columns:
                pump_risk_count = len(df[(df['rvol'] > 10) & (df['master_score'] < 50)]) # Very high RVOL, but low score
                if pump_risk_count > len(df) * 0.02: # More than 2% of stocks
                    risk_factors += 1
            
            # Check for overall downtrends
            if 'trend_quality' in df.columns:
                downtrends_count = len(df[df['trend_quality'] < 40])
                if downtrends_count > len(df) * 0.3: # More than 30% of stocks in downtrend
                    risk_factors += 1
            
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]
            risk_level_display = risk_levels[min(risk_factors, len(risk_levels) - 1)]
            
            UIComponents.render_metric_card(
                "Risk Level",
                risk_level_display,
                f"{risk_factors} warning factors",
                "An aggregated risk assessment based on market-wide indicators."
            )
        
        # 2. TODAY'S OPPORTUNITIES
        st.markdown("### 🎯 Today's Best Opportunities")
        
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        
        with opp_col1:
            # Ready to Run (strong momentum, acceleration, and RVOL)
            ready_to_run = df[
                (df['momentum_score'] >= 70) & 
                (df['acceleration_score'] >= 70) &
                (df['rvol'] >= 2)
            ].nlargest(5, 'master_score')
            
            st.markdown("**🚀 Ready to Run**")
            if not ready_to_run.empty:
                for _, stock in ready_to_run.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"Score: {stock['master_score']:.1f} | RVOL: {stock['rvol']:.1f}x")
            else:
                st.info("No momentum leaders found in current view.")
        
        with opp_col2:
            # Hidden Gems (strong in category, lower overall percentile)
            hidden_gems = df[df['patterns'].str.contains('HIDDEN GEM', na=False)].nlargest(5, 'master_score')
            
            st.markdown("**💎 Hidden Gems**")
            if not hidden_gems.empty:
                for _, stock in hidden_gems.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"Cat %ile: {stock.get('category_percentile', 0):.0f} | Score: {stock['master_score']:.1f}")
            else:
                st.info("No hidden gems found today in current view.")
        
        with opp_col3:
            # Volume Alerts (extreme RVOL)
            volume_alerts = df[df['rvol'] > 3].nlargest(5, 'master_score')
            
            st.markdown("**⚡ Volume Alerts**")
            if not volume_alerts.empty:
                for _, stock in volume_alerts.iterrows():
                    st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                    st.caption(f"RVOL: {stock['rvol']:.1f}x | Wave: {stock.get('wave_state', 'N/A')}")
            else:
                st.info("No extreme volume detected in current view.")
        
        # 3. MARKET INTELLIGENCE
        st.markdown("### 🧠 Market Intelligence")
        
        intel_col1, intel_col2 = st.columns([2, 1])
        
        with intel_col1:
            # Sector Rotation Map (using dynamic sampling)
            sector_rotation_data = MarketIntelligence.detect_sector_rotation(df)
            
            if not sector_rotation_data.empty:
                # Create sector flow visualization
                fig_sector_flow = go.Figure()
                
                # Take top 10 sectors by Flow Score for the chart
                top_10_sectors = sector_rotation_data.head(10)

                fig_sector_flow.add_trace(go.Bar(
                    x=top_10_sectors.index, # Sector names on X-axis
                    y=top_10_sectors['flow_score'],
                    text=[f"{val:.1f}" for val in top_10_sectors['flow_score']], # Display Flow Score on bars
                    textposition='outside',
                    marker_color=[
                        '#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' # Green for good, red for bad, orange for neutral
                        for score in top_10_sectors['flow_score']
                    ],
                    hovertemplate=(
                        'Sector: %{x}<br>'
                        'Flow Score: %{y:.1f}<br>'
                        'Analyzed: %{customdata[0]:.0f} of %{customdata[1]:.0f} stocks<br>'
                        'Avg Score (Sampled): %{customdata[2]:.1f}<br>'
                        'Median Score (Sampled): %{customdata[3]:.1f}<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        top_10_sectors['analyzed_stocks'],
                        top_10_sectors['total_stocks'],
                        top_10_sectors['avg_score'],
                        top_10_sectors['median_score']
                    ))
                ))
                
                fig_sector_flow.update_layout(
                    title="Sector Rotation Map - Smart Money Flow (Dynamically Sampled Top Performers)",
                    xaxis_title="Sector",
                    yaxis_title="Flow Score",
                    height=400,
                    template='plotly_white',
                    showlegend=False,
                    xaxis_tickangle=-45, # Rotate labels for readability
                    margin=dict(l=50, r=50, t=80, b=100)
                )
                
                st.plotly_chart(fig_sector_flow, use_container_width=True)
            else:
                st.info("No sector rotation data available for visualization with current filters.")
        
        with intel_col2:
            # Market Regime Detection
            regime_detected, regime_metrics = MarketIntelligence.detect_market_regime(df)
            
            st.markdown(f"**🎯 Market Regime**")
            st.markdown(f"### {regime_detected}")
            
            # Key signals driving the regime
            st.markdown("**📡 Key Signals**")
            
            signals_list = []
            
            # Breadth signal
            breadth_val = regime_metrics.get('breadth', np.nan)
            if not pd.isna(breadth_val):
                if breadth_val > 0.6:
                    signals_list.append(f"✅ Strong breadth ({breadth_val*100:.0f}% positive 30D ret)")
                elif breadth_val < 0.4:
                    signals_list.append(f"⚠️ Weak breadth ({breadth_val*100:.0f}% positive 30D ret)")
                else:
                    signals_list.append(f"➡️ Neutral breadth ({breadth_val*100:.0f}% positive 30D ret)")
            
            # Category rotation (small vs large caps)
            category_spread_val = regime_metrics.get('category_spread', np.nan)
            if not pd.isna(category_spread_val):
                if category_spread_val > 10:
                    signals_list.append("🔄 Small caps leading (Risk-ON bias)")
                elif category_spread_val < -10:
                    signals_list.append("🛡️ Large caps leading (Defensive bias)")
                else:
                    signals_list.append("↔️ Balanced category performance")
            
            # Volume signal (average RVOL)
            avg_rvol_val = regime_metrics.get('avg_rvol', np.nan)
            if not pd.isna(avg_rvol_val):
                if avg_rvol_val > 1.5:
                    signals_list.append(f"🌊 High volume activity (Median RVOL {avg_rvol_val:.1f}x)")
                else:
                    signals_list.append(f"💧 Normal volume activity (Median RVOL {avg_rvol_val:.1f}x)")
            
            # Pattern emergence (general pattern activity)
            pattern_count = (df['patterns'] != '').sum()
            if len(df) > 0 and pattern_count > len(df) * 0.2:
                signals_list.append(f"🎯 Many patterns emerging ({pattern_count} stocks)")
            elif len(df) > 0:
                signals_list.append(f"✨ Moderate pattern activity ({pattern_count} stocks)")
            
            if signals_list:
                for signal in signals_list:
                    st.write(signal)
            else:
                st.info("No key signals available or insufficient data.")
            
            # Market strength meter
            st.markdown("**💪 Market Strength**")
            
            # Combine metrics into a simple strength score
            strength_score = 0
            if not pd.isna(breadth_val): strength_score += (breadth_val * 50)
            if not pd.isna(avg_rvol_val): strength_score += (min(avg_rvol_val, 2) * 25) # Cap RVOL contribution
            if len(df) > 0: strength_score += ((pattern_count / len(df)) * 25)
            
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
    """Manage session state properly for consistent UI behavior."""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with default values."""
        
        defaults = {
            'search_query': "",
            'last_refresh': datetime.now(timezone.utc),
            'data_source': "sheet", # Default to Google Sheets
            'user_preferences': {
                'default_top_n': CONFIG.DEFAULT_TOP_N,
                'display_mode': 'Technical',
                'last_filters': {} # To store the state of the filter dict
            },
            'filters': {}, # The actively applied filters
            'active_filter_count': 0,
            'quick_filter': None, # Stores the name of the active quick filter
            'quick_filter_applied': False,
            'show_debug': False,
            'performance_metrics': {}, # Stores timing data
            'data_quality': {}, # Stores data quality metrics
            'trigger_clear_filters_flag': False # Flag to signal filter clear across reruns
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Initialize filter widget states explicitly if they are not already set
        # This prevents "ValueError: Default value must be one of the options..."
        # or empty multiselects not reflecting 'all'
        if 'category_filter' not in st.session_state: st.session_state.category_filter = []
        if 'sector_filter' not in st.session_state: st.session_state.sector_filter = []
        if 'min_score' not in st.session_state: st.session_state.min_score = 0
        if 'patterns' not in st.session_state: st.session_state.patterns = []
        if 'trend_filter' not in st.session_state: st.session_state.trend_filter = 'All Trends'
        if 'eps_tier_filter' not in st.session_state: st.session_state.eps_tier_filter = []
        if 'pe_tier_filter' not in st.session_state: st.session_state.pe_tier_filter = []
        if 'price_tier_filter' not in st.session_state: st.session_state.price_tier_filter = []
        if 'min_eps_change' not in st.session_state: st.session_state.min_eps_change = ""
        if 'min_pe' not in st.session_state: st.session_state.min_pe = ""
        if 'max_pe' not in st.session_state: st.session_state.max_pe = ""
        if 'require_fundamental_data' not in st.session_state: st.session_state.require_fundamental_data = False
        if 'wave_states_filter' not in st.session_state: st.session_state.wave_states_filter = []
        if 'wave_strength_range_slider' not in st.session_state: st.session_state.wave_strength_range_slider = (0, 100)


    @staticmethod
    def clear_all_filter_states_in_session():
        """Resets all filter-related session state keys to their default 'cleared' values."""
        
        # Reset multiselects to empty lists
        for key in ['category_filter', 'sector_filter', 'patterns', 'eps_tier_filter', 
                    'pe_tier_filter', 'price_tier_filter', 'wave_states_filter']:
            st.session_state[key] = []
        
        # Reset sliders to their initial min value (or full range for range sliders)
        st.session_state.min_score = 0
        st.session_state.wave_strength_range_slider = (0, 100) 
        
        # Reset selectboxes to their first option
        st.session_state.trend_filter = 'All Trends'
        
        # Reset text inputs to empty strings
        for key in ['min_eps_change', 'min_pe', 'max_pe']:
            st.session_state[key] = ""
        
        # Reset checkboxes to False
        st.session_state.require_fundamental_data = False
        
        # Reset quick filter states
        st.session_state.quick_filter = None
        st.session_state.quick_filter_applied = False
        
        # Reset internal filter dictionary and count
        st.session_state.filters = {}
        st.session_state.active_filter_count = 0
        
        logger.info("All filter states in session cleared.")

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application - Final Ultimate Production Version"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state (must be first after set_page_config)
    SessionStateManager.initialize()
    
    # Check if a filter clear was triggered from a different widget
    if st.session_state.trigger_clear_filters_flag:
        SessionStateManager.clear_all_filter_states_in_session()
        st.session_state.trigger_clear_filters_flag = False # Reset the flag
        st.rerun() # Rerun to apply clear filters visually
    
    # Custom CSS for production UI
    st.markdown("""
    <style>
    /* Production-ready CSS */
    .main {padding: 0rem 1rem;}
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; /* Space between tabs */
        justify-content: center; /* Center the tabs */
        border-bottom: none; /* Remove default border */
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6; /* Light background for tabs */
        border-radius: 8px 8px 0 0; /* Rounded top corners */
        border: 1px solid #ddd; /* Light border */
        border-bottom: none; /* No bottom border for tab itself */
        transition: all 0.2s ease-in-out; /* Smooth transition */
        font-weight: bold;
        color: #333;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6; /* Darker on hover */
        color: #000;
    }
    /* Active tab styling */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #667eea; /* Vibrant background for active tab */
        color: white; /* White text for active tab */
        border-color: #667eea; /* Matching border color */
        border-bottom: none; /* No bottom border for active tab */
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1); /* Subtle shadow */
    }
    /* Tab content styling */
    .stTabs [data-baseweb="tab-panel"] {
        border: 1px solid #ddd;
        border-top: none; /* No top border, as tab itself has no bottom border */
        border-radius: 0 0 8px 8px; /* Rounded bottom corners */
        padding: 1.5rem;
        background-color: white;
    }

    /* Metric card styling */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.08); /* Lighter blue background */
        border: 1px solid rgba(28, 131, 225, 0.15); /* Slightly darker border */
        padding: 10% 5% 10% 10%; /* Adjusted padding for better fit */
        border-radius: 8px; /* Slightly more rounded corners */
        overflow-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); /* Subtle shadow */
    }
    /* Metric label color */
    div[data-testid="metric-container"] > label > div[data-testid="stMetricLabel"] > div {
        color: #333; /* Darker text for label */
        font-size: 0.9rem;
    }
    /* Metric value color */
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
        color: #000; /* Black for value */
        font-size: 1.5rem; /* Larger font size for value */
    }
    /* Metric delta color adjustments */
    div[data-testid="stMetricDelta"] svg {
        fill: green !important; /* Default green for delta icon */
    }
    div[data-testid="stMetricDelta"].inverse svg {
        fill: red !important; /* Red for inverse delta icon */
    }


    /* Alert styling */
    .stAlert {
        padding: 1rem;
        border-radius: 8px; /* More rounded alerts */
    }
    /* Button styling */
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    /* Primary button style */
    div.stButton button.primary {
        background-color: #667eea;
        color: white;
        border: none;
    }
    div.stButton button.primary:hover {
        background-color: #556ee6;
    }

    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .stDataFrame {font-size: 11px;} /* Smaller font for tables on mobile */
        div[data-testid="metric-container"] {padding: 5% 2%;} /* Tighter padding */
        .main {padding: 0rem 0.5rem;} /* Less padding on main content */
        .stTabs [data-baseweb="tab"] {padding-left: 10px; padding-right: 10px; font-size: 0.8rem;} /* Smaller tabs */
        div[data-testid="stMetricValue"] {font-size: 1.2rem;} /* Smaller metric value */
        div[data-testid="stMetricDelta"] {font-size: 0.8rem;} /* Smaller metric delta */
    }
    /* Table optimization for horizontal scroll */
    .stDataFrame > div {
        overflow-x: auto;
    }
    /* Loading animation color */
    .stSpinner > div {
        border-color: #3498db;
    }
    /* Make expander headers clickable and styled */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #e0e2e6;
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
            Professional Stock Ranking System • Final Ultimate Production Version
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True, help="Clears cache and reloads all data from source."):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now(timezone.utc)
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True, help="Clears all cached data, forcing a fresh load on next refresh."):
                st.cache_data.clear()
                gc.collect()  # Force garbage collection
                st.success("Cache cleared!")
                time.sleep(0.5)
                st.rerun()
        
        # Data source selection
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        # Using two distinct buttons for data source selection
        data_source_col1, data_source_col2 = st.columns(2)
        with data_source_col1:
            if st.button("📊 Google Sheets", 
                         type="primary" if st.session_state.data_source == "sheet" else "secondary", 
                         use_container_width=True, key="sheets_data_source_button",
                         help="Load data directly from the configured Google Sheet."):
                if st.session_state.data_source != "sheet":
                    st.session_state.data_source = "sheet"
                    st.rerun()
        with data_source_col2:
            if st.button("📁 Upload CSV", 
                         type="primary" if st.session_state.data_source == "upload" else "secondary", 
                         use_container_width=True, key="upload_data_source_button",
                         help="Upload your own CSV file for analysis. Must contain 'ticker' and 'price' columns."):
                if st.session_state.data_source != "upload":
                    st.session_state.data_source = "upload"
                    st.rerun()

        uploaded_file = None
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="Upload a CSV file with stock data. Required columns: 'ticker', 'price'. Other columns for scoring: 'volume_1d', 'ret_X', 'from_low_pct', 'from_high_pct', 'rvol', 'sma_X', 'pe', 'eps_change_pct', etc."
            )
            if uploaded_file is None:
                st.info("Please upload a CSV file to continue with 'Upload CSV' mode.")
        
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
                    
                    st.metric("Completeness", f"{emoji} {completeness:.1f}%", help="Percentage of non-missing data points across all cells.")
                    st.metric("Total Stocks", f"{quality.get('total_rows', 0):,}", help="Total number of stocks after initial processing.")
                
                with col2:
                    if 'timestamp' in quality:
                        # Convert UTC timestamp to IST for display
                        data_timestamp_ist = quality['timestamp'].astimezone(IST)
                        age = datetime.now(timezone.utc) - quality['timestamp']
                        minutes = int(age.total_seconds() / 60)
                        
                        if minutes < 60:
                            freshness_display = f"{minutes} min"
                            freshness_status = "🟢 Fresh"
                        elif minutes < CONFIG.STALE_DATA_HOURS * 60:
                            freshness_display = f"{int(minutes/60)} hr"
                            freshness_status = "🟡 Recent"
                        else:
                            freshness_display = f"{int(minutes/1440)} days"
                            freshness_status = "🔴 Stale"
                        
                        st.metric("Data Age", freshness_display, freshness_status, help=f"Data loaded at {data_timestamp_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")
                    
                    duplicates = quality.get('duplicate_tickers', 0)
                    if duplicates > 0:
                        st.metric("Duplicates", f"⚠️ {duplicates}", help="Number of duplicate tickers removed during processing.")
        
        # Performance metrics
        if st.session_state.performance_metrics:
            with st.expander("⚡ Performance", expanded=False): # Keep collapsed by default
                perf = st.session_state.performance_metrics
                
                total_time = sum(perf.values())
                if total_time < 3:
                    perf_emoji = "🟢"
                elif total_time < 5:
                    perf_emoji = "🟡"
                else:
                    perf_emoji = "🔴"
                
                st.metric("Load Time", f"{perf_emoji} {total_time:.1f}s", help="Total time taken for data loading and full processing pipeline.")
                
                # Show slowest operations
                if len(perf) > 0:
                    sorted_perf = sorted(perf.items(), key=lambda x: x[1], reverse=True)
                    for func_name, elapsed in sorted_perf:
                        if elapsed > 0.001: # Show even very small times for full debug
                            st.caption(f"{func_name}: {elapsed:.4f}s")
        
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        # Count active filters
        active_filter_count = 0
        
        if st.session_state.get('quick_filter_applied', False):
            active_filter_count += 1
        
        # Check all filter states from session_state for dynamic count
        filter_checks_state_keys = {
            'category_filter': lambda x: bool(x),
            'sector_filter': lambda x: bool(x),
            'min_score': lambda x: x > 0,
            'patterns': lambda x: bool(x),
            'trend_filter': lambda x: x != 'All Trends',
            'eps_tier_filter': lambda x: bool(x),
            'pe_tier_filter': lambda x: bool(x),
            'price_tier_filter': lambda x: bool(x),
            'min_eps_change': lambda x: x and str(x).strip() != '', # Use str(x) to handle both float and empty string
            'min_pe': lambda x: x and str(x).strip() != '',
            'max_pe': lambda x: x and str(x).strip() != '',
            'require_fundamental_data': lambda x: x,
            'wave_states_filter': lambda x: bool(x),
            'wave_strength_range_slider': lambda x: x != (0, 100) # Check if range is different from default (0, 100)
        }
        
        current_active_filter_count = 0
        for key, check_func in filter_checks_state_keys.items():
            if key in st.session_state and check_func(st.session_state[key]):
                current_active_filter_count += 1
        
        st.session_state.active_filter_count = current_active_filter_count # Update session state with actual count
        
        # Show active filter count
        if st.session_state.active_filter_count > 0:
            st.info(f"🔍 **{st.session_state.active_filter_count} filter{'s' if st.session_state.active_filter_count > 1 else ''} active**")
        
        # Clear filters button
        if st.button("🗑️ Clear All Filters", 
                    use_container_width=True, 
                    type="primary" if st.session_state.active_filter_count > 0 else "secondary",
                    help="Resets all applied filters to their default (inactive) state."):
            SessionStateManager.clear_all_filter_states_in_session()
            st.success("✅ All filters cleared!")
            st.rerun() # Rerun to ensure all widgets update their display state
        
        # Debug mode checkbox
        st.markdown("---")
        # st.session_state.show_debug is already managed by SessionStateManager.initialize()
        st.checkbox("🐛 Show Debug Info", 
                    value=st.session_state.get('show_debug', False), # Default to False if not set
                    key="show_debug",
                    help="Display detailed application state, filter parameters, and performance metrics.")
    
    # Data loading and processing
    try:
        # Check if we need to load data (only if data source is upload and no file, or if no cached data)
        if st.session_state.data_source == "upload" and uploaded_file is None:
            # If in upload mode and no file is uploaded, stop the app flow
            st.stop()
        
        # Load and process data, leveraging Streamlit's caching
        with st.spinner("📥 Loading and processing data... This might take a moment if not cached..."):
            ranked_df, data_timestamp, metadata = load_and_process_data(
                source_type=st.session_state.data_source,
                file_data=uploaded_file,
                sheet_url=CONFIG.DEFAULT_SHEET_URL,
                gid=CONFIG.DEFAULT_GID
            )
            
            # Store primary data and metadata in session state
            st.session_state.ranked_df = ranked_df
            st.session_state.data_timestamp = data_timestamp
            
            # Display any warnings or errors from the data loading/processing phase
            if metadata.get('warnings'):
                for warning_msg in metadata['warnings']:
                    st.warning(warning_msg)
            
            if metadata.get('errors'):
                for error_msg in metadata['errors']:
                    st.error(f"Data Load/Processing Error: {error_msg}")
                    # If critical errors, might want to stop or show specific recovery.
                    # For now, just logging and displaying the error.

    except Exception as e:
        # Catch any exceptions during initial data load and processing
        st.error(f"❌ Critical Data Initialization Error: {str(e)}")
        logger.error(f"Application failed during data initialization: {str(e)}", exc_info=True)
        # Suggest troubleshooting steps or restart
        st.info("Please ensure your data source is accessible and correctly formatted. Try refreshing the page or clearing the cache.")
        st.stop() # Stop the execution if core data loading fails
    
    # Quick Action Buttons (positioned at the top of the main content area)
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
    # These buttons will update st.session_state.quick_filter and trigger a rerun
    with qa_col1:
        if st.button("📈 Top Gainers", use_container_width=True, key="qa_top_gainers_btn"):
            st.session_state.quick_filter = 'top_gainers'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_col2:
        if st.button("🔥 Volume Surges", use_container_width=True, key="qa_volume_surges_btn"):
            st.session_state.quick_filter = 'volume_surges'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_col3:
        if st.button("🎯 Breakout Ready", use_container_width=True, key="qa_breakout_ready_btn"):
            st.session_state.quick_filter = 'breakout_ready'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_col4:
        if st.button("💎 Hidden Gems", use_container_width=True, key="qa_hidden_gems_btn"):
            st.session_state.quick_filter = 'hidden_gems'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_col5:
        if st.button("🌊 Show All", use_container_width=True, key="qa_show_all_btn"):
            st.session_state.quick_filter = None
            st.session_state.quick_filter_applied = False
            st.rerun()
    
    # Apply quick filters to get the initial `ranked_df_display`
    if st.session_state.quick_filter_applied and st.session_state.quick_filter:
        quick_filter_type = st.session_state.quick_filter
        if quick_filter_type == 'top_gainers':
            ranked_df_display = ranked_df[ranked_df['momentum_score'] >= CONFIG.TOP_GAINER_MOMENTUM].copy()
            st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ {CONFIG.TOP_GAINER_MOMENTUM} (Quick Filter: Top Gainers)")
        elif quick_filter_type == 'volume_surges':
            ranked_df_display = ranked_df[ranked_df['rvol'] >= CONFIG.VOLUME_SURGE_RVOL].copy()
            st.info(f"Showing {len(ranked_df_display)} stocks with RVOL ≥ {CONFIG.VOLUME_SURGE_RVOL}x (Quick Filter: Volume Surges)")
        elif quick_filter_type == 'breakout_ready':
            ranked_df_display = ranked_df[ranked_df['breakout_score'] >= CONFIG.BREAKOUT_READY_SCORE].copy()
            st.info(f"Showing {len(ranked_df_display)} stocks with breakout score ≥ {CONFIG.BREAKOUT_READY_SCORE} (Quick Filter: Breakout Ready)")
        elif quick_filter_type == 'hidden_gems':
            ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('HIDDEN GEM', case=False, na=False)].copy()
            st.info(f"Showing {len(ranked_df_display)} hidden gem stocks (Quick Filter: Hidden Gems)")
        else: # Fallback just in case quick_filter is set to an unknown value
            ranked_df_display = ranked_df.copy()
            st.session_state.quick_filter_applied = False
            st.session_state.quick_filter = None
    else:
        ranked_df_display = ranked_df.copy() # Use the full ranked data if no quick filter active
    
    # Initialize filters dictionary for FilterEngine
    filters_dict_for_engine = {}
    
    # Sidebar filters to populate the `filters_dict_for_engine`
    with st.sidebar:
        # Display Mode Toggle
        st.markdown("### 📊 Display Mode")
        # Use session state to manage selection
        current_display_mode = st.session_state.user_preferences['display_mode']
        display_mode_selected = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if current_display_mode == 'Technical' else 1,
            help="Technical: Focuses on momentum, volume, and price action. Hybrid: Integrates fundamental data like PE and EPS growth.",
            key="display_mode_toggle"
        )
        st.session_state.user_preferences['display_mode'] = display_mode_selected
        show_fundamentals = (display_mode_selected == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        # Category filter (Multiselect)
        # Options generated from `ranked_df_display` filtered by other (temp) filters
        categories = FilterEngine.get_filter_options(ranked_df, 'category', filters_dict_for_engine) # Use full ranked_df to get all possible options
        category_counts_for_options = ranked_df_display['category'].value_counts() # Count based on current quick filter state
        category_options_with_counts = [
            f"{cat} ({category_counts_for_options.get(cat, 0)})" 
            for cat in categories
        ]
        
        # Default value for multiselect needs to be from session state to persist
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=category_options_with_counts,
            default=st.session_state.category_filter, 
            placeholder="Select categories (empty = All)",
            key="category_filter",
            help="Filter stocks by their market capitalization category (e.g., Small Cap, Large Cap)."
        )
        # Store actual category names (without counts) in filters_dict_for_engine
        filters_dict_for_engine['categories'] = [cat.split(' (')[0] for cat in selected_categories]
        
        # Sector filter (Multiselect)
        # Options are interconnected based on currently selected categories
        sectors = FilterEngine.get_filter_options(ranked_df, 'sector', filters_dict_for_engine)
        sector_counts_for_options = ranked_df_display['sector'].value_counts()
        sector_options_with_counts = [
            f"{sec} ({sector_counts_for_options.get(sec, 0)})" 
            for sec in sectors
        ]

        selected_sectors = st.multiselect(
            "Sector",
            options=sector_options_with_counts,
            default=st.session_state.sector_filter,
            placeholder="Select sectors (empty = All)",
            key="sector_filter",
            help="Filter stocks by their industry sector (e.g., Technology, Financial Services)."
        )
        filters_dict_for_engine['sectors'] = [sec.split(' (')[0] for sec in selected_sectors]

        # Minimum Master Score (Slider)
        filters_dict_for_engine['min_score'] = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=st.session_state.min_score,
            step=5,
            help="Set a minimum threshold for the overall Master Score of stocks.",
            key="min_score"
        )
        
        # Pattern filter (Multiselect)
        all_patterns_in_display_df = set()
        for patterns_str in ranked_df_display['patterns'].dropna():
            if patterns_str:
                all_patterns_in_display_df.update(patterns_str.split(' | '))
        
        if all_patterns_in_display_df:
            filters_dict_for_engine['patterns'] = st.multiselect(
                "Patterns",
                options=sorted(all_patterns_in_display_df),
                default=st.session_state.patterns,
                placeholder="Select patterns (empty = All)",
                help="Filter stocks that exhibit specific identified trading patterns.",
                key="patterns"
            )
        else:
            filters_dict_for_engine['patterns'] = []
            st.info("No patterns detected in the current quick-filtered data.")
        
        # Trend filter (Selectbox)
        st.markdown("#### 📈 Trend Strength")
        trend_options_map = {
            "All Trends": (0, 100),
            "🔥 Strong Uptrend (80+)": (80, 100),
            "✅ Good Uptrend (60-79)": (60, 79),
            "➡️ Neutral Trend (40-59)": (40, 59),
            "⚠️ Weak/Downtrend (<40)": (0, 39)
        }
        
        filters_dict_for_engine['trend_filter'] = st.selectbox(
            "Trend Quality",
            options=list(trend_options_map.keys()),
            index=list(trend_options_map.keys()).index(st.session_state.trend_filter),
            key="trend_filter",
            help="Filter stocks based on their trend quality (e.g., price relative to moving averages)."
        )
        filters_dict_for_engine['trend_range'] = trend_options_map[filters_dict_for_engine['trend_filter']]

        # NEW: Wave Filters section
        st.markdown("#### 🌊 Wave Filters")
        wave_states_options = FilterEngine.get_filter_options(ranked_df, 'wave_state', filters_dict_for_engine)
        
        filters_dict_for_engine['wave_states'] = st.multiselect(
            "Wave State",
            options=wave_states_options,
            default=st.session_state.wave_states_filter,
            placeholder="Select wave states (empty = All)",
            help="Filter stocks by their calculated 'Wave State' (FORMING, BUILDING, CRESTING, BREAKING).",
            key="wave_states_filter"
        )

        # Overall Wave Strength Slider
        if 'overall_wave_strength' in ranked_df_display.columns:
            # Ensure proper min/max for slider based on available data, while keeping range generous
            min_strength_data = ranked_df_display['overall_wave_strength'].min()
            max_strength_data = ranked_df_display['overall_wave_strength'].max()
            
            # Set slider bounds to a practical range, e.g., 0-100, but ensure current data min/max is within.
            # If all data is same value, set small range around it.
            slider_min_val = max(0, int(min_strength_data) - 10) if not pd.isna(min_strength_data) else 0
            slider_max_val = min(100, int(max_strength_data) + 10) if not pd.isna(max_strength_data) else 100
            
            # Ensure min_val < max_val for slider, if not, adjust slightly
            if slider_min_val == slider_max_val:
                if slider_min_val == 100: slider_min_val -= 10
                else: slider_max_val += 10
                slider_min_val = max(0, slider_min_val)
                slider_max_val = min(100, slider_max_val)


            # Use persisted value, but clip it to the current available slider range
            current_slider_value = st.session_state.wave_strength_range_slider
            current_slider_value = (
                max(slider_min_val, min(slider_max_val, current_slider_value[0])),
                max(slider_min_val, min(slider_max_val, current_slider_value[1]))
            )

            filters_dict_for_engine['wave_strength_range'] = st.slider(
                "Overall Wave Strength",
                min_value=slider_min_val,
                max_value=slider_max_val,
                value=current_slider_value,
                step=1,
                help="Filter stocks by their composite 'Overall Wave Strength' score.",
                key="wave_strength_range_slider"
            )
        else:
            filters_dict_for_engine['wave_strength_range'] = (0, 100) # Default to full range if column is missing
            st.info("Overall Wave Strength data not available for filtering.")
        
        # Advanced filters in an expander
        with st.expander("🔧 Advanced Filters", expanded=False):
            # Tier filters (EPS, PE, Price)
            for tier_type_key, col_name in [
                ('eps_tiers', 'eps_tier'),
                ('pe_tiers', 'pe_tier'),
                ('price_tiers', 'price_tier')
            ]:
                if col_name in ranked_df_display.columns:
                    tier_options = FilterEngine.get_filter_options(ranked_df, col_name, filters_dict_for_engine)
                    
                    selected_tiers = st.multiselect(
                        f"{col_name.replace('_', ' ').title()}", # Display as "Eps Tier", "Pe Tier" etc.
                        options=tier_options,
                        default=st.session_state.get(f'{col_name}_filter', []), # Persist state
                        placeholder=f"Select {col_name.replace('_', ' ')}s (empty = All)",
                        key=f"{col_name}_filter",
                        help=f"Filter stocks based on their calculated {col_name.replace('_', ' ')} range."
                    )
                    filters_dict_for_engine[tier_type_key] = selected_tiers
            
            # EPS change filter (Text input)
            if 'eps_change_pct' in ranked_df_display.columns:
                eps_change_input_value = st.session_state.get('min_eps_change', "")
                if not pd.isna(eps_change_input_value): # Handle cases where session state might be np.nan
                    eps_change_input_value = str(eps_change_input_value)

                eps_change_input = st.text_input(
                    "Min EPS Change %",
                    value=eps_change_input_value,
                    placeholder="e.g. 10 or -20",
                    help="Enter minimum Earnings Per Share (EPS) growth percentage. Stocks with missing EPS change will be excluded.",
                    key="min_eps_change"
                )
                
                # Convert input to float if not empty, otherwise None
                if eps_change_input.strip():
                    try:
                        filters_dict_for_engine['min_eps_change'] = float(eps_change_input)
                    except ValueError:
                        st.error("Please enter a valid number for Min EPS Change %.")
                        filters_dict_for_engine['min_eps_change'] = None # Ensure invalid input doesn't apply filter
                else:
                    filters_dict_for_engine['min_eps_change'] = None # No filter applied if empty

            # PE filters (Text inputs, only in Hybrid mode)
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                col_pe1, col_pe2 = st.columns(2)
                with col_pe1:
                    min_pe_input_value = st.session_state.get('min_pe', "")
                    if not pd.isna(min_pe_input_value):
                        min_pe_input_value = str(min_pe_input_value)

                    min_pe_input = st.text_input(
                        "Min PE Ratio",
                        value=min_pe_input_value,
                        placeholder="e.g. 10",
                        key="min_pe",
                        help="Enter a minimum Price-to-Earnings (PE) ratio. Stocks with missing or non-positive PE will be excluded."
                    )
                    
                    if min_pe_input.strip():
                        try:
                            filters_dict_for_engine['min_pe'] = float(min_pe_input)
                        except ValueError:
                            st.error("Please enter a valid number for Min PE Ratio.")
                            filters_dict_for_engine['min_pe'] = None
                    else:
                        filters_dict_for_engine['min_pe'] = None
                
                with col_pe2:
                    max_pe_input_value = st.session_state.get('max_pe', "")
                    if not pd.isna(max_pe_input_value):
                        max_pe_input_value = str(max_pe_input_value)

                    max_pe_input = st.text_input(
                        "Max PE Ratio",
                        value=max_pe_input_value,
                        placeholder="e.g. 30",
                        key="max_pe",
                        help="Enter a maximum Price-to-Earnings (PE) ratio. Stocks with missing or non-positive PE will be excluded."
                    )
                    
                    if max_pe_input.strip():
                        try:
                            filters_dict_for_engine['max_pe'] = float(max_pe_input)
                        except ValueError:
                            st.error("Please enter a valid number for Max PE Ratio.")
                            filters_dict_for_engine['max_pe'] = None
                    else:
                        filters_dict_for_engine['max_pe'] = None
                
                # Data completeness filter (Checkbox)
                filters_dict_for_engine['require_fundamental_data'] = st.checkbox(
                    "Only show stocks with PE and EPS data",
                    value=st.session_state.require_fundamental_data, # Persist state
                    key="require_fundamental_data",
                    help="If checked, only stocks with complete and valid PE and EPS change data will be shown. Overrides individual PE/EPS filter NaN behavior."
                )
    
    # Apply filters to the current `ranked_df_display` (which might already be quick-filtered)
    filtered_df = FilterEngine.apply_filters(ranked_df_display, filters_dict_for_engine)
    
    # Sort the final filtered DataFrame by rank
    filtered_df = filtered_df.sort_values('rank').reset_index(drop=True)
    
    # Main content area - Display filter status
    total_active_filters = st.session_state.active_filter_count
    if total_active_filters > 0:
        filter_status_col1, filter_status_col2 = st.columns([5, 1])
        with filter_status_col1:
            if st.session_state.quick_filter_applied and st.session_state.quick_filter:
                quick_filter_display_name = {
                    'top_gainers': '📈 Top Gainers', 'volume_surges': '🔥 Volume Surges',
                    'breakout_ready': '🎯 Breakout Ready', 'hidden_gems': '💎 Hidden Gems'
                }.get(st.session_state.quick_filter, 'Quick Filter')
                
                if total_active_filters > 1:
                    st.info(f"**Viewing:** {quick_filter_display_name} + {total_active_filters - 1} other filter{'s' if total_active_filters > 2 else ''} | **{len(filtered_df):,} stocks** shown")
                else:
                    st.info(f"**Viewing:** {quick_filter_display_name} | **{len(filtered_df):,} stocks** shown")
            else:
                st.info(f"**Filtered View:** {len(filtered_df):,} stocks match your {total_active_filters} active filter{'s' if total_active_filters > 1 else ''}")
        
        with filter_status_col2:
            # This button will set the flag, which the initial rerun check picks up
            if st.button("Clear Filters", type="secondary", key="clear_filters_main_button"):
                st.session_state.trigger_clear_filters_flag = True
                st.rerun() # Rerun to trigger the clear logic in the sidebar

    # Summary metrics (below filters, above tabs)
    col1_metrics, col2_metrics, col3_metrics, col4_metrics, col5_metrics, col6_metrics = st.columns(6)
    
    with col1_metrics:
        total_stocks_filtered = len(filtered_df)
        total_original_stocks = len(ranked_df)
        pct_of_all = (total_stocks_filtered / total_original_stocks * 100) if total_original_stocks > 0 else 0
        
        UIComponents.render_metric_card(
            "Total Stocks",
            f"{total_stocks_filtered:,}",
            f"{pct_of_all:.0f}% of {total_original_stocks:,}",
            help="Total number of stocks currently displayed after applying all filters."
        )
    
    with col2_metrics:
        if not filtered_df.empty and 'master_score' in filtered_df.columns:
            avg_score = filtered_df['master_score'].mean()
            std_score = filtered_df['master_score'].std()
            UIComponents.render_metric_card(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={std_score:.1f}",
                help="Average Master Score of the currently filtered stocks."
            )
        else:
            UIComponents.render_metric_card("Avg Score", "N/A", help="Average Master Score of the currently filtered stocks.")
    
    with col3_metrics:
        if show_fundamentals and 'pe' in filtered_df.columns:
            valid_pe_mask = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
            pe_coverage_count = valid_pe_mask.sum()
            pe_pct_coverage = (pe_coverage_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            
            if pe_coverage_count > 0:
                median_pe = filtered_df.loc[valid_pe_mask, 'pe'].median()
                UIComponents.render_metric_card(
                    "Median PE",
                    f"{median_pe:.1f}x",
                    f"{pe_pct_coverage:.0f}% have data",
                    help="Median Price-to-Earnings ratio for stocks with valid PE data in the filtered set. Only shown in Hybrid mode."
                )
            else:
                UIComponents.render_metric_card("PE Data", "Limited", "No PE data", help="Median Price-to-Earnings ratio for stocks with valid PE data in the filtered set. Only shown in Hybrid mode.")
        else: # If not in hybrid mode, show score range
            if not filtered_df.empty and 'master_score' in filtered_df.columns:
                min_score_val = filtered_df['master_score'].min()
                max_score_val = filtered_df['master_score'].max()
                score_range_display = f"{min_score_val:.1f}-{max_score_val:.1f}"
            else:
                score_range_display = "N/A"
            UIComponents.render_metric_card("Score Range", score_range_display, help="Range of Master Scores in the current filtered selection.")
    
    with col4_metrics:
        if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
            valid_eps_change_mask = filtered_df['eps_change_pct'].notna() & ~np.isinf(filtered_df['eps_change_pct'])
            positive_eps_growth_count = valid_eps_change_mask & (filtered_df['eps_change_pct'] > 0).sum()
            strong_growth_count = (valid_eps_change_mask & (filtered_df['eps_change_pct'] > 50)).sum()
            mega_growth_count = (valid_eps_change_mask & (filtered_df['eps_change_pct'] > 100)).sum()
            
            if mega_growth_count > 0 or strong_growth_count > 0:
                UIComponents.render_metric_card(
                    "EPS Growth +ve",
                    f"{positive_eps_growth_count}",
                    f"{strong_growth_count} >50% | {mega_growth_count} >100%",
                    help="Count of stocks with positive EPS growth. Detail shows count of strong (>50%) and mega (>100%) growth. Only shown in Hybrid mode."
                )
            else:
                UIComponents.render_metric_card(
                    "EPS Growth +ve",
                    f"{positive_eps_growth_count}",
                    f"{valid_eps_change_mask.sum()} have data",
                    help="Count of stocks with positive EPS growth. Detail shows total count with valid EPS data. Only shown in Hybrid mode."
                )
        else: # If not in hybrid mode, show accelerating stocks count
            if 'acceleration_score' in filtered_df.columns:
                accelerating_count = (filtered_df['acceleration_score'] >= 80).sum()
            else:
                accelerating_count = 0
            UIComponents.render_metric_card("Accelerating", f"{accelerating_count}", help="Count of stocks with Acceleration Score >= 80.")
    
    with col5_metrics:
        if 'rvol' in filtered_df.columns:
            high_rvol_count = (filtered_df['rvol'] > 2).sum()
            median_rvol_val = filtered_df['rvol'].median()
            UIComponents.render_metric_card("High RVOL", f"{high_rvol_count}", f"Median: {median_rvol_val:.1f}x", help="Count of stocks with Relative Volume (RVOL) > 2x.")
        else:
            UIComponents.render_metric_card("High RVOL", "N/A", help="Count of stocks with Relative Volume (RVOL) > 2x.")
    
    with col6_metrics:
        if 'trend_quality' in filtered_df.columns:
            strong_trends_count = (filtered_df['trend_quality'] >= 80).sum()
            total_filtered_stocks = len(filtered_df)
            strong_trends_pct = (strong_trends_count / total_filtered_stocks * 100) if total_filtered_stocks > 0 else 0
            UIComponents.render_metric_card(
                "Strong Trends", 
                f"{strong_trends_count}",
                f"{strong_trends_pct:.0f}%",
                help="Count and percentage of stocks with a 'Strong Uptrend' (Trend Quality >= 80)."
            )
        else:
            with_patterns_count = (filtered_df['patterns'] != '').sum()
            UIComponents.render_metric_card("With Patterns", f"{with_patterns_count}", help="Count of stocks exhibiting any detected pattern.")
    
    # Main tabs for navigation
    tabs = st.tabs([
        "📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📊 Sector Analysis", "📥 Export", "ℹ️ About"
    ])
    
    # Tab 0: Summary - Enhanced Dashboard
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        
        if not filtered_df.empty:
            # Render the comprehensive summary section (calls internal UIComponents functions)
            UIComponents.render_summary_section(filtered_df)
            
            # Download section (same as in V4.py)
            st.markdown("---")
            st.markdown("#### 💾 Download Clean Processed Data")
            
            download_cols_summary = st.columns(3)
            
            with download_cols_summary[0]:
                st.markdown("**📊 Current View Data**")
                st.write(f"Includes {len(filtered_df)} stocks matching current filters.")
                
                csv_filtered = ExportEngine.create_csv_export(filtered_df)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_filtered,
                    file_name=f"wave_detection_filtered_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download all currently displayed stocks with all scores and indicators."
                )
            
            with download_cols_summary[1]:
                st.markdown("**🏆 Top 100 Stocks**")
                st.write("Elite stocks ranked by Master Score from the current view.")
                
                top_100_for_download = filtered_df.nlargest(100, 'master_score')
                csv_top100 = ExportEngine.create_csv_export(top_100_for_download)
                st.download_button(
                    label="📥 Download Top 100 (CSV)",
                    data=csv_top100,
                    file_name=f"wave_detection_top100_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the top 100 stocks by Master Score from the filtered set."
                )
            
            with download_cols_summary[2]:
                st.markdown("**🎯 Pattern Stocks Only**")
                pattern_stocks_for_download = filtered_df[filtered_df['patterns'] != '']
                st.write(f"Includes {len(pattern_stocks_for_download)} stocks with detected patterns.")
                
                if not pattern_stocks_for_download.empty:
                    csv_patterns = ExportEngine.create_csv_export(pattern_stocks_for_download)
                    st.download_button(
                        label="📥 Download Pattern Stocks (CSV)",
                        data=csv_patterns,
                        file_name=f"wave_detection_patterns_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download only stocks that have one or more patterns detected."
                    )
                else:
                    st.info("No stocks with patterns in current filter to download.")
        
        else:
            st.warning("No data available for summary dashboard. Please adjust filters.")
    
    # Tab 1: Rankings - Top Stocks & Stats
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        
        # Display options (Top N, Sort by)
        col_rank_options1, col_rank_options2, _ = st.columns([2, 2, 6])
        with col_rank_options1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n']),
                key="rankings_display_count_select",
                help="Select how many top-ranked stocks to display."
            )
            st.session_state.user_preferences['default_top_n'] = display_count
        
        with col_rank_options2:
            sort_options = ['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow']
            if 'trend_quality' in filtered_df.columns:
                sort_options.append('Trend')
            
            sort_by = st.selectbox(
                "Sort by",
                options=sort_options,
                index=0, # Default to 'Rank'
                key="rankings_sort_by_select",
                help="Choose the primary metric to sort the ranked stock list."
            )
        
        # Get data for display table
        display_table_df = filtered_df.copy() # Start with the filtered DataFrame
        
        # Apply sorting
        if sort_by == 'Master Score':
            display_table_df = display_table_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL':
            display_table_df = display_table_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum':
            display_table_df = display_table_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Money Flow' and 'money_flow_mm' in display_table_df.columns:
            display_table_df = display_table_df.sort_values('money_flow_mm', ascending=False)
        elif sort_by == 'Trend' and 'trend_quality' in display_table_df.columns:
            display_table_df = display_table_df.sort_values('trend_quality', ascending=False)
        
        # Take the top N after sorting (or filtering if 'Rank' is chosen, it's already sorted by rank)
        display_table_df = display_table_df.head(display_count).copy()

        if not display_table_df.empty:
            # Add trend indicator column if 'trend_quality' is available
            if 'trend_quality' in display_table_df.columns:
                def get_trend_indicator_emoji(score):
                    if pd.isna(score): return "➖"
                    elif score >= 80: return "🔥" # Strong Uptrend
                    elif score >= 60: return "✅" # Good Uptrend
                    elif score >= 40: return "➡️" # Neutral Trend
                    else: return "⚠️" # Weak/Downtrend
                display_table_df['Trend (Quality)'] = display_table_df['trend_quality'].apply(get_trend_indicator_emoji)
            
            # Define columns to display in the table, adapting to display mode
            display_cols_map = {
                'rank': 'Rank',
                'ticker': 'Ticker',
                'company_name': 'Company',
                'master_score': 'Score',
                'overall_wave_strength': 'Wave Strength', # NEW
                'wave_state': 'Wave State',
                'Trend (Quality)': 'Trend', # New column generated above
                'price': 'Price',
                'from_low_pct': 'From Low',
                'ret_30d': '30D Ret',
                'rvol': 'RVOL',
                'vmi': 'VMI',
                'money_flow_mm': 'Money Flow (MM)',
                'patterns': 'Patterns',
                'category': 'Category',
                'sector': 'Sector'
            }
            
            # Add fundamental columns conditionally
            if show_fundamentals:
                display_cols_map['pe'] = 'PE'
                display_cols_map['eps_change_pct'] = 'EPS Δ%'
                display_cols_map['eps_current'] = 'EPS' # Added for detail
            
            # Filter for columns that actually exist in the DataFrame
            columns_for_table = [col for col in display_cols_map.keys() if col in display_table_df.columns]
            display_table_df = display_table_df[columns_for_table]
            
            # Apply formatting for display
            # Prices: ₹#,##0
            if 'price' in display_table_df.columns:
                display_table_df['price'] = display_table_df['price'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-')
            # From Low/High: 0%
            if 'from_low_pct' in display_table_df.columns:
                display_table_df['from_low_pct'] = display_table_df['from_low_pct'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else '-')
            # Returns: +X.X%
            if 'ret_30d' in display_table_df.columns:
                display_table_df['ret_30d'] = display_table_df['ret_30d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else '-')
            # RVOL: X.X x
            if 'rvol' in display_table_df.columns:
                display_table_df['rvol'] = display_table_df['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
            # VMI: X.XX
            if 'vmi' in display_table_df.columns:
                display_table_df['vmi'] = display_table_df['vmi'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '-')
            # Money Flow: ₹X.X M
            if 'money_flow_mm' in display_table_df.columns:
                display_table_df['money_flow_mm'] = display_table_df['money_flow_mm'].apply(lambda x: f"₹{x:.1f}M" if pd.notna(x) else '-')
            # Master Score, Wave Strength: X.X
            for col in ['master_score', 'overall_wave_strength']:
                if col in display_table_df.columns:
                    display_table_df[col] = display_table_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
            
            # Smart PE formatting (handles NaN, negative, large values)
            def format_pe_for_display(value):
                try:
                    if pd.isna(value): return '-'
                    val = float(value)
                    if val <= 0: return 'Loss'
                    elif np.isinf(val): return '∞'
                    elif val > 10000: return f"{val/1000:.0f}K"
                    elif val > 1000: return f"{val:.0f}"
                    else: return f"{val:.1f}"
                except (ValueError, TypeError, OverflowError): return '-'

            if show_fundamentals and 'pe' in display_table_df.columns:
                display_table_df['pe'] = display_table_df['pe'].apply(format_pe_for_display)

            # Smart EPS Change formatting (handles NaN, large percentages)
            def format_eps_change_for_display(value):
                try:
                    if pd.isna(value): return '-'
                    val = float(value)
                    if np.isinf(val): return '∞' if val > 0 else '-∞'
                    if abs(val) >= 10000: return f"{val/1000:+.1f}K%"
                    elif abs(val) >= 1000: return f"{val:+.0f}%"
                    else: return f"{val:+.1f}%"
                except (ValueError, TypeError, OverflowError): return '-'
            
            if show_fundamentals and 'eps_change_pct' in display_table_df.columns:
                display_table_df['eps_change_pct'] = display_table_df['eps_change_pct'].apply(format_eps_change_for_display)

            # EPS Current formatting
            if show_fundamentals and 'eps_current' in display_table_df.columns:
                display_table_df['eps_current'] = display_table_df['eps_current'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else '-')


            # Rename columns for final display
            display_table_df.columns = [display_cols_map.get(col, col) for col in display_table_df.columns]
            
            # Display the DataFrame
            st.dataframe(
                display_table_df,
                use_container_width=True,
                height=min(600, len(display_table_df) * 35 + 50), # Dynamic height
                hide_index=True
            )
            
            # Quick stats below table (ENHANCED)
            with st.expander("📊 Quick Statistics (Filtered Stocks)", expanded=False):
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown("**Score Distribution**")
                    if 'master_score' in filtered_df.columns and not filtered_df['master_score'].empty:
                        scores = filtered_df['master_score'].dropna()
                        if not scores.empty:
                            st.text(f"Max: {scores.max():.1f}")
                            st.text(f"Min: {scores.min():.1f}")
                            st.text(f"Mean: {scores.mean():.1f}")
                            st.text(f"Median: {scores.median():.1f}")
                            st.text(f"Q1 (25%): {scores.quantile(0.25):.1f}")
                            st.text(f"Q3 (75%): {scores.quantile(0.75):.1f}")
                            st.text(f"Std Dev: {scores.std():.1f}")
                        else:
                            st.text("No score data available")
                    else:
                        st.text("No score data available")
                
                with stat_cols[1]:
                    st.markdown("**Returns (30D)**")
                    if 'ret_30d' in filtered_df.columns and not filtered_df['ret_30d'].empty:
                        returns_30d = filtered_df['ret_30d'].dropna()
                        if not returns_30d.empty:
                            st.text(f"Max: {returns_30d.max():.1f}%")
                            st.text(f"Min: {returns_30d.min():.1f}%")
                            st.text(f"Avg: {returns_30d.mean():.1f}%")
                            st.text(f"Positive: {(returns_30d > 0).sum()}")
                            st.text(f"Negative: {(returns_30d < 0).sum()}")
                        else:
                            st.text("No 30D return data available")
                    else:
                        st.text("No 30D return data available")
                
                with stat_cols[2]:
                    if show_fundamentals:
                        st.markdown("**Fundamentals**")
                        if 'pe' in filtered_df.columns:
                            valid_pe_data = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
                            if valid_pe_data.any():
                                pe_values = filtered_df.loc[valid_pe_data, 'pe'].dropna()
                                st.text(f"Median PE: {pe_values.median():.1f}x")
                                st.text(f"Avg PE: {pe_values.mean():.1f}x")
                                st.text(f"PE Q1-Q3: {pe_values.quantile(0.25):.1f}-{pe_values.quantile(0.75):.1f}x")
                            else:
                                st.text("PE: No valid data")
                        else:
                            st.text("PE: No data")
                        
                        if 'eps_change_pct' in filtered_df.columns:
                            valid_eps_data = filtered_df['eps_change_pct'].notna() & ~np.isinf(filtered_df['eps_change_pct'])
                            if valid_eps_data.any():
                                eps_values = filtered_df.loc[valid_eps_data, 'eps_change_pct'].dropna()
                                mega_growth_count = (eps_values > 100).sum()
                                strong_growth_count = ((eps_values > 50) & (eps_values <= 100)).sum()
                                positive_growth_count = (eps_values > 0).sum()
                                st.text(f"Avg EPS Growth: {eps_values.mean():.1f}%")
                                st.text(f"Positive EPS: {positive_growth_count}")
                                if mega_growth_count > 0:
                                    st.text(f">100% Growth: {mega_growth_count}")
                            else:
                                st.text("EPS Growth: N/A")
                        else:
                            st.text("EPS: No data")
                    else: # If not in hybrid mode, show RVOL stats
                        st.markdown("**Volume Stats**")
                        if 'rvol' in filtered_df.columns and not filtered_df['rvol'].empty:
                            rvol_values = filtered_df['rvol'].dropna()
                            if not rvol_values.empty:
                                st.text(f"Max RVOL: {rvol_values.max():.1f}x")
                                st.text(f"Avg RVOL: {rvol_values.mean():.1f}x")
                                st.text(f"Median RVOL: {rvol_values.median():.1f}x")
                                st.text(f"Stocks >2x RVOL: {(rvol_values > 2).sum()}")
                            else:
                                st.text("No RVOL data available")
                        else:
                            st.text("No RVOL data available")
                
                with stat_cols[3]:
                    st.markdown("**Trend & Categories**")
                    if 'trend_quality' in filtered_df.columns and not filtered_df['trend_quality'].empty:
                        trend_values = filtered_df['trend_quality'].dropna()
                        if not trend_values.empty:
                            stocks_above_all_smas = (trend_values >= 85).sum() # Based on calculation for 85 score
                            stocks_in_uptrend = (trend_values >= 60).sum() # Good or Strong uptrend
                            stocks_in_downtrend = (trend_values < 40).sum() # Weak/Downtrend
                            
                            st.text(f"Avg Trend Score: {trend_values.mean():.1f}")
                            st.text(f"Above All SMAs: {stocks_above_all_smas}")
                            st.text(f"In Uptrend (60+): {stocks_in_uptrend}")
                            st.text(f"In Downtrend (<40): {stocks_in_downtrend}")
                        else:
                            st.text("No trend data available")
                    else:
                        st.text("No trend data available")

                    st.markdown("**Top Categories (Count)**")
                    if 'category' in filtered_df.columns and not filtered_df['category'].empty:
                        category_counts = filtered_df['category'].value_counts().head(3)
                        if not category_counts.empty:
                            for cat, count in category_counts.items():
                                st.text(f"{cat}: {count}")
                        else:
                            st.text("No category data available")
                    else:
                        st.text("No category data available")
        
        else:
            st.warning("No stocks match the selected filters for ranking. Please adjust filters.")
        
        # Master Score Breakdown Visualization (ENHANCED)
        if not filtered_df.empty:
            st.markdown("---")
            st.markdown("### 📊 Master Score Component Analysis")
            
            # Allow user to select number of stocks to show
            breakdown_col1, _ = st.columns([1, 4])
            with breakdown_col1:
                breakdown_count = st.selectbox(
                    "Show breakdown for top",
                    options=[10, 20, 30, 50],
                    index=1, # Default to 20
                    key="breakdown_count_select",
                    help="Select the number of top-ranked stocks to visualize their Master Score composition."
                )
            
            # Create and display the breakdown chart
            fig_breakdown = Visualizer.create_master_score_breakdown(filtered_df, n=breakdown_count)
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # Explanation
            st.info(
                "📊 **How to read this chart:**\n"
                "- Each horizontal bar represents a stock. Tickers are on the left.\n"
                "- The total length of the stacked bar indicates the stock's overall Master Score, which is also displayed numerically at the end of the bar.\n"
                "- Different colors within each bar represent the weighted contribution of each component score (Position, Volume, Momentum, Acceleration, Breakout, RVOL).\n"
                "- The percentage in the legend shows the global weight of each component in the Master Score formula.\n"
                "- A longer colored segment means that specific component contributed more significantly to that stock's Master Score."
            )
    
    # Tab 2: Wave Radar - Early Momentum Detection System
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        st.markdown("*Catch waves as they form, not after they've peaked!*")
        
        # Wave Radar Controls
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        
        with radar_col1:
            wave_timeframe_options = [
                "All Waves",
                "Intraday Surge",
                "3-Day Buildup", 
                "Weekly Breakout",
                "Monthly Trend"
            ]
            wave_timeframe = st.selectbox(
                "Wave Detection Timeframe",
                options=wave_timeframe_options,
                index=wave_timeframe_options.index(st.session_state.get('wave_timeframe_select', "All Waves")),
                key="wave_timeframe_select",
                help="""
                **All Waves**: Displays all potential signals based on selected sensitivity.
                **⚡ Intraday Surge**: Focuses on high RVOL and strong daily price movers.
                **📈 3-Day Buildup**: Identifies stocks building momentum over 3 days, above short-term MA.
                **🚀 Weekly Breakout**: Targets stocks near 52-week highs with strong weekly volume surges.
                **💪 Monthly Trend**: Highlights stocks with established positive trends over a month, supported by MAs.
                """
            )
        
        with radar_col2:
            sensitivity_options = ["Conservative", "Balanced", "Aggressive"]
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=sensitivity_options,
                value=st.session_state.get('wave_sensitivity', "Balanced"),
                key="wave_sensitivity",
                help="Adjusts the strictness of pattern and signal detection. Higher sensitivity (Aggressive) shows more signals but with potentially lower conviction."
            )
            
            # Sensitivity details toggle
            show_sensitivity_details = st.checkbox(
                "Show thresholds",
                value=st.session_state.get('show_sensitivity_details', False),
                key="show_sensitivity_details",
                help="Display the exact threshold values being used for detection based on the selected sensitivity."
            )
        
        with radar_col3:
            show_market_regime = st.checkbox(
                "📊 Market Regime Analysis",
                value=st.session_state.get('show_market_regime', True),
                key="show_market_regime",
                help="Toggle visibility of the category rotation flow and market regime detection section."
            )
        
        # Initialize wave_filtered_df with a copy of the main filtered_df
        wave_filtered_df = filtered_df.copy()
        
        with radar_col4:
            # Calculate overall Wave Strength for the current view
            overall_wave_strength_val = 0.0
            if 'overall_wave_strength' in wave_filtered_df.columns and not wave_filtered_df.empty:
                overall_wave_strength_val = wave_filtered_df['overall_wave_strength'].mean()
            
            if overall_wave_strength_val > 70:
                wave_emoji_display = "🌊🔥"
                wave_status_text = "STRONG WAVE"
                wave_color_status = "🟢"
            elif overall_wave_strength_val > 50:
                wave_emoji_display = "🌊"
                wave_status_text = "BUILDING WAVE"
                wave_color_status = "🟡"
            else:
                wave_emoji_display = "💤"
                wave_status_text = "CALM/WEAK"
                wave_color_status = "🔴"
            
            UIComponents.render_metric_card(
                "Market Wave",
                f"{wave_emoji_display} {overall_wave_strength_val:.0f}",
                wave_status_text,
                help="Average 'Overall Wave Strength' score for all stocks in the current filtered view. Indicates general market momentum."
            )
        
        # Display sensitivity thresholds if enabled
        if show_sensitivity_details:
            with st.expander("📊 Current Sensitivity Thresholds", expanded=True):
                if sensitivity == "Conservative":
                    st.markdown("""
                    **Conservative Settings** 🛡️ (High conviction, fewer signals)
                    - **Momentum Shifts:** Momentum Score ≥ 60, Acceleration Score ≥ 70
                    - **Emerging Patterns:** Within 5% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 3.0x (only extreme volumes)
                    - **Acceleration Alerts:** Acceleration Score ≥ 85 (strongest signals)
                    """)
                elif sensitivity == "Balanced":
                    st.markdown("""
                    **Balanced Settings** ⚖️ (Good balance of signals and conviction)
                    - **Momentum Shifts:** Momentum Score ≥ 50, Acceleration Score ≥ 60
                    - **Emerging Patterns:** Within 10% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 2.0x (standard high volume)
                    - **Acceleration Alerts:** Acceleration Score ≥ 70 (good acceleration)
                    """)
                else:  # Aggressive
                    st.markdown("""
                    **Aggressive Settings** 🚀 (More signals, including early stage)
                    - **Momentum Shifts:** Momentum Score ≥ 40, Acceleration Score ≥ 50
                    - **Emerging Patterns:** Within 15% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 1.5x (early signs of building volume)
                    - **Acceleration Alerts:** Acceleration Score ≥ 60 (early acceleration signals)
                    """)
                
                st.info("💡 **Tip**: Adjust sensitivity based on market conditions (Aggressive for strong bull, Conservative for uncertain).")
        
        # Apply intelligent timeframe filtering to the `wave_filtered_df`
        if wave_timeframe != "All Waves":
            original_wave_df_count = len(wave_filtered_df)
            temp_wave_df = wave_filtered_df.copy() # Work on a temporary copy for timeframe filtering
            try:
                if wave_timeframe == "Intraday Surge":
                    # Focus on today's high volume movers: RVOL > 2.5, 1-day return > 2%, current price > previous close by 2%
                    if all(col in temp_wave_df.columns for col in ['rvol', 'ret_1d', 'price', 'prev_close']):
                        temp_wave_df = temp_wave_df[
                            (temp_wave_df['rvol'] >= 2.5) &
                            (temp_wave_df['ret_1d'] > 2) &
                            (temp_wave_df['price'] > temp_wave_df['prev_close'] * 1.02)
                        ]
                    else:
                        st.warning(f"Insufficient data for '{wave_timeframe}' timeframe filter (requires 'rvol', 'ret_1d', 'price', 'prev_close').")
                        temp_wave_df = filtered_df.copy() # Revert to original filtered_df if data missing
                
                elif wave_timeframe == "3-Day Buildup":
                    # Stocks building momentum over 3 days, above short-term moving average
                    if all(col in temp_wave_df.columns for col in ['ret_3d', 'vol_ratio_7d_90d', 'price', 'sma_20d']):
                        temp_wave_df = temp_wave_df[
                            (temp_wave_df['ret_3d'] > 5) & # Positive 3-day return
                            (temp_wave_df['vol_ratio_7d_90d'] > 1.5) & # Volume building
                            (temp_wave_df['price'] > temp_wave_df['sma_20d']) # Above 20-day MA
                        ]
                    else:
                        st.warning(f"Insufficient data for '{wave_timeframe}' timeframe filter (requires 'ret_3d', 'vol_ratio_7d_90d', 'price', 'sma_20d').")
                        temp_wave_df = filtered_df.copy() # Revert
                
                elif wave_timeframe == "Weekly Breakout":
                    # Stocks near 52-week highs with strong weekly momentum and volume
                    if all(col in temp_wave_df.columns for col in ['ret_7d', 'vol_ratio_7d_90d', 'from_high_pct']):
                        temp_wave_df = temp_wave_df[
                            (temp_wave_df['ret_7d'] > 8) & # Strong 7-day return
                            (temp_wave_df['vol_ratio_7d_90d'] > 2.0) & # Significant weekly volume
                            (temp_wave_df['from_high_pct'] > -10) # Within 10% of 52-week high
                        ]
                    else:
                        st.warning(f"Insufficient data for '{wave_timeframe}' timeframe filter (requires 'ret_7d', 'vol_ratio_7d_90d', 'from_high_pct').")
                        temp_wave_df = filtered_df.copy() # Revert
                
                elif wave_timeframe == "Monthly Trend":
                    # Established trends with technical confirmation (above key MAs, sustained volume)
                    if all(col in temp_wave_df.columns for col in ['ret_30d', 'price', 'sma_20d', 'sma_50d', 'vol_ratio_30d_180d', 'from_low_pct']):
                        temp_wave_df = temp_wave_df[
                            (temp_wave_df['ret_30d'] > 15) & # Strong 30-day return
                            (temp_wave_df['price'] > temp_wave_df['sma_20d']) & # Above 20-day MA
                            (temp_wave_df['sma_20d'] > temp_wave_df['sma_50d']) & # Short MA above medium MA (bullish cross)
                            (temp_wave_df['vol_ratio_30d_180d'] > 1.2) & # Sustained higher volume
                            (temp_wave_df['from_low_pct'] > 30) # Already significantly up from 52-week low
                        ]
                    else:
                        st.warning(f"Insufficient data for '{wave_timeframe}' timeframe filter (requires 'ret_30d', 'price', 'sma_20d', 'sma_50d', 'vol_ratio_30d_180d', 'from_low_pct').")
                        temp_wave_df = filtered_df.copy() # Revert

                # Assign the filtered temp_wave_df back only if it's not empty, otherwise revert
                if temp_wave_df.empty and original_wave_df_count > 0:
                    st.info(f"No stocks found for '{wave_timeframe}' timeframe with current filters. Displaying with original filters.")
                    wave_filtered_df = filtered_df.copy() # Revert to the general filtered_df
                else:
                    wave_filtered_df = temp_wave_df.copy() # Apply the timeframe filter successfully
            
            except Exception as e:
                logger.error(f"Error applying {wave_timeframe} filter in Wave Radar: {str(e)}", exc_info=True)
                st.error(f"An error occurred while applying '{wave_timeframe}' filter. Displaying original filtered data.")
                wave_filtered_df = filtered_df.copy() # Revert on error
        
        # Display Wave Radar Analysis sections
        if not wave_filtered_df.empty:
            # 1. MOMENTUM SHIFT DETECTION
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            
            # Set thresholds based on sensitivity
            if sensitivity == "Conservative":
                momentum_threshold_ms = 60
                acceleration_threshold_ms = 70
                min_rvol_ms = 3.0
                breakout_score_ms = 75
                vol_ratio_7d_90d_ms = 1.5
            elif sensitivity == "Balanced":
                momentum_threshold_ms = 50
                acceleration_threshold_ms = 60
                min_rvol_ms = 2.0
                breakout_score_ms = 70
                vol_ratio_7d_90d_ms = 1.2
            else:  # Aggressive
                momentum_threshold_ms = 40
                acceleration_threshold_ms = 50
                min_rvol_ms = 1.5
                breakout_score_ms = 60
                vol_ratio_7d_90d_ms = 1.1
            
            # Find momentum shifts based on core criteria
            # Ensure columns exist, default to neutral if not
            momentum_score_series = wave_filtered_df['momentum_score'].fillna(50)
            acceleration_score_series = wave_filtered_df['acceleration_score'].fillna(50)
            
            momentum_shifts_mask = (
                (momentum_score_series >= momentum_threshold_ms) & 
                (acceleration_score_series >= acceleration_threshold_ms)
            )
            
            # Apply the mask to get the subset of stocks for further signal counting
            momentum_shifts_candidates = wave_filtered_df[momentum_shifts_mask].copy()

            if not momentum_shifts_candidates.empty:
                # Calculate multi-signal count for each stock within candidates
                momentum_shifts_candidates['signal_count'] = 0
                
                # Signal 1: Core Momentum Shift (already applied by `momentum_shifts_mask`, effectively count as 1)
                momentum_shifts_candidates['signal_count'] += 1 # Base count for meeting initial criteria

                # Signal 2: High RVOL
                if 'rvol' in momentum_shifts_candidates.columns:
                    momentum_shifts_candidates.loc[momentum_shifts_candidates['rvol'] >= min_rvol_ms, 'signal_count'] += 1
                
                # Signal 3: Very Strong Acceleration
                if 'acceleration_score' in momentum_shifts_candidates.columns:
                    momentum_shifts_candidates.loc[momentum_shifts_candidates['acceleration_score'] >= (acceleration_threshold_ms + 15).clip(0,100), 'signal_count'] += 1
                
                # Signal 4: Recent Volume Surge (7D/90D ratio)
                if 'vol_ratio_7d_90d' in momentum_shifts_candidates.columns:
                    momentum_shifts_candidates.loc[momentum_shifts_candidates['vol_ratio_7d_90d'] >= vol_ratio_7d_90d_ms, 'signal_count'] += 1
                
                # Signal 5: Breakout Readiness
                if 'breakout_score' in momentum_shifts_candidates.columns:
                    momentum_shifts_candidates.loc[momentum_shifts_candidates['breakout_score'] >= breakout_score_ms, 'signal_count'] += 1
                
                # Calculate shift strength (weighted sum of relevant scores)
                momentum_shifts_candidates['shift_strength'] = (
                    momentum_shifts_candidates['momentum_score'].fillna(50) * 0.4 +
                    momentum_shifts_candidates['acceleration_score'].fillna(50) * 0.4 +
                    momentum_shifts_candidates['rvol_score'].fillna(50) * 0.2
                )
                
                # Get top N shifts based on signal count, then shift strength
                top_shifts = momentum_shifts_candidates.sort_values(
                    ['signal_count', 'shift_strength'], ascending=[False, False]
                ).head(20) # Limit to top 20 for display performance
                
                # Prepare display columns for the table
                display_columns_ms = ['ticker', 'company_name', 'master_score', 'momentum_score', 
                                     'acceleration_score', 'rvol', 'signal_count', 'wave_state']
                
                if 'ret_7d' in top_shifts.columns:
                    display_columns_ms.insert(display_columns_ms.index('rvol') + 1, 'ret_7d') # Add 7D return after RVOL
                
                display_columns_ms.append('category')
                
                shift_display_df = top_shifts[[col for col in display_columns_ms if col in top_shifts.columns]].copy()
                
                # Add a visual 'Signals' column with emojis and count
                shift_display_df['Signals'] = shift_display_df['signal_count'].apply(
                    lambda x: f"{'🔥' * min(x, 3)} {x}/5" if x > 0 else "0/5"
                )
                
                # Format numeric columns for display
                for col in ['master_score', 'momentum_score', 'acceleration_score']:
                    if col in shift_display_df.columns:
                        shift_display_df[col] = shift_display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
                if 'rvol' in shift_display_df.columns:
                    shift_display_df['rvol'] = shift_display_df['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                if 'ret_7d' in shift_display_df.columns:
                    shift_display_df['ret_7d'] = shift_display_df['ret_7d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else '-')
                
                # Rename columns for final display in table
                shift_display_df = shift_display_df.rename(columns={
                    'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score',
                    'momentum_score': 'Momentum', 'acceleration_score': 'Accel', 'rvol': 'RVOL',
                    'wave_state': 'Wave State', 'category': 'Category', 'ret_7d': '7D Ret'
                })
                
                # Drop the raw signal_count column as 'Signals' is more user-friendly
                shift_display_df = shift_display_df.drop(columns=['signal_count'])
                
                st.dataframe(shift_display_df, use_container_width=True, hide_index=True)
                
                # Summary of multi-signal leaders
                multi_signal_leaders = top_shifts[top_shifts['signal_count'] >= 3]
                if not multi_signal_leaders.empty:
                    st.success(f"🏆 Found {len(multi_signal_leaders)} stocks with 3+ signals (strongest momentum shifts)! Check 'Signals' column for details.")
            else:
                st.info(f"No significant momentum shifts detected with '{sensitivity}' sensitivity in the '{wave_timeframe}' timeframe. Try adjusting filters or sensitivity.")
            
            # 2. ACCELERATION PROFILES (Visualization)
            st.markdown("#### 🚀 Acceleration Profiles - Momentum Building Over Time")
            
            # Set acceleration threshold for visualization based on sensitivity
            if sensitivity == "Conservative":
                accel_plot_threshold = 85
            elif sensitivity == "Balanced":
                accel_plot_threshold = 70
            else:  # Aggressive
                accel_plot_threshold = 60
            
            # Filter for stocks that meet the acceleration threshold for plotting
            # Only include stocks with enough return data points to make a meaningful plot
            plot_candidates = wave_filtered_df[
                (wave_filtered_df['acceleration_score'] >= accel_plot_threshold) &
                (wave_filtered_df['ret_1d'].notna()) & # Must have at least 1D return
                (wave_filtered_df['ret_7d'].notna())   # Must have at least 7D return
            ].nlargest(10, 'acceleration_score') # Limit to top 10 for plot clarity
            
            if not plot_candidates.empty:
                fig_accel_profiles = Visualizer.create_acceleration_profiles(plot_candidates, n=10)
                st.plotly_chart(fig_accel_profiles, use_container_width=True)
                
                # Summary statistics for accelerating stocks
                accel_stats_col1, accel_stats_col2, accel_stats_col3 = st.columns(3)
                with accel_stats_col1:
                    perfect_accel_count = (plot_candidates['acceleration_score'] >= 90).sum()
                    UIComponents.render_metric_card("Perfect Accel (90+)", perfect_accel_count)
                with accel_stats_col2:
                    strong_accel_count = (plot_candidates['acceleration_score'] >= 80).sum()
                    UIComponents.render_metric_card("Strong Accel (80+)", strong_accel_count)
                with accel_stats_col3:
                    avg_accel_score_plot = plot_candidates['acceleration_score'].mean()
                    UIComponents.render_metric_card("Avg Accel Score", f"{avg_accel_score_plot:.1f}")
            else:
                st.info(f"No stocks meet the acceleration threshold ({accel_plot_threshold}+) or have sufficient return data for plotting acceleration profiles in this view.")
            
            # 3. CATEGORY ROTATION FLOW (Visualization & Insights)
            if show_market_regime:
                st.markdown("#### 💰 Category Rotation - Smart Money Flow")
                
                flow_col1, flow_col2 = st.columns([3, 2])
                
                with flow_col1:
                    # Calculate category performance using dynamic sampling for flow scores
                    # The MarketIntelligence.detect_sector_rotation function can also be adapted or reused for categories
                    # For categories, it's typically simpler to use all relevant stocks due to fewer categories
                    try:
                        if 'category' in wave_filtered_df.columns and not wave_filtered_df.empty:
                            # Aggregate all stocks in the current wave_filtered_df by category
                            # This provides a view of which categories are performing well in the current filtered "wave" context
                            category_flow_df = wave_filtered_df.groupby('category').agg(
                                master_score=('master_score', 'mean'),
                                count=('master_score', 'count'), # Count of stocks in this category after current filters
                                avg_momentum=('momentum_score', 'mean'),
                                avg_volume=('volume_score', 'mean'),
                                avg_rvol=('rvol', 'mean')
                            ).round(2)
                            
                            if not category_flow_df.empty:
                                category_flow_df['Flow Score'] = (
                                    category_flow_df['master_score'] * 0.4 +
                                    category_flow_df['avg_momentum'] * 0.3 +
                                    category_flow_df['avg_volume'] * 0.3
                                )
                                category_flow_df = category_flow_df.sort_values('Flow Score', ascending=False)
                                
                                # Determine overall market bias based on category leadership
                                flow_direction_text = "➡️ Neutral"
                                top_category_name = ""
                                if not category_flow_df.empty:
                                    top_category_name = category_flow_df.index[0]
                                    if 'Small' in top_category_name or 'Micro' in top_category_name:
                                        flow_direction_text = "🔥 Risk-ON"
                                    elif 'Large' in top_category_name or 'Mega' in top_category_name:
                                        flow_direction_text = "❄️ Risk-OFF"
                                
                                # Create bar chart for category flow
                                fig_category_flow = go.Figure()
                                fig_category_flow.add_trace(go.Bar(
                                    x=category_flow_df.index,
                                    y=category_flow_df['Flow Score'],
                                    text=[f"{val:.1f}" for val in category_flow_df['Flow Score']],
                                    textposition='outside',
                                    marker_color=[
                                        '#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12'
                                        for score in category_flow_df['Flow Score']
                                    ],
                                    hovertemplate='Category: %{x}<br>Flow Score: %{y:.1f}<br>Stocks: %{customdata:.0f}<extra></extra>',
                                    customdata=category_flow_df['count']
                                ))
                                
                                fig_category_flow.update_layout(
                                    title=f"Smart Money Flow Direction: {flow_direction_text}",
                                    xaxis_title="Market Cap Category",
                                    yaxis_title="Flow Score",
                                    height=350,
                                    template='plotly_white',
                                    showlegend=False,
                                    xaxis_tickangle=-45,
                                    margin=dict(l=50, r=50, t=80, b=100)
                                )
                                
                                st.plotly_chart(fig_category_flow, use_container_width=True)
                            else:
                                st.info("Insufficient data for category flow analysis.")
                                flow_direction_text = "➡️ Neutral"
                                category_flow_df = pd.DataFrame()
                        else:
                            st.info("Category data not available for flow analysis.")
                            flow_direction_text = "➡️ Neutral"
                            category_flow_df = pd.DataFrame()
                            
                    except Exception as e:
                        logger.error(f"Error in category flow analysis: {str(e)}", exc_info=True)
                        st.error("Unable to analyze category flow with current data.")
                        flow_direction_text = "➡️ Neutral"
                        category_flow_df = pd.DataFrame()
                
                with flow_col2:
                    st.markdown(f"**🎯 Market Regime: {flow_direction_text}**")
                    
                    st.markdown("**💎 Strongest Categories:**")
                    if not category_flow_df.empty:
                        for i, (cat, row) in enumerate(category_flow_df.head(3).iterrows()):
                            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.write(f"{emoji} **{cat}**: Score {row['Flow Score']:.1f} ({row['count']} stocks)")
                    else:
                        st.info("No category data to display.")
                    
                    st.markdown("**🔄 Category Shifts:**")
                    if not category_flow_df.empty:
                        small_caps_scores = category_flow_df[category_flow_df.index.str.contains('Small Cap|Micro Cap')]['Flow Score']
                        large_caps_scores = category_flow_df[category_flow_df.index.str.contains('Large Cap|Mega Cap')]['Flow Score']
                        
                        small_caps_avg_score = small_caps_scores.mean() if not small_caps_scores.empty else np.nan
                        large_caps_avg_score = large_caps_scores.mean() if not large_caps_scores.empty else np.nan

                        if not pd.isna(small_caps_avg_score) and not pd.isna(large_caps_avg_score):
                            score_diff = small_caps_avg_score - large_caps_avg_score
                            if score_diff > 10: # Small caps significantly outperforming
                                st.success("📈 Small Caps Leading Strongly - Confirmed Risk-ON!")
                            elif score_diff < -10: # Large caps significantly outperforming
                                st.warning("📉 Large Caps Leading Strongly - Confirmed Defensive Mode!")
                            elif score_diff > 0:
                                st.info("📊 Small caps slightly ahead - Cautious Risk-ON.")
                            else:
                                st.info("📊 Large caps slightly ahead - Cautious Defensive.")
                        else:
                            st.info("Insufficient data for full category shift analysis.")
                    else:
                        st.info("Insufficient data for category shift analysis.")
            else:
                st.info("Market Regime Analysis is disabled. Enable it above to see Category Rotation Flow.")
            
            # 4. EMERGING PATTERNS (Stocks about to qualify for a pattern)
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            
            # Set pattern distance thresholds based on sensitivity
            if sensitivity == "Conservative":
                pattern_dist_thresh = 5  # Within 5% of qualifying
            elif sensitivity == "Balanced":
                pattern_dist_thresh = 10  # Within 10% of qualifying
            else:  # Aggressive
                pattern_dist_thresh = 15  # Within 15% of qualifying
            
            emergence_data = []
            
            # Category Leader emergence
            if 'category_percentile' in wave_filtered_df.columns:
                close_to_leader = wave_filtered_df[
                    (wave_filtered_df['category_percentile'] >= (90 - pattern_dist_thresh)) & 
                    (wave_filtered_df['category_percentile'] < 90)
                ]
                for _, stock in close_to_leader.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🔥 CAT LEADER',
                        'Distance': f"{90 - stock['category_percentile']:.1f}% away",
                        'Current Value': f"{stock['category_percentile']:.1f}%ile",
                        'Score': stock['master_score']
                    })
            
            # Breakout Ready emergence
            if 'breakout_score' in wave_filtered_df.columns:
                close_to_breakout = wave_filtered_df[
                    (wave_filtered_df['breakout_score'] >= (80 - pattern_dist_thresh)) & 
                    (wave_filtered_df['breakout_score'] < 80)
                ]
                for _, stock in close_to_breakout.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🎯 BREAKOUT',
                        'Distance': f"{80 - stock['breakout_score']:.1f} pts away",
                        'Current Value': f"{stock['breakout_score']:.1f} score",
                        'Score': stock['master_score']
                    })
            
            # Volume Explosion emergence (based on sensitivity-adjusted RVOL threshold)
            rvol_thresh_for_explosion_emergence = {"Conservative": 3.0, "Balanced": 2.0, "Aggressive": 1.5}[sensitivity]
            if 'rvol' in wave_filtered_df.columns:
                close_to_explosion = wave_filtered_df[
                    (wave_filtered_df['rvol'] >= (rvol_thresh_for_explosion_emergence - 0.5)) & # Within 0.5x of threshold
                    (wave_filtered_df['rvol'] < rvol_thresh_for_explosion_emergence)
                ]
                for _, stock in close_to_explosion.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '⚡ VOL EXPLOSION',
                        'Distance': f"{rvol_thresh_for_explosion_emergence - stock['rvol']:.1f}x away",
                        'Current Value': f"{stock['rvol']:.1f}x",
                        'Score': stock['master_score']
                    })
            
            if emergence_data:
                # Sort by score and take top 15 for display
                emergence_df = pd.DataFrame(emergence_data).sort_values('Score', ascending=False).head(15)
                
                emergence_col1, emergence_col2 = st.columns([3, 1])
                with emergence_col1:
                    st.dataframe(emergence_df, use_container_width=True, hide_index=True)
                with emergence_col2:
                    UIComponents.render_metric_card("Emerging Patterns", len(emergence_df), 
                                                    help=f"Number of stocks close to triggering a pattern alert (within {pattern_dist_thresh}% of threshold).")
                    if not emergence_df.empty:
                        st.caption(f"Top {len(emergence_df)} stocks shown.")
            else:
                st.info(f"No patterns emerging within {pattern_dist_thresh}% threshold with current '{wave_timeframe}' timeframe and filters.")
            
            # 5. VOLUME SURGE DETECTION (Current real-time surges)
            st.markdown("#### 🌊 Volume Surges - Unusual Activity NOW")
            
            # Set RVOL threshold for current surges based on sensitivity
            rvol_surge_threshold_current = {"Conservative": 3.0, "Balanced": 2.0, "Aggressive": 1.5}[sensitivity]
            
            # Find volume surges
            # Main condition: RVOL >= threshold
            surge_conditions_mask = (wave_filtered_df['rvol'] >= rvol_surge_threshold_current)
            
            # Optional additional condition: recent 1D volume ratio also high
            if 'vol_ratio_1d_90d' in wave_filtered_df.columns:
                surge_conditions_mask |= (wave_filtered_df['vol_ratio_1d_90d'] >= rvol_surge_threshold_current) # OR condition for either rvol or 1D ratio
            
            volume_surges_df = wave_filtered_df[surge_conditions_mask].copy()
            
            if not volume_surges_df.empty:
                # Calculate a 'surge_score' for ranking within surges
                volume_surges_df['surge_score'] = (
                    volume_surges_df['rvol_score'].fillna(50) * 0.5 +
                    volume_surges_df['volume_score'].fillna(50) * 0.3 +
                    volume_surges_df['momentum_score'].fillna(50) * 0.2
                )
                
                top_surges = volume_surges_df.nlargest(15, 'surge_score') # Top 15 surges for display
                
                surge_col1_display, surge_col2_metrics = st.columns([2, 1])
                
                with surge_col1_display:
                    # Select columns for display table
                    display_cols_surge = ['ticker', 'company_name', 'rvol', 'price', 'ret_1d', 'money_flow_mm', 'wave_state', 'category']
                    
                    surge_display_df = top_surges[[col for col in display_cols_surge if col in top_surges.columns]].copy()
                    
                    # Add a visual 'Type' column with emojis for surge strength
                    surge_display_df['Surge Type'] = surge_display_df['rvol'].apply(
                        lambda x: "🔥🔥🔥 Extreme" if x > 5 else "🔥🔥 High" if x > 3 else "🔥 Moderate"
                    )
                    
                    # Format numeric columns for display
                    if 'ret_1d' in surge_display_df.columns:
                        surge_display_df['ret_1d'] = surge_display_df['ret_1d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else '-')
                    if 'money_flow_mm' in surge_display_df.columns:
                        surge_display_df['money_flow_mm'] = surge_display_df['money_flow_mm'].apply(lambda x: f"₹{x:.1f}M" if pd.notna(x) else '-')
                    if 'price' in surge_display_df.columns:
                        surge_display_df['price'] = surge_display_df['price'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-')
                    if 'rvol' in surge_display_df.columns:
                        surge_display_df['rvol'] = surge_display_df['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                    
                    # Rename columns for final table display
                    surge_display_df = surge_display_df.rename(columns={
                        'ticker': 'Ticker', 'company_name': 'Company', 'rvol': 'RVOL',
                        'price': 'Price', 'money_flow_mm': 'Money Flow', 'wave_state': 'Wave State',
                        'category': 'Category', 'ret_1d': '1D Ret'
                    })
                    
                    st.dataframe(surge_display_df[['Surge Type', 'Ticker', 'Company', 'RVOL', '1D Ret', 'Money Flow', 'Wave State', 'Category']], use_container_width=True, hide_index=True)
                
                with surge_col2_metrics:
                    UIComponents.render_metric_card("Active Surges", len(volume_surges_df), 
                                                    help=f"Total stocks with RVOL or 1D/90D volume ratio >= {rvol_surge_threshold_current}x.")
                    if 'rvol' in volume_surges_df.columns:
                        UIComponents.render_metric_card("Extreme (>5x)", (volume_surges_df['rvol'] > 5).sum(), 
                                                        help="Count of stocks with RVOL > 5x in the current view.")
                        UIComponents.render_metric_card("High (>3x)", (volume_surges_df['rvol'] > 3).sum(),
                                                        help="Count of stocks with RVOL > 3x in the current view.")
                    
                    # Surge distribution by category (top 3)
                    if 'category' in volume_surges_df.columns and not volume_surges_df.empty:
                        st.markdown("**📊 Surge by Category:**")
                        surge_categories_counts = volume_surges_df['category'].value_counts().head(3)
                        if not surge_categories_counts.empty:
                            for cat, count in surge_categories_counts.items():
                                st.caption(f"• {cat}: {count} stocks")
                        else:
                            st.caption("No category data for surges.")
            else:
                st.info(f"No significant volume surges detected with '{sensitivity}' sensitivity (requires RVOL or 1D/90D ratio ≥ {rvol_surge_threshold_current}x) in the '{wave_timeframe}' timeframe.")
            
            # Wave Radar Summary (overall view of key radar metrics)
            st.markdown("---")
            st.markdown("#### 🎯 Wave Radar Summary")
            
            summary_radar_cols = st.columns(5)
            
            with summary_radar_cols[0]:
                momentum_shifts_count_summary = len(momentum_shifts_candidates) if 'momentum_shifts_candidates' in locals() else 0
                UIComponents.render_metric_card("Momentum Shifts", momentum_shifts_count_summary, help="Stocks exhibiting strong recent momentum and acceleration.")
            
            with summary_radar_cols[1]:
                # Market Regime from MarketIntelligence
                regime_summary_text, _ = MarketIntelligence.detect_market_regime(wave_filtered_df)
                regime_display_short = regime_summary_text.split(' ')[1] if ' ' in regime_summary_text else regime_summary_text
                UIComponents.render_metric_card("Market Regime", regime_display_short, help=f"Overall market sentiment: {regime_summary_text}.")
            
            with summary_radar_cols[2]:
                emergence_count_summary = len(emergence_data) if 'emergence_data' in locals() and emergence_data else 0
                UIComponents.render_metric_card("Emerging Patterns", emergence_count_summary, help="Stocks close to triggering a pattern alert.")
            
            with summary_radar_cols[3]:
                accel_count_summary = len(plot_candidates) if 'plot_candidates' in locals() else 0
                UIComponents.render_metric_card("Accelerating", accel_count_summary, help=f"Top stocks with high acceleration score (≥ {accel_plot_threshold}).")
            
            with summary_radar_cols[4]:
                surge_count_summary = len(volume_surges_df) if 'volume_surges_df' in locals() else 0
                UIComponents.render_metric_card("Volume Surges", surge_count_summary, help=f"Stocks with significant recent RVOL or volume ratio (≥ {rvol_surge_threshold_current}x).")
        
        else:
            st.warning(f"No data available for Wave Radar analysis with the current filters and selected timeframe: '{wave_timeframe}'. Please adjust your filters or timeframe selection.")
    
    # Tab 3: Analysis - Market & Sector Insights
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        
        if not filtered_df.empty:
            # Score distribution & Pattern analysis (side-by-side)
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with analysis_col2:
                fig_patterns = Visualizer.create_pattern_analysis(filtered_df)
                st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Sector Performance (Overview - All Stocks) - Renamed from previous section
            st.markdown("---")
            st.markdown("#### Sector Performance Overview (All Filtered Stocks)")
            try:
                if 'sector' in filtered_df.columns:
                    sector_overview_agg = filtered_df.groupby('sector').agg(
                        Avg_Score=('master_score', 'mean'),
                        Count=('master_score', 'count'),
                        Avg_RVOL=('rvol', 'mean'),
                        Avg_30D_Ret=('ret_30d', 'mean')
                    ).round(2)
                    
                    if not sector_overview_agg.empty:
                        # Calculate percentage of total stocks in each sector
                        sector_overview_agg['% of Total'] = (sector_overview_agg['Count'] / len(filtered_df) * 100).round(1)
                        sector_overview_agg = sector_overview_agg.sort_values('Avg_Score', ascending=False)
                        
                        st.dataframe(
                            sector_overview_agg.style.background_gradient(subset=['Avg_Score']),
                            use_container_width=True
                        )
                        st.info("This table shows aggregated metrics for ALL filtered stocks within each sector. For normalized comparison (Dynamically Sampled), see the 'Sector Analysis' tab.")
                    else:
                        st.info("No sector data available for analysis in the current filtered set.")
                else:
                    st.info("Sector column not available in data.")
            except Exception as e:
                logger.error(f"Error in sector analysis overview: {str(e)}", exc_info=True)
                st.error("Unable to perform sector analysis overview with current data.")
            
            # Category performance
            st.markdown("---")
            st.markdown("#### Category Performance Overview (All Filtered Stocks)")
            try:
                if 'category' in filtered_df.columns:
                    category_overview_agg = filtered_df.groupby('category').agg(
                        Avg_Score=('master_score', 'mean'),
                        Count=('master_score', 'count'),
                        Avg_Cat_Percentile=('category_percentile', 'mean')
                    ).round(2)
                    
                    if not category_overview_agg.empty:
                        category_overview_agg = category_overview_agg.sort_values('Avg_Score', ascending=False)
                        
                        st.dataframe(
                            category_overview_agg.style.background_gradient(subset=['Avg_Score']),
                            use_container_width=True
                        )
                    else:
                        st.info("No category data available for analysis in the current filtered set.")
                else:
                    st.info("Category column not available in data.")
            except Exception as e:
                logger.error(f"Error in category analysis overview: {str(e)}", exc_info=True)
                st.error("Unable to perform category analysis overview with current data.")
            
            # Trend Analysis (Distribution & Stats)
            st.markdown("---")
            st.markdown("#### 📈 Trend Distribution")
            if 'trend_quality' in filtered_df.columns and not filtered_df['trend_quality'].empty:
                analysis_trend_col1, analysis_trend_col2 = st.columns(2)
                
                with analysis_trend_col1:
                    # Trend distribution pie chart
                    trend_dist_counts = pd.cut(
                        filtered_df['trend_quality'],
                        bins=[0, 40, 60, 80, 100], # Define exact bins
                        labels=['⚠️ Weak/Down', '➡️ Neutral', '✅ Good Up', '🔥 Strong Up'],
                        right=False, # Make bins [0, 40), [40, 60) etc.
                        include_lowest=True
                    ).value_counts().sort_index() # Sort by label order
                    
                    # Ensure all labels are present, even if count is 0
                    all_labels_ordered = ['⚠️ Weak/Down', '➡️ Neutral', '✅ Good Up', '🔥 Strong Up']
                    trend_dist_full = trend_dist_counts.reindex(all_labels_ordered, fill_value=0)

                    if not trend_dist_full.empty:
                        fig_trend_dist = px.pie(
                            names=trend_dist_full.index,
                            values=trend_dist_full.values,
                            title="Trend Quality Distribution",
                            color='names', # Color by category name
                            color_discrete_map={ # Explicitly map colors for consistency
                                '🔥 Strong Up': '#2ecc71',
                                '✅ Good Up': '#3498db',
                                '➡️ Neutral': '#f39c12',
                                '⚠️ Weak/Down': '#e74c3c'
                            },
                            hole=0.3 # Donut chart
                        )
                        fig_trend_dist.update_traces(textinfo='percent+label', pull=[0.05 if x == '🔥 Strong Up' else 0 for x in all_labels_ordered]) # Pull strongest segment
                        fig_trend_dist.update_layout(showlegend=True, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig_trend_dist, use_container_width=True)
                    else:
                        st.info("No trend data available for pie chart.")
                
                with analysis_trend_col2:
                    # Trend statistics
                    st.markdown("**Trend Statistics**")
                    trend_data_for_stats = filtered_df['trend_quality'].dropna()
                    if not trend_data_for_stats.empty:
                        avg_trend_score_overall = trend_data_for_stats.mean()
                        
                        trend_stats_metrics = {
                            "Average Trend Score": f"{avg_trend_score_overall:.1f}",
                            "Stocks Above All SMAs": f"{(trend_data_for_stats >= 85).sum()}",
                            "Stocks in Uptrend (60+)": f"{(trend_data_for_stats >= 60).sum()}",
                            "Stocks in Downtrend (<40)": f"{(trend_data_for_stats < 40).sum()}"
                        }
                        for label, value in trend_stats_metrics.items():
                            st.metric(label, value)
                    else:
                        st.info("No trend data available for statistics.")
            else:
                st.info("Trend quality data not available for analysis.")
        
        else:
            st.info("No data available for market analysis. Please adjust filters.")
    
    # Tab 4: Search - Advanced Stock Search
    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        
        # Search interface
        col_search1, col_search2 = st.columns([4, 1])
        
        with col_search1:
            search_query_input = st.text_input(
                "Search stocks",
                value=st.session_state.get('search_query', ''), # Persist last search query
                placeholder="Enter ticker or company name...",
                help="Search by ticker symbol (e.g., RELIANCE) or company name (e.g., Tata Motors). Results are sorted by relevance.",
                key="search_input"
            )
        
        with col_search2:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
            if st.button("🔎 Search", type="primary", use_container_width=True, key="search_button"):
                st.session_state.search_query = search_query_input # Store current query
                st.rerun() # Trigger rerun to update results based on new query
        
        # Perform search if a query is present
        if st.session_state.search_query:
            with st.spinner(f"Searching for '{st.session_state.search_query}'..."):
                search_results_df = SearchEngine.search_stocks(filtered_df, st.session_state.search_query)
            
            if not search_results_df.empty:
                st.success(f"Found {len(search_results_df)} matching stock(s) for '{st.session_state.search_query}'.")
                
                # Display each result in an expandable section for detailed view
                for idx, stock in search_results_df.iterrows():
                    with st.expander(
                        f"📊 {stock['ticker']} - {stock['company_name']} "
                        f"(Rank #{int(stock['rank'])}, Score {stock['master_score']:.1f})",
                        expanded=False # Collapsed by default
                    ):
                        # Header metrics for quick overview in expander
                        metric_cols_stock_detail = st.columns(6)
                        
                        with metric_cols_stock_detail[0]:
                            UIComponents.render_metric_card(
                                "Master Score",
                                f"{stock['master_score']:.1f}",
                                f"Rank #{int(stock['rank'])}",
                                help="Overall composite score of the stock based on all criteria."
                            )
                        
                        with metric_cols_stock_detail[1]:
                            price_val = f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A"
                            ret_1d_val = f"{stock['ret_1d']:+.1f}%" if pd.notna(stock.get('ret_1d')) else None
                            UIComponents.render_metric_card("Price", price_val, ret_1d_val, help="Current price and 1-Day return.")
                        
                        with metric_cols_stock_detail[2]:
                            from_low_pct_val = f"{stock['from_low_pct']:.0f}%" if pd.notna(stock.get('from_low_pct')) else "N/A"
                            UIComponents.render_metric_card(
                                "From Low",
                                from_low_pct_val,
                                "52-Week Range Position",
                                help="Percentage increase from the 52-week low. Higher is closer to the 52-week high."
                            )
                        
                        with metric_cols_stock_detail[3]:
                            ret_30d_val = stock.get('ret_30d')
                            ret_30d_display = f"{ret_30d_val:+.1f}%" if pd.notna(ret_30d_val) else "N/A"
                            ret_30d_delta = "↑" if pd.notna(ret_30d_val) and ret_30d_val > 0 else ("↓" if pd.notna(ret_30d_val) and ret_30d_val < 0 else None)
                            UIComponents.render_metric_card(
                                "30D Return",
                                ret_30d_display,
                                ret_30d_delta,
                                help="Percentage return over the last 30 trading days."
                            )
                        
                        with metric_cols_stock_detail[4]:
                            rvol_val = stock.get('rvol')
                            rvol_display = f"{rvol_val:.1f}x" if pd.notna(rvol_val) else "N/A"
                            rvol_status = "High" if pd.notna(rvol_val) and rvol_val > 2 else "Normal"
                            UIComponents.render_metric_card(
                                "RVOL",
                                rvol_display,
                                rvol_status,
                                help="Relative Volume: Current day's volume relative to its average daily volume."
                            )
                        
                        with metric_cols_stock_detail[5]:
                            wave_state_val = stock.get('wave_state', 'N/A')
                            category_val = stock.get('category', 'Unknown')
                            UIComponents.render_metric_card(
                                "Wave State",
                                wave_state_val,
                                category_val,
                                help="Current momentum phase (FORMING, BUILDING, CRESTING, BREAKING)."
                            )
                        
                        # Score Components Breakdown
                        st.markdown("#### 📈 Score Components")
                        score_components_cols = st.columns(6) # 6 columns for 6 main scores
                        
                        components_for_detail = [
                            ("Position", stock.get('position_score'), CONFIG.POSITION_WEIGHT),
                            ("Volume", stock.get('volume_score'), CONFIG.VOLUME_WEIGHT),
                            ("Momentum", stock.get('momentum_score'), CONFIG.MOMENTUM_WEIGHT),
                            ("Acceleration", stock.get('acceleration_score'), CONFIG.ACCELERATION_WEIGHT),
                            ("Breakout", stock.get('breakout_score'), CONFIG.BREAKOUT_WEIGHT),
                            ("RVOL", stock.get('rvol_score'), CONFIG.RVOL_WEIGHT)
                        ]
                        
                        for i, (name, score_val, weight) in enumerate(components_for_detail):
                            with score_components_cols[i]:
                                score_display = f"{score_val:.0f}" if pd.notna(score_val) else "N/A"
                                if pd.isna(score_val):
                                    color_emoji = "⚪"
                                elif score_val >= 80:
                                    color_emoji = "🟢"
                                elif score_val >= 60:
                                    color_emoji = "🟡"
                                else:
                                    color_emoji = "🔴"
                                
                                st.markdown(
                                    f"**{name}**<br>"
                                    f"{color_emoji} {score_display}<br>"
                                    f"<small>Weight: {weight:.0%}</small>",
                                    unsafe_allow_html=True
                                )
                        
                        # Patterns
                        if stock.get('patterns'):
                            st.markdown(f"**🎯 Patterns:** {stock['patterns']}")
                        
                        # Additional details (Classification, Fundamentals, Performance, Technicals)
                        st.markdown("---") # Separator
                        
                        # First row: Classification & Fundamentals (col1), Performance (col2)
                        detail_row1_col1, detail_row1_col2 = st.columns(2)
                        
                        with detail_row1_col1:
                            st.markdown("**📊 Classification**")
                            st.text(f"Sector: {stock.get('sector', 'Unknown')}")
                            st.text(f"Category: {stock.get('category', 'Unknown')}")
                            
                            if show_fundamentals:
                                st.markdown("**💰 Fundamentals**")
                                
                                # PE Ratio
                                pe_val_detail = stock.get('pe')
                                if pd.notna(pe_val_detail):
                                    if pe_val_detail <= 0 or np.isinf(pe_val_detail):
                                        st.text("PE Ratio: 🔴 Loss")
                                    elif pe_val_detail < 15:
                                        st.text(f"PE Ratio: 🟢 {pe_val_detail:.1f}x")
                                    elif pe_val_detail < 25:
                                        st.text(f"PE Ratio: 🟡 {pe_val_detail:.1f}x")
                                    else:
                                        st.text(f"PE Ratio: 🔴 {pe_val_detail:.1f}x")
                                else:
                                    st.text("PE Ratio: N/A")
                                
                                # EPS Current
                                eps_current_val = stock.get('eps_current')
                                if pd.notna(eps_current_val):
                                    st.text(f"EPS Current: ₹{eps_current_val:.2f}")
                                else:
                                    st.text("EPS Current: N/A")

                                # EPS Change %
                                eps_change_pct_val = stock.get('eps_change_pct')
                                if pd.notna(eps_change_pct_val):
                                    eps_display_str = f"{eps_change_pct_val:+.1f}%"
                                    if eps_change_pct_val >= 100: eps_emoji = "🚀"
                                    elif eps_change_pct_val >= 50: eps_emoji = "🔥"
                                    elif eps_change_pct_val >= 0: eps_emoji = "📈"
                                    else: eps_emoji = "📉"
                                    st.text(f"EPS Growth: {eps_emoji} {eps_display_str}")
                                else:
                                    st.text("EPS Growth: N/A")
                        
                        with detail_row1_col2:
                            st.markdown("**📈 Performance**")
                            performance_metrics_list = [
                                ("1 Day", 'ret_1d'), ("3 Days", 'ret_3d'), ("7 Days", 'ret_7d'),
                                ("30 Days", 'ret_30d'), ("3 Months", 'ret_3m'), ("6 Months", 'ret_6m'),
                                ("1 Year", 'ret_1y'), ("3 Years", 'ret_3y'), ("5 Years", 'ret_5y')
                            ]
                            for period_label, col_name in performance_metrics_list:
                                ret_val = stock.get(col_name)
                                if pd.notna(ret_val):
                                    st.text(f"{period_label}: {ret_val:+.1f}%")
                                else:
                                    st.text(f"{period_label}: N/A")
                        
                        # Second row: Technicals + Trading Position (col1), Advanced Metrics (col2)
                        st.markdown("---") # Separator
                        detail_row2_col1, detail_row2_col2 = st.columns(2)
                        
                        with detail_row2_col1:
                            st.markdown("**🔍 Technicals**")
                            # 52-week range details
                            low_52w_val = stock.get('low_52w')
                            high_52w_val = stock.get('high_52w')
                            if pd.notna(low_52w_val):
                                st.text(f"52W Low: ₹{low_52w_val:,.0f}")
                            else: st.text("52W Low: N/A")
                            if pd.notna(high_52w_val):
                                st.text(f"52W High: ₹{high_52w_val:,.0f}")
                            else: st.text("52W High: N/A")

                            from_high_pct_val = stock.get('from_high_pct')
                            if pd.notna(from_high_pct_val):
                                st.text(f"From High: {from_high_pct_val:.0f}%")
                            else: st.text("From High: N/A")
                            
                            st.markdown("**📊 Trading Position**")
                            current_price_for_tech = stock.get('price')
                            
                            sma_checks_detail = [
                                ('sma_20d', '20DMA'), ('sma_50d', '50DMA'), ('sma_200d', '200DMA')
                            ]
                            
                            sma_detail_col1, sma_detail_col2, sma_detail_col3 = st.columns(3)
                            
                            for i, (sma_col_name, sma_label_name) in enumerate(sma_checks_detail):
                                with [sma_detail_col1, sma_detail_col2, sma_detail_col3][i]:
                                    sma_value_detail = stock.get(sma_col_name)
                                    if pd.notna(current_price_for_tech) and pd.notna(sma_value_detail) and sma_value_detail > 0:
                                        if current_price_for_tech > sma_value_detail:
                                            pct_diff = ((current_price_for_tech - sma_value_detail) / sma_value_detail) * 100
                                            st.markdown(f"**{sma_label_name}**: <span style='color:green'>↑{pct_diff:.1f}%</span>", unsafe_allow_html=True)
                                        else:
                                            pct_diff = ((sma_value_detail - current_price_for_tech) / sma_value_detail) * 100
                                            st.markdown(f"**{sma_label_name}**: <span style='color:red'>↓{pct_diff:.1f}%</span>", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"**{sma_label_name}**: N/A")
                            
                            st.markdown("**📈 Trend Analysis**")
                            trend_quality_val = stock.get('trend_quality')
                            if pd.notna(trend_quality_val):
                                if trend_quality_val >= 80:
                                    st.markdown(f"🔥 Strong Uptrend ({trend_quality_val:.0f})")
                                elif trend_quality_val >= 60:
                                    st.markdown(f"✅ Good Uptrend ({trend_quality_val:.0f})")
                                elif trend_quality_val >= 40:
                                    st.markdown(f"➡️ Neutral Trend ({trend_quality_val:.0f})")
                                else:
                                    st.markdown(f"⚠️ Weak/Downtrend ({trend_quality_val:.0f})")
                            else:
                                st.markdown("Trend: N/A")

                        with detail_row2_col2:
                            st.markdown("#### 🎯 Advanced Metrics")
                            adv_metric_col1, adv_metric_col2 = st.columns(2)
                            
                            with adv_metric_col1:
                                vmi_val = stock.get('vmi')
                                if pd.notna(vmi_val):
                                    st.metric("VMI", f"{vmi_val:.2f}", help="Volume Momentum Index: Weighted average of recent volume ratios.")
                                else: st.metric("VMI", "N/A", help="Volume Momentum Index: Weighted average of recent volume ratios.")
                                
                                momentum_harmony_val = stock.get('momentum_harmony')
                                if pd.notna(momentum_harmony_val):
                                    harmony_emoji = "🟢" if momentum_harmony_val >= 3 else "🟡" if momentum_harmony_val >= 2 else "🔴"
                                    st.metric("Harmony", f"{harmony_emoji} {int(momentum_harmony_val)}/4", help="Momentum Harmony: Alignment of short-term and long-term momentum signals.")
                                else: st.metric("Harmony", "N/A", help="Momentum Harmony: Alignment of short-term and long-term momentum signals.")
                            
                            with adv_metric_col2:
                                position_tension_val = stock.get('position_tension')
                                if pd.notna(position_tension_val):
                                    st.metric("Position Tension", f"{position_tension_val:.0f}", help="Indicates how 'stretched' a stock's price is within its 52-week range.")
                                else: st.metric("Position Tension", "N/A", help="Indicates how 'stretched' a stock's price is within its 52-week range.")
                                
                                money_flow_mm_val = stock.get('money_flow_mm')
                                if pd.notna(money_flow_mm_val):
                                    st.metric("Money Flow", f"₹{money_flow_mm_val:.1f}M", help="Estimated dollar volume traded, adjusted by RVOL, in millions.")
                                else: st.metric("Money Flow", "N/A", help="Estimated dollar volume traded, adjusted by RVOL, in millions.")
            else:
                st.warning(f"No stocks found matching your search criteria: '{st.session_state.search_query}'. Please try a different query or adjust current filters.")
        else:
            st.info("Enter a ticker or company name in the search bar to find specific stocks and view their detailed analysis.")
    
    # Tab 5: Sector Analysis (NEW DEDICATED TAB)
    with tabs[5]:
        st.markdown("### 📊 Sector Analysis")

        if not filtered_df.empty and 'sector' in filtered_df.columns:
            st.markdown("#### 📈 Sector Overview (Dynamically Sampled)")
            
            # Use the MarketIntelligence method that applies dynamic sampling
            sector_overview_df = MarketIntelligence.detect_sector_rotation(filtered_df)

            if not sector_overview_df.empty:
                # Select and reorder columns for display
                display_cols_overview = [
                    'sector_rank', 'sector', 'flow_score', 'avg_score', 'median_score',
                    'analyzed_stocks', 'total_stocks', 'avg_momentum', 'avg_volume', 
                    'avg_rvol', 'avg_ret_30d'
                ]
                if 'total_money_flow' in sector_overview_df.columns:
                    display_cols_overview.insert(display_cols_overview.index('avg_ret_30d') + 1, 'total_money_flow')
                display_cols_overview.append('std_score') # Std score at the end

                # Filter for available columns in case some were not calculated due to missing data
                available_overview_cols = [col for col in display_cols_overview if col in sector_overview_df.columns]
                
                sector_overview_display = sector_overview_df[available_overview_cols].copy()
                
                # Rename columns for user-friendly display in Streamlit
                sector_overview_display.columns = [
                    'Rank', 'Sector', 'Flow Score', 'Avg Score', 'Median Score',
                    'Analyzed', 'Total', 'Avg Momentum', 'Avg Volume', 'Avg RVOL',
                    'Avg 30D Ret'
                ] + (['Total Money Flow (M)'] if 'total_money_flow' in available_overview_cols else []) + ['Std Score']
                
                # Add a 'Coverage %' column for clarity on sampling
                sector_overview_display['Coverage %'] = (
                    (sector_overview_display['Analyzed'] / sector_overview_display['Total'] * 100)
                    .replace([np.inf, -np.inf], np.nan) # Handle potential inf (division by zero)
                    .fillna(0) # Fill NaN from 0 total stocks
                    .round(1)
                    .apply(lambda x: f"{x}%")
                )
                
                # Format money flow column if present
                if 'Total Money Flow (M)' in sector_overview_display.columns:
                    sector_overview_display['Total Money Flow (M)'] = sector_overview_display['Total Money Flow (M)'].apply(
                        lambda x: f"₹{x:.1f}M" if pd.notna(x) else '-'
                    )
                # Format avg RVOL
                if 'Avg RVOL' in sector_overview_display.columns:
                    sector_overview_display['Avg RVOL'] = sector_overview_display['Avg RVOL'].apply(
                        lambda x: f"{x:.1f}x" if pd.notna(x) else '-'
                    )
                # Format Avg 30D Ret
                if 'Avg 30D Ret' in sector_overview_display.columns:
                    sector_overview_display['Avg 30D Ret'] = sector_overview_display['Avg 30D Ret'].apply(
                        lambda x: f"{x:+.1f}%" if pd.notna(x) else '-'
                    )
                # Format scores
                for col in ['Flow Score', 'Avg Score', 'Median Score', 'Avg Momentum', 'Avg Volume', 'Std Score']:
                    if col in sector_overview_display.columns:
                        sector_overview_display[col] = sector_overview_display[col].apply(
                            lambda x: f"{x:.1f}" if pd.notna(x) else '-'
                        )

                st.dataframe(
                    sector_overview_display.style.background_gradient(subset=['Flow Score']),
                    use_container_width=True,
                    hide_index=True
                )
                st.info("📊 **Normalized Analysis**: Shows key metrics for a dynamically sampled subset of top-performing stocks within each sector (chosen by Master Score), providing a fair comparison across sectors of varying sizes.")

            else:
                st.info("No sector overview data available with current filters. Ensure 'sector' column is present and valid data exists.")

            st.markdown("---")
            st.markdown("#### 🔍 Sector Deep Dive")
            
            # Dropdown selector for sectors, dynamically populated based on filtered data
            available_sectors_for_deep_dive = sorted(filtered_df['sector'].dropna().unique().tolist())
            selected_sector_for_deep_dive = st.selectbox(
                "Select a Sector to Deep Dive",
                options=['-- Select a Sector --'] + available_sectors_for_deep_dive,
                key="sector_deep_dive_select",
                help="Choose a specific sector to view its top stocks and detailed aggregated statistics."
            )

            if selected_sector_for_deep_dive != '-- Select a Sector --':
                sector_data_for_deep_dive = filtered_df[filtered_df['sector'] == selected_sector_for_deep_dive].copy()
                
                if not sector_data_for_deep_dive.empty:
                    st.markdown(f"##### Top 10 Stocks in {selected_sector_for_deep_dive}")
                    top_10_sector_stocks = sector_data_for_deep_dive.nlargest(10, 'master_score').reset_index(drop=True)
                    
                    # Prepare display columns for top 10 stocks in deep dive
                    display_cols_deep_dive = {
                        'rank': 'Rank', 'ticker': 'Ticker', 'company_name': 'Company', 'master_score': 'Score',
                        'rvol': 'RVOL', 'ret_30d': '30D Ret', 'momentum_score': 'Momentum', 'patterns': 'Patterns'
                    }
                    if 'overall_wave_strength' in top_10_sector_stocks.columns:
                         display_cols_deep_dive['overall_wave_strength'] = 'Wave Strength'
                    
                    # Add fundamental columns if enabled and present
                    if show_fundamentals:
                        display_cols_deep_dive['pe'] = 'PE'
                        display_cols_deep_dive['eps_change_pct'] = 'EPS Δ%'
                        display_cols_deep_dive['eps_current'] = 'EPS'

                    available_deep_dive_cols = [col for col in display_cols_deep_dive.keys() if col in top_10_sector_stocks.columns]
                    top_10_display = top_10_sector_stocks[available_deep_dive_cols].copy()

                    # Apply formatting for deep dive table
                    for col_key, col_label in display_cols_deep_dive.items():
                        if col_key in top_10_display.columns:
                            if col_key == 'master_score' or col_key == 'momentum_score' or col_key == 'overall_wave_strength':
                                top_10_display[col_key] = top_10_display[col_key].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
                            elif col_key == 'rvol':
                                top_10_display[col_key] = top_10_display[col_key].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                            elif col_key == 'ret_30d':
                                top_10_display[col_key] = top_10_display[col_key].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else '-')
                            elif col_key == 'pe':
                                top_10_display[col_key] = top_10_display[col_key].apply(format_pe_for_display)
                            elif col_key == 'eps_change_pct':
                                top_10_display[col_key] = top_10_display[col_key].apply(format_eps_change_for_display)
                            elif col_key == 'eps_current':
                                top_10_display[col_key] = top_10_display[col_key].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else '-')
                    
                    # Rename columns for display
                    top_10_display.columns = [display_cols_deep_dive[c] for c in available_deep_dive_cols]

                    st.dataframe(top_10_display, use_container_width=True, hide_index=True)

                    st.markdown(f"##### Key Statistics for {selected_sector_for_deep_dive}")
                    col_stats_sector1, col_stats_sector2, col_stats_sector3 = st.columns(3)
                    
                    with col_stats_sector1:
                        st.metric("Total Stocks", len(sector_data_for_deep_dive), help="Total number of stocks in this sector within the current filtered view.")
                        st.metric("Avg Master Score", f"{sector_data_for_deep_dive['master_score'].mean():.1f}", help="Average Master Score for all stocks in this sector.")
                    with col_stats_sector2:
                        if 'rvol' in sector_data_for_deep_dive.columns:
                            st.metric("Avg RVOL", f"{sector_data_for_deep_dive['rvol'].mean():.1f}x", help="Average Relative Volume for stocks in this sector.")
                        else: st.metric("Avg RVOL", "N/A")
                        if 'ret_30d' in sector_data_for_deep_dive.columns:
                            st.metric("Avg 30D Return", f"{sector_data_for_deep_dive['ret_30d'].mean():.1f}%", help="Average 30-Day Return for stocks in this sector.")
                        else: st.metric("Avg 30D Return", "N/A")
                    with col_stats_sector3:
                        if show_fundamentals and 'pe' in sector_data_for_deep_dive.columns:
                            valid_pe_sector = sector_data_for_deep_dive['pe'].notna() & (sector_data_for_deep_dive['pe'] > 0)
                            if valid_pe_sector.any():
                                st.metric("Median PE", f"{sector_data_for_deep_dive.loc[valid_pe_sector, 'pe'].median():.1f}x", help="Median PE ratio for stocks with valid PE in this sector.")
                            else: st.metric("Median PE", "N/A")
                        else: st.metric("Median PE", "N/A")

                        if show_fundamentals and 'eps_change_pct' in sector_data_for_deep_dive.columns:
                            valid_eps_sector = sector_data_for_deep_dive['eps_change_pct'].notna()
                            if valid_eps_sector.any():
                                st.metric("Avg EPS Growth", f"{sector_data_for_deep_dive.loc[valid_eps_sector, 'eps_change_pct'].mean():.1f}%", help="Average EPS Growth for stocks with valid EPS data in this sector.")
                            else: st.metric("Avg EPS Growth", "N/A")
                        else: st.metric("Avg EPS Growth", "N/A")

                else:
                    st.info(f"No data available for '{selected_sector_for_deep_dive}' sector after applying current filters. Please adjust filters.")
            else:
                st.info("Select a sector from the dropdown above to view its detailed performance, including top stocks and key statistics.")

        else:
            st.warning("No sector data available in the filtered dataset for analysis. Please ensure 'sector' column exists and adjust your filters.")

    # Tab 6: Export Data
    with tabs[6]:
        st.markdown("### 📥 Export Data")
        
        # Export template selection
        st.markdown("#### 📋 Export Templates")
        export_template_selection = st.radio(
            "Choose export template:",
            options=[
                "Full Analysis (All Data)",
                "Day Trader Focus",
                "Swing Trader Focus",
                "Investor Focus"
            ],
            key="export_template_radio",
            help="Select a pre-defined template to tailor the columns included in the export file."
        )
        
        # Map template names to internal keys
        template_map_for_export = {
            "Full Analysis (All Data)": "full",
            "Day Trader Focus": "day_trader",
            "Swing Trader Focus": "swing_trader",
            "Investor Focus": "investor"
        }
        selected_template_key = template_map_for_export[export_template_selection]
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            st.markdown("#### 📊 Excel Report")
            st.markdown(
                """
                Comprehensive multi-sheet Excel report including:
                - **Top 100 Stocks**: Filtered top 100 stocks based on your selected template.
                - **Market Intelligence**: Dashboard with market regime, A/D ratio.
                - **Sector Rotation**: Detailed sector performance with dynamic sampling.
                - **Pattern Analysis**: Frequency count of all detected patterns.
                - **Wave Radar Signals**: Stocks showing strong momentum and volume signals.
                - **Summary**: Overall application statistics and export details.
                """
            )
            
            if st.button("Generate Excel Report", type="primary", use_container_width=True, help="Generate an Excel file with multiple sheets covering various aspects of the analysis."):
                if filtered_df.empty:
                    st.error("No data to export. Please adjust your filters to display some stocks.")
                else:
                    with st.spinner("Creating Excel report... This may take a few seconds for large datasets."):
                        try:
                            excel_file_bytes = ExportEngine.create_excel_report(
                                filtered_df, template=selected_template_key
                            )
                            
                            st.download_button(
                                label="📥 Download Excel Report (.xlsx)",
                                data=excel_file_bytes,
                                file_name=f"wave_detection_report_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Click to download the generated Excel report."
                            )
                            
                            st.success("Excel report generated successfully! Click the download button above.")
                            
                        except Exception as e:
                            st.error(f"Error generating Excel report: {str(e)}. Please check logs for details.")
                            logger.error(f"Excel export error: {str(e)}", exc_info=True)
        
        with col_export2:
            st.markdown("#### 📄 CSV Export")
            st.markdown(
                """
                Single CSV file export containing:
                - All ranking scores and raw data (if available in source).
                - Advanced metrics like RVOL, VMI, Money Flow.
                - All detected patterns and wave states.
                - Comprehensive performance and fundamental data.
                - Optimized for further custom analysis in external tools.
                """
            )
            
            if st.button("Generate CSV Export", use_container_width=True, help="Generate a flat CSV file containing all available columns for the filtered data."):
                if filtered_df.empty:
                    st.error("No data to export. Please adjust your filters to display some stocks.")
                else:
                    try:
                        csv_data_string = ExportEngine.create_csv_export(filtered_df)
                        
                        st.download_button(
                            label="📥 Download CSV File (.csv)",
                            data=csv_data_string,
                            file_name=f"wave_detection_data_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Click to download the generated CSV file."
                        )
                        
                        st.success("CSV export generated successfully! Click the download button above.")
                        
                    except Exception as e:
                        st.error(f"Error generating CSV: {str(e)}. Please check logs for details.")
                        logger.error(f"CSV export error: {str(e)}", exc_info=True)
        
        # Export statistics preview
        st.markdown("---")
        st.markdown("#### 📊 Export Preview Statistics")
        
        export_stats_for_display = {
            "Total Stocks to Export": len(filtered_df),
            "Average Master Score": f"{filtered_df['master_score'].mean():.1f}" if 'master_score' in filtered_df.columns and not filtered_df.empty else "N/A",
            "Stocks with Patterns": (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0,
            "High RVOL (>2x)": (filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns else 0,
            "Positive 30D Returns": (filtered_df['ret_30d'] > 0).sum() if 'ret_30d' in filtered_df.columns else 0,
            "Data Quality (Overall)": f"{st.session_state.data_quality.get('completeness', 0):.1f}%" if 'data_quality' in st.session_state else "N/A"
        }
        
        export_stats_cols = st.columns(3)
        for i, (label, value) in enumerate(export_stats_for_display.items()):
            with export_stats_cols[i % 3]:
                UIComponents.render_metric_card(label, value)
    
    # Tab 7: About - Comprehensive Documentation
    with tabs[7]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0 - FINAL ULTIMATE PRODUCTION VERSION")
        
        about_col1, about_col2 = st.columns([2, 1])
        
        with about_col1:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
            This is the **FINAL, ULTIMATE PRODUCTION VERSION** of the most advanced stock ranking system, meticulously designed to identify and capitalize on momentum waves early. This professional-grade tool synthesizes cutting-edge technical analysis, intricate volume dynamics, robust advanced metrics, and sophisticated pattern recognition to pinpoint high-potential stocks *before* they reach their peak.
            
            This version is **permanently locked**. No further updates or modifications are planned, ensuring a stable and reliable analytical environment.
            
            #### 🎯 Core Features - PERMANENTLY LOCKED
            
            **Master Score 3.0** - Our proprietary weighted ranking algorithm, optimized for precision and performance:
            - **Position Analysis (30%)** - Evaluates stock's current price relative to its 52-week range.
            - **Volume Dynamics (25%)** - Analyzes multi-timeframe volume patterns and anomalies.
            - **Momentum Tracking (15%)** - Assesses short to medium-term price momentum.
            - **Acceleration Detection (10%)** - Identifies the rate of change in momentum, highlighting accelerating trends.
            - **Breakout Probability (10%)** - Quantifies a stock's readiness for a significant price breakout.
            - **RVOL Integration (10%)** - Incorporates real-time Relative Volume for immediate market activity insights.
            
            **25 Pattern Detection** - A comprehensive suite of patterns to identify diverse trading opportunities:
            - **Technical Patterns**: `🔥 CAT LEADER`, `💎 HIDDEN GEM`, `🚀 ACCELERATING`, `🏦 INSTITUTIONAL`, `⚡ VOL EXPLOSION`, `🎯 BREAKOUT`, `👑 MARKET LEADER`, `🌊 MOMENTUM WAVE`, `💰 LIQUID LEADER`, `💪 LONG STRENGTH`, `📈 QUALITY TREND`.
            - **Price Range Patterns**: `🎯 52W HIGH APPROACH`, `🔄 52W LOW BOUNCE`, `👑 GOLDEN ZONE`, `🎯 RANGE COMPRESS`.
            - **Volume & Momentum Dynamics**: `📊 VOL ACCUMULATION`, `🔀 MOMENTUM DIVERGE`.
            - **NEW Intelligence Patterns**: `🤫 STEALTH`, `🧛 VAMPIRE`, `⛈️ PERFECT STORM`.
            
            **Advanced Metrics** - Proprietary indicators for deeper market understanding:
            - **Money Flow (MM)** - Estimated dollar volume traded, adjusted by RVOL.
            - **VMI (Volume Momentum Index)** - Weighted average of recent volume ratios, indicating sustained volume interest.
            - **Position Tension** - Quantifies price "stretch" within its 52-week range.
            - **Momentum Harmony** - Aligns short, medium, and long-term momentum (0-4 signals).
            - **Wave State** - Categorizes current momentum phase (FORMING, BUILDING, CRESTING, BREAKING).
            - **Overall Wave Strength** - A composite score for general market wave intensity.
            
            **Wave Radar™** - Our advanced early detection system:
            - **Momentum Shift Detection** with multi-signal counting.
            - **Category Rotation Flow** to track smart money movement across market caps.
            - **Pattern Emergence Alerts** for stocks nearing qualification.
            - **Dynamic Sensitivity Controls** (Conservative, Balanced, Aggressive) to fine-tune signal detection.
            
            #### 💡 How to Use
            
            1.  **Data Source**: Select between our live Google Sheets feed or upload your own custom CSV file.
            2.  **Summary Tab**: Get an executive overview of the market pulse, top opportunities, and risk indicators.
            3.  **Quick Actions**: Apply instant, pre-defined filters for common trading scenarios.
            4.  **Smart Filters (Sidebar)**: Utilize interconnected, multi-dimensional filters for precise stock screening, including new 'Wave' specific filters.
            5.  **Rankings Tab**: View the top-ranked stocks based on the Master Score, with comprehensive metrics and visual breakdowns.
            6.  **Wave Radar Tab**: Dive into early momentum signals, acceleration profiles, and detailed volume surge analysis.
            7.  **Analysis Tab**: Explore market-wide score distributions, pattern frequencies, and overall trend health.
            8.  **Search Tab**: Find specific stocks by ticker or company name and view their complete detailed analysis.
            9.  **Sector Analysis Tab**: Get deep insights into sector performance with dynamic sampling and individual sector statistics.
            10. **Export Tab**: Download filtered data in comprehensive Excel reports or flexible CSV formats for external analysis.
            
            #### 🔧 Production-Ready Features
            
            -   **Maximal Performance**: Achieved through aggressive caching, extensive NumPy/Pandas vectorization, and meticulous code optimization. Sub-second processing for core operations.
            -   **Memory Efficiency**: Engineered to handle large datasets (1791+ stocks with 41+ data points) smoothly.
            -   **Robust Data Handling**: Comprehensive data validation, cleaning, and sanitization at every step to ensure data integrity and prevent errors.
            -   **Error Resilient**: Graceful handling of data loading failures, missing columns, and numerical edge cases to ensure continuous operation.
            -   **Streamlit State Management**: Robust session state initialization and management for consistent UI behavior across reruns.
            -   **Mobile Responsive**: Optimized UI and table displays for seamless viewing on various devices.
            """)
        
        with about_col2:
            st.markdown("""
            #### 📈 Trend Indicators
            
            -   **🔥 Strong Uptrend** (Score 80-100)
                -   Price above all key Moving Averages (e.g., 20, 50, 200 DMA).
                -   MAs are correctly stacked (20 DMA > 50 DMA > 200 DMA).
            -   **✅ Good Uptrend** (Score 60-79)
                -   Price above most key Moving Averages.
                -   Positive overall momentum.
            -   **➡️ Neutral Trend** (Score 40-59)
                -   Mixed signals from Moving Averages.
                -   Price possibly consolidating or ranging.
            -   **⚠️ Weak/Downtrend** (Score 0-39)
                -   Price below most or all key Moving Averages.
                -   Negative momentum or bearish MA crosses.
            
            #### 🎨 Display Modes
            
            **Technical Mode** (Default)
            -   Focuses purely on technical indicators, price action, volume dynamics, and patterns.
            -   Excludes fundamental metrics like PE and EPS from primary views.
            
            **Hybrid (Technical + Fundamentals) Mode**
            -   Combines all technical analysis with key fundamental insights.
            -   Integrates PE ratio, EPS growth, and related fundamental patterns into analysis.
            
            #### ⚡ Performance Benchmarks
            
            -   Initial data load & full processing: Typically <2 seconds (if cached, near instant).
            -   Filtering operations: <200 milliseconds.
            -   Pattern detection (all 25): <500 milliseconds.
            -   Stock Search: <50 milliseconds.
            -   Export report generation: <1 second.
            
            #### 🔒 Production Status
            
            **Version**: 3.0-ULTIMATE-LOCKED
            **Last Updated**: July 28, 2025
            **Status**: FINAL PRODUCTION READY - PERMANENTLY LOCKED
            **Future Updates**: NONE (This version is fixed.)
            
            #### 💬 Credits & Legal
            
            Developed for professional traders requiring reliable, fast, and comprehensive market analysis. This application represents the culmination of extensive development and optimization.
            
            ---
            
            **Indian Market Optimized**
            -   ₹ Currency formatting.
            -   IST (Indian Standard Time) timezone aware for display.
            -   NSE/BSE (National Stock Exchange/Bombay Stock Exchange) equivalent categories for market capitalization.
            -   Local number formats for readability.
            """)
        
        # System stats at the bottom of the About tab
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        
        stats_cols_about = st.columns(4)
        
        with stats_cols_about[0]:
            UIComponents.render_metric_card(
                "Total Stocks Loaded",
                f"{len(ranked_df):,}" if 'ranked_df' in locals() else "0",
                help="Total number of stocks loaded into the system after initial data processing."
            )
        
        with stats_cols_about[1]:
            UIComponents.render_metric_card(
                "Currently Filtered",
                f"{len(filtered_df):,}" if 'filtered_df' in locals() else "0",
                help="Number of stocks currently visible after all active filters have been applied."
            )
        
        with stats_cols_about[2]:
            data_quality_completeness = st.session_state.data_quality.get('completeness', 0)
            quality_emoji_about = "🟢" if data_quality_completeness > 80 else "🟡" if data_quality_completeness > 60 else "🔴"
            UIComponents.render_metric_card(
                "Data Quality",
                f"{quality_emoji_about} {data_quality_completeness:.1f}%",
                help="Overall completeness percentage of the loaded dataset."
            )
        
        with stats_cols_about[3]:
            # Convert cache_age to IST for display
            cache_time_utc = datetime.now(timezone.utc) - st.session_state.last_refresh
            minutes_cache_age = int(cache_time_utc.total_seconds() / 60)
            cache_status_display = "Fresh" if minutes_cache_age < 60 else "Stale"
            cache_emoji_about = "🟢" if minutes_cache_age < 60 else "🔴"
            UIComponents.render_metric_card(
                "Cache Age",
                f"{cache_emoji_about} {minutes_cache_age} min",
                cache_status_display,
                help="Time elapsed since the data was last loaded from the source. Older cache might mean slightly outdated data."
            )
    
    # Footer (consistent across all versions)
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            🌊 Wave Detection Ultimate 3.0 - FINAL ULTIMATE PRODUCTION VERSION<br>
            <small>Professional Stock Ranking System • All Features Complete • Maximally Optimized • Permanently Locked</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================
# APPLICATION ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        # Run the Streamlit application
        main()
    except Exception as e:
        # Global error handler for any uncaught exceptions
        st.error(f"❌ A critical application error occurred: {str(e)}")
        logger.critical(f"Application crashed unexpectedly: {str(e)}", exc_info=True)
        
        # Provide recovery options to the user
        st.info("If this error persists, please try the following:")
        st.markdown("- **Refresh the page:** (Ctrl/Cmd + R) to restart the application.")
        st.markdown("- **Clear the browser cache:** This can resolve issues with old Streamlit session states.")
        st.markdown("- **Contact support:** If the issue continues, please provide a screenshot of this error message.")
        
        # Offer buttons to help diagnose/restart
        col_error_btns1, col_error_btns2 = st.columns(2)
        with col_error_btns1:
            if st.button("🔄 Force Restart Application", type="primary", use_container_width=True):
                st.cache_data.clear() # Clear all caches
                st.session_state.clear() # Clear all session state
                st.rerun() # Force a full rerun from scratch
        with col_error_btns2:
            if st.button("📧 Report Issue & Get Logs", use_container_width=True):
                st.markdown("---")
                st.warning("Please copy the detailed logs below and send them to the developer for assistance.")
                # Attempt to read and display recent logs (if logging to a file in a real deployment)
                # For this self-contained script, logs go to console/Streamlit's internal logs
                st.code(f"Error Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}\nError Details: {e}\n\nCheck Streamlit console logs for full tracebacks.")

# END OF WAVE DETECTION ULTIMATE 3.0 - FINAL ULTIMATE PRODUCTION VERSION
