"""
Wave Detection Ultimate 3.0 - FINAL ENHANCED PRODUCTION VERSION
===============================================================
Professional Stock Ranking System with Advanced Analytics
All bugs fixed, optimized for Streamlit Community Cloud
Enhanced with Industry Filter and Analysis Features

Version: 3.0.8-FINAL-INDUSTRY
Last Updated: August 2025
Status: PRODUCTION READY - Feature Complete
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
    
    # Data source - Default Google Sheets (users can change)
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM/edit?usp=sharing"
    DEFAULT_GID: str = "1823439984"
    SPREADSHEET_ID_LENGTH: int = 44  # Standard Google Sheets ID length
    
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
        "golden_cross": 75,
        "volume_surge": 85,
        "consistent_performer": 80,
        "range_bound_breakout": 75,
        "quiet_accumulation": 70,
        "earnings_beat": 85,
        "fundamental_value": 75,
        "recovery_play": 80,
        "smart_money": 85,
        "stealth_mode": 75,
        "position_builder": 80,
        "harmony_wave": 85,
        "apex_predator": 95
    })

# Create singleton instance
CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """Monitor and optimize performance"""
    
    @staticmethod
    def timer(target_time: float = 1.0):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                if elapsed > target_time:
                    logger.warning(f"{func.__name__} took {elapsed:.2f}s (target: {target_time}s)")
                else:
                    logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
                
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """Get current memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }

# ============================================
# DATA VALIDATION AND PROCESSING
# ============================================

class DataValidator:
    """Validate data integrity and quality"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                          context: str = "") -> Tuple[bool, str]:
        """Validate dataframe has required structure"""
        
        if df is None or df.empty:
            return False, f"{context}: Empty dataframe"
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"{context}: Missing columns: {missing_cols}"
        
        if len(df) < 10:
            return False, f"{context}: Too few rows ({len(df)})"
        
        return True, "Valid"
    
    @staticmethod
    def sanitize_numeric(value: Any, default: float = 0.0, 
                        bounds: Optional[Tuple[float, float]] = None) -> float:
        """Safely convert to numeric with bounds checking"""
        
        if pd.isna(value) or value is None:
            return default
        
        try:
            # Handle string representations
            if isinstance(value, str):
                cleaned = value.strip().upper()
                
                # Handle special cases
                if cleaned in ['', 'N/A', 'NA', 'NAN', 'NULL', 'NONE', '-', '#N/A', '#ERROR!', '#DIV/0!', 'INF', '-INF']:
                    return default
                
                # Remove common symbols and spaces
                cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
                
                # Convert to float
                result = float(cleaned)
            else:
                result = float(value)
            
            # Apply bounds if specified
            if bounds:
                min_val, max_val = bounds
                result = np.clip(result, min_val, max_val)
            
            # Check for unreasonable values
            if np.isnan(result) or np.isinf(result):
                return default
            
            return result
            
        except (ValueError, TypeError, AttributeError):
            return default
    
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
                         spreadsheet_id: str = None, gid: str = None,
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
            # Use provided spreadsheet ID or default
            if not spreadsheet_id:
                # Extract from URL if provided
                if gid and "docs.google.com" in str(gid):
                    try:
                        spreadsheet_id = str(gid).split("/d/")[1].split("/")[0]
                    except:
                        spreadsheet_id = None
                
                if not spreadsheet_id:
                    # Use default
                    url_parts = CONFIG.DEFAULT_SHEET_URL.split("/d/")[1].split("/")[0]
                    spreadsheet_id = url_parts
            
            # Use default GID if not provided
            if not gid or "docs.google.com" in str(gid):
                gid = CONFIG.DEFAULT_GID
            
            # Construct CSV URL
            base_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from Google Sheets: {spreadsheet_id[:8]}...")
            
            try:
                df = pd.read_csv(csv_url, low_memory=False)
                metadata['source'] = f"Google Sheets (ID: {spreadsheet_id[:8]}...)"
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
        raise

# ============================================
# DATA PROCESSOR - ENHANCED WITH SAFE OPERATIONS
# ============================================

class DataProcessor:
    """Process raw data into clean format with enhanced safety"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=2.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Process dataframe with all cleaning and transformations"""
        
        logger.info(f"Processing {len(df)} rows with {len(df.columns)} columns")
        
        # Store original shape
        original_shape = df.shape
        
        # Basic cleaning
        df = DataProcessor._basic_cleaning(df)
        
        # Process numeric columns with safety
        df = DataProcessor._process_numeric_columns(df, metadata)
        
        # Process text columns
        df = DataProcessor._process_text_columns(df)
        
        # Add derived columns
        df = DataProcessor._add_derived_columns(df)
        
        # Validate tickers
        df = DataProcessor._validate_tickers(df, metadata)
        
        # Final quality check
        df = DataProcessor._final_quality_check(df, metadata)
        
        logger.info(f"Processing complete: {original_shape} -> {df.shape}")
        
        return df
    
    @staticmethod
    def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        # Remove completely empty rows/columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Remove duplicate tickers (keep first)
        if 'ticker' in df.columns:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=['ticker'], keep='first')
            if len(df) < before_dedup:
                logger.warning(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    @staticmethod
    def _process_numeric_columns(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Process numeric columns with enhanced safety"""
        
        numeric_columns = {
            # Price columns
            'price': (0.01, 100000),
            'low_52w': (0.01, 100000),
            'high_52w': (0.01, 100000),
            'prev_close': (0.01, 100000),
            
            # Return columns (percentages)
            'ret_1d': (-50, 50),
            'ret_3d': (-75, 75),
            'ret_7d': (-90, 90),
            'ret_30d': (-95, 200),
            'ret_3m': (-95, 500),
            'ret_6m': (-95, 1000),
            'ret_1y': (-95, 2000),
            'ret_3y': (-95, 5000),
            'ret_5y': (-95, 10000),
            
            # Volume columns
            'volume_1d': (0, 1e12),
            'volume_7d': (0, 1e13),
            'volume_30d': (0, 1e14),
            'volume_90d': (0, 1e15),
            'volume_180d': (0, 1e15),
            
            # Ratios
            'rvol': (0, 100),
            'pe': (-1000, 1000),
            'eps_current': (-100, 100),
            'eps_last_qtr': (-100, 100),
            
            # Volume ratios
            'vol_ratio_1d_90d': (0, 10),
            'vol_ratio_7d_90d': (0, 10),
            'vol_ratio_30d_90d': (0, 10),
            'vol_ratio_1d_180d': (0, 10),
            'vol_ratio_7d_180d': (0, 10),
            'vol_ratio_30d_180d': (0, 10),
            'vol_ratio_90d_180d': (0, 10),
        }
        
        for col, bounds in numeric_columns.items():
            if col in df.columns:
                before_nulls = df[col].isna().sum()
                df[col] = df[col].apply(lambda x: DataValidator.sanitize_numeric(x, bounds=bounds))
                after_nulls = df[col].isna().sum()
                
                if after_nulls > before_nulls:
                    new_nulls = after_nulls - before_nulls
                    metadata['warnings'].append(f"{col}: {new_nulls} invalid values converted to NaN")
        
        # Handle percentage columns specially
        for col in CONFIG.PERCENTAGE_COLUMNS:
            if col in df.columns and col not in numeric_columns:
                df[col] = df[col].apply(lambda x: DataValidator.sanitize_numeric(x, bounds=(-100, 10000)))
        
        # Handle from_low_pct and from_high_pct with special logic
        if 'from_low_pct' in df.columns:
            df['from_low_pct'] = df['from_low_pct'].apply(
                lambda x: DataValidator.sanitize_numeric(x, bounds=(0, 10000))
            )
        
        if 'from_high_pct' in df.columns:
            # from_high_pct should be negative or zero
            df['from_high_pct'] = df['from_high_pct'].apply(
                lambda x: DataValidator.sanitize_numeric(x, bounds=(-99.99, 0))
            )
        
        return df
    
    @staticmethod
    def _process_text_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Process text columns"""
        
        text_columns = ['ticker', 'company_name', 'category', 'sector', 'industry']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
                
                # Special handling for ticker
                if col == 'ticker':
                    # Ensure uppercase and remove special characters
                    df[col] = df[col].str.upper().str.replace(r'[^A-Z0-9\-\.]', '', regex=True)
        
        return df
    
    @staticmethod
    def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns if missing"""
        
        # Ensure critical calculated columns exist
        if 'rvol' not in df.columns and all(col in df.columns for col in ['volume_1d', 'volume_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_vol_30d = df['volume_30d'] / 30
                df['rvol'] = np.where(avg_vol_30d > 0, df['volume_1d'] / avg_vol_30d, 1.0)
                df['rvol'] = df['rvol'].fillna(1.0).clip(0, 100)
        
        # Add year if missing
        if 'year' not in df.columns:
            df['year'] = datetime.now().year
        
        # Calculate market cap if missing
        if 'market_cap' not in df.columns and all(col in df.columns for col in ['price', 'volume_1d']):
            # Rough estimation - not accurate but better than nothing
            df['market_cap'] = df['price'] * df['volume_1d'] * 10  # Very rough estimate
        
        return df
    
    @staticmethod
    def _validate_tickers(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Validate and clean ticker symbols"""
        
        if 'ticker' not in df.columns:
            metadata['errors'].append("No ticker column found!")
            return df
        
        # Remove invalid tickers
        initial_count = len(df)
        
        # Basic ticker validation
        valid_ticker_mask = (
            df['ticker'].notna() & 
            (df['ticker'] != '') & 
            (df['ticker'].str.len() <= 10) &  # Max ticker length
            (df['ticker'].str.len() >= 1)      # Min ticker length
        )
        
        df = df[valid_ticker_mask]
        
        if len(df) < initial_count:
            removed = initial_count - len(df)
            metadata['warnings'].append(f"Removed {removed} invalid tickers")
            logger.warning(f"Removed {removed} invalid tickers")
        
        return df
    
    @staticmethod
    def _final_quality_check(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Final quality checks and reporting"""
        
        # Calculate data quality metrics
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.notna().sum().sum()
        completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
        
        metadata['data_quality'] = {
            'completeness': completeness,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'critical_columns_present': all(col in df.columns for col in CONFIG.CRITICAL_COLUMNS),
            'important_columns_present': sum(col in df.columns for col in CONFIG.IMPORTANT_COLUMNS)
        }
        
        # Log quality metrics
        logger.info(f"Data quality: {completeness:.1f}% complete, {len(df)} valid rows")
        
        # Store quality metrics in session state for UI display
        if 'data_quality' not in st.session_state:
            st.session_state.data_quality = {}
        st.session_state.data_quality.update(metadata['data_quality'])
        
        return df

# ============================================
# SCORING ENGINE - PRODUCTION READY
# ============================================

class RankingEngine:
    """Calculate all scores and rankings with proper error handling"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores and master score"""
        
        if df.empty:
            return df
        
        logger.info("Calculating component scores...")
        
        # Calculate each component score with safety
        df['position_score'] = RankingEngine._calculate_position_score(df)
        df['volume_score'] = RankingEngine._calculate_volume_score(df)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df)
        
        # Calculate Master Score 3.0
        df['master_score'] = RankingEngine._calculate_master_score(df)
        
        # Add rankings
        df = RankingEngine._add_rankings(df)
        
        logger.info("Score calculation complete")
        
        return df
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score (30% weight) with enhanced logic"""
        
        position_score = pd.Series(50, index=df.index, dtype=float)  # Default score
        
        if 'from_low_pct' not in df.columns or 'from_high_pct' not in df.columns:
            logger.warning("Position data missing, using default scores")
            return position_score
        
        # Get clean data with safe operations
        from_low = df['from_low_pct'].fillna(0)
        from_high = df['from_high_pct'].fillna(-50)  # Default to middle
        
        # Calculate position ratio (0 = at low, 1 = at high)
        # Safe division to avoid divide by zero
        range_size = from_low - from_high
        range_size = range_size.replace(0, 1)  # Avoid division by zero
        
        position_ratio = (0 - from_high) / range_size
        position_ratio = position_ratio.clip(0, 1)
        
        # Non-linear scoring for better differentiation
        # Stocks near 52w high (>80% range) get bonus
        high_position = position_ratio > 0.8
        position_score[high_position] = 80 + (position_ratio[high_position] - 0.8) * 100
        
        # Middle range (20-80%)
        middle_position = (position_ratio >= 0.2) & (position_ratio <= 0.8)
        position_score[middle_position] = 40 + position_ratio[middle_position] * 50
        
        # Low range (<20%) - penalized
        low_position = position_ratio < 0.2
        position_score[low_position] = position_ratio[low_position] * 200
        
        # Extra bonus for extreme positions
        very_high = from_high > -5  # Within 5% of 52w high
        position_score[very_high] = 100
        
        very_low = from_low < 10  # Within 10% of 52w low
        position_score[very_low] = 0
        
        return position_score.clip(0, 100)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate volume score (25% weight) with sophisticated logic"""
        
        volume_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check for volume ratio columns
        vol_ratio_cols = [col for col in CONFIG.VOLUME_RATIO_COLUMNS if col in df.columns]
        
        if not vol_ratio_cols:
            logger.warning("No volume ratio data available")
            return volume_score
        
        # Calculate composite volume momentum
        volume_momentum = pd.Series(0, index=df.index, dtype=float)
        weights = {
            'vol_ratio_1d_90d': 0.35,    # Recent volume most important
            'vol_ratio_7d_90d': 0.25,
            'vol_ratio_30d_90d': 0.20,
            'vol_ratio_1d_180d': 0.10,
            'vol_ratio_7d_180d': 0.05,
            'vol_ratio_30d_180d': 0.05
        }
        
        for col, weight in weights.items():
            if col in df.columns:
                ratio = df[col].fillna(1.0).clip(0, 10)  # Cap at 10x
                # Non-linear scoring: reward ratios > 1
                score_contribution = np.where(
                    ratio > 1,
                    50 + (ratio - 1) * 5.5,  # Positive momentum
                    ratio * 50               # Negative momentum
                )
                volume_momentum += score_contribution * weight
        
        # Normalize to 0-100
        volume_score = volume_momentum.clip(0, 100)
        
        # Bonus for extreme volume events
        if 'vol_ratio_1d_90d' in df.columns:
            extreme_volume = df['vol_ratio_1d_90d'] > 3
            volume_score[extreme_volume] = volume_score[extreme_volume].clip(lower=80)
        
        return volume_score
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score (15% weight) with safe operations"""
        
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        # Define momentum factors with weights
        momentum_factors = {
            'ret_1d': 0.20,
            'ret_7d': 0.25,
            'ret_30d': 0.35,
            'ret_3m': 0.20
        }
        
        available_factors = {k: v for k, v in momentum_factors.items() if k in df.columns}
        
        if not available_factors:
            logger.warning("No return data available for momentum calculation")
            return momentum_score
        
        # Normalize weights
        total_weight = sum(available_factors.values())
        normalized_weights = {k: v/total_weight for k, v in available_factors.items()}
        
        # Calculate weighted momentum
        for col, weight in normalized_weights.items():
            ret_data = df[col].fillna(0)
            
            # Non-linear scoring
            # Map returns to scores (roughly: -50% = 0, 0% = 50, +50% = 100)
            score_contribution = 50 + ret_data.clip(-50, 50)
            momentum_score += (score_contribution - 50) * weight
        
        # Add consistency bonus
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            all_positive = (df['ret_1d'] > 0) & (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            momentum_score[all_positive] += 10
            
            # Acceleration bonus
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            momentum_score[accelerating] += 5
        
        return momentum_score.clip(0, 100)
    
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
        ret_1d = df['ret_1d'].fillna(0) if 'ret_1d' in df.columns else pd.Series(0, index=df.index)
        ret_7d = df['ret_7d'].fillna(0) if 'ret_7d' in df.columns else pd.Series(0, index=df.index)
        ret_30d = df['ret_30d'].fillna(0) if 'ret_30d' in df.columns else pd.Series(0, index=df.index)
        
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
        
        # Factor 1: Distance from 52w high (40% weight)
        if 'from_high_pct' in df.columns:
            distance_score = (df['from_high_pct'].fillna(-50) + 50) * 2  # Convert to 0-100
            distance_score = distance_score.clip(0, 100)
            breakout_score = breakout_score * 0.6 + distance_score * 0.4
        
        # Factor 2: Volume surge (30% weight)
        if 'vol_ratio_1d_90d' in df.columns:
            volume_surge = df['vol_ratio_1d_90d'].fillna(1.0)
            volume_score = volume_surge.clip(0, 5) * 20  # 5x volume = 100 score
            breakout_score = breakout_score * 0.7 + volume_score * 0.3
        
        # Factor 3: Momentum consistency (30% weight)
        if all(col in df.columns for col in ['ret_1d', 'ret_7d']):
            consistent = (df['ret_1d'] > 0) & (df['ret_7d'] > 0)
            consistency_score = consistent.astype(float) * 100
            breakout_score = breakout_score * 0.7 + consistency_score * 0.3
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate relative volume score"""
        rvol_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'rvol' in df.columns:
            rvol = df['rvol'].fillna(1.0)
            
            # Non-linear scoring
            # RVOL < 0.5: 0-25
            # RVOL 0.5-1: 25-50  
            # RVOL 1-2: 50-75
            # RVOL 2-5: 75-95
            # RVOL > 5: 95-100
            
            rvol_score = np.select(
                [rvol < 0.5, rvol < 1, rvol < 2, rvol < 5, rvol >= 5],
                [rvol * 50, 25 + rvol * 25, 50 + (rvol - 1) * 25, 75 + (rvol - 2) * 6.67, 95 + (rvol - 5) * 1],
                default=50
            )
        
        return rvol_score.clip(0, 100)
    
    @staticmethod
    def _calculate_master_score(df: pd.DataFrame) -> pd.Series:
        """Calculate Master Score 3.0 with validated weights"""
        
        # Ensure all component scores exist
        required_scores = ['position_score', 'volume_score', 'momentum_score', 
                          'acceleration_score', 'breakout_score', 'rvol_score']
        
        for score in required_scores:
            if score not in df.columns:
                logger.warning(f"Missing {score}, using default value of 50")
                df[score] = 50
        
        # Calculate weighted master score
        master_score = (
            df['position_score'] * CONFIG.POSITION_WEIGHT +
            df['volume_score'] * CONFIG.VOLUME_WEIGHT +
            df['momentum_score'] * CONFIG.MOMENTUM_WEIGHT +
            df['acceleration_score'] * CONFIG.ACCELERATION_WEIGHT +
            df['breakout_score'] * CONFIG.BREAKOUT_WEIGHT +
            df['rvol_score'] * CONFIG.RVOL_WEIGHT
        )
        
        # Ensure weights sum to 1.0 (they do in CONFIG)
        weight_sum = (CONFIG.POSITION_WEIGHT + CONFIG.VOLUME_WEIGHT + 
                     CONFIG.MOMENTUM_WEIGHT + CONFIG.ACCELERATION_WEIGHT + 
                     CONFIG.BREAKOUT_WEIGHT + CONFIG.RVOL_WEIGHT)
        
        if abs(weight_sum - 1.0) > 0.001:
            logger.error(f"Weight sum is {weight_sum}, not 1.0!")
            master_score = master_score / weight_sum
        
        return master_score.clip(0, 100).round(1)
    
    @staticmethod
    def _add_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Add various ranking columns"""
        
        if df.empty:
            return df
        
        # Overall rank
        df['rank'] = df['master_score'].rank(method='dense', ascending=False).astype(int)
        
        # Category rank
        if 'category' in df.columns:
            df['category_rank'] = df.groupby('category')['master_score'].rank(
                method='dense', ascending=False
            ).astype(int)
        
        # Sector rank  
        if 'sector' in df.columns:
            df['sector_rank'] = df.groupby('sector')['master_score'].rank(
                method='dense', ascending=False
            ).astype(int)
        
        # Industry rank
        if 'industry' in df.columns:
            df['industry_rank'] = df.groupby('industry')['master_score'].rank(
                method='dense', ascending=False
            ).astype(int)
        
        return df

# ============================================
# PATTERN DETECTION ENGINE
# ============================================

class PatternDetector:
    """Detect trading patterns with metadata"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns and add to dataframe"""
        
        if df.empty:
            return df
        
        logger.info("Detecting patterns...")
        
        patterns_data = []
        pattern_metadata = {}
        
        # Define pattern detection functions
        pattern_functions = {
            "category_leader": PatternDetector._is_category_leader,
            "hidden_gem": PatternDetector._is_hidden_gem,
            "acceleration": PatternDetector._is_acceleration_pattern,
            "institutional": PatternDetector._is_institutional_accumulation,
            "vol_explosion": PatternDetector._is_volume_explosion,
            "breakout_ready": PatternDetector._is_breakout_ready,
            "market_leader": PatternDetector._is_market_leader,
            "momentum_wave": PatternDetector._is_momentum_wave,
            "liquid_leader": PatternDetector._is_liquid_leader,
            "long_strength": PatternDetector._is_long_term_strength,
            "52w_high_approach": PatternDetector._is_52w_high_approaching,
            "52w_low_bounce": PatternDetector._is_52w_low_bounce,
            "golden_cross": PatternDetector._is_golden_cross_pattern,
            "volume_surge": PatternDetector._is_volume_surge,
            "consistent_performer": PatternDetector._is_consistent_performer,
            "range_bound_breakout": PatternDetector._is_range_bound_breakout,
            "quiet_accumulation": PatternDetector._is_quiet_accumulation,
            "earnings_beat": PatternDetector._is_earnings_beat,
            "fundamental_value": PatternDetector._is_fundamental_value,
            "recovery_play": PatternDetector._is_recovery_play,
            "smart_money": PatternDetector._is_smart_money_flow,
            "stealth_mode": PatternDetector._is_stealth_mode,
            "position_builder": PatternDetector._is_position_builder,
            "harmony_wave": PatternDetector._is_harmony_wave,
            "apex_predator": PatternDetector._is_apex_predator
        }
        
        # Detect each pattern
        for pattern_name, detect_func in pattern_functions.items():
            try:
                mask, metadata = detect_func(df)
                if mask.any():
                    count = mask.sum()
                    pattern_metadata[pattern_name] = {
                        'count': count,
                        'metadata': metadata
                    }
                    
                    # Store pattern for each stock
                    for idx in df[mask].index:
                        if idx >= len(patterns_data):
                            patterns_data.extend([''] * (idx - len(patterns_data) + 1))
                        
                        if patterns_data[idx]:
                            patterns_data[idx] += f", {pattern_name}"
                        else:
                            patterns_data[idx] = pattern_name
            
            except Exception as e:
                logger.error(f"Error detecting {pattern_name}: {str(e)}")
                continue
        
        # Ensure patterns_data matches dataframe length
        if len(patterns_data) < len(df):
            patterns_data.extend([''] * (len(df) - len(patterns_data)))
        elif len(patterns_data) > len(df):
            patterns_data = patterns_data[:len(df)]
        
        # Add patterns column
        df['patterns'] = patterns_data
        
        # Store metadata in session state
        st.session_state.pattern_metadata = pattern_metadata
        
        logger.info(f"Pattern detection complete: {len(pattern_metadata)} patterns found")
        
        return df
    
    # Individual pattern detection methods
    @staticmethod
    def _is_category_leader(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Top performer in category"""
        mask = pd.Series(False, index=df.index)
        
        if 'category_rank' in df.columns and 'master_score' in df.columns:
            mask = (df['category_rank'] == 1) & (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['category_leader'])
        
        metadata = {
            'description': 'Top ranked stock in its market cap category',
            'importance': 'high',
            'reliability': 0.85
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_hidden_gem(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """High score but low visibility"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['master_score', 'volume_1d', 'ret_30d']):
            # High score, positive returns, but relatively low volume
            mask = (
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) &
                (df['ret_30d'] > 10) &
                (df['volume_1d'] < df['volume_1d'].quantile(0.5))
            )
        
        metadata = {
            'description': 'High potential stock flying under the radar',
            'importance': 'medium',
            'reliability': 0.75
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_acceleration_pattern(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Momentum accelerating"""
        mask = pd.Series(False, index=df.index)
        
        if 'acceleration_score' in df.columns:
            mask = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
        
        metadata = {
            'description': 'Price momentum is accelerating',
            'importance': 'high',
            'reliability': 0.80
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_institutional_accumulation(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Large volume with steady price increase"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['volume_score', 'ret_30d', 'ret_7d']):
            mask = (
                (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['ret_30d'] > 5) &
                (df['ret_7d'] > 0) &
                (df['ret_30d'] < 50)  # Not parabolic
            )
        
        metadata = {
            'description': 'Possible institutional accumulation',
            'importance': 'high',
            'reliability': 0.70
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_volume_explosion(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Extreme volume spike"""
        mask = pd.Series(False, index=df.index)
        
        if 'rvol' in df.columns and 'rvol_score' in df.columns:
            mask = (
                (df['rvol'] > 5) | 
                (df['rvol_score'] >= CONFIG.PATTERN_THRESHOLDS['vol_explosion'])
            )
        
        metadata = {
            'description': 'Massive volume spike detected',
            'importance': 'high',
            'reliability': 0.85,
            'risk': 'high'
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_breakout_ready(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Near resistance with volume build"""
        mask = pd.Series(False, index=df.index)
        
        if 'breakout_score' in df.columns:
            mask = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
        
        metadata = {
            'description': 'Approaching breakout conditions',
            'importance': 'high',
            'reliability': 0.75
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_market_leader(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Top overall performer"""
        mask = pd.Series(False, index=df.index)
        
        if 'rank' in df.columns and 'master_score' in df.columns:
            mask = (df['rank'] <= 10) & (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['market_leader'])
        
        metadata = {
            'description': 'Overall market leader',
            'importance': 'very_high',
            'reliability': 0.90
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_momentum_wave(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Strong trending momentum"""
        mask = pd.Series(False, index=df.index)
        
        if 'momentum_score' in df.columns:
            mask = df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']
        
        metadata = {
            'description': 'Riding strong momentum wave',
            'importance': 'high',
            'reliability': 0.80
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_liquid_leader(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """High score with excellent liquidity"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['master_score', 'volume_1d']):
            mask = (
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
                (df['volume_1d'] > df['volume_1d'].quantile(0.8))
            )
        
        metadata = {
            'description': 'High performer with excellent liquidity',
            'importance': 'high',
            'reliability': 0.85
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_long_term_strength(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Strong long-term performance"""
        mask = pd.Series(False, index=df.index)
        
        long_term_cols = ['ret_1y', 'ret_3y', 'ret_5y']
        available_cols = [col for col in long_term_cols if col in df.columns]
        
        if available_cols and 'master_score' in df.columns:
            long_term_positive = True
            for col in available_cols:
                long_term_positive = long_term_positive & (df[col] > 20)
            
            mask = long_term_positive & (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['long_strength'])
        
        metadata = {
            'description': 'Consistent long-term outperformer',
            'importance': 'medium',
            'reliability': 0.85
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_52w_high_approaching(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Near 52-week high"""
        mask = pd.Series(False, index=df.index)
        
        if 'from_high_pct' in df.columns and 'momentum_score' in df.columns:
            mask = (
                (df['from_high_pct'] > -5) &  # Within 5% of high
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['52w_high_approach'])
            )
        
        metadata = {
            'description': 'Approaching 52-week high',
            'importance': 'high',
            'reliability': 0.80
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_52w_low_bounce(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Bouncing from 52-week low"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['from_low_pct', 'ret_30d', 'volume_score']):
            mask = (
                (df['from_low_pct'] < 20) &  # Near low
                (df['ret_30d'] > 10) &       # But bouncing
                (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['52w_low_bounce'])
            )
        
        metadata = {
            'description': 'Strong bounce from 52-week low',
            'importance': 'medium',
            'reliability': 0.70,
            'risk': 'high'
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_golden_cross_pattern(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Price above key moving averages"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']):
            mask = (
                (df['price'] > df['sma_50d']) &
                (df['sma_50d'] > df['sma_200d']) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['golden_cross'])
            )
        
        metadata = {
            'description': 'Golden cross pattern detected',
            'importance': 'medium',
            'reliability': 0.75
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_volume_surge(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Sustained volume increase"""
        mask = pd.Series(False, index=df.index)
        
        vol_cols = ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        if all(col in df.columns for col in vol_cols):
            mask = (
                (df['vol_ratio_7d_90d'] > 2) &
                (df['vol_ratio_30d_90d'] > 1.5) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['volume_surge'])
            )
        
        metadata = {
            'description': 'Sustained volume surge',
            'importance': 'high',
            'reliability': 0.80
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_consistent_performer(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Steady gains across timeframes"""
        mask = pd.Series(False, index=df.index)
        
        ret_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m']
        available = [col for col in ret_cols if col in df.columns]
        
        if len(available) >= 3 and 'master_score' in df.columns:
            all_positive = df[available[0]] > 0
            for col in available[1:]:
                all_positive = all_positive & (df[col] > 0)
            
            mask = all_positive & (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['consistent_performer'])
        
        metadata = {
            'description': 'Consistent gains across all timeframes',
            'importance': 'high',
            'reliability': 0.85
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_range_bound_breakout(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Breaking out of range"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'rvol']):
            # In upper part of range with volume
            mask = (
                (df['from_high_pct'] > -10) &
                (df['from_low_pct'] > 30) &
                (df['rvol'] > 1.5) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['range_bound_breakout'])
            )
        
        metadata = {
            'description': 'Breaking out of trading range',
            'importance': 'medium',
            'reliability': 0.70
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_quiet_accumulation(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Steady rise on normal volume"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['ret_30d', 'rvol', 'momentum_score']):
            mask = (
                (df['ret_30d'] > 10) &
                (df['ret_30d'] < 30) &  # Not too fast
                (df['rvol'] < 2) &      # Normal volume
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['quiet_accumulation'])
            )
        
        metadata = {
            'description': 'Quiet accumulation phase',
            'importance': 'medium',
            'reliability': 0.65
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_earnings_beat(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Strong earnings growth"""
        mask = pd.Series(False, index=df.index)
        
        if 'eps_change_pct' in df.columns and 'master_score' in df.columns:
            mask = (
                (df['eps_change_pct'] > 20) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['earnings_beat'])
            )
        
        metadata = {
            'description': 'Strong earnings growth',
            'importance': 'high',
            'reliability': 0.80
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_fundamental_value(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Good value metrics"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['pe', 'eps_current', 'master_score']):
            mask = (
                (df['pe'] > 0) &
                (df['pe'] < 25) &
                (df['eps_current'] > 0) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['fundamental_value'])
            )
        
        metadata = {
            'description': 'Strong fundamental value',
            'importance': 'medium',
            'reliability': 0.75
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_recovery_play(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Recovering from decline"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['ret_3m', 'ret_30d', 'ret_7d', 'volume_score']):
            mask = (
                (df['ret_3m'] < -10) &   # Was down
                (df['ret_30d'] > 5) &    # Now recovering  
                (df['ret_7d'] > 0) &     # Recent positive
                (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['recovery_play'])
            )
        
        metadata = {
            'description': 'Recovery from recent decline',
            'importance': 'medium',
            'reliability': 0.70,
            'risk': 'medium'
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_smart_money_flow(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Institutional interest indicators"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['volume_score', 'position_score', 'master_score']):
            mask = (
                (df['volume_score'] >= 70) &
                (df['position_score'] >= 70) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['smart_money'])
            )
        
        metadata = {
            'description': 'Smart money flow detected',
            'importance': 'high',
            'reliability': 0.75
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_stealth_mode(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Strong technicals, low attention"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['master_score', 'volume_1d']):
            mask = (
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['stealth_mode']) &
                (df['volume_1d'] < df['volume_1d'].quantile(0.3))
            )
        
        metadata = {
            'description': 'Under-the-radar opportunity',
            'importance': 'medium',
            'reliability': 0.65
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_position_builder(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Ideal for position building"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['position_score', 'volume_score', 'momentum_score']):
            mask = (
                (df['position_score'] >= 60) &
                (df['position_score'] <= 80) &  # Not overextended
                (df['volume_score'] >= 60) &
                (df['momentum_score'] >= 60) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['position_builder'])
            )
        
        metadata = {
            'description': 'Ideal setup for position building',
            'importance': 'high',
            'reliability': 0.80
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_harmony_wave(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Perfect technical harmony"""
        mask = pd.Series(False, index=df.index)
        
        score_cols = ['position_score', 'volume_score', 'momentum_score', 
                     'acceleration_score', 'breakout_score', 'rvol_score']
        
        if all(col in df.columns for col in score_cols):
            # All scores above 70
            all_strong = df[score_cols[0]] >= 70
            for col in score_cols[1:]:
                all_strong = all_strong & (df[col] >= 70)
            
            mask = all_strong & (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['harmony_wave'])
        
        metadata = {
            'description': 'Perfect technical harmony across all metrics',
            'importance': 'very_high',
            'reliability': 0.90,
            'rarity': 'very_rare'
        }
        
        return mask, metadata
    
    @staticmethod
    def _is_apex_predator(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Ultimate pattern - best of everything"""
        mask = pd.Series(False, index=df.index)
        
        if all(col in df.columns for col in ['master_score', 'rank', 'patterns']):
            # Top 5 rank with multiple patterns
            mask = (
                (df['rank'] <= 5) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['apex_predator']) &
                (df['patterns'].str.count(',') >= 3)  # Multiple patterns
            )
        
        metadata = {
            'description': 'Apex predator - absolute market dominator',
            'importance': 'extreme',
            'reliability': 0.95,
            'rarity': 'extremely_rare'
        }
        
        return mask, metadata

# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    """Calculate advanced technical and market metrics"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics"""
        
        if df.empty:
            return df
        
        logger.info("Calculating advanced metrics...")
        
        # Volume Momentum Index (VMI)
        if all(col in df.columns for col in CONFIG.VOLUME_RATIO_COLUMNS[:3]):
            df['vmi'] = (
                df['vol_ratio_1d_90d'] * 0.5 +
                df['vol_ratio_7d_90d'] * 0.3 +
                df['vol_ratio_30d_90d'] * 0.2
            ).round(2)
        else:
            df['vmi'] = 1.0
        
        # Position Tension (0-100)
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            range_size = df['from_low_pct'] - df['from_high_pct']
            range_size = range_size.replace(0, 1)  # Avoid division by zero
            position_in_range = (0 - df['from_high_pct']) / range_size * 100
            df['position_tension'] = position_in_range.clip(0, 100).round(1)
        else:
            df['position_tension'] = 50.0
        
        # Smart Money Flow Indicator
        if all(col in df.columns for col in ['volume_score', 'position_score', 'rvol']):
            df['smart_money_flow'] = (
                (df['volume_score'] > 70).astype(int) +
                (df['position_score'] > 80).astype(int) +
                (df['rvol'] > 2).astype(int)
            )
        else:
            df['smart_money_flow'] = 0
        
        # Momentum Harmony
        df['momentum_harmony'] = 0
        
        if 'ret_1d' in df.columns:
            df['momentum_harmony'] += (df['ret_1d'] > 0).astype(int)
        
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
        
        if 'ret_3m' in df.columns:
            df['momentum_harmony'] += (df['ret_3m'] > 0).astype(int)
        
        # Wave State
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)

        # Overall Wave Strength (for filtering)
        if all(col in df.columns for col in ['momentum_score', 'acceleration_score', 'rvol_score', 'breakout_score']):
            df['overall_wave_strength'] = (
                df['momentum_score'] * 0.3 +
                df['acceleration_score'] * 0.3 +
                df['rvol_score'] * 0.2 +
                df['breakout_score'] * 0.2
            )
        else:
            df['overall_wave_strength'] = 50.0
        
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
# FILTERING ENGINE - ENHANCED WITH INDUSTRY
# ============================================

class FilterEngine:
    """Apply filters to ranked data with industry support"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all active filters"""
        
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Category filter
        if filters.get('category_filter') and 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'].isin(filters['category_filter'])]
        
        # Sector filter
        if filters.get('sector_filter') and 'sector' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sector'].isin(filters['sector_filter'])]
        
        # Industry filter (NEW)
        if filters.get('industry_filter') and 'industry' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['industry'].isin(filters['industry_filter'])]
        
        # Score filter
        if filters.get('min_score', 0) > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= filters['min_score']]
        
        # Pattern filter
        if filters.get('patterns') and 'patterns' in filtered_df.columns:
            pattern_mask = filtered_df['patterns'].str.contains('|'.join(filters['patterns']), 
                                                               case=False, na=False)
            filtered_df = filtered_df[pattern_mask]
        
        # Trend filter
        trend_filter = filters.get('trend_filter', 'All Trends')
        if trend_filter != 'All Trends' and 'ret_30d' in filtered_df.columns:
            if trend_filter == 'Strong Uptrend (>20%)':
                filtered_df = filtered_df[filtered_df['ret_30d'] > 20]
            elif trend_filter == 'Uptrend (5-20%)':
                filtered_df = filtered_df[(filtered_df['ret_30d'] >= 5) & 
                                        (filtered_df['ret_30d'] <= 20)]
            elif trend_filter == 'Sideways (-5% to 5%)':
                filtered_df = filtered_df[(filtered_df['ret_30d'] >= -5) & 
                                        (filtered_df['ret_30d'] <= 5)]
            elif trend_filter == 'Downtrend (<-5%)':
                filtered_df = filtered_df[filtered_df['ret_30d'] < -5]
        
        # Wave state filter
        if filters.get('wave_states_filter') and 'wave_state' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['wave_state'].isin(filters['wave_states_filter'])]
        
        # Wave strength range filter
        if 'wave_strength_range_slider' in filters and 'overall_wave_strength' in filtered_df.columns:
            min_strength, max_strength = filters['wave_strength_range_slider']
            filtered_df = filtered_df[
                (filtered_df['overall_wave_strength'] >= min_strength) &
                (filtered_df['overall_wave_strength'] <= max_strength)
            ]
        
        # EPS change filter
        if filters.get('min_eps_change') is not None and 'eps_change_pct' in filtered_df.columns:
            try:
                min_eps = float(filters['min_eps_change'])
                filtered_df = filtered_df[filtered_df['eps_change_pct'] >= min_eps]
            except (ValueError, TypeError):
                pass
        
        # PE filter
        if 'pe' in filtered_df.columns:
            if filters.get('min_pe') is not None:
                try:
                    min_pe = float(filters['min_pe'])
                    filtered_df = filtered_df[filtered_df['pe'] >= min_pe]
                except (ValueError, TypeError):
                    pass
            
            if filters.get('max_pe') is not None:
                try:
                    max_pe = float(filters['max_pe'])
                    filtered_df = filtered_df[filtered_df['pe'] <= max_pe]
                except (ValueError, TypeError):
                    pass
        
        # Require fundamental data
        if filters.get('require_fundamental_data', False):
            if all(col in filtered_df.columns for col in ['pe', 'eps_current']):
                filtered_df = filtered_df[
                    filtered_df['pe'].notna() & 
                    filtered_df['eps_current'].notna() &
                    (filtered_df['pe'] != 0) &
                    (filtered_df['eps_current'] != 0)
                ]
        
        # Quick filters
        quick_filter = filters.get('quick_filter')
        if quick_filter == "🔥 Hot Stocks":
            filtered_df = filtered_df[
                (filtered_df['master_score'] >= 80) & 
                (filtered_df['rvol'] > 2)
            ]
        elif quick_filter == "💎 Hidden Gems":
            if 'patterns' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['patterns'].str.contains('hidden_gem', case=False, na=False)
                ]
        elif quick_filter == "🚀 Momentum Leaders":
            filtered_df = filtered_df[filtered_df['momentum_score'] >= 85]
        elif quick_filter == "📈 Breakout Ready":
            if 'patterns' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['patterns'].str.contains('breakout', case=False, na=False)
                ]
        elif quick_filter == "🏆 Top 10 Only":
            filtered_df = filtered_df[filtered_df['rank'] <= 10]
        
        return filtered_df

# ============================================
# SESSION STATE MANAGER
# ============================================

class SessionStateManager:
    """Manage Streamlit session state"""
    
    @staticmethod
    def initialize():
        """Initialize session state variables"""
        
        # Data state
        if 'ranked_df' not in st.session_state:
            st.session_state.ranked_df = pd.DataFrame()
        
        if 'data_timestamp' not in st.session_state:
            st.session_state.data_timestamp = None
        
        if 'data_source' not in st.session_state:
            st.session_state.data_source = "sheet"
        
        if 'user_spreadsheet_id' not in st.session_state:
            st.session_state.user_spreadsheet_id = ""
        
        # Filter state
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
        
        if 'category_filter' not in st.session_state:
            st.session_state.category_filter = []
        
        if 'sector_filter' not in st.session_state:
            st.session_state.sector_filter = []
        
        if 'industry_filter' not in st.session_state:
            st.session_state.industry_filter = []
        
        if 'min_score' not in st.session_state:
            st.session_state.min_score = 0
        
        if 'patterns' not in st.session_state:
            st.session_state.patterns = []
        
        if 'trend_filter' not in st.session_state:
            st.session_state.trend_filter = 'All Trends'
        
        if 'quick_filter' not in st.session_state:
            st.session_state.quick_filter = None
        
        if 'wave_states_filter' not in st.session_state:
            st.session_state.wave_states_filter = []
        
        if 'wave_strength_range_slider' not in st.session_state:
            st.session_state.wave_strength_range_slider = (0, 100)
        
        # UI state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        
        if 'rows_per_page' not in st.session_state:
            st.session_state.rows_per_page = 20
        
        # Analysis state
        if 'selected_analysis' not in st.session_state:
            st.session_state.selected_analysis = "Overview"
        
        # Additional filter states
        additional_filters = [
            'eps_tier_filter', 'pe_tier_filter', 'price_tier_filter',
            'min_eps_change', 'min_pe', 'max_pe', 'require_fundamental_data'
        ]
        
        for filter_name in additional_filters:
            if filter_name not in st.session_state:
                if 'tier' in filter_name:
                    st.session_state[filter_name] = []
                elif filter_name == 'require_fundamental_data':
                    st.session_state[filter_name] = False
                else:
                    st.session_state[filter_name] = None
    
    @staticmethod
    def get_active_filters() -> Dict[str, Any]:
        """Get all active filters"""
        
        filters = {}
        
        # List filters
        for filter_name in ['category_filter', 'sector_filter', 'industry_filter', 
                           'patterns', 'eps_tier_filter', 'pe_tier_filter', 
                           'price_tier_filter', 'wave_states_filter']:
            if st.session_state.get(filter_name):
                filters[filter_name] = st.session_state[filter_name]
        
        # Value filters
        if st.session_state.get('min_score', 0) > 0:
            filters['min_score'] = st.session_state.min_score
        
        if st.session_state.get('trend_filter') != 'All Trends':
            filters['trend_filter'] = st.session_state.trend_filter
        
        if st.session_state.get('quick_filter'):
            filters['quick_filter'] = st.session_state.quick_filter
        
        # Numeric filters
        for filter_name in ['min_eps_change', 'min_pe', 'max_pe']:
            value = st.session_state.get(filter_name)
            if value is not None and str(value).strip():
                filters[filter_name] = value
        
        # Boolean filters
        if st.session_state.get('require_fundamental_data', False):
            filters['require_fundamental_data'] = True
        
        # Range filters
        if st.session_state.get('wave_strength_range_slider') != (0, 100):
            filters['wave_strength_range_slider'] = st.session_state.wave_strength_range_slider
        
        return filters
    
    @staticmethod
    def update_filter(filter_name: str, value: Any):
        """Update a specific filter"""
        st.session_state[filter_name] = value
        st.session_state.filters = SessionStateManager.get_active_filters()
    
    @staticmethod
    def clear_filters():
        """Clear all filters"""
        
        filter_defaults = {
            'category_filter': [],
            'sector_filter': [],
            'industry_filter': [],
            'min_score': 0,
            'patterns': [],
            'trend_filter': 'All Trends',
            'quick_filter': None,
            'eps_tier_filter': [],
            'pe_tier_filter': [],
            'price_tier_filter': [],
            'min_eps_change': None,
            'min_pe': None,
            'max_pe': None,
            'require_fundamental_data': False,
            'wave_states_filter': [],
            'wave_strength_range_slider': (0, 100)
        }
        
        for filter_name, default_value in filter_defaults.items():
            st.session_state[filter_name] = default_value
        
        st.session_state.filters = {}
        st.session_state.current_page = 0

# ============================================
# UI COMPONENTS - ENHANCED
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #2E86AB; margin-bottom: 0;'>
                🌊 Wave Detection Ultimate 3.0
            </h1>
            <p style='color: #666; font-size: 1.1em;'>
                Professional Stock Ranking System with Advanced Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric_card(label: str, value: str, delta: str = None, 
                          delta_color: str = "normal"):
        """Render a metric card"""
        
        st.metric(
            label=label,
            value=value,
            delta=delta,
            delta_color=delta_color
        )
    
    @staticmethod
    def render_score_gauge(score: float, title: str = "Score"):
        """Render a score gauge chart"""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': UIComponents._get_score_color(score)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        
        return fig
    
    @staticmethod
    def _get_score_color(score: float) -> str:
        """Get color based on score"""
        if score >= 90:
            return "#00D26A"  # Bright green
        elif score >= 80:
            return "#92D050"  # Green
        elif score >= 70:
            return "#FFC000"  # Orange
        elif score >= 60:
            return "#FF9500"  # Dark orange
        else:
            return "#FF0000"  # Red
    
    @staticmethod
    def format_number(value: float, format_type: str = "general") -> str:
        """Format numbers for display"""
        
        if pd.isna(value):
            return "N/A"
        
        if format_type == "currency":
            if abs(value) >= 1e9:
                return f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.2f}K"
            else:
                return f"${value:.2f}"
        
        elif format_type == "percentage":
            return f"{value:.1f}%"
        
        elif format_type == "volume":
            if abs(value) >= 1e9:
                return f"{value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"{value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"{value/1e3:.2f}K"
            else:
                return f"{value:.0f}"
        
        else:  # general
            if abs(value) >= 1e6:
                return f"{value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"{value/1e3:.2f}K"
            else:
                return f"{value:.2f}"

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
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e2e6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    SessionStateManager.initialize()
    
    # Render header
    UIComponents.render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Data Source")
        
        # Enhanced data source selection
        data_source = st.radio(
            "Select data source:",
            ["Google Sheets", "Upload CSV"],
            index=0 if st.session_state.data_source == "sheet" else 1,
            key="data_source_radio"
        )
        
        st.session_state.data_source = "sheet" if data_source == "Google Sheets" else "upload"
        
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your watchlist CSV file"
            )
        else:
            # Google Sheets configuration with user input
            st.markdown("### 🔗 Spreadsheet Configuration")
            
            # Input for custom spreadsheet ID
            user_spreadsheet_id = st.text_input(
                "Enter Spreadsheet ID:",
                value=st.session_state.user_spreadsheet_id,
                placeholder="e.g., 1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM",
                help="Find this in your Google Sheets URL between /d/ and /edit"
            )
            
            # Validate and store
            if user_spreadsheet_id:
                if len(user_spreadsheet_id) == CONFIG.SPREADSHEET_ID_LENGTH and user_spreadsheet_id.isalnum():
                    st.session_state.user_spreadsheet_id = user_spreadsheet_id
                    st.success("✅ Valid Spreadsheet ID")
                else:
                    st.error(f"❌ Invalid ID. Must be {CONFIG.SPREADSHEET_ID_LENGTH} alphanumeric characters")
            
            # Option to use default
            if st.button("Use Demo Spreadsheet", type="secondary"):
                default_id = CONFIG.DEFAULT_SHEET_URL.split("/d/")[1].split("/")[0]
                st.session_state.user_spreadsheet_id = default_id
                st.rerun()
            
            # GID input (optional)
            gid = st.text_input(
                "Sheet GID (optional):",
                value=CONFIG.DEFAULT_GID,
                help="Found in the URL after #gid="
            )
            
            uploaded_file = None
        
        # Load data button
        if st.button("🔄 Load/Refresh Data", type="primary", use_container_width=True):
            with st.spinner("Loading and processing data..."):
                try:
                    if st.session_state.data_source == "upload" and uploaded_file is not None:
                        df, timestamp, metadata = load_and_process_data(
                            source_type="upload",
                            file_data=uploaded_file
                        )
                    else:
                        # Use custom spreadsheet ID or default
                        spreadsheet_id = st.session_state.user_spreadsheet_id or None
                        df, timestamp, metadata = load_and_process_data(
                            source_type="sheet",
                            spreadsheet_id=spreadsheet_id,
                            gid=gid if gid else CONFIG.DEFAULT_GID
                        )
                    
                    st.session_state.ranked_df = df
                    st.session_state.data_timestamp = timestamp
                    st.session_state.metadata = metadata
                    
                    st.success(f"✅ Loaded {len(df)} stocks successfully!")
                    
                    # Show data quality
                    if 'data_quality' in metadata:
                        st.info(f"Data Quality: {metadata['data_quality']['completeness']:.1f}% complete")
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    logger.error(f"Data load error: {str(e)}", exc_info=True)
        
        # Data info
        if not st.session_state.ranked_df.empty:
            st.markdown("---")
            st.markdown("### 📈 Data Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                UIComponents.render_metric_card(
                    "Total Stocks",
                    f"{len(st.session_state.ranked_df):,}"
                )
            with col2:
                if st.session_state.data_timestamp:
                    age_minutes = (datetime.now(timezone.utc) - 
                                 st.session_state.data_timestamp).seconds // 60
                    UIComponents.render_metric_card(
                        "Data Age",
                        f"{age_minutes} min"
                    )
        
        # Filters section
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        
        # Clear filters button
        if st.button("🗑️ Clear All Filters", use_container_width=True):
            SessionStateManager.clear_filters()
            st.success("✅ All filters cleared!")
            st.rerun()
        
        # Quick filters
        st.markdown("#### ⚡ Quick Filters")
        quick_filter_options = [
            "None",
            "🔥 Hot Stocks",
            "💎 Hidden Gems", 
            "🚀 Momentum Leaders",
            "📈 Breakout Ready",
            "🏆 Top 10 Only"
        ]
        
        quick_filter = st.selectbox(
            "Apply quick filter:",
            quick_filter_options,
            index=0
        )
        
        if quick_filter != "None":
            st.session_state.quick_filter = quick_filter
        else:
            st.session_state.quick_filter = None
        
        # Category filter
        if 'category' in st.session_state.ranked_df.columns:
            categories = sorted(st.session_state.ranked_df['category'].unique())
            selected_categories = st.multiselect(
                "Categories:",
                categories,
                default=st.session_state.category_filter
            )
            st.session_state.category_filter = selected_categories
        
        # Sector filter
        if 'sector' in st.session_state.ranked_df.columns:
            sectors = sorted(st.session_state.ranked_df['sector'].unique())
            selected_sectors = st.multiselect(
                "Sectors:",
                sectors,
                default=st.session_state.sector_filter
            )
            st.session_state.sector_filter = selected_sectors
        
        # Industry filter (NEW)
        if 'industry' in st.session_state.ranked_df.columns:
            industries = sorted(st.session_state.ranked_df['industry'].unique())
            
            # Show count for each industry
            industry_counts = st.session_state.ranked_df['industry'].value_counts()
            industry_options = [f"{ind} ({industry_counts.get(ind, 0)})" for ind in industries]
            
            selected_industries_with_counts = st.multiselect(
                "Industries:",
                industry_options,
                default=[f"{ind} ({industry_counts.get(ind, 0)})" for ind in st.session_state.industry_filter]
            )
            
            # Extract industry names without counts
            st.session_state.industry_filter = [ind.split(' (')[0] for ind in selected_industries_with_counts]
        
        # Score filter
        min_score = st.slider(
            "Minimum Master Score:",
            min_value=0,
            max_value=100,
            value=st.session_state.min_score,
            step=5
        )
        st.session_state.min_score = min_score
        
        # Pattern filter
        if st.session_state.pattern_metadata:
            pattern_names = list(st.session_state.pattern_metadata.keys())
            selected_patterns = st.multiselect(
                "Patterns:",
                pattern_names,
                default=st.session_state.patterns
            )
            st.session_state.patterns = selected_patterns
        
        # Trend filter
        trend_options = [
            "All Trends",
            "Strong Uptrend (>20%)",
            "Uptrend (5-20%)",
            "Sideways (-5% to 5%)",
            "Downtrend (<-5%)"
        ]
        
        trend_filter = st.selectbox(
            "30-Day Trend:",
            trend_options,
            index=trend_options.index(st.session_state.trend_filter)
        )
        st.session_state.trend_filter = trend_filter
        
        # Advanced filters in expander
        with st.expander("⚙️ Advanced Filters"):
            # Wave state filter
            if 'wave_state' in st.session_state.ranked_df.columns:
                wave_states = sorted(st.session_state.ranked_df['wave_state'].unique())
                selected_wave_states = st.multiselect(
                    "Wave States:",
                    wave_states,
                    default=st.session_state.wave_states_filter
                )
                st.session_state.wave_states_filter = selected_wave_states
            
            # Wave strength range
            if 'overall_wave_strength' in st.session_state.ranked_df.columns:
                wave_strength_range = st.slider(
                    "Wave Strength Range:",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.wave_strength_range_slider,
                    step=5
                )
                st.session_state.wave_strength_range_slider = wave_strength_range
            
            # EPS filters
            if 'eps_change_pct' in st.session_state.ranked_df.columns:
                min_eps_change = st.number_input(
                    "Min EPS Change %:",
                    value=st.session_state.min_eps_change,
                    placeholder="e.g., 10"
                )
                st.session_state.min_eps_change = min_eps_change
            
            # PE filters
            if 'pe' in st.session_state.ranked_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    min_pe = st.number_input(
                        "Min P/E:",
                        value=st.session_state.min_pe,
                        placeholder="e.g., 0"
                    )
                    st.session_state.min_pe = min_pe
                
                with col2:
                    max_pe = st.number_input(
                        "Max P/E:",
                        value=st.session_state.max_pe,
                        placeholder="e.g., 50"
                    )
                    st.session_state.max_pe = max_pe
            
            # Require fundamental data
            require_fundamental = st.checkbox(
                "Require fundamental data",
                value=st.session_state.require_fundamental_data
            )
            st.session_state.require_fundamental_data = require_fundamental
        
        # Active filter count
        active_filters = SessionStateManager.get_active_filters()
        if active_filters:
            st.info(f"🎯 {len(active_filters)} active filter(s)")
    
    # Main content area
    if st.session_state.ranked_df.empty:
        st.info("👆 Please load data using the sidebar to begin")
        return
    
    # Apply filters
    filters = SessionStateManager.get_active_filters()
    filtered_df = FilterEngine.apply_filters(st.session_state.ranked_df, filters)
    
    # Show filtering results
    if len(filtered_df) < len(st.session_state.ranked_df):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.info(f"Showing {len(filtered_df)} of {len(st.session_state.ranked_df)} stocks after filtering")
        with col3:
            if st.button("Clear Filters"):
                SessionStateManager.clear_filters()
                st.rerun()
    
    # Main tabs
    tabs = st.tabs(["🏆 Rankings", "📊 Analysis", "🔍 Stock Details", 
                    "📈 Charts", "💾 Export", "📚 Patterns", "ℹ️ About"])
    
    # Tab 1: Rankings
    with tabs[0]:
        st.markdown("### 🏆 Master Rankings")
        
        # Display options
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            show_top_n = st.selectbox(
                "Show top:",
                CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(CONFIG.DEFAULT_TOP_N)
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Master Score", "Momentum", "Volume", "Position", "RVOL"],
                index=0
            )
        
        # Get top N stocks
        sort_column_map = {
            "Master Score": "master_score",
            "Momentum": "momentum_score",
            "Volume": "volume_score",
            "Position": "position_score",
            "RVOL": "rvol"
        }
        
        sort_col = sort_column_map[sort_by]
        if sort_col in filtered_df.columns:
            display_df = filtered_df.nlargest(show_top_n, sort_col)
        else:
            display_df = filtered_df.head(show_top_n)
        
        # Summary metrics
        if not display_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_score = display_df['master_score'].mean()
                UIComponents.render_metric_card(
                    "Avg Master Score",
                    f"{avg_score:.1f}",
                    delta=f"{avg_score - 50:.1f}" if avg_score > 50 else None,
                    delta_color="normal"
                )
            
            with col2:
                if 'ret_30d' in display_df.columns:
                    avg_return = display_df['ret_30d'].mean()
                    UIComponents.render_metric_card(
                        "Avg 30D Return",
                        f"{avg_return:.1f}%",
                        delta=f"{avg_return:.1f}%",
                        delta_color="normal"
                    )
            
            with col3:
                if 'rvol' in display_df.columns:
                    high_rvol = (display_df['rvol'] > 2).sum()
                    UIComponents.render_metric_card(
                        "High RVOL (>2x)",
                        f"{high_rvol}",
                        delta=f"{high_rvol/len(display_df)*100:.0f}%" if len(display_df) > 0 else None
                    )
            
            with col4:
                pattern_count = (display_df['patterns'] != '').sum() if 'patterns' in display_df.columns else 0
                UIComponents.render_metric_card(
                    "With Patterns",
                    f"{pattern_count}",
                    delta=f"{pattern_count/len(display_df)*100:.0f}%" if len(display_df) > 0 else None
                )
        
        # Display the rankings table
        st.markdown("---")
        
        # Select columns to display
        display_columns = ['rank', 'ticker', 'company_name', 'master_score', 
                          'price', 'ret_30d', 'rvol', 'patterns', 
                          'wave_state', 'category', 'sector', 'industry']
        
        # Only include columns that exist
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        # Format the dataframe for display
        display_df_formatted = display_df[display_columns].copy()
        
        # Format numeric columns
        if 'master_score' in display_df_formatted.columns:
            display_df_formatted['master_score'] = display_df_formatted['master_score'].round(1)
        if 'price' in display_df_formatted.columns:
            display_df_formatted['price'] = display_df_formatted['price'].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
            )
        if 'ret_30d' in display_df_formatted.columns:
            display_df_formatted['ret_30d'] = display_df_formatted['ret_30d'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
            )
        if 'rvol' in display_df_formatted.columns:
            display_df_formatted['rvol'] = display_df_formatted['rvol'].round(2)
        
        # Display with custom styling
        st.dataframe(
            display_df_formatted,
            use_container_width=True,
            height=600,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "company_name": st.column_config.TextColumn("Company", width="medium"),
                "master_score": st.column_config.NumberColumn("Score", width="small"),
                "price": st.column_config.TextColumn("Price", width="small"),
                "ret_30d": st.column_config.TextColumn("30D Return", width="small"),
                "rvol": st.column_config.NumberColumn("RVOL", width="small"),
                "patterns": st.column_config.TextColumn("Patterns", width="large"),
                "wave_state": st.column_config.TextColumn("Wave", width="medium"),
                "category": st.column_config.TextColumn("Category", width="small"),
                "sector": st.column_config.TextColumn("Sector", width="medium"),
                "industry": st.column_config.TextColumn("Industry", width="medium")
            }
        )
    
    # Tab 2: Analysis (Enhanced with Industry)
    with tabs[1]:
        st.markdown("### 📊 Market Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis:",
            ["Overview", "Sector Performance", "Industry Performance", 
             "Pattern Analysis", "Score Distribution", "Correlation Matrix"]
        )
        
        if analysis_type == "Overview":
            # Market overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📈 Market Direction")
                if 'ret_30d' in filtered_df.columns:
                    positive = (filtered_df['ret_30d'] > 0).sum()
                    negative = (filtered_df['ret_30d'] <= 0).sum()
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=['Positive', 'Negative'],
                            values=[positive, negative],
                            hole=.3,
                            marker_colors=['#00D26A', '#FF4B4B']
                        )
                    ])
                    fig.update_layout(height=300, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🌊 Wave Distribution")
                if 'wave_state' in filtered_df.columns:
                    wave_counts = filtered_df['wave_state'].value_counts()
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=wave_counts.index,
                            y=wave_counts.values,
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                        )
                    ])
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.markdown("#### 🎯 Score Ranges")
                score_bins = pd.cut(filtered_df['master_score'], 
                                  bins=[0, 50, 70, 85, 100],
                                  labels=['Low (0-50)', 'Medium (50-70)', 
                                         'High (70-85)', 'Elite (85+)'])
                score_dist = score_bins.value_counts()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=score_dist.index,
                        y=score_dist.values,
                        marker_color=['#FF4B4B', '#FFA500', '#4ECDC4', '#00D26A']
                    )
                ])
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Sector Performance":
            if 'sector' in filtered_df.columns:
                # Sector analysis
                sector_stats = filtered_df.groupby('sector').agg({
                    'master_score': ['mean', 'std', 'count'],
                    'ret_30d': 'mean',
                    'rvol': 'mean'
                }).round(2)
                
                sector_stats.columns = ['Avg Score', 'Score Std', 'Count', 'Avg 30D Return', 'Avg RVOL']
                sector_stats = sector_stats.sort_values('Avg Score', ascending=False)
                
                # Sector performance chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=sector_stats.index[:15],  # Top 15 sectors
                    y=sector_stats['Avg Score'][:15],
                    name='Avg Score',
                    marker_color='lightblue',
                    yaxis='y'
                ))
                
                fig.add_trace(go.Scatter(
                    x=sector_stats.index[:15],
                    y=sector_stats['Avg 30D Return'][:15],
                    name='Avg 30D Return',
                    line=dict(color='red', width=3),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="Top 15 Sectors by Average Score",
                    xaxis_title="Sector",
                    yaxis=dict(title="Average Score", side="left"),
                    yaxis2=dict(title="Average 30D Return %", overlaying="y", side="right"),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sector details table
                st.markdown("#### Sector Statistics")
                st.dataframe(
                    sector_stats.style.background_gradient(subset=['Avg Score', 'Avg 30D Return'], cmap='RdYlGn'),
                    use_container_width=True
                )
        
        elif analysis_type == "Industry Performance":
            if 'industry' in filtered_df.columns:
                # Industry analysis (NEW)
                st.markdown("#### 🏭 Industry Performance Analysis")
                
                # Calculate industry statistics
                industry_stats = filtered_df.groupby('industry').agg({
                    'master_score': ['mean', 'std', 'count'],
                    'ret_30d': 'mean',
                    'rvol': 'mean',
                    'volume_1d': 'sum'
                }).round(2)
                
                industry_stats.columns = ['Avg Score', 'Score Std', 'Count', 
                                        'Avg 30D Return', 'Avg RVOL', 'Total Volume']
                
                # Filter industries with minimum stocks
                min_stocks = st.slider("Minimum stocks per industry:", 1, 10, 3)
                industry_stats = industry_stats[industry_stats['Count'] >= min_stocks]
                industry_stats = industry_stats.sort_values('Avg Score', ascending=False)
                
                # Top industries chart
                top_n = st.slider("Show top N industries:", 10, 50, 20)
                top_industries = industry_stats.head(top_n)
                
                # Create visualization
                fig = go.Figure()
                
                # Bar chart for average score
                fig.add_trace(go.Bar(
                    x=top_industries['Avg Score'],
                    y=top_industries.index,
                    orientation='h',
                    name='Avg Score',
                    marker=dict(
                        color=top_industries['Avg Score'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Score")
                    ),
                    text=top_industries['Avg Score'].round(1),
                    textposition='inside'
                ))
                
                fig.update_layout(
                    title=f"Top {top_n} Industries by Average Score (Min {min_stocks} stocks)",
                    xaxis_title="Average Master Score",
                    yaxis_title="Industry",
                    height=max(500, top_n * 25),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Industry comparison metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Returns vs Score scatter
                    fig_scatter = go.Figure()
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=industry_stats['Avg Score'],
                        y=industry_stats['Avg 30D Return'],
                        mode='markers',
                        marker=dict(
                            size=industry_stats['Count'] * 2,
                            color=industry_stats['Avg RVOL'],
                            colorscale='Portland',
                            showscale=True,
                            colorbar=dict(title="Avg RVOL")
                        ),
                        text=industry_stats.index,
                        hovertemplate='<b>%{text}</b><br>' +
                                     'Score: %{x:.1f}<br>' +
                                     'Return: %{y:.1f}%<br>' +
                                     '<extra></extra>'
                    ))
                    
                    fig_scatter.update_layout(
                        title="Industry Score vs 30D Return",
                        xaxis_title="Average Score",
                        yaxis_title="Average 30D Return %",
                        height=400
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Industry concentration
                    fig_conc = go.Figure(data=[
                        go.Pie(
                            labels=industry_stats.head(10).index,
                            values=industry_stats.head(10)['Count'],
                            hole=.3,
                            textinfo='label+percent'
                        )
                    ])
                    
                    fig_conc.update_layout(
                        title="Top 10 Industries by Stock Count",
                        height=400
                    )
                    
                    st.plotly_chart(fig_conc, use_container_width=True)
                
                # Industry details table
                st.markdown("#### Detailed Industry Statistics")
                
                # Format the display
                industry_display = industry_stats.copy()
                industry_display['Total Volume'] = industry_display['Total Volume'].apply(
                    lambda x: UIComponents.format_number(x, 'volume')
                )
                
                st.dataframe(
                    industry_display.style.background_gradient(
                        subset=['Avg Score', 'Avg 30D Return'], 
                        cmap='RdYlGn'
                    ).format({
                        'Avg Score': '{:.1f}',
                        'Score Std': '{:.1f}',
                        'Avg 30D Return': '{:.1f}%',
                        'Avg RVOL': '{:.2f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Industry insights
                st.markdown("#### 💡 Industry Insights")
                
                if len(industry_stats) > 0:
                    best_industry = industry_stats.index[0]
                    best_score = industry_stats.iloc[0]['Avg Score']
                    best_return = industry_stats.iloc[0]['Avg 30D Return']
                    
                    # Find most improved industry
                    positive_returns = industry_stats[industry_stats['Avg 30D Return'] > 0]
                    if len(positive_returns) > 0:
                        most_improved = positive_returns['Avg 30D Return'].idxmax()
                        improved_return = positive_returns.loc[most_improved, 'Avg 30D Return']
                        
                        insights_col1, insights_col2 = st.columns(2)
                        
                        with insights_col1:
                            st.info(f"""
                            **🏆 Top Performing Industry:**  
                            {best_industry}  
                            - Average Score: {best_score:.1f}  
                            - 30D Return: {best_return:.1f}%
                            """)
                        
                        with insights_col2:
                            st.success(f"""
                            **📈 Most Improved Industry:**  
                            {most_improved}  
                            - 30D Return: {improved_return:.1f}%  
                            - Stocks: {industry_stats.loc[most_improved, 'Count']:.0f}
                            """)
        
        elif analysis_type == "Pattern Analysis":
            if st.session_state.pattern_metadata:
                # Pattern frequency
                pattern_counts = {}
                for pattern, data in st.session_state.pattern_metadata.items():
                    pattern_counts[pattern] = data['count']
                
                pattern_df = pd.DataFrame.from_dict(pattern_counts, orient='index', 
                                                  columns=['Count'])
                pattern_df = pattern_df.sort_values('Count', ascending=False)
                
                # Pattern distribution chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=pattern_df.index[:15],
                        y=pattern_df['Count'][:15],
                        marker_color=px.colors.qualitative.Set3[:15]
                    )
                ])
                
                fig.update_layout(
                    title="Top 15 Pattern Distribution",
                    xaxis_title="Pattern",
                    yaxis_title="Count",
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Pattern details
                st.markdown("#### Pattern Details")
                
                pattern_details = []
                for pattern, data in st.session_state.pattern_metadata.items():
                    metadata = data['metadata']
                    pattern_details.append({
                        'Pattern': pattern,
                        'Count': data['count'],
                        'Importance': metadata.get('importance', 'medium'),
                        'Reliability': f"{metadata.get('reliability', 0.7) * 100:.0f}%",
                        'Description': metadata.get('description', '')
                    })
                
                pattern_details_df = pd.DataFrame(pattern_details)
                pattern_details_df = pattern_details_df.sort_values('Count', ascending=False)
                
                st.dataframe(
                    pattern_details_df,
                    use_container_width=True,
                    height=400
                )
        
        elif analysis_type == "Score Distribution":
            # Score distribution analysis
            fig = go.Figure()
            
            # Histogram of master scores
            fig.add_trace(go.Histogram(
                x=filtered_df['master_score'],
                nbinsx=20,
                name='Master Score',
                marker_color='lightblue'
            ))
            
            # Add component score distributions
            score_cols = ['position_score', 'volume_score', 'momentum_score', 
                         'acceleration_score', 'breakout_score', 'rvol_score']
            
            for col in score_cols:
                if col in filtered_df.columns:
                    fig.add_trace(go.Box(
                        y=filtered_df[col],
                        name=col.replace('_score', '').title(),
                        boxpoints='outliers'
                    ))
            
            fig.update_layout(
                title="Score Distributions",
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Correlation Matrix":
            # Select numeric columns for correlation
            numeric_cols = ['master_score', 'position_score', 'volume_score', 
                           'momentum_score', 'acceleration_score', 'breakout_score', 
                           'rvol_score', 'ret_1d', 'ret_7d', 'ret_30d', 'rvol']
            
            available_cols = [col for col in numeric_cols if col in filtered_df.columns]
            
            if len(available_cols) > 2:
                corr_matrix = filtered_df[available_cols].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title="Correlation Matrix",
                    height=600,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Stock Details
    with tabs[2]:
        st.markdown("### 🔍 Stock Details")
        
        # Stock selector
        if not filtered_df.empty:
            ticker_list = filtered_df['ticker'].tolist()
            selected_ticker = st.selectbox(
                "Select a stock:",
                ticker_list,
                index=0
            )
            
            # Get stock data
            stock_data = filtered_df[filtered_df['ticker'] == selected_ticker].iloc[0]
            
            # Layout
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"## {selected_ticker}")
                if 'company_name' in stock_data:
                    st.markdown(f"### {stock_data['company_name']}")
                
                # Categories
                categories = []
                if 'category' in stock_data:
                    categories.append(f"**Category:** {stock_data['category']}")
                if 'sector' in stock_data:
                    categories.append(f"**Sector:** {stock_data['sector']}")
                if 'industry' in stock_data:
                    categories.append(f"**Industry:** {stock_data['industry']}")
                
                if categories:
                    st.markdown(" | ".join(categories))
            
            with col2:
                # Master score gauge
                if 'master_score' in stock_data:
                    fig = UIComponents.render_score_gauge(
                        stock_data['master_score'],
                        "Master Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # Key metrics
                st.markdown("#### Key Metrics")
                if 'price' in stock_data:
                    st.metric("Price", f"${stock_data['price']:.2f}")
                if 'ret_30d' in stock_data:
                    st.metric("30D Return", f"{stock_data['ret_30d']:.1f}%",
                             delta=f"{stock_data['ret_30d']:.1f}%")
                if 'rvol' in stock_data:
                    st.metric("RVOL", f"{stock_data['rvol']:.2f}x")
            
            # Component scores
            st.markdown("---")
            st.markdown("#### 📊 Component Scores")
            
            score_cols = ['position_score', 'volume_score', 'momentum_score',
                         'acceleration_score', 'breakout_score', 'rvol_score']
            
            available_scores = [(col, stock_data[col]) for col in score_cols if col in stock_data]
            
            if available_scores:
                # Create radar chart
                categories = [col.replace('_score', '').title() for col, _ in available_scores]
                values = [score for _, score in available_scores]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_ticker,
                    line_color='blue'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Price performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Price Performance")
                
                perf_metrics = []
                for period in ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']:
                    if period in stock_data and pd.notna(stock_data[period]):
                        period_label = period.replace('ret_', '').upper()
                        perf_metrics.append({
                            'Period': period_label,
                            'Return': f"{stock_data[period]:.1f}%"
                        })
                
                if perf_metrics:
                    perf_df = pd.DataFrame(perf_metrics)
                    st.dataframe(perf_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 📊 Trading Data")
                
                trading_metrics = []
                
                # Volume data
                if 'volume_1d' in stock_data:
                    trading_metrics.append({
                        'Metric': 'Daily Volume',
                        'Value': UIComponents.format_number(stock_data['volume_1d'], 'volume')
                    })
                
                # 52-week range
                if all(col in stock_data for col in ['low_52w', 'high_52w']):
                    trading_metrics.append({
                        'Metric': '52W Range',
                        'Value': f"${stock_data['low_52w']:.2f} - ${stock_data['high_52w']:.2f}"
                    })
                
                # Position in range
                if all(col in stock_data for col in ['from_low_pct', 'from_high_pct']):
                    trading_metrics.append({
                        'Metric': 'From 52W Low',
                        'Value': f"{stock_data['from_low_pct']:.1f}%"
                    })
                    trading_metrics.append({
                        'Metric': 'From 52W High',
                        'Value': f"{stock_data['from_high_pct']:.1f}%"
                    })
                
                if trading_metrics:
                    trading_df = pd.DataFrame(trading_metrics)
                    st.dataframe(trading_df, use_container_width=True, hide_index=True)
            
            # Patterns and advanced metrics
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Patterns Detected")
                if 'patterns' in stock_data and stock_data['patterns']:
                    patterns = stock_data['patterns'].split(', ')
                    for pattern in patterns:
                        # Get pattern metadata if available
                        if hasattr(st.session_state, 'pattern_metadata') and pattern in st.session_state.pattern_metadata:
                            metadata = st.session_state.pattern_metadata[pattern]['metadata']
                            importance = metadata.get('importance', 'medium')
                            
                            # Color code by importance
                            if importance == 'very_high' or importance == 'extreme':
                                st.error(f"🔴 {pattern}")
                            elif importance == 'high':
                                st.warning(f"🟡 {pattern}")
                            else:
                                st.info(f"🔵 {pattern}")
                            
                            # Show description in expander
                            with st.expander(f"About {pattern}"):
                                st.write(metadata.get('description', 'No description available'))
                                st.write(f"**Reliability:** {metadata.get('reliability', 0.7) * 100:.0f}%")
                        else:
                            st.info(f"• {pattern}")
                else:
                    st.info("No patterns detected")
            
            with col2:
                st.markdown("#### 🌊 Advanced Metrics")
                
                adv_metrics = []
                
                if 'wave_state' in stock_data:
                    adv_metrics.append({
                        'Metric': 'Wave State',
                        'Value': stock_data['wave_state']
                    })
                
                if 'vmi' in stock_data:
                    adv_metrics.append({
                        'Metric': 'Volume Momentum Index',
                        'Value': f"{stock_data['vmi']:.2f}"
                    })
                
                if 'position_tension' in stock_data:
                    adv_metrics.append({
                        'Metric': 'Position Tension',
                        'Value': f"{stock_data['position_tension']:.1f}"
                    })
                
                if 'momentum_harmony' in stock_data:
                    adv_metrics.append({
                        'Metric': 'Momentum Harmony',
                        'Value': f"{stock_data['momentum_harmony']}/4"
                    })
                
                if 'smart_money_flow' in stock_data:
                    adv_metrics.append({
                        'Metric': 'Smart Money Signals',
                        'Value': f"{stock_data['smart_money_flow']}/3"
                    })
                
                if adv_metrics:
                    adv_df = pd.DataFrame(adv_metrics)
                    st.dataframe(adv_df, use_container_width=True, hide_index=True)
            
            # Fundamental data
            if any(col in stock_data for col in ['pe', 'eps_current', 'eps_change_pct', 'market_cap']):
                st.markdown("---")
                st.markdown("#### 💰 Fundamental Data")
                
                fund_cols = st.columns(4)
                
                with fund_cols[0]:
                    if 'pe' in stock_data and pd.notna(stock_data['pe']):
                        st.metric("P/E Ratio", f"{stock_data['pe']:.1f}")
                
                with fund_cols[1]:
                    if 'eps_current' in stock_data and pd.notna(stock_data['eps_current']):
                        st.metric("EPS", f"${stock_data['eps_current']:.2f}")
                
                with fund_cols[2]:
                    if 'eps_change_pct' in stock_data and pd.notna(stock_data['eps_change_pct']):
                        st.metric("EPS Change", f"{stock_data['eps_change_pct']:.1f}%",
                                 delta=f"{stock_data['eps_change_pct']:.1f}%")
                
                with fund_cols[3]:
                    if 'market_cap' in stock_data and pd.notna(stock_data['market_cap']):
                        st.metric("Market Cap", UIComponents.format_number(stock_data['market_cap'], 'currency'))
    
    # Tab 4: Charts
    with tabs[3]:
        st.markdown("### 📈 Interactive Charts")
        
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Score vs Return Scatter", "Volume Analysis", "Sector Heatmap", 
             "Pattern Co-occurrence", "Time Series Comparison"]
        )
        
        if chart_type == "Score vs Return Scatter":
            # Scatter plot of master score vs 30d return
            if 'ret_30d' in filtered_df.columns:
                fig = go.Figure()
                
                # Color by category if available
                if 'category' in filtered_df.columns:
                    categories = filtered_df['category'].unique()
                    colors = px.colors.qualitative.Set3[:len(categories)]
                    
                    for i, cat in enumerate(categories):
                        cat_data = filtered_df[filtered_df['category'] == cat]
                        fig.add_trace(go.Scatter(
                            x=cat_data['master_score'],
                            y=cat_data['ret_30d'],
                            mode='markers',
                            name=cat,
                            marker=dict(
                                size=10,
                                color=colors[i % len(colors)],
                                line=dict(width=1, color='white')
                            ),
                            text=cat_data['ticker'],
                            hovertemplate='<b>%{text}</b><br>' +
                                         'Score: %{x:.1f}<br>' +
                                         'Return: %{y:.1f}%<br>' +
                                         '<extra></extra>'
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['master_score'],
                        y=filtered_df['ret_30d'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=filtered_df['master_score'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=filtered_df['ticker'],
                        hovertemplate='<b>%{text}</b><br>' +
                                     'Score: %{x:.1f}<br>' +
                                     'Return: %{y:.1f}%<br>' +
                                     '<extra></extra>'
                    ))
                
                # Add trend line
                z = np.polyfit(filtered_df['master_score'], filtered_df['ret_30d'], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=filtered_df['master_score'],
                    y=p(filtered_df['master_score']),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="Master Score vs 30-Day Return",
                    xaxis_title="Master Score",
                    yaxis_title="30-Day Return %",
                    height=600,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation
                correlation = filtered_df['master_score'].corr(filtered_df['ret_30d'])
                st.info(f"Correlation between Master Score and 30D Return: {correlation:.3f}")
        
        elif chart_type == "Volume Analysis":
            # Volume analysis charts
            col1, col2 = st.columns(2)
            
            with col1:
                # RVOL distribution
                if 'rvol' in filtered_df.columns:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=filtered_df['rvol'],
                        nbinsx=30,
                        name='RVOL Distribution',
                        marker_color='lightblue'
                    ))
                    
                    fig.add_vline(x=2, line_dash="dash", line_color="red",
                                 annotation_text="2x Average")
                    
                    fig.update_layout(
                        title="Relative Volume Distribution",
                        xaxis_title="RVOL",
                        yaxis_title="Count",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Volume score vs RVOL
                if all(col in filtered_df.columns for col in ['volume_score', 'rvol']):
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=filtered_df['rvol'],
                        y=filtered_df['volume_score'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=filtered_df['master_score'],
                            colorscale='Portland',
                            showscale=True,
                            colorbar=dict(title="Master Score")
                        ),
                        text=filtered_df['ticker'],
                        hovertemplate='<b>%{text}</b><br>' +
                                     'RVOL: %{x:.2f}<br>' +
                                     'Volume Score: %{y:.1f}<br>' +
                                     '<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Volume Score vs RVOL",
                        xaxis_title="RVOL",
                        yaxis_title="Volume Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Sector Heatmap":
            # Sector performance heatmap
            if 'sector' in filtered_df.columns:
                # Create pivot table
                metrics = ['master_score', 'ret_30d', 'rvol', 'momentum_score']
                available_metrics = [m for m in metrics if m in filtered_df.columns]
                
                if available_metrics:
                    sector_pivot = filtered_df.groupby('sector')[available_metrics].mean()
                    
                    # Normalize to 0-1 scale for heatmap
                    sector_pivot_norm = (sector_pivot - sector_pivot.min()) / (sector_pivot.max() - sector_pivot.min())
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=sector_pivot_norm.T.values,
                        x=sector_pivot_norm.index,
                        y=[col.replace('_', ' ').title() for col in sector_pivot_norm.columns],
                        colorscale='RdYlGn',
                        text=sector_pivot.T.values.round(1),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    
                    fig.update_layout(
                        title="Sector Performance Heatmap",
                        height=400,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Pattern Co-occurrence":
            # Pattern co-occurrence matrix
            if 'patterns' in filtered_df.columns:
                # Extract all unique patterns
                all_patterns = set()
                for patterns in filtered_df['patterns'].dropna():
                    if patterns:
                        all_patterns.update([p.strip() for p in patterns.split(',')])
                
                if len(all_patterns) > 1:
                    # Create co-occurrence matrix
                    pattern_list = sorted(list(all_patterns))[:15]  # Top 15 patterns
                    cooc_matrix = np.zeros((len(pattern_list), len(pattern_list)))
                    
                    for patterns in filtered_df['patterns'].dropna():
                        if patterns:
                            stock_patterns = [p.strip() for p in patterns.split(',')]
                            for i, p1 in enumerate(pattern_list):
                                if p1 in stock_patterns:
                                    for j, p2 in enumerate(pattern_list):
                                        if p2 in stock_patterns:
                                            cooc_matrix[i, j] += 1
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cooc_matrix,
                        x=pattern_list,
                        y=pattern_list,
                        colorscale='Blues',
                        text=cooc_matrix.astype(int),
                        texttemplate='%{text}',
                        textfont={"size": 10}
                    ))
                    
                    fig.update_layout(
                        title="Pattern Co-occurrence Matrix",
                        height=600,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough patterns for co-occurrence analysis")
        
        elif chart_type == "Time Series Comparison":
            # Compare returns across different time periods
            return_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']
            available_returns = [col for col in return_cols if col in filtered_df.columns]
            
            if len(available_returns) >= 2:
                # Get top performers
                top_n = st.slider("Number of stocks to compare:", 5, 20, 10)
                top_stocks = filtered_df.nlargest(top_n, 'master_score')
                
                # Prepare data for plotting
                return_data = []
                for _, stock in top_stocks.iterrows():
                    for col in available_returns:
                        if pd.notna(stock[col]):
                            return_data.append({
                                'Ticker': stock['ticker'],
                                'Period': col.replace('ret_', '').upper(),
                                'Return': stock[col]
                            })
                
                return_df = pd.DataFrame(return_data)
                
                # Create grouped bar chart
                fig = px.bar(
                    return_df,
                    x='Period',
                    y='Return',
                    color='Ticker',
                    barmode='group',
                    title=f"Return Comparison - Top {top_n} Stocks",
                    labels={'Return': 'Return %'},
                    height=600
                )
                
                fig.update_layout(xaxis_tickangle=0)
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Export
    with tabs[4]:
        st.markdown("### 💾 Export Data")
        
        # Export options
        export_format = st.selectbox(
            "Select export format:",
            ["CSV", "Excel", "JSON"]
        )
        
        # Column selection
        st.markdown("#### Select columns to export:")
        
        # Predefined column groups
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            include_basic = st.checkbox("Basic Info", value=True)
            include_scores = st.checkbox("All Scores", value=True)
        
        with col2:
            include_returns = st.checkbox("Returns", value=True)
            include_volume = st.checkbox("Volume Data", value=True)
        
        with col3:
            include_patterns = st.checkbox("Patterns", value=True)
            include_advanced = st.checkbox("Advanced Metrics", value=True)
        
        with col4:
            include_fundamental = st.checkbox("Fundamentals", value=False)
            include_all = st.checkbox("All Columns", value=False)
        
        # Build column list
        if include_all:
            export_columns = filtered_df.columns.tolist()
        else:
            export_columns = []
            
            if include_basic:
                basic_cols = ['rank', 'ticker', 'company_name', 'price', 
                             'category', 'sector', 'industry']
                export_columns.extend([col for col in basic_cols if col in filtered_df.columns])
            
            if include_scores:
                score_cols = ['master_score', 'position_score', 'volume_score',
                             'momentum_score', 'acceleration_score', 'breakout_score',
                             'rvol_score']
                export_columns.extend([col for col in score_cols if col in filtered_df.columns])
            
            if include_returns:
                return_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']
                export_columns.extend([col for col in return_cols if col in filtered_df.columns])
            
            if include_volume:
                vol_cols = ['volume_1d', 'rvol'] + CONFIG.VOLUME_RATIO_COLUMNS
                export_columns.extend([col for col in vol_cols if col in filtered_df.columns])
            
            if include_patterns:
                pattern_cols = ['patterns', 'wave_state']
                export_columns.extend([col for col in pattern_cols if col in filtered_df.columns])
            
            if include_advanced:
                adv_cols = ['vmi', 'position_tension', 'momentum_harmony', 
                           'smart_money_flow', 'overall_wave_strength']
                export_columns.extend([col for col in adv_cols if col in filtered_df.columns])
            
            if include_fundamental:
                fund_cols = ['pe', 'eps_current', 'eps_change_pct', 'market_cap']
                export_columns.extend([col for col in fund_cols if col in filtered_df.columns])
            
            # Remove duplicates while preserving order
            export_columns = list(dict.fromkeys(export_columns))
        
        # Prepare export data
        export_df = filtered_df[export_columns].copy()
        
        # Show preview
        st.markdown("#### Preview (first 10 rows):")
        st.dataframe(export_df.head(10), use_container_width=True)
        
        # Export statistics
        st.markdown("#### Export Summary:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(export_df))
        with col2:
            st.metric("Total Columns", len(export_columns))
        with col3:
            st.metric("File Size (est.)", f"{len(export_df) * len(export_columns) * 20 / 1024:.1f} KB")
        
        # Generate export file
        st.markdown("---")
        
        if export_format == "CSV":
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"wave_detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif export_format == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, sheet_name='Wave Detection', index=False)
                
                # Get workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Wave Detection']
                
                # Add formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BD',
                    'border': 1
                })
                
                # Write headers with formatting
                for col_num, value in enumerate(export_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Auto-fit columns
                for i, col in enumerate(export_df.columns):
                    column_width = max(export_df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, min(column_width, 50))
            
            output.seek(0)
            
            st.download_button(
                label="📥 Download Excel",
                data=output,
                file_name=f"wave_detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        elif export_format == "JSON":
            json_str = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"wave_detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Tab 6: Patterns
    with tabs[5]:
        st.markdown("### 📚 Pattern Library")
        
        if hasattr(st.session_state, 'pattern_metadata') and st.session_state.pattern_metadata:
            # Pattern search
            search_term = st.text_input("Search patterns:", placeholder="e.g., momentum, breakout")
            
            # Filter patterns
            filtered_patterns = {}
            for pattern, data in st.session_state.pattern_metadata.items():
                if not search_term or search_term.lower() in pattern.lower() or \
                   search_term.lower() in data['metadata'].get('description', '').lower():
                    filtered_patterns[pattern] = data
            
            # Display patterns in grid
            cols_per_row = 3
            pattern_items = list(filtered_patterns.items())
            
            for i in range(0, len(pattern_items), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j in range(cols_per_row):
                    if i + j < len(pattern_items):
                        pattern_name, pattern_data = pattern_items[i + j]
                        metadata = pattern_data['metadata']
                        
                        with cols[j]:
                            # Pattern card
                            with st.container():
                                # Header with importance color
                                importance = metadata.get('importance', 'medium')
                                if importance in ['very_high', 'extreme']:
                                    st.error(f"### {pattern_name}")
                                elif importance == 'high':
                                    st.warning(f"### {pattern_name}")
                                else:
                                    st.info(f"### {pattern_name}")
                                
                                # Description
                                st.write(metadata.get('description', 'No description'))
                                
                                # Metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    reliability = metadata.get('reliability', 0.7)
                                    st.metric("Reliability", f"{reliability * 100:.0f}%")
                                with col2:
                                    st.metric("Count", pattern_data['count'])
                                
                                # Additional info
                                if 'risk' in metadata:
                                    st.write(f"**Risk:** {metadata['risk']}")
                                if 'rarity' in metadata:
                                    st.write(f"**Rarity:** {metadata['rarity']}")
                                
                                # Stocks with this pattern
                                with st.expander("View stocks"):
                                    pattern_stocks = filtered_df[
                                        filtered_df['patterns'].str.contains(
                                            pattern_name, case=False, na=False
                                        )
                                    ][['ticker', 'master_score', 'ret_30d']].head(10)
                                    
                                    if not pattern_stocks.empty:
                                        st.dataframe(pattern_stocks, use_container_width=True)
            
            # Pattern statistics
            st.markdown("---")
            st.markdown("### 📊 Pattern Statistics")
            
            # Most common patterns
            pattern_counts = pd.DataFrame.from_dict(
                {k: v['count'] for k, v in filtered_patterns.items()},
                orient='index',
                columns=['Count']
            ).sort_values('Count', ascending=False)
            
            fig = px.bar(
                x=pattern_counts.index[:10],
                y=pattern_counts['Count'][:10],
                title="Top 10 Most Common Patterns",
                labels={'x': 'Pattern', 'y': 'Count'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pattern data available. Please load data first.")
    
    # Tab 7: About
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
            The professional-grade stock ranking system designed to identify high-potential 
            stocks before they make major moves. This enhanced version includes comprehensive
            industry analysis and filtering capabilities.
            
            #### 🎯 Key Features
            
            - **Master Score 3.0** - Advanced composite scoring algorithm
            - **25 Pattern Detection** - Comprehensive pattern recognition
            - **Industry Analysis** - Deep dive into industry performance
            - **Real-time Processing** - Sub-2 second analysis
            - **Smart Filtering** - Advanced multi-criteria filtering
            - **Professional Export** - Multiple format support
            
            #### 📊 How It Works
            
            1. **Data Input** - Load from Google Sheets or CSV
            2. **Score Calculation** - 6 component scores weighted into Master Score
            3. **Pattern Detection** - 25 patterns with metadata
            4. **Ranking** - Multi-level ranking (overall, sector, industry)
            5. **Analysis** - Advanced metrics and visualizations
            
            #### 🔧 Component Scores
            
            - **Position Score (30%)** - Position within 52-week range
            - **Volume Score (25%)** - Volume momentum analysis
            - **Momentum Score (15%)** - Price momentum across timeframes
            - **Acceleration Score (10%)** - Momentum acceleration
            - **Breakout Score (10%)** - Breakout probability
            - **RVOL Score (10%)** - Relative volume analysis
            
            #### 📈 Pattern Categories
            
            - **Momentum Patterns** - Trend and momentum based
            - **Volume Patterns** - Volume anomaly detection
            - **Technical Patterns** - Classic technical setups
            - **Fundamental Patterns** - Value and growth signals
            - **Composite Patterns** - Multi-factor patterns
            """)
        
        with col2:
            st.markdown("""
            #### 🚀 Quick Start
            
            1. **Load Data**
               - Enter your Google Sheets ID
               - Or upload a CSV file
            
            2. **Apply Filters**
               - Use quick filters for common scenarios
               - Apply advanced filters for precision
            
            3. **Analyze Results**
               - Review top-ranked stocks
               - Check pattern alerts
               - Analyze by sector/industry
            
            4. **Export Data**
               - Choose your columns
               - Select format (CSV/Excel/JSON)
               - Download results
            
            #### 💡 Pro Tips
            
            - Focus on stocks with multiple patterns
            - Check industry performance trends
            - Monitor RVOL > 2 for activity
            - Use wave states for timing
            - Combine filters for precision
            
            #### ⚡ Performance
            
            - Handles 2000+ stocks smoothly
            - 1-hour intelligent caching
            - Optimized for cloud deployment
            - Mobile responsive design
            
            #### 🔒 Data Requirements
            
            **Critical Columns:**
            - ticker
            - price
            - volume_1d
            
            **Important Columns:**
            - ret_30d
            - from_low_pct
            - from_high_pct
            - volume ratios
            - industry (for industry analysis)
            """)
        
        # Version info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Version", "3.0.8-FINAL")
        with col2:
            st.metric("Last Updated", "August 2025")
        with col3:
            st.metric("Status", "Production Ready")
        
        # Performance metrics
        if hasattr(st.session_state, 'metadata') and st.session_state.metadata:
            st.markdown("---")
            st.markdown("#### 🎛️ Current Session Stats")
            
            metadata = st.session_state.metadata
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'processing_time' in metadata:
                    st.metric("Processing Time", f"{metadata['processing_time']:.2f}s")
            
            with col2:
                if 'data_quality' in metadata:
                    st.metric("Data Quality", f"{metadata['data_quality']['completeness']:.1f}%")
            
            with col3:
                st.metric("Patterns Detected", len(st.session_state.pattern_metadata) if hasattr(st.session_state, 'pattern_metadata') else 0)
            
            with col4:
                memory = PerformanceMonitor.memory_usage()
                st.metric("Memory Usage", f"{memory['rss_mb']:.1f} MB")

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
