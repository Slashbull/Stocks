"""
Wave Detection Ultimate 3.0 - FINAL ENHANCED VERSION
===================================================
Professional Stock Ranking System with Advanced Analytics
All bugs fixed, optimized for Streamlit Community Cloud
Enhanced with all valuable features from previous versions

Version: 3.0.7-FINAL-COMPLETE
Last Updated: December 2024
Status: PRODUCTION READY - Feature Complete
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import time
from io import BytesIO
import warnings
import json

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# LOGGING CONFIGURATION
# ============================================

# Configure logging for production
log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights"""
    
    # Data source - HARDCODED for production
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM/edit?usp=sharing"
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings optimized for Streamlit Community Cloud
    CACHE_TTL: int = 3600  # 1 hour for better performance
    
    # Master Score 3.0 weights (total = 100%)
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Thresholds for patterns
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
    })
    
    # Tier definitions
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {
            "Loss": (-float('inf'), 0),
            "0-5": (0.01, 5),
            "5-10": (5.01, 10),
            "10-20": (10.01, 20),
            "20-50": (20.01, 50),
            "50-100": (50.01, 100),
            "100+": (100.01, float('inf'))
        },
        "pe": {
            "Negative/NA": (-float('inf'), 0),
            "0-10": (0.01, 10),
            "10-15": (10.01, 15),
            "15-20": (15.01, 20),
            "20-30": (20.01, 30),
            "30-50": (30.01, 50),
            "50+": (50.01, float('inf'))
        },
        "price": {
            "0-100": (0, 100),
            "100-250": (100.01, 250),
            "250-500": (250.01, 500),
            "500-1000": (500.01, 1000),
            "1000-2500": (1000.01, 2500),
            "2500-5000": (2500.01, 5000),
            "5000+": (5000.01, float('inf'))
        }
    })
    
    # Performance thresholds - FIXED: Allow extreme RVOL values
    RVOL_MAX_THRESHOLD: float = 1000000.0  # Allow up to 1M RVOL
    MIN_VALID_PRICE: float = 0.01  # Minimum valid price
    
    # Quick action thresholds
    TOP_GAINER_MOMENTUM: float = 80
    VOLUME_SURGE_RVOL: float = 3.0
    BREAKOUT_READY_SCORE: float = 80
    
    # Data quality thresholds
    MIN_DATA_COMPLETENESS: float = 0.8  # 80% data completeness required
    STALE_DATA_HOURS: int = 24  # Data older than 24 hours is stale

# Global configuration instance
CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

def timer(func):
    """Performance timing decorator with logging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if elapsed > 1.0:
                logger.warning(f"{func.__name__} took {elapsed:.2f}s")
            # Store timing for performance monitoring
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            st.session_state.performance_metrics[func.__name__] = elapsed
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
            raise
    return wrapper

# ============================================
# DATA VALIDATION
# ============================================

class DataValidator:
    """Validate data at each step"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> bool:
        """Validate dataframe has required columns and data"""
        if df is None or df.empty:
            logger.error(f"{context}: Empty or None dataframe")
            return False
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"{context}: Missing required columns: {missing_cols}")
        
        logger.info(f"{context}: Found {len(df.columns)} columns, {len(df)} rows")
        
        # Calculate data quality metrics
        completeness = df.notna().sum().sum() / (len(df) * len(df.columns))
        if 'data_quality' not in st.session_state:
            st.session_state.data_quality = {}
        st.session_state.data_quality['completeness'] = completeness
        st.session_state.data_quality['total_rows'] = len(df)
        st.session_state.data_quality['total_columns'] = len(df.columns)
        
        return True
    
    @staticmethod
    def validate_numeric_column(series: pd.Series, col_name: str, 
                              min_val: Optional[float] = None, 
                              max_val: Optional[float] = None) -> pd.Series:
        """Validate and clean numeric column"""
        if series is None:
            return pd.Series(dtype=float)
        
        # Convert to numeric, coercing errors
        series = pd.to_numeric(series, errors='coerce')
        
        # Apply bounds if specified
        if min_val is not None:
            series = series.clip(lower=min_val)
        if max_val is not None:
            series = series.clip(upper=max_val)
        
        # Log if too many NaN values
        nan_pct = series.isna().sum() / len(series) * 100
        if nan_pct > 50:
            logger.warning(f"{col_name}: {nan_pct:.1f}% NaN values")
        
        return series

# ============================================
# SMART CACHING FOR DATA LOADING - ENHANCED
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, sheet_url: str = None, gid: str = None) -> Tuple[pd.DataFrame, datetime]:
    """Load and process data with smart caching - supports both Google Sheets and CSV upload"""
    try:
        # Record start time
        start_time = time.perf_counter()
        
        if source_type == "upload" and file_data is not None:
            # Load from uploaded file
            logger.info("Loading data from uploaded CSV")
            df = pd.read_csv(file_data)
            logger.info(f"Successfully loaded {len(df)} rows from uploaded CSV")
        else:
            # Load from Google Sheets
            # Validate inputs
            if not sheet_url:
                sheet_url = CONFIG.DEFAULT_SHEET_URL
            if not gid:
                gid = CONFIG.DEFAULT_GID
                
            if not sheet_url or not gid:
                raise ValueError("Sheet URL and GID are required")
            
            # Construct CSV URL
            base_url = sheet_url.split('/edit')[0]
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from Google Sheets")
            
            # Load with timeout and error handling
            try:
                df = pd.read_csv(csv_url, low_memory=False)
            except Exception as e:
                logger.error(f"Failed to load from Google Sheets: {str(e)}")
                # Fallback to cached data if available
                if 'last_good_data' in st.session_state:
                    logger.info("Using cached data as fallback")
                    return st.session_state.last_good_data
                raise
        
        if df.empty:
            raise ValueError("Loaded empty dataframe")
        
        logger.info(f"Successfully loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Process the data
        df = DataProcessor.process_dataframe(df)
        
        # Calculate rankings
        df = RankingEngine.calculate_rankings(df)
        
        # Store as last good data
        st.session_state.last_good_data = (df.copy(), datetime.now())
        
        # Record processing time
        processing_time = time.perf_counter() - start_time
        logger.info(f"Total data processing time: {processing_time:.2f}s")
        
        # Return processed data with timestamp
        return df, datetime.now()
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        raise

# ============================================
# DATA PROCESSING
# ============================================

class DataProcessor:
    """Handle all data processing with validation and error handling"""
    
    # Define all expected columns
    NUMERIC_COLUMNS = [
        'price', 'prev_close', 'low_52w', 'high_52w',
        'from_low_pct', 'from_high_pct',
        'sma_20d', 'sma_50d', 'sma_200d',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
        'vol_ratio_90d_180d',
        'rvol', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'
    ]
    
    CATEGORICAL_COLUMNS = ['ticker', 'company_name', 'category', 'sector']
    
    REQUIRED_COLUMNS = ['ticker', 'price']
    
    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False, is_volume_ratio: bool = False) -> Optional[float]:
        """Clean and convert Indian number format to float - FIXED for percentages"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            # Convert to string and clean
            cleaned = str(value).strip()
            
            # Quick check for invalid values
            if cleaned in ['', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None', '#VALUE!', '#ERROR!', '#DIV/0!']:
                return np.nan
            
            # Handle scientific notation
            if 'e' in cleaned.lower() or 'E' in cleaned:
                try:
                    return float(cleaned)
                except:
                    return np.nan
            
            # Remove currency symbols and special characters
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '')
            
            # FIXED: Handle percentage values properly
            if is_percentage:
                # Data is stored as percentage values directly (e.g., "-56.61" means -56.61%)
                # NO division by 100 needed
                if '%' in cleaned:
                    cleaned = cleaned.replace('%', '')
                return float(cleaned)
            
            # For non-percentage values, remove % if present
            cleaned = cleaned.replace('%', '')
            
            return float(cleaned)
        except (ValueError, AttributeError):
            return np.nan
    
    @staticmethod
    @timer
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline with validation"""
        if not DataValidator.validate_dataframe(df, DataProcessor.REQUIRED_COLUMNS, "Initial data"):
            return pd.DataFrame()
        
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Define which columns are percentages (stored as percentage values, not decimals)
        percentage_columns = [
            'from_low_pct', 'from_high_pct',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
            'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
            'eps_change_pct'
        ]
        
        # Volume ratios are also percentages
        volume_ratio_columns = [
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'vol_ratio_90d_180d'
        ]
        
        # Process numeric columns
        for col in DataProcessor.NUMERIC_COLUMNS:
            if col in df.columns:
                # FIXED: Properly identify percentage columns
                is_pct = col in percentage_columns
                is_vol_ratio = col in volume_ratio_columns
                
                # Clean numeric values
                df[col] = df[col].apply(
                    lambda x: DataProcessor.clean_numeric_value(x, is_percentage=is_pct, is_volume_ratio=is_vol_ratio)
                )
                
                # Additional validation
                df[col] = DataValidator.validate_numeric_column(df[col], col)
        
        # Process categorical columns
        for col in DataProcessor.CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', '', 'N/A', 'NaN'], 'Unknown')
                # Remove extra whitespace
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # FIXED: Volume ratios conversion
        for col in volume_ratio_columns:
            if col in df.columns:
                # Data is stored as percentage change (e.g., -56.61 means 56.61% decrease)
                # Convert to ratio: 100% + change% = new ratio
                # -56.61% means 43.39% of original = 0.4339 ratio
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].fillna(1.0)
                df[col] = df[col].clip(0.01, 100.0)  # Allow wider range but prevent negative
        
        # Validate data quality
        initial_count = len(df)
        
        # Remove rows with critical missing data
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > CONFIG.MIN_VALID_PRICE]
        
        # For position data, fill NaN with reasonable defaults
        if 'from_low_pct' in df.columns:
            df['from_low_pct'] = df['from_low_pct'].fillna(50)
        else:
            df['from_low_pct'] = 50
            
        if 'from_high_pct' in df.columns:
            df['from_high_pct'] = df['from_high_pct'].fillna(-50)
        else:
            df['from_high_pct'] = -50
        
        # Remove duplicate tickers (keep first)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            logger.info(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid/duplicate rows")
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        # FIXED: Don't cap RVOL values
        if 'rvol' not in df.columns:
            df['rvol'] = 1.0
        else:
            df['rvol'] = pd.to_numeric(df['rvol'], errors='coerce')
            df['rvol'] = df['rvol'].fillna(1.0).clip(lower=0.01)
            # Don't cap extreme values - they're legitimate
        
        logger.info(f"Processed {len(df)} valid stocks")
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications for EPS, PE, and Price - FIXED boundaries"""
        
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Classify a value into appropriate tier - FIXED boundary handling"""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                # FIXED: Use < for max boundary to prevent overlap
                if min_val <= value < max_val:
                    return tier_name
                # Special case for the last tier
                if max_val == float('inf') and value >= min_val:
                    return tier_name
            return "Unknown"
        
        # Add tier columns
        df['eps_tier'] = df['eps_current'].apply(
            lambda x: classify_tier(x, CONFIG.TIERS['eps'])
        )
        
        df['pe_tier'] = df['pe'].apply(
            lambda x: "Negative/NA" if pd.isna(x) or x <= 0 
            else classify_tier(x, CONFIG.TIERS['pe'])
        )
        
        df['price_tier'] = df['price'].apply(
            lambda x: classify_tier(x, CONFIG.TIERS['price'])
        )
        
        return df

# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    """Core ranking calculations - optimized and vectorized"""
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safely rank a series with proper handling of edge cases"""
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        # Create a copy to avoid modifying original
        series = series.copy()
        
        # Replace inf values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Count valid values
        valid_count = series.notna().sum()
        if valid_count == 0:
            # Add small random variation to avoid identical scores
            return pd.Series(50 + np.random.uniform(-2, 2, size=len(series)), index=series.index)
        
        # For percentage ranks, ensure 0-100 scale
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom')
            ranks = ranks * 100
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
        
        # For NaN values, assign worst rank
        if pct:
            ranks = ranks.fillna(0 if ascending else 100)
        else:
            ranks = ranks.fillna(valid_count + 1)
        
        return ranks
    
    @staticmethod
    def calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score from 52-week range - FIXED logic"""
        # Initialize with neutral score
        position_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check if we have the required columns
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
            logger.warning("No position data available, using neutral position scores")
            return position_score + np.random.uniform(-5, 5, size=len(df))
        
        # Get data with reasonable defaults
        from_low = df['from_low_pct'].fillna(50) if has_from_low else pd.Series(50, index=df.index)
        from_high = df['from_high_pct'].fillna(-50) if has_from_high else pd.Series(-50, index=df.index)
        
        # Rank distance from low (higher % from low is better)
        if has_from_low:
            rank_from_low = RankingEngine.safe_rank(from_low, pct=True, ascending=True)
        else:
            rank_from_low = pd.Series(50, index=df.index)
        
        # FIXED: For distance from high, closer to high is better
        if has_from_high:
            # from_high is negative, so more negative = further from high
            # We want stocks closer to high (less negative) to rank higher
            rank_from_high = RankingEngine.safe_rank(from_high, pct=True, ascending=False)
        else:
            rank_from_high = pd.Series(50, index=df.index)
        
        # Combined position score
        if has_from_low and has_from_high:
            position_score = (rank_from_low * 0.6 + rank_from_high * 0.4)
        elif has_from_low:
            position_score = rank_from_low
        else:
            position_score = rank_from_high
        
        return position_score.clip(0, 100)
    
    @staticmethod
    def calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive volume score"""
        volume_score = pd.Series(50, index=df.index, dtype=float)
        
        vol_cols = [
            ('vol_ratio_1d_90d', 0.20),
            ('vol_ratio_7d_90d', 0.20),
            ('vol_ratio_30d_90d', 0.20),
            ('vol_ratio_30d_180d', 0.15),
            ('vol_ratio_90d_180d', 0.25)
        ]
        
        total_weight = 0
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        has_any_vol_data = False
        
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                has_any_vol_data = True
                col_data = df[col].copy()
                col_data = col_data.fillna(1.0)
                col_data = col_data.clip(lower=0.1)
                col_rank = RankingEngine.safe_rank(col_data, pct=True, ascending=True)
                weighted_score += col_rank * weight
                total_weight += weight
        
        if total_weight > 0 and has_any_vol_data:
            volume_score = weighted_score / total_weight
        else:
            logger.warning("No volume ratio data available, using neutral scores")
            volume_score = pd.Series(50, index=df.index, dtype=float)
            volume_score += np.random.uniform(-5, 5, size=len(df))
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns - FIXED fallback"""
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            logger.warning("No 30-day return data available, using neutral momentum scores")
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                # Use 7-day returns directly without scaling
                ret_7d = df['ret_7d'].fillna(0)
                momentum_score = RankingEngine.safe_rank(ret_7d, pct=True, ascending=True)
                logger.info("Using 7-day returns for momentum score")
            else:
                momentum_score += np.random.uniform(-5, 5, size=len(df))
            
            return momentum_score.clip(0, 100)
        
        ret_30d = df['ret_30d'].fillna(0)
        momentum_score = RankingEngine.safe_rank(ret_30d, pct=True, ascending=True)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            consistency_bonus[all_positive] = 5
            
            # Safe division to avoid divide by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = df['ret_7d'] / 7
                daily_ret_30d = df['ret_30d'] / 30
                
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            consistency_bonus[accelerating] = 10
            
            momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
        
        return momentum_score
    
    @staticmethod
    def calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating - FIXED math"""
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in req_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient return data for acceleration calculation")
            return acceleration_score + np.random.uniform(-5, 5, size=len(df))
        
        ret_1d = df['ret_1d'].fillna(0) if 'ret_1d' in df.columns else pd.Series(0, index=df.index)
        ret_7d = df['ret_7d'].fillna(0) if 'ret_7d' in df.columns else pd.Series(0, index=df.index)
        ret_30d = df['ret_30d'].fillna(0) if 'ret_30d' in df.columns else pd.Series(0, index=df.index)
        
        # FIXED: Compare daily average returns, not annualized
        with np.errstate(divide='ignore', invalid='ignore'):
            # Average daily returns
            avg_daily_1d = ret_1d  # Already daily
            avg_daily_7d = ret_7d / 7
            avg_daily_30d = ret_30d / 30
        
        if all(col in df.columns for col in req_cols):
            # Perfect acceleration: recent returns accelerating at each timeframe
            perfect = (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
            acceleration_score.loc[perfect] = 100
            
            # Good acceleration: today beats week average
            good = (~perfect) & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0)
            acceleration_score.loc[good] = 80
            
            # Moderate: positive today
            moderate = (~perfect) & (~good) & (ret_1d > 0)
            acceleration_score.loc[moderate] = 60
            
            # Slight deceleration
            slight_decel = (ret_1d <= 0) & (ret_7d > 0)
            acceleration_score.loc[slight_decel] = 40
            
            # Strong deceleration
            strong_decel = (ret_1d <= 0) & (ret_7d <= 0)
            acceleration_score.loc[strong_decel] = 20
        else:
            if 'ret_1d' in df.columns and 'ret_7d' in df.columns:
                accelerating = avg_daily_1d > avg_daily_7d
                acceleration_score.loc[accelerating & (ret_1d > 0)] = 75
                acceleration_score.loc[~accelerating & (ret_1d > 0)] = 55
                acceleration_score.loc[ret_1d <= 0] = 35
        
        return acceleration_score
    
    @staticmethod
    def calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout probability - FIXED distance calculation"""
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        
        # Factor 1: Distance from high (40% weight) - FIXED
        if 'from_high_pct' in df.columns:
            # from_high_pct is negative (e.g., -20 means 20% below high)
            # Closer to high (less negative) is better
            distance_from_high = -df['from_high_pct'].fillna(-50)  # Convert to positive distance
            # Now 0 = at high, 100 = far from high
            distance_factor = (100 - distance_from_high).clip(0, 100)
        else:
            distance_factor = pd.Series(50, index=df.index)
        
        # Factor 2: Volume surge (40% weight)
        volume_factor = pd.Series(50, index=df.index)
        if 'vol_ratio_7d_90d' in df.columns:
            vol_ratio = df['vol_ratio_7d_90d'].fillna(1.0)
            # vol_ratio > 1 means increasing volume
            volume_factor = ((vol_ratio - 1) * 100).clip(0, 100)
        
        # Factor 3: Trend support (20% weight)
        trend_factor = pd.Series(0, index=df.index, dtype=float)
        trend_count = 0
        
        if 'price' in df.columns:
            current_price = df['price']
            
            if 'sma_20d' in df.columns:
                above_20 = (current_price > df['sma_20d']).fillna(False)
                trend_factor += above_20.astype(float) * 33.33
                trend_count += 1
            
            if 'sma_50d' in df.columns:
                above_50 = (current_price > df['sma_50d']).fillna(False)
                trend_factor += above_50.astype(float) * 33.33
                trend_count += 1
            
            if 'sma_200d' in df.columns:
                above_200 = (current_price > df['sma_200d']).fillna(False)
                trend_factor += above_200.astype(float) * 33.34
                trend_count += 1
        
        if trend_count > 0 and trend_count < 3:
            trend_factor = trend_factor * (3 / trend_count)
        
        trend_factor = trend_factor.clip(0, 100)
        
        breakout_score = (
            distance_factor * 0.4 +
            volume_factor * 0.4 +
            trend_factor * 0.2
        )
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate RVOL-based score - FIXED for extreme values"""
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        rvol = df['rvol'].fillna(1.0)
        rvol_score = pd.Series(50, index=df.index, dtype=float)
        
        # FIXED: Handle extreme RVOL values properly
        rvol_score.loc[rvol > 1000] = 100  # Extreme surge
        rvol_score.loc[(rvol > 100) & (rvol <= 1000)] = 95
        rvol_score.loc[(rvol > 10) & (rvol <= 100)] = 90
        rvol_score.loc[(rvol > 5) & (rvol <= 10)] = 85
        rvol_score.loc[(rvol > 3) & (rvol <= 5)] = 80
        rvol_score.loc[(rvol > 2) & (rvol <= 3)] = 75
        rvol_score.loc[(rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score.loc[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score.loc[(rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score.loc[(rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score.loc[(rvol > 0.3) & (rvol <= 0.5)] = 30
        rvol_score.loc[rvol <= 0.3] = 20
        
        return rvol_score
    
    @staticmethod
    def calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality score based on SMA alignment"""
        trend_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'price' not in df.columns:
            return trend_score
        
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        available_smas = [col for col in sma_cols if col in df.columns and df[col].notna().any()]
        
        if len(available_smas) == 0:
            return trend_score
        
        current_price = df['price']
        
        if len(available_smas) >= 3:
            # Perfect trend alignment
            perfect_trend = (
                (current_price > df['sma_20d']) & 
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            )
            trend_score.loc[perfect_trend] = 100
            
            # Strong trend - price above all SMAs
            strong_trend = (
                (~perfect_trend) &
                (current_price > df['sma_20d']) & 
                (current_price > df['sma_50d']) & 
                (current_price > df['sma_200d'])
            )
            trend_score.loc[strong_trend] = 85
            
            # Count how many SMAs price is above
            above_count = pd.Series(0, index=df.index)
            for sma in available_smas:
                above_count += (current_price > df[sma]).astype(int)
            
            # Good trend - above 2 SMAs
            good_trend = (above_count == 2) & (~perfect_trend) & (~strong_trend)
            trend_score.loc[good_trend] = 70
            
            # Weak trend - above 1 SMA
            weak_trend = (above_count == 1)
            trend_score.loc[weak_trend] = 40
            
            # Poor trend - below all SMAs
            poor_trend = (above_count == 0)
            trend_score.loc[poor_trend] = 20
        
        elif len(available_smas) == 2:
            above_all = pd.Series(True, index=df.index)
            for sma in available_smas:
                above_all &= (current_price > df[sma])
            
            trend_score.loc[above_all] = 80
            trend_score.loc[~above_all] = 30
        
        elif len(available_smas) == 1:
            sma = available_smas[0]
            trend_score.loc[current_price > df[sma]] = 65
            trend_score.loc[current_price <= df[sma]] = 35
        
        return trend_score
    
    @staticmethod
    def calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score"""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        
        if not available_cols:
            return strength_score
        
        # Calculate average long-term return
        lt_returns = df[available_cols].fillna(0)
        avg_return = lt_returns.mean(axis=1)
        
        # Check if returns are improving
        if len(available_cols) >= 2:
            if 'ret_3m' in available_cols and 'ret_1y' in available_cols:
                # Compare 3-month annualized vs 1-year
                with np.errstate(divide='ignore', invalid='ignore'):
                    annualized_3m = df['ret_3m'] * 4  # Rough annualization
                improving = annualized_3m > df['ret_1y']
            else:
                improving = pd.Series(False, index=df.index)
        else:
            improving = pd.Series(False, index=df.index)
        
        # Categorize based on average return
        exceptional = avg_return > 100
        strength_score.loc[exceptional] = 100
        
        very_strong = (avg_return > 50) & (avg_return <= 100)
        strength_score.loc[very_strong] = 90
        
        strong = (avg_return > 30) & (avg_return <= 50)
        strength_score.loc[strong] = 80
        
        good = (avg_return > 15) & (avg_return <= 30)
        strength_score.loc[good] = 70
        
        moderate = (avg_return > 5) & (avg_return <= 15)
        strength_score.loc[moderate] = 60
        
        weak = (avg_return > 0) & (avg_return <= 5)
        strength_score.loc[weak] = 50
        
        recovering = (avg_return > -10) & (avg_return <= 0)
        strength_score.loc[recovering] = 40
        
        poor = (avg_return > -25) & (avg_return <= -10)
        strength_score.loc[poor] = 30
        
        very_poor = avg_return <= -25
        strength_score.loc[very_poor] = 20
        
        # Bonus for improving returns
        strength_score.loc[improving] += 5
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score based on trading volume - FIXED"""
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'volume_30d' in df.columns and 'price' in df.columns:
            # Calculate dollar volume for true liquidity measure
            dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
            
            # Rank based on dollar volume
            liquidity_score = RankingEngine.safe_rank(
                dollar_volume, pct=True, ascending=True
            )
            
            # Add consistency bonus if multiple volume periods available
            if all(col in df.columns for col in ['volume_7d', 'volume_30d', 'volume_90d']):
                vol_data = df[['volume_7d', 'volume_30d', 'volume_90d']]
                
                # Calculate coefficient of variation
                with np.errstate(divide='ignore', invalid='ignore'):
                    vol_mean = vol_data.mean(axis=1)
                    vol_std = vol_data.std(axis=1)
                    
                    # Avoid division by zero
                    vol_cv = pd.Series(1.0, index=df.index)
                    valid_mask = vol_mean > 0
                    vol_cv[valid_mask] = vol_std[valid_mask] / vol_mean[valid_mask]
                
                # Lower CV means more consistent volume
                consistency_score = RankingEngine.safe_rank(
                    vol_cv, pct=True, ascending=False
                )
                
                liquidity_score = liquidity_score * 0.8 + consistency_score * 0.2
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    @timer
    def calculate_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Main ranking calculation with all components"""
        if df.empty:
            return df
        
        logger.info("Starting ranking calculations...")
        
        # Calculate all component scores with error handling
        try:
            df['position_score'] = RankingEngine.calculate_position_score(df)
        except Exception as e:
            logger.error(f"Error calculating position score: {str(e)}")
            df['position_score'] = 50
        
        try:
            df['volume_score'] = RankingEngine.calculate_volume_score(df)
        except Exception as e:
            logger.error(f"Error calculating volume score: {str(e)}")
            df['volume_score'] = 50
        
        try:
            df['momentum_score'] = RankingEngine.calculate_momentum_score(df)
        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            df['momentum_score'] = 50
        
        try:
            df['acceleration_score'] = RankingEngine.calculate_acceleration_score(df)
        except Exception as e:
            logger.error(f"Error calculating acceleration score: {str(e)}")
            df['acceleration_score'] = 50
        
        try:
            df['breakout_score'] = RankingEngine.calculate_breakout_score(df)
        except Exception as e:
            logger.error(f"Error calculating breakout score: {str(e)}")
            df['breakout_score'] = 50
        
        try:
            df['rvol_score'] = RankingEngine.calculate_rvol_score(df)
        except Exception as e:
            logger.error(f"Error calculating rvol score: {str(e)}")
            df['rvol_score'] = 50
        
        # Calculate auxiliary scores
        try:
            df['trend_quality'] = RankingEngine.calculate_trend_quality(df)
        except Exception as e:
            logger.error(f"Error calculating trend quality: {str(e)}")
            df['trend_quality'] = 50
        
        try:
            df['long_term_strength'] = RankingEngine.calculate_long_term_strength(df)
        except Exception as e:
            logger.error(f"Error calculating long term strength: {str(e)}")
            df['long_term_strength'] = 50
        
        try:
            df['liquidity_score'] = RankingEngine.calculate_liquidity_score(df)
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            df['liquidity_score'] = 50
        
        # MASTER SCORE 3.0
        components = {
            'position_score': CONFIG.POSITION_WEIGHT,
            'volume_score': CONFIG.VOLUME_WEIGHT,
            'momentum_score': CONFIG.MOMENTUM_WEIGHT,
            'acceleration_score': CONFIG.ACCELERATION_WEIGHT,
            'breakout_score': CONFIG.BREAKOUT_WEIGHT,
            'rvol_score': CONFIG.RVOL_WEIGHT
        }
        
        # Calculate master score
        df['master_score'] = 0
        total_weight = 0
        
        for component, weight in components.items():
            if component in df.columns:
                df['master_score'] += df[component].fillna(50) * weight
                total_weight += weight
            else:
                logger.warning(f"Missing component: {component}")
        
        # Normalize if weights don't sum to 1.0
        if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Total weight is {total_weight}, normalizing...")
            df['master_score'] = df['master_score'] / total_weight
        
        df['master_score'] = df['master_score'].clip(0, 100)
        
        # Calculate ranks with proper handling
        valid_scores = df['master_score'].notna()
        
        if valid_scores.sum() == 0:
            logger.error("No valid master scores calculated!")
            df['rank'] = 9999
            df['percentile'] = 0
        else:
            # Use method='first' to ensure unique ranks
            df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom')
            df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
            
            df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
            df['percentile'] = df['percentile'].fillna(0)
        
        # Calculate category-specific ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        # Detect patterns
        df = RankingEngine._detect_patterns(df)
        
        logger.info(f"Ranking complete: {len(df)} stocks processed")
        
        return df
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks within each category"""
        # Get unique categories
        categories = df['category'].unique()
        
        # Initialize category rank columns
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        # Rank within each category
        for category in categories:
            if category != 'Unknown':
                mask = df['category'] == category
                cat_df = df[mask]
                
                if len(cat_df) > 0:
                    # Calculate ranks within category
                    cat_ranks = RankingEngine.safe_rank(
                        cat_df['master_score'], pct=False, ascending=False
                    )
                    df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                    
                    # Calculate percentiles within category
                    cat_percentiles = RankingEngine.safe_rank(
                        cat_df['master_score'], pct=True, ascending=True
                    )
                    df.loc[mask, 'category_percentile'] = cat_percentiles
        
        return df
    
    @staticmethod
    def _detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect patterns using vectorized operations - FIXED string handling"""
        # Initialize pattern column properly
        df['patterns'] = ''
        
        # Use vectorized operations for pattern detection
        patterns_list = []
        
        # 1. Category Leader
        if 'category_percentile' in df.columns:
            cat_leader_mask = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
            patterns_list.append((cat_leader_mask, '🔥 CAT LEADER'))
        
        # 2. Hidden Gem
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            hidden_gem_mask = (
                (df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
                (df['percentile'] < 70)
            )
            patterns_list.append((hidden_gem_mask, '💎 HIDDEN GEM'))
        
        # 3. Accelerating
        if 'acceleration_score' in df.columns:
            accel_mask = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
            patterns_list.append((accel_mask, '🚀 ACCELERATING'))
        
        # 4. Institutional
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            inst_mask = (
                (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['vol_ratio_90d_180d'] > 1.1)
            )
            patterns_list.append((inst_mask, '🏦 INSTITUTIONAL'))
        
        # 5. Volume Explosion
        if 'rvol' in df.columns:
            vol_explosion_mask = df['rvol'] > 3
            patterns_list.append((vol_explosion_mask, '⚡ VOL EXPLOSION'))
        
        # 6. Breakout Ready
        if 'breakout_score' in df.columns:
            breakout_mask = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
            patterns_list.append((breakout_mask, '🎯 BREAKOUT'))
        
        # 7. Market Leader
        if 'percentile' in df.columns:
            market_leader_mask = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
            patterns_list.append((market_leader_mask, '👑 MARKET LEADER'))
        
        # 8. Momentum Wave
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            momentum_wave_mask = (
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                (df['acceleration_score'] >= 70)
            )
            patterns_list.append((momentum_wave_mask, '🌊 MOMENTUM WAVE'))
        
        # 9. Liquid Leader
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            liquid_mask = (
                (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
                (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
            )
            patterns_list.append((liquid_mask, '💰 LIQUID LEADER'))
        
        # 10. Long-term Strength
        if 'long_term_strength' in df.columns:
            long_strength_mask = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            patterns_list.append((long_strength_mask, '💪 LONG STRENGTH'))
        
        # 11. Quality Trend
        if 'trend_quality' in df.columns:
            quality_trend_mask = df['trend_quality'] >= 80
            patterns_list.append((quality_trend_mask, '📈 QUALITY TREND'))
        
        # SMART FUNDAMENTAL PATTERNS - FIXED thresholds for percentage data
        # 12. Value Momentum
        if 'pe' in df.columns and 'percentile' in df.columns:
            has_valid_pe = (
                df['pe'].notna() & 
                (df['pe'] > 0) & 
                (df['pe'] < 10000) &
                ~np.isinf(df['pe'])
            )
            value_momentum_mask = has_valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
            patterns_list.append((value_momentum_mask, '💎 VALUE MOMENTUM'))
        
        # 13. Earnings Rocket - FIXED for percentage data
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
            # eps_change_pct is stored as percentage (100 = 100% growth)
            extreme_growth = has_eps_growth & (df['eps_change_pct'] > 1000)  # >1000%
            normal_growth = has_eps_growth & (df['eps_change_pct'] > 50) & (df['eps_change_pct'] <= 1000)
            
            earnings_rocket_mask = (
                (extreme_growth & (df['acceleration_score'] >= 80)) |
                (normal_growth & (df['acceleration_score'] >= 70))
            )
            patterns_list.append((earnings_rocket_mask, '📊 EARNINGS ROCKET'))
        
        # 14. Quality Leader - FIXED for percentage data
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = (
                df['pe'].notna() & 
                df['eps_change_pct'].notna() & 
                (df['pe'] > 0) &
                (df['pe'] < 10000) &
                ~np.isinf(df['pe']) &
                ~np.isinf(df['eps_change_pct'])
            )
            quality_leader_mask = (
                has_complete_data &
                (df['pe'] >= 10) & (df['pe'] <= 25) &
                (df['eps_change_pct'] > 20) &  # >20% growth
                (df['percentile'] >= 80)
            )
            patterns_list.append((quality_leader_mask, '🏆 QUALITY LEADER'))
        
        # 15. Turnaround Play - FIXED for percentage data
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
            mega_turnaround = has_eps & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60)  # >500%
            strong_turnaround = has_eps & (df['eps_change_pct'] > 100) & (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70)
            
            turnaround_mask = mega_turnaround | strong_turnaround
            patterns_list.append((turnaround_mask, '⚡ TURNAROUND'))
        
        # 16. Overvalued Warning
        if 'pe' in df.columns:
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0) & ~np.isinf(df['pe'])
            extreme_pe_mask = has_valid_pe & (df['pe'] > 100)
            patterns_list.append((extreme_pe_mask, '⚠️ HIGH PE'))
        
        # ============================================
        # ADD THESE NEW PATTERNS AFTER EXISTING ONES
        # ============================================

        # 17. 52W High Approach - VERY RELIABLE
        if 'from_high_pct' in df.columns and 'volume_score' in df.columns and 'momentum_score' in df.columns:
            high_approach_mask = (
                (df['from_high_pct'] > -5) & 
                (df['volume_score'] >= 70) & 
                (df['momentum_score'] >= 60)
            )
            patterns_list.append((high_approach_mask, '🎯 52W HIGH APPROACH'))

        # 18. 52W Low Bounce - REVERSAL PLAY
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            low_bounce_mask = (
                (df['from_low_pct'] < 20) & 
                (df['acceleration_score'] >= 80) & 
                (df['ret_30d'] > 10)
            )
            patterns_list.append((low_bounce_mask, '🔄 52W LOW BOUNCE'))

        # 19. Golden Zone - OPTIMAL RANGE
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            golden_zone_mask = (
                (df['from_low_pct'] > 60) & 
                (df['from_high_pct'] > -40) & 
                (df['trend_quality'] >= 70)
            )
            patterns_list.append((golden_zone_mask, '👑 GOLDEN ZONE'))

        # 20. Volume Accumulation - SMART MONEY
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            vol_accum_mask = (
                (df['vol_ratio_30d_90d'] > 1.2) & 
                (df['vol_ratio_90d_180d'] > 1.1) & 
                (df['ret_30d'] > 5)
            )
            patterns_list.append((vol_accum_mask, '📊 VOL ACCUMULATION'))

        # 21. Momentum Divergence - ACCELERATION
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            # Calculate daily pace safely
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = df['ret_7d'] / 7
                daily_30d_pace = df['ret_30d'] / 30
                
            divergence_mask = (
                (daily_7d_pace > daily_30d_pace * 1.5) & 
                (df['acceleration_score'] >= 85) & 
                (df['rvol'] > 2)
            )
            divergence_mask = divergence_mask.fillna(False)
            patterns_list.append((divergence_mask, '🔀 MOMENTUM DIVERGE'))

        # 22. Range Compression - BREAKOUT SETUP
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            # Calculate range safely
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = ((df['high_52w'] - df['low_52w']) / df['low_52w']) * 100
                range_pct = range_pct.fillna(100)  # Default to 100 if calculation fails
                
            compression_mask = (
                (range_pct < 50) & 
                (df['from_low_pct'] > 30)
            )
            patterns_list.append((compression_mask, '🎯 RANGE COMPRESS'))
        
        # Efficiently combine all patterns
        for mask, pattern_name in patterns_list:
            df.loc[mask, 'patterns'] = df.loc[mask, 'patterns'].apply(
                lambda x: f"{x} | {pattern_name}" if x else pattern_name
            )
        
        return df

# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    """Handle all filtering operations with smart interconnected filters"""
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column: str, 
                         exclude_unknown: bool = True,
                         filters: Dict[str, Any] = None) -> List[str]:
        """Get sorted unique values for a column with smart filtering"""
        if df.empty or column not in df.columns:
            return []
        
        # Apply any existing filters first (for interconnected filtering)
        if filters:
            filtered_df = FilterEngine._apply_filter_subset(df, filters, exclude_cols=[column])
        else:
            filtered_df = df
        
        try:
            values = filtered_df[column].dropna().unique().tolist()
            
            # Convert to strings to ensure consistency
            values = [str(v) for v in values]
            
            if exclude_unknown:
                values = [v for v in values if v not in ['Unknown', 'unknown', 'nan', 'NaN', '']]
            
            return sorted(values)
        except Exception as e:
            logger.error(f"Error getting unique values for {column}: {str(e)}")
            return []
    
    @staticmethod
    def _apply_filter_subset(df: pd.DataFrame, filters: Dict[str, Any], exclude_cols: List[str]) -> pd.DataFrame:
        """Apply filters excluding specific columns (for interconnected filtering)"""
        filtered_df = df.copy()
        
        # Apply each filter except excluded ones
        if 'categories' in filters and 'category' not in exclude_cols:
            categories = filters.get('categories', [])
            if categories and 'All' not in categories and categories != ['']:
                filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        if 'sectors' in filters and 'sector' not in exclude_cols:
            sectors = filters.get('sectors', [])
            if sectors and 'All' not in sectors and sectors != ['']:
                filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
        
        if 'eps_tiers' in filters and 'eps_tier' not in exclude_cols:
            eps_tiers = filters.get('eps_tiers', [])
            if eps_tiers and 'All' not in eps_tiers and eps_tiers != ['']:
                filtered_df = filtered_df[filtered_df['eps_tier'].isin(eps_tiers)]
        
        if 'pe_tiers' in filters and 'pe_tier' not in exclude_cols:
            pe_tiers = filters.get('pe_tiers', [])
            if pe_tiers and 'All' not in pe_tiers and pe_tiers != ['']:
                filtered_df = filtered_df[filtered_df['pe_tier'].isin(pe_tiers)]
        
        if 'price_tiers' in filters and 'price_tier' not in exclude_cols:
            price_tiers = filters.get('price_tiers', [])
            if price_tiers and 'All' not in price_tiers and price_tiers != ['']:
                filtered_df = filtered_df[filtered_df['price_tier'].isin(price_tiers)]
        
        return filtered_df
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with validation"""
        if df.empty:
            return df
        
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Category filter
        categories = filters.get('categories', [])
        if categories and 'All' not in categories and categories != ['']:
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        # Sector filter
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors and sectors != ['']:
            filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
        
        # EPS tier filter
        eps_tiers = filters.get('eps_tiers', [])
        if eps_tiers and 'All' not in eps_tiers and eps_tiers != ['']:
            filtered_df = filtered_df[filtered_df['eps_tier'].isin(eps_tiers)]
        
        # PE tier filter
        pe_tiers = filters.get('pe_tiers', [])
        if pe_tiers and 'All' not in pe_tiers and pe_tiers != ['']:
            filtered_df = filtered_df[filtered_df['pe_tier'].isin(pe_tiers)]
        
        # Price tier filter
        price_tiers = filters.get('price_tiers', [])
        if price_tiers and 'All' not in price_tiers and price_tiers != ['']:
            filtered_df = filtered_df[filtered_df['price_tier'].isin(price_tiers)]
        
        # Score filter
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= min_score]
        
        # EPS change filter - FIXED: Data is already in percentage
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in filtered_df.columns:
            # User enters percentage, data is in percentage - no conversion needed
            filtered_df = filtered_df[
                (filtered_df['eps_change_pct'] >= min_eps_change) | 
                (filtered_df['eps_change_pct'].isna())
            ]
        
        # Pattern filter
        patterns = filters.get('patterns', [])
        if patterns:
            pattern_mask = filtered_df['patterns'].str.contains(
                '|'.join(patterns), 
                case=False, 
                na=False
            )
            filtered_df = filtered_df[pattern_mask]
        
        # Trend filter
        if filters.get('trend_range') and filters.get('trend_filter') != 'All Trends':
            min_trend, max_trend = filters['trend_range']
            if 'trend_quality' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['trend_quality'] >= min_trend) & 
                    (filtered_df['trend_quality'] <= max_trend)
                ]
        
        # PE filters
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['pe'].isna()) |
                ((filtered_df['pe'] > 0) & 
                 (filtered_df['pe'] >= min_pe) & 
                 ~np.isinf(filtered_df['pe']))
            ]
        
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['pe'].isna()) |
                ((filtered_df['pe'] > 0) & 
                 (filtered_df['pe'] <= max_pe) & 
                 ~np.isinf(filtered_df['pe']))
            ]
        
        # Data completeness filter
        if filters.get('require_fundamental_data', False):
            if 'pe' in filtered_df.columns and 'eps_change_pct' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['pe'].notna() & 
                    (filtered_df['pe'] > 0) &
                    ~np.isinf(filtered_df['pe']) &
                    filtered_df['eps_change_pct'].notna() &
                    ~np.isinf(filtered_df['eps_change_pct'])
                ]
        
        filtered_count = len(filtered_df)
        if filtered_count < initial_count:
            logger.info(f"Filters reduced stocks from {initial_count} to {filtered_count}")
        
        return filtered_df

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
        """Create enhanced top stocks breakdown chart"""
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
        
        for name, score_col, weight, color in components:
            if score_col in top_df.columns:
                weighted_contrib = top_df[score_col] * weight
                
                fig.add_trace(go.Bar(
                    name=f'{name} ({weight:.0%})',
                    y=top_df['ticker'],
                    x=weighted_contrib,
                    orientation='h',
                    marker_color=color,
                    text=[f"{val:.1f}" for val in top_df[score_col]],
                    textposition='inside',
                    hovertemplate=f'{name}<br>Score: %{{text}}<br>Contribution: %{{x:.1f}}<extra></extra>'
                ))
        
        # Add master score annotation
        for i, (idx, row) in enumerate(top_df.iterrows()):
            fig.add_annotation(
                x=row['master_score'],
                y=i,
                text=f"{row['master_score']:.1f}",
                showarrow=False,
                xanchor='left',
                bgcolor='rgba(255,255,255,0.8)'
            )
        
        fig.update_layout(
            title=f"Top {len(top_df)} Stocks - Master Score 3.0 Breakdown",
            xaxis_title="Weighted Score Contribution",
            xaxis_range=[0, 105],
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
            )
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance_scatter(df: pd.DataFrame) -> go.Figure:
        """Create sector performance scatter plot"""
        # Aggregate by sector
        try:
            sector_stats = df.groupby('sector').agg({
                'master_score': ['mean', 'std', 'count'],
                'percentile': 'mean',
                'rvol': 'mean'
            }).reset_index()
            
            # Flatten column names
            sector_stats.columns = ['sector', 'avg_score', 'std_score', 'count', 'avg_percentile', 'avg_rvol']
            
            # Filter sectors with at least 3 stocks
            sector_stats = sector_stats[sector_stats['count'] >= 3]
            
            if len(sector_stats) == 0:
                return go.Figure()
            
            # Create scatter plot
            fig = px.scatter(
                sector_stats,
                x='avg_percentile',
                y='avg_score',
                size='count',
                color='avg_rvol',
                hover_data={
                    'count': True,
                    'std_score': ':.1f',
                    'avg_rvol': ':.2f'
                },
                text='sector',
                title='Sector Performance Analysis',
                labels={
                    'avg_percentile': 'Average Percentile Rank',
                    'avg_score': 'Average Master Score',
                    'count': 'Number of Stocks',
                    'avg_rvol': 'Avg RVOL'
                },
                color_continuous_scale='Viridis'
            )
            
            fig.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=1, color='white'))
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating sector scatter: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def create_pattern_analysis(df: pd.DataFrame) -> go.Figure:
        """Create pattern frequency analysis"""
        # Extract all patterns
        all_patterns = []
        
        if not df.empty and 'patterns' in df.columns:
            for patterns in df['patterns'].dropna():
                if patterns:
                    all_patterns.extend(patterns.split(' | '))
        
        if not all_patterns:
            fig = go.Figure()
            fig.add_annotation(
                text="No patterns detected in current selection",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Pattern Frequency Analysis",
                template='plotly_white',
                height=400
            )
            return fig
        
        # Count pattern frequencies
        pattern_counts = pd.Series(all_patterns).value_counts()
        
        # Create bar chart
        fig = go.Figure([
            go.Bar(
                x=pattern_counts.values,
                y=pattern_counts.index,
                orientation='h',
                marker_color='#3498db',
                text=pattern_counts.values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Pattern Frequency Analysis",
            xaxis_title="Number of Stocks",
            yaxis_title="Pattern",
            template='plotly_white',
            height=max(400, len(pattern_counts) * 30),
            margin=dict(l=150)
        )
        
        return fig

# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """Advanced search functionality with improved robustness"""
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with simple string matching - no complex indexing"""
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query = query.upper().strip()
            
            # Direct ticker match (exact)
            ticker_match = df[df['ticker'].str.upper() == query]
            if not ticker_match.empty:
                return ticker_match
            
            # Ticker contains query
            ticker_contains = df[df['ticker'].str.upper().str.contains(query, na=False)]
            
            # Company name contains query
            company_contains = df[df['company_name'].str.upper().str.contains(query, na=False)]
            
            # Combine results, remove duplicates
            results = pd.concat([ticker_contains, company_contains]).drop_duplicates()
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle all export operations"""
    
    @staticmethod
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        """Create comprehensive Excel report with templates"""
        output = BytesIO()
        
        # Define export templates
        templates = {
            'day_trader': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 
                          'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 
                          'volume_score', 'patterns', 'category'],
            'swing_trader': ['rank', 'ticker', 'company_name', 'master_score', 
                           'breakout_score', 'position_score', 'from_high_pct', 
                           'from_low_pct', 'trend_quality', 'patterns'],
            'investor': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 
                        'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 
                        'long_term_strength', 'category', 'sector'],
            'full': None  # Use all columns
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
                
                # 1. Top 100 Stocks
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                
                # Select columns based on template
                if template in templates and templates[template]:
                    export_cols = templates[template]
                else:
                    # Full export columns
                    export_cols = [
                        'rank', 'ticker', 'company_name', 'master_score',
                        'position_score', 'volume_score', 'momentum_score',
                        'acceleration_score', 'breakout_score', 'rvol_score',
                        'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
                        'from_low_pct', 'from_high_pct',
                        'ret_1d', 'ret_7d', 'ret_30d', 'rvol',
                        'patterns', 'category', 'sector'
                    ]
                
                available_cols = [col for col in export_cols if col in top_100.columns]
                
                # Create a copy for export to avoid modifying original
                export_df = top_100[available_cols].copy()
                
                # eps_change_pct is already in percentage format, no conversion needed
                
                export_df.to_excel(
                    writer, sheet_name='Top 100', index=False
                )
                
                # Format the sheet
                worksheet = writer.sheets['Top 100']
                for i, col in enumerate(available_cols):
                    worksheet.write(0, i, col, header_format)
                
                # 2. All Stocks Summary
                summary_cols = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'trend_quality', 'price', 'pe', 'eps_change_pct',
                    'ret_30d', 'rvol', 
                    'patterns', 'category', 'sector'
                ]
                available_summary = [col for col in summary_cols if col in df.columns]
                
                summary_df = df[available_summary].copy()
                
                summary_df.to_excel(
                    writer, sheet_name='All Stocks', index=False
                )
                
                # 3. Sector Analysis
                if 'sector' in df.columns:
                    try:
                        sector_analysis = df.groupby('sector').agg({
                            'master_score': ['mean', 'std', 'min', 'max', 'count'],
                            'rvol': 'mean',
                            'ret_30d': 'mean'
                        }).round(2)
                        
                        # Add PE analysis if available
                        if 'pe' in df.columns:
                            pe_stats = df[df['pe'] > 0].groupby('sector')['pe'].agg(['mean', 'median'])
                            sector_analysis = pd.concat([sector_analysis, pe_stats], axis=1)
                        
                        # Add EPS change if available
                        if 'eps_change_pct' in df.columns:
                            eps_stats = df.groupby('sector')['eps_change_pct'].agg(['mean', 'median'])
                            sector_analysis = pd.concat([sector_analysis, eps_stats], axis=1)
                        
                        sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
                    except Exception as e:
                        logger.warning(f"Unable to create sector analysis: {str(e)}")
                
                # 4. Category Analysis
                if 'category' in df.columns:
                    try:
                        category_analysis = df.groupby('category').agg({
                            'master_score': ['mean', 'std', 'min', 'max', 'count'],
                            'rvol': 'mean',
                            'ret_30d': 'mean'
                        }).round(2)
                        
                        category_analysis.to_excel(writer, sheet_name='Category Analysis')
                    except Exception as e:
                        logger.warning(f"Unable to create category analysis: {str(e)}")
                
                # 5. Pattern Analysis
                pattern_data = []
                for patterns in df['patterns'].dropna():
                    if patterns:
                        for p in patterns.split(' | '):
                            pattern_data.append(p)
                
                if pattern_data:
                    pattern_df = pd.DataFrame(
                        pd.Series(pattern_data).value_counts()
                    ).reset_index()
                    pattern_df.columns = ['Pattern', 'Count']
                    pattern_df.to_excel(
                        writer, sheet_name='Pattern Analysis', index=False
                    )
                
                # 6. Wave Radar Signals
                momentum_shifts = df[
                    (df['momentum_score'] >= 50) & 
                    (df['acceleration_score'] >= 60)
                ].head(20)
                
                if len(momentum_shifts) > 0:
                    wave_cols = ['ticker', 'company_name', 'master_score', 
                                'momentum_score', 'acceleration_score', 'rvol',
                                'pe', 'eps_change_pct', 
                                'category', 'sector']
                    available_wave_cols = [col for col in wave_cols if col in momentum_shifts.columns]
                    
                    wave_df = momentum_shifts[available_wave_cols].copy()
                    
                    wave_df.to_excel(
                        writer, sheet_name='Wave Radar Signals', index=False
                    )
                
                logger.info("Excel report created successfully")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export with selected columns"""
        export_cols = [
            'rank', 'ticker', 'company_name', 'master_score',
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score',
            'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
            'from_low_pct', 'from_high_pct',
            'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
            'rvol', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d',
            'patterns', 'category', 'sector', 'eps_tier', 'pe_tier'
        ]
        
        available_cols = [col for col in export_cols if col in df.columns]
        
        # Create a copy for export
        export_df = df[available_cols].copy()
        
        # eps_change_pct and returns are already in percentage format
        # Volume ratios need to be converted back to percentage for display
        vol_ratio_cols = [col for col in export_df.columns if 'vol_ratio' in col]
        for col in vol_ratio_cols:
            export_df[col] = (export_df[col] - 1) * 100  # Convert back to percentage change
        
        return export_df.to_csv(index=False)

# ============================================
# DATA QUALITY MONITORING
# ============================================

def calculate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data quality metrics"""
    quality_metrics = {}
    
    # Completeness
    total_cells = len(df) * len(df.columns)
    filled_cells = df.notna().sum().sum()
    quality_metrics['completeness'] = (filled_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Freshness check
    if 'price' in df.columns and 'prev_close' in df.columns:
        unchanged_prices = (df['price'] == df['prev_close']).sum()
        quality_metrics['price_changes'] = len(df) - unchanged_prices
        quality_metrics['freshness'] = ((len(df) - unchanged_prices) / len(df)) * 100 if len(df) > 0 else 0
    else:
        quality_metrics['price_changes'] = 0
        quality_metrics['freshness'] = 0
    
    # Data coverage by category
    if 'pe' in df.columns:
        quality_metrics['pe_coverage'] = (df['pe'].notna() & (df['pe'] > 0) & ~np.isinf(df['pe'])).sum()
    else:
        quality_metrics['pe_coverage'] = 0
        
    if 'eps_change_pct' in df.columns:
        quality_metrics['eps_coverage'] = (df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])).sum()
    else:
        quality_metrics['eps_coverage'] = 0
    
    # Volume data coverage
    volume_cols = ['volume_1d', 'volume_30d', 'volume_90d']
    vol_coverage = 0
    vol_count = 0
    for col in volume_cols:
        if col in df.columns:
            vol_coverage += df[col].notna().sum()
            vol_count += len(df)
    quality_metrics['volume_coverage'] = (vol_coverage / vol_count * 100) if vol_count > 0 else 0
    
    # Last update time
    quality_metrics['last_update'] = datetime.now()
    
    return quality_metrics

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI - Fixed for mobile responsiveness
    st.markdown("""
    <style>
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
    /* Quick action button styling */
    div.stButton > button {
        width: 100%;
    }
    div.quick-action > button {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
    }
    div.quick-action > button:hover {
        background-color: #e0e2e6;
        border-color: #bbb;
    }
    /* Mobile responsive tables */
    @media (max-width: 768px) {
        .stDataFrame {
            font-size: 12px;
        }
        div[data-testid="metric-container"] {
            padding: 3%;
        }
    }
    /* Fix table overflow */
    .stDataFrame > div {
        overflow-x: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with gradient
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
            Professional Stock Ranking System with Wave Radar™ Early Detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'default_top_n': CONFIG.DEFAULT_TOP_N,
            'display_mode': 'Technical',
            'last_filters': {}
        }
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "sheet"
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
                time.sleep(0.5)
                st.rerun()
        
        # Data source selection - ENHANCED
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Google Sheets", "Upload CSV"],
            index=0 if st.session_state.data_source == "sheet" else 1,
            help="Load data from Google Sheets or upload your own CSV file"
        )
        st.session_state.data_source = "sheet" if data_source == "Google Sheets" else "upload"
        
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
        if 'data_quality' in st.session_state:
            with st.expander("📊 Data Quality", expanded=True):
                quality = st.session_state.data_quality
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Completeness", f"{quality.get('completeness', 0):.1f}%")
                    st.metric("PE Coverage", f"{quality.get('pe_coverage', 0):,}")
                
                with col2:
                    st.metric("Freshness", f"{quality.get('freshness', 0):.1f}%")
                    st.metric("EPS Coverage", f"{quality.get('eps_coverage', 0):,}")
                
                # Last update time
                if 'data_timestamp' in st.session_state:
                    st.caption(f"Data loaded: {st.session_state.data_timestamp.strftime('%I:%M %p')}")
        
        # Performance metrics - ENHANCED
        if 'performance_metrics' in st.session_state and st.session_state.performance_metrics:
            with st.expander("⚡ Performance Stats"):
                perf = st.session_state.performance_metrics
                
                total_time = sum(perf.values())
                st.metric("Total Load Time", f"{total_time:.2f}s")
                
                # Show slowest operations
                slowest = sorted(perf.items(), key=lambda x: x[1], reverse=True)[:3]
                for func_name, elapsed in slowest:
                    st.caption(f"{func_name}: {elapsed:.2f}s")
        
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        # Check if any filters are active and count them
        active_filter_count = 0
        filters_active = False
        
        # Count quick filter if active
        if st.session_state.get('quick_filter_applied', False):
            active_filter_count += 1
            filters_active = True
        
        if st.session_state.get('category_filter', []):
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('sector_filter', []):
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('min_score', 0) > 0:
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('patterns', []):
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('trend_filter', 'All Trends') != 'All Trends':
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('eps_tier_filter', []):
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('pe_tier_filter', []):
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('price_tier_filter', []):
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('min_eps_change', '').strip():
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('min_pe', '').strip():
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('max_pe', '').strip():
            active_filter_count += 1
            filters_active = True
        if st.session_state.get('require_fundamental_data', False):
            active_filter_count += 1
            filters_active = True
        
        # Show active filter count if any
        if filters_active:
            quick_note = " (incl. quick filter)" if st.session_state.get('quick_filter_applied', False) else ""
            st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active{quick_note}**")
        
        # Store filter count for use in main area
        st.session_state['active_filter_count'] = active_filter_count
        
        # FIXED: Clear Filters button - properly reset ALL filter values for frontend sync
        button_type = "primary" if filters_active else "secondary"
        
        # Check if clear was triggered from main content area
        if st.session_state.get('trigger_clear', False):
            st.session_state['trigger_clear'] = False
            clear_button_clicked = True
        else:
            clear_button_clicked = st.button("🗑️ Clear All Filters", use_container_width=True, type=button_type)
        
        if clear_button_clicked:
            # Reset multiselect widgets to empty lists
            multiselect_keys = [
                'category_filter', 'sector_filter', 'eps_tier_filter', 
                'pe_tier_filter', 'price_tier_filter', 'patterns'
            ]
            for key in multiselect_keys:
                if key in st.session_state:
                    st.session_state[key] = []
            
            # Reset slider to default value (0)
            if 'min_score' in st.session_state:
                st.session_state['min_score'] = 0
            
            # Reset selectbox to first option (index 0)
            if 'trend_filter' in st.session_state:
                st.session_state['trend_filter'] = 'All Trends'
            
            # Reset text inputs to empty strings
            text_input_keys = ['min_eps_change', 'min_pe', 'max_pe']
            for key in text_input_keys:
                if key in st.session_state:
                    st.session_state[key] = ""
            
            # Reset checkboxes to False
            checkbox_keys = ['require_fundamental_data', 'show_debug']
            for key in checkbox_keys:
                if key in st.session_state:
                    st.session_state[key] = False
            
            # Reset radio buttons to default
            if 'display_mode_toggle' in st.session_state:
                st.session_state['display_mode_toggle'] = "Technical"
            
            # Clear any other filter-related keys that might exist
            for key in list(st.session_state.keys()):
                if any(filter_keyword in key.lower() for filter_keyword in 
                      ['filter', 'select', 'slider', 'min_', 'max_', 'trend', 'require']):
                    # If we haven't already handled it, remove it
                    if key not in multiselect_keys and key not in text_input_keys and key not in checkbox_keys:
                        if 'multiselect' in str(type(st.session_state[key])):
                            st.session_state[key] = []
                        elif isinstance(st.session_state[key], bool):
                            st.session_state[key] = False
                        elif isinstance(st.session_state[key], str):
                            st.session_state[key] = ""
                        elif isinstance(st.session_state[key], (int, float)):
                            st.session_state[key] = 0
            
            # Reset quick filter state
            if 'quick_filter' in st.session_state:
                st.session_state['quick_filter'] = None
            if 'quick_filter_applied' in st.session_state:
                st.session_state['quick_filter_applied'] = False
            
            # Clear filter dictionary
            if 'filters' in st.session_state:
                st.session_state.filters = {}
            
            # Show success message at the top
            st.success("✅ All filters cleared!")
            st.rerun()
        
        # Debug checkbox at the bottom of sidebar - ENHANCED
        st.markdown("---")
        show_debug = st.checkbox("🐛 Show Debug Info", value=st.session_state.get('show_debug', False), key="show_debug")
    
    # Data loading and processing with smart caching
    try:
        # Check if we need to load data
        should_load_data = True
        
        if st.session_state.data_source == "upload" and uploaded_file is None:
            should_load_data = False
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        # Check if we have cached data
        if 'ranked_df' in st.session_state and (datetime.now() - st.session_state.last_refresh).seconds < 3600:
            # Use cached data
            ranked_df = st.session_state.ranked_df
            st.caption(f"Using cached data from {st.session_state.last_refresh.strftime('%I:%M %p')}")
        else:
            # Load and process data with caching
            with st.spinner("📥 Loading and processing data..."):
                try:
                    if st.session_state.data_source == "upload" and uploaded_file is not None:
                        ranked_df, data_timestamp = load_and_process_data("upload", file_data=uploaded_file)
                    else:
                        ranked_df, data_timestamp = load_and_process_data("sheet", sheet_url=CONFIG.DEFAULT_SHEET_URL, gid=CONFIG.DEFAULT_GID)
                    
                    st.session_state.ranked_df = ranked_df
                    st.session_state.data_timestamp = data_timestamp
                    st.session_state.last_refresh = datetime.now()
                    
                    # Calculate data quality
                    st.session_state.data_quality = calculate_data_quality(ranked_df)
                except Exception as e:
                    logger.error(f"Failed to load data: {str(e)}")
                    # Try to use last good data if available
                    if 'last_good_data' in st.session_state:
                        ranked_df, data_timestamp = st.session_state.last_good_data
                        st.warning("Failed to load fresh data, using cached version")
                    else:
                        raise
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
            st.info("Common issues:\n- Network connectivity\n- Google Sheets permissions\n- Data format issues\n- Invalid CSV structure")
        st.stop()
    
    # Quick Action Buttons (Top of page)
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
    # Check for quick filter state from session
    quick_filter_applied = st.session_state.get('quick_filter_applied', False)
    quick_filter = st.session_state.get('quick_filter', None)
    
    with qa_col1:
        if st.button("📈 Top Gainers", use_container_width=True):
            quick_filter = 'top_gainers'
            quick_filter_applied = True
            st.session_state['quick_filter'] = quick_filter
            st.session_state['quick_filter_applied'] = True
    
    with qa_col2:
        if st.button("🔥 Volume Surges", use_container_width=True):
            quick_filter = 'volume_surges'
            quick_filter_applied = True
            st.session_state['quick_filter'] = quick_filter
            st.session_state['quick_filter_applied'] = True
    
    with qa_col3:
        if st.button("🎯 Breakout Ready", use_container_width=True):
            quick_filter = 'breakout_ready'
            quick_filter_applied = True
            st.session_state['quick_filter'] = quick_filter
            st.session_state['quick_filter_applied'] = True
    
    with qa_col4:
        if st.button("💎 Hidden Gems", use_container_width=True):
            quick_filter = 'hidden_gems'
            quick_filter_applied = True
            st.session_state['quick_filter'] = quick_filter
            st.session_state['quick_filter_applied'] = True
    
    with qa_col5:
        if st.button("🌊 Show All", use_container_width=True):
            quick_filter = None
            quick_filter_applied = False
            st.session_state['quick_filter'] = None
            st.session_state['quick_filter_applied'] = False
    
    # Apply quick filters if clicked
    if quick_filter:
        if quick_filter == 'top_gainers':
            ranked_df_display = ranked_df[ranked_df['momentum_score'] >= CONFIG.TOP_GAINER_MOMENTUM]
            st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ {CONFIG.TOP_GAINER_MOMENTUM}")
        elif quick_filter == 'volume_surges':
            ranked_df_display = ranked_df[ranked_df['rvol'] >= CONFIG.VOLUME_SURGE_RVOL]
            st.info(f"Showing {len(ranked_df_display)} stocks with RVOL ≥ {CONFIG.VOLUME_SURGE_RVOL}x")
        elif quick_filter == 'breakout_ready':
            ranked_df_display = ranked_df[ranked_df['breakout_score'] >= CONFIG.BREAKOUT_READY_SCORE]
            st.info(f"Showing {len(ranked_df_display)} stocks with breakout score ≥ {CONFIG.BREAKOUT_READY_SCORE}")
        elif quick_filter == 'hidden_gems':
            ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('HIDDEN GEM', na=False)]
            st.info(f"Showing {len(ranked_df_display)} hidden gem stocks")
    else:
        ranked_df_display = ranked_df
    
    # Get filter options
    filter_engine = FilterEngine()
    
    # Sidebar filters with smart interconnection
    with st.sidebar:
        # Initialize filters dict
        filters = {}
        
        # Display Mode Toggle
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if st.session_state.user_preferences['display_mode'] == 'Technical' else 1,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data",
            key="display_mode_toggle"
        )
        
        # Update preference
        st.session_state.user_preferences['display_mode'] = display_mode
        
        # Store display preference
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        # Category filter with smart updates
        categories = filter_engine.get_unique_values(ranked_df_display, 'category', filters=filters)
        category_counts = ranked_df_display['category'].value_counts()
        category_options = [
            f"{cat} ({category_counts.get(cat, 0)})" 
            for cat in categories
        ]
        
        # Default to empty selection (which means "All")
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=category_options,
            default=[],  # Empty default
            placeholder="Select categories (empty = All)",
            key="category_filter"
        )
        
        # Extract actual category names
        filters['categories'] = [
            cat.split(' (')[0] for cat in selected_categories
        ] if selected_categories else []
        
        # Sector filter with smart updates based on selected categories
        sectors = filter_engine.get_unique_values(ranked_df_display, 'sector', filters=filters)
        
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=[],  # Empty default
            placeholder="Select sectors (empty = All)",
            key="sector_filter"
        )
        
        filters['sectors'] = selected_sectors if selected_sectors else []
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=st.session_state.get('min_score', 0),
            step=5,
            help="Filter stocks by minimum score",
            key="min_score"
        )
        
        # Pattern filter
        all_patterns = set()
        for patterns in ranked_df_display['patterns'].dropna():
            if patterns:
                all_patterns.update(patterns.split(' | '))
        
        if all_patterns:
            filters['patterns'] = st.multiselect(
                "Patterns",
                options=sorted(all_patterns),
                default=st.session_state.get('patterns', []),
                placeholder="Select patterns (empty = All)",
                help="Filter by specific patterns",
                key="patterns"
            )
        
        # Trend filter
        st.markdown("#### 📈 Trend Strength")
        trend_options = {
            "All Trends": (0, 100),
            "🔥 Strong Uptrend (80+)": (80, 100),
            "✅ Good Uptrend (60-79)": (60, 79),
            "➡️ Neutral Trend (40-59)": (40, 59),
            "⚠️ Weak/Downtrend (<40)": (0, 39)
        }
        
        filters['trend_filter'] = st.selectbox(
            "Trend Quality",
            options=list(trend_options.keys()),
            index=0,
            help="Filter stocks by trend strength based on SMA alignment"
        )
        filters['trend_range'] = trend_options[filters['trend_filter']]
        
        # Advanced filters in expander
        with st.expander("🔧 Advanced Filters"):
            # EPS tier filter with smart updates
            eps_tiers = filter_engine.get_unique_values(ranked_df_display, 'eps_tier', filters=filters)
            
            selected_eps_tiers = st.multiselect(
                "EPS Tier",
                options=eps_tiers,
                default=[],
                placeholder="Select EPS tiers (empty = All)",
                key="eps_tier_filter"
            )
            filters['eps_tiers'] = selected_eps_tiers if selected_eps_tiers else []
            
            # PE tier filter with smart updates
            pe_tiers = filter_engine.get_unique_values(ranked_df_display, 'pe_tier', filters=filters)
            
            selected_pe_tiers = st.multiselect(
                "PE Tier",
                options=pe_tiers,
                default=[],
                placeholder="Select PE tiers (empty = All)",
                key="pe_tier_filter"
            )
            filters['pe_tiers'] = selected_pe_tiers if selected_pe_tiers else []
            
            # Price tier filter with smart updates
            price_tiers = filter_engine.get_unique_values(ranked_df_display, 'price_tier', filters=filters)
            
            selected_price_tiers = st.multiselect(
                "Price Range",
                options=price_tiers,
                default=[],
                placeholder="Select price ranges (empty = All)",
                key="price_tier_filter"
            )
            filters['price_tiers'] = selected_price_tiers if selected_price_tiers else []
            
            # EPS change filter - FIXED: Data is already in percentage
            if 'eps_change_pct' in ranked_df_display.columns:
                eps_change_input = st.text_input(
                    "Min EPS Change %",
                    value=st.session_state.get('min_eps_change', ""),
                    placeholder="e.g. -50 or leave empty",
                    help="Enter minimum EPS growth percentage (e.g., -50 for -50% or higher), or leave empty to include all stocks",
                    key="min_eps_change"
                )
                
                if eps_change_input.strip():
                    try:
                        filters['min_eps_change'] = float(eps_change_input)
                    except ValueError:
                        st.error("Please enter a valid number for EPS change")
                        filters['min_eps_change'] = None
                else:
                    filters['min_eps_change'] = None
            
            # PE filters - Only show in hybrid mode
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_pe_input = st.text_input(
                        "Min PE Ratio",
                        value=st.session_state.get('min_pe', ""),
                        placeholder="e.g. 10",
                        help="Minimum PE ratio (leave empty for no minimum)",
                        key="min_pe"
                    )
                    
                    if min_pe_input.strip():
                        try:
                            filters['min_pe'] = float(min_pe_input)
                        except ValueError:
                            st.error("Please enter a valid number for Min PE")
                            filters['min_pe'] = None
                    else:
                        filters['min_pe'] = None
                
                with col2:
                    max_pe_input = st.text_input(
                        "Max PE Ratio",
                        value=st.session_state.get('max_pe', ""),
                        placeholder="e.g. 30",
                        help="Maximum PE ratio (leave empty for no maximum)",
                        key="max_pe"
                    )
                    
                    if max_pe_input.strip():
                        try:
                            filters['max_pe'] = float(max_pe_input)
                        except ValueError:
                            st.error("Please enter a valid number for Max PE")
                            filters['max_pe'] = None
                    else:
                        filters['max_pe'] = None
                
                # Data completeness filter
                filters['require_fundamental_data'] = st.checkbox(
                    "Only show stocks with PE and EPS data",
                    value=False,
                    help="Filter out stocks missing fundamental data"
                )
    
    # FIXED: Apply filters properly - allow combination with quick filters
    if quick_filter_applied:
        # Apply filters to the quick-filtered dataset
        filtered_df = filter_engine.apply_filters(ranked_df_display, filters)
    else:
        # Apply filters to the full dataset
        filtered_df = filter_engine.apply_filters(ranked_df, filters)
    
    filtered_df = filtered_df.sort_values('rank')
    
    # Save current filters to preferences
    st.session_state.user_preferences['last_filters'] = filters
    
    # Debug filter information - ENHANCED
    if show_debug:
        with st.sidebar.expander("🐛 Filter Debug Info", expanded=True):
            st.write("**Active Filters:**")
            
            # Show active filters with actual values
            active_filters = []
            
            # Quick filter
            if st.session_state.get('quick_filter_applied', False) and st.session_state.get('quick_filter'):
                quick_filter_names = {
                    'top_gainers': '📈 Top Gainers',
                    'volume_surges': '🔥 Volume Surges',
                    'breakout_ready': '🎯 Breakout Ready',
                    'hidden_gems': '💎 Hidden Gems'
                }
                active_filters.append(f"Quick Filter: {quick_filter_names.get(st.session_state['quick_filter'], st.session_state['quick_filter'])}")
            
            if filters.get('categories', []):
                active_filters.append(f"Categories: {', '.join(filters['categories'])}")
            if filters.get('sectors', []):
                active_filters.append(f"Sectors: {', '.join(filters['sectors'])}")
            if filters.get('min_score', 0) > 0:
                active_filters.append(f"Min Score: ≥{filters['min_score']}")
            if filters.get('patterns', []):
                active_filters.append(f"Patterns: {', '.join(filters['patterns'][:3])}{'...' if len(filters['patterns']) > 3 else ''}")
            if filters.get('trend_filter') != 'All Trends':
                active_filters.append(f"Trend: {filters['trend_filter']}")
            if filters.get('eps_tiers', []):
                active_filters.append(f"EPS Tiers: {', '.join(filters['eps_tiers'])}")
            if filters.get('pe_tiers', []):
                active_filters.append(f"PE Tiers: {', '.join(filters['pe_tiers'])}")
            if filters.get('price_tiers', []):
                active_filters.append(f"Price Tiers: {', '.join(filters['price_tiers'])}")
            if filters.get('min_eps_change') is not None:
                active_filters.append(f"Min EPS Change: ≥{filters['min_eps_change']}%")
            if show_fundamentals:
                if filters.get('min_pe') is not None:
                    active_filters.append(f"Min PE: ≥{filters['min_pe']}")
                if filters.get('max_pe') is not None:
                    active_filters.append(f"Max PE: ≤{filters['max_pe']}")
                if filters.get('require_fundamental_data', False):
                    active_filters.append("✓ Require Fundamental Data")
            
            if active_filters:
                for filter_desc in active_filters:
                    st.write(f"• {filter_desc}")
            else:
                st.write("No filters active")
            
            st.write(f"\n**Filter Result:**")
            st.write(f"Before: {len(ranked_df)} stocks")
            st.write(f"After: {len(filtered_df)} stocks")
            st.write(f"Filtered: {len(ranked_df) - len(filtered_df)} stocks ({(len(ranked_df) - len(filtered_df))/len(ranked_df)*100:.1f}%)")
            
            # Additional debug info
            st.write(f"\n**Data Source:**")
            st.write(f"Type: {st.session_state.data_source}")
            if st.session_state.data_source == "upload":
                st.write(f"File: {uploaded_file.name if uploaded_file else 'None'}")
            else:
                st.write(f"Sheet: Google Sheets")
            
            # Data quality debug
            if st.session_state.get('data_quality'):
                st.write(f"\n**Data Quality Metrics:**")
                quality = st.session_state.data_quality
                st.write(f"• Completeness: {quality.get('completeness', 0):.1f}%")
                st.write(f"• Freshness: {quality.get('freshness', 0):.1f}%")
                st.write(f"• Volume Coverage: {quality.get('volume_coverage', 0):.1f}%")
    
    # Main content area
    # Show filter status if any filters are active
    total_active_filters = st.session_state.get('active_filter_count', 0)
    if filters_active or quick_filter_applied:
        filter_status_col1, filter_status_col2 = st.columns([5, 1])
        with filter_status_col1:
            if quick_filter:
                quick_filter_display = {
                    'top_gainers': '📈 Top Gainers',
                    'volume_surges': '🔥 Volume Surges',
                    'breakout_ready': '🎯 Breakout Ready',
                    'hidden_gems': '💎 Hidden Gems'
                }
                if total_active_filters > 1:
                    st.info(f"**Viewing:** {quick_filter_display.get(quick_filter, 'Filtered')} + {total_active_filters - 1} other filter{'s' if total_active_filters > 2 else ''} | **{len(filtered_df):,} stocks** shown")
                else:
                    st.info(f"**Viewing:** {quick_filter_display.get(quick_filter, 'Filtered')} | **{len(filtered_df):,} stocks** shown")
            else:
                st.info(f"**Filtered View:** {len(filtered_df):,} stocks match your {total_active_filters} active filter{'s' if total_active_filters > 1 else ''}")
        with filter_status_col2:
            if st.button("Clear Filters", type="secondary"):
                # Trigger the main clear filters button logic
                st.session_state['trigger_clear'] = True
                st.rerun()
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = len(filtered_df)
        total_original = len(ranked_df)
        pct_of_all = (total_stocks/total_original*100) if total_original > 0 else 0
        
        st.metric(
            "Total Stocks",
            f"{total_stocks:,}",
            f"{pct_of_all:.0f}% of {total_original:,}"
        )
    
    with col2:
        if not filtered_df.empty and 'master_score' in filtered_df.columns:
            avg_score = filtered_df['master_score'].mean()
            std_score = filtered_df['master_score'].std()
            st.metric(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={std_score:.1f}"
            )
        else:
            st.metric("Avg Score", "N/A")
    
    with col3:
        if show_fundamentals and 'pe' in filtered_df.columns:
            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
            pe_coverage = valid_pe.sum()
            pe_pct = (pe_coverage / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            
            if pe_coverage > 0:
                median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                st.metric(
                    "Median PE",
                    f"{median_pe:.1f}x",
                    f"{pe_pct:.0f}% have data"
                )
            else:
                st.metric("PE Data", "Limited", "No PE data")
        else:
            if not filtered_df.empty and 'master_score' in filtered_df.columns:
                min_score = filtered_df['master_score'].min()
                max_score = filtered_df['master_score'].max()
                score_range = f"{min_score:.1f}-{max_score:.1f}"
            else:
                score_range = "N/A"
            st.metric("Score Range", score_range)
    
    with col4:
        if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
            # FIXED: Data is in percentage format
            valid_eps_change = filtered_df['eps_change_pct'].notna() & ~np.isinf(filtered_df['eps_change_pct'])
            positive_eps_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 0)
            strong_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 50)  # >50%
            mega_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 100)  # >100%
            
            growth_count = positive_eps_growth.sum()
            strong_count = strong_growth.sum()
            
            if mega_growth.sum() > 0:
                st.metric(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{strong_count} >50% | {mega_growth.sum()} >100%"
                )
            else:
                st.metric(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{valid_eps_change.sum()} have data"
                )
        else:
            if 'acceleration_score' in filtered_df.columns:
                accelerating = (filtered_df['acceleration_score'] >= 80).sum()
            else:
                accelerating = 0
            st.metric("Accelerating", f"{accelerating}")
    
    with col5:
        if 'rvol' in filtered_df.columns:
            high_rvol = (filtered_df['rvol'] > 2).sum()
        else:
            high_rvol = 0
        st.metric("High RVOL", f"{high_rvol}")
    
    with col6:
        if 'trend_quality' in filtered_df.columns:
            strong_trends = (filtered_df['trend_quality'] >= 80).sum()
            total = len(filtered_df)
            st.metric(
                "Strong Trends", 
                f"{strong_trends}",
                f"{strong_trends/total*100:.0f}%" if total > 0 else "0%"
            )
        else:
            with_patterns = (filtered_df['patterns'] != '').sum()
            st.metric("With Patterns", f"{with_patterns}")
    
    # Main tabs
    tabs = st.tabs([
        "📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"
    ])
    
    # Tab 0: Summary - ENHANCED VERSION WITH ALL FEATURES
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        
        if not filtered_df.empty:
            # 1. MARKET PULSE METRICS
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Trend Distribution
                strong_up = len(filtered_df[filtered_df['trend_quality'] >= 80])
                good_up = len(filtered_df[(filtered_df['trend_quality'] >= 60) & (filtered_df['trend_quality'] < 80)])
                neutral = len(filtered_df[(filtered_df['trend_quality'] >= 40) & (filtered_df['trend_quality'] < 60)])
                weak_down = len(filtered_df[filtered_df['trend_quality'] < 40])
                
                total_uptrend = strong_up + good_up
                total_stocks = len(filtered_df)
                uptrend_pct = (total_uptrend / total_stocks * 100) if total_stocks > 0 else 0
                
                st.metric(
                    "Market Breadth",
                    f"{total_uptrend}/{total_stocks}",
                    f"{uptrend_pct:.0f}% in uptrend"
                )
            
            with col2:
                # Average Score
                avg_score = filtered_df['master_score'].mean()
                median_score = filtered_df['master_score'].median()
                st.metric(
                    "Avg Master Score",
                    f"{avg_score:.1f}",
                    f"Median: {median_score:.1f}"
                )
            
            with col3:
                # High Momentum Count
                high_momentum = len(filtered_df[filtered_df['momentum_score'] >= 70])
                momentum_pct = (high_momentum / total_stocks * 100) if total_stocks > 0 else 0
                st.metric(
                    "High Momentum",
                    f"{high_momentum}",
                    f"{momentum_pct:.0f}% of stocks"
                )
            
            with col4:
                # Market Regime
                if strong_up > weak_down:
                    regime = "🔥 BULLISH"
                    regime_color = "inverse"
                elif weak_down > strong_up:
                    regime = "❄️ BEARISH"
                    regime_color = "normal"
                else:
                    regime = "➡️ NEUTRAL"
                    regime_color = "off"
                
                st.metric(
                    "Market Regime",
                    regime,
                    f"Score: {strong_up - weak_down}",
                    delta_color=regime_color
                )
            
            # 2. TREND BREAKDOWN
            st.markdown("#### 📈 Trend Analysis")
            
            trend_col1, trend_col2 = st.columns([2, 1])
            
            with trend_col1:
                # Trend distribution chart
                trend_data = pd.DataFrame({
                    'Trend': ['🔥 Strong Up', '✅ Good Up', '➡️ Neutral', '⚠️ Weak/Down'],
                    'Count': [strong_up, good_up, neutral, weak_down],
                    'Percentage': [
                        strong_up/total_stocks*100,
                        good_up/total_stocks*100,
                        neutral/total_stocks*100,
                        weak_down/total_stocks*100
                    ]
                })
                
                fig_trend = px.bar(
                    trend_data,
                    x='Count',
                    y='Trend',
                    orientation='h',
                    text='Count',
                    title='Trend Distribution',
                    color='Trend',
                    color_discrete_map={
                        '🔥 Strong Up': '#2ecc71',
                        '✅ Good Up': '#3498db',
                        '➡️ Neutral': '#f39c12',
                        '⚠️ Weak/Down': '#e74c3c'
                    }
                )
                fig_trend.update_layout(
                    height=300,
                    showlegend=False,
                    template='plotly_white'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with trend_col2:
                # Trend Leaders
                st.markdown("**🏆 Trend Leaders**")
                
                # Strong uptrend leaders
                strong_leaders = filtered_df[filtered_df['trend_quality'] >= 80].nlargest(5, 'master_score')
                if len(strong_leaders) > 0:
                    st.markdown("**🔥 Strong Uptrend:**")
                    for _, stock in strong_leaders.iterrows():
                        st.write(f"• {stock['ticker']} ({stock['master_score']:.1f})")
                
                # Pattern count stats
                st.markdown("**📊 Pattern Stats**")
                multi_pattern = len(filtered_df[filtered_df['patterns'].str.count('\\|') >= 2])
                st.write(f"• Multi-Pattern: {multi_pattern}")
                
                high_rvol = len(filtered_df[filtered_df['rvol'] > 3])
                st.write(f"• High RVOL (>3x): {high_rvol}")
            
            # 3. TOP OPPORTUNITIES
            st.markdown("#### 🎯 Today's Best Opportunities")
            
            opp_col1, opp_col2, opp_col3 = st.columns(3)
            
            with opp_col1:
                # Strong Trend + Breakout
                strong_breakout = filtered_df[
                    (filtered_df['trend_quality'] >= 80) & 
                    (filtered_df['breakout_score'] >= 80)
                ].nlargest(5, 'master_score')
                
                st.markdown("**🚀 Trend + Breakout**")
                if len(strong_breakout) > 0:
                    for _, stock in strong_breakout.iterrows():
                        st.write(f"• {stock['ticker']} - {stock['company_name'][:20]}")
                        st.caption(f"  Score: {stock['master_score']:.1f} | Trend: {stock['trend_quality']:.0f}")
                else:
                    st.info("No stocks match criteria")
            
            with opp_col2:
                # New Pattern: 52W High Approach
                if '52W HIGH APPROACH' in filtered_df['patterns'].str.cat():
                    high_approach = filtered_df[
                        filtered_df['patterns'].str.contains('52W HIGH APPROACH', na=False)
                    ].nlargest(5, 'master_score')
                    
                    st.markdown("**🎯 Near 52W High**")
                    if len(high_approach) > 0:
                        for _, stock in high_approach.iterrows():
                            st.write(f"• {stock['ticker']} - {stock['company_name'][:20]}")
                            st.caption(f"  From High: {stock['from_high_pct']:.1f}%")
                    else:
                        st.info("No stocks near 52W high")
                else:
                    st.info("No 52W High Approach patterns")
            
            with opp_col3:
                # Volume Accumulation
                if 'VOL ACCUMULATION' in filtered_df['patterns'].str.cat():
                    vol_accum = filtered_df[
                        filtered_df['patterns'].str.contains('VOL ACCUMULATION', na=False)
                    ].nlargest(5, 'master_score')
                    
                    st.markdown("**📊 Volume Accumulation**")
                    if len(vol_accum) > 0:
                        for _, stock in vol_accum.iterrows():
                            st.write(f"• {stock['ticker']} - {stock['company_name'][:20]}")
                            st.caption(f"  RVOL: {stock['rvol']:.1f}x")
                    else:
                        st.info("No volume accumulation")
                else:
                    st.info("No volume accumulation patterns")
            
            # 4. RISK INDICATORS - ENHANCED
            st.markdown("#### ⚠️ Risk Indicators")
            
            risk_cols = st.columns(4)
            
            with risk_cols[0]:
                # Overextended stocks
                if 'from_high_pct' in filtered_df.columns and 'momentum_score' in filtered_df.columns:
                    at_highs_risky = len(filtered_df[
                        (filtered_df['from_high_pct'] >= 0) & 
                        (filtered_df['momentum_score'] < 50)
                    ])
                    st.metric("Overextended", f"{at_highs_risky}", "At high, low momentum")
                else:
                    st.metric("Overextended", "N/A")
            
            with risk_cols[1]:
                # Extreme RVOL (pump risk)
                if 'rvol' in filtered_df.columns and 'master_score' in filtered_df.columns:
                    extreme_rvol_risky = len(filtered_df[
                        (filtered_df['rvol'] > 10) & 
                        (filtered_df['master_score'] < 50)
                    ])
                    st.metric("Pump Risk", f"{extreme_rvol_risky}", "High RVOL, low score")
                else:
                    st.metric("Pump Risk", "N/A")
            
            with risk_cols[2]:
                # Downtrend count
                if 'trend_quality' in filtered_df.columns:
                    downtrend = len(filtered_df[filtered_df['trend_quality'] < 40])
                    st.metric("Downtrends", f"{downtrend}")
                else:
                    st.metric("Downtrends", "N/A")
            
            with risk_cols[3]:
                # Overall risk score
                risk_factors = 0
                if 'at_highs_risky' in locals() and at_highs_risky > 20: 
                    risk_factors += 1
                if 'downtrend' in locals() and downtrend > total_stocks * 0.3: 
                    risk_factors += 1
                if 'extreme_rvol_risky' in locals() and extreme_rvol_risky > 10: 
                    risk_factors += 1
                
                risk_level = ["LOW", "MODERATE", "HIGH", "EXTREME"][min(risk_factors, 3)]
                risk_color = ["🟢", "🟡", "🟠", "🔴"][min(risk_factors, 3)]
                
                st.metric("Risk Level", f"{risk_color} {risk_level}", f"{risk_factors} factors")
            
            # 5. KEY INSIGHTS
            st.markdown("#### 💡 Key Market Insights")
            
            insights = []
            
            # Trend insight
            if uptrend_pct > 60:
                insights.append(f"🔥 {uptrend_pct:.0f}% stocks in uptrend - STRONG BULL MARKET!")
            elif uptrend_pct < 40:
                insights.append(f"⚠️ Only {uptrend_pct:.0f}% in uptrend - DEFENSIVE MARKET")
            
            # Momentum insight
            if high_momentum > 100:
                insights.append(f"⚡ {high_momentum} stocks with high momentum - ACTIVE MARKET!")
            
            # Pattern insights
            if multi_pattern > 20:
                insights.append(f"💎 {multi_pattern} stocks showing multiple patterns - OPPORTUNITIES ABOUND!")
            
            # New pattern insights
            new_patterns_count = 0
            for pattern in ['52W HIGH APPROACH', '52W LOW BOUNCE', 'GOLDEN ZONE', 'VOL ACCUMULATION']:
                if pattern in filtered_df['patterns'].str.cat():
                    new_patterns_count += len(filtered_df[filtered_df['patterns'].str.contains(pattern, na=False)])
            
            if new_patterns_count > 10:
                insights.append(f"🎯 {new_patterns_count} stocks with new advanced patterns detected!")
            
            # Display insights
            for insight in insights:
                st.info(insight)
            
            # 6. DOWNLOAD CLEAN DATA SECTION - ENHANCED
            st.markdown("---")
            st.markdown("#### 💾 Download Clean Processed Data")
            
            download_cols = st.columns(3)
            
            with download_cols[0]:
                # Download filtered data
                st.markdown("**📊 Current View Data**")
                st.write(f"Includes {len(filtered_df)} stocks matching current filters")
                
                csv_filtered = ExportEngine.create_csv_export(filtered_df)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_filtered,
                    file_name=f"wave_detection_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download currently filtered stocks with all scores and indicators"
                )
            
            with download_cols[1]:
                # Download top 100
                st.markdown("**🏆 Top 100 Stocks**")
                st.write("Elite stocks ranked by Master Score")
                
                top_100 = filtered_df.nlargest(100, 'master_score')
                csv_top100 = ExportEngine.create_csv_export(top_100)
                st.download_button(
                    label="📥 Download Top 100 (CSV)",
                    data=csv_top100,
                    file_name=f"wave_detection_top100_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download top 100 stocks by Master Score"
                )
            
            with download_cols[2]:
                # Download patterns only
                st.markdown("**🎯 Pattern Stocks Only**")
                pattern_stocks = filtered_df[filtered_df['patterns'] != '']
                st.write(f"Includes {len(pattern_stocks)} stocks with patterns")
                
                if len(pattern_stocks) > 0:
                    csv_patterns = ExportEngine.create_csv_export(pattern_stocks)
                    st.download_button(
                        label="📥 Download Pattern Stocks (CSV)",
                        data=csv_patterns,
                        file_name=f"wave_detection_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download only stocks showing patterns"
                    )
                else:
                    st.info("No stocks with patterns in current filter")
            
            # 7. QUICK STATS TABLE
            st.markdown("---")
            st.markdown("#### 📊 Quick Statistics")
            
            # Create comprehensive stats
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.markdown("**Returns**")
                positive_1d = len(filtered_df[filtered_df['ret_1d'] > 0])
                positive_30d = len(filtered_df[filtered_df['ret_30d'] > 0])
                st.write(f"1D Positive: {positive_1d}")
                st.write(f"30D Positive: {positive_30d}")
            
            with stats_col2:
                st.markdown("**Volume**")
                avg_rvol = filtered_df['rvol'].mean()
                high_vol = len(filtered_df[filtered_df['rvol'] > 2])
                st.write(f"Avg RVOL: {avg_rvol:.2f}x")
                st.write(f"High Vol (>2x): {high_vol}")
            
            with stats_col3:
                st.markdown("**Categories**")
                top_cat = filtered_df['category'].value_counts().head(2)
                for cat, count in top_cat.items():
                    st.write(f"{cat}: {count}")
            
            with stats_col4:
                st.markdown("**Patterns**")
                total_patterns = filtered_df['patterns'].str.count('\\|').sum() + len(filtered_df[filtered_df['patterns'] != ''])
                avg_patterns = total_patterns / len(filtered_df) if len(filtered_df) > 0 else 0
                st.write(f"Total: {total_patterns}")
                st.write(f"Avg/Stock: {avg_patterns:.1f}")
        
        else:
            st.warning("No data available for summary. Please adjust filters.")
    
    # Tab 1: Rankings
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        
        # Display options
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n'])
            )
            # Update preference
            st.session_state.user_preferences['default_top_n'] = display_count
        
        with col2:
            sort_options = ['Rank', 'Master Score', 'RVOL', 'Momentum']
            if 'trend_quality' in filtered_df.columns:
                sort_options.append('Trend')
            
            sort_by = st.selectbox(
                "Sort by",
                options=sort_options,
                index=0
            )
        
        # Get display data
        display_df = filtered_df.head(display_count).copy()
        
        # Apply sorting
        if sort_by == 'Master Score':
            display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL':
            display_df = display_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum':
            display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Trend' and 'trend_quality' in display_df.columns:
            display_df = display_df.sort_values('trend_quality', ascending=False)
        
        if not display_df.empty:
            # Add trend indicator column if trend_quality exists
            if 'trend_quality' in display_df.columns:
                def get_trend_indicator(score):
                    if pd.isna(score):
                        return "➖"
                    elif score >= 80:
                        return "🔥"
                    elif score >= 60:
                        return "✅"
                    elif score >= 40:
                        return "➡️"
                    else:
                        return "⚠️"
                
                display_df['trend_indicator'] = display_df['trend_quality'].apply(get_trend_indicator)
            
            # Prepare display columns
            display_cols = {
                'rank': 'Rank',
                'ticker': 'Ticker',
                'company_name': 'Company',
                'master_score': 'Score'
            }
            
            if 'trend_indicator' in display_df.columns:
                display_cols['trend_indicator'] = 'Trend'
            
            display_cols['price'] = 'Price'
            
            # Add fundamental columns if enabled
            if show_fundamentals:
                if 'pe' in display_df.columns:
                    display_cols['pe'] = 'PE'
                
                if 'eps_change_pct' in display_df.columns:
                    display_cols['eps_change_pct'] = 'EPS Δ%'
            
            # Add remaining columns
            display_cols.update({
                'from_low_pct': 'From Low',
                'ret_30d': '30D Ret',
                'rvol': 'RVOL',
                'patterns': 'Patterns',
                'category': 'Category',
                'sector': 'Sector'
            })
            
            # Format numeric columns
            format_rules = {
                'master_score': '{:.1f}',
                'price': '₹{:,.0f}',
                'from_low_pct': '{:.0f}%',
                'ret_30d': '{:+.1f}%',
                'rvol': '{:.1f}x'
            }
            
            # Smart PE formatting function - FIXED
            def format_pe(value):
                try:
                    if pd.isna(value) or value == 'N/A' or value == '':
                        return '-'
                    
                    val = float(value)
                    
                    if val <= 0:
                        return 'Loss'
                    elif np.isinf(val):
                        return '∞'
                    elif val > 10000:
                        return f"{val/1000:.0f}K"
                    elif val > 1000:
                        return f"{val:.0f}"
                    elif val > 100:
                        return f"{val:.1f}"
                    else:
                        return f"{val:.1f}"
                except (ValueError, TypeError, OverflowError):
                    return '-'
            
            # FIXED: Smart EPS change formatting function for percentage data
            def format_eps_change(value):
                try:
                    if pd.isna(value) or value == 'N/A' or value == '':
                        return '-'
                    
                    # Value is already in percentage (e.g., -56.61 for -56.61%)
                    val = float(value)
                    
                    if np.isinf(val):
                        return '∞' if val > 0 else '-∞'
                    
                    if abs(val) >= 10000:
                        return f"{val/1000:+.1f}K%"
                    elif abs(val) >= 1000:
                        return f"{val:+.0f}%"
                    elif abs(val) >= 100:
                        return f"{val:+.1f}%"
                    elif abs(val) >= 10:
                        return f"{val:+.1f}%"
                    elif abs(val) >= 0.1:
                        return f"{val:+.1f}%"
                    else:
                        return f"{val:+.2f}%"
                        
                except (ValueError, TypeError, OverflowError):
                    return '-'
            
            # Apply formatting
            for col, fmt in format_rules.items():
                if col in display_df.columns:
                    try:
                        if col == 'ret_30d':
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:+.1f}%" if pd.notna(x) and x != 'N/A' and isinstance(x, (int, float)) else '-'
                            )
                        else:
                            display_df[col] = display_df[col].apply(
                                lambda x: fmt.format(x) if pd.notna(x) and x != 'N/A' and isinstance(x, (int, float)) else '-'
                            )
                    except Exception as e:
                        logger.warning(f"Error formatting {col}: {str(e)}")
                        display_df[col] = display_df[col].fillna('-')
            
            # Apply special formatting for fundamentals when enabled
            if show_fundamentals:
                if 'pe' in display_df.columns:
                    display_df['pe'] = display_df['pe'].apply(format_pe)
                
                if 'eps_change_pct' in display_df.columns:
                    display_df['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change)
            
            # Rename columns for display
            display_df = display_df[[c for c in display_cols.keys() if c in display_df.columns]]
            display_df.columns = [display_cols[c] for c in display_df.columns]
            
            # Display with styling
            st.dataframe(
                display_df,
                use_container_width=True,
                height=min(600, len(display_df) * 35 + 50),
                hide_index=True
            )
            
            # Quick stats below table
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown("**Score Distribution**")
                    if 'master_score' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['master_score'].max():.1f}")
                        st.text(f"Min: {filtered_df['master_score'].min():.1f}")
                        st.text(f"Std: {filtered_df['master_score'].std():.1f}")
                    else:
                        st.text("No score data available")
                
                with stat_cols[1]:
                    st.markdown("**Returns (30D)**")
                    if 'ret_30d' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['ret_30d'].max():.1f}%")
                        st.text(f"Avg: {filtered_df['ret_30d'].mean():.1f}%")
                        st.text(f"Positive: {(filtered_df['ret_30d'] > 0).sum()}")
                    else:
                        st.text("No 30D return data available")
                
                with stat_cols[2]:
                    if show_fundamentals:
                        st.markdown("**Fundamentals**")
                        if 'pe' in filtered_df.columns:
                            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
                            if valid_pe.any():
                                median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                                q1_pe = filtered_df.loc[valid_pe, 'pe'].quantile(0.25)
                                q3_pe = filtered_df.loc[valid_pe, 'pe'].quantile(0.75)
                                
                                st.text(f"Median PE: {median_pe:.1f}x")
                                st.text(f"PE Range: {q1_pe:.1f}-{q3_pe:.1f}")
                            else:
                                st.text("PE: No valid data")
                        else:
                            st.text("PE: No data")
                        
                        if 'eps_change_pct' in filtered_df.columns:
                            valid_eps = filtered_df['eps_change_pct'].notna() & ~np.isinf(filtered_df['eps_change_pct'])
                            if valid_eps.any():
                                eps_data = filtered_df.loc[valid_eps, 'eps_change_pct']
                                
                                # FIXED: Data is in percentage format
                                mega_growth = (eps_data > 100).sum()  # >100%
                                strong_growth = ((eps_data > 50) & (eps_data <= 100)).sum()  # 50-100%
                                moderate_growth = ((eps_data > 0) & (eps_data <= 50)).sum()  # 0-50%
                                declining = (eps_data < 0).sum()
                                
                                if mega_growth > 0:
                                    st.text(f">100%: {mega_growth} stocks")
                                st.text(f"Positive: {moderate_growth + strong_growth + mega_growth}")
                                st.text(f"Negative: {declining}")
                            else:
                                st.text("EPS Growth: N/A")
                        else:
                            st.text("EPS: No data")
                    else:
                        st.markdown("**RVOL Stats**")
                        if 'rvol' in filtered_df.columns:
                            st.text(f"Max: {filtered_df['rvol'].max():.1f}x")
                            st.text(f"Avg: {filtered_df['rvol'].mean():.1f}x")
                            st.text(f">2x: {(filtered_df['rvol'] > 2).sum()}")
                        else:
                            st.text("No RVOL data available")
                
                with stat_cols[3]:
                    st.markdown("**Categories**")
                    if 'category' in filtered_df.columns:
                        for cat, count in filtered_df['category'].value_counts().head(3).items():
                            st.text(f"{cat}: {count}")
                    else:
                        st.text("No category data available")
        
        else:
            st.warning("No stocks match the selected filters.")
    
    # Tab 2: Wave Radar
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        st.markdown("*Catch waves as they form, not after they've peaked!*")
        
        # Wave Radar Controls
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        
        with radar_col1:
            # Enhanced timeframe options with intelligent filtering
            wave_timeframe = st.selectbox(
                "Wave Detection Timeframe",
                options=[
                    "All Waves",
                    "Intraday Surge",
                    "3-Day Buildup", 
                    "Weekly Breakout",
                    "Monthly Trend"
                ],
                index=0,
                help="""
                🌊 All Waves: Complete unfiltered view
                ⚡ Intraday Surge: High RVOL & today's movers
                📈 3-Day Buildup: Building momentum patterns
                🚀 Weekly Breakout: Near 52w highs with volume
                💪 Monthly Trend: Established trends with SMAs
                """
            )
        
        with radar_col2:
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=["Conservative", "Balanced", "Aggressive"],
                value="Balanced",
                help="Conservative = Stronger signals, Aggressive = More signals"
            )
            
            # Sensitivity details toggle
            show_sensitivity_details = st.checkbox(
                "Show thresholds",
                value=False,
                help="Display exact threshold values for current sensitivity"
            )
        
        with radar_col3:
            # Market Regime Toggle
            show_market_regime = st.checkbox(
                "📊 Market Regime Analysis",
                value=True,
                help="Show category rotation flow and market regime detection"
            )
        
        # Initialize wave_filtered_df before using it
        wave_filtered_df = filtered_df.copy()
        
        with radar_col4:
            # Calculate overall Wave Strength
            if not wave_filtered_df.empty:
                try:
                    momentum_count = len(wave_filtered_df[wave_filtered_df['momentum_score'] >= 60]) if 'momentum_score' in wave_filtered_df.columns else 0
                    accel_count = len(wave_filtered_df[wave_filtered_df['acceleration_score'] >= 70]) if 'acceleration_score' in wave_filtered_df.columns else 0
                    rvol_count = len(wave_filtered_df[wave_filtered_df['rvol'] >= 2]) if 'rvol' in wave_filtered_df.columns else 0
                    breakout_count = len(wave_filtered_df[wave_filtered_df['breakout_score'] >= 70]) if 'breakout_score' in wave_filtered_df.columns else 0
                    
                    total_stocks = len(wave_filtered_df)
                    if total_stocks > 0:
                        wave_strength = (
                            momentum_count * 0.3 +
                            accel_count * 0.3 +
                            rvol_count * 0.2 +
                            breakout_count * 0.2
                        ) / total_stocks * 100
                    else:
                        wave_strength = 0
                    
                    if wave_strength > 20:
                        wave_emoji = "🌊🔥"
                        wave_color = "🟢"
                    elif wave_strength > 10:
                        wave_emoji = "🌊"
                        wave_color = "🟡"
                    else:
                        wave_emoji = "💤"
                        wave_color = "🔴"
                    
                    # If market regime is hidden, still calculate and show regime in metric
                    regime_indicator = ""
                    if not show_market_regime and not wave_filtered_df.empty and 'category' in wave_filtered_df.columns:
                        # Quick regime calculation for display
                        try:
                            quick_flow = wave_filtered_df.groupby('category')['master_score'].mean()
                            if len(quick_flow) > 0:
                                top_category = quick_flow.idxmax()
                                if 'Small' in top_category or 'Micro' in top_category:
                                    regime_indicator = " | Risk-ON"
                                elif 'Large' in top_category or 'Mega' in top_category:
                                    regime_indicator = " | Risk-OFF"
                        except:
                            pass
                    
                    st.metric(
                        "Wave Strength",
                        f"{wave_emoji} {wave_strength:.0f}%",
                        f"{wave_color} Market{regime_indicator}"
                    )
                except Exception as e:
                    logger.error(f"Error calculating wave strength: {str(e)}")
                    st.metric("Wave Strength", "N/A", "Error")
        
        # Display sensitivity thresholds if enabled
        if show_sensitivity_details:
            with st.expander("📊 Current Sensitivity Thresholds", expanded=True):
                if sensitivity == "Conservative":
                    st.markdown("""
                    **Conservative Settings** 🛡️
                    - **Momentum Shifts:** Score ≥ 60, Acceleration ≥ 70
                    - **Emerging Patterns:** Within 5% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 3.0x (extreme volumes only)
                    - **Acceleration Alerts:** Score ≥ 85 (strongest signals)
                    """)
                elif sensitivity == "Balanced":
                    st.markdown("""
                    **Balanced Settings** ⚖️
                    - **Momentum Shifts:** Score ≥ 50, Acceleration ≥ 60
                    - **Emerging Patterns:** Within 10% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 2.0x (standard threshold)
                    - **Acceleration Alerts:** Score ≥ 70 (good acceleration)
                    """)
                else:  # Aggressive
                    st.markdown("""
                    **Aggressive Settings** 🚀
                    - **Momentum Shifts:** Score ≥ 40, Acceleration ≥ 50
                    - **Emerging Patterns:** Within 15% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 1.5x (building volume)
                    - **Acceleration Alerts:** Score ≥ 60 (early signals)
                    """)
        
        # Apply intelligent timeframe filtering to the already initialized wave_filtered_df
        if wave_timeframe != "All Waves":
            try:
                if wave_timeframe == "Intraday Surge":
                    # Focus on today's high volume movers
                    if all(col in wave_filtered_df.columns for col in ['rvol', 'ret_1d', 'price', 'prev_close']):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['rvol'] >= 2.5) &
                            (wave_filtered_df['ret_1d'] > 2) &
                            (wave_filtered_df['price'] > wave_filtered_df['prev_close'] * 1.02)
                        ]
                    
                elif wave_timeframe == "3-Day Buildup":
                    # Stocks building momentum over 3 days
                    if all(col in wave_filtered_df.columns for col in ['ret_3d', 'vol_ratio_7d_90d', 'price', 'sma_20d']):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_3d'] > 5) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 1.5) &
                            (wave_filtered_df['price'] > wave_filtered_df['sma_20d'])
                        ]
                    
                elif wave_timeframe == "Weekly Breakout":
                    # Stocks near highs with strong weekly momentum
                    if all(col in wave_filtered_df.columns for col in ['ret_7d', 'vol_ratio_7d_90d', 'from_high_pct']):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_7d'] > 8) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 2.0) &
                            (wave_filtered_df['from_high_pct'] > -10)
                        ]
                    
                elif wave_timeframe == "Monthly Trend":
                    # Established trends with technical confirmation
                    if all(col in wave_filtered_df.columns for col in ['ret_30d', 'price', 'sma_20d', 'sma_50d', 'vol_ratio_30d_180d', 'from_low_pct']):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_30d'] > 15) &
                            (wave_filtered_df['price'] > wave_filtered_df['sma_20d']) &
                            (wave_filtered_df['sma_20d'] > wave_filtered_df['sma_50d']) &
                            (wave_filtered_df['vol_ratio_30d_180d'] > 1.2) &
                            (wave_filtered_df['from_low_pct'] > 30)
                        ]
                        
            except KeyError as e:
                logger.warning(f"Column missing for {wave_timeframe} filter: {str(e)}")
                st.warning(f"Some data not available for {wave_timeframe} filter")
                # Reset to original filtered data if timeframe filtering fails
                wave_filtered_df = filtered_df.copy()
        
        # Wave Radar Analysis sections
        if not wave_filtered_df.empty:
            # 1. MOMENTUM SHIFT DETECTION
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            
            # Calculate momentum shifts
            momentum_shifts = wave_filtered_df.copy()
            
            # Identify crossing points based on sensitivity
            if sensitivity == "Conservative":
                cross_threshold = 60
                min_acceleration = 70
                min_rvol_threshold = 3.0
                acceleration_alert_threshold = 85
            elif sensitivity == "Balanced":
                cross_threshold = 50
                min_acceleration = 60
                min_rvol_threshold = 2.0
                acceleration_alert_threshold = 70
            else:  # Aggressive
                cross_threshold = 40
                min_acceleration = 50
                min_rvol_threshold = 1.5
                acceleration_alert_threshold = 60
            
            # Find stocks crossing into strength
            if 'ret_30d' in momentum_shifts.columns:
                median_return = momentum_shifts['ret_30d'].median()
                return_condition = momentum_shifts['ret_30d'] > median_return
            elif 'ret_7d' in momentum_shifts.columns:
                median_return = momentum_shifts['ret_7d'].median()
                return_condition = momentum_shifts['ret_7d'] > median_return
            else:
                return_condition = pd.Series(True, index=momentum_shifts.index)
            
            momentum_shifts['momentum_shift'] = (
                (momentum_shifts['momentum_score'] >= cross_threshold) & 
                (momentum_shifts['acceleration_score'] >= min_acceleration) &
                return_condition
            )
            
            # Calculate multi-signal count for each stock
            momentum_shifts['signal_count'] = 0
            
            # Signal 1: Momentum shift
            momentum_shifts.loc[momentum_shifts['momentum_shift'], 'signal_count'] += 1
            
            # Signal 2: High RVOL
            if 'rvol' in momentum_shifts.columns:
                momentum_shifts.loc[momentum_shifts['rvol'] >= min_rvol_threshold, 'signal_count'] += 1
            
            # Signal 3: Strong acceleration
            momentum_shifts.loc[momentum_shifts['acceleration_score'] >= acceleration_alert_threshold, 'signal_count'] += 1
            
            # Signal 4: Volume surge
            if 'vol_ratio_7d_90d' in momentum_shifts.columns:
                momentum_shifts.loc[momentum_shifts['vol_ratio_7d_90d'] >= 1.5, 'signal_count'] += 1
            
            # Signal 5: Breakout ready
            if 'breakout_score' in momentum_shifts.columns:
                momentum_shifts.loc[momentum_shifts['breakout_score'] >= 75, 'signal_count'] += 1
            
            # Calculate shift strength
            momentum_shifts['shift_strength'] = (
                momentum_shifts['momentum_score'] * 0.4 +
                momentum_shifts['acceleration_score'] * 0.4 +
                momentum_shifts['rvol_score'] * 0.2
            )
            
            # Get top momentum shifts
            top_shifts = momentum_shifts[momentum_shifts['momentum_shift']].nlargest(20, 'shift_strength')
            
            if len(top_shifts) > 0:
                # Sort by signal count first, then by shift strength
                top_shifts = top_shifts.sort_values(['signal_count', 'shift_strength'], ascending=[False, False])
                
                # Select available columns for display
                display_columns = ['ticker', 'company_name', 'master_score', 'momentum_score', 
                                 'acceleration_score', 'rvol', 'signal_count']
                
                if 'ret_7d' in top_shifts.columns:
                    display_columns.insert(-1, 'ret_7d')
                
                display_columns.append('category')
                
                shift_display = top_shifts[display_columns].copy()
                
                # Add shift indicators with multi-signal emoji
                shift_display['Signal'] = shift_display.apply(
                    lambda x: f"{'🔥' * min(x['signal_count'], 3)} {x['signal_count']}/5", axis=1
                )
                
                # Format for display
                if 'ret_7d' in shift_display.columns:
                    shift_display['ret_7d'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
                
                shift_display['rvol'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x")
                
                # Format signal count for display
                shift_display['signal_count'] = shift_display['signal_count'].apply(
                    lambda x: f"{x} {'🏆' if x >= 4 else '✨' if x >= 3 else ''}"
                )
                
                # Rename columns
                rename_dict = {
                    'ticker': 'Ticker',
                    'company_name': 'Company',
                    'master_score': 'Score',
                    'momentum_score': 'Momentum',
                    'acceleration_score': 'Acceleration',
                    'rvol': 'RVOL',
                    'signal_count': 'Signals',
                    'category': 'Category'
                }
                
                if 'ret_7d' in shift_display.columns:
                    rename_dict['ret_7d'] = '7D Return'
                
                shift_display = shift_display.rename(columns=rename_dict)
                
                st.dataframe(
                    shift_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show multi-signal leaders summary
                multi_signal_leaders = top_shifts[top_shifts['signal_count'] >= 3]
                if len(multi_signal_leaders) > 0:
                    st.success(f"🏆 Found {len(multi_signal_leaders)} stocks with 3+ signals (strongest momentum)")
            else:
                st.info(f"No momentum shifts detected in {wave_timeframe} timeframe. Try 'Aggressive' sensitivity or 'All Waves' timeframe.")
            
            # 2. CATEGORY ROTATION FLOW
            if show_market_regime:
                st.markdown("#### 💰 Category Rotation - Smart Money Flow")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Calculate category performance
                    try:
                        if not wave_filtered_df.empty and 'category' in wave_filtered_df.columns:
                            category_flow = wave_filtered_df.groupby('category').agg({
                                'master_score': ['mean', 'count'],
                                'momentum_score': 'mean',
                                'volume_score': 'mean',
                                'rvol': 'mean'
                            }).round(2)
                            
                            if not category_flow.empty:
                                category_flow.columns = ['Avg Score', 'Count', 'Avg Momentum', 'Avg Volume', 'Avg RVOL']
                                category_flow['Flow Score'] = (
                                    category_flow['Avg Score'] * 0.4 +
                                    category_flow['Avg Momentum'] * 0.3 +
                                    category_flow['Avg Volume'] * 0.3
                                )
                                
                                # Determine flow direction
                                category_flow = category_flow.sort_values('Flow Score', ascending=False)
                                if len(category_flow) > 0:
                                    top_category = category_flow.index[0]
                                    if 'Small' in top_category or 'Micro' in top_category:
                                        flow_direction = "🔥 Risk-ON"
                                    elif 'Large' in top_category or 'Mega' in top_category:
                                        flow_direction = "❄️ Risk-OFF"
                                    else:
                                        flow_direction = "➡️ Neutral"
                                else:
                                    flow_direction = "➡️ Neutral"
                                
                                # Create flow visualization
                                fig_flow = go.Figure()
                                
                                fig_flow.add_trace(go.Bar(
                                    x=category_flow.index,
                                    y=category_flow['Flow Score'],
                                    text=[f"{val:.1f}" for val in category_flow['Flow Score']],
                                    textposition='outside',
                                    marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' 
                                                 for score in category_flow['Flow Score']],
                                    hovertemplate='Category: %{x}<br>Flow Score: %{y:.1f}<br>Stocks: %{customdata}<extra></extra>',
                                    customdata=category_flow['Count']
                                ))
                                
                                fig_flow.update_layout(
                                    title=f"Smart Money Flow Direction: {flow_direction}",
                                    xaxis_title="Market Cap Category",
                                    yaxis_title="Flow Score",
                                    height=300,
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig_flow, use_container_width=True)
                            else:
                                st.info("Insufficient data for category flow analysis")
                                flow_direction = "➡️ Neutral"
                                category_flow = pd.DataFrame()
                        else:
                            st.info("Category data not available")
                            flow_direction = "➡️ Neutral"
                            category_flow = pd.DataFrame()
                            
                    except Exception as e:
                        logger.error(f"Error in category flow analysis: {str(e)}")
                        st.error("Unable to analyze category flow")
                        flow_direction = "➡️ Neutral"
                        category_flow = pd.DataFrame()
                
                with col2:
                    st.markdown(f"**🎯 Market Regime: {flow_direction}**")
                    
                    # Top categories
                    st.markdown("**💎 Strongest Categories:**")
                    if not category_flow.empty:
                        for i, (cat, row) in enumerate(category_flow.head(3).iterrows()):
                            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            try:
                                st.write(f"{emoji} **{cat}**: Score {row['Flow Score']:.1f}")
                            except:
                                st.write(f"{emoji} **{cat}**: Score N/A")
                    else:
                        st.info("No category data available")
                    
                    # Category shift detection
                    st.markdown("**🔄 Category Shifts:**")
                    if not category_flow.empty:
                        # Look for categories containing 'Small' and 'Large'
                        small_cats = [cat for cat in category_flow.index if 'Small' in cat]
                        large_cats = [cat for cat in category_flow.index if 'Large' in cat]
                        
                        if small_cats and large_cats:
                            try:
                                # Use the first matching category
                                small_score = category_flow.loc[small_cats[0], 'Flow Score']
                                large_score = category_flow.loc[large_cats[0], 'Flow Score']
                                
                                # Better regime detection logic
                                score_diff = abs(small_score - large_score)
                                if score_diff < 5:  # Within 5 points
                                    st.info("➡️ Balanced Market - No Clear Leader")
                                elif small_score > large_score + 10:  # Small caps leading by 10+
                                    st.success("📈 Small Caps Leading - Early Bull Signal!")
                                elif large_score > small_score + 10:  # Large caps leading by 10+
                                    st.warning("📉 Large Caps Leading - Defensive Mode")
                                else:
                                    # Small difference
                                    if small_score > large_score:
                                        st.info("📊 Small caps slightly ahead")
                                    else:
                                        st.info("📊 Large caps slightly ahead")
                            except Exception as e:
                                logger.error(f"Error in category shift detection: {str(e)}")
                                st.info("Unable to determine category shifts")
                        else:
                            st.info("Need both Small and Large cap categories for shift analysis")
                    else:
                        st.info("Insufficient data for category shift analysis")
            else:
                # Market regime is hidden
                flow_direction = "➡️ Neutral"
                category_flow = pd.DataFrame()
            
            # 3. EMERGING PATTERNS
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            
            # Calculate pattern emergence based on sensitivity
            pattern_emergence = wave_filtered_df.copy()
            
            # Set pattern distance thresholds based on sensitivity
            if sensitivity == "Conservative":
                pattern_distance = 5  # Within 5% of qualifying
            elif sensitivity == "Balanced":
                pattern_distance = 10  # Within 10% of qualifying
            else:  # Aggressive
                pattern_distance = 15  # Within 15% of qualifying
            
            # Check how close to pattern thresholds
            emergence_data = []
            
            # Category Leader emergence
            if 'category_percentile' in pattern_emergence.columns:
                close_to_leader = pattern_emergence[
                    (pattern_emergence['category_percentile'] >= (90 - pattern_distance)) & 
                    (pattern_emergence['category_percentile'] < 90)
                ]
                for _, stock in close_to_leader.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🔥 CAT LEADER',
                        'Distance': f"{90 - stock['category_percentile']:.1f}% away",
                        'Current': f"{stock['category_percentile']:.1f}%ile",
                        'Score': stock['master_score']
                    })
            
            # Breakout Ready emergence
            if 'breakout_score' in pattern_emergence.columns:
                close_to_breakout = pattern_emergence[
                    (pattern_emergence['breakout_score'] >= (80 - pattern_distance)) & 
                    (pattern_emergence['breakout_score'] < 80)
                ]
                for _, stock in close_to_breakout.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🎯 BREAKOUT',
                        'Distance': f"{80 - stock['breakout_score']:.1f} pts away",
                        'Current': f"{stock['breakout_score']:.1f} score",
                        'Score': stock['master_score']
                    })
            
            # Volume Explosion emergence (based on sensitivity)
            if sensitivity == "Conservative":
                rvol_threshold = 3.0
            elif sensitivity == "Balanced":
                rvol_threshold = 2.0
            else:  # Aggressive
                rvol_threshold = 1.5
                
            close_to_explosion = pattern_emergence[
                (pattern_emergence['rvol'] >= (rvol_threshold - 0.5)) & 
                (pattern_emergence['rvol'] < rvol_threshold)
            ]
            for _, stock in close_to_explosion.iterrows():
                emergence_data.append({
                    'Ticker': stock['ticker'],
                    'Company': stock['company_name'],
                    'Pattern': '⚡ VOL EXPLOSION',
                    'Distance': f"{rvol_threshold - stock['rvol']:.1f}x away",
                    'Current': f"{stock['rvol']:.1f}x",
                    'Score': stock['master_score']
                })
            
            if emergence_data:
                emergence_df = pd.DataFrame(emergence_data).sort_values('Score', ascending=False).head(15)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(emergence_df, use_container_width=True, hide_index=True)
                with col2:
                    st.metric("Emerging Patterns", len(emergence_df))
                    st.caption("Stocks about to trigger pattern alerts")
            else:
                st.info(f"No patterns emerging within {pattern_distance}% threshold with current {wave_timeframe} timeframe.")
            
            # 4. ACCELERATION ALERTS
            st.markdown("#### ⚡ Acceleration Alerts - Momentum Building")
            
            # Set acceleration threshold based on sensitivity
            if sensitivity == "Conservative":
                accel_threshold = 85
                momentum_threshold = 70
            elif sensitivity == "Balanced":
                accel_threshold = 70
                momentum_threshold = 60
            else:  # Aggressive
                accel_threshold = 60
                momentum_threshold = 50
            
            # Find accelerating stocks
            accel_conditions = (
                (wave_filtered_df['acceleration_score'] >= accel_threshold) &
                (wave_filtered_df['momentum_score'] >= momentum_threshold)
            )
            
            # Add return pace condition if data available
            if 'ret_7d' in wave_filtered_df.columns and 'ret_30d' in wave_filtered_df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Compare daily average returns
                    avg_daily_7d = wave_filtered_df['ret_7d'] / 7
                    avg_daily_30d = wave_filtered_df['ret_30d'] / 30
                    accel_conditions &= (avg_daily_7d > avg_daily_30d)
            
            accelerating = wave_filtered_df[accel_conditions].nlargest(10, 'acceleration_score')
            
            if len(accelerating) > 0:
                # Create acceleration visualization
                fig_accel = go.Figure()
                
                for _, stock in accelerating.iterrows():
                    # Create mini momentum chart
                    returns = [0]  # Start point
                    x_points = ['Start']
                    
                    if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']):
                        returns.append(stock['ret_30d'])
                        x_points.append('30D')
                    
                    if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']):
                        returns.append(stock['ret_7d'])
                        x_points.append('7D')
                    
                    if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']):
                        returns.append(stock['ret_1d'])
                        x_points.append('1D')
                    
                    if len(returns) > 1:  # Only plot if we have data
                        fig_accel.add_trace(go.Scatter(
                            x=x_points,
                            y=returns,
                            mode='lines+markers',
                            name=stock['ticker'],
                            line=dict(width=2),
                            hovertemplate='%{y:.1f}%<extra></extra>'
                        ))
                
                fig_accel.update_layout(
                    title=f"Acceleration Profiles - Momentum Building (Score ≥ {accel_threshold})",
                    xaxis_title="Time Frame",
                    yaxis_title="Return %",
                    height=350,
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                st.plotly_chart(fig_accel, use_container_width=True)
            else:
                st.info(f"No acceleration signals detected with {sensitivity} sensitivity (requires score ≥ {accel_threshold}).")
            
            # 5. VOLUME SURGE DETECTION
            st.markdown("#### 🌊 Volume Surges - Unusual Activity NOW")
            
            # Set RVOL threshold based on sensitivity
            if sensitivity == "Conservative":
                rvol_surge_threshold = 3.0
            elif sensitivity == "Balanced":
                rvol_surge_threshold = 2.0
            else:  # Aggressive
                rvol_surge_threshold = 1.5
            
            # Find volume surges
            surge_conditions = (wave_filtered_df['rvol'] >= rvol_surge_threshold)
            
            if 'vol_ratio_1d_90d' in wave_filtered_df.columns:
                surge_conditions |= (wave_filtered_df['vol_ratio_1d_90d'] >= rvol_surge_threshold)
            
            volume_surges = wave_filtered_df[surge_conditions].copy()
            
            if len(volume_surges) > 0:
                # Calculate surge score
                volume_surges['surge_score'] = (
                    volume_surges['rvol_score'] * 0.5 +
                    volume_surges['volume_score'] * 0.3 +
                    volume_surges['momentum_score'] * 0.2
                )
                
                top_surges = volume_surges.nlargest(15, 'surge_score')
                
                # Create surge visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Select available columns for display
                    display_columns = ['ticker', 'company_name', 'rvol', 'price', 'category']
                    
                    # Add optional columns if they exist
                    if 'ret_1d' in top_surges.columns:
                        display_columns.insert(3, 'ret_1d')
                    
                    surge_display = top_surges[display_columns].copy()
                    
                    # Add surge type
                    surge_display['Type'] = surge_display['rvol'].apply(
                        lambda x: "🔥🔥🔥" if x > 5 else "🔥🔥" if x > 3 else "🔥"
                    )
                    
                    # Format columns
                    if 'ret_1d' in surge_display.columns:
                        surge_display['ret_1d'] = surge_display['ret_1d'].apply(
                            lambda x: f"{x:+.1f}%" if pd.notna(x) else "0.0%"
                        )
                    
                    surge_display['price'] = surge_display['price'].apply(lambda x: f"₹{x:,.0f}")
                    surge_display['rvol'] = surge_display['rvol'].apply(lambda x: f"{x:.1f}x")
                    
                    # Rename columns
                    rename_dict = {
                        'ticker': 'Ticker',
                        'company_name': 'Company',
                        'rvol': 'RVOL',
                        'price': 'Price',
                        'category': 'Category'
                    }
                    
                    if 'ret_1d' in surge_display.columns:
                        rename_dict['ret_1d'] = '1D Ret'
                    
                    surge_display = surge_display.rename(columns=rename_dict)
                    
                    st.dataframe(surge_display, use_container_width=True, hide_index=True)
                
                with col2:
                    # Volume statistics
                    st.metric("Active Surges", len(volume_surges))
                    st.metric("Extreme (>5x)", len(volume_surges[volume_surges['rvol'] > 5]))
                    st.metric("High (>3x)", len(volume_surges[volume_surges['rvol'] > 3]))
                    
                    # Surge distribution
                    surge_categories = volume_surges['category'].value_counts()
                    if len(surge_categories) > 0:
                        st.markdown("**Surge by Category:**")
                        for cat, count in surge_categories.head(3).items():
                            st.caption(f"{cat}: {count} stocks")
            else:
                st.info(f"No volume surges detected with {sensitivity} sensitivity (requires RVOL ≥ {rvol_surge_threshold}x).")
            
            # Wave Radar Summary
            st.markdown("---")
            st.markdown("#### 🎯 Wave Radar Summary")
            
            summary_cols = st.columns(5)
            
            with summary_cols[0]:
                momentum_count = len(top_shifts) if 'top_shifts' in locals() else 0
                st.metric("Momentum Shifts", momentum_count)
            
            with summary_cols[1]:
                # Show flow direction if available
                if 'flow_direction' in locals() and flow_direction != "➡️ Neutral":
                    regime_display = flow_direction.split()[1] if flow_direction != "N/A" else "Unknown"
                else:
                    # Quick regime calculation if market regime is hidden
                    try:
                        if not wave_filtered_df.empty and 'category' in wave_filtered_df.columns:
                            quick_flow = wave_filtered_df.groupby('category')['master_score'].mean()
                            if len(quick_flow) > 0:
                                top_cat = quick_flow.idxmax()
                                if 'Small' in top_cat or 'Micro' in top_cat:
                                    regime_display = "Risk-ON"
                                elif 'Large' in top_cat or 'Mega' in top_cat:
                                    regime_display = "Risk-OFF"
                                else:
                                    regime_display = "Neutral"
                            else:
                                regime_display = "Unknown"
                        else:
                            regime_display = "Unknown"
                    except:
                        regime_display = "Unknown"
                
                st.metric("Market Regime", regime_display)
            
            with summary_cols[2]:
                emergence_count = len(emergence_data) if 'emergence_data' in locals() and emergence_data else 0
                st.metric("Emerging Patterns", emergence_count)
            
            with summary_cols[3]:
                if 'acceleration_score' in wave_filtered_df.columns:
                    # Use appropriate threshold based on sensitivity
                    if sensitivity == "Conservative":
                        accel_threshold = 85
                    elif sensitivity == "Balanced":
                        accel_threshold = 70
                    else:  # Aggressive
                        accel_threshold = 60
                    accel_count = len(wave_filtered_df[wave_filtered_df['acceleration_score'] >= accel_threshold])
                else:
                    accel_count = 0
                st.metric("Accelerating", accel_count)
            
            with summary_cols[4]:
                if 'rvol' in wave_filtered_df.columns:
                    # Use appropriate threshold based on sensitivity
                    if sensitivity == "Conservative":
                        surge_threshold = 3.0
                    elif sensitivity == "Balanced":
                        surge_threshold = 2.0
                    else:  # Aggressive
                        surge_threshold = 1.5
                    surge_count = len(wave_filtered_df[wave_filtered_df['rvol'] >= surge_threshold])
                else:
                    surge_count = 0
                st.metric("Volume Surges", surge_count)
        
        else:
            st.warning(f"No data available for Wave Radar analysis with {wave_timeframe} timeframe.")
    
    # Tab 3: Analysis
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        
        if not filtered_df.empty:
            # Score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Pattern analysis
                fig_patterns = Visualizer.create_pattern_analysis(filtered_df)
                st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Sector performance
            st.markdown("#### Sector Performance")
            try:
                if 'sector' in filtered_df.columns:
                    sector_df = filtered_df.groupby('sector').agg({
                        'master_score': ['mean', 'count'],
                        'rvol': 'mean',
                        'ret_30d': 'mean'
                    }).round(2)
                    
                    if not sector_df.empty:
                        sector_df.columns = ['Avg Score', 'Count', 'Avg RVOL', 'Avg 30D Ret']
                        sector_df = sector_df.sort_values('Avg Score', ascending=False)
                        
                        # Add percentage column
                        sector_df['% of Total'] = (sector_df['Count'] / len(filtered_df) * 100).round(1)
                        
                        st.dataframe(
                            sector_df.style.background_gradient(subset=['Avg Score']),
                            use_container_width=True
                        )
                    else:
                        st.info("No sector data available for analysis.")
                else:
                    st.info("Sector column not available in data.")
            except Exception as e:
                logger.error(f"Error in sector analysis: {str(e)}")
                st.error("Unable to perform sector analysis with current data.")
            
            # Category performance
            st.markdown("#### Category Performance")
            try:
                if 'category' in filtered_df.columns:
                    category_df = filtered_df.groupby('category').agg({
                        'master_score': ['mean', 'count'],
                        'category_percentile': 'mean'
                    }).round(2)
                    
                    if not category_df.empty:
                        category_df.columns = ['Avg Score', 'Count', 'Avg Cat %ile']
                        category_df = category_df.sort_values('Avg Score', ascending=False)
                        
                        st.dataframe(
                            category_df.style.background_gradient(subset=['Avg Score']),
                            use_container_width=True
                        )
                    else:
                        st.info("No category data available for analysis.")
                else:
                    st.info("Category column not available in data.")
            except Exception as e:
                logger.error(f"Error in category analysis: {str(e)}")
                st.error("Unable to perform category analysis with current data.")
            
            # Trend Analysis
            if 'trend_quality' in filtered_df.columns:
                st.markdown("#### 📈 Trend Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trend distribution pie chart
                    trend_dist = pd.cut(
                        filtered_df['trend_quality'],
                        bins=[0, 40, 60, 80, 100],
                        labels=['⚠️ Weak/Down', '➡️ Neutral', '✅ Good Up', '🔥 Strong Up']
                    ).value_counts()
                    
                    fig_trend = px.pie(
                        values=trend_dist.values,
                        names=trend_dist.index,
                        title="Trend Quality Distribution",
                        color_discrete_map={
                            '🔥 Strong Up': '#2ecc71',
                            '✅ Good Up': '#3498db',
                            '➡️ Neutral': '#f39c12',
                            '⚠️ Weak/Down': '#e74c3c'
                        }
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                with col2:
                    # Trend statistics
                    st.markdown("**Trend Statistics**")
                    trend_stats = {
                        "Average Trend Score": f"{filtered_df['trend_quality'].mean():.1f}",
                        "Stocks Above All SMAs": f"{(filtered_df['trend_quality'] >= 85).sum()}",
                        "Stocks in Uptrend (60+)": f"{(filtered_df['trend_quality'] >= 60).sum()}",
                        "Stocks in Downtrend (<40)": f"{(filtered_df['trend_quality'] < 40).sum()}"
                    }
                    for label, value in trend_stats.items():
                        st.metric(label, value)
        
        else:
            st.info("No data available for analysis.")
    
    # Tab 4: Search
    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search stocks",
                placeholder="Enter ticker or company name...",
                help="Search by ticker symbol or company name",
                key="search_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        
        # Perform search
        if search_query or search_clicked:
            search_results = SearchEngine.search_stocks(filtered_df, search_query)
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                
                # Display each result in detail
                for idx, stock in search_results.iterrows():
                    with st.expander(
                        f"📊 {stock['ticker']} - {stock['company_name']} "
                        f"(Rank #{int(stock['rank'])})",
                        expanded=True
                    ):
                        # Header metrics
                        metric_cols = st.columns(6)
                        
                        with metric_cols[0]:
                            st.metric(
                                "Master Score",
                                f"{stock['master_score']:.1f}",
                                f"Rank #{int(stock['rank'])}"
                            )
                        
                        with metric_cols[1]:
                            price_value = f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A"
                            ret_1d_value = f"{stock['ret_1d']:.1f}%" if pd.notna(stock.get('ret_1d')) else None
                            st.metric(
                                "Price",
                                price_value,
                                ret_1d_value
                            )
                        
                        with metric_cols[2]:
                            st.metric(
                                "From Low",
                                f"{stock['from_low_pct']:.0f}%",
                                "52-week range position"
                            )
                        
                        with metric_cols[3]:
                            st.metric(
                                "30D Return",
                                f"{stock['ret_30d']:.1f}%",
                                "↑" if stock['ret_30d'] > 0 else "↓"
                            )
                        
                        with metric_cols[4]:
                            st.metric(
                                "RVOL",
                                f"{stock['rvol']:.1f}x",
                                "High" if stock['rvol'] > 2 else "Normal"
                            )
                        
                        with metric_cols[5]:
                            st.metric(
                                "Category %ile",
                                f"{stock.get('category_percentile', 0):.0f}",
                                stock['category']
                            )
                        
                        # Score breakdown
                        st.markdown("#### 📈 Score Components")
                        score_cols = st.columns(6)
                        
                        components = [
                            ("Position", stock['position_score'], CONFIG.POSITION_WEIGHT),
                            ("Volume", stock['volume_score'], CONFIG.VOLUME_WEIGHT),
                            ("Momentum", stock['momentum_score'], CONFIG.MOMENTUM_WEIGHT),
                            ("Acceleration", stock['acceleration_score'], CONFIG.ACCELERATION_WEIGHT),
                            ("Breakout", stock['breakout_score'], CONFIG.BREAKOUT_WEIGHT),
                            ("RVOL", stock['rvol_score'], CONFIG.RVOL_WEIGHT)
                        ]
                        
                        for i, (name, score, weight) in enumerate(components):
                            with score_cols[i]:
                                # Color coding
                                if score >= 80:
                                    color = "🟢"
                                elif score >= 60:
                                    color = "🟡"
                                else:
                                    color = "🔴"
                                
                                st.markdown(
                                    f"**{name}**<br>"
                                    f"{color} {score:.0f}<br>"
                                    f"<small>Weight: {weight:.0%}</small>",
                                    unsafe_allow_html=True
                                )
                        
                        # Patterns
                        if stock.get('patterns'):
                            st.markdown(f"**🎯 Patterns:** {stock['patterns']}")
                        
                        # Additional details in columns
                        detail_cols = st.columns(3)
                        
                        with detail_cols[0]:
                            st.markdown("**📊 Classification**")
                            st.text(f"Sector: {stock['sector']}")
                            st.text(f"Category: {stock['category']}")
                            if 'eps_tier' in stock:
                                st.text(f"EPS Tier: {stock['eps_tier']}")
                            if 'pe_tier' in stock:
                                st.text(f"PE Tier: {stock['pe_tier']}")
                            
                            # Smart fundamental display
                            if show_fundamentals:
                                st.markdown("**💰 Fundamentals**")
                                
                                # PE Ratio
                                if 'pe' in stock and pd.notna(stock['pe']):
                                    try:
                                        pe_val = float(stock['pe'])
                                        if pe_val <= 0 or np.isinf(pe_val):
                                            pe_display = "Loss"
                                            pe_color = "🔴"
                                        elif pe_val < 10:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🟢"
                                        elif pe_val < 15:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🟢"
                                        elif pe_val < 25:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🟡"
                                        elif pe_val < 50:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🟠"
                                        elif pe_val < 100:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🔴"
                                        else:
                                            if pe_val > 10000:
                                                pe_display = f"{pe_val/1000:.0f}Kx"
                                            else:
                                                pe_display = f"{pe_val:.0f}x"
                                            pe_color = "🔴"
                                        st.text(f"PE Ratio: {pe_color} {pe_display}")
                                    except (ValueError, TypeError, OverflowError):
                                        st.text("PE Ratio: - (Error)")
                                else:
                                    st.text("PE Ratio: - (N/A)")
                                
                                # EPS Current
                                if 'eps_current' in stock and pd.notna(stock['eps_current']):
                                    try:
                                        eps_val = float(stock['eps_current'])
                                        if abs(eps_val) >= 1000:
                                            eps_display = f"₹{eps_val/1000:.1f}K"
                                        elif abs(eps_val) >= 100:
                                            eps_display = f"₹{eps_val:.0f}"
                                        else:
                                            eps_display = f"₹{eps_val:.2f}"
                                        st.text(f"EPS: {eps_display}")
                                    except (ValueError, TypeError):
                                        st.text("EPS: - (Error)")
                                else:
                                    st.text("EPS: - (N/A)")
                                
                                # EPS Change - FIXED for percentage data
                                if 'eps_change_pct' in stock and pd.notna(stock['eps_change_pct']):
                                    try:
                                        # Value is already in percentage
                                        eps_chg = float(stock['eps_change_pct'])
                                        
                                        if np.isinf(eps_chg):
                                            eps_display = "∞" if eps_chg > 0 else "-∞"
                                        elif abs(eps_chg) >= 10000:
                                            eps_display = f"{eps_chg/1000:+.1f}K%"
                                        elif abs(eps_chg) >= 1000:
                                            eps_display = f"{eps_chg:+.0f}%"
                                        else:
                                            eps_display = f"{eps_chg:+.1f}%"
                                        
                                        if eps_chg >= 100:
                                            eps_emoji = "🚀"
                                        elif eps_chg >= 50:
                                            eps_emoji = "🔥"
                                        elif eps_chg >= 20:
                                            eps_emoji = "📈"
                                        elif eps_chg >= 0:
                                            eps_emoji = "➕"
                                        elif eps_chg >= -20:
                                            eps_emoji = "➖"
                                        elif eps_chg >= -50:
                                            eps_emoji = "📉"
                                        else:
                                            eps_emoji = "⚠️"
                                        
                                        st.text(f"EPS Growth: {eps_emoji} {eps_display}")
                                    except (ValueError, TypeError, OverflowError):
                                        st.text("EPS Growth: - (Error)")
                                else:
                                    st.text("EPS Growth: - (N/A)")
                        
                        with detail_cols[1]:
                            st.markdown("**📈 Performance**")
                            for period, col in [
                                ("1 Day", 'ret_1d'),
                                ("7 Days", 'ret_7d'),
                                ("30 Days", 'ret_30d'),
                                ("3 Months", 'ret_3m'),
                                ("6 Months", 'ret_6m')
                            ]:
                                if col in stock.index and pd.notna(stock[col]):
                                    st.text(f"{period}: {stock[col]:.1f}%")
                        
                        with detail_cols[2]:
                            st.markdown("**🔍 Technicals**")
                            st.text(f"52W Low: ₹{stock.get('low_52w', 0):,.0f}")
                            st.text(f"52W High: ₹{stock.get('high_52w', 0):,.0f}")
                            st.text(f"From High: {stock.get('from_high_pct', 0):.0f}%")
                            
                            # Trading Above/Below SMAs
                            st.markdown("**📊 Trading Position**")
                            
                            # Helper function for clean SMA display
                            def get_sma_position(price, sma_value, sma_name):
                                """Calculate position relative to SMA with clean formatting"""
                                if pd.isna(sma_value) or sma_value <= 0:
                                    return f"{sma_name}: No data"
                                
                                if price > sma_value:
                                    pct_above = ((price - sma_value) / sma_value) * 100
                                    return f"{sma_name}: ₹{sma_value:,.0f} (↑ {pct_above:.1f}%)"
                                else:
                                    pct_below = ((sma_value - price) / sma_value) * 100
                                    return f"{sma_name}: ₹{sma_value:,.0f} (↓ {pct_below:.1f}%)"
                            
                            # Check each SMA
                            current_price = stock.get('price', 0)
                            trading_above = []
                            trading_below = []
                            
                            sma_checks = [
                                ('sma_20d', '20 DMA'),
                                ('sma_50d', '50 DMA'),
                                ('sma_200d', '200 DMA')
                            ]
                            
                            for sma_col, sma_label in sma_checks:
                                if sma_col in stock and pd.notna(stock[sma_col]) and stock[sma_col] > 0:
                                    if current_price > stock[sma_col]:
                                        trading_above.append(sma_label)
                                    else:
                                        trading_below.append(sma_label)
                                    
                                    # Show detailed position
                                    position_text = get_sma_position(current_price, stock[sma_col], sma_label)
                                    
                                    # Color code the output
                                    if current_price > stock[sma_col]:
                                        st.text(f"✅ {position_text}")
                                    else:
                                        st.text(f"❌ {position_text}")
                            
                            # Summary of trading position
                            if trading_above:
                                st.success(f"Above: {', '.join(trading_above)}")
                            if trading_below:
                                st.warning(f"Below: {', '.join(trading_below)}")
                            
                            # Trend quality
                            if 'trend_quality' in stock:
                                tq = stock['trend_quality']
                                if tq >= 80:
                                    st.text(f"Trend: 💪 Strong ({tq:.0f})")
                                elif tq >= 60:
                                    st.text(f"Trend: 👍 Good ({tq:.0f})")
                                else:
                                    st.text(f"Trend: 👎 Weak ({tq:.0f})")
            
            else:
                st.warning("No stocks found matching your search criteria.")
    
    # Tab 5: Export
    with tabs[5]:
        st.markdown("### 📥 Export Data")
        
        # Export template selection
        st.markdown("#### 📋 Export Templates")
        export_template = st.radio(
            "Choose export template:",
            options=[
                "Full Analysis (All Data)",
                "Day Trader Focus",
                "Swing Trader Focus",
                "Investor Focus"
            ],
            help="Select a template based on your trading style"
        )
        
        # Map template names to keys
        template_map = {
            "Full Analysis (All Data)": "full",
            "Day Trader Focus": "day_trader",
            "Swing Trader Focus": "swing_trader",
            "Investor Focus": "investor"
        }
        
        selected_template = template_map[export_template]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Excel Report")
            st.markdown(
                "Comprehensive multi-sheet report including:\n"
                "- Top 100 stocks with all scores\n"
                "- Complete stock list\n"
                "- Sector analysis\n"
                "- Category analysis\n"
                "- Pattern frequency analysis\n"
                "- Wave Radar signals (momentum shifts)\n"
                "- Smart money flow tracking"
            )
            
            if st.button("Generate Excel Report", type="primary", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export. Please adjust your filters.")
                else:
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_file = ExportEngine.create_excel_report(filtered_df, template=selected_template)
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_file,
                                file_name=f"wave_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("Excel report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating Excel report: {str(e)}")
                            logger.error(f"Excel export error: {str(e)}", exc_info=True)
        
        with col2:
            st.markdown("#### 📄 CSV Export")
            st.markdown(
                "Enhanced CSV format with:\n"
                "- All ranking scores\n"
                "- Price and return data\n"
                "- Pattern detections\n"
                "- Category classifications\n"
                "- Trend quality scores\n"
                "- RVOL and volume metrics\n"
                "- Perfect for Wave Radar analysis"
            )
            
            if st.button("Generate CSV Export", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export. Please adjust your filters.")
                else:
                    try:
                        csv_data = ExportEngine.create_csv_export(filtered_df)
                        
                        st.download_button(
                            label="📥 Download CSV File",
                            data=csv_data,
                            file_name=f"wave_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("CSV export generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating CSV: {str(e)}")
                        logger.error(f"CSV export error: {str(e)}", exc_info=True)
        
        # Export statistics
        st.markdown("---")
        st.markdown("#### 📊 Export Preview")
        
        export_stats = {
            "Total Stocks": len(filtered_df),
            "Average Score": f"{filtered_df['master_score'].mean():.1f}" if not filtered_df.empty and 'master_score' in filtered_df.columns else "N/A",
            "Stocks with Patterns": (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0,
            "High RVOL (>2x)": (filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns else 0,
            "Positive 30D Returns": (filtered_df['ret_30d'] > 0).sum() if 'ret_30d' in filtered_df.columns else 0,
            "Data Quality": f"{(1 - filtered_df['master_score'].isna().sum() / len(filtered_df)) * 100:.1f}%" if not filtered_df.empty and 'master_score' in filtered_df.columns else "N/A"
        }
        
        stat_cols = st.columns(3)
        for i, (label, value) in enumerate(export_stats.items()):
            with stat_cols[i % 3]:
                st.metric(label, value)
    
    # Tab 6: About - ENHANCED WITH COMPLETE PATTERN DOCUMENTATION
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
            The most advanced stock ranking system designed to catch momentum waves early.
            This professional-grade tool combines technical analysis, volume dynamics, and 
            smart pattern recognition to identify high-potential stocks before they peak.
            
            #### 🎯 Core Features
            
            **Master Score 3.0** - Our proprietary ranking algorithm combines:
            - **Position Analysis (30%)** - 52-week range positioning
            - **Volume Dynamics (25%)** - Multi-timeframe volume patterns
            - **Momentum Tracking (15%)** - 30-day price momentum
            - **Acceleration Detection (10%)** - Momentum acceleration signals
            - **Breakout Probability (10%)** - Technical breakout readiness
            - **RVOL Integration (10%)** - Real-time relative volume
            
            **Wave Radar™** - Early detection system featuring:
            - Momentum shift detection
            - Smart money flow tracking
            - Pattern emergence alerts
            - Acceleration monitoring
            - Volume surge detection
            
            **Smart Filters** - Interconnected filtering system:
            - Dynamic filter updates based on selections
            - Multi-dimensional screening
            - Pattern-based filtering
            - Trend quality filtering
            
            #### 💡 How to Use
            
            1. **Data Source** - Choose between Google Sheets or upload your CSV
            2. **Summary Tab** - Executive dashboard with market overview
            3. **Quick Actions** - Use buttons for instant insights
            4. **Rankings Tab** - View top-ranked stocks with comprehensive metrics
            5. **Wave Radar** - Monitor early momentum signals and market shifts
            6. **Analysis Tab** - Deep dive into market sectors and patterns
            7. **Search Tab** - Find specific stocks with detailed analysis
            8. **Export Tab** - Download data for further analysis
            
            #### 🔧 Pro Tips
            
            - Use **Hybrid Mode** to see both technical and fundamental data
            - Combine multiple filters for precision screening
            - Watch for stocks with multiple pattern detections
            - Monitor Wave Radar for early entry opportunities
            - Export data regularly for historical tracking
            - Use CSV upload for custom data analysis
            
            #### 📊 Complete Pattern Legend
            
            **Technical Patterns:**
            - 🔥 **CAT LEADER** - Top 10% in category, category outperformer
            - 💎 **HIDDEN GEM** - Strong in category but undervalued overall
            - 🚀 **ACCELERATING** - Momentum building rapidly (85+ acceleration)
            - 🏦 **INSTITUTIONAL** - Smart money accumulation patterns
            - ⚡ **VOL EXPLOSION** - Extreme volume surge (RVOL > 3x)
            - 🎯 **BREAKOUT** - Ready for technical breakout (80+ score)
            - 👑 **MARKET LEADER** - Top 5% overall ranking
            - 🌊 **MOMENTUM WAVE** - Sustained momentum with acceleration
            - 💰 **LIQUID LEADER** - High liquidity with strong performance
            - 💪 **LONG STRENGTH** - Strong long-term performance (80+ score)
            - 📈 **QUALITY TREND** - Perfect SMA alignment or strong trend
            
            **Price Range Patterns:**
            - 🎯 **52W HIGH APPROACH** - Within 5% of 52-week high with momentum
            - 🔄 **52W LOW BOUNCE** - Bouncing from 52-week low with acceleration
            - 👑 **GOLDEN ZONE** - Optimal position (60%+ from low, <40% from high)
            - 🎯 **RANGE COMPRESS** - Range compression setup for breakout
            
            **Volume Patterns:**
            - 📊 **VOL ACCUMULATION** - Smart money accumulation (rising volume ratios)
            - 🔀 **MOMENTUM DIVERGE** - Acceleration divergence pattern
            
            **Fundamental Patterns (Hybrid Mode):**
            - 💎 **VALUE MOMENTUM** - Low PE (<15) with high momentum
            - 📊 **EARNINGS ROCKET** - Explosive EPS growth with acceleration
            - 🏆 **QUALITY LEADER** - Balanced PE (10-25) with growth
            - ⚡ **TURNAROUND** - Major EPS improvement with volume
            - ⚠️ **HIGH PE** - Overvaluation warning (PE > 100)
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Trend Indicators
            
            - 🔥 **Strong Uptrend** (80-100)
              - Price above all SMAs
              - Perfect SMA alignment
            - ✅ **Good Uptrend** (60-79)
              - Price above 2+ SMAs
              - Positive momentum
            - ➡️ **Neutral Trend** (40-59)
              - Mixed signals
              - Consolidation phase
            - ⚠️ **Weak/Downtrend** (0-39)
              - Price below most SMAs
              - Negative momentum
            
            #### 🎨 Display Modes
            
            **Technical Mode**
            - Pure momentum analysis
            - Technical indicators only
            - Pattern detection
            - Volume dynamics
            
            **Hybrid Mode**
            - Technical + Fundamentals
            - PE ratio display
            - EPS growth tracking
            - Value patterns
            
            #### ⚡ Performance
            
            - Real-time data processing
            - 1-hour intelligent caching
            - Performance monitoring
            - Cloud-ready architecture
            
            #### 🔒 Data Sources
            
            **Google Sheets Integration**
            - Live data connection
            - 1790+ stocks coverage
            - 41 data points per stock
            - Daily updates
            
            **CSV Upload Support**
            - Custom data analysis
            - Same processing pipeline
            - Flexible column mapping
            - Export ready
            
            #### 💬 Support
            
            For best results:
            - Check filters if no data shows
            - Clear cache for fresh data
            - Use search for specific stocks
            - Export data for records
            - Enable debug mode for issues
            
            ---
            
            **Version**: 3.0.7-FINAL-COMPLETE
            **Last Updated**: Dec 2024
            **Status**: Feature Complete
            **New**: CSV Upload + All Features
            """)
        
        # System stats
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.metric(
                "Total Stocks Loaded",
                f"{len(ranked_df):,}" if 'ranked_df' in locals() else "0"
            )
        
        with stats_cols[1]:
            st.metric(
                "Currently Filtered",
                f"{len(filtered_df):,}" if 'filtered_df' in locals() else "0"
            )
        
        with stats_cols[2]:
            st.metric(
                "Data Quality",
                f"{st.session_state.data_quality.get('completeness', 0):.1f}%" if 'data_quality' in st.session_state else "N/A"
            )
        
        with stats_cols[3]:
            cache_time = datetime.now() - st.session_state.last_refresh
            minutes = int(cache_time.total_seconds() / 60)
            st.metric(
                "Cache Age",
                f"{minutes} min",
                "Refresh recommended" if minutes > 60 else "Fresh"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            Wave Detection Ultimate 3.0 Enhanced | Professional Edition with All Features<br>
            <small>Real-time momentum detection • Early entry signals • Smart money flow tracking • Complete Feature Set</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the application
if __name__ == "__main__":
    main()
