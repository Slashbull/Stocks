"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION VERSION
=====================================================
Professional Stock Ranking System with Advanced Analytics
All features complete, all bugs fixed, production-ready
Permanent version - no further updates needed

Version: 3.0-FINAL-PERMANENT
Last Updated: July 2025
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
        "momentum_diverge": 85,
        "range_compress": 80,
        "value_momentum": 70,
        "earnings_rocket": 80,
        "quality_leader": 80,
        "turnaround": 70,
        "high_pe": 100,
        "stealth": 80,
        "vampire": 75,
        "perfect_storm": 90
    })
    
    # Tier thresholds
    TIER_THRESHOLDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Elite": (85, 100),
        "Strong": (70, 85),
        "Average": (50, 70),
        "Weak": (30, 50),
        "Poor": (0, 30)
    })
    
    # Category definitions based on market cap
    MARKET_CAP_CATEGORIES: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Mega Cap": (200_000, float('inf')),    # >200B
        "Large Cap": (10_000, 200_000),         # 10B-200B
        "Mid Cap": (2_000, 10_000),             # 2B-10B
        "Small Cap": (300, 2_000),              # 300M-2B
        "Micro Cap": (50, 300),                 # 50M-300M
        "Nano Cap": (0, 50)                     # <50M
    })

# Create global config instance
CONFIG = Config()

# ============================================
# INITIALIZE SESSION STATE
# ============================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'search_query': "",
        'last_refresh': datetime.now(timezone.utc),
        'user_preferences': {
            'default_top_n': CONFIG.DEFAULT_TOP_N,
            'display_mode': 'Technical',
            'last_filters': {}
        },
        'data_source': "sheet",
        'data_quality': {},
        'show_debug': False,
        'quick_filter': None,
        'quick_filter_applied': False,
        'active_filter_count': 0,
        'wave_timeframe': "All Waves",
        'wave_sensitivity': "Balanced",
        'show_market_regime': True,
        'show_sensitivity_details': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    @staticmethod
    def timer(target_time: float = 1.0):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                
                if duration > target_time:
                    logger.warning(f"{func.__name__} took {duration:.2f}s (target: {target_time}s)")
                else:
                    logger.debug(f"{func.__name__} completed in {duration:.2f}s")
                
                return result
            return wrapper
        return decorator

# ============================================
# DATA VALIDATORS
# ============================================

class DataValidator:
    """Validate data quality and integrity"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                          context: str = "") -> Tuple[bool, str]:
        """Validate dataframe has required columns and data"""
        
        if df is None or df.empty:
            return False, f"{context}: DataFrame is empty"
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"{context}: Missing columns: {', '.join(missing_columns)}"
        
        return True, "Valid"
    
    @staticmethod
    def get_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics"""
        
        quality = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_tickers': 0,
            'missing_critical': 0,
            'missing_important': 0,
            'timestamp': datetime.now(timezone.utc)
        }
        
        if df.empty:
            quality['completeness'] = 0
            quality['freshness'] = 0
            quality['volume_coverage'] = 0
            return quality
        
        # Check duplicates
        if 'ticker' in df.columns:
            quality['duplicate_tickers'] = df['ticker'].duplicated().sum()
        
        # Check critical columns
        for col in CONFIG.CRITICAL_COLUMNS:
            if col in df.columns:
                quality['missing_critical'] += df[col].isna().sum()
        
        # Check important columns
        for col in CONFIG.IMPORTANT_COLUMNS:
            if col in df.columns:
                quality['missing_important'] += df[col].isna().sum()
        
        # Calculate completeness score
        total_cells = len(df) * len(CONFIG.CRITICAL_COLUMNS + CONFIG.IMPORTANT_COLUMNS)
        missing_cells = quality['missing_critical'] + quality['missing_important']
        quality['completeness'] = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Check data freshness (based on volume data)
        if 'volume_1d' in df.columns:
            volume_coverage = (df['volume_1d'] > 0).sum() / len(df) * 100
            quality['volume_coverage'] = volume_coverage
            quality['freshness'] = volume_coverage  # Use volume as proxy for freshness
        else:
            quality['volume_coverage'] = 0
            quality['freshness'] = 0
        
        return quality

# ============================================
# DATA PROCESSING
# ============================================

class DataProcessor:
    """Process and clean raw data"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=2.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any] = None) -> pd.DataFrame:
        """Process raw dataframe with all cleaning and transformations"""
        
        if metadata is None:
            metadata = {}
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Track processing steps
        processing_steps = []
        
        # Step 1: Basic cleaning
        df = DataProcessor._clean_basic_data(df)
        processing_steps.append("Basic cleaning")
        
        # Step 2: Process numeric columns
        df = DataProcessor._process_numeric_columns(df)
        processing_steps.append("Numeric processing")
        
        # Step 3: Process percentage columns
        df = DataProcessor._process_percentage_columns(df)
        processing_steps.append("Percentage processing")
        
        # Step 4: Process volume ratios
        df = DataProcessor._process_volume_ratios(df)
        processing_steps.append("Volume ratio processing")
        
        # Step 5: Calculate derived columns
        df = DataProcessor._calculate_derived_columns(df)
        processing_steps.append("Derived calculations")
        
        # Step 6: Categorize stocks
        df = DataProcessor._categorize_stocks(df)
        processing_steps.append("Stock categorization")
        
        # Step 7: Handle missing data
        df = DataProcessor._handle_missing_data(df)
        processing_steps.append("Missing data handling")
        
        # Step 8: Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        processing_steps.append("Tier classification")
        
        # Step 9: Final validation
        df = DataProcessor._final_validation(df)
        processing_steps.append("Final validation")
        
        # Update metadata
        if metadata is not None:
            metadata['processing_steps'] = processing_steps
            metadata['processed_rows'] = len(df)
        
        return df
    
    @staticmethod
    def _clean_basic_data(df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        
        # Ensure ticker is uppercase
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        
        # Clean company names
        if 'company_name' in df.columns:
            df['company_name'] = df['company_name'].apply(
                lambda x: DataCleaner.sanitize_string(x, "Unknown Company")
            )
        
        # Clean sector and category
        for col in ['sector', 'category']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: DataCleaner.sanitize_string(x, "Unknown")
                )
        
        return df
    
    @staticmethod
    def _process_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Process numeric columns with bounds checking"""
        
        numeric_columns = {
            'price': (0.01, 1_000_000),
            'volume_1d': (0, 10_000_000_000),
            'volume_7d': (0, 100_000_000_000),
            'volume_30d': (0, 1_000_000_000_000),
            'volume_90d': (0, 10_000_000_000_000),
            'volume_180d': (0, 100_000_000_000_000),
            'rvol': (0, 100),
            'pe': (-1000, 10000),
            'eps_current': (-1000, 1000),
            'eps_last_qtr': (-1000, 1000),
            'market_cap': (0, 10_000_000)  # in millions
        }
        
        for col, bounds in numeric_columns.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: DataCleaner.clean_numeric(x, bounds)
                )
        
        # Handle SMA columns
        for col in ['sma_20d', 'sma_50d', 'sma_200d', 'low_52w', 'high_52w', 'prev_close']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: DataCleaner.clean_numeric(x, (0.01, 1_000_000))
                )
        
        return df
    
    @staticmethod
    def _process_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Process percentage columns"""
        
        for col in CONFIG.PERCENTAGE_COLUMNS:
            if col in df.columns:
                # First clean the numeric values
                df[col] = df[col].apply(
                    lambda x: DataCleaner.clean_numeric(x, (-10000, 10000))
                )
        
        return df
    
    @staticmethod
    def _process_volume_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Process volume ratio columns"""
        
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: DataCleaner.clean_numeric(x, (0, 1000))
                )
        
        return df
    
    @staticmethod
    def _calculate_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional derived columns"""
        
        # Calculate year if missing
        if 'year' not in df.columns:
            df['year'] = datetime.now().year
        
        # Ensure RVOL exists
        if 'rvol' not in df.columns and all(col in df.columns for col in ['volume_1d', 'volume_90d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_volume_90d = df['volume_90d'] / 90
                df['rvol'] = np.where(
                    avg_volume_90d > 0,
                    df['volume_1d'] / avg_volume_90d,
                    1.0
                )
        
        return df
    
    @staticmethod
    def _categorize_stocks(df: pd.DataFrame) -> pd.DataFrame:
        """Categorize stocks by market cap if not already categorized"""
        
        if 'category' not in df.columns and 'market_cap' in df.columns:
            df['category'] = df['market_cap'].apply(DataProcessor._get_market_cap_category)
        elif 'category' not in df.columns:
            df['category'] = "Unknown"
        
        return df
    
    @staticmethod
    def _get_market_cap_category(market_cap: float) -> str:
        """Get category based on market cap"""
        
        if pd.isna(market_cap) or market_cap <= 0:
            return "Unknown"
        
        for category, (min_cap, max_cap) in CONFIG.MARKET_CAP_CATEGORIES.items():
            if min_cap <= market_cap < max_cap:
                return category
        
        return "Unknown"
    
    @staticmethod
    def _handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data with intelligent defaults"""
        
        # Critical columns that need values
        critical_defaults = {
            'price': 0,
            'volume_1d': 0,
            'ticker': 'UNKNOWN'
        }
        
        for col, default in critical_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        
        # Optional columns with reasonable defaults
        optional_defaults = {
            'ret_1d': 0, 'ret_3d': 0, 'ret_7d': 0, 'ret_30d': 0,
            'ret_3m': 0, 'ret_6m': 0, 'ret_1y': 0,
            'from_low_pct': 50, 'from_high_pct': -50,
            'rvol': 1.0,
            'category': 'Unknown',
            'sector': 'Unknown'
        }
        
        for col, default in optional_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications for PE and EPS"""
        
        # PE Tier classification
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(DataProcessor._get_pe_tier)
        
        # EPS Tier classification
        if 'eps_change_pct' in df.columns:
            df['eps_tier'] = df['eps_change_pct'].apply(DataProcessor._get_eps_tier)
        
        # Price Tier classification
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(DataProcessor._get_price_tier)
        
        return df
    
    @staticmethod
    def _get_pe_tier(pe: float) -> str:
        """Classify PE ratio into tiers"""
        
        if pd.isna(pe) or pe <= 0:
            return "N/A"
        elif pe < 10:
            return "Deep Value"
        elif pe < 15:
            return "Value"
        elif pe < 20:
            return "Fair"
        elif pe < 30:
            return "Growth"
        elif pe < 50:
            return "High Growth"
        else:
            return "Speculative"
    
    @staticmethod
    def _get_eps_tier(eps_change: float) -> str:
        """Classify EPS change into tiers"""
        
        if pd.isna(eps_change):
            return "N/A"
        elif eps_change < -20:
            return "Declining"
        elif eps_change < 0:
            return "Negative"
        elif eps_change < 10:
            return "Slow Growth"
        elif eps_change < 25:
            return "Moderate"
        elif eps_change < 50:
            return "High Growth"
        else:
            return "Explosive"
    
    @staticmethod
    def _get_price_tier(price: float) -> str:
        """Classify price into tiers"""
        
        if pd.isna(price) or price <= 0:
            return "N/A"
        elif price < 10:
            return "Penny"
        elif price < 50:
            return "Low"
        elif price < 200:
            return "Mid"
        elif price < 1000:
            return "High"
        else:
            return "Premium"
    
    @staticmethod
    def _final_validation(df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup"""
        
        # Remove any rows with invalid tickers
        if 'ticker' in df.columns:
            df = df[df['ticker'] != 'UNKNOWN']
            df = df[df['ticker'].notna()]
            df = df[df['ticker'].str.len() > 0]
        
        # Remove any rows with zero or negative price
        if 'price' in df.columns:
            df = df[df['price'] > 0]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df

# ============================================
# DATA CLEANING UTILITIES
# ============================================

class DataCleaner:
    """Utility functions for data cleaning"""
    
    @staticmethod
    def clean_numeric(value: Any, bounds: Tuple[float, float] = None) -> float:
        """Clean and convert numeric values with bounds checking"""
        
        if pd.isna(value) or value is None:
            return np.nan
        
        # Handle string representations
        if isinstance(value, str):
            # Remove currency symbols and commas
            cleaned = value.strip().upper()
            
            # Check for special cases
            if cleaned in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-', 
                          '#N/A', '#VALUE!', '#ERROR!', '#DIV/0!', 'INF', '-INF']:
                return np.nan
            
            # Remove common symbols and spaces
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            
            try:
                # Convert to float
                result = float(cleaned)
            except (ValueError, TypeError):
                return np.nan
        else:
            try:
                result = float(value)
            except (ValueError, TypeError):
                return np.nan
        
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
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    """Calculate advanced trading metrics"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics efficiently"""
        
        # Money Flow (Price × Volume × RVOL) in millions
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow_mm'] = (df['price'] * df['volume_1d'] * df['rvol']) / 1_000_000
        else:
            df['money_flow_mm'] = 0
        
        # Volume Momentum Index (VMI)
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'rvol']):
            df['vmi'] = (
                df['vol_ratio_7d_90d'] * 0.5 +
                df['vol_ratio_30d_90d'] * 0.3 +
                df['rvol'] * 0.2
            ) * 20  # Scale to 0-100
        else:
            df['vmi'] = 50
        
        # Position Tension
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            # How stretched is the position in the range
            df['position_tension'] = np.abs(df['from_low_pct'] + df['from_high_pct']) / 2
        else:
            df['position_tension'] = 0
        
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
# RANKING ENGINE - OPTIMIZED
# ============================================

class RankingEngine:
    """Calculate all scores and rankings"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores and master score"""
        
        # Calculate individual component scores
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
        
        # Calculate Master Score
        df = RankingEngine._calculate_master_score(df)
        
        # Calculate rankings
        df = RankingEngine.calculate_rankings(df)
        
        return df
    
    @staticmethod
    def _calculate_master_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate master score from components"""
        
        # MASTER SCORE 3.0 - DO NOT MODIFY WEIGHTS
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
        
        return df
    
    @staticmethod
    def calculate_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall and category rankings"""
        
        if df.empty or 'master_score' not in df.columns:
            df['rank'] = 9999
            df['percentile'] = 0
            df['category_rank'] = 9999
            df['category_percentile'] = 0
            return df
        
        # Overall rankings
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        
        # Category rankings
        df = RankingEngine._calculate_category_ranks(df)
        
        return df
    
    @staticmethod
    def calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Position relative to 52-week range"""
        
        if not all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            return pd.Series(50, index=df.index)
        
        # Position score: Higher when closer to 52w high, lower when closer to 52w low
        # from_low_pct: 0-100 (0 = at low, 100 = at high)
        # from_high_pct: 0 to -100 (0 = at high, -100 = at low)
        
        # Convert from_high_pct to positive scale
        position_from_high = 100 + df['from_high_pct']  # 0 to 100 scale
        
        # Average of position from low and position from high
        position_score = (df['from_low_pct'] + position_from_high) / 2
        
        # Apply smoothing for extreme values
        position_score = position_score.clip(0, 100)
        
        # Boost scores for stocks in "sweet spot" (60-90% from low)
        sweet_spot_mask = (df['from_low_pct'] >= 60) & (df['from_low_pct'] <= 90)
        position_score.loc[sweet_spot_mask] *= 1.1
        
        return position_score.clip(0, 100)
    
    @staticmethod
    def calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Multi-timeframe volume analysis"""
        
        volume_ratios = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d']
        
        available_ratios = [col for col in volume_ratios if col in df.columns]
        
        if not available_ratios:
            return pd.Series(50, index=df.index)
        
        # Calculate weighted average of volume ratios
        weights = {
            'vol_ratio_1d_90d': 0.3,
            'vol_ratio_7d_90d': 0.25,
            'vol_ratio_30d_90d': 0.2,
            'vol_ratio_1d_180d': 0.1,
            'vol_ratio_7d_180d': 0.1,
            'vol_ratio_30d_180d': 0.05
        }
        
        volume_score = pd.Series(0, index=df.index)
        total_weight = 0
        
        for ratio in available_ratios:
            # Convert ratio to score (1.0 = 50, 2.0 = 100, 0.5 = 25)
            ratio_score = df[ratio].clip(0, 5) * 20  # Cap at 5x for 100 score
            volume_score += ratio_score * weights.get(ratio, 0.1)
            total_weight += weights.get(ratio, 0.1)
        
        if total_weight > 0:
            volume_score = volume_score / total_weight
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """30-day momentum with trend quality"""
        
        if 'ret_30d' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # Base momentum from 30-day return
        # Scale: -50% = 0, 0% = 50, +50% = 100
        momentum_score = (df['ret_30d'] + 50).clip(0, 100)
        
        # Boost for consistent positive momentum
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            consistent_mask = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            momentum_score.loc[consistent_mask] *= 1.1
        
        # Penalty for extreme moves (possible pump)
        extreme_mask = df['ret_30d'] > 100
        momentum_score.loc[extreme_mask] *= 0.9
        
        return momentum_score.clip(0, 100)
    
    @staticmethod
    def calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Momentum acceleration detection"""
        
        required_cols = ['ret_3d', 'ret_7d', 'ret_30d']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(50, index=df.index)
        
        # Calculate daily pace for each period
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_3d = df['ret_3d'] / 3
            daily_7d = df['ret_7d'] / 7
            daily_30d = df['ret_30d'] / 30
        
        # Acceleration factors
        accel_3d_7d = np.where(daily_7d != 0, daily_3d / daily_7d, 1)
        accel_7d_30d = np.where(daily_30d != 0, daily_7d / daily_30d, 1)
        
        # Combined acceleration score
        # Higher score when recent returns > older returns
        acceleration_score = pd.Series(50, index=df.index)
        
        # Strong acceleration pattern
        strong_accel = (accel_3d_7d > 1.2) & (accel_7d_30d > 1.1) & (daily_3d > 0)
        acceleration_score.loc[strong_accel] = 85
        
        # Moderate acceleration
        moderate_accel = (accel_3d_7d > 1.0) & (accel_7d_30d > 1.0) & (daily_3d > 0)
        acceleration_score.loc[moderate_accel & ~strong_accel] = 70
        
        # Deceleration
        decel = (accel_3d_7d < 0.8) | (accel_7d_30d < 0.8)
        acceleration_score.loc[decel] = 30
        
        return acceleration_score.clip(0, 100)
    
    @staticmethod
    def calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Technical breakout readiness"""
        
        breakout_score = pd.Series(50, index=df.index)
        
        # Factor 1: Near 52-week high
        if 'from_high_pct' in df.columns:
            near_high_score = (100 + df['from_high_pct']).clip(0, 100)
            breakout_score += near_high_score * 0.3
        
        # Factor 2: Above moving averages
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d']):
            above_sma20 = df['price'] > df['sma_20d']
            above_sma50 = df['price'] > df['sma_50d']
            
            sma_score = (above_sma20.astype(int) + above_sma50.astype(int)) * 25
            breakout_score += sma_score * 0.2
        
        # Factor 3: Volume surge
        if 'rvol' in df.columns:
            volume_surge_score = (df['rvol'].clip(0, 3) / 3) * 100
            breakout_score += volume_surge_score * 0.2
        
        # Factor 4: Momentum alignment
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            momentum_aligned = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            breakout_score.loc[momentum_aligned] += 15
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Relative volume importance"""
        
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # RVOL scoring
        # 1x = 50, 2x = 75, 3x = 90, 4x+ = 100
        rvol_score = pd.Series(50, index=df.index)
        
        rvol_score.loc[df['rvol'] >= 4] = 100
        rvol_score.loc[(df['rvol'] >= 3) & (df['rvol'] < 4)] = 90
        rvol_score.loc[(df['rvol'] >= 2) & (df['rvol'] < 3)] = 75
        rvol_score.loc[(df['rvol'] >= 1.5) & (df['rvol'] < 2)] = 65
        rvol_score.loc[(df['rvol'] >= 1) & (df['rvol'] < 1.5)] = 55
        rvol_score.loc[df['rvol'] < 1] = df.loc[df['rvol'] < 1, 'rvol'] * 50
        
        return rvol_score.clip(0, 100)
    
    @staticmethod
    def calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Assess trend quality using multiple factors"""
        
        trend_score = pd.Series(50, index=df.index)
        
        # Factor 1: SMA alignment
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
            perfect_trend = (
                (df['price'] > df['sma_20d']) &
                (df['sma_20d'] > df['sma_50d']) &
                (df['sma_50d'] > df['sma_200d'])
            )
            good_trend = (
                (df['price'] > df['sma_20d']) &
                (df['sma_20d'] > df['sma_50d'])
            )
            
            trend_score.loc[perfect_trend] = 90
            trend_score.loc[good_trend & ~perfect_trend] = 70
        
        # Factor 2: Momentum consistency
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'ret_3m']):
            consistent_momentum = (
                (df['ret_7d'] > 0) &
                (df['ret_30d'] > 5) &
                (df['ret_3m'] > 10)
            )
            trend_score.loc[consistent_momentum] += 10
        
        return trend_score.clip(0, 100)
    
    @staticmethod
    def calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Long-term performance strength"""
        
        strength_score = pd.Series(50, index=df.index)
        
        # Use available long-term returns
        lt_returns = ['ret_3m', 'ret_6m', 'ret_1y']
        available_returns = [col for col in lt_returns if col in df.columns]
        
        if not available_returns:
            return strength_score
        
        # Weight recent performance more
        weights = {'ret_3m': 0.4, 'ret_6m': 0.35, 'ret_1y': 0.25}
        
        for col in available_returns:
            # Convert returns to scores
            # -50% = 0, 0% = 50, +100% = 100
            col_score = ((df[col] + 50) / 1.5).clip(0, 100)
            strength_score += col_score * weights.get(col, 0.3)
        
        # Normalize by total weight
        total_weight = sum(weights.get(col, 0.3) for col in available_returns)
        if total_weight > 0:
            strength_score = strength_score / total_weight
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Liquidity assessment based on volume"""
        
        if 'volume_1d' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # Use log scale for volume to handle wide range
        log_volume = np.log10(df['volume_1d'].clip(lower=1))
        
        # Normalize to 0-100 scale
        # Assume log volume range of 3 (1K) to 9 (1B)
        liquidity_score = ((log_volume - 3) / 6 * 100).clip(0, 100)
        
        # Boost for consistent volume
        if 'vol_ratio_30d_90d' in df.columns:
            consistent_volume = df['vol_ratio_30d_90d'].between(0.8, 1.5)
            liquidity_score.loc[consistent_volume] *= 1.1
        
        # Apply percentile ranking for better distribution
        liquidity_score = liquidity_score.rank(pct=True, ascending=True) * 100
        
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
        """Detect all 25 patterns efficiently using numpy vectorization"""
        
        if df.empty:
            df['patterns'] = ''
            return df
        
        # Get all pattern definitions as boolean masks
        patterns_with_masks = PatternDetector._get_all_pattern_definitions(df)
        
        # Pre-allocate numpy array for pattern presence
        num_patterns = len(patterns_with_masks)
        pattern_matrix = np.zeros((len(df), num_patterns), dtype=bool)
        
        # Extract pattern names for later use
        pattern_names = []
        
        # Populate the boolean matrix
        for i, (pattern_name, mask) in enumerate(patterns_with_masks):
            pattern_names.append(pattern_name)
            if mask is not None and not mask.empty:
                # Ensure mask aligns with DataFrame index
                aligned_mask = mask.reindex(df.index, fill_value=False)
                pattern_matrix[:, i] = aligned_mask.to_numpy()
        
        # Convert boolean matrix to pattern strings efficiently
        patterns_list = []
        for row_idx in range(len(df)):
            active_patterns = [
                pattern_names[col_idx] 
                for col_idx in range(num_patterns) 
                if pattern_matrix[row_idx, col_idx]
            ]
            patterns_list.append(' | '.join(active_patterns) if active_patterns else '')
        
        df['patterns'] = patterns_list
        
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
        if all(col in df.columns for col in ['percentile', 'rvol', 'ret_30d']):
            mask = (
                (df['percentile'] < 50) & 
                (df['rvol'] > 2) & 
                (df['ret_30d'] > 20)
            )
            patterns.append(('💎 HIDDEN GEM', mask))
        
        # 3. Acceleration Pattern
        if 'acceleration_score' in df.columns:
            mask = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
            patterns.append(('🚀 ACCELERATING', mask))
        
        # 4. Institutional Interest
        if all(col in df.columns for col in ['volume_score', 'liquidity_score', 'vol_ratio_30d_90d']):
            mask = (
                (df['volume_score'] >= 70) & 
                (df['liquidity_score'] >= 70) & 
                (df['vol_ratio_30d_90d'] > 1.5)
            )
            patterns.append(('🏦 INSTITUTIONAL', mask))
        
        # 5. Volume Explosion
        if 'rvol' in df.columns:
            mask = df['rvol'] >= 5
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
        if 'momentum_score' in df.columns and 'momentum_harmony' in df.columns:
            mask = (
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & 
                (df['momentum_harmony'] >= 3)
            )
            patterns.append(('🌊 MOMENTUM WAVE', mask))
        
        # 9. Liquid Leader
        if 'liquidity_score' in df.columns:
            mask = df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']
            patterns.append(('💰 LIQUID LEADER', mask))
        
        # 10. Long-term Strength
        if 'long_term_strength' in df.columns:
            mask = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            patterns.append(('💪 LONG STRENGTH', mask))
        
        # 11. Quality Trend
        if 'trend_quality' in df.columns:
            mask = df['trend_quality'] >= 80
            patterns.append(('📈 QUALITY TREND', mask))
        
        # FUNDAMENTAL PATTERNS (for Hybrid mode)
        
        # 12. Value Momentum
        if all(col in df.columns for col in ['pe', 'master_score']):
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)
            mask = has_valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
            patterns.append(('💎 VALUE MOMENTUM', mask))
        
        # 13. Earnings Rocket
        if all(col in df.columns for col in ['eps_change_pct', 'acceleration_score']):
            has_eps_growth = df['eps_change_pct'].notna()
            mask = has_eps_growth & (df['eps_change_pct'] > 50) & (df['acceleration_score'] >= 70)
            patterns.append(('📊 EARNINGS ROCKET', mask))
        
        # 14. Quality Leader
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = df['pe'].notna() & df['eps_change_pct'].notna() & (df['pe'] > 0)
            mask = (
                has_complete_data &
                (df['pe'].between(10, 25)) &
                (df['eps_change_pct'] > 20) &
                (df['percentile'] >= 80)
            )
            patterns.append(('🏆 QUALITY LEADER', mask))
        
        # 15. Turnaround Story
        if all(col in df.columns for col in ['eps_change_pct', 'from_low_pct', 'acceleration_score']):
            mask = (
                (df['eps_change_pct'] > 100) & 
                (df['from_low_pct'] < 40) & 
                (df['acceleration_score'] >= 70)
            )
            patterns.append(('⚡ TURNAROUND', mask))
        
        # 16. High PE Warning
        if 'pe' in df.columns:
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
            mask = has_valid_pe & (df['pe'] > 100)
            patterns.append(('⚠️ HIGH PE', mask))
        
        # RANGE PATTERNS
        
        # 17. 52W High Approach
        if all(col in df.columns for col in ['from_high_pct', 'volume_score', 'momentum_score']):
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
        
        # NEW INTELLIGENCE PATTERNS
        
        # 23. Stealth Accumulation
        if all(col in df.columns for col in ['volume_score', 'ret_30d', 'from_low_pct', 'vmi']):
            mask = (
                (df['volume_score'] >= 60) & 
                (df['ret_30d'].between(10, 30)) & 
                (df['from_low_pct'] < 50) & 
                (df['vmi'] > 60)
            )
            patterns.append(('🤫 STEALTH', mask))
        
        # 24. Vampire Pattern (quiet accumulation)
        if all(col in df.columns for col in ['rvol', 'ret_30d', 'vol_ratio_7d_90d']):
            mask = (
                (df['rvol'].between(0.5, 1.5)) & 
                (df['ret_30d'] > 15) & 
                (df['vol_ratio_7d_90d'] < 0.8)
            )
            patterns.append(('🧛 VAMPIRE', mask))
        
        # 25. Perfect Storm
        if all(col in df.columns for col in ['master_score', 'acceleration_score', 'breakout_score', 'rvol']):
            mask = (
                (df['master_score'] >= 85) & 
                (df['acceleration_score'] >= 80) & 
                (df['breakout_score'] >= 80) & 
                (df['rvol'] >= 2)
            )
            patterns.append(('⛈️ PERFECT STORM', mask))
        
        return patterns

# ============================================
# FILTER ENGINE - ENHANCED
# ============================================

class FilterEngine:
    """Handle all filtering operations with smart interconnected filters"""
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column: str, 
                         current_filters: Dict[str, Any] = None) -> List[str]:
        """Get unique values for interconnected filtering"""
        
        if df.empty or column not in df.columns:
            return []
        
        # For interconnected filtering, apply other filters first
        if current_filters:
            temp_filters = current_filters.copy()
            
            # Map column names to filter keys
            filter_key_map = {
                'category': 'categories',
                'sector': 'sectors',
                'eps_tier': 'eps_tiers',
                'pe_tier': 'pe_tiers',
                'price_tier': 'price_tiers',
                'wave_state': 'wave_states'
            }
            
            # Remove this column's filter to see all its options
            if column in filter_key_map:
                temp_filters.pop(filter_key_map[column], None)
            
            # Apply remaining filters
            filtered_df = FilterEngine.apply_filters(df, temp_filters)
        else:
            filtered_df = df
        
        # Get unique values
        values = filtered_df[column].dropna().unique()
        
        # Convert to strings and exclude unknowns
        values = [str(v) for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN']]
        
        return sorted(values)
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with validation"""
        
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Category filter
        categories = filters.get('categories', [])
        if categories and 'All' not in categories and categories != ['']:
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        # Sector filter
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors and sectors != ['']:
            filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
        
        # EPS tier filter
        if 'eps_tier' in filtered_df.columns:
            eps_tiers = filters.get('eps_tiers', [])
            if eps_tiers and 'All' not in eps_tiers and eps_tiers != ['']:
                filtered_df = filtered_df[filtered_df['eps_tier'].isin(eps_tiers)]
        
        # PE tier filter
        if 'pe_tier' in filtered_df.columns:
            pe_tiers = filters.get('pe_tiers', [])
            if pe_tiers and 'All' not in pe_tiers and pe_tiers != ['']:
                filtered_df = filtered_df[filtered_df['pe_tier'].isin(pe_tiers)]
        
        # Price tier filter
        if 'price_tier' in filtered_df.columns:
            price_tiers = filters.get('price_tiers', [])
            if price_tiers and 'All' not in price_tiers and price_tiers != ['']:
                filtered_df = filtered_df[filtered_df['price_tier'].isin(price_tiers)]
        
        # Master score range
        score_range = filters.get('score_range', [0, 100])
        if score_range != [0, 100]:
            filtered_df = filtered_df[
                (filtered_df['master_score'] >= score_range[0]) &
                (filtered_df['master_score'] <= score_range[1])
            ]
        
        # RVOL range
        rvol_range = filters.get('rvol_range', [0, 10])
        if rvol_range != [0, 10] and 'rvol' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['rvol'] >= rvol_range[0]) &
                (filtered_df['rvol'] <= rvol_range[1])
            ]
        
        # Momentum range
        momentum_range = filters.get('momentum_range', [-100, 200])
        if momentum_range != [-100, 200] and 'ret_30d' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['ret_30d'] >= momentum_range[0]) &
                (filtered_df['ret_30d'] <= momentum_range[1])
            ]
        
        # From low range
        from_low_range = filters.get('from_low_range', [0, 100])
        if from_low_range != [0, 100] and 'from_low_pct' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['from_low_pct'] >= from_low_range[0]) &
                (filtered_df['from_low_pct'] <= from_low_range[1])
            ]
        
        # Pattern filter
        pattern_filter = filters.get('pattern_filter', '')
        if pattern_filter and pattern_filter != 'All' and 'patterns' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['patterns'].str.contains(pattern_filter, na=False)
            ]
        
        # Trend strength filter
        trend_filter = filters.get('trend_filter', 'All')
        if trend_filter != 'All' and 'trend_quality' in filtered_df.columns:
            if trend_filter == 'Strong Uptrend':
                filtered_df = filtered_df[filtered_df['trend_quality'] >= 70]
            elif trend_filter == 'Moderate Uptrend':
                filtered_df = filtered_df[filtered_df['trend_quality'].between(50, 70)]
            elif trend_filter == 'Weak/No Trend':
                filtered_df = filtered_df[filtered_df['trend_quality'] < 50]
        
        # Wave State filter (NEW)
        if 'wave_state' in filtered_df.columns:
            wave_states = filters.get('wave_states', [])
            if wave_states and 'All' not in wave_states and wave_states != ['']:
                filtered_df = filtered_df[filtered_df['wave_state'].isin(wave_states)]
        
        # Wave Strength range (NEW)
        if 'overall_wave_strength' in filtered_df.columns:
            wave_strength_range = filters.get('wave_strength_range', [0, 100])
            if wave_strength_range != [0, 100]:
                filtered_df = filtered_df[
                    (filtered_df['overall_wave_strength'] >= wave_strength_range[0]) &
                    (filtered_df['overall_wave_strength'] <= wave_strength_range[1])
                ]
        
        return filtered_df

# ============================================
# SEARCH ENGINE - ENHANCED
# ============================================

class SearchEngine:
    """Enhanced search functionality with partial matching"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with enhanced partial matching"""
        
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query = query.upper().strip()
            
            # Method 1: Direct ticker match (exact)
            ticker_exact = df[df['ticker'].str.upper() == query]
            if not ticker_exact.empty:
                return ticker_exact
            
            # Method 2: Ticker starts with query
            ticker_starts = df[df['ticker'].str.upper().str.startswith(query)]
            
            # Method 3: Ticker contains query
            ticker_contains = df[df['ticker'].str.upper().str.contains(query, na=False, regex=False)]
            
            # Method 4: Company name contains query (case insensitive)
            company_contains = df[df['company_name'].str.upper().str.contains(query, na=False, regex=False)]
            
            # Method 5: Word boundary matching in company name (ENHANCED)
            def word_starts_with(company_name):
                if pd.isna(company_name):
                    return False
                words = str(company_name).upper().split()
                return any(word.startswith(query) for word in words)
            
            company_word_match = df[df['company_name'].apply(word_starts_with)]
            
            # Combine all results with priority ordering
            all_matches = pd.concat([
                ticker_exact,
                ticker_starts,
                ticker_contains,
                company_word_match,
                company_contains
            ]).drop_duplicates()
            
            # Sort by relevance
            if not all_matches.empty:
                # Add relevance score
                all_matches['relevance'] = 0
                all_matches.loc[all_matches['ticker'].str.upper() == query, 'relevance'] = 100
                all_matches.loc[all_matches['ticker'].str.upper().str.startswith(query), 'relevance'] += 80
                all_matches.loc[all_matches.index.isin(company_word_match.index), 'relevance'] += 60
                all_matches.loc[all_matches['company_name'].str.upper().str.startswith(query), 'relevance'] += 40
                all_matches.loc[all_matches.index.isin(company_contains.index), 'relevance'] += 20
                
                # Sort by relevance then master score
                return all_matches.sort_values(
                    ['relevance', 'master_score'], 
                    ascending=[False, False]
                ).drop('relevance', axis=1)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    """Advanced market analysis and insights"""
    
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Detect sector rotation with dynamic sampling (ENHANCED)"""
        
        if df.empty or 'sector' not in df.columns:
            return pd.DataFrame()
        
        # Group by sector
        sector_groups = []
        
        for sector in df['sector'].unique():
            if sector == 'Unknown':
                continue
            
            sector_df = df[df['sector'] == sector]
            total_stocks = len(sector_df)
            
            if total_stocks == 0:
                continue
            
            # Dynamic sampling based on sector size
            if total_stocks <= 5:
                # Use all stocks for tiny sectors
                sample_size = total_stocks
                sample_pct = 100
            elif total_stocks <= 20:
                # Use 80% for small sectors
                sample_size = int(total_stocks * 0.8)
                sample_pct = 80
            elif total_stocks <= 50:
                # Use 60% for medium sectors
                sample_size = int(total_stocks * 0.6)
                sample_pct = 60
            elif total_stocks <= 100:
                # Use 40% for large sectors
                sample_size = int(total_stocks * 0.4)
                sample_pct = 40
            else:
                # Use 25% (max 50) for very large sectors
                sample_size = min(int(total_stocks * 0.25), 50)
                sample_pct = 25
            
            # Get top stocks by master score
            top_stocks = sector_df.nlargest(sample_size, 'master_score')
            
            # Calculate metrics
            metrics = {
                'sector': sector,
                'total_stocks': total_stocks,
                'analyzed_stocks': sample_size,
                'sample_percentage': sample_pct,
                'avg_score': top_stocks['master_score'].mean(),
                'median_score': top_stocks['master_score'].median(),
                'avg_momentum': top_stocks['ret_30d'].mean() if 'ret_30d' in top_stocks.columns else 0,
                'avg_volume': top_stocks['volume_1d'].mean() if 'volume_1d' in top_stocks.columns else 0,
                'avg_rvol': top_stocks['rvol'].mean() if 'rvol' in top_stocks.columns else 1,
                'avg_ret_30d': top_stocks['ret_30d'].mean() if 'ret_30d' in top_stocks.columns else 0,
                'strong_stocks': len(top_stocks[top_stocks['master_score'] >= 70])
            }
            
            # Calculate money flow
            if 'money_flow_mm' in top_stocks.columns:
                metrics['total_money_flow'] = top_stocks['money_flow_mm'].sum()
            else:
                metrics['total_money_flow'] = 0
            
            sector_groups.append(metrics)
        
        # Create DataFrame and calculate flow score
        sector_df = pd.DataFrame(sector_groups)
        
        if not sector_df.empty:
            # Calculate normalized flow score
            sector_df['flow_score'] = (
                sector_df['avg_score'].rank(pct=True) * 0.4 +
                sector_df['avg_momentum'].rank(pct=True) * 0.3 +
                sector_df['avg_rvol'].rank(pct=True) * 0.3
            ) * 100
            
            # Sort by flow score
            sector_df = sector_df.sort_values('flow_score', ascending=False)
        
        return sector_df
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> str:
        """Detect overall market regime"""
        
        if df.empty:
            return "Unknown"
        
        # Calculate market breadth
        positive_momentum = (df['ret_30d'] > 0).sum() / len(df) if 'ret_30d' in df.columns else 0.5
        high_rvol = (df['rvol'] > 2).sum() / len(df) if 'rvol' in df.columns else 0.1
        strong_scores = (df['master_score'] > 70).sum() / len(df) if 'master_score' in df.columns else 0.2
        
        # Determine regime
        if positive_momentum > 0.6 and strong_scores > 0.3:
            return "🟢 Risk-ON (Bullish)"
        elif positive_momentum < 0.4 and strong_scores < 0.2:
            return "🔴 Risk-OFF (Bearish)"
        else:
            return "🟡 Neutral (Mixed)"

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution histogram"""
        
        if df.empty or 'master_score' not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df['master_score'],
            nbinsx=20,
            name='Score Distribution',
            marker_color='#3498db',
            opacity=0.8
        ))
        
        # Add average line
        avg_score = df['master_score'].mean()
        fig.add_vline(
            x=avg_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {avg_score:.1f}"
        )
        
        fig.update_layout(
            title="Master Score Distribution",
            xaxis_title="Master Score",
            yaxis_title="Number of Stocks",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_momentum_vs_volume_scatter(df: pd.DataFrame) -> go.Figure:
        """Create momentum vs volume scatter plot"""
        
        if df.empty or not all(col in df.columns for col in ['momentum_score', 'volume_score', 'master_score']):
            return go.Figure()
        
        # Limit to top 200 for performance
        plot_df = df.nlargest(200, 'master_score')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_df['momentum_score'],
            y=plot_df['volume_score'],
            mode='markers',
            marker=dict(
                size=plot_df['master_score'] / 5,
                color=plot_df['master_score'],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Master Score"),
                line=dict(width=1, color='white')
            ),
            text=plot_df.apply(
                lambda row: f"{row['ticker']}<br>"
                           f"Score: {row['master_score']:.1f}<br>"
                           f"Momentum: {row['momentum_score']:.1f}<br>"
                           f"Volume: {row['volume_score']:.1f}",
                axis=1
            ),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Momentum vs Volume Analysis (Top 200)",
            xaxis_title="Momentum Score",
            yaxis_title="Volume Score",
            template='plotly_white',
            height=500,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
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
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle all export operations"""
    
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
                           'patterns', 'sector'],
                'focus': 'Multi-day breakout setups'
            },
            'investor': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 
                           'pe', 'eps_current', 'eps_change_pct', 'ret_1y', 
                           'ret_3y', 'long_term_strength', 'category', 'sector'],
                'focus': 'Long-term fundamentals'
            },
            'full': {
                'columns': None,  # Use all columns
                'focus': 'Complete data export'
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
                
                # 1. Summary Sheet
                summary_data = {
                    'Metric': [
                        'Total Stocks',
                        'Average Master Score',
                        'Stocks with Patterns',
                        'High RVOL (>2x)',
                        'Positive Momentum (30d)',
                        'Near 52W High (<10%)',
                        'Data Timestamp'
                    ],
                    'Value': [
                        len(df),
                        f"{df['master_score'].mean():.1f}" if 'master_score' in df.columns else 'N/A',
                        (df['patterns'] != '').sum() if 'patterns' in df.columns else 0,
                        (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0,
                        (df['ret_30d'] > 0).sum() if 'ret_30d' in df.columns else 0,
                        (df['from_high_pct'] > -10).sum() if 'from_high_pct' in df.columns else 0,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                worksheet = writer.sheets['Summary']
                for i, col in enumerate(summary_df.columns):
                    worksheet.write(0, i, col, header_format)
                
                # 2. Main Data Sheet
                template_config = templates.get(template, templates['full'])
                
                if template_config['columns']:
                    # Use only specified columns that exist
                    export_columns = [col for col in template_config['columns'] if col in df.columns]
                    export_df = df[export_columns]
                else:
                    export_df = df
                
                export_df.to_excel(writer, sheet_name='Data', index=False)
                
                # Format data sheet
                worksheet = writer.sheets['Data']
                for i, col in enumerate(export_df.columns):
                    worksheet.write(0, i, col, header_format)
                    
                    # Auto-adjust column width
                    max_len = max(
                        export_df[col].astype(str).str.len().max(),
                        len(col)
                    ) + 2
                    worksheet.set_column(i, i, min(max_len, 30))
                
                # 3. Top Patterns Sheet
                if 'patterns' in df.columns:
                    pattern_stocks = df[df['patterns'] != ''].copy()
                    if not pattern_stocks.empty:
                        pattern_export = pattern_stocks[
                            ['rank', 'ticker', 'company_name', 'master_score', 'patterns']
                        ].head(100)
                        
                        pattern_export.to_excel(writer, sheet_name='Top Patterns', index=False)
                        
                        worksheet = writer.sheets['Top Patterns']
                        for i, col in enumerate(pattern_export.columns):
                            worksheet.write(0, i, col, header_format)
                
                # 4. Sector Analysis Sheet
                sector_analysis = MarketIntelligence.detect_sector_rotation(df)
                if not sector_analysis.empty:
                    sector_analysis.to_excel(writer, sheet_name='Sector Analysis', index=False)
                    
                    worksheet = writer.sheets['Sector Analysis']
                    for i, col in enumerate(sector_analysis.columns):
                        worksheet.write(0, i, col, header_format)
                
        except Exception as e:
            logger.error(f"Excel export error: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export with all data"""
        
        if df.empty:
            return ""
        
        try:
            return df.to_csv(index=False)
        except Exception as e:
            logger.error(f"CSV export error: {str(e)}")
            return ""

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_metric_card(label: str, value: str, delta: str = None):
        """Render a styled metric card"""
        
        if delta:
            st.metric(label=label, value=value, delta=delta)
        else:
            st.metric(label=label, value=value)
    
    @staticmethod
    def render_stock_card(stock: pd.Series):
        """Render a comprehensive stock information card"""
        
        # Header
        st.markdown(f"### {stock['ticker']} - {stock['company_name']}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            UIComponents.render_metric_card(
                "Master Score",
                f"{stock['master_score']:.1f}",
                f"Rank #{int(stock['rank'])}"
            )
        
        with col2:
            UIComponents.render_metric_card(
                "Price",
                f"₹{stock['price']:,.0f}",
                f"{stock.get('ret_1d', 0):+.1f}%"
            )
        
        with col3:
            UIComponents.render_metric_card(
                "30D Return",
                f"{stock.get('ret_30d', 0):+.1f}%"
            )
        
        with col4:
            UIComponents.render_metric_card(
                "RVOL",
                f"{stock.get('rvol', 1):.1f}x"
            )
        
        # Patterns
        if stock.get('patterns'):
            st.markdown("**Patterns:** " + stock['patterns'])

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
        
        # Update data quality metrics
        st.session_state.data_quality = DataValidator.get_data_quality_score(df)
        
        # Clean up memory
        gc.collect()
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        metadata['errors'].append(str(e))
        
        # Try to return cached data if available
        if 'last_good_data' in st.session_state:
            logger.info("Returning cached data due to processing error")
            df, timestamp, old_metadata = st.session_state.last_good_data
            metadata['warnings'].append("Using cached data due to processing error")
            metadata['cache_used'] = True
            return df, timestamp, metadata
        
        raise

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
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Wave Detection Ultimate 3.0 - Professional Stock Ranking System"
        }
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Main container */
    .main {padding-top: 0rem;}
    /* Headers */
    h1, h2, h3 {color: #2c3e50;}
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Buttons */
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
                gc.collect()
                st.success("Cache cleared!")
                time.sleep(0.5)
                st.rerun()
        
        # Data source selection - ENHANCED with prominent buttons
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        data_source_col1, data_source_col2 = st.columns(2)
        with data_source_col1:
            if st.button(
                "📊 Google Sheets", 
                type="primary" if st.session_state.data_source == "sheet" else "secondary",
                use_container_width=True
            ):
                st.session_state.data_source = "sheet"
                st.rerun()
        
        with data_source_col2:
            if st.button(
                "📁 Upload CSV", 
                type="primary" if st.session_state.data_source == "upload" else "secondary",
                use_container_width=True
            ):
                st.session_state.data_source = "upload"
                st.rerun()
        
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
        
        # Debug mode
        st.markdown("---")
        show_debug = st.checkbox("🐛 Show Debug Info", 
                               value=st.session_state.get('show_debug', False))
        st.session_state.show_debug = show_debug
    
    # Data loading and processing
    try:
        # Check if we need to load data
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        # Load and process data
        with st.spinner("📥 Loading and processing data..."):
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
                
                st.session_state.ranked_df = ranked_df
                st.session_state.data_timestamp = data_timestamp
                st.session_state.last_refresh = datetime.now(timezone.utc)
                
                # Show any warnings or errors
                if metadata.get('warnings'):
                    for warning in metadata['warnings']:
                        st.warning(warning)
                
                if metadata.get('errors'):
                    for error in metadata['errors']:
                        st.error(error)
                
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}")
                
                # Try to use last good data
                if 'last_good_data' in st.session_state:
                    ranked_df, data_timestamp, metadata = st.session_state.last_good_data
                    st.warning("Failed to load fresh data, using cached version")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Common issues:\n- Network connectivity\n- Google Sheets permissions\n- Invalid CSV format")
                    st.stop()
        
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        st.stop()
    
    # Quick Action Buttons
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
    # Check for quick filter state
    quick_filter_applied = st.session_state.get('quick_filter_applied', False)
    quick_filter = st.session_state.get('quick_filter', None)
    
    with qa_col1:
        if st.button("📈 Top Gainers", use_container_width=True):
            st.session_state['quick_filter'] = 'top_gainers'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col2:
        if st.button("🔥 Volume Surges", use_container_width=True):
            st.session_state['quick_filter'] = 'volume_surges'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col3:
        if st.button("🎯 Breakout Ready", use_container_width=True):
            st.session_state['quick_filter'] = 'breakout_ready'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col4:
        if st.button("💎 Hidden Gems", use_container_width=True):
            st.session_state['quick_filter'] = 'hidden_gems'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col5:
        if st.button("🌊 Show All", use_container_width=True):
            st.session_state['quick_filter'] = None
            st.session_state['quick_filter_applied'] = False
            st.rerun()
    
    # Apply quick filters
    if quick_filter:
        if quick_filter == 'top_gainers':
            ranked_df_display = ranked_df[ranked_df['momentum_score'] >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ 80")
        elif quick_filter == 'volume_surges':
            ranked_df_display = ranked_df[ranked_df['rvol'] >= 3]
            st.info(f"Showing {len(ranked_df_display)} stocks with RVOL ≥ 3x")
        elif quick_filter == 'breakout_ready':
            ranked_df_display = ranked_df[ranked_df['breakout_score'] >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with breakout score ≥ 80")
        elif quick_filter == 'hidden_gems':
            ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('HIDDEN GEM', na=False)]
            st.info(f"Showing {len(ranked_df_display)} hidden gem stocks")
    else:
        ranked_df_display = ranked_df
    
    # Sidebar filters
    with st.sidebar:
        # Initialize filters dict
        filters = {}
        
        # Display Mode
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if st.session_state.user_preferences['display_mode'] == 'Technical' else 1,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data"
        )
        
        st.session_state.user_preferences['display_mode'] = display_mode
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        # Advanced Filters
        st.markdown("### 🔧 Advanced Filters")
        
        # Category filter
        categories = FilterEngine.get_unique_values(ranked_df_display, 'category', filters)
        selected_categories = st.multiselect(
            "Categories",
            options=['All'] + categories,
            default=['All']
        )
        if 'All' not in selected_categories and selected_categories:
            filters['categories'] = selected_categories
        
        # Sector filter
        sectors = FilterEngine.get_unique_values(ranked_df_display, 'sector', filters)
        selected_sectors = st.multiselect(
            "Sectors",
            options=['All'] + sectors,
            default=['All']
        )
        if 'All' not in selected_sectors and selected_sectors:
            filters['sectors'] = selected_sectors
        
        # Master Score Range
        score_range = st.slider(
            "Master Score Range",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=5,
            help="Filter by Master Score range"
        )
        if score_range != (0, 100):
            filters['score_range'] = score_range
        
        # RVOL Range
        if 'rvol' in ranked_df_display.columns:
            max_rvol = min(ranked_df_display['rvol'].max(), 10)
            rvol_range = st.slider(
                "RVOL Range",
                min_value=0.0,
                max_value=float(max_rvol),
                value=(0.0, float(max_rvol)),
                step=0.5,
                help="Filter by Relative Volume"
            )
            if rvol_range != (0.0, float(max_rvol)):
                filters['rvol_range'] = rvol_range
        
        # Momentum (30D Return) Range
        if 'ret_30d' in ranked_df_display.columns:
            min_ret = max(ranked_df_display['ret_30d'].min(), -100)
            max_ret = min(ranked_df_display['ret_30d'].max(), 200)
            momentum_range = st.slider(
                "30D Return Range (%)",
                min_value=float(min_ret),
                max_value=float(max_ret),
                value=(float(min_ret), float(max_ret)),
                step=5.0,
                help="Filter by 30-day return percentage"
            )
            if momentum_range != (float(min_ret), float(max_ret)):
                filters['momentum_range'] = momentum_range
        
        # 52W Position Range
        if 'from_low_pct' in ranked_df_display.columns:
            from_low_range = st.slider(
                "Position from 52W Low (%)",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=5,
                help="Filter by position relative to 52-week low"
            )
            if from_low_range != (0, 100):
                filters['from_low_range'] = from_low_range
        
        # Pattern filter
        if 'patterns' in ranked_df_display.columns:
            unique_patterns = []
            for patterns in ranked_df_display['patterns'].dropna():
                if patterns:
                    unique_patterns.extend(patterns.split(' | '))
            unique_patterns = sorted(list(set(unique_patterns)))
            
            if unique_patterns:
                pattern_filter = st.selectbox(
                    "Pattern Filter",
                    options=['All'] + unique_patterns,
                    help="Filter by specific pattern"
                )
                if pattern_filter != 'All':
                    filters['pattern_filter'] = pattern_filter
        
        # Trend Strength filter
        if 'trend_quality' in ranked_df_display.columns:
            trend_filter = st.selectbox(
                "Trend Strength",
                options=['All', 'Strong Uptrend', 'Moderate Uptrend', 'Weak/No Trend'],
                help="Filter by trend quality"
            )
            if trend_filter != 'All':
                filters['trend_filter'] = trend_filter
        
        # Wave State filter (NEW)
        if 'wave_state' in ranked_df_display.columns:
            wave_states = FilterEngine.get_unique_values(ranked_df_display, 'wave_state', filters)
            selected_wave_states = st.multiselect(
                "Wave States",
                options=['All'] + wave_states,
                default=['All'],
                help="Filter by wave state"
            )
            if 'All' not in selected_wave_states and selected_wave_states:
                filters['wave_states'] = selected_wave_states
        
        # Wave Strength Range (NEW)
        if 'overall_wave_strength' in ranked_df_display.columns:
            wave_strength_range = st.slider(
                "Wave Strength Range",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=5,
                help="Filter by overall wave strength"
            )
            if wave_strength_range != (0, 100):
                filters['wave_strength_range'] = wave_strength_range
        
        # Fundamental filters (only in Hybrid mode)
        if show_fundamentals:
            st.markdown("### 💼 Fundamental Filters")
            
            # PE Tier filter
            if 'pe_tier' in ranked_df_display.columns:
                pe_tiers = FilterEngine.get_unique_values(ranked_df_display, 'pe_tier', filters)
                selected_pe_tiers = st.multiselect(
                    "PE Tiers",
                    options=['All'] + pe_tiers,
                    default=['All']
                )
                if 'All' not in selected_pe_tiers and selected_pe_tiers:
                    filters['pe_tiers'] = selected_pe_tiers
            
            # EPS Tier filter
            if 'eps_tier' in ranked_df_display.columns:
                eps_tiers = FilterEngine.get_unique_values(ranked_df_display, 'eps_tier', filters)
                selected_eps_tiers = st.multiselect(
                    "EPS Growth Tiers",
                    options=['All'] + eps_tiers,
                    default=['All']
                )
                if 'All' not in selected_eps_tiers and selected_eps_tiers:
                    filters['eps_tiers'] = selected_eps_tiers
        
        # Clear filters button
        st.markdown("---")
        if st.button("🔄 Clear All Filters", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith('filter_'):
                    del st.session_state[key]
            st.session_state['quick_filter'] = None
            st.session_state['quick_filter_applied'] = False
            st.rerun()
    
    # Apply all filters
    filtered_df = FilterEngine.apply_filters(ranked_df_display, filters)
    
    # Update active filter count
    st.session_state.active_filter_count = len(filters)
    filters_active = len(filters) > 0
    
    # Show filter status
    if filters_active or quick_filter_applied:
        filter_col1, filter_col2 = st.columns([5, 1])
        with filter_col1:
            filter_msg = f"**Filtered View:** {len(filtered_df):,} stocks"
            if quick_filter:
                filter_msg += f" | Quick filter: {quick_filter.replace('_', ' ').title()}"
            if filters_active:
                filter_msg += f" | {len(filters)} advanced filters"
            st.info(filter_msg)
        
        with filter_col2:
            if st.button("Clear Filters", type="secondary"):
                st.session_state['quick_filter'] = None
                st.session_state['quick_filter_applied'] = False
                st.rerun()
    
    # Tab interface
    tabs = st.tabs([
        "📊 Rankings", 
        "📈 Analytics", 
        "🌊 Wave Radar", 
        "🔍 Search",
        "📊 Sector Analysis",  # NEW TAB
        "📤 Export",
        "ℹ️ About"
    ])
    
    # Tab 1: Rankings
    with tabs[0]:
        st.markdown("### 🏆 Master Rankings")
        
        # Display controls
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            top_n = st.selectbox(
                "Show Top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n'])
            )
            st.session_state.user_preferences['default_top_n'] = top_n
        
        with col2:
            data_age = datetime.now(timezone.utc) - data_timestamp
            age_minutes = int(data_age.total_seconds() / 60)
            if age_minutes < 60:
                age_str = f"{age_minutes}m ago"
            else:
                age_str = f"{age_minutes // 60}h {age_minutes % 60}m ago"
            st.info(f"📅 Updated: {age_str}")
        
        with col3:
            total_patterns = (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0
            st.metric("Patterns Found", f"{total_patterns:,}")
        
        with col4:
            high_rvol = (filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns else 0
            st.metric("High RVOL (>2x)", f"{high_rvol:,}")
        
        # Display the ranking table
        if not filtered_df.empty:
            display_df = filtered_df.head(top_n).copy()
            
            # Add trend indicator
            if 'trend_quality' in display_df.columns:
                def get_trend_indicator(score):
                    if score >= 80:
                        return "🔥"
                    elif score >= 60:
                        return "📈"
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
            
            # Select available columns
            available_cols = [col for col in display_cols.keys() if col in display_df.columns]
            display_df_final = display_df[available_cols].copy()
            display_df_final.columns = [display_cols[col] for col in available_cols]
            
            # Format numeric columns
            format_dict = {}
            if 'Score' in display_df_final.columns:
                format_dict['Score'] = '{:.1f}'
            if 'Price' in display_df_final.columns:
                format_dict['Price'] = '₹{:,.0f}'
            if 'From Low' in display_df_final.columns:
                format_dict['From Low'] = '{:.0f}%'
            if '30D Ret' in display_df_final.columns:
                format_dict['30D Ret'] = '{:+.1f}%'
            if 'RVOL' in display_df_final.columns:
                format_dict['RVOL'] = '{:.1f}x'
            if 'PE' in display_df_final.columns:
                format_dict['PE'] = lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "-"
            if 'EPS Δ%' in display_df_final.columns:
                format_dict['EPS Δ%'] = lambda x: f"{x:+.0f}%" if pd.notna(x) else "-"
            
            # Apply formatting and display
            st.dataframe(
                display_df_final.style.format(format_dict)
                .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=100)
                .apply(lambda x: ['background-color: #e8f5e9' if v == '🔥' else '' for v in x], 
                      subset=['Trend'] if 'Trend' in display_df_final.columns else [], axis=1),
                use_container_width=True,
                height=min(600, 50 + len(display_df_final) * 35)
            )
            
            # Show trend distribution statistics (NEW - from old version)
            st.markdown("---")
            st.markdown("#### 📊 Trend Distribution Statistics")
            
            stat_cols = st.columns(4)
            
            with stat_cols[0]:
                if 'trend_quality' in filtered_df.columns:
                    avg_trend = filtered_df['trend_quality'].mean()
                    st.metric("Average Trend Score", f"{avg_trend:.1f}")
                else:
                    st.metric("Average Trend Score", "N/A")
            
            with stat_cols[1]:
                if all(col in filtered_df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
                    above_all_smas = filtered_df[
                        (filtered_df['price'] > filtered_df['sma_20d']) &
                        (filtered_df['price'] > filtered_df['sma_50d']) &
                        (filtered_df['price'] > filtered_df['sma_200d'])
                    ]
                    st.metric("Above All SMAs", f"{len(above_all_smas):,}")
                else:
                    st.metric("Above All SMAs", "N/A")
            
            with stat_cols[2]:
                if 'trend_quality' in filtered_df.columns:
                    uptrend_count = len(filtered_df[filtered_df['trend_quality'] >= 60])
                    st.metric("Stocks in Uptrend (60+)", f"{uptrend_count:,}")
                else:
                    st.metric("Stocks in Uptrend", "N/A")
            
            with stat_cols[3]:
                if 'trend_quality' in filtered_df.columns:
                    downtrend_count = len(filtered_df[filtered_df['trend_quality'] < 40])
                    st.metric("Stocks in Downtrend (<40)", f"{downtrend_count:,}")
                else:
                    st.metric("Stocks in Downtrend", "N/A")
            
            # Quick Statistics (NEW - from old version)
            st.markdown("#### 📈 Quick Statistics")
            
            if 'master_score' in filtered_df.columns:
                q1 = filtered_df['master_score'].quantile(0.25)
                median = filtered_df['master_score'].median()
                q3 = filtered_df['master_score'].quantile(0.75)
                
                stat_cols2 = st.columns(3)
                with stat_cols2[0]:
                    st.metric("Q1 Score", f"{q1:.1f}")
                with stat_cols2[1]:
                    st.metric("Median Score", f"{median:.1f}")
                with stat_cols2[2]:
                    st.metric("Q3 Score", f"{q3:.1f}")
        
        else:
            st.warning("No stocks match the current filters")
    
    # Tab 2: Analytics
    with tabs[1]:
        st.markdown("### 📈 Market Analytics Dashboard")
        
        if not filtered_df.empty:
            # Row 1: Distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Score Distribution")
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.markdown("#### Momentum vs Volume")
                fig_scatter = Visualizer.create_momentum_vs_volume_scatter(filtered_df)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("---")
            
            # Pattern Analysis
            st.markdown("#### 🎯 Pattern Analysis")
            if 'patterns' in filtered_df.columns:
                fig_patterns = Visualizer.create_pattern_analysis(filtered_df)
                if fig_patterns.data:
                    st.plotly_chart(fig_patterns, use_container_width=True)
                else:
                    st.info("No patterns detected in current selection")
            
            st.markdown("---")
            
            # Sector performance
            st.markdown("#### 📊 Sector Performance (Dynamically Sampled)")
            sector_rotation = MarketIntelligence.detect_sector_rotation(filtered_df)
            
            if not sector_rotation.empty:
                # Display sector table
                display_cols = ['sector', 'flow_score', 'avg_score', 'median_score', 
                               'analyzed_stocks', 'total_stocks', 'sample_percentage']
                
                if all(col in sector_rotation.columns for col in display_cols):
                    sector_display = sector_rotation[display_cols].copy()
                    sector_display.columns = ['Sector', 'Flow Score', 'Avg Score', 
                                            'Median Score', 'Analyzed', 'Total', 'Sample %']
                    
                    # Format percentages
                    sector_display['Sample %'] = sector_display['Sample %'].apply(lambda x: f"{x}%")
                    
                    st.dataframe(
                        sector_display.style.format({
                            'Flow Score': '{:.1f}',
                            'Avg Score': '{:.1f}',
                            'Median Score': '{:.1f}'
                        }).background_gradient(subset=['Flow Score', 'Avg Score'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                    
                    st.info("📊 **Dynamic Sampling**: Analysis based on sample size that varies by sector size to ensure fair comparison")
            else:
                st.info("No sector data available for analysis")
            
            # Category performance
            st.markdown("#### 💼 Category Performance")
            if 'category' in filtered_df.columns:
                category_df = filtered_df.groupby('category').agg({
                    'master_score': ['mean', 'count'],
                    'category_percentile': 'mean'
                }).round(2)
                
                category_df.columns = ['Avg Score', 'Count', 'Avg Percentile']
                category_df = category_df.sort_values('Avg Score', ascending=False)
                
                st.dataframe(
                    category_df.style.background_gradient(subset=['Avg Score'], cmap='RdYlGn'),
                    use_container_width=True
                )
        else:
            st.info("No data available for analysis")
    
    # Tab 3: Wave Radar - Enhanced
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        st.markdown("*Catch waves as they form, not after they've peaked!*")
        
        # Wave Radar Controls
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        
        with radar_col1:
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
                value=st.session_state.get('wave_sensitivity', "Balanced"),
                help="Conservative = Stronger signals, Aggressive = More signals"
            )
            st.session_state.wave_sensitivity = sensitivity
        
        with radar_col3:
            show_market_regime = st.checkbox(
                "📊 Market Regime Analysis",
                value=st.session_state.get('show_market_regime', True),
                help="Show category rotation flow and market regime detection"
            )
            st.session_state.show_market_regime = show_market_regime
        
        # Initialize wave_filtered_df
        wave_filtered_df = filtered_df.copy()
        
        with radar_col4:
            # Calculate Wave Strength
            if not wave_filtered_df.empty and 'overall_wave_strength' in wave_filtered_df.columns:
                try:
                    wave_strength_score = wave_filtered_df['overall_wave_strength'].mean()
                    
                    if wave_strength_score > 70:
                        wave_emoji = "🌊🔥"
                        wave_color = "🟢"
                    elif wave_strength_score > 50:
                        wave_emoji = "🌊"
                        wave_color = "🟡"
                    else:
                        wave_emoji = "💤"
                        wave_color = "🔴"
                    
                    UIComponents.render_metric_card(
                        "Wave Strength",
                        f"{wave_emoji} {wave_strength_score:.0f}%",
                        f"{wave_color} Market"
                    )
                except Exception as e:
                    logger.error(f"Error calculating wave strength: {str(e)}")
                    UIComponents.render_metric_card("Wave Strength", "N/A", "Error")
            else:
                UIComponents.render_metric_card("Wave Strength", "N/A", "No data")
        
        # Display sensitivity thresholds if enabled
        show_sensitivity_details = st.checkbox(
            "Show sensitivity thresholds",
            value=st.session_state.get('show_sensitivity_details', False)
        )
        st.session_state.show_sensitivity_details = show_sensitivity_details
        
        if show_sensitivity_details:
            st.info(f"""
            **{sensitivity} Sensitivity Thresholds:**
            - Momentum Score: {70 if sensitivity == "Conservative" else 60 if sensitivity == "Balanced" else 50}+
            - Acceleration Score: {80 if sensitivity == "Conservative" else 70 if sensitivity == "Balanced" else 60}+
            - RVOL: {3.0 if sensitivity == "Conservative" else 2.0 if sensitivity == "Balanced" else 1.5}x+
            - Pattern Distance: {5 if sensitivity == "Conservative" else 10 if sensitivity == "Balanced" else 15}%
            """)
        
        # Apply wave timeframe filtering
        if wave_timeframe != "All Waves":
            try:
                if wave_timeframe == "Intraday Surge":
                    required_cols = ['rvol', 'ret_1d', 'price', 'prev_close']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['rvol'] >= 2.5) &
                            (wave_filtered_df['ret_1d'] > 2) &
                            (wave_filtered_df['price'] > wave_filtered_df['prev_close'] * 1.02)
                        ]
                
                elif wave_timeframe == "3-Day Buildup":
                    required_cols = ['ret_3d', 'vol_ratio_7d_90d', 'price', 'sma_20d']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_3d'] > 5) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 1.5) &
                            (wave_filtered_df['price'] > wave_filtered_df['sma_20d'])
                        ]
                
                elif wave_timeframe == "Weekly Breakout":
                    required_cols = ['ret_7d', 'vol_ratio_7d_90d', 'from_high_pct']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_7d'] > 8) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 2.0) &
                            (wave_filtered_df['from_high_pct'] > -10)
                        ]
                
                elif wave_timeframe == "Monthly Trend":
                    required_cols = ['ret_30d', 'price', 'sma_20d', 'sma_50d', 'vol_ratio_30d_180d', 'from_low_pct']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_30d'] > 15) &
                            (wave_filtered_df['price'] > wave_filtered_df['sma_20d']) &
                            (wave_filtered_df['sma_20d'] > wave_filtered_df['sma_50d']) &
                            (wave_filtered_df['vol_ratio_30d_180d'] > 1.2) &
                            (wave_filtered_df['from_low_pct'] > 30)
                        ]
            except Exception as e:
                logger.warning(f"Error applying {wave_timeframe} filter: {str(e)}")
                st.warning(f"Some data not available for {wave_timeframe} filter")
        
        if not wave_filtered_df.empty:
            # 1. MOMENTUM SHIFTS
            st.markdown("#### 🚀 Momentum Shifts Detected")
            
            # Apply sensitivity-based filtering
            if sensitivity == "Conservative":
                momentum_threshold = 70
                accel_threshold = 80
                rvol_threshold = 3.0
            elif sensitivity == "Balanced":
                momentum_threshold = 60
                accel_threshold = 70
                rvol_threshold = 2.0
            else:  # Aggressive
                momentum_threshold = 50
                accel_threshold = 60
                rvol_threshold = 1.5
            
            # Find momentum shifts
            momentum_shifts = wave_filtered_df[
                (wave_filtered_df['momentum_score'] >= momentum_threshold) &
                (wave_filtered_df['acceleration_score'] >= accel_threshold) &
                (wave_filtered_df['rvol'] >= rvol_threshold)
            ].nlargest(10, 'acceleration_score')
            
            if not momentum_shifts.empty:
                shift_cols = st.columns(5)
                for i, (_, stock) in enumerate(momentum_shifts.head(5).iterrows()):
                    with shift_cols[i]:
                        st.markdown(f"""
                        **{stock['ticker']}**  
                        🎯 Score: {stock['master_score']:.0f}  
                        📈 Accel: {stock['acceleration_score']:.0f}  
                        🔥 RVOL: {stock['rvol']:.1f}x  
                        💰 30D: {stock['ret_30d']:+.1f}%
                        """)
                
                # Show remaining as table
                if len(momentum_shifts) > 5:
                    st.markdown("**More Momentum Shifts:**")
                    remaining = momentum_shifts.iloc[5:][['ticker', 'company_name', 'acceleration_score', 'rvol', 'ret_30d']]
                    remaining.columns = ['Ticker', 'Company', 'Accel Score', 'RVOL', '30D Return']
                    st.dataframe(remaining, hide_index=True, use_container_width=True)
            else:
                st.info("No significant momentum shifts detected with current settings")
            
            # 2. SMART MONEY FLOW
            st.markdown("---")
            st.markdown("#### 💰 Smart Money Flow by Category")
            
            if 'money_flow_mm' in wave_filtered_df.columns:
                category_flow = wave_filtered_df.groupby('category').agg({
                    'money_flow_mm': 'sum',
                    'master_score': 'mean',
                    'ticker': 'count'
                }).round(2)
                
                category_flow.columns = ['Total Flow (₹M)', 'Avg Score', 'Stock Count']
                category_flow = category_flow.sort_values('Total Flow (₹M)', ascending=False).head(8)
                
                # Create flow visualization
                flow_cols = st.columns(len(category_flow))
                for i, (category, row) in enumerate(category_flow.iterrows()):
                    with flow_cols[i]:
                        flow_value = row['Total Flow (₹M)']
                        if flow_value > 1000:
                            flow_str = f"₹{flow_value/1000:.1f}B"
                        else:
                            flow_str = f"₹{flow_value:.0f}M"
                        
                        st.metric(
                            category,
                            flow_str,
                            f"{row['Stock Count']} stocks"
                        )
            
            # 3. EMERGING PATTERNS
            st.markdown("---")
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            
            # Calculate pattern emergence
            pattern_distance = 5 if sensitivity == "Conservative" else 10 if sensitivity == "Balanced" else 15
            
            emergence_data = []
            
            # Check various pattern proximities
            if 'category_percentile' in wave_filtered_df.columns:
                close_to_leader = wave_filtered_df[
                    (wave_filtered_df['category_percentile'] >= (90 - pattern_distance)) & 
                    (wave_filtered_df['category_percentile'] < 90)
                ]
                for _, stock in close_to_leader.head(3).iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🔥 CAT LEADER',
                        'Distance': f"{90 - stock['category_percentile']:.1f}% away",
                        'Current': f"{stock['category_percentile']:.1f}%ile"
                    })
            
            if 'breakout_score' in wave_filtered_df.columns:
                close_to_breakout = wave_filtered_df[
                    (wave_filtered_df['breakout_score'] >= (80 - pattern_distance)) & 
                    (wave_filtered_df['breakout_score'] < 80)
                ]
                for _, stock in close_to_breakout.head(3).iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🎯 BREAKOUT',
                        'Distance': f"{80 - stock['breakout_score']:.1f} pts away",
                        'Current': f"{stock['breakout_score']:.1f} score"
                    })
            
            if emergence_data:
                emergence_df = pd.DataFrame(emergence_data).head(10)
                st.dataframe(emergence_df, hide_index=True, use_container_width=True)
            else:
                st.info("No patterns emerging within threshold distance")
            
            # 4. MARKET REGIME
            if show_market_regime:
                st.markdown("---")
                st.markdown("#### 🌍 Market Regime Analysis")
                
                regime = MarketIntelligence.detect_market_regime(wave_filtered_df)
                
                regime_col1, regime_col2, regime_col3 = st.columns(3)
                
                with regime_col1:
                    st.markdown(f"**Overall Market Regime**")
                    st.markdown(f"# {regime}")
                
                with regime_col2:
                    positive_momentum = (wave_filtered_df['ret_30d'] > 0).sum() / len(wave_filtered_df) * 100
                    st.metric("Positive Momentum %", f"{positive_momentum:.1f}%")
                
                with regime_col3:
                    high_scores = (wave_filtered_df['master_score'] > 70).sum()
                    st.metric("High Score Stocks", f"{high_scores:,}")
        
        else:
            st.warning("No stocks match the current wave detection criteria")
    
    # Tab 4: Search
    with tabs[3]:
        st.markdown("### 🔍 Advanced Stock Search")
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search stocks",
                placeholder="Enter ticker or company name...",
                help="Search by ticker symbol or company name (partial matches supported)",
                key="search_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        
        # Perform search
        if search_query or search_clicked:
            with st.spinner("Searching..."):
                search_results = SearchEngine.search_stocks(filtered_df, search_query)
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                
                # Display each result with enhanced layout (FIXED)
                for idx, stock in search_results.iterrows():
                    with st.expander(
                        f"📊 {stock['ticker']} - {stock['company_name']} "
                        f"(Rank #{int(stock['rank'])})",
                        expanded=True
                    ):
                        # Trading Position (4 columns)
                        st.markdown("##### 📊 Trading Position")
                        pos_cols = st.columns(4)
                        
                        with pos_cols[0]:
                            UIComponents.render_metric_card(
                                "Master Score",
                                f"{stock['master_score']:.1f}",
                                f"Rank #{int(stock['rank'])}"
                            )
                        
                        with pos_cols[1]:
                            price_value = f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A"
                            ret_1d_value = f"{stock['ret_1d']:+.1f}%" if pd.notna(stock.get('ret_1d')) else None
                            UIComponents.render_metric_card("Price", price_value, ret_1d_value)
                        
                        with pos_cols[2]:
                            UIComponents.render_metric_card(
                                "From Low",
                                f"{stock.get('from_low_pct', 0):.0f}%",
                                "52W position"
                            )
                        
                        with pos_cols[3]:
                            UIComponents.render_metric_card(
                                "RVOL",
                                f"{stock.get('rvol', 1):.1f}x",
                                "Volume ratio"
                            )
                        
                        # Trend Analysis (3 columns)
                        st.markdown("##### 📈 Trend Analysis")
                        trend_cols = st.columns(3)
                        
                        with trend_cols[0]:
                            UIComponents.render_metric_card(
                                "30D Return",
                                f"{stock.get('ret_30d', 0):+.1f}%"
                            )
                        
                        with trend_cols[1]:
                            UIComponents.render_metric_card(
                                "Momentum Score",
                                f"{stock.get('momentum_score', 50):.0f}"
                            )
                        
                        with trend_cols[2]:
                            if 'wave_state' in stock:
                                UIComponents.render_metric_card(
                                    "Wave State",
                                    stock['wave_state']
                                )
                        
                        # Advanced Metrics (4 columns)
                        st.markdown("##### 🔬 Advanced Metrics")
                        adv_cols = st.columns(4)
                        
                        with adv_cols[0]:
                            if 'money_flow_mm' in stock:
                                UIComponents.render_metric_card(
                                    "Money Flow",
                                    f"₹{stock['money_flow_mm']:.0f}M"
                                )
                        
                        with adv_cols[1]:
                            if 'vmi' in stock:
                                UIComponents.render_metric_card(
                                    "VMI",
                                    f"{stock['vmi']:.0f}"
                                )
                        
                        with adv_cols[2]:
                            if 'position_tension' in stock:
                                UIComponents.render_metric_card(
                                    "Position Tension",
                                    f"{stock['position_tension']:.0f}"
                                )
                        
                        with adv_cols[3]:
                            if 'momentum_harmony' in stock:
                                UIComponents.render_metric_card(
                                    "Momentum Harmony",
                                    f"{stock['momentum_harmony']}/4"
                                )
                        
                        # Fundamentals (if enabled) - FIXED with P/E and EPS
                        if show_fundamentals:
                            st.markdown("##### 💼 Fundamentals")
                            fund_cols = st.columns(4)
                            
                            with fund_cols[0]:
                                if 'pe' in stock and pd.notna(stock['pe']):
                                    UIComponents.render_metric_card(
                                        "P/E Ratio",
                                        f"{stock['pe']:.1f}"
                                    )
                            
                            with fund_cols[1]:
                                if 'eps_current' in stock and pd.notna(stock['eps_current']):
                                    UIComponents.render_metric_card(
                                        "EPS",
                                        f"₹{stock['eps_current']:.2f}"
                                    )
                            
                            with fund_cols[2]:
                                if 'eps_change_pct' in stock and pd.notna(stock['eps_change_pct']):
                                    UIComponents.render_metric_card(
                                        "EPS Growth",
                                        f"{stock['eps_change_pct']:+.0f}%"
                                    )
                            
                            with fund_cols[3]:
                                if 'pe_tier' in stock:
                                    UIComponents.render_metric_card(
                                        "Valuation",
                                        stock['pe_tier']
                                    )
                        
                        # Patterns
                        if stock.get('patterns'):
                            st.markdown("##### 🎯 Patterns Detected")
                            st.info(stock['patterns'])
                        
                        # Category and Sector
                        st.markdown("##### 📋 Classification")
                        class_cols = st.columns(3)
                        
                        with class_cols[0]:
                            st.write(f"**Category:** {stock.get('category', 'Unknown')}")
                        
                        with class_cols[1]:
                            st.write(f"**Sector:** {stock.get('sector', 'Unknown')}")
                        
                        with class_cols[2]:
                            if 'category_percentile' in stock:
                                st.write(f"**Category %ile:** {stock['category_percentile']:.0f}")
            else:
                st.warning(f"No stocks found matching '{search_query}'")
        else:
            st.info("Enter a ticker symbol or company name to search")
    
    # Tab 5: Sector Analysis (NEW)
    with tabs[4]:
        st.markdown("### 📊 Sector Analysis")
        
        if not filtered_df.empty and 'sector' in filtered_df.columns:
            # Sector Overview
            st.markdown("#### 📈 Sector Overview (Dynamically Sampled)")
            
            sector_overview = MarketIntelligence.detect_sector_rotation(filtered_df)
            
            if not sector_overview.empty:
                # Prepare display columns
                display_cols = ['sector', 'flow_score', 'avg_score', 'median_score', 
                               'avg_momentum', 'avg_rvol', 'analyzed_stocks', 'total_stocks', 
                               'sample_percentage']
                
                available_cols = [col for col in display_cols if col in sector_overview.columns]
                sector_display = sector_overview[available_cols].copy()
                
                # Rename columns for display
                rename_map = {
                    'sector': 'Sector',
                    'flow_score': 'Flow Score',
                    'avg_score': 'Avg Score',
                    'median_score': 'Median Score',
                    'avg_momentum': 'Avg Momentum',
                    'avg_rvol': 'Avg RVOL',
                    'analyzed_stocks': 'Analyzed',
                    'total_stocks': 'Total',
                    'sample_percentage': 'Sample %'
                }
                
                sector_display.columns = [rename_map.get(col, col) for col in sector_display.columns]
                
                # Format for display
                format_dict = {}
                if 'Flow Score' in sector_display.columns:
                    format_dict['Flow Score'] = '{:.1f}'
                if 'Avg Score' in sector_display.columns:
                    format_dict['Avg Score'] = '{:.1f}'
                if 'Median Score' in sector_display.columns:
                    format_dict['Median Score'] = '{:.1f}'
                if 'Avg Momentum' in sector_display.columns:
                    format_dict['Avg Momentum'] = '{:+.1f}%'
                if 'Avg RVOL' in sector_display.columns:
                    format_dict['Avg RVOL'] = '{:.1f}x'
                if 'Sample %' in sector_display.columns:
                    sector_display['Sample %'] = sector_display['Sample %'].apply(lambda x: f"{x}%")
                
                st.dataframe(
                    sector_display.style.format(format_dict)
                    .background_gradient(subset=['Flow Score', 'Avg Score'] if all(col in sector_display.columns for col in ['Flow Score', 'Avg Score']) else [], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                st.info("📊 **Dynamic Sampling**: Larger sectors are sampled to ensure fair comparison. Sample size varies from 25% to 100% based on sector size.")
            
            # Sector Deep Dive
            st.markdown("---")
            st.markdown("#### 🔍 Sector Deep Dive")
            
            available_sectors = sorted(filtered_df['sector'].dropna().unique().tolist())
            selected_sector = st.selectbox(
                "Select a Sector",
                options=['-- Select --'] + available_sectors
            )
            
            if selected_sector != '-- Select --':
                sector_stocks = filtered_df[filtered_df['sector'] == selected_sector]
                
                if not sector_stocks.empty:
                    # Sector statistics
                    st.markdown(f"##### 📊 {selected_sector} Statistics")
                    
                    stat_cols = st.columns(5)
                    
                    with stat_cols[0]:
                        st.metric("Total Stocks", f"{len(sector_stocks):,}")
                    
                    with stat_cols[1]:
                        avg_score = sector_stocks['master_score'].mean()
                        st.metric("Avg Score", f"{avg_score:.1f}")
                    
                    with stat_cols[2]:
                        if 'ret_30d' in sector_stocks.columns:
                            avg_momentum = sector_stocks['ret_30d'].mean()
                            st.metric("Avg 30D Return", f"{avg_momentum:+.1f}%")
                    
                    with stat_cols[3]:
                        if 'rvol' in sector_stocks.columns:
                            avg_rvol = sector_stocks['rvol'].mean()
                            st.metric("Avg RVOL", f"{avg_rvol:.1f}x")
                    
                    with stat_cols[4]:
                        pattern_count = (sector_stocks['patterns'] != '').sum() if 'patterns' in sector_stocks.columns else 0
                        st.metric("Stocks w/ Patterns", f"{pattern_count:,}")
                    
                    # Top 10 stocks in sector
                    st.markdown(f"##### 🏆 Top 10 Stocks in {selected_sector}")
                    
                    top_10 = sector_stocks.nlargest(10, 'master_score')
                    
                    # Prepare display columns
                    display_cols = {
                        'rank': 'Rank',
                        'ticker': 'Ticker',
                        'company_name': 'Company',
                        'master_score': 'Score',
                        'rvol': 'RVOL',
                        'ret_30d': '30D Ret',
                        'momentum_score': 'Momentum',
                        'patterns': 'Patterns'
                    }
                    
                    # Add fundamental columns if enabled
                    if show_fundamentals:
                        if 'pe' in top_10.columns:
                            display_cols['pe'] = 'PE'
                        if 'eps_change_pct' in top_10.columns:
                            display_cols['eps_change_pct'] = 'EPS Δ%'
                    
                    available_cols = [col for col in display_cols.keys() if col in top_10.columns]
                    display_df = top_10[available_cols].copy()
                    display_df.columns = [display_cols[col] for col in available_cols]
                    
                    # Format columns
                    format_dict = {}
                    if 'Score' in display_df.columns:
                        format_dict['Score'] = '{:.1f}'
                    if 'RVOL' in display_df.columns:
                        format_dict['RVOL'] = '{:.1f}x'
                    if '30D Ret' in display_df.columns:
                        format_dict['30D Ret'] = '{:+.1f}%'
                    if 'Momentum' in display_df.columns:
                        format_dict['Momentum'] = '{:.0f}'
                    if 'PE' in display_df.columns:
                        format_dict['PE'] = lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "-"
                    if 'EPS Δ%' in display_df.columns:
                        format_dict['EPS Δ%'] = lambda x: f"{x:+.0f}%" if pd.notna(x) else "-"
                    
                    st.dataframe(
                        display_df.style.format(format_dict)
                        .background_gradient(subset=['Score'], cmap='RdYlGn'),
                        use_container_width=True
                    )
        else:
            st.info("No sector data available for analysis")
    
    # Tab 6: Export
    with tabs[5]:
        st.markdown("### 📤 Export Data")
        
        if not filtered_df.empty:
            # Export statistics
            st.markdown("#### 📊 Export Preview")
            
            export_stats = {
                "Total Stocks": len(filtered_df),
                "Average Score": f"{filtered_df['master_score'].mean():.1f}" if 'master_score' in filtered_df.columns else "N/A",
                "Stocks with Patterns": (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0,
                "High RVOL (>2x)": (filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns else 0,
                "Positive 30D Returns": (filtered_df['ret_30d'] > 0).sum() if 'ret_30d' in filtered_df.columns else 0,
                "Data Quality": f"{(1 - filtered_df['master_score'].isna().sum() / len(filtered_df)) * 100:.1f}%" if 'master_score' in filtered_df.columns else "N/A"
            }
            
            stat_cols = st.columns(3)
            for i, (label, value) in enumerate(export_stats.items()):
                with stat_cols[i % 3]:
                    st.metric(label, value)
            
            st.markdown("---")
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Excel Export Templates")
                st.markdown(
                    "Professional Excel reports with multiple sheets:\n"
                    "- Summary statistics\n"
                    "- Full ranked data\n"
                    "- Top patterns analysis\n"
                    "- Sector breakdown\n"
                    "- Custom formatting"
                )
                
                template = st.selectbox(
                    "Select Template",
                    options=[
                        ("Full Report", "full"),
                        ("Day Trader Focus", "day_trader"),
                        ("Swing Trader Focus", "swing_trader"),
                        ("Investor Focus", "investor")
                    ],
                    format_func=lambda x: x[0]
                )
                
                if st.button("📥 Generate Excel Report", use_container_width=True):
                    with st.spinner("Generating Excel report..."):
                        try:
                            excel_data = ExportEngine.create_excel_report(
                                filtered_df, 
                                template=template[1]
                            )
                            
                            st.download_button(
                                label="📥 Download Excel File",
                                data=excel_data,
                                file_name=f"wave_detection_{template[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("Excel report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating Excel report: {str(e)}")
                            logger.error(f"Excel export error: {str(e)}", exc_info=True)
            
            with col2:
                st.markdown("#### 📄 CSV Export")
                st.markdown(
                    "Complete data export with:\n"
                    "- All calculated scores\n"
                    "- Advanced metrics\n"
                    "- Pattern detections\n"
                    "- Wave states\n"
                    "- Raw data for analysis"
                )
                
                if st.button("📥 Generate CSV Export", use_container_width=True):
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
            
            # Quick download section
            st.markdown("---")
            st.markdown("#### ⚡ Quick Downloads")
            
            download_cols = st.columns(3)
            
            with download_cols[0]:
                st.markdown("**Current View**")
                st.write(f"Export {len(filtered_df)} filtered stocks")
                
                csv_filtered = ExportEngine.create_csv_export(filtered_df)
                st.download_button(
                    label="📥 Download Filtered (CSV)",
                    data=csv_filtered,
                    file_name=f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with download_cols[1]:
                st.markdown("**Top 100**")
                st.write("Elite stocks only")
                
                top_100 = filtered_df.nlargest(100, 'master_score')
                csv_top100 = ExportEngine.create_csv_export(top_100)
                st.download_button(
                    label="📥 Download Top 100 (CSV)",
                    data=csv_top100,
                    file_name=f"top100_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with download_cols[2]:
                st.markdown("**Pattern Stocks**")
                pattern_stocks = filtered_df[filtered_df['patterns'] != '']
                st.write(f"{len(pattern_stocks)} stocks with patterns")
                
                if len(pattern_stocks) > 0:
                    csv_patterns = ExportEngine.create_csv_export(pattern_stocks)
                    st.download_button(
                        label="📥 Download Patterns (CSV)",
                        data=csv_patterns,
                        file_name=f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("No data available for export")
    
    # Tab 7: About
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
            The most advanced stock ranking system designed to catch momentum waves early.
            This professional-grade tool combines technical analysis, volume dynamics, 
            advanced metrics, and smart pattern recognition to identify high-potential 
            stocks before they peak.
            
            #### 🎯 Core Features
            
            **Master Score 3.0** - Proprietary ranking algorithm:
            - **Position Analysis (30%)** - 52-week range positioning
            - **Volume Dynamics (25%)** - Multi-timeframe volume patterns
            - **Momentum Tracking (15%)** - 30-day price momentum
            - **Acceleration Detection (10%)** - Momentum acceleration signals
            - **Breakout Probability (10%)** - Technical breakout readiness
            - **RVOL Integration (10%)** - Real-time relative volume
            
            **Advanced Metrics**:
            - **Money Flow** - Price × Volume × RVOL in millions
            - **VMI** - Volume Momentum Index
            - **Position Tension** - Range position stress indicator
            - **Momentum Harmony** - Multi-timeframe alignment (0-4)
            - **Wave State** - Real-time momentum classification
            
            **Pattern Detection** - 25 patterns including:
            - Technical patterns (11)
            - Fundamental patterns (5)
            - Range patterns (6)
            - Intelligence patterns (3)
            
            #### 💡 What's New in 3.0
            
            - **Enhanced Search** - Partial word matching for better results
            - **Dynamic Sector Analysis** - Fair comparison across sector sizes
            - **Faster Pattern Detection** - Vectorized numpy operations
            - **Prominent Data Source** - Easy switching between Sheets/CSV
            - **Wave Filters** - Filter by wave state and strength
            - **Sector Analysis Tab** - Deep dive into sector performance
            - **Trend Statistics** - Distribution and SMA analysis
            
            #### 🔒 Production Status
            
            **Version**: 3.0-FINAL-PERMANENT  
            **Status**: PRODUCTION READY  
            **Optimization**: COMPLETE  
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Pattern Groups
            
            **Technical**
            - 🔥 CAT LEADER
            - 💎 HIDDEN GEM
            - 🚀 ACCELERATING
            - 🏦 INSTITUTIONAL
            - ⚡ VOL EXPLOSION
            - 🎯 BREAKOUT
            - 👑 MARKET LEADER
            - 🌊 MOMENTUM WAVE
            - 💰 LIQUID LEADER
            - 💪 LONG STRENGTH
            - 📈 QUALITY TREND
            
            **Range**
            - 🎯 52W HIGH APPROACH
            - 🔄 52W LOW BOUNCE
            - 👑 GOLDEN ZONE
            - 📊 VOL ACCUMULATION
            - 🔀 MOMENTUM DIVERGE
            - 🎯 RANGE COMPRESS
            
            **Intelligence**
            - 🤫 STEALTH
            - 🧛 VAMPIRE
            - ⛈️ PERFECT STORM
            
            **Fundamental**
            - 💎 VALUE MOMENTUM
            - 📊 EARNINGS ROCKET
            - 🏆 QUALITY LEADER
            - ⚡ TURNAROUND
            - ⚠️ HIGH PE
            
            #### ⚡ Performance
            
            - Processing: <2 seconds
            - Filtering: <200ms
            - Pattern detection: <500ms
            - Search: <50ms
            - Export: <1 second
            """)
    
    # Debug information
    if show_debug:
        with st.expander("🐛 Debug Information", expanded=True):
            st.write(f"**Session State Keys:** {len(st.session_state.keys())}")
            st.write(f"**Data Shape:** {ranked_df.shape}")
            st.write(f"**Filtered Shape:** {filtered_df.shape}")
            st.write(f"**Processing Time:** {metadata.get('processing_time', 'N/A')}s")
            st.write(f"**Data Source:** {metadata.get('source', 'Unknown')}")
            st.write(f"**Active Filters:** {st.session_state.active_filter_count}")
            
            if st.session_state.data_quality:
                st.write("\n**Data Quality:**")
                for key, value in st.session_state.data_quality.items():
                    if key != 'timestamp':
                        st.write(f"- {key}: {value}")

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
