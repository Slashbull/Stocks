"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION VERSION
=====================================================
Professional Stock Ranking System with Advanced Analytics
Combines V1 reliability with V2 enhancements - All bugs fixed
Optimized for Streamlit Community Cloud deployment

Version: 3.0.0-FINAL
Last Updated: December 2024
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
    
    # Data source defaults
    DEFAULT_SHEET_ID: str = "1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM"
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    STALE_DATA_HOURS: int = 24
    
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
    
    # Critical columns
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    
    # Important columns
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'ret_30d', 'from_low_pct', 'from_high_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d', 'rvol', 'category', 'sector', 'industry',
        'market_cap', 'pe', 'eps_current', 'eps_change_pct',
        'sma_20d', 'sma_50d', 'sma_200d'
    ])
    
    # Wave analysis thresholds
    WAVE_STRONG_THRESHOLD: float = 70.0
    WAVE_MEDIUM_THRESHOLD: float = 40.0
    
    # Pattern detection thresholds
    BREAKOUT_VOLUME_THRESHOLD: float = 2.0
    ACCUMULATION_VOLUME_RATIO: float = 1.5
    DISTRIBUTION_VOLUME_RATIO: float = 0.7

# Create global config instance
CONFIG = Config()

# ============================================
# SESSION STATE MANAGER
# ============================================

class SessionStateManager:
    """Clean session state management without over-engineering"""
    
    @staticmethod
    def initialize():
        """Initialize essential session state variables"""
        defaults = {
            'search_query': "",
            'last_refresh': datetime.now(timezone.utc),
            'data_source': "sheet",
            'sheet_id': CONFIG.DEFAULT_SHEET_ID,
            'gid': CONFIG.DEFAULT_GID,
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
            'data_quality': {},
            'trigger_clear': False,
            
            # Initialize all filter-related keys
            'category_filter': [],
            'sector_filter': [],
            'industry_filter': [],
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
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        """Clear all filter states properly"""
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter',
            'min_score', 'patterns', 'trend_filter',
            'eps_tier_filter', 'pe_tier_filter', 'price_tier_filter',
            'min_eps_change', 'min_pe', 'max_pe',
            'require_fundamental_data', 'wave_states_filter',
            'wave_strength_range_slider', 'quick_filter',
            'quick_filter_applied'
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = "All Trends"
                    else:
                        st.session_state[key] = ""
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], tuple):
                    if key == 'wave_strength_range_slider':
                        st.session_state[key] = (0, 100)
                elif isinstance(st.session_state[key], (int, float)):
                    st.session_state[key] = 0
        
        st.session_state.filters = {}
        st.session_state.active_filter_count = 0
        st.session_state.trigger_clear = False

# ============================================
# DATA LOADING AND PROCESSING
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False)
def load_and_process_data(data_source: str, 
                         sheet_id: Optional[str] = None,
                         gid: Optional[str] = None,
                         file_data: Optional[Any] = None) -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """Load and process data with proper error handling"""
    
    start_time = time.time()
    metadata = {
        'source': data_source,
        'warnings': [],
        'errors': [],
        'processing_time': 0
    }
    
    try:
        # Load data based on source
        if data_source == "upload" and file_data is not None:
            df = pd.read_csv(file_data, low_memory=False)
            metadata['source'] = "CSV Upload"
        else:
            # Use sheet_id and gid, with defaults if not provided
            if not sheet_id:
                sheet_id = CONFIG.DEFAULT_SHEET_ID
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            # Construct Google Sheets CSV export URL
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading from Google Sheets: ID={sheet_id}, GID={gid}")
            
            try:
                df = pd.read_csv(csv_url, low_memory=False)
                metadata['source'] = "Google Sheets"
                metadata['sheet_id'] = sheet_id
                metadata['gid'] = gid
            except Exception as e:
                logger.error(f"Failed to load Google Sheets: {str(e)}")
                metadata['errors'].append(f"Sheet load error: {str(e)}")
                
                # Try cached data as fallback
                if hasattr(st.session_state, 'last_good_data'):
                    logger.info("Using cached data as fallback")
                    return st.session_state.last_good_data
                raise
        
        # Validate critical columns
        missing_critical = [col for col in CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            raise ValueError(f"Missing critical columns: {missing_critical}")
        
        # Process data
        df = process_dataframe(df, metadata)
        
        # Calculate scores and rankings
        df = calculate_all_scores(df)
        
        # Detect patterns
        df = detect_all_patterns(df)
        
        # Add advanced metrics
        df = calculate_advanced_metrics(df)
        
        # Final validation
        if 'master_score' not in df.columns or 'rank' not in df.columns:
            raise ValueError("Failed to calculate rankings")
        
        # Store as last good data
        timestamp = datetime.now(timezone.utc)
        st.session_state.last_good_data = (df, timestamp, metadata)
        
        metadata['processing_time'] = time.time() - start_time
        logger.info(f"Data loaded successfully: {len(df)} stocks in {metadata['processing_time']:.2f}s")
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        metadata['errors'].append(str(e))
        raise

def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Process and clean the dataframe"""
    
    original_count = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['ticker'], keep='first')
    if len(df) < original_count:
        metadata['warnings'].append(f"Removed {original_count - len(df)} duplicate tickers")
    
    # Clean numeric columns
    numeric_columns = [
        'price', 'volume_1d', 'ret_30d', 'from_low_pct', 'from_high_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d', 'rvol', 'pe', 'eps_current', 'eps_change_pct'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Add missing columns with defaults
    if 'rvol' not in df.columns and 'volume_1d' in df.columns and 'volume_90d' in df.columns:
        df['rvol'] = df['volume_1d'] / df['volume_90d'].replace(0, np.nan)
        metadata['warnings'].append("Calculated RVOL from volume data")
    
    # Clean category names
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('Unknown')
    
    if 'sector' in df.columns:
        df['sector'] = df['sector'].fillna('Unknown')
        
    if 'industry' in df.columns:
        df['industry'] = df['industry'].fillna('Unknown')
    
    # Remove invalid stocks
    df = df[df['price'] > 0]
    df = df[df['volume_1d'] > 0]
    
    if len(df) < original_count * 0.5:
        metadata['warnings'].append(f"Significant data reduction: {original_count} → {len(df)} stocks")
    
    return df

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Clean numeric data handling various formats"""
    if series.dtype == 'object':
        # Remove currency symbols and percentage signs
        series = series.astype(str).str.replace('₹', '', regex=False)
        series = series.str.replace(',', '', regex=False)
        series = series.str.replace('%', '', regex=False)
        series = series.str.strip()
        
        # Handle special cases
        series = series.replace(['', 'NaN', 'nan', 'None', '-'], np.nan)
        
        # Convert to numeric
        series = pd.to_numeric(series, errors='coerce')
    
    return series

# ============================================
# SCORING AND RANKING ENGINE
# ============================================

def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all scoring components and final rankings"""
    
    # Position Score (30%)
    df['position_score'] = calculate_position_score(
        df['from_low_pct'].fillna(0),
        df['from_high_pct'].fillna(0)
    )
    
    # Volume Score (25%)
    df['volume_score'] = calculate_volume_score(df)
    
    # Momentum Score (15%)
    df['momentum_score'] = calculate_momentum_score(
        df['ret_30d'].fillna(0)
    )
    
    # Acceleration Score (10%)
    df['acceleration_score'] = calculate_acceleration_score(df)
    
    # Breakout Score (10%)
    df['breakout_score'] = calculate_breakout_score(df)
    
    # RVOL Score (10%)
    df['rvol_score'] = calculate_rvol_score(
        df['rvol'].fillna(1)
    )
    
    # Master Score
    df['master_score'] = (
        df['position_score'] * CONFIG.POSITION_WEIGHT +
        df['volume_score'] * CONFIG.VOLUME_WEIGHT +
        df['momentum_score'] * CONFIG.MOMENTUM_WEIGHT +
        df['acceleration_score'] * CONFIG.ACCELERATION_WEIGHT +
        df['breakout_score'] * CONFIG.BREAKOUT_WEIGHT +
        df['rvol_score'] * CONFIG.RVOL_WEIGHT
    )
    
    # Ensure score is between 0-100
    df['master_score'] = df['master_score'].clip(0, 100)
    
    # Calculate rank
    df['rank'] = df['master_score'].rank(ascending=False, method='min').astype(int)
    
    return df

def calculate_position_score(from_low: pd.Series, from_high: pd.Series) -> pd.Series:
    """Calculate position score (0-100)"""
    # Closer to high is better
    high_distance = 100 - from_high.abs()
    
    # Bonus for being above 52w midpoint
    midpoint_bonus = (from_low > 50).astype(float) * 20
    
    # Penalty for being too far from high
    high_penalty = (from_high < -30).astype(float) * 10
    
    score = high_distance + midpoint_bonus - high_penalty
    return score.clip(0, 100)

def calculate_volume_score(df: pd.DataFrame) -> pd.Series:
    """Calculate volume score based on multiple ratios"""
    score = pd.Series(50.0, index=df.index)  # Base score
    
    # Recent volume surge
    if 'vol_ratio_1d_90d' in df.columns:
        surge_1d = df['vol_ratio_1d_90d'].fillna(0)
        score += np.where(surge_1d > 50, 20, 0)
        score += np.where(surge_1d > 100, 10, 0)
    
    # Weekly volume trend
    if 'vol_ratio_7d_90d' in df.columns:
        surge_7d = df['vol_ratio_7d_90d'].fillna(0)
        score += np.where(surge_7d > 20, 10, 0)
    
    # Monthly volume consistency
    if 'vol_ratio_30d_90d' in df.columns:
        surge_30d = df['vol_ratio_30d_90d'].fillna(0)
        score += np.where(surge_30d > 0, 10, 0)
    
    return score.clip(0, 100)

def calculate_momentum_score(ret_30d: pd.Series) -> pd.Series:
    """Calculate momentum score (0-100)"""
    # Scale 30-day returns to 0-100
    # Assume -50% to +100% maps to 0-100
    score = ((ret_30d + 50) / 150) * 100
    return score.clip(0, 100)

def calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
    """Calculate acceleration score"""
    score = pd.Series(50.0, index=df.index)
    
    # Compare different timeframes
    if all(col in df.columns for col in ['ret_3d', 'ret_7d', 'ret_30d']):
        ret_3d = df['ret_3d'].fillna(0)
        ret_7d = df['ret_7d'].fillna(0)
        ret_30d = df['ret_30d'].fillna(0)
        
        # Acceleration: recent > medium > long
        accel_short = (ret_3d > ret_7d).astype(float) * 25
        accel_medium = (ret_7d > ret_30d).astype(float) * 25
        
        score = 50 + accel_short + accel_medium
    
    return score.clip(0, 100)

def calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
    """Calculate breakout potential score"""
    score = pd.Series(50.0, index=df.index)
    
    # Near 52-week high with volume
    if 'from_high_pct' in df.columns and 'rvol' in df.columns:
        near_high = (df['from_high_pct'] > -10).astype(float)
        high_volume = (df['rvol'] > 1.5).astype(float)
        score += near_high * high_volume * 50
    
    return score.clip(0, 100)

def calculate_rvol_score(rvol: pd.Series) -> pd.Series:
    """Calculate relative volume score"""
    # Map RVOL to score
    # 0-1: 0-50, 1-3: 50-80, 3+: 80-100
    score = pd.Series(0.0, index=rvol.index)
    
    score = np.where(rvol <= 1, rvol * 50,
            np.where(rvol <= 3, 50 + (rvol - 1) * 15,
                    80 + np.minimum((rvol - 3) * 10, 20)))
    
    return pd.Series(score, index=rvol.index).clip(0, 100)

# ============================================
# PATTERN DETECTION
# ============================================

def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect all trading patterns"""
    
    patterns_dict = {}
    
    # Volume patterns
    if 'rvol' in df.columns:
        patterns_dict['Volume Surge'] = df['rvol'] > CONFIG.BREAKOUT_VOLUME_THRESHOLD
        patterns_dict['Accumulation'] = df['rvol'] > CONFIG.ACCUMULATION_VOLUME_RATIO
    
    # Price patterns
    if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
        patterns_dict['Near 52W High'] = df['from_high_pct'] > -5
        patterns_dict['Breakout Ready'] = (df['from_high_pct'] > -10) & (df['from_low_pct'] > 70)
        patterns_dict['Mid-Range Consolidation'] = (df['from_low_pct'].between(40, 60))
    
    # Momentum patterns
    if 'ret_30d' in df.columns:
        patterns_dict['Strong Momentum'] = df['ret_30d'] > 20
        patterns_dict['Steady Uptrend'] = df['ret_30d'].between(10, 20)
    
    # SMA patterns
    if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
        price_num = clean_numeric_column(df['price'])
        sma20_num = clean_numeric_column(df['sma_20d'])
        sma50_num = clean_numeric_column(df['sma_50d'])
        sma200_num = clean_numeric_column(df['sma_200d'])
        
        patterns_dict['Above All SMAs'] = (
            (price_num > sma20_num) & 
            (price_num > sma50_num) & 
            (price_num > sma200_num)
        )
        patterns_dict['Golden Cross Setup'] = (sma50_num > sma200_num)
    
    # Combine patterns
    pattern_names = []
    for pattern, mask in patterns_dict.items():
        if mask.any():
            pattern_names.append(pattern)
    
    # Create pattern strings
    df['patterns'] = ''
    for idx in df.index:
        patterns = []
        for pattern, mask in patterns_dict.items():
            if mask.loc[idx]:
                patterns.append(pattern)
        df.loc[idx, 'patterns'] = ' | '.join(patterns)
    
    return df

# ============================================
# ADVANCED METRICS
# ============================================

def calculate_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced metrics and indicators"""
    
    # EPS Tiers
    if 'eps_change_pct' in df.columns:
        df['eps_tier'] = pd.cut(
            df['eps_change_pct'],
            bins=[-np.inf, -10, 0, 10, 25, 50, np.inf],
            labels=['Declining', 'Negative', 'Stable', 'Growing', 'High Growth', 'Hyper Growth']
        )
    
    # PE Tiers
    if 'pe' in df.columns:
        df['pe_tier'] = pd.cut(
            df['pe'],
            bins=[-np.inf, 0, 15, 25, 35, 50, np.inf],
            labels=['Negative', 'Low', 'Fair', 'High', 'Very High', 'Extreme']
        )
    
    # Price Tiers
    if 'price' in df.columns:
        price_num = clean_numeric_column(df['price'])
        df['price_tier'] = pd.cut(
            price_num,
            bins=[0, 100, 500, 1000, 5000, 10000, np.inf],
            labels=['Penny', 'Low', 'Mid', 'High', 'Premium', 'Super Premium']
        )
    
    # Volatility metrics
    if all(col in df.columns for col in ['high_52w', 'low_52w']):
        high_num = clean_numeric_column(df['high_52w'])
        low_num = clean_numeric_column(df['low_52w'])
        df['volatility_52w'] = ((high_num - low_num) / low_num * 100).round(2)
    
    # Risk metrics
    df['risk_score'] = 50  # Base risk
    if 'volatility_52w' in df.columns:
        df.loc[df['volatility_52w'] > 100, 'risk_score'] = 80
        df.loc[df['volatility_52w'] > 200, 'risk_score'] = 90
    
    return df

# ============================================
# WAVE ANALYSIS
# ============================================

class WaveAnalyzer:
    """Advanced wave pattern analysis"""
    
    @staticmethod
    def calculate_wave_strength(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall wave strength"""
        
        wave_score = pd.Series(50.0, index=df.index)
        
        # Volume wave
        if 'rvol' in df.columns:
            wave_score += (df['rvol'] > 2).astype(float) * 15
            wave_score += (df['rvol'] > 3).astype(float) * 10
        
        # Price wave
        if 'ret_30d' in df.columns:
            wave_score += (df['ret_30d'] > 15).astype(float) * 15
            wave_score += (df['ret_30d'] > 30).astype(float) * 10
        
        df['overall_wave_strength'] = wave_score.clip(0, 100)
        
        # Wave state
        df['wave_state'] = pd.cut(
            df['overall_wave_strength'],
            bins=[0, 40, 70, 100],
            labels=['Calm', 'Building', 'Strong']
        )
        
        return df
    
    @staticmethod
    def detect_wave_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect market-wide wave patterns"""
        
        patterns = {
            'bull_wave': False,
            'distribution': False,
            'accumulation': False,
            'rotation': False
        }
        
        if len(df) == 0:
            return patterns
        
        # Bull wave: >40% stocks in strong uptrend
        if 'ret_30d' in df.columns:
            bull_stocks = (df['ret_30d'] > 20).sum()
            patterns['bull_wave'] = (bull_stocks / len(df)) > 0.4
        
        # High volume accumulation
        if 'rvol' in df.columns:
            high_vol_stocks = (df['rvol'] > 2).sum()
            patterns['accumulation'] = (high_vol_stocks / len(df)) > 0.3
        
        return patterns

# ============================================
# FILTERING ENGINE
# ============================================

class FilterEngine:
    """Unified filtering system with proper interconnection"""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters to dataframe"""
        
        if df.empty:
            return df
        
        mask = pd.Series(True, index=df.index)
        
        # Category filter
        categories = filters.get('categories', [])
        if categories and 'All' not in categories and 'category' in df.columns:
            mask &= df['category'].isin(categories)
        
        # Sector filter
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors and 'sector' in df.columns:
            mask &= df['sector'].isin(sectors)
        
        # Industry filter
        industries = filters.get('industries', [])
        if industries and 'All' not in industries and 'industry' in df.columns:
            mask &= df['industry'].isin(industries)
        
        # Score filter
        min_score = filters.get('min_score', 0)
        if min_score > 0 and 'master_score' in df.columns:
            mask &= df['master_score'] >= min_score
        
        # Pattern filter
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            pattern_mask = pd.Series(False, index=df.index)
            for pattern in patterns:
                pattern_mask |= df['patterns'].str.contains(pattern, case=False, na=False)
            mask &= pattern_mask
        
        # Trend filter
        trend_range = filters.get('trend_range')
        if trend_range and 'momentum_score' in df.columns:
            min_trend, max_trend = trend_range
            mask &= (df['momentum_score'] >= min_trend) & (df['momentum_score'] <= max_trend)
        
        # EPS tier filter
        eps_tiers = filters.get('eps_tiers', [])
        if eps_tiers and 'eps_tier' in df.columns:
            mask &= df['eps_tier'].isin(eps_tiers)
        
        # PE tier filter
        pe_tiers = filters.get('pe_tiers', [])
        if pe_tiers and 'pe_tier' in df.columns:
            mask &= df['pe_tier'].isin(pe_tiers)
        
        # Price tier filter
        price_tiers = filters.get('price_tiers', [])
        if price_tiers and 'price_tier' in df.columns:
            mask &= df['price_tier'].isin(price_tiers)
        
        # EPS change filter
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            mask &= df['eps_change_pct'] >= min_eps_change
        
        # PE range filter
        min_pe = filters.get('min_pe')
        max_pe = filters.get('max_pe')
        if 'pe' in df.columns:
            if min_pe is not None:
                mask &= df['pe'] >= min_pe
            if max_pe is not None:
                mask &= df['pe'] <= max_pe
        
        # Fundamental data filter
        if filters.get('require_fundamental_data', False):
            if 'pe' in df.columns and 'eps_current' in df.columns:
                mask &= df['pe'].notna() & df['eps_current'].notna()
        
        # Wave state filter
        wave_states = filters.get('wave_states', [])
        if wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)
        
        # Wave strength filter
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            mask &= (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws)
        
        filtered_df = df[mask].copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        """Get available filter options with proper interconnection"""
        
        if df.empty or column not in df.columns:
            return []
        
        # Apply all OTHER filters to see interconnected options
        temp_filters = current_filters.copy()
        
        # Map column names to filter keys
        filter_key_map = {
            'category': 'categories',
            'sector': 'sectors',
            'industry': 'industries',
            'eps_tier': 'eps_tiers',
            'pe_tier': 'pe_tiers',
            'price_tier': 'price_tiers',
            'wave_state': 'wave_states'
        }
        
        # Remove current column's filter to see all its options
        if column in filter_key_map:
            temp_filters.pop(filter_key_map[column], None)
        
        # Apply remaining filters
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        
        # Get unique values
        values = filtered_df[column].dropna().unique()
        
        # Clean and sort
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN', 'None']]
        
        # Sort intelligently
        try:
            values = sorted(values, key=lambda x: float(str(x).replace(',', '')) if str(x).replace(',', '').replace('.', '').isdigit() else x)
        except:
            values = sorted(values)
        
        return values

# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """Enhanced search with relevance scoring from V2"""
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with enhanced relevance scoring"""
        
        if not query or df.empty:
            return df
        
        query_lower = query.lower().strip()
        
        # Calculate relevance scores
        scores = pd.Series(0, index=df.index)
        
        # Exact ticker match (highest priority)
        if 'ticker' in df.columns:
            exact_ticker = df['ticker'].str.lower() == query_lower
            scores[exact_ticker] = 100
            
            # Ticker starts with query
            ticker_starts = df['ticker'].str.lower().str.startswith(query_lower)
            scores[ticker_starts & ~exact_ticker] = 80
            
            # Ticker contains query
            ticker_contains = df['ticker'].str.lower().str.contains(query_lower, na=False)
            scores[ticker_contains & ~ticker_starts & ~exact_ticker] = 60
        
        # Company name match
        if 'company_name' in df.columns:
            name_lower = df['company_name'].str.lower()
            
            # Name starts with query
            name_starts = name_lower.str.startswith(query_lower)
            scores[name_starts & (scores < 50)] = 50
            
            # Name contains query
            name_contains = name_lower.str.contains(query_lower, na=False)
            scores[name_contains & (scores < 30)] = 30
        
        # Sector/Industry match
        if 'sector' in df.columns:
            sector_match = df['sector'].str.lower().str.contains(query_lower, na=False)
            scores[sector_match & (scores < 20)] = 20
        
        if 'industry' in df.columns:
            industry_match = df['industry'].str.lower().str.contains(query_lower, na=False)
            scores[industry_match & (scores < 20)] = 20
        
        # Filter and sort by relevance
        matched_df = df[scores > 0].copy()
        matched_df['search_relevance'] = scores[scores > 0]
        matched_df = matched_df.sort_values('search_relevance', ascending=False)
        
        # Remove temporary column
        matched_df = matched_df.drop('search_relevance', axis=1)
        
        return matched_df

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    """Advanced market analysis from V2"""
    
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame, lookback_days: int = 30) -> Dict[str, Any]:
        """Detect sector rotation patterns"""
        
        if df.empty or 'sector' not in df.columns:
            return {}
        
        sector_performance = df.groupby('sector').agg({
            'ret_30d': 'mean',
            'volume_1d': 'sum',
            'master_score': 'mean'
        }).round(2)
        
        sector_performance = sector_performance.sort_values('ret_30d', ascending=False)
        
        rotation_data = {
            'leading_sectors': sector_performance.head(3).index.tolist(),
            'lagging_sectors': sector_performance.tail(3).index.tolist(),
            'sector_scores': sector_performance.to_dict()
        }
        
        return rotation_data
    
    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect industry rotation within sectors"""
        
        if df.empty or 'industry' not in df.columns:
            return {}
        
        industry_performance = df.groupby(['sector', 'industry']).agg({
            'ret_30d': 'mean',
            'master_score': 'mean',
            'ticker': 'count'
        }).round(2)
        
        industry_performance = industry_performance.rename(columns={'ticker': 'stock_count'})
        industry_performance = industry_performance.sort_values('master_score', ascending=False)
        
        # Top industries per sector
        top_industries = {}
        for sector in df['sector'].unique():
            if sector in industry_performance.index:
                sector_industries = industry_performance.loc[sector].head(3)
                top_industries[sector] = sector_industries.index.tolist()
        
        return {
            'top_industries_by_sector': top_industries,
            'industry_metrics': industry_performance.head(10).to_dict()
        }
    
    @staticmethod
    def calculate_market_breadth(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market breadth indicators"""
        
        if df.empty:
            return {}
        
        breadth = {}
        
        # Advance/Decline ratio
        if 'ret_1d' in df.columns:
            advances = (df['ret_1d'] > 0).sum()
            declines = (df['ret_1d'] < 0).sum()
            breadth['advance_decline_ratio'] = advances / max(declines, 1)
        
        # Stocks above key SMAs
        if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']):
            price_num = clean_numeric_column(df['price'])
            sma50_num = clean_numeric_column(df['sma_50d'])
            sma200_num = clean_numeric_column(df['sma_200d'])
            
            breadth['pct_above_50ma'] = (price_num > sma50_num).mean() * 100
            breadth['pct_above_200ma'] = (price_num > sma200_num).mean() * 100
        
        # New highs/lows
        if 'from_high_pct' in df.columns:
            breadth['new_highs'] = (df['from_high_pct'] > -1).sum()
            breadth['near_lows'] = (df['from_low_pct'] < 10).sum()
        
        return breadth

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_metric_card(label: str, value: str, delta: str = None):
        """Render a metric card"""
        if delta:
            st.metric(label=label, value=value, delta=delta)
        else:
            st.metric(label=label, value=value)
    
    @staticmethod
    def render_stock_card(row: pd.Series, show_fundamentals: bool = False):
        """Render a detailed stock card"""
        
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{row['ticker']}** - {row.get('company_name', 'N/A')}")
                st.caption(f"{row.get('category', 'N/A')} | {row.get('sector', 'N/A')}")
            
            with col2:
                price_str = row.get('price', 'N/A')
                ret_30d = row.get('ret_30d', 0)
                delta_color = "green" if ret_30d > 0 else "red"
                st.markdown(f"**{price_str}**")
                st.markdown(f"<span style='color:{delta_color}'>{ret_30d:+.1f}%</span>", 
                           unsafe_allow_html=True)
            
            with col3:
                st.metric("Score", f"{row.get('master_score', 0):.1f}")
            
            with col4:
                st.metric("Rank", f"#{row.get('rank', 'N/A')}")
            
            if show_fundamentals and all(col in row.index for col in ['pe', 'eps_current']):
                st.caption(f"PE: {row['pe']:.1f} | EPS: ₹{row['eps_current']:.1f}")

# ============================================
# DATA EXPORT
# ============================================

def prepare_export_data(df: pd.DataFrame, template: str) -> pd.DataFrame:
    """Prepare data for export based on template"""
    
    export_columns = {
        'essential': [
            'rank', 'ticker', 'company_name', 'price', 'master_score',
            'ret_1d', 'ret_30d', 'rvol', 'patterns'
        ],
        'detailed': [
            'rank', 'ticker', 'company_name', 'category', 'sector', 'industry',
            'price', 'master_score', 'position_score', 'volume_score',
            'momentum_score', 'ret_1d', 'ret_7d', 'ret_30d',
            'from_low_pct', 'from_high_pct', 'rvol', 'patterns'
        ],
        'full': df.columns.tolist()
    }
    
    if template == "Essential (Top 20)":
        columns = [col for col in export_columns['essential'] if col in df.columns]
        return df.head(20)[columns]
    elif template == "Detailed (Top 50)":
        columns = [col for col in export_columns['detailed'] if col in df.columns]
        return df.head(50)[columns]
    else:  # Full Analysis
        return df

def generate_excel_report(dfs: Dict[str, pd.DataFrame]) -> BytesIO:
    """Generate Excel report with multiple sheets"""
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BD',
                'border': 1
            })
            
            # Format headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-fit columns
            for i, col in enumerate(df.columns):
                column_width = max(df[col].astype(str).str.len().max(), len(col)) + 2
                worksheet.set_column(i, i, min(column_width, 50))
    
    output.seek(0)
    return output

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
    
    # Initialize session state
    SessionStateManager.initialize()
    
    # Custom CSS
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
    /* Button styling */
    div.stButton > button {
        width: 100%;
    }
    /* Metric container enhancements */
    [data-testid="metric-container"] > div {
        width: fit-content;
        margin: auto;
    }
    /* Sidebar enhancements */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
        width: 300px !important;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"] {
            background-color: #262730;
        }
    }
    /* Search box styling */
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    /* DataFrame enhancements */
    .dataframe {
        font-size: 14px;
    }
    .dataframe tbody tr:hover {
        background-color: rgba(28, 131, 225, 0.1);
    }
    /* Tab panel styling */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    /* Card styling */
    .stock-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .stock-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    /* Mobile responsive */
    @media (max-width: 768px) {
        .stDataFrame {font-size: 12px;}
        div[data-testid="metric-container"] {padding: 3%;}
        .main {padding: 0rem 0.5rem;}
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
    
    # Sidebar
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
        
        # Data source selection
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        data_source_col1, data_source_col2 = st.columns(2)
        with data_source_col1:
            if st.button("📊 Google Sheets", 
                        type="primary" if st.session_state.data_source == "sheet" else "secondary", 
                        use_container_width=True):
                st.session_state.data_source = "sheet"
                st.rerun()
        
        with data_source_col2:
            if st.button("📁 Upload CSV", 
                        type="primary" if st.session_state.data_source == "upload" else "secondary", 
                        use_container_width=True):
                st.session_state.data_source = "upload"
                st.rerun()

        uploaded_file = None
        sheet_id = None
        gid = None
        
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="Upload a CSV file with stock data. Must contain 'ticker' and 'price' columns."
            )
            if uploaded_file is None:
                st.info("Please upload a CSV file to continue")
        else:
            # Google Sheets configuration
            st.markdown("#### 📊 Google Sheets Config")
            
            # Sheet ID/URL input
            sheet_input = st.text_input(
                "Google Sheets ID or URL",
                value=st.session_state.get('sheet_id', CONFIG.DEFAULT_SHEET_ID),
                placeholder="Enter Sheet ID or full URL",
                help="Paste the full Google Sheets URL or just the ID"
            )
            
            if sheet_input:
                # Extract sheet ID from URL if needed
                if sheet_input.startswith('http'):
                    match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_input)
                    if match:
                        sheet_id = match.group(1)
                    else:
                        st.error("Invalid Google Sheets URL")
                else:
                    sheet_id = sheet_input.strip()
                
                if sheet_id:
                    st.session_state.sheet_id = sheet_id
            
            # GID input
            gid_input = st.text_input(
                "Sheet Tab GID (Optional)",
                value=st.session_state.get('gid', CONFIG.DEFAULT_GID),
                placeholder=f"Default: {CONFIG.DEFAULT_GID}",
                help="The GID identifies the specific sheet tab"
            )
            
            if gid_input:
                gid = gid_input.strip()
                st.session_state.gid = gid
        
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
                    
                    duplicates = quality.get('duplicates', 0)
                    st.metric("Duplicates", f"{duplicates}")
        
        # Filters Section
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        
        # Check if clear was triggered from main area
        if st.session_state.get('trigger_clear', False):
            SessionStateManager.clear_filters()
            st.rerun()
        
        # Debug mode
        st.markdown("---")
        show_debug = st.checkbox("🐛 Show Debug Info", 
                               value=st.session_state.get('show_debug', False),
                               key="show_debug")
    
    # Data loading
    try:
        # Check if we need to load data
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        if st.session_state.data_source == "sheet" and not sheet_id:
            sheet_id = CONFIG.DEFAULT_SHEET_ID
        
        # Load and process data
        with st.spinner("📥 Loading and processing data..."):
            try:
                ranked_df, data_timestamp, metadata = load_and_process_data(
                    st.session_state.data_source,
                    sheet_id=sheet_id,
                    gid=gid,
                    file_data=uploaded_file
                )
                
                st.session_state.ranked_df = ranked_df
                st.session_state.data_timestamp = data_timestamp
                st.session_state.last_refresh = datetime.now(timezone.utc)
                
                # Calculate data quality
                total_cells = len(ranked_df) * len(ranked_df.columns)
                non_null_cells = ranked_df.count().sum()
                completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
                
                st.session_state.data_quality = {
                    'completeness': completeness,
                    'total_rows': len(ranked_df),
                    'duplicates': metadata.get('duplicates', 0),
                    'timestamp': data_timestamp
                }
                
                # Show warnings/errors
                if metadata.get('warnings'):
                    for warning in metadata['warnings']:
                        st.warning(warning)
                
                if metadata.get('errors'):
                    for error in metadata['errors']:
                        st.error(error)
                
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}")
                
                # Try cached data
                if hasattr(st.session_state, 'last_good_data'):
                    ranked_df, data_timestamp, metadata = st.session_state.last_good_data
                    st.warning("Using cached data due to load failure")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please check:\n- Google Sheets ID is correct\n- Sheet is publicly accessible\n- CSV format is valid")
                    st.stop()
        
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        st.stop()
    
    # Quick Action Buttons
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
    quick_filter_applied = st.session_state.get('quick_filter_applied', False)
    quick_filter = st.session_state.get('quick_filter', None)
    
    with qa_col1:
        if st.button("📈 Top Gainers", use_container_width=True):
            st.session_state.quick_filter = 'top_gainers'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_col2:
        if st.button("🔥 Volume Surges", use_container_width=True):
            st.session_state.quick_filter = 'volume_surges'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_col3:
        if st.button("🎯 Breakout Ready", use_container_width=True):
            st.session_state.quick_filter = 'breakout_ready'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_col4:
        if st.button("💎 Hidden Gems", use_container_width=True):
            st.session_state.quick_filter = 'hidden_gems'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_col5:
        if st.button("🌊 Show All", use_container_width=True):
            st.session_state.quick_filter = None
            st.session_state.quick_filter_applied = False
            st.rerun()
    
    # Apply quick filters
    if quick_filter:
        if quick_filter == 'top_gainers':
            ranked_df_display = ranked_df[ranked_df['momentum_score'] >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ 80")
        elif quick_filter == 'volume_surges':
            ranked_df_display = ranked_df[ranked_df['rvol'] > 2]
            st.info(f"Showing {len(ranked_df_display)} stocks with RVOL > 2x")
        elif quick_filter == 'breakout_ready':
            ranked_df_display = ranked_df[
                (ranked_df['from_high_pct'] > -10) & 
                (ranked_df['volume_score'] > 70)
            ]
            st.info(f"Showing {len(ranked_df_display)} stocks near 52W high with strong volume")
        elif quick_filter == 'hidden_gems':
            ranked_df_display = ranked_df[
                (ranked_df['master_score'] > 70) & 
                (ranked_df['rank'] > 50)
            ]
            st.info(f"Showing {len(ranked_df_display)} high-scoring stocks ranked >50")
        else:
            ranked_df_display = ranked_df
    else:
        ranked_df_display = ranked_df
    
    # Add wave analysis
    ranked_df_display = WaveAnalyzer.calculate_wave_strength(ranked_df_display)
    
    # Main Tabs
    tabs = st.tabs([
        "📊 Rankings", 
        "📈 Analysis", 
        "🔍 Deep Dive",
        "🌊 Wave Radar",
        "🧪 Pattern Lab",
        "📥 Export",
        "ℹ️ About"
    ])
    
    # Tab 1: Rankings
    with tabs[0]:
        # Search and filter controls
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search stocks",
                value=st.session_state.get('search_query', ''),
                placeholder="Enter ticker, name, sector...",
                key="search_input"
            )
            st.session_state.search_query = search_query
        
        with col2:
            top_n = st.selectbox(
                "Show Top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n'])
            )
            st.session_state.user_preferences['default_top_n'] = top_n
        
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=['Rank', 'Score', 'Returns', 'Volume'],
                index=0
            )
        
        # Apply search
        if search_query:
            display_df = SearchEngine.search_stocks(ranked_df_display, search_query)
        else:
            display_df = ranked_df_display
        
        # Apply sorting
        if sort_by == 'Rank':
            display_df = display_df.sort_values('rank')
        elif sort_by == 'Score':
            display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'Returns':
            display_df = display_df.sort_values('ret_30d', ascending=False)
        elif sort_by == 'Volume':
            display_df = display_df.sort_values('rvol', ascending=False)
        
        # Limit to top N
        display_df = display_df.head(top_n)
        
        # Summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Stocks", f"{len(display_df):,}")
        
        with col2:
            avg_score = display_df['master_score'].mean() if not display_df.empty else 0
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        with col3:
            gainers = (display_df['ret_30d'] > 0).sum() if 'ret_30d' in display_df.columns else 0
            st.metric("Gainers", f"{gainers}")
        
        with col4:
            high_rvol = (display_df['rvol'] > 2).sum() if 'rvol' in display_df.columns else 0
            st.metric("High RVOL", f"{high_rvol}")
        
        with col5:
            with_patterns = (display_df['patterns'] != '').sum() if 'patterns' in display_df.columns else 0
            st.metric("With Patterns", f"{with_patterns}")
        
        with col6:
            near_high = (display_df['from_high_pct'] > -10).sum() if 'from_high_pct' in display_df.columns else 0
            st.metric("Near 52W High", f"{near_high}")
        
        # Display table
        st.markdown("---")
        
        if not display_df.empty:
            # Prepare display columns
            display_columns = [
                'rank', 'ticker', 'company_name', 'category', 'sector', 'industry',
                'price', 'master_score', 'ret_1d', 'ret_30d', 'rvol', 
                'from_high_pct', 'patterns'
            ]
            
            # Only show columns that exist
            display_columns = [col for col in display_columns if col in display_df.columns]
            
            # Format display
            formatted_df = display_df[display_columns].copy()
            
            # Format numeric columns
            if 'master_score' in formatted_df.columns:
                formatted_df['master_score'] = formatted_df['master_score'].round(1)
            if 'ret_1d' in formatted_df.columns:
                formatted_df['ret_1d'] = formatted_df['ret_1d'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
            if 'ret_30d' in formatted_df.columns:
                formatted_df['ret_30d'] = formatted_df['ret_30d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
            if 'rvol' in formatted_df.columns:
                formatted_df['rvol'] = formatted_df['rvol'].round(1)
            if 'from_high_pct' in formatted_df.columns:
                formatted_df['from_high_pct'] = formatted_df['from_high_pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
            
            # Display with styling
            st.dataframe(
                formatted_df,
                use_container_width=True,
                height=600,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "company_name": st.column_config.TextColumn("Company", width="large"),
                    "master_score": st.column_config.ProgressColumn(
                        "Score",
                        min_value=0,
                        max_value=100,
                        format="%.1f"
                    ),
                    "price": st.column_config.TextColumn("Price", width="small"),
                    "ret_1d": st.column_config.TextColumn("1D %", width="small"),
                    "ret_30d": st.column_config.TextColumn("30D %", width="small"),
                    "rvol": st.column_config.NumberColumn("RVOL", format="%.1f", width="small"),
                }
            )
        else:
            st.info("No stocks match your criteria")
    
    # Tab 2: Analysis
    with tabs[1]:
        st.markdown("### 📊 Market Analysis Dashboard")
        
        # Market breadth
        breadth = MarketIntelligence.calculate_market_breadth(ranked_df)
        
        if breadth:
            st.markdown("#### Market Breadth Indicators")
            breadth_cols = st.columns(4)
            
            with breadth_cols[0]:
                adv_dec = breadth.get('advance_decline_ratio', 0)
                st.metric("Advance/Decline", f"{adv_dec:.2f}")
            
            with breadth_cols[1]:
                above_50 = breadth.get('pct_above_50ma', 0)
                st.metric("Above 50 MA", f"{above_50:.1f}%")
            
            with breadth_cols[2]:
                above_200 = breadth.get('pct_above_200ma', 0)
                st.metric("Above 200 MA", f"{above_200:.1f}%")
            
            with breadth_cols[3]:
                new_highs = breadth.get('new_highs', 0)
                st.metric("Near 52W High", f"{new_highs}")
        
        # Sector rotation
        st.markdown("---")
        st.markdown("#### 🔄 Sector Rotation Analysis")
        
        sector_data = MarketIntelligence.detect_sector_rotation(ranked_df)
        
        if sector_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Leading Sectors:**")
                for sector in sector_data.get('leading_sectors', []):
                    scores = sector_data['sector_scores']
                    ret = scores.get('ret_30d', {}).get(sector, 0)
                    st.write(f"✅ {sector} ({ret:+.1f}%)")
            
            with col2:
                st.markdown("**Lagging Sectors:**")
                for sector in sector_data.get('lagging_sectors', []):
                    scores = sector_data['sector_scores']
                    ret = scores.get('ret_30d', {}).get(sector, 0)
                    st.write(f"⚠️ {sector} ({ret:+.1f}%)")
        
        # Industry rotation
        industry_data = MarketIntelligence.detect_industry_rotation(ranked_df)
        
        if industry_data and industry_data.get('top_industries_by_sector'):
            st.markdown("---")
            st.markdown("#### 🏭 Top Industries by Sector")
            
            for sector, industries in industry_data['top_industries_by_sector'].items():
                with st.expander(f"{sector}"):
                    for industry in industries[:3]:
                        st.write(f"• {industry}")
        
        # Performance distribution
        st.markdown("---")
        st.markdown("#### 📈 Performance Distribution")
        
        if 'ret_30d' in ranked_df.columns:
            fig = px.histogram(
                ranked_df,
                x='ret_30d',
                nbins=50,
                title="30-Day Returns Distribution",
                labels={'ret_30d': '30-Day Return (%)', 'count': 'Number of Stocks'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Apply filters in sidebar
    with st.sidebar:
        filters = {}
        
        # Display mode
        display_mode = st.radio(
            "Display Mode",
            ["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if st.session_state.user_preferences['display_mode'] == "Technical" else 1,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data",
            key="display_mode_toggle"
        )
        
        st.session_state.user_preferences['display_mode'] = display_mode
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        # Category filter
        categories = FilterEngine.get_filter_options(ranked_df_display, 'category', filters)
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=categories,
            default=st.session_state.get('category_filter', []),
            placeholder="Select categories",
            key="category_filter"
        )
        filters['categories'] = selected_categories
        
        # Sector filter
        sectors = FilterEngine.get_filter_options(ranked_df_display, 'sector', filters)
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=st.session_state.get('sector_filter', []),
            placeholder="Select sectors",
            key="sector_filter"
        )
        filters['sectors'] = selected_sectors
        
        # Industry filter
        if 'industry' in ranked_df_display.columns:
            industries = FilterEngine.get_filter_options(ranked_df_display, 'industry', filters)
            selected_industries = st.multiselect(
                "Industry",
                options=industries,
                default=st.session_state.get('industry_filter', []),
                placeholder="Select industries",
                key="industry_filter"
            )
            filters['industries'] = selected_industries
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=st.session_state.get('min_score', 0),
            step=5,
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
                placeholder="Select patterns",
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
        
        trend_filter = st.selectbox(
            "Trend Filter",
            options=list(trend_options.keys()),
            index=list(trend_options.keys()).index(st.session_state.get('trend_filter', 'All Trends')),
            key="trend_filter"
        )
        filters['trend_range'] = trend_options[trend_filter]
        
        # Fundamental filters
        if show_fundamentals:
            st.markdown("#### 💰 Fundamental Filters")
            
            # EPS tier filter
            if 'eps_tier' in ranked_df_display.columns:
                eps_tiers = FilterEngine.get_filter_options(ranked_df_display, 'eps_tier', filters)
                filters['eps_tiers'] = st.multiselect(
                    "EPS Growth Tier",
                    options=eps_tiers,
                    default=st.session_state.get('eps_tier_filter', []),
                    key="eps_tier_filter"
                )
            
            # PE tier filter
            if 'pe_tier' in ranked_df_display.columns:
                pe_tiers = FilterEngine.get_filter_options(ranked_df_display, 'pe_tier', filters)
                filters['pe_tiers'] = st.multiselect(
                    "PE Valuation Tier",
                    options=pe_tiers,
                    default=st.session_state.get('pe_tier_filter', []),
                    key="pe_tier_filter"
                )
            
            # Price tier filter
            if 'price_tier' in ranked_df_display.columns:
                price_tiers = FilterEngine.get_filter_options(ranked_df_display, 'price_tier', filters)
                filters['price_tiers'] = st.multiselect(
                    "Price Range Tier",
                    options=price_tiers,
                    default=st.session_state.get('price_tier_filter', []),
                    key="price_tier_filter"
                )
            
            # EPS change filter
            min_eps_input = st.text_input(
                "Min EPS Change %",
                value=st.session_state.get('min_eps_change', ''),
                placeholder="e.g. 20",
                key="min_eps_change"
            )
            
            if min_eps_input.strip():
                try:
                    filters['min_eps_change'] = float(min_eps_input)
                except ValueError:
                    st.error("Invalid EPS change value")
            
            # PE range filter
            col1, col2 = st.columns(2)
            
            with col1:
                min_pe_input = st.text_input(
                    "Min PE Ratio",
                    value=st.session_state.get('min_pe', ''),
                    placeholder="e.g. 10",
                    key="min_pe"
                )
                
                if min_pe_input.strip():
                    try:
                        filters['min_pe'] = float(min_pe_input)
                    except ValueError:
                        st.error("Invalid Min PE")
            
            with col2:
                max_pe_input = st.text_input(
                    "Max PE Ratio",
                    value=st.session_state.get('max_pe', ''),
                    placeholder="e.g. 30",
                    key="max_pe"
                )
                
                if max_pe_input.strip():
                    try:
                        filters['max_pe'] = float(max_pe_input)
                    except ValueError:
                        st.error("Invalid Max PE")
            
            # Data completeness filter
            filters['require_fundamental_data'] = st.checkbox(
                "Only show stocks with PE and EPS data",
                value=st.session_state.get('require_fundamental_data', False),
                key="require_fundamental_data"
            )
        
        # Count active filters
        active_filter_count = 0
        filter_checks = [
            ('categories', lambda x: x and len(x) > 0),
            ('sectors', lambda x: x and len(x) > 0),
            ('industries', lambda x: x and len(x) > 0),
            ('min_score', lambda x: x > 0),
            ('patterns', lambda x: x and len(x) > 0),
            ('trend_filter', lambda x: x != 'All Trends'),
            ('eps_tiers', lambda x: x and len(x) > 0),
            ('pe_tiers', lambda x: x and len(x) > 0),
            ('price_tiers', lambda x: x and len(x) > 0),
            ('min_eps_change', lambda x: x is not None),
            ('min_pe', lambda x: x is not None),
            ('max_pe', lambda x: x is not None),
            ('require_fundamental_data', lambda x: x),
        ]
        
        for key, check_func in filter_checks:
            value = filters.get(key) or st.session_state.get(key.replace('s', '_filter') if key.endswith('s') else key)
            if value is not None and check_func(value):
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
            time.sleep(0.5)
            st.rerun()
    
    # Apply filters
    if quick_filter_applied:
        filtered_df = FilterEngine.apply_filters(ranked_df_display, filters)
    else:
        filtered_df = FilterEngine.apply_filters(ranked_df, filters)
    
    filtered_df = filtered_df.sort_values('rank')
    
    # Save filters
    st.session_state.user_preferences['last_filters'] = filters
    
    # Debug info
    if show_debug:
        with st.sidebar.expander("🐛 Debug Info", expanded=True):
            st.write("**Active Filters:**")
            for key, value in filters.items():
                if value:
                    st.write(f"• {key}: {value}")
            
            st.write(f"\n**Filter Result:**")
            st.write(f"Before: {len(ranked_df)} stocks")
            st.write(f"After: {len(filtered_df)} stocks")
            
            if st.session_state.performance_metrics:
                st.write(f"\n**Performance:**")
                for func, time_taken in st.session_state.performance_metrics.items():
                    st.write(f"• {func}: {time_taken:.4f}s")
    
    # Filter status in main area
    if st.session_state.active_filter_count > 0 or quick_filter_applied:
        filter_status_col1, filter_status_col2 = st.columns([5, 1])
        with filter_status_col1:
            filter_msg = f"**{len(filtered_df):,} stocks** shown"
            if quick_filter:
                filter_msg = f"Quick filter: {quick_filter.replace('_', ' ').title()} | " + filter_msg
            if st.session_state.active_filter_count > 0:
                filter_msg += f" | {st.session_state.active_filter_count} filter{'s' if st.session_state.active_filter_count > 1 else ''} active"
            st.info(filter_msg)
        
        with filter_status_col2:
            if st.button("Clear Filters", type="secondary"):
                st.session_state.trigger_clear = True
                st.rerun()
    
    # Tab 3: Deep Dive
    with tabs[2]:
        st.markdown("### 🔬 Stock Deep Dive")
        
        if not filtered_df.empty:
            # Stock selector
            selected_ticker = st.selectbox(
                "Select a stock to analyze",
                options=filtered_df['ticker'].tolist(),
                format_func=lambda x: f"{x} - {filtered_df[filtered_df['ticker']==x]['company_name'].iloc[0]}"
            )
            
            if selected_ticker:
                stock_data = filtered_df[filtered_df['ticker'] == selected_ticker].iloc[0]
                
                # Header metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Rank", f"#{stock_data['rank']}")
                
                with col2:
                    st.metric("Master Score", f"{stock_data['master_score']:.1f}")
                
                with col3:
                    price = stock_data.get('price', 'N/A')
                    st.metric("Price", price)
                
                with col4:
                    ret_30d = stock_data.get('ret_30d', 0)
                    st.metric("30D Return", f"{ret_30d:+.1f}%")
                
                with col5:
                    rvol = stock_data.get('rvol', 1)
                    st.metric("RVOL", f"{rvol:.1f}x")
                
                # Detailed analysis
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Score Breakdown")
                    scores = {
                        'Position Score': stock_data.get('position_score', 0),
                        'Volume Score': stock_data.get('volume_score', 0),
                        'Momentum Score': stock_data.get('momentum_score', 0),
                        'Acceleration Score': stock_data.get('acceleration_score', 0),
                        'Breakout Score': stock_data.get('breakout_score', 0),
                        'RVOL Score': stock_data.get('rvol_score', 0)
                    }
                    
                    for score_name, score_value in scores.items():
                        st.progress(score_value/100, text=f"{score_name}: {score_value:.1f}")
                
                with col2:
                    st.markdown("#### 🎯 Key Metrics")
                    
                    metrics_data = {
                        'Category': stock_data.get('category', 'N/A'),
                        'Sector': stock_data.get('sector', 'N/A'),
                        'Industry': stock_data.get('industry', 'N/A'),
                        '52W High %': f"{stock_data.get('from_high_pct', 0):.1f}%",
                        '52W Low %': f"{stock_data.get('from_low_pct', 0):.1f}%",
                        'Patterns': stock_data.get('patterns', 'None')
                    }
                    
                    if show_fundamentals:
                        metrics_data.update({
                            'PE Ratio': f"{stock_data.get('pe', 'N/A')}",
                            'EPS': f"₹{stock_data.get('eps_current', 'N/A')}",
                            'EPS Change': f"{stock_data.get('eps_change_pct', 0):.1f}%"
                        })
                    
                    for metric, value in metrics_data.items():
                        st.write(f"**{metric}:** {value}")
                
                # Price action chart
                st.markdown("---")
                st.markdown("#### 📈 Technical Overview")
                
                # Create simple price visualization
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=['52W Low', 'Current', '52W High'],
                    y=[
                        clean_numeric_column(pd.Series([stock_data.get('low_52w', '0')])).iloc[0],
                        clean_numeric_column(pd.Series([stock_data.get('price', '0')])).iloc[0],
                        clean_numeric_column(pd.Series([stock_data.get('high_52w', '0')])).iloc[0]
                    ],
                    mode='lines+markers',
                    name='Price Position',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title=f"{selected_ticker} - 52 Week Range Position",
                    xaxis_title="",
                    yaxis_title="Price (₹)",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stocks available for analysis")
    
    # Tab 4: Wave Radar
    with tabs[3]:
        st.markdown("### 🌊 Wave Pattern Radar")
        
        # Wave analysis
        wave_patterns = WaveAnalyzer.detect_wave_patterns(filtered_df)
        
        # Wave indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bull_wave = wave_patterns.get('bull_wave', False)
            st.metric("Bull Wave", "🟢 Active" if bull_wave else "⚫ Inactive")
        
        with col2:
            accumulation = wave_patterns.get('accumulation', False)
            st.metric("Accumulation", "🟢 Yes" if accumulation else "⚫ No")
        
        with col3:
            if 'overall_wave_strength' in filtered_df.columns:
                avg_wave = filtered_df['overall_wave_strength'].mean()
                st.metric("Avg Wave Strength", f"{avg_wave:.1f}")
        
        with col4:
            strong_waves = (filtered_df['wave_state'] == 'Strong').sum() if 'wave_state' in filtered_df.columns else 0
            st.metric("Strong Waves", f"{strong_waves}")
        
        # Wave distribution
        st.markdown("---")
        
        if 'wave_state' in filtered_df.columns:
            wave_dist = filtered_df['wave_state'].value_counts()
            
            fig = px.pie(
                values=wave_dist.values,
                names=wave_dist.index,
                title="Wave State Distribution",
                color_discrete_map={'Calm': '#3498db', 'Building': '#f39c12', 'Strong': '#e74c3c'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top wave stocks
        st.markdown("---")
        st.markdown("#### 🏄 Top Wave Riders")
        
        if 'overall_wave_strength' in filtered_df.columns:
            top_waves = filtered_df.nlargest(10, 'overall_wave_strength')[
                ['ticker', 'company_name', 'overall_wave_strength', 'wave_state', 'ret_30d', 'rvol']
            ]
            
            st.dataframe(
                top_waves,
                use_container_width=True,
                column_config={
                    "overall_wave_strength": st.column_config.ProgressColumn(
                        "Wave Strength",
                        min_value=0,
                        max_value=100,
                        format="%.1f"
                    ),
                    "ret_30d": st.column_config.NumberColumn("30D Return", format="%.1f%%"),
                    "rvol": st.column_config.NumberColumn("RVOL", format="%.1fx")
                }
            )
    
    # Tab 5: Pattern Lab
    with tabs[4]:
        st.markdown("### 🧪 Pattern Detection Laboratory")
        
        # Pattern statistics
        if 'patterns' in filtered_df.columns:
            all_patterns = []
            for patterns in filtered_df['patterns'].dropna():
                if patterns:
                    all_patterns.extend(patterns.split(' | '))
            
            if all_patterns:
                pattern_counts = pd.Series(all_patterns).value_counts()
                
                # Pattern frequency chart
                fig = px.bar(
                    x=pattern_counts.values,
                    y=pattern_counts.index,
                    orientation='h',
                    title="Pattern Frequency Analysis",
                    labels={'x': 'Number of Stocks', 'y': 'Pattern'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stocks by pattern
                st.markdown("---")
                selected_pattern = st.selectbox(
                    "Select a pattern to view stocks",
                    options=pattern_counts.index.tolist()
                )
                
                if selected_pattern:
                    pattern_stocks = filtered_df[
                        filtered_df['patterns'].str.contains(selected_pattern, na=False)
                    ].head(20)
                    
                    st.markdown(f"#### Stocks with '{selected_pattern}' pattern")
                    
                    display_cols = ['ticker', 'company_name', 'master_score', 'ret_30d', 'rvol', 'patterns']
                    display_cols = [col for col in display_cols if col in pattern_stocks.columns]
                    
                    st.dataframe(
                        pattern_stocks[display_cols],
                        use_container_width=True,
                        column_config={
                            "master_score": st.column_config.ProgressColumn(
                                "Score",
                                min_value=0,
                                max_value=100,
                                format="%.1f"
                            ),
                            "ret_30d": st.column_config.NumberColumn("30D %", format="%.1f%%"),
                            "rvol": st.column_config.NumberColumn("RVOL", format="%.1fx")
                        }
                    )
    
    # Tab 6: Export
    with tabs[5]:
        st.markdown("### 📥 Export Data")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.radio(
                "Export Format",
                ["CSV", "Excel"],
                index=0
            )
        
        with col2:
            export_template = st.radio(
                "Export Template",
                ["Essential (Top 20)", "Detailed (Top 50)", "Full Analysis (All Data)"],
                index=1,
                key="export_template_radio"
            )
        
        # Prepare export data
        export_df = prepare_export_data(filtered_df, export_template)
        
        # Export buttons
        st.markdown("---")
        
        if export_format == "CSV":
            csv = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {export_template} as CSV",
                data=csv,
                file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:  # Excel
            # Create Excel with multiple sheets
            excel_data = {
                'Rankings': export_df,
                'Summary': pd.DataFrame({
                    'Metric': ['Total Stocks', 'Average Score', 'Top Gainer', 'Highest RVOL'],
                    'Value': [
                        len(filtered_df),
                        filtered_df['master_score'].mean(),
                        filtered_df.nlargest(1, 'ret_30d')['ticker'].iloc[0] if not filtered_df.empty else 'N/A',
                        filtered_df.nlargest(1, 'rvol')['ticker'].iloc[0] if not filtered_df.empty else 'N/A'
                    ]
                })
            }
            
            if 'sector' in filtered_df.columns:
                sector_summary = filtered_df.groupby('sector').agg({
                    'ticker': 'count',
                    'master_score': 'mean',
                    'ret_30d': 'mean'
                }).round(2).rename(columns={'ticker': 'count'})
                excel_data['Sector Analysis'] = sector_summary
            
            excel_file = generate_excel_report(excel_data)
            
            st.download_button(
                label=f"📥 Download {export_template} as Excel",
                data=excel_file,
                file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Export preview
        st.markdown("---")
        st.markdown("#### 📊 Export Preview")
        
        preview_stats = {
            "Total Stocks": len(export_df),
            "Columns": len(export_df.columns),
            "File Size (est.)": f"{len(export_df) * len(export_df.columns) * 50 / 1024:.1f} KB"
        }
        
        stat_cols = st.columns(len(preview_stats))
        for i, (label, value) in enumerate(preview_stats.items()):
            with stat_cols[i]:
                st.metric(label, value)
        
        # Show first few rows
        st.markdown("First 5 rows:")
        st.dataframe(export_df.head(), use_container_width=True)
    
    # Tab 7: About
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
            The most advanced stock ranking system designed to catch momentum waves early.
            This professional-grade tool combines technical analysis, volume dynamics, 
            advanced metrics, and pattern recognition to identify high-potential stocks.
            
            **Key Features:**
            - **Master Score 3.0**: Multi-factor ranking algorithm
            - **Wave Analysis**: Advanced momentum detection
            - **Pattern Recognition**: 10+ technical patterns
            - **Smart Filters**: Interconnected filtering system
            - **Real-time Data**: Google Sheets integration
            - **Market Intelligence**: Sector rotation analysis
            
            **Score Components:**
            - Position Score (30%): 52-week range analysis
            - Volume Score (25%): Multi-timeframe volume analysis
            - Momentum Score (15%): Price momentum metrics
            - Acceleration Score (10%): Trend acceleration
            - Breakout Score (10%): Breakout potential
            - RVOL Score (10%): Relative volume strength
            
            **Data Requirements:**
            - Ticker symbol
            - Current price
            - Volume data (1d, 7d, 30d, 90d, 180d)
            - Price metrics (52w high/low, returns)
            - Optional: Fundamentals (PE, EPS)
            """)
        
        with col2:
            st.markdown("""
            #### 🎯 Quick Tips
            
            1. **Quick Filters**: Use preset filters for rapid analysis
            2. **Search**: Find stocks by ticker, name, or sector
            3. **Filters**: Combine multiple filters for precision
            4. **Export**: Download results in CSV or Excel
            5. **Wave Radar**: Monitor market momentum
            
            #### 📊 Data Sources
            
            - Google Sheets (Live)
            - CSV Upload
            - IST Timezone
            - ₹ INR Currency
            
            #### 🚀 Performance
            
            - Handles 1000+ stocks
            - Real-time calculations
            - Optimized for mobile
            - Cloud-ready
            
            ---
            
            **Version**: 3.0.0 Final
            **Status**: Production Ready
            """)
        
        # System stats
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.metric("Total Stocks", f"{len(ranked_df):,}")
        
        with stats_cols[1]:
            st.metric("Filtered Stocks", f"{len(filtered_df):,}")
        
        with stats_cols[2]:
            quality = st.session_state.data_quality.get('completeness', 0)
            st.metric("Data Quality", f"{quality:.1f}%")
        
        with stats_cols[3]:
            age = datetime.now(timezone.utc) - st.session_state.last_refresh
            minutes = int(age.total_seconds() / 60)
            st.metric("Cache Age", f"{minutes} min")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            🌊 Wave Detection Ultimate 3.0 - Final Production Version<br>
            <small>Professional Stock Ranking System • All Features Complete</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================
# APPLICATION ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}")
        logger.error(f"Application crashed: {str(e)}", exc_info=True)
        
        if st.button("🔄 Restart Application"):
            st.cache_data.clear()
            st.rerun()

# END OF WAVE DETECTION ULTIMATE 3.0
