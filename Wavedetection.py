#!/usr/bin/env python3
"""
Wave Detection Ultimate 3.0 - Final Production Version
A professional-grade stock screening and ranking system

Architecture: Clean, Simple, Bulletproof
Philosophy: Every line must earn its place
"""

# ============================================
# IMPORTS - Only What's Needed
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import gc
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from functools import wraps
import logging
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# CONFIGURATION - Production Settings
# ============================================
@dataclass
class Config:
    """Global configuration - DO NOT MODIFY IN PRODUCTION"""
    
    # Data source
    DEFAULT_GID: str = "0"  # Default Google Sheets tab
    
    # Critical columns that must exist
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: [
        'ticker', 'company_name', 'price'
    ])
    
    # Core scoring weights - PROVEN IN PRODUCTION
    SCORING_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'position_score': 0.30,
        'volume_score': 0.25,
        'momentum_score': 0.15,
        'acceleration_score': 0.10,
        'breakout_score': 0.10,
        'rvol_score': 0.10
    })
    
    # Pattern thresholds - TESTED AND OPTIMIZED
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'category_leader': 90,
        'hidden_gem': 85,
        'acceleration': 85,
        'institutional': 80,
        'breakout_ready': 80,
        'market_leader': 95,
        'momentum_wave': 80,
        'liquid_leader': 90,
        'long_strength': 80,
        'trend_quality': 80,
        'value_momentum': 15,
        'earnings_rocket': 50,
        'quality_value': 20,
        'turnaround': 100
    })
    
    # Tier definitions - FLEXIBLE SYSTEM
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        'eps': {
            'Negative': (-float('inf'), 0),
            'Low (0-20%)': (0, 20),
            'Medium (20-50%)': (20, 50),
            'High (50-100%)': (50, 100),
            'Extreme (>100%)': (100, float('inf'))
        },
        'pe': {
            'Negative/Zero': (-float('inf'), 0),
            'Low (0-15)': (0, 15),
            'Moderate (15-25)': (15, 25),
            'High (25-50)': (25, 50),
            'Very High (>50)': (50, float('inf'))
        },
        'price': {
            'Penny (<10)': (0, 10),
            'Low (10-100)': (10, 100),
            'Mid (100-500)': (100, 500),
            'High (500-2000)': (500, 2000),
            'Premium (>2000)': (2000, float('inf'))
        }
    })
    
    # Columns that contain percentage values
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_3d', 'ret_7d',
        'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ])
    
    # Volume ratio columns
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    # Reasonable bounds for data validation
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1000000),
        'volume': (0, 1e12),
        'rvol': (0, 1000),
        'pe': (-1000, 1000),
        'returns': (-99.99, 10000)
    })
    
    # Cache settings
    CACHE_DURATION_HOURS: int = 1
    
    # Display settings
    ROWS_PER_PAGE: int = 50
    MAX_EXPORT_ROWS: int = 2000

# Create global configuration instance
CONFIG = Config()

# ============================================
# LOGGING - Simple and Effective
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# SESSION STATE MANAGEMENT
# ============================================
class SessionStateManager:
    """Manage Streamlit session state professionally"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with defaults"""
        defaults = {
            # Data state
            'ranked_df': pd.DataFrame(),
            'last_update': None,
            'last_good_data': None,
            'cache_key': None,
            
            # User inputs
            'user_spreadsheet_id': '',  # Persistent spreadsheet ID
            'data_source': 'sheet',
            'search_query': '',
            'current_page': 0,
            
            # Filters
            'filters': {},
            'active_filter_count': 0,
            'quick_filter': None,
            'quick_filter_applied': False,
            'trigger_clear': False,
            
            # Display settings
            'display_mode': 'Technical',
            'show_debug': False,
            
            # Filter-specific states
            'category_filter': [],
            'sector_filter': [],
            'industry_filter': [],
            'patterns': [],
            'min_score': 0,
            'trend_filter': 'All Trends',
            'trend_range': (0, 100),
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
        
        # Initialize only if not exists
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        """Clear all filter-related session state"""
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter', 'patterns',
            'min_score', 'trend_filter', 'trend_range', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'min_eps_change',
            'min_pe', 'max_pe', 'require_fundamental_data',
            'wave_states_filter', 'wave_strength_range_slider',
            'quick_filter', 'quick_filter_applied'
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], tuple):
                    if key == 'wave_strength_range_slider':
                        st.session_state[key] = (0, 100)
                    else:
                        st.session_state[key] = (0, 100)
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = 'All Trends'
                    else:
                        st.session_state[key] = ''
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], (int, float)):
                    st.session_state[key] = 0 if key == 'min_score' else None
                else:
                    st.session_state[key] = None
        
        # Reset filter tracking
        st.session_state.filters = {}
        st.session_state.active_filter_count = 0
        st.session_state.trigger_clear = False

# ============================================
# DATA VALIDATION
# ============================================
class DataValidator:
    """Validate and clean data professionally"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], 
                          context: str) -> Tuple[bool, str]:
        """Validate dataframe structure and content"""
        if df is None:
            return False, f"{context}: DataFrame is None"
        
        if df.empty:
            return False, f"{context}: DataFrame is empty"
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"{context}: Missing columns: {missing_cols}"
        
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False,
                           bounds: Optional[Tuple[float, float]] = None) -> float:
        """Clean and validate numeric values"""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        # Handle string inputs
        if isinstance(value, str):
            value = value.strip().replace(',', '').replace('%', '')
            if value in ['', '-', 'N/A', 'nan', 'NaN']:
                return np.nan
        
        try:
            num_val = float(value)
            
            # Handle infinity
            if np.isinf(num_val):
                return np.nan
            
            # Apply bounds if specified
            if bounds:
                num_val = np.clip(num_val, bounds[0], bounds[1])
            
            return num_val
        except (ValueError, TypeError):
            return np.nan
    
    @staticmethod
    def sanitize_string(value: Any) -> str:
        """Clean string values"""
        if pd.isna(value) or value is None:
            return "Unknown"
        
        str_val = str(value).strip()
        return str_val if str_val else "Unknown"

# ============================================
# DATA LOADING WITH RETRY LOGIC
# ============================================
def load_data_with_retry(source: str, spreadsheet_id: str = None, 
                        uploaded_file: Any = None, max_retries: int = 3) -> pd.DataFrame:
    """Load data with retry logic and clear error messages"""
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            if source == "upload" and uploaded_file is not None:
                # Load from uploaded file
                df = pd.read_csv(uploaded_file, low_memory=False)
                logger.info(f"Loaded {len(df)} rows from uploaded file")
                return df
            
            elif source == "sheet" and spreadsheet_id:
                # Construct Google Sheets CSV URL
                csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={CONFIG.DEFAULT_GID}"
                
                # Attempt to load
                df = pd.read_csv(csv_url, low_memory=False)
                logger.info(f"Loaded {len(df)} rows from Google Sheets")
                return df
            
            else:
                raise ValueError("Invalid data source configuration")
        
        except Exception as e:
            attempt += 1
            last_error = str(e)
            
            # Determine error type and message
            if "404" in last_error or "Not Found" in last_error:
                error_msg = "Spreadsheet not found. Please check the ID and ensure it's publicly accessible."
            elif "timeout" in last_error.lower():
                error_msg = "Connection timeout. Please check your internet connection."
            elif "403" in last_error or "Forbidden" in last_error:
                error_msg = "Access denied. Please make sure the sheet is set to 'Anyone with link can view'."
            else:
                error_msg = f"Error loading data: {last_error}"
            
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed: {error_msg}. Retrying...")
                time.sleep(1)  # Brief pause before retry
            else:
                logger.error(f"All attempts failed: {error_msg}")
                raise Exception(error_msg)
    
    raise Exception(f"Failed to load data after {max_retries} attempts")

# ============================================
# DATA PROCESSING ENGINE
# ============================================
class DataProcessor:
    """Process raw data into analysis-ready format"""
    
    @staticmethod
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline"""
        df = df.copy()
        initial_count = len(df)
        
        # Process numeric columns
        numeric_cols = [col for col in df.columns 
                       if col not in ['ticker', 'company_name', 'category', 
                                     'sector', 'industry', 'year', 'market_cap']]
        
        for col in numeric_cols:
            if col in df.columns:
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
                
                # Clean values
                df[col] = df[col].apply(
                    lambda x: DataValidator.clean_numeric_value(x, is_pct, bounds)
                )
        
        # Process categorical columns
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
        # Fix volume ratios
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 1000.0)
                df[col] = df[col].fillna(1.0)
        
        # Validate critical data
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        # Fill missing values
        df = DataProcessor._fill_missing_values(df)
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        # Money flow calculation (millions)
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow_mm'] = (df['price'] * df['volume_1d'] * df['rvol']) / 1_000_000
        else:
            df['money_flow_mm'] = 0
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows")
        return df
    
    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults"""
        # Position data defaults
        df['from_low_pct'] = df.get('from_low_pct', 50).fillna(50)
        df['from_high_pct'] = df.get('from_high_pct', -50).fillna(-50)
        
        # RVOL default
        df['rvol'] = df.get('rvol', 1.0).fillna(1.0)
        
        # Return defaults
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
        """Add flexible tier classifications"""
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Classify value into tier"""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
            
            return "Unknown"
        
        # EPS tier
        if 'eps_change_pct' in df.columns:
            df['eps_tier'] = df['eps_change_pct'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['eps'])
            )
        
        # PE tier
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['pe'])
            )
        
        # Price tier
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['price'])
            )
        
        return df

# ============================================
# RANKING ENGINE - Core Scoring Logic
# ============================================
class RankingEngine:
    """Calculate all scores and rankings"""
    
    @staticmethod
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores and master score"""
        if df.empty:
            return df
        
        # Calculate individual component scores
        df['position_score'] = RankingEngine._calculate_position_score(df)
        df['volume_score'] = RankingEngine._calculate_volume_score(df)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df)
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df)
        
        # Calculate master score
        df['master_score'] = (
            df['position_score'] * CONFIG.SCORING_WEIGHTS['position_score'] +
            df['volume_score'] * CONFIG.SCORING_WEIGHTS['volume_score'] +
            df['momentum_score'] * CONFIG.SCORING_WEIGHTS['momentum_score'] +
            df['acceleration_score'] * CONFIG.SCORING_WEIGHTS['acceleration_score'] +
            df['breakout_score'] * CONFIG.SCORING_WEIGHTS['breakout_score'] +
            df['rvol_score'] * CONFIG.SCORING_WEIGHTS['rvol_score']
        ).clip(0, 100)
        
        # Calculate ranks
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        
        # Category ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        return df
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score with non-linear transformation"""
        if 'from_low_pct' not in df.columns or 'from_high_pct' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # Non-linear transformation for better extremes
        from_low = df['from_low_pct'].fillna(50)
        from_low_transformed = np.tanh(from_low / 100) * 100
        
        from_high = df['from_high_pct'].fillna(-50)
        from_high_score = (100 + from_high).clip(0, 100)
        
        # Weighted combination
        position_score = (from_low_transformed * 0.6 + from_high_score * 0.4).clip(0, 100)
        
        return position_score
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate VMI (Volume Momentum Index) with enhancements"""
        vmi_score = pd.Series(50, index=df.index, dtype=float)
        
        # Volume ratio contributions
        volume_ratios = {
            'vol_ratio_1d_90d': 0.4,
            'vol_ratio_7d_90d': 0.3,
            'vol_ratio_30d_90d': 0.2,
            'vol_ratio_90d_180d': 0.1
        }
        
        for col, weight in volume_ratios.items():
            if col in df.columns:
                ratio = df[col].fillna(1.0)
                # Non-linear response for volume spikes
                contribution = np.tanh((ratio - 1) * 0.5) * 50 + 50
                vmi_score += (contribution - 50) * weight
        
        return vmi_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate 30-day momentum score"""
        if 'ret_30d' not in df.columns:
            return pd.Series(50, index=df.index)
        
        ret_30d = df['ret_30d'].fillna(0)
        
        # Scale returns to score
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        momentum_score[ret_30d > 50] = 90
        momentum_score[(ret_30d > 30) & (ret_30d <= 50)] = 80
        momentum_score[(ret_30d > 15) & (ret_30d <= 30)] = 70
        momentum_score[(ret_30d > 5) & (ret_30d <= 15)] = 60
        momentum_score[(ret_30d > -5) & (ret_30d <= 5)] = 50
        momentum_score[(ret_30d > -15) & (ret_30d <= -5)] = 40
        momentum_score[ret_30d <= -15] = 30
        
        return momentum_score.clip(0, 100)
    
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum acceleration"""
        accel_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
            weekly_pace = df['ret_7d'] / 7
            monthly_pace = df['ret_30d'] / 30
            
            acceleration = weekly_pace - monthly_pace
            
            # Convert to score
            accel_score[acceleration > 3] = 90
            accel_score[(acceleration > 2) & (acceleration <= 3)] = 80
            accel_score[(acceleration > 1) & (acceleration <= 2)] = 70
            accel_score[(acceleration > 0) & (acceleration <= 1)] = 60
            accel_score[(acceleration > -1) & (acceleration <= 0)] = 50
            accel_score[acceleration <= -1] = 40
        
        return accel_score.clip(0, 100)
    
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout readiness"""
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'from_high_pct' in df.columns:
            distance_from_high = abs(df['from_high_pct'])
            
            breakout_score[distance_from_high < 3] = 95
            breakout_score[(distance_from_high >= 3) & (distance_from_high < 5)] = 85
            breakout_score[(distance_from_high >= 5) & (distance_from_high < 10)] = 75
            breakout_score[(distance_from_high >= 10) & (distance_from_high < 20)] = 60
            breakout_score[distance_from_high >= 20] = 40
        
        if 'volume_score' in df.columns:
            breakout_score = (breakout_score * 0.7 + df['volume_score'] * 0.3).clip(0, 100)
        
        return breakout_score
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate relative volume score"""
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        rvol = df['rvol'].fillna(1.0)
        
        rvol_score = pd.Series(50, index=df.index, dtype=float)
        rvol_score[rvol > 5] = 100
        rvol_score[(rvol > 3) & (rvol <= 5)] = 90
        rvol_score[(rvol > 2) & (rvol <= 3)] = 80
        rvol_score[(rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score[(rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score[rvol <= 0.8] = 40
        
        return rvol_score.clip(0, 100)
    
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality based on SMA alignment"""
        trend_quality = pd.Series(50, index=df.index, dtype=float)
        
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
            price = df['price']
            sma_20 = df['sma_20d']
            sma_50 = df['sma_50d']
            sma_200 = df['sma_200d']
            
            # Perfect uptrend
            perfect_uptrend = (price > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
            trend_quality[perfect_uptrend] = 100
            
            # Strong uptrend
            strong_uptrend = (price > sma_20) & (sma_20 > sma_50)
            trend_quality[strong_uptrend & ~perfect_uptrend] = 80
            
            # Moderate uptrend
            moderate_uptrend = price > sma_50
            trend_quality[moderate_uptrend & ~strong_uptrend & ~perfect_uptrend] = 60
            
            # Downtrend
            downtrend = (price < sma_20) & (sma_20 < sma_50)
            trend_quality[downtrend] = 20
        
        return trend_quality.clip(0, 100)
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score"""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        long_term_returns = []
        if 'ret_6m' in df.columns:
            long_term_returns.append(df['ret_6m'])
        if 'ret_1y' in df.columns:
            long_term_returns.append(df['ret_1y'])
        
        if long_term_returns:
            avg_return = pd.concat(long_term_returns, axis=1).mean(axis=1)
            
            strength_score[avg_return > 100] = 90
            strength_score[(avg_return > 50) & (avg_return <= 100)] = 80
            strength_score[(avg_return > 20) & (avg_return <= 50)] = 70
            strength_score[(avg_return > 0) & (avg_return <= 20)] = 60
            strength_score[(avg_return > -20) & (avg_return <= 0)] = 40
            strength_score[avg_return <= -20] = 20
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score"""
        if 'volume_30d' in df.columns and 'price' in df.columns:
            dollar_volume = df['volume_30d'] * df['price']
            
            # Rank-based scoring
            liquidity_score = dollar_volume.rank(pct=True, ascending=True) * 100
            return liquidity_score.fillna(50).clip(0, 100)
        
        return pd.Series(50, index=df.index)
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ranks within categories"""
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        for category in df['category'].unique():
            if category != 'Unknown':
                mask = df['category'] == category
                cat_df = df[mask]
                
                if len(cat_df) > 0:
                    cat_ranks = cat_df['master_score'].rank(
                        method='first', ascending=False, na_option='bottom'
                    )
                    df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                    
                    cat_percentiles = cat_df['master_score'].rank(
                        pct=True, ascending=True, na_option='bottom'
                    ) * 100
                    df.loc[mask, 'category_percentile'] = cat_percentiles
        
        return df

# ============================================
# PATTERN DETECTION ENGINE
# ============================================
class PatternDetector:
    """Detect all 25 trading patterns"""
    
    @staticmethod
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns using vectorized operations"""
        if df.empty:
            df['patterns'] = ''
            return df
        
        patterns = []
        
        # Technical Patterns (11)
        patterns.extend(PatternDetector._detect_technical_patterns(df))
        
        # Fundamental Patterns (5)
        patterns.extend(PatternDetector._detect_fundamental_patterns(df))
        
        # Range Patterns (6)
        patterns.extend(PatternDetector._detect_range_patterns(df))
        
        # Intelligence Patterns (3)
        patterns.extend(PatternDetector._detect_intelligence_patterns(df))
        
        # Combine all patterns
        if patterns:
            pattern_matrix = np.zeros((len(df), len(patterns)), dtype=bool)
            pattern_names = []
            
            for i, (name, mask) in enumerate(patterns):
                if mask is not None and isinstance(mask, pd.Series):
                    pattern_matrix[:, i] = mask.values
                    pattern_names.append(name)
            
            # Create pattern strings
            df['patterns'] = [
                ' | '.join([pattern_names[j] for j in range(len(pattern_names)) 
                           if pattern_matrix[i, j]])
                for i in range(len(df))
            ]
        else:
            df['patterns'] = ''
        
        return df
    
    @staticmethod
    def _detect_technical_patterns(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        """Detect technical patterns"""
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
            patterns.append(('👑 MKT LEADER', mask))
        
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
            mask = df['trend_quality'] >= CONFIG.PATTERN_THRESHOLDS['trend_quality']
            patterns.append(('📈 QUALITY TREND', mask))
        
        return patterns
    
    @staticmethod
    def _detect_fundamental_patterns(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        """Detect fundamental patterns"""
        patterns = []
        
        # Skip if no fundamental data
        if 'pe' not in df.columns or 'eps_change_pct' not in df.columns:
            return patterns
        
        # 12. Value Momentum
        if all(col in df.columns for col in ['pe', 'momentum_score']):
            mask = (
                (df['pe'] > 0) & 
                (df['pe'] < CONFIG.PATTERN_THRESHOLDS['value_momentum']) & 
                (df['momentum_score'] >= 70)
            )
            patterns.append(('💎 VALUE MOMENTUM', mask))
        
        # 13. Earnings Rocket
        if 'eps_change_pct' in df.columns:
            mask = df['eps_change_pct'] >= CONFIG.PATTERN_THRESHOLDS['earnings_rocket']
            patterns.append(('📊 EARNINGS ROCKET', mask))
        
        # 14. Quality Leader
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'liquidity_score']):
            mask = (
                (df['pe'] > 0) & 
                (df['pe'] < CONFIG.PATTERN_THRESHOLDS['quality_value']) & 
                (df['eps_change_pct'] > 10) & 
                (df['liquidity_score'] >= 70)
            )
            patterns.append(('🏆 QUALITY LEADER', mask))
        
        # 15. Turnaround
        if 'eps_change_pct' in df.columns and 'momentum_score' in df.columns:
            mask = (
                (df['eps_change_pct'] >= CONFIG.PATTERN_THRESHOLDS['turnaround']) & 
                (df['momentum_score'] >= 60)
            )
            patterns.append(('⚡ TURNAROUND', mask))
        
        # 16. High PE Warning
        if 'pe' in df.columns:
            mask = (df['pe'] > 0) & (df['pe'] > 100)
            patterns.append(('⚠️ HIGH PE', mask))
        
        return patterns
    
    @staticmethod
    def _detect_range_patterns(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        """Detect price range patterns"""
        patterns = []
        
        # 17. 52W High Approach
        if 'from_high_pct' in df.columns and 'volume_score' in df.columns:
            mask = (
                (df['from_high_pct'] > -5) & 
                (df['volume_score'] >= 70) & 
                (df.get('momentum_score', 50) >= 60)
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
            daily_7d_pace = df['ret_7d'] / 7
            daily_30d_pace = df['ret_30d'] / 30
            
            mask = (
                (daily_7d_pace > daily_30d_pace * 1.5) & 
                (df['acceleration_score'] >= 85) & 
                (df['rvol'] > 2)
            )
            patterns.append(('🔀 MOMENTUM DIVERGE', mask))
        
        # 22. Range Compression
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            range_pct = ((df['high_52w'] - df['low_52w']) / df['low_52w'].clip(lower=0.01)) * 100
            mask = (range_pct < 50) & (df['from_low_pct'] > 30)
            patterns.append(('🎯 RANGE COMPRESS', mask))
        
        return patterns
    
    @staticmethod
    def _detect_intelligence_patterns(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        """Detect advanced intelligence patterns"""
        patterns = []
        
        # 23. Stealth Accumulation
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d', 
                                             'ret_30d', 'momentum_score']):
            mask = (
                (df['vol_ratio_7d_90d'] > 1.3) & 
                (df['vol_ratio_30d_90d'] > 1.2) & 
                (df['ret_30d'].between(-5, 5)) & 
                (df['momentum_score'] < 60)
            )
            patterns.append(('🤫 STEALTH', mask))
        
        # 24. Vampire Pattern (small cap surge)
        if all(col in df.columns for col in ['category', 'ret_7d', 'ret_30d', 'rvol']):
            daily_7d = df['ret_7d'] / 7
            daily_30d = df['ret_30d'] / 30
            
            mask = (
                (df['category'].isin(['Small Cap', 'Micro Cap'])) & 
                (daily_7d > daily_30d * 2) & 
                (df['rvol'] > 3)
            )
            patterns.append(('🧛 VAMPIRE', mask))
        
        # 25. Perfect Storm
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            mask = (
                (df['momentum_harmony'] == 4) & 
                (df['master_score'] > 80)
            )
            patterns.append(('⛈️ PERFECT STORM', mask))
        
        return patterns

# ============================================
# ADVANCED METRICS
# ============================================
class AdvancedMetrics:
    """Calculate advanced metrics and indicators"""
    
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics"""
        if df.empty:
            return df
        
        # Position Tension
        if 'from_low_pct' in df.columns and 'from_high_pct' in df.columns:
            low_tension = np.exp(-df['from_low_pct'] / 20) * 100
            high_tension = (1 - np.exp(df['from_high_pct'] / 20)) * 100
            df['position_tension'] = (low_tension + high_tension) / 2
        else:
            df['position_tension'] = 50
        
        # Momentum Harmony
        df['momentum_harmony'] = 0
        
        if 'ret_1d' in df.columns:
            df['momentum_harmony'] += (df['ret_1d'] > 0).astype(int)
        
        if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
            daily_ret_7d = df['ret_7d'] / 7
            daily_ret_30d = df['ret_30d'] / 30
            df['momentum_harmony'] += (daily_ret_7d > daily_ret_30d).astype(int)
        
        if 'ret_30d' in df.columns and 'ret_3m' in df.columns:
            daily_ret_30d = df['ret_30d'] / 30
            daily_ret_3m = df['ret_3m'] / 90
            df['momentum_harmony'] += (daily_ret_30d > daily_ret_3m).astype(int)
        
        if 'ret_3m' in df.columns:
            df['momentum_harmony'] += (df['ret_3m'] > 0).astype(int)
        
        # Wave State
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)
        
        # Wave Strength (simple signal counting)
        df['wave_strength'] = df.apply(AdvancedMetrics._calculate_wave_strength, axis=1)
        
        # Smart Money Flow
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d', 
                                             'ret_30d', 'volume_score']):
            vol_persist = ((df['vol_ratio_7d_90d'] > 1.2) & 
                          (df['vol_ratio_30d_90d'] > 1.1)).astype(int) * 20
            divergence = np.where((df['ret_30d'].abs() < 5) & (df['volume_score'] > 70), 20, 0)
            df['smart_money_flow'] = 50 + vol_persist + divergence
        else:
            df['smart_money_flow'] = 50
        
        # Market Regime (simplified)
        breadth = (df['ret_30d'] > 0).mean()
        if breadth > 0.6 and (df['ret_30d'] > 10).mean() > 0.3:
            df['market_regime'] = "RISK-ON 🔥"
        elif breadth < 0.4:
            df['market_regime'] = "RISK-OFF ❄️"
        else:
            df['market_regime'] = "NEUTRAL ⚖️"
        
        return df
    
    @staticmethod
    def _get_wave_state(row: pd.Series) -> str:
        """Determine wave state based on signals"""
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
    
    @staticmethod
    def _calculate_wave_strength(row: pd.Series) -> float:
        """Calculate wave strength (0-100)"""
        signals = 0
        
        if row.get('momentum_score', 0) > 70:
            signals += 1
        if row.get('volume_score', 0) > 70:
            signals += 1
        if row.get('acceleration_score', 0) > 70:
            signals += 1
        if row.get('rvol', 0) > 2:
            signals += 1
        
        return (signals / 4) * 100

# ============================================
# FILTER ENGINE
# ============================================
class FilterEngine:
    """Handle all filtering operations efficiently"""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with optimized performance"""
        if df.empty:
            return df
        
        # Start with boolean mask
        mask = pd.Series(True, index=df.index)
        
        # Category filter
        categories = filters.get('categories', [])
        if categories and 'All' not in categories:
            mask &= df['category'].isin(categories)
        
        # Sector filter
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors:
            mask &= df['sector'].isin(sectors)
        
        # Industry filter
        industries = filters.get('industries', [])
        if industries and 'All' not in industries and 'industry' in df.columns:
            mask &= df['industry'].isin(industries)
        
        # Score filter
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            mask &= df['master_score'] >= min_score
        
        # Pattern filter
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            pattern_regex = '|'.join([re.escape(p) for p in patterns])
            mask &= df['patterns'].str.contains(pattern_regex, case=False, na=False, regex=True)
        
        # EPS change filter
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            mask &= (df['eps_change_pct'] >= min_eps_change) | df['eps_change_pct'].isna()
        
        # PE filters
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in df.columns:
            mask &= df['pe'].isna() | ((df['pe'] > 0) & (df['pe'] >= min_pe))
        
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in df.columns:
            mask &= df['pe'].isna() | ((df['pe'] > 0) & (df['pe'] <= max_pe))
        
        # Tier filters
        for tier_type, col_name in [
            ('eps_tiers', 'eps_tier'),
            ('pe_tiers', 'pe_tier'),
            ('price_tiers', 'price_tier')
        ]:
            tier_values = filters.get(tier_type, [])
            if tier_values and 'All' not in tier_values and col_name in df.columns:
                mask &= df[col_name].isin(tier_values)
        
        # Data completeness filter
        if filters.get('require_fundamental_data', False):
            if 'pe' in df.columns and 'eps_change_pct' in df.columns:
                mask &= df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna()
        
        # Wave state filter
        wave_states = filters.get('wave_states', [])
        if wave_states and 'All' not in wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)
        
        # Wave strength filter
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and 'wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            mask &= (df['wave_strength'] >= min_ws) & (df['wave_strength'] <= max_ws)
        
        # Trend filter
        if filters.get('trend_range') and filters.get('trend_filter') != 'All Trends':
            min_trend, max_trend = filters['trend_range']
            if 'trend_quality' in df.columns:
                mask &= (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend)
        
        # Apply mask
        filtered_df = df[mask].copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, 
                          current_filters: Dict[str, Any]) -> List[str]:
        """Get available options for a filter based on other active filters"""
        if df.empty or column not in df.columns:
            return []
        
        # Apply other filters first (interconnected filtering)
        temp_filters = current_filters.copy()
        
        # Remove the current column's filter to avoid circular dependency
        filter_key_map = {
            'category': 'categories',
            'sector': 'sectors',
            'industry': 'industries',
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
# SEARCH ENGINE
# ============================================
class SearchEngine:
    """Handle stock search functionality"""
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks by ticker or company name"""
        if not query or df.empty:
            return pd.DataFrame()
        
        query = query.upper().strip()
        
        # Direct ticker match
        ticker_exact = df[df['ticker'].str.upper() == query]
        if not ticker_exact.empty:
            return ticker_exact
        
        # Ticker contains
        ticker_contains = df[df['ticker'].str.upper().str.contains(query, na=False, regex=False)]
        
        # Company name contains
        company_contains = df[df['company_name'].str.upper().str.contains(query, na=False, regex=False)]
        
        # Combine results (remove duplicates)
        results = pd.concat([ticker_contains, company_contains]).drop_duplicates()
        
        return results

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
    def render_stock_details(stock_data: pd.Series, display_mode: str):
        """Render detailed stock information"""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown(f"### {stock_data['ticker']} - {stock_data['company_name']}")
            st.markdown(f"**Category:** {stock_data.get('category', 'N/A')} | "
                       f"**Sector:** {stock_data.get('sector', 'N/A')}")
            
            if stock_data.get('patterns'):
                st.markdown(f"**Patterns:** {stock_data['patterns']}")
        
        with col2:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                UIComponents.render_metric_card(
                    "Master Score",
                    f"{stock_data.get('master_score', 0):.1f}",
                    f"Rank #{stock_data.get('rank', 'N/A')}"
                )
            
            with metrics_col2:
                UIComponents.render_metric_card(
                    "Price",
                    f"₹{stock_data.get('price', 0):.2f}",
                    f"{stock_data.get('ret_1d', 0):+.2f}%"
                )
            
            with metrics_col3:
                UIComponents.render_metric_card(
                    "RVOL",
                    f"{stock_data.get('rvol', 0):.2f}x"
                )
        
        with col3:
            wave_state = stock_data.get('wave_state', 'Unknown')
            wave_strength = stock_data.get('wave_strength', 0)
            
            st.markdown(f"**Wave State**")
            st.markdown(f"# {wave_state}")
            st.progress(wave_strength / 100)
            st.caption(f"Strength: {wave_strength:.0f}%")
    
    @staticmethod
    def render_summary_section(df: pd.DataFrame):
        """Render the summary dashboard section"""
        # Top movers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Top Gainers Today")
            if 'ret_1d' in df.columns:
                top_gainers = df.nlargest(5, 'ret_1d')[['ticker', 'company_name', 'ret_1d', 'master_score']]
                for _, stock in top_gainers.iterrows():
                    st.markdown(f"**{stock['ticker']}** - {stock['company_name'][:30]}... "
                              f"`+{stock['ret_1d']:.2f}%` (Score: {stock['master_score']:.1f})")
        
        with col2:
            st.markdown("#### 🔥 High Volume Activity")
            if 'rvol' in df.columns:
                high_vol = df.nlargest(5, 'rvol')[['ticker', 'company_name', 'rvol', 'master_score']]
                for _, stock in high_vol.iterrows():
                    st.markdown(f"**{stock['ticker']}** - {stock['company_name'][:30]}... "
                              f"`{stock['rvol']:.1f}x` (Score: {stock['master_score']:.1f})")
        
        # Wave distribution
        st.markdown("---")
        st.markdown("#### 🌊 Wave State Distribution")
        
        if 'wave_state' in df.columns:
            wave_counts = df['wave_state'].value_counts()
            
            # Create wave state chart
            fig = px.bar(
                x=wave_counts.index,
                y=wave_counts.values,
                labels={'x': 'Wave State', 'y': 'Count'},
                color=wave_counts.index,
                color_discrete_map={
                    '🌊🌊🌊 CRESTING': '#00cc00',
                    '🌊🌊 BUILDING': '#66b3ff',
                    '🌊 FORMING': '#99ccff',
                    '💥 BREAKING': '#ff6666'
                }
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# EXPORT FUNCTIONS
# ============================================
def prepare_export_data(df: pd.DataFrame, export_type: str) -> pd.DataFrame:
    """Prepare data for export based on type"""
    if df.empty:
        return df
    
    # Base columns for all exports
    base_cols = [
        'rank', 'ticker', 'company_name', 'category', 'sector', 'industry',
        'master_score', 'price', 'ret_1d', 'ret_30d', 'rvol', 'patterns', 
        'wave_state', 'wave_strength', 'money_flow_mm'
    ]
    
    if export_type == "Detailed Analysis":
        # Include all scores and metrics
        score_cols = [
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score',
            'trend_quality', 'long_term_strength', 'liquidity_score'
        ]
        
        metric_cols = [
            'from_low_pct', 'from_high_pct', 'position_tension',
            'momentum_harmony', 'smart_money_flow'
        ]
        
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        volume_cols = [col for col in df.columns if col.startswith('vol_')]
        
        export_cols = base_cols + score_cols + metric_cols + return_cols + volume_cols
    
    elif export_type == "Trading Signals":
        # Focus on actionable data
        signal_cols = [
            'momentum_score', 'acceleration_score', 'breakout_score',
            'from_high_pct', 'ret_7d', 'ret_30d', 'vol_ratio_7d_90d'
        ]
        
        export_cols = base_cols + signal_cols
    
    else:  # Quick Summary
        export_cols = base_cols
    
    # Select available columns
    available_cols = [col for col in export_cols if col in df.columns]
    
    return df[available_cols].copy()

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
    
    # Apply custom CSS
    st.markdown("""
    <style>
    /* Professional styling */
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
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🌊 Wave Detection Ultimate 3.0")
    st.markdown("*Professional Stock Screening & Ranking System*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Google Sheets", "Upload CSV"],
            index=0 if st.session_state.data_source == "sheet" else 1,
            key="data_source_radio"
        )
        
        st.session_state.data_source = "sheet" if data_source == "Google Sheets" else "upload"
        
        uploaded_file = None
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        else:
            st.markdown("### 📊 Spreadsheet Configuration")
            
            # Persistent spreadsheet ID input
            current_id = st.session_state.get('user_spreadsheet_id', '')
            
            # Spreadsheet ID input with validation
            new_id = st.text_input(
                "Spreadsheet ID",
                value=current_id,
                placeholder="Enter your Google Sheets ID",
                help="The ID is the long string in your Google Sheets URL between /d/ and /edit"
            )
            
            # Validate and update only if changed and valid
            if new_id != current_id:
                # Clean the input
                cleaned_id = new_id.strip()
                
                # Validate format (44 chars, alphanumeric with - and _)
                if re.match(r'^[a-zA-Z0-9_-]{44}$', cleaned_id):
                    st.session_state.user_spreadsheet_id = cleaned_id
                    st.success("✅ Valid Spreadsheet ID")
                    st.rerun()
                elif len(cleaned_id) > 0:
                    st.error("❌ Invalid ID format. Must be 44 characters (letters, numbers, -, _)")
        
        # Display mode
        st.markdown("### 📊 Display Settings")
        st.session_state.display_mode = st.radio(
            "Display Mode",
            ["Technical", "Hybrid (with Fundamentals)"],
            index=0 if st.session_state.display_mode == "Technical" else 1
        )
        
        # Filter count
        active_filter_count = 0
        filter_checks = [
            ('category_filter', lambda x: x and len(x) > 0),
            ('sector_filter', lambda x: x and len(x) > 0),
            ('industry_filter', lambda x: x and len(x) > 0),
            ('min_score', lambda x: x > 0),
            ('patterns', lambda x: x and len(x) > 0),
            ('trend_filter', lambda x: x != 'All Trends'),
            ('eps_tier_filter', lambda x: x and len(x) > 0),
            ('pe_tier_filter', lambda x: x and len(x) > 0),
            ('price_tier_filter', lambda x: x and len(x) > 0),
            ('min_eps_change', lambda x: x is not None and str(x).strip() != ''),
            ('min_pe', lambda x: x is not None and str(x).strip() != ''),
            ('max_pe', lambda x: x is not None and str(x).strip() != ''),
            ('require_fundamental_data', lambda x: x),
            ('wave_states_filter', lambda x: x and len(x) > 0),
            ('wave_strength_range_slider', lambda x: x != (0, 100))
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
    
    # Data loading
    try:
        # Check if we need to load data
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        # For Google Sheets, check if ID is provided
        if st.session_state.data_source == "sheet" and not st.session_state.get('user_spreadsheet_id'):
            st.info("👋 Welcome! Please enter your Google Spreadsheet ID in the sidebar to begin.")
            st.markdown("""
            ### 🚀 Getting Started
            1. Open your Google Sheet
            2. Copy the ID from the URL (the long string between `/d/` and `/edit`)
            3. Paste it in the sidebar
            4. Make sure your sheet is set to "Anyone with link can view"
            """)
            st.stop()
        
        # Check cache
        cache_key = f"{st.session_state.data_source}_{st.session_state.get('user_spreadsheet_id', '')}"
        cache_valid = False
        
        if ('last_update' in st.session_state and 
            st.session_state.get('cache_key') == cache_key and
            st.session_state.last_update):
            time_diff = datetime.now(timezone.utc) - st.session_state.last_update
            cache_valid = time_diff < timedelta(hours=CONFIG.CACHE_DURATION_HOURS)
        
        # Load data if needed
        if not cache_valid or st.session_state.ranked_df.empty:
            with st.spinner("🔄 Loading and processing data..."):
                # Load data with retry
                df = load_data_with_retry(
                    st.session_state.data_source,
                    st.session_state.get('user_spreadsheet_id'),
                    uploaded_file
                )
                
                # Validate
                is_valid, msg = DataValidator.validate_dataframe(
                    df, CONFIG.CRITICAL_COLUMNS, "Initial load"
                )
                if not is_valid:
                    st.error(f"❌ Data validation failed: {msg}")
                    st.stop()
                
                # Process data
                df = DataProcessor.process_dataframe(df)
                df = RankingEngine.calculate_all_scores(df)
                df = PatternDetector.detect_all_patterns(df)
                df = AdvancedMetrics.calculate_all_metrics(df)
                
                # Store in session state
                st.session_state.ranked_df = df
                st.session_state.last_update = datetime.now(timezone.utc)
                st.session_state.cache_key = cache_key
                
                logger.info(f"Successfully loaded {len(df)} stocks")
        
        # Get data from session state
        ranked_df = st.session_state.ranked_df.copy()
        
        if ranked_df.empty:
            st.error("❌ No data available")
            st.stop()
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        if st.session_state.get('show_debug'):
            st.exception(e)
        st.stop()
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🔍 Filters")
        
        # Build filter dictionary
        filters = {}
        
        # Category filter (interconnected)
        categories = FilterEngine.get_filter_options(ranked_df, 'category', filters)
        selected_categories = st.multiselect(
            "Category",
            options=categories,
            default=st.session_state.get('category_filter', []),
            key="category_filter"
        )
        filters['categories'] = selected_categories
        
        # Sector filter (interconnected)
        sectors = FilterEngine.get_filter_options(ranked_df, 'sector', filters)
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=st.session_state.get('sector_filter', []),
            key="sector_filter"
        )
        filters['sectors'] = selected_sectors
        
        # Industry filter (interconnected)
        if 'industry' in ranked_df.columns:
            industries = FilterEngine.get_filter_options(ranked_df, 'industry', filters)
            selected_industries = st.multiselect(
                "Industry",
                options=industries,
                default=st.session_state.get('industry_filter', []),
                key="industry_filter"
            )
            filters['industries'] = selected_industries
        
        # Score filter
        min_score = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=st.session_state.get('min_score', 0),
            key="min_score"
        )
        filters['min_score'] = min_score
        
        # Pattern filter
        if 'patterns' in ranked_df.columns:
            all_patterns = set()
            for patterns in ranked_df['patterns']:
                if patterns:
                    pattern_list = [p.strip() for p in patterns.split('|')]
                    all_patterns.update(pattern_list)
            
            if all_patterns:
                selected_patterns = st.multiselect(
                    "Patterns",
                    options=sorted(all_patterns),
                    default=st.session_state.get('patterns', []),
                    key="patterns"
                )
                filters['patterns'] = selected_patterns
        
        # Wave filters
        with st.expander("🌊 Wave Filters", expanded=False):
            # Wave states
            if 'wave_state' in ranked_df.columns:
                wave_states = ranked_df['wave_state'].unique().tolist()
                selected_wave_states = st.multiselect(
                    "Wave States",
                    options=wave_states,
                    default=st.session_state.get('wave_states_filter', []),
                    key="wave_states_filter"
                )
                filters['wave_states'] = selected_wave_states
            
            # Wave strength
            if 'wave_strength' in ranked_df.columns:
                wave_strength_range = st.slider(
                    "Wave Strength Range",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.get('wave_strength_range_slider', (0, 100)),
                    key="wave_strength_range_slider"
                )
                filters['wave_strength_range'] = wave_strength_range
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters", expanded=False):
            # Tier filters
            if st.session_state.display_mode == "Hybrid (with Fundamentals)":
                # EPS tier
                if 'eps_tier' in ranked_df.columns:
                    eps_tiers = FilterEngine.get_filter_options(ranked_df, 'eps_tier', filters)
                    selected_eps_tiers = st.multiselect(
                        "EPS Growth Tier",
                        options=eps_tiers,
                        default=st.session_state.get('eps_tier_filter', []),
                        key="eps_tier_filter"
                    )
                    filters['eps_tiers'] = selected_eps_tiers
                
                # PE tier
                if 'pe_tier' in ranked_df.columns:
                    pe_tiers = FilterEngine.get_filter_options(ranked_df, 'pe_tier', filters)
                    selected_pe_tiers = st.multiselect(
                        "PE Ratio Tier",
                        options=pe_tiers,
                        default=st.session_state.get('pe_tier_filter', []),
                        key="pe_tier_filter"
                    )
                    filters['pe_tiers'] = selected_pe_tiers
                
                # Fundamental value filters
                col1, col2 = st.columns(2)
                with col1:
                    min_pe = st.number_input(
                        "Min PE",
                        value=st.session_state.get('min_pe'),
                        key="min_pe"
                    )
                    filters['min_pe'] = min_pe
                
                with col2:
                    max_pe = st.number_input(
                        "Max PE",
                        value=st.session_state.get('max_pe'),
                        key="max_pe"
                    )
                    filters['max_pe'] = max_pe
                
                min_eps_change = st.number_input(
                    "Min EPS Change %",
                    value=st.session_state.get('min_eps_change'),
                    key="min_eps_change"
                )
                filters['min_eps_change'] = min_eps_change
                
                require_fundamental = st.checkbox(
                    "Require fundamental data",
                    value=st.session_state.get('require_fundamental_data', False),
                    key="require_fundamental_data"
                )
                filters['require_fundamental_data'] = require_fundamental
            
            # Price tier
            if 'price_tier' in ranked_df.columns:
                price_tiers = FilterEngine.get_filter_options(ranked_df, 'price_tier', filters)
                selected_price_tiers = st.multiselect(
                    "Price Range",
                    options=price_tiers,
                    default=st.session_state.get('price_tier_filter', []),
                    key="price_tier_filter"
                )
                filters['price_tiers'] = selected_price_tiers
            
            # Trend filter
            trend_filter = st.selectbox(
                "Trend Filter",
                options=["All Trends", "Bullish", "Bearish", "Strong Bullish", "Strong Bearish"],
                index=0,
                key="trend_filter"
            )
            
            if trend_filter != "All Trends":
                trend_ranges = {
                    "Bullish": (60, 100),
                    "Bearish": (0, 40),
                    "Strong Bullish": (80, 100),
                    "Strong Bearish": (0, 20)
                }
                filters['trend_filter'] = trend_filter
                filters['trend_range'] = trend_ranges.get(trend_filter, (0, 100))
        
        # Store filters
        st.session_state.filters = filters
    
    # Apply filters
    filtered_df = FilterEngine.apply_filters(ranked_df, filters)
    
    # Handle quick filters
    quick_filter = st.session_state.get('quick_filter')
    if quick_filter:
        if quick_filter == 'top_gainers':
            filtered_df = filtered_df.nlargest(20, 'ret_1d')
        elif quick_filter == 'volume_surges':
            filtered_df = filtered_df[filtered_df['rvol'] > 2.5].nlargest(50, 'rvol')
        elif quick_filter == 'breakout_ready':
            filtered_df = filtered_df[filtered_df['breakout_score'] >= 80].nlargest(50, 'master_score')
        elif quick_filter == 'hidden_gems':
            if 'patterns' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['patterns'].str.contains('HIDDEN GEM', na=False)]
    
    # Clear button sync
    if st.session_state.get('trigger_clear', False):
        SessionStateManager.clear_filters()
        st.session_state.trigger_clear = False
        st.rerun()
    
    # Quick action buttons
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
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
    
    # Show active filters
    if st.session_state.active_filter_count > 0 or st.session_state.get('quick_filter_applied'):
        filter_col1, filter_col2 = st.columns([5, 1])
        
        with filter_col1:
            filter_text = []
            if st.session_state.get('quick_filter'):
                filter_text.append(f"Quick: {st.session_state.quick_filter.replace('_', ' ').title()}")
            if st.session_state.active_filter_count > 0:
                filter_text.append(f"{st.session_state.active_filter_count} custom filter(s)")
            
            st.info(f"🔍 Active filters: {' + '.join(filter_text)} | **{len(filtered_df):,} stocks** shown")
        
        with filter_col2:
            if st.button("Clear Filters", type="secondary"):
                st.session_state.trigger_clear = True
                st.rerun()
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        UIComponents.render_metric_card(
            "Total Stocks",
            f"{len(filtered_df):,}",
            f"{len(filtered_df)/len(ranked_df)*100:.0f}% of all"
        )
    
    with col2:
        if not filtered_df.empty:
            avg_score = filtered_df['master_score'].mean()
            UIComponents.render_metric_card(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={filtered_df['master_score'].std():.1f}"
            )
    
    with col3:
        top_scorers = (filtered_df['master_score'] >= 80).sum()
        UIComponents.render_metric_card("Top Scorers", f"{top_scorers}")
    
    with col4:
        if st.session_state.display_mode == "Hybrid (with Fundamentals)" and 'eps_change_pct' in filtered_df.columns:
            growth_count = (filtered_df['eps_change_pct'] > 0).sum()
            UIComponents.render_metric_card("EPS Growth +ve", f"{growth_count}")
        else:
            accelerating = (filtered_df.get('acceleration_score', 0) >= 80).sum()
            UIComponents.render_metric_card("Accelerating", f"{accelerating}")
    
    with col5:
        high_rvol = (filtered_df.get('rvol', 0) > 2).sum()
        UIComponents.render_metric_card("High RVOL", f"{high_rvol}")
    
    with col6:
        if 'money_flow_mm' in filtered_df.columns:
            total_flow = filtered_df['money_flow_mm'].sum()
            UIComponents.render_metric_card(
                "Total Flow",
                f"₹{total_flow:,.0f}M"
            )
        else:
            with_patterns = (filtered_df.get('patterns', '') != '').sum()
            UIComponents.render_metric_card("With Patterns", f"{with_patterns}")
    
    # Main tabs
    tabs = st.tabs([
        "📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", 
        "🔍 Search", "📥 Export", "ℹ️ About"
    ])
    
    # Tab 1: Summary
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        
        if not filtered_df.empty:
            UIComponents.render_summary_section(filtered_df)
            
            # Market regime indicator
            if 'market_regime' in filtered_df.columns:
                regime = filtered_df['market_regime'].iloc[0]
                st.markdown("---")
                st.markdown(f"### 🌡️ Market Regime: **{regime}**")
                
                breadth = (filtered_df['ret_30d'] > 0).mean() * 100
                st.progress(breadth / 100)
                st.caption(f"Market Breadth: {breadth:.1f}% positive")
        else:
            st.info("No data to display. Try adjusting your filters.")
    
    # Tab 2: Rankings
    with tabs[1]:
        st.markdown("### 🏆 Stock Rankings")
        
        if not filtered_df.empty:
            # Pagination
            items_per_page = CONFIG.ROWS_PER_PAGE
            total_pages = len(filtered_df) // items_per_page + (1 if len(filtered_df) % items_per_page > 0 else 0)
            
            page_col1, page_col2, page_col3 = st.columns([2, 3, 2])
            
            with page_col2:
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.get('current_page', 1),
                    key="page_selector"
                )
                st.session_state.current_page = current_page - 1
            
            # Calculate slice
            start_idx = st.session_state.current_page * items_per_page
            end_idx = start_idx + items_per_page
            
            # Display columns based on mode
            if st.session_state.display_mode == "Technical":
                display_cols = [
                    'rank', 'ticker', 'company_name', 'category', 'sector',
                    'master_score', 'price', 'ret_1d', 'ret_30d', 'rvol',
                    'money_flow_mm', 'patterns', 'wave_state', 'wave_strength'
                ]
            else:
                display_cols = [
                    'rank', 'ticker', 'company_name', 'category', 'sector',
                    'master_score', 'price', 'ret_1d', 'pe', 'eps_change_pct',
                    'money_flow_mm', 'patterns', 'wave_state'
                ]
            
            # Filter available columns
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Format and display
            display_df = filtered_df[display_cols].iloc[start_idx:end_idx].copy()
            
            # Format numeric columns
            for col in display_df.columns:
                if col in ['price', 'money_flow_mm']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
                elif col in ['master_score', 'ret_1d', 'ret_30d', 'pe', 'eps_change_pct', 'wave_strength']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
                elif col == 'rvol':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "")
            
            # Display table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600,
                hide_index=True
            )
            
            # Show page info
            st.caption(f"Showing {start_idx+1} to {min(end_idx, len(filtered_df))} of {len(filtered_df)} stocks")
        else:
            st.info("No stocks match the current filters.")
    
    # Tab 3: Wave Radar
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection")
        
        if not filtered_df.empty and 'wave_state' in filtered_df.columns:
            # Wave controls
            radar_col1, radar_col2, radar_col3 = st.columns([2, 2, 2])
            
            with radar_col1:
                wave_filter = st.selectbox(
                    "Wave State Filter",
                    options=["All Waves", "🌊🌊🌊 CRESTING", "🌊🌊 BUILDING", 
                            "🌊 FORMING", "💥 BREAKING"],
                    index=0
                )
            
            with radar_col2:
                sensitivity = st.select_slider(
                    "Detection Sensitivity",
                    options=["Conservative", "Balanced", "Aggressive"],
                    value="Balanced"
                )
            
            with radar_col3:
                sort_by = st.selectbox(
                    "Sort By",
                    options=["Wave Strength", "Master Score", "RVOL", "Money Flow"],
                    index=0
                )
            
            # Apply wave filter
            wave_df = filtered_df.copy()
            if wave_filter != "All Waves":
                wave_df = wave_df[wave_df['wave_state'] == wave_filter]
            
            # Apply sensitivity filter
            if sensitivity == "Conservative":
                wave_df = wave_df[wave_df['wave_strength'] >= 75]
            elif sensitivity == "Aggressive":
                wave_df = wave_df[wave_df['wave_strength'] >= 25]
            else:  # Balanced
                wave_df = wave_df[wave_df['wave_strength'] >= 50]
            
            # Sort
            sort_map = {
                "Wave Strength": "wave_strength",
                "Master Score": "master_score",
                "RVOL": "rvol",
                "Money Flow": "money_flow_mm"
            }
            wave_df = wave_df.sort_values(sort_map[sort_by], ascending=False)
            
            # Display wave radar
            if not wave_df.empty:
                # Top momentum movers
                st.markdown("#### 🚀 Top Momentum Movers")
                
                for idx, stock in wave_df.head(10).iterrows():
                    with st.container():
                        UIComponents.render_stock_details(stock, st.session_state.display_mode)
                        st.markdown("---")
                
                # Wave statistics
                st.markdown("#### 📊 Wave Statistics")
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    cresting_count = (wave_df['wave_state'] == "🌊🌊🌊 CRESTING").sum()
                    st.metric("Cresting", cresting_count)
                
                with stat_col2:
                    building_count = (wave_df['wave_state'] == "🌊🌊 BUILDING").sum()
                    st.metric("Building", building_count)
                
                with stat_col3:
                    avg_strength = wave_df['wave_strength'].mean()
                    st.metric("Avg Strength", f"{avg_strength:.1f}%")
                
                with stat_col4:
                    high_momentum = (wave_df['momentum_score'] >= 80).sum()
                    st.metric("High Momentum", high_momentum)
            else:
                st.info("No stocks detected in current wave radar settings.")
        else:
            st.info("Wave analysis requires wave state data.")
    
    # Tab 4: Analysis
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        
        if not filtered_df.empty:
            analysis_tabs = st.tabs(["Sector Analysis", "Pattern Analysis", "Industry Analysis"])
            
            # Sector Analysis
            with analysis_tabs[0]:
                if 'sector' in filtered_df.columns:
                    sector_stats = filtered_df.groupby('sector').agg({
                        'master_score': ['mean', 'count'],
                        'ret_30d': 'mean',
                        'rvol': 'mean',
                        'money_flow_mm': 'sum'
                    }).round(2)
                    
                    sector_stats.columns = ['Avg Score', 'Count', 'Avg 30D Return', 'Avg RVOL', 'Total Flow (M)']
                    sector_stats = sector_stats.sort_values('Avg Score', ascending=False)
                    
                    # Sector performance chart
                    fig = px.bar(
                        sector_stats,
                        x=sector_stats.index,
                        y='Avg Score',
                        color='Avg 30D Return',
                        title="Sector Performance Analysis",
                        labels={'x': 'Sector', 'y': 'Average Score'},
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sector table
                    st.markdown("#### Detailed Sector Statistics")
                    st.dataframe(sector_stats, use_container_width=True)
            
            # Pattern Analysis
            with analysis_tabs[1]:
                if 'patterns' in filtered_df.columns:
                    # Extract all patterns
                    all_patterns = []
                    for patterns in filtered_df['patterns']:
                        if patterns:
                            pattern_list = [p.strip() for p in patterns.split('|')]
                            all_patterns.extend(pattern_list)
                    
                    if all_patterns:
                        pattern_counts = pd.Series(all_patterns).value_counts()
                        
                        # Pattern frequency chart
                        fig = px.bar(
                            x=pattern_counts.values,
                            y=pattern_counts.index,
                            orientation='h',
                            title="Pattern Frequency Analysis",
                            labels={'x': 'Count', 'y': 'Pattern'}
                        )
                        fig.update_layout(height=max(400, len(pattern_counts) * 30))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Pattern performance
                        st.markdown("#### Pattern Performance")
                        
                        pattern_perf = []
                        for pattern in pattern_counts.index[:10]:  # Top 10 patterns
                            mask = filtered_df['patterns'].str.contains(pattern, na=False)
                            if mask.any():
                                perf_data = {
                                    'Pattern': pattern,
                                    'Count': mask.sum(),
                                    'Avg Score': filtered_df[mask]['master_score'].mean(),
                                    'Avg 30D Return': filtered_df[mask].get('ret_30d', 0).mean(),
                                    'Avg RVOL': filtered_df[mask].get('rvol', 1).mean()
                                }
                                pattern_perf.append(perf_data)
                        
                        if pattern_perf:
                            pattern_perf_df = pd.DataFrame(pattern_perf)
                            st.dataframe(pattern_perf_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No patterns detected in current selection.")
            
            # Industry Analysis
            with analysis_tabs[2]:
                if 'industry' in filtered_df.columns:
                    # Industry statistics
                    industry_stats = filtered_df.groupby('industry').agg({
                        'master_score': ['mean', 'count'],
                        'ret_30d': 'mean',
                        'money_flow_mm': 'sum'
                    }).round(2)
                    
                    industry_stats.columns = ['Avg Score', 'Count', 'Avg 30D Return', 'Total Flow (M)']
                    
                    # Filter industries with meaningful data (3+ stocks)
                    meaningful_industries = industry_stats[industry_stats['Count'] >= 3]
                    
                    if not meaningful_industries.empty:
                        meaningful_industries = meaningful_industries.sort_values('Avg Score', ascending=False)
                        
                        # Top industries chart
                        top_industries = meaningful_industries.head(15)
                        
                        fig = px.bar(
                            top_industries,
                            x='Avg Score',
                            y=top_industries.index,
                            orientation='h',
                            color='Avg 30D Return',
                            title="Top 15 Industries by Score (3+ stocks)",
                            labels={'y': 'Industry'},
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=max(400, len(top_industries) * 30))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Industry table
                        st.markdown("#### Industry Statistics (3+ stocks)")
                        st.dataframe(meaningful_industries, use_container_width=True)
                    
                    # Show all industries summary
                    st.markdown("#### All Industries Summary")
                    total_industries = len(industry_stats)
                    industries_with_data = len(meaningful_industries)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Industries", total_industries)
                    with col2:
                        st.metric("Industries (3+ stocks)", industries_with_data)
                    with col3:
                        st.metric("Single-stock Industries", total_industries - industries_with_data)
                    
                    # Option to view all
                    if st.checkbox("Show all industries (including single-stock)"):
                        st.dataframe(industry_stats.sort_values('Count', ascending=False), 
                                   use_container_width=True)
        else:
            st.info("No data available for analysis.")
    
    # Tab 5: Search
    with tabs[4]:
        st.markdown("### 🔍 Stock Search")
        
        search_query = st.text_input(
            "Search by ticker or company name",
            value=st.session_state.get('search_query', ''),
            key="search_input"
        )
        
        if search_query:
            st.session_state.search_query = search_query
            search_results = SearchEngine.search_stocks(ranked_df, search_query)
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} result(s)")
                
                for idx, stock in search_results.iterrows():
                    with st.container():
                        UIComponents.render_stock_details(stock, st.session_state.display_mode)
                        
                        # Additional details
                        with st.expander("View All Details"):
                            detail_cols = st.columns(2)
                            
                            with detail_cols[0]:
                                st.markdown("**Scores:**")
                                for score in ['position_score', 'volume_score', 'momentum_score',
                                            'acceleration_score', 'breakout_score', 'rvol_score']:
                                    if score in stock:
                                        st.write(f"- {score.replace('_', ' ').title()}: {stock[score]:.1f}")
                            
                            with detail_cols[1]:
                                st.markdown("**Metrics:**")
                                if 'position_tension' in stock:
                                    st.write(f"- Position Tension: {stock['position_tension']:.1f}")
                                if 'momentum_harmony' in stock:
                                    st.write(f"- Momentum Harmony: {stock['momentum_harmony']}")
                                if 'smart_money_flow' in stock:
                                    st.write(f"- Smart Money Flow: {stock['smart_money_flow']:.1f}")
                        
                        st.markdown("---")
            else:
                st.warning(f"No results found for '{search_query}'")
    
    # Tab 6: Export
    with tabs[5]:
        st.markdown("### 📥 Export Data")
        
        export_col1, export_col2 = st.columns([2, 1])
        
        with export_col1:
            export_type = st.radio(
                "Export Type",
                ["Quick Summary", "Trading Signals", "Detailed Analysis"],
                help="""
                - Quick Summary: Essential data only
                - Trading Signals: Actionable metrics for trading
                - Detailed Analysis: All scores and metrics
                """
            )
        
        with export_col2:
            st.markdown("#### Export Stats")
            st.write(f"- Total rows: {len(filtered_df)}")
            
            if export_type == "Quick Summary":
                st.write("- Columns: ~15")
            elif export_type == "Trading Signals":
                st.write("- Columns: ~20")
            else:
                st.write("- Columns: 40+")
        
        # Prepare export data
        export_df = prepare_export_data(filtered_df, export_type)
        
        # Generate CSV
        csv = export_df.to_csv(index=False)
        
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wave_detection_{export_type.lower().replace(' ', '_')}_{timestamp}.csv"
        
        st.download_button(
            label=f"📥 Download {export_type} CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )
        
        # Preview
        with st.expander("Preview Export Data"):
            st.dataframe(export_df.head(10), use_container_width=True)
    
    # Tab 7: About
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        st.markdown("""
        #### 🌊 Welcome to Wave Detection Ultimate 3.0
        
        This professional-grade tool combines technical analysis, volume dynamics, 
        and advanced metrics to identify high-potential stocks before they peak.
        
        #### 🎯 Core Features
        
        - **Master Score 3.0**: Proprietary 6-component ranking algorithm
        - **25 Pattern Detection**: Technical, fundamental, range, and intelligence patterns
        - **Wave State Analysis**: Real-time momentum classification
        - **Smart Filtering**: Interconnected filters for precise screening
        - **Money Flow Tracking**: Volume-weighted capital movement
        - **Market Regime Detection**: Risk-on/off market analysis
        
        #### 📊 Scoring Components
        
        1. **Position Score (30%)**: 52-week range positioning with non-linear transformation
        2. **Volume Score (25%)**: Multi-timeframe volume momentum (VMI)
        3. **Momentum Score (15%)**: 30-day price momentum
        4. **Acceleration Score (10%)**: Momentum rate of change
        5. **Breakout Score (10%)**: Technical breakout readiness
        6. **RVOL Score (10%)**: Real-time relative volume
        
        #### 🌊 Wave States
        
        - **🌊🌊🌊 CRESTING**: Maximum momentum (4/4 signals)
        - **🌊🌊 BUILDING**: Strong momentum (3/4 signals)
        - **🌊 FORMING**: Early momentum (1-2/4 signals)
        - **💥 BREAKING**: Momentum exhausted (0/4 signals)
        
        #### 💡 Best Practices
        
        1. **Start with Quick Actions** for instant insights
        2. **Use Wave Radar** for early momentum detection
        3. **Combine multiple patterns** for higher confidence
        4. **Monitor wave state changes** for entry/exit signals
        5. **Export data** for offline analysis
        
        #### 🔧 Technical Details
        
        - **Data Source**: Google Sheets or CSV upload
        - **Update Frequency**: Daily after market close
        - **Cache Duration**: 1 hour
        - **Max Stocks**: 2000+
        - **Processing Time**: <2 seconds
        
        ---
        
        *Built for traders who demand precision, speed, and reliability.*
        
        **Version**: 3.0.0-FINAL  
        **Last Updated**: Production Ready  
        **Status**: ✅ All Systems Operational
        """)
    
    # Debug info
    if st.session_state.get('show_debug'):
        with st.expander("🐛 Debug Information"):
            st.write("**Session State:**")
            debug_state = {
                'Data Source': st.session_state.data_source,
                'Spreadsheet ID': st.session_state.get('user_spreadsheet_id', 'Not set'),
                'Total Stocks': len(ranked_df),
                'Filtered Stocks': len(filtered_df),
                'Active Filters': st.session_state.active_filter_count,
                'Quick Filter': st.session_state.get('quick_filter', 'None'),
                'Display Mode': st.session_state.display_mode,
                'Last Update': st.session_state.get('last_update', 'Never'),
                'Cache Valid': cache_valid if 'cache_valid' in locals() else 'Unknown'
            }
            
            for key, value in debug_state.items():
                st.write(f"- {key}: {value}")

# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    main()
