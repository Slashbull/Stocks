#!/usr/bin/env python3
"""
Wave Detection Ultimate 3.0 - Final Production Version
A professional-grade stock screening tool for Indian markets
Combines the reliability of V1 with smart enhancements from V2
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import gc
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import Counter
from io import BytesIO
from functools import wraps

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class Config:
    """Centralized configuration - all constants in one place"""
    
    # Default spreadsheet configuration
    DEFAULT_GID: str = "0"
    VALID_SHEET_ID_PATTERN: str = r'^[a-zA-Z0-9_-]{44}$'  # Exactly 44 chars, allows - and _
    
    # Performance settings
    CACHE_TTL_SECONDS: int = 3600  # 1 hour cache
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # Data processing settings
    MIN_VALID_PRICE: float = 0.01
    RVOL_DEFAULT: float = 1.0
    
    # Pattern detection thresholds
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'category_leader': 90,
        'hidden_gem': 80,
        'acceleration': 85,
        'institutional': 70,
        'breakout_ready': 80,
        'market_leader': 95,
        'momentum_wave': 80,
        'liquid_leader': 80,
        'long_strength': 80,
        'quality_trend': 75,
        'value_momentum': 0,
        'turnaround': 100,
        'quality_leader': 10,
        'stealth': 1.3,
        'vampire': 2,
        'perfect_storm': 80
    })
    
    # Tier definitions
    TIER_DEFINITIONS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        'eps': {
            'Negative': (-float('inf'), 0),
            'Low (0-20%)': (0, 20),
            'Medium (20-50%)': (20, 50),
            'High (50-100%)': (50, 100),
            'Extreme (>100%)': (100, float('inf'))
        },
        'pe': {
            'Negative/Loss': (-float('inf'), 0),
            'Low (0-15)': (0, 15),
            'Moderate (15-25)': (15, 25),
            'High (25-40)': (25, 40),
            'Very High (>40)': (40, float('inf'))
        },
        'price': {
            'Penny (<10)': (-float('inf'), 10),
            'Low (10-100)': (10, 100),
            'Mid (100-500)': (100, 500),
            'High (500-2000)': (500, 2000),
            'Premium (>2000)': (2000, float('inf'))
        }
    })
    
    # Pattern metadata for better display
    PATTERN_INFO: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        '🔥 CAT LEADER': {'risk': 'low', 'timeframe': 'medium'},
        '💎 HIDDEN GEM': {'risk': 'medium', 'timeframe': 'medium'},
        '🚀 ACCELERATING': {'risk': 'medium', 'timeframe': 'short'},
        '🏦 INSTITUTIONAL': {'risk': 'low', 'timeframe': 'long'},
        '⚡ VOL EXPLOSION': {'risk': 'high', 'timeframe': 'short'},
        '🎯 BREAKOUT': {'risk': 'medium', 'timeframe': 'short'},
        '👑 MARKET LEADER': {'risk': 'low', 'timeframe': 'long'},
        '🌊 MOMENTUM WAVE': {'risk': 'medium', 'timeframe': 'medium'},
        '💰 LIQUID LEADER': {'risk': 'low', 'timeframe': 'long'},
        '💪 LONG STRENGTH': {'risk': 'low', 'timeframe': 'long'},
        '📈 QUALITY TREND': {'risk': 'low', 'timeframe': 'medium'},
        '💎 VALUE MOMENTUM': {'risk': 'low', 'timeframe': 'long'},
        '📊 EARNINGS ROCKET': {'risk': 'medium', 'timeframe': 'medium'},
        '🏆 QUALITY LEADER': {'risk': 'low', 'timeframe': 'long'},
        '⚡ TURNAROUND': {'risk': 'high', 'timeframe': 'medium'},
        '⚠️ HIGH PE': {'risk': 'high', 'timeframe': 'medium'},
        '🎯 52W HIGH APPROACH': {'risk': 'medium', 'timeframe': 'short'},
        '🔄 52W LOW BOUNCE': {'risk': 'high', 'timeframe': 'medium'},
        '👑 GOLDEN ZONE': {'risk': 'medium', 'timeframe': 'medium'},
        '📊 VOL ACCUMULATION': {'risk': 'low', 'timeframe': 'medium'},
        '🔀 MOMENTUM DIVERGE': {'risk': 'medium', 'timeframe': 'short'},
        '🎯 RANGE COMPRESS': {'risk': 'low', 'timeframe': 'medium'},
        '🤫 STEALTH': {'risk': 'medium', 'timeframe': 'medium'},
        '🧛 VAMPIRE': {'risk': 'very_high', 'timeframe': 'short'},
        '⛈️ PERFECT STORM': {'risk': 'medium', 'timeframe': 'short'}
    })
    
    # Column definitions
    CRITICAL_COLUMNS: List[str] = ['ticker', 'price']
    PERCENTAGE_COLUMNS: List[str] = [
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 
        'ret_1y', 'ret_3y', 'ret_5y', 'from_low_pct', 'from_high_pct', 
        'eps_change_pct'
    ]
    VOLUME_RATIO_COLUMNS: List[str] = [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
        'vol_ratio_90d_180d'
    ]
    
    # Value bounds for validation
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000),
        'volume': (0, 1e12),
        'rvol': (0, 1000),
        'pe': (-1000, 1000),
        'returns': (-99.99, 10000)
    })

# Global configuration instance
CONFIG = Config()

# ============================================
# LOGGING SETUP
# ============================================

class SimpleLogger:
    """Simple logging without over-engineering"""
    
    def __init__(self):
        self.logger = logging.getLogger('WaveDetection')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)

logger = SimpleLogger()

# ============================================
# SESSION STATE MANAGEMENT
# ============================================

class SessionStateManager:
    """Manage session state with persistence"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with proper defaults"""
        
        default_states = {
            # Data related
            'data_source': 'sheet',
            'last_good_data': None,
            'ranked_df': pd.DataFrame(),
            'last_loaded_url': '',
            'user_spreadsheet_id': '',  # Persistent spreadsheet ID
            
            # Filters
            'category_filter': [],
            'sector_filter': [],
            'industry_filter': [],
            'min_score': 0,
            'patterns': [],
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
            'wave_strength_range_slider': (0, 100),
            
            # UI state
            'current_page': 0,
            'search_query': '',
            'active_filter_count': 0,
            'trigger_clear': False,
            'show_debug': False,
            'quick_filter': None,
            'quick_filter_applied': False,
            
            # Performance metrics
            'performance_metrics': {}
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        """Clear all filters and reset to defaults"""
        
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter', 'min_score',
            'patterns', 'trend_filter', 'trend_range', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'min_eps_change', 
            'min_pe', 'max_pe', 'require_fundamental_data', 'wave_states_filter',
            'wave_strength_range_slider', 'quick_filter', 'quick_filter_applied'
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = 'All Trends'
                    else:
                        st.session_state[key] = ''
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], tuple):
                    if key == 'trend_range':
                        st.session_state[key] = (0, 100)
                    elif key == 'wave_strength_range_slider':
                        st.session_state[key] = (0, 100)
                elif isinstance(st.session_state[key], (int, float)):
                    st.session_state[key] = 0 if key == 'min_score' else None
                else:
                    st.session_state[key] = None
        
        st.session_state.filters = {}
        st.session_state.active_filter_count = 0
        st.session_state.trigger_clear = False

# ============================================
# DATA VALIDATION
# ============================================

class DataValidator:
    """Validate and clean data"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> Tuple[bool, str]:
        """Validate dataframe structure"""
        if df is None:
            return False, f"{context}: DataFrame is None"
        
        if df.empty:
            return False, f"{context}: DataFrame is empty"
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return False, f"{context}: Missing columns: {missing_cols}"
        
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False, 
                          bounds: Optional[Tuple[float, float]] = None) -> float:
        """Clean and validate numeric values"""
        if pd.isna(value) or value is None:
            return 0.0
        
        try:
            # Handle string representations
            if isinstance(value, str):
                value = value.strip()
                if value in ['', '-', 'N/A', 'n/a', '#N/A']:
                    return 0.0
                # Remove % sign if present
                if is_percentage and value.endswith('%'):
                    value = value[:-1]
                value = float(value)
            else:
                value = float(value)
            
            # Handle infinity
            if np.isinf(value):
                return 0.0
            
            # Apply bounds if specified
            if bounds:
                value = np.clip(value, bounds[0], bounds[1])
            
            return value
            
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def sanitize_string(value: Any) -> str:
        """Clean string values"""
        if pd.isna(value) or value is None:
            return "Unknown"
        
        value = str(value).strip()
        if value.lower() in ['', '-', 'n/a', 'nan', 'none', 'null']:
            return "Unknown"
        
        return value

# ============================================
# DATA LOADING WITH RETRY LOGIC
# ============================================

def load_data_with_retry(source: str, uploaded_file=None) -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """Load data with retry logic and proper error handling"""
    
    metadata = {
        'source': source,
        'rows': 0,
        'columns': 0,
        'processing_start': datetime.now(timezone.utc),
        'errors': [],
        'warnings': []
    }
    
    for attempt in range(CONFIG.MAX_RETRIES):
        try:
            if source == "upload" and uploaded_file is not None:
                logger.info(f"Loading data from uploaded file (attempt {attempt + 1})")
                df = pd.read_csv(uploaded_file, low_memory=False)
                metadata['source'] = "CSV Upload"
                
            elif source == "sheet":
                spreadsheet_id = st.session_state.get('user_spreadsheet_id', '').strip()
                
                if not spreadsheet_id:
                    raise ValueError("Please enter a Google Spreadsheet ID")
                
                # Validate spreadsheet ID
                if not re.match(CONFIG.VALID_SHEET_ID_PATTERN, spreadsheet_id):
                    raise ValueError("Invalid Spreadsheet ID format. Must be exactly 44 characters.")
                
                # Construct URL
                csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={CONFIG.DEFAULT_GID}"
                
                logger.info(f"Loading data from Google Sheets (attempt {attempt + 1})")
                df = pd.read_csv(csv_url, low_memory=False)
                metadata['source'] = "Google Sheets"
                metadata['spreadsheet_id'] = spreadsheet_id
                st.session_state.last_loaded_url = csv_url
                
            else:
                raise ValueError(f"Unknown data source: {source}")
            
            # If successful, break the retry loop
            metadata['rows'] = len(df)
            metadata['columns'] = len(df.columns)
            return df, datetime.now(timezone.utc), metadata
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < CONFIG.MAX_RETRIES - 1:
                time.sleep(CONFIG.RETRY_DELAY)
            else:
                # Provide user-friendly error messages
                if "404" in error_msg or "Not Found" in error_msg:
                    raise ValueError("Spreadsheet not found. Please check the ID and ensure it's set to 'Anyone with link can view'.")
                elif "timeout" in error_msg.lower():
                    raise ValueError("Connection timed out. Please check your internet connection and try again.")
                else:
                    raise ValueError(f"Failed to load data after {CONFIG.MAX_RETRIES} attempts: {error_msg}")

# ============================================
# DATA PROCESSING
# ============================================

class DataProcessor:
    """Process and clean data"""
    
    @staticmethod
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Complete data processing pipeline"""
        
        df = df.copy()
        initial_count = len(df)
        
        # Process numeric columns
        numeric_cols = [col for col in df.columns if col not in 
                       ['ticker', 'company_name', 'category', 'sector', 'industry', 'year', 'market_cap']]
        
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
                df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, is_pct, bounds))
        
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
        df = df[df['price'] > CONFIG.MIN_VALID_PRICE]
        
        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        # Fill missing values
        df = DataProcessor._fill_missing_values(df)
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        # Log processing results
        removed = initial_count - len(df)
        if removed > 0:
            metadata['warnings'].append(f"Removed {removed} invalid rows")
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows")
        
        return df
    
    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults"""
        
        # Position data
        df['from_low_pct'] = df.get('from_low_pct', pd.Series(50, index=df.index)).fillna(50)
        df['from_high_pct'] = df.get('from_high_pct', pd.Series(-50, index=df.index)).fillna(-50)
        
        # RVOL
        df['rvol'] = df.get('rvol', pd.Series(1.0, index=df.index)).fillna(CONFIG.RVOL_DEFAULT)
        
        # Returns
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols:
            df[col] = df[col].fillna(0)
        
        # Volumes
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols:
            df[col] = df[col].fillna(0)
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications using flexible system"""
        
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Classify value into tier"""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
            
            return "Unknown"
        
        # Add EPS tier
        if 'eps_change_pct' in df.columns:
            df['eps_tier'] = df['eps_change_pct'].apply(
                lambda x: classify_tier(x, CONFIG.TIER_DEFINITIONS['eps'])
            )
        
        # Add PE tier
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: classify_tier(x, CONFIG.TIER_DEFINITIONS['pe'])
            )
        
        # Add price tier
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(
                lambda x: classify_tier(x, CONFIG.TIER_DEFINITIONS['price'])
            )
        
        return df

# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    """Calculate all scores and rankings"""
    
    @staticmethod
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores and master score"""
        
        if df.empty:
            return df
        
        # Calculate component scores
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
            df['position_score'] * 0.30 +
            df['volume_score'] * 0.25 +
            df['momentum_score'] * 0.15 +
            df['acceleration_score'] * 0.10 +
            df['breakout_score'] * 0.10 +
            df['rvol_score'] * 0.10
        ).clip(0, 100)
        
        # Calculate rankings
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        
        # Calculate category ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        return df
    
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safe ranking that handles all edge cases"""
        if series.empty or series.isna().all():
            return pd.Series(50.0 if pct else 1, index=series.index)
        
        ranks = series.rank(pct=pct, ascending=ascending, na_option='bottom')
        
        if pct:
            return (ranks * 100).fillna(50)
        else:
            return ranks.fillna(series.count() + 1)
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score with non-linear enhancement"""
        if 'from_low_pct' not in df.columns or 'from_high_pct' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # Non-linear transformation for better extremes
        from_low_transformed = np.tanh(df['from_low_pct'] / 100) * 100
        
        # Calculate components
        low_score = RankingEngine._safe_rank(from_low_transformed, pct=True, ascending=True)
        high_score = 100 - RankingEngine._safe_rank(df['from_high_pct'].abs(), pct=True, ascending=True)
        
        # Weighted average (60% low, 40% high)
        return (low_score * 0.6 + high_score * 0.4).clip(0, 100)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate volume score with VMI enhancement"""
        volume_score = pd.Series(50, index=df.index)
        
        volume_metrics = []
        weights = []
        
        # Volume ratios with non-linear response
        for col, weight in [
            ('vol_ratio_1d_90d', 0.35),
            ('vol_ratio_7d_90d', 0.25),
            ('vol_ratio_30d_90d', 0.20),
            ('vol_ratio_90d_180d', 0.20)
        ]:
            if col in df.columns:
                # Non-linear response for volume spikes
                vol_ratio = df[col].fillna(1.0)
                contribution = np.tanh((vol_ratio - 1) * 0.5) * 50 + 50
                volume_metrics.append(contribution)
                weights.append(weight)
        
        if volume_metrics:
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            
            for metric, weight in zip(volume_metrics, normalized_weights):
                volume_score += (metric - 50) * weight
        
        # RVOL boost
        if 'rvol' in df.columns:
            rvol_boost = np.where(df['rvol'] > 2, 10, 0)
            volume_score += rvol_boost
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score"""
        if 'ret_30d' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # Base momentum from 30-day returns
        momentum_score = pd.Series(50, index=df.index)
        ret_30d = df['ret_30d'].fillna(0)
        
        # Non-linear scoring
        momentum_score[ret_30d > 50] = 90
        momentum_score[(ret_30d > 30) & (ret_30d <= 50)] = 80
        momentum_score[(ret_30d > 15) & (ret_30d <= 30)] = 70
        momentum_score[(ret_30d > 5) & (ret_30d <= 15)] = 60
        momentum_score[(ret_30d > 0) & (ret_30d <= 5)] = 50
        momentum_score[(ret_30d > -10) & (ret_30d <= 0)] = 40
        momentum_score[ret_30d <= -10] = 30
        
        return momentum_score.clip(0, 100)
    
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate acceleration score"""
        acceleration_score = pd.Series(50, index=df.index)
        
        # Calculate short vs long momentum
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(df['ret_30d'] != 0, df['ret_7d'] / 7, 0)
                daily_30d_pace = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
                
                acceleration_ratio = np.where(
                    daily_30d_pace != 0,
                    daily_7d_pace / daily_30d_pace,
                    1.0
                )
                
                # Score based on acceleration
                acceleration_score = 50 + (acceleration_ratio - 1) * 25
        
        return acceleration_score.clip(0, 100)
    
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout readiness score"""
        breakout_score = pd.Series(50, index=df.index)
        
        # Multiple breakout signals
        if 'from_high_pct' in df.columns:
            near_high = (df['from_high_pct'] > -10).astype(float) * 30
            breakout_score += near_high
        
        if 'volume_score' in df.columns:
            volume_signal = (df['volume_score'] > 70).astype(float) * 20
            breakout_score += volume_signal
        
        if 'momentum_score' in df.columns:
            momentum_signal = (df['momentum_score'] > 60).astype(float) * 20
            breakout_score += momentum_signal
        
        if 'trend_quality' in df.columns:
            trend_signal = (df['trend_quality'] > 70).astype(float) * 20
            breakout_score += trend_signal
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate RVOL-based score"""
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        rvol_score = pd.Series(50, index=df.index)
        rvol = df['rvol'].fillna(1.0)
        
        # Non-linear RVOL scoring
        rvol_score[rvol > 5] = 100
        rvol_score[(rvol > 3) & (rvol <= 5)] = 90
        rvol_score[(rvol > 2) & (rvol <= 3)] = 80
        rvol_score[(rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score[(rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score[(rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score[rvol <= 0.5] = 30
        
        return rvol_score.clip(0, 100)
    
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality score"""
        trend_score = pd.Series(50, index=df.index)
        
        # Price above SMAs
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
            above_sma20 = (df['price'] > df['sma_20d']).astype(float) * 20
            above_sma50 = (df['price'] > df['sma_50d']).astype(float) * 20
            above_sma200 = (df['price'] > df['sma_200d']).astype(float) * 20
            
            # SMA alignment
            sma_aligned = ((df['sma_20d'] > df['sma_50d']) & 
                          (df['sma_50d'] > df['sma_200d'])).astype(float) * 20
            
            trend_score = 20 + above_sma20 + above_sma50 + above_sma200 + sma_aligned
        
        return trend_score.clip(0, 100)
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score"""
        strength_score = pd.Series(50, index=df.index)
        
        # Average of long-term returns
        long_returns = []
        for col in ['ret_3m', 'ret_6m', 'ret_1y']:
            if col in df.columns:
                long_returns.append(df[col].fillna(0))
        
        if long_returns:
            avg_return = pd.concat(long_returns, axis=1).mean(axis=1)
            
            # Score based on average long-term return
            strength_score[avg_return > 100] = 90
            strength_score[(avg_return > 50) & (avg_return <= 100)] = 80
            strength_score[(avg_return > 30) & (avg_return <= 50)] = 70
            strength_score[(avg_return > 15) & (avg_return <= 30)] = 60
            strength_score[(avg_return > 0) & (avg_return <= 15)] = 50
            strength_score[avg_return <= 0] = 40
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score"""
        if 'volume_30d' not in df.columns or 'price' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # Calculate dollar volume
        dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
        
        # Rank based on dollar volume
        liquidity_score = RankingEngine._safe_rank(dollar_volume, pct=True, ascending=True)
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks within each category"""
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        categories = df['category'].unique()
        
        for category in categories:
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
# ADVANCED METRICS
# ============================================

class AdvancedMetrics:
    """Calculate advanced metrics and wave analysis"""
    
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics"""
        
        if df.empty:
            return df
        
        # Money Flow
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow_mm'] = (df['price'] * df['volume_1d'] * df['rvol']) / 1_000_000
        else:
            df['money_flow_mm'] = 0
        
        # VMI (Volume Momentum Index)
        df['vmi'] = AdvancedMetrics._calculate_vmi(df)
        
        # Position Tension
        df['position_tension'] = AdvancedMetrics._calculate_position_tension(df)
        
        # Momentum Harmony
        df['momentum_harmony'] = AdvancedMetrics._calculate_momentum_harmony(df)
        
        # Wave Strength
        df['wave_strength'] = df.apply(AdvancedMetrics._calculate_wave_strength, axis=1)
        
        # Wave State
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)
        
        # Market Regime
        df['market_regime'] = AdvancedMetrics._detect_market_regime(df)
        
        # Smart Money Flow
        df['smart_money_flow'] = AdvancedMetrics._calculate_smart_money_flow(df)
        
        return df
    
    @staticmethod
    def _calculate_vmi(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Momentum Index"""
        vmi = pd.Series(50, index=df.index)
        
        # Volume persistence check
        vol_cols = ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        if all(col in df.columns for col in vol_cols):
            vol_persist = ((df['vol_ratio_7d_90d'] > 1.2) & 
                          (df['vol_ratio_30d_90d'] > 1.1)).astype(float) * 30
            vmi += vol_persist
        
        # Volume acceleration
        if 'volume_score' in df.columns:
            vol_accel = (df['volume_score'] > 70).astype(float) * 20
            vmi += vol_accel
        
        return vmi.clip(0, 100)
    
    @staticmethod
    def _calculate_position_tension(df: pd.DataFrame) -> pd.Series:
        """Calculate position tension with exponential response"""
        if 'from_low_pct' not in df.columns or 'from_high_pct' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # Exponential response for extremes
        low_tension = np.exp(-df['from_low_pct'] / 20) * 100
        high_tension = (1 - np.exp(df['from_high_pct'] / 20)) * 100
        
        # Average tension
        position_tension = (low_tension + high_tension) / 2
        
        return position_tension.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_harmony(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum harmony (0-4)"""
        harmony = pd.Series(0, index=df.index)
        
        # Check if multiple timeframes are positive
        for col in ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m']:
            if col in df.columns:
                harmony += (df[col] > 0).astype(int)
        
        return harmony
    
    @staticmethod
    def _calculate_wave_strength(row: pd.Series) -> float:
        """Calculate wave strength based on signals firing"""
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
    
    @staticmethod
    def _get_wave_state(row: pd.Series) -> str:
        """Determine wave state for a stock"""
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
    def _detect_market_regime(df: pd.DataFrame) -> str:
        """Simple market regime detection"""
        if 'ret_30d' not in df.columns:
            return "NEUTRAL ⚖️"
        
        # Calculate breadth
        breadth = (df['ret_30d'] > 0).mean()
        strong_gainers = (df['ret_30d'] > 10).mean()
        
        if breadth > 0.6 and strong_gainers > 0.3:
            return "RISK-ON 🔥"
        elif breadth < 0.4:
            return "RISK-OFF ❄️"
        else:
            return "NEUTRAL ⚖️"
    
    @staticmethod
    def _calculate_smart_money_flow(df: pd.DataFrame) -> pd.Series:
        """Calculate smart money flow indicator"""
        smart_flow = pd.Series(50, index=df.index)
        
        # Volume persistence
        if 'vol_ratio_7d_90d' in df.columns and 'vol_ratio_30d_90d' in df.columns:
            vol_persist = ((df['vol_ratio_7d_90d'] > 1.2) & 
                          (df['vol_ratio_30d_90d'] > 1.1)).astype(int) * 20
            smart_flow += vol_persist
        
        # Price-volume divergence
        if 'ret_30d' in df.columns and 'volume_score' in df.columns:
            divergence = np.where(
                (abs(df['ret_30d']) < 5) & (df['volume_score'] > 70), 
                20, 0
            )
            smart_flow += divergence
        
        return smart_flow.clip(0, 100)

# ============================================
# PATTERN DETECTION
# ============================================

class PatternDetector:
    """Detect all 25 patterns efficiently"""
    
    @staticmethod
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns using vectorized operations"""
        
        if df.empty:
            df['patterns'] = ''
            return df
        
        # Get all pattern definitions
        patterns = PatternDetector._get_all_pattern_definitions(df)
        
        # Vectorized pattern detection
        pattern_matrix = np.zeros((len(df), len(patterns)), dtype=bool)
        pattern_names = []
        
        for i, (pattern_name, mask) in enumerate(patterns):
            if mask is not None and isinstance(mask, pd.Series) and not mask.empty:
                pattern_matrix[:, i] = mask.values
                pattern_names.append(pattern_name)
        
        # Convert to string format
        df['patterns'] = PatternDetector._matrix_to_patterns(pattern_matrix, pattern_names)
        
        return df
    
    @staticmethod
    def _get_all_pattern_definitions(df: pd.DataFrame) -> List[Tuple[str, Optional[pd.Series]]]:
        """Get all 25 pattern definitions"""
        patterns = []
        
        # Technical Patterns (11)
        
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
        if 'trend_quality' in df.columns and 'master_score' in df.columns:
            mask = (
                (df['trend_quality'] >= CONFIG.PATTERN_THRESHOLDS['quality_trend']) &
                (df['master_score'] >= 70)
            )
            patterns.append(('📈 QUALITY TREND', mask))
        
        # Fundamental Patterns (5)
        if st.session_state.get('display_mode', 'technical') == 'hybrid':
            
            # 12. Value Momentum
            if all(col in df.columns for col in ['pe', 'momentum_score']):
                has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
                mask = (
                    has_valid_pe & 
                    (df['pe'] < 20) & 
                    (df['momentum_score'] >= 70)
                )
                patterns.append(('💎 VALUE MOMENTUM', mask))
            
            # 13. Earnings Rocket
            if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
                mask = (
                    (df['eps_change_pct'] > CONFIG.PATTERN_THRESHOLDS['turnaround']) &
                    (df['acceleration_score'] >= 70)
                )
                patterns.append(('📊 EARNINGS ROCKET', mask))
            
            # 14. Quality Leader
            if all(col in df.columns for col in ['pe', 'master_score', 'trend_quality']):
                has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
                mask = (
                    has_valid_pe &
                    (df['pe'] > 10) & (df['pe'] < 30) &
                    (df['master_score'] >= 85) &
                    (df['trend_quality'] >= 80)
                )
                patterns.append(('🏆 QUALITY LEADER', mask))
            
            # 15. Turnaround
            if 'eps_change_pct' in df.columns and 'from_low_pct' in df.columns:
                mask = (
                    (df['eps_change_pct'] > 50) &
                    (df['from_low_pct'] < 30)
                )
                patterns.append(('⚡ TURNAROUND', mask))
            
            # 16. High PE Warning
            if 'pe' in df.columns:
                has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
                mask = has_valid_pe & (df['pe'] > 100)
                patterns.append(('⚠️ HIGH PE', mask))
        
        # Range Patterns (6)
        
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
        
        # Intelligence Patterns (3)
        
        # 23. Stealth Accumulation
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'ret_30d']):
            mask = (
                (df['vol_ratio_7d_90d'] > CONFIG.PATTERN_THRESHOLDS['stealth']) &
                (df['vol_ratio_30d_90d'] > 1.2) &
                (abs(df['ret_30d']) < 10)
            )
            patterns.append(('🤫 STEALTH', mask))
        
        # 24. Vampire Pattern
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'rvol', 'category']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_30d_pace = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            
            mask = (
                (daily_7d_pace > daily_30d_pace * CONFIG.PATTERN_THRESHOLDS['vampire']) &
                (df['rvol'] > 3) &
                (df['category'].isin(['Small Cap', 'Micro Cap']))
            )
            patterns.append(('🧛 VAMPIRE', mask))
        
        # 25. Perfect Storm
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            mask = (
                (df['momentum_harmony'] == 4) &
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['perfect_storm'])
            )
            patterns.append(('⛈️ PERFECT STORM', mask))
        
        return patterns
    
    @staticmethod
    def _matrix_to_patterns(matrix: np.ndarray, pattern_names: List[str]) -> pd.Series:
        """Convert boolean matrix to pattern strings"""
        patterns = []
        
        for row in matrix:
            active_patterns = [pattern_names[i] for i, active in enumerate(row) if active]
            patterns.append(' | '.join(active_patterns) if active_patterns else '')
        
        return pd.Series(patterns)

# ============================================
# FILTERING ENGINE
# ============================================

class FilterEngine:
    """Handle all filtering operations efficiently"""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with optimized performance"""
        
        if df.empty:
            return df
        
        # Start with all rows
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
        
        # Tier filters
        for tier_type, col_name in [
            ('eps_tiers', 'eps_tier'),
            ('pe_tiers', 'pe_tier'),
            ('price_tiers', 'price_tier')
        ]:
            tier_values = filters.get(tier_type, [])
            if tier_values and 'All' not in tier_values:
                col_name = tier_type.replace('_tiers', '_tier')
                if col_name in df.columns:
                    mask &= df[col_name].isin(tier_values)
        
        # Data completeness filter
        if filters.get('require_fundamental_data', False):
            if 'pe' in df.columns and 'eps_change_pct' in df.columns:
                mask &= df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna()
        
        # Wave State filter
        wave_states = filters.get('wave_states', [])
        if wave_states and 'All' not in wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)
        
        # Wave Strength filter
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and 'wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            mask &= (df['wave_strength'] >= min_ws) & (df['wave_strength'] <= max_ws)
        
        # Apply mask
        filtered_df = df[mask].copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        """Get available filter options with smart interconnection"""
        
        if df.empty or column not in df.columns:
            return []
        
        # Apply other filters first
        temp_filters = current_filters.copy()
        
        # Remove current column's filter to avoid circular dependency
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
    """Handle search operations"""
    
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
        
        # Combine results
        combined = pd.concat([ticker_contains, company_contains]).drop_duplicates(subset=['ticker'])
        
        # Sort by relevance
        combined['relevance'] = 0
        combined.loc[combined['ticker'].str.upper() == query, 'relevance'] = 3
        combined.loc[combined['ticker'].str.upper().str.startswith(query), 'relevance'] = 2
        combined.loc[combined['ticker'].str.upper().str.contains(query), 'relevance'] = 1
        
        return combined.sort_values(['relevance', 'master_score'], ascending=[False, False])

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_metric_card(label: str, value: str, delta: str = None):
        """Render a metric card"""
        st.metric(label=label, value=value, delta=delta)
    
    @staticmethod
    def render_stock_card(row: pd.Series, show_fundamentals: bool = False):
        """Render a detailed stock card"""
        
        with st.container():
            # Header
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"### {row['ticker']} - {row['company_name']}")
                st.caption(f"{row['category']} | {row['sector']}")
            
            with col2:
                st.metric("Price", f"₹{row['price']:.2f}", 
                         f"{row.get('ret_1d', 0):.1f}%" if 'ret_1d' in row else None)
            
            with col3:
                st.metric("Rank", f"#{int(row['rank'])}", 
                         f"Score: {row['master_score']:.1f}")
            
            # Metrics
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.metric("RVOL", f"{row.get('rvol', 1):.1f}x")
            
            with metrics_cols[1]:
                st.metric("Money Flow", f"₹{row.get('money_flow_mm', 0):.1f}M")
            
            with metrics_cols[2]:
                st.metric("Wave", row.get('wave_state', 'Unknown'))
            
            with metrics_cols[3]:
                st.metric("Strength", f"{row.get('wave_strength', 0):.0f}%")
            
            # Patterns
            if row.get('patterns'):
                st.markdown("**Patterns:** " + row['patterns'])
            
            # Fundamentals
            if show_fundamentals:
                fund_cols = st.columns(3)
                
                with fund_cols[0]:
                    if 'pe' in row and pd.notna(row['pe']):
                        st.metric("P/E", f"{row['pe']:.1f}")
                
                with fund_cols[1]:
                    if 'eps_change_pct' in row and pd.notna(row['eps_change_pct']):
                        st.metric("EPS Change", f"{row['eps_change_pct']:.1f}%")
                
                with fund_cols[2]:
                    if 'pe_tier' in row:
                        st.metric("PE Tier", row['pe_tier'])
    
    @staticmethod
    def render_summary_section(df: pd.DataFrame):
        """Render executive summary"""
        
        # Market Overview
        st.markdown("#### 🌡️ Market Overview")
        
        overview_cols = st.columns(4)
        
        with overview_cols[0]:
            gaining = (df['ret_1d'] > 0).sum() if 'ret_1d' in df.columns else 0
            total = len(df)
            st.metric("Gainers", f"{gaining}/{total}", 
                     f"{gaining/total*100:.0f}%" if total > 0 else "0%")
        
        with overview_cols[1]:
            high_volume = (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0
            st.metric("High Volume", f"{high_volume}", 
                     f"{high_volume/total*100:.0f}%" if total > 0 else "0%")
        
        with overview_cols[2]:
            if 'market_regime' in df.columns:
                regime = df['market_regime'].iloc[0] if not df.empty else "Unknown"
                st.metric("Market Regime", regime)
        
        with overview_cols[3]:
            avg_strength = df['wave_strength'].mean() if 'wave_strength' in df.columns else 0
            st.metric("Avg Wave Strength", f"{avg_strength:.0f}%")
        
        # Top Movers
        if not df.empty and 'ret_1d' in df.columns:
            st.markdown("---")
            st.markdown("#### 🚀 Top Movers Today")
            
            movers_cols = st.columns(2)
            
            with movers_cols[0]:
                st.markdown("**📈 Top Gainers**")
                top_gainers = df.nlargest(5, 'ret_1d')[['ticker', 'ret_1d', 'master_score']]
                for _, stock in top_gainers.iterrows():
                    st.markdown(f"• **{stock['ticker']}** +{stock['ret_1d']:.1f}% (Score: {stock['master_score']:.0f})")
            
            with movers_cols[1]:
                st.markdown("**📉 Top Losers**")
                top_losers = df.nsmallest(5, 'ret_1d')[['ticker', 'ret_1d', 'master_score']]
                for _, stock in top_losers.iterrows():
                    st.markdown(f"• **{stock['ticker']}** {stock['ret_1d']:.1f}% (Score: {stock['master_score']:.0f})")
        
        # Wave Distribution
        if 'wave_state' in df.columns:
            st.markdown("---")
            st.markdown("#### 🌊 Wave State Distribution")
            
            wave_counts = df['wave_state'].value_counts()
            wave_cols = st.columns(len(wave_counts))
            
            for i, (state, count) in enumerate(wave_counts.items()):
                with wave_cols[i]:
                    pct = count / len(df) * 100
                    st.metric(state, f"{count}", f"{pct:.0f}%")

# ============================================
# EXPORT FUNCTIONS
# ============================================

def export_to_csv(df: pd.DataFrame) -> bytes:
    """Export dataframe to CSV"""
    return df.to_csv(index=False).encode('utf-8')

def export_to_excel(df: pd.DataFrame) -> bytes:
    """Export dataframe to Excel with formatting"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Wave Detection', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Wave Detection']
        
        # Format headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BD',
            'border': 1
        })
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-fit columns
        for i, col in enumerate(df.columns):
            column_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, min(column_width, 50))
    
    return output.getvalue()

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
    .stAlert {padding: 1rem; border-radius: 5px;}
    div.stButton > button {
        width: 100%;
        background-color: #1c83e1;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🌊 Wave Detection Ultimate 3.0")
    st.markdown("*Professional Stock Screening & Ranking System*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings & Filters")
        
        # Data source selection
        st.markdown("### 📊 Data Source")
        data_source = st.radio(
            "Select data source:",
            ["Google Sheets", "Upload CSV"],
            index=0 if st.session_state.data_source == "sheet" else 1,
            key="data_source_radio"
        )
        
        st.session_state.data_source = "sheet" if data_source == "Google Sheets" else "upload"
        
        uploaded_file = None
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        else:
            # Spreadsheet ID input with persistence
            st.markdown("### 📋 Google Sheets Configuration")
            
            current_id = st.session_state.get('user_spreadsheet_id', '')
            new_id = st.text_input(
                "Spreadsheet ID (44 characters):",
                value=current_id,
                placeholder="Enter your Google Sheets ID",
                help="The ID is the long string in your Google Sheets URL",
                key="spreadsheet_id_input"
            )
            
            # Only update if valid and changed
            if new_id != current_id:
                if new_id and re.match(CONFIG.VALID_SHEET_ID_PATTERN, new_id.strip()):
                    st.session_state.user_spreadsheet_id = new_id.strip()
                    st.success("✅ Valid Spreadsheet ID")
                    st.rerun()
                elif new_id:
                    st.error("❌ Invalid ID format. Must be exactly 44 characters.")
        
        # Display mode
        st.markdown("---")
        st.markdown("### 🎯 Display Mode")
        display_mode = st.radio(
            "Select display mode:",
            ["Technical Only", "Hybrid (with Fundamentals)"],
            index=0 if st.session_state.get('display_mode', 'technical') == 'technical' else 1
        )
        st.session_state.display_mode = "technical" if display_mode == "Technical Only" else "hybrid"
        
        # Filter counter
        st.markdown("---")
        st.markdown("### 🔍 Active Filters")
        
        # Count active filters
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
    
    # Check for clear trigger from main area
    if st.session_state.get('trigger_clear', False):
        SessionStateManager.clear_filters()
        st.session_state.trigger_clear = False
        st.rerun()
    
    # Data loading
    try:
        # Check prerequisites
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        if st.session_state.data_source == "sheet" and not st.session_state.get('user_spreadsheet_id'):
            st.info("Please enter a Google Spreadsheet ID in the sidebar to load data.")
            st.stop()
        
        # Load data with retry logic
        with st.spinner("Loading data..."):
            df, timestamp, metadata = load_data_with_retry(
                st.session_state.data_source, 
                uploaded_file
            )
        
        # Process data
        with st.spinner("Processing data..."):
            df = DataProcessor.process_dataframe(df, metadata)
            df = RankingEngine.calculate_all_scores(df)
            df = PatternDetector.detect_all_patterns(df)
            df = AdvancedMetrics.calculate_all_metrics(df)
        
        # Store in session state
        st.session_state.ranked_df = df
        st.session_state.last_data_update = timestamp
        
        # Display metadata
        if metadata.get('warnings'):
            for warning in metadata['warnings']:
                st.warning(warning)
        
        st.success(f"✅ Loaded {len(df):,} stocks successfully!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        
        # Try to use cached data
        if 'last_good_data' in st.session_state:
            st.warning("Using cached data from previous successful load")
            df, timestamp, _ = st.session_state.last_good_data
            st.session_state.ranked_df = df
        else:
            st.stop()
    
    # Get data
    ranked_df = st.session_state.get('ranked_df', pd.DataFrame())
    
    if ranked_df.empty:
        st.error("No data available")
        st.stop()
    
    # Prepare filters
    filters = {}
    
    # Apply quick filters if active
    quick_filter = st.session_state.get('quick_filter')
    if quick_filter == 'top_gainers':
        filters['quick'] = lambda df: df.nlargest(50, 'ret_1d') if 'ret_1d' in df.columns else df
    elif quick_filter == 'volume_surges':
        filters['quick'] = lambda df: df[df['rvol'] > 2] if 'rvol' in df.columns else df
    elif quick_filter == 'breakout_ready':
        filters['quick'] = lambda df: df[df['breakout_score'] >= 80] if 'breakout_score' in df.columns else df
    elif quick_filter == 'hidden_gems':
        filters['quick'] = lambda df: df[df['patterns'].str.contains('HIDDEN GEM', na=False)] if 'patterns' in df.columns else df
    
    # Apply quick filter first if exists
    if 'quick' in filters:
        ranked_df_display = filters['quick'](ranked_df)
    else:
        ranked_df_display = ranked_df
    
    # Build standard filters
    if st.session_state.get('category_filter'):
        filters['categories'] = st.session_state.category_filter
    if st.session_state.get('sector_filter'):
        filters['sectors'] = st.session_state.sector_filter
    if st.session_state.get('industry_filter'):
        filters['industries'] = st.session_state.industry_filter
    if st.session_state.get('min_score', 0) > 0:
        filters['min_score'] = st.session_state.min_score
    if st.session_state.get('patterns'):
        filters['patterns'] = st.session_state.patterns
    if st.session_state.get('trend_filter') != 'All Trends':
        filters['trend_filter'] = st.session_state.trend_filter
        filters['trend_range'] = st.session_state.get('trend_range', (0, 100))
    if st.session_state.get('eps_tier_filter'):
        filters['eps_tiers'] = st.session_state.eps_tier_filter
    if st.session_state.get('pe_tier_filter'):
        filters['pe_tiers'] = st.session_state.pe_tier_filter
    if st.session_state.get('price_tier_filter'):
        filters['price_tiers'] = st.session_state.price_tier_filter
    if st.session_state.get('min_eps_change') is not None:
        filters['min_eps_change'] = st.session_state.min_eps_change
    if st.session_state.get('min_pe') is not None:
        filters['min_pe'] = st.session_state.min_pe
    if st.session_state.get('max_pe') is not None:
        filters['max_pe'] = st.session_state.max_pe
    if st.session_state.get('require_fundamental_data', False):
        filters['require_fundamental_data'] = True
    if st.session_state.get('wave_states_filter'):
        filters['wave_states'] = st.session_state.wave_states_filter
    if st.session_state.get('wave_strength_range_slider') != (0, 100):
        filters['wave_strength_range'] = st.session_state.wave_strength_range_slider
    
    # Apply all filters
    filtered_df = FilterEngine.apply_filters(ranked_df_display, filters)
    
    # Store filtered data
    st.session_state.filters = filters
    
    # Main content area
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    qa_cols = st.columns(5)
    
    with qa_cols[0]:
        if st.button("📈 Top Gainers", use_container_width=True):
            st.session_state.quick_filter = 'top_gainers'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_cols[1]:
        if st.button("🔥 Volume Surges", use_container_width=True):
            st.session_state.quick_filter = 'volume_surges'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_cols[2]:
        if st.button("🎯 Breakout Ready", use_container_width=True):
            st.session_state.quick_filter = 'breakout_ready'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_cols[3]:
        if st.button("💎 Hidden Gems", use_container_width=True):
            st.session_state.quick_filter = 'hidden_gems'
            st.session_state.quick_filter_applied = True
            st.rerun()
    
    with qa_cols[4]:
        if st.button("🌊 Show All", use_container_width=True):
            st.session_state.quick_filter = None
            st.session_state.quick_filter_applied = False
            st.rerun()
    
    # Filter status
    if st.session_state.active_filter_count > 0 or st.session_state.get('quick_filter_applied', False):
        filter_cols = st.columns([5, 1])
        
        with filter_cols[0]:
            if st.session_state.get('quick_filter'):
                quick_names = {
                    'top_gainers': '📈 Top Gainers',
                    'volume_surges': '🔥 Volume Surges',
                    'breakout_ready': '🎯 Breakout Ready',
                    'hidden_gems': '💎 Hidden Gems'
                }
                filter_display = quick_names.get(st.session_state.quick_filter, 'Filtered')
                
                if st.session_state.active_filter_count > 0:
                    st.info(f"**Viewing:** {filter_display} + {st.session_state.active_filter_count} other filter{'s' if st.session_state.active_filter_count > 1 else ''} | **{len(filtered_df):,} stocks** shown")
                else:
                    st.info(f"**Viewing:** {filter_display} | **{len(filtered_df):,} stocks** shown")
            else:
                st.info(f"**{st.session_state.active_filter_count} filter{'s' if st.session_state.active_filter_count > 1 else ''} active** | **{len(filtered_df):,} stocks** shown")
        
        with filter_cols[1]:
            if st.button("Clear Filters", type="secondary"):
                st.session_state.trigger_clear = True
                st.rerun()
    
    # Summary metrics
    metric_cols = st.columns(6)
    
    with metric_cols[0]:
        total_stocks = len(filtered_df)
        total_original = len(ranked_df)
        pct_of_all = (total_stocks/total_original*100) if total_original > 0 else 0
        UIComponents.render_metric_card(
            "Total Stocks",
            f"{total_stocks:,}",
            f"{pct_of_all:.0f}% of {total_original:,}"
        )
    
    with metric_cols[1]:
        if not filtered_df.empty and 'master_score' in filtered_df.columns:
            avg_score = filtered_df['master_score'].mean()
            UIComponents.render_metric_card("Avg Score", f"{avg_score:.1f}")
        else:
            UIComponents.render_metric_card("Avg Score", "N/A")
    
    with metric_cols[2]:
        above_70 = (filtered_df['master_score'] >= 70).sum() if not filtered_df.empty else 0
        UIComponents.render_metric_card("Score ≥70", f"{above_70}")
    
    with metric_cols[3]:
        if 'eps_change_pct' in filtered_df.columns:
            positive_eps = (filtered_df['eps_change_pct'] > 0).sum()
            UIComponents.render_metric_card("EPS +ve", f"{positive_eps}")
        else:
            if 'acceleration_score' in filtered_df.columns:
                accelerating = (filtered_df['acceleration_score'] >= 80).sum()
            else:
                accelerating = 0
            UIComponents.render_metric_card("Accelerating", f"{accelerating}")
    
    with metric_cols[4]:
        if 'rvol' in filtered_df.columns:
            high_rvol = (filtered_df['rvol'] > 2).sum()
        else:
            high_rvol = 0
        UIComponents.render_metric_card("High RVOL", f"{high_rvol}")
    
    with metric_cols[5]:
        with_patterns = (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0
        UIComponents.render_metric_card("With Patterns", f"{with_patterns}")
    
    # Main tabs
    tabs = st.tabs([
        "📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", 
        "🔍 Search", "📥 Export", "ℹ️ About"
    ])
    
    # Tab 0: Summary
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        
        if not filtered_df.empty:
            UIComponents.render_summary_section(filtered_df)
            
            # Download section
            st.markdown("---")
            st.markdown("#### 💾 Quick Download")
            
            csv_data = export_to_csv(filtered_df)
            st.download_button(
                label="📥 Download Current View (CSV)",
                data=csv_data,
                file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data to display. Adjust filters or check data source.")
    
    # Tab 1: Rankings
    with tabs[1]:
        st.markdown("### 🏆 Stock Rankings")
        
        # Filters section
        with st.expander("🔧 Filters", expanded=True):
            filter_cols = st.columns(4)
            
            with filter_cols[0]:
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
                
                # Minimum score
                min_score = st.slider(
                    "Minimum Master Score",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.get('min_score', 0),
                    step=5,
                    key="min_score"
                )
                filters['min_score'] = min_score
            
            with filter_cols[1]:
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
            
            with filter_cols[2]:
                # Pattern filter
                if 'patterns' in filtered_df.columns:
                    all_patterns = set()
                    pattern_counts = Counter()
                    
                    for patterns in filtered_df['patterns']:
                        if patterns:
                            pattern_list = [p.strip() for p in patterns.split('|')]
                            all_patterns.update(pattern_list)
                            pattern_counts.update(pattern_list)
                    
                    if all_patterns:
                        sorted_patterns = sorted(
                            pattern_counts.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        
                        pattern_options = [f"{pattern} ({count})" for pattern, count in sorted_patterns]
                        
                        selected_patterns_with_counts = st.multiselect(
                            "Patterns",
                            pattern_options,
                            key="patterns_display"
                        )
                        
                        st.session_state.patterns = [
                            pattern.split(' (')[0] for pattern in selected_patterns_with_counts
                        ]
                
                # Trend filter
                trend_options = ["All Trends", "Bullish", "Bearish", "Strong Bullish", "Strong Bearish"]
                selected_trend = st.selectbox(
                    "Trend Filter",
                    trend_options,
                    index=trend_options.index(st.session_state.get('trend_filter', 'All Trends')),
                    key="trend_filter"
                )
                
                if selected_trend != "All Trends":
                    trend_ranges = {
                        "Bullish": (60, 100),
                        "Bearish": (0, 40),
                        "Strong Bullish": (80, 100),
                        "Strong Bearish": (0, 20)
                    }
                    st.session_state.trend_range = trend_ranges[selected_trend]
            
            with filter_cols[3]:
                # Wave filters
                st.markdown("**Wave Analysis**")
                
                # Wave states
                if 'wave_state' in filtered_df.columns:
                    wave_states = FilterEngine.get_filter_options(ranked_df_display, 'wave_state', filters)
                    selected_wave_states = st.multiselect(
                        "Wave States",
                        options=wave_states,
                        default=st.session_state.get('wave_states_filter', []),
                        key="wave_states_filter"
                    )
                    filters['wave_states'] = selected_wave_states
                
                # Wave strength range
                if 'wave_strength' in filtered_df.columns:
                    wave_strength_range = st.slider(
                        "Wave Strength Range",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.get('wave_strength_range_slider', (0, 100)),
                        step=10,
                        key="wave_strength_range_slider"
                    )
                    filters['wave_strength_range'] = wave_strength_range
        
        # Advanced filters
        if st.session_state.display_mode == 'hybrid':
            with st.expander("🔧 Advanced Filters", expanded=False):
                adv_cols = st.columns(4)
                
                with adv_cols[0]:
                    # Tier filters
                    st.markdown("**Tier Filters**")
                    
                    for tier_type, col_name in [
                        ('eps_tier', 'eps_tier'),
                        ('pe_tier', 'pe_tier'),
                        ('price_tier', 'price_tier')
                    ]:
                        if col_name in filtered_df.columns:
                            tier_options = FilterEngine.get_filter_options(
                                ranked_df_display, col_name, filters
                            )
                            
                            if tier_options:
                                # Add counts to options
                                tier_counts = filtered_df[col_name].value_counts()
                                display_options = []
                                for tier in tier_options:
                                    count = tier_counts.get(tier, 0)
                                    display_options.append(f"{tier} ({count})")
                                
                                selected_tiers = st.multiselect(
                                    f"{col_name.replace('_', ' ').title()}",
                                    options=display_options,
                                    key=f"{col_name}_filter"
                                )
                                
                                # Extract tier names without counts
                                actual_tiers = [t.split(' (')[0] for t in selected_tiers]
                                filters[f"{tier_type}s"] = actual_tiers
                
                with adv_cols[1]:
                    st.markdown("**EPS Filters**")
                    
                    min_eps_change = st.number_input(
                        "Min EPS Change %",
                        value=st.session_state.get('min_eps_change'),
                        placeholder="Any",
                        key="min_eps_change"
                    )
                    if min_eps_change is not None:
                        filters['min_eps_change'] = min_eps_change
                
                with adv_cols[2]:
                    st.markdown("**PE Filters**")
                    
                    pe_col1, pe_col2 = st.columns(2)
                    with pe_col1:
                        min_pe = st.number_input(
                            "Min PE",
                            value=st.session_state.get('min_pe'),
                            placeholder="Any",
                            key="min_pe"
                        )
                        if min_pe is not None:
                            filters['min_pe'] = min_pe
                    
                    with pe_col2:
                        max_pe = st.number_input(
                            "Max PE",
                            value=st.session_state.get('max_pe'),
                            placeholder="Any",
                            key="max_pe"
                        )
                        if max_pe is not None:
                            filters['max_pe'] = max_pe
                
                with adv_cols[3]:
                    st.markdown("**Data Requirements**")
                    
                    require_fundamental = st.checkbox(
                        "Only stocks with PE & EPS data",
                        value=st.session_state.get('require_fundamental_data', False),
                        key="require_fundamental_data"
                    )
                    filters['require_fundamental_data'] = require_fundamental
        
        # Display options
        st.markdown("---")
        display_cols = st.columns(4)
        
        with display_cols[0]:
            items_per_page = st.selectbox(
                "Items per page",
                [25, 50, 100, 200],
                index=1
            )
        
        with display_cols[1]:
            sort_by = st.selectbox(
                "Sort by",
                ["Rank", "Master Score", "Momentum", "Volume", "RVOL", "Wave Strength"],
                index=0
            )
        
        with display_cols[2]:
            sort_order = st.radio(
                "Sort order",
                ["Ascending", "Descending"],
                index=1 if sort_by == "Rank" else 0,
                horizontal=True
            )
        
        # Apply sorting
        sort_column_map = {
            "Rank": "rank",
            "Master Score": "master_score",
            "Momentum": "momentum_score",
            "Volume": "volume_score",
            "RVOL": "rvol",
            "Wave Strength": "wave_strength"
        }
        
        sort_col = sort_column_map.get(sort_by, "rank")
        if sort_col in filtered_df.columns:
            filtered_df = filtered_df.sort_values(
                sort_col,
                ascending=(sort_order == "Ascending")
            )
        
        # Pagination
        total_items = len(filtered_df)
        total_pages = (total_items + items_per_page - 1) // items_per_page
        
        if total_pages > 1:
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.get('current_page', 1),
                key="current_page"
            )
        else:
            page = 1
        
        # Display data
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        if not filtered_df.empty:
            st.info(f"Showing {start_idx + 1}-{end_idx} of {total_items} stocks")
            
            # Prepare display columns
            display_columns = [
                'rank', 'ticker', 'company_name', 'category', 'sector',
                'price', 'ret_1d', 'master_score', 'momentum_score',
                'volume_score', 'rvol', 'money_flow_mm', 'wave_state',
                'wave_strength', 'patterns'
            ]
            
            if st.session_state.display_mode == 'hybrid':
                display_columns.extend(['pe', 'eps_change_pct', 'pe_tier'])
            
            # Filter to available columns
            display_columns = [col for col in display_columns if col in filtered_df.columns]
            
            # Display dataframe
            display_df = filtered_df[display_columns].iloc[start_idx:end_idx]
            
            # Format columns
            format_dict = {
                'price': '{:.2f}',
                'ret_1d': '{:.1f}%',
                'master_score': '{:.1f}',
                'momentum_score': '{:.1f}',
                'volume_score': '{:.1f}',
                'rvol': '{:.1f}x',
                'money_flow_mm': '₹{:.1f}M',
                'wave_strength': '{:.0f}%',
                'pe': '{:.1f}',
                'eps_change_pct': '{:.1f}%'
            }
            
            st.dataframe(
                display_df.style.format(format_dict, na_rep='—'),
                use_container_width=True,
                height=400
            )
        else:
            st.warning("No stocks match the current filters")
    
    # Tab 2: Wave Radar
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection")
        st.markdown("*Catch waves as they form, not after they've peaked!*")
        
        # Wave Radar controls
        radar_cols = st.columns([2, 2, 2, 1])
        
        with radar_cols[0]:
            wave_timeframe = st.selectbox(
                "Detection Timeframe",
                ["All Waves", "Intraday Surge", "3-Day Buildup", 
                 "Weekly Breakout", "Monthly Trend"],
                index=0
            )
        
        with radar_cols[1]:
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=["Conservative", "Balanced", "Aggressive"],
                value="Balanced"
            )
        
        with radar_cols[2]:
            show_details = st.checkbox("Show threshold details", value=False)
        
        # Apply wave filters
        wave_df = filtered_df.copy()
        
        # Timeframe filters
        if wave_timeframe == "Intraday Surge":
            wave_df = wave_df[(wave_df['rvol'] > 2) & (wave_df['ret_1d'] > 2)]
        elif wave_timeframe == "3-Day Buildup":
            if 'ret_3d' in wave_df.columns:
                wave_df = wave_df[(wave_df['ret_3d'] > 5) & (wave_df['acceleration_score'] > 70)]
        elif wave_timeframe == "Weekly Breakout":
            wave_df = wave_df[(wave_df['from_high_pct'] > -10) & (wave_df['volume_score'] > 70)]
        elif wave_timeframe == "Monthly Trend":
            wave_df = wave_df[(wave_df['trend_quality'] >= 70) & (wave_df['ret_30d'] > 10)]
        
        # Sensitivity adjustments
        if sensitivity == "Conservative":
            wave_df = wave_df[wave_df['master_score'] >= 75]
        elif sensitivity == "Aggressive":
            wave_df = wave_df[wave_df['master_score'] >= 60]
        else:  # Balanced
            wave_df = wave_df[wave_df['master_score'] >= 70]
        
        if show_details:
            detail_cols = st.columns(3)
            with detail_cols[0]:
                st.info(f"**Conservative:** Score ≥75, Strict signals")
            with detail_cols[1]:
                st.info(f"**Balanced:** Score ≥70, Standard signals")
            with detail_cols[2]:
                st.info(f"**Aggressive:** Score ≥60, More signals")
        
        # Wave statistics
        st.markdown("---")
        wave_stat_cols = st.columns(4)
        
        with wave_stat_cols[0]:
            cresting = (wave_df['wave_state'] == "🌊🌊🌊 CRESTING").sum()
            st.metric("Cresting Now", f"{cresting}", "Immediate action")
        
        with wave_stat_cols[1]:
            building = (wave_df['wave_state'] == "🌊🌊 BUILDING").sum()
            st.metric("Building", f"{building}", "Prepare positions")
        
        with wave_stat_cols[2]:
            forming = (wave_df['wave_state'] == "🌊 FORMING").sum()
            st.metric("Forming", f"{forming}", "Watch closely")
        
        with wave_stat_cols[3]:
            avg_strength = wave_df['wave_strength'].mean() if not wave_df.empty else 0
            st.metric("Avg Strength", f"{avg_strength:.0f}%")
        
        # Wave detection results
        st.markdown("---")
        
        if not wave_df.empty:
            # Group by wave state
            for state in ["🌊🌊🌊 CRESTING", "🌊🌊 BUILDING", "🌊 FORMING"]:
                state_df = wave_df[wave_df['wave_state'] == state]
                
                if not state_df.empty:
                    st.markdown(f"#### {state}")
                    
                    # Show top stocks in this state
                    top_in_state = state_df.nlargest(10, 'master_score')
                    
                    for _, stock in top_in_state.iterrows():
                        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
                        
                        with col1:
                            st.markdown(f"**{stock['ticker']}** - {stock['company_name']}")
                        
                        with col2:
                            st.markdown(f"Score: **{stock['master_score']:.0f}**")
                        
                        with col3:
                            st.markdown(f"RVOL: **{stock['rvol']:.1f}x**")
                        
                        with col4:
                            st.markdown(f"Strength: **{stock['wave_strength']:.0f}%**")
                        
                        with col5:
                            if stock['patterns']:
                                st.markdown(f"*{stock['patterns'][:50]}{'...' if len(stock['patterns']) > 50 else ''}*")
            
            # Momentum shifts
            st.markdown("---")
            st.markdown("#### 🔄 Momentum Shifts Detected")
            
            shift_cols = st.columns(2)
            
            with shift_cols[0]:
                st.markdown("##### 🚀 Acceleration Leaders")
                if 'acceleration_score' in wave_df.columns:
                    accel_leaders = wave_df[wave_df['acceleration_score'] >= 85].nlargest(5, 'acceleration_score')
                    
                    if not accel_leaders.empty:
                        for _, stock in accel_leaders.iterrows():
                            st.markdown(
                                f"• **{stock['ticker']}** - "
                                f"Accel: {stock['acceleration_score']:.0f} | "
                                f"30D: {stock.get('ret_30d', 0):+.1f}%"
                            )
                    else:
                        st.info("No significant acceleration detected")
            
            with shift_cols[1]:
                st.markdown("##### ⚡ Volume Explosions")
                vol_explosions = wave_df[wave_df['rvol'] > 3].nlargest(5, 'rvol')
                
                if not vol_explosions.empty:
                    for _, stock in vol_explosions.iterrows():
                        st.markdown(
                            f"• **{stock['ticker']}** - "
                            f"RVOL: {stock['rvol']:.1f}x | "
                            f"Vol Score: {stock.get('volume_score', 0):.0f}"
                        )
                else:
                    st.info("No volume explosions detected")
            
            # Smart Money Flow
            if 'smart_money_flow' in wave_df.columns:
                st.markdown("---")
                st.markdown("#### 💰 Smart Money Flow Analysis")
                
                smart_money_leaders = wave_df[wave_df['smart_money_flow'] >= 70].nlargest(10, 'smart_money_flow')
                
                if not smart_money_leaders.empty:
                    money_cols = st.columns(2)
                    
                    with money_cols[0]:
                        st.markdown("**Top Smart Money Flows**")
                        for i, (_, stock) in enumerate(smart_money_leaders.iterrows()):
                            if i < 5:
                                st.markdown(
                                    f"{i+1}. **{stock['ticker']}** - "
                                    f"Flow: {stock['smart_money_flow']:.0f} | "
                                    f"₹{stock['money_flow_mm']:.1f}M"
                                )
                    
                    with money_cols[1]:
                        st.markdown("**By Category**")
                        category_flows = smart_money_leaders.groupby('category')['smart_money_flow'].mean()
                        for cat, flow in category_flows.items():
                            st.markdown(f"• {cat}: {flow:.0f}")
                else:
                    st.info("No significant smart money flows detected")
        else:
            st.warning("No stocks detected in current wave radar settings")
    
    # Tab 3: Analysis
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        
        analysis_tabs = st.tabs(["Sector Analysis", "Pattern Analysis", "Industry Analysis"])
        
        # Sector Analysis
        with analysis_tabs[0]:
            st.markdown("#### 🏢 Sector Performance Analysis")
            
            if not filtered_df.empty and 'sector' in filtered_df.columns:
                # Sector statistics
                sector_stats = filtered_df.groupby('sector').agg({
                    'master_score': ['mean', 'count'],
                    'ret_1d': 'mean',
                    'ret_30d': 'mean',
                    'rvol': 'mean',
                    'wave_strength': 'mean'
                }).round(1)
                
                sector_stats.columns = ['Avg Score', 'Count', 'Avg 1D %', 'Avg 30D %', 'Avg RVOL', 'Avg Wave']
                sector_stats = sector_stats.sort_values('Avg Score', ascending=False)
                
                # Display sector table
                st.dataframe(
                    sector_stats.style.format({
                        'Avg Score': '{:.1f}',
                        'Avg 1D %': '{:.1f}%',
                        'Avg 30D %': '{:.1f}%',
                        'Avg RVOL': '{:.1f}x',
                        'Avg Wave': '{:.0f}%'
                    }),
                    use_container_width=True
                )
                
                # Sector rotation analysis
                st.markdown("---")
                st.markdown("##### 🔄 Sector Rotation Signals")
                
                rotation_cols = st.columns(2)
                
                with rotation_cols[0]:
                    st.markdown("**📈 Improving Sectors (30D momentum)**")
                    improving = sector_stats[sector_stats['Avg 30D %'] > 10].sort_values('Avg 30D %', ascending=False)
                    for sector, row in improving.iterrows():
                        st.markdown(f"• **{sector}**: +{row['Avg 30D %']:.1f}% ({int(row['Count'])} stocks)")
                
                with rotation_cols[1]:
                    st.markdown("**📉 Weakening Sectors**")
                    weakening = sector_stats[sector_stats['Avg 30D %'] < 0].sort_values('Avg 30D %')
                    for sector, row in weakening.iterrows():
                        st.markdown(f"• **{sector}**: {row['Avg 30D %']:.1f}% ({int(row['Count'])} stocks)")
            else:
                st.info("No sector data available")
        
        # Pattern Analysis
        with analysis_tabs[1]:
            st.markdown("#### 🎯 Pattern Performance Analysis")
            
            if 'patterns' in filtered_df.columns:
                # Extract all patterns
                all_patterns = []
                for patterns in filtered_df['patterns']:
                    if patterns:
                        all_patterns.extend([p.strip() for p in patterns.split('|')])
                
                if all_patterns:
                    pattern_counts = Counter(all_patterns)
                    
                    # Pattern statistics
                    pattern_data = []
                    for pattern, count in pattern_counts.most_common(20):
                        # Get stocks with this pattern
                        pattern_stocks = filtered_df[filtered_df['patterns'].str.contains(pattern, na=False)]
                        
                        if not pattern_stocks.empty:
                            avg_score = pattern_stocks['master_score'].mean()
                            avg_ret_1d = pattern_stocks['ret_1d'].mean() if 'ret_1d' in pattern_stocks else 0
                            avg_ret_30d = pattern_stocks['ret_30d'].mean() if 'ret_30d' in pattern_stocks else 0
                            
                            # Get pattern info
                            pattern_info = CONFIG.PATTERN_INFO.get(pattern, {})
                            risk = pattern_info.get('risk', 'unknown')
                            timeframe = pattern_info.get('timeframe', 'unknown')
                            
                            pattern_data.append({
                                'Pattern': pattern,
                                'Count': count,
                                'Avg Score': avg_score,
                                'Avg 1D %': avg_ret_1d,
                                'Avg 30D %': avg_ret_30d,
                                'Risk': risk,
                                'Timeframe': timeframe
                            })
                    
                    if pattern_data:
                        pattern_df = pd.DataFrame(pattern_data)
                        
                        # Display pattern statistics
                        st.dataframe(
                            pattern_df.style.format({
                                'Avg Score': '{:.1f}',
                                'Avg 1D %': '{:.1f}%',
                                'Avg 30D %': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
                        
                        # Pattern insights
                        st.markdown("---")
                        st.markdown("##### 💡 Pattern Insights")
                        
                        insight_cols = st.columns(3)
                        
                        with insight_cols[0]:
                            st.markdown("**🔥 Hot Patterns (High frequency)**")
                            for _, row in pattern_df.head(5).iterrows():
                                st.markdown(f"• {row['Pattern']} ({row['Count']}x)")
                        
                        with insight_cols[1]:
                            st.markdown("**💎 Quality Patterns (High score)**")
                            quality_patterns = pattern_df.nlargest(5, 'Avg Score')
                            for _, row in quality_patterns.iterrows():
                                st.markdown(f"• {row['Pattern']} (Score: {row['Avg Score']:.0f})")
                        
                        with insight_cols[2]:
                            st.markdown("**🚀 Momentum Patterns (High returns)**")
                            momentum_patterns = pattern_df.nlargest(5, 'Avg 30D %')
                            for _, row in momentum_patterns.iterrows():
                                st.markdown(f"• {row['Pattern']} (+{row['Avg 30D %']:.1f}%)")
                else:
                    st.info("No patterns detected in current selection")
            else:
                st.info("Pattern data not available")
        
        # Industry Analysis
        with analysis_tabs[2]:
            st.markdown("#### 🏭 Industry Deep Dive")
            
            if 'industry' in filtered_df.columns:
                # Industry statistics
                industry_stats = filtered_df.groupby('industry').agg({
                    'ticker': 'count',
                    'master_score': 'mean',
                    'ret_30d': 'mean',
                    'wave_strength': 'mean'
                }).round(1)
                
                industry_stats.columns = ['Count', 'Avg Score', 'Avg 30D %', 'Avg Wave']
                
                # Show all industries in stats
                st.markdown("##### All Industries Overview")
                st.dataframe(
                    industry_stats.sort_values('Avg Score', ascending=False).style.format({
                        'Avg Score': '{:.1f}',
                        'Avg 30D %': '{:.1f}%',
                        'Avg Wave': '{:.0f}%'
                    }),
                    use_container_width=True
                )
                
                # Detailed analysis only for industries with 3+ stocks
                meaningful_industries = industry_stats[industry_stats['Count'] >= 3]
                
                if not meaningful_industries.empty:
                    st.markdown("---")
                    st.markdown("##### 🔍 Detailed Industry Analysis (3+ stocks)")
                    
                    # Select industry for deep dive
                    selected_industry = st.selectbox(
                        "Select industry for detailed view:",
                        meaningful_industries.index.tolist()
                    )
                    
                    if selected_industry:
                        industry_stocks = filtered_df[filtered_df['industry'] == selected_industry]
                        
                        ind_cols = st.columns(3)
                        
                        with ind_cols[0]:
                            st.metric("Stocks", len(industry_stocks))
                            st.metric("Avg Score", f"{industry_stocks['master_score'].mean():.1f}")
                        
                        with ind_cols[1]:
                            st.metric("Avg 30D Return", f"{industry_stocks['ret_30d'].mean():.1f}%")
                            if 'rvol' in industry_stocks:
                                st.metric("Avg RVOL", f"{industry_stocks['rvol'].mean():.1f}x")
                        
                        with ind_cols[2]:
                            gaining = (industry_stocks['ret_1d'] > 0).sum() if 'ret_1d' in industry_stocks else 0
                            st.metric("Gainers Today", f"{gaining}/{len(industry_stocks)}")
                            st.metric("Avg Wave", f"{industry_stocks['wave_strength'].mean():.0f}%")
                        
                        # Top stocks in industry
                        st.markdown(f"**Top 5 Stocks in {selected_industry}:**")
                        top_industry = industry_stocks.nlargest(5, 'master_score')[
                            ['ticker', 'company_name', 'master_score', 'ret_30d', 'wave_state']
                        ]
                        
                        for _, stock in top_industry.iterrows():
                            st.markdown(
                                f"• **{stock['ticker']}** - Score: {stock['master_score']:.0f}, "
                                f"30D: {stock['ret_30d']:+.1f}%, {stock['wave_state']}"
                            )
                else:
                    st.info("No industries with 3+ stocks for detailed analysis. See overview table above for all industries.")
            else:
                st.info("Industry data not available")
    
    # Tab 4: Search
    with tabs[4]:
        st.markdown("### 🔍 Stock Search")
        
        search_query = st.text_input(
            "Search by ticker or company name:",
            value=st.session_state.get('search_query', ''),
            placeholder="Enter ticker or company name...",
            key="search_query"
        )
        
        if search_query:
            # Search for stocks
            search_results = SearchEngine.search_stocks(ranked_df, search_query)
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} result(s) for '{search_query}'")
                
                # Display search results
                for idx, (_, stock) in enumerate(search_results.iterrows()):
                    with st.expander(f"**{stock['ticker']}** - {stock['company_name']}", expanded=(idx == 0)):
                        UIComponents.render_stock_card(stock, st.session_state.display_mode == 'hybrid')
                        
                        # Additional details
                        st.markdown("---")
                        detail_cols = st.columns(4)
                        
                        with detail_cols[0]:
                            st.markdown("**📊 Score Components**")
                            st.markdown(f"• Position: {stock.get('position_score', 0):.0f}")
                            st.markdown(f"• Volume: {stock.get('volume_score', 0):.0f}")
                            st.markdown(f"• Momentum: {stock.get('momentum_score', 0):.0f}")
                        
                        with detail_cols[1]:
                            st.markdown("**📈 Returns**")
                            st.markdown(f"• 7D: {stock.get('ret_7d', 0):+.1f}%")
                            st.markdown(f"• 30D: {stock.get('ret_30d', 0):+.1f}%")
                            st.markdown(f"• 3M: {stock.get('ret_3m', 0):+.1f}%")
                        
                        with detail_cols[2]:
                            st.markdown("**🎯 Position**")
                            st.markdown(f"• From Low: {stock.get('from_low_pct', 0):.0f}%")
                            st.markdown(f"• From High: {stock.get('from_high_pct', 0):.0f}%")
                            st.markdown(f"• Tension: {stock.get('position_tension', 0):.0f}")
                        
                        with detail_cols[3]:
                            st.markdown("**🌊 Wave Analysis**")
                            st.markdown(f"• State: {stock.get('wave_state', 'Unknown')}")
                            st.markdown(f"• Strength: {stock.get('wave_strength', 0):.0f}%")
                            st.markdown(f"• Harmony: {stock.get('momentum_harmony', 0)}/4")
            else:
                st.warning(f"No results found for '{search_query}'")
                st.info("💡 Try searching with partial ticker or company name")
        else:
            st.info("Enter a ticker symbol or company name to search")
            
            # Show some examples
            st.markdown("**Popular searches:**")
            example_cols = st.columns(5)
            
            examples = ["RELIANCE", "TCS", "HDFC", "INFY", "ICICI"]
            for i, example in enumerate(examples):
                with example_cols[i]:
                    if st.button(example, key=f"example_{example}"):
                        st.session_state.search_query = example
                        st.rerun()
    
    # Tab 5: Export
    with tabs[5]:
        st.markdown("### 📥 Data Export")
        st.markdown("*Export filtered data for further analysis*")
        
        export_cols = st.columns(2)
        
        with export_cols[0]:
            st.markdown("#### 📊 Export Options")
            
            # Export format
            export_format = st.radio(
                "Select export format:",
                ["CSV", "Excel"],
                help="CSV is faster, Excel includes formatting"
            )
            
            # Column selection
            st.markdown("**Select columns to export:**")
            
            available_columns = list(filtered_df.columns)
            default_columns = [
                'rank', 'ticker', 'company_name', 'category', 'sector',
                'price', 'ret_1d', 'ret_30d', 'master_score', 
                'rvol', 'money_flow_mm', 'wave_state', 'patterns'
            ]
            
            # Only include default columns that exist
            default_columns = [col for col in default_columns if col in available_columns]
            
            selected_columns = st.multiselect(
                "Columns to export:",
                available_columns,
                default=default_columns,
                help="Select the columns you want in your export"
            )
            
            # Export options
            include_all = st.checkbox(
                "Export all stocks (ignore current filters)",
                value=False,
                help="Check to export entire dataset instead of filtered view"
            )
            
        with export_cols[1]:
            st.markdown("#### 📋 Export Summary")
            
            export_df = ranked_df if include_all else filtered_df
            
            if selected_columns:
                export_df = export_df[selected_columns]
            
            # Summary stats
            st.info(f"**Rows to export:** {len(export_df):,}")
            st.info(f"**Columns selected:** {len(selected_columns)}")
            
            if not export_df.empty:
                # File size estimate
                if export_format == "CSV":
                    estimated_size = len(export_df) * len(selected_columns) * 50 / 1024  # Rough estimate in KB
                    st.info(f"**Estimated size:** ~{estimated_size:.0f} KB")
                
                # Generate export
                if export_format == "CSV":
                    csv_data = export_to_csv(export_df)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"wave_detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:  # Excel
                    excel_data = export_to_excel(export_df)
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"wave_detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                # Quick export templates
                st.markdown("---")
                st.markdown("#### 🎯 Quick Export Templates")
                
                template_cols = st.columns(3)
                
                with template_cols[0]:
                    st.markdown("**Day Trading**")
                    if st.button("Export Day Trade List"):
                        day_trade_df = filtered_df[
                            (filtered_df['wave_state'] == "🌊🌊🌊 CRESTING") & 
                            (filtered_df['rvol'] > 2)
                        ][['ticker', 'price', 'ret_1d', 'rvol', 'wave_strength', 'patterns']]
                        
                        csv_data = export_to_csv(day_trade_df)
                        st.download_button(
                            label="📥 Download Day Trade List",
                            data=csv_data,
                            file_name=f"day_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                with template_cols[1]:
                    st.markdown("**Swing Trading**")
                    if st.button("Export Swing List"):
                        swing_df = filtered_df[
                            (filtered_df['wave_state'].isin(["🌊🌊 BUILDING", "🌊 FORMING"])) & 
                            (filtered_df['master_score'] >= 75)
                        ][['ticker', 'price', 'master_score', 'momentum_score', 'wave_state', 'patterns']]
                        
                        csv_data = export_to_csv(swing_df)
                        st.download_button(
                            label="📥 Download Swing List",
                            data=csv_data,
                            file_name=f"swing_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                with template_cols[2]:
                    st.markdown("**Investment**")
                    if st.button("Export Investment List"):
                        if st.session_state.display_mode == 'hybrid':
                            invest_df = filtered_df[
                                (filtered_df['master_score'] >= 80) & 
                                (filtered_df['pe'] > 0) & 
                                (filtered_df['pe'] < 30)
                            ][['ticker', 'price', 'pe', 'eps_change_pct', 'master_score', 'patterns']]
                        else:
                            invest_df = filtered_df[
                                filtered_df['master_score'] >= 80
                            ][['ticker', 'price', 'master_score', 'long_term_strength', 'patterns']]
                        
                        csv_data = export_to_csv(invest_df)
                        st.download_button(
                            label="📥 Download Investment List",
                            data=csv_data,
                            file_name=f"investments_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("No data to export")
    
    # Tab 6: About
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        about_cols = st.columns([2, 1])
        
        with about_cols[0]:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
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
            - **Wave Strength** - Signal-based momentum indicator (0-100%)
            - **Wave State** - Real-time momentum classification
            - **Smart Money Flow** - Institutional activity tracking
            - **Market Regime** - Overall market condition detection
            
            **25 Pattern Detection System**:
            - 11 Technical patterns
            - 5 Fundamental patterns (Hybrid mode)
            - 6 Price range patterns
            - 3 Intelligence patterns (STEALTH, VAMPIRE, PERFECT STORM)
            
            #### 💡 How to Use
            
            1. **Data Source** - Enter Google Sheets ID or upload CSV
            2. **Quick Actions** - Instant filtering for common scenarios
            3. **Smart Filters** - Interconnected filtering system
            4. **Display Modes** - Technical only or Hybrid (with fundamentals)
            5. **Wave Radar** - Monitor early momentum signals
            6. **Export Options** - Download filtered data for analysis
            
            #### 🔧 Technical Details
            
            - **Performance** - Handles 2000+ stocks in <2 seconds
            - **Accuracy** - Multiple validation layers ensure data quality
            - **Reliability** - Retry logic and fallback mechanisms
            - **Flexibility** - Works with any Google Sheets data source
            - **Mobile Ready** - Responsive design for all devices
            """)
        
        with about_cols[1]:
            st.markdown("""
            #### 📊 Data Processing
            
            1. **Load** - CSV/Sheets with validation
            2. **Clean** - Handle missing/invalid data
            3. **Calculate** - All scores and metrics
            4. **Detect** - Pattern recognition
            5. **Rank** - Final scoring
            6. **Filter** - Smart interconnection
            
            #### 🎨 Pattern Categories
            
            **Technical**
            - Momentum based
            - Volume based
            - Trend based
            
            **Fundamental**
            - Value focused
            - Growth focused
            - Quality focused
            
            **Range**
            - Position based
            - Breakout focused
            
            **Intelligence**
            - Algorithm driven
            - Multi-signal based
            
            #### 📈 Wave States
            
            - **🌊🌊🌊 CRESTING** - Peak momentum
            - **🌊🌊 BUILDING** - Growing strength
            - **🌊 FORMING** - Early signals
            - **💥 BREAKING** - Momentum lost
            
            #### 🏆 Best Practices
            
            1. Check daily at market open
            2. Focus on your strategy
            3. Use appropriate filters
            4. Monitor wave states
            5. Export for tracking
            """)
        
        # Version info
        st.markdown("---")
        version_cols = st.columns(3)
        
        with version_cols[0]:
            st.metric("Version", "3.0 Final")
        
        with version_cols[1]:
            if st.session_state.get('last_data_update'):
                update_time = st.session_state.last_data_update
                st.metric("Data Updated", update_time.strftime("%H:%M:%S"))
        
        with version_cols[2]:
            st.metric("Stocks Loaded", f"{len(ranked_df):,}")
        
        # Debug info
        if st.session_state.get('show_debug'):
            st.markdown("---")
            st.markdown("#### 🐛 Debug Information")
            
            debug_cols = st.columns(2)
            
            with debug_cols[0]:
                st.json({
                    "session_id": st.session_state.get('session_id', 'Unknown'),
                    "data_source": st.session_state.data_source,
                    "display_mode": st.session_state.display_mode,
                    "active_filters": st.session_state.active_filter_count,
                    "filtered_stocks": len(filtered_df),
                    "total_stocks": len(ranked_df)
                })
            
            with debug_cols[1]:
                if 'performance_metrics' in st.session_state:
                    st.markdown("**Performance Metrics:**")
                    for func, time_taken in st.session_state.performance_metrics.items():
                        st.markdown(f"• {func}: {time_taken:.3f}s")

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
