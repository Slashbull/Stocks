"""
Wave Detection Ultimate 3.0 - FINAL ENHANCED VERSION
===================================================
Professional Stock Ranking System with Advanced Analytics
Optimized for Streamlit Community Cloud - Production Ready

Version: 3.0.6-FINAL-ENHANCED
Status: PERMANENT PRODUCTION VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from functools import lru_cache
import time
from io import BytesIO
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# STREAMLIT CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Wave Detection Ultimate 3.0",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
@dataclass(frozen=True)
class Config:
    """Immutable system configuration"""
    
    # Data source settings
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM/edit?usp=sharing"
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    
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
    
    # Performance thresholds
    MIN_VALID_PRICE: float = 0.01
    RVOL_MAX_THRESHOLD: float = 1000000.0
    MIN_DATA_COMPLETENESS: float = 0.8
    STALE_DATA_HOURS: int = 24
    
    # Quick action thresholds
    TOP_GAINER_MOMENTUM: float = 80
    VOLUME_SURGE_RVOL: float = 3.0
    BREAKOUT_READY_SCORE: float = 80

# Global configuration instance
CONFIG = Config()

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
/* Main layout optimization */
.main {padding: 0rem 1rem;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}

/* Metric styling */
div[data-testid="metric-container"] {
    background-color: rgba(28, 131, 225, 0.1);
    border: 1px solid rgba(28, 131, 225, 0.2);
    padding: 5% 5% 5% 10%;
    border-radius: 5px;
    overflow-wrap: break-word;
}

/* Alert styling */
.stAlert {
    padding: 1rem;
    border-radius: 5px;
}

/* Button styling */
div.stButton > button {
    width: 100%;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .stDataFrame {font-size: 12px;}
    div[data-testid="metric-container"] {padding: 3%;}
}

/* Table overflow fix */
.stDataFrame > div {
    overflow-x: auto;
}

/* Performance optimization */
.element-container {
    overflow: hidden;
}

/* Smooth transitions */
* {
    transition: background-color 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
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

# ============================================
# DATA VALIDATION
# ============================================
class DataValidator:
    """Validate and clean data efficiently"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, context: str) -> bool:
        """Quick validation of dataframe"""
        if df is None or df.empty:
            logger.error(f"{context}: Empty dataframe")
            return False
        logger.info(f"{context}: {len(df)} rows, {len(df.columns)} columns")
        return True
    
    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False) -> Optional[float]:
        """Clean Indian number format to float - optimized"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            # Quick type check
            if isinstance(value, (int, float)):
                return float(value)
            
            # String cleaning
            cleaned = str(value).strip()
            if cleaned in ['', '-', 'N/A', 'nan', 'None', '#VALUE!']:
                return np.nan
            
            # Remove formatting
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            
            return float(cleaned)
        except:
            return np.nan

# ============================================
# DATA PROCESSING - OPTIMIZED
# ============================================
class DataProcessor:
    """Optimized data processing"""
    
    # Column definitions
    NUMERIC_COLUMNS = [
        'price', 'prev_close', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
        'sma_20d', 'sma_50d', 'sma_200d',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
        'rvol', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'
    ]
    
    CATEGORICAL_COLUMNS = ['ticker', 'company_name', 'category', 'sector']
    
    PERCENTAGE_COLUMNS = [
        'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_3d', 'ret_7d', 
        'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'eps_change_pct'
    ]
    
    VOLUME_RATIO_COLUMNS = [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d'
    ]
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe with optimized operations"""
        if not DataValidator.validate_dataframe(df, "Processing"):
            return pd.DataFrame()
        
        # Work on copy
        df = df.copy()
        
        # Vectorized numeric cleaning
        for col in DataProcessor.NUMERIC_COLUMNS:
            if col in df.columns:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Special handling for volume ratios
                if col in DataProcessor.VOLUME_RATIO_COLUMNS:
                    # Convert percentage change to ratio
                    df[col] = ((100 + df[col]) / 100).fillna(1.0).clip(0.01, 100.0)
        
        # Vectorized categorical cleaning
        for col in DataProcessor.CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', '', 'N/A', 'NaN'], 'Unknown')
                # Remove extra whitespace
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Remove invalid rows
        df = df.dropna(subset=['ticker', 'price'])
        df = df[df['price'] > CONFIG.MIN_VALID_PRICE]
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        # Fill missing values efficiently
        df['from_low_pct'] = df.get('from_low_pct', pd.Series(50, index=df.index)).fillna(50)
        df['from_high_pct'] = df.get('from_high_pct', pd.Series(-50, index=df.index)).fillna(-50)
        df['rvol'] = df.get('rvol', pd.Series(1.0, index=df.index)).fillna(1.0).clip(lower=0.01)
        
        # Add tier classifications
        df = DataProcessor._add_tiers_vectorized(df)
        
        logger.info(f"Processed {len(df)} valid stocks")
        return df
    
    @staticmethod
    def _add_tiers_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications using vectorized operations"""
        # EPS tiers
        if 'eps_current' in df.columns:
            df['eps_tier'] = pd.cut(
                df['eps_current'],
                bins=[-np.inf, 0, 5, 10, 20, 50, 100, np.inf],
                labels=['Loss', '0-5', '5-10', '10-20', '20-50', '50-100', '100+'],
                include_lowest=True
            )
        else:
            df['eps_tier'] = 'Unknown'
        
        # PE tiers
        if 'pe' in df.columns:
            pe_series = df['pe'].copy()
            pe_series = pe_series.where(pe_series > 0, -1)
            df['pe_tier'] = pd.cut(
                pe_series,
                bins=[-np.inf, 0, 10, 15, 20, 30, 50, np.inf],
                labels=['Negative/NA', '0-10', '10-15', '15-20', '20-30', '30-50', '50+'],
                include_lowest=True
            )
        else:
            df['pe_tier'] = 'Unknown'
        
        # Price tiers
        if 'price' in df.columns:
            df['price_tier'] = pd.cut(
                df['price'],
                bins=[0, 100, 250, 500, 1000, 2500, 5000, np.inf],
                labels=['0-100', '100-250', '250-500', '500-1000', '1000-2500', '2500-5000', '5000+'],
                include_lowest=True
            )
        else:
            df['price_tier'] = 'Unknown'
        
        return df

# ============================================
# RANKING ENGINE - VECTORIZED
# ============================================
class RankingEngine:
    """Fully vectorized ranking calculations"""
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safely rank a series with proper handling"""
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        # Replace inf values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # For percentage ranks
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom') * 100
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
        
        # Fill NaN values
        if pct:
            ranks = ranks.fillna(0 if ascending else 100)
        else:
            ranks = ranks.fillna(series.notna().sum() + 1)
        
        return ranks
    
    @staticmethod
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all scores in one pass - fully vectorized"""
        if df.empty:
            return df
        
        # Position Score (30%)
        from_low = df['from_low_pct'].fillna(50)
        from_high = df['from_high_pct'].fillna(-50)
        
        rank_from_low = RankingEngine.safe_rank(from_low, pct=True, ascending=True)
        rank_from_high = RankingEngine.safe_rank(from_high, pct=True, ascending=False)
        
        df['position_score'] = (rank_from_low * 0.6 + rank_from_high * 0.4).clip(0, 100)
        
        # Volume Score (25%)
        vol_components = []
        vol_weights = []
        
        vol_mappings = [
            ('vol_ratio_1d_90d', 0.20),
            ('vol_ratio_7d_90d', 0.20),
            ('vol_ratio_30d_90d', 0.20),
            ('vol_ratio_30d_180d', 0.15),
            ('vol_ratio_90d_180d', 0.25)
        ]
        
        for col, weight in vol_mappings:
            if col in df.columns:
                data = df[col].fillna(1.0).clip(lower=0.1)
                vol_components.append(RankingEngine.safe_rank(data, pct=True, ascending=True) * weight)
                vol_weights.append(weight)
        
        if vol_components:
            df['volume_score'] = sum(vol_components) / sum(vol_weights)
        else:
            df['volume_score'] = 50
        
        # Momentum Score (15%)
        if 'ret_30d' in df.columns:
            df['momentum_score'] = RankingEngine.safe_rank(df['ret_30d'].fillna(0), pct=True, ascending=True)
        elif 'ret_7d' in df.columns:
            df['momentum_score'] = RankingEngine.safe_rank(df['ret_7d'].fillna(0), pct=True, ascending=True)
        else:
            df['momentum_score'] = 50
        
        # Acceleration Score (10%) - Vectorized
        df['acceleration_score'] = 50  # Default
        
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            # Calculate daily averages
            avg_1d = df['ret_1d'].fillna(0)
            avg_7d = df['ret_7d'].fillna(0) / 7
            avg_30d = df['ret_30d'].fillna(0) / 30
            
            # Vectorized conditions
            perfect = (avg_1d > avg_7d) & (avg_7d > avg_30d) & (df['ret_1d'] > 0)
            good = (~perfect) & (avg_1d > avg_7d) & (df['ret_1d'] > 0)
            moderate = (~perfect) & (~good) & (df['ret_1d'] > 0)
            slight_decel = (df['ret_1d'] <= 0) & (df['ret_7d'] > 0)
            
            # Apply scores
            df.loc[perfect, 'acceleration_score'] = 100
            df.loc[good, 'acceleration_score'] = 80
            df.loc[moderate, 'acceleration_score'] = 60
            df.loc[slight_decel, 'acceleration_score'] = 40
            df.loc[~(perfect | good | moderate | slight_decel), 'acceleration_score'] = 20
        
        # Breakout Score (10%) - Vectorized
        distance_from_high = (-df['from_high_pct'].fillna(-50)).clip(0, 100)
        distance_factor = (100 - distance_from_high).clip(0, 100)
        
        volume_factor = 50
        if 'vol_ratio_7d_90d' in df.columns:
            vol_ratio = df['vol_ratio_7d_90d'].fillna(1.0)
            volume_factor = ((vol_ratio - 1) * 100).clip(0, 100)
        
        trend_factor = 0
        trend_count = 0
        
        if 'price' in df.columns:
            current_price = df['price']
            for sma_col in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma_col in df.columns:
                    trend_factor += (current_price > df[sma_col]).astype(float) * 33.33
                    trend_count += 1
        
        if trend_count > 0:
            trend_factor = trend_factor.clip(0, 100)
        
        df['breakout_score'] = (
            distance_factor * 0.4 +
            volume_factor * 0.4 +
            trend_factor * 0.2
        ).clip(0, 100)
        
        # RVOL Score (10%) - Vectorized
        if 'rvol' in df.columns:
            rvol = df['rvol'].fillna(1.0)
            
            # Vectorized scoring
            df['rvol_score'] = 50  # Default
            
            # Apply score ranges
            df.loc[rvol > 1000, 'rvol_score'] = 100
            df.loc[(rvol > 100) & (rvol <= 1000), 'rvol_score'] = 95
            df.loc[(rvol > 10) & (rvol <= 100), 'rvol_score'] = 90
            df.loc[(rvol > 5) & (rvol <= 10), 'rvol_score'] = 85
            df.loc[(rvol > 3) & (rvol <= 5), 'rvol_score'] = 80
            df.loc[(rvol > 2) & (rvol <= 3), 'rvol_score'] = 75
            df.loc[(rvol > 1.5) & (rvol <= 2), 'rvol_score'] = 70
            df.loc[(rvol > 1.2) & (rvol <= 1.5), 'rvol_score'] = 60
            df.loc[(rvol > 0.8) & (rvol <= 1.2), 'rvol_score'] = 50
            df.loc[(rvol > 0.5) & (rvol <= 0.8), 'rvol_score'] = 40
            df.loc[(rvol > 0.3) & (rvol <= 0.5), 'rvol_score'] = 30
            df.loc[rvol <= 0.3, 'rvol_score'] = 20
        else:
            df['rvol_score'] = 50
        
        # Calculate Master Score
        df['master_score'] = (
            df['position_score'] * CONFIG.POSITION_WEIGHT +
            df['volume_score'] * CONFIG.VOLUME_WEIGHT +
            df['momentum_score'] * CONFIG.MOMENTUM_WEIGHT +
            df['acceleration_score'] * CONFIG.ACCELERATION_WEIGHT +
            df['breakout_score'] * CONFIG.BREAKOUT_WEIGHT +
            df['rvol_score'] * CONFIG.RVOL_WEIGHT
        ).clip(0, 100)
        
        # Calculate ranks
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').fillna(len(df) + 1).astype(int)
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom').fillna(0) * 100
        
        # Calculate auxiliary scores
        df = RankingEngine._calculate_auxiliary_scores(df)
        
        # Category ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        # Pattern detection
        df = RankingEngine._detect_patterns_vectorized(df)
        
        return df
    
    @staticmethod
    def _calculate_auxiliary_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend quality, long-term strength, and liquidity scores"""
        # Trend Quality Score
        df['trend_quality'] = 50
        
        if 'price' in df.columns:
            current_price = df['price']
            
            # Count SMAs price is above
            above_count = pd.Series(0, index=df.index)
            for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
                if sma in df.columns:
                    above_count += (current_price > df[sma]).astype(int)
            
            # Perfect alignment check
            if all(sma in df.columns for sma in ['sma_20d', 'sma_50d', 'sma_200d']):
                perfect = (current_price > df['sma_20d']) & (df['sma_20d'] > df['sma_50d']) & (df['sma_50d'] > df['sma_200d'])
                df.loc[perfect, 'trend_quality'] = 100
                
                strong = (~perfect) & (above_count == 3)
                df.loc[strong, 'trend_quality'] = 85
            
            # Set scores based on count
            df.loc[above_count == 2, 'trend_quality'] = 70
            df.loc[above_count == 1, 'trend_quality'] = 40
            df.loc[above_count == 0, 'trend_quality'] = 20
        
        # Long-term Strength Score
        df['long_term_strength'] = 50
        
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_lt = [col for col in lt_cols if col in df.columns]
        
        if available_lt:
            avg_return = df[available_lt].fillna(0).mean(axis=1)
            
            # Vectorized scoring
            df.loc[avg_return > 100, 'long_term_strength'] = 100
            df.loc[(avg_return > 50) & (avg_return <= 100), 'long_term_strength'] = 90
            df.loc[(avg_return > 30) & (avg_return <= 50), 'long_term_strength'] = 80
            df.loc[(avg_return > 15) & (avg_return <= 30), 'long_term_strength'] = 70
            df.loc[(avg_return > 5) & (avg_return <= 15), 'long_term_strength'] = 60
            df.loc[(avg_return > 0) & (avg_return <= 5), 'long_term_strength'] = 50
            df.loc[(avg_return > -10) & (avg_return <= 0), 'long_term_strength'] = 40
            df.loc[(avg_return > -25) & (avg_return <= -10), 'long_term_strength'] = 30
            df.loc[avg_return <= -25, 'long_term_strength'] = 20
        
        # Liquidity Score
        df['liquidity_score'] = 50
        
        if 'volume_30d' in df.columns and 'price' in df.columns:
            dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
            df['liquidity_score'] = RankingEngine.safe_rank(dollar_volume, pct=True, ascending=True)
        
        return df
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate category-specific ranks efficiently"""
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        # Group operation
        for category, group in df.groupby('category'):
            if category != 'Unknown' and len(group) > 0:
                cat_ranks = group['master_score'].rank(method='first', ascending=False, na_option='bottom')
                cat_percentiles = group['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
                
                df.loc[group.index, 'category_rank'] = cat_ranks.astype(int)
                df.loc[group.index, 'category_percentile'] = cat_percentiles
        
        return df
    
    @staticmethod
    def _detect_patterns_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns using vectorized operations"""
        # Initialize patterns list for each row
        patterns = [[] for _ in range(len(df))]
        
        # Pattern 1: Category Leader
        if 'category_percentile' in df.columns:
            mask = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
            for idx in df[mask].index:
                patterns[idx].append('🔥 CAT LEADER')
        
        # Pattern 2: Hidden Gem
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            mask = (df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (df['percentile'] < 70)
            for idx in df[mask].index:
                patterns[idx].append('💎 HIDDEN GEM')
        
        # Pattern 3: Accelerating
        if 'acceleration_score' in df.columns:
            mask = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
            for idx in df[mask].index:
                patterns[idx].append('🚀 ACCELERATING')
        
        # Pattern 4: Institutional
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            mask = (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (df['vol_ratio_90d_180d'] > 1.1)
            for idx in df[mask].index:
                patterns[idx].append('🏦 INSTITUTIONAL')
        
        # Pattern 5: Volume Explosion
        if 'rvol' in df.columns:
            mask = df['rvol'] > 3
            for idx in df[mask].index:
                patterns[idx].append('⚡ VOL EXPLOSION')
        
        # Pattern 6: Breakout Ready
        if 'breakout_score' in df.columns:
            mask = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
            for idx in df[mask].index:
                patterns[idx].append('🎯 BREAKOUT')
        
        # Pattern 7: Market Leader
        if 'percentile' in df.columns:
            mask = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
            for idx in df[mask].index:
                patterns[idx].append('👑 MARKET LEADER')
        
        # Pattern 8: Momentum Wave
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            mask = (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (df['acceleration_score'] >= 70)
            for idx in df[mask].index:
                patterns[idx].append('🌊 MOMENTUM WAVE')
        
        # Pattern 9: Liquid Leader
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            mask = (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
            for idx in df[mask].index:
                patterns[idx].append('💰 LIQUID LEADER')
        
        # Pattern 10: Long-term Strength
        if 'long_term_strength' in df.columns:
            mask = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            for idx in df[mask].index:
                patterns[idx].append('💪 LONG STRENGTH')
        
        # Pattern 11: Quality Trend
        if 'trend_quality' in df.columns:
            mask = df['trend_quality'] >= 80
            for idx in df[mask].index:
                patterns[idx].append('📈 QUALITY TREND')
        
        # Pattern 12: Value Momentum
        if 'pe' in df.columns and 'percentile' in df.columns:
            mask = (df['pe'].notna()) & (df['pe'] > 0) & (df['pe'] < 15) & (df['master_score'] >= 70)
            for idx in df[mask].index:
                patterns[idx].append('💎 VALUE MOMENTUM')
        
        # Pattern 13: Earnings Rocket
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            mask1 = (df['eps_change_pct'].notna()) & (df['eps_change_pct'] > 1000) & (df['acceleration_score'] >= 80)
            mask2 = (df['eps_change_pct'].notna()) & (df['eps_change_pct'] > 50) & (df['eps_change_pct'] <= 1000) & (df['acceleration_score'] >= 70)
            for idx in df[mask1 | mask2].index:
                patterns[idx].append('📊 EARNINGS ROCKET')
        
        # Pattern 14: Quality Leader
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            mask = (df['pe'].notna()) & (df['pe'] > 10) & (df['pe'] <= 25) & (df['eps_change_pct'] > 20) & (df['percentile'] >= 80)
            for idx in df[mask].index:
                patterns[idx].append('🏆 QUALITY LEADER')
        
        # Pattern 15: Turnaround Play
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            mask1 = (df['eps_change_pct'].notna()) & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60)
            mask2 = (df['eps_change_pct'].notna()) & (df['eps_change_pct'] > 100) & (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70)
            for idx in df[mask1 | mask2].index:
                patterns[idx].append('⚡ TURNAROUND')
        
        # Pattern 16: Overvalued Warning
        if 'pe' in df.columns:
            mask = (df['pe'].notna()) & (df['pe'] > 100)
            for idx in df[mask].index:
                patterns[idx].append('⚠️ HIGH PE')
        
        # New Pattern 17: 52W High Approach
        if 'from_high_pct' in df.columns and 'volume_score' in df.columns and 'momentum_score' in df.columns:
            mask = (df['from_high_pct'] > -5) & (df['volume_score'] >= 70) & (df['momentum_score'] >= 60)
            for idx in df[mask].index:
                patterns[idx].append('🎯 52W HIGH APPROACH')
        
        # New Pattern 18: 52W Low Bounce
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            mask = (df['from_low_pct'] < 20) & (df['acceleration_score'] >= 80) & (df['ret_30d'] > 10)
            for idx in df[mask].index:
                patterns[idx].append('🔄 52W LOW BOUNCE')
        
        # New Pattern 19: Golden Zone
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            mask = (df['from_low_pct'] > 60) & (df['from_high_pct'] > -40) & (df['trend_quality'] >= 70)
            for idx in df[mask].index:
                patterns[idx].append('👑 GOLDEN ZONE')
        
        # New Pattern 20: Volume Accumulation
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            mask = (df['vol_ratio_30d_90d'] > 1.2) & (df['vol_ratio_90d_180d'] > 1.1) & (df['ret_30d'] > 5)
            for idx in df[mask].index:
                patterns[idx].append('📊 VOL ACCUMULATION')
        
        # New Pattern 21: Momentum Divergence
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = df['ret_7d'] / 7
                daily_30d_pace = df['ret_30d'] / 30
                
            mask = (daily_7d_pace > daily_30d_pace * 1.5) & (df['acceleration_score'] >= 85) & (df['rvol'] > 2)
            mask = mask.fillna(False)
            for idx in df[mask].index:
                patterns[idx].append('🔀 MOMENTUM DIVERGE')
        
        # New Pattern 22: Range Compression
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = ((df['high_52w'] - df['low_52w']) / df['low_52w']) * 100
                range_pct = range_pct.fillna(100)
                
            mask = (range_pct < 50) & (df['from_low_pct'] > 30)
            for idx in df[mask].index:
                patterns[idx].append('🎯 RANGE COMPRESS')
        
        # Convert patterns list to string
        df['patterns'] = [' | '.join(p) if p else '' for p in patterns]
        
        return df

# ============================================
# DATA LOADING - FLEXIBLE
# ============================================
@st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False)
def load_data(source_type: str = "sheet", file_data=None, sheet_url: str = None, gid: str = None) -> Tuple[pd.DataFrame, datetime]:
    """Load data from either Google Sheets or uploaded CSV"""
    try:
        if source_type == "upload" and file_data is not None:
            # Load from uploaded file
            df = pd.read_csv(file_data)
            logger.info(f"Loaded {len(df)} rows from uploaded CSV")
        else:
            # Load from Google Sheets
            if not sheet_url or not gid:
                sheet_url = CONFIG.DEFAULT_SHEET_URL
                gid = CONFIG.DEFAULT_GID
            
            base_url = sheet_url.split('/edit')[0]
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            df = pd.read_csv(csv_url, low_memory=False)
            logger.info(f"Loaded {len(df)} rows from Google Sheets")
        
        # Process data
        df = DataProcessor.process_dataframe(df)
        
        # Calculate rankings
        df = RankingEngine.calculate_all_scores(df)
        
        return df, datetime.now()
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

# ============================================
# FILTER ENGINE
# ============================================
class FilterEngine:
    """Optimized filtering operations"""
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column: str, filters: Dict = None) -> List[str]:
        """Get unique values with smart filtering"""
        if df.empty or column not in df.columns:
            return []
        
        # Apply existing filters for interconnected behavior
        if filters:
            temp_df = FilterEngine.apply_filters(df, {k: v for k, v in filters.items() if k != column})
        else:
            temp_df = df
        
        values = temp_df[column].dropna().unique().tolist()
        return sorted([str(v) for v in values if str(v) not in ['Unknown', 'nan', '']])
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters efficiently"""
        if df.empty:
            return df
        
        filtered_df = df
        
        # Category filter
        if filters.get('categories'):
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # Sector filter
        if filters.get('sectors'):
            filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
        
        # Score filter
        if filters.get('min_score', 0) > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= filters['min_score']]
        
        # Pattern filter
        if filters.get('patterns'):
            pattern_regex = '|'.join(filters['patterns'])
            filtered_df = filtered_df[filtered_df['patterns'].str.contains(pattern_regex, case=False, na=False)]
        
        # Trend filter
        if filters.get('trend_range') and 'trend_quality' in filtered_df.columns:
            min_trend, max_trend = filters['trend_range']
            filtered_df = filtered_df[(filtered_df['trend_quality'] >= min_trend) & 
                                    (filtered_df['trend_quality'] <= max_trend)]
        
        # EPS change filter
        if filters.get('min_eps_change') is not None and 'eps_change_pct' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['eps_change_pct'] >= filters['min_eps_change']) | 
                                    (filtered_df['eps_change_pct'].isna())]
        
        # PE filters
        if filters.get('min_pe') is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['pe'].isna()) | 
                                    ((filtered_df['pe'] >= filters['min_pe']) & (filtered_df['pe'] < np.inf))]
        
        if filters.get('max_pe') is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['pe'].isna()) | 
                                    ((filtered_df['pe'] <= filters['max_pe']) & (filtered_df['pe'] > 0))]
        
        # Tier filters
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            if filters.get(tier_type):
                col_name = tier_type.replace('_tiers', '_tier')
                if col_name in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[col_name].isin(filters[tier_type])]
        
        # Require fundamental data filter
        if filters.get('require_fundamental_data', False):
            if 'pe' in filtered_df.columns and 'eps_change_pct' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['pe'].notna() & 
                    (filtered_df['pe'] > 0) &
                    ~np.isinf(filtered_df['pe']) &
                    filtered_df['eps_change_pct'].notna() &
                    ~np.isinf(filtered_df['eps_change_pct'])
                ]
        
        return filtered_df

# ============================================
# SEARCH ENGINE
# ============================================
class SearchEngine:
    """Fast stock search"""
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks by ticker or name"""
        if not query or df.empty:
            return pd.DataFrame()
        
        query = query.upper().strip()
        
        # Exact ticker match
        exact = df[df['ticker'].str.upper() == query]
        if not exact.empty:
            return exact
        
        # Contains search
        ticker_match = df[df['ticker'].str.upper().str.contains(query, na=False)]
        name_match = df[df['company_name'].str.upper().str.contains(query, na=False)]
        
        return pd.concat([ticker_match, name_match]).drop_duplicates()

# ============================================
# VISUALIZATION ENGINE
# ============================================
class Visualizer:
    """Create optimized visualizations"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Score distribution box plot"""
        fig = go.Figure()
        
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
                fig.add_trace(go.Box(
                    y=df[score_col],
                    name=label,
                    marker_color=color,
                    boxpoints='outliers'
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
        """Create top stocks breakdown chart"""
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
    def create_pattern_analysis(df: pd.DataFrame) -> go.Figure:
        """Pattern frequency chart"""
        all_patterns = []
        
        for patterns in df['patterns'].dropna():
            if patterns:
                all_patterns.extend(patterns.split(' | '))
        
        if not all_patterns:
            fig = go.Figure()
            fig.add_annotation(text="No patterns detected", x=0.5, y=0.5)
            return fig
        
        pattern_counts = pd.Series(all_patterns).value_counts()
        
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
            template='plotly_white',
            height=max(400, len(pattern_counts) * 30),
            margin=dict(l=150)
        )
        
        return fig

# ============================================
# EXPORT ENGINE
# ============================================
class ExportEngine:
    """Handle exports efficiently"""
    
    @staticmethod
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        """Create Excel report with multiple sheets"""
        output = BytesIO()
        
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
            'full': None
        }
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#3498db',
                'font_color': 'white',
                'border': 1
            })
            
            # Top 100 sheet
            top_100 = df.nlargest(min(100, len(df)), 'master_score')
            
            if template in templates and templates[template]:
                cols = [c for c in templates[template] if c in top_100.columns]
            else:
                cols = ['rank', 'ticker', 'company_name', 'master_score',
                       'position_score', 'volume_score', 'momentum_score',
                       'acceleration_score', 'breakout_score', 'rvol_score',
                       'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
                       'from_low_pct', 'from_high_pct',
                       'ret_1d', 'ret_7d', 'ret_30d', 'rvol',
                       'patterns', 'category', 'sector']
                cols = [c for c in cols if c in top_100.columns]
            
            export_df = top_100[cols].copy()
            export_df.to_excel(writer, sheet_name='Top 100', index=False)
            
            # Format the sheet
            worksheet = writer.sheets['Top 100']
            for i, col in enumerate(cols):
                worksheet.write(0, i, col, header_format)
            
            # All stocks summary
            summary_cols = ['rank', 'ticker', 'company_name', 'master_score', 'price', 
                          'ret_30d', 'rvol', 'patterns', 'category', 'sector']
            available_summary = [c for c in summary_cols if c in df.columns]
            df[available_summary].to_excel(writer, sheet_name='All Stocks', index=False)
            
            # Sector analysis
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
            
            # Category analysis
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
            
            # Pattern analysis
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
                pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
            
            # Wave Radar signals
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
                
                momentum_shifts[available_wave_cols].to_excel(
                    writer, sheet_name='Wave Radar Signals', index=False
                )
            
            logger.info("Excel report created successfully")
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export"""
        export_cols = ['rank', 'ticker', 'company_name', 'master_score',
                      'position_score', 'volume_score', 'momentum_score',
                      'acceleration_score', 'breakout_score', 'rvol_score',
                      'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
                      'from_low_pct', 'from_high_pct',
                      'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
                      'rvol', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d',
                      'patterns', 'category', 'sector', 'eps_tier', 'pe_tier']
        
        available = [c for c in export_cols if c in df.columns]
        
        # Create a copy for export
        export_df = df[available].copy()
        
        # Volume ratios need to be converted back to percentage for display
        vol_ratio_cols = [col for col in export_df.columns if 'vol_ratio' in col]
        for col in vol_ratio_cols:
            export_df[col] = (export_df[col] - 1) * 100  # Convert back to percentage change
        
        return export_df.to_csv(index=False)

# ============================================
# FORMATTERS
# ============================================
def format_pe(value):
    """Format PE ratio for display"""
    try:
        if pd.isna(value) or value == 'N/A':
            return '-'
        val = float(value)
        if val <= 0:
            return 'Loss'
        elif np.isinf(val):
            return '∞'
        elif val > 10000:
            return f"{val/1000:.0f}K"
        elif val > 100:
            return f"{val:.0f}"
        else:
            return f"{val:.1f}"
    except:
        return '-'

def format_eps_change(value):
    """Format EPS change percentage"""
    try:
        if pd.isna(value):
            return '-'
        val = float(value)
        if np.isinf(val):
            return '∞' if val > 0 else '-∞'
        if abs(val) >= 10000:
            return f"{val/1000:+.1f}K%"
        elif abs(val) >= 1000:
            return f"{val:+.0f}%"
        else:
            return f"{val:+.1f}%"
    except:
        return '-'

# ============================================
# QUICK STATS
# ============================================
def calculate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate data quality metrics"""
    quality = {}
    
    # Completeness
    total_cells = len(df) * len(df.columns)
    filled_cells = df.notna().sum().sum()
    quality['completeness'] = (filled_cells / total_cells * 100) if total_cells > 0 else 0
    
    # Freshness
    if 'price' in df.columns and 'prev_close' in df.columns:
        unchanged = (df['price'] == df['prev_close']).sum()
        quality['freshness'] = ((len(df) - unchanged) / len(df) * 100) if len(df) > 0 else 0
    else:
        quality['freshness'] = 0
    
    # Coverage
    if 'pe' in df.columns:
        quality['pe_coverage'] = (df['pe'].notna() & (df['pe'] > 0)).sum()
    else:
        quality['pe_coverage'] = 0
        
    if 'eps_change_pct' in df.columns:
        quality['eps_coverage'] = df['eps_change_pct'].notna().sum()
    else:
        quality['eps_coverage'] = 0
    
    return quality

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application with all features"""
    
    # Initialize session state
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "sheet"
    if 'display_mode' not in st.session_state:
        st.session_state.display_mode = "Technical"
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'quick_filter' not in st.session_state:
        st.session_state.quick_filter = None
    if 'quick_filter_applied' not in st.session_state:
        st.session_state.quick_filter_applied = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
                time.sleep(0.5)
                st.rerun()
        
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
        
        st.markdown("---")
        
        # Data source selection
        st.markdown("### 📂 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Google Sheets", "Upload CSV"],
            index=0 if st.session_state.data_source == "sheet" else 1
        )
        st.session_state.data_source = "sheet" if data_source == "Google Sheets" else "upload"
        
        uploaded_file = None
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        # Display mode
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose view:",
            ["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if st.session_state.display_mode == "Technical" else 1,
            key="display_mode_toggle"
        )
        st.session_state.display_mode = display_mode
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
    
    # Load data
    try:
        with st.spinner("📥 Loading data..."):
            if st.session_state.data_source == "upload" and uploaded_file is not None:
                df, timestamp = load_data("upload", file_data=uploaded_file)
            else:
                df, timestamp = load_data("sheet")
            
            st.session_state.data_timestamp = timestamp
            st.session_state.data_quality = calculate_data_quality(df)
            
            st.caption(f"✅ Loaded {len(df):,} stocks at {timestamp.strftime('%I:%M %p')}")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    qa_cols = st.columns(5)
    
    # Check for quick filter state
    quick_filter_applied = st.session_state.get('quick_filter_applied', False)
    quick_filter = st.session_state.get('quick_filter', None)
    
    with qa_cols[0]:
        if st.button("📈 Top Gainers", use_container_width=True):
            quick_filter = 'top_gainers'
            quick_filter_applied = True
            st.session_state['quick_filter'] = quick_filter
            st.session_state['quick_filter_applied'] = True
    
    with qa_cols[1]:
        if st.button("🔥 Volume Surges", use_container_width=True):
            quick_filter = 'volume_surges'
            quick_filter_applied = True
            st.session_state['quick_filter'] = quick_filter
            st.session_state['quick_filter_applied'] = True
    
    with qa_cols[2]:
        if st.button("🎯 Breakout Ready", use_container_width=True):
            quick_filter = 'breakout_ready'
            quick_filter_applied = True
            st.session_state['quick_filter'] = quick_filter
            st.session_state['quick_filter_applied'] = True
    
    with qa_cols[3]:
        if st.button("💎 Hidden Gems", use_container_width=True):
            quick_filter = 'hidden_gems'
            quick_filter_applied = True
            st.session_state['quick_filter'] = quick_filter
            st.session_state['quick_filter_applied'] = True
    
    with qa_cols[4]:
        if st.button("🌊 Show All", use_container_width=True):
            quick_filter = None
            quick_filter_applied = False
            st.session_state['quick_filter'] = None
            st.session_state['quick_filter_applied'] = False
    
    # Apply quick filter
    if quick_filter:
        if quick_filter == 'top_gainers':
            display_df = df[df['momentum_score'] >= CONFIG.TOP_GAINER_MOMENTUM]
            st.info(f"Showing {len(display_df)} stocks with momentum score ≥ {CONFIG.TOP_GAINER_MOMENTUM}")
        elif quick_filter == 'volume_surges':
            display_df = df[df['rvol'] >= CONFIG.VOLUME_SURGE_RVOL]
            st.info(f"Showing {len(display_df)} stocks with RVOL ≥ {CONFIG.VOLUME_SURGE_RVOL}x")
        elif quick_filter == 'breakout_ready':
            display_df = df[df['breakout_score'] >= CONFIG.BREAKOUT_READY_SCORE]
            st.info(f"Showing {len(display_df)} stocks with breakout score ≥ {CONFIG.BREAKOUT_READY_SCORE}")
        elif quick_filter == 'hidden_gems':
            display_df = df[df['patterns'].str.contains('HIDDEN GEM', na=False)]
            st.info(f"Showing {len(display_df)} hidden gem stocks")
    else:
        display_df = df
    
    # Sidebar filters
    with st.sidebar:
        filters = {}
        
        # Category filter
        categories = FilterEngine.get_unique_values(display_df, 'category', filters)
        category_counts = display_df['category'].value_counts()
        category_options = [
            f"{cat} ({category_counts.get(cat, 0)})" 
            for cat in categories
        ]
        
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=category_options,
            default=[],
            placeholder="Select categories (empty = All)",
            key="category_filter"
        )
        
        # Extract actual category names
        filters['categories'] = [
            cat.split(' (')[0] for cat in selected_categories
        ] if selected_categories else []
        
        # Sector filter
        sectors = FilterEngine.get_unique_values(display_df, 'sector', filters)
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=[],
            placeholder="Select sectors (empty = All)",
            key="sector_filter"
        )
        filters['sectors'] = selected_sectors if selected_sectors else []
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="min_score"
        )
        
        # Pattern filter
        all_patterns = set()
        for patterns in display_df['patterns'].dropna():
            if patterns:
                all_patterns.update(patterns.split(' | '))
        
        if all_patterns:
            filters['patterns'] = st.multiselect(
                "Patterns",
                options=sorted(all_patterns),
                default=[],
                placeholder="Select patterns (empty = All)",
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
            index=0
        )
        filters['trend_range'] = trend_options[filters['trend_filter']]
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            # EPS tier filter
            eps_tiers = FilterEngine.get_unique_values(display_df, 'eps_tier', filters)
            selected_eps_tiers = st.multiselect(
                "EPS Tier",
                options=eps_tiers,
                default=[],
                placeholder="Select EPS tiers (empty = All)",
                key="eps_tier_filter"
            )
            filters['eps_tiers'] = selected_eps_tiers if selected_eps_tiers else []
            
            # PE tier filter
            pe_tiers = FilterEngine.get_unique_values(display_df, 'pe_tier', filters)
            selected_pe_tiers = st.multiselect(
                "PE Tier",
                options=pe_tiers,
                default=[],
                placeholder="Select PE tiers (empty = All)",
                key="pe_tier_filter"
            )
            filters['pe_tiers'] = selected_pe_tiers if selected_pe_tiers else []
            
            # Price tier filter
            price_tiers = FilterEngine.get_unique_values(display_df, 'price_tier', filters)
            selected_price_tiers = st.multiselect(
                "Price Range",
                options=price_tiers,
                default=[],
                placeholder="Select price ranges (empty = All)",
                key="price_tier_filter"
            )
            filters['price_tiers'] = selected_price_tiers if selected_price_tiers else []
            
            # EPS change filter
            if 'eps_change_pct' in display_df.columns:
                eps_change_input = st.text_input(
                    "Min EPS Change %",
                    value="",
                    placeholder="e.g. -50 or leave empty",
                    help="Enter minimum EPS growth percentage",
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
            if show_fundamentals and 'pe' in display_df.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_pe_input = st.text_input(
                        "Min PE Ratio",
                        value="",
                        placeholder="e.g. 10",
                        key="min_pe"
                    )
                    
                    if min_pe_input.strip():
                        try:
                            filters['min_pe'] = float(min_pe_input)
                        except ValueError:
                            st.error("Invalid Min PE")
                            filters['min_pe'] = None
                    else:
                        filters['min_pe'] = None
                
                with col2:
                    max_pe_input = st.text_input(
                        "Max PE Ratio",
                        value="",
                        placeholder="e.g. 30",
                        key="max_pe"
                    )
                    
                    if max_pe_input.strip():
                        try:
                            filters['max_pe'] = float(max_pe_input)
                        except ValueError:
                            st.error("Invalid Max PE")
                            filters['max_pe'] = None
                    else:
                        filters['max_pe'] = None
                
                # Data completeness filter
                filters['require_fundamental_data'] = st.checkbox(
                    "Only show stocks with PE and EPS data",
                    value=False,
                    help="Filter out stocks missing fundamental data"
                )
        
        # Clear filters button
        if st.button("🗑️ Clear All Filters", use_container_width=True, type="secondary"):
            st.session_state['quick_filter'] = None
            st.session_state['quick_filter_applied'] = False
            st.rerun()
    
    # Apply filters
    if quick_filter_applied:
        filtered_df = FilterEngine.apply_filters(display_df, filters)
    else:
        filtered_df = FilterEngine.apply_filters(df, filters)
    
    filtered_df = filtered_df.sort_values('rank')
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = len(filtered_df)
        total_original = len(df)
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
            valid_eps_change = filtered_df['eps_change_pct'].notna() & ~np.isinf(filtered_df['eps_change_pct'])
            positive_eps_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 0)
            strong_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 50)
            mega_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 100)
            
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
        "📊 Summary", "🏆 Rankings", "🌊 Wave Radar", 
        "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"
    ])
    
    # Tab 0: Summary
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        
        if not filtered_df.empty:
            # Market Pulse
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Trend distribution
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
                avg_score = filtered_df['master_score'].mean()
                median_score = filtered_df['master_score'].median()
                st.metric(
                    "Avg Master Score",
                    f"{avg_score:.1f}",
                    f"Median: {median_score:.1f}"
                )
            
            with col3:
                high_momentum = len(filtered_df[filtered_df['momentum_score'] >= 70])
                momentum_pct = (high_momentum / total_stocks * 100) if total_stocks > 0 else 0
                st.metric(
                    "High Momentum",
                    f"{high_momentum}",
                    f"{momentum_pct:.0f}% of stocks"
                )
            
            with col4:
                # Market regime
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
            
            # Trend Analysis
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
                
                strong_leaders = filtered_df[filtered_df['trend_quality'] >= 80].nlargest(5, 'master_score')
                if len(strong_leaders) > 0:
                    st.markdown("**🔥 Strong Uptrend:**")
                    for _, stock in strong_leaders.iterrows():
                        st.write(f"• {stock['ticker']} ({stock['master_score']:.1f})")
                
                # Pattern stats
                st.markdown("**📊 Pattern Stats**")
                multi_pattern = len(filtered_df[filtered_df['patterns'].str.count('\\|') >= 2])
                st.write(f"• Multi-Pattern: {multi_pattern}")
                
                high_rvol = len(filtered_df[filtered_df['rvol'] > 3])
                st.write(f"• High RVOL (>3x): {high_rvol}")
            
            # Top Opportunities
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
                # Near 52W High
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
            
            # Key Insights
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
            
            # Quick Stats Table
            st.markdown("#### 📊 Quick Statistics")
            
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
                index=CONFIG.AVAILABLE_TOP_N.index(CONFIG.DEFAULT_TOP_N)
            )
        
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
            # Add trend indicator
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
            
            # Apply formatting
            for col, fmt in format_rules.items():
                if col in display_df.columns:
                    try:
                        if col == 'ret_30d':
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:+.1f}%" if pd.notna(x) else '-'
                            )
                        else:
                            display_df[col] = display_df[col].apply(
                                lambda x: fmt.format(x) if pd.notna(x) else '-'
                            )
                    except Exception as e:
                        logger.warning(f"Error formatting {col}: {str(e)}")
            
            # Apply special formatting for fundamentals
            if show_fundamentals:
                if 'pe' in display_df.columns:
                    display_df['pe'] = display_df['pe'].apply(format_pe)
                
                if 'eps_change_pct' in display_df.columns:
                    display_df['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change)
            
            # Select and rename columns
            available_cols = [c for c in display_cols.keys() if c in display_df.columns]
            display_df = display_df[available_cols]
            display_df.columns = [display_cols[c] for c in available_cols]
            
            # Display
            st.dataframe(
                display_df,
                use_container_width=True,
                height=min(600, len(display_df) * 35 + 50),
                hide_index=True
            )
            
            # Quick stats
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown("**Score Distribution**")
                    st.text(f"Max: {filtered_df['master_score'].max():.1f}")
                    st.text(f"Min: {filtered_df['master_score'].min():.1f}")
                    st.text(f"Std: {filtered_df['master_score'].std():.1f}")
                
                with stat_cols[1]:
                    st.markdown("**Returns (30D)**")
                    if 'ret_30d' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['ret_30d'].max():.1f}%")
                        st.text(f"Avg: {filtered_df['ret_30d'].mean():.1f}%")
                        st.text(f"Positive: {(filtered_df['ret_30d'] > 0).sum()}")
                
                with stat_cols[2]:
                    if show_fundamentals:
                        st.markdown("**Fundamentals**")
                        if 'pe' in filtered_df.columns:
                            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
                            if valid_pe.any():
                                median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                                st.text(f"Median PE: {median_pe:.1f}x")
                    else:
                        st.markdown("**RVOL Stats**")
                        if 'rvol' in filtered_df.columns:
                            st.text(f"Max: {filtered_df['rvol'].max():.1f}x")
                            st.text(f"Avg: {filtered_df['rvol'].mean():.1f}x")
                            st.text(f">2x: {(filtered_df['rvol'] > 2).sum()}")
                
                with stat_cols[3]:
                    st.markdown("**Categories**")
                    for cat, count in filtered_df['category'].value_counts().head(3).items():
                        st.text(f"{cat}: {count}")
        
        else:
            st.warning("No stocks match the selected filters.")
    
    # Tab 2: Wave Radar
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
                value="Balanced",
                help="Conservative = Stronger signals, Aggressive = More signals"
            )
        
        with radar_col3:
            show_market_regime = st.checkbox(
                "📊 Market Regime Analysis",
                value=True,
                help="Show category rotation flow and market regime detection"
            )
        
        # Initialize wave_filtered_df
        wave_filtered_df = filtered_df.copy()
        
        with radar_col4:
            # Calculate Wave Strength
            if not wave_filtered_df.empty:
                try:
                    momentum_count = len(wave_filtered_df[wave_filtered_df['momentum_score'] >= 60])
                    accel_count = len(wave_filtered_df[wave_filtered_df['acceleration_score'] >= 70])
                    rvol_count = len(wave_filtered_df[wave_filtered_df['rvol'] >= 2])
                    breakout_count = len(wave_filtered_df[wave_filtered_df['breakout_score'] >= 70])
                    
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
                    
                    st.metric(
                        "Wave Strength",
                        f"{wave_emoji} {wave_strength:.0f}%",
                        f"{wave_color} Market"
                    )
                except Exception as e:
                    logger.error(f"Error calculating wave strength: {str(e)}")
                    st.metric("Wave Strength", "N/A", "Error")
        
        # Apply timeframe filtering
        if wave_timeframe != "All Waves":
            try:
                if wave_timeframe == "Intraday Surge":
                    if all(col in wave_filtered_df.columns for col in ['rvol', 'ret_1d', 'price', 'prev_close']):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['rvol'] >= 2.5) &
                            (wave_filtered_df['ret_1d'] > 2) &
                            (wave_filtered_df['price'] > wave_filtered_df['prev_close'] * 1.02)
                        ]
                    
                elif wave_timeframe == "3-Day Buildup":
                    if all(col in wave_filtered_df.columns for col in ['ret_3d', 'vol_ratio_7d_90d', 'price', 'sma_20d']):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_3d'] > 5) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 1.5) &
                            (wave_filtered_df['price'] > wave_filtered_df['sma_20d'])
                        ]
                    
                elif wave_timeframe == "Weekly Breakout":
                    if all(col in wave_filtered_df.columns for col in ['ret_7d', 'vol_ratio_7d_90d', 'from_high_pct']):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_7d'] > 8) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 2.0) &
                            (wave_filtered_df['from_high_pct'] > -10)
                        ]
                    
                elif wave_timeframe == "Monthly Trend":
                    if all(col in wave_filtered_df.columns for col in ['ret_30d', 'price', 'sma_20d', 'sma_50d']):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_30d'] > 15) &
                            (wave_filtered_df['price'] > wave_filtered_df['sma_20d']) &
                            (wave_filtered_df['sma_20d'] > wave_filtered_df['sma_50d'])
                        ]
                        
            except KeyError as e:
                logger.warning(f"Column missing for {wave_timeframe} filter: {str(e)}")
                st.warning(f"Some data not available for {wave_timeframe} filter")
                wave_filtered_df = filtered_df.copy()
        
        # Wave Radar Analysis
        if not wave_filtered_df.empty:
            # 1. MOMENTUM SHIFTS
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            
            # Set thresholds based on sensitivity
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
            
            # Find momentum shifts
            momentum_shifts = wave_filtered_df.copy()
            
            momentum_shifts['momentum_shift'] = (
                (momentum_shifts['momentum_score'] >= cross_threshold) & 
                (momentum_shifts['acceleration_score'] >= min_acceleration)
            )
            
            # Calculate signal count
            momentum_shifts['signal_count'] = 0
            
            # Count signals
            momentum_shifts.loc[momentum_shifts['momentum_shift'], 'signal_count'] += 1
            
            if 'rvol' in momentum_shifts.columns:
                momentum_shifts.loc[momentum_shifts['rvol'] >= min_rvol_threshold, 'signal_count'] += 1
            
            momentum_shifts.loc[momentum_shifts['acceleration_score'] >= acceleration_alert_threshold, 'signal_count'] += 1
            
            if 'vol_ratio_7d_90d' in momentum_shifts.columns:
                momentum_shifts.loc[momentum_shifts['vol_ratio_7d_90d'] >= 1.5, 'signal_count'] += 1
            
            if 'breakout_score' in momentum_shifts.columns:
                momentum_shifts.loc[momentum_shifts['breakout_score'] >= 75, 'signal_count'] += 1
            
            # Calculate shift strength
            momentum_shifts['shift_strength'] = (
                momentum_shifts['momentum_score'] * 0.4 +
                momentum_shifts['acceleration_score'] * 0.4 +
                momentum_shifts['rvol_score'] * 0.2
            )
            
            # Get top shifts
            top_shifts = momentum_shifts[momentum_shifts['momentum_shift']].sort_values(
                ['signal_count', 'shift_strength'], ascending=[False, False]
            ).head(20)
            
            if len(top_shifts) > 0:
                display_columns = ['ticker', 'company_name', 'master_score', 'momentum_score', 
                                 'acceleration_score', 'rvol', 'signal_count', 'category']
                
                if 'ret_7d' in top_shifts.columns:
                    display_columns.insert(-2, 'ret_7d')
                
                shift_display = top_shifts[display_columns].copy()
                
                # Add signal indicator
                shift_display['Signal'] = shift_display['signal_count'].apply(
                    lambda x: f"{'🔥' * min(x, 3)} {x}/5"
                )
                
                # Format display
                if 'ret_7d' in shift_display.columns:
                    shift_display['ret_7d'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%")
                
                shift_display['rvol'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x")
                
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
                shift_display = shift_display.drop(columns=['Signals'])  # Remove numeric signals column
                
                st.dataframe(
                    shift_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Multi-signal leaders
                multi_signal_leaders = top_shifts[top_shifts['signal_count'] >= 3]
                if len(multi_signal_leaders) > 0:
                    st.success(f"🏆 Found {len(multi_signal_leaders)} stocks with 3+ signals")
            else:
                st.info(f"No momentum shifts detected in {wave_timeframe} timeframe")
            
            # 2. CATEGORY ROTATION FLOW
            if show_market_regime:
                st.markdown("#### 💰 Category Rotation - Smart Money Flow")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Calculate category performance
                    try:
                        if 'category' in wave_filtered_df.columns:
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
                                
                                category_flow = category_flow.sort_values('Flow Score', ascending=False)
                                
                                # Determine flow direction
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
                            st.write(f"{emoji} **{cat}**: Score {row['Flow Score']:.1f}")
                    else:
                        st.info("No category data available")
            
            # 3. EMERGING PATTERNS
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            
            # Set pattern distance thresholds
            if sensitivity == "Conservative":
                pattern_distance = 5
            elif sensitivity == "Balanced":
                pattern_distance = 10
            else:  # Aggressive
                pattern_distance = 15
            
            # Check for emerging patterns
            emergence_data = []
            
            # Category Leader emergence
            if 'category_percentile' in wave_filtered_df.columns:
                close_to_leader = wave_filtered_df[
                    (wave_filtered_df['category_percentile'] >= (90 - pattern_distance)) & 
                    (wave_filtered_df['category_percentile'] < 90)
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
            if 'breakout_score' in wave_filtered_df.columns:
                close_to_breakout = wave_filtered_df[
                    (wave_filtered_df['breakout_score'] >= (80 - pattern_distance)) & 
                    (wave_filtered_df['breakout_score'] < 80)
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
            
            if emergence_data:
                emergence_df = pd.DataFrame(emergence_data).sort_values('Score', ascending=False).head(15)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(emergence_df, use_container_width=True, hide_index=True)
                with col2:
                    st.metric("Emerging Patterns", len(emergence_df))
                    st.caption("Stocks about to trigger pattern alerts")
            else:
                st.info(f"No patterns emerging within {pattern_distance}% threshold")
            
            # 4. VOLUME SURGE DETECTION
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
                    display_columns = ['ticker', 'company_name', 'rvol', 'price', 'category']
                    
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
                # Show flow direction
                if 'flow_direction' in locals() and flow_direction != "➡️ Neutral":
                    regime_display = flow_direction.split()[1] if flow_direction != "N/A" else "Unknown"
                else:
                    regime_display = "Neutral"
                
                st.metric("Market Regime", regime_display)
            
            with summary_cols[2]:
                emergence_count = len(emergence_data) if 'emergence_data' in locals() and emergence_data else 0
                st.metric("Emerging Patterns", emergence_count)
            
            with summary_cols[3]:
                if 'acceleration_score' in wave_filtered_df.columns:
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
            
            # Master Score Breakdown
            st.markdown("#### 🏆 Top Stocks Master Score Breakdown")
            fig_breakdown = Visualizer.create_master_score_breakdown(filtered_df, n=20)
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
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
                            
                            # Fundamentals if enabled
                            if show_fundamentals:
                                st.markdown("**💰 Fundamentals**")
                                
                                # PE Ratio
                                if 'pe' in stock and pd.notna(stock['pe']):
                                    pe_display = format_pe(stock['pe'])
                                    st.text(f"PE Ratio: {pe_display}")
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
                                    except:
                                        st.text("EPS: - (Error)")
                                else:
                                    st.text("EPS: - (N/A)")
                                
                                # EPS Change
                                if 'eps_change_pct' in stock and pd.notna(stock['eps_change_pct']):
                                    eps_change_display = format_eps_change(stock['eps_change_pct'])
                                    st.text(f"EPS Growth: {eps_change_display}")
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
                            
                            # SMA Analysis
                            st.markdown("**📊 Trading Position**")
                            
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
                                        pct_above = ((current_price - stock[sma_col]) / stock[sma_col]) * 100
                                        st.text(f"✅ {sma_label}: ₹{stock[sma_col]:,.0f} (↑ {pct_above:.1f}%)")
                                    else:
                                        trading_below.append(sma_label)
                                        pct_below = ((stock[sma_col] - current_price) / stock[sma_col]) * 100
                                        st.text(f"❌ {sma_label}: ₹{stock[sma_col]:,.0f} (↓ {pct_below:.1f}%)")
                            
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
            "Average Score": f"{filtered_df['master_score'].mean():.1f}" if not filtered_df.empty else "N/A",
            "Stocks with Patterns": (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0,
            "High RVOL (>2x)": (filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns else 0,
            "Positive 30D Returns": (filtered_df['ret_30d'] > 0).sum() if 'ret_30d' in filtered_df.columns else 0,
            "Data Quality": f"{(1 - filtered_df['master_score'].isna().sum() / len(filtered_df)) * 100:.1f}%" if not filtered_df.empty else "N/A"
        }
        
        stat_cols = st.columns(3)
        for i, (label, value) in enumerate(export_stats.items()):
            with stat_cols[i % 3]:
                st.metric(label, value)
    
    # Tab 6: About
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
            
            1. **Summary Tab** - Executive dashboard with market overview
            2. **Quick Actions** - Use buttons for instant insights
            3. **Rankings Tab** - View top-ranked stocks with comprehensive metrics
            4. **Wave Radar** - Monitor early momentum signals and market shifts
            5. **Analysis Tab** - Deep dive into market sectors and patterns
            6. **Search Tab** - Find specific stocks with detailed analysis
            7. **Export Tab** - Download data for further analysis
            
            #### 🔧 Pro Tips
            
            - Use **Hybrid Mode** to see both technical and fundamental data
            - Combine multiple filters for precision screening
            - Watch for stocks with multiple pattern detections
            - Monitor Wave Radar for early entry opportunities
            - Export data regularly for historical tracking
            
            #### 📊 Pattern Legend
            
            - 🔥 **CAT LEADER** - Top 10% in category
            - 💎 **HIDDEN GEM** - Strong in category, undervalued overall
            - 🚀 **ACCELERATING** - Momentum building rapidly
            - 🏦 **INSTITUTIONAL** - Smart money accumulation
            - ⚡ **VOL EXPLOSION** - Extreme volume surge
            - 🎯 **BREAKOUT** - Ready for technical breakout
            - 👑 **MARKET LEADER** - Top 5% overall
            - 🌊 **MOMENTUM WAVE** - Sustained momentum with acceleration
            - 💰 **LIQUID LEADER** - High liquidity with performance
            - 💪 **LONG STRENGTH** - Strong long-term performance
            - 🎯 **52W HIGH APPROACH** - Near 52-week high with momentum
            - 🔄 **52W LOW BOUNCE** - Bouncing from 52-week low
            - 👑 **GOLDEN ZONE** - Optimal range position
            - 📊 **VOL ACCUMULATION** - Smart money accumulation
            - 🔀 **MOMENTUM DIVERGE** - Acceleration divergence
            - 🎯 **RANGE COMPRESS** - Range compression setup
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Trend Indicators
            
            - 🔥 **Strong Uptrend** (80-100)
            - ✅ **Good Uptrend** (60-79)
            - ➡️ **Neutral Trend** (40-59)
            - ⚠️ **Weak/Downtrend** (0-39)
            
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
            - Optimized calculations
            - Cloud-ready architecture
            
            #### 🔒 Data Source
            
            - Live Google Sheets integration
            - CSV upload support
            - 1790+ stocks coverage
            - 41 data points per stock
            - Daily updates
            
            #### 💬 Support
            
            For questions or feedback:
            - Check filters if no data shows
            - Clear cache for fresh data
            - Use search for specific stocks
            - Export data for records
            
            ---
            
            **Version**: 3.0.6-FINAL-ENHANCED
            **Last Updated**: Dec 2024
            **Status**: PERMANENT PRODUCTION
            **Engine**: Fully Vectorized
            """)
        
        # System stats
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.metric(
                "Total Stocks Loaded",
                f"{len(df):,}"
            )
        
        with stats_cols[1]:
            st.metric(
                "Currently Filtered",
                f"{len(filtered_df):,}"
            )
        
        with stats_cols[2]:
            st.metric(
                "Data Quality",
                f"{st.session_state.data_quality.get('completeness', 0):.1f}%"
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
            Wave Detection Ultimate 3.0 Enhanced | FINAL PERMANENT VERSION<br>
            <small>Real-time momentum detection • Early entry signals • Smart money flow tracking • Advanced Pattern Recognition</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the application
if __name__ == "__main__":
    main()
