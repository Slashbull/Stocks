"""
Wave Detection Ultimate 3.0 - Final Production Version
Professional-grade stock momentum detection system
No further updates will be made - this is the permanent version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from functools import wraps
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION - DO NOT MODIFY
# ============================================

class CONFIG:
    """Master configuration - All settings locked for production"""
    
    # Master Score Weights - LOCKED
    POSITION_WEIGHT = 0.30
    VOLUME_WEIGHT = 0.25
    MOMENTUM_WEIGHT = 0.15
    ACCELERATION_WEIGHT = 0.10
    BREAKOUT_WEIGHT = 0.10
    RVOL_WEIGHT = 0.10
    
    # Pattern Thresholds - OPTIMIZED
    PATTERN_THRESHOLDS = {
        'category_leader': 85,
        'hidden_gem': 75,
        'accelerating': 85,
        'institutional': 80,
        'vol_explosion': 3.0,
        'breakout': 75,
        'market_leader': 90,
        'momentum_wave': 85,
        'liquid_leader': 80,
        'long_strength': 75,
        'high_approach': 90,
        'low_bounce': 10,
        'golden_zone': (60, 80),
        'vol_accumulation': 1.5,
        'momentum_diverge': 70,
        'range_compress': 25,
        'stealth': 70,
        'vampire': 75,
        'perfect_storm': 85
    }
    
    # Wave States - FINAL
    WAVE_STATES = {
        'CRESTING': {'min_score': 85, 'momentum': 80, 'color': '#FF4136'},
        'BUILDING': {'min_score': 70, 'momentum': 60, 'color': '#FF851B'},
        'FORMING': {'min_score': 55, 'momentum': 40, 'color': '#FFDC00'},
        'BREAKING': {'min_score': 0, 'momentum': 0, 'color': '#B10DC9'}
    }
    
    # Display Settings
    TIER_COLORS = {
        'ELITE': '#FFD700',
        'PREMIUM': '#C0C0C0',
        'STANDARD': '#CD7F32',
        'DEVELOPING': '#4169E1',
        'EMERGING': '#32CD32'
    }
    
    # Cache Settings
    CACHE_TTL = 3600  # 1 hour
    
    # Performance Targets
    PERFORMANCE_TARGETS = {
        'data_load': 2.0,
        'pattern_detection': 0.5,
        'ranking': 0.3,
        'filter': 0.2,
        'search': 0.05
    }
    
    # Export Columns - Professional Order
    EXPORT_COLUMNS = [
        'ticker', 'company_name', 'master_score', 'tier', 'wave_state',
        'patterns', 'price', 'ret_30d', 'rvol', 'money_flow_millions',
        'vmi', 'position_tension', 'momentum_harmony', 'category',
        'category_percentile', 'sector', 'market_cap', 'pe', 'eps_change_pct'
    ]

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """Track and optimize performance"""
    
    @staticmethod
    def timer(target_time: float = 1.0):
        """Decorator to time functions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if elapsed > target_time:
                    logging.warning(f"{func.__name__} took {elapsed:.2f}s (target: {target_time}s)")
                
                return result
            return wrapper
        return decorator

# ============================================
# LOGGER SETUP
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CORE DATA ENGINE
# ============================================

class DataEngine:
    """Handle all data operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['data_load'])
    def load_data(source: str = 'sheet', uploaded_file=None) -> pd.DataFrame:
        """Load data from Google Sheets or CSV"""
        try:
            if source == 'upload' and uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                logger.info(f"Loaded {len(df)} rows from uploaded CSV")
            else:
                # Default Google Sheets URL
                sheet_url = st.session_state.get('sheet_url', 
                    "https://docs.google.com/spreadsheets/d/1Gg3tF64tAYq0yTBDhsCF0zH9PDzQlRMgjLIzXA5_d6Y/export?format=csv&gid=1883862674")
                df = pd.read_csv(sheet_url)
                logger.info(f"Loaded {len(df)} rows from Google Sheets")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Failed to load data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data validation"""
        quality = {
            'total_rows': len(df),
            'valid_rows': 0,
            'missing_critical': [],
            'data_issues': [],
            'completeness': 0.0
        }
        
        # Critical columns
        critical_cols = ['ticker', 'company_name', 'price']
        
        for col in critical_cols:
            if col not in df.columns:
                quality['missing_critical'].append(col)
            elif df[col].isna().sum() > len(df) * 0.1:
                quality['data_issues'].append(f"{col} has >10% missing values")
        
        # Calculate completeness
        total_cells = len(df) * len(df.columns)
        non_null_cells = df.notna().sum().sum()
        quality['completeness'] = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Count valid rows
        quality['valid_rows'] = len(df[df['ticker'].notna() & df['price'].notna()])
        
        return quality
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        df = df.copy()
        
        # Clean numeric columns
        numeric_cols = [
            'price', 'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 
            'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'rvol', 'pe', 
            'eps_current', 'eps_last_qtr', 'eps_change_pct'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # Remove % signs and convert
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean percentage columns (already in percentage form)
        pct_cols = ['from_low_pct', 'from_high_pct']
        for col in pct_cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure positive values where needed
        positive_cols = ['price', 'rvol', 'volume_1d', 'volume_7d', 'volume_30d']
        for col in positive_cols:
            if col in df.columns:
                df[col] = df[col].abs()
        
        # Handle infinity values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Standardize text columns
        text_cols = ['ticker', 'company_name', 'category', 'sector']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], np.nan)
        
        return df

# ============================================
# TECHNICAL INDICATORS ENGINE
# ============================================

class TechnicalIndicators:
    """Calculate all technical indicators"""
    
    @staticmethod
    def calculate_sma_positions(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA positions and scores"""
        df = df.copy()
        
        # SMA columns mapping
        sma_cols = {
            '20': 'sma_20d',
            '50': 'sma_50d', 
            '200': 'sma_200d'
        }
        
        # Calculate positions relative to SMAs
        for period, col in sma_cols.items():
            if col in df.columns and 'price' in df.columns:
                df[f'above_sma_{period}'] = df['price'] > df[col]
                df[f'sma_{period}_dist'] = ((df['price'] - df[col]) / df[col] * 100).fillna(0)
        
        # Count SMAs above
        sma_above_cols = [f'above_sma_{p}' for p in sma_cols.keys()]
        existing_cols = [col for col in sma_above_cols if col in df.columns]
        if existing_cols:
            df['smas_above'] = df[existing_cols].sum(axis=1)
        else:
            df['smas_above'] = 0
        
        return df
    
    @staticmethod
    def calculate_trend_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive trend score"""
        df = df.copy()
        
        # Initialize trend score
        df['trend_score'] = 50.0
        
        # Price trend component (40%)
        if 'ret_30d' in df.columns:
            price_trend = df['ret_30d'].clip(-50, 100)
            df['trend_score'] += price_trend * 0.2
        
        # SMA alignment component (30%)
        if 'smas_above' in df.columns:
            sma_score = df['smas_above'] * 10  # 0, 10, 20, 30
            df['trend_score'] += sma_score * 0.3
        
        # Momentum consistency (30%)
        momentum_cols = ['ret_7d', 'ret_30d', 'ret_3m']
        positive_momentum = 0
        for col in momentum_cols:
            if col in df.columns:
                positive_momentum += (df[col] > 0).astype(int)
        
        df['trend_score'] += (positive_momentum / len(momentum_cols) * 30) * 0.3
        
        # Clip to valid range
        df['trend_score'] = df['trend_score'].clip(0, 100)
        
        return df

# ============================================
# ADVANCED METRICS ENGINE
# ============================================

class AdvancedMetrics:
    """Calculate advanced proprietary metrics"""
    
    @staticmethod
    def calculate_money_flow(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate money flow in millions"""
        df = df.copy()
        
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            # Money Flow = Price × Volume × RVOL / 1,000,000
            df['money_flow_millions'] = (
                df['price'] * df['volume_1d'] * df['rvol'] / 1_000_000
            ).fillna(0)
        else:
            df['money_flow_millions'] = 0
        
        return df
    
    @staticmethod
    def calculate_vmi(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Momentum Index"""
        df = df.copy()
        
        # VMI = Weighted average of volume ratios
        df['vmi'] = 50.0
        
        vol_ratios = {
            'vol_ratio_1d_90d': 0.4,
            'vol_ratio_7d_90d': 0.3,
            'vol_ratio_30d_90d': 0.3
        }
        
        total_weight = 0
        for col, weight in vol_ratios.items():
            if col in df.columns:
                # Convert ratio to score (0-100)
                ratio_score = df[col].clip(0, 3) * 33.33
                df['vmi'] += ratio_score * weight
                total_weight += weight
        
        if total_weight > 0:
            df['vmi'] = (df['vmi'] - 50) / total_weight + 50
        
        df['vmi'] = df['vmi'].clip(0, 100).fillna(50)
        
        return df
    
    @staticmethod
    def calculate_position_tension(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position tension indicator"""
        df = df.copy()
        
        if 'from_high_pct' in df.columns and 'from_low_pct' in df.columns:
            # High tension when near 52w high with strong momentum
            high_tension = (100 - df['from_high_pct']) * 0.6
            
            # Low tension when far from high
            low_tension = df['from_low_pct'] * 0.4
            
            df['position_tension'] = (high_tension + low_tension).clip(0, 100)
        else:
            df['position_tension'] = 50
        
        return df
    
    @staticmethod
    def calculate_momentum_harmony(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-timeframe momentum alignment"""
        df = df.copy()
        
        # Count positive momentum across timeframes
        momentum_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m']
        
        harmony_count = 0
        for col in momentum_cols:
            if col in df.columns:
                harmony_count += (df[col] > 0).astype(int)
        
        df['momentum_harmony'] = harmony_count
        
        return df
    
    @staticmethod
    def detect_wave_state(df: pd.DataFrame) -> pd.DataFrame:
        """Classify stocks into wave states"""
        df = df.copy()
        
        # Default state
        df['wave_state'] = 'BREAKING'
        df['wave_state_score'] = 0
        
        # Calculate wave state based on multiple factors
        for state, criteria in CONFIG.WAVE_STATES.items():
            mask = (
                (df['master_score'] >= criteria['min_score']) &
                (df['momentum_score'] >= criteria['momentum'])
            )
            df.loc[mask, 'wave_state'] = state
            df.loc[mask, 'wave_state_score'] = criteria['min_score']
        
        return df

# ============================================
# RANKING ENGINE - CORE ALGORITHM
# ============================================

class RankingEngine:
    """Master scoring and ranking system"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['ranking'])
    def calculate_master_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores and master score"""
        
        # Calculate component scores
        df = RankingEngine._calculate_position_score(df)
        df = RankingEngine._calculate_volume_score(df)
        df = RankingEngine._calculate_momentum_score(df)
        df = RankingEngine._calculate_acceleration_score(df)
        df = RankingEngine._calculate_breakout_score(df)
        df = RankingEngine._calculate_rvol_score(df)
        
        # Calculate auxiliary scores
        df = RankingEngine._calculate_trend_quality(df)
        df = RankingEngine._calculate_long_term_strength(df)
        df = RankingEngine._calculate_liquidity_score(df)
        
        # MASTER SCORE 3.0 CALCULATION
        components = {
            'position_score': CONFIG.POSITION_WEIGHT,
            'volume_score': CONFIG.VOLUME_WEIGHT,
            'momentum_score': CONFIG.MOMENTUM_WEIGHT,
            'acceleration_score': CONFIG.ACCELERATION_WEIGHT,
            'breakout_score': CONFIG.BREAKOUT_WEIGHT,
            'rvol_score': CONFIG.RVOL_WEIGHT
        }
        
        df['master_score'] = 0
        for component, weight in components.items():
            if component in df.columns:
                df['master_score'] += df[component].fillna(50) * weight
        
        df['master_score'] = df['master_score'].clip(0, 100)
        
        # Calculate ranks and percentiles
        df = RankingEngine._calculate_ranks(df)
        df = RankingEngine._calculate_category_ranks(df)
        df = RankingEngine._assign_tiers(df)
        
        return df
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position score (30% weight)"""
        df['position_score'] = 50.0
        
        if 'from_low_pct' in df.columns:
            # Reward stocks moving up from lows
            df['position_score'] = df['from_low_pct'].clip(0, 100)
            
            # Bonus for golden zone (60-80% from low)
            golden_mask = df['from_low_pct'].between(60, 80)
            df.loc[golden_mask, 'position_score'] += 10
        
        df['position_score'] = df['position_score'].clip(0, 100)
        return df
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume dynamics score (25% weight)"""
        df['volume_score'] = 50.0
        
        # Multi-timeframe volume analysis
        vol_components = []
        
        if 'vol_ratio_1d_90d' in df.columns:
            vol_components.append(df['vol_ratio_1d_90d'].clip(0, 3) * 33.33 * 0.4)
        
        if 'vol_ratio_7d_90d' in df.columns:
            vol_components.append(df['vol_ratio_7d_90d'].clip(0, 3) * 33.33 * 0.3)
        
        if 'vol_ratio_30d_90d' in df.columns:
            vol_components.append(df['vol_ratio_30d_90d'].clip(0, 3) * 33.33 * 0.3)
        
        if vol_components:
            df['volume_score'] = sum(vol_components)
        
        df['volume_score'] = df['volume_score'].clip(0, 100)
        return df
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum score (15% weight)"""
        df['momentum_score'] = 50.0
        
        if 'ret_30d' in df.columns:
            # Convert 30-day return to score
            # -50% = 0 score, 0% = 50 score, +50% = 100 score
            df['momentum_score'] = (df['ret_30d'] + 50).clip(0, 100)
        
        return df
    
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate acceleration score (10% weight)"""
        df['acceleration_score'] = 50.0
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            # Weekly return as % of monthly return
            # If weekly > 50% of monthly, momentum accelerating
            monthly_abs = df['ret_30d'].abs() + 1  # Avoid division by zero
            weekly_contribution = (df['ret_7d'] / monthly_abs * 100).fillna(0)
            
            # Convert to score
            df['acceleration_score'] = weekly_contribution.clip(0, 100)
        
        return df
    
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout probability score (10% weight)"""
        df['breakout_score'] = 50.0
        
        factors = []
        
        # Near 52-week high
        if 'from_high_pct' in df.columns:
            high_proximity = (100 - df['from_high_pct']).clip(0, 100) * 0.4
            factors.append(high_proximity)
        
        # Strong recent momentum
        if 'ret_30d' in df.columns:
            momentum_factor = df['ret_30d'].clip(0, 50) * 2 * 0.3
            factors.append(momentum_factor)
        
        # Volume surge
        if 'rvol' in df.columns:
            volume_factor = (df['rvol'] - 1).clip(0, 2) * 50 * 0.3
            factors.append(volume_factor)
        
        if factors:
            df['breakout_score'] = sum(factors)
        
        df['breakout_score'] = df['breakout_score'].clip(0, 100)
        return df
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relative volume score (10% weight)"""
        df['rvol_score'] = 50.0
        
        if 'rvol' in df.columns:
            # Convert RVOL to score
            # 0.5x = 25, 1x = 50, 2x = 75, 3x+ = 100
            df['rvol_score'] = (df['rvol'] * 25).clip(0, 100)
        
        return df
    
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend quality score"""
        df['trend_quality'] = 50.0
        
        factors = []
        
        # SMA alignment
        if 'smas_above' in df.columns:
            sma_factor = df['smas_above'] * 33.33 * 0.5
            factors.append(sma_factor)
        
        # Consistent gains
        gain_cols = ['ret_7d', 'ret_30d', 'ret_3m']
        gain_count = 0
        for col in gain_cols:
            if col in df.columns:
                gain_count += (df[col] > 0).astype(int)
        
        if gain_cols:
            consistency_factor = (gain_count / len(gain_cols) * 100) * 0.5
            factors.append(consistency_factor)
        
        if factors:
            df['trend_quality'] = sum(factors)
        
        df['trend_quality'] = df['trend_quality'].clip(0, 100)
        return df
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate long-term strength score"""
        df['long_term_strength'] = 50.0
        
        lt_returns = {
            'ret_3m': 0.3,
            'ret_6m': 0.3,
            'ret_1y': 0.4
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for col, weight in lt_returns.items():
            if col in df.columns:
                # Normalize returns to 0-100 scale
                norm_return = (df[col] + 50).clip(0, 100)
                weighted_sum += norm_return * weight
                total_weight += weight
        
        if total_weight > 0:
            df['long_term_strength'] = weighted_sum / total_weight
        
        df['long_term_strength'] = df['long_term_strength'].clip(0, 100)
        return df
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity score"""
        df['liquidity_score'] = 50.0
        
        if 'volume_30d' in df.columns:
            # Use percentile ranking for liquidity
            df['liquidity_score'] = df['volume_30d'].rank(pct=True) * 100
        
        df['liquidity_score'] = df['liquidity_score'].clip(0, 100)
        return df
    
    @staticmethod
    def _calculate_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall ranks and percentiles"""
        # Overall rank
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom').astype(int)
        
        # Overall percentile
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        
        return df
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ranks within categories"""
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        if 'category' in df.columns:
            for category in df['category'].unique():
                if pd.notna(category) and category != 'Unknown':
                    mask = df['category'] == category
                    cat_df = df[mask]
                    
                    if len(cat_df) > 0:
                        # Category rank
                        df.loc[mask, 'category_rank'] = cat_df['master_score'].rank(
                            method='first', ascending=False, na_option='bottom'
                        ).astype(int)
                        
                        # Category percentile
                        df.loc[mask, 'category_percentile'] = cat_df['master_score'].rank(
                            pct=True, ascending=True, na_option='bottom'
                        ) * 100
        
        return df
    
    @staticmethod
    def _assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
        """Assign stocks to tiers based on percentile"""
        df['tier'] = 'EMERGING'
        
        conditions = [
            (df['percentile'] >= 95, 'ELITE'),
            (df['percentile'] >= 85, 'PREMIUM'),
            (df['percentile'] >= 70, 'STANDARD'),
            (df['percentile'] >= 50, 'DEVELOPING'),
            (df['percentile'] < 50, 'EMERGING')
        ]
        
        for condition, tier in conditions:
            if isinstance(condition, tuple):
                mask, tier_name = condition, tier
                df.loc[mask, 'tier'] = tier_name
            else:
                df.loc[condition, 'tier'] = tier
        
        return df

# ============================================
# PATTERN DETECTION ENGINE
# ============================================

class PatternDetector:
    """Detect all 25 patterns using vectorized operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['pattern_detection'])
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns efficiently using numpy vectorization"""
        
        n = len(df)
        pattern_masks = np.zeros((n, 25), dtype=bool)
        pattern_names = []
        
        # Technical Patterns (0-10)
        idx = 0
        
        # 1. Category Leader
        if 'category_percentile' in df.columns:
            pattern_masks[:, idx] = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
            pattern_names.append('🔥 CAT LEADER')
            idx += 1
        
        # 2. Hidden Gem
        if all(col in df.columns for col in ['percentile', 'rvol']):
            pattern_masks[:, idx] = (
                (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) &
                (df['rvol'] >= 1.5)
            )
            pattern_names.append('💎 HIDDEN GEM')
            idx += 1
        
        # 3. Accelerating
        if 'acceleration_score' in df.columns:
            pattern_masks[:, idx] = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['accelerating']
            pattern_names.append('🚀 ACCELERATING')
            idx += 1
        
        # 4. Institutional
        if all(col in df.columns for col in ['volume_30d', 'liquidity_score']):
            pattern_masks[:, idx] = (
                (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['volume_30d'] > df['volume_30d'].quantile(0.8))
            )
            pattern_names.append('🏦 INSTITUTIONAL')
            idx += 1
        
        # 5. Volume Explosion
        if 'rvol' in df.columns:
            pattern_masks[:, idx] = df['rvol'] >= CONFIG.PATTERN_THRESHOLDS['vol_explosion']
            pattern_names.append('⚡ VOL EXPLOSION')
            idx += 1
        
        # 6. Breakout
        if 'breakout_score' in df.columns:
            pattern_masks[:, idx] = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout']
            pattern_names.append('🎯 BREAKOUT')
            idx += 1
        
        # 7. Market Leader
        if 'percentile' in df.columns:
            pattern_masks[:, idx] = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
            pattern_names.append('👑 MARKET LEADER')
            idx += 1
        
        # 8. Momentum Wave
        if all(col in df.columns for col in ['momentum_score', 'trend_quality']):
            pattern_masks[:, idx] = (
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                (df['trend_quality'] >= 70)
            )
            pattern_names.append('🌊 MOMENTUM WAVE')
            idx += 1
        
        # 9. Liquid Leader
        if 'liquidity_score' in df.columns:
            pattern_masks[:, idx] = df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']
            pattern_names.append('💰 LIQUID LEADER')
            idx += 1
        
        # 10. Long Strength
        if 'long_term_strength' in df.columns:
            pattern_masks[:, idx] = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            pattern_names.append('💪 LONG STRENGTH')
            idx += 1
        
        # 11. Quality Trend
        if 'trend_quality' in df.columns:
            pattern_masks[:, idx] = df['trend_quality'] >= 80
            pattern_names.append('📈 QUALITY TREND')
            idx += 1
        
        # Range Patterns (11-16)
        
        # 12. 52W High Approach
        if 'from_high_pct' in df.columns:
            pattern_masks[:, idx] = (100 - df['from_high_pct']) >= CONFIG.PATTERN_THRESHOLDS['high_approach']
            pattern_names.append('🎯 52W HIGH APPROACH')
            idx += 1
        
        # 13. 52W Low Bounce
        if 'from_low_pct' in df.columns:
            pattern_masks[:, idx] = df['from_low_pct'] <= CONFIG.PATTERN_THRESHOLDS['low_bounce']
            pattern_names.append('🔄 52W LOW BOUNCE')
            idx += 1
        
        # 14. Golden Zone
        if 'from_low_pct' in df.columns:
            low, high = CONFIG.PATTERN_THRESHOLDS['golden_zone']
            pattern_masks[:, idx] = df['from_low_pct'].between(low, high)
            pattern_names.append('👑 GOLDEN ZONE')
            idx += 1
        
        # 15. Volume Accumulation
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
            pattern_masks[:, idx] = (
                (df['vol_ratio_7d_90d'] >= CONFIG.PATTERN_THRESHOLDS['vol_accumulation']) &
                (df['vol_ratio_30d_90d'] >= 1.2)
            )
            pattern_names.append('📊 VOL ACCUMULATION')
            idx += 1
        
        # 16. Momentum Divergence
        if all(col in df.columns for col in ['momentum_score', 'rvol']):
            pattern_masks[:, idx] = (
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_diverge']) &
                (df['rvol'] < 0.8)
            )
            pattern_names.append('🔀 MOMENTUM DIVERGE')
            idx += 1
        
        # 17. Range Compression
        if 'from_low_pct' in df.columns and 'from_high_pct' in df.columns:
            range_size = df['from_low_pct'] + df['from_high_pct']
            pattern_masks[:, idx] = range_size <= CONFIG.PATTERN_THRESHOLDS['range_compress']
            pattern_names.append('🎯 RANGE COMPRESS')
            idx += 1
        
        # Intelligence Patterns (17-19)
        
        # 18. Stealth Mode
        if all(col in df.columns for col in ['master_score', 'rvol', 'ret_30d']):
            pattern_masks[:, idx] = (
                (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['stealth']) &
                (df['rvol'] < 1.2) &
                (df['ret_30d'] > 10)
            )
            pattern_names.append('🤫 STEALTH')
            idx += 1
        
        # 19. Vampire Pattern
        if all(col in df.columns for col in ['ret_1d', 'rvol', 'momentum_score']):
            pattern_masks[:, idx] = (
                (df['ret_1d'] < -2) &
                (df['rvol'] > 2) &
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['vampire'])
            )
            pattern_names.append('🧛 VAMPIRE')
            idx += 1
        
        # 20. Perfect Storm
        if all(col in df.columns for col in ['position_score', 'volume_score', 'momentum_score']):
            pattern_masks[:, idx] = (
                (df['position_score'] >= CONFIG.PATTERN_THRESHOLDS['perfect_storm']) &
                (df['volume_score'] >= 80) &
                (df['momentum_score'] >= 75)
            )
            pattern_names.append('⛈️ PERFECT STORM')
            idx += 1
        
        # Fundamental Patterns (20-24) - Only in Hybrid Mode
        if st.session_state.get('display_mode', 'Technical') == 'Hybrid':
            
            # 21. Value Momentum
            if all(col in df.columns for col in ['pe', 'master_score']):
                valid_pe = df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)
                pattern_masks[:, idx] = valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
                pattern_names.append('💎 VALUE MOMENTUM')
                idx += 1
            
            # 22. Earnings Rocket
            if all(col in df.columns for col in ['eps_change_pct', 'acceleration_score']):
                valid_eps = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
                pattern_masks[:, idx] = (
                    valid_eps &
                    (df['eps_change_pct'] > 50) &
                    (df['acceleration_score'] >= 70)
                )
                pattern_names.append('📊 EARNINGS ROCKET')
                idx += 1
            
            # 23. Quality Leader
            if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
                valid_data = (
                    df['pe'].notna() & df['eps_change_pct'].notna() &
                    (df['pe'] > 0) & (df['pe'] < 10000)
                )
                pattern_masks[:, idx] = (
                    valid_data &
                    df['pe'].between(10, 25) &
                    (df['eps_change_pct'] > 20) &
                    (df['percentile'] >= 80)
                )
                pattern_names.append('🏆 QUALITY LEADER')
                idx += 1
            
            # 24. Turnaround
            if all(col in df.columns for col in ['eps_change_pct', 'ret_30d']):
                valid_eps = df['eps_change_pct'].notna()
                pattern_masks[:, idx] = (
                    valid_eps &
                    (df['eps_change_pct'] > 100) &
                    (df['ret_30d'] > 0)
                )
                pattern_names.append('⚡ TURNAROUND')
                idx += 1
            
            # 25. High PE Warning
            if 'pe' in df.columns:
                valid_pe = df['pe'].notna() & (df['pe'] > 0)
                pattern_masks[:, idx] = valid_pe & (df['pe'] > 50)
                pattern_names.append('⚠️ HIGH PE')
                idx += 1
        
        # Combine patterns efficiently
        pattern_results = []
        for i in range(n):
            row_patterns = [pattern_names[j] for j in range(len(pattern_names)) if pattern_masks[i, j]]
            pattern_results.append(' | '.join(row_patterns) if row_patterns else '')
        
        df['patterns'] = pattern_results
        
        return df

# ============================================
# MARKET INTELLIGENCE ENGINE
# ============================================

class MarketIntelligence:
    """Advanced market analysis and regime detection"""
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> str:
        """Detect current market regime"""
        if df.empty:
            return "NEUTRAL"
        
        # Calculate market breadth
        advances = (df['ret_1d'] > 0).sum() if 'ret_1d' in df.columns else 0
        declines = (df['ret_1d'] < 0).sum() if 'ret_1d' in df.columns else 0
        
        advance_ratio = advances / (advances + declines) if (advances + declines) > 0 else 0.5
        
        # Calculate average momentum
        avg_momentum = df['ret_30d'].mean() if 'ret_30d' in df.columns else 0
        
        # Determine regime
        if advance_ratio > 0.65 and avg_momentum > 10:
            return "RISK-ON"
        elif advance_ratio < 0.35 and avg_momentum < -10:
            return "RISK-OFF"
        else:
            return "NEUTRAL"
    
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Detect sector rotation with dynamic sampling"""
        if 'sector' not in df.columns:
            return pd.DataFrame()
        
        sector_analysis = []
        
        for sector in df['sector'].unique():
            if pd.isna(sector) or sector == 'Unknown':
                continue
            
            sector_df = df[df['sector'] == sector]
            sector_size = len(sector_df)
            
            if sector_size == 0:
                continue
            
            # Dynamic sampling based on sector size
            if sector_size <= 5:
                sample_size = sector_size
            elif sector_size <= 20:
                sample_size = int(sector_size * 0.8)
            elif sector_size <= 50:
                sample_size = int(sector_size * 0.6)
            elif sector_size <= 100:
                sample_size = int(sector_size * 0.4)
            else:
                sample_size = min(50, int(sector_size * 0.25))
            
            # Get top stocks by master score
            top_stocks = sector_df.nlargest(sample_size, 'master_score')
            
            # Calculate sector metrics
            metrics = {
                'Sector': sector,
                'Total Stocks': sector_size,
                'Sample Size': sample_size,
                'Avg Score': top_stocks['master_score'].mean(),
                'Avg 30D Return': top_stocks['ret_30d'].mean() if 'ret_30d' in top_stocks.columns else 0,
                'Avg RVOL': top_stocks['rvol'].mean() if 'rvol' in top_stocks.columns else 0,
                'Top Stock': top_stocks.iloc[0]['ticker'] if len(top_stocks) > 0 else 'N/A',
                'Momentum': 'Strong' if top_stocks['ret_30d'].mean() > 10 else 'Weak'
            }
            
            sector_analysis.append(metrics)
        
        return pd.DataFrame(sector_analysis).sort_values('Avg Score', ascending=False)

# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    """Advanced filtering system"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['filter'])
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters efficiently"""
        filtered_df = df.copy()
        
        # Categories filter
        if filters.get('categories') and filters['categories']:
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # Sectors filter
        if filters.get('sectors') and filters['sectors']:
            filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
        
        # Market cap filter
        if filters.get('market_caps') and filters['market_caps']:
            # Parse market cap values
            cap_masks = []
            for cap in filters['market_caps']:
                if cap == 'Large Cap':
                    cap_masks.append(filtered_df['market_cap'].str.contains('B', na=False))
                elif cap == 'Mid Cap':
                    cap_masks.append(
                        filtered_df['market_cap'].str.contains('M', na=False) &
                        (filtered_df['market_cap'].str.extract(r'(\d+\.?\d*)')[0].astype(float) >= 100)
                    )
                elif cap == 'Small Cap':
                    cap_masks.append(
                        filtered_df['market_cap'].str.contains('M', na=False) &
                        (filtered_df['market_cap'].str.extract(r'(\d+\.?\d*)')[0].astype(float) < 100)
                    )
            
            if cap_masks:
                combined_mask = cap_masks[0]
                for mask in cap_masks[1:]:
                    combined_mask |= mask
                filtered_df = filtered_df[combined_mask]
        
        # Master score range
        if 'score_range' in filters:
            min_score, max_score = filters['score_range']
            filtered_df = filtered_df[
                (filtered_df['master_score'] >= min_score) &
                (filtered_df['master_score'] <= max_score)
            ]
        
        # RVOL range
        if 'rvol_range' in filters:
            min_rvol, max_rvol = filters['rvol_range']
            filtered_df = filtered_df[
                (filtered_df['rvol'] >= min_rvol) &
                (filtered_df['rvol'] <= max_rvol)
            ]
        
        # 30D return range
        if 'ret_30d_range' in filters:
            min_ret, max_ret = filters['ret_30d_range']
            filtered_df = filtered_df[
                (filtered_df['ret_30d'] >= min_ret) &
                (filtered_df['ret_30d'] <= max_ret)
            ]
        
        # Trend strength filter
        if 'trend_strength' in filters and filters['trend_strength']:
            if filters['trend_strength'] == 'Strong Uptrend (60+)':
                filtered_df = filtered_df[filtered_df['trend_score'] >= 60]
            elif filters['trend_strength'] == 'Weak Uptrend (40-60)':
                filtered_df = filtered_df[filtered_df['trend_score'].between(40, 60)]
            elif filters['trend_strength'] == 'Downtrend (<40)':
                filtered_df = filtered_df[filtered_df['trend_score'] < 40]
        
        # Wave filter
        if filters.get('wave_states') and filters['wave_states']:
            filtered_df = filtered_df[filtered_df['wave_state'].isin(filters['wave_states'])]
        
        # Wave strength range
        if 'wave_strength_range' in filters:
            min_strength, max_strength = filters['wave_strength_range']
            filtered_df = filtered_df[
                (filtered_df['wave_state_score'] >= min_strength) &
                (filtered_df['wave_state_score'] <= max_strength)
            ]
        
        # Pattern filter
        if filters.get('has_patterns'):
            filtered_df = filtered_df[filtered_df['patterns'] != '']
        
        # PE filter (Hybrid mode)
        if 'pe_range' in filters and st.session_state.get('display_mode') == 'Hybrid':
            min_pe, max_pe = filters['pe_range']
            filtered_df = filtered_df[
                (filtered_df['pe'] >= min_pe) &
                (filtered_df['pe'] <= max_pe) &
                (filtered_df['pe'] > 0)
            ]
        
        # EPS growth filter (Hybrid mode)
        if 'eps_growth_range' in filters and st.session_state.get('display_mode') == 'Hybrid':
            min_eps, max_eps = filters['eps_growth_range']
            filtered_df = filtered_df[
                (filtered_df['eps_change_pct'] >= min_eps) &
                (filtered_df['eps_change_pct'] <= max_eps)
            ]
        
        return filtered_df

# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """Fast and flexible search functionality"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=CONFIG.PERFORMANCE_TARGETS['search'])
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks by ticker or company name with word boundary matching"""
        if not query:
            return pd.DataFrame()
        
        query = query.upper().strip()
        
        # Search in ticker (exact match)
        ticker_match = df[df['ticker'].str.upper() == query]
        
        # Search in company name (word boundary matching)
        def word_starts_with(company_name):
            if pd.isna(company_name):
                return False
            words = str(company_name).upper().split()
            return any(word.startswith(query) for word in words)
        
        company_match = df[df['company_name'].apply(word_starts_with)]
        
        # Combine results (remove duplicates)
        results = pd.concat([ticker_match, company_match]).drop_duplicates()
        
        # Sort by relevance (exact ticker match first, then by master score)
        if not results.empty:
            results['exact_match'] = results['ticker'].str.upper() == query
            results = results.sort_values(['exact_match', 'master_score'], ascending=[False, False])
            results = results.drop('exact_match', axis=1)
        
        return results

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle data exports"""
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export with selected columns"""
        export_df = df.copy()
        
        # Select columns that exist
        export_cols = [col for col in CONFIG.EXPORT_COLUMNS if col in export_df.columns]
        export_df = export_df[export_cols]
        
        # Format numeric columns
        numeric_formats = {
            'master_score': '{:.1f}',
            'ret_30d': '{:.1f}%',
            'rvol': '{:.1f}x',
            'money_flow_millions': '{:.1f}M',
            'pe': '{:.1f}',
            'eps_change_pct': '{:.0f}%',
            'category_percentile': '{:.0f}'
        }
        
        for col, fmt in numeric_formats.items():
            if col in export_df.columns:
                if '%' in fmt:
                    export_df[col] = export_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')
                elif 'M' in fmt:
                    export_df[col] = export_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')
                elif 'x' in fmt:
                    export_df[col] = export_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')
                else:
                    export_df[col] = export_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')
        
        return export_df.to_csv(index=False)

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: str = None, color: str = None):
        """Render a styled metric card"""
        if color:
            st.markdown(f"<p style='color: {color};'><b>{label}</b><br>{value}</p>", 
                       unsafe_allow_html=True)
        else:
            st.metric(label, value, delta)
    
    @staticmethod
    def render_progress_bar(value: float, label: str = "", max_value: float = 100):
        """Render a progress bar"""
        progress = value / max_value
        st.progress(progress)
        if label:
            st.caption(f"{label}: {value:.1f}/{max_value}")
    
    @staticmethod
    def render_wave_state_badge(state: str):
        """Render wave state with color"""
        color = CONFIG.WAVE_STATES.get(state, {}).get('color', '#808080')
        st.markdown(
            f"<span style='background-color: {color}; color: white; "
            f"padding: 2px 8px; border-radius: 4px;'>{state}</span>",
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_pattern_pills(patterns: str):
        """Render patterns as pills"""
        if not patterns:
            return
        
        pattern_list = patterns.split(' | ')
        pills_html = ""
        for pattern in pattern_list:
            pills_html += f"<span style='background-color: #E0E0E0; padding: 2px 6px; "
            pills_html += f"margin: 2px; border-radius: 12px; font-size: 12px;'>{pattern}</span>"
        
        st.markdown(pills_html, unsafe_allow_html=True)

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
    
    # Custom CSS
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
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
        margin-bottom: 10px;
    }
    .search-result-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = pd.DataFrame()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'display_mode' not in st.session_state:
        st.session_state.display_mode = 'Technical'
    if 'data_quality' not in st.session_state:
        st.session_state.data_quality = {}
    
    # Header
    st.markdown("# 🌊 Wave Detection Ultimate 3.0")
    st.markdown("**Professional Momentum Detection System** - Final Production Version")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        
        # Data Source Selection with prominent buttons
        st.markdown("### 📊 Data Source")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Google Sheets", 
                        type="primary" if st.session_state.get('data_source', 'sheet') == 'sheet' else "secondary",
                        use_container_width=True):
                st.session_state.data_source = 'sheet'
        
        with col2:
            if st.button("📁 Upload CSV", 
                        type="primary" if st.session_state.get('data_source', 'sheet') == 'upload' else "secondary",
                        use_container_width=True):
                st.session_state.data_source = 'upload'
        
        # Show upload interface if CSV selected
        uploaded_file = None
        if st.session_state.get('data_source', 'sheet') == 'upload':
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        # Load Data Button
        if st.button("🔄 Load/Refresh Data", type="primary", use_container_width=True):
            with st.spinner("Loading data..."):
                df = DataEngine.load_data(
                    source=st.session_state.get('data_source', 'sheet'),
                    uploaded_file=uploaded_file
                )
                
                if not df.empty:
                    # Clean data
                    df = DataEngine.clean_data(df)
                    
                    # Validate data
                    st.session_state.data_quality = DataEngine.validate_data(df)
                    
                    # Calculate all metrics
                    df = TechnicalIndicators.calculate_sma_positions(df)
                    df = TechnicalIndicators.calculate_trend_score(df)
                    df = AdvancedMetrics.calculate_money_flow(df)
                    df = AdvancedMetrics.calculate_vmi(df)
                    df = AdvancedMetrics.calculate_position_tension(df)
                    df = AdvancedMetrics.calculate_momentum_harmony(df)
                    
                    # Calculate master score and rankings
                    df = RankingEngine.calculate_master_score(df)
                    
                    # Detect patterns
                    df = PatternDetector.detect_all_patterns(df)
                    
                    # Detect wave states
                    df = AdvancedMetrics.detect_wave_state(df)
                    
                    # Store in session state
                    st.session_state.data = df
                    st.session_state.filtered_data = df
                    st.session_state.data_loaded = True
                    st.session_state.last_update = datetime.now()
                    
                    st.success(f"✅ Loaded {len(df)} stocks successfully!")
                else:
                    st.error("Failed to load data")
        
        # Display mode selection
        st.markdown("---")
        st.markdown("### 🎨 Display Mode")
        display_mode = st.radio(
            "Select mode:",
            ["Technical", "Hybrid"],
            index=0 if st.session_state.display_mode == "Technical" else 1
        )
        st.session_state.display_mode = display_mode
        
        if display_mode == "Hybrid":
            st.info("📊 Showing fundamental data (P/E, EPS)")
        
        # Quick Actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        quick_actions = {
            "🚀 Top 100": lambda df: df.nlargest(100, 'master_score'),
            "🔥 Hot Patterns": lambda df: df[df['patterns'] != ''],
            "💎 Hidden Gems": lambda df: df[
                (df['percentile'] >= 70) & 
                (df['rvol'] >= 1.5) & 
                (df['volume_30d'] < df['volume_30d'].quantile(0.5))
            ],
            "📈 Strong Momentum": lambda df: df[df['ret_30d'] > 20],
            "⚡ High RVOL": lambda df: df[df['rvol'] > 2],
            "🌊 Wave Cresting": lambda df: df[df['wave_state'] == 'CRESTING']
        }
        
        for action_name, action_func in quick_actions.items():
            if st.button(action_name, use_container_width=True):
                if st.session_state.data_loaded:
                    st.session_state.filtered_data = action_func(st.session_state.data)
        
        # Filters
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        filters = {}
        
        # Categories
        if st.session_state.data_loaded and 'category' in st.session_state.data.columns:
            categories = sorted(st.session_state.data['category'].dropna().unique())
            filters['categories'] = st.multiselect("Categories", categories)
        
        # Sectors
        if st.session_state.data_loaded and 'sector' in st.session_state.data.columns:
            sectors = sorted(st.session_state.data['sector'].dropna().unique())
            filters['sectors'] = st.multiselect("Sectors", sectors)
        
        # Master Score Range
        filters['score_range'] = st.slider(
            "Master Score Range",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=5
        )
        
        # RVOL Range
        filters['rvol_range'] = st.slider(
            "RVOL Range",
            min_value=0.0,
            max_value=5.0,
            value=(0.0, 5.0),
            step=0.1
        )
        
        # 30D Return Range
        filters['ret_30d_range'] = st.slider(
            "30D Return % Range",
            min_value=-50,
            max_value=100,
            value=(-50, 100),
            step=5
        )
        
        # Trend Strength
        filters['trend_strength'] = st.selectbox(
            "Trend Strength",
            ["All", "Strong Uptrend (60+)", "Weak Uptrend (40-60)", "Downtrend (<40)"]
        )
        
        # Wave Filter
        filters['wave_states'] = st.multiselect(
            "Wave States",
            ["CRESTING", "BUILDING", "FORMING", "BREAKING"],
            help="Filter by wave momentum state"
        )
        
        # Wave Strength Range
        if filters['wave_states']:
            filters['wave_strength_range'] = st.slider(
                "Wave Strength Range",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=5
            )
        
        # Pattern Filter
        filters['has_patterns'] = st.checkbox("Has Patterns Only")
        
        # Market Cap Filter
        filters['market_caps'] = st.multiselect(
            "Market Cap",
            ["Large Cap", "Mid Cap", "Small Cap"]
        )
        
        # Fundamental Filters (Hybrid mode only)
        if st.session_state.display_mode == "Hybrid":
            st.markdown("#### 📊 Fundamental Filters")
            
            filters['pe_range'] = st.slider(
                "P/E Ratio Range",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=5
            )
            
            filters['eps_growth_range'] = st.slider(
                "EPS Growth % Range",
                min_value=-100,
                max_value=500,
                value=(-100, 500),
                step=10
            )
        
        # Apply Filters Button
        if st.button("🔍 Apply Filters", type="primary", use_container_width=True):
            if st.session_state.data_loaded:
                st.session_state.filtered_data = FilterEngine.apply_filters(
                    st.session_state.data, filters
                )
                st.success(f"Filtered to {len(st.session_state.filtered_data)} stocks")
        
        # Reset Filters
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state.filtered_data = st.session_state.data
            st.experimental_rerun()
        
        # Data Quality Indicator
        if st.session_state.data_quality:
            st.markdown("---")
            st.markdown("### 📊 Data Quality")
            quality = st.session_state.data_quality
            
            st.metric("Total Rows", f"{quality['total_rows']:,}")
            st.metric("Valid Rows", f"{quality['valid_rows']:,}")
            st.metric("Completeness", f"{quality['completeness']:.1f}%")
            
            if quality['data_issues']:
                with st.expander("⚠️ Data Issues"):
                    for issue in quality['data_issues']:
                        st.warning(issue)
    
    # Main Content Area
    if not st.session_state.data_loaded:
        st.info("👈 Please load data using the sidebar to begin")
        return
    
    # Get filtered data
    df = st.session_state.filtered_data
    
    # Search Box
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input(
            "🔍 Search by ticker or company name",
            placeholder="e.g., AAPL or Apple",
            help="Search for stocks by ticker symbol or company name"
        )
    
    with col2:
        market_regime = MarketIntelligence.detect_market_regime(df)
        regime_color = {
            "RISK-ON": "#2ECC71",
            "RISK-OFF": "#E74C3C",
            "NEUTRAL": "#F39C12"
        }.get(market_regime, "#808080")
        
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; background-color: {regime_color}; 
                    color: white; border-radius: 5px; margin-top: 20px;'>
            <b>Market: {market_regime}</b>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric("Active Filters", 
                 sum(1 for v in filters.values() if v and v != "All" and v != (0, 100)))
    
    # Search Results
    if search_query:
        search_results = SearchEngine.search_stocks(df, search_query)
        if not search_results.empty:
            st.markdown(f"### Search Results ({len(search_results)} found)")
            
            for idx, stock in search_results.iterrows():
                with st.container():
                    st.markdown('<div class="search-result-card">', unsafe_allow_html=True)
                    
                    # Header
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown(f"## {stock['ticker']} - {stock['company_name']}")
                        if stock['patterns']:
                            UIComponents.render_pattern_pills(stock['patterns'])
                    
                    with col2:
                        st.metric("Master Score", f"{stock['master_score']:.1f}")
                        UIComponents.render_wave_state_badge(stock['wave_state'])
                    
                    with col3:
                        st.metric("Rank", f"#{stock['rank']}")
                        st.caption(f"Top {100 - stock['percentile']:.1f}%")
                    
                    # Detailed metrics in 4 columns
                    st.markdown("#### 📊 Trading Position")
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric("Price", f"${stock['price']:.2f}")
                        st.caption(f"From Low: {stock['from_low_pct']:.0f}%")
                    
                    with metric_cols[1]:
                        st.metric("30D Return", f"{stock['ret_30d']:.1f}%",
                                 delta="↑" if stock['ret_30d'] > 0 else "↓")
                    
                    with metric_cols[2]:
                        st.metric("RVOL", f"{stock['rvol']:.1f}x")
                        st.caption("Volume vs 90D avg")
                    
                    with metric_cols[3]:
                        st.metric("Category %ile", f"{stock.get('category_percentile', 0):.0f}")
                        st.caption(stock['category'])
                    
                    # Trend Analysis in 3 columns
                    st.markdown("#### 📈 Trend Analysis")
                    trend_cols = st.columns(3)
                    
                    with trend_cols[0]:
                        st.metric("Trend Score", f"{stock['trend_score']:.0f}")
                        st.caption(f"SMAs Above: {stock['smas_above']}/3")
                    
                    with trend_cols[1]:
                        st.metric("Momentum", f"{stock['momentum_score']:.0f}")
                        st.caption(f"Harmony: {stock['momentum_harmony']}/4")
                    
                    with trend_cols[2]:
                        st.metric("Quality", f"{stock['trend_quality']:.0f}")
                        st.caption("Trend consistency")
                    
                    # Advanced Metrics in 4 columns
                    st.markdown("#### 💎 Advanced Metrics")
                    adv_cols = st.columns(4)
                    
                    with adv_cols[0]:
                        st.metric("Money Flow", f"${stock['money_flow_millions']:.1f}M")
                    
                    with adv_cols[1]:
                        st.metric("VMI", f"{stock['vmi']:.0f}")
                        st.caption("Volume momentum")
                    
                    with adv_cols[2]:
                        st.metric("Position Tension", f"{stock['position_tension']:.0f}")
                    
                    with adv_cols[3]:
                        st.metric("Breakout Score", f"{stock['breakout_score']:.0f}")
                    
                    # Fundamentals (Hybrid mode)
                    if st.session_state.display_mode == "Hybrid":
                        st.markdown("#### 💰 Fundamentals")
                        fund_cols = st.columns(4)
                        
                        with fund_cols[0]:
                            pe_value = stock['pe'] if pd.notna(stock['pe']) and stock['pe'] > 0 else "N/A"
                            if pe_value != "N/A":
                                st.metric("P/E Ratio", f"{pe_value:.1f}")
                            else:
                                st.metric("P/E Ratio", pe_value)
                        
                        with fund_cols[1]:
                            eps_value = stock['eps_current'] if pd.notna(stock['eps_current']) else "N/A"
                            if eps_value != "N/A":
                                st.metric("EPS", f"${eps_value:.2f}")
                            else:
                                st.metric("EPS", eps_value)
                        
                        with fund_cols[2]:
                            eps_growth = stock['eps_change_pct'] if pd.notna(stock['eps_change_pct']) else "N/A"
                            if eps_growth != "N/A":
                                st.metric("EPS Growth", f"{eps_growth:.0f}%")
                            else:
                                st.metric("EPS Growth", eps_growth)
                        
                        with fund_cols[3]:
                            st.metric("Sector", stock['sector'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"No results found for '{search_query}'")
    
    # Main Tabs
    tabs = st.tabs([
        "📊 Rankings",
        "🌊 Wave Radar",
        "📈 Market Intelligence", 
        "📉 Charts",
        "💾 Export",
        "🏢 Sector Analysis",
        "ℹ️ About"
    ])
    
    # Tab 1: Rankings
    with tabs[0]:
        st.markdown("### 🏆 Master Rankings")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Stocks", f"{len(df):,}")
        
        with col2:
            avg_score = df['master_score'].mean() if not df.empty else 0
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        with col3:
            patterns_count = (df['patterns'] != '').sum() if 'patterns' in df.columns else 0
            st.metric("With Patterns", f"{patterns_count:,}")
        
        with col4:
            high_rvol = (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0
            st.metric("High RVOL (>2x)", f"{high_rvol:,}")
        
        with col5:
            positive_momentum = (df['ret_30d'] > 0).sum() if 'ret_30d' in df.columns else 0
            st.metric("Positive 30D", f"{positive_momentum:,}")
        
        # Trend Distribution Statistics
        if not df.empty and 'trend_score' in df.columns:
            st.markdown("### 📊 Trend Distribution")
            
            trend_cols = st.columns(4)
            
            with trend_cols[0]:
                avg_trend = df['trend_score'].mean()
                st.metric("Average Trend Score", f"{avg_trend:.1f}")
            
            with trend_cols[1]:
                above_all_smas = (df['smas_above'] == 3).sum() if 'smas_above' in df.columns else 0
                st.metric("Above All SMAs", f"{above_all_smas:,}")
            
            with trend_cols[2]:
                uptrend = (df['trend_score'] >= 60).sum()
                st.metric("In Uptrend (60+)", f"{uptrend:,}")
            
            with trend_cols[3]:
                downtrend = (df['trend_score'] < 40).sum()
                st.metric("In Downtrend (<40)", f"{downtrend:,}")
        
        # Quick Statistics
        if not df.empty:
            st.markdown("### 📈 Quick Statistics")
            
            stat_cols = st.columns(5)
            
            # Master Score quartiles
            q1 = df['master_score'].quantile(0.25)
            median = df['master_score'].quantile(0.5)
            q3 = df['master_score'].quantile(0.75)
            
            with stat_cols[0]:
                st.metric("Q1 Score", f"{q1:.1f}")
            
            with stat_cols[1]:
                st.metric("Median Score", f"{median:.1f}")
            
            with stat_cols[2]:
                st.metric("Q3 Score", f"{q3:.1f}")
            
            with stat_cols[3]:
                if 'ret_30d' in df.columns:
                    median_return = df['ret_30d'].median()
                    st.metric("Median 30D Return", f"{median_return:.1f}%")
            
            with stat_cols[4]:
                if 'rvol' in df.columns:
                    median_rvol = df['rvol'].median()
                    st.metric("Median RVOL", f"{median_rvol:.1f}x")
        
        # Display options
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_patterns = st.checkbox("Show Patterns", value=True)
        
        with col2:
            show_fundamentals = st.checkbox("Show Fundamentals", 
                                          value=st.session_state.display_mode == "Hybrid")
        
        with col3:
            rows_per_page = st.selectbox("Rows per page", [25, 50, 100, 200], index=1)
        
        with col4:
            sort_by = st.selectbox("Sort by", ["Master Score", "30D Return", "RVOL", "Category %ile"])
        
        # Prepare display dataframe
        display_cols = [
            'rank', 'ticker', 'company_name', 'master_score', 'tier', 'wave_state',
            'price', 'ret_30d', 'rvol', 'trend_score', 'category', 'category_percentile'
        ]
        
        if show_patterns and 'patterns' in df.columns:
            display_cols.append('patterns')
        
        if show_fundamentals and st.session_state.display_mode == "Hybrid":
            display_cols.extend(['pe', 'eps_change_pct'])
        
        # Add advanced metrics
        display_cols.extend(['money_flow_millions', 'vmi', 'momentum_harmony'])
        
        # Filter to existing columns
        display_cols = [col for col in display_cols if col in df.columns]
        
        # Sort data
        sort_column_map = {
            "Master Score": "master_score",
            "30D Return": "ret_30d",
            "RVOL": "rvol",
            "Category %ile": "category_percentile"
        }
        
        sort_col = sort_column_map.get(sort_by, "master_score")
        display_df = df[display_cols].sort_values(sort_col, ascending=False).head(rows_per_page)
        
        # Format display
        format_dict = {
            'master_score': '{:.1f}',
            'price': '${:.2f}',
            'ret_30d': '{:.1f}%',
            'rvol': '{:.1f}x',
            'trend_score': '{:.0f}',
            'category_percentile': '{:.0f}',
            'money_flow_millions': '${:.1f}M',
            'vmi': '{:.0f}',
            'momentum_harmony': '{}/4',
            'pe': '{:.1f}',
            'eps_change_pct': '{:.0f}%'
        }
        
        # Apply formatting
        styled_df = display_df.style.format(format_dict, na_rep='—')
        
        # Apply color gradients
        styled_df = styled_df.background_gradient(subset=['master_score'], cmap='RdYlGn')
        styled_df = styled_df.background_gradient(subset=['ret_30d'], cmap='RdBu', vmin=-20, vmax=20)
        
        # Display table
        st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Tab 2: Wave Radar
    with tabs[1]:
        st.markdown("### 🌊 Wave Radar - Early Detection System")
        
        # Sensitivity selector
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            sensitivity = st.selectbox(
                "Detection Sensitivity",
                ["Conservative", "Balanced", "Aggressive"],
                index=1
            )
        
        with col2:
            min_score_threshold = {
                "Conservative": 75,
                "Balanced": 65,
                "Aggressive": 55
            }[sensitivity]
            
            st.metric("Min Score Threshold", min_score_threshold)
        
        with col3:
            wave_filter = st.multiselect(
                "Wave States",
                ["CRESTING", "BUILDING", "FORMING"],
                default=["CRESTING", "BUILDING"]
            )
        
        # Filter data for wave radar
        wave_df = df[
            (df['master_score'] >= min_score_threshold) &
            (df['wave_state'].isin(wave_filter))
        ]
        
        if not wave_df.empty:
            # Wave state distribution
            st.markdown("#### 🌊 Wave State Distribution")
            
            wave_counts = wave_df['wave_state'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=wave_counts.index,
                    y=wave_counts.values,
                    marker_color=[CONFIG.WAVE_STATES[state]['color'] for state in wave_counts.index],
                    text=wave_counts.values,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Stocks by Wave State",
                xaxis_title="Wave State",
                yaxis_title="Count",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top momentum shifts
            st.markdown("#### 🚀 Top Momentum Shifts")
            
            momentum_df = wave_df.nlargest(20, 'momentum_score')[
                ['ticker', 'company_name', 'wave_state', 'momentum_score', 
                 'acceleration_score', 'ret_30d', 'rvol', 'patterns']
            ]
            
            st.dataframe(
                momentum_df.style.format({
                    'momentum_score': '{:.0f}',
                    'acceleration_score': '{:.0f}',
                    'ret_30d': '{:.1f}%',
                    'rvol': '{:.1f}x'
                }),
                use_container_width=True
            )
            
            # Pattern emergence
            st.markdown("#### 💎 Pattern Emergence")
            
            pattern_stocks = wave_df[wave_df['patterns'] != '']
            
            if not pattern_stocks.empty:
                # Count patterns
                all_patterns = []
                for patterns in pattern_stocks['patterns']:
                    all_patterns.extend(patterns.split(' | '))
                
                pattern_counts = pd.Series(all_patterns).value_counts().head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=pattern_counts.values,
                        y=pattern_counts.index,
                        orientation='h',
                        marker_color='lightblue',
                        text=pattern_counts.values,
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Most Common Patterns in Wave Stocks",
                    xaxis_title="Count",
                    yaxis_title="Pattern",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Smart money flow by category
            if 'money_flow_millions' in wave_df.columns:
                st.markdown("#### 💰 Smart Money Flow by Category")
                
                category_flow = wave_df.groupby('category')['money_flow_millions'].sum().sort_values(ascending=False).head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=category_flow.index,
                        y=category_flow.values,
                        marker_color='green',
                        text=[f'${x:.0f}M' for x in category_flow.values],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Money Flow by Category (Top 10)",
                    xaxis_title="Category",
                    yaxis_title="Money Flow (Millions $)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stocks found matching wave criteria. Try adjusting sensitivity or filters.")
    
    # Tab 3: Market Intelligence
    with tabs[2]:
        st.markdown("### 📊 Market Intelligence Dashboard")
        
        # Market Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            advances = (df['ret_1d'] > 0).sum() if 'ret_1d' in df.columns else 0
            declines = (df['ret_1d'] < 0).sum() if 'ret_1d' in df.columns else 0
            st.metric("Advance/Decline", f"{advances}/{declines}")
        
        with col2:
            avg_ret_30d = df['ret_30d'].mean() if 'ret_30d' in df.columns else 0
            st.metric("Avg 30D Return", f"{avg_ret_30d:.1f}%")
        
        with col3:
            avg_rvol = df['rvol'].mean() if 'rvol' in df.columns else 0
            st.metric("Avg RVOL", f"{avg_rvol:.2f}x")
        
        with col4:
            high_momentum = (df['momentum_score'] > 70).sum() if 'momentum_score' in df.columns else 0
            st.metric("High Momentum", f"{high_momentum:,}")
        
        # Sector Rotation Analysis
        st.markdown("#### 🔄 Sector Rotation Analysis")
        
        sector_df = MarketIntelligence.detect_sector_rotation(df)
        
        if not sector_df.empty:
            # Format sector analysis table
            sector_display = sector_df[['Sector', 'Total Stocks', 'Sample Size', 
                                       'Avg Score', 'Avg 30D Return', 'Avg RVOL', 
                                       'Top Stock', 'Momentum']]
            
            st.dataframe(
                sector_display.style.format({
                    'Avg Score': '{:.1f}',
                    'Avg 30D Return': '{:.1f}%',
                    'Avg RVOL': '{:.2f}x'
                }).background_gradient(subset=['Avg Score'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Sector performance visualization
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=sector_df['Avg 30D Return'],
                y=sector_df['Avg Score'],
                mode='markers+text',
                marker=dict(
                    size=sector_df['Total Stocks'],
                    sizemode='area',
                    sizeref=2.*max(sector_df['Total Stocks'])/(40.**2),
                    sizemin=4,
                    color=sector_df['Avg RVOL'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg RVOL")
                ),
                text=sector_df['Sector'],
                textposition="top center"
            ))
            
            fig.update_layout(
                title="Sector Performance Map (Bubble size = Total Stocks)",
                xaxis_title="Average 30D Return (%)",
                yaxis_title="Average Master Score",
                height=600,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Category Performance
        if 'category' in df.columns:
            st.markdown("#### 📊 Category Performance")
            
            category_stats = df.groupby('category').agg({
                'master_score': ['mean', 'count'],
                'ret_30d': 'mean',
                'rvol': 'mean'
            }).round(2)
            
            category_stats.columns = ['Avg Score', 'Count', 'Avg 30D Return', 'Avg RVOL']
            category_stats = category_stats.sort_values('Avg Score', ascending=False).head(20)
            
            st.dataframe(
                category_stats.style.format({
                    'Avg Score': '{:.1f}',
                    'Avg 30D Return': '{:.1f}%',
                    'Avg RVOL': '{:.2f}x'
                }).background_gradient(subset=['Avg Score'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        # Volume Analysis
        st.markdown("#### 📊 Volume Analysis")
        
        vol_cols = st.columns(3)
        
        with vol_cols[0]:
            extreme_vol = (df['rvol'] > 3).sum() if 'rvol' in df.columns else 0
            st.metric("Extreme Volume (>3x)", f"{extreme_vol:,}")
        
        with vol_cols[1]:
            if 'money_flow_millions' in df.columns:
                total_flow = df['money_flow_millions'].sum()
                st.metric("Total Money Flow", f"${total_flow:,.0f}M")
        
        with vol_cols[2]:
            if 'vmi' in df.columns:
                high_vmi = (df['vmi'] > 70).sum()
                st.metric("High VMI (>70)", f"{high_vmi:,}")
    
    # Tab 4: Charts
    with tabs[3]:
        st.markdown("### 📉 Visual Analytics")
        
        # Score Distribution
        st.markdown("#### 📊 Master Score Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['master_score'],
            nbinsx=20,
            marker_color='lightblue',
            name='Master Score'
        ))
        
        fig.update_layout(
            title="Master Score Distribution",
            xaxis_title="Master Score",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Return vs Score Scatter
        if 'ret_30d' in df.columns:
            st.markdown("#### 📈 Return vs Score Analysis")
            
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=df['master_score'],
                y=df['ret_30d'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df['rvol'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="RVOL"),
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                text=df['ticker'],
                hovertemplate='<b>%{text}</b><br>Score: %{x:.1f}<br>30D Return: %{y:.1f}%<br>'
            ))
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ))
            
            fig.update_layout(
                title="30D Return vs Master Score (Color = RVOL)",
                xaxis_title="Master Score",
                yaxis_title="30D Return (%)",
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Wave State Sunburst
        st.markdown("#### 🌊 Wave State Composition")
        
        # Prepare hierarchical data
        wave_data = []
        for tier in df['tier'].unique():
            tier_df = df[df['tier'] == tier]
            for state in tier_df['wave_state'].unique():
                count = len(tier_df[tier_df['wave_state'] == state])
                wave_data.append({
                    'tier': tier,
                    'state': state,
                    'count': count
                })
        
        wave_df_chart = pd.DataFrame(wave_data)
        
        fig = px.sunburst(
            wave_df_chart,
            path=['tier', 'state'],
            values='count',
            color='count',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            title="Stock Distribution by Tier and Wave State",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top Categories Performance
        if 'category' in df.columns:
            st.markdown("#### 🏆 Top Categories by Average Score")
            
            top_categories = df.groupby('category')['master_score'].agg(['mean', 'count'])
            top_categories = top_categories[top_categories['count'] >= 5]  # At least 5 stocks
            top_categories = top_categories.sort_values('mean', ascending=False).head(15)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_categories.index,
                y=top_categories['mean'],
                text=[f"{score:.1f}<br>({count} stocks)" 
                      for score, count in zip(top_categories['mean'], top_categories['count'])],
                textposition='auto',
                marker_color=top_categories['mean'],
                marker_colorscale='RdYlGn',
                marker_showscale=True,
                marker_colorbar_title="Avg Score"
            ))
            
            fig.update_layout(
                title="Top 15 Categories by Average Master Score (Min 5 stocks)",
                xaxis_title="Category",
                yaxis_title="Average Master Score",
                height=500,
                showlegend=False
            )
            
            fig.update_xaxis(tickangle=-45)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Export
    with tabs[4]:
        st.markdown("### 💾 Export Center")
        
        # Export templates
        st.markdown("#### 📋 Export Templates")
        
        template = st.selectbox(
            "Select Export Template",
            [
                "Full Analysis (All Columns)",
                "Trading Focus (Key Metrics)",
                "Pattern Stocks Only",
                "Top 100 Elite",
                "Wave Momentum (Cresting/Building)",
                "High RVOL Alert (>2x)",
                "Custom Selection"
            ]
        )
        
        # Apply template
        export_df = df.copy()
        
        if template == "Trading Focus (Key Metrics)":
            export_cols = ['ticker', 'company_name', 'master_score', 'wave_state',
                          'price', 'ret_30d', 'rvol', 'patterns', 'category']
            export_df = export_df[export_cols]
        
        elif template == "Pattern Stocks Only":
            export_df = export_df[export_df['patterns'] != '']
        
        elif template == "Top 100 Elite":
            export_df = export_df.nlargest(100, 'master_score')
        
        elif template == "Wave Momentum (Cresting/Building)":
            export_df = export_df[export_df['wave_state'].isin(['CRESTING', 'BUILDING'])]
        
        elif template == "High RVOL Alert (>2x)":
            export_df = export_df[export_df['rvol'] > 2]
        
        elif template == "Custom Selection":
            available_cols = st.multiselect(
                "Select columns to export",
                df.columns.tolist(),
                default=CONFIG.EXPORT_COLUMNS[:10]
            )
            if available_cols:
                export_df = export_df[available_cols]
        
        # Export preview
        st.markdown("#### 👁️ Export Preview")
        st.write(f"**Rows:** {len(export_df):,} | **Columns:** {len(export_df.columns)}")
        
        st.dataframe(export_df.head(10), use_container_width=True)
        
        # Download buttons
        st.markdown("#### 📥 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = ExportEngine.create_csv_export(export_df)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"wave_detection_{template.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel export would go here
            st.button("📊 Download Excel (Coming Soon)", disabled=True, use_container_width=True)
        
        with col3:
            # JSON export
            json_data = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="🔧 Download JSON",
                data=json_data,
                file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Export statistics
        st.markdown("#### 📊 Export Summary")
        
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            st.metric("Total Stocks", f"{len(export_df):,}")
        
        with summary_cols[1]:
            avg_score = export_df['master_score'].mean() if 'master_score' in export_df.columns else 0
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        with summary_cols[2]:
            with_patterns = (export_df['patterns'] != '').sum() if 'patterns' in export_df.columns else 0
            st.metric("With Patterns", f"{with_patterns:,}")
        
        with summary_cols[3]:
            if 'ret_30d' in export_df.columns:
                positive_ret = (export_df['ret_30d'] > 0).sum()
                st.metric("Positive Returns", f"{positive_ret:,}")
    
    # Tab 6: Sector Analysis
    with tabs[5]:
        st.markdown("### 🏢 Sector Deep Dive")
        
        if 'sector' not in df.columns:
            st.warning("Sector data not available")
            return
        
        # Sector Overview
        st.markdown("#### 📊 Sector Overview")
        
        sector_summary = MarketIntelligence.detect_sector_rotation(df)
        
        if not sector_summary.empty:
            # Display sector summary table
            st.dataframe(
                sector_summary.style.format({
                    'Avg Score': '{:.1f}',
                    'Avg 30D Return': '{:.1f}%',
                    'Avg RVOL': '{:.2f}x'
                }).background_gradient(subset=['Avg Score'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
            
            # Sector selector for deep dive
            st.markdown("#### 🔍 Sector Deep Dive")
            
            selected_sector = st.selectbox(
                "Select a sector to analyze",
                sector_summary['Sector'].tolist()
            )
            
            if selected_sector:
                # Get sector data
                sector_stocks = df[df['sector'] == selected_sector]
                
                # Sector metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Stocks", f"{len(sector_stocks):,}")
                
                with col2:
                    avg_score = sector_stocks['master_score'].mean()
                    st.metric("Avg Master Score", f"{avg_score:.1f}")
                
                with col3:
                    avg_ret = sector_stocks['ret_30d'].mean() if 'ret_30d' in sector_stocks.columns else 0
                    st.metric("Avg 30D Return", f"{avg_ret:.1f}%")
                
                with col4:
                    with_patterns = (sector_stocks['patterns'] != '').sum() if 'patterns' in sector_stocks.columns else 0
                    st.metric("Stocks with Patterns", f"{with_patterns:,}")
                
                # Top 10 stocks in sector
                st.markdown(f"#### 🏆 Top 10 Stocks in {selected_sector}")
                
                top_sector_stocks = sector_stocks.nlargest(10, 'master_score')
                
                display_cols = ['rank', 'ticker', 'company_name', 'master_score', 
                               'wave_state', 'price', 'ret_30d', 'rvol', 'patterns']
                
                if st.session_state.display_mode == 'Hybrid':
                    display_cols.extend(['pe', 'eps_change_pct'])
                
                display_cols = [col for col in display_cols if col in top_sector_stocks.columns]
                
                st.dataframe(
                    top_sector_stocks[display_cols].style.format({
                        'master_score': '{:.1f}',
                        'price': '${:.2f}',
                        'ret_30d': '{:.1f}%',
                        'rvol': '{:.1f}x',
                        'pe': '{:.1f}',
                        'eps_change_pct': '{:.0f}%'
                    }).background_gradient(subset=['master_score'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Sector distribution charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=sector_stocks['master_score'],
                        nbinsx=15,
                        marker_color='lightblue',
                        name='Master Score'
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_sector} - Score Distribution",
                        xaxis_title="Master Score",
                        yaxis_title="Count",
                        height=350,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Return distribution
                    if 'ret_30d' in sector_stocks.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=sector_stocks['ret_30d'],
                            marker_color='lightgreen',
                            name='30D Return'
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_sector} - 30D Return Distribution",
                            yaxis_title="30D Return (%)",
                            height=350,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Pattern analysis for sector
                if 'patterns' in sector_stocks.columns:
                    pattern_stocks = sector_stocks[sector_stocks['patterns'] != '']
                    if not pattern_stocks.empty:
                        st.markdown(f"#### 💎 Common Patterns in {selected_sector}")
                        
                        # Extract and count patterns
                        all_patterns = []
                        for patterns in pattern_stocks['patterns']:
                            all_patterns.extend(patterns.split(' | '))
                        
                        pattern_counts = pd.Series(all_patterns).value_counts().head(10)
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=pattern_counts.values,
                                y=pattern_counts.index,
                                orientation='h',
                                marker_color='purple',
                                text=pattern_counts.values,
                                textposition='auto',
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"Top Patterns in {selected_sector}",
                            xaxis_title="Count",
                            yaxis_title="Pattern",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 7: About
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
            The FINAL production version of the most advanced stock momentum detection system.
            This tool combines technical analysis, volume dynamics, and pattern recognition
            to identify high-potential stocks before they peak.
            
            #### 🎯 Core Features
            
            **Master Score 3.0** - Proprietary ranking algorithm:
            - Position Analysis (30%)
            - Volume Dynamics (25%)
            - Momentum Tracking (15%)
            - Acceleration Detection (10%)
            - Breakout Probability (10%)
            - RVOL Integration (10%)
            
            **25 Pattern Detection System**:
            - 11 Technical patterns
            - 6 Range patterns
            - 3 Intelligence patterns
            - 5 Fundamental patterns (Hybrid mode)
            
            **Wave State Classification**:
            - CRESTING: Peak momentum (85+ score)
            - BUILDING: Strong momentum (70+ score)
            - FORMING: Developing momentum (55+ score)
            - BREAKING: Weak momentum (<55 score)
            
            #### 📊 Data Processing
            
            1. Load from Google Sheets or CSV
            2. Validate and clean 41 columns
            3. Calculate technical indicators
            4. Generate component scores
            5. Compute Master Score 3.0
            6. Detect all patterns
            7. Classify wave states
            8. Rank and categorize
            
            #### ⚡ Performance
            
            - Initial load: <2 seconds
            - Pattern detection: <500ms
            - Filtering: <200ms
            - Search: <50ms
            - Export: <1 second
            """)
        
        with col2:
            st.markdown("""
            #### 🔧 Technical Details
            
            **Version**: 3.0-FINAL
            **Status**: PRODUCTION
            **Updates**: LOCKED
            
            #### 📈 Pattern Groups
            
            **Technical**:
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
            
            **Range**:
            - 🎯 52W HIGH APPROACH
            - 🔄 52W LOW BOUNCE
            - 👑 GOLDEN ZONE
            - 📊 VOL ACCUMULATION
            - 🔀 MOMENTUM DIVERGE
            - 🎯 RANGE COMPRESS
            
            **Intelligence**:
            - 🤫 STEALTH
            - 🧛 VAMPIRE
            - ⛈️ PERFECT STORM
            
            **Fundamental**:
            - 💎 VALUE MOMENTUM
            - 📊 EARNINGS ROCKET
            - 🏆 QUALITY LEADER
            - ⚡ TURNAROUND
            - ⚠️ HIGH PE
            
            #### 💡 Tips
            
            - Use Quick Actions for instant filtering
            - Combine filters for precision
            - Export templates save time
            - Wave Radar finds early movers
            - Check sector rotation daily
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        Wave Detection Ultimate 3.0 - Final Production Version<br>
        No further updates will be made - All features are permanent<br>
        Built for professional traders requiring reliable momentum detection
        </div>
        """, unsafe_allow_html=True)

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
