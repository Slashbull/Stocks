"""
Wave Detection Ultimate 3.0 - Core Engine
==========================================
Production-Ready Mathematical Heart & Business Logic
All scoring algorithms, pattern detection, and configuration

Version: 3.0.6-PRODUCTION-BULLETPROOF
Status: PRODUCTION READY - Zero Bug Tolerance
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from functools import lru_cache
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure logging
logger = logging.getLogger(__name__)

# ============================================
# PRODUCTION CONFIGURATION
# ============================================

@dataclass(frozen=True)
class ProductionConfig:
    """Production-grade configuration with bulletproof validation"""
    
    # Data source configuration
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM/edit?usp=sharing"
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings optimized for production
    CACHE_TTL: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 128  # Maximum cache entries
    
    # Master Score 3.0 weights (validated to sum to 1.0)
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Pattern thresholds (optimized for signal quality)
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
        "long_strength": 80
    })
    
    # Tier definitions (fixed boundary overlaps)
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
    MAX_VALID_PE: float = 10000.0
    MAX_VALID_EPS_CHANGE: float = 100000.0
    
    # Quick action thresholds
    TOP_GAINER_MOMENTUM: float = 80
    VOLUME_SURGE_RVOL: float = 3.0
    BREAKOUT_READY_SCORE: float = 80
    
    def __post_init__(self):
        """Validate configuration on initialization"""
        # Validate weights sum to 1.0
        total_weight = (
            self.POSITION_WEIGHT + self.VOLUME_WEIGHT + self.MOMENTUM_WEIGHT +
            self.ACCELERATION_WEIGHT + self.BREAKOUT_WEIGHT + self.RVOL_WEIGHT
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

# Global configuration instance
CONFIG = ProductionConfig()

# ============================================
# SAFE MATHEMATICAL OPERATIONS
# ============================================

class SafeMath:
    """Production-grade mathematical operations with bulletproof error handling"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with proper error handling"""
        try:
            if pd.isna(numerator) or pd.isna(denominator):
                return default
            if abs(denominator) < 1e-10:  # Avoid division by very small numbers
                return default
            result = numerator / denominator
            if np.isinf(result) or np.isnan(result):
                return default
            return result
        except (ZeroDivisionError, TypeError, OverflowError):
            return default
    
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """Safe float conversion with comprehensive error handling"""
        if value is None or value == '':
            return default
        try:
            if pd.isna(value):
                return default
            result = float(value)
            if np.isinf(result) or np.isnan(result):
                return default
            return result
        except (ValueError, TypeError, OverflowError):
            return default
    
    @staticmethod
    def safe_percentage(value: float, default: float = 0.0) -> float:
        """Safe percentage handling with bounds checking"""
        safe_val = SafeMath.safe_float(value, default)
        # Clamp extreme percentages to reasonable bounds
        return max(-1000, min(10000, safe_val))
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Bulletproof ranking with comprehensive error handling"""
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        try:
            # Create a copy to avoid modifying original
            series_clean = series.copy()
            
            # Replace inf values with NaN
            series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
            
            # Count valid values
            valid_count = series_clean.notna().sum()
            if valid_count == 0:
                # Return neutral scores with small random variation
                return pd.Series(
                    50 + np.random.uniform(-2, 2, size=len(series)), 
                    index=series.index
                )
            
            # Calculate ranks
            if pct:
                ranks = series_clean.rank(pct=True, ascending=ascending, na_option='bottom')
                ranks = ranks * 100
                # Fill NaN values appropriately
                ranks = ranks.fillna(0 if ascending else 100)
            else:
                ranks = series_clean.rank(ascending=ascending, method='min', na_option='bottom')
                ranks = ranks.fillna(valid_count + 1)
            
            # Ensure ranks are within bounds
            if pct:
                ranks = ranks.clip(0, 100)
            
            return ranks
            
        except Exception as e:
            logger.error(f"Error in safe_rank: {str(e)}")
            # Return neutral scores on any error
            return pd.Series(50, index=series.index, dtype=float)

# ============================================
# CORE SCORING ENGINE
# ============================================

class ProductionScoringEngine:
    """Production-grade scoring engine with bulletproof calculations"""
    
    @staticmethod
    @lru_cache(maxsize=CONFIG.MAX_CACHE_SIZE)
    def calculate_position_score(from_low_pct: Tuple[float, ...], from_high_pct: Tuple[float, ...]) -> Tuple[float, ...]:
        """Calculate position score with caching and error handling"""
        try:
            # Convert tuples back to series for calculation
            from_low = pd.Series(from_low_pct)
            from_high = pd.Series(from_high_pct)
            
            # Validate inputs
            from_low = from_low.apply(lambda x: SafeMath.safe_percentage(x, 50))
            from_high = from_high.apply(lambda x: SafeMath.safe_percentage(x, -50))
            
            # Calculate rank components
            rank_from_low = SafeMath.safe_rank(from_low, pct=True, ascending=True)
            rank_from_high = SafeMath.safe_rank(from_high, pct=True, ascending=False)
            
            # Combined position score
            position_score = (rank_from_low * 0.6 + rank_from_high * 0.4).clip(0, 100)
            
            return tuple(position_score)
            
        except Exception as e:
            logger.error(f"Error in position score calculation: {str(e)}")
            return tuple([50.0] * len(from_low_pct))
    
    @staticmethod
    def calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive volume score with error handling"""
        try:
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
            
            for col, weight in vol_cols:
                if col in df.columns:
                    col_data = df[col].copy()
                    # Safe data cleaning
                    col_data = col_data.apply(lambda x: SafeMath.safe_float(x, 1.0))
                    col_data = col_data.clip(lower=0.1)  # Prevent negative ratios
                    
                    col_rank = SafeMath.safe_rank(col_data, pct=True, ascending=True)
                    weighted_score += col_rank * weight
                    total_weight += weight
            
            if total_weight > 0:
                volume_score = (weighted_score / total_weight).clip(0, 100)
            
            return volume_score
            
        except Exception as e:
            logger.error(f"Error in volume score calculation: {str(e)}")
            return pd.Series(50, index=df.index, dtype=float)
    
    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score with fallback logic"""
        try:
            momentum_score = pd.Series(50, index=df.index, dtype=float)
            
            # Primary: Use 30-day returns
            if 'ret_30d' in df.columns and df['ret_30d'].notna().sum() > 0:
                ret_30d = df['ret_30d'].apply(lambda x: SafeMath.safe_percentage(x, 0))
                momentum_score = SafeMath.safe_rank(ret_30d, pct=True, ascending=True)
                
                # Consistency bonus if 7-day data available
                if 'ret_7d' in df.columns:
                    ret_7d = df['ret_7d'].apply(lambda x: SafeMath.safe_percentage(x, 0))
                    
                    # Both positive momentum
                    both_positive = (ret_7d > 0) & (ret_30d > 0)
                    momentum_score.loc[both_positive] += 5
                    
                    # Accelerating momentum
                    daily_7d = ret_7d / 7
                    daily_30d = ret_30d / 30
                    accelerating = both_positive & (daily_7d > daily_30d)
                    momentum_score.loc[accelerating] += 5
                    
                    momentum_score = momentum_score.clip(0, 100)
            
            # Fallback: Use 7-day returns
            elif 'ret_7d' in df.columns and df['ret_7d'].notna().sum() > 0:
                ret_7d = df['ret_7d'].apply(lambda x: SafeMath.safe_percentage(x, 0))
                momentum_score = SafeMath.safe_rank(ret_7d, pct=True, ascending=True)
                
            return momentum_score
            
        except Exception as e:
            logger.error(f"Error in momentum score calculation: {str(e)}")
            return pd.Series(50, index=df.index, dtype=float)
    
    @staticmethod
    def calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate acceleration with bulletproof math"""
        try:
            acceleration_score = pd.Series(50, index=df.index, dtype=float)
            
            req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
            available_cols = [col for col in req_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return acceleration_score
            
            # Safe data extraction
            ret_1d = df['ret_1d'].apply(lambda x: SafeMath.safe_percentage(x, 0)) if 'ret_1d' in df.columns else pd.Series(0, index=df.index)
            ret_7d = df['ret_7d'].apply(lambda x: SafeMath.safe_percentage(x, 0)) if 'ret_7d' in df.columns else pd.Series(0, index=df.index)
            ret_30d = df['ret_30d'].apply(lambda x: SafeMath.safe_percentage(x, 0)) if 'ret_30d' in df.columns else pd.Series(0, index=df.index)
            
            # Calculate daily averages safely
            avg_daily_1d = ret_1d
            avg_daily_7d = ret_7d.apply(lambda x: SafeMath.safe_divide(x, 7, 0))
            avg_daily_30d = ret_30d.apply(lambda x: SafeMath.safe_divide(x, 30, 0))
            
            if len(available_cols) >= 3:
                # Perfect acceleration
                perfect = (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
                acceleration_score.loc[perfect] = 100
                
                # Good acceleration
                good = (~perfect) & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0)
                acceleration_score.loc[good] = 80
                
                # Moderate
                moderate = (~perfect) & (~good) & (ret_1d > 0)
                acceleration_score.loc[moderate] = 60
                
                # Deceleration levels
                slight_decel = (ret_1d <= 0) & (ret_7d > 0)
                acceleration_score.loc[slight_decel] = 40
                
                strong_decel = (ret_1d <= 0) & (ret_7d <= 0)
                acceleration_score.loc[strong_decel] = 20
            
            return acceleration_score
            
        except Exception as e:
            logger.error(f"Error in acceleration score calculation: {str(e)}")
            return pd.Series(50, index=df.index, dtype=float)
    
    @staticmethod
    def calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout probability with safe math"""
        try:
            breakout_score = pd.Series(50, index=df.index, dtype=float)
            
            # Distance from high (40% weight)
            distance_factor = pd.Series(50, index=df.index)
            if 'from_high_pct' in df.columns:
                from_high = df['from_high_pct'].apply(lambda x: SafeMath.safe_percentage(x, -50))
                distance_from_high = -from_high  # Convert to positive distance
                distance_factor = (100 - distance_from_high).clip(0, 100)
            
            # Volume surge (40% weight)
            volume_factor = pd.Series(50, index=df.index)
            if 'vol_ratio_7d_90d' in df.columns:
                vol_ratio = df['vol_ratio_7d_90d'].apply(lambda x: SafeMath.safe_float(x, 1.0))
                volume_factor = ((vol_ratio - 1) * 100).clip(0, 100)
            
            # Trend support (20% weight)
            trend_factor = pd.Series(0, index=df.index, dtype=float)
            if 'price' in df.columns:
                current_price = df['price'].apply(lambda x: SafeMath.safe_float(x, 0))
                trend_count = 0
                
                for sma_col, weight in [('sma_20d', 33.33), ('sma_50d', 33.33), ('sma_200d', 33.34)]:
                    if sma_col in df.columns:
                        sma_values = df[sma_col].apply(lambda x: SafeMath.safe_float(x, 0))
                        above_sma = (current_price > sma_values) & (sma_values > 0)
                        trend_factor += above_sma.astype(float) * weight
                        trend_count += 1
                
                if trend_count > 0 and trend_count < 3:
                    trend_factor = trend_factor * (3 / trend_count)
            
            trend_factor = trend_factor.clip(0, 100)
            
            # Combined breakout score
            breakout_score = (
                distance_factor * 0.4 +
                volume_factor * 0.4 +
                trend_factor * 0.2
            ).clip(0, 100)
            
            return breakout_score
            
        except Exception as e:
            logger.error(f"Error in breakout score calculation: {str(e)}")
            return pd.Series(50, index=df.index, dtype=float)
    
    @staticmethod
    def calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate RVOL-based score with safe handling of extreme values"""
        try:
            if 'rvol' not in df.columns:
                return pd.Series(50, index=df.index)
            
            rvol = df['rvol'].apply(lambda x: SafeMath.safe_float(x, 1.0))
            rvol_score = pd.Series(50, index=df.index, dtype=float)
            
            # Handle extreme RVOL values safely
            rvol_score.loc[rvol > 1000] = 100
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
            
        except Exception as e:
            logger.error(f"Error in RVOL score calculation: {str(e)}")
            return pd.Series(50, index=df.index)

# ============================================
# PATTERN DETECTION ENGINE
# ============================================

class PatternDetectionEngine:
    """Production-grade pattern detection with bulletproof logic"""
    
    @staticmethod
    def detect_all_patterns(df: pd.DataFrame) -> pd.Series:
        """Detect all patterns with comprehensive error handling"""
        try:
            # Initialize pattern column
            patterns = pd.Series('', index=df.index, dtype=str)
            
            # Define pattern detection functions
            pattern_detectors = [
                PatternDetectionEngine._detect_category_leader,
                PatternDetectionEngine._detect_hidden_gem,
                PatternDetectionEngine._detect_accelerating,
                PatternDetectionEngine._detect_volume_explosion,
                PatternDetectionEngine._detect_breakout_ready,
                PatternDetectionEngine._detect_market_leader,
                PatternDetectionEngine._detect_momentum_wave,
                PatternDetectionEngine._detect_value_momentum,
                PatternDetectionEngine._detect_earnings_rocket,
            ]
            
            # Apply each detector safely
            for detector in pattern_detectors:
                try:
                    pattern_name, mask = detector(df)
                    if mask.any():
                        current_patterns = patterns.loc[mask]
                        patterns.loc[mask] = current_patterns.apply(
                            lambda x: f"{x} | {pattern_name}" if x else pattern_name
                        )
                except Exception as e:
                    logger.error(f"Error in pattern detector {detector.__name__}: {str(e)}")
                    continue
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            return pd.Series('', index=df.index, dtype=str)
    
    @staticmethod
    def _detect_category_leader(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect category leaders"""
        if 'category_percentile' not in df.columns:
            return '🔥 CAT LEADER', pd.Series(False, index=df.index)
        
        threshold = CONFIG.PATTERN_THRESHOLDS['category_leader']
        mask = df['category_percentile'] >= threshold
        return '🔥 CAT LEADER', mask
    
    @staticmethod
    def _detect_hidden_gem(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect hidden gems"""
        if not all(col in df.columns for col in ['category_percentile', 'percentile']):
            return '💎 HIDDEN GEM', pd.Series(False, index=df.index)
        
        mask = (
            (df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
            (df['percentile'] < 70)
        )
        return '💎 HIDDEN GEM', mask
    
    @staticmethod
    def _detect_accelerating(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect accelerating stocks"""
        if 'acceleration_score' not in df.columns:
            return '🚀 ACCELERATING', pd.Series(False, index=df.index)
        
        mask = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
        return '🚀 ACCELERATING', mask
    
    @staticmethod
    def _detect_volume_explosion(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect volume explosions"""
        if 'rvol' not in df.columns:
            return '⚡ VOL EXPLOSION', pd.Series(False, index=df.index)
        
        rvol_safe = df['rvol'].apply(lambda x: SafeMath.safe_float(x, 1.0))
        mask = rvol_safe > 3
        return '⚡ VOL EXPLOSION', mask
    
    @staticmethod
    def _detect_breakout_ready(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect breakout ready stocks"""
        if 'breakout_score' not in df.columns:
            return '🎯 BREAKOUT', pd.Series(False, index=df.index)
        
        mask = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
        return '🎯 BREAKOUT', mask
    
    @staticmethod
    def _detect_market_leader(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect market leaders"""
        if 'percentile' not in df.columns:
            return '👑 MARKET LEADER', pd.Series(False, index=df.index)
        
        mask = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
        return '👑 MARKET LEADER', mask
    
    @staticmethod
    def _detect_momentum_wave(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect momentum waves"""
        if not all(col in df.columns for col in ['momentum_score', 'acceleration_score']):
            return '🌊 MOMENTUM WAVE', pd.Series(False, index=df.index)
        
        mask = (
            (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
            (df['acceleration_score'] >= 70)
        )
        return '🌊 MOMENTUM WAVE', mask
    
    @staticmethod
    def _detect_value_momentum(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect value momentum stocks"""
        if not all(col in df.columns for col in ['pe', 'master_score']):
            return '💎 VALUE MOMENTUM', pd.Series(False, index=df.index)
        
        pe_safe = df['pe'].apply(lambda x: SafeMath.safe_float(x, float('inf')))
        valid_pe = (pe_safe > 0) & (pe_safe < CONFIG.MAX_VALID_PE)
        mask = valid_pe & (pe_safe < 15) & (df['master_score'] >= 70)
        return '💎 VALUE MOMENTUM', mask
    
    @staticmethod
    def _detect_earnings_rocket(df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """Detect earnings rockets"""
        if not all(col in df.columns for col in ['eps_change_pct', 'acceleration_score']):
            return '📊 EARNINGS ROCKET', pd.Series(False, index=df.index)
        
        eps_change = df['eps_change_pct'].apply(lambda x: SafeMath.safe_percentage(x, 0))
        extreme_growth = eps_change > 1000  # >1000%
        normal_growth = (eps_change > 50) & (eps_change <= 1000)
        
        mask = (
            (extreme_growth & (df['acceleration_score'] >= 80)) |
            (normal_growth & (df['acceleration_score'] >= 70))
        )
        return '📊 EARNINGS ROCKET', mask

# ============================================
# TIER CLASSIFICATION ENGINE
# ============================================

class TierClassificationEngine:
    """Production-grade tier classification with bulletproof logic"""
    
    @staticmethod
    def classify_all_tiers(df: pd.DataFrame) -> pd.DataFrame:
        """Add all tier classifications"""
        try:
            df = df.copy()
            
            # EPS tier
            if 'eps_current' in df.columns:
                df['eps_tier'] = df['eps_current'].apply(
                    lambda x: TierClassificationEngine._classify_eps_tier(x)
                )
            else:
                df['eps_tier'] = 'Unknown'
            
            # PE tier
            if 'pe' in df.columns:
                df['pe_tier'] = df['pe'].apply(
                    lambda x: TierClassificationEngine._classify_pe_tier(x)
                )
            else:
                df['pe_tier'] = 'Unknown'
            
            # Price tier
            if 'price' in df.columns:
                df['price_tier'] = df['price'].apply(
                    lambda x: TierClassificationEngine._classify_price_tier(x)
                )
            else:
                df['price_tier'] = 'Unknown'
            
            return df
            
        except Exception as e:
            logger.error(f"Error in tier classification: {str(e)}")
            return df
    
    @staticmethod
    def _classify_eps_tier(value: Any) -> str:
        """Classify EPS into tiers"""
        safe_value = SafeMath.safe_float(value, float('nan'))
        if pd.isna(safe_value):
            return "Unknown"
        
        for tier_name, (min_val, max_val) in CONFIG.TIERS['eps'].items():
            if min_val <= safe_value < max_val:
                return tier_name
            if max_val == float('inf') and safe_value >= min_val:
                return tier_name
        return "Unknown"
    
    @staticmethod
    def _classify_pe_tier(value: Any) -> str:
        """Classify PE into tiers"""
        safe_value = SafeMath.safe_float(value, float('nan'))
        if pd.isna(safe_value) or safe_value <= 0:
            return "Negative/NA"
        
        for tier_name, (min_val, max_val) in CONFIG.TIERS['pe'].items():
            if tier_name == "Negative/NA":
                continue
            if min_val <= safe_value < max_val:
                return tier_name
            if max_val == float('inf') and safe_value >= min_val:
                return tier_name
        return "Unknown"
    
    @staticmethod
    def _classify_price_tier(value: Any) -> str:
        """Classify price into tiers"""
        safe_value = SafeMath.safe_float(value, float('nan'))
        if pd.isna(safe_value) or safe_value < CONFIG.MIN_VALID_PRICE:
            return "Unknown"
        
        for tier_name, (min_val, max_val) in CONFIG.TIERS['price'].items():
            if min_val <= safe_value < max_val:
                return tier_name
            if max_val == float('inf') and safe_value >= min_val:
                return tier_name
        return "Unknown"

# ============================================
# MASTER SCORING ENGINE
# ============================================

class MasterScoringEngine:
    """Production-grade master scoring with comprehensive validation"""
    
    @staticmethod
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all scores with bulletproof error handling"""
        try:
            if df.empty:
                return df
            
            logger.info("Starting master scoring calculations...")
            df = df.copy()
            
            # Calculate component scores with error handling
            scoring_functions = [
                ('position_score', MasterScoringEngine._calculate_position_score),
                ('volume_score', ProductionScoringEngine.calculate_volume_score),
                ('momentum_score', ProductionScoringEngine.calculate_momentum_score),
                ('acceleration_score', ProductionScoringEngine.calculate_acceleration_score),
                ('breakout_score', ProductionScoringEngine.calculate_breakout_score),
                ('rvol_score', ProductionScoringEngine.calculate_rvol_score),
            ]
            
            for score_name, score_func in scoring_functions:
                try:
                    df[score_name] = score_func(df)
                    # Ensure scores are within bounds
                    df[score_name] = df[score_name].clip(0, 100).fillna(50)
                except Exception as e:
                    logger.error(f"Error calculating {score_name}: {str(e)}")
                    df[score_name] = 50
            
            # Calculate master score
            df = MasterScoringEngine._calculate_master_score(df)
            
            # Calculate ranks
            df = MasterScoringEngine._calculate_ranks(df)
            
            # Add tier classifications
            df = TierClassificationEngine.classify_all_tiers(df)
            
            # Detect patterns
            df['patterns'] = PatternDetectionEngine.detect_all_patterns(df)
            
            logger.info(f"Master scoring complete: {len(df)} stocks processed")
            return df
            
        except Exception as e:
            logger.error(f"Critical error in master scoring: {str(e)}")
            # Return original dataframe with minimal scoring
            df['master_score'] = 50
            df['rank'] = range(1, len(df) + 1)
            df['percentile'] = 50
            df['patterns'] = ''
            return df
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score with bulletproof handling"""
        try:
            position_score = pd.Series(50, index=df.index, dtype=float)
            
            # Get position data safely
            from_low = df.get('from_low_pct', pd.Series(50, index=df.index))
            from_high = df.get('from_high_pct', pd.Series(-50, index=df.index))
            
            from_low = from_low.apply(lambda x: SafeMath.safe_percentage(x, 50))
            from_high = from_high.apply(lambda x: SafeMath.safe_percentage(x, -50))
            
            # Calculate rankings
            rank_from_low = SafeMath.safe_rank(from_low, pct=True, ascending=True)
            rank_from_high = SafeMath.safe_rank(from_high, pct=True, ascending=False)
            
            # Combined score
            position_score = (rank_from_low * 0.6 + rank_from_high * 0.4).clip(0, 100)
            
            return position_score
            
        except Exception as e:
            logger.error(f"Error in position score calculation: {str(e)}")
            return pd.Series(50, index=df.index, dtype=float)
    
    @staticmethod
    def _calculate_master_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate master score with validated weights"""
        try:
            components = {
                'position_score': CONFIG.POSITION_WEIGHT,
                'volume_score': CONFIG.VOLUME_WEIGHT,
                'momentum_score': CONFIG.MOMENTUM_WEIGHT,
                'acceleration_score': CONFIG.ACCELERATION_WEIGHT,
                'breakout_score': CONFIG.BREAKOUT_WEIGHT,
                'rvol_score': CONFIG.RVOL_WEIGHT
            }
            
            df['master_score'] = 0
            total_weight = 0
            
            for component, weight in components.items():
                if component in df.columns:
                    component_score = df[component].fillna(50).clip(0, 100)
                    df['master_score'] += component_score * weight
                    total_weight += weight
                else:
                    logger.warning(f"Missing component: {component}")
                    df['master_score'] += 50 * weight  # Use neutral score
                    total_weight += weight
            
            # Normalize if needed
            if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
                df['master_score'] = df['master_score'] / total_weight
            
            df['master_score'] = df['master_score'].clip(0, 100)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in master score calculation: {str(e)}")
            df['master_score'] = 50
            return df
    
    @staticmethod
    def _calculate_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ranks with bulletproof logic"""
        try:
            valid_scores = df['master_score'].notna()
            
            if valid_scores.sum() == 0:
                logger.error("No valid master scores found!")
                df['rank'] = 9999
                df['percentile'] = 0
                df['category_rank'] = 9999
                df['category_percentile'] = 0
                return df
            
            # Overall ranks
            df['rank'] = df['master_score'].rank(
                method='first', ascending=False, na_option='bottom'
            ).fillna(len(df) + 1).astype(int)
            
            df['percentile'] = df['master_score'].rank(
                pct=True, ascending=True, na_option='bottom'
            ).fillna(0) * 100
            
            # Category ranks
            df['category_rank'] = 9999
            df['category_percentile'] = 0.0
            
            if 'category' in df.columns:
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
            
        except Exception as e:
            logger.error(f"Error in rank calculation: {str(e)}")
            df['rank'] = range(1, len(df) + 1)
            df['percentile'] = 50
            df['category_rank'] = range(1, len(df) + 1)
            df['category_percentile'] = 50
            return df
