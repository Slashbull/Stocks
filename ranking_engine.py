"""
Wave Detection Ultimate 3.0 - Ranking Engine
============================================
Clean, efficient ranking and pattern detection system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from config import CONFIG, logger
from utils import timer, RankingUtils

@dataclass
class ScoreComponents:
    """Container for score components"""
    position: pd.Series
    volume: pd.Series  
    momentum: pd.Series
    acceleration: pd.Series
    breakout: pd.Series
    rvol: pd.Series
    
    def to_dict(self) -> Dict[str, pd.Series]:
        """Convert to dictionary for easy processing"""
        return {
            'position_score': self.position,
            'volume_score': self.volume,
            'momentum_score': self.momentum,
            'acceleration_score': self.acceleration,
            'breakout_score': self.breakout,
            'rvol_score': self.rvol
        }

class ScoreCalculator(ABC):
    """Abstract base for score calculators"""
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate score component"""
        pass

class PositionScoreCalculator(ScoreCalculator):
    """Calculate position score from 52-week range"""
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate position score based on 52-week range positioning"""
        position_score = pd.Series(50.0, index=df.index, dtype=float)
        
        # Check data availability
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
            logger.warning("No position data available, using neutral scores")
            return position_score + np.random.uniform(-5, 5, size=len(df))
        
        # Get position data
        if has_from_low:
            from_low = df['from_low_pct'].fillna(50)
            rank_from_low = RankingUtils.safe_rank(from_low, pct=True, ascending=True)
        else:
            rank_from_low = pd.Series(50.0, index=df.index)
        
        if has_from_high:
            # Convert from_high_pct to distance from high
            distance_from_high = 100 + df['from_high_pct'].fillna(-50)
            rank_from_high = RankingUtils.safe_rank(distance_from_high, pct=True, ascending=True)
        else:
            rank_from_high = pd.Series(50.0, index=df.index)
        
        # Weighted combination (distance from low is more important)
        if has_from_low and has_from_high:
            position_score = rank_from_low * 0.6 + rank_from_high * 0.4
        elif has_from_low:
            position_score = rank_from_low
        else:
            position_score = rank_from_high
        
        return position_score.clip(0, 100)

class VolumeScoreCalculator(ScoreCalculator):
    """Calculate comprehensive volume score"""
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume score from multiple volume ratios"""
        volume_score = pd.Series(50.0, index=df.index, dtype=float)
        
        # Volume ratio columns with weights
        vol_ratios = {
            'vol_ratio_1d_90d': 0.20,
            'vol_ratio_7d_90d': 0.20, 
            'vol_ratio_30d_90d': 0.20,
            'vol_ratio_30d_180d': 0.15,
            'vol_ratio_90d_180d': 0.25
        }
        
        total_weight = 0
        weighted_score = pd.Series(0.0, index=df.index, dtype=float)
        
        for col, weight in vol_ratios.items():
            if col in df.columns and df[col].notna().any():
                # Clean and rank the data
                col_data = df[col].fillna(1.0).clip(lower=0.1)  # Neutral = 1.0, min = 0.1
                col_rank = RankingUtils.safe_rank(col_data, pct=True, ascending=True)
                
                weighted_score += col_rank * weight
                total_weight += weight
        
        if total_weight > 0:
            volume_score = weighted_score / total_weight
        else:
            logger.warning("No volume ratio data available")
            volume_score += np.random.uniform(-5, 5, size=len(df))
        
        return volume_score.clip(0, 100)

class MomentumScoreCalculator(ScoreCalculator):
    """Calculate momentum score with consistency bonuses"""
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns"""
        momentum_score = pd.Series(50.0, index=df.index, dtype=float)
        
        # Primary momentum from 30-day returns
        if 'ret_30d' in df.columns and df['ret_30d'].notna().any():
            ret_30d = df['ret_30d'].fillna(0)
            momentum_score = RankingUtils.safe_rank(ret_30d, pct=True, ascending=True)
            
            # Add consistency bonuses
            if 'ret_7d' in df.columns:
                consistency_bonus = self._calculate_consistency_bonus(df)
                momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
                
        elif 'ret_7d' in df.columns and df['ret_7d'].notna().any():
            # Fallback to 7-day returns
            ret_7d = df['ret_7d'].fillna(0)
            momentum_score = RankingUtils.safe_rank(ret_7d, pct=True, ascending=True)
            logger.info("Using 7-day returns for momentum (30-day not available)")
        else:
            logger.warning("No return data available for momentum calculation")
            momentum_score += np.random.uniform(-5, 5, size=len(df))
        
        return momentum_score.clip(0, 100)
    
    def _calculate_consistency_bonus(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bonus for consistent positive momentum"""
        bonus = pd.Series(0.0, index=df.index, dtype=float)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            # Bonus for positive momentum in both periods
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            bonus[all_positive] = 5
            
            # Extra bonus for accelerating momentum
            daily_ret_7d = df['ret_7d'] / 7
            daily_ret_30d = df['ret_30d'] / 30
            
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            bonus[accelerating] = 10
        
        return bonus

class AccelerationScoreCalculator(ScoreCalculator):
    """Calculate acceleration score from momentum trends"""
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating"""
        acceleration_score = pd.Series(50.0, index=df.index, dtype=float)
        
        # Required columns for acceleration analysis
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in req_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient return data for acceleration calculation")
            return acceleration_score + np.random.uniform(-5, 5, size=len(df))
        
        # Get return data
        ret_1d = df.get('ret_1d', pd.Series(0, index=df.index)).fillna(0)
        ret_7d = df.get('ret_7d', pd.Series(0, index=df.index)).fillna(0) 
        ret_30d = df.get('ret_30d', pd.Series(0, index=df.index)).fillna(0)
        
        # Calculate daily averages
        daily_avg_7d = ret_7d / 7
        daily_avg_30d = ret_30d / 30
        
        # Score acceleration patterns
        if len(available_cols) >= 3:
            acceleration_score = self._score_full_acceleration(
                ret_1d, daily_avg_7d, daily_avg_30d
            )
        else:
            acceleration_score = self._score_partial_acceleration(
                ret_1d, daily_avg_7d
            )
        
        return acceleration_score.clip(0, 100)
    
    def _score_full_acceleration(self, ret_1d: pd.Series, daily_7d: pd.Series, 
                               daily_30d: pd.Series) -> pd.Series:
        """Score with full acceleration data"""
        scores = pd.Series(50.0, index=ret_1d.index)
        
        # Perfect acceleration: 1d > 7d avg > 30d avg, all positive
        perfect = (ret_1d > daily_7d) & (daily_7d > daily_30d) & (ret_1d > 0)
        scores[perfect] = 100
        
        # Good acceleration: 1d > 7d avg, both positive
        good = (~perfect) & (ret_1d > daily_7d) & (ret_1d > 0)
        scores[good] = 80
        
        # Moderate: positive 1d return
        moderate = (~perfect) & (~good) & (ret_1d > 0)
        scores[moderate] = 60
        
        # Slight deceleration: negative 1d, positive 7d
        slight_decel = (ret_1d <= 0) & (daily_7d > 0)
        scores[slight_decel] = 40
        
        # Strong deceleration: both negative
        strong_decel = (ret_1d <= 0) & (daily_7d <= 0)
        scores[strong_decel] = 20
        
        return scores
    
    def _score_partial_acceleration(self, ret_1d: pd.Series, 
                                  daily_7d: pd.Series) -> pd.Series:
        """Score with limited acceleration data"""
        scores = pd.Series(50.0, index=ret_1d.index)
        
        accelerating = ret_1d > daily_7d
        scores[accelerating & (ret_1d > 0)] = 75
        scores[~accelerating & (ret_1d > 0)] = 55
        scores[ret_1d <= 0] = 35
        
        return scores

class BreakoutScoreCalculator(ScoreCalculator):
    """Calculate breakout probability score"""
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate breakout readiness score"""
        # Three factors for breakout probability
        distance_factor = self._calculate_distance_factor(df)
        volume_factor = self._calculate_volume_factor(df)
        trend_factor = self._calculate_trend_factor(df)
        
        # Weighted combination
        breakout_score = (
            distance_factor * 0.4 +    # Distance from high
            volume_factor * 0.4 +      # Volume surge
            trend_factor * 0.2         # Trend support
        )
        
        return breakout_score.clip(0, 100)
    
    def _calculate_distance_factor(self, df: pd.DataFrame) -> pd.Series:
        """Factor based on distance from 52-week high"""
        if 'from_high_pct' in df.columns:
            distance_from_high = 100 + df['from_high_pct'].fillna(-50)
            return distance_from_high.clip(0, 100)
        else:
            return pd.Series(50.0, index=df.index)
    
    def _calculate_volume_factor(self, df: pd.DataFrame) -> pd.Series:
        """Factor based on recent volume surge"""
        if 'vol_ratio_7d_90d' in df.columns:
            vol_ratio = df['vol_ratio_7d_90d'].fillna(1.0)
            # Convert ratio to score (1.0 = neutral, >1.0 = increasing volume)
            return ((vol_ratio - 1) * 100).clip(0, 100)
        else:
            return pd.Series(50.0, index=df.index)
    
    def _calculate_trend_factor(self, df: pd.DataFrame) -> pd.Series:
        """Factor based on moving average support"""
        trend_factor = pd.Series(0.0, index=df.index, dtype=float)
        
        # Check position relative to key moving averages
        sma_checks = [
            ('sma_20d', 33.33),
            ('sma_50d', 33.33), 
            ('sma_200d', 33.34)
        ]
        
        for sma_col, weight in sma_checks:
            if sma_col in df.columns and 'price' in df.columns:
                above_sma = (df['price'] > df[sma_col]).fillna(False)
                trend_factor += above_sma.astype(float) * weight
        
        return trend_factor.clip(0, 100)

class RVOLScoreCalculator(ScoreCalculator):
    """Calculate RVOL-based score"""
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Score based on relative volume levels"""
        if 'rvol' not in df.columns:
            return pd.Series(50.0, index=df.index)
        
        rvol = df['rvol'].fillna(1.0)
        scores = pd.Series(50.0, index=df.index, dtype=float)
        
        # Score based on RVOL levels
        score_mapping = [
            (5.0, 100),    # Extreme volume
            (3.0, 90),     # Very high volume
            (2.0, 80),     # High volume
            (1.5, 70),     # Above average
            (1.2, 60),     # Slightly above average
            (0.8, 50),     # Normal (default)
            (0.5, 40),     # Below average
            (0.3, 30),     # Low volume
            (0.0, 20)      # Very low volume
        ]
        
        for threshold, score in score_mapping:
            if threshold > 0:
                mask = rvol > threshold
                scores[mask] = score
                rvol = rvol[~mask]  # Remove processed values
            else:
                scores[rvol <= 0.3] = score
        
        return scores

class PatternDetector:
    """Detect various stock patterns using vectorized operations"""
    
    def __init__(self):
        self.patterns = []
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns and add to dataframe"""
        df = df.copy()
        df['patterns'] = ''
        
        # Technical patterns
        self._detect_technical_patterns(df)
        
        # Fundamental patterns (if data available)
        if self._has_fundamental_data(df):
            self._detect_fundamental_patterns(df)
        
        # Clean up pattern strings
        df['patterns'] = df['patterns'].str.rstrip(' | ')
        
        return df
    
    def _detect_technical_patterns(self, df: pd.DataFrame):
        """Detect technical analysis patterns"""
        patterns = [
            ('🔥 CAT LEADER', self._category_leader_pattern(df)),
            ('💎 HIDDEN GEM', self._hidden_gem_pattern(df)),
            ('🚀 ACCELERATING', self._acceleration_pattern(df)),
            ('🏦 INSTITUTIONAL', self._institutional_pattern(df)),
            ('⚡ VOL EXPLOSION', self._volume_explosion_pattern(df)),
            ('🎯 BREAKOUT', self._breakout_ready_pattern(df)),
            ('👑 MARKET LEADER', self._market_leader_pattern(df)),
            ('🌊 MOMENTUM WAVE', self._momentum_wave_pattern(df)),
            ('💰 LIQUID LEADER', self._liquid_leader_pattern(df)),
            ('💪 LONG STRENGTH', self._long_strength_pattern(df)),
            ('📈 QUALITY TREND', self._quality_trend_pattern(df))
        ]
        
        self._apply_patterns(df, patterns)
    
    def _detect_fundamental_patterns(self, df: pd.DataFrame):
        """Detect fundamental analysis patterns"""
        patterns = [
            ('💎 VALUE MOMENTUM', self._value_momentum_pattern(df)),
            ('📊 EARNINGS ROCKET', self._earnings_rocket_pattern(df)),
            ('🏆 QUALITY LEADER', self._quality_leader_pattern(df)),
            ('⚡ TURNAROUND', self._turnaround_pattern(df)),
            ('⚠️ HIGH PE', self._high_pe_warning(df))
        ]
        
        self._apply_patterns(df, patterns)
    
    def _apply_patterns(self, df: pd.DataFrame, patterns: List[Tuple[str, pd.Series]]):
        """Apply pattern masks to dataframe"""
        for pattern_name, mask in patterns:
            if mask is not None and mask.any():
                df.loc[mask, 'patterns'] += pattern_name + ' | '
    
    # Technical Pattern Methods
    def _category_leader_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'category_percentile' in df.columns:
            return df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
        return None
    
    def _hidden_gem_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if all(col in df.columns for col in ['category_percentile', 'percentile']):
            return ((df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
                   (df['percentile'] < 70))
        return None
    
    def _acceleration_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'acceleration_score' in df.columns:
            return df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
        return None
    
    def _institutional_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if all(col in df.columns for col in ['volume_score', 'vol_ratio_90d_180d']):
            return ((df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                   (df['vol_ratio_90d_180d'] > 1.1))
        return None
    
    def _volume_explosion_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'rvol' in df.columns:
            return df['rvol'] > 3
        return None
    
    def _breakout_ready_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'breakout_score' in df.columns:
            return df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
        return None
    
    def _market_leader_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'percentile' in df.columns:
            return df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
        return None
    
    def _momentum_wave_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if all(col in df.columns for col in ['momentum_score', 'acceleration_score']):
            return ((df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                   (df['acceleration_score'] >= 70))
        return None
    
    def _liquid_leader_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if all(col in df.columns for col in ['liquidity_score', 'percentile']):
            threshold = CONFIG.PATTERN_THRESHOLDS['liquid_leader']
            return ((df['liquidity_score'] >= threshold) & (df['percentile'] >= threshold))
        return None
    
    def _long_strength_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'long_term_strength' in df.columns:
            return df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
        return None
    
    def _quality_trend_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'trend_quality' in df.columns:
            return df['trend_quality'] >= 80
        return None
    
    # Fundamental Pattern Methods
    def _value_momentum_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if all(col in df.columns for col in ['pe', 'master_score']):
            valid_pe = ((df['pe'].notna()) & (df['pe'] > 0) & 
                       (df['pe'] < 10000) & (~np.isinf(df['pe'])))
            return valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
        return None
    
    def _earnings_rocket_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if all(col in df.columns for col in ['eps_change_pct', 'acceleration_score']):
            valid_eps = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
            extreme_growth = valid_eps & (df['eps_change_pct'] > 1000) & (df['acceleration_score'] >= 80)
            strong_growth = (valid_eps & (df['eps_change_pct'] > 50) & 
                           (df['eps_change_pct'] <= 1000) & (df['acceleration_score'] >= 70))
            return extreme_growth | strong_growth
        return None
    
    def _quality_leader_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        required_cols = ['pe', 'eps_change_pct', 'percentile']
        if all(col in df.columns for col in required_cols):
            valid_data = ((df['pe'].notna()) & (df['eps_change_pct'].notna()) & 
                         (df['pe'] > 0) & (df['pe'] < 10000) & 
                         (~np.isinf(df['pe'])) & (~np.isinf(df['eps_change_pct'])))
            return (valid_data & (df['pe'] >= 10) & (df['pe'] <= 25) & 
                   (df['eps_change_pct'] > 20) & (df['percentile'] >= 80))
        return None
    
    def _turnaround_pattern(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if all(col in df.columns for col in ['eps_change_pct', 'volume_score']):
            valid_eps = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
            mega_turnaround = valid_eps & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60)
            strong_turnaround = (valid_eps & (df['eps_change_pct'] > 100) & 
                               (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70))
            return mega_turnaround | strong_turnaround
        return None
    
    def _high_pe_warning(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'pe' in df.columns:
            valid_pe = df['pe'].notna() & (df['pe'] > 0) & ~np.isinf(df['pe'])
            return valid_pe & (df['pe'] > 100)
        return None
    
    def _has_fundamental_data(self, df: pd.DataFrame) -> bool:
        """Check if fundamental data is available"""
        fundamental_cols = ['pe', 'eps_change_pct']
        return any(col in df.columns and df[col].notna().any() for col in fundamental_cols)

class RankingEngine:
    """Main ranking engine that orchestrates all calculations"""
    
    def __init__(self):
        self.calculators = {
            'position': PositionScoreCalculator(),
            'volume': VolumeScoreCalculator(), 
            'momentum': MomentumScoreCalculator(),
            'acceleration': AccelerationScoreCalculator(),
            'breakout': BreakoutScoreCalculator(),
            'rvol': RVOLScoreCalculator()
        }
        self.pattern_detector = PatternDetector()
    
    @timer
    def calculate_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main ranking calculation with all components"""
        if df.empty:
            return df
        
        logger.info("Starting ranking calculations...")
        result_df = df.copy()
        
        # Calculate all score components
        score_components = self._calculate_score_components(result_df)
        
        # Add component scores to dataframe
        for name, score in score_components.to_dict().items():
            result_df[name] = score
        
        # Calculate master score
        result_df['master_score'] = self._calculate_master_score(score_components)
        
        # Calculate ranks and percentiles
        result_df = self._calculate_ranks(result_df)
        
        # Calculate category-specific ranks
        result_df = self._calculate_category_ranks(result_df)
        
        # Detect patterns
        result_df = self.pattern_detector.detect_patterns(result_df)
        
        logger.info(f"Ranking complete: {len(result_df)} stocks processed")
        return result_df
    
    def _calculate_score_components(self, df: pd.DataFrame) -> ScoreComponents:
        """Calculate all score components"""
        return ScoreComponents(
            position=self.calculators['position'].calculate(df),
            volume=self.calculators['volume'].calculate(df),
            momentum=self.calculators['momentum'].calculate(df),
            acceleration=self.calculators['acceleration'].calculate(df),
            breakout=self.calculators['breakout'].calculate(df),
            rvol=self.calculators['rvol'].calculate(df)
        )
    
    def _calculate_master_score(self, components: ScoreComponents) -> pd.Series:
        """Calculate weighted master score"""
        weights = {
            'position_score': CONFIG.POSITION_WEIGHT,
            'volume_score': CONFIG.VOLUME_WEIGHT,
            'momentum_score': CONFIG.MOMENTUM_WEIGHT,
            'acceleration_score': CONFIG.ACCELERATION_WEIGHT,
            'breakout_score': CONFIG.BREAKOUT_WEIGHT,
            'rvol_score': CONFIG.RVOL_WEIGHT
        }
        
        return RankingUtils.weighted_score(components.to_dict(), weights)
    
    def _calculate_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall ranks and percentiles"""
        valid_scores = df['master_score'].notna()
        
        if valid_scores.sum() == 0:
            logger.error("No valid master scores calculated!")
            df['rank'] = 9999
            df['percentile'] = 0
        else:
            # Calculate ranks (method='first' ensures unique ranks)
            df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom')
            df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
            
            # Calculate percentiles
            df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
            df['percentile'] = df['percentile'].fillna(0)
        
        return df
    
    def _calculate_category_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks within each category"""
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        if 'category' not in df.columns:
            return df
        
        # Rank within each category
        for category in df['category'].unique():
            if category != 'Unknown':
                mask = df['category'] == category
                cat_df = df[mask]
                
                if len(cat_df) > 0:
                    # Calculate ranks within category
                    cat_ranks = RankingUtils.safe_rank(
                        cat_df['master_score'], pct=False, ascending=False
                    )
                    df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                    
                    # Calculate percentiles within category
                    cat_percentiles = RankingUtils.safe_rank(
                        cat_df['master_score'], pct=True, ascending=True
                    )
                    df.loc[mask, 'category_percentile'] = cat_percentiles
        
        return df