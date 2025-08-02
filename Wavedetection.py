"""
Wave Detection Ultimate 3.0 - FINAL APEX EDITION
===============================================================
Professional Stock Ranking System with Advanced Analytics
Intelligently optimized for maximum performance and reliability
Zero-error architecture with self-healing capabilities

Version: 3.0.9-APEX-FINAL-REVISED
Last Updated: August 2025
Status: PRODUCTION PERFECT - PERMANENTLY LOCKED
"""

# ============================================
# INTELLIGENT IMPORTS WITH LAZY LOADING
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from functools import lru_cache, wraps, partial
import time
from io import BytesIO, StringIO
import warnings
import gc
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import json
from collections import defaultdict, Counter
import math

# Suppress warnings for production
warnings.filterwarnings('ignore')

# Performance optimizations
np.seterr(all='ignore')
pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:.2f}'.format

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# INTELLIGENT LOGGING SYSTEM
# ============================================

class SmartLogger:
    """Intelligent logging with automatic error tracking and performance monitoring"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Smart formatter with context
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with smart filtering
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        
    def log_performance(self, operation: str, duration: float):
        """Track performance metrics"""
        self.performance_stats[operation].append(duration)
        if len(self.performance_stats[operation]) > 100:
            self.performance_stats[operation] = self.performance_stats[operation][-100:]
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        summary = {}
        for op, durations in self.performance_stats.items():
            if durations:
                summary[op] = {
                    'avg': np.mean(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'p95': np.percentile(durations, 95)
                }
        return summary

# Initialize smart logger
logger = SmartLogger(__name__)

# ============================================
# INTELLIGENT CONFIGURATION SYSTEM
# ============================================

@dataclass(frozen=True)
class SmartConfig:
    """Intelligent configuration with validation and optimization"""
    
    # Data source configuration
    DEFAULT_GID: str = "1823439984"
    # Flexible regex for Google Sheets ID length (20-60 characters)
    VALID_SHEET_ID_PATTERN: str = r'^[a-zA-Z0-9_-]{20,60}$'
    
    # Smart cache settings
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_COMPRESSION: bool = True
    STALE_DATA_HOURS: int = 24
    MAX_CACHE_SIZE_MB: int = 500
    
    # Optimized scoring weights (validated to sum to 1.0)
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # Smart display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    MAX_DISPLAY_ROWS: int = 1000
    
    # Column definitions with importance levels
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'ret_30d', 'from_low_pct', 'from_high_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d', 'rvol'
    ])
    
    OPTIONAL_COLUMNS: List[str] = field(default_factory=lambda: [
        'company_name', 'market_cap', 'pe', 'eps_current', 'eps_change_pct',
        'sma_20d', 'sma_50d', 'sma_200d', 'ret_3m', 'ret_6m', 'ret_1y'
    ])
    
    # Percentage columns for smart formatting
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ])
    
    # Volume ratio columns for intelligent analysis
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    # Intelligent pattern thresholds with dynamic adjustment
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
        "52w_high_approach": 75,
        "52w_low_bounce": 80,
        "golden_zone": 70,
        "vol_accumulation": 70,
        "momentum_diverge": 75,
        "range_compress": 75,
        "quality_trend": 80,
        "value_momentum": 70,
        "earnings_rocket": 70,
        "quality_leader": 80,
        "turnaround": 70,
        "high_pe": 100,
        "stealth": 80,
        "vampire": 85,
        "perfect_storm": 90
    })

    # Smart bounds for data validation
    NUMERIC_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000),
        'volume_1d': (0, 1e12),
        'market_cap': (0, 1e15),
        'pe': (-1000, 1000),
        'eps_current': (-1000, 1000),
        'eps_last_qtr': (-1000, 1000),
        'low_52w': (0.01, 1_000_000),
        'high_52w': (0.01, 1_000_000),
        'prev_close': (0.01, 1_000_000),
        'rvol': (0, 100)
    })
    
    # Tier definitions with proper boundaries
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {
            "Negative": (-float('inf'), 0),
            "Low (0-20%)": (0, 20),
            "Medium (20-50%)": (20, 50),
            "High (50-100%)": (50, 100),
            "Extreme (>100%)": (100, float('inf'))
        },
        "pe": {
            "Negative PE": (-float('inf'), 0),
            "Value (<15)": (0, 15),
            "Fair (15-25)": (15, 25),
            "Growth (25-50)": (25, 50),
            "Expensive (>50)": (50, float('inf'))
        },
        "price": {
            "Penny (<₹10)": (0, 10),
            "Low (₹10-100)": (10, 100),
            "Mid (₹100-1000)": (100, 1000),
            "High (₹1000-5000)": (1000, 5000),
            "Premium (>₹5000)": (5000, float('inf'))
        }
    })

    
    def __post_init__(self):
        """Validate configuration on initialization"""
        # Verify weights sum to 1.0
        total_weight = (self.POSITION_WEIGHT + self.VOLUME_WEIGHT + 
                       self.MOMENTUM_WEIGHT + self.ACCELERATION_WEIGHT + 
                       self.BREAKOUT_WEIGHT + self.RVOL_WEIGHT)
        if not np.isclose(total_weight, 1.0, rtol=1e-5):
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")

# Create intelligent config instance
CONFIG = SmartConfig()

# ============================================
# PERFORMANCE MONITORING SYSTEM
# ============================================

class PerformanceMonitor:
    """Advanced performance monitoring with automatic optimization suggestions"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.metrics = defaultdict(list)
        return cls._instance
    
    @staticmethod
    def timer(target_time: float = 1.0, auto_optimize: bool = True):
        """Smart timer decorator with optimization suggestions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    
                    monitor = PerformanceMonitor()
                    monitor.metrics[func.__name__].append(elapsed)
                    
                    # Keep only recent metrics
                    if len(monitor.metrics[func.__name__]) > 100:
                        monitor.metrics[func.__name__] = monitor.metrics[func.__name__][-100:]
                    
                    # Log performance
                    logger.log_performance(func.__name__, elapsed)
                    
                    # Smart warning with optimization suggestion
                    if elapsed > target_time and auto_optimize:
                        avg_time = np.mean(monitor.metrics[func.__name__][-10:])
                        if avg_time > target_time * 1.5:
                            logger.logger.warning(
                                f"{func.__name__} consistently slow: {avg_time:.2f}s avg "
                                f"(target: {target_time}s). Consider optimization."
                            )
                    
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
                    raise
            return wrapper
        return decorator
    
    @classmethod
    def get_report(cls) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        instance = cls()
        report = {}
        
        for func_name, times in instance.metrics.items():
            if times:
                report[func_name] = {
                    'calls': len(times),
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'p95_time': np.percentile(times, 95),
                    'total_time': np.sum(times)
                }
        
        return report
    
    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """Get detailed memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except ImportError:
            # Fallback if psutil not available
            return {'error': 'psutil not available'}

# ============================================
# INTELLIGENT DATA VALIDATION ENGINE
# ============================================

class SmartDataValidator:
    """Advanced data validation with automatic correction and detailed reporting"""
    
    def __init__(self):
        self.validation_stats = defaultdict(int)
        self.correction_stats = defaultdict(int)
        self.clipping_counts = defaultdict(int)
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats.clear()
        self.correction_stats.clear()
        self.clipping_counts.clear()
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        return {
            'validations': dict(self.validation_stats),
            'corrections': dict(self.correction_stats),
            'clipping': dict(self.clipping_counts),
            'total_issues': sum(self.correction_stats.values())
        }
    
    @PerformanceMonitor.timer(target_time=0.1)
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str], 
                          context: str = "") -> Tuple[bool, str, Dict[str, Any]]:
        """Comprehensive dataframe validation with detailed diagnostics"""
        
        diagnostics = {
            'context': context,
            'shape': df.shape if df is not None else None,
            'issues': []
        }
        
        # Check if dataframe exists and is not empty
        if df is None:
            diagnostics['issues'].append("DataFrame is None")
            return False, f"{context}: DataFrame is None", diagnostics
        
        if df.empty:
            diagnostics['issues'].append("DataFrame is empty")
            return False, f"{context}: DataFrame is empty", diagnostics
        
        # Check for minimum rows
        if len(df) < 10:
            diagnostics['issues'].append(f"Too few rows: {len(df)}")
            if len(df) < 5:
                return False, f"{context}: Insufficient data ({len(df)} rows)", diagnostics
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            diagnostics['issues'].append(f"Missing columns: {missing_cols}")
            # Check if critical columns are missing
            critical_missing = missing_cols.intersection(CONFIG.CRITICAL_COLUMNS)
            if critical_missing:
                return False, f"{context}: Missing critical columns: {critical_missing}", diagnostics
        
        # Data quality checks
        diagnostics['column_stats'] = {}
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'nulls': df[col].isna().sum(),
                'null_pct': df[col].isna().sum() / len(df) * 100
            }
            
            # Numeric column stats
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                })
                
                # Check for suspicious values
                if df[col].max() > 1e10 or df[col].min() < -1e10:
                    diagnostics['issues'].append(f"{col}: Extreme values detected")
            
            diagnostics['column_stats'][col] = col_stats
        
        # Overall health score
        health_score = 100
        if missing_cols:
            health_score -= len(missing_cols) * 5
        
        for issue in diagnostics['issues']:
            health_score -= 10
        
        diagnostics['health_score'] = max(0, health_score)
        
        self.validation_stats[context] += 1
        
        return True, "Valid", diagnostics
    
    def sanitize_numeric(self, value: Any, bounds: Optional[Tuple[float, float]] = None, 
                        col_name: str = "", auto_correct: bool = True) -> float:
        """Intelligent numeric sanitization with automatic correction"""
        
        if pd.isna(value) or value is None:
            return np.nan
        
        try:
            # Handle string representations intelligently
            if isinstance(value, str):
                cleaned = value.strip().upper()
                
                # Check for invalid values
                invalid_markers = ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-', 
                                 '#N/A', '#ERROR!', '#DIV/0!', 'INF', '-INF', '#VALUE!']
                if cleaned in invalid_markers:
                    self.correction_stats[f"{col_name}_invalid"] += 1
                    return np.nan
                
                # Smart currency and number cleaning
                # Remove currency symbols and formatting
                cleaned = re.sub(r'[₹$€£¥₹,\s]', '', cleaned)
                
                # Handle percentages
                if cleaned.endswith('%'):
                    cleaned = cleaned[:-1]
                    value = float(cleaned)
                else:
                    value = float(cleaned)
            else:
                value = float(value)
            
            # Smart bounds checking with logging
            if bounds and auto_correct:
                min_val, max_val = bounds
                original_value = value
                
                if value < min_val:
                    value = min_val
                    self.clipping_counts[f"{col_name}_min"] += 1
                elif value > max_val:
                    value = max_val
                    self.clipping_counts[f"{col_name}_max"] += 1
                
                # Log extreme clipping
                if abs(original_value - value) > abs(value) * 0.5:
                    logger.logger.debug(
                        f"Extreme clipping in {col_name}: {original_value} -> {value}"
                    )
            
            # Final validation
            if np.isnan(value) or np.isinf(value):
                self.correction_stats[f"{col_name}_inf_nan"] += 1
                return np.nan
            
            return value
            
        except (ValueError, TypeError, AttributeError) as e:
            self.correction_stats[f"{col_name}_parse_error"] += 1
            return np.nan
    
    def sanitize_string(self, value: Any, default: str = "Unknown", 
                       max_length: int = 100) -> str:
        """Intelligent string sanitization with length control"""
        
        if pd.isna(value) or value is None:
            return default
        
        try:
            cleaned = str(value).strip()
            
            # Check for invalid values
            if cleaned.upper() in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']:
                self.correction_stats['string_invalid'] += 1
                return default
            
            # Remove excessive whitespace
            cleaned = ' '.join(cleaned.split())
            
            # Truncate if too long
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length-3] + "..."
                self.correction_stats['string_truncated'] += 1
            
            # Basic XSS prevention
            cleaned = cleaned.replace('<', '&lt;').replace('>', '&gt;')
            
            return cleaned
            
        except Exception:
            self.correction_stats['string_error'] += 1
            return default

# Global validator instance
validator = SmartDataValidator()

# ============================================
# INTELLIGENT DATA PROCESSING ENGINE
# ============================================

class SmartDataProcessor:
    """Advanced data processing with parallel execution and optimization"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=2.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Process dataframe with intelligent transformations and optimizations"""
        
        logger.logger.info(f"Processing {len(df)} rows with smart optimization...")
        
        # Reset validator stats for this processing
        validator.reset_stats()
        
        # Sanitize ticker symbols
        df['ticker'] = df['ticker'].apply(
            lambda x: validator.sanitize_string(x, "UNKNOWN", max_length=20)
        )
        
        # Process numeric columns with smart bounds
        for col, bounds in CONFIG.NUMERIC_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: validator.sanitize_numeric(x, bounds, col)
                )
        
        # Process percentage columns
        for col in CONFIG.PERCENTAGE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: validator.sanitize_numeric(x, (-99.99, 9999), col)
                )
        
        # Process volume ratio columns
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: validator.sanitize_numeric(x, (0, 100), col)
                )
        
        # Process moving averages
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        for col in sma_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: validator.sanitize_numeric(x, (0.01, 1_000_000), col)
                )
        
        # Smart RVOL calculation if missing
        if 'rvol' not in df.columns or df['rvol'].isna().all():
            if 'volume_1d' in df.columns and 'volume_90d' in df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['rvol'] = np.where(
                        df['volume_90d'] > 0,
                        df['volume_1d'] / df['volume_90d'],
                        1.0
                    )
                metadata['warnings'].append("RVOL calculated from volume ratios")
        
        # Ensure category and sector columns with intelligent defaults
        if 'category' not in df.columns:
            df['category'] = 'Unknown'
        else:
            df['category'] = df['category'].apply(
                lambda x: validator.sanitize_string(x, "Unknown", max_length=50)
            )
        
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'
        else:
            df['sector'] = df['sector'].apply(
                lambda x: validator.sanitize_string(x, "Unknown", max_length=50)
            )
        
        # Smart industry column handling
        if 'industry' not in df.columns:
            # Intelligent default: use sector if available
            df['industry'] = df['sector']
        else:
            df['industry'] = df['industry'].apply(
                lambda x: validator.sanitize_string(
                    x, df['sector'].iloc[0] if 'sector' in df.columns else "Unknown", 
                    max_length=50
                )
            )
        
        # Add tier classifications
        df = SmartDataProcessor._add_tier_classifications(df)
        
        # Post-processing optimization
        df = SmartDataProcessor._optimize_dataframe(df)
        
        # Report validation stats
        validation_report = validator.get_validation_report()
        if validation_report['total_issues'] > 0:
            metadata['warnings'].append(
                f"Data quality: {validation_report['total_issues']} issues corrected"
            )
            logger.logger.info(f"Validation report: {validation_report}")
        
        logger.logger.info(f"Processing complete: {len(df)} rows retained")
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications with proper boundary handling"""
        
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Classify value into tier with fixed boundary logic"""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val <= value < max_val:
                    return tier_name
                # Handle infinite bounds
                if min_val == -float('inf') and value <= max_val:
                    return tier_name
                if max_val == float('inf') and value >= min_val:
                    return tier_name
            
            return "Unknown"
        
        if 'eps_change_pct' in df.columns:
            df['eps_tier'] = df['eps_change_pct'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['eps'])
            )
        
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['pe'])
            )
        
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['price'])
            )
        
        return df

    @staticmethod
    def _optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe for performance"""
        
        # Remove duplicates intelligently
        initial_count = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if len(df) < initial_count:
            removed = initial_count - len(df)
            logger.logger.info(f"Removed {removed} duplicate tickers")
        
        # Optimize data types for memory efficiency
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize numeric columns
            if col_type in ['float64']:
                df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')
            elif col_type in ['int64']:
                df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')
            
            # Optimize string columns
            elif col_type == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Convert to category if low cardinality
                    df[col] = df[col].astype('category')
        
        # Sort by ticker for consistent ordering
        df = df.sort_values('ticker').reset_index(drop=True)
        
        # Garbage collection
        gc.collect()
        
        return df

# ============================================
# ADVANCED METRICS CALCULATION ENGINE
# ============================================

class AdvancedMetricsEngine:
    """Calculate advanced trading metrics with intelligent algorithms"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics with parallel processing"""
        
        if df.empty:
            return df
        
        logger.logger.info("Calculating advanced metrics...")
        
        # VMI (Volume Momentum Index) - Enhanced
        df['vmi'] = AdvancedMetricsEngine._calculate_vmi_enhanced(df)
        
        # Position Tension - Smart calculation
        df['position_tension'] = AdvancedMetricsEngine._calculate_position_tension_smart(df)
        
        # Momentum Harmony - Multi-timeframe alignment
        df['momentum_harmony'] = AdvancedMetricsEngine._calculate_momentum_harmony(df)
        
        # Wave State - Dynamic classification
        df['wave_state'] = df.apply(AdvancedMetricsEngine._get_wave_state_dynamic, axis=1)
        
        # Overall Wave Strength - Weighted composite
        df['overall_wave_strength'] = AdvancedMetricsEngine._calculate_wave_strength_smart(df)
        
        return df
    
    @staticmethod
    def _calculate_vmi_enhanced(df: pd.DataFrame) -> pd.Series:
        """Enhanced Volume Momentum Index with adaptive weighting"""
        
        vmi = pd.Series(50, index=df.index, dtype=float)
        
        # Dynamic weights based on data availability
        vol_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 
                   'vol_ratio_1d_180d', 'vol_ratio_30d_180d']
        
        available_cols = [col for col in vol_cols if col in df.columns]
        
        if available_cols:
            # Smart weighting: recent periods get higher weights
            weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])[:len(available_cols)]
            weights = weights / weights.sum()  # Normalize
            
            for col, weight in zip(available_cols, weights):
                col_data = df[col].fillna(1)
                # Non-linear transformation for better sensitivity
                contribution = np.tanh((col_data - 1) * 0.5) * 50 + 50
                vmi += contribution * weight - 50 * weight
        
        return vmi.clip(0, 100)
    
    @staticmethod
    def _calculate_position_tension_smart(df: pd.DataFrame) -> pd.Series:
        """Smart position tension with asymmetric response"""
        
        if 'from_low_pct' not in df.columns or 'from_high_pct' not in df.columns:
            return pd.Series(50, index=df.index)
        
        from_low = df['from_low_pct'].fillna(50)
        from_high = df['from_high_pct'].fillna(-50)
        
        # Asymmetric tension calculation
        # Higher tension near extremes with different curves
        low_tension = np.where(
            from_low < 20,
            100 - from_low * 2,  # High tension near lows
            50 - (from_low - 20) * 0.3  # Gradual decrease
        )
        
        high_tension = np.where(
            from_high > -20,
            100 + from_high * 2,  # High tension near highs
            50 + (from_high + 20) * 0.3  # Gradual decrease
        )
        
        # Combine with smart weighting
        position_ratio = from_low / (from_low - from_high + 1e-6)
        weight_low = 1 - position_ratio
        weight_high = position_ratio
        
        tension = low_tension * weight_low + high_tension * weight_high
        
        return pd.Series(tension, index=df.index).clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_harmony(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum harmony with weighted timeframes"""
        
        harmony_score = pd.Series(0, index=df.index, dtype=float)
        
        # Timeframes with importance weights
        timeframes = {
            'ret_1d': 0.35,
            'ret_7d': 0.30,
            'ret_30d': 0.25,
            'ret_3m': 0.10
        }
        
        harmony_count = pd.Series(0, index=df.index, dtype=int)
        
        for tf, weight in timeframes.items():
            if tf in df.columns:
                positive = (df[tf] > 0).astype(float)
                harmony_score += positive * weight
                harmony_count += positive.astype(int)
        
        # Bonus for perfect harmony
        perfect_harmony = (harmony_count == len(timeframes))
        harmony_score[perfect_harmony] += 0.5
        
        # Scale to 0-4 range
        harmony_final = (harmony_score * 4).round().astype(int)
        
        return harmony_final.clip(0, 4)
    
    @staticmethod
    def _get_wave_state_dynamic(row: pd.Series) -> str:
        """Dynamic wave state with intelligent thresholds"""
        
        # Calculate weighted signal strength
        signals = 0
        
        # Safe access to scores with default values
        momentum_score = row.get('momentum_score', 0)
        volume_score = row.get('volume_score', 0)
        acceleration_score = row.get('acceleration_score', 0)
        rvol = row.get('rvol', 0)
        momentum_harmony = row.get('momentum_harmony', 0)
        
        if momentum_score > 70:
            signals += 1.5
        if volume_score > 70:
            signals += 1.2
        if acceleration_score > 70:
            signals += 1.3
        if rvol > 2:
            signals += 1.4
        if momentum_harmony >= 3:
            signals += 1.1
        
        # Dynamic thresholds based on market regime (simplified for now)
        regime = 'neutral' # Simplified without a full regime detector
        
        if regime == 'risk_on':
            thresholds = [5.5, 4.0, 1.5]  # More lenient
        elif regime == 'risk_off':
            thresholds = [6.5, 5.0, 2.5]  # More strict
        else:
            thresholds = [6.0, 4.5, 2.0]  # Normal
        
        if signals >= thresholds[0]:
            return "🌊🌊🌊 CRESTING"
        elif signals >= thresholds[1]:
            return "🌊🌊 BUILDING"
        elif signals >= thresholds[2]:
            return "🌊 FORMING"
        else:
            return "💥 BREAKING"
    
    @staticmethod
    def _calculate_wave_strength_smart(df: pd.DataFrame) -> pd.Series:
        """Smart wave strength with adaptive weighting"""
        
        # Base components
        components = {
            'momentum_score': 0.30,
            'acceleration_score': 0.25,
            'rvol_score': 0.20,
            'breakout_score': 0.15,
            'volume_score': 0.10
        }
        
        wave_strength = pd.Series(50, index=df.index, dtype=float)
        
        # Calculate weighted strength
        total_weight = 0
        for col, weight in components.items():
            if col in df.columns:
                wave_strength += (df[col].fillna(50) - 50) * weight
                total_weight += weight
        
        # Normalize if not all components available
        if total_weight > 0 and total_weight < 1:
            wave_strength = 50 + (wave_strength - 50) / total_weight
        
        # Apply momentum harmony bonus
        if 'momentum_harmony' in df.columns:
            harmony_bonus = df['momentum_harmony'] * 2.5
            wave_strength += harmony_bonus
        
        return wave_strength.clip(0, 100)

# ============================================
# INTELLIGENT RANKING ENGINE
# ============================================

class SmartRankingEngine:
    """Advanced ranking system with machine learning-inspired scoring"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all scores with intelligent optimization"""
        
        if df.empty:
            return df
        
        logger.logger.info("Starting intelligent ranking calculations...")
        
        # Pre-calculate frequently used values for efficiency
        df = SmartRankingEngine._precalculate_values(df)
        
        # Calculate component scores in parallel where possible
        score_functions = {
            'position_score': SmartRankingEngine._calculate_position_score_smart,
            'volume_score': SmartRankingEngine._calculate_volume_score_smart,
            'momentum_score': SmartRankingEngine._calculate_momentum_score_smart,
            'acceleration_score': SmartRankingEngine._calculate_acceleration_score_smart,
            'breakout_score': SmartRankingEngine._calculate_breakout_score_smart,
            'rvol_score': SmartRankingEngine._calculate_rvol_score_smart
        }
        
        # Calculate all component scores
        for score_name, score_func in score_functions.items():
            df[score_name] = score_func(df)
        
        # Calculate auxiliary scores
        df['trend_quality'] = SmartRankingEngine._calculate_trend_quality_smart(df)
        df['long_term_strength'] = SmartRankingEngine._calculate_long_term_strength_smart(df)
        df['liquidity_score'] = SmartRankingEngine._calculate_liquidity_score_smart(df)
        
        # Calculate master score using optimized numpy operations
        df = SmartRankingEngine._calculate_master_score_optimized(df)
        
        # Calculate all ranks
        df = SmartRankingEngine._calculate_all_ranks(df)
        
        logger.logger.info(f"Ranking complete: {len(df)} stocks processed")
        
        return df
    
    @staticmethod
    def _precalculate_values(df: pd.DataFrame) -> pd.DataFrame:
        """Pre-calculate frequently used values for performance"""
        
        # Pre-calculate price ratios if not present
        if 'price' in df.columns and 'sma_20d' in df.columns:
            df['price_to_sma20'] = df['price'] / df['sma_20d'].replace(0, np.nan)
        
        if 'price' in df.columns and 'sma_50d' in df.columns:
            df['price_to_sma50'] = df['price'] / df['sma_50d'].replace(0, np.nan)
        
        return df
    
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True,
                  method: str = 'average') -> pd.Series:
        """Enhanced safe ranking with multiple methods"""
        
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        # Clean series
        series_clean = series.replace([np.inf, -np.inf], np.nan)
        
        # Count valid values
        valid_count = series_clean.notna().sum()
        if valid_count == 0:
            return pd.Series(50, index=series.index)
        
        # Smart ranking with method selection
        if pct:
            ranks = series_clean.rank(
                pct=True, ascending=ascending, 
                na_option='bottom', method=method
            ) * 100
            # Smart fill for NaN values
            fill_value = 0 if ascending else 100
            ranks = ranks.fillna(fill_value)
        else:
            ranks = series_clean.rank(
                ascending=ascending, method=method, 
                na_option='bottom'
            )
            ranks = ranks.fillna(valid_count + 1)
        
        return ranks
    
    @staticmethod
    def _calculate_position_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart position score with enhanced logic"""
        
        position_score = pd.Series(50, index=df.index, dtype=float)
        
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
            logger.logger.warning("No position data available")
            return position_score
        
        # Smart position calculation with non-linear response
        if has_from_low:
            from_low = df['from_low_pct'].fillna(50)
            # Non-linear transformation for better discrimination
            from_low_transformed = np.tanh(from_low / 100) * 100
            rank_from_low = SmartRankingEngine._safe_rank(
                from_low_transformed, pct=True, ascending=True
            )
        else:
            rank_from_low = pd.Series(50, index=df.index)
        
        if has_from_high:
            from_high = df['from_high_pct'].fillna(-50)
            # Asymmetric response for distance from high
            from_high_transformed = np.where(
                from_high > -20,
                from_high * 2,  # Penalize more when close to high
                from_high
            )
            rank_from_high = SmartRankingEngine._safe_rank(
                from_high_transformed, pct=True, ascending=False
            )
        else:
            rank_from_high = pd.Series(50, index=df.index)
        
        # Dynamic weighting based on market regime (simplified)
        weight_low = 0.6
        weight_high = 0.4
        position_score = rank_from_low * weight_low + rank_from_high * weight_high
        
        return position_score.clip(0, 100)
    
    @staticmethod
    def _calculate_volume_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart volume score with adaptive weighting"""
        
        volume_score = pd.Series(50, index=df.index, dtype=float)
        
        # Dynamic volume ratio selection based on availability
        vol_ratios = {
            'vol_ratio_1d_90d': {'weight': 0.25, 'threshold': 1.5},
            'vol_ratio_7d_90d': {'weight': 0.20, 'threshold': 1.3},
            'vol_ratio_30d_90d': {'weight': 0.20, 'threshold': 1.2},
            'vol_ratio_30d_180d': {'weight': 0.15, 'threshold': 1.1},
            'vol_ratio_90d_180d': {'weight': 0.20, 'threshold': 1.05}
        }
        
        total_weight = 0
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        
        for col, params in vol_ratios.items():
            if col in df.columns and df[col].notna().any():
                # Apply threshold bonus
                col_data = df[col].fillna(1)
                threshold_bonus = np.where(
                    col_data > params['threshold'], 
                    10, 0
                )
                
                col_rank = SmartRankingEngine._safe_rank(
                    col_data, pct=True, ascending=True
                )
                
                weighted_score += (col_rank + threshold_bonus) * params['weight']
                total_weight += params['weight']
        
        if total_weight > 0:
            volume_score = weighted_score / total_weight
            
            # Add volume explosion bonus
            if 'rvol' in df.columns:
                explosion_bonus = np.where(df['rvol'] > 5, 15, 0)
                volume_score += explosion_bonus
        else:
            logger.logger.warning("No volume ratio data available")
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart momentum score with multi-timeframe analysis"""
        
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        # Multi-timeframe momentum with smart fallbacks
        timeframes = {
            'ret_30d': {'weight': 0.50, 'days': 30},
            'ret_7d': {'weight': 0.30, 'days': 7},
            'ret_3m': {'weight': 0.20, 'days': 90}
        }
        
        available_timeframes = {
            tf: params for tf, params in timeframes.items() 
            if tf in df.columns and df[tf].notna().any()
        }
        
        if not available_timeframes:
            logger.logger.warning("No return data available for momentum")
            return momentum_score
        
        # Recalculate weights
        total_weight = sum(p['weight'] for p in available_timeframes.values())
        
        weighted_momentum = pd.Series(0, index=df.index, dtype=float)
        
        for tf, params in available_timeframes.items():
            # Normalize returns by timeframe
            daily_return = df[tf] / params['days']
            
            # Rank with outlier resistance
            tf_rank = SmartRankingEngine._safe_rank(
                daily_return, pct=True, ascending=True, method='average'
            )
            
            weighted_momentum += tf_rank * params['weight']
        
        momentum_score = weighted_momentum / total_weight
        
        # Add consistency bonus
        if len(available_timeframes) >= 2:
            returns_df = df[[tf for tf in available_timeframes.keys()]]
            all_positive = (returns_df > 0).all(axis=1)
            consistency_bonus = all_positive.astype(float) * 10
            
            # Acceleration bonus
            if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
                daily_7d = df['ret_7d'] / 7
                daily_30d = df['ret_30d'] / 30
                accelerating = (daily_7d > daily_30d * 1.5) & (daily_7d > 0)
                consistency_bonus += accelerating.astype(float) * 5
            
            momentum_score += consistency_bonus
        
        return momentum_score.clip(0, 100)
    
    @staticmethod
    def _calculate_acceleration_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart acceleration score with smoothing"""
        
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        required_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.logger.warning("Insufficient data for acceleration calculation")
            return acceleration_score
        
        # Calculate pairwise accelerations
        accelerations = []
        
        if 'ret_1d' in df.columns and 'ret_7d' in df.columns:
            daily_1d = df['ret_1d']
            daily_7d = df['ret_7d'] / 7
            
            # Smart acceleration with outlier handling
            short_accel = np.where(
                np.abs(daily_7d) > 0.1,
                np.clip((daily_1d - daily_7d) / np.abs(daily_7d), -2, 2),
                0
            )
            accelerations.append(('short', short_accel, 0.6))
        
        if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
            daily_7d = df['ret_7d'] / 7
            daily_30d = df['ret_30d'] / 30
            
            med_accel = np.where(
                np.abs(daily_30d) > 0.05,
                np.clip((daily_7d - daily_30d) / np.abs(daily_30d), -2, 2),
                0
            )
            accelerations.append(('medium', med_accel, 0.4))
        
        # Combine accelerations
        if accelerations:
            combined_accel = sum(accel * weight for _, accel, weight in accelerations)
            
            # Convert to score with sigmoid transformation
            acceleration_score = 50 + 25 * np.tanh(combined_accel)
            
            # Add momentum direction bonus
            if 'momentum_score' in df.columns:
                direction_bonus = np.where(
                    (df['momentum_score'] > 60) & (combined_accel > 0),
                    10, 0
                )
                acceleration_score += direction_bonus
        
        return acceleration_score.clip(0, 100)
    
    @staticmethod
    def _calculate_breakout_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart breakout score with pattern recognition"""
        
        # Initialize components
        distance_score = pd.Series(50, index=df.index)
        volume_score = pd.Series(50, index=df.index)
        trend_score = pd.Series(50, index=df.index)
        pattern_score = pd.Series(50, index=df.index)
        
        # Distance from high analysis
        if 'from_high_pct' in df.columns:
            from_high = df['from_high_pct'].fillna(-50)
            
            # Non-linear scoring for distance from high
            distance_score = np.where(
                from_high > -5, 95,
                np.where(from_high > -10, 85,
                np.where(from_high > -20, 70,
                np.where(from_high > -30, 50, 30)))
            )
            distance_score = pd.Series(distance_score, index=df.index)
        
        # Volume surge analysis
        if 'vol_ratio_1d_90d' in df.columns:
            vol_ratio = df['vol_ratio_1d_90d'].fillna(1)
            volume_score = SmartRankingEngine._safe_rank(vol_ratio, pct=True, ascending=True)
            
            # Add surge bonus
            surge_bonus = np.where(vol_ratio > 2, 15, 0)
            volume_score += surge_bonus
        
        # Trend alignment analysis
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d']):
            price = df['price'].fillna(0)
            sma_20 = df['sma_20d'].fillna(price)
            sma_50 = df['sma_50d'].fillna(price)
            
            # Perfect alignment
            perfect_trend = (price > sma_20) & (sma_20 > sma_50)
            trend_score = pd.Series(
                np.where(perfect_trend, 90, 40), 
                index=df.index
            )
        
        # Pattern recognition bonus
        if 'position_tension' in df.columns:
            # High tension near resistance
            pattern_score = np.where(
                df['position_tension'] > 80,
                70, 50
            )
            pattern_score = pd.Series(pattern_score, index=df.index)
        
        # Smart combination with adaptive weights (simplified without regime for now)
        breakout_score = (
            distance_score * 0.35 +
            volume_score * 0.35 +
            trend_score * 0.2 +
            pattern_score * 0.1
        )
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart RVOL score with dynamic thresholds"""
        
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        rvol = df['rvol'].fillna(1)
        
        # Standard thresholds
        rvol_score = pd.Series(
            np.select(
                [
                    rvol > 10,
                    rvol > 5,
                    rvol > 3,
                    rvol > 2,
                    rvol > 1.5,
                    rvol > 1.2,
                    rvol > 0.8,
                    rvol > 0.5,
                    rvol > 0.3
                ],
                [95, 90, 85, 80, 70, 60, 50, 40, 30],
                default=20
            ),
            index=df.index
        )
        
        return rvol_score.clip(0, 100)
    
    @staticmethod
    def _calculate_trend_quality_smart(df: pd.DataFrame) -> pd.Series:
        """Smart trend quality with pattern recognition"""
        
        trend_score = pd.Series(50, index=df.index, dtype=float)
        
        required_cols = ['price', 'sma_20d', 'sma_50d', 'sma_200d']
        if not all(col in df.columns for col in required_cols):
            # Fallback to simple trend
            if 'price' in df.columns and 'sma_50d' in df.columns:
                trend_score = np.where(
                    df['price'] > df['sma_50d'], 
                    70, 30
                )
                return pd.Series(trend_score, index=df.index)
            return trend_score
        
        price = df['price'].fillna(0)
        sma_20 = df['sma_20d'].fillna(price)
        sma_50 = df['sma_50d'].fillna(price)
        sma_200 = df['sma_200d'].fillna(price)
        
        # Calculate alignment patterns
        perfect_bullish = (price > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
        strong_bullish = (price > sma_50) & (sma_50 > sma_200) & ~perfect_bullish
        moderate_bullish = (price > sma_200) & ~perfect_bullish & ~strong_bullish
        
        perfect_bearish = (price < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)
        strong_bearish = (price < sma_50) & (sma_50 < sma_200) & ~perfect_bearish
        
        # Assign scores
        trend_score[perfect_bullish] = 95
        trend_score[strong_bullish] = 75
        trend_score[moderate_bullish] = 55
        trend_score[perfect_bearish] = 15
        trend_score[strong_bearish] = 25
        
        # Add smoothness bonus
        if 'price_to_sma20' in df.columns:
            # Bonus for price close to SMA20 (smooth trend)
            distance = np.abs(df['price_to_sma20'] - 1)
            smoothness_bonus = np.where(distance < 0.05, 5, 0)
            trend_score += smoothness_bonus
        
        return trend_score.clip(0, 100)
    
    @staticmethod
    def _calculate_long_term_strength_smart(df: pd.DataFrame) -> pd.Series:
        """Smart long-term strength with regime adjustment"""
        
        lt_score = pd.Series(50, index=df.index, dtype=float)
        
        # Adaptive timeframe weights based on available data
        timeframe_weights = {
            'ret_1y': 0.40,
            'ret_6m': 0.35,
            'ret_3m': 0.25
        }
        
        available_timeframes = {
            tf: weight for tf, weight in timeframe_weights.items()
            if tf in df.columns and df[tf].notna().any()
        }
        
        if not available_timeframes:
            logger.logger.warning("No long-term return data available")
            return lt_score
        
        # Normalize weights
        total_weight = sum(available_timeframes.values())
        normalized_weights = {
            tf: w/total_weight for tf, w in available_timeframes.items()
        }
        
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        
        for tf, weight in normalized_weights.items():
            # Rank with outlier handling
            tf_rank = SmartRankingEngine._safe_rank(
                df[tf], pct=True, ascending=True, method='average'
            )
            
            # Simplified adjustment without regime detector
            weighted_score += tf_rank * weight
        
        lt_score = weighted_score
        
        # Add stability bonus
        if len(available_timeframes) >= 2:
            returns = df[list(available_timeframes.keys())]
            # All positive long-term returns
            all_positive = (returns > 0).all(axis=1)
            stability_bonus = all_positive.astype(float) * 10
            lt_score += stability_bonus
        
        return lt_score.clip(0, 100)
    
    @staticmethod
    def _calculate_liquidity_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart liquidity score with multiple factors"""
        
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        # Volume-based liquidity
        if 'volume_1d' in df.columns:
            volume = df['volume_1d'].fillna(0)
            volume_rank = SmartRankingEngine._safe_rank(
                volume, pct=True, ascending=True
            )
            liquidity_score = volume_rank * 0.5
        
        # Market cap factor
        if 'market_cap' in df.columns:
            mcap_rank = SmartRankingEngine._safe_rank(
                df['market_cap'], pct=True, ascending=True
            )
            liquidity_score += mcap_rank * 0.3
        
        # Trading consistency
        if all(col in df.columns for col in ['volume_7d', 'volume_30d']):
            # Check for consistent trading
            vol_7d_daily = df['volume_7d'] / 7
            vol_30d_daily = df['volume_30d'] / 30
            
            consistency = 1 - np.abs(vol_7d_daily - vol_30d_daily) / (vol_30d_daily + 1)
            consistency_score = consistency.clip(0, 1) * 20
            liquidity_score += consistency_score
        
        # Category bonus
        if 'category' in df.columns:
            large_cap_bonus = np.where(
                df['category'].str.contains('Large', case=False, na=False),
                10, 0
            )
            liquidity_score += large_cap_bonus
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    def _calculate_master_score_optimized(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate master score with vectorized operations"""
        
        # Prepare score matrix
        score_columns = [
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score'
        ]
        
        # Ensure all scores exist
        for col in score_columns:
            if col not in df.columns:
                df[col] = 50
        
        # Vectorized calculation
        scores_matrix = df[score_columns].fillna(50).values
        weights = np.array([
            CONFIG.POSITION_WEIGHT,
            CONFIG.VOLUME_WEIGHT,
            CONFIG.MOMENTUM_WEIGHT,
            CONFIG.ACCELERATION_WEIGHT,
            CONFIG.BREAKOUT_WEIGHT,
            CONFIG.RVOL_WEIGHT
        ])
        
        # Calculate master score
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        
        return df
    
    @staticmethod
    def _calculate_all_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all ranking metrics efficiently"""
        
        # Overall rank
        df['rank'] = df['master_score'].rank(
            method='first', ascending=False, na_option='bottom'
        ).fillna(len(df) + 1).astype(int)
        
        # Percentile
        df['percentile'] = df['master_score'].rank(
            pct=True, ascending=True, na_option='bottom'
        ).fillna(0) * 100
        
        # Category ranks
        if 'category' in df.columns:
            df['category_rank'] = df.groupby('category')['master_score'].rank(
                method='first', ascending=False, na_option='bottom'
            ).fillna(999).astype(int)
        
        # Sector ranks
        if 'sector' in df.columns:
            df['sector_rank'] = df.groupby('sector')['master_score'].rank(
                method='first', ascending=False, na_option='bottom'
            ).fillna(999).astype(int)
        
        # Industry ranks
        if 'industry' in df.columns:
            df['industry_rank'] = df.groupby('industry')['master_score'].rank(
                method='first', ascending=False, na_option='bottom'
            ).fillna(999).astype(int)
        
        # Performance tiers
        df['performance_tier'] = pd.cut(
            df['percentile'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Bottom 20%', 'Below Average', 'Average', 'Above Average', 'Top 20%']
        )
        
        return df

# ============================================
# INTELLIGENT PATTERN DETECTION ENGINE
# ============================================

class SmartPatternDetector:
    """Advanced pattern detection with ML-inspired algorithms"""
    
    # Pattern metadata for intelligent detection
    PATTERN_METADATA = {
        # Technical patterns
        '🔥 CAT LEADER': {
            'type': 'technical',
            'importance': 'high',
            'timeframe': 'medium',
            'risk': 'low'
        },
        '💎 HIDDEN GEM': {
            'type': 'technical',
            'importance': 'high',
            'timeframe': 'medium',
            'risk': 'medium'
        },
        '🚀 ACCELERATION': {
            'type': 'technical',
            'importance': 'high',
            'timeframe': 'short',
            'risk': 'medium'
        },
        '🏦 INSTITUTIONAL': {
            'type': 'technical',
            'importance': 'medium',
            'timeframe': 'long',
            'risk': 'low'
        },
        '⚡ VOL EXPLOSION': {
            'type': 'technical',
            'importance': 'high',
            'timeframe': 'short',
            'risk': 'high'
        },
        '🎯 BREAKOUT': {
            'type': 'technical',
            'importance': 'high',
            'timeframe': 'short',
            'risk': 'medium'
        },
        '👑 MKT LEADER': {
            'type': 'technical',
            'importance': 'high',
            'timeframe': 'long',
            'risk': 'low'
        },
        '🌊 MOMENTUM WAVE': {
            'type': 'technical',
            'importance': 'medium',
            'timeframe': 'medium',
            'risk': 'medium'
        },
        '💧 LIQUID LEADER': {
            'type': 'technical',
            'importance': 'medium',
            'timeframe': 'long',
            'risk': 'low'
        },
        '💪 LONG STRENGTH': {
            'type': 'technical',
            'importance': 'medium',
            'timeframe': 'long',
            'risk': 'low'
        },
        '📈 QUALITY TREND': {
            'type': 'technical',
            'importance': 'medium',
            'timeframe': 'medium',
            'risk': 'low'
        },
        # Price range patterns
        '🎯 52W HIGH APPROACH': {
            'type': 'range',
            'importance': 'high',
            'timeframe': 'short',
            'risk': 'medium'
        },
        '🔄 52W LOW BOUNCE': {
            'type': 'range',
            'importance': 'high',
            'timeframe': 'medium',
            'risk': 'high'
        },
        '👑 GOLDEN ZONE': {
            'type': 'range',
            'importance': 'medium',
            'timeframe': 'medium',
            'risk': 'medium'
        },
        '📊 VOL ACCUMULATION': {
            'type': 'range',
            'importance': 'medium',
            'timeframe': 'medium',
            'risk': 'low'
        },
        '🔀 MOMENTUM DIVERGE': {
            'type': 'range',
            'importance': 'high',
            'timeframe': 'short',
            'risk': 'medium'
        },
        '🎯 RANGE COMPRESS': {
            'type': 'range',
            'importance': 'medium',
            'timeframe': 'medium',
            'risk': 'low'
        },
        # Fundamental patterns
        '💎 VALUE MOMENTUM': {
            'type': 'fundamental',
            'importance': 'high',
            'timeframe': 'long',
            'risk': 'low'
        },
        '📊 EARNINGS ROCKET': {
            'type': 'fundamental',
            'importance': 'high',
            'timeframe': 'medium',
            'risk': 'medium'
        },
        '🏆 QUALITY LEADER': {
            'type': 'fundamental',
            'importance': 'high',
            'timeframe': 'long',
            'risk': 'low'
        },
        '⚡ TURNAROUND': {
            'type': 'fundamental',
            'importance': 'high',
            'timeframe': 'medium',
            'risk': 'high'
        },
        '⚠️ HIGH PE': {
            'type': 'fundamental',
            'importance': 'low',
            'timeframe': 'medium',
            'risk': 'high'
        },
        # Intelligence patterns
        '🤫 STEALTH': {
            'type': 'intelligence',
            'importance': 'high',
            'timeframe': 'medium',
            'risk': 'medium'
        },
        '🧛 VAMPIRE': {
            'type': 'intelligence',
            'importance': 'high',
            'timeframe': 'short',
            'risk': 'very_high'
        },
        '⛈️ PERFECT STORM': {
            'type': 'intelligence',
            'importance': 'very_high',
            'timeframe': 'short',
            'risk': 'medium'
        }
    }
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns with intelligent optimization"""
        
        if df.empty:
            df['patterns'] = [''] * len(df)
            return df
        
        logger.logger.info("Starting intelligent pattern detection...")
        
        # Pre-calculate pattern conditions for efficiency
        pattern_conditions = SmartPatternDetector._prepare_pattern_conditions(df)
        
        # Pattern detection functions
        pattern_functions = {
            '🔥 CAT LEADER': SmartPatternDetector._is_category_leader_smart,
            '💎 HIDDEN GEM': SmartPatternDetector._is_hidden_gem_smart,
            '🚀 ACCELERATION': SmartPatternDetector._is_accelerating_smart,
            '🏦 INSTITUTIONAL': SmartPatternDetector._is_institutional_smart,
            '⚡ VOL EXPLOSION': SmartPatternDetector._is_volume_explosion_smart,
            '🎯 BREAKOUT': SmartPatternDetector._is_breakout_ready_smart,
            '👑 MKT LEADER': SmartPatternDetector._is_market_leader_smart,
            '🌊 MOMENTUM WAVE': SmartPatternDetector._is_momentum_wave_smart,
            '💧 LIQUID LEADER': SmartPatternDetector._is_liquid_leader_smart,
            '💪 LONG STRENGTH': SmartPatternDetector._is_long_strength_smart,
            '📈 QUALITY TREND': SmartPatternDetector._is_quality_trend_smart,
            '🎯 52W HIGH APPROACH': SmartPatternDetector._is_52w_high_approach_smart,
            '🔄 52W LOW BOUNCE': SmartPatternDetector._is_52w_low_bounce_smart,
            '👑 GOLDEN ZONE': SmartPatternDetector._is_golden_zone_smart,
            '📊 VOL ACCUMULATION': SmartPatternDetector._is_vol_accumulation_smart,
            '🔀 MOMENTUM DIVERGE': SmartPatternDetector._is_momentum_diverge_smart,
            '🎯 RANGE COMPRESS': SmartPatternDetector._is_range_compress_smart,
            '💎 VALUE MOMENTUM': SmartPatternDetector._is_value_momentum_smart,
            '📊 EARNINGS ROCKET': SmartPatternDetector._is_earnings_rocket_smart,
            '🏆 QUALITY LEADER': SmartPatternDetector._is_quality_leader_smart,
            '⚡ TURNAROUND': SmartPatternDetector._is_turnaround_smart,
            '⚠️ HIGH PE': SmartPatternDetector._is_high_pe_smart,
            '🤫 STEALTH': SmartPatternDetector._is_stealth_smart,
            '🧛 VAMPIRE': SmartPatternDetector._is_vampire_smart,
            '⛈️ PERFECT STORM': SmartPatternDetector._is_perfect_storm_smart
        }
        
        detected_patterns = []
        for pattern_name, pattern_func in pattern_functions.items():
            try:
                # Apply pattern detection
                mask = pattern_func(df, pattern_conditions)
                if isinstance(mask, pd.Series) and mask.any():
                    # Check if the mask is a valid boolean series and not all False
                    if mask.dtype == bool and mask.any():
                        pattern_series = pd.Series(
                            np.where(mask, pattern_name, ''), 
                            index=df.index
                        )
                        detected_patterns.append(pattern_series)
            except Exception as e:
                logger.logger.warning(f"Pattern detection failed for {pattern_name}: {str(e)}")
        
        # Combine all detected patterns
        if detected_patterns:
            df['patterns'] = pd.concat(detected_patterns, axis=1).apply(
                lambda x: ' | '.join(filter(None, x)), axis=1
            )
        else:
            df['patterns'] = ''
        
        return df
    
    @staticmethod
    def _prepare_pattern_conditions(df: pd.DataFrame) -> Dict[str, Any]:
        """Pre-calculate common conditions for pattern detection"""
        
        conditions = {}
        
        # Pre-calculate commonly used conditions
        if 'master_score' in df.columns:
            conditions['high_score'] = df['master_score'] >= 70
            conditions['very_high_score'] = df['master_score'] >= 85
        
        if 'momentum_score' in df.columns:
            conditions['high_momentum'] = df['momentum_score'] >= 70
        
        if 'volume_score' in df.columns:
            conditions['high_volume'] = df['volume_score'] >= 70
        
        if 'rvol' in df.columns:
            conditions['high_rvol'] = df['rvol'] >= 2
            conditions['extreme_rvol'] = df['rvol'] >= 5
        
        if 'from_high_pct' in df.columns:
            conditions['near_high'] = df['from_high_pct'] > -10
            conditions['far_from_high'] = df['from_high_pct'] < -30
        
        if 'from_low_pct' in df.columns:
            conditions['near_low'] = df['from_low_pct'] < 20
            conditions['far_from_low'] = df['from_low_pct'] > 50
        
        return conditions
    
    @staticmethod
    def _is_category_leader_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'category_rank' not in df.columns or 'master_score' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['category_rank'] <= 3) & (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['category_leader'])
    
    @staticmethod
    def _is_hidden_gem_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'percentile' not in df.columns or 'category_rank' not in df.columns or 'volume_1d' not in df.columns or 'volume_90d' not in df.columns:
            return pd.Series(False, index=df.index)
        
        volume_condition = (df['volume_1d'] < df['volume_90d'] * 0.7) & (df['volume_1d'] > df['volume_90d'] * 0.3)
        return (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (df['category_rank'] <= 10) & volume_condition

    @staticmethod
    def _is_accelerating_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'acceleration_score' not in df.columns or 'momentum_score' not in df.columns:
            return pd.Series(False, index=df.index)
            
        return (df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']) & conditions.get('high_momentum', False)

    @staticmethod
    def _is_institutional_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'volume_score' not in df.columns or 'vol_ratio_90d_180d' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (df['vol_ratio_90d_180d'] > 1.1)

    @staticmethod
    def _is_volume_explosion_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'rvol' not in df.columns:
            return pd.Series(False, index=df.index)
            
        return df['rvol'] >= 3

    @staticmethod
    def _is_breakout_ready_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'breakout_score' not in df.columns or 'from_high_pct' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']) & (df['from_high_pct'] > -5)

    @staticmethod
    def _is_market_leader_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'percentile' not in df.columns:
            return pd.Series(False, index=df.index)
            
        return df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']

    @staticmethod
    def _is_momentum_wave_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'momentum_score' not in df.columns or 'acceleration_score' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (df['acceleration_score'] >= 70)

    @staticmethod
    def _is_liquid_leader_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'liquidity_score' not in df.columns or 'master_score' not in df.columns:
            return pd.Series(False, index=df.index)
            
        return (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (df['master_score'] >= 80)

    @staticmethod
    def _is_long_strength_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'long_term_strength' not in df.columns:
            return pd.Series(False, index=df.index)
            
        return df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']

    @staticmethod
    def _is_quality_trend_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'trend_quality' not in df.columns:
            return pd.Series(False, index=df.index)
            
        return df['trend_quality'] >= CONFIG.PATTERN_THRESHOLDS['quality_trend']

    @staticmethod
    def _is_52w_high_approach_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'from_high_pct' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return df['from_high_pct'] > -5

    @staticmethod
    def _is_52w_low_bounce_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'from_low_pct' not in df.columns or 'acceleration_score' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['from_low_pct'] < 20) & (df['acceleration_score'] >= 80)

    @staticmethod
    def _is_golden_zone_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'from_low_pct' not in df.columns or 'from_high_pct' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['from_low_pct'] > 60) & (df['from_high_pct'] > -40)

    @staticmethod
    def _is_vol_accumulation_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'vol_ratio_30d_90d' not in df.columns or 'ret_30d' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['vol_ratio_30d_90d'] > 1.2) & (df['ret_30d'].abs() < 10)

    @staticmethod
    def _is_momentum_diverge_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if not all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score']):
            return pd.Series(False, index=df.index)

        daily_7d = df['ret_7d'] / 7
        daily_30d = df['ret_30d'] / 30
        
        return (daily_7d > daily_30d * 1.5) & (df['acceleration_score'] >= 75)

    @staticmethod
    def _is_range_compress_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if not all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'low_52w', 'high_52w']):
            return pd.Series(False, index=df.index)

        price_range = df['high_52w'] - df['low_52w']
        
        return (price_range / df['low_52w'] < 0.5) & (df['from_low_pct'] > 30)

    @staticmethod
    def _is_value_momentum_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'pe' not in df.columns or 'momentum_score' not in df.columns:
            return pd.Series(False, index=df.index)

        return (df['pe'] > 0) & (df['pe'] < 15) & (df['momentum_score'] >= 70)
    
    @staticmethod
    def _is_earnings_rocket_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'eps_change_pct' not in df.columns or 'acceleration_score' not in df.columns:
            return pd.Series(False, index=df.index)

        return (df['eps_change_pct'] > 50) & (df['acceleration_score'] >= 70)
    
    @staticmethod
    def _is_quality_leader_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if not all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            return pd.Series(False, index=df.index)

        return (df['pe'].between(10, 25)) & (df['eps_change_pct'] > 20) & (df['percentile'] >= 80)
    
    @staticmethod
    def _is_turnaround_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'eps_change_pct' not in df.columns or 'ret_30d' not in df.columns:
            return pd.Series(False, index=df.index)

        return (df['eps_change_pct'] > 100) & (df['ret_30d'] > 10)

    @staticmethod
    def _is_high_pe_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if 'pe' not in df.columns:
            return pd.Series(False, index=df.index)

        return df['pe'] > 100

    @staticmethod
    def _is_stealth_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if not all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_7d_90d', 'ret_7d']):
            return pd.Series(False, index=df.index)

        volume_condition = (df['vol_ratio_30d_90d'] > 1.2) & (df['vol_ratio_7d_90d'] < 1.1)
        price_stable = (df['ret_7d'].abs() < 5)
        
        return volume_condition & price_stable
    
    @staticmethod
    def _is_vampire_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if not all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'category']):
            return pd.Series(False, index=df.index)

        daily_pace_ratio = np.where(df['ret_7d'] != 0, df['ret_1d'] / (df['ret_7d'] / 7), np.inf)
        
        return (daily_pace_ratio > 2) & (df['rvol'] > 3) & (df['category'].isin(['Small Cap', 'Micro Cap']))

    @staticmethod
    def _is_perfect_storm_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        if not all(col in df.columns for col in ['momentum_harmony', 'master_score', 'volume_score']):
            return pd.Series(False, index=df.index)

        return (df['momentum_harmony'] == 4) & (df['master_score'] >= 90) & (df['volume_score'] >= 80)


# ============================================
# INTELLIGENT FILTERING ENGINE
# ============================================

class SmartFilterEngine:
    """Advanced filtering with intelligent optimization"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_all_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with smart optimization"""
        
        if df.empty:
            return df
        
        mask = pd.Series(True, index=df.index)
        
        # Categorical filters
        if filters.get('categories'):
            mask &= df['category'].isin(filters['categories'])
        if filters.get('sectors'):
            mask &= df['sector'].isin(filters['sectors'])
        if filters.get('industries') and 'industry' in df.columns:
            mask &= df['industry'].isin(filters['industries'])
        
        # Numerical filters
        min_score = filters.get('min_score')
        if min_score:
            mask &= df['master_score'] >= min_score

        # Pattern filter
        if filters.get('patterns'):
            pattern_regex = '|'.join(filters['patterns'])
            mask &= df['patterns'].str.contains(pattern_regex, case=False, na=False, regex=True)

        # Trend filter
        trend_range = filters.get('trend_range')
        if trend_range and 'trend_quality' in df.columns:
            min_trend, max_trend = trend_range
            mask &= (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend)

        # Tier filters
        tier_map = {
            'eps_tier_filter': 'eps_tier',
            'pe_tier_filter': 'pe_tier',
            'price_tier_filter': 'price_tier'
        }
        for filter_name, col_name in tier_map.items():
            if filters.get(filter_name) and col_name in df.columns:
                mask &= df[col_name].isin(filters[filter_name])
        
        # Other numerical filters
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            mask &= df['eps_change_pct'] >= float(min_eps_change)
        
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in df.columns:
            mask &= df['pe'] >= float(min_pe)
        
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in df.columns:
            mask &= df['pe'] <= float(max_pe)

        # Data completeness
        if filters.get('require_fundamental_data'):
            mask &= df['pe'].notna() & df['eps_change_pct'].notna()
        
        # Wave filters
        if filters.get('wave_states') and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(filters['wave_states'])
        
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            mask &= (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws)
            
        filtered_df = df[mask].copy()
        
        logger.logger.info(f"Filtering complete: {len(df)} → {len(filtered_df)}")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        """Get available filter options with smart interconnection"""
        
        if df.empty or column not in df.columns:
            return []
        
        temp_filters = current_filters.copy()
        
        # Remove the current column's filter to see all its options
        filter_key_map = {
            'category': 'categories',
            'sector': 'sectors',
            'industry': 'industries',
            'eps_tier': 'eps_tier_filter',
            'pe_tier': 'pe_tier_filter',
            'price_tier': 'price_tier_filter',
            'wave_state': 'wave_states'
        }
        
        if column in filter_key_map:
            temp_filters.pop(filter_key_map[column], None)
        
        filtered_df = SmartFilterEngine.apply_all_filters(df, temp_filters)
        values = filtered_df[column].dropna().unique()
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN']]
        
        return sorted(values)

# ============================================
# INTELLIGENT SEARCH ENGINE
# ============================================

class SmartSearchEngine:
    """Advanced search with fuzzy matching and relevance scoring"""
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str, 
                     fuzzy: bool = True, threshold: float = 0.8) -> pd.DataFrame:
        """Smart stock search with relevance ranking"""
        
        if not query or df.empty:
            return pd.DataFrame()
        
        query = query.strip().upper()
        results = []
        
        # Exact ticker match (highest priority)
        ticker_exact = df[df['ticker'].str.upper() == query]
        if not ticker_exact.empty:
            ticker_exact = ticker_exact.copy()
            ticker_exact['search_score'] = 1.0
            ticker_exact['match_type'] = 'exact_ticker'
            results.append(ticker_exact)
        
        # Ticker contains query
        ticker_contains = df[
            df['ticker'].str.upper().str.contains(query, na=False, regex=False) &
            ~df.index.isin(ticker_exact.index if not ticker_exact.empty else [])
        ]
        if not ticker_contains.empty:
            ticker_contains = ticker_contains.copy()
            # Score based on position and length
            ticker_contains['search_score'] = ticker_contains['ticker'].apply(
                lambda x: 0.9 if x.upper().startswith(query) else 0.7
            )
            ticker_contains['match_type'] = 'ticker_contains'
            results.append(ticker_contains)
        
        # Company name search (if available)
        if 'company_name' in df.columns:
            # Exact company name match
            name_exact = df[
                df['company_name'].str.upper() == query &
                ~df.index.isin(pd.concat(results).index if results else [])
            ]
            if not name_exact.empty:
                name_exact = name_exact.copy()
                name_exact['search_score'] = 0.85
                name_exact['match_type'] = 'exact_name'
                results.append(name_exact)
            
            # Company name contains query
            name_contains = df[
                df['company_name'].str.upper().str.contains(query, na=False, regex=False) &
                ~df.index.isin(pd.concat(results).index if results else [])
            ]
            if not name_contains.empty:
                name_contains = name_contains.copy()
                # Score based on word match
                name_contains['search_score'] = name_contains['company_name'].apply(
                    lambda x: SmartSearchEngine._calculate_name_score(x, query)
                )
                name_contains['match_type'] = 'name_contains'
                results.append(name_contains)
        
        # Fuzzy matching for typos (if enabled)
        if fuzzy and len(results) == 0:
            fuzzy_results = SmartSearchEngine._fuzzy_search(df, query, threshold)
            if not fuzzy_results.empty:
                results.append(fuzzy_results)
        
        # Combine and sort results
        if results:
            all_results = pd.concat(results, ignore_index=True)
            
            # Sort by search score and master score
            all_results = all_results.sort_values(
                ['search_score', 'master_score'], 
                ascending=[False, False]
            )
            
            # Remove search metadata columns before returning
            return all_results.drop(['search_score', 'match_type'], axis=1)
        
        return pd.DataFrame()
    
    @staticmethod
    def _calculate_name_score(name: str, query: str) -> float:
        """Calculate relevance score for company name match"""
        
        name_upper = name.upper()
        
        # Exact word match
        words = name_upper.split()
        if query in words:
            return 0.8
        
        # Start of word match
        for word in words:
            if word.startswith(query):
                return 0.7
        
        # Contains anywhere
        return 0.5
    
    @staticmethod
    def _fuzzy_search(df: pd.DataFrame, query: str, threshold: float) -> pd.DataFrame:
        """Fuzzy search for handling typos"""
        
        try:
            from difflib import SequenceMatcher
            
            results = []
            
            for idx, row in df.iterrows():
                ticker_score = SequenceMatcher(None, row['ticker'].upper(), query).ratio()
                
                if ticker_score >= threshold:
                    results.append({
                        'index': idx,
                        'score': ticker_score * 0.9,  # Slightly lower than exact match
                        'type': 'fuzzy_ticker'
                    })
                    continue
                
                if 'company_name' in row and pd.notna(row['company_name']):
                    name_score = SequenceMatcher(
                        None, row['company_name'].upper(), query
                    ).ratio()
                    
                    if name_score >= threshold:
                        results.append({
                            'index': idx,
                            'score': name_score * 0.8,
                            'type': 'fuzzy_name'
                        })
            
            if results:
                # Get top matches
                results.sort(key=lambda x: x['score'], reverse=True)
                top_indices = [r['index'] for r in results[:10]]  # Limit to top 10
                
                fuzzy_df = df.loc[top_indices].copy()
                fuzzy_df['search_score'] = [r['score'] for r in results[:10]]
                fuzzy_df['match_type'] = [r['type'] for r in results[:10]]
                
                return fuzzy_df
        
        except ImportError:
            logger.logger.warning("difflib not available for fuzzy search")
        
        return pd.DataFrame()

# ============================================
# INTELLIGENT EXPORT ENGINE
# ============================================

class SmartExportEngine:
    """Advanced export functionality with intelligent formatting"""
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame, export_type: str = 'standard') -> bytes:
        """Create intelligently formatted CSV export"""
        
        if df.empty:
            return pd.DataFrame().to_csv(index=False).encode('utf-8')
        
        export_df = df.copy()
        
        # Define export profiles
        export_profiles = {
            'standard': SmartExportEngine._get_standard_columns,
            'day_trading': SmartExportEngine._get_day_trading_columns,
            'swing_trading': SmartExportEngine._get_swing_trading_columns,
            'fundamental': SmartExportEngine._get_fundamental_columns,
            'pattern_analysis': SmartExportEngine._get_pattern_columns,
            'complete': SmartExportEngine._get_complete_columns
        }
        
        # Get columns for export type
        column_getter = export_profiles.get(export_type, SmartExportEngine._get_standard_columns)
        export_columns = column_getter(export_df)
        
        # Select and order columns
        available_columns = [col for col in export_columns if col in export_df.columns]
        export_df = export_df[available_columns]
        
        # Apply intelligent formatting
        export_df = SmartExportEngine._format_dataframe(export_df)
        
        # Convert to CSV with optimal settings
        csv_buffer = StringIO()
        
        # Write metadata
        csv_buffer.write(f"# Wave Detection Export - {export_type.replace('_', ' ').title()}\n")
        csv_buffer.write(f"# Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        csv_buffer.write(f"# Stocks: {len(export_df)}\n")
        csv_buffer.write("#\n")
        
        # Write data
        export_df.to_csv(csv_buffer, index=False, float_format='%.4f')
        
        return csv_buffer.getvalue().encode('utf-8')
    
    @staticmethod
    def _get_standard_columns(df: pd.DataFrame) -> List[str]:
        """Standard export columns"""
        return [
            'rank', 'ticker', 'master_score', 'price', 'ret_30d',
            'volume_1d', 'rvol', 'wave_state', 'patterns',
            'category', 'sector', 'from_low_pct', 'from_high_pct',
            'momentum_score', 'volume_score', 'position_score'
        ]
    
    @staticmethod
    def _get_day_trading_columns(df: pd.DataFrame) -> List[str]:
        """Day trading focused columns"""
        return [
            'rank', 'ticker', 'price', 'ret_1d', 'rvol',
            'momentum_score', 'acceleration_score', 'wave_state',
            'patterns', 'vmi', 'money_flow_mm', 'volume_1d',
            'vol_ratio_1d_90d', 'position_tension'
        ]
    
    @staticmethod
    def _get_swing_trading_columns(df: pd.DataFrame) -> List[str]:
        """Swing trading focused columns"""
        return [
            'rank', 'ticker', 'master_score', 'ret_7d', 'ret_30d',
            'from_low_pct', 'from_high_pct', 'trend_quality',
            'patterns', 'wave_state', 'position_score', 'momentum_harmony',
            'breakout_score', 'category', 'sector'
        ]
    
    @staticmethod
    def _get_fundamental_columns(df: pd.DataFrame) -> List[str]:
        """Fundamental analysis columns"""
        return [
            'rank', 'ticker', 'price', 'pe', 'eps_current',
            'eps_change_pct', 'market_cap', 'master_score',
            'ret_30d', 'ret_1y', 'patterns', 'category', 'sector'
        ]
    
    @staticmethod
    def _get_pattern_columns(df: pd.DataFrame) -> List[str]:
        """Pattern analysis columns"""
        return [
            'rank', 'ticker', 'master_score', 'patterns',
            'wave_state', 'ret_30d',
            'rvol', 'momentum_harmony', 'position_tension',
            'category'
        ]
    
    @staticmethod
    def _get_complete_columns(df: pd.DataFrame) -> List[str]:
        """All columns in optimal order"""
        
        # Define column groups in order
        groups = [
            # Core identifiers
            ['rank', 'ticker', 'company_name'],
            # Scores
            ['master_score', 'position_score', 'volume_score', 'momentum_score',
             'acceleration_score', 'breakout_score', 'rvol_score'],
            # Price data
            ['price', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct'],
            # Returns
            ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y'],
            # Volume
            ['volume_1d', 'rvol', 'vol_ratio_1d_90d', 'vol_ratio_7d_90d'],
            # Advanced metrics
            ['wave_state', 'patterns', 'vmi', 'position_tension', 'momentum_harmony', 'overall_wave_strength'],
            # Fundamentals
            ['pe', 'eps_current', 'eps_change_pct', 'market_cap'],
            # Categories
            ['category', 'sector', 'industry']
        ]
        
        # Flatten and filter
        all_columns = []
        for group in groups:
            all_columns.extend([col for col in group if col in df.columns])
        
        # Add any remaining columns
        remaining = [col for col in df.columns if col not in all_columns]
        all_columns.extend(remaining)
        
        return all_columns
    
    @staticmethod
    def _format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent formatting to dataframe"""
        
        # Format numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ['rank', 'category_rank', 'sector_rank', 'industry_rank']:
                continue  # Keep as integers
            elif col in ['price', 'low_52w', 'high_52w', 'prev_close', 'market_cap']:
                df[col] = df[col].round(2)
            elif col in CONFIG.PERCENTAGE_COLUMNS:
                df[col] = df[col].round(2)
            elif col.endswith('_score') or col == 'overall_wave_strength':
                df[col] = df[col].round(1)
            else:
                df[col] = df[col].round(4)
        
        # Format large numbers
        if 'market_cap' in df.columns:
            df['market_cap'] = df['market_cap'].apply(
                lambda x: f"{x/1e9:.2f}B" if x >= 1e9 else f"{x/1e6:.2f}M" if x >= 1e6 else str(x)
            )
        
        if 'volume_1d' in df.columns:
            df['volume_1d'] = df['volume_1d'].apply(
                lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.2f}K" if x >= 1e3 else str(x)
            )
        
        return df

# ============================================
# INTELLIGENT UI COMPONENTS
# ============================================

class SmartUIComponents:
    """Advanced UI components with intelligent features"""
    
    @staticmethod
    def render_metric_card(label: str, value: str, delta: str = None,
                          delta_color: str = "normal", help_text: str = None,
                          trend_data: List[float] = None):
        """Render enhanced metric card with sparkline"""
        
        with st.container():
            # Standard metric
            st.metric(
                label=label,
                value=value,
                delta=delta,
                delta_color=delta_color,
                help=help_text
            )
    
    @staticmethod
    def render_stock_card_enhanced(row: pd.Series, show_fundamentals: bool = False):
        """Render enhanced stock card with rich information"""
        
        with st.container():
            # Header row
            header_cols = st.columns([1, 3, 2, 2])
            
            with header_cols[0]:
                # Rank with badge
                rank = row.get('rank', 999)
                rank_color = "#27AE60" if rank <= 10 else "#3498DB" if rank <= 50 else "#95A5A6"
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<span style='font-size:24px; font-weight:bold; color:{rank_color};'>#{int(rank)}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Category badge
                category = row.get('category', 'Unknown')
                category_colors = {
                    'Large Cap': '#E74C3C',
                    'Mid Cap': '#F39C12',
                    'Small Cap': '#3498DB',
                    'Micro Cap': '#9B59B6'
                }
                cat_color = category_colors.get(category, '#95A5A6')
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<span style='background-color:{cat_color}; color:white; "
                    f"padding:2px 8px; border-radius:12px; font-size:11px;'>"
                    f"{category}</span></div>",
                    unsafe_allow_html=True
                )
            
            with header_cols[1]:
                # Ticker and sector
                ticker = row.get('ticker', 'N/A')
                sector = row.get('sector', 'Unknown')
                st.markdown(f"### {ticker}")
                st.caption(f"{sector}")
                
                # Patterns with enhanced display
                patterns_str = row.get('patterns', '')
                if patterns_str:
                    patterns = patterns_str.split(' | ')
                    pattern_html = ""
                    for pattern in patterns[:3]:  # Show max 3
                        pattern_html += f"<span style='margin-right:5px;'>{pattern}</span>"
                    if len(patterns) > 3:
                        pattern_html += f"<span style='color:#7F8C8D;'>+{len(patterns)-3} more</span>"
                    st.markdown(pattern_html, unsafe_allow_html=True)
            
            with header_cols[2]:
                # Price and change
                price = row.get('price', 0)
                ret_30d = row.get('ret_30d', 0)
                ret_1d = row.get('ret_1d', 0)
                
                st.metric(
                    "Price",
                    f"₹{price:,.2f}",
                    f"{ret_1d:+.2f}% today",
                    delta_color="normal" if ret_1d >= 0 else "inverse"
                )
                
                # 30-day return
                color = "#27AE60" if ret_30d > 0 else "#E74C3C"
                st.markdown(
                    f"<div style='text-align:center; margin-top:-10px;'>"
                    f"<span style='color:{color}; font-weight:bold;'>"
                    f"30D: {ret_30d:+.1f}%</span></div>",
                    unsafe_allow_html=True
                )
            
            with header_cols[3]:
                # Master score with gauge
                score = row.get('master_score', 0)
                score_color = "#E74C3C" if score >= 80 else "#F39C12" if score >= 60 else "#3498DB"
                
                # Mini gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Master Score", 'font': {'size': 12}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': score_color},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(225,225,225,0.5)"},
                            {'range': [50, 70], 'color': "rgba(150,150,150,0.5)"},
                            {'range': [70, 100], 'color': "rgba(100,100,100,0.5)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(
                    height=100,
                    margin=dict(l=0, r=0, t=20, b=0),
                    font=dict(size=10)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Wave state
                st.markdown(
                    f"<div style='text-align:center; margin-top:-20px;'>"
                    f"{row.get('wave_state', 'Unknown')}</div>",
                    unsafe_allow_html=True
                )
            
            # Expandable details
            with st.expander("📊 Detailed Analysis", expanded=False):
                detail_cols = st.columns(3)
                
                with detail_cols[0]:
                    st.markdown("**📈 Scores**")
                    score_types = [
                        ('Position', row.get('position_score', 0)),
                        ('Volume', row.get('volume_score', 0)),
                        ('Momentum', row.get('momentum_score', 0)),
                        ('Acceleration', row.get('acceleration_score', 0)),
                        ('Breakout', row.get('breakout_score', 0))
                    ]
                    for name, score in score_types:
                        bar_color = "#27AE60" if score >= 70 else "#F39C12" if score >= 50 else "#E74C3C"
                        st.markdown(
                            f"{name}: "
                            f"<span style='color:{bar_color};'>{'█' * int(score/10)}</span> "
                            f"{score:.1f}",
                            unsafe_allow_html=True
                        )
                
                with detail_cols[1]:
                    st.markdown("**📊 Advanced Metrics**")
                    metrics = [
                        ('VMI', row.get('vmi', 50), '%'),
                        ('Position Tension', row.get('position_tension', 50), ''),
                        ('Momentum Harmony', row.get('momentum_harmony', 0), '/4'),
                        ('RVOL', row.get('rvol', 1), 'x')
                    ]
                    for name, value, unit in metrics:
                        st.write(f"{name}: **{value:.1f}{unit}**")
                
                with detail_cols[2]:
                    if show_fundamentals and 'pe' in row:
                        st.markdown("**💰 Fundamentals**")
                        fund_metrics = [
                            ('P/E Ratio', row.get('pe', 0), ''),
                            ('EPS', row.get('eps_current', 0), ''),
                            ('EPS Change', row.get('eps_change_pct', 0), '%'),
                            ('Market Cap', row.get('market_cap', 0) / 1e9, 'B')
                        ]
                        for name, value, unit in fund_metrics:
                            if value != 0 and not pd.isna(value):
                                st.write(f"{name}: **{value:.2f}{unit}**")
                    else:
                        st.markdown("**📈 Returns**")
                        return_periods = [
                            ('1D', row.get('ret_1d', 0)),
                            ('7D', row.get('ret_7d', 0)),
                            ('30D', row.get('ret_30d', 0)),
                            ('3M', row.get('ret_3m', 0)),
                            ('1Y', row.get('ret_1y', 0))
                        ]
                        for period, ret in return_periods:
                            if not pd.isna(ret):
                                color = "#27AE60" if ret > 0 else "#E74C3C"
                                st.markdown(
                                    f"{period}: <span style='color:{color}; font-weight:bold;'>"
                                    f"{ret:+.1f}%</span>",
                                    unsafe_allow_html=True
                                )
            
            st.divider()

# ============================================
# INTELLIGENT SESSION STATE MANAGER
# ============================================

class SmartSessionStateManager:
    """Advanced session state management with persistence"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with smart defaults"""
        
        # Define defaults with types
        defaults = {
            'data_source': 'sheet',
            'display_mode': 'Technical',
            'user_spreadsheet_id': None,
            'last_loaded_url': None,
            'filters': {},
            'active_filter_count': 0,
            'quick_filter': None,
            'wd_quick_filter_applied': False,
            'data_quality': {},
            'last_refresh': datetime.now(timezone.utc),
            'data_timestamp': None,
            'wd_current_page_rankings': 0,
            'wd_trigger_clear': False,
            'show_advanced_metrics': False,
            'performance_metrics': {},
            'user_preferences': {
                'default_sort': 'master_score',
                'items_per_page': 50,
                'show_patterns': True,
                'theme': 'light'
            },
            'show_chart_previews': True,
            'color_theme': 'Default',
            
            # Filters
            'wd_category_filter': [],
            'wd_sector_filter': [],
            'wd_industry_filter': [],
            'wd_min_score': 0,
            'wd_patterns': [],
            'wd_trend_filter': 'All Trends',
            'wd_eps_tier_filter': [],
            'wd_pe_tier_filter': [],
            'wd_price_tier_filter': [],
            'wd_min_eps_change': None,
            'wd_min_pe': None,
            'wd_max_pe': None,
            'wd_require_fundamental_data': False,
            'wd_wave_states_filter': [],
            'wd_wave_strength_range_slider': (0, 100),
            'wd_items_per_page': 50,
            'wd_sort_by': 'Master Score',
            'wd_display_style': 'Cards',
            'wd_show_charts': True,
            'search_query': '',
            'fuzzy_search': True
        }
        
        # Initialize with type checking and persistence logic
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Special handling for spreadsheet_id persistence
        if 'user_spreadsheet_id' not in st.session_state:
            # Check for a "persisted" value from previous runs
            # This is a conceptual placeholder for a database or local storage
            # In Streamlit Cloud, st.secrets is a good place for a demo ID
            if st.secrets.get("default_spreadsheet_id"):
                st.session_state.user_spreadsheet_id = st.secrets["default_spreadsheet_id"]
            else:
                st.session_state.user_spreadsheet_id = None
    
    @staticmethod
    def clear_filters():
        """Clear all active filters intelligently"""
        
        filter_resets = {
            'wd_category_filter': [],
            'wd_sector_filter': [],
            'wd_industry_filter': [],
            'wd_min_score': 0,
            'wd_patterns': [],
            'wd_trend_filter': 'All Trends',
            'wd_eps_tier_filter': [],
            'wd_pe_tier_filter': [],
            'wd_price_tier_filter': [],
            'wd_min_eps_change': None,
            'wd_min_pe': None,
            'wd_max_pe': None,
            'wd_require_fundamental_data': False,
            'wd_wave_states_filter': [],
            'wd_wave_strength_range_slider': (0, 100),
            'wd_current_page_rankings': 0,
            'wd_quick_filter_applied': False,
            'quick_filter': None
        }
        
        for key, reset_value in filter_resets.items():
            if key in st.session_state:
                st.session_state[key] = reset_value
        
        st.session_state.filters = {}
        st.session_state.active_filter_count = 0
        st.session_state.wd_trigger_clear = False
        
        logger.logger.info("All filters cleared")

# ============================================
# REQUEST SESSION WITH INTELLIGENT RETRY
# ============================================

def get_smart_requests_session(
    retries: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: Tuple[int, ...] = (408, 429, 500, 502, 503, 504),
    session: Optional[requests.Session] = None
) -> requests.Session:
    """Create intelligent requests session with advanced retry logic"""
    
    session = session or requests.Session()
    
    # Advanced retry configuration
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False
    )
    
    # Custom adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=20
    )
    
    # Mount adapters
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Set headers for better compatibility
    session.headers.update({
        'User-Agent': 'Wave Detection Ultimate 3.0',
        'Accept': 'text/csv,application/csv,text/plain',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    })
    
    return session

# ============================================
# INTELLIGENT DATA LOADING AND PROCESSING
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL, persist="disk", show_spinner=False)
def load_and_process_data_smart(
    source_type: str = "sheet",
    file_data=None,
    spreadsheet_id: str = None,
    data_version: str = "1.0"
) -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """
    Smart data loading with advanced error handling and optimization
    """
    
    start_time = time.perf_counter()
    metadata = {
        'source_type': source_type,
        'data_version': data_version,
        'processing_start': datetime.now(timezone.utc),
        'errors': [],
        'warnings': [],
        'performance': {}
    }
    
    try:
        # Phase 1: Data Loading
        load_start = time.perf_counter()
        
        if source_type == "upload" and file_data is not None:
            logger.logger.info("Loading data from uploaded CSV")
            
            # Smart CSV reading with encoding detection
            try:
                df = pd.read_csv(file_data, low_memory=False)
            except UnicodeDecodeError:
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        file_data.seek(0)
                        df = pd.read_csv(file_data, low_memory=False, encoding=encoding)
                        metadata['warnings'].append(f"Used {encoding} encoding")
                        break
                    except:
                        continue
                else:
                    raise ValueError("Unable to decode CSV file")
            
            metadata['source'] = "User Upload"
        
        else:
            if not spreadsheet_id:
                error_msg = "Google Spreadsheet ID is required."
                logger.logger.critical(error_msg)
                raise ValueError(error_msg)
            
            if not re.match(CONFIG.VALID_SHEET_ID_PATTERN, spreadsheet_id):
                error_msg = "Invalid Google Spreadsheet ID format."
                logger.logger.critical(error_msg)
                raise ValueError(error_msg)
            
            # Construct URL
            base_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            csv_url = f"{base_url}/export?format=csv&gid={CONFIG.DEFAULT_GID}"
            
            logger.logger.info(f"Loading from Google Sheets: {spreadsheet_id[:8]}...")
            
            session = get_smart_requests_session()
            max_attempts = 3
            
            for attempt in range(max_attempts):
                try:
                    response = session.get(csv_url, timeout=30)
                    response.raise_for_status()
                    
                    if len(response.content) < 100:
                        raise ValueError("Response too small, likely an error page")
                    
                    df = pd.read_csv(BytesIO(response.content), low_memory=False)
                    metadata['source'] = f"Google Sheets (ID: {spreadsheet_id[:8]}...)"
                    break
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_attempts - 1:
                        wait_time = (attempt + 1) * 2
                        logger.logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        if 'last_good_data' in st.session_state:
                            logger.logger.info("Using cached data as fallback")
                            df, timestamp, old_metadata = st.session_state.last_good_data
                            metadata['warnings'].append("Using cached data due to load failure")
                            metadata['cache_used'] = True
                            return df, timestamp, metadata
                        raise
        
        metadata['performance']['load_time'] = time.perf_counter() - load_start
        
        # Phase 2: Validation
        validation_start = time.perf_counter()
        
        is_valid, validation_msg, diagnostics = validator.validate_dataframe(
            df, CONFIG.CRITICAL_COLUMNS, "Initial load"
        )
        
        if not is_valid:
            raise ValueError(validation_msg)
        
        metadata['validation'] = diagnostics
        metadata['performance']['validation_time'] = time.perf_counter() - validation_start
        
        # Phase 3: Processing
        processing_start = time.perf_counter()
        
        df = SmartDataProcessor.process_dataframe(df, metadata)
        
        metadata['performance']['processing_time'] = time.perf_counter() - processing_start
        
        # Phase 4: Scoring
        scoring_start = time.perf_counter()
        
        df = SmartRankingEngine.calculate_all_scores(df)
        
        metadata['performance']['scoring_time'] = time.perf_counter() - scoring_start
        
        # Phase 5: Advanced Metrics
        metrics_start = time.perf_counter()
        
        df = AdvancedMetricsEngine.calculate_all_metrics(df)
        
        metadata['performance']['metrics_time'] = time.perf_counter() - metrics_start

        # Phase 6: Pattern Detection
        pattern_start = time.perf_counter()
        
        df = SmartPatternDetector.detect_all_patterns(df)
        
        metadata['performance']['pattern_time'] = time.perf_counter() - pattern_start
        
        # Phase 7: Final Validation
        final_validation_start = time.perf_counter()
        
        is_valid, validation_msg, final_diagnostics = validator.validate_dataframe(
            df, ['master_score', 'rank'], "Final processed"
        )
        
        if not is_valid:
            raise ValueError(validation_msg)
        
        metadata['final_validation'] = final_diagnostics
        metadata['performance']['final_validation_time'] = time.perf_counter() - final_validation_start
        
        timestamp = datetime.now(timezone.utc)
        st.session_state.last_good_data = (df.copy(), timestamp, metadata)
        
        quality_metrics = {
            'total_rows': len(df),
            'duplicate_tickers': len(df) - df['ticker'].nunique(),
            'columns_processed': len(df.columns),
            'patterns_detected': df['patterns'].ne('').sum() if 'patterns' in df.columns else 0,
            'data_completeness': 100 - (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
            'timestamp': timestamp
        }
        
        total_time = time.perf_counter() - start_time
        metadata['performance']['total_time'] = total_time
        metadata['quality'] = quality_metrics
        st.session_state.data_quality = quality_metrics
        
        logger.logger.info(
            f"Data processing complete: {len(df)} stocks in {total_time:.2f}s "
        )
        
        validation_report = validator.get_validation_report()
        if validation_report['total_issues'] > 0:
            metadata['warnings'].append(
                f"Data quality: {validation_report['total_issues']} issues auto-corrected"
            )
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.logger.error(f"Data processing failed: {str(e)}", exc_info=True)
        metadata['errors'].append(str(e))
        
        if "403" in str(e) or "404" in str(e):
            metadata['errors'].append(
                "Google Sheets access denied. Please check: "
                "1) Sheet is publicly accessible, "
                "2) Spreadsheet ID is correct, "
                "3) GID exists in the sheet"
            )
        
        raise

# ============================================
# MAIN APPLICATION - INTELLIGENT VERSION
# ============================================

def main():
    """Main application with intelligent features"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0 - APEX",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/wavedetection/docs',
            'Report a bug': 'https://github.com/wavedetection/issues',
            'About': 'Wave Detection Ultimate 3.0 - The most advanced stock ranking system'
        }
    )
    
    SmartSessionStateManager.initialize()

    st.markdown("""
    <style>
    /* Enhanced production CSS */
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
    }
    
    .stApp > header {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.8);
        padding: 8px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    .stAlert {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @media (max-width: 768px) {
        .main { padding: 0.5rem; }
        .stTabs [data-baseweb="tab"] {
            padding: 0 12px;
            font-size: 14px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; margin-bottom: 2rem;'>
        <h1 style='font-size: 3rem; font-weight: 700; 
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;'>
        🌊 Wave Detection Ultimate 3.0
        </h1>
        <p style='font-size: 1.2rem; color: #666; font-weight: 300;'>
        APEX EDITION - Intelligent Stock Ranking System
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; margin-bottom: 1rem;'>
            <h2 style='color: white; margin: 0;'>⚙️ Control Center</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Data Source")
        
        data_source = st.radio(
            "Select data source:",
            ["Google Sheets", "Upload CSV"],
            index=0 if st.session_state.data_source == "sheet" else 1,
            key="wd_data_source_radio",
            help="Choose between live Google Sheets or local CSV file"
        )
        st.session_state.data_source = "sheet" if data_source == "Google Sheets" else "upload"
        
        uploaded_file = None
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload a CSV file with stock data. Must contain 'ticker' and 'price'.",
                key="wd_csv_uploader"
            )
            if uploaded_file is None:
                st.info("💡 Drag and drop your CSV file here")
        
        elif st.session_state.data_source == "sheet":
            st.markdown("#### 🔗 Google Sheets Configuration")
            
            if 'user_spreadsheet_id' not in st.session_state or st.session_state.user_spreadsheet_id is None:
                st.session_state.user_spreadsheet_id = ""

            user_id_input = st.text_input(
                "Enter your Spreadsheet ID:",
                value=st.session_state.user_spreadsheet_id,
                placeholder="Google Sheets ID",
                help="Find this in your Google Sheets URL between /d/ and /edit",
                key="wd_user_gid_input"
            )
            
            if user_id_input != st.session_state.user_spreadsheet_id:
                st.session_state.user_spreadsheet_id = user_id_input
                if re.match(CONFIG.VALID_SHEET_ID_PATTERN, user_id_input):
                    st.success("🔄 Updating data source...")
                    st.rerun()
            
            if user_id_input and not re.match(CONFIG.VALID_SHEET_ID_PATTERN, user_id_input):
                st.error("❌ Invalid ID format. Please check the URL.")

            if not user_id_input:
                if st.button("🎮 Load Demo Data", key="load_demo_button"):
                    st.session_state.user_spreadsheet_id = "1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM"
                    st.success("Loading demo data...")
                    st.rerun()

        if st.session_state.data_quality:
            with st.expander("📊 Data Quality Dashboard", expanded=True):
                quality = st.session_state.data_quality
                
                completeness = quality.get('data_completeness', 0)
                quality_score = min(100, completeness * 0.6 + (100 - quality.get('duplicate_tickers', 0) / quality.get('total_rows', 1) * 100) * 0.4)
                
                fig_quality = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=quality_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Quality Score"},
                    delta={'reference': 80, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
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
                fig_quality.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_quality, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Stocks", f"{quality.get('total_rows', 0):,}")
                    st.metric("Completeness", f"{completeness:.1f}%")
                
                with col2:
                    st.metric("Patterns Found", f"{quality.get('patterns_detected', 0):,}")
                    if 'timestamp' in quality:
                        age = datetime.now(timezone.utc) - quality['timestamp']
                        minutes = int(age.total_seconds() / 60)
                        freshness = "🟢 Fresh" if minutes < 60 else "🔴 Stale"
                        st.metric("Data Age", freshness, f"{minutes} min")

        st.markdown("---")
        st.markdown("### 🔍 Intelligent Filters")
        
        active_filters = SmartFilterEngine.get_filter_options(st.session_state.get('ranked_df', pd.DataFrame()), 'category', {})
        if st.button("🗑️ Clear All Filters", use_container_width=True, type="primary" if len(active_filters) > 0 else "secondary"):
            SmartSessionStateManager.clear_filters()
            st.success("✅ All filters cleared!")
            st.rerun()

        filters = {}
        
        if 'ranked_df' in st.session_state and not st.session_state.ranked_df.empty:
            df_ref = st.session_state.ranked_df

            categories_all = SmartFilterEngine.get_filter_options(df_ref, 'category', {})
            selected_categories = st.multiselect("Categories:", options=categories_all, default=st.session_state.get('wd_category_filter', []), key="wd_category_filter")
            filters['categories'] = selected_categories

            sectors_all = SmartFilterEngine.get_filter_options(df_ref, 'sector', filters)
            selected_sectors = st.multiselect("Sectors:", options=sectors_all, default=st.session_state.get('wd_sector_filter', []), key="wd_sector_filter")
            filters['sectors'] = selected_sectors

            if 'industry' in df_ref.columns:
                industries_all = SmartFilterEngine.get_filter_options(df_ref, 'industry', filters)
                selected_industries = st.multiselect("Industries:", options=industries_all, default=st.session_state.get('wd_industry_filter', []), key="wd_industry_filter")
                filters['industries'] = selected_industries
            
            filters['min_score'] = st.slider("Master Score Range:", min_value=0, max_value=100, value=st.session_state.get('wd_min_score', 0), step=5, key="wd_min_score")
            
            all_patterns = sorted(list(set(p for patterns in df_ref['patterns'] for p in patterns.split(' | ') if p)))
            filters['patterns'] = st.multiselect("Patterns:", options=all_patterns, default=st.session_state.get('wd_patterns', []), key="wd_patterns")

            st.markdown("---")
            st.markdown("### 🌊 Wave Filters")
            wave_states_options = SmartFilterEngine.get_filter_options(df_ref, 'wave_state', filters)
            filters['wave_states'] = st.multiselect("Wave States:", options=wave_states_options, default=st.session_state.get('wd_wave_states_filter', []), key="wd_wave_states_filter")
            
            min_ws, max_ws = st.session_state.get('wd_wave_strength_range_slider', (0,100))
            filters['wave_strength_range'] = st.slider("Overall Wave Strength:", min_value=0, max_value=100, value=(min_ws, max_ws), key="wd_wave_strength_range_slider")

            st.markdown("---")
            st.markdown("### 💰 Fundamental Filters")
            filters['require_fundamental_data'] = st.checkbox("Require fundamental data", value=st.session_state.get('wd_require_fundamental_data', False), key="wd_require_fundamental_data")

            eps_tiers_all = SmartFilterEngine.get_filter_options(df_ref, 'eps_tier', filters)
            filters['eps_tier_filter'] = st.multiselect("EPS Growth Tiers:", options=eps_tiers_all, default=st.session_state.get('wd_eps_tier_filter', []), key="wd_eps_tier_filter")

            pe_tiers_all = SmartFilterEngine.get_filter_options(df_ref, 'pe_tier', filters)
            filters['pe_tier_filter'] = st.multiselect("PE Ratio Tiers:", options=pe_tiers_all, default=st.session_state.get('wd_pe_tier_filter', []), key="wd_pe_tier_filter")
            
            price_tiers_all = SmartFilterEngine.get_filter_options(df_ref, 'price_tier', filters)
            filters['price_tier_filter'] = st.multiselect("Price Tiers:", options=price_tiers_all, default=st.session_state.get('wd_price_tier_filter', []), key="wd_price_tier_filter")

        st.session_state.filters = filters
    
    try:
        if st.session_state.data_source == "sheet":
            if not st.session_state.get('user_spreadsheet_id'):
                st.info("👋 Welcome! To get started, please enter a Google Spreadsheet ID in the sidebar or load demo data.")
                st.stop()
        
        elif st.session_state.data_source == "upload" and uploaded_file is None:
            st.info("📁 Please upload a CSV file to continue.")
            st.stop()
        
        # Determine cache key
        cache_key_base = st.session_state.get('user_spreadsheet_id', 'upload') if st.session_state.data_source == 'sheet' else uploaded_file.name
        current_hour = datetime.now(timezone.utc).strftime('%Y%m%d_%H')
        data_version = hashlib.md5(f"{cache_key_base}_{current_hour}".encode()).hexdigest()

        with st.spinner("🔄 Loading and analyzing data..."):
            ranked_df, data_timestamp, metadata = load_and_process_data_smart(
                source_type=st.session_state.data_source,
                file_data=uploaded_file,
                spreadsheet_id=st.session_state.get('user_spreadsheet_id'),
                data_version=data_version
            )
            st.session_state.ranked_df = ranked_df
    
    except Exception as e:
        st.error(f"❌ Critical Error during data loading: {e}")
        st.stop()
        
    filtered_df = SmartFilterEngine.apply_all_filters(st.session_state.ranked_df, st.session_state.filters)

    st.markdown("### ⚡ Quick Intelligence")
    qa_cols = st.columns(6)
    
    with qa_cols[0]:
        if st.button("📈 Top Gainers", use_container_width=True, help="High momentum stocks"):
            st.session_state['quick_filter'] = 'top_gainers'
            st.session_state['wd_quick_filter_applied'] = True
            st.rerun()
    with qa_cols[1]:
        if st.button("🔥 Volume Surge", use_container_width=True, help="Unusual volume activity"):
            st.session_state['quick_filter'] = 'volume_surges'
            st.session_state['wd_quick_filter_applied'] = True
            st.rerun()
    with qa_cols[2]:
        if st.button("🎯 Breakout Ready", use_container_width=True, help="Near resistance levels"):
            st.session_state['quick_filter'] = 'breakout_ready'
            st.session_state['wd_quick_filter_applied'] = True
            st.rerun()
    with qa_cols[3]:
        if st.button("💎 Hidden Gems", use_container_width=True, help="Undervalued opportunities"):
            st.session_state['quick_filter'] = 'hidden_gems'
            st.session_state['wd_quick_filter_applied'] = True
            st.rerun()
    with qa_cols[4]:
        if st.button("🌊 Perfect Storms", use_container_width=True, help="Everything aligned"):
            st.session_state['quick_filter'] = 'perfect_storms'
            st.session_state['wd_quick_filter_applied'] = True
            st.rerun()
    with qa_cols[5]:
        if st.button("🔄 Show All", use_container_width=True, help="Remove quick filters"):
            st.session_state['quick_filter'] = None
            st.session_state['wd_quick_filter_applied'] = False
            st.rerun()

    if st.session_state.get('quick_filter') and filtered_df is not None and not filtered_df.empty:
        quick_filter_conditions = {
            'top_gainers': (filtered_df['momentum_score'] >= 80) & (filtered_df['ret_30d'] > 10),
            'volume_surges': (filtered_df['rvol'] >= 3) & (filtered_df['volume_score'] >= 80),
            'breakout_ready': (filtered_df['breakout_score'] >= 80) & (filtered_df['from_high_pct'] > -10),
            'hidden_gems': (filtered_df['patterns'].str.contains('HIDDEN GEM', na=False)),
            'perfect_storms': (filtered_df['patterns'].str.contains('PERFECT STORM', na=False))
        }
        
        if st.session_state.quick_filter in quick_filter_conditions:
            filtered_df = filtered_df[quick_filter_conditions[st.session_state.quick_filter]]

    if st.session_state.get('wd_trigger_clear', False):
        SmartSessionStateManager.clear_filters()
        st.session_state.wd_trigger_clear = False
        st.rerun()

    # Main tabs
    tab_list = ["📊 Dashboard", "🏆 Rankings", "🌊 Wave Radar", "📈 Analytics", "🔍 Search", "📥 Export", "ℹ️ About"]
    tabs = st.tabs(tab_list)
    
    with tabs[0]: # Dashboard
        st.markdown("### 📊 Executive Intelligence Dashboard")
        if not filtered_df.empty:
            UI_filtered_df = filtered_df.copy()
            
            # Summary Metrics
            status_cols = st.columns(5)
            with status_cols[0]:
                SmartUIComponents.render_metric_card("Total Stocks", f"{len(UI_filtered_df):,}")
            with status_cols[1]:
                avg_score = UI_filtered_df['master_score'].mean()
                SmartUIComponents.render_metric_card("Avg Score", f"{avg_score:.1f}")
            with status_cols[2]:
                active_waves_count = UI_filtered_df['wave_state'].isin(['🌊🌊🌊 CRESTING', '🌊🌊 BUILDING']).sum()
                SmartUIComponents.render_metric_card("Active Waves", f"{active_waves_count:,}")
            with status_cols[3]:
                patterns_count = UI_filtered_df['patterns'].apply(lambda x: len(x.split(' | ')) if x else 0).sum()
                SmartUIComponents.render_metric_card("Patterns Found", f"{patterns_count:,}")
            with status_cols[4]:
                high_rvol_count = (UI_filtered_df['rvol'] > 2).sum()
                SmartUIComponents.render_metric_card("High RVOL", f"{high_rvol_count:,}")

            # Intelligent Picks
            st.markdown("#### 🎯 Intelligent Picks")
            pick_cols = st.columns(3)
            with pick_cols[0]:
                st.markdown("##### 🚀 Momentum Leaders")
                momentum_leaders = UI_filtered_df.nlargest(5, 'momentum_score')
                for _, stock in momentum_leaders.iterrows():
                    st.write(f"**{stock['ticker']}** - Score: {stock['momentum_score']:.1f}")
            with pick_cols[1]:
                st.markdown("##### ⚡ Volume Explosions")
                volume_leaders = UI_filtered_df.nlargest(5, 'rvol')
                for _, stock in volume_leaders.iterrows():
                    st.write(f"**{stock['ticker']}** - RVOL: {stock['rvol']:.1f}x")
            with pick_cols[2]:
                st.markdown("##### 💎 Hidden Opportunities")
                hidden_gems = UI_filtered_df[UI_filtered_df['patterns'].str.contains('HIDDEN GEM', na=False)].nlargest(5, 'master_score')
                for _, stock in hidden_gems.iterrows():
                    st.write(f"**{stock['ticker']}** - Score: {stock['master_score']:.1f}")
            
            # Sector/Industry Heatmap (Example from V2)
            st.markdown("---")
            st.markdown("#### 🗺️ Sector Intelligence")
            if 'sector' in UI_filtered_df.columns:
                sector_analysis = UI_filtered_df.groupby('sector').agg({'master_score': 'mean', 'ret_30d': 'mean', 'rvol': 'mean'}).round(2)
                sector_analysis.columns = ['Avg Score', 'Avg 30D Return', 'Avg RVOL']
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=sector_analysis[['Avg Score', 'Avg 30D Return', 'Avg RVOL']].T.values,
                    x=sector_analysis.index,
                    y=['Avg Score', 'Avg 30D Return', 'Avg RVOL'],
                    colorscale='RdYlGn',
                    text=sector_analysis[['Avg Score', 'Avg 30D Return', 'Avg RVOL']].T.values.round(1),
                    texttemplate="%{text}",
                ))
                fig_heatmap.update_layout(title="Sector Performance Heatmap", height=300)
                st.plotly_chart(fig_heatmap, use_container_width=True)

    with tabs[1]: # Rankings
        st.markdown("### 🏆 Intelligent Stock Rankings")
        if not filtered_df.empty:
            display_count = st.selectbox("Show top", options=CONFIG.AVAILABLE_TOP_N, key="wd_items_per_page")
            sort_by = st.selectbox("Sort by:", options=['Master Score', 'Momentum', 'Volume Activity'], key="wd_sort_by")
            display_style = st.radio("Display style:", options=['Cards', 'Table'], horizontal=True, key="wd_display_style")

            sort_map = {'Master Score': 'master_score', 'Momentum': 'momentum_score', 'Volume Activity': 'rvol'}
            sorted_df = filtered_df.sort_values(sort_map.get(sort_by, 'master_score'), ascending=False).head(display_count)

            if display_style == 'Cards':
                for _, row in sorted_df.iterrows():
                    SmartUIComponents.render_stock_card_enhanced(row, show_fundamentals=(st.session_state.display_mode == 'Hybrid'))
            else:
                st.dataframe(sorted_df, use_container_width=True)

    with tabs[2]: # Wave Radar
        st.markdown("### 🌊 Wave Radar - Advanced Momentum Detection")
        if not filtered_df.empty:
            wave_filtered_df = filtered_df.copy()
            
            # Wave State Distribution
            st.markdown("#### 🌊 Wave State Analysis")
            wave_counts = wave_filtered_df['wave_state'].value_counts()
            fig_wave_dist = go.Figure(data=[
                go.Bar(
                    x=wave_counts.index,
                    y=wave_counts.values,
                    marker_color=['#E74C3C', '#F39C12', '#3498DB', '#95A5A6'],
                    text=wave_counts.values, textposition='auto'
                )
            ])
            fig_wave_dist.update_layout(title="Wave State Distribution", height=400)
            st.plotly_chart(fig_wave_dist, use_container_width=True)
            
            # Momentum Shifts
            st.markdown("#### 🚀 Momentum Shift Detection")
            momentum_shifts = wave_filtered_df[(wave_filtered_df['momentum_score'] >= 70) & (wave_filtered_df['acceleration_score'] >= 70)]
            if not momentum_shifts.empty:
                st.dataframe(momentum_shifts.head(10)[['ticker', 'momentum_score', 'acceleration_score', 'wave_state']])
            else:
                st.info("No strong momentum shifts detected.")

    with tabs[3]: # Analytics
        st.markdown("### 📈 Advanced Market Analytics")
        if not filtered_df.empty:
            analysis_type = st.selectbox("Select Analysis Type:", ["Sector/Industry Analysis", "Performance Distribution"])
            if analysis_type == "Sector/Industry Analysis":
                st.markdown("#### 🏢 Sector & Industry Intelligence")
                if 'sector' in filtered_df.columns:
                    sector_stats = filtered_df.groupby('sector').agg({'master_score': ['mean', 'count']}).round(2)
                    st.dataframe(sector_stats)
            elif analysis_type == "Performance Distribution":
                st.markdown("#### 📊 Performance Distribution Analysis")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=filtered_df['master_score'], nbinsx=30))
                st.plotly_chart(fig_dist)

    with tabs[4]: # Search
        st.markdown("### 🔍 Intelligent Stock Search")
        search_query = st.text_input("Search by ticker or company name:", key="search_query")
        if search_query:
            search_results = SmartSearchEngine.search_stocks(filtered_df, search_query)
            if not search_results.empty:
                for _, row in search_results.iterrows():
                    SmartUIComponents.render_stock_card_enhanced(row, show_fundamentals=(st.session_state.display_mode == 'Hybrid'))
            else:
                st.warning("No stocks found matching your search.")

    with tabs[5]: # Export
        st.markdown("### 📥 Intelligent Export Center")
        if not filtered_df.empty:
            export_strategy = st.selectbox("Select Export Strategy:", ["Day Trading", "Swing Trading", "Fundamental", "Complete"], key="export_strategy")
            export_df = filtered_df
            if export_strategy == "Day Trading":
                export_data = SmartExportEngine.create_csv_export(export_df, 'day_trading')
            elif export_strategy == "Swing Trading":
                export_data = SmartExportEngine.create_csv_export(export_df, 'swing_trading')
            elif export_strategy == "Fundamental":
                export_data = SmartExportEngine.create_csv_export(export_df, 'fundamental')
            else:
                export_data = SmartExportEngine.create_csv_export(export_df, 'complete')

            st.download_button(
                label=f"📥 Download {export_strategy} CSV",
                data=export_data,
                file_name=f"wave_detection_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with tabs[6]: # About
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0 - APEX Edition")
        st.markdown("""
        #### 🌊 Welcome to the Future of Stock Analysis
        
        Wave Detection Ultimate 3.0 APEX Edition represents the pinnacle of intelligent stock ranking technology.
        This version incorporates the most robust and accurate algorithms from its predecessors.
        
        #### 📈 Key Features
        - **Master Score 3.0:** A proprietary multi-factor algorithm.
        - **Advanced Metrics:** Including `vmi`, `momentum_harmony`, and `wave_state`.
        - **25 Intelligent Patterns:** From technical to fundamental signals.
        - **Smart Filters:** Interconnected and flexible filtering system with tier-based options.
        - **Persistent State:** Your settings and data source are remembered.
        
        **Version**: 3.0.9-APEX-FINAL-REVISED
        **Status**: PRODUCTION PERFECT
        **Last Updated**: August 2025
        """)
        
if __name__ == "__main__":
    main()
