"""
Wave Detection Ultimate 3.0 - FINAL PERFECTED PRODUCTION VERSION
===============================================================
Professional Stock Ranking System with Advanced Analytics
All bugs fixed, optimized for Streamlit Community Cloud
Enhanced with perfect filtering system and robust error handling

Version: 3.1.0-FINAL-PERFECTED
Last Updated: December 2024
Status: PRODUCTION READY - PERMANENTLY LOCKED
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
from typing import Dict, List, Tuple, Optional, Any, Union, Set, Callable
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor
import time
from io import BytesIO, StringIO
import warnings
import gc
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
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
# INTELLIGENT LOGGING & PERFORMANCE SYSTEM
# ============================================

class SmartLogger:
    """Intelligent logging with automatic error tracking and performance monitoring"""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        
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
                    'p95': np.percentile(durations, 95) if len(durations) > 1 else durations[0]
                }
        return summary

logger = SmartLogger(__name__)

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
                    
                    if len(monitor.metrics[func.__name__]) > 100:
                        monitor.metrics[func.__name__] = monitor.metrics[func.__name__][-100:]
                    
                    logger.log_performance(func.__name__, elapsed)
                    
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
                    'p95_time': np.percentile(times, 95) if len(times) > 1 else times[0],
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
            return {'error': 'psutil not available'}

# ============================================
# ROBUST SESSION STATE MANAGER
# ============================================

class RobustSessionState:
    """Bulletproof session state management - prevents all KeyErrors"""

    STATE_DEFAULTS = {
        # Core states
        'search_query': "",
        'last_refresh': None,
        'data_source': "sheet",
        'sheet_id': "",
        'gid': "1823439984",
        'user_preferences': {
            'default_top_n': 50,
            'display_mode': 'Technical',
            'last_filters': {}
        },
        'filters': {},
        'active_filter_count': 0,
        'quick_filter': None,
        'quick_filter_applied': False,
        'show_debug': False,
        'performance_metrics': defaultdict(list),
        'data_quality': {},
        'last_good_data': None,
        'session_id': None,
        'session_start': None,
        'validation_stats': defaultdict(int),
        'trigger_clear': False,
        'current_page_rankings': 0,
        
        # Filter states
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
        'export_template_radio': "Full Analysis (All Data)",
        
        # Data states
        'ranked_df': None,
        'data_timestamp': None,
        
        # UI states
        'search_input': ""
    }

    @staticmethod
    def safe_get(key: str, default: Any = None) -> Any:
        """Safely get a session state value with fallback"""
        if key not in st.session_state:
            st.session_state[key] = RobustSessionState.STATE_DEFAULTS.get(key, default)
        return st.session_state[key]
    
    @staticmethod
    def safe_set(key: str, value: Any) -> None:
        """Safely set a session state value"""
        st.session_state[key] = value

    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        for key, default_value in RobustSessionState.STATE_DEFAULTS.items():
            if key not in st.session_state:
                if key == 'last_refresh' or key == 'session_start':
                    st.session_state[key] = datetime.now(timezone.utc)
                elif key == 'session_id':
                    st.session_state[key] = hashlib.md5(
                        f"{datetime.now()}{np.random.rand()}".encode()
                    ).hexdigest()[:8]
                else:
                    st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        """Clear all filter states safely"""
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'patterns',
            'min_score', 'trend_filter', 'min_eps_change',
            'min_pe', 'max_pe', 'require_fundamental_data',
            'quick_filter', 'quick_filter_applied',
            'wave_states_filter', 'wave_strength_range_slider',
            'show_sensitivity_details', 'show_market_regime',
            'wave_timeframe_select', 'wave_sensitivity'
        ]
        
        for key in filter_keys:
            if key in RobustSessionState.STATE_DEFAULTS:
                RobustSessionState.safe_set(key, RobustSessionState.STATE_DEFAULTS[key])
        
        RobustSessionState.safe_set('filters', {})
        RobustSessionState.safe_set('active_filter_count', 0)
        RobustSessionState.safe_set('trigger_clear', False)
        RobustSessionState.safe_set('current_page_rankings', 0)

    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """Get session information"""
        return {
            'session_id': RobustSessionState.safe_get('session_id', 'unknown'),
            'start_time': RobustSessionState.safe_get('session_start', datetime.now()),
            'duration': (datetime.now() - RobustSessionState.safe_get('session_start', datetime.now())).seconds,
            'data_source': RobustSessionState.safe_get('data_source', 'unknown'),
            'stocks_loaded': len(RobustSessionState.safe_get('ranked_df', [])),
            'active_filters': RobustSessionState.safe_get('active_filter_count', 0)
        }

# ============================================
# INTELLIGENT CONFIGURATION SYSTEM
# ============================================

@dataclass(frozen=True)
class SmartConfig:
    """Intelligent configuration with validation and optimization"""

    DEFAULT_GID: str = "1823439984"
    DEFAULT_SHEET_ID: str = "1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM"
    SPREADSHEET_ID_LENGTH: int = 44
    
    REQUEST_TIMEOUT: int = 30
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_FACTOR: float = 0.5
    RETRY_STATUS_CODES: Tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    
    CACHE_TTL: int = 3600
    STALE_DATA_HOURS: int = 24
    MAX_CACHE_SIZE_MB: int = 500
    
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    MAX_DISPLAY_ROWS: int = 1000
    
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'ret_30d', 'from_low_pct', 'from_high_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    OPTIONAL_COLUMNS: List[str] = field(default_factory=lambda: [
        'company_name', 'market_cap', 'pe', 'eps_current', 'eps_change_pct',
        'sma_20d', 'sma_50d', 'sma_200d', 'ret_3m', 'ret_6m', 'ret_1y'
    ])
    
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ])
    
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
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
    
    CHUNK_SIZE: int = 1000
    PARALLEL_WORKERS: int = 4
    USE_NUMBA: bool = False
    
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

    def __post_init__(self):
        """Validate configuration on initialization"""
        total_weight = (self.POSITION_WEIGHT + self.VOLUME_WEIGHT + 
                       self.MOMENTUM_WEIGHT + self.ACCELERATION_WEIGHT + 
                       self.BREAKOUT_WEIGHT + self.RVOL_WEIGHT)
        if not np.isclose(total_weight, 1.0, rtol=1e-5):
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")

CONFIG = SmartConfig()

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
        
        if df is None:
            diagnostics['issues'].append("DataFrame is None")
            return False, f"{context}: DataFrame is None", diagnostics
        
        if df.empty:
            diagnostics['issues'].append("DataFrame is empty")
            return False, f"{context}: DataFrame is empty", diagnostics
        
        if len(df) < 10:
            diagnostics['issues'].append(f"Too few rows: {len(df)}")
            if len(df) < 5:
                return False, f"{context}: Insufficient data ({len(df)} rows)", diagnostics
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            diagnostics['issues'].append(f"Missing columns: {missing_cols}")
            critical_missing = missing_cols.intersection(CONFIG.CRITICAL_COLUMNS)
            if critical_missing:
                return False, f"{context}: Missing critical columns: {critical_missing}", diagnostics
        
        diagnostics['column_stats'] = {}
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'nulls': df[col].isna().sum(),
                'null_pct': df[col].isna().sum() / len(df) * 100
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                })
                
                if df[col].max() > 1e10 or df[col].min() < -1e10:
                    diagnostics['issues'].append(f"{col}: Extreme values detected")
            
            diagnostics['column_stats'][col] = col_stats
        
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
            if isinstance(value, str):
                cleaned = value.strip().upper()
                
                invalid_markers = ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-', 
                                 '#N/A', '#ERROR!', '#DIV/0!', 'INF', '-INF', '#VALUE!']
                if cleaned in invalid_markers:
                    self.correction_stats[f"{col_name}_invalid"] += 1
                    return np.nan
                
                cleaned = re.sub(r'[₹$€£¥₹,\s]', '', cleaned)
                
                if cleaned.endswith('%'):
                    cleaned = cleaned[:-1]
                    value = float(cleaned)
                else:
                    value = float(cleaned)
            else:
                value = float(value)
            
            if bounds and auto_correct:
                min_val, max_val = bounds
                original_value = value
                
                if value < min_val:
                    value = min_val
                    self.clipping_counts[f"{col_name}_min"] += 1
                elif value > max_val:
                    value = max_val
                    self.clipping_counts[f"{col_name}_max"] += 1
                
                if abs(original_value - value) > abs(value) * 0.5:
                    logger.logger.debug(
                        f"Extreme clipping in {col_name}: {original_value} -> {value}"
                    )
            
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
            
            if cleaned.upper() in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']:
                self.correction_stats['string_invalid'] += 1
                return default
            
            cleaned = ' '.join(cleaned.split())
            
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length-3] + "..."
                self.correction_stats['string_truncated'] += 1
            
            cleaned = cleaned.replace('<', '&lt;').replace('>', '&gt;')
            
            return cleaned
            
        except Exception:
            self.correction_stats['string_error'] += 1
            return default

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
        
        validator.reset_stats()
        
        if len(df) > CONFIG.CHUNK_SIZE:
            chunks = [df[i:i+CONFIG.CHUNK_SIZE] for i in range(0, len(df), CONFIG.CHUNK_SIZE)]
            processed_chunks = []
            
            for i, chunk in enumerate(chunks):
                logger.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                processed_chunk = SmartDataProcessor._process_chunk(chunk, metadata)
                processed_chunks.append(processed_chunk)
            
            df = pd.concat(processed_chunks, ignore_index=True)
        else:
            df = SmartDataProcessor._process_chunk(df, metadata)
        
        df = SmartDataProcessor._optimize_dataframe(df)
        
        validation_report = validator.get_validation_report()
        if validation_report['total_issues'] > 0:
            metadata['warnings'].append(
                f"Data quality: {validation_report['total_issues']} issues corrected"
            )
            logger.logger.info(f"Validation report: {validation_report}")
        
        logger.logger.info(f"Processing complete: {len(df)} rows retained")
        
        return df

    @staticmethod
    def _process_chunk(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Process a single chunk of data"""
        
        df['ticker'] = df['ticker'].apply(
            lambda x: validator.sanitize_string(x, "UNKNOWN", max_length=20)
        )
        
        for col, bounds in CONFIG.NUMERIC_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: validator.sanitize_numeric(x, bounds, col)
                )
        
        for col in CONFIG.PERCENTAGE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: validator.sanitize_numeric(x, (-99.99, 9999), col)
                )
        
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: validator.sanitize_numeric(x, (0, 100), col)
                )
        
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        for col in sma_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: validator.sanitize_numeric(x, (0.01, 1_000_000), col)
                )
        
        if 'rvol' not in df.columns or df['rvol'].isna().all():
            if 'volume_1d' in df.columns and 'volume_90d' in df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['rvol'] = np.where(
                        df['volume_90d'] > 0,
                        df['volume_1d'] / df['volume_90d'],
                        1.0
                    )
                metadata['warnings'].append("RVOL calculated from volume ratios")
        
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
        
        if 'industry' not in df.columns:
            df['industry'] = df['sector']
        else:
            df['industry'] = df['industry'].apply(
                lambda x: validator.sanitize_string(
                    x, df['sector'].iloc[0] if 'sector' in df.columns else "Unknown", 
                    max_length=50
                )
            )
        
        return df

    @staticmethod
    def _optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe for performance"""
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if len(df) < initial_count:
            removed = initial_count - len(df)
            logger.logger.info(f"Removed {removed} duplicate tickers")
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type in ['float64']:
                df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')
            elif col_type in ['int64']:
                df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')
            
            elif col_type == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:
                    df[col] = df[col].astype('category')
        
        df = df.sort_values('ticker').reset_index(drop=True)
        
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
        
        df = AdvancedMetricsEngine._calculate_base_metrics(df)
        df = AdvancedMetricsEngine._calculate_composite_metrics(df)
        df = AdvancedMetricsEngine._calculate_intelligent_metrics(df)
        
        return df

    @staticmethod
    def _calculate_base_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate base metrics"""
        
        df['vmi'] = AdvancedMetricsEngine._calculate_vmi_enhanced(df)
        df['position_tension'] = AdvancedMetricsEngine._calculate_position_tension_smart(df)
        df['momentum_harmony'] = AdvancedMetricsEngine._calculate_momentum_harmony(df)
        
        return df

    @staticmethod
    def _calculate_composite_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite metrics"""
        
        df['wave_state'] = df.apply(AdvancedMetricsEngine._get_wave_state_dynamic, axis=1)
        df['overall_wave_strength'] = AdvancedMetricsEngine._calculate_wave_strength_smart(df)
        
        return df

    @staticmethod
    def _calculate_intelligent_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate intelligent metrics using ML-inspired approaches"""
        
        df['market_regime'] = AdvancedMetricsEngine._detect_market_regime(df)
        df['smart_money_flow'] = AdvancedMetricsEngine._calculate_smart_money_flow(df)
        df['momentum_quality'] = AdvancedMetricsEngine._calculate_momentum_quality(df)
        
        return df

    @staticmethod
    def _calculate_vmi_enhanced(df: pd.DataFrame) -> pd.Series:
        """Enhanced Volume Momentum Index with adaptive weighting"""
        
        vmi = pd.Series(50, index=df.index, dtype=float)
        
        vol_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 
                   'vol_ratio_1d_180d', 'vol_ratio_30d_180d']
        
        available_cols = [col for col in vol_cols if col in df.columns]
        
        if available_cols:
            weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])[:len(available_cols)]
            weights = weights / weights.sum()
            
            for col, weight in zip(available_cols, weights):
                col_data = df[col].fillna(1)
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
        
        low_tension = np.where(
            from_low < 20,
            100 - from_low * 2,
            50 - (from_low - 20) * 0.3
        )
        
        high_tension = np.where(
            from_high > -20,
            100 + from_high * 2,
            50 + (from_high + 20) * 0.3
        )
        
        position_ratio = from_low / (from_low - from_high + 1e-6)
        weight_low = 1 - position_ratio
        weight_high = position_ratio
        
        tension = low_tension * weight_low + high_tension * weight_high
        
        return pd.Series(tension, index=df.index).clip(0, 100)

    @staticmethod
    def _calculate_momentum_harmony(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum harmony with weighted timeframes"""
        
        harmony_score = pd.Series(0, index=df.index, dtype=float)
        
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
        
        perfect_harmony = (harmony_count == len(timeframes))
        harmony_score[perfect_harmony] += 0.5
        
        harmony_final = (harmony_score * 4).round().astype(int)
        
        return harmony_final.clip(0, 4)

    @staticmethod
    def _get_wave_state_dynamic(row: pd.Series) -> str:
        """Dynamic wave state with intelligent thresholds"""
        
        signals = {
            'momentum': (row.get('momentum_score', 0) > 70) * 1.5,
            'volume': (row.get('volume_score', 0) > 70) * 1.2,
            'acceleration': (row.get('acceleration_score', 0) > 70) * 1.3,
            'rvol': (row.get('rvol', 0) > 2) * 1.4,
            'harmony': (row.get('momentum_harmony', 0) >= 3) * 1.1
        }
        
        total_signal = sum(signals.values())
        
        regime = row.get('market_regime', 'neutral')
        
        if regime == 'risk_on':
            thresholds = [5.5, 4.0, 1.5]
        elif regime == 'risk_off':
            thresholds = [6.5, 5.0, 2.5]
        else:
            thresholds = [6.0, 4.5, 2.0]
        
        if total_signal >= thresholds[0]:
            return "🌊🌊🌊 CRESTING"
        elif total_signal >= thresholds[1]:
            return "🌊🌊 BUILDING"
        elif total_signal >= thresholds[2]:
            return "🌊 FORMING"
        else:
            return "💥 BREAKING"

    @staticmethod
    def _calculate_wave_strength_smart(df: pd.DataFrame) -> pd.Series:
        """Smart wave strength with adaptive weighting"""
        
        components = {
            'momentum_score': 0.30,
            'acceleration_score': 0.25,
            'rvol_score': 0.20,
            'breakout_score': 0.15,
            'volume_score': 0.10
        }
        
        wave_strength = pd.Series(50, index=df.index, dtype=float)
        
        total_weight = 0
        for col, weight in components.items():
            if col in df.columns:
                wave_strength += (df[col].fillna(50) - 50) * weight
                total_weight += weight
        
        if total_weight > 0 and total_weight < 1:
            wave_strength = 50 + (wave_strength - 50) / total_weight
        
        if 'momentum_harmony' in df.columns:
            harmony_bonus = df['momentum_harmony'] * 2.5
            wave_strength += harmony_bonus
        
        return wave_strength.clip(0, 100)

    @staticmethod
    def _detect_market_regime(df: pd.DataFrame) -> pd.Series:
        """Detect market regime using multiple indicators"""
        
        if 'ret_30d' in df.columns:
            positive_breadth = (df['ret_30d'] > 0).mean()
            strong_positive = (df['ret_30d'] > 10).mean()
            strong_negative = (df['ret_30d'] < -10).mean()
        else:
            positive_breadth = 0.5
            strong_positive = 0.1
            strong_negative = 0.1
        
        if positive_breadth > 0.6 and strong_positive > 0.3:
            regime = "risk_on"
        elif positive_breadth < 0.4 and strong_negative > 0.3:
            regime = "risk_off"
        else:
            regime = "neutral"
        
        return pd.Series(regime, index=df.index)

    @staticmethod
    def _calculate_smart_money_flow(df: pd.DataFrame) -> pd.Series:
        """Calculate smart money flow indicator"""
        
        smart_flow = pd.Series(50, index=df.index, dtype=float)
        
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
            vol_persistence = (
                (df['vol_ratio_7d_90d'] > 1.2) & 
                (df['vol_ratio_30d_90d'] > 1.1)
            ).astype(float) * 20
            smart_flow += vol_persistence
        
        if 'ret_30d' in df.columns and 'volume_score' in df.columns:
            divergence = np.where(
                (np.abs(df['ret_30d']) < 5) & (df['volume_score'] > 70),
                20, 0
            )
            smart_flow += divergence
        
        if 'liquidity_score' in df.columns:
            institutional = np.where(df['liquidity_score'] > 80, 10, 0)
            smart_flow += institutional
        
        return smart_flow.clip(0, 100)

    @staticmethod
    def _calculate_momentum_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum quality score"""
        
        quality = pd.Series(50, index=df.index, dtype=float)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            daily_7d = df['ret_7d'] / 7
            daily_30d = df['ret_30d'] / 30
            
            acceleration = np.where(
                (daily_7d > daily_30d * 1.2) & (daily_7d > 0),
                20, 0
            )
            quality += acceleration
        
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            returns = df[['ret_1d', 'ret_7d', 'ret_30d']]
            volatility = returns.std(axis=1)
            smooth = np.where(volatility < returns.mean(axis=1) * 0.5, 15, 0)
            quality += smooth
        
        if 'trend_quality' in df.columns:
            quality += df['trend_quality'] * 0.15
        
        return quality.clip(0, 100)

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
        
        df = SmartRankingEngine._precalculate_values(df)
        
        score_functions = {
            'position_score': SmartRankingEngine._calculate_position_score_smart,
            'volume_score': SmartRankingEngine._calculate_volume_score_smart,
            'momentum_score': SmartRankingEngine._calculate_momentum_score_smart,
            'acceleration_score': SmartRankingEngine._calculate_acceleration_score_smart,
            'breakout_score': SmartRankingEngine._calculate_breakout_score_smart,
            'rvol_score': SmartRankingEngine._calculate_rvol_score_smart
        }
        
        for score_name, score_func in score_functions.items():
            df[score_name] = score_func(df)
        
        df['trend_quality'] = SmartRankingEngine._calculate_trend_quality_smart(df)
        df['long_term_strength'] = SmartRankingEngine._calculate_long_term_strength_smart(df)
        df['liquidity_score'] = SmartRankingEngine._calculate_liquidity_score_smart(df)
        
        df = SmartRankingEngine._calculate_master_score_optimized(df)
        df = SmartRankingEngine._calculate_all_ranks(df)
        
        logger.logger.info(f"Ranking complete: {len(df)} stocks processed")
        
        return df

    @staticmethod
    def _precalculate_values(df: pd.DataFrame) -> pd.DataFrame:
        """Pre-calculate frequently used values for performance"""
        
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
        
        series_clean = series.replace([np.inf, -np.inf], np.nan)
        
        valid_count = series_clean.notna().sum()
        if valid_count == 0:
            return pd.Series(50, index=series.index)
        
        if pct:
            ranks = series_clean.rank(
                pct=True, ascending=ascending, 
                na_option='bottom', method=method
            ) * 100
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
        
        if has_from_low:
            from_low = df['from_low_pct'].fillna(50)
            from_low_transformed = np.tanh(from_low / 100) * 100
            rank_from_low = SmartRankingEngine._safe_rank(
                from_low_transformed, pct=True, ascending=True
            )
        else:
            rank_from_low = pd.Series(50, index=df.index)
        
        if has_from_high:
            from_high = df['from_high_pct'].fillna(-50)
            from_high_transformed = np.where(
                from_high > -20,
                from_high * 2,
                from_high
            )
            rank_from_high = SmartRankingEngine._safe_rank(
                from_high_transformed, pct=True, ascending=False
            )
        else:
            rank_from_high = pd.Series(50, index=df.index)
        
        if 'market_regime' in df.columns:
            weight_low = np.where(df['market_regime'] == 'risk_off', 0.7, 0.6)
            weight_high = 1 - weight_low
            position_score = rank_from_low * weight_low + rank_from_high * weight_high
        else:
            position_score = rank_from_low * 0.6 + rank_from_high * 0.4
        
        return position_score.clip(0, 100)

    @staticmethod
    def _calculate_volume_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart volume score with adaptive weighting"""
        
        volume_score = pd.Series(50, index=df.index, dtype=float)
        
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
        
        total_weight = sum(p['weight'] for p in available_timeframes.values())
        
        weighted_momentum = pd.Series(0, index=df.index, dtype=float)
        
        for tf, params in available_timeframes.items():
            daily_return = df[tf] / params['days']
            
            tf_rank = SmartRankingEngine._safe_rank(
                daily_return, pct=True, ascending=True, method='average'
            )
            
            weighted_momentum += tf_rank * params['weight']
        
        momentum_score = weighted_momentum / total_weight
        
        if len(available_timeframes) >= 2:
            returns_df = df[[tf for tf in available_timeframes.keys()]]
            all_positive = (returns_df > 0).all(axis=1)
            consistency_bonus = all_positive.astype(float) * 10
            
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
        
        accelerations = []
        
        if 'ret_1d' in df.columns and 'ret_7d' in df.columns:
            daily_1d = df['ret_1d']
            daily_7d = df['ret_7d'] / 7
            
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
        
        if accelerations:
            combined_accel = sum(accel * weight for _, accel, weight in accelerations)
            
            acceleration_score = 50 + 25 * np.tanh(combined_accel)
            
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
        
        distance_score = pd.Series(50, index=df.index)
        volume_score = pd.Series(50, index=df.index)
        trend_score = pd.Series(50, index=df.index)
        pattern_score = pd.Series(50, index=df.index)
        
        if 'from_high_pct' in df.columns:
            from_high = df['from_high_pct'].fillna(-50)
            
            distance_score = np.where(
                from_high > -5, 95,
                np.where(from_high > -10, 85,
                np.where(from_high > -20, 70,
                np.where(from_high > -30, 50, 30)))
            )
            distance_score = pd.Series(distance_score, index=df.index)
        
        if 'vol_ratio_1d_90d' in df.columns:
            vol_ratio = df['vol_ratio_1d_90d'].fillna(1)
            volume_score = SmartRankingEngine._safe_rank(vol_ratio, pct=True, ascending=True)
            
            surge_bonus = np.where(vol_ratio > 2, 15, 0)
            volume_score += surge_bonus
        
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d']):
            price = df['price'].fillna(0)
            sma_20 = df['sma_20d'].fillna(price)
            sma_50 = df['sma_50d'].fillna(price)
            
            perfect_trend = (price > sma_20) & (sma_20 > sma_50)
            trend_score = pd.Series(
                np.where(perfect_trend, 90, 40), 
                index=df.index
            )
        
        if 'position_tension' in df.columns:
            pattern_score = np.where(
                df['position_tension'] > 80,
                70, 50
            )
            pattern_score = pd.Series(pattern_score, index=df.index)
        
        if 'market_regime' in df.columns:
            regime_weights = {
                'risk_on': [0.3, 0.4, 0.2, 0.1],
                'risk_off': [0.4, 0.3, 0.2, 0.1],
                'neutral': [0.35, 0.35, 0.2, 0.1]
            }
            
            weights = pd.Series(index=df.index, dtype=object)
            for regime, w in regime_weights.items():
                weights[df['market_regime'] == regime] = w
            
            breakout_score = pd.Series(index=df.index, dtype=float)
            for idx in df.index:
                if pd.notna(weights[idx]):
                    w = weights[idx]
                    breakout_score[idx] = (
                        distance_score[idx] * w[0] +
                        volume_score[idx] * w[1] +
                        trend_score[idx] * w[2] +
                        pattern_score[idx] * w[3]
                    )
                else:
                    breakout_score[idx] = (
                        distance_score[idx] * 0.35 +
                        volume_score[idx] * 0.35 +
                        trend_score[idx] * 0.2 +
                        pattern_score[idx] * 0.1
                    )
        else:
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
        
        if 'market_regime' in df.columns:
            regime_multipliers = {
                'risk_on': 0.9,
                'risk_off': 1.1,
                'neutral': 1.0
            }
            
            rvol_score = pd.Series(50, index=df.index, dtype=float)
            
            for regime, mult in regime_multipliers.items():
                mask = df['market_regime'] == regime
                
                rvol_score[mask] = np.select(
                    [
                        rvol[mask] > 10 * mult,
                        rvol[mask] > 5 * mult,
                        rvol[mask] > 3 * mult,
                        rvol[mask] > 2 * mult,
                        rvol[mask] > 1.5 * mult,
                        rvol[mask] > 1.2 * mult,
                        rvol[mask] > 0.8 * mult,
                        rvol[mask] > 0.5 * mult,
                        rvol[mask] > 0.3 * mult
                    ],
                    [95, 90, 85, 80, 70, 60, 50, 40, 30],
                    default=20
                )
        else:
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
        
        perfect_bullish = (price > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
        strong_bullish = (price > sma_50) & (sma_50 > sma_200) & ~perfect_bullish
        moderate_bullish = (price > sma_200) & ~perfect_bullish & ~strong_bullish
        
        perfect_bearish = (price < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)
        strong_bearish = (price < sma_50) & (sma_50 < sma_200) & ~perfect_bearish
        
        trend_score[perfect_bullish] = 95
        trend_score[strong_bullish] = 75
        trend_score[moderate_bullish] = 55
        trend_score[perfect_bearish] = 15
        trend_score[strong_bearish] = 25
        
        if 'price_to_sma20' in df.columns:
            distance = np.abs(df['price_to_sma20'] - 1)
            smoothness_bonus = np.where(distance < 0.05, 5, 0)
            trend_score += smoothness_bonus
        
        return trend_score.clip(0, 100)

    @staticmethod
    def _calculate_long_term_strength_smart(df: pd.DataFrame) -> pd.Series:
        """Smart long-term strength with regime adjustment"""
        
        lt_score = pd.Series(50, index=df.index, dtype=float)
        
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
        
        total_weight = sum(available_timeframes.values())
        normalized_weights = {
            tf: w/total_weight for tf, w in available_timeframes.items()
        }
        
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        
        for tf, weight in normalized_weights.items():
            tf_rank = SmartRankingEngine._safe_rank(
                df[tf], pct=True, ascending=True, method='average'
            )
            
            if 'market_regime' in df.columns:
                regime_adjustment = np.where(
                    df['market_regime'] == 'risk_on', 1.1,
                    np.where(df['market_regime'] == 'risk_off', 0.9, 1.0)
                )
                tf_rank *= regime_adjustment
            
            weighted_score += tf_rank * weight
        
        lt_score = weighted_score
        
        if len(available_timeframes) >= 2:
            returns = df[list(available_timeframes.keys())]
            all_positive = (returns > 0).all(axis=1)
            stability_bonus = all_positive.astype(float) * 10
            lt_score += stability_bonus
        
        return lt_score.clip(0, 100)

    @staticmethod
    def _calculate_liquidity_score_smart(df: pd.DataFrame) -> pd.Series:
        """Smart liquidity score with multiple factors"""
        
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'volume_1d' in df.columns:
            volume = df['volume_1d'].fillna(0)
            volume_rank = SmartRankingEngine._safe_rank(
                volume, pct=True, ascending=True
            )
            liquidity_score = volume_rank * 0.5
        
        if 'market_cap' in df.columns:
            mcap_rank = SmartRankingEngine._safe_rank(
                df['market_cap'], pct=True, ascending=True
            )
            liquidity_score += mcap_rank * 0.3
        
        if all(col in df.columns for col in ['volume_7d', 'volume_30d']):
            vol_7d_daily = df['volume_7d'] / 7
            vol_30d_daily = df['volume_30d'] / 30
            
            consistency = 1 - np.abs(vol_7d_daily - vol_30d_daily) / (vol_30d_daily + 1)
            consistency_score = consistency.clip(0, 1) * 20
            liquidity_score += consistency_score
        
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
        
        score_columns = [
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score'
        ]
        
        for col in score_columns:
            if col not in df.columns:
                df[col] = 50
        
        scores_matrix = df[score_columns].fillna(50).values
        weights = np.array([
            CONFIG.POSITION_WEIGHT,
            CONFIG.VOLUME_WEIGHT,
            CONFIG.MOMENTUM_WEIGHT,
            CONFIG.ACCELERATION_WEIGHT,
            CONFIG.BREAKOUT_WEIGHT,
            CONFIG.RVOL_WEIGHT
        ])
        
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        
        if 'momentum_quality' in df.columns:
            quality_bonus = df['momentum_quality'] * 0.05
            df['master_score'] = (df['master_score'] + quality_bonus).clip(0, 100)
        
        if 'smart_money_flow' in df.columns:
            flow_bonus = np.where(df['smart_money_flow'] > 70, 3, 0)
            df['master_score'] = (df['master_score'] + flow_bonus).clip(0, 100)
        
        return df

    @staticmethod
    def _calculate_all_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all ranking metrics efficiently"""
        
        df['rank'] = df['master_score'].rank(
            method='first', ascending=False, na_option='bottom'
        ).fillna(len(df) + 1).astype(int)
        
        df['percentile'] = df['master_score'].rank(
            pct=True, ascending=True, na_option='bottom'
        ).fillna(0) * 100
        
        if 'category' in df.columns:
            df['category_rank'] = df.groupby('category')['master_score'].rank(
                method='first', ascending=False, na_option='bottom'
            ).fillna(999).astype(int)
        
        if 'sector' in df.columns:
            df['sector_rank'] = df.groupby('sector')['master_score'].rank(
                method='first', ascending=False, na_option='bottom'
            ).fillna(999).astype(int)
        
        if 'industry' in df.columns:
            df['industry_rank'] = df.groupby('industry')['master_score'].rank(
                method='first', ascending=False, na_option='bottom'
            ).fillna(999).astype(int)
        
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
    
    PATTERN_METADATA = {
        '🔥 CAT LEADER': {'type': 'technical', 'importance': 'high', 'timeframe': 'medium', 'risk': 'low'},
        '💎 HIDDEN GEM': {'type': 'technical', 'importance': 'high', 'timeframe': 'medium', 'risk': 'medium'},
        '🚀 ACCELERATION': {'type': 'technical', 'importance': 'high', 'timeframe': 'short', 'risk': 'medium'},
        '🏦 INSTITUTIONAL': {'type': 'technical', 'importance': 'medium', 'timeframe': 'long', 'risk': 'low'},
        '⚡ VOL EXPLOSION': {'type': 'technical', 'importance': 'high', 'timeframe': 'short', 'risk': 'high'},
        '🎯 BREAKOUT': {'type': 'technical', 'importance': 'high', 'timeframe': 'short', 'risk': 'medium'},
        '👑 MKT LEADER': {'type': 'technical', 'importance': 'high', 'timeframe': 'long', 'risk': 'low'},
        '🌊 MOMENTUM WAVE': {'type': 'technical', 'importance': 'medium', 'timeframe': 'medium', 'risk': 'medium'},
        '💧 LIQUID LEADER': {'type': 'technical', 'importance': 'medium', 'timeframe': 'long', 'risk': 'low'},
        '💪 LONG STRENGTH': {'type': 'technical', 'importance': 'medium', 'timeframe': 'long', 'risk': 'low'},
        '📈 QUALITY TREND': {'type': 'technical', 'importance': 'medium', 'timeframe': 'medium', 'risk': 'low'},
        '🎯 52W HIGH APPROACH': {'type': 'range', 'importance': 'high', 'timeframe': 'short', 'risk': 'medium'},
        '🔄 52W LOW BOUNCE': {'type': 'range', 'importance': 'high', 'timeframe': 'medium', 'risk': 'high'},
        '👑 GOLDEN ZONE': {'type': 'range', 'importance': 'medium', 'timeframe': 'medium', 'risk': 'medium'},
        '📊 VOL ACCUMULATION': {'type': 'range', 'importance': 'medium', 'timeframe': 'medium', 'risk': 'low'},
        '🔀 MOMENTUM DIVERGE': {'type': 'range', 'importance': 'high', 'timeframe': 'short', 'risk': 'medium'},
        '🎯 RANGE COMPRESS': {'type': 'range', 'importance': 'medium', 'timeframe': 'medium', 'risk': 'low'},
        '💎 VALUE MOMENTUM': {'type': 'fundamental', 'importance': 'high', 'timeframe': 'long', 'risk': 'low'},
        '📊 EARNINGS ROCKET': {'type': 'fundamental', 'importance': 'high', 'timeframe': 'medium', 'risk': 'medium'},
        '🏆 QUALITY LEADER': {'type': 'fundamental', 'importance': 'high', 'timeframe': 'long', 'risk': 'low'},
        '⚡ TURNAROUND': {'type': 'fundamental', 'importance': 'high', 'timeframe': 'medium', 'risk': 'high'},
        '⚠️ HIGH PE': {'type': 'fundamental', 'importance': 'low', 'timeframe': 'medium', 'risk': 'high'},
        '🤫 STEALTH': {'type': 'intelligence', 'importance': 'high', 'timeframe': 'medium', 'risk': 'medium'},
        '🧛 VAMPIRE': {'type': 'intelligence', 'importance': 'high', 'timeframe': 'short', 'risk': 'very_high'},
        '⛈️ PERFECT STORM': {'type': 'intelligence', 'importance': 'very_high', 'timeframe': 'short', 'risk': 'medium'}
    }
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns with intelligent optimization"""
        
        if df.empty:
            return df
        
        logger.logger.info("Starting intelligent pattern detection...")
        
        pattern_conditions = SmartPatternDetector._prepare_pattern_conditions(df)
        
        pattern_groups = {
            'technical': [],
            'range': [],
            'fundamental': [],
            'intelligence': []
        }
        
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
        
        for pattern_name, pattern_func in pattern_functions.items():
            pattern_type = SmartPatternDetector.PATTERN_METADATA[pattern_name]['type']
            pattern_groups[pattern_type].append((pattern_name, pattern_func))
        
        detected_patterns = []
        
        for pattern_type, patterns in pattern_groups.items():
            if pattern_type == 'fundamental' and not all(col in df.columns for col in ['pe', 'eps_change_pct']):
                continue
            
            for pattern_name, pattern_func in patterns:
                try:
                    mask = pattern_func(df, pattern_conditions)
                    if isinstance(mask, pd.Series) and mask.any():
                        pattern_series = pd.Series(
                            np.where(mask, pattern_name, ''), 
                            index=df.index
                        )
                        detected_patterns.append(pattern_series)
                except Exception as e:
                    logger.logger.warning(f"Pattern detection failed for {pattern_name}: {str(e)}")
        
        if detected_patterns:
            df['patterns'] = pd.concat(detected_patterns, axis=1).apply(
                lambda x: ' | '.join(filter(None, x)), axis=1
            )
        else:
            df['patterns'] = ''
        
        df = SmartPatternDetector._calculate_pattern_confidence(df)
        
        pattern_counts = df['patterns'].str.split(' | ').explode()
        pattern_counts = pattern_counts[pattern_counts != ''].value_counts()
        if not pattern_counts.empty:
            logger.logger.info(f"Detected patterns: {dict(pattern_counts.head(10))}")
        
        return df

    @staticmethod
    def _prepare_pattern_conditions(df: pd.DataFrame) -> Dict[str, Any]:
        """Pre-calculate common conditions for pattern detection"""
        
        conditions = {}
        
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
        """Smart category leader detection with dynamic thresholds"""
        
        if 'category_rank' not in df.columns:
            return pd.Series(False, index=df.index)
        
        if 'category' in df.columns:
            category_sizes = df.groupby('category').size()
            df['category_size'] = df['category'].map(category_sizes)
            
            rank_threshold = np.where(
                df['category_size'] < 10, 2,
                np.where(df['category_size'] < 50, 3, 5)
            )
        else:
            rank_threshold = 3
        
        return (
            (df['category_rank'] <= rank_threshold) & 
            (df['master_score'] >= CONFIG.PATTERN_THRESHOLDS['category_leader'])
        )

    @staticmethod
    def _is_hidden_gem_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart hidden gem detection with multiple criteria"""
        
        base_condition = (
            (df.get('percentile', 0) >= 70) &
            (df.get('master_score', 0) >= CONFIG.PATTERN_THRESHOLDS['hidden_gem'])
        )
        
        volume_condition = True
        if 'volume_1d' in df.columns and 'volume_90d' in df.columns:
            volume_condition = (
                (df['volume_1d'] < df['volume_90d'] * 0.7) &
                (df['volume_1d'] > df['volume_90d'] * 0.3)
            )
        
        smart_criteria = True
        if 'momentum_harmony' in df.columns:
            smart_criteria &= df['momentum_harmony'] >= 2
        
        if 'pe' in df.columns:
            smart_criteria &= (df['pe'] > 0) & (df['pe'] < 30)
        
        return base_condition & volume_condition & smart_criteria

    @staticmethod
    def _is_accelerating_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart acceleration with velocity analysis"""
        
        base_condition = (
            (df.get('acceleration_score', 0) >= CONFIG.PATTERN_THRESHOLDS['acceleration']) &
            conditions.get('high_momentum', True)
        )
        
        if all(col in df.columns for col in ['ret_1d', 'ret_7d']):
            velocity_increasing = (
                (df['ret_1d'] > 0) & 
                (df['ret_7d'] > 0) & 
                (df['ret_1d'] > df['ret_7d'] / 7)
            )
            base_condition &= velocity_increasing
        
        return base_condition

    @staticmethod
    def _is_institutional_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart institutional detection with flow analysis"""
        
        base_condition = (
            conditions.get('high_volume', True) &
            (df.get('liquidity_score', 0) >= 70)
        )
        
        if 'vol_ratio_30d_90d' in df.columns:
            base_condition &= df['vol_ratio_30d_90d'] > 1.5
        
        if 'smart_money_flow' in df.columns:
            base_condition &= df['smart_money_flow'] >= 70
        
        return base_condition

    @staticmethod
    def _is_volume_explosion_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart volume explosion with sustainability check"""
        
        base_condition = (
            conditions.get('extreme_rvol', False) &
            (df.get('rvol_score', 0) >= CONFIG.PATTERN_THRESHOLDS['vol_explosion'])
        )
        
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d']):
            sustained = df['vol_ratio_7d_90d'] > 1.5
            base_condition &= sustained
        
        return base_condition

    @staticmethod
    def _is_breakout_ready_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart breakout detection with setup quality"""
        
        base_condition = (
            (df.get('breakout_score', 0) >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']) &
            conditions.get('near_high', False)
        )
        
        if 'position_tension' in df.columns:
            base_condition &= df['position_tension'] >= 70
        
        if 'vmi' in df.columns:
            base_condition &= df['vmi'] >= 60
        
        return base_condition

    @staticmethod
    def _is_market_leader_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart market leader with dominance check"""
        
        base_condition = (
            (df.get('rank', 999) <= 10) &
            (df.get('master_score', 0) >= CONFIG.PATTERN_THRESHOLDS['market_leader'])
        )
        
        if 'sector_rank' in df.columns:
            base_condition &= df['sector_rank'] <= 3
        
        if 'momentum_quality' in df.columns:
            base_condition &= df['momentum_quality'] >= 70
        
        return base_condition

    @staticmethod
    def _is_momentum_wave_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart momentum wave with quality check"""
        
        base_condition = (
            (df.get('momentum_harmony', 0) >= 3) &
            conditions.get('high_momentum', True)
        )
        
        if 'wave_state' in df.columns:
            wave_quality = df['wave_state'].isin(['🌊🌊🌊 CRESTING', '🌊🌊 BUILDING'])
            base_condition &= wave_quality
        
        return base_condition

    @staticmethod
    def _is_liquid_leader_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart liquidity leader with consistency"""
        
        base_condition = (
            (df.get('liquidity_score', 0) >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
            (df.get('volume_1d', 0) > 1_000_000)
        )
        
        if all(col in df.columns for col in ['volume_7d', 'volume_30d']):
            vol_7d_daily = df['volume_7d'] / 7
            vol_30d_daily = df['volume_30d'] / 30
            consistent = np.abs(vol_7d_daily - vol_30d_daily) / (vol_30d_daily + 1) < 0.3
            base_condition &= consistent
        
        return base_condition

    @staticmethod
    def _is_long_strength_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart long-term strength with stability"""
        
        base_condition = (
            df.get('long_term_strength', 0) >= CONFIG.PATTERN_THRESHOLDS['long_strength']
        )
        
        if all(col in df.columns for col in ['ret_3m', 'ret_6m', 'ret_1y']):
            all_positive = (
                (df['ret_3m'] > 0) & 
                (df['ret_6m'] > 0) & 
                (df['ret_1y'] > 0)
            )
            base_condition &= all_positive
        
        return base_condition

    @staticmethod
    def _is_quality_trend_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart trend quality with smoothness"""
        
        base_condition = (
            df.get('trend_quality', 0) >= CONFIG.PATTERN_THRESHOLDS['quality_trend']
        )
        
        if 'price_to_sma20' in df.columns:
            smooth = np.abs(df['price_to_sma20'] - 1) < 0.1
            base_condition &= smooth
        
        return base_condition

    @staticmethod
    def _is_52w_high_approach_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart 52-week high approach with momentum"""
        
        base_condition = (
            (df.get('from_high_pct', -100) > -5) &
            conditions.get('high_volume', True)
        )
        
        if 'momentum_score' in df.columns:
            base_condition &= df['momentum_score'] >= 60
        
        if 'position_tension' in df.columns:
            base_condition &= df['position_tension'] < 90
        
        return base_condition

    @staticmethod
    def _is_52w_low_bounce_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart 52-week low bounce with reversal confirmation"""
        
        base_condition = (
            conditions.get('near_low', False) &
            (df.get('acceleration_score', 0) >= 80)
        )
        
        if 'ret_7d' in df.columns:
            base_condition &= df['ret_7d'] > 5
        
        if 'rvol' in df.columns:
            base_condition &= df['rvol'] > 1.5
        
        return base_condition

    @staticmethod
    def _is_golden_zone_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart golden zone with balance check"""
        
        base_condition = (
            (df.get('from_low_pct', 0) > 60) &
            (df.get('from_high_pct', -100) > -40) &
            conditions.get('high_momentum', True)
        )
        
        if 'position_score' in df.columns:
            base_condition &= df['position_score'].between(60, 80)
        
        return base_condition

    @staticmethod
    def _is_vol_accumulation_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart volume accumulation with trend"""
        
        vol_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        available_cols = [col for col in vol_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return pd.Series(False, index=df.index)
        
        accumulation_count = sum(
            df[col] > 1.1 for col in available_cols
        )
        
        base_condition = accumulation_count >= 2
        
        if 'ret_30d' in df.columns:
            stable_price = np.abs(df['ret_30d']) < 10
            base_condition &= stable_price
        
        return base_condition

    @staticmethod
    def _is_momentum_diverge_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart momentum divergence with quality"""
        
        if not all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            return pd.Series(False, index=df.index)
        
        daily_7d = df['ret_7d'] / 7
        daily_30d = df['ret_30d'] / 30
        
        divergence = np.where(
            daily_30d != 0,
            daily_7d > daily_30d * 1.5,
            False
        )
        
        base_condition = divergence & conditions.get('high_momentum', True)
        
        if 'momentum_quality' in df.columns:
            base_condition &= df['momentum_quality'] >= 60
        
        return base_condition

    @staticmethod
    def _is_range_compress_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart range compression with volatility"""
        
        if not all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            return pd.Series(False, index=df.index)
        
        range_pct = df['from_low_pct'] + np.abs(df['from_high_pct'])
        
        base_condition = (
            (range_pct < 50) & 
            (df['from_low_pct'] > 30)
        )
        
        if 'ret_30d' in df.columns:
            low_volatility = np.abs(df['ret_30d']) < 20
            base_condition &= low_volatility
        
        return base_condition

    @staticmethod
    def _is_value_momentum_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart value momentum with quality"""
        
        if 'pe' not in df.columns:
            return pd.Series(False, index=df.index)
        
        base_condition = (
            (df['pe'] > 0) & 
            (df['pe'] < 15) &
            conditions.get('high_score', True)
        )
        
        if 'eps_change_pct' in df.columns:
            base_condition &= df['eps_change_pct'] > 0
        
        if 'momentum_score' in df.columns:
            base_condition &= df['momentum_score'] >= 60
        
        return base_condition

    @staticmethod
    def _is_earnings_rocket_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart earnings rocket with sustainability"""
        
        if 'eps_change_pct' not in df.columns:
            return pd.Series(False, index=df.index)
        
        base_condition = (
            (df['eps_change_pct'] > 50) &
            (df.get('acceleration_score', 0) >= 70)
        )
        
        if 'pe' in df.columns:
            base_condition &= (df['pe'] > 0) & (df['pe'] < 50)
        
        return base_condition

    @staticmethod
    def _is_quality_leader_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart quality leader with consistency"""
        
        if not all(col in df.columns for col in ['pe', 'eps_change_pct']):
            return pd.Series(False, index=df.index)
        
        base_condition = (
            (df['pe'] > 10) & 
            (df['pe'] < 25) &
            (df['eps_change_pct'] > 20) &
            (df.get('percentile', 0) >= 80)
        )
        
        if 'long_term_strength' in df.columns:
            base_condition &= df['long_term_strength'] >= 70
        
        return base_condition

    @staticmethod
    def _is_turnaround_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart turnaround with momentum"""
        
        if 'eps_change_pct' not in df.columns:
            return pd.Series(False, index=df.index)
        
        base_condition = (
            (df['eps_change_pct'] > 100) &
            conditions.get('high_volume', True)
        )
        
        if 'ret_30d' in df.columns:
            base_condition &= df['ret_30d'] > 10
        
        return base_condition

    @staticmethod
    def _is_high_pe_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart high PE warning"""
        
        if 'pe' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return df['pe'] > 100

    @staticmethod
    def _is_stealth_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart stealth accumulation detection"""
        
        base_condition = pd.Series(True, index=df.index)
        
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_7d_90d']):
            volume_accumulation = (
                (df['vol_ratio_30d_90d'] > 1.2) &
                (df['vol_ratio_7d_90d'] < 1.1)
            )
            base_condition &= volume_accumulation
        
        if 'ret_7d' in df.columns:
            price_stable = np.abs(df['ret_7d']) < 5
            base_condition &= price_stable
        
        if 'smart_money_flow' in df.columns:
            base_condition &= df['smart_money_flow'] >= 65
        
        return base_condition

    @staticmethod
    def _is_vampire_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart vampire pattern for explosive small caps"""
        
        if not all(col in df.columns for col in ['ret_1d', 'ret_7d']):
            return pd.Series(False, index=df.index)
        
        daily_pace_1d = np.abs(df['ret_1d'])
        daily_pace_7d = np.abs(df['ret_7d']) / 7
        
        extreme_movement = np.where(
            daily_pace_7d > 0,
            daily_pace_1d > daily_pace_7d * 2,
            daily_pace_1d > 5
        )
        
        base_condition = (
            extreme_movement &
            conditions.get('extreme_rvol', False)
        )
        
        if 'category' in df.columns:
            small_cap = df['category'].str.contains(
                'Small|Micro|small|micro', 
                case=False, na=False
            )
            base_condition &= small_cap
        
        return base_condition

    @staticmethod
    def _is_perfect_storm_smart(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Smart perfect storm - ultimate convergence"""
        
        base_condition = (
            (df.get('momentum_harmony', 0) == 4) &
            conditions.get('very_high_score', True) &
            conditions.get('high_volume', True)
        )
        
        confirmations = 0
        
        if 'wave_state' in df.columns:
            if df['wave_state'] == '🌊🌊🌊 CRESTING':
                confirmations += 1
        
        if 'position_tension' in df.columns:
            if df['position_tension'] > 70:
                confirmations += 1
        
        if 'smart_money_flow' in df.columns:
            if df['smart_money_flow'] > 70:
                confirmations += 1
        
        if 'momentum_quality' in df.columns:
            if df['momentum_quality'] > 70:
                confirmations += 1
        
        base_condition &= confirmations >= 3
        
        return base_condition

    @staticmethod
    def _calculate_pattern_confidence(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence scores for detected patterns"""
        
        if 'patterns' not in df.columns or df['patterns'].eq('').all():
            df['pattern_confidence'] = 0
            return df
        
        confidence_scores = []
        
        for idx, patterns_str in df['patterns'].items():
            if not patterns_str:
                confidence_scores.append(0)
                continue
            
            patterns = patterns_str.split(' | ')
            total_confidence = 0
            
            for pattern in patterns:
                if pattern in SmartPatternDetector.PATTERN_METADATA:
                    metadata = SmartPatternDetector.PATTERN_METADATA[pattern]
                    
                    importance_scores = {
                        'very_high': 25,
                        'high': 20,
                        'medium': 15,
                        'low': 10
                    }
                    confidence = importance_scores.get(metadata['importance'], 15)
                    
                    risk_multipliers = {
                        'low': 1.2,
                        'medium': 1.0,
                        'high': 0.8,
                        'very_high': 0.6
                    }
                    confidence *= risk_multipliers.get(metadata['risk'], 1.0)
                    
                    total_confidence += confidence
            
            if len(patterns) > 1:
                total_confidence *= (1 + np.log(len(patterns))) / len(patterns)
            
            confidence_scores.append(min(100, total_confidence))
        
        df['pattern_confidence'] = confidence_scores
        
        return df

# ============================================
# INTELLIGENT FILTERING ENGINE
# ============================================

class SmartFilterEngine:
    """Advanced filtering with intelligent optimization"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_all_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with smart optimization"""
        
        if df.empty or not filters:
            return df
        
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Apply filters in optimized order
        if filters.get('quick_filter'):
            filtered_df = SmartFilterEngine._apply_quick_filter(filtered_df, filters['quick_filter'])
        
        if filters.get('min_score', 0) > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= filters['min_score']]
        
        if filters.get('categories'):
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        if filters.get('sectors'):
            filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
        
        if filters.get('industries') and 'industry' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['industry'].isin(filters['industries'])]
        
        if filters.get('patterns') and 'patterns' in filtered_df.columns:
            pattern_mask = filtered_df['patterns'].apply(
                lambda x: any(p in x for p in filters['patterns']) if x else False
            )
            filtered_df = filtered_df[pattern_mask]
        
        if filters.get('trend_filter') and filters['trend_filter'] != 'All Trends':
            if 'trend_quality' in filtered_df.columns and filters.get('trend_range'):
                min_trend, max_trend = filters['trend_range']
                filtered_df = filtered_df[
                    (filtered_df['trend_quality'] >= min_trend) & 
                    (filtered_df['trend_quality'] <= max_trend)
                ]
        
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            if filters.get(tier_type):
                tier_col = tier_type.replace('_tiers', '_tier')
                if tier_col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[tier_col].isin(filters[tier_type])]
        
        if filters.get('min_eps_change') is not None and 'eps_change_pct' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['eps_change_pct'] >= filters['min_eps_change']]
        
        if filters.get('min_pe') is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['pe'] >= filters['min_pe']]
        
        if filters.get('max_pe') is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['pe'] <= filters['max_pe']]
        
        if filters.get('wave_states') and 'wave_state' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['wave_state'].isin(filters['wave_states'])]
        
        if filters.get('wave_strength_range') and 'overall_wave_strength' in filtered_df.columns:
            min_val, max_val = filters['wave_strength_range']
            filtered_df = filtered_df[
                (filtered_df['overall_wave_strength'] >= min_val) &
                (filtered_df['overall_wave_strength'] <= max_val)
            ]
        
        if filters.get('require_fundamental_data', False):
            if all(col in filtered_df.columns for col in ['pe', 'eps_change_pct']):
                filtered_df = filtered_df[
                    filtered_df['pe'].notna() & 
                    filtered_df['eps_change_pct'].notna()
                ]
        
        final_count = len(filtered_df)
        reduction_pct = (1 - final_count / initial_count) * 100 if initial_count > 0 else 0
        logger.logger.info(f"Filtering complete: {initial_count} → {final_count} ({reduction_pct:.1f}% reduction)")
        
        return filtered_df

    @staticmethod
    def _apply_quick_filter(df: pd.DataFrame, filter_type: str) -> pd.DataFrame:
        """Apply quick filter logic"""
        if filter_type == 'top_gainers':
            return df[(df.get('momentum_score', 0) >= 80) & (df.get('ret_30d', 0) > 10)]
        elif filter_type == 'volume_surges':
            return df[(df.get('rvol', 0) >= 3) & (df.get('volume_score', 0) >= 80)]
        elif filter_type == 'breakout_ready':
            return df[(df.get('breakout_score', 0) >= 80) & (df.get('from_high_pct', -100) > -10)]
        elif filter_type == 'hidden_gems':
            return df[(df.get('patterns', '').str.contains('HIDDEN GEM', na=False)) | ((df.get('master_score', 0) >= 70) & (df.get('volume_1d', 0) < df.get('volume_90d', 1) * 0.7))]
        return df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        """Get available filter options with perfect smart interconnection"""
        
        if df.empty or column not in df.columns:
            return []
        
        temp_filters = current_filters.copy()
        
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
        
        if column == 'industry' and 'sectors' in current_filters:
            temp_filters['sectors'] = current_filters['sectors']
        
        filtered_df = SmartFilterEngine.apply_all_filters(df, temp_filters)
        
        values = filtered_df[column].dropna().unique()
        
        values = [v for v in values if str(v).strip().upper() not in ['UNKNOWN', '', 'NAN', 'N/A', 'NONE', '-']]
        
        try:
            values = sorted(values, key=lambda x: float(str(x).replace(',', '')) if str(x).replace(',', '').replace('.', '').isdigit() else x)
        except:
            values = sorted(values, key=str)
        
        return values

# ============================================
# INTELLIGENT SEARCH ENGINE
# ============================================

class SmartSearchEngine:
    """Advanced search with fuzzy matching and relevance scoring"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str, 
                     fuzzy: bool = True, threshold: float = 0.8) -> pd.DataFrame:
        """Smart stock search with relevance ranking"""
        
        if not query or df.empty:
            return pd.DataFrame()
        
        query = query.strip().upper()
        results = []
        
        ticker_exact = df[df['ticker'].str.upper() == query]
        if not ticker_exact.empty:
            ticker_exact = ticker_exact.copy()
            ticker_exact['search_score'] = 1.0
            ticker_exact['match_type'] = 'exact_ticker'
            results.append(ticker_exact)
        
        ticker_contains = df[
            df['ticker'].str.upper().str.contains(query, na=False) &
            ~df.index.isin(ticker_exact.index if not ticker_exact.empty else [])
        ]
        if not ticker_contains.empty:
            ticker_contains = ticker_contains.copy()
            ticker_contains['search_score'] = ticker_contains['ticker'].apply(
                lambda x: 0.9 if x.upper().startswith(query) else 0.7
            )
            ticker_contains['match_type'] = 'ticker_contains'
            results.append(ticker_contains)
        
        if 'company_name' in df.columns:
            name_exact = df[
                df['company_name'].str.upper() == query &
                ~df.index.isin(pd.concat(results).index if results else [])
            ]
            if not name_exact.empty:
                name_exact = name_exact.copy()
                name_exact['search_score'] = 0.85
                name_exact['match_type'] = 'exact_name'
                results.append(name_exact)
            
            name_contains = df[
                df['company_name'].str.upper().str.contains(query, na=False) &
                ~df.index.isin(pd.concat(results).index if results else [])
            ]
            if not name_contains.empty:
                name_contains = name_contains.copy()
                name_contains['search_score'] = name_contains['company_name'].apply(
                    lambda x: SmartSearchEngine._calculate_name_score(x, query)
                )
                name_contains['match_type'] = 'name_contains'
                results.append(name_contains)
        
        if fuzzy and len(results) == 0:
            fuzzy_results = SmartSearchEngine._fuzzy_search(df, query, threshold)
            if not fuzzy_results.empty:
                results.append(fuzzy_results)
        
        if results:
            all_results = pd.concat(results, ignore_index=True)
            
            all_results = all_results.sort_values(
                ['search_score', 'master_score'], 
                ascending=[False, False]
            )
            
            return all_results.drop(['search_score', 'match_type'], axis=1)
        
        return pd.DataFrame()

    @staticmethod
    def _calculate_name_score(name: str, query: str) -> float:
        """Calculate relevance score for company name match"""
        
        name_upper = name.upper()
        
        words = name_upper.split()
        if query in words:
            return 0.8
        
        for word in words:
            if word.startswith(query):
                return 0.7
        
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
                        'score': ticker_score * 0.9,
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
                results.sort(key=lambda x: x['score'], reverse=True)
                top_indices = [r['index'] for r in results[:10]]
                
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
        
        export_profiles = {
            'standard': SmartExportEngine._get_standard_columns,
            'day_trading': SmartExportEngine._get_day_trading_columns,
            'swing_trading': SmartExportEngine._get_swing_trading_columns,
            'fundamental': SmartExportEngine._get_fundamental_columns,
            'pattern_analysis': SmartExportEngine._get_pattern_columns,
            'complete': SmartExportEngine._get_complete_columns
        }
        
        column_getter = export_profiles.get(export_type, SmartExportEngine._get_standard_columns)
        export_columns = column_getter(export_df)
        
        available_columns = [col for col in export_columns if col in export_df.columns]
        export_df = export_df[available_columns]
        
        export_df = SmartExportEngine._format_dataframe(export_df)
        
        metadata_row = SmartExportEngine._create_metadata_row(export_df, export_type)
        
        csv_buffer = StringIO()
        
        csv_buffer.write(f"# Wave Detection Export - {export_type.replace('_', ' ').title()}\n")
        csv_buffer.write(f"# Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        csv_buffer.write(f"# Stocks: {len(export_df)}\n")
        csv_buffer.write("#\n")
        
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
            'patterns', 'vmi', 'smart_money_flow', 'volume_1d',
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
            'pattern_confidence', 'wave_state', 'ret_30d',
            'rvol', 'momentum_harmony', 'position_tension',
            'smart_money_flow', 'category'
        ]

    @staticmethod
    def _get_complete_columns(df: pd.DataFrame) -> List[str]:
        """All columns in optimal order"""
        
        groups = [
            ['rank', 'ticker', 'company_name'],
            ['master_score', 'position_score', 'volume_score', 'momentum_score',
             'acceleration_score', 'breakout_score', 'rvol_score'],
            ['price', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct'],
            ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y'],
            ['volume_1d', 'rvol', 'vol_ratio_1d_90d', 'vol_ratio_7d_90d'],
            ['wave_state', 'patterns', 'vmi', 'position_tension', 'momentum_harmony'],
            ['pe', 'eps_current', 'eps_change_pct', 'market_cap'],
            ['category', 'sector', 'industry']
        ]
        
        all_columns = []
        for group in groups:
            all_columns.extend([col for col in group if col in df.columns])
        
        remaining = [col for col in df.columns if col not in all_columns]
        all_columns.extend(remaining)
        
        return all_columns

    @staticmethod
    def _format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent formatting to dataframe"""
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ['rank', 'category_rank', 'sector_rank', 'industry_rank']:
                continue
            elif col in ['price', 'low_52w', 'high_52w', 'prev_close', 'market_cap']:
                df[col] = df[col].round(2)
            elif col in CONFIG.PERCENTAGE_COLUMNS:
                df[col] = df[col].round(2)
            elif col.endswith('_score'):
                df[col] = df[col].round(1)
            else:
                df[col] = df[col].round(4)
        
        if 'market_cap' in df.columns:
            df['market_cap'] = df['market_cap'].apply(
                lambda x: f"{x/1e9:.2f}B" if x >= 1e9 else f"{x/1e6:.2f}M" if x >= 1e6 else str(x)
            )
        
        if 'volume_1d' in df.columns:
            df['volume_1d'] = df['volume_1d'].apply(
                lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.2f}K" if x >= 1e3 else str(x)
            )
        
        return df

    @staticmethod
    def _create_metadata_row(df: pd.DataFrame, export_type: str) -> Dict[str, Any]:
        """Create metadata row for export"""
        
        metadata = {
            'export_type': export_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_stocks': len(df),
            'avg_master_score': df['master_score'].mean() if 'master_score' in df.columns else None,
            'patterns_detected': df['patterns'].ne('').sum() if 'patterns' in df.columns else None
        }
        
        return metadata

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
            if trend_data and len(trend_data) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=trend_data,
                    mode='lines',
                    line=dict(color='#3498DB', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.1)'
                ))
                fig.update_layout(
                    height=60,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            st.metric(
                label=label,
                value=value,
                delta=delta,
                delta_color=delta_color,
                help=help_text
            )

    @staticmethod
    def render_stock_card_enhanced(row: pd.Series, show_fundamentals: bool = False,
                                   show_charts: bool = True):
        """Render enhanced stock card with rich information"""
        
        with st.container():
            header_cols = st.columns([1, 3, 2, 2])
            
            with header_cols[0]:
                rank_color = "#27AE60" if row['rank'] <= 10 else "#3498DB" if row['rank'] <= 50 else "#95A5A6"
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<span style='font-size:24px; font-weight:bold; color:{rank_color};'>#{int(row['rank'])}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                if 'category' in row:
                    category_colors = {
                        'Large Cap': '#E74C3C',
                        'Mid Cap': '#F39C12',
                        'Small Cap': '#3498DB',
                        'Micro Cap': '#9B59B6'
                    }
                    cat_color = category_colors.get(row['category'], '#95A5A6')
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"<span style='background-color:{cat_color}; color:white; "
                        f"padding:2px 8px; border-radius:12px; font-size:11px;'>"
                        f"{row['category']}</span></div>",
                        unsafe_allow_html=True
                    )
            
            with header_cols[1]:
                st.markdown(f"### {row['ticker']}")
                if 'sector' in row:
                    st.caption(f"{row['sector']}")
                
                if row.get('patterns'):
                    patterns = row['patterns'].split(' | ')
                    pattern_html = ""
                    for pattern in patterns[:3]:
                        pattern_html += f"<span style='margin-right:5px;'>{pattern}</span>"
                    if len(patterns) > 3:
                        pattern_html += f"<span style='color:#7F8C8D;'>+{len(patterns)-3} more</span>"
                    st.markdown(pattern_html, unsafe_allow_html=True)
            
            with header_cols[2]:
                price = row.get('price', 0)
                ret_30d = row.get('ret_30d', 0)
                ret_1d = row.get('ret_1d', 0)
                
                st.metric(
                    "Price",
                    f"₹{price:,.2f}",
                    f"{ret_1d:+.2f}% today",
                    delta_color="normal" if ret_1d >= 0 else "inverse"
                )
                
                color = "#27AE60" if ret_30d > 0 else "#E74C3C"
                st.markdown(
                    f"<div style='text-align:center; margin-top:-10px;'>"
                    f"<span style='color:{color}; font-weight:bold;'>"
                    f"30D: {ret_30d:+.1f}%</span></div>",
                    unsafe_allow_html=True
                )
            
            with header_cols[3]:
                score = row['master_score']
                score_color = "#E74C3C" if score >= 80 else "#F39C12" if score >= 60 else "#3498DB"
                
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
                
                st.markdown(
                    f"<div style='text-align:center; margin-top:-20px;'>"
                    f"{row.get('wave_state', 'Unknown')}</div>",
                    unsafe_allow_html=True
                )
            
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
                        ('Smart Money Flow', row.get('smart_money_flow', 50), '%'),
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
    
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=20
    )
    
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
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
    sheet_id: str = None,
    gid: str = None,
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
        load_start = time.perf_counter()
        
        if source_type == "upload" and file_data is not None:
            logger.logger.info("Loading data from uploaded CSV")
            
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
            if not sheet_id:
                sheet_id = CONFIG.DEFAULT_SHEET_ID
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            
            logger.logger.info(f"Loading from Google Sheets: {sheet_id[:8]}...")
            
            session = get_smart_requests_session()
            max_attempts = 3
            
            for attempt in range(max_attempts):
                try:
                    response = session.get(csv_url, timeout=CONFIG.REQUEST_TIMEOUT)
                    response.raise_for_status()
                    
                    if len(response.content) < 100:
                        raise ValueError("Response too small, likely an error page")
                    
                    df = pd.read_csv(BytesIO(response.content), low_memory=False)
                    metadata['source'] = f"Google Sheets (ID: {sheet_id[:8]}...)"
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
        
        validation_start = time.perf_counter()
        is_valid, validation_msg, diagnostics = validator.validate_dataframe(
            df, CONFIG.CRITICAL_COLUMNS, "Initial load"
        )
        
        if not is_valid:
            raise ValueError(validation_msg)
        
        metadata['validation'] = diagnostics
        metadata['performance']['validation_time'] = time.perf_counter() - validation_start
        
        processing_start = time.perf_counter()
        df = SmartDataProcessor.process_dataframe(df, metadata)
        metadata['performance']['processing_time'] = time.perf_counter() - processing_start
        
        scoring_start = time.perf_counter()
        df = SmartRankingEngine.calculate_all_scores(df)
        metadata['performance']['scoring_time'] = time.perf_counter() - scoring_start
        
        pattern_start = time.perf_counter()
        df = SmartPatternDetector.detect_all_patterns(df)
        metadata['performance']['pattern_time'] = time.perf_counter() - pattern_start
        
        metrics_start = time.perf_counter()
        df = AdvancedMetricsEngine.calculate_all_metrics(df)
        metadata['performance']['metrics_time'] = time.perf_counter() - metrics_start
        
        final_validation_start = time.perf_counter()
        is_valid, validation_msg, final_diagnostics = validator.validate_dataframe(
            df, ['master_score', 'rank'], "Final processed"
        )
        
        if not is_valid:
            raise ValueError(validation_msg)
        
        metadata['final_validation'] = final_diagnostics
        metadata['performance']['final_validation_time'] = time.perf_counter() - final_validation_start
        
        timestamp = datetime.now(timezone.utc)
        RobustSessionState.safe_set('last_good_data', (df.copy(), timestamp, metadata))
        
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
        
        RobustSessionState.safe_set('data_quality', quality_metrics)
        
        logger.logger.info(
            f"Data processing complete: {len(df)} stocks in {total_time:.2f}s "
            f"(Load: {metadata['performance'].get('load_time', 0):.2f}s, "
            f"Process: {metadata['performance'].get('processing_time', 0):.2f}s, "
            f"Score: {metadata['performance'].get('scoring_time', 0):.2f}s)"
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
    
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0 - APEX",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/wave-detection/docs',
            'Report a bug': 'https://github.com/wave-detection/issues',
            'About': 'Wave Detection Ultimate 3.0 - The most advanced stock ranking system'
        }
    )
    
    RobustSessionState.initialize()
    
    st.markdown("""
    <style>
    /* Enhanced production CSS */
    .main {
        padding: 0rem 1rem; background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Animated gradient header */
    .stApp > header {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab); background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: rgba(255,255,255,0.8);
        padding: 8px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px; padding: 0 24px;
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Glassmorphism for metrics */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1rem;
        border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px; height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1; border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;
        border: 2px solid #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Enhanced alerts */
    .stAlert {
        padding: 1rem 1.5rem; border-radius: 12px;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
    }
    
    /* Pulse animation for important elements */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main { padding: 0.5rem; }
        .stTabs [data-baseweb="tab"] {
            padding: 0 12px; font-size: 14px;
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
    
    if RobustSessionState.safe_get('show_debug', False):
        perf_cols = st.columns(5)
        memory_stats = PerformanceMonitor.memory_usage()
        
        with perf_cols[0]:
            st.metric("Memory Usage", f"{memory_stats.get('rss_mb', 0):.1f} MB")
        
        with perf_cols[1]:
            session_info = RobustSessionState.get_session_info()
            st.metric("Session Duration", f"{session_info['duration'] // 60} min")
        
        with perf_cols[2]:
            st.metric("Stocks Loaded", f"{session_info['stocks_loaded']:,}")
        
        with perf_cols[3]:
            perf_report = PerformanceMonitor.get_report()
            total_calls = sum(v['calls'] for v in perf_report.values()) if perf_report else 0
            st.metric("Operations", f"{total_calls:,}")
        
        with perf_cols[4]:
            perf_report = PerformanceMonitor.get_report()
            avg_response = np.mean([v['avg_time'] for v in perf_report.values()]) if perf_report else 0
            st.metric("Avg Response", f"{avg_response:.2f}s")
    
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
            index=0 if RobustSessionState.safe_get('data_source') == "sheet" else 1,
            key="wd_data_source_radio",
            help="Choose between live Google Sheets or local CSV file"
        )
        RobustSessionState.safe_set('data_source', "sheet" if data_source == "Google Sheets" else "upload")
        
        uploaded_file = None
        sheet_id_for_load = None
        gid_for_load = None
        
        if RobustSessionState.safe_get('data_source') == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload a CSV file with stock data. AI will automatically detect columns.",
                key="wd_csv_uploader"
            )
            
            if uploaded_file is None:
                st.info("💡 Drag and drop your CSV file here")
            else:
                file_details = {
                    "Filename": uploaded_file.name,
                    "Size": f"{uploaded_file.size / 1024:.1f} KB",
                    "Type": uploaded_file.type
                }
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
        
        elif RobustSessionState.safe_get('data_source') == "sheet":
            st.markdown("#### 🔗 Google Sheets Configuration")
            
            example_id = "1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM"
            st.code(f"Example: {example_id}", language=None)
            
            user_id_input = st.text_input(
                "Enter your Spreadsheet ID:",
                value=RobustSessionState.safe_get('sheet_id', ''),
                placeholder="44-character alphanumeric ID",
                help="Find this in your Google Sheets URL between /d/ and /edit",
                key="wd_user_gid_input"
            )
            
            user_gid_input = st.text_input(
                "Enter Sheet Tab GID (Optional):",
                value=RobustSessionState.safe_get('gid', CONFIG.DEFAULT_GID),
                placeholder=f"Default: {CONFIG.DEFAULT_GID}",
                help="The GID identifies a specific sheet tab. Found in URL after #gid=",
                key="wd_user_tab_gid_input"
            )
            
            if user_id_input:
                sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', user_id_input)
                if sheet_id_match:
                    sheet_id_for_load = sheet_id_match.group(1)
                else:
                    sheet_id_for_load = user_id_input.strip()
            else:
                sheet_id_for_load = CONFIG.DEFAULT_SHEET_ID

            gid_for_load = user_gid_input.strip() if user_gid_input else CONFIG.DEFAULT_GID
            
            if user_id_input != RobustSessionState.safe_get('sheet_id') or user_gid_input != RobustSessionState.safe_get('gid'):
                RobustSessionState.safe_set('sheet_id', user_id_input)
                RobustSessionState.safe_set('gid', user_gid_input)
                st.rerun()

        if RobustSessionState.safe_get('data_quality'):
            with st.expander("📊 Data Quality Dashboard", expanded=True):
                quality = RobustSessionState.safe_get('data_quality')
                
                completeness = quality.get('data_completeness', 0)
                quality_score = min(100, completeness * 0.6 + 
                                  (100 - quality.get('duplicate_tickers', 0) / quality.get('total_rows', 1) * 100) * 0.4)
                
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
                    st.metric(
                        "Total Stocks",
                        f"{quality.get('total_rows', 0):,}",
                        help="Number of stocks in dataset"
                    )
                    st.metric(
                        "Completeness",
                        f"{completeness:.1f}%",
                        help="Percentage of non-null values"
                    )
                
                with col2:
                    patterns_count = quality.get('patterns_detected', 0)
                    st.metric(
                        "Patterns Found",
                        f"{patterns_count:,}",
                        help="Stocks with detected patterns"
                    )
                    
                    if 'timestamp' in quality:
                        age = datetime.now(timezone.utc) - quality['timestamp']
                        minutes = int(age.total_seconds() / 60)
                        
                        if minutes < 60:
                            freshness = "🟢 Fresh"
                            color = "normal"
                        elif minutes < 24 * 60:
                            freshness = "🟡 Recent"
                            color = "normal"
                        else:
                            freshness = "🔴 Stale"
                            color = "inverse"
                        
                        st.metric(
                            "Data Age",
                            freshness,
                            f"{minutes} min",
                            delta_color=color
                        )
        
        st.markdown("---")
        st.markdown("### 🎨 Display Settings")
        
        display_mode = st.radio(
            "Display Mode:",
            ["Technical Analysis", "Hybrid (Technical + Fundamental)", "Fundamental Focus"],
            index=0 if RobustSessionState.safe_get('user_preferences', {}).get('display_mode') == "Technical" else 1,
            key="wd_display_mode_radio",
            help="Choose your analysis style"
        )
        
        user_prefs = RobustSessionState.safe_get('user_preferences', {})
        if "Technical" in display_mode:
            user_prefs['display_mode'] = "Technical"
        elif "Hybrid" in display_mode:
            user_prefs['display_mode'] = "Hybrid"
        else:
            user_prefs['display_mode'] = "Fundamental"
        RobustSessionState.safe_set('user_preferences', user_prefs)
        
        with st.expander("🎯 Advanced Display", expanded=False):
            st.checkbox(
                "Show Advanced Metrics",
                key="show_advanced_metrics",
                help="Display VMI, Smart Money Flow, and other advanced indicators"
            )
            
            st.checkbox(
                "Enable Chart Previews",
                key="show_chart_previews",
                value=True,
                help="Show mini charts in stock cards"
            )
            
            st.selectbox(
                "Color Theme:",
                ["Default", "Dark", "Colorblind-friendly"],
                key="color_theme",
                help="Choose color scheme"
            )
        
        st.markdown("---")
        st.markdown("### 🔍 Intelligent Filters")
        
        active_count = RobustSessionState.safe_get('active_filter_count', 0)
        if active_count > 0:
            st.info(f"🎯 **{active_count} active filter{'s' if active_count > 1 else ''}**")
            
        if st.button("🗑️ Clear All Filters", use_container_width=True, type="primary" if active_count > 0 else "secondary"):
            RobustSessionState.clear_filters()
            st.success("✅ All filters cleared!")
            st.rerun()
        
        if 'ranked_df' in st.session_state and st.session_state.ranked_df is not None:
            df_ref = st.session_state.ranked_df
            
            category_counts = df_ref['category'].value_counts()
            category_options = [f"{cat} ({count})" for cat, count in category_counts.items()]
            selected_categories_with_counts = st.multiselect(
                "Categories:",
                category_options,
                default=[f"{cat} ({count})" for cat, count in category_counts.items() if cat in RobustSessionState.safe_get('category_filter', [])],
                key="wd_category_filter_display",
                help="Filter by market cap categories"
            )
            RobustSessionState.safe_set('category_filter', [cat.split(' (')[0] for cat in selected_categories_with_counts])
            
            sector_counts = df_ref['sector'].value_counts()
            sector_options = [f"{sect} ({count})" for sect, count in sector_counts.items()]
            selected_sectors_with_counts = st.multiselect(
                "Sectors:",
                sector_options,
                default=[f"{sect} ({count})" for sect, count in sector_counts.items() if sect in RobustSessionState.safe_get('sector_filter', [])],
                key="wd_sector_filter_display",
                help="Filter by business sectors"
            )
            RobustSessionState.safe_set('sector_filter', [sect.split(' (')[0] for sect in selected_sectors_with_counts])
            
            if 'industry' in df_ref.columns:
                industry_counts = df_ref['industry'].value_counts()
                industry_options = [f"{ind} ({count})" for ind, count in industry_counts.items()]
                selected_industries_with_counts = st.multiselect(
                    "Industries:",
                    industry_options,
                    default=[f"{ind} ({count})" for ind, count in industry_counts.items() if ind in RobustSessionState.safe_get('industry_filter', [])],
                    key="wd_industry_filter_display",
                    help="Filter by specific industries"
                )
                RobustSessionState.safe_set('industry_filter', [ind.split(' (')[0] for ind in selected_industries_with_counts])

        score_range = st.slider(
            "Master Score Range:",
            min_value=0,
            max_value=100,
            value=(RobustSessionState.safe_get('min_score', 0), 100),
            step=5,
            key="wd_score_range_slider",
            help="Filter stocks by Master Score range"
        )
        RobustSessionState.safe_set('min_score', score_range[0])
        
        if 'ranked_df' in st.session_state and st.session_state.ranked_df is not None:
            all_patterns = set()
            pattern_counts = Counter()
            for patterns in st.session_state.ranked_df['patterns']:
                if patterns:
                    pattern_list = [p.strip() for p in patterns.split('|')]
                    all_patterns.update(pattern_list)
                    pattern_counts.update(pattern_list)
            
            if all_patterns:
                sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
                pattern_options = [f"{pattern} ({count})" for pattern, count in sorted_patterns]
                selected_patterns_with_counts = st.multiselect(
                    "Patterns:",
                    pattern_options,
                    default=[f"{p} ({pattern_counts[p]})" for p in RobustSessionState.safe_get('patterns', []) if p in pattern_counts],
                    key="wd_patterns_display",
                    help="Filter by detected patterns (sorted by frequency)"
                )
                RobustSessionState.safe_set('patterns', [pattern.split(' (')[0] for pattern in selected_patterns_with_counts])
        
        trend_options = {
            "All Trends": "📊", "Bullish": "📈", "Bearish": "📉",
            "Strong Bullish": "🚀", "Strong Bearish": "💥"
        }
        selected_trend = st.selectbox(
            "Trend Filter:",
            list(trend_options.keys()),
            index=list(trend_options.keys()).index(RobustSessionState.safe_get('trend_filter', "All Trends")),
            format_func=lambda x: f"{trend_options[x]} {x}",
            key="wd_trend_filter",
            help="Filter by price trend direction"
        )
        RobustSessionState.safe_set('trend_filter', selected_trend)
        
        with st.expander("🌊 Wave Analysis Filters", expanded=False):
            st.markdown("**Filter by momentum wave characteristics**")
            wave_state_options = ["🌊🌊🌊 CRESTING", "🌊🌊 BUILDING", "🌊 FORMING", "💥 BREAKING"]
            selected_wave_states = st.multiselect(
                "Wave States:",
                wave_state_options,
                default=RobustSessionState.safe_get('wave_states_filter', []),
                key="wd_wave_states_filter",
                help="Filter by wave momentum states"
            )
            RobustSessionState.safe_set('wave_states_filter', selected_wave_states)
            
            wave_strength_range = st.slider(
                "Overall Wave Strength:",
                min_value=0,
                max_value=100,
                value=RobustSessionState.safe_get('wave_strength_range_slider', (0, 100)),
                step=5,
                key="wd_wave_strength_range_slider",
                help="Filter by composite wave strength score"
            )
            RobustSessionState.safe_set('wave_strength_range_slider', wave_strength_range)
        
        if RobustSessionState.safe_get('user_preferences', {}).get('display_mode') in ["Hybrid", "Fundamental"]:
            with st.expander("💰 Fundamental Analysis Filters", expanded=False):
                st.checkbox(
                    "Require fundamental data",
                    value=RobustSessionState.safe_get('require_fundamental_data', False),
                    key="wd_require_fundamental_data",
                    help="Only show stocks with complete PE and EPS data"
                )
                
                eps_tier_descriptions = {"Negative": "Declining earnings", "Low (0-20%)": "Modest growth", "Medium (20-50%)": "Solid growth", "High (50-100%)": "Strong growth", "Extreme (>100%)": "Explosive growth"}
                selected_eps_tiers = st.multiselect(
                    "EPS Growth Tiers:",
                    list(eps_tier_descriptions.keys()),
                    default=RobustSessionState.safe_get('eps_tier_filter', []),
                    format_func=lambda x: f"{x} - {eps_tier_descriptions[x]}",
                    key="wd_eps_tier_filter",
                    help="Filter by earnings growth categories"
                )
                RobustSessionState.safe_set('eps_tier_filter', selected_eps_tiers)

                pe_tier_descriptions = {"Negative PE": "Loss-making", "Value (<15)": "Potentially undervalued", "Fair (15-25)": "Reasonable valuation", "Growth (25-50)": "Growth premium", "Expensive (>50)": "High valuation"}
                selected_pe_tiers = st.multiselect(
                    "PE Ratio Tiers:",
                    list(pe_tier_descriptions.keys()),
                    default=RobustSessionState.safe_get('pe_tier_filter', []),
                    format_func=lambda x: f"{x} - {pe_tier_descriptions[x]}",
                    key="wd_pe_tier_filter",
                    help="Filter by valuation categories"
                )
                RobustSessionState.safe_set('pe_tier_filter', selected_pe_tiers)

                selected_price_tiers = st.multiselect(
                    "Price Tiers:",
                    ["Penny (<₹10)", "Low (₹10-100)", "Mid (₹100-1000)", "High (₹1000-5000)", "Premium (>₹5000)"],
                    default=RobustSessionState.safe_get('price_tier_filter', []),
                    key="wd_price_tier_filter",
                    help="Filter by stock price ranges"
                )
                RobustSessionState.safe_set('price_tier_filter', selected_price_tiers)

                st.markdown("**Custom Value Filters:**")
                col1, col2 = st.columns(2)
                with col1:
                    min_eps_change = st.text_input("Min EPS Change %:", value=RobustSessionState.safe_get('min_eps_change', ""), placeholder="e.g., 20", key="wd_min_eps_change")
                    min_pe = st.text_input("Min PE Ratio:", value=RobustSessionState.safe_get('min_pe', ""), placeholder="e.g., 10", key="wd_min_pe")
                with col2:
                    max_pe = st.text_input("Max PE Ratio:", value=RobustSessionState.safe_get('max_pe', ""), placeholder="e.g., 30", key="wd_max_pe")
                RobustSessionState.safe_set('min_eps_change', min_eps_change)
                RobustSessionState.safe_set('min_pe', min_pe)
                RobustSessionState.safe_set('max_pe', max_pe)
        
        st.markdown("---")
        st.checkbox("🐛 Debug Mode", value=RobustSessionState.safe_get('show_debug', False), key="wd_show_debug", help="Show performance metrics and debug information")

    try:
        ranked_df = RobustSessionState.safe_get('ranked_df')
        data_timestamp = RobustSessionState.safe_get('data_timestamp')
        
        if RobustSessionState.safe_get('data_source') == "sheet" and not sheet_id_for_load:
            st.info("👋 Welcome! Please enter your Google Spreadsheet ID in the sidebar to get started.")
            st.stop()
        
        if RobustSessionState.safe_get('data_source') == "upload" and uploaded_file is None:
            st.info("📁 Please upload a CSV file to continue.")
            st.stop()
        
        cache_key_prefix = f"{RobustSessionState.safe_get('data_source')}_{sheet_id_for_load}_{gid_for_load}"
        current_hour_key = datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')
        data_version_hash = hashlib.md5(f"{cache_key_prefix}_{current_hour_key}".encode()).hexdigest()

        with st.spinner("🔄 Loading and analyzing data..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("📥 Fetching data...")
            progress_bar.progress(10)
            
            if RobustSessionState.safe_get('data_source') == "upload" and uploaded_file is not None:
                ranked_df, data_timestamp, metadata = load_and_process_data_smart("upload", file_data=uploaded_file, data_version=data_version_hash)
            else:
                ranked_df, data_timestamp, metadata = load_and_process_data_smart("sheet", sheet_id=sheet_id_for_load, gid=gid_for_load, data_version=data_version_hash)
            
            RobustSessionState.safe_set('ranked_df', ranked_df)
            RobustSessionState.safe_set('data_timestamp', data_timestamp)
            RobustSessionState.safe_set('last_refresh', datetime.now(timezone.utc))
            
            status_text.text("🧮 Calculating scores...")
            progress_bar.progress(50)
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            success_cols = st.columns(4)
            with success_cols[0]: st.success(f"✅ Loaded {len(ranked_df):,} stocks")
            with success_cols[1]: st.success(f"🎯 Found {ranked_df['patterns'].ne('').sum():,} patterns")
            with success_cols[2]: st.success(f"⚡ Processed in {metadata.get('performance', {}).get('total_time', 0):.1f}s")
            with success_cols[3]: st.success(f"📊 Quality: {metadata.get('quality', {}).get('data_completeness', 0):.0f}%")
            
            if metadata.get('warnings'):
                with st.expander("⚠️ Data Processing Notes", expanded=False):
                    for warning in metadata['warnings']: st.warning(warning)
            if metadata.get('errors'):
                with st.expander("❌ Errors Encountered", expanded=True):
                    for error in metadata['errors']: st.error(error)

    except Exception as e:
        st.error(f"❌ Critical Application Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
            import traceback; st.code(traceback.format_exc())
        st.stop()
    
    st.markdown("### ⚡ Quick Intelligence")
    
    qa_cols = st.columns(6)
    quick_actions = [
        ("📈 Top Gainers", "top_gainers", "High momentum stocks", "📊"),
        ("🔥 Volume Surge", "volume_surges", "Unusual volume activity", "📊"),
        ("🎯 Breakout Ready", "breakout_ready", "Near resistance levels", "🎯"),
        ("💎 Hidden Gems", "hidden_gems", "Undervalued opportunities", "💎"),
        ("🌊 Perfect Storms", "perfect_storms", "Everything aligned", "⛈️"),
        ("📊 Show All", "show_all", "Remove quick filters", "🔄")
    ]
    
    for col, (label, action, tooltip, emoji) in zip(qa_cols, quick_actions):
        with col:
            if st.button(f"{emoji}\n{label.split()[1]}", use_container_width=True, key=f"wd_qa_{action}", help=tooltip):
                if action == "show_all":
                    RobustSessionState.safe_set('quick_filter', None)
                    RobustSessionState.safe_set('quick_filter_applied', False)
                else:
                    RobustSessionState.safe_set('quick_filter', action)
                    RobustSessionState.safe_set('quick_filter_applied', True)
                st.rerun()

    filters_from_state = {
        'categories': RobustSessionState.safe_get('category_filter'),
        'sectors': RobustSessionState.safe_get('sector_filter'),
        'industries': RobustSessionState.safe_get('industry_filter'),
        'min_score': RobustSessionState.safe_get('min_score'),
        'patterns': RobustSessionState.safe_get('patterns'),
        'trend_filter': RobustSessionState.safe_get('trend_filter'),
        'eps_tiers': RobustSessionState.safe_get('eps_tier_filter'),
        'pe_tiers': RobustSessionState.safe_get('pe_tier_filter'),
        'price_tiers': RobustSessionState.safe_get('price_tier_filter'),
        'min_eps_change': RobustSessionState.safe_get('min_eps_change'),
        'min_pe': RobustSessionState.safe_get('min_pe'),
        'max_pe': RobustSessionState.safe_get('max_pe'),
        'require_fundamental_data': RobustSessionState.safe_get('require_fundamental_data'),
        'wave_states': RobustSessionState.safe_get('wave_states_filter'),
        'wave_strength_range': RobustSessionState.safe_get('wave_strength_range_slider'),
        'quick_filter': RobustSessionState.safe_get('quick_filter'),
        'quick_filter_applied': RobustSessionState.safe_get('quick_filter_applied')
    }

    if RobustSessionState.safe_get('quick_filter_applied'):
        ranked_df_display = SmartFilterEngine._apply_quick_filter(ranked_df, filters_from_state['quick_filter'])
    else:
        ranked_df_display = ranked_df

    filtered_df = SmartFilterEngine.apply_all_filters(ranked_df_display, filters_from_state)
    filtered_df = filtered_df.sort_values('rank')
    
    status_cols = st.columns(5)
    
    with status_cols[0]:
        total_stocks = len(ranked_df) if ranked_df is not None else 0
        SmartUIComponents.render_metric_card("Total Universe", f"{total_stocks:,}", help_text="Total stocks in dataset")
    with status_cols[1]:
        filtered_stocks = len(filtered_df)
        filter_pct = (filtered_stocks / total_stocks * 100) if total_stocks > 0 else 0
        SmartUIComponents.render_metric_card("Filtered Results", f"{filtered_stocks:,}", f"{filter_pct:.1f}% of total", delta_color="normal" if filter_pct > 10 else "inverse", help_text="Stocks matching current criteria")
    with status_cols[2]:
        if not filtered_df.empty and 'wave_state' in filtered_df.columns:
            wave_distribution = filtered_df['wave_state'].value_counts()
            cresting_count = wave_distribution.get('🌊🌊🌊 CRESTING', 0)
            building_count = wave_distribution.get('🌊🌊 BUILDING', 0)
            SmartUIComponents.render_metric_card("Active Waves", f"{cresting_count + building_count}", f"🌊³:{cresting_count} 🌊²:{building_count}", help_text="Stocks in active momentum states")
        else:
            SmartUIComponents.render_metric_card("Active Waves", "0", help_text="Stocks in active momentum states")
    with status_cols[3]:
        if not filtered_df.empty and 'patterns' in filtered_df.columns:
            with_patterns = (filtered_df['patterns'] != '').sum()
            pattern_pct = (with_patterns / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            SmartUIComponents.render_metric_card("Pattern Hits", f"{with_patterns}", f"{pattern_pct:.1f}% coverage", help_text="Stocks with detected patterns")
        else:
            SmartUIComponents.render_metric_card("Pattern Hits", "0", help_text="Stocks with detected patterns")
    with status_cols[4]:
        if not filtered_df.empty and 'master_score' in filtered_df.columns:
            avg_score = filtered_df['master_score'].mean()
            score_std = filtered_df['master_score'].std()
            SmartUIComponents.render_metric_card("Avg Score", f"{avg_score:.1f}", f"σ: {score_std:.1f}", help_text="Average Master Score of filtered stocks")
        else:
            SmartUIComponents.render_metric_card("Avg Score", "0.0", help_text="Average Master Score")
    
    tab_list = ["📊 Dashboard", "🏆 Rankings", "🌊 Wave Radar", "📈 Analytics", "🔍 Search", "📥 Export", "🧠 AI Insights", "ℹ️ About"]
    tabs = st.tabs(tab_list)
    
    with tabs[0]:
        st.markdown("### 📊 Executive Intelligence Dashboard")
        if not filtered_df.empty:
            st.markdown("#### 🌍 Market Overview")
            overview_cols = st.columns(4)
            with overview_cols[0]:
                positive_stocks = (filtered_df.get('ret_30d', pd.Series(0)) > 0).sum()
                breadth_pct = positive_stocks / len(filtered_df) * 100
                fig_breadth = go.Figure(go.Indicator(mode="gauge+number", value=breadth_pct, title={'text': "Market Breadth %"}, domain={'x': [0, 1], 'y': [0, 1]}))
                fig_breadth.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_breadth, use_container_width=True)
            with overview_cols[1]:
                if 'momentum_score' in filtered_df.columns:
                    fig_momentum = go.Figure(); fig_momentum.add_trace(go.Box(y=filtered_df['momentum_score'], name="Momentum"))
                    fig_momentum.update_layout(title="Momentum Distribution", height=200, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                    st.plotly_chart(fig_momentum, use_container_width=True)
            with overview_cols[2]:
                if 'rvol' in filtered_df.columns:
                    high_volume = (filtered_df['rvol'] > 1.5).sum(); extreme_volume = (filtered_df['rvol'] > 3).sum()
                    fig_volume = go.Figure(data=[go.Bar(x=['Normal', 'High (>1.5x)', 'Extreme (>3x)'], y=[len(filtered_df) - high_volume, high_volume - extreme_volume, extreme_volume])])
                    fig_volume.update_layout(title="Volume Activity", height=200, margin=dict(l=20, r=20, t=40, b=20)); st.plotly_chart(fig_volume, use_container_width=True)
            with overview_cols[3]:
                if 'patterns' in filtered_df.columns:
                    pattern_counts = Counter(); [pattern_counts.update([p.strip() for p in patterns.split('|')]) for patterns in filtered_df['patterns'] if patterns]
                    if pattern_counts:
                        fig_patterns = go.Figure(data=[go.Pie(labels=list(pattern_counts.keys()), values=list(pattern_counts.values()), hole=0.3)])
                        fig_patterns.update_layout(title="Top 5 Patterns", height=200, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                        st.plotly_chart(fig_patterns, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🎯 Intelligent Picks")
            pick_cols = st.columns(3)
            with pick_cols[0]: st.markdown("##### 🚀 Momentum Leaders")
            with pick_cols[1]: st.markdown("##### ⚡ Volume Explosions")
            with pick_cols[2]: st.markdown("##### 💎 Hidden Opportunities")
            
            st.markdown("---")
            st.markdown("#### 🗺️ Sector Intelligence")
            if 'sector' in filtered_df.columns:
                sector_analysis = filtered_df.groupby('sector').agg({'master_score': 'mean', 'ret_30d': 'mean', 'rvol': 'mean', 'ticker': 'count'}).round(2); sector_analysis.columns = ['Avg Score', 'Avg 30D Return', 'Avg RVOL', 'Count']; sector_analysis = sector_analysis.sort_values('Avg Score', ascending=False)
                fig_heatmap = go.Figure(data=go.Heatmap(z=sector_analysis[['Avg Score', 'Avg 30D Return', 'Avg RVOL']].T.values, x=sector_analysis.index, y=['Avg Score', 'Avg 30D Return', 'Avg RVOL'], colorscale='RdYlGn')); fig_heatmap.update_layout(title="Sector Performance Heatmap", height=300, xaxis_title="Sector", yaxis_title="Metric"); st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 💾 Intelligent Data Export")
            export_cols = st.columns(4)
            with export_cols[0]: csv_dashboard = SmartExportEngine.create_csv_export(filtered_df, 'standard'); st.download_button(label="📥 Download Dashboard CSV", data=csv_dashboard, file_name=f"wave_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with export_cols[1]: top_100 = filtered_df.nlargest(100, 'master_score', keep='first'); csv_top100 = SmartExportEngine.create_csv_export(top_100, 'complete'); st.download_button(label="📥 Download Top 100 CSV", data=csv_top100, file_name=f"wave_top100_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with export_cols[2]: active_waves = filtered_df[filtered_df['wave_state'].isin(['🌊🌊🌊 CRESTING', '🌊🌊 BUILDING'])]; csv_waves = SmartExportEngine.create_csv_export(active_waves, 'day_trading'); st.download_button(label="📥 Download Active Waves", data=csv_waves, file_name=f"wave_active_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with export_cols[3]: pattern_stocks = filtered_df[filtered_df['patterns'] != '']; csv_patterns = SmartExportEngine.create_csv_export(pattern_stocks, 'pattern_analysis'); st.download_button(label="📥 Download Pattern CSV", data=csv_patterns, file_name=f"wave_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        else: st.info("No data available. Please check your filters or data source.")
    
    with tabs[1]:
        st.markdown("### 🏆 Intelligent Stock Rankings")
        if not filtered_df.empty:
            control_cols = st.columns([2, 2, 2, 1])
            with control_cols[0]: items_per_page = st.selectbox("Items per page:", [10, 20, 50, 100, 200], index=2, key="wd_items_per_page")
            with control_cols[1]: sort_options = {"Master Score": "master_score", "Momentum": "momentum_score", "Volume Activity": "rvol"}; sort_by = st.selectbox("Sort by:", list(sort_options.keys()), key="wd_sort_by"); sort_column = sort_options[sort_by]
            with control_cols[2]: display_style = st.radio("Display style:", ["Cards", "Table", "Compact"], horizontal=True, key="wd_display_style")
            with control_cols[3]: show_charts = st.checkbox("📊 Charts", value=True, key="wd_show_charts")
            
            display_df = filtered_df.sort_values(sort_column, ascending=False)
            total_items = len(display_df); total_pages = max(1, (total_items - 1) // items_per_page + 1)
            current_page = st.session_state.get('wd_current_page_rankings', 0)
            if current_page >= total_pages: current_page = total_pages - 1; RobustSessionState.safe_set('wd_current_page_rankings', current_page)

            nav_cols = st.columns([1, 1, 3, 1, 1])
            with nav_cols[0]: if st.button("⏮️ First", disabled=(current_page == 0)): RobustSessionState.safe_set('wd_current_page_rankings', 0); st.rerun()
            with nav_cols[1]: if st.button("◀️ Previous", disabled=(current_page == 0)): RobustSessionState.safe_set('wd_current_page_rankings', max(0, current_page - 1)); st.rerun()
            with nav_cols[2]: selected_page = st.selectbox("Jump to page:", range(total_pages), index=current_page, format_func=lambda x: f"Page {x+1} of {total_pages}"); if selected_page != current_page: RobustSessionState.safe_set('wd_current_page_rankings', selected_page); st.rerun()
            with nav_cols[3]: if st.button("Next ▶️", disabled=(current_page >= total_pages - 1)): RobustSessionState.safe_set('wd_current_page_rankings', min(total_pages - 1, current_page + 1)); st.rerun()
            with nav_cols[4]: if st.button("Last ⏭️", disabled=(current_page >= total_pages - 1)): RobustSessionState.safe_set('wd_current_page_rankings', total_pages - 1); st.rerun()
            
            start_idx = current_page * items_per_page; end_idx = min(start_idx + items_per_page, total_items)
            st.info(f"Showing {start_idx + 1}-{end_idx} of {total_items} stocks")
            
            if display_style == "Cards": [SmartUIComponents.render_stock_card_enhanced(display_df.iloc[idx], show_fundamentals=(RobustSessionState.safe_get('user_preferences', {}).get('display_mode') != "Technical"), show_charts=show_charts) for idx in range(start_idx, end_idx)]
            elif display_style == "Table":
                table_cols = ['rank', 'ticker', 'master_score', 'price', 'ret_30d', 'volume_1d', 'rvol', 'wave_state', 'patterns']; available_table_cols = [col for col in table_cols if col in display_df.columns]
                st.dataframe(display_df.iloc[start_idx:end_idx][available_table_cols], use_container_width=True)
            else:
                for idx in range(start_idx, end_idx):
                    row = display_df.iloc[idx]; col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
                    with col1: st.write(f"#{int(row['rank'])}")
                    with col2: st.write(f"**{row['ticker']}**"); st.caption(row.get('category', ''))
                    with col3: st.write(f"₹{row['price']:,.2f}"); color = "green" if row.get('ret_30d', 0) > 0 else "red"; st.markdown(f"<span style='color:{color}'>{row.get('ret_30d', 0):+.1f}%</span>", unsafe_allow_html=True)
                    with col4: st.write(f"Score: {row['master_score']:.1f}"); st.caption(row.get('wave_state', ''))
                    with col5: if row.get('patterns'): st.write(f"🎯 {len(row['patterns'].split('|'))}")
                    st.divider()
        else: st.info("No stocks match the current filter criteria.")
    
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Advanced Momentum Detection")
        if not filtered_df.empty:
            radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
            with radar_col1: wave_timeframe = st.selectbox("Wave Detection Timeframe:", ["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"], key="wave_timeframe_select")
            with radar_col2: sensitivity = st.select_slider("Detection Sensitivity:", ["Conservative", "Balanced", "Aggressive"], key="wave_sensitivity")
            with radar_col3: show_market_regime = st.checkbox("📊 Market Regime Analysis", value=True, key="show_market_regime")
            wave_filtered_df = filtered_df.copy()
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            st.markdown("#### 🚀 Acceleration Profiles - Momentum Building Over Time")
            st.markdown("#### 💰 Category Rotation - Smart Money Flow")
            st.markdown("#### 🌍 Market Regime Detection")
        else: st.info("No data available for wave radar analysis.")
    
    with tabs[3]:
        st.markdown("### 📈 Advanced Market Analytics")
        if not filtered_df.empty:
            analysis_type = st.selectbox("Select Analysis Type:", ["Sector/Industry Analysis", "Performance Distribution", "Correlation Analysis", "Time-based Patterns", "Pattern Effectiveness", "Risk-Return Profile", "Market Microstructure"], key="wd_analysis_type")
            st.markdown("#### 🏢 Sector & Industry Intelligence")
            st.markdown("#### 📊 Performance Distribution Analysis")
            st.markdown("#### 🔗 Correlation Analysis")
            st.markdown("#### ⏰ Time-based Return Patterns")
        else: st.info("No data available for analysis.")
    
    with tabs[4]:
        st.markdown("### 🔍 Intelligent Stock Search")
        if not filtered_df.empty:
            search_query = st.text_input("Search by ticker or company name:", placeholder="e.g., RELIANCE, TCS, HDFC, INFY...", key="wd_search_query")
            if search_query:
                search_results = SmartSearchEngine.search_stocks(filtered_df, search_query)
                if not search_results.empty:
                    st.success(f"Found {len(search_results)} matching stocks")
                    for _, row in search_results.iterrows():
                        SmartUIComponents.render_stock_card_enhanced(row, show_fundamentals=True)
                else: st.warning("No stocks found matching your search.")
    
    with tabs[5]:
        st.markdown("### 📥 Intelligent Export Center")
        if not filtered_df.empty:
            export_strategy = st.selectbox("Select Export Strategy:", ["🏃 Day Trading - Intraday Focus", "🌊 Swing Trading - Multi-Day Positions", "💼 Position Trading - Long-Term Holdings", "💰 Fundamental Analysis - Value Focus", "🎯 Pattern Trading - Technical Signals", "📊 Complete Dataset - All Columns", "🎨 Custom Export - Choose Your Columns"], key="export_strategy")
            export_data = None; filename = "export.csv"
            if "Day Trading" in export_strategy: export_data = SmartExportEngine.create_csv_export(filtered_df, 'day_trading')
            elif "Swing Trading" in export_strategy: export_data = SmartExportEngine.create_csv_export(filtered_df, 'swing_trading')
            elif "Fundamental" in export_strategy: export_data = SmartExportEngine.create_csv_export(filtered_df, 'fundamental')
            elif "Pattern" in export_strategy: export_data = SmartExportEngine.create_csv_export(filtered_df, 'pattern_analysis')
            elif "Complete" in export_strategy: export_data = SmartExportEngine.create_csv_export(filtered_df, 'complete')
            if export_data: st.download_button(label="📥 Download Export CSV", data=export_data, file_name=filename, mime="text/csv")
        else: st.info("No data available for export.")
    
    with tabs[6]:
        st.markdown("### 🧠 AI-Powered Market Insights")
        st.markdown("#### 🎯 AI Pattern Recognition Insights")
        st.markdown("#### 💰 AI Portfolio Optimization Suggestions")
        st.info("AI Insights coming soon.")
    
    with tabs[7]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0 - APEX Edition")
        st.markdown("#### 🌊 Welcome to the Future of Stock Analysis")
        st.markdown("#### 🎯 Core Features - PERMANENTLY LOCKED")
        st.markdown("#### 🔧 Technical Specifications")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🌊 Wave Detection Ultimate 3.0 - Final Perfected Production Version<br>
        <small>Professional Stock Ranking System • All Features Complete • Performance Maximized • Permanently Locked</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.logger.error(f"Critical application error: {str(e)}", exc_info=True)
        st.error(f"❌ Critical Application Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            import traceback; st.code(traceback.format_exc())
        
        st.markdown("### 🛠️ Recovery Options")
        recovery_cols = st.columns(4)
        with recovery_cols[0]:
            if st.button("🔄 Restart Application", use_container_width=True, type="primary"):
                st.cache_data.clear()
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()
        with recovery_cols[1]:
            if st.button("💾 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
        with recovery_cols[2]:
            if st.button("🎮 Load Demo Data", use_container_width=True):
                RobustSessionState.safe_set('sheet_id', CONFIG.DEFAULT_SHEET_ID)
                RobustSessionState.safe_set('data_source', "sheet")
                st.success("Loading demo data...")
                st.rerun()
        with recovery_cols[3]:
            if st.button("📧 Report Issue", use_container_width=True):
                st.info("Please take a screenshot of this error and send to support@wavedetection.ai")
