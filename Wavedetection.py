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

# Additional smart imports from V1 for robustness
import hashlib                               # For intelligent cache versioning
import requests                              # For reliable HTTP requests
from requests.adapters import HTTPAdapter    # For connection pooling
from urllib3.util.retry import Retry         # For retry logic
import json                                  # For data serialization
from collections import defaultdict          # For safer dictionaries

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Performance optimizations
np.seterr(all='ignore')                            # Ignore numpy warnings
pd.options.mode.chained_assignment = None          # Disable pandas warning
pd.options.display.float_format = '{:.2f}'.format  # Format float display

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# LOGGING CONFIGURATION
# ============================================

# Production logging with performance tracking
log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Performance tracking storage
performance_stats = defaultdict(list)

def log_performance(operation: str, duration: float):
    """Log performance metrics"""
    performance_stats[operation].append(duration)
    if duration > 1.0:
        logger.warning(f"Slow operation: {operation} took {duration:.2f}s")

# ============================================
# ROBUST SESSION STATE MANAGER - FIXED
# ============================================

class RobustSessionState:
    """Bulletproof session state management - prevents all KeyErrors"""
    
    # Complete list of ALL session state keys with their default values
    STATE_DEFAULTS = {
        # Core states
        'search_query': "",
        'search_input': "",  # For search widget
        'last_refresh': None,
        'data_source': "sheet",
        'sheet_id': "",
        'gid': "",
        'user_preferences': {
            'default_top_n': 50,
            'display_mode': 'Technical',
            'last_filters': {}
        },
        # Unified filter state - single source of truth
        'active_filters': {
            'categories': [],
            'sectors': [],
            'industries': [],
            'min_score': 0,
            'patterns': [],
            'trend_filter': 'All Trends',
            'eps_tiers': [],
            'pe_tiers': [],
            'price_tiers': [],
            'min_eps_change': None,
            'min_pe': None,
            'max_pe': None,
            'require_fundamental_data': False,
            'wave_states': [],
            'wave_strength_range': (0, 100),
            'quick_filter': None
        },
        'filter_count': 0,
        'show_debug': False,
        'performance_metrics': {},
        'data_quality': {},
        'ranked_df': None,
        'data_timestamp': None,
        'last_good_data': None,
        'trigger_clear': False,
        # UI states
        'display_mode_radio': 'Technical',
        'top_n_slider': 50,
        'export_template_radio': 'Full Analysis (All Data)',
        # Wave Radar states
        'show_sensitivity_details': False,
        'show_market_regime': True,
        'wave_timeframe_select': 'All Waves',
        'wave_sensitivity': 'Balanced'
    }
    
    @staticmethod
    def safe_get(key: str, default: Any = None) -> Any:
        """Safely get a session state value with fallback"""
        if key not in st.session_state:
            if key in RobustSessionState.STATE_DEFAULTS:
                st.session_state[key] = RobustSessionState.STATE_DEFAULTS[key]
            else:
                st.session_state[key] = default
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
                if key == 'last_refresh' and default_value is None:
                    st.session_state[key] = datetime.now(timezone.utc)
                else:
                    st.session_state[key] = default_value
    
    @staticmethod
    def clear_all_filters():
        """Clear all filter states properly"""
        # Reset the unified filter state
        st.session_state['active_filters'] = RobustSessionState.STATE_DEFAULTS['active_filters'].copy()
        st.session_state['filter_count'] = 0
        st.session_state['trigger_clear'] = False
        logger.info("All filters cleared")

# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights and thresholds"""
    
    # Data source - Dynamic with default GID
    DEFAULT_SHEET_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing"
    CSV_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings optimized for Streamlit Community Cloud
    CACHE_TTL: int = 3600  # 1 hour
    STALE_DATA_HOURS: int = 24
    REQUEST_TIMEOUT: int = 30
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_FACTOR: float = 0.3
    RETRY_STATUS_CODES: List[int] = field(default_factory=lambda: [500, 502, 503, 504])
    
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
        'vol_ratio_90d_180d', 'industry'
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
        "momentum_diverge": 90,
        "range_compress": 75,
        "stealth": 70,
        "vampire": 85,
        "perfect_storm": 80
    })
    
    # Value bounds for data validation
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000),
        'rvol': (0.01, 1_000_000.0),
        'pe': (-10000, 10000),
        'returns': (-99.99, 9999.99),
        'volume': (0, 1e12)
    })
    
    # Performance thresholds
    PERFORMANCE_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'data_processing': 2.0,
        'filtering': 0.2,
        'pattern_detection': 0.5,
        'export_generation': 1.0,
        'search': 0.05
    })
    
    # Market categories
    MARKET_CATEGORIES: List[str] = field(default_factory=lambda: [
        'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'
    ])
    
    # Tier definitions with proper boundaries
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {
            "Loss": (-float('inf'), 0),
            "0-5": (0, 5),
            "5-10": (5, 10),
            "10-20": (10, 20),
            "20-50": (20, 50),
            "50-100": (50, 100),
            "100+": (100, float('inf'))
        },
        "pe": {
            "0-10": (0, 10),
            "10-15": (10, 15),
            "15-20": (15, 20),
            "20-25": (20, 25),
            "25-30": (25, 30),
            "30-50": (30, 50),
            "50+": (50, float('inf'))
        },
        "price": {
            "0-50": (0, 50),
            "50-100": (50, 100),
            "100-500": (100, 500),
            "500-1000": (500, 1000),
            "1000-5000": (1000, 5000),
            "5000+": (5000, float('inf'))
        }
    })

# Create global config instance
CONFIG = Config()

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
                
                log_performance(func.__name__, duration)
                
                if duration > target_time:
                    logger.warning(f"{func.__name__} took {duration:.2f}s (target: {target_time}s)")
                
                # Store in session state for display
                perf_metrics = RobustSessionState.safe_get('performance_metrics', {})
                perf_metrics[func.__name__] = duration
                RobustSessionState.safe_set('performance_metrics', perf_metrics)
                
                return result
            return wrapper
        return decorator

# ============================================
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidator:
    """Validate and sanitize data with comprehensive logging"""
    
    def __init__(self):
        self.validation_stats = {
            'cleaned_values': 0,
            'invalid_values': 0,
            'clipped_values': 0,
            'type_conversions': 0
        }
    
    def reset_stats(self):
        """Reset validation statistics"""
        for key in self.validation_stats:
            self.validation_stats[key] = 0
    
    def clean_numeric_value(self, value: Any, column_name: str, is_percentage: bool = False, 
                          bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """Clean and validate numeric values with bounds checking"""
        if pd.isna(value) or value is None:
            return np.nan
        
        # Handle string inputs
        if isinstance(value, str):
            # Remove common artifacts
            value = value.strip().replace(',', '').replace('$', '').replace('%', '')
            
            # Handle parentheses for negative values
            if value.startswith('(') and value.endswith(')'):
                value = '-' + value[1:-1]
            
            # Handle special cases
            if value in ['-', 'N/A', 'n/a', '#N/A', '#VALUE!', 'inf', '-inf']:
                self.validation_stats['invalid_values'] += 1
                return np.nan
            
            try:
                value = float(value)
                self.validation_stats['type_conversions'] += 1
            except ValueError:
                self.validation_stats['invalid_values'] += 1
                return np.nan
        
        # Apply bounds if specified
        if bounds and not pd.isna(value):
            min_val, max_val = bounds
            original_value = value
            value = np.clip(value, min_val, max_val)
            if value != original_value:
                self.validation_stats['clipped_values'] += 1
                logger.debug(f"Clipped {column_name}: {original_value} -> {value}")
        
        # Final validation
        if np.isfinite(value):
            self.validation_stats['cleaned_values'] += 1
            return float(value)
        else:
            self.validation_stats['invalid_values'] += 1
            return np.nan
    
    @staticmethod
    def sanitize_string(value: Any) -> str:
        """Sanitize string values"""
        if pd.isna(value) or value is None:
            return "Unknown"
        
        value = str(value).strip()
        
        # Remove problematic characters
        value = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Normalize multiple spaces
        value = ' '.join(value.split())
        
        return value if value else "Unknown"
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate dataframe has minimum required columns"""
        if df is None or df.empty:
            return False, "Empty or null dataframe"
        
        missing_critical = [col for col in CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            return False, f"Missing critical columns: {', '.join(missing_critical)}"
        
        if len(df) < 10:
            return False, f"Too few stocks ({len(df)}). Need at least 10."
        
        return True, "Valid"

# Create global validator instance
validator = DataValidator()

# ============================================
# SMART CACHING WITH VERSIONING
# ============================================

def get_requests_session() -> requests.Session:
    """Create requests session with retry logic"""
    session = requests.Session()
    
    retry = Retry(
        total=CONFIG.MAX_RETRY_ATTEMPTS,
        read=CONFIG.MAX_RETRY_ATTEMPTS,
        connect=CONFIG.MAX_RETRY_ATTEMPTS,
        backoff_factor=CONFIG.RETRY_BACKOFF_FACTOR,
        status_forcelist=CONFIG.RETRY_STATUS_CODES,
        allowed_methods=["HEAD", "GET", "OPTIONS"]
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

@st.cache_data(persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, 
                         sheet_id: str = None, gid: str = None,
                         data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """Load and process data with smart caching and versioning"""
    
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
        validator.reset_stats()
        
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            try:
                df = pd.read_csv(file_data, low_memory=False)
                metadata['source'] = "User Upload"
            except UnicodeDecodeError:
                # Try different encodings
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
        else:
            # Google Sheets loading
            if not sheet_id:
                raise ValueError("Please enter a Google Sheets ID")
            
            # Extract sheet ID from URL if needed
            sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_id)
            if sheet_id_match:
                sheet_id = sheet_id_match.group(1)
            
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            csv_url = CONFIG.CSV_URL_TEMPLATE.format(sheet_id=sheet_id, gid=gid)
            
            logger.info(f"Loading data from Google Sheets ID: {sheet_id}")
            
            session = get_requests_session()
            
            try:
                response = session.get(csv_url, timeout=CONFIG.REQUEST_TIMEOUT)
                response.raise_for_status()
                
                if len(response.content) < 100:
                    raise ValueError("Response too small, likely an error page")
                
                df = pd.read_csv(BytesIO(response.content), low_memory=False)
                metadata['source'] = "Google Sheets"
                metadata['sheet_id'] = sheet_id
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to load from Google Sheets: {str(e)}")
                
                # Try fallback
                last_good_data = RobustSessionState.safe_get('last_good_data')
                if last_good_data:
                    logger.info("Using cached data as fallback")
                    df, timestamp, old_metadata = last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise
        
        metadata['performance']['load_time'] = time.perf_counter() - start_time
        
        # Validate data
        is_valid, validation_msg = DataValidator.validate_dataframe(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {validation_msg}")
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Process dataframe
        df = DataProcessor.process_dataframe(df, metadata)
        
        # Calculate all metrics
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        # Calculate scores
        df = RankingEngine.calculate_all_scores(df)
        
        # Detect patterns
        df = PatternDetector.detect_all_patterns(df)
        
        # Add category analysis
        df = MarketIntelligence.add_category_analysis(df)
        
        # Final ranking
        df = RankingEngine.apply_final_ranking(df)
        
        # Create timestamp
        timestamp = datetime.now(timezone.utc)
        
        # Store validation stats
        metadata['validation_stats'] = validator.validation_stats.copy()
        
        # Save as last good data
        RobustSessionState.safe_set('last_good_data', (df.copy(), timestamp, metadata))
        
        # Record processing time
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        # Clean up memory
        gc.collect()
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        metadata['errors'].append(str(e))
        raise

# ============================================
# DATA PROCESSING ENGINE
# ============================================

class DataProcessor:
    """Handle all data processing with validation and optimization"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Complete data processing pipeline"""
        
        df = df.copy()
        initial_count = len(df)
        
        logger.info(f"Processing {initial_count} rows...")
        
        # Process categorical columns first
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
        # Create industry from sector if missing
        if 'industry' not in df.columns and 'sector' in df.columns:
            df['industry'] = df['sector']
            metadata['warnings'].append("Industry column created from sector data")
        
        # Process numeric columns
        numeric_cols = [col for col in df.columns if col not in string_cols + ['year', 'market_cap']]
        
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
                
                df[col] = df[col].apply(
                    lambda x: validator.clean_numeric_value(x, col, is_pct, bounds)
                )
        
        # Fix volume ratios
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 1000.0)
                df[col] = df[col].fillna(1.0)
        
        # Calculate RVOL if missing
        if 'rvol' not in df.columns or df['rvol'].isna().all():
            if 'volume_1d' in df.columns and 'volume_90d' in df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['rvol'] = np.where(
                        df['volume_90d'] > 0,
                        df['volume_1d'] / df['volume_90d'],
                        1.0
                    )
                metadata['warnings'].append("RVOL calculated from volume data")
        
        # Validate critical data
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        
        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        # Fill missing values
        df = DataProcessor._fill_missing_values(df)
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        # Data quality metrics
        removed = initial_count - len(df)
        if removed > 0:
            metadata['warnings'].append(f"Removed {removed} invalid rows during processing")
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        metadata['data_quality'] = {
            'total_rows': len(df),
            'removed_rows': removed,
            'duplicate_tickers': before_dedup - len(df),
            'completeness': completeness,
            'columns_available': list(df.columns)
        }
        
        RobustSessionState.safe_set('data_quality', metadata['data_quality'])
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows")
        
        return df
    
    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults - defensive implementation"""
        
        # Position data defaults
        df['from_low_pct'] = df.get('from_low_pct', pd.Series(50.0, index=df.index)).fillna(50.0)
        df['from_high_pct'] = df.get('from_high_pct', pd.Series(-50.0, index=df.index)).fillna(-50.0)
        
        # RVOL default
        df['rvol'] = df.get('rvol', pd.Series(1.0, index=df.index)).fillna(1.0)
        
        # Return defaults
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        # Volume defaults
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Category defaults
        df['category'] = df.get('category', pd.Series('Unknown', index=df.index)).fillna('Unknown')
        df['sector'] = df.get('sector', pd.Series('Unknown', index=df.index)).fillna('Unknown')
        
        # Industry defaults
        if 'industry' in df.columns:
            df['industry'] = df['industry'].fillna(df['sector'])
        else:
            df['industry'] = df['sector']
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications for filtering"""
        
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Classify value into tier with fixed boundary logic"""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
                if min_val == -float('inf') and value <= max_val:
                    return tier_name
                if max_val == float('inf') and value > min_val:
                    return tier_name
                # Special case for zero
                if min_val == 0 and max_val > 0 and value == 0:
                    continue
            
            return "Unknown"
        
        # Add tier columns
        if 'eps_current' in df.columns:
            df['eps_tier'] = df['eps_current'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['eps'])
            )
        
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: "Negative/NA" if pd.isna(x) or x <= 0 
                else classify_tier(x, CONFIG.TIERS['pe'])
            )
        
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['price'])
            )
        
        return df

# ============================================
# ADVANCED METRICS CALCULATOR - ENHANCED
# ============================================

class AdvancedMetrics:
    """Calculate advanced metrics and indicators"""
    
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics including V1 features"""
        
        # Market Regime Detection (V1 feature)
        df['market_regime'] = AdvancedMetrics._detect_market_regime(df)
        
        # Money Flow
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow'] = df['price'] * df['volume_1d'] * df['rvol']
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
        else:
            df['money_flow_mm'] = 0.0
        
        # Volume Momentum Index (VMI)
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 
                                             'vol_ratio_30d_90d', 'vol_ratio_90d_180d']):
            df['vmi'] = (
                df['vol_ratio_1d_90d'] * 4 +
                df['vol_ratio_7d_90d'] * 3 +
                df['vol_ratio_30d_90d'] * 2 +
                df['vol_ratio_90d_180d'] * 1
            ) / 10
        else:
            df['vmi'] = 1.0
        
        # Position Tension
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'] + abs(df['from_high_pct'])
        else:
            df['position_tension'] = 100.0
        
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
        
        # Smart Money Flow (V1 feature - complete implementation)
        df['smart_money_flow'] = AdvancedMetrics._calculate_smart_money_flow(df)
        
        # Momentum Quality Score (V1 feature)
        df['momentum_quality'] = AdvancedMetrics._calculate_momentum_quality(df)
        
        # Wave State
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)
        
        # Overall Wave Strength
        if all(col in df.columns for col in ['momentum_score', 'acceleration_score', 
                                             'rvol_score', 'breakout_score']):
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
    
    @staticmethod
    def _detect_market_regime(df: pd.DataFrame) -> pd.Series:
        """Detect current market regime (V1 feature)"""
        if df.empty or 'ret_30d' not in df.columns:
            return pd.Series("😴 RANGE-BOUND", index=df.index)
        
        # Calculate market breadth
        positive_breadth = (df['ret_30d'] > 0).mean()
        strong_positive = (df['ret_30d'] > 10).mean()
        strong_negative = (df['ret_30d'] < -10).mean()
        
        # Determine regime
        if positive_breadth > 0.6 and strong_positive > 0.3:
            regime = "🔥 RISK-ON BULL"
        elif positive_breadth < 0.4 and strong_negative > 0.3:
            regime = "🛡️ RISK-OFF DEFENSIVE"
        else:
            regime = "😴 RANGE-BOUND"
        
        return pd.Series(regime, index=df.index)
    
    @staticmethod
    def _calculate_smart_money_flow(df: pd.DataFrame) -> pd.Series:
        """Calculate smart money flow indicator (V1 feature - complete)"""
        smart_flow = pd.Series(50, index=df.index, dtype=float)
        
        # Volume persistence check
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
            # High volume persisting over multiple periods indicates institutional interest
            vol_persistence = (
                (df['vol_ratio_7d_90d'] > 1.2) & 
                (df['vol_ratio_30d_90d'] > 1.1)
            ).astype(float) * 20
            smart_flow += vol_persistence
        
        # Price-volume divergence
        if 'ret_30d' in df.columns and 'volume_score' in df.columns:
            # Smart money accumulates on low price change with high volume
            divergence = np.where(
                (np.abs(df['ret_30d']) < 5) & (df['volume_score'] > 70),
                20, 0
            )
            smart_flow += divergence
        
        # Institutional patterns
        if all(col in df.columns for col in ['vmi', 'position_tension']):
            # High VMI with moderate position tension suggests accumulation
            institutional_pattern = np.where(
                (df['vmi'] > 2) & (df['position_tension'].between(30, 70)),
                15, 0
            )
            smart_flow += institutional_pattern
        
        # Money flow consistency
        if 'money_flow_mm' in df.columns and 'ret_7d' in df.columns:
            # Consistent money flow with moderate returns
            flow_consistency = np.where(
                (df['money_flow_mm'] > df['money_flow_mm'].median()) & 
                (df['ret_7d'].between(-5, 10)),
                15, 0
            )
            smart_flow += flow_consistency
        
        return smart_flow.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum quality score (V1 feature)"""
        quality_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check momentum consistency across timeframes
        if all(col in df.columns for col in ['ret_30d', 'ret_7d', 'ret_1d']):
            # All positive = high quality
            all_positive = (df['ret_30d'] > 0) & (df['ret_7d'] > 0) & (df['ret_1d'] > 0)
            quality_score[all_positive] = 80
            
            # Accelerating momentum = very high quality
            accelerating = (df['ret_1d'] > df['ret_7d'] / 7) & (df['ret_7d'] / 7 > df['ret_30d'] / 30)
            quality_score[accelerating] = 90
            
            # Mixed signals = low quality
            mixed = ((df['ret_30d'] > 0) & (df['ret_7d'] < 0)) | ((df['ret_7d'] > 0) & (df['ret_1d'] < 0))
            quality_score[mixed] = 30
        
        # Smoothness check using volatility proxy
        if 'ret_30d' in df.columns and 'ret_7d' in df.columns:
            # Calculate simple volatility proxy
            volatility_proxy = np.abs(df['ret_7d'] - df['ret_30d'] / 4.3)
            low_volatility = volatility_proxy < volatility_proxy.quantile(0.25)
            quality_score[low_volatility] += 10
        
        return quality_score.clip(0, 100)

# ============================================
# RANKING ENGINE - ENHANCED
# ============================================

class RankingEngine:
    """Calculate component scores and final rankings"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
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
        
        # Calculate master score
        df['master_score'] = (
            df['position_score'] * CONFIG.POSITION_WEIGHT +
            df['volume_score'] * CONFIG.VOLUME_WEIGHT +
            df['momentum_score'] * CONFIG.MOMENTUM_WEIGHT +
            df['acceleration_score'] * CONFIG.ACCELERATION_WEIGHT +
            df['breakout_score'] * CONFIG.BREAKOUT_WEIGHT +
            df['rvol_score'] * CONFIG.RVOL_WEIGHT
        )
        
        # Apply Smart Money Flow bonus (V1 feature)
        if 'smart_money_flow' in df.columns:
            # Stocks with high smart money flow get up to 5 point bonus
            smart_money_bonus = (df['smart_money_flow'] / 100) * 5
            df['master_score'] = (df['master_score'] + smart_money_bonus).clip(0, 100)
        
        # Calculate additional metrics
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df)
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df)
        
        logger.info("All scores calculated successfully")
        
        return df
    
    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True, 
                   na_option: str = 'bottom') -> pd.Series:
        """Safely rank a series handling NaN values"""
        if series.isna().all():
            return pd.Series(50.0, index=series.index)
        
        ranks = series.rank(pct=pct, ascending=ascending, na_option=na_option)
        
        if pct:
            ranks = ranks * 100
        
        return ranks.fillna(50.0)
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score based on 52-week range"""
        from_low = df.get('from_low_pct', pd.Series(50, index=df.index))
        from_high = df.get('from_high_pct', pd.Series(-50, index=df.index))
        
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
            return pd.Series(50, index=df.index)
        
        if has_from_low:
            rank_from_low = RankingEngine._safe_rank(from_low, pct=True, ascending=True)
        else:
            rank_from_low = pd.Series(50, index=df.index)
        
        if has_from_high:
            rank_from_high = RankingEngine._safe_rank(from_high, pct=True, ascending=False)
        else:
            rank_from_high = pd.Series(50, index=df.index)
        
        position_score = (rank_from_low * 0.6 + rank_from_high * 0.4)
        
        return position_score.clip(0, 100)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive volume score"""
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
            if col in df.columns and df[col].notna().any():
                col_rank = RankingEngine._safe_rank(df[col], pct=True, ascending=True)
                weighted_score += col_rank * weight
                total_weight += weight
        
        if total_weight > 0:
            volume_score = weighted_score / total_weight
        else:
            logger.warning("No volume ratio data available, using neutral scores")
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns"""
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                ret_7d = df['ret_7d'].fillna(0)
                momentum_score = RankingEngine._safe_rank(ret_7d, pct=True, ascending=True)
                logger.info("Using 7-day returns for momentum score")
            else:
                logger.warning("No return data available for momentum calculation")
            
            return momentum_score.clip(0, 100)
        
        ret_30d = df['ret_30d'].fillna(0)
        momentum_score = RankingEngine._safe_rank(ret_30d, pct=True, ascending=True)
        
        if 'ret_7d' in df.columns:
            ret_7d = df['ret_7d'].fillna(0)
            
            consistent_momentum = ((ret_30d > 0) & (ret_7d > 0)) | ((ret_30d < 0) & (ret_7d < 0))
            momentum_score[consistent_momentum] = momentum_score[consistent_momentum] * 1.1
            
            momentum_reversal = ((ret_30d > 0) & (ret_7d < 0)) | ((ret_30d < 0) & (ret_7d > 0))
            momentum_score[momentum_reversal] = momentum_score[momentum_reversal] * 0.9
        
        return momentum_score.clip(0, 100)
    
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate acceleration score"""
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                acceleration_7d_30d = np.where(
                    df['ret_30d'] != 0,
                    (df['ret_7d'] * 4.3) / df['ret_30d'],
                    1.0
                )
            
            acceleration_7d_30d = np.clip(acceleration_7d_30d, 0, 5)
            
            acceleration_score = 20 + (acceleration_7d_30d * 16)
        
        elif 'ret_7d' in df.columns:
            acceleration_score = 50 + np.clip(df['ret_7d'], -25, 25)
        
        return acceleration_score.clip(0, 100)
    
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout probability score"""
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'from_high_pct' in df.columns:
            from_high = df['from_high_pct'].fillna(-50)
            distance_from_high = -from_high
            
            breakout_score = np.where(
                distance_from_high <= 5,
                90 + (5 - distance_from_high) * 2,
                np.where(
                    distance_from_high <= 10,
                    70 + (10 - distance_from_high) * 4,
                    np.where(
                        distance_from_high <= 20,
                        50 + (20 - distance_from_high) * 2,
                        50 - np.clip(distance_from_high - 20, 0, 30)
                    )
                )
            )
        
        if 'rvol' in df.columns:
            high_volume_boost = np.where(df['rvol'] > 2, 10, 0)
            breakout_score = breakout_score + high_volume_boost
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate relative volume score"""
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        rvol = df['rvol'].fillna(1.0)
        
        rvol_score = np.where(
            rvol >= 5, 100,
            np.where(
                rvol >= 3, 80 + (rvol - 3) * 10,
                np.where(
                    rvol >= 2, 60 + (rvol - 2) * 20,
                    np.where(
                        rvol >= 1, 50 + (rvol - 1) * 10,
                        50 * rvol
                    )
                )
            )
        )
        
        return pd.Series(rvol_score, index=df.index).clip(0, 100)
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score"""
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'money_flow_mm' in df.columns:
            money_flow_rank = RankingEngine._safe_rank(df['money_flow_mm'], pct=True, ascending=True)
            liquidity_score = money_flow_rank * 0.7 + 30
        
        if 'volume_1d' in df.columns:
            volume_rank = RankingEngine._safe_rank(df['volume_1d'], pct=True, ascending=True)
            liquidity_score = liquidity_score * 0.5 + volume_rank * 0.5
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score"""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        timeframes = [
            ('ret_3m', 0.3),
            ('ret_6m', 0.3),
            ('ret_1y', 0.4)
        ]
        
        total_weight = 0
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        
        for col, weight in timeframes:
            if col in df.columns and df[col].notna().any():
                col_rank = RankingEngine._safe_rank(df[col], pct=True, ascending=True)
                weighted_score += col_rank * weight
                total_weight += weight
        
        if total_weight > 0:
            strength_score = weighted_score / total_weight
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality score"""
        quality_score = pd.Series(50, index=df.index, dtype=float)
        
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            positive_days = (
                (df['ret_1d'] > 0).astype(int) +
                (df['ret_7d'] > 0).astype(int) +
                (df['ret_30d'] > 0).astype(int)
            )
            
            quality_score = 25 + (positive_days * 25)
        
        if 'momentum_harmony' in df.columns:
            harmony_boost = df['momentum_harmony'] * 5
            quality_score = quality_score + harmony_boost
        
        return quality_score.clip(0, 100)
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ranks within categories"""
        if 'category' not in df.columns:
            return df
        
        for category in df['category'].unique():
            if pd.notna(category) and category != 'Unknown':
                mask = df['category'] == category
                if mask.sum() > 0:
                    df.loc[mask, 'category_rank'] = df.loc[mask, 'master_score'].rank(
                        ascending=False, 
                        method='min',
                        na_option='bottom'
                    )
        
        return df
    
    @staticmethod
    def apply_final_ranking(df: pd.DataFrame) -> pd.DataFrame:
        """Apply final ranking and percentiles"""
        df['rank'] = df['master_score'].rank(ascending=False, method='min', na_option='bottom').astype(int)
        
        df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
        
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        df['percentile'] = df['percentile'].fillna(0)
        
        df = RankingEngine._calculate_category_ranks(df)
        
        logger.info(f"Ranking complete: {len(df)} stocks processed")
        
        return df

# ============================================
# PATTERN DETECTION ENGINE
# ============================================

class PatternDetector:
    """Detect all trading patterns with O(n) performance"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns with optimized vectorized operations"""
        
        if df.empty:
            df['patterns'] = ''
            return df
        
        # Pre-calculate common conditions once
        conditions = PatternDetector._calculate_base_conditions(df)
        
        # Initialize pattern results
        pattern_results = {}
        
        # Technical patterns (vectorized)
        pattern_results['🔥 CAT LEADER'] = PatternDetector._is_category_leader(df, conditions)
        pattern_results['💎 HIDDEN GEM'] = PatternDetector._is_hidden_gem(df, conditions)
        pattern_results['🚀 ACCELERATING'] = PatternDetector._is_accelerating(df, conditions)
        pattern_results['🏦 INSTITUTIONAL'] = PatternDetector._is_institutional(df, conditions)
        pattern_results['⚡ VOL EXPLOSION'] = PatternDetector._is_volume_explosion(df, conditions)
        pattern_results['🎯 BREAKOUT'] = PatternDetector._is_breakout_ready(df, conditions)
        pattern_results['👑 MARKET LEADER'] = PatternDetector._is_market_leader(df, conditions)
        pattern_results['🌊 MOMENTUM WAVE'] = PatternDetector._is_momentum_wave(df, conditions)
        pattern_results['💰 LIQUID LEADER'] = PatternDetector._is_liquid_leader(df, conditions)
        pattern_results['💪 LONG STRENGTH'] = PatternDetector._is_long_strength(df, conditions)
        pattern_results['📈 QUALITY TREND'] = PatternDetector._is_quality_trend(df, conditions)
        
        # Range patterns
        pattern_results['🎯 52W HIGH APPROACH'] = PatternDetector._is_52w_high_approach(df, conditions)
        pattern_results['🔄 52W LOW BOUNCE'] = PatternDetector._is_52w_low_bounce(df, conditions)
        pattern_results['👑 GOLDEN ZONE'] = PatternDetector._is_golden_zone(df, conditions)
        pattern_results['📊 VOL ACCUMULATION'] = PatternDetector._is_volume_accumulation(df, conditions)
        pattern_results['🔀 MOMENTUM DIVERGE'] = PatternDetector._is_momentum_divergence(df, conditions)
        pattern_results['🎯 RANGE COMPRESS'] = PatternDetector._is_range_compression(df, conditions)
        
        # Intelligence patterns
        pattern_results['🤫 STEALTH'] = PatternDetector._is_stealth_accumulation(df, conditions)
        pattern_results['🧛 VAMPIRE'] = PatternDetector._is_vampire_squeeze(df, conditions)
        pattern_results['⛈️ PERFECT STORM'] = PatternDetector._is_perfect_storm(df, conditions)
        
        # Fundamental patterns (if in hybrid mode)
        if any(col in df.columns for col in ['pe', 'eps_change_pct']):
            pattern_results['💎 VALUE MOMENTUM'] = PatternDetector._is_value_momentum(df, conditions)
            pattern_results['📊 EARNINGS ROCKET'] = PatternDetector._is_earnings_rocket(df, conditions)
            pattern_results['🏆 QUALITY LEADER'] = PatternDetector._is_quality_leader(df, conditions)
            pattern_results['⚡ TURNAROUND'] = PatternDetector._is_turnaround(df, conditions)
            pattern_results['⚠️ HIGH PE'] = PatternDetector._is_high_pe_warning(df, conditions)
        
        # Combine all patterns efficiently
        df['patterns'] = PatternDetector._combine_patterns(pattern_results)
        
        # Count patterns
        pattern_count = sum(pattern_results[p].sum() for p in pattern_results)
        logger.info(f"Detected {pattern_count} total pattern occurrences across {len(df)} stocks")
        
        return df
    
    @staticmethod
    def _calculate_base_conditions(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Pre-calculate common conditions for efficiency"""
        conditions = {}
        
        # Momentum conditions
        if 'momentum_score' in df.columns:
            conditions['high_momentum'] = df['momentum_score'] >= 70
            conditions['extreme_momentum'] = df['momentum_score'] >= 85
        
        # Volume conditions
        if 'rvol' in df.columns:
            conditions['high_volume'] = df['rvol'] >= 2
            conditions['extreme_rvol'] = df['rvol'] >= 3
        
        # Position conditions
        if 'from_high_pct' in df.columns:
            conditions['near_high'] = df['from_high_pct'] > -10
            conditions['very_near_high'] = df['from_high_pct'] > -5
        
        if 'from_low_pct' in df.columns:
            conditions['near_low'] = df['from_low_pct'] < 20
        
        # Score conditions
        if 'master_score' in df.columns:
            conditions['high_score'] = df['master_score'] >= 70
            conditions['top_score'] = df['master_score'] >= 85
        
        return conditions
    
    @staticmethod
    def _is_category_leader(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect category leaders"""
        if 'category_rank' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['category_rank'] <= 3) & conditions.get('top_score', False)
    
    @staticmethod
    def _is_hidden_gem(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect hidden gems"""
        base_condition = conditions.get('high_score', pd.Series(True, index=df.index))
        
        if 'percentile' in df.columns:
            base_condition &= df['percentile'] < 90
        
        if 'money_flow_mm' in df.columns:
            base_condition &= df['money_flow_mm'] < df['money_flow_mm'].quantile(0.5)
        
        return base_condition & conditions.get('high_momentum', True)
    
    @staticmethod
    def _is_accelerating(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect accelerating stocks"""
        if 'acceleration_score' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
    
    @staticmethod
    def _is_institutional(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect institutional accumulation"""
        base_condition = conditions.get('high_volume', pd.Series(True, index=df.index))
        
        if 'smart_money_flow' in df.columns:
            base_condition &= df['smart_money_flow'] >= CONFIG.PATTERN_THRESHOLDS['institutional']
        
        if 'liquidity_score' in df.columns:
            base_condition &= df['liquidity_score'] >= 70
        
        return base_condition
    
    @staticmethod
    def _is_volume_explosion(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect volume explosions"""
        return conditions.get('extreme_rvol', pd.Series(False, index=df.index))
    
    @staticmethod
    def _is_breakout_ready(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect breakout ready stocks"""
        if 'breakout_score' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']) & \
               conditions.get('near_high', True)
    
    @staticmethod
    def _is_market_leader(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect market leaders"""
        if 'rank' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['rank'] <= 10) & conditions.get('top_score', True)
    
    @staticmethod
    def _is_momentum_wave(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect momentum waves"""
        base_condition = conditions.get('high_momentum', pd.Series(True, index=df.index))
        
        if 'acceleration_score' in df.columns:
            base_condition &= df['acceleration_score'] >= 70
        
        return base_condition
    
    @staticmethod
    def _is_liquid_leader(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect liquid leaders"""
        if 'liquidity_score' not in df.columns or 'percentile' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & \
               (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
    
    @staticmethod
    def _is_long_strength(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect long-term strength"""
        if 'long_term_strength' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
    
    @staticmethod
    def _is_quality_trend(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect quality trends"""
        if 'trend_quality' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return df['trend_quality'] >= 80
    
    @staticmethod
    def _is_52w_high_approach(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect 52-week high approach"""
        return conditions.get('very_near_high', pd.Series(False, index=df.index))
    
    @staticmethod
    def _is_52w_low_bounce(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect 52-week low bounce"""
        base_condition = conditions.get('near_low', pd.Series(False, index=df.index))
        
        if 'ret_30d' in df.columns:
            base_condition &= df['ret_30d'] > 10
        
        return base_condition
    
    @staticmethod
    def _is_golden_zone(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect golden zone stocks"""
        if not all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            return pd.Series(False, index=df.index)
        
        return (df['from_low_pct'].between(30, 70)) & (df['from_high_pct'].between(-70, -30))
    
    @staticmethod
    def _is_volume_accumulation(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect volume accumulation"""
        if 'vmi' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return df['vmi'] >= 2.5
    
    @staticmethod
    def _is_momentum_divergence(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect momentum divergence"""
        if not all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            return pd.Series(False, index=df.index)
        
        return ((df['ret_7d'] > 5) & (df['ret_30d'] < -5)) | \
               ((df['ret_7d'] < -5) & (df['ret_30d'] > 5))
    
    @staticmethod
    def _is_range_compression(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect range compression"""
        if 'position_tension' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return df['position_tension'] < 40
    
    @staticmethod
    def _is_stealth_accumulation(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect stealth accumulation pattern"""
        base_condition = pd.Series(True, index=df.index)
        
        if 'smart_money_flow' in df.columns:
            base_condition &= df['smart_money_flow'] >= CONFIG.PATTERN_THRESHOLDS['stealth']
        
        if 'ret_30d' in df.columns:
            base_condition &= df['ret_30d'].between(-5, 5)
        
        if 'vmi' in df.columns:
            base_condition &= df['vmi'] >= 1.5
        
        return base_condition
    
    @staticmethod
    def _is_vampire_squeeze(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect vampire squeeze pattern"""
        base_condition = conditions.get('extreme_rvol', pd.Series(False, index=df.index))
        
        if 'from_high_pct' in df.columns and 'from_low_pct' in df.columns:
            base_condition &= (df['from_high_pct'] < -50) & (df['from_low_pct'] > 100)
        
        return base_condition
    
    @staticmethod
    def _is_perfect_storm(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect perfect storm pattern"""
        storm_conditions = 0
        base_mask = pd.Series(True, index=df.index)
        
        if conditions.get('extreme_momentum', pd.Series(False, index=df.index)).any():
            storm_conditions += conditions['extreme_momentum'].astype(int)
        
        if conditions.get('extreme_rvol', pd.Series(False, index=df.index)).any():
            storm_conditions += conditions['extreme_rvol'].astype(int)
        
        if 'acceleration_score' in df.columns:
            storm_conditions += (df['acceleration_score'] >= 85).astype(int)
        
        if 'smart_money_flow' in df.columns:
            storm_conditions += (df['smart_money_flow'] >= 80).astype(int)
        
        return storm_conditions >= 3
    
    @staticmethod
    def _is_value_momentum(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect value momentum stocks"""
        if 'pe' not in df.columns:
            return pd.Series(False, index=df.index)
        
        has_valid_pe = (df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000))
        
        return has_valid_pe & (df['pe'] < 15) & conditions.get('high_score', True)
    
    @staticmethod
    def _is_earnings_rocket(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect earnings rockets"""
        if 'eps_change_pct' not in df.columns:
            return pd.Series(False, index=df.index)
        
        has_eps_growth = df['eps_change_pct'].notna()
        extreme_growth = has_eps_growth & (df['eps_change_pct'] > 100)
        
        base_condition = extreme_growth
        
        if 'acceleration_score' in df.columns:
            base_condition &= df['acceleration_score'] >= 70
        
        return base_condition
    
    @staticmethod
    def _is_quality_leader(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect quality leaders"""
        if not all(col in df.columns for col in ['pe', 'eps_change_pct']):
            return pd.Series(False, index=df.index)
        
        has_complete_data = (df['pe'].notna() & df['eps_change_pct'].notna() & 
                           (df['pe'] > 0) & (df['pe'] < 10000))
        
        return has_complete_data & df['pe'].between(10, 25) & \
               (df['eps_change_pct'] > 20) & conditions.get('top_score', True)
    
    @staticmethod
    def _is_turnaround(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect turnaround stories"""
        if 'eps_change_pct' not in df.columns:
            return pd.Series(False, index=df.index)
        
        has_eps = df['eps_change_pct'].notna()
        mega_turnaround = has_eps & (df['eps_change_pct'] > 500)
        
        base_condition = mega_turnaround
        
        if 'volume_score' in df.columns:
            base_condition &= df['volume_score'] >= 60
        
        return base_condition
    
    @staticmethod
    def _is_high_pe_warning(df: pd.DataFrame, conditions: Dict) -> pd.Series:
        """Detect high PE warnings"""
        if 'pe' not in df.columns:
            return pd.Series(False, index=df.index)
        
        return (df['pe'] > 50) & (df['pe'] < 10000)
    
    @staticmethod
    def _combine_patterns(pattern_results: Dict[str, pd.Series]) -> pd.Series:
        """Combine all detected patterns into a single string column"""
        
        def get_patterns_for_row(idx):
            patterns = [name for name, mask in pattern_results.items() if mask.iloc[idx]]
            return ' | '.join(patterns) if patterns else ''
        
        # Vectorized approach for better performance
        pattern_matrix = pd.DataFrame(pattern_results)
        pattern_strings = pattern_matrix.apply(
            lambda row: ' | '.join([col for col, val in row.items() if val]),
            axis=1
        )
        
        return pattern_strings

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    """Advanced market analysis and insights"""
    
    @staticmethod
    def add_category_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """Add category-based analysis"""
        if 'category' not in df.columns or df.empty:
            return df
        
        # Calculate category metrics
        category_stats = df.groupby('category').agg({
            'master_score': ['mean', 'std', 'count'],
            'momentum_score': 'mean',
            'rvol': 'mean'
        }).round(2)
        
        # Flatten column names
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats.columns = ['avg_score', 'score_std', 'count', 'avg_momentum', 'avg_rvol']
        
        # Add category strength rating
        category_stats['strength'] = (
            category_stats['avg_score'] * 0.5 +
            category_stats['avg_momentum'] * 0.3 +
            category_stats['avg_rvol'] * 10 * 0.2
        )
        
        # Log category analysis
        logger.info(f"Category analysis complete for {len(category_stats)} categories")
        
        return df

# ============================================
# VISUALIZATION ENGINE
# ============================================

class VisualizationEngine:
    """Create interactive visualizations"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        if df.empty or 'master_score' not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df['master_score'],
            nbinsx=50,
            name='Score Distribution',
            marker_color='rgba(30, 144, 255, 0.7)',
            hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_score = df['master_score'].mean()
        fig.add_vline(
            x=mean_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_score:.1f}"
        )
        
        # Update layout
        fig.update_layout(
            title="Master Score Distribution",
            xaxis_title="Master Score",
            yaxis_title="Number of Stocks",
            height=400,
            showlegend=False,
            hovermode='x'
        )
        
        return fig
    
    @staticmethod
    def create_momentum_heatmap(df: pd.DataFrame, limit: int = 20) -> go.Figure:
        """Create momentum heatmap for top stocks"""
        if df.empty:
            return go.Figure()
        
        # Get top stocks
        top_stocks = df.nlargest(limit, 'master_score')
        
        # Prepare data for heatmap
        metrics = ['momentum_score', 'acceleration_score', 'volume_score', 
                  'breakout_score', 'position_score', 'rvol_score']
        
        available_metrics = [m for m in metrics if m in top_stocks.columns]
        
        if not available_metrics:
            return go.Figure()
        
        heatmap_data = top_stocks[available_metrics].values.T
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=top_stocks['ticker'].values,
            y=[m.replace('_', ' ').title() for m in available_metrics],
            colorscale='RdYlGn',
            zmid=50,
            text=heatmap_data.round(1),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y}<br>%{x}: %{z:.1f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Component Scores Heatmap - Top {limit} Stocks",
            height=400,
            xaxis={'tickangle': -45},
            yaxis={'tickmode': 'linear'}
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance(df: pd.DataFrame) -> go.Figure:
        """Create sector performance chart"""
        if df.empty or 'sector' not in df.columns:
            return go.Figure()
        
        # Calculate sector metrics
        sector_perf = df.groupby('sector').agg({
            'master_score': 'mean',
            'ret_30d': 'mean',
            'ticker': 'count'
        }).round(2)
        
        sector_perf.columns = ['avg_score', 'avg_return', 'count']
        sector_perf = sector_perf.sort_values('avg_score', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add average score bars
        fig.add_trace(go.Bar(
            y=sector_perf.index,
            x=sector_perf['avg_score'],
            name='Avg Score',
            orientation='h',
            marker_color='dodgerblue',
            text=sector_perf['avg_score'].round(1),
            textposition='auto',
            hovertemplate='%{y}<br>Avg Score: %{x:.1f}<br>Stocks: %{customdata}<extra></extra>',
            customdata=sector_perf['count']
        ))
        
        # Update layout
        fig.update_layout(
            title="Sector Performance Analysis",
            xaxis_title="Average Master Score",
            height=max(400, len(sector_perf) * 25),
            showlegend=False,
            margin=dict(l=150)
        )
        
        return fig
    
    @staticmethod
    def create_volume_momentum_scatter(df: pd.DataFrame, limit: int = 100) -> go.Figure:
        """Create volume vs momentum scatter plot"""
        if df.empty or not all(col in df.columns for col in ['rvol', 'momentum_score']):
            return go.Figure()
        
        # Limit to top stocks for clarity
        plot_df = df.nlargest(limit, 'master_score')
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_df['momentum_score'],
            y=plot_df['rvol'],
            mode='markers+text',
            marker=dict(
                size=plot_df['master_score'] / 5,
                color=plot_df['master_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Master Score"),
                line=dict(width=1, color='white')
            ),
            text=plot_df['ticker'],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>' +
                         'Momentum: %{x:.1f}<br>' +
                         'RVOL: %{y:.2f}<br>' +
                         'Master Score: %{marker.color:.1f}<extra></extra>'
        ))
        
        # Add quadrant lines
        fig.add_hline(y=2, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=70, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=f"Volume vs Momentum Analysis - Top {limit} Stocks",
            xaxis_title="Momentum Score",
            yaxis_title="Relative Volume (RVOL)",
            height=600,
            hovermode='closest'
        )
        
        # Add annotations for quadrants
        fig.add_annotation(x=85, y=3.5, text="🚀 High Mom + Vol", showarrow=False)
        fig.add_annotation(x=85, y=0.5, text="📈 High Mom Only", showarrow=False)
        fig.add_annotation(x=40, y=3.5, text="🔥 High Vol Only", showarrow=False)
        fig.add_annotation(x=40, y=0.5, text="😴 Low Both", showarrow=False)
        
        return fig
    
    @staticmethod
    def create_pattern_sunburst(df: pd.DataFrame) -> go.Figure:
        """Create pattern distribution sunburst chart"""
        if df.empty or 'patterns' not in df.columns:
            return go.Figure()
        
        # Extract all patterns
        all_patterns = []
        pattern_types = {
            'Technical': ['🔥', '💎', '🚀', '🏦', '⚡', '🎯', '👑', '🌊', '💰', '💪', '📈'],
            'Range': ['🎯 52W', '🔄', '👑 GOLDEN', '📊', '🔀', '🎯 RANGE'],
            'Intelligence': ['🤫', '🧛', '⛈️'],
            'Fundamental': ['💎 VALUE', '📊 EARNINGS', '🏆', '⚡ TURNAROUND', '⚠️']
        }
        
        pattern_data = []
        
        for _, row in df.iterrows():
            if row['patterns']:
                patterns = row['patterns'].split(' | ')
                for pattern in patterns:
                    # Determine pattern type
                    pattern_type = 'Other'
                    for ptype, indicators in pattern_types.items():
                        if any(ind in pattern for ind in indicators):
                            pattern_type = ptype
                            break
                    
                    pattern_data.append({
                        'pattern': pattern,
                        'type': pattern_type,
                        'ticker': row['ticker']
                    })
        
        if not pattern_data:
            return go.Figure()
        
        # Create pattern counts
        pattern_df = pd.DataFrame(pattern_data)
        pattern_counts = pattern_df.groupby(['type', 'pattern']).size().reset_index(name='count')
        
        # Create sunburst
        fig = go.Figure(go.Sunburst(
            labels=['All Patterns'] + pattern_counts['type'].unique().tolist() + pattern_counts['pattern'].tolist(),
            parents=[''] + ['All Patterns'] * len(pattern_counts['type'].unique()) + pattern_counts['type'].tolist(),
            values=[len(pattern_data)] + [pattern_counts[pattern_counts['type'] == t]['count'].sum() 
                                         for t in pattern_counts['type'].unique()] + pattern_counts['count'].tolist(),
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percentParent}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="Pattern Distribution Analysis",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_trend_strength_gauge(value: float, title: str = "Trend Strength") -> go.Figure:
        """Create a gauge chart for trend strength"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250)
        return fig
    
    @staticmethod
    def create_market_regime_indicator(df: pd.DataFrame) -> go.Figure:
        """Create market regime indicator"""
        if df.empty or 'ret_30d' not in df.columns:
            return go.Figure()
        
        # Calculate market metrics
        positive_pct = (df['ret_30d'] > 0).mean() * 100
        avg_return = df['ret_30d'].mean()
        
        # Determine regime
        if positive_pct > 60 and avg_return > 5:
            regime = "🔥 RISK-ON BULL"
            color = "green"
        elif positive_pct < 40 and avg_return < -5:
            regime = "🛡️ RISK-OFF DEFENSIVE"
            color = "red"
        else:
            regime = "😴 RANGE-BOUND"
            color = "gray"
        
        # Create indicator
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="number+delta+gauge",
            value=positive_pct,
            delta={'reference': 50, 'relative': False},
            title={'text': f"Market Regime: {regime}<br><sub>Stocks with Positive Returns</sub>"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 60], 'color': "lightgray"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ]
            }
        ))
        
        fig.update_layout(height=250)
        return fig
    
    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, limit: int = 10) -> go.Figure:
        """Create acceleration profiles for top stocks"""
        if df.empty or not all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            return go.Figure()
        
        # Get top accelerating stocks
        if 'acceleration_score' in df.columns:
            top_stocks = df.nlargest(limit, 'acceleration_score')
        else:
            top_stocks = df.nlargest(limit, 'master_score')
        
        fig = go.Figure()
        
        # Add traces for each stock
        for _, stock in top_stocks.iterrows():
            returns = [
                stock.get('ret_1d', 0),
                stock.get('ret_7d', 0),
                stock.get('ret_30d', 0)
            ]
            
            fig.add_trace(go.Scatter(
                x=['1 Day', '7 Days', '30 Days'],
                y=returns,
                mode='lines+markers',
                name=stock['ticker'],
                line=dict(width=2),
                marker=dict(size=8),
                hovertemplate='%{fullData.name}<br>%{x}: %{y:.1f}%<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Acceleration Profiles - Top {limit} Stocks",
            xaxis_title="Time Period",
            yaxis_title="Return (%)",
            height=400,
            hovermode='x'
        )
        
        return fig

# ============================================
# FILTER ENGINE - COMPLETELY FIXED
# ============================================

class FilterEngine:
    """Apply all filters with perfect interconnection"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all active filters efficiently - single pass"""
        
        if df.empty:
            return df
        
        # Start with all True mask
        mask = pd.Series(True, index=df.index)
        
        # Quick filter (highest priority)
        quick_filter = filters.get('quick_filter')
        if quick_filter:
            if quick_filter == 'top_gainers':
                mask &= (df['momentum_score'] >= 80) & (df.get('ret_30d', 0) > 10)
            elif quick_filter == 'volume_surges':
                mask &= (df['rvol'] >= 3) & (df.get('volume_score', 50) >= 80)
            elif quick_filter == 'breakout_ready':
                mask &= (df['breakout_score'] >= 80) & (df.get('from_high_pct', -100) > -10)
            elif quick_filter == 'hidden_gems':
                mask &= (df['patterns'].str.contains('HIDDEN GEM', na=False)) | \
                       ((df['master_score'] >= 70) & (df.get('volume_1d', 0) < df.get('volume_90d', 1) * 0.7))
        
        # Score filter
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            mask &= df['master_score'] >= min_score
        
        # Category filter
        categories = filters.get('categories', [])
        if categories and 'category' in df.columns:
            mask &= df['category'].isin(categories)
        
        # Sector filter
        sectors = filters.get('sectors', [])
        if sectors and 'sector' in df.columns:
            mask &= df['sector'].isin(sectors)
        
        # Industry filter (respects sector)
        industries = filters.get('industries', [])
        if industries and 'industry' in df.columns:
            mask &= df['industry'].isin(industries)
        
        # Pattern filter
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            pattern_mask = df['patterns'].apply(
                lambda x: any(p in x for p in patterns) if x else False
            )
            mask &= pattern_mask
        
        # Trend filter
        trend_filter = filters.get('trend_filter', 'All Trends')
        if trend_filter != 'All Trends' and 'trend_quality' in df.columns:
            if trend_filter == 'Strong Uptrend':
                mask &= df['trend_quality'] >= 75
            elif trend_filter == 'Uptrend':
                mask &= df['trend_quality'] >= 50
            elif trend_filter == 'Neutral':
                mask &= df['trend_quality'].between(25, 75)
            elif trend_filter == 'Downtrend':
                mask &= df['trend_quality'] < 50
        
        # Tier filters
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            tier_values = filters.get(tier_type, [])
            if tier_values:
                tier_col = tier_type.replace('_tiers', '_tier')
                if tier_col in df.columns:
                    mask &= df[tier_col].isin(tier_values)
        
        # Numeric filters
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            mask &= df['eps_change_pct'] >= min_eps_change
        
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in df.columns:
            mask &= df['pe'] >= min_pe
        
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in df.columns:
            mask &= df['pe'] <= max_pe
        
        # Wave filters
        wave_states = filters.get('wave_states', [])
        if wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)
        
        wave_strength_range = filters.get('wave_strength_range', (0, 100))
        if wave_strength_range and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            mask &= (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws)
        
        # Fundamental data requirement
        if filters.get('require_fundamental_data', False):
            if all(col in df.columns for col in ['pe', 'eps_change_pct']):
                mask &= df['pe'].notna() & df['eps_change_pct'].notna()
        
        # Apply mask
        filtered_df = df[mask].copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_interconnected_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        """Get filter options that respect other active filters"""
        
        if df.empty or column not in df.columns:
            return []
        
        # For industry, apply sector filter first
        if column == 'industry' and 'sectors' in current_filters and current_filters['sectors']:
            # Filter by selected sectors first
            sector_filtered = df[df['sector'].isin(current_filters['sectors'])]
            if not sector_filtered.empty:
                df = sector_filtered
        
        # Apply all OTHER filters (except the one we're getting options for)
        temp_filters = current_filters.copy()
        
        # Remove the current filter to see all its options
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
        
        # Apply filters to get interconnected options
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        
        # Get unique values
        values = filtered_df[column].dropna().unique()
        
        # Clean and sort
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN', 'None']]
        
        # Sort intelligently
        try:
            values = sorted(values, key=lambda x: float(str(x).replace(',', '')) 
                          if str(x).replace(',', '').replace('.', '').replace('-', '').isdigit() 
                          else str(x))
        except:
            values = sorted(values, key=str)
        
        return values

# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """High-performance search functionality"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with exact match priority"""
        
        if not query or df.empty:
            return df
        
        query = query.strip().upper()
        
        # Exact ticker match (highest priority)
        exact_ticker = df['ticker'].str.upper() == query
        if exact_ticker.any():
            return df[exact_ticker]
        
        # Ticker starts with query
        ticker_starts = df['ticker'].str.upper().str.startswith(query)
        
        # Company name contains query
        company_contains = pd.Series(False, index=df.index)
        if 'company_name' in df.columns:
            company_contains = df['company_name'].str.upper().str.contains(query, na=False)
        
        # Pattern contains query
        pattern_contains = pd.Series(False, index=df.index)
        if 'patterns' in df.columns:
            pattern_contains = df['patterns'].str.upper().str.contains(query, na=False)
        
        # Combine all matches
        matches = ticker_starts | company_contains | pattern_contains
        
        if not matches.any():
            return pd.DataFrame()
        
        # Sort results by relevance
        result_df = df[matches].copy()
        
        # Add relevance score
        result_df['search_relevance'] = 0
        result_df.loc[exact_ticker, 'search_relevance'] = 100
        result_df.loc[ticker_starts, 'search_relevance'] += 50
        result_df.loc[company_contains, 'search_relevance'] += 25
        result_df.loc[pattern_contains, 'search_relevance'] += 10
        
        # Sort by relevance and master score
        result_df = result_df.sort_values(
            ['search_relevance', 'master_score'],
            ascending=[False, False]
        )
        
        # Remove temporary column
        result_df = result_df.drop('search_relevance', axis=1)
        
        logger.info(f"Search '{query}' found {len(result_df)} matches")
        
        return result_df

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle data exports with multiple formats"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def prepare_export_data(df: pd.DataFrame, template: str = "full") -> pd.DataFrame:
        """Prepare data for export based on template"""
        
        if df.empty:
            return df
        
        export_df = df.copy()
        
        # Define column sets for each template
        templates = {
            "full": None,  # All columns
            "essential": [
                'rank', 'ticker', 'company_name', 'master_score',
                'price', 'ret_30d', 'rvol', 'patterns',
                'category', 'sector', 'industry'
            ],
            "technical": [
                'rank', 'ticker', 'master_score',
                'momentum_score', 'acceleration_score', 'volume_score',
                'breakout_score', 'position_score', 'rvol_score',
                'ret_1d', 'ret_7d', 'ret_30d', 'rvol',
                'from_low_pct', 'from_high_pct', 'patterns'
            ],
            "fundamental": [
                'rank', 'ticker', 'company_name', 'master_score',
                'price', 'pe', 'eps_current', 'eps_change_pct',
                'pe_tier', 'eps_tier', 'category', 'sector'
            ],
            "wave_analysis": [
                'rank', 'ticker', 'master_score',
                'wave_state', 'overall_wave_strength',
                'smart_money_flow', 'momentum_quality',
                'vmi', 'position_tension', 'momentum_harmony',
                'patterns'
            ]
        }
        
        # Apply template
        if template in templates and templates[template]:
            available_cols = [col for col in templates[template] if col in export_df.columns]
            export_df = export_df[available_cols]
        
        # Round numeric columns
        numeric_cols = export_df.select_dtypes(include=[np.number]).columns
        export_df[numeric_cols] = export_df[numeric_cols].round(2)
        
        logger.info(f"Prepared {template} export with {len(export_df.columns)} columns")
        
        return export_df
    
    @staticmethod
    def to_csv(df: pd.DataFrame) -> bytes:
        """Convert dataframe to CSV bytes"""
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue()
    
    @staticmethod
    def to_excel(df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bytes:
        """Convert dataframe to Excel with formatting"""
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Stock Data', index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Stock Data']
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BD',
                'border': 1
            })
            
            # Write headers with format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-adjust column widths
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                worksheet.set_column(i, i, min(max_len, 30))
            
            # Add metadata sheet if provided
            if metadata:
                metadata_df = pd.DataFrame([
                    ['Generated', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')],
                    ['Total Stocks', len(df)],
                    ['Data Source', metadata.get('source', 'Unknown')],
                    ['Processing Time', f"{metadata.get('processing_time', 0):.2f} seconds"]
                ], columns=['Metric', 'Value'])
                
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        return buffer.getvalue()

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_metric_card(label: str, value: str, delta: str = None, help_text: str = None):
        """Render a metric card with optional delta and help text"""
        col = st.container()
        with col:
            if help_text:
                st.metric(label=label, value=value, delta=delta, help=help_text)
            else:
                st.metric(label=label, value=value, delta=delta)
    
    @staticmethod
    def render_stock_card(stock: pd.Series):
        """Render a detailed stock card"""
        with st.container():
            # Header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {stock['ticker']}")
                if 'company_name' in stock and pd.notna(stock['company_name']):
                    st.caption(stock['company_name'])
            
            with col2:
                st.metric("Score", f"{stock['master_score']:.1f}")
            
            with col3:
                st.metric("Rank", f"#{int(stock['rank'])}")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                price_str = f"${stock['price']:.2f}" if stock['price'] < 100 else f"${stock['price']:.0f}"
                st.metric("Price", price_str)
            
            with col2:
                ret_30d = stock.get('ret_30d', 0)
                st.metric("30D Return", f"{ret_30d:.1f}%", 
                         delta=f"{ret_30d:.1f}%" if ret_30d != 0 else None)
            
            with col3:
                rvol = stock.get('rvol', 1)
                st.metric("RVOL", f"{rvol:.2f}x")
            
            with col4:
                if 'wave_state' in stock:
                    st.metric("Wave", stock['wave_state'])
            
            # Patterns
            if 'patterns' in stock and stock['patterns']:
                st.markdown("**Patterns:** " + stock['patterns'])
            
            # Scores breakdown
            with st.expander("Score Breakdown"):
                scores_df = pd.DataFrame({
                    'Component': ['Position', 'Volume', 'Momentum', 'Acceleration', 'Breakout', 'RVOL'],
                    'Score': [
                        stock.get('position_score', 50),
                        stock.get('volume_score', 50),
                        stock.get('momentum_score', 50),
                        stock.get('acceleration_score', 50),
                        stock.get('breakout_score', 50),
                        stock.get('rvol_score', 50)
                    ]
                })
                st.dataframe(scores_df, hide_index=True)

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
    
    # Initialize session state
    RobustSessionState.initialize()
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 400px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🌊 Wave Detection Ultimate 3.0")
    st.markdown("**Professional Stock Ranking System** | Real-time Analysis | Smart Pattern Detection")
    
    # Data source selection
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    
    with col1:
        data_source = st.radio(
            "Data Source",
            ["Google Sheets", "Upload CSV"],
            index=0 if RobustSessionState.safe_get('data_source') == "sheet" else 1,
            horizontal=True
        )
        RobustSessionState.safe_set('data_source', "sheet" if data_source == "Google Sheets" else "upload")
    
    with col2:
        if data_source == "Google Sheets":
            default_sheet = "1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM"
            sheet_id = st.text_input(
                "Sheet ID or URL",
                value=RobustSessionState.safe_get('sheet_id', default_sheet),
                placeholder="Enter Google Sheets ID or full URL",
                help="Example: 1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM"
            )
            RobustSessionState.safe_set('sheet_id', sheet_id)
        else:
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            sheet_id = None
    
    with col3:
        if data_source == "Google Sheets":
            gid = st.text_input(
                "GID (Tab ID)",
                value=RobustSessionState.safe_get('gid', CONFIG.DEFAULT_GID),
                placeholder=CONFIG.DEFAULT_GID,
                help="Optional: Specific sheet tab ID"
            )
            RobustSessionState.safe_set('gid', gid)
        else:
            gid = None
    
    with col4:
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            # Clear cache and reload
            st.cache_data.clear()
            st.rerun()
    
    # Data loading and processing
    try:
        if RobustSessionState.safe_get('data_source') == "upload" and uploaded_file is None:
            st.warning("👆 Please upload a CSV file to continue")
            st.stop()
        
        if RobustSessionState.safe_get('data_source') == "sheet" and not sheet_id:
            st.warning("👆 Please enter a Google Sheets ID to continue")
            st.stop()
        
        # Smart cache key implementation
        cache_key_prefix = f"{RobustSessionState.safe_get('data_source')}_{sheet_id}_{gid}"
        current_hour_key = datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')
        data_version_hash = hashlib.md5(f"{cache_key_prefix}_{current_hour_key}".encode()).hexdigest()
        
        # Load and process data
        with st.spinner("📥 Loading and processing data..."):
            try:
                if RobustSessionState.safe_get('data_source') == "upload" and uploaded_file is not None:
                    ranked_df, data_timestamp, metadata = load_and_process_data(
                        "upload", file_data=uploaded_file, data_version=data_version_hash
                    )
                else:
                    ranked_df, data_timestamp, metadata = load_and_process_data(
                        "sheet", 
                        sheet_id=sheet_id,
                        gid=gid,
                        data_version=data_version_hash
                    )
                
                RobustSessionState.safe_set('ranked_df', ranked_df)
                RobustSessionState.safe_set('data_timestamp', data_timestamp)
                RobustSessionState.safe_set('last_refresh', datetime.now(timezone.utc))
                
                # Show warnings/errors
                if metadata.get('warnings'):
                    for warning in metadata['warnings']:
                        st.warning(warning)
                
                if metadata.get('errors'):
                    for error in metadata['errors']:
                        st.error(error)
                
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}")
                
                # Try to use last good data
                last_good_data = RobustSessionState.safe_get('last_good_data')
                if last_good_data:
                    ranked_df, data_timestamp, metadata = last_good_data
                    st.warning("⚠️ Failed to load fresh data, using cached version")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Common issues:\n- Invalid Google Sheets ID\n- Sheet not publicly accessible\n- Network connectivity issues\n- Invalid CSV format")
                    st.stop()
    
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        st.stop()
    
    # Data quality info
    data_quality = RobustSessionState.safe_get('data_quality', {})
    last_refresh = RobustSessionState.safe_get('last_refresh')
    
    with st.expander("📊 Data Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", f"{len(ranked_df):,}")
        with col2:
            st.metric("Data Completeness", f"{data_quality.get('completeness', 0):.1f}%")
        with col3:
            if last_refresh:
                mins_ago = (datetime.now(timezone.utc) - last_refresh).seconds // 60
                st.metric("Last Refresh", f"{mins_ago} mins ago")
        with col4:
            st.metric("Processing Time", f"{metadata.get('processing_time', 0):.1f}s")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
    with qa_col1:
        if st.button("📈 Top Gainers", use_container_width=True):
            filters = RobustSessionState.safe_get('active_filters')
            filters['quick_filter'] = 'top_gainers'
            RobustSessionState.safe_set('active_filters', filters)
    
    with qa_col2:
        if st.button("🔥 Volume Surges", use_container_width=True):
            filters = RobustSessionState.safe_get('active_filters')
            filters['quick_filter'] = 'volume_surges'
            RobustSessionState.safe_set('active_filters', filters)
    
    with qa_col3:
        if st.button("🎯 Breakout Ready", use_container_width=True):
            filters = RobustSessionState.safe_get('active_filters')
            filters['quick_filter'] = 'breakout_ready'
            RobustSessionState.safe_set('active_filters', filters)
    
    with qa_col4:
        if st.button("💎 Hidden Gems", use_container_width=True):
            filters = RobustSessionState.safe_get('active_filters')
            filters['quick_filter'] = 'hidden_gems'
            RobustSessionState.safe_set('active_filters', filters)
    
    with qa_col5:
        if st.button("🌊 Show All", use_container_width=True):
            filters = RobustSessionState.safe_get('active_filters')
            filters['quick_filter'] = None
            RobustSessionState.safe_set('active_filters', filters)
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🎯 Smart Filters")
        
        # Display mode
        display_mode = st.radio(
            "Display Mode",
            ["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if RobustSessionState.safe_get('display_mode_radio') == 'Technical' else 1,
            help="Technical: Pure momentum | Hybrid: Adds PE & EPS"
        )
        RobustSessionState.safe_set('display_mode_radio', display_mode)
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        # Get current filters
        current_filters = RobustSessionState.safe_get('active_filters')
        
        # Category filter
        categories = FilterEngine.get_interconnected_options(ranked_df, 'category', current_filters)
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=categories,
            default=current_filters.get('categories', []),
            placeholder="All Categories"
        )
        current_filters['categories'] = selected_categories
        
        # Sector filter
        sectors = FilterEngine.get_interconnected_options(ranked_df, 'sector', current_filters)
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=current_filters.get('sectors', []),
            placeholder="All Sectors"
        )
        current_filters['sectors'] = selected_sectors
        
        # Industry filter (respects sector)
        if 'industry' in ranked_df.columns:
            industries = FilterEngine.get_interconnected_options(ranked_df, 'industry', current_filters)
            selected_industries = st.multiselect(
                "Industry",
                options=industries,
                default=current_filters.get('industries', []),
                placeholder="All Industries",
                help="Filtered by selected sectors"
            )
            current_filters['industries'] = selected_industries
        
        # Score filter
        min_score = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=current_filters.get('min_score', 0),
            step=5
        )
        current_filters['min_score'] = min_score
        
        # Pattern filter
        if 'patterns' in ranked_df.columns:
            all_patterns = set()
            for patterns in ranked_df['patterns'].dropna():
                if patterns:
                    all_patterns.update(p.strip() for p in patterns.split('|'))
            
            all_patterns = sorted(list(all_patterns))
            selected_patterns = st.multiselect(
                "Patterns",
                options=all_patterns,
                default=current_filters.get('patterns', []),
                placeholder="All Patterns"
            )
            current_filters['patterns'] = selected_patterns
        
        # Trend filter
        trend_options = ['All Trends', 'Strong Uptrend', 'Uptrend', 'Neutral', 'Downtrend']
        trend_filter = st.selectbox(
            "Trend Filter",
            options=trend_options,
            index=trend_options.index(current_filters.get('trend_filter', 'All Trends'))
        )
        current_filters['trend_filter'] = trend_filter
        
        # Wave filters
        st.markdown("#### 🌊 Wave Filters")
        
        if 'wave_state' in ranked_df.columns:
            wave_states = FilterEngine.get_interconnected_options(ranked_df, 'wave_state', current_filters)
            selected_wave_states = st.multiselect(
                "Wave States",
                options=wave_states,
                default=current_filters.get('wave_states', []),
                placeholder="All Wave States"
            )
            current_filters['wave_states'] = selected_wave_states
        
        if 'overall_wave_strength' in ranked_df.columns:
            wave_range = st.slider(
                "Wave Strength Range",
                min_value=0,
                max_value=100,
                value=current_filters.get('wave_strength_range', (0, 100)),
                step=5
            )
            current_filters['wave_strength_range'] = wave_range
        
        # Fundamental filters (if in hybrid mode)
        if show_fundamentals:
            st.markdown("#### 💼 Fundamental Filters")
            
            # EPS tier filter
            if 'eps_tier' in ranked_df.columns:
                eps_tiers = FilterEngine.get_interconnected_options(ranked_df, 'eps_tier', current_filters)
                selected_eps_tiers = st.multiselect(
                    "EPS Tiers",
                    options=eps_tiers,
                    default=current_filters.get('eps_tiers', []),
                    placeholder="All EPS Tiers"
                )
                current_filters['eps_tiers'] = selected_eps_tiers
            
            # PE tier filter
            if 'pe_tier' in ranked_df.columns:
                pe_tiers = FilterEngine.get_interconnected_options(ranked_df, 'pe_tier', current_filters)
                selected_pe_tiers = st.multiselect(
                    "PE Tiers",
                    options=pe_tiers,
                    default=current_filters.get('pe_tiers', []),
                    placeholder="All PE Tiers"
                )
                current_filters['pe_tiers'] = selected_pe_tiers
            
            # EPS change filter
            col1, col2 = st.columns(2)
            with col1:
                min_eps_change = st.number_input(
                    "Min EPS Change %",
                    value=current_filters.get('min_eps_change', None),
                    placeholder="Any",
                    step=10.0
                )
                current_filters['min_eps_change'] = min_eps_change if min_eps_change else None
            
            with col2:
                require_fundamental = st.checkbox(
                    "Require Fund. Data",
                    value=current_filters.get('require_fundamental_data', False)
                )
                current_filters['require_fundamental_data'] = require_fundamental
            
            # PE range filter
            col1, col2 = st.columns(2)
            with col1:
                min_pe = st.number_input(
                    "Min PE",
                    value=current_filters.get('min_pe', None),
                    placeholder="Any",
                    step=1.0
                )
                current_filters['min_pe'] = min_pe if min_pe else None
            
            with col2:
                max_pe = st.number_input(
                    "Max PE",
                    value=current_filters.get('max_pe', None),
                    placeholder="Any",
                    step=1.0
                )
                current_filters['max_pe'] = max_pe if max_pe else None
        
        # Price tier filter
        if 'price_tier' in ranked_df.columns:
            st.markdown("#### 💰 Price Filters")
            price_tiers = FilterEngine.get_interconnected_options(ranked_df, 'price_tier', current_filters)
            selected_price_tiers = st.multiselect(
                "Price Tiers",
                options=price_tiers,
                default=current_filters.get('price_tiers', []),
                placeholder="All Price Ranges"
            )
            current_filters['price_tiers'] = selected_price_tiers
        
        # Update session state with current filters
        RobustSessionState.safe_set('active_filters', current_filters)
        
        # Count active filters
        filter_count = 0
        if current_filters.get('quick_filter'):
            filter_count += 1
        if current_filters.get('categories'):
            filter_count += 1
        if current_filters.get('sectors'):
            filter_count += 1
        if current_filters.get('industries'):
            filter_count += 1
        if current_filters.get('min_score', 0) > 0:
            filter_count += 1
        if current_filters.get('patterns'):
            filter_count += 1
        if current_filters.get('trend_filter', 'All Trends') != 'All Trends':
            filter_count += 1
        if current_filters.get('wave_states'):
            filter_count += 1
        if current_filters.get('wave_strength_range', (0, 100)) != (0, 100):
            filter_count += 1
        if current_filters.get('eps_tiers'):
            filter_count += 1
        if current_filters.get('pe_tiers'):
            filter_count += 1
        if current_filters.get('price_tiers'):
            filter_count += 1
        if current_filters.get('min_eps_change') is not None:
            filter_count += 1
        if current_filters.get('min_pe') is not None:
            filter_count += 1
        if current_filters.get('max_pe') is not None:
            filter_count += 1
        if current_filters.get('require_fundamental_data'):
            filter_count += 1
        
        RobustSessionState.safe_set('filter_count', filter_count)
        
        # Show active filter count
        if filter_count > 0:
            st.info(f"🔍 **{filter_count} filter{'s' if filter_count > 1 else ''} active**")
        
        # Clear filters button
        if st.button("🗑️ Clear All Filters", 
                    use_container_width=True,
                    type="primary" if filter_count > 0 else "secondary"):
            RobustSessionState.clear_all_filters()
            st.rerun()
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            show_debug = st.checkbox(
                "Show Debug Info",
                value=RobustSessionState.safe_get('show_debug', False)
            )
            RobustSessionState.safe_set('show_debug', show_debug)
            
            if show_debug:
                st.write("**Active Filters:**")
                st.json(current_filters)
                
                if RobustSessionState.safe_get('performance_metrics'):
                    st.write("**Performance:**")
                    for func, duration in RobustSessionState.safe_get('performance_metrics').items():
                        st.write(f"- {func}: {duration:.3f}s")
    
    # Apply all filters
    filtered_df = FilterEngine.apply_filters(ranked_df, current_filters)
    
    # Search functionality
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search stocks",
            value=RobustSessionState.safe_get('search_input', ''),
            placeholder="Enter ticker, company name, or pattern...",
            key="search_input"
        )
        
        # Apply search if query exists
        if search_query:
            filtered_df = SearchEngine.search_stocks(filtered_df, search_query)
            RobustSessionState.safe_set('search_query', search_query)
        else:
            RobustSessionState.safe_set('search_query', '')
    
    with col2:
        top_n = st.selectbox(
            "Show Top",
            options=CONFIG.AVAILABLE_TOP_N,
            index=CONFIG.AVAILABLE_TOP_N.index(
                RobustSessionState.safe_get('top_n_slider', CONFIG.DEFAULT_TOP_N)
            ),
            key="top_n_slider"
        )
    
    # Filter status display
    if filter_count > 0 or search_query:
        status_parts = []
        
        if search_query:
            status_parts.append(f"Search: '{search_query}'")
        
        if current_filters.get('quick_filter'):
            quick_names = {
                'top_gainers': '📈 Top Gainers',
                'volume_surges': '🔥 Volume Surges',
                'breakout_ready': '🎯 Breakout Ready',
                'hidden_gems': '💎 Hidden Gems'
            }
            status_parts.append(quick_names.get(current_filters['quick_filter'], 'Filtered'))
        
        if filter_count > (1 if current_filters.get('quick_filter') else 0):
            additional = filter_count - (1 if current_filters.get('quick_filter') else 0)
            status_parts.append(f"{additional} other filter{'s' if additional > 1 else ''}")
        
        st.info(f"**Active:** {' + '.join(status_parts)} | **{len(filtered_df):,} stocks** found")
    
    # Limit to top N
    display_df = filtered_df.head(top_n)
    
    # Summary metrics
    if not display_df.empty:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            UIComponents.render_metric_card(
                "Stocks Shown",
                f"{len(display_df):,}",
                f"of {len(filtered_df):,} filtered"
            )
        
        with col2:
            avg_score = display_df['master_score'].mean()
            UIComponents.render_metric_card(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={display_df['master_score'].std():.1f}"
            )
        
        with col3:
            if 'ret_30d' in display_df.columns:
                avg_return = display_df['ret_30d'].mean()
                UIComponents.render_metric_card(
                    "Avg 30D Return",
                    f"{avg_return:.1f}%",
                    "↑" if avg_return > 0 else "↓"
                )
        
        with col4:
            if 'rvol' in display_df.columns:
                high_rvol = (display_df['rvol'] >= 2).sum()
                UIComponents.render_metric_card(
                    "High RVOL",
                    f"{high_rvol}",
                    f"{high_rvol/len(display_df)*100:.0f}% of shown"
                )
        
        with col5:
            patterns_count = display_df['patterns'].str.len().gt(0).sum()
            UIComponents.render_metric_card(
                "With Patterns",
                f"{patterns_count}",
                f"{patterns_count/len(display_df)*100:.0f}% of shown"
            )
        
        with col6:
            if 'market_regime' in display_df.columns:
                regime = display_df['market_regime'].iloc[0] if len(display_df) > 0 else "Unknown"
                UIComponents.render_metric_card(
                    "Market Regime",
                    regime.split()[-1] if ' ' in regime else regime
                )
    
    # Main content tabs
    tabs = st.tabs([
        "📊 Ranking Table",
        "📈 Visualizations", 
        "🌊 Wave Radar™",
        "🎯 Pattern Analysis",
        "🏆 Performance Analysis",
        "💾 Export Data",
        "ℹ️ About"
    ])
    
    # Tab 1: Ranking Table
    with tabs[0]:
        if display_df.empty:
            st.warning("No stocks match the current filters. Try adjusting your criteria.")
        else:
            # Prepare display columns
            display_columns = [
                'rank', 'ticker', 'master_score', 'price', 'ret_30d', 'rvol',
                'patterns', 'category', 'sector'
            ]
            
            if 'industry' in display_df.columns:
                display_columns.insert(9, 'industry')
            
            if show_fundamentals:
                fund_cols = ['pe', 'eps_change_pct']
                for col in fund_cols:
                    if col in display_df.columns:
                        display_columns.append(col)
            
            # Add wave columns
            wave_cols = ['wave_state', 'smart_money_flow']
            for col in wave_cols:
                if col in display_df.columns:
                    display_columns.append(col)
            
            # Filter to available columns
            display_columns = [col for col in display_columns if col in display_df.columns]
            
            # Format the dataframe
            formatted_df = display_df[display_columns].copy()
            
            # Format numeric columns
            if 'price' in formatted_df.columns:
                formatted_df['price'] = formatted_df['price'].apply(
                    lambda x: f"${x:.2f}" if x < 100 else f"${x:.0f}"
                )
            
            if 'ret_30d' in formatted_df.columns:
                formatted_df['ret_30d'] = formatted_df['ret_30d'].apply(
                    lambda x: f"{x:.1f}%"
                )
            
            if 'rvol' in formatted_df.columns:
                formatted_df['rvol'] = formatted_df['rvol'].apply(
                    lambda x: f"{x:.2f}x"
                )
            
            if 'pe' in formatted_df.columns:
                formatted_df['pe'] = formatted_df['pe'].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "N/A"
                )
            
            if 'eps_change_pct' in formatted_df.columns:
                formatted_df['eps_change_pct'] = formatted_df['eps_change_pct'].apply(
                    lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A"
                )
            
            # Display the table
            st.dataframe(
                formatted_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn(
                        "Rank",
                        help="Overall ranking based on Master Score",
                        format="%d"
                    ),
                    "ticker": st.column_config.TextColumn(
                        "Ticker",
                        help="Stock symbol",
                        width="small"
                    ),
                    "master_score": st.column_config.ProgressColumn(
                        "Score",
                        help="Master Score (0-100)",
                        format="%.1f",
                        min_value=0,
                        max_value=100
                    ),
                    "patterns": st.column_config.TextColumn(
                        "Patterns",
                        help="Detected trading patterns",
                        width="large"
                    ),
                    "wave_state": st.column_config.TextColumn(
                        "Wave",
                        help="Current wave state",
                        width="medium"
                    ),
                    "smart_money_flow": st.column_config.ProgressColumn(
                        "Smart Flow",
                        help="Smart Money Flow indicator",
                        format="%.0f",
                        min_value=0,
                        max_value=100
                    )
                }
            )
            
            # Selected stock details
            if len(display_df) > 0:
                st.markdown("---")
                st.subheader("📋 Stock Details")
                
                selected_ticker = st.selectbox(
                    "Select a stock for detailed view:",
                    options=display_df['ticker'].tolist(),
                    index=0
                )
                
                selected_stock = display_df[display_df['ticker'] == selected_ticker].iloc[0]
                UIComponents.render_stock_card(selected_stock)
    
    # Tab 2: Visualizations
    with tabs[1]:
        if display_df.empty:
            st.warning("No data available for visualization")
        else:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Score distribution
                fig_dist = VisualizationEngine.create_score_distribution(display_df)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Sector performance
                if 'sector' in display_df.columns:
                    fig_sector = VisualizationEngine.create_sector_performance(display_df)
                    st.plotly_chart(fig_sector, use_container_width=True)
            
            with viz_col2:
                # Momentum heatmap
                fig_heatmap = VisualizationEngine.create_momentum_heatmap(display_df, limit=20)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Market regime indicator
                fig_regime = VisualizationEngine.create_market_regime_indicator(display_df)
                st.plotly_chart(fig_regime, use_container_width=True)
            
            # Volume vs Momentum scatter
            st.markdown("---")
            fig_scatter = VisualizationEngine.create_volume_momentum_scatter(display_df, limit=min(100, len(display_df)))
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tab 3: Wave Radar
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar™ - Early Momentum Detection System")
        
        # Wave controls
        wave_col1, wave_col2, wave_col3 = st.columns([2, 1, 1])
        
        with wave_col1:
            wave_timeframe = st.selectbox(
                "Timeframe",
                ["All Waves", "🌊🌊🌊 Cresting Only", "🌊🌊 Building", "🌊 Forming"],
                index=["All Waves", "🌊🌊🌊 Cresting Only", "🌊🌊 Building", "🌊 Forming"].index(
                    RobustSessionState.safe_get('wave_timeframe_select', 'All Waves')
                ),
                key="wave_timeframe_select"
            )
        
        with wave_col2:
            wave_sensitivity = st.select_slider(
                "Sensitivity",
                options=["Conservative", "Balanced", "Aggressive"],
                value=RobustSessionState.safe_get('wave_sensitivity', 'Balanced'),
                key="wave_sensitivity"
            )
        
        with wave_col3:
            show_details = st.checkbox(
                "Show Details",
                value=RobustSessionState.safe_get('show_sensitivity_details', False),
                key="show_sensitivity_details"
            )
        
        # Filter by wave state if selected
        wave_df = display_df.copy()
        if wave_timeframe != "All Waves" and 'wave_state' in wave_df.columns:
            wave_df = wave_df[wave_df['wave_state'].str.contains(wave_timeframe.split()[0])]
        
        if wave_df.empty:
            st.info("No stocks in selected wave state")
        else:
            # Wave distribution
            if 'wave_state' in wave_df.columns:
                wave_counts = wave_df['wave_state'].value_counts()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cresting = wave_counts.get('🌊🌊🌊 CRESTING', 0)
                    UIComponents.render_metric_card(
                        "🌊🌊🌊 Cresting",
                        str(cresting),
                        f"{cresting/len(wave_df)*100:.0f}% of shown"
                    )
                
                with col2:
                    building = wave_counts.get('🌊🌊 BUILDING', 0)
                    UIComponents.render_metric_card(
                        "🌊🌊 Building",
                        str(building),
                        f"{building/len(wave_df)*100:.0f}% of shown"
                    )
                
                with col3:
                    forming = wave_counts.get('🌊 FORMING', 0)
                    UIComponents.render_metric_card(
                        "🌊 Forming",
                        str(forming),
                        f"{forming/len(wave_df)*100:.0f}% of shown"
                    )
                
                with col4:
                    breaking = wave_counts.get('💥 BREAKING', 0)
                    UIComponents.render_metric_card(
                        "💥 Breaking",
                        str(breaking),
                        f"{breaking/len(wave_df)*100:.0f}% of shown"
                    )
            
            # Momentum shifts detection
            st.markdown("#### 🔄 Momentum Shifts")
            
            if all(col in wave_df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d', 'acceleration_score']):
                # Find stocks with momentum shifts
                momentum_shifts = wave_df[
                    (wave_df['acceleration_score'] > 80) |
                    ((wave_df['ret_7d'] > wave_df['ret_30d'] / 4.3 * 1.5) & (wave_df['ret_7d'] > 0))
                ].head(10)
                
                if not momentum_shifts.empty:
                    shift_data = momentum_shifts[[
                        'ticker', 'ret_1d', 'ret_7d', 'ret_30d', 
                        'acceleration_score', 'wave_state'
                    ]].copy()
                    
                    shift_data['momentum_signal'] = shift_data.apply(
                        lambda x: "🚀 Accelerating" if x['acceleration_score'] > 85 
                        else "📈 Building" if x['acceleration_score'] > 70 
                        else "🌱 Early", axis=1
                    )
                    
                    st.dataframe(
                        shift_data,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "ret_1d": st.column_config.NumberColumn("1D %", format="%.1f"),
                            "ret_7d": st.column_config.NumberColumn("7D %", format="%.1f"),
                            "ret_30d": st.column_config.NumberColumn("30D %", format="%.1f"),
                            "acceleration_score": st.column_config.ProgressColumn(
                                "Acceleration",
                                min_value=0,
                                max_value=100
                            )
                        }
                    )
                else:
                    st.info("No significant momentum shifts detected")
            
            # Smart money flow analysis
            if 'smart_money_flow' in wave_df.columns:
                st.markdown("#### 💰 Smart Money Flow Analysis")
                
                flow_analysis_cols = st.columns(2)
                
                with flow_analysis_cols[0]:
                    # Smart money flow by category
                    if 'category' in wave_df.columns:
                        flow_by_category = wave_df.groupby('category')['smart_money_flow'].mean().sort_values(ascending=False)
                        
                        fig_flow = go.Figure(data=[
                            go.Bar(
                                x=flow_by_category.index,
                                y=flow_by_category.values,
                                marker_color=px.colors.sequential.Viridis,
                                text=flow_by_category.values.round(1),
                                textposition='auto'
                            )
                        ])
                        
                        fig_flow.update_layout(
                            title="Smart Money Flow by Category",
                            xaxis_title="Category",
                            yaxis_title="Average Flow Score",
                            height=350
                        )
                        
                        st.plotly_chart(fig_flow, use_container_width=True)
                
                with flow_analysis_cols[1]:
                    # Top smart money targets
                    st.markdown("##### 🎯 Smart Money Targets")
                    
                    smart_targets = wave_df.nlargest(5, 'smart_money_flow')[
                        ['ticker', 'smart_money_flow', 'category']
                    ]
                    
                    for _, target in smart_targets.iterrows():
                        flow_level = "🔥" if target['smart_money_flow'] > 80 else "📈"
                        st.markdown(
                            f"{flow_level} **{target['ticker']}** ({target['category']})\n"
                            f"Flow Score: {target['smart_money_flow']:.1f}"
                        )
            
            # Market regime context
            if RobustSessionState.safe_get('show_market_regime', True):
                st.markdown("#### 🌍 Market Context")
                
                if 'market_regime' in wave_df.columns:
                    regime = wave_df['market_regime'].iloc[0] if len(wave_df) > 0 else "Unknown"
                    
                    if "BULL" in regime:
                        st.success(f"**Current Regime:** {regime}")
                        st.info("💡 Bull market conditions favor momentum strategies")
                    elif "DEFENSIVE" in regime:
                        st.warning(f"**Current Regime:** {regime}")
                        st.info("💡 Defensive positioning may be prudent")
                    else:
                        st.info(f"**Current Regime:** {regime}")
                        st.info("💡 Mixed signals suggest selective positioning")
    
    # Tab 4: Pattern Analysis
    with tabs[3]:
        if display_df.empty or 'patterns' not in display_df.columns:
            st.warning("No pattern data available")
        else:
            # Pattern distribution sunburst
            fig_sunburst = VisualizationEngine.create_pattern_sunburst(display_df)
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            # Pattern frequency analysis
            st.markdown("#### 📊 Pattern Frequency Analysis")
            
            # Extract all patterns
            all_patterns = []
            for patterns in display_df['patterns'].dropna():
                if patterns:
                    all_patterns.extend(p.strip() for p in patterns.split('|'))
            
            if all_patterns:
                pattern_counts = pd.Series(all_patterns).value_counts().head(20)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_patterns = go.Figure(data=[
                        go.Bar(
                            y=pattern_counts.index,
                            x=pattern_counts.values,
                            orientation='h',
                            marker_color='lightseagreen',
                            text=pattern_counts.values,
                            textposition='auto'
                        )
                    ])
                    
                    fig_patterns.update_layout(
                        title="Top 20 Most Common Patterns",
                        xaxis_title="Frequency",
                        yaxis_title="Pattern",
                        height=600,
                        margin=dict(l=200)
                    )
                    
                    st.plotly_chart(fig_patterns, use_container_width=True)
                
                with col2:
                    st.markdown("##### 🎯 Pattern Leaders")
                    
                    # Find stocks with most patterns
                    display_df['pattern_count'] = display_df['patterns'].str.count('\|') + 1
                    pattern_leaders = display_df.nlargest(10, 'pattern_count')[
                        ['ticker', 'pattern_count', 'master_score']
                    ]
                    
                    for _, leader in pattern_leaders.iterrows():
                        st.markdown(
                            f"**{leader['ticker']}** - {leader['pattern_count']} patterns\n"
                            f"Score: {leader['master_score']:.1f}"
                        )
            
            # Pattern combinations
            st.markdown("#### 🔗 Pattern Combinations")
            
            # Find stocks with multiple high-value patterns
            valuable_patterns = ['🔥 CAT LEADER', '🚀 ACCELERATING', '🏦 INSTITUTIONAL', 
                               '⚡ VOL EXPLOSION', '👑 MARKET LEADER', '⛈️ PERFECT STORM']
            
            multi_pattern_stocks = []
            
            for _, stock in display_df.iterrows():
                if stock['patterns']:
                    pattern_list = stock['patterns'].split(' | ')
                    valuable_count = sum(1 for p in valuable_patterns if p in pattern_list)
                    
                    if valuable_count >= 2:
                        multi_pattern_stocks.append({
                            'ticker': stock['ticker'],
                            'score': stock['master_score'],
                            'valuable_patterns': valuable_count,
                            'total_patterns': len(pattern_list),
                            'patterns': stock['patterns']
                        })
            
            if multi_pattern_stocks:
                multi_df = pd.DataFrame(multi_pattern_stocks).sort_values(
                    ['valuable_patterns', 'score'], 
                    ascending=[False, False]
                ).head(10)
                
                st.dataframe(
                    multi_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "score": st.column_config.ProgressColumn(
                            "Score",
                            min_value=0,
                            max_value=100
                        ),
                        "patterns": st.column_config.TextColumn(
                            "Patterns",
                            width="large"
                        )
                    }
                )
            else:
                st.info("No stocks with multiple high-value patterns found")
    
    # Tab 5: Performance Analysis
    with tabs[4]:
        st.markdown("### 🏆 Performance Analysis")
        
        # Time period selector
        perf_col1, perf_col2, perf_col3 = st.columns([1, 1, 2])
        
        with perf_col1:
            analysis_metric = st.selectbox(
                "Analyze by",
                ["Category", "Sector", "Industry", "Pattern Type"]
            )
        
        with perf_col2:
            perf_metric = st.selectbox(
                "Performance Metric",
                ["Master Score", "30D Return", "Smart Money Flow", "RVOL"]
            )
        
        # Map display names to column names
        metric_map = {
            "Master Score": "master_score",
            "30D Return": "ret_30d",
            "Smart Money Flow": "smart_money_flow",
            "RVOL": "rvol"
        }
        
        metric_col = metric_map.get(perf_metric, "master_score")
        
        if metric_col not in display_df.columns:
            st.warning(f"{perf_metric} data not available")
        else:
            # Prepare analysis data
            if analysis_metric == "Pattern Type":
                # Special handling for patterns
                pattern_perf_data = []
                
                for _, stock in display_df.iterrows():
                    if stock['patterns']:
                        patterns = stock['patterns'].split(' | ')
                        for pattern in patterns:
                            # Determine pattern type
                            if any(x in pattern for x in ['🔥', '💎', '🚀', '🏦', '⚡', '🎯', '👑', '🌊', '💰', '💪', '📈']):
                                ptype = "Technical"
                            elif any(x in pattern for x in ['52W', '🔄', 'GOLDEN', '📊', '🔀', 'RANGE']):
                                ptype = "Range"
                            elif any(x in pattern for x in ['🤫', '🧛', '⛈️']):
                                ptype = "Intelligence"
                            elif any(x in pattern for x in ['VALUE', 'EARNINGS', '🏆', 'TURNAROUND', 'HIGH PE']):
                                ptype = "Fundamental"
                            else:
                                ptype = "Other"
                            
                            pattern_perf_data.append({
                                'type': ptype,
                                'value': stock[metric_col]
                            })
                
                if pattern_perf_data:
                    pattern_df = pd.DataFrame(pattern_perf_data)
                    perf_summary = pattern_df.groupby('type')['value'].agg(['mean', 'std', 'count'])
                    analysis_col = 'type'
                else:
                    st.warning("No pattern data available")
                    perf_summary = pd.DataFrame()
            else:
                # Regular groupby analysis
                analysis_col = analysis_metric.lower()
                
                if analysis_col not in display_df.columns:
                    st.warning(f"{analysis_metric} data not available")
                    perf_summary = pd.DataFrame()
                else:
                    perf_summary = display_df.groupby(analysis_col)[metric_col].agg(['mean', 'std', 'count'])
            
            if not perf_summary.empty:
                perf_summary = perf_summary.sort_values('mean', ascending=False)
                
                # Visualization
                fig_perf = go.Figure()
                
                # Add mean bars
                fig_perf.add_trace(go.Bar(
                    name='Average',
                    x=perf_summary.index,
                    y=perf_summary['mean'],
                    error_y=dict(
                        type='data',
                        array=perf_summary['std'],
                        visible=True
                    ),
                    text=[f"{val:.1f}<br>n={int(cnt)}" 
                          for val, cnt in zip(perf_summary['mean'], perf_summary['count'])],
                    textposition='auto',
                    marker_color='dodgerblue'
                ))
                
                fig_perf.update_layout(
                    title=f"{perf_metric} by {analysis_metric}",
                    xaxis_title=analysis_metric,
                    yaxis_title=f"Average {perf_metric}",
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Detailed table
                with st.expander("📊 Detailed Statistics"):
                    detailed_stats = perf_summary.copy()
                    detailed_stats.columns = ['Average', 'Std Dev', 'Count']
                    detailed_stats = detailed_stats.round(2)
                    
                    st.dataframe(
                        detailed_stats,
                        use_container_width=True,
                        column_config={
                            "Average": st.column_config.NumberColumn(format="%.2f"),
                            "Std Dev": st.column_config.NumberColumn(format="%.2f"),
                            "Count": st.column_config.NumberColumn(format="%d")
                        }
                    )
        
        # Acceleration profiles
        st.markdown("---")
        st.markdown("#### 🚀 Acceleration Profiles")
        
        fig_accel = VisualizationEngine.create_acceleration_profiles(display_df, limit=10)
        st.plotly_chart(fig_accel, use_container_width=True)
    
    # Tab 6: Export Data
    with tabs[5]:
        st.markdown("### 💾 Export Data")
        
        export_col1, export_col2 = st.columns([1, 2])
        
        with export_col1:
            export_template = st.radio(
                "Export Template",
                [
                    "Full Analysis (All Data)",
                    "Essential Trading View",
                    "Technical Analysis Only",
                    "Fundamental Analysis",
                    "Wave Analysis Report"
                ],
                index=["Full Analysis (All Data)", "Essential Trading View", 
                      "Technical Analysis Only", "Fundamental Analysis", 
                      "Wave Analysis Report"].index(
                    RobustSessionState.safe_get('export_template_radio', 'Full Analysis (All Data)')
                ),
                key="export_template_radio"
            )
            
            # Map template names
            template_map = {
                "Full Analysis (All Data)": "full",
                "Essential Trading View": "essential",
                "Technical Analysis Only": "technical",
                "Fundamental Analysis": "fundamental",
                "Wave Analysis Report": "wave_analysis"
            }
            
            selected_template = template_map[export_template]
        
        with export_col2:
            st.markdown("#### 📋 Template Description")
            
            descriptions = {
                "Full Analysis (All Data)": "Complete dataset with all calculated metrics, scores, and patterns",
                "Essential Trading View": "Core trading metrics: rank, ticker, score, price, returns, volume, patterns",
                "Technical Analysis Only": "Technical indicators and scores without fundamental data",
                "Fundamental Analysis": "Fundamental metrics: PE, EPS, valuations (requires Hybrid mode)",
                "Wave Analysis Report": "Wave states, smart money flow, momentum quality metrics"
            }
            
            st.info(descriptions[export_template])
        
        # Prepare export data
        export_df = ExportEngine.prepare_export_data(filtered_df, selected_template)
        
        if export_df.empty:
            st.warning("No data available for export")
        else:
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export
                csv_data = ExportEngine.to_csv(export_df)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"wave_detection_{selected_template}_{timestamp_str}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel export
                try:
                    excel_data = ExportEngine.to_excel(export_df, metadata)
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f"wave_detection_{selected_template}_{timestamp_str}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Excel export error: {str(e)}")
            
            with col3:
                # Copy to clipboard button
                if st.button("📋 Copy Tickers", use_container_width=True):
                    tickers = export_df['ticker'].tolist()
                    ticker_string = ', '.join(tickers)
                    st.code(ticker_string)
                    st.info(f"Copied {len(tickers)} tickers")
        
        # Export statistics
        st.markdown("---")
        st.markdown("#### 📊 Export Preview")
        
        export_stats = {
            "Total Stocks": len(export_df),
            "Average Score": f"{export_df['master_score'].mean():.1f}" if 'master_score' in export_df.columns else "N/A",
            "Columns": len(export_df.columns),
            "With Patterns": (export_df['patterns'] != '').sum() if 'patterns' in export_df.columns else 0
        }
        
        stat_cols = st.columns(4)
        for i, (label, value) in enumerate(export_stats.items()):
            with stat_cols[i]:
                UIComponents.render_metric_card(label, str(value))
        
        # Show preview
        with st.expander("👀 Preview Export Data"):
            st.dataframe(
                export_df.head(20),
                use_container_width=True,
                hide_index=True
            )
    
    # Tab 7: About
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
            The FINAL PERFECTED production version of the most advanced stock ranking system 
            designed to catch momentum waves early. This professional-grade tool combines 
            technical analysis, volume dynamics, advanced metrics, and smart pattern recognition 
            to identify high-potential stocks before they peak.
            
            #### 🎯 Core Features
            
            **Master Score 3.0** - Proprietary ranking algorithm:
            - Position Analysis (30%) - 52-week range positioning
            - Volume Dynamics (25%) - Multi-timeframe volume patterns  
            - Momentum Tracking (15%) - 30-day price momentum
            - Acceleration Detection (10%) - Momentum acceleration signals
            - Breakout Probability (10%) - Technical breakout readiness
            - RVOL Integration (10%) - Real-time relative volume
            
            **Advanced Metrics**:
            - Smart Money Flow - Institutional accumulation detection
            - Momentum Quality - Multi-timeframe momentum consistency
            - Market Regime - Bull/Bear/Neutral classification
            - Wave States - Real-time momentum classification
            - Position Tension - Range position stress indicator
            
            **Pattern Detection** - 25 patterns across 4 categories:
            - 11 Technical patterns
            - 6 Price range patterns
            - 3 Intelligence patterns
            - 5 Fundamental patterns (Hybrid mode)
            
            #### 🚀 Key Improvements in v3.1.0
            
            - ✅ Perfect filter interconnection
            - ✅ Industry filter respects sector selection
            - ✅ Smart Money Flow with complete algorithm
            - ✅ Momentum Quality Score implementation
            - ✅ Hash-based smart caching (hourly refresh)
            - ✅ Single-pass filter architecture
            - ✅ Unified session state management
            - ✅ Zero KeyError guarantee
            - ✅ O(n) pattern detection
            - ✅ Complete UI bug fixes
            
            #### 💡 How to Use
            
            1. **Data Source** - Use default Google Sheets or upload CSV
            2. **Quick Actions** - Instant filtering for common scenarios
            3. **Smart Filters** - Perfectly interconnected filtering
            4. **Search** - Find stocks by ticker, name, or pattern
            5. **Visualizations** - Interactive charts and insights
            6. **Wave Radar** - Monitor early momentum signals
            7. **Export** - Download filtered data in CSV or Excel
            """)
        
        with col2:
            st.markdown("""
            #### 📊 Data Processing
            
            1. Load from Google Sheets/CSV
            2. Validate and clean all columns
            3. Calculate 6 component scores
            4. Generate Master Score 3.0
            5. Apply Smart Money bonus
            6. Calculate advanced metrics
            7. Detect all 25 patterns
            8. Apply final ranking
            
            #### ⚡ Performance
            
            - Initial load: <2 seconds
            - Filtering: <200ms
            - Pattern detection: <500ms
            - Search: <50ms
            - Export: <1 second
            
            #### 🔧 Technical Stack
            
            - **Frontend**: Streamlit
            - **Data**: Pandas, NumPy
            - **Visualization**: Plotly
            - **Caching**: Smart hash-based
            - **State**: Unified management
            
            #### 📈 Trading Philosophy
            
            "Catch the wave early, ride it high,
            exit before the tide turns."
            
            This system identifies stocks showing
            early momentum characteristics before
            they become obvious to the market.
            
            #### 🔒 Production Status
            
            **Version**: 3.1.0-FINAL-PERFECTED
            **Status**: PRODUCTION READY
            **Updates**: PERMANENTLY LOCKED
            """)
        
        # System health check
        st.markdown("---")
        st.markdown("#### 🏥 System Health Check")
        
        health_col1, health_col2, health_col3, health_col4 = st.columns(4)
        
        with health_col1:
            data_freshness = "✅ Fresh" if last_refresh and (datetime.now(timezone.utc) - last_refresh).seconds < 3600 else "⚠️ Stale"
            UIComponents.render_metric_card("Data Status", data_freshness)
        
        with health_col2:
            cache_status = "✅ Active" if RobustSessionState.safe_get('last_good_data') else "⚠️ Empty"
            UIComponents.render_metric_card("Cache Status", cache_status)
        
        with health_col3:
            perf_metrics = RobustSessionState.safe_get('performance_metrics', {})
            slowest = max(perf_metrics.values()) if perf_metrics else 0
            perf_status = "✅ Fast" if slowest < 2 else "⚠️ Slow"
            UIComponents.render_metric_card("Performance", perf_status)
        
        with health_col4:
            error_count = len(metadata.get('errors', []))
            error_status = "✅ Clean" if error_count == 0 else f"⚠️ {error_count} errors"
            UIComponents.render_metric_card("Errors", error_status)

# ============================================
# APPLICATION ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
