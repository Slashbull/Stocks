"""
Wave Detection Ultimate 3.0 - Data Pipeline
===========================================
Production-Ready Data Loading & Processing Infrastructure
Google Sheets integration, data validation, and quality monitoring

Version: 3.0.6-PRODUCTION-BULLETPROOF
Status: PRODUCTION READY - Zero Data Corruption Tolerance
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import warnings
from functools import wraps
from io import StringIO
import requests
from urllib.parse import urlparse

# Import core engine for configuration and scoring
from core_engine import CONFIG, SafeMath, MasterScoringEngine

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# ============================================
# PERFORMANCE MONITORING
# ============================================

def performance_timer(func):
    """Production-grade performance timing decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            
            # Log slow operations
            if elapsed > 2.0:
                logger.warning(f"{func.__name__} took {elapsed:.2f}s (slow)")
            elif elapsed > 5.0:
                logger.error(f"{func.__name__} took {elapsed:.2f}s (critical)")
            
            # Store metrics for monitoring
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            st.session_state.performance_metrics[func.__name__] = elapsed
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
            raise
            
    return wrapper

# ============================================
# DATA VALIDATION ENGINE
# ============================================

class ProductionDataValidator:
    """Production-grade data validation with comprehensive checks"""
    
    # Define expected columns and their types
    REQUIRED_COLUMNS = ['ticker', 'price']
    
    NUMERIC_COLUMNS = [
        'price', 'prev_close', 'low_52w', 'high_52w',
        'from_low_pct', 'from_high_pct',
        'sma_20d', 'sma_50d', 'sma_200d',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
        'vol_ratio_90d_180d',
        'rvol', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'
    ]
    
    CATEGORICAL_COLUMNS = ['ticker', 'company_name', 'category', 'sector']
    
    PERCENTAGE_COLUMNS = [
        'from_low_pct', 'from_high_pct',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ]
    
    VOLUME_RATIO_COLUMNS = [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ]
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, context: str) -> bool:
        """Comprehensive dataframe validation"""
        try:
            if df is None:
                logger.error(f"{context}: DataFrame is None")
                return False
            
            if df.empty:
                logger.error(f"{context}: DataFrame is empty")
                return False
            
            # Check required columns
            missing_required = [col for col in ProductionDataValidator.REQUIRED_COLUMNS 
                              if col not in df.columns]
            if missing_required:
                logger.error(f"{context}: Missing required columns: {missing_required}")
                return False
            
            # Log data overview
            logger.info(f"{context}: {len(df)} rows, {len(df.columns)} columns")
            
            # Calculate and store data quality metrics
            ProductionDataValidator._calculate_quality_metrics(df, context)
            
            return True
            
        except Exception as e:
            logger.error(f"{context}: Validation error: {str(e)}")
            return False
    
    @staticmethod
    def validate_numeric_column(series: pd.Series, col_name: str, 
                              min_val: Optional[float] = None, 
                              max_val: Optional[float] = None) -> pd.Series:
        """Bulletproof numeric column validation"""
        try:
            if series is None or series.empty:
                return pd.Series(dtype=float)
            
            # Convert to numeric with comprehensive error handling
            original_count = len(series)
            series_clean = pd.to_numeric(series, errors='coerce')
            
            # Apply bounds if specified
            if min_val is not None:
                out_of_bounds = series_clean < min_val
                if out_of_bounds.any():
                    logger.warning(f"{col_name}: {out_of_bounds.sum()} values below minimum {min_val}")
                series_clean = series_clean.clip(lower=min_val)
            
            if max_val is not None:
                out_of_bounds = series_clean > max_val
                if out_of_bounds.any():
                    logger.warning(f"{col_name}: {out_of_bounds.sum()} values above maximum {max_val}")
                series_clean = series_clean.clip(upper=max_val)
            
            # Check for data quality issues
            nan_count = series_clean.isna().sum()
            nan_pct = (nan_count / original_count) * 100 if original_count > 0 else 0
            
            if nan_pct > 50:
                logger.warning(f"{col_name}: High NaN percentage: {nan_pct:.1f}%")
            elif nan_pct > 0:
                logger.info(f"{col_name}: {nan_pct:.1f}% NaN values")
            
            return series_clean
            
        except Exception as e:
            logger.error(f"Error validating {col_name}: {str(e)}")
            return pd.Series(0.0, index=series.index if series is not None else [])
    
    @staticmethod
    def _calculate_quality_metrics(df: pd.DataFrame, context: str):
        """Calculate and store comprehensive data quality metrics"""
        try:
            if 'data_quality' not in st.session_state:
                st.session_state.data_quality = {}
            
            # Overall completeness
            total_cells = len(df) * len(df.columns)
            filled_cells = df.notna().sum().sum()
            completeness = (filled_cells / total_cells) * 100 if total_cells > 0 else 0
            
            # Column-specific completeness
            column_completeness = {}
            for col in df.columns:
                if col in ProductionDataValidator.NUMERIC_COLUMNS:
                    non_na = df[col].notna().sum()
                    completeness_pct = (non_na / len(df)) * 100 if len(df) > 0 else 0
                    column_completeness[col] = completeness_pct
            
            # Price data freshness (if available)
            freshness = 0
            if all(col in df.columns for col in ['price', 'prev_close']):
                price_changes = (df['price'] != df['prev_close']).sum()
                freshness = (price_changes / len(df)) * 100 if len(df) > 0 else 0
            
            # Store metrics
            st.session_state.data_quality.update({
                'context': context,
                'completeness': completeness,
                'column_completeness': column_completeness,
                'freshness': freshness,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'last_update': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {str(e)}")

# ============================================
# DATA CLEANING ENGINE
# ============================================

class ProductionDataCleaner:
    """Production-grade data cleaning with bulletproof transformations"""
    
    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False, 
                          is_volume_ratio: bool = False) -> float:
        """Bulletproof numeric value cleaning for Indian data formats"""
        try:
            # Handle None and empty values
            if value is None or value == '' or pd.isna(value):
                return np.nan
            
            # Convert to string for processing
            try:
                cleaned = str(value).strip()
            except (AttributeError, TypeError):
                return np.nan
            
            # Handle known invalid values
            invalid_values = {'', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None', 
                            '#VALUE!', '#ERROR!', '#DIV/0!', 'null', 'NULL'}
            if cleaned.lower() in [v.lower() for v in invalid_values]:
                return np.nan
            
            # Handle scientific notation
            if 'e' in cleaned.lower():
                try:
                    return float(cleaned)
                except (ValueError, OverflowError):
                    return np.nan
            
            # Remove currency symbols and formatting
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '')
            
            # Handle percentage signs
            has_percent = '%' in cleaned
            cleaned = cleaned.replace('%', '')
            
            # Convert to float
            try:
                result = float(cleaned)
            except (ValueError, OverflowError):
                return np.nan
            
            # Handle percentage data
            if is_percentage and not has_percent:
                # Data is already in percentage format (e.g., -56.61 for -56.61%)
                pass  # No conversion needed
            elif has_percent and not is_percentage:
                # Convert percentage to decimal if needed
                result = result / 100
            
            # Validate result
            if np.isinf(result):
                return np.nan
            
            # Apply reasonable bounds
            if is_percentage:
                result = max(-10000, min(100000, result))  # Allow extreme but not infinite
            elif is_volume_ratio:
                result = max(0.001, min(1000000, result))  # Positive ratios only
            
            return result
            
        except Exception as e:
            logger.debug(f"Error cleaning value '{value}': {str(e)}")
            return np.nan
    
    @staticmethod
    @performance_timer
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete dataframe processing with comprehensive error handling"""
        try:
            if not ProductionDataValidator.validate_dataframe(df, "Raw input"):
                logger.error("Input validation failed")
                return pd.DataFrame()
            
            # Create working copy
            df_processed = df.copy()
            initial_count = len(df_processed)
            
            logger.info(f"Processing {initial_count} rows...")
            
            # Process numeric columns with appropriate handling
            ProductionDataCleaner._process_numeric_columns(df_processed)
            
            # Process categorical columns
            ProductionDataCleaner._process_categorical_columns(df_processed)
            
            # Handle volume ratios conversion
            ProductionDataCleaner._process_volume_ratios(df_processed)
            
            # Data quality filtering
            df_processed = ProductionDataCleaner._apply_quality_filters(df_processed)
            
            # Add calculated fields
            df_processed = ProductionDataCleaner._add_calculated_fields(df_processed)
            
            # Remove duplicates
            df_processed = ProductionDataCleaner._remove_duplicates(df_processed)
            
            final_count = len(df_processed)
            removed_count = initial_count - final_count
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} invalid/duplicate rows ({removed_count/initial_count*100:.1f}%)")
            
            logger.info(f"Data processing complete: {final_count} valid stocks")
            return df_processed
            
        except Exception as e:
            logger.error(f"Critical error in data processing: {str(e)}")
            # Return minimal valid dataframe to prevent complete failure
            return pd.DataFrame(columns=ProductionDataValidator.REQUIRED_COLUMNS)
    
    @staticmethod
    def _process_numeric_columns(df: pd.DataFrame):
        """Process all numeric columns with type-specific handling"""
        try:
            for col in ProductionDataValidator.NUMERIC_COLUMNS:
                if col in df.columns:
                    is_pct = col in ProductionDataValidator.PERCENTAGE_COLUMNS
                    is_vol_ratio = col in ProductionDataValidator.VOLUME_RATIO_COLUMNS
                    
                    # Clean values
                    df[col] = df[col].apply(
                        lambda x: ProductionDataCleaner.clean_numeric_value(
                            x, is_percentage=is_pct, is_volume_ratio=is_vol_ratio
                        )
                    )
                    
                    # Apply column-specific validation
                    if col == 'price':
                        df[col] = ProductionDataValidator.validate_numeric_column(
                            df[col], col, min_val=CONFIG.MIN_VALID_PRICE
                        )
                    elif col == 'pe':
                        df[col] = ProductionDataValidator.validate_numeric_column(
                            df[col], col, min_val=0, max_val=CONFIG.MAX_VALID_PE
                        )
                    elif col == 'eps_change_pct':
                        df[col] = ProductionDataValidator.validate_numeric_column(
                            df[col], col, min_val=-1000, max_val=CONFIG.MAX_VALID_EPS_CHANGE
                        )
                    else:
                        df[col] = ProductionDataValidator.validate_numeric_column(df[col], col)
                        
        except Exception as e:
            logger.error(f"Error processing numeric columns: {str(e)}")
    
    @staticmethod
    def _process_categorical_columns(df: pd.DataFrame):
        """Process categorical columns with standardization"""
        try:
            for col in ProductionDataCleaner.CATEGORICAL_COLUMNS:
                if col in df.columns:
                    # Convert to string and clean
                    df[col] = df[col].astype(str).str.strip()
                    
                    # Replace various null representations
                    null_values = ['nan', 'None', '', 'N/A', 'NaN', 'null', 'NULL']
                    df[col] = df[col].replace(null_values, 'Unknown')
                    
                    # Clean whitespace
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                    
                    # Capitalize properly for company names
                    if col == 'company_name':
                        df[col] = df[col].str.title()
                        
        except Exception as e:
            logger.error(f"Error processing categorical columns: {str(e)}")
    
    @staticmethod
    def _process_volume_ratios(df: pd.DataFrame):
        """Convert volume ratio percentages to actual ratios"""
        try:
            for col in ProductionDataCleaner.VOLUME_RATIO_COLUMNS:
                if col in df.columns:
                    # Convert percentage change to ratio
                    # -56.61% change means 43.39% of original = 0.4339 ratio
                    df[col] = (100 + df[col].fillna(0)) / 100
                    df[col] = df[col].fillna(1.0).clip(0.01, 100.0)
                    
        except Exception as e:
            logger.error(f"Error processing volume ratios: {str(e)}")
    
    @staticmethod
    def _apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters"""
        try:
            # Remove rows with critical missing data
            essential_cols = ['ticker', 'price']
            for col in essential_cols:
                if col in df.columns:
                    df = df.dropna(subset=[col])
            
            # Remove invalid prices
            if 'price' in df.columns:
                df = df[df['price'] > CONFIG.MIN_VALID_PRICE]
            
            # Fill missing position data with reasonable defaults
            if 'from_low_pct' in df.columns:
                df['from_low_pct'] = df['from_low_pct'].fillna(50)
            else:
                df['from_low_pct'] = 50
                
            if 'from_high_pct' in df.columns:
                df['from_high_pct'] = df['from_high_pct'].fillna(-50)
            else:
                df['from_high_pct'] = -50
            
            # Handle RVOL
            if 'rvol' in df.columns:
                df['rvol'] = df['rvol'].fillna(1.0).clip(lower=0.01)
            else:
                df['rvol'] = 1.0
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying quality filters: {str(e)}")
            return df
    
    @staticmethod
    def _add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields safely"""
        try:
            # Add year column if not present
            if 'year' not in df.columns:
                df['year'] = datetime.now().year
            
            # Ensure all required columns exist
            required_for_scoring = ['from_low_pct', 'from_high_pct', 'rvol']
            for col in required_for_scoring:
                if col not in df.columns:
                    if col == 'rvol':
                        df[col] = 1.0
                    elif col == 'from_low_pct':
                        df[col] = 50.0
                    elif col == 'from_high_pct':
                        df[col] = -50.0
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding calculated fields: {str(e)}")
            return df
    
    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate tickers safely"""
        try:
            if 'ticker' in df.columns:
                initial_count = len(df)
                df = df.drop_duplicates(subset=['ticker'], keep='first')
                removed = initial_count - len(df)
                if removed > 0:
                    logger.info(f"Removed {removed} duplicate tickers")
            return df
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}")
            return df

# ============================================
# DATA LOADING ENGINE
# ============================================

class ProductionDataLoader:
    """Production-grade data loading with bulletproof error handling"""
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False, max_entries=10)
    def load_from_google_sheets(sheet_url: str, gid: str) -> pd.DataFrame:
        """Load data from Google Sheets with comprehensive error handling"""
        try:
            # Validate inputs
            if not sheet_url or not gid:
                raise ValueError("Sheet URL and GID are required")
            
            # Validate URL format
            parsed_url = urlparse(sheet_url)
            if not parsed_url.netloc:
                raise ValueError(f"Invalid sheet URL: {sheet_url}")
            
            # Construct CSV URL
            base_url = sheet_url.split('/edit')[0]
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from Google Sheets: {csv_url[:100]}...")
            
            # Load with timeout and retries
            max_retries = 3
            timeout = 30
            
            for attempt in range(max_retries):
                try:
                    # Use requests for better control
                    response = requests.get(csv_url, timeout=timeout)
                    response.raise_for_status()
                    
                    # Parse CSV
                    csv_data = StringIO(response.text)
                    df = pd.read_csv(csv_data, low_memory=False)
                    
                    if df.empty:
                        raise ValueError("Loaded empty dataframe")
                    
                    logger.info(f"Successfully loaded {len(df):,} rows with {len(df.columns)} columns")
                    return df
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed on attempt {attempt + 1}/{max_retries}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
            
            raise Exception("All retry attempts failed")
            
        except Exception as e:
            logger.error(f"Failed to load from Google Sheets: {str(e)}")
            # Try to return cached data if available
            if 'last_good_data' in st.session_state:
                logger.info("Returning cached data as fallback")
                cached_df, _ = st.session_state.last_good_data
                return cached_df
            raise
    
    @staticmethod
    @performance_timer
    def load_and_process_data(sheet_url: str = None, gid: str = None) -> Tuple[pd.DataFrame, datetime]:
        """Main data loading and processing pipeline"""
        try:
            # Use default values if not provided
            sheet_url = sheet_url or CONFIG.DEFAULT_SHEET_URL
            gid = gid or CONFIG.DEFAULT_GID
            
            # Record start time
            start_time = time.perf_counter()
            
            logger.info("Starting data pipeline...")
            
            # Load raw data
            raw_df = ProductionDataLoader.load_from_google_sheets(sheet_url, gid)
            
            # Process data
            processed_df = ProductionDataCleaner.process_dataframe(raw_df)
            
            if processed_df.empty:
                raise ValueError("Data processing resulted in empty dataframe")
            
            # Calculate scores
            scored_df = MasterScoringEngine.calculate_all_scores(processed_df)
            
            # Store as last good data
            timestamp = datetime.now()
            st.session_state.last_good_data = (scored_df.copy(), timestamp)
            
            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            logger.info(f"Data pipeline complete: {processing_time:.2f}s")
            
            return scored_df, timestamp
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {str(e)}")
            
            # Try to return last good data
            if 'last_good_data' in st.session_state:
                logger.info("Returning last good data due to pipeline failure")
                return st.session_state.last_good_data
            
            # Return empty dataframe with proper structure
            empty_df = pd.DataFrame(columns=['ticker', 'company_name', 'master_score', 'rank'])
            return empty_df, datetime.now()

# ============================================
# CACHE MANAGEMENT
# ============================================

class CacheManager:
    """Production-grade cache management with cleanup"""
    
    @staticmethod
    def cleanup_session_state():
        """Clean up session state to prevent memory bloat"""
        try:
            # Clean up performance metrics (keep only last 10 entries)
            if 'performance_metrics' in st.session_state:
                metrics = st.session_state.performance_metrics
                if len(metrics) > 10:
                    # Keep only the 10 most recent entries
                    recent_metrics = dict(list(metrics.items())[-10:])
                    st.session_state.performance_metrics = recent_metrics
            
            # Clean up old data quality entries
            if 'data_quality' in st.session_state:
                quality = st.session_state.data_quality
                if 'last_update' in quality:
                    # Remove if older than 2 hours
                    if (datetime.now() - quality['last_update']).seconds > 7200:
                        del st.session_state.data_quality
            
            # Clean up temporary variables
            temp_keys = [key for key in st.session_state.keys() 
                        if key.startswith('temp_') or key.startswith('_temp')]
            for key in temp_keys:
                del st.session_state[key]
                
        except Exception as e:
            logger.error(f"Error in cache cleanup: {str(e)}")
    
    @staticmethod
    def clear_all_caches():
        """Clear all Streamlit caches"""
        try:
            st.cache_data.clear()
            logger.info("All caches cleared")
        except Exception as e:
            logger.error(f"Error clearing caches: {str(e)}")
    
    @staticmethod
    def get_cache_info() -> Dict[str, Any]:
        """Get cache information for monitoring"""
        try:
            info = {
                'last_data_load': None,
                'cache_size': 0,
                'session_keys': len(st.session_state.keys()),
                'performance_entries': 0
            }
            
            if 'last_good_data' in st.session_state:
                _, timestamp = st.session_state.last_good_data
                info['last_data_load'] = timestamp
            
            if 'performance_metrics' in st.session_state:
                info['performance_entries'] = len(st.session_state.performance_metrics)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
            return {'error': str(e)}

# ============================================
# DATA QUALITY MONITORING
# ============================================

def calculate_comprehensive_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data quality metrics for monitoring"""
    try:
        if df.empty:
            return {'error': 'Empty dataframe'}
        
        quality_metrics = {
            'timestamp': datetime.now(),
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        
        # Overall completeness
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        quality_metrics['completeness'] = (filled_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Critical column completeness
        critical_columns = ['ticker', 'price', 'master_score']
        critical_completeness = {}
        for col in critical_columns:
            if col in df.columns:
                completeness = (df[col].notna().sum() / len(df)) * 100
                critical_completeness[col] = completeness
        quality_metrics['critical_completeness'] = critical_completeness
        
        # Data freshness (price changes)
        if all(col in df.columns for col in ['price', 'prev_close']):
            price_changes = (df['price'] != df['prev_close']).sum()
            quality_metrics['freshness'] = (price_changes / len(df)) * 100
        else:
            quality_metrics['freshness'] = 0
        
        # Fundamental data coverage
        if 'pe' in df.columns:
            valid_pe = (df['pe'].notna() & (df['pe'] > 0) & ~np.isinf(df['pe'])).sum()
            quality_metrics['pe_coverage'] = valid_pe
        else:
            quality_metrics['pe_coverage'] = 0
            
        if 'eps_change_pct' in df.columns:
            valid_eps = (df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])).sum()
            quality_metrics['eps_coverage'] = valid_eps
        else:
            quality_metrics['eps_coverage'] = 0
        
        # Volume data quality
        volume_cols = ['volume_1d', 'volume_30d', 'volume_90d']
        volume_quality = 0
        volume_count = 0
        for col in volume_cols:
            if col in df.columns:
                volume_quality += df[col].notna().sum()
                volume_count += len(df)
        quality_metrics['volume_coverage'] = (volume_quality / volume_count * 100) if volume_count > 0 else 0
        
        # Score distribution quality
        if 'master_score' in df.columns:
            scores = df['master_score'].dropna()
            if len(scores) > 0:
                quality_metrics['score_distribution'] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'valid_count': len(scores)
                }
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Error calculating data quality: {str(e)}")
        return {'error': str(e), 'timestamp': datetime.now()}
