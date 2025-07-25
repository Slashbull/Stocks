"""
Wave Detection Ultimate 3.0 - Utilities Module
===============================================
Helper functions and utilities for the stock ranking system.
"""

import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Any, Union
from config import logger, DATA_CONFIG

def timer(func):
    """Performance timing decorator with logging and session state storage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            # Log slow operations
            if elapsed > 1.0:
                logger.warning(f"{func.__name__} took {elapsed:.2f}s")
            
            # Store timing for performance monitoring
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            st.session_state.performance_metrics[func.__name__] = elapsed
            
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
            raise
    return wrapper

class DataValidator:
    """Comprehensive data validation utilities"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> bool:
        """Validate dataframe has required columns and data"""
        if df is None or df.empty:
            logger.error(f"{context}: Empty or None dataframe")
            return False
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"{context}: Missing required columns: {missing_cols}")
        
        logger.info(f"{context}: Found {len(df.columns)} columns, {len(df)} rows")
        
        # Calculate and store data quality metrics
        completeness = df.notna().sum().sum() / (len(df) * len(df.columns))
        DataValidator._update_data_quality_metrics(df, completeness)
        
        return True
    
    @staticmethod
    def _update_data_quality_metrics(df: pd.DataFrame, completeness: float):
        """Update session state with data quality metrics"""
        if 'data_quality' not in st.session_state:
            st.session_state.data_quality = {}
        
        quality_metrics = st.session_state.data_quality
        quality_metrics['completeness'] = completeness
        quality_metrics['total_rows'] = len(df)
        quality_metrics['total_columns'] = len(df.columns)
        
        # Calculate coverage for important columns
        if 'pe' in df.columns:
            pe_coverage = df['pe'].notna().sum()
            quality_metrics['pe_coverage'] = pe_coverage
        
        if 'eps_change_pct' in df.columns:
            eps_coverage = df['eps_change_pct'].notna().sum()
            quality_metrics['eps_coverage'] = eps_coverage
        
        # Calculate freshness (placeholder - would need timestamp data)
        quality_metrics['freshness'] = 95.0  # Assume good freshness
    
    @staticmethod
    def validate_numeric_column(series: pd.Series, col_name: str, 
                              min_val: Optional[float] = None, 
                              max_val: Optional[float] = None) -> pd.Series:
        """Validate and clean numeric column with bounds checking"""
        if series is None:
            return pd.Series(dtype=float)
        
        # Convert to numeric, coercing errors
        series = pd.to_numeric(series, errors='coerce')
        
        # Apply bounds if specified
        if min_val is not None:
            series = series.clip(lower=min_val)
        if max_val is not None:
            series = series.clip(upper=max_val)
        
        # Log quality issues
        nan_pct = series.isna().sum() / len(series) * 100
        if nan_pct > 50:
            logger.warning(f"{col_name}: {nan_pct:.1f}% NaN values")
        
        return series

class NumericCleaner:
    """Efficient numeric data cleaning utilities"""
    
    @staticmethod
    def clean_indian_number_format(value: Any) -> Optional[float]:
        """Clean and convert Indian number format to float - optimized"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            # Convert to string and clean
            cleaned = str(value).strip()
            
            # Quick check for invalid values
            invalid_values = {'', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None', '#VALUE!', '#ERROR!'}
            if cleaned in invalid_values:
                return np.nan
            
            # Remove currency symbols and special characters efficiently
            cleaned = cleaned.replace('₹', '').replace('$', '').replace('%', '').replace(',', '').replace(' ', '')
            
            # Handle 'Cr' (Crores) and 'L' (Lakhs) suffixes
            if cleaned.endswith('Cr'):
                return float(cleaned[:-2]) * 10_000_000  # 1 Crore = 10M
            elif cleaned.endswith('L'):
                return float(cleaned[:-1]) * 100_000     # 1 Lakh = 100K
            
            return float(cleaned)
            
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    @staticmethod
    def safe_percentage_to_float(value: Any) -> Optional[float]:
        """Convert percentage string to float"""
        if pd.isna(value):
            return np.nan
        
        try:
            if isinstance(value, str) and value.endswith('%'):
                return float(value[:-1])
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    
    @staticmethod
    def batch_clean_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Efficiently clean multiple numeric columns"""
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df_cleaned.columns:
                if col.endswith('_pct') or col.startswith('ret_'):
                    df_cleaned[col] = df_cleaned[col].apply(NumericCleaner.safe_percentage_to_float)
                else:
                    df_cleaned[col] = df_cleaned[col].apply(NumericCleaner.clean_indian_number_format)
                
                # Validate the cleaned column
                df_cleaned[col] = DataValidator.validate_numeric_column(df_cleaned[col], col)
        
        return df_cleaned

class TierAssigner:
    """Utility for assigning categorical tiers to numeric data"""
    
    @staticmethod
    def assign_tier(value: float, tier_config: Dict[str, tuple], default: str = "Unknown") -> str:
        """Assign a tier based on value and tier configuration"""
        if pd.isna(value) or np.isinf(value):
            return default
        
        for tier_name, (min_val, max_val) in tier_config.items():
            if min_val <= value < max_val:
                return tier_name
        
        return default
    
    @staticmethod
    def batch_assign_tiers(df: pd.DataFrame, tier_configs: Dict[str, Dict[str, tuple]]) -> pd.DataFrame:
        """Assign tiers for multiple columns efficiently"""
        df_tiered = df.copy()
        
        for col_base, tier_config in tier_configs.items():
            if col_base in df_tiered.columns:
                tier_col = f"{col_base}_tier"
                df_tiered[tier_col] = df_tiered[col_base].apply(
                    lambda x: TierAssigner.assign_tier(x, tier_config)
                )
        
        return df_tiered

class RankingUtils:
    """Utilities for ranking calculations"""
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safely rank a series with proper handling of edge cases"""
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        # Create a copy to avoid modifying original
        series = series.copy()
        
        # Replace inf values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Count valid values
        valid_count = series.notna().sum()
        if valid_count == 0:
            # Add small random variation to avoid identical scores
            return pd.Series(50 + np.random.uniform(-2, 2, size=len(series)), index=series.index)
        
        # Calculate ranks
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom')
            ranks = ranks * 100
            # Fill NaN values with appropriate default
            ranks = ranks.fillna(0 if ascending else 100)
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
            ranks = ranks.fillna(valid_count + 1)
        
        return ranks
    
    @staticmethod
    def weighted_score(components: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
        """Calculate weighted score from components"""
        if not components:
            raise ValueError("No components provided for scoring")
        
        # Get index from first component
        index = next(iter(components.values())).index
        weighted_score = pd.Series(0.0, index=index)
        total_weight = 0
        
        for component_name, weight in weights.items():
            if component_name in components:
                component_values = components[component_name].fillna(50)  # Neutral score for missing
                weighted_score += component_values * weight
                total_weight += weight
            else:
                logger.warning(f"Missing component: {component_name}")
        
        # Normalize if weights don't sum to 1.0
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Total weight is {total_weight}, normalizing...")
            weighted_score = weighted_score / total_weight
        
        return weighted_score.clip(0, 100)

class SessionManager:
    """Utilities for managing Streamlit session state"""
    
    @staticmethod
    def initialize_session_defaults():
        """Initialize default session state values"""
        defaults = {
            'search_index': None,
            'last_refresh': datetime.now(),
            'user_preferences': {
                'default_top_n': 50,
                'display_mode': 'Technical',
                'last_filters': {}
            },
            'performance_metrics': {},
            'data_quality': {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def clear_filter_state():
        """Clear all filter-related session state"""
        filter_keys = [
            'category_filter', 'sector_filter', 'eps_tier_filter', 
            'pe_tier_filter', 'price_tier_filter'
        ]
        for key in filter_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def update_user_preferences(updates: Dict[str, Any]):
        """Update user preferences in session state"""
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        
        st.session_state.user_preferences.update(updates)

def calculate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data quality metrics"""
    if df.empty:
        return {'completeness': 0, 'total_rows': 0, 'total_columns': 0}
    
    quality_metrics = {
        'completeness': df.notna().sum().sum() / (len(df) * len(df.columns)),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'freshness': 95.0  # Placeholder - would need timestamp data
    }
    
    # Calculate coverage for important columns
    important_cols = ['pe', 'eps_change_pct', 'rvol', 'volume_1d']
    for col in important_cols:
        if col in df.columns:
            coverage = df[col].notna().sum()
            quality_metrics[f'{col}_coverage'] = coverage
    
    return quality_metrics

def format_number(value: float, format_type: str = 'auto') -> str:
    """Format numbers for display with appropriate units"""
    if pd.isna(value) or np.isinf(value):
        return "N/A"
    
    if format_type == 'currency':
        if value >= 1_00_00_000:  # 1 Crore
            return f"₹{value/1_00_00_000:.1f} Cr"
        elif value >= 1_00_000:  # 1 Lakh
            return f"₹{value/1_00_000:.1f} L"
        else:
            return f"₹{value:,.0f}"
    
    elif format_type == 'percentage':
        return f"{value:.1f}%"
    
    elif format_type == 'score':
        return f"{value:.1f}"
    
    elif format_type == 'auto':
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        else:
            return f"{value:.1f}"
    
    return str(value)

def get_score_color(score: float) -> str:
    """Get color based on score value"""
    if score >= 80:
        return '#2ecc71'  # Green
    elif score >= 60:
        return '#f39c12'  # Orange  
    elif score >= 40:
        return '#95a5a6'  # Gray
    else:
        return '#e74c3c'  # Red