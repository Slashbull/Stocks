"""
Wave Detection Ultimate 3.0 - Data Processing Module
====================================================
Comprehensive data processing for Indian stock market data with proper format handling.
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
import requests

from config import CONFIG, DATA_CONFIG, logger
from utils import (
    timer, DataValidator, NumericCleaner, TierAssigner, 
    calculate_data_quality
)

warnings.filterwarnings('ignore')

class DataLoader:
    """Handles data loading from various sources with robust error handling"""
    
    @staticmethod
    @timer
    def load_from_google_sheets(sheet_url: str, gid: str) -> pd.DataFrame:
        """Load data from Google Sheets with proper error handling"""
        try:
            # Validate inputs
            if not sheet_url or not gid:
                raise ValueError("Sheet URL and GID are required")
            
            # Construct CSV URL
            base_url = sheet_url.split('/edit')[0]
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from Google Sheets: {csv_url}")
            
            # Load with timeout and error handling
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            
            # Read CSV from response content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), low_memory=False)
            
            if df.empty:
                raise ValueError("Loaded empty dataframe")
            
            logger.info(f"Successfully loaded {len(df):,} rows with {len(df.columns)} columns")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error loading from Google Sheets: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to load data from Google Sheets: {str(e)}")
            raise
    
    @staticmethod
    @timer
    def load_from_csv(file_path: str) -> pd.DataFrame:
        """Load data from local CSV file"""
        try:
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"Loaded {len(df):,} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {str(e)}")
            raise

class IndianDataCleaner:
    """Specialized cleaner for Indian stock market data formats"""
    
    @staticmethod
    def clean_indian_currency(value: Any) -> Optional[float]:
        """Clean Indian currency format (₹1,23,456.78 or ₹1,234 Cr)"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            cleaned = str(value).strip()
            
            # Handle empty or invalid values
            invalid_values = {'', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None', '#VALUE!', '#ERROR!'}
            if cleaned in invalid_values:
                return np.nan
            
            # Remove currency symbols
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').strip()
            
            # Handle Indian numbering suffixes
            if cleaned.endswith('Cr'):
                return float(cleaned[:-2].strip()) * 1_00_00_000  # 1 Crore = 10M
            elif cleaned.endswith('L'):
                return float(cleaned[:-1].strip()) * 1_00_000     # 1 Lakh = 100K
            elif cleaned.endswith('K'):
                return float(cleaned[:-1].strip()) * 1_000        # 1 Thousand
            
            return float(cleaned)
            
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    @staticmethod
    def clean_percentage(value: Any) -> Optional[float]:
        """Clean percentage values (5.25% -> 5.25)"""
        if pd.isna(value):
            return np.nan
        
        try:
            cleaned = str(value).strip()
            
            if cleaned in {'', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None'}:
                return np.nan
            
            # Remove percentage sign
            if cleaned.endswith('%'):
                cleaned = cleaned[:-1].strip()
            
            return float(cleaned)
        except (ValueError, TypeError):
            return np.nan
    
    @staticmethod
    def clean_volume_data(value: Any) -> Optional[float]:
        """Clean volume data with proper handling of large numbers"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            cleaned = str(value).strip()
            
            if cleaned in {'', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None'}:
                return np.nan
            
            # Remove commas and handle scientific notation
            cleaned = cleaned.replace(',', '')
            
            # Convert to float
            result = float(cleaned)
            
            # Cap extreme values to prevent calculation issues
            if result > 1e12:  # Cap at 1 trillion
                return 1e12
            
            return result if result >= 0 else np.nan
            
        except (ValueError, TypeError):
            return np.nan

class StockDataProcessor:
    """Main processor for stock market data with comprehensive cleaning and validation"""
    
    def __init__(self):
        self.cleaner = IndianDataCleaner()
        self.validator = DataValidator()
        self.tier_assigner = TierAssigner()
    
    @timer
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete processing pipeline for stock data"""
        logger.info("Starting comprehensive data processing...")
        
        # Validate input
        if not self.validator.validate_dataframe(df, DATA_CONFIG.REQUIRED_COLUMNS, "Raw Data"):
            raise ValueError("Invalid input dataframe")
        
        # Create working copy
        processed_df = df.copy()
        
        # Step 1: Clean column names
        processed_df = self._clean_column_names(processed_df)
        
        # Step 2: Process different data types
        processed_df = self._process_price_data(processed_df)
        processed_df = self._process_return_data(processed_df)
        processed_df = self._process_volume_data(processed_df)
        processed_df = self._process_fundamental_data(processed_df)
        
        # Step 3: Handle missing values intelligently
        processed_df = self._handle_missing_values(processed_df)
        
        # Step 4: Create derived columns
        processed_df = self._create_derived_columns(processed_df)
        
        # Step 5: Assign tier classifications
        processed_df = self._assign_tiers(processed_df)
        
        # Step 6: Final validation and cleanup
        processed_df = self._final_cleanup(processed_df)
        
        logger.info(f"Data processing complete: {len(processed_df)} stocks processed")
        return processed_df
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        # Remove extra spaces and standardize naming
        df.columns = df.columns.str.strip().str.lower()
        
        # Map any alternate column names
        column_mapping = {
            'symbol': 'ticker',
            'company': 'company_name',
            'market_cap_cr': 'market_cap',
            'current_price': 'price'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        return df
    
    def _process_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all price-related columns"""
        price_columns = ['price', 'prev_close', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d']
        
        for col in price_columns:
            if col in df.columns:
                logger.debug(f"Processing price column: {col}")
                df[col] = df[col].apply(self.cleaner.clean_indian_currency)
                
                # Validate price data
                df[col] = self.validator.validate_numeric_column(
                    df[col], col, min_val=CONFIG.MIN_VALID_PRICE
                )
        
        return df
    
    def _process_return_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all return/percentage columns"""
        return_columns = [col for col in df.columns if col.startswith('ret_') or col.endswith('_pct')]
        return_columns.extend(['from_low_pct', 'from_high_pct'])
        
        for col in return_columns:
            if col in df.columns:
                logger.debug(f"Processing return column: {col}")
                df[col] = df[col].apply(self.cleaner.clean_percentage)
                
                # Cap extreme returns to prevent calculation issues
                if col.startswith('ret_'):
                    df[col] = df[col].clip(-99.9, 10000)  # -99.9% to 10,000%
                elif col == 'eps_change_pct':
                    df[col] = df[col].clip(-100, 50000)   # Cap EPS growth
        
        return df
    
    def _process_volume_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process volume and volume ratio columns"""
        volume_columns = [col for col in df.columns if 'volume' in col]
        volume_ratio_columns = [col for col in df.columns if 'vol_ratio' in col]
        
        # Clean volume data
        for col in volume_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.cleaner.clean_volume_data)
        
        # Clean volume ratios (these are percentages)
        for col in volume_ratio_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.cleaner.clean_percentage)
                # Convert to ratio (divide by 100)
                df[col] = df[col] / 100
        
        # Process RVOL
        if 'rvol' in df.columns:
            df['rvol'] = pd.to_numeric(df['rvol'], errors='coerce')
            df['rvol'] = df['rvol'].clip(0, CONFIG.RVOL_MAX_THRESHOLD)
        
        return df
    
    def _process_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process fundamental analysis columns (PE, EPS, etc.)"""
        # Process PE ratio
        if 'pe' in df.columns:
            df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
            # Cap extreme PE values
            df['pe'] = df['pe'].clip(0, 10000)
        
        # Process EPS data
        eps_columns = ['eps_current', 'eps_last_qtr']
        for col in eps_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process EPS change percentage
        if 'eps_change_pct' in df.columns:
            df['eps_change_pct'] = df['eps_change_pct'].apply(self.cleaner.clean_percentage)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent handling of missing values based on data type"""
        
        # For categorical data, fill with 'Unknown'
        categorical_cols = ['category', 'sector']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # For price data, forward fill from similar columns when possible
        if 'price' in df.columns and 'prev_close' in df.columns:
            df['price'] = df['price'].fillna(df['prev_close'])
        
        # For volume ratios, fill with 1.0 (neutral ratio)
        vol_ratio_cols = [col for col in df.columns if 'vol_ratio' in col]
        for col in vol_ratio_cols:
            df[col] = df[col].fillna(1.0)
        
        # For RVOL, fill with 1.0 (average volume)
        if 'rvol' in df.columns:
            df['rvol'] = df['rvol'].fillna(1.0)
        
        return df
    
    def _create_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create useful derived columns"""
        
        # Calculate market cap numeric value for sorting
        if 'market_cap' in df.columns:
            df['market_cap_numeric'] = df['market_cap'].apply(
                lambda x: self.cleaner.clean_indian_currency(x) if pd.notna(x) else np.nan
            )
        
        # Calculate price change from previous close
        if 'price' in df.columns and 'prev_close' in df.columns:
            df['price_change'] = df['price'] - df['prev_close']
            df['price_change_pct'] = ((df['price'] - df['prev_close']) / df['prev_close'] * 100).round(2)
        
        # Calculate average volume for better analysis
        volume_cols = ['volume_7d', 'volume_30d', 'volume_90d']
        available_vol_cols = [col for col in volume_cols if col in df.columns]
        if available_vol_cols:
            df['avg_volume'] = df[available_vol_cols].mean(axis=1, skipna=True)
        
        # Calculate position in 52-week range
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            df['position_52w'] = ((df['price'] - df['low_52w']) / 
                                 (df['high_52w'] - df['low_52w']) * 100).round(1)
        
        return df
    
    def _assign_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign tier classifications for filtering"""
        
        # Assign tiers using configuration
        tier_configs = {
            'pe': CONFIG.TIERS['pe'],
            'price': CONFIG.TIERS['price']
        }
        
        # Add EPS tier based on EPS change
        if 'eps_change_pct' in df.columns:
            tier_configs['eps_change_pct'] = CONFIG.TIERS['eps']
        
        df = self.tier_assigner.batch_assign_tiers(df, tier_configs)
        
        return df
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup"""
        
        # Remove any duplicate tickers
        if 'ticker' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['ticker'], keep='first')
            removed = initial_count - len(df)
            if removed > 0:
                logger.warning(f"Removed {removed} duplicate tickers")
        
        # Ensure numeric columns are properly typed
        numeric_cols = [col for col in DATA_CONFIG.NUMERIC_COLUMNS if col in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add data quality metrics
        df['data_completeness'] = df.notna().sum(axis=1) / len(df.columns) * 100
        
        # Sort by market cap (largest first) for consistent ordering
        if 'market_cap_numeric' in df.columns:
            df = df.sort_values('market_cap_numeric', ascending=False, na_position='last')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df

# Factory function for easy use
@timer
def load_and_process_data(sheet_url: str = None, gid: str = None, csv_file: str = None) -> Tuple[pd.DataFrame, datetime]:
    """
    Load and process stock data from Google Sheets or CSV file
    
    Args:
        sheet_url: Google Sheets URL (optional)
        gid: Google Sheets GID (optional) 
        csv_file: Local CSV file path (optional)
    
    Returns:
        Tuple of (processed_dataframe, timestamp)
    """
    try:
        # Load data
        if csv_file:
            raw_df = DataLoader.load_from_csv(csv_file)
        elif sheet_url and gid:
            raw_df = DataLoader.load_from_google_sheets(sheet_url, gid)
        else:
            # Use default configuration
            raw_df = DataLoader.load_from_google_sheets(CONFIG.DEFAULT_SHEET_URL, CONFIG.DEFAULT_GID)
        
        # Process data
        processor = StockDataProcessor()
        processed_df = processor.process_dataframe(raw_df)
        
        # Validate final result
        if processed_df.empty:
            raise ValueError("Processing resulted in empty dataframe")
        
        return processed_df, datetime.now()
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        raise

# Streamlit cached version
@st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False)
def cached_load_and_process_data(sheet_url: str, gid: str) -> Tuple[pd.DataFrame, datetime]:
    """Cached version for Streamlit"""
    return load_and_process_data(sheet_url, gid)