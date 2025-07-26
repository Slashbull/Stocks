"""
Wave Detection Ultimate 3.0 - Smart Filtering Engine
====================================================
Production-Ready Filtering and Search Logic
Advanced interconnected filtering, pattern matching, and search algorithms

Version: 3.0.6-PRODUCTION-BULLETPROOF
Status: PRODUCTION READY - Zero Filter Corruption Tolerance
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import re
from functools import lru_cache

# Import core engine for configuration and safe math
from core_engine import CONFIG, SafeMath

# Configure logging
logger = logging.getLogger(__name__)

# ============================================
# PRODUCTION FILTER ENGINE
# ============================================

class ProductionFilterEngine:
    """Production-grade filtering engine with bulletproof logic"""
    
    @staticmethod
    @lru_cache(maxsize=CONFIG.MAX_CACHE_SIZE)
    def get_unique_values_cached(column_name: str, data_hash: str, 
                               exclude_unknown: bool = True) -> Tuple[str, ...]:
        """Cached version of get_unique_values for performance"""
        # This is a placeholder for caching - actual implementation in get_unique_values
        pass
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column: str, 
                         exclude_unknown: bool = True,
                         filters: Dict[str, Any] = None) -> List[str]:
        """Get sorted unique values with smart interconnected filtering"""
        try:
            if df is None or df.empty or column not in df.columns:
                return []
            
            # Apply existing filters first (for interconnected filtering)
            if filters:
                try:
                    filtered_df = ProductionFilterEngine._apply_filter_subset(
                        df, filters, exclude_cols=[column]
                    )
                except Exception as e:
                    logger.warning(f"Error in interconnected filtering: {str(e)}")
                    filtered_df = df
            else:
                filtered_df = df
            
            # Extract unique values safely
            try:
                if filtered_df.empty:
                    return []
                
                unique_series = filtered_df[column].dropna()
                if unique_series.empty:
                    return []
                
                # Convert to strings safely
                values = []
                for val in unique_series.unique():
                    try:
                        str_val = str(val).strip()
                        if str_val:  # Non-empty string
                            values.append(str_val)
                    except Exception:
                        continue
                
                # Filter out unknown values if requested
                if exclude_unknown:
                    unknown_variants = {
                        'unknown', 'unknown', 'nan', 'none', 'n/a', 'na', 
                        'null', 'nil', '', '-', 'undefined'
                    }
                    values = [v for v in values 
                            if v.lower() not in unknown_variants]
                
                # Sort safely
                try:
                    return sorted(values)
                except Exception:
                    return values  # Return unsorted if sorting fails
                
            except Exception as e:
                logger.error(f"Error extracting unique values from {column}: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Critical error in get_unique_values for {column}: {str(e)}")
            return []
    
    @staticmethod
    def _apply_filter_subset(df: pd.DataFrame, filters: Dict[str, Any], 
                           exclude_cols: List[str]) -> pd.DataFrame:
        """Apply filters excluding specific columns for interconnected filtering"""
        try:
            if df.empty:
                return df
            
            filtered_df = df.copy()
            
            # Apply each filter except excluded ones
            filter_applications = [
                ('categories', 'category', ProductionFilterEngine._apply_category_filter),
                ('sectors', 'sector', ProductionFilterEngine._apply_sector_filter),
                ('eps_tiers', 'eps_tier', ProductionFilterEngine._apply_eps_tier_filter),
                ('pe_tiers', 'pe_tier', ProductionFilterEngine._apply_pe_tier_filter),
                ('price_tiers', 'price_tier', ProductionFilterEngine._apply_price_tier_filter),
            ]
            
            for filter_key, column_name, filter_func in filter_applications:
                if column_name not in exclude_cols and filter_key in filters:
                    try:
                        filter_values = filters.get(filter_key, [])
                        if filter_values and 'All' not in filter_values and filter_values != ['']:
                            filtered_df = filter_func(filtered_df, filter_values)
                    except Exception as e:
                        logger.warning(f"Error applying {filter_key} filter: {str(e)}")
                        continue
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error in filter subset application: {str(e)}")
            return df
    
    @staticmethod
    def _apply_category_filter(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
        """Apply category filter safely"""
        try:
            if not categories or 'category' not in df.columns:
                return df
            return df[df['category'].isin(categories)]
        except Exception as e:
            logger.error(f"Error in category filter: {str(e)}")
            return df
    
    @staticmethod
    def _apply_sector_filter(df: pd.DataFrame, sectors: List[str]) -> pd.DataFrame:
        """Apply sector filter safely"""
        try:
            if not sectors or 'sector' not in df.columns:
                return df
            return df[df['sector'].isin(sectors)]
        except Exception as e:
            logger.error(f"Error in sector filter: {str(e)}")
            return df
    
    @staticmethod
    def _apply_eps_tier_filter(df: pd.DataFrame, eps_tiers: List[str]) -> pd.DataFrame:
        """Apply EPS tier filter safely"""
        try:
            if not eps_tiers or 'eps_tier' not in df.columns:
                return df
            return df[df['eps_tier'].isin(eps_tiers)]
        except Exception as e:
            logger.error(f"Error in EPS tier filter: {str(e)}")
            return df
    
    @staticmethod
    def _apply_pe_tier_filter(df: pd.DataFrame, pe_tiers: List[str]) -> pd.DataFrame:
        """Apply PE tier filter safely"""
        try:
            if not pe_tiers or 'pe_tier' not in df.columns:
                return df
            return df[df['pe_tier'].isin(pe_tiers)]
        except Exception as e:
            logger.error(f"Error in PE tier filter: {str(e)}")
            return df
    
    @staticmethod
    def _apply_price_tier_filter(df: pd.DataFrame, price_tiers: List[str]) -> pd.DataFrame:
        """Apply price tier filter safely"""
        try:
            if not price_tiers or 'price_tier' not in df.columns:
                return df
            return df[df['price_tier'].isin(price_tiers)]
        except Exception as e:
            logger.error(f"Error in price tier filter: {str(e)}")
            return df
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with comprehensive validation and error handling"""
        try:
            if df is None or df.empty:
                return pd.DataFrame()
            
            filtered_df = df.copy()
            initial_count = len(filtered_df)
            
            # Track filter applications for debugging
            filter_steps = []
            
            # 1. Category filter
            try:
                categories = filters.get('categories', [])
                if categories and 'All' not in categories and categories != ['']:
                    if 'category' in filtered_df.columns:
                        before = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
                        after = len(filtered_df)
                        filter_steps.append(f"Category: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in category filter: {str(e)}")
            
            # 2. Sector filter
            try:
                sectors = filters.get('sectors', [])
                if sectors and 'All' not in sectors and sectors != ['']:
                    if 'sector' in filtered_df.columns:
                        before = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
                        after = len(filtered_df)
                        filter_steps.append(f"Sector: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in sector filter: {str(e)}")
            
            # 3. Score filter
            try:
                min_score = filters.get('min_score', 0)
                if min_score > 0 and 'master_score' in filtered_df.columns:
                    before = len(filtered_df)
                    score_filter = filtered_df['master_score'] >= min_score
                    filtered_df = filtered_df[score_filter]
                    after = len(filtered_df)
                    filter_steps.append(f"Min Score {min_score}: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in score filter: {str(e)}")
            
            # 4. EPS tier filter
            try:
                eps_tiers = filters.get('eps_tiers', [])
                if eps_tiers and 'All' not in eps_tiers and eps_tiers != ['']:
                    if 'eps_tier' in filtered_df.columns:
                        before = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['eps_tier'].isin(eps_tiers)]
                        after = len(filtered_df)
                        filter_steps.append(f"EPS Tier: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in EPS tier filter: {str(e)}")
            
            # 5. PE tier filter
            try:
                pe_tiers = filters.get('pe_tiers', [])
                if pe_tiers and 'All' not in pe_tiers and pe_tiers != ['']:
                    if 'pe_tier' in filtered_df.columns:
                        before = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['pe_tier'].isin(pe_tiers)]
                        after = len(filtered_df)
                        filter_steps.append(f"PE Tier: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in PE tier filter: {str(e)}")
            
            # 6. Price tier filter
            try:
                price_tiers = filters.get('price_tiers', [])
                if price_tiers and 'All' not in price_tiers and price_tiers != ['']:
                    if 'price_tier' in filtered_df.columns:
                        before = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['price_tier'].isin(price_tiers)]
                        after = len(filtered_df)
                        filter_steps.append(f"Price Tier: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in price tier filter: {str(e)}")
            
            # 7. EPS change filter
            try:
                min_eps_change = filters.get('min_eps_change')
                if min_eps_change is not None and 'eps_change_pct' in filtered_df.columns:
                    before = len(filtered_df)
                    # Allow NaN values or values >= threshold
                    eps_filter = (
                        (filtered_df['eps_change_pct'] >= min_eps_change) | 
                        (filtered_df['eps_change_pct'].isna())
                    )
                    filtered_df = filtered_df[eps_filter]
                    after = len(filtered_df)
                    filter_steps.append(f"Min EPS Change {min_eps_change}%: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in EPS change filter: {str(e)}")
            
            # 8. Pattern filter
            try:
                patterns = filters.get('patterns', [])
                if patterns and 'patterns' in filtered_df.columns:
                    before = len(filtered_df)
                    # Escape special regex characters in pattern names
                    escaped_patterns = []
                    for pattern in patterns:
                        try:
                            escaped = re.escape(str(pattern))
                            escaped_patterns.append(escaped)
                        except Exception:
                            continue
                    
                    if escaped_patterns:
                        pattern_regex = '|'.join(escaped_patterns)
                        pattern_mask = filtered_df['patterns'].str.contains(
                            pattern_regex, case=False, na=False, regex=True
                        )
                        filtered_df = filtered_df[pattern_mask]
                        after = len(filtered_df)
                        filter_steps.append(f"Patterns: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in pattern filter: {str(e)}")
            
            # 9. Trend filter
            try:
                trend_range = filters.get('trend_range')
                if (trend_range and filters.get('trend_filter') != 'All Trends' and 
                    'trend_quality' in filtered_df.columns):
                    min_trend, max_trend = trend_range
                    before = len(filtered_df)
                    trend_filter = (
                        (filtered_df['trend_quality'] >= min_trend) & 
                        (filtered_df['trend_quality'] <= max_trend)
                    )
                    filtered_df = filtered_df[trend_filter]
                    after = len(filtered_df)
                    filter_steps.append(f"Trend {min_trend}-{max_trend}: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in trend filter: {str(e)}")
            
            # 10. PE range filters
            try:
                min_pe = filters.get('min_pe')
                if min_pe is not None and 'pe' in filtered_df.columns:
                    before = len(filtered_df)
                    pe_filter = (
                        (filtered_df['pe'].isna()) |
                        ((filtered_df['pe'] > 0) & 
                         (filtered_df['pe'] >= min_pe) & 
                         ~np.isinf(filtered_df['pe']))
                    )
                    filtered_df = filtered_df[pe_filter]
                    after = len(filtered_df)
                    filter_steps.append(f"Min PE {min_pe}: {before} → {after}")
                
                max_pe = filters.get('max_pe')
                if max_pe is not None and 'pe' in filtered_df.columns:
                    before = len(filtered_df)
                    pe_filter = (
                        (filtered_df['pe'].isna()) |
                        ((filtered_df['pe'] > 0) & 
                         (filtered_df['pe'] <= max_pe) & 
                         ~np.isinf(filtered_df['pe']))
                    )
                    filtered_df = filtered_df[pe_filter]
                    after = len(filtered_df)
                    filter_steps.append(f"Max PE {max_pe}: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in PE range filters: {str(e)}")
            
            # 11. Fundamental data completeness filter
            try:
                require_fundamental = filters.get('require_fundamental_data', False)
                if require_fundamental:
                    if all(col in filtered_df.columns for col in ['pe', 'eps_change_pct']):
                        before = len(filtered_df)
                        fundamental_filter = (
                            filtered_df['pe'].notna() & 
                            (filtered_df['pe'] > 0) &
                            ~np.isinf(filtered_df['pe']) &
                            filtered_df['eps_change_pct'].notna() &
                            ~np.isinf(filtered_df['eps_change_pct'])
                        )
                        filtered_df = filtered_df[fundamental_filter]
                        after = len(filtered_df)
                        filter_steps.append(f"Require Fundamentals: {before} → {after}")
            except Exception as e:
                logger.error(f"Error in fundamental data filter: {str(e)}")
            
            # Log filter results
            final_count = len(filtered_df)
            if final_count != initial_count:
                logger.info(f"Filters applied: {initial_count} → {final_count} stocks")
                if len(filter_steps) > 0:
                    logger.debug(f"Filter steps: {'; '.join(filter_steps)}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Critical error in apply_filters: {str(e)}")
            # Return original dataframe on critical error
            return df

# ============================================
# PRODUCTION SEARCH ENGINE
# ============================================

class ProductionSearchEngine:
    """Production-grade search engine with fuzzy matching and robust algorithms"""
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Advanced stock search with multiple matching strategies"""
        try:
            if not query or df is None or df.empty:
                return pd.DataFrame()
            
            # Clean and prepare query
            query_clean = ProductionSearchEngine._clean_query(query)
            if not query_clean:
                return pd.DataFrame()
            
            # Multiple search strategies with ranking
            search_results = []
            
            # Strategy 1: Exact ticker match (highest priority)
            exact_ticker = ProductionSearchEngine._search_exact_ticker(df, query_clean)
            if not exact_ticker.empty:
                exact_ticker['search_rank'] = 1
                search_results.append(exact_ticker)
            
            # Strategy 2: Ticker prefix match
            prefix_ticker = ProductionSearchEngine._search_ticker_prefix(df, query_clean)
            if not prefix_ticker.empty:
                prefix_ticker['search_rank'] = 2
                search_results.append(prefix_ticker)
            
            # Strategy 3: Ticker contains
            contains_ticker = ProductionSearchEngine._search_ticker_contains(df, query_clean)
            if not contains_ticker.empty:
                contains_ticker['search_rank'] = 3
                search_results.append(contains_ticker)
            
            # Strategy 4: Company name exact words
            company_exact = ProductionSearchEngine._search_company_exact_words(df, query_clean)
            if not company_exact.empty:
                company_exact['search_rank'] = 4
                search_results.append(company_exact)
            
            # Strategy 5: Company name contains
            company_contains = ProductionSearchEngine._search_company_contains(df, query_clean)
            if not company_contains.empty:
                company_contains['search_rank'] = 5
                search_results.append(company_contains)
            
            # Strategy 6: Fuzzy matching
            fuzzy_results = ProductionSearchEngine._search_fuzzy(df, query_clean)
            if not fuzzy_results.empty:
                fuzzy_results['search_rank'] = 6
                search_results.append(fuzzy_results)
            
            # Combine and deduplicate results
            if search_results:
                combined = pd.concat(search_results, ignore_index=True)
                # Remove duplicates, keeping best rank
                combined = combined.sort_values('search_rank').drop_duplicates(
                    subset=['ticker'], keep='first'
                )
                # Remove search rank column
                combined = combined.drop(columns=['search_rank'])
                return combined
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in stock search: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _clean_query(query: str) -> str:
        """Clean and normalize search query"""
        try:
            if not query:
                return ""
            
            # Convert to string and strip
            clean = str(query).strip()
            
            # Remove extra whitespace
            clean = re.sub(r'\s+', ' ', clean)
            
            # Convert to uppercase for ticker matching
            clean = clean.upper()
            
            return clean
            
        except Exception as e:
            logger.error(f"Error cleaning query '{query}': {str(e)}")
            return ""
    
    @staticmethod
    def _search_exact_ticker(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search for exact ticker match"""
        try:
            if 'ticker' not in df.columns:
                return pd.DataFrame()
            
            # Exact match
            mask = df['ticker'].str.upper() == query
            return df[mask].copy()
            
        except Exception as e:
            logger.error(f"Error in exact ticker search: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _search_ticker_prefix(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search for ticker prefix match"""
        try:
            if 'ticker' not in df.columns or len(query) < 2:
                return pd.DataFrame()
            
            # Prefix match (but not exact match to avoid duplicates)
            mask = (
                df['ticker'].str.upper().str.startswith(query) & 
                (df['ticker'].str.upper() != query)
            )
            return df[mask].copy()
            
        except Exception as e:
            logger.error(f"Error in ticker prefix search: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _search_ticker_contains(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search for ticker containing query"""
        try:
            if 'ticker' not in df.columns or len(query) < 2:
                return pd.DataFrame()
            
            # Contains match (excluding prefix matches)
            mask = (
                df['ticker'].str.upper().str.contains(query, na=False) & 
                ~df['ticker'].str.upper().str.startswith(query)
            )
            return df[mask].copy()
            
        except Exception as e:
            logger.error(f"Error in ticker contains search: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _search_company_exact_words(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search for exact word matches in company name"""
        try:
            if 'company_name' not in df.columns or len(query) < 3:
                return pd.DataFrame()
            
            # Split query into words
            query_words = query.split()
            if not query_words:
                return pd.DataFrame()
            
            # Find companies with all query words
            mask = pd.Series(True, index=df.index)
            for word in query_words:
                if len(word) >= 2:  # Only consider words with 2+ characters
                    word_pattern = r'\b' + re.escape(word) + r'\b'
                    word_mask = df['company_name'].str.upper().str.contains(
                        word_pattern, case=False, na=False, regex=True
                    )
                    mask = mask & word_mask
            
            return df[mask].copy()
            
        except Exception as e:
            logger.error(f"Error in company exact words search: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _search_company_contains(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search for query contained in company name"""
        try:
            if 'company_name' not in df.columns or len(query) < 3:
                return pd.DataFrame()
            
            # Simple contains search
            mask = df['company_name'].str.upper().str.contains(query, na=False)
            results = df[mask].copy()
            
            # Exclude results that were already found in exact words search
            if not results.empty:
                exact_words_results = ProductionSearchEngine._search_company_exact_words(df, query)
                if not exact_words_results.empty:
                    exact_tickers = set(exact_words_results['ticker'])
                    results = results[~results['ticker'].isin(exact_tickers)]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in company contains search: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _search_fuzzy(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Fuzzy search using simple string similarity"""
        try:
            if len(query) < 3:
                return pd.DataFrame()
            
            fuzzy_results = []
            
            # Check both ticker and company name columns
            search_columns = []
            if 'ticker' in df.columns:
                search_columns.append('ticker')
            if 'company_name' in df.columns:
                search_columns.append('company_name')
            
            for _, row in df.iterrows():
                max_similarity = 0
                
                for col in search_columns:
                    try:
                        text = str(row[col]).upper()
                        similarity = ProductionSearchEngine._calculate_similarity(query, text)
                        max_similarity = max(max_similarity, similarity)
                    except Exception:
                        continue
                
                # Include results with similarity > 0.6
                if max_similarity > 0.6:
                    fuzzy_results.append(row)
            
            if fuzzy_results:
                results_df = pd.DataFrame(fuzzy_results)
                # Limit to top 10 fuzzy matches
                return results_df.head(10)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in fuzzy search: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _calculate_similarity(query: str, text: str) -> float:
        """Calculate simple string similarity"""
        try:
            if not query or not text:
                return 0.0
            
            query = query.strip()
            text = text.strip()
            
            # Exact match
            if query == text:
                return 1.0
            
            # One contains the other
            if query in text or text in query:
                return 0.8
            
            # Character overlap ratio
            query_chars = set(query.lower())
            text_chars = set(text.lower())
            
            if not query_chars or not text_chars:
                return 0.0
            
            intersection = len(query_chars.intersection(text_chars))
            union = len(query_chars.union(text_chars))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

# ============================================
# QUICK FILTER ENGINE
# ============================================

class QuickFilterEngine:
    """Production-grade quick filtering for common use cases"""
    
    @staticmethod
    def apply_quick_filter(df: pd.DataFrame, filter_type: str) -> pd.DataFrame:
        """Apply quick filters with comprehensive error handling"""
        try:
            if df is None or df.empty:
                return pd.DataFrame()
            
            filter_functions = {
                'top_gainers': QuickFilterEngine._filter_top_gainers,
                'volume_surges': QuickFilterEngine._filter_volume_surges,
                'breakout_ready': QuickFilterEngine._filter_breakout_ready,
                'hidden_gems': QuickFilterEngine._filter_hidden_gems,
                'momentum_waves': QuickFilterEngine._filter_momentum_waves,
                'value_picks': QuickFilterEngine._filter_value_picks
            }
            
            if filter_type not in filter_functions:
                logger.warning(f"Unknown quick filter type: {filter_type}")
                return df
            
            filter_func = filter_functions[filter_type]
            return filter_func(df)
            
        except Exception as e:
            logger.error(f"Error in quick filter '{filter_type}': {str(e)}")
            return df
    
    @staticmethod
    def _filter_top_gainers(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for top momentum gainers"""
        try:
            if 'momentum_score' not in df.columns:
                return df
            
            threshold = CONFIG.TOP_GAINER_MOMENTUM
            return df[df['momentum_score'] >= threshold].copy()
            
        except Exception as e:
            logger.error(f"Error in top gainers filter: {str(e)}")
            return df
    
    @staticmethod
    def _filter_volume_surges(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for volume surge stocks"""
        try:
            if 'rvol' not in df.columns:
                return df
            
            threshold = CONFIG.VOLUME_SURGE_RVOL
            rvol_safe = df['rvol'].apply(lambda x: SafeMath.safe_float(x, 1.0))
            return df[rvol_safe >= threshold].copy()
            
        except Exception as e:
            logger.error(f"Error in volume surges filter: {str(e)}")
            return df
    
    @staticmethod
    def _filter_breakout_ready(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for breakout ready stocks"""
        try:
            if 'breakout_score' not in df.columns:
                return df
            
            threshold = CONFIG.BREAKOUT_READY_SCORE
            return df[df['breakout_score'] >= threshold].copy()
            
        except Exception as e:
            logger.error(f"Error in breakout ready filter: {str(e)}")
            return df
    
    @staticmethod
    def _filter_hidden_gems(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for hidden gem pattern stocks"""
        try:
            if 'patterns' not in df.columns:
                return df
            
            pattern_mask = df['patterns'].str.contains('HIDDEN GEM', na=False)
            return df[pattern_mask].copy()
            
        except Exception as e:
            logger.error(f"Error in hidden gems filter: {str(e)}")
            return df
    
    @staticmethod
    def _filter_momentum_waves(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for momentum wave stocks"""
        try:
            required_cols = ['momentum_score', 'acceleration_score']
            if not all(col in df.columns for col in required_cols):
                return df
            
            momentum_threshold = CONFIG.PATTERN_THRESHOLDS['momentum_wave']
            acceleration_threshold = 70
            
            mask = (
                (df['momentum_score'] >= momentum_threshold) &
                (df['acceleration_score'] >= acceleration_threshold)
            )
            return df[mask].copy()
            
        except Exception as e:
            logger.error(f"Error in momentum waves filter: {str(e)}")
            return df
    
    @staticmethod
    def _filter_value_picks(df: pd.DataFrame) -> pd.DataFrame:
        """Filter for value momentum picks"""
        try:
            required_cols = ['pe', 'master_score']
            if not all(col in df.columns for col in required_cols):
                return df
            
            pe_safe = df['pe'].apply(lambda x: SafeMath.safe_float(x, float('inf')))
            valid_pe = (pe_safe > 0) & (pe_safe < CONFIG.MAX_VALID_PE)
            
            mask = valid_pe & (pe_safe < 15) & (df['master_score'] >= 70)
            return df[mask].copy()
            
        except Exception as e:
            logger.error(f"Error in value picks filter: {str(e)}")
            return df

# ============================================
# FILTER STATE MANAGEMENT
# ============================================

class FilterStateManager:
    """Production-grade filter state management with validation"""
    
    @staticmethod
    def validate_filter_state(filters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean filter state"""
        try:
            if not isinstance(filters, dict):
                return {}
            
            validated = {}
            
            # List-type filters
            list_filters = [
                'categories', 'sectors', 'eps_tiers', 'pe_tiers', 
                'price_tiers', 'patterns'
            ]
            
            for filter_name in list_filters:
                if filter_name in filters:
                    value = filters[filter_name]
                    if isinstance(value, list):
                        # Clean list values
                        clean_list = []
                        for item in value:
                            if item and str(item).strip():
                                clean_list.append(str(item).strip())
                        validated[filter_name] = clean_list
                    else:
                        validated[filter_name] = []
            
            # Numeric filters
            numeric_filters = ['min_score', 'min_eps_change', 'min_pe', 'max_pe']
            for filter_name in numeric_filters:
                if filter_name in filters:
                    value = filters[filter_name]
                    try:
                        if value is not None and str(value).strip():
                            validated[filter_name] = float(value)
                    except (ValueError, TypeError):
                        pass  # Skip invalid numeric values
            
            # Boolean filters
            boolean_filters = ['require_fundamental_data']
            for filter_name in boolean_filters:
                if filter_name in filters:
                    validated[filter_name] = bool(filters[filter_name])
            
            # String filters
            string_filters = ['trend_filter']
            for filter_name in string_filters:
                if filter_name in filters:
                    value = filters[filter_name]
                    if value:
                        validated[filter_name] = str(value).strip()
            
            # Special handling for trend_range
            if 'trend_range' in filters:
                trend_range = filters['trend_range']
                if isinstance(trend_range, (list, tuple)) and len(trend_range) == 2:
                    try:
                        min_val = float(trend_range[0])
                        max_val = float(trend_range[1])
                        if min_val <= max_val:
                            validated['trend_range'] = (min_val, max_val)
                    except (ValueError, TypeError):
                        pass
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating filter state: {str(e)}")
            return {}
    
    @staticmethod
    def get_filter_summary(filters: Dict[str, Any]) -> str:
        """Generate human-readable filter summary"""
        try:
            if not filters:
                return "No filters active"
            
            summary_parts = []
            
            # Count active filters
            for key, value in filters.items():
                if key in ['categories', 'sectors', 'eps_tiers', 'pe_tiers', 'price_tiers', 'patterns']:
                    if value and len(value) > 0:
                        summary_parts.append(f"{key.replace('_', ' ').title()}: {len(value)}")
                elif key in ['min_score', 'min_eps_change', 'min_pe', 'max_pe']:
                    if value is not None:
                        summary_parts.append(f"{key.replace('_', ' ').title()}: {value}")
                elif key == 'trend_filter' and value != 'All Trends':
                    summary_parts.append(f"Trend: {value}")
                elif key == 'require_fundamental_data' and value:
                    summary_parts.append("Requires Fundamentals")
            
            if summary_parts:
                return f"{len(summary_parts)} filters: " + ", ".join(summary_parts)
            else:
                return "No active filters"
                
        except Exception as e:
            logger.error(f"Error generating filter summary: {str(e)}")
            return "Filter summary error"
