"""
Wave Detection Ultimate 3.0 - UI Components
===========================================
Modular Streamlit UI components for clean, maintainable interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from config import CONFIG, UI_CONFIG, logger
from utils import format_number, get_score_color, SessionManager

class UIStyler:
    """Handle all UI styling and theming"""
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS for better UI"""
        st.markdown("""
        <style>
        .main {padding: 0rem 1rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 8px;}
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.2);
            padding: 5% 5% 5% 10%;
            border-radius: 5px;
            overflow-wrap: break-word;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 5px;
        }
        div.stButton > button {
            width: 100%;
        }
        .score-excellent { color: #2ecc71; font-weight: bold; }
        .score-good { color: #f39c12; font-weight: bold; }
        .score-average { color: #95a5a6; font-weight: bold; }
        .score-poor { color: #e74c3c; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_header():
        """Create the main application header"""
        st.markdown("""
        <div style="
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 3.0</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Professional Stock Ranking System with Wave Radar™ Early Detection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def format_score_display(score: float) -> str:
        """Format score with color coding"""
        if pd.isna(score):
            return "N/A"
        
        color_class = UIStyler.get_score_class(score)
        return f'<span class="{color_class}">{score:.1f}</span>'
    
    @staticmethod
    def get_score_class(score: float) -> str:
        """Get CSS class for score"""
        if score >= 80:
            return "score-excellent"
        elif score >= 60:
            return "score-good"
        elif score >= 40:
            return "score-average"
        else:
            return "score-poor"

class SidebarManager:
    """Manage sidebar components"""
    
    @staticmethod
    def render_sidebar(ranked_df: pd.DataFrame) -> Dict[str, Any]:
        """Render complete sidebar and return filters"""
        with st.sidebar:
            SidebarManager._render_quick_actions()
            SidebarManager._render_data_quality()
            
            st.markdown("---")
            st.markdown("### 🔍 Smart Filters")
            
            filters = SidebarManager._render_filters(ranked_df)
            
            SidebarManager._render_performance_stats()
            
            return filters
    
    @staticmethod
    def _render_quick_actions():
        """Render quick action buttons"""
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.cache_data.clear()
                SessionManager.clear_filter_state()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.rerun()
    
    @staticmethod
    def _render_data_quality():
        """Render data quality indicators"""
        if 'data_quality' in st.session_state:
            with st.expander("📊 Data Quality", expanded=True):
                quality = st.session_state.data_quality
                
                col1, col2 = st.columns(2)
                with col1:
                    completeness = quality.get('completeness', 0) * 100
                    st.metric("Completeness", f"{completeness:.1f}%")
                    pe_coverage = quality.get('pe_coverage', 0)
                    st.metric("PE Coverage", f"{pe_coverage:,}")
                
                with col2:
                    freshness = quality.get('freshness', 0)
                    st.metric("Freshness", f"{freshness:.1f}%")
                    eps_coverage = quality.get('eps_coverage', 0)
                    st.metric("EPS Coverage", f"{eps_coverage:,}")
                
                if 'data_timestamp' in st.session_state:
                    timestamp = st.session_state.data_timestamp
                    st.caption(f"Data loaded: {timestamp.strftime('%I:%M %p')}")
    
    @staticmethod
    def _render_filters(df: pd.DataFrame) -> Dict[str, Any]:
        """Render filter controls"""
        filters = {}
        
        # Clear filters button
        if st.button("🗑️ Clear All Filters", use_container_width=True):
            SessionManager.clear_filter_state()
            st.rerun()
        
        # Basic filters
        filters.update(SidebarManager._render_basic_filters(df))
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            filters.update(SidebarManager._render_advanced_filters(df))
        
        return filters
    
    @staticmethod
    def _render_basic_filters(df: pd.DataFrame) -> Dict[str, Any]:
        """Render basic filter controls"""
        filters = {}
        
        # Category filter
        if 'category' in df.columns:
            categories = sorted(df['category'].unique())
            filters['categories'] = st.multiselect(
                "Market Cap Category",
                options=categories,
                default=[],
                placeholder="All Categories",
                key="category_filter"
            )
        
        # Sector filter
        if 'sector' in df.columns:
            sectors = sorted(df['sector'].unique())
            filters['sectors'] = st.multiselect(
                "Sector",
                options=sectors,
                default=[],
                placeholder="All Sectors",
                key="sector_filter"
            )
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Master Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            help="Filter stocks by minimum master score"
        )
        
        return filters
    
    @staticmethod
    def _render_advanced_filters(df: pd.DataFrame) -> Dict[str, Any]:
        """Render advanced filter controls"""
        filters = {}
        
        # PE filter
        if 'pe' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                filters['min_pe'] = st.number_input(
                    "Min PE", min_value=0.0, value=0.0, step=1.0
                )
            with col2:
                filters['max_pe'] = st.number_input(
                    "Max PE", min_value=0.0, value=1000.0, step=1.0
                )
        
        # EPS change filter
        if 'eps_change_pct' in df.columns:
            filters['min_eps_change'] = st.number_input(
                "Min EPS Change %",
                value=-100.0,
                step=10.0,
                help="Minimum EPS growth percentage"
            )
        
        # RVOL filter
        if 'rvol' in df.columns:
            filters['min_rvol'] = st.number_input(
                "Min RVOL",
                min_value=0.0,
                value=0.0,
                step=0.1,
                help="Minimum relative volume"
            )
        
        return filters
    
    @staticmethod
    def _render_performance_stats():
        """Render performance statistics"""
        if 'performance_metrics' in st.session_state:
            with st.expander("⚡ Performance Stats"):
                perf = st.session_state.performance_metrics
                
                total_time = sum(perf.values())
                st.metric("Total Load Time", f"{total_time:.2f}s")
                
                # Show slowest operations
                slowest = sorted(perf.items(), key=lambda x: x[1], reverse=True)[:3]
                for func_name, elapsed in slowest:
                    st.caption(f"{func_name}: {elapsed:.2f}s")

class MetricsDisplay:
    """Handle metrics and summary displays"""
    
    @staticmethod
    def render_summary_metrics(filtered_df: pd.DataFrame, total_df: pd.DataFrame):
        """Render top-level summary metrics"""
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            MetricsDisplay._render_stock_count_metric(filtered_df, total_df)
        
        with col2:
            MetricsDisplay._render_score_metric(filtered_df)
        
        with col3:
            MetricsDisplay._render_pe_or_range_metric(filtered_df)
        
        with col4:
            MetricsDisplay._render_growth_metric(filtered_df)
        
        with col5:
            MetricsDisplay._render_rvol_metric(filtered_df)
        
        with col6:
            MetricsDisplay._render_trend_metric(filtered_df)
    
    @staticmethod
    def _render_stock_count_metric(filtered_df: pd.DataFrame, total_df: pd.DataFrame):
        """Render stock count metric"""
        total_stocks = len(filtered_df)
        total_original = len(total_df)
        pct_of_all = (total_stocks/total_original*100) if total_original > 0 else 0
        
        st.metric(
            "Total Stocks",
            f"{total_stocks:,}",
            f"{pct_of_all:.0f}% of {total_original:,}"
        )
    
    @staticmethod
    def _render_score_metric(df: pd.DataFrame):
        """Render average score metric"""
        if not df.empty and 'master_score' in df.columns:
            avg_score = df['master_score'].mean()
            std_score = df['master_score'].std()
            st.metric(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={std_score:.1f}"
            )
        else:
            st.metric("Avg Score", "N/A")
    
    @staticmethod
    def _render_pe_or_range_metric(df: pd.DataFrame):
        """Render PE or score range metric"""
        if 'pe' in df.columns:
            valid_pe = df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000)
            if valid_pe.any():
                median_pe = df.loc[valid_pe, 'pe'].median()
                pe_count = valid_pe.sum()
                pe_pct = (pe_count / len(df) * 100) if len(df) > 0 else 0
                st.metric(
                    "Median PE",
                    f"{median_pe:.1f}x",
                    f"{pe_pct:.0f}% have data"
                )
            else:
                st.metric("PE Data", "Limited")
        else:
            if not df.empty and 'master_score' in df.columns:
                min_score = df['master_score'].min()
                max_score = df['master_score'].max()
                st.metric("Score Range", f"{min_score:.1f}-{max_score:.1f}")
            else:
                st.metric("Score Range", "N/A")
    
    @staticmethod
    def _render_growth_metric(df: pd.DataFrame):
        """Render growth metric"""
        if 'eps_change_pct' in df.columns:
            valid_eps = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
            positive_growth = valid_eps & (df['eps_change_pct'] > 0)
            strong_growth = valid_eps & (df['eps_change_pct'] > 50)
            mega_growth = valid_eps & (df['eps_change_pct'] > 100)
            
            growth_count = positive_growth.sum()
            if mega_growth.sum() > 0:
                st.metric(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{strong_growth.sum()} >50% | {mega_growth.sum()} >100%"
                )
            else:
                st.metric(
                    "EPS Growth +ve", 
                    f"{growth_count}",
                    f"{valid_eps.sum()} have data"
                )
        else:
            accelerating = (df.get('acceleration_score', 0) >= 80).sum()
            st.metric("Accelerating", f"{accelerating}")
    
    @staticmethod
    def _render_rvol_metric(df: pd.DataFrame):
        """Render RVOL metric"""
        if 'rvol' in df.columns:
            high_rvol = (df['rvol'] > 2).sum()
            st.metric("High RVOL", f"{high_rvol}")
        else:
            st.metric("High RVOL", "0")
    
    @staticmethod
    def _render_trend_metric(df: pd.DataFrame):
        """Render trend strength metric"""
        if 'trend_quality' in df.columns:
            strong_trends = (df['trend_quality'] >= 80).sum()
            total = len(df)
            pct = (strong_trends/total*100) if total > 0 else 0
            st.metric(
                "Strong Trends",
                f"{strong_trends}",
                f"{pct:.0f}%"
            )
        else:
            with_patterns = (df.get('patterns', '') != '').sum()
            st.metric("With Patterns", f"{with_patterns}")

class DataTableRenderer:
    """Handle data table rendering with formatting"""
    
    @staticmethod
    def render_rankings_table(df: pd.DataFrame, display_count: int, 
                            show_fundamentals: bool = False) -> pd.DataFrame:
        """Render the main rankings table"""
        if df.empty:
            st.warning("No data to display with current filters.")
            return df
        
        # Select columns based on display mode
        if show_fundamentals:
            display_cols = UI_CONFIG.HYBRID_COLUMNS
        else:
            display_cols = UI_CONFIG.TECHNICAL_COLUMNS
        
        # Filter available columns
        available_cols = [col for col in display_cols if col in df.columns]
        
        # Get display data
        display_df = df.head(display_count)[available_cols].copy()
        
        # Format the data for display
        display_df = DataTableRenderer._format_display_data(display_df, show_fundamentals)
        
        # Render table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=DataTableRenderer._get_column_config(show_fundamentals)
        )
        
        return display_df
    
    @staticmethod
    def _format_display_data(df: pd.DataFrame, show_fundamentals: bool) -> pd.DataFrame:
        """Format data for display"""
        formatted_df = df.copy()
        
        # Format price columns
        price_cols = ['price', 'prev_close', 'low_52w', 'high_52w']
        for col in price_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: format_number(x, 'currency') if pd.notna(x) else 'N/A'
                )
        
        # Format score columns
        score_cols = ['master_score', 'position_score', 'volume_score', 'momentum_score']
        for col in score_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A'
                )
        
        # Format percentage columns
        if show_fundamentals:
            if 'eps_change_pct' in formatted_df.columns:
                formatted_df['eps_change_pct'] = formatted_df['eps_change_pct'].apply(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else 'N/A'
                )
            
            if 'pe' in formatted_df.columns:
                formatted_df['pe'] = formatted_df['pe'].apply(
                    lambda x: f"{x:.1f}x" if pd.notna(x) else 'N/A'
                )
        
        # Format RVOL
        if 'rvol' in formatted_df.columns:
            formatted_df['rvol'] = formatted_df['rvol'].apply(
                lambda x: f"{x:.1f}x" if pd.notna(x) else 'N/A'
            )
        
        return formatted_df
    
    @staticmethod
    def _get_column_config(show_fundamentals: bool) -> Dict[str, Any]:
        """Get column configuration for dataframe"""
        config = {
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "price": st.column_config.TextColumn("Price", width="small"),
            "master_score": st.column_config.TextColumn("Score", width="small"),
            "patterns": st.column_config.TextColumn("Patterns", width="large")
        }
        
        if show_fundamentals:
            config.update({
                "pe": st.column_config.TextColumn("PE", width="small"),
                "eps_change_pct": st.column_config.TextColumn("EPS Δ%", width="small")
            })
        else:
            config.update({
                "position_score": st.column_config.TextColumn("Position", width="small"),
                "volume_score": st.column_config.TextColumn("Volume", width="small"),
                "momentum_score": st.column_config.TextColumn("Momentum", width="small")
            })
        
        return config

class QuickActionsBar:
    """Handle quick action buttons"""
    
    @staticmethod
    def render_quick_actions() -> Optional[str]:
        """Render quick action buttons and return selected action"""
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📈 Top Gainers", use_container_width=True):
                return 'top_gainers'
        
        with col2:
            if st.button("🔥 Volume Surges", use_container_width=True):
                return 'volume_surges'
        
        with col3:
            if st.button("🎯 Breakout Ready", use_container_width=True):
                return 'breakout_ready'
        
        with col4:
            if st.button("💎 Hidden Gems", use_container_width=True):
                return 'hidden_gems'
        
        with col5:
            if st.button("🚀 Accelerating", use_container_width=True):
                return 'accelerating'
        
        return None

class FilterEngine:
    """Handle data filtering logic"""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters to dataframe"""
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Category filter
        if filters.get('categories'):
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # Sector filter
        if filters.get('sectors'):
            filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
        
        # Score filter
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= min_score]
        
        # PE filters
        if filters.get('min_pe') is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['pe'] >= filters['min_pe']) | filtered_df['pe'].isna()
            ]
        
        if filters.get('max_pe') is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['pe'] <= filters['max_pe']) | filtered_df['pe'].isna()
            ]
        
        # EPS change filter
        if filters.get('min_eps_change') is not None and 'eps_change_pct' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['eps_change_pct'] >= filters['min_eps_change']) | 
                filtered_df['eps_change_pct'].isna()
            ]
        
        # RVOL filter
        if filters.get('min_rvol') is not None and 'rvol' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['rvol'] >= filters['min_rvol']) | 
                filtered_df['rvol'].isna()
            ]
        
        return filtered_df.sort_values('rank')
    
    @staticmethod
    def apply_quick_action(df: pd.DataFrame, action: str) -> pd.DataFrame:
        """Apply quick action filters"""
        if action == 'top_gainers':
            return df[df.get('momentum_score', 0) >= CONFIG.TOP_GAINER_MOMENTUM]
        elif action == 'volume_surges':
            return df[df.get('rvol', 0) >= CONFIG.VOLUME_SURGE_RVOL]
        elif action == 'breakout_ready':
            return df[df.get('breakout_score', 0) >= CONFIG.BREAKOUT_READY_SCORE]
        elif action == 'hidden_gems':
            return df[df.get('patterns', '').str.contains('HIDDEN GEM', na=False)]
        elif action == 'accelerating':
            return df[df.get('patterns', '').str.contains('ACCELERATING', na=False)]
        else:
            return df

class ChartRenderer:
    """Handle chart and visualization rendering"""
    
    @staticmethod
    def render_sector_distribution(df: pd.DataFrame):
        """Render sector distribution chart"""
        if df.empty or 'sector' not in df.columns:
            return
        
        sector_counts = df['sector'].value_counts().head(10)
        
        fig = px.bar(
            x=sector_counts.values,
            y=sector_counts.index,
            orientation='h',
            title="Top 10 Sectors by Stock Count",
            labels={'x': 'Number of Stocks', 'y': 'Sector'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_score_distribution(df: pd.DataFrame):
        """Render score distribution histogram"""
        if df.empty or 'master_score' not in df.columns:
            return
        
        fig = px.histogram(
            df,
            x='master_score',
            nbins=20,
            title="Master Score Distribution",
            labels={'master_score': 'Master Score', 'count': 'Number of Stocks'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_category_performance(df: pd.DataFrame):
        """Render category performance chart"""
        if df.empty or 'category' not in df.columns or 'master_score' not in df.columns:
            return
        
        category_stats = df.groupby('category')['master_score'].agg(['mean', 'count']).reset_index()
        category_stats = category_stats[category_stats['count'] >= 5]  # Only categories with 5+ stocks
        
        fig = px.scatter(
            category_stats,
            x='count',
            y='mean',
            size='count',
            hover_data=['category'],
            title="Category Performance vs Count",
            labels={'count': 'Number of Stocks', 'mean': 'Average Master Score'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)