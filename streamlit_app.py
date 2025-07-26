"""
Wave Detection Ultimate 3.0 - Main Streamlit Application
========================================================
Production-Ready User Interface and Experience
Complete UI orchestration, visualizations, and user interactions

Version: 3.0.6-PRODUCTION-BULLETPROOF
Status: PRODUCTION READY - Zero UI Bug Tolerance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from io import BytesIO
import json
import warnings

# Import our production modules
from core_engine import CONFIG, SafeMath, ProductionConfig
from data_pipeline import (
    ProductionDataLoader, CacheManager, 
    calculate_comprehensive_data_quality, performance_timer
)
from filters import (
    ProductionFilterEngine, ProductionSearchEngine, 
    QuickFilterEngine, FilterStateManager
)

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# PRODUCTION VISUALIZATION ENGINE
# ============================================

class ProductionVisualizer:
    """Production-grade visualization engine with bulletproof chart generation"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart with comprehensive error handling"""
        try:
            fig = go.Figure()
            
            if df is None or df.empty:
                fig.add_annotation(
                    text="No data available for visualization",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="gray")
                )
                fig.update_layout(
                    title="Score Distribution",
                    template='plotly_white',
                    height=400
                )
                return fig
            
            # Score components with safe access
            score_configs = [
                ('position_score', 'Position', '#3498db'),
                ('volume_score', 'Volume', '#e74c3c'),
                ('momentum_score', 'Momentum', '#2ecc71'),
                ('acceleration_score', 'Acceleration', '#f39c12'),
                ('breakout_score', 'Breakout', '#9b59b6'),
                ('rvol_score', 'RVOL', '#e67e22')
            ]
            
            for score_col, label, color in score_configs:
                try:
                    if score_col in df.columns:
                        score_data = df[score_col].dropna()
                        if len(score_data) > 0:
                            fig.add_trace(go.Box(
                                y=score_data,
                                name=label,
                                marker_color=color,
                                boxpoints='outliers',
                                hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'
                            ))
                except Exception as e:
                    logger.warning(f"Error adding {label} to score distribution: {str(e)}")
                    continue
            
            fig.update_layout(
                title="Score Component Distribution",
                yaxis_title="Score (0-100)",
                template='plotly_white',
                height=400,
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating score distribution chart: {str(e)}")
            # Return empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Chart Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    @staticmethod
    def create_master_score_breakdown(df: pd.DataFrame, n: int = 20) -> go.Figure:
        """Create enhanced top stocks breakdown chart"""
        try:
            if df is None or df.empty:
                return go.Figure()
            
            # Get top stocks safely
            if 'master_score' not in df.columns:
                return go.Figure()
            
            top_df = df.nlargest(min(n, len(df)), 'master_score').copy()
            
            if len(top_df) == 0:
                return go.Figure()
            
            # Component configurations
            components = [
                ('Position', 'position_score', CONFIG.POSITION_WEIGHT, '#3498db'),
                ('Volume', 'volume_score', CONFIG.VOLUME_WEIGHT, '#e74c3c'),
                ('Momentum', 'momentum_score', CONFIG.MOMENTUM_WEIGHT, '#2ecc71'),
                ('Acceleration', 'acceleration_score', CONFIG.ACCELERATION_WEIGHT, '#f39c12'),
                ('Breakout', 'breakout_score', CONFIG.BREAKOUT_WEIGHT, '#9b59b6'),
                ('RVOL', 'rvol_score', CONFIG.RVOL_WEIGHT, '#e67e22')
            ]
            
            fig = go.Figure()
            
            for name, score_col, weight, color in components:
                try:
                    if score_col in top_df.columns:
                        weighted_contrib = top_df[score_col].fillna(50) * weight
                        
                        fig.add_trace(go.Bar(
                            name=f'{name} ({weight:.0%})',
                            y=top_df['ticker'] if 'ticker' in top_df.columns else range(len(top_df)),
                            x=weighted_contrib,
                            orientation='h',
                            marker_color=color,
                            text=[f"{val:.1f}" for val in top_df[score_col].fillna(50)],
                            textposition='inside',
                            hovertemplate=f'{name}<br>Score: %{{text}}<br>Contribution: %{{x:.1f}}<extra></extra>'
                        ))
                except Exception as e:
                    logger.warning(f"Error adding {name} component: {str(e)}")
                    continue
            
            # Add master score annotations
            try:
                for i, (idx, row) in enumerate(top_df.iterrows()):
                    master_score = SafeMath.safe_float(row.get('master_score', 50), 50)
                    fig.add_annotation(
                        x=master_score,
                        y=i,
                        text=f"{master_score:.1f}",
                        showarrow=False,
                        xanchor='left',
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='gray',
                        borderwidth=1
                    )
            except Exception as e:
                logger.warning(f"Error adding score annotations: {str(e)}")
            
            fig.update_layout(
                title=f"Top {len(top_df)} Stocks - Master Score 3.0 Breakdown",
                xaxis_title="Weighted Score Contribution",
                xaxis_range=[0, 105],
                barmode='stack',
                template='plotly_white',
                height=max(400, len(top_df) * 35),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=80, b=50, l=150, r=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating master score breakdown: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def create_sector_performance_scatter(df: pd.DataFrame) -> go.Figure:
        """Create sector performance scatter plot with error handling"""
        try:
            if df is None or df.empty or 'sector' not in df.columns:
                return go.Figure()
            
            # Aggregate by sector safely
            sector_stats = df.groupby('sector').agg({
                'master_score': ['mean', 'std', 'count'],
                'percentile': 'mean',
                'rvol': 'mean'
            }).reset_index()
            
            # Flatten column names
            sector_stats.columns = ['sector', 'avg_score', 'std_score', 'count', 'avg_percentile', 'avg_rvol']
            
            # Filter sectors with meaningful data
            sector_stats = sector_stats[sector_stats['count'] >= 3]
            
            if len(sector_stats) == 0:
                fig = go.Figure()
                fig.add_annotation(
                    text="Insufficient sector data for analysis",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Create scatter plot
            fig = px.scatter(
                sector_stats,
                x='avg_percentile',
                y='avg_score',
                size='count',
                color='avg_rvol',
                hover_data={
                    'count': True,
                    'std_score': ':.1f',
                    'avg_rvol': ':.2f'
                },
                text='sector',
                title='Sector Performance Analysis',
                labels={
                    'avg_percentile': 'Average Percentile Rank',
                    'avg_score': 'Average Master Score',
                    'count': 'Number of Stocks',
                    'avg_rvol': 'Avg RVOL'
                },
                color_continuous_scale='Viridis'
            )
            
            fig.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=1, color='white'))
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sector scatter plot: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def create_pattern_analysis(df: pd.DataFrame) -> go.Figure:
        """Create pattern frequency analysis with robust error handling"""
        try:
            if df is None or df.empty or 'patterns' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="No pattern data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title="Pattern Frequency Analysis",
                    template='plotly_white',
                    height=400
                )
                return fig
            
            # Extract all patterns safely
            all_patterns = []
            for patterns in df['patterns'].dropna():
                try:
                    if patterns and isinstance(patterns, str):
                        pattern_list = patterns.split(' | ')
                        all_patterns.extend([p.strip() for p in pattern_list if p.strip()])
                except Exception:
                    continue
            
            if not all_patterns:
                fig = go.Figure()
                fig.add_annotation(
                    text="No patterns detected in current selection",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title="Pattern Frequency Analysis",
                    template='plotly_white',
                    height=400
                )
                return fig
            
            # Count pattern frequencies
            pattern_counts = pd.Series(all_patterns).value_counts()
            
            # Create horizontal bar chart
            fig = go.Figure([
                go.Bar(
                    x=pattern_counts.values,
                    y=pattern_counts.index,
                    orientation='h',
                    marker_color='#3498db',
                    text=pattern_counts.values,
                    textposition='outside',
                    hovertemplate='Pattern: %{y}<br>Count: %{x}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Pattern Frequency Analysis",
                xaxis_title="Number of Stocks",
                yaxis_title="Pattern",
                template='plotly_white',
                height=max(400, len(pattern_counts) * 30),
                margin=dict(l=200, r=50, t=50, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern analysis: {str(e)}")
            return go.Figure()

# ============================================
# PRODUCTION EXPORT ENGINE
# ============================================

class ProductionExportEngine:
    """Production-grade export engine with comprehensive error handling"""
    
    @staticmethod
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        """Create comprehensive Excel report with bulletproof error handling"""
        try:
            if df is None or df.empty:
                raise ValueError("No data available for export")
            
            output = BytesIO()
            
            # Define export templates
            templates = {
                'day_trader': [
                    'rank', 'ticker', 'company_name', 'master_score', 'rvol', 
                    'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 
                    'volume_score', 'patterns', 'category'
                ],
                'swing_trader': [
                    'rank', 'ticker', 'company_name', 'master_score', 
                    'breakout_score', 'position_score', 'from_high_pct', 
                    'from_low_pct', 'trend_quality', 'patterns'
                ],
                'investor': [
                    'rank', 'ticker', 'company_name', 'master_score', 'pe', 
                    'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 
                    'long_term_strength', 'category', 'sector'
                ],
                'full': None  # Use all available columns
            }
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#3498db',
                    'font_color': 'white',
                    'border': 1,
                    'align': 'center'
                })
                
                number_format = workbook.add_format({'num_format': '#,##0.00'})
                percent_format = workbook.add_format({'num_format': '0.00%'})
                
                try:
                    # 1. Top 100 Stocks
                    top_100 = df.nlargest(min(100, len(df)), 'master_score')
                    
                    # Select columns based on template
                    if template in templates and templates[template]:
                        export_cols = templates[template]
                    else:
                        # Full export columns
                        export_cols = [col for col in [
                            'rank', 'ticker', 'company_name', 'master_score',
                            'position_score', 'volume_score', 'momentum_score',
                            'acceleration_score', 'breakout_score', 'rvol_score',
                            'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
                            'from_low_pct', 'from_high_pct',
                            'ret_1d', 'ret_7d', 'ret_30d', 'rvol',
                            'patterns', 'category', 'sector'
                        ] if col in df.columns]
                    
                    available_cols = [col for col in export_cols if col in top_100.columns]
                    export_df = top_100[available_cols].copy()
                    
                    # Clean data for export
                    export_df = ProductionExportEngine._clean_export_data(export_df)
                    
                    export_df.to_excel(writer, sheet_name='Top 100', index=False)
                    
                    # Format the sheet
                    worksheet = writer.sheets['Top 100']
                    for i, col in enumerate(available_cols):
                        worksheet.write(0, i, col, header_format)
                        
                except Exception as e:
                    logger.error(f"Error creating Top 100 sheet: {str(e)}")
                
                try:
                    # 2. Summary Sheet
                    summary_cols = [col for col in [
                        'rank', 'ticker', 'company_name', 'master_score',
                        'price', 'pe', 'eps_change_pct', 'ret_30d', 'rvol', 
                        'patterns', 'category', 'sector'
                    ] if col in df.columns]
                    
                    summary_df = df[summary_cols].copy()
                    summary_df = ProductionExportEngine._clean_export_data(summary_df)
                    summary_df.to_excel(writer, sheet_name='All Stocks', index=False)
                    
                except Exception as e:
                    logger.error(f"Error creating summary sheet: {str(e)}")
                
                try:
                    # 3. Sector Analysis
                    if 'sector' in df.columns:
                        sector_analysis = df.groupby('sector').agg({
                            'master_score': ['mean', 'std', 'min', 'max', 'count'],
                            'rvol': 'mean',
                            'ret_30d': 'mean'
                        }).round(2)
                        
                        sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
                        
                except Exception as e:
                    logger.warning(f"Unable to create sector analysis: {str(e)}")
                
                try:
                    # 4. Pattern Analysis
                    if 'patterns' in df.columns:
                        pattern_data = []
                        for patterns in df['patterns'].dropna():
                            if patterns:
                                for p in str(patterns).split(' | '):
                                    if p.strip():
                                        pattern_data.append(p.strip())
                        
                        if pattern_data:
                            pattern_df = pd.DataFrame(
                                pd.Series(pattern_data).value_counts()
                            ).reset_index()
                            pattern_df.columns = ['Pattern', 'Count']
                            pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
                            
                except Exception as e:
                    logger.warning(f"Unable to create pattern analysis: {str(e)}")
            
            output.seek(0)
            logger.info("Excel report created successfully")
            return output
            
        except Exception as e:
            logger.error(f"Critical error creating Excel report: {str(e)}")
            raise
    
    @staticmethod
    def _clean_export_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean data for export with safe handling"""
        try:
            df_clean = df.copy()
            
            # Handle percentage columns
            percentage_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'eps_change_pct']
            for col in percentage_cols:
                if col in df_clean.columns:
                    # Data is already in percentage format, just ensure it's numeric
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Handle score columns
            score_cols = [col for col in df_clean.columns if 'score' in col]
            for col in score_cols:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').round(1)
            
            # Handle financial columns
            if 'pe' in df_clean.columns:
                df_clean['pe'] = pd.to_numeric(df_clean['pe'], errors='coerce').round(2)
            
            if 'price' in df_clean.columns:
                df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce').round(2)
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning export data: {str(e)}")
            return df
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export with comprehensive error handling"""
        try:
            if df is None or df.empty:
                raise ValueError("No data available for CSV export")
            
            export_cols = [col for col in [
                'rank', 'ticker', 'company_name', 'master_score',
                'position_score', 'volume_score', 'momentum_score',
                'acceleration_score', 'breakout_score', 'rvol_score',
                'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
                'from_low_pct', 'from_high_pct',
                'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
                'rvol', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d',
                'patterns', 'category', 'sector', 'eps_tier', 'pe_tier'
            ] if col in df.columns]
            
            export_df = df[export_cols].copy()
            export_df = ProductionExportEngine._clean_export_data(export_df)
            
            return export_df.to_csv(index=False, encoding='utf-8')
            
        except Exception as e:
            logger.error(f"Error creating CSV export: {str(e)}")
            raise

# ============================================
# UI COMPONENT GENERATORS
# ============================================

class UIComponents:
    """Production-grade UI component generators"""
    
    @staticmethod
    def render_stock_metrics(stock_data: pd.Series) -> None:
        """Render stock metrics with bulletproof formatting"""
        try:
            cols = st.columns(6)
            
            with cols[0]:
                master_score = SafeMath.safe_float(stock_data.get('master_score', 50), 50)
                rank = int(SafeMath.safe_float(stock_data.get('rank', 9999), 9999))
                st.metric(
                    "Master Score",
                    f"{master_score:.1f}",
                    f"Rank #{rank}"
                )
            
            with cols[1]:
                price = SafeMath.safe_float(stock_data.get('price', 0), 0)
                ret_1d = SafeMath.safe_float(stock_data.get('ret_1d', 0), 0)
                price_str = f"₹{price:,.0f}" if price > 0 else "N/A"
                ret_str = f"{ret_1d:+.1f}%" if abs(ret_1d) < 10000 else None
                st.metric("Price", price_str, ret_str)
            
            with cols[2]:
                from_low = SafeMath.safe_float(stock_data.get('from_low_pct', 0), 0)
                st.metric(
                    "From Low",
                    f"{from_low:.0f}%",
                    "52-week position"
                )
            
            with cols[3]:
                ret_30d = SafeMath.safe_float(stock_data.get('ret_30d', 0), 0)
                direction = "↑" if ret_30d > 0 else "↓" if ret_30d < 0 else "→"
                st.metric(
                    "30D Return",
                    f"{ret_30d:.1f}%",
                    direction
                )
            
            with cols[4]:
                rvol = SafeMath.safe_float(stock_data.get('rvol', 1), 1)
                rvol_status = "High" if rvol > 2 else "Normal"
                st.metric(
                    "RVOL",
                    f"{rvol:.1f}x",
                    rvol_status
                )
            
            with cols[5]:
                cat_percentile = SafeMath.safe_float(stock_data.get('category_percentile', 0), 0)
                category = str(stock_data.get('category', 'Unknown'))
                st.metric(
                    "Category %ile",
                    f"{cat_percentile:.0f}",
                    category
                )
                
        except Exception as e:
            st.error(f"Error rendering metrics: {str(e)}")
    
    @staticmethod
    def format_currency(value: Any, currency: str = "₹") -> str:
        """Format currency values safely"""
        try:
            num_value = SafeMath.safe_float(value, 0)
            if num_value == 0:
                return "N/A"
            
            if num_value >= 10000000:  # 1 crore
                return f"{currency}{num_value/10000000:.1f}Cr"
            elif num_value >= 100000:  # 1 lakh
                return f"{currency}{num_value/100000:.1f}L"
            elif num_value >= 1000:
                return f"{currency}{num_value/1000:.0f}K"
            else:
                return f"{currency}{num_value:,.0f}"
                
        except Exception:
            return "N/A"
    
    @staticmethod
    def format_percentage(value: Any, decimal_places: int = 1) -> str:
        """Format percentage values safely"""
        try:
            num_value = SafeMath.safe_float(value, None)
            if num_value is None:
                return "N/A"
            
            if abs(num_value) >= 10000:
                return f"{num_value/1000:+.1f}K%"
            elif abs(num_value) >= 1000:
                return f"{num_value:+.0f}%"
            else:
                return f"{num_value:+.{decimal_places}f}%"
                
        except Exception:
            return "N/A"
    
    @staticmethod
    def format_pe_ratio(value: Any) -> str:
        """Format PE ratio values safely"""
        try:
            pe_value = SafeMath.safe_float(value, None)
            if pe_value is None or pe_value <= 0:
                return "Loss"
            
            if np.isinf(pe_value):
                return "∞"
            elif pe_value > 10000:
                return f"{pe_value/1000:.0f}K"
            elif pe_value > 1000:
                return f"{pe_value:.0f}"
            else:
                return f"{pe_value:.1f}x"
                
        except Exception:
            return "N/A"

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application with production-grade error handling"""
    try:
        # Page configuration
        st.set_page_config(
            page_title="Wave Detection Ultimate 3.0",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Apply custom CSS
        apply_custom_css()
        
        # Render header
        render_header()
        
        # Clean up cache periodically
        CacheManager.cleanup_session_state()
        
        # Load and process data
        ranked_df, data_timestamp = load_application_data()
        
        if ranked_df.empty:
            st.error("❌ Unable to load data. Please check your connection and try again.")
            st.stop()
        
        # Render sidebar
        render_sidebar(ranked_df)
        
        # Apply quick actions
        filtered_df = apply_quick_actions(ranked_df)
        
        # Apply filters
        filters = get_current_filters()
        final_df = ProductionFilterEngine.apply_filters(filtered_df, filters)
        
        # Show filter status
        show_filter_status(ranked_df, final_df, filters)
        
        # Render summary metrics
        render_summary_metrics(final_df, ranked_df)
        
        # Render main tabs
        render_main_tabs(final_df, ranked_df)
        
        # Render footer
        render_footer()
        
    except Exception as e:
        logger.error(f"Critical application error: {str(e)}")
        st.error(f"💥 Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")

def initialize_session_state():
    """Initialize session state with default values"""
    try:
        defaults = {
            'search_query': "",
            'last_refresh': datetime.now(),
            'user_preferences': {
                'default_top_n': CONFIG.DEFAULT_TOP_N,
                'display_mode': 'Technical',
                'last_filters': {}
            },
            'quick_filter': None,
            'quick_filter_applied': False,
            'performance_metrics': {},
            'active_filter_count': 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")

def apply_custom_css():
    """Apply custom CSS with mobile responsiveness"""
    try:
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
        @media (max-width: 768px) {
            .stDataFrame {
                font-size: 12px;
            }
            div[data-testid="metric-container"] {
                padding: 3%;
            }
        }
        .stDataFrame > div {
            overflow-x: auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error applying CSS: {str(e)}")

def render_header():
    """Render application header"""
    try:
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
                Production-Ready Stock Ranking System with Bulletproof Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error rendering header: {str(e)}")
        st.title("🌊 Wave Detection Ultimate 3.0")

@performance_timer
def load_application_data() -> Tuple[pd.DataFrame, datetime]:
    """Load application data with caching and error handling"""
    try:
        # Check for cached data
        if ('ranked_df' in st.session_state and 
            (datetime.now() - st.session_state.last_refresh).seconds < 3600):
            return st.session_state.ranked_df, st.session_state.data_timestamp
        
        # Load fresh data
        with st.spinner("📥 Loading and processing data..."):
            ranked_df, data_timestamp = ProductionDataLoader.load_and_process_data()
            
            if not ranked_df.empty:
                st.session_state.ranked_df = ranked_df
                st.session_state.data_timestamp = data_timestamp
                st.session_state.last_refresh = datetime.now()
                
                # Calculate data quality
                st.session_state.data_quality = calculate_comprehensive_data_quality(ranked_df)
                
                return ranked_df, data_timestamp
            
        return pd.DataFrame(), datetime.now()
        
    except Exception as e:
        logger.error(f"Error loading application data: {str(e)}")
        # Return cached data if available
        if 'ranked_df' in st.session_state:
            return st.session_state.ranked_df, st.session_state.data_timestamp
        return pd.DataFrame(), datetime.now()

def render_sidebar(df: pd.DataFrame):
    """Render sidebar with filters and controls"""
    try:
        with st.sidebar:
            st.markdown("### 🎯 Quick Actions")
            
            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", type="primary", use_container_width=True):
                    st.cache_data.clear()
                    st.session_state.last_refresh = datetime.now()
                    st.rerun()
            
            with col2:
                if st.button("🧹 Clear Cache", use_container_width=True):
                    CacheManager.clear_all_caches()
                    st.success("Cache cleared!")
                    time.sleep(0.5)
                    st.rerun()
            
            # Data quality display
            display_data_quality()
            
            st.markdown("---")
            
            # Display mode toggle
            render_display_mode_selector()
            
            st.markdown("---")
            st.markdown("### 🔍 Smart Filters")
            
            # Show active filter count
            show_active_filter_summary()
            
            # Render filter controls
            render_filter_controls(df)
            
            # Clear filters button
            render_clear_filters_button()
            
    except Exception as e:
        logger.error(f"Error rendering sidebar: {str(e)}")

def display_data_quality():
    """Display data quality metrics in sidebar"""
    try:
        if 'data_quality' in st.session_state:
            with st.expander("📊 Data Quality", expanded=True):
                quality = st.session_state.data_quality
                
                col1, col2 = st.columns(2)
                with col1:
                    completeness = quality.get('completeness', 0)
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
                    st.caption(f"Last updated: {timestamp.strftime('%I:%M %p')}")
                    
    except Exception as e:
        logger.error(f"Error displaying data quality: {str(e)}")

def render_display_mode_selector():
    """Render display mode selector"""
    try:
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if st.session_state.user_preferences['display_mode'] == 'Technical' else 1,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data",
            key="display_mode_toggle"
        )
        
        st.session_state.user_preferences['display_mode'] = display_mode
        
    except Exception as e:
        logger.error(f"Error rendering display mode selector: {str(e)}")

def show_active_filter_summary():
    """Show summary of active filters"""
    try:
        active_count = st.session_state.get('active_filter_count', 0)
        quick_filter_active = st.session_state.get('quick_filter_applied', False)
        
        if active_count > 0 or quick_filter_active:
            filter_text = f"{active_count} filter{'s' if active_count != 1 else ''}"
            if quick_filter_active:
                filter_text += " + quick filter"
            st.info(f"🔍 **{filter_text} active**")
            
    except Exception as e:
        logger.error(f"Error showing filter summary: {str(e)}")

def render_filter_controls(df: pd.DataFrame):
    """Render all filter controls"""
    try:
        if df.empty:
            st.warning("No data available for filtering")
            return
        
        # Category filter
        render_category_filter(df)
        
        # Sector filter
        render_sector_filter(df)
        
        # Score filter
        render_score_filter()
        
        # Pattern filter
        render_pattern_filter(df)
        
        # Trend filter
        render_trend_filter()
        
        # Advanced filters
        render_advanced_filters(df)
        
    except Exception as e:
        logger.error(f"Error rendering filter controls: {str(e)}")

def render_category_filter(df: pd.DataFrame):
    """Render category filter with smart updates"""
    try:
        current_filters = get_current_filters()
        categories = ProductionFilterEngine.get_unique_values(
            df, 'category', filters=current_filters
        )
        
        if categories:
            category_counts = df['category'].value_counts()
            category_options = [
                f"{cat} ({category_counts.get(cat, 0)})" 
                for cat in categories
            ]
            
            selected_categories = st.multiselect(
                "Market Cap Category",
                options=category_options,
                default=st.session_state.get('category_filter', []),
                placeholder="Select categories (empty = All)",
                key="category_filter"
            )
            
    except Exception as e:
        logger.error(f"Error rendering category filter: {str(e)}")

def render_sector_filter(df: pd.DataFrame):
    """Render sector filter with smart updates"""
    try:
        current_filters = get_current_filters()
        sectors = ProductionFilterEngine.get_unique_values(
            df, 'sector', filters=current_filters
        )
        
        if sectors:
            selected_sectors = st.multiselect(
                "Sector",
                options=sectors,
                default=st.session_state.get('sector_filter', []),
                placeholder="Select sectors (empty = All)",
                key="sector_filter"
            )
            
    except Exception as e:
        logger.error(f"Error rendering sector filter: {str(e)}")

def render_score_filter():
    """Render score filter"""
    try:
        min_score = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=st.session_state.get('min_score', 0),
            step=5,
            help="Filter stocks by minimum score",
            key="min_score"
        )
        
    except Exception as e:
        logger.error(f"Error rendering score filter: {str(e)}")

def render_pattern_filter(df: pd.DataFrame):
    """Render pattern filter"""
    try:
        if 'patterns' in df.columns:
            all_patterns = set()
            for patterns in df['patterns'].dropna():
                if patterns:
                    all_patterns.update(str(patterns).split(' | '))
            
            if all_patterns:
                selected_patterns = st.multiselect(
                    "Patterns",
                    options=sorted(all_patterns),
                    default=st.session_state.get('patterns', []),
                    placeholder="Select patterns (empty = All)",
                    help="Filter by specific patterns",
                    key="patterns"
                )
                
    except Exception as e:
        logger.error(f"Error rendering pattern filter: {str(e)}")

def render_trend_filter():
    """Render trend filter"""
    try:
        st.markdown("#### 📈 Trend Strength")
        trend_options = {
            "All Trends": (0, 100),
            "🔥 Strong Uptrend (80+)": (80, 100),
            "✅ Good Uptrend (60-79)": (60, 79),
            "➡️ Neutral Trend (40-59)": (40, 59),
            "⚠️ Weak/Downtrend (<40)": (0, 39)
        }
        
        selected_trend = st.selectbox(
            "Trend Quality",
            options=list(trend_options.keys()),
            index=0,
            help="Filter stocks by trend strength",
            key="trend_filter"
        )
        
        st.session_state['trend_range'] = trend_options[selected_trend]
        
    except Exception as e:
        logger.error(f"Error rendering trend filter: {str(e)}")

def render_advanced_filters(df: pd.DataFrame):
    """Render advanced filters in expander"""
    try:
        with st.expander("🔧 Advanced Filters"):
            # Show fundamentals only in hybrid mode
            show_fundamentals = (
                st.session_state.user_preferences['display_mode'] == 
                "Hybrid (Technical + Fundamentals)"
            )
            
            if show_fundamentals:
                render_fundamental_filters(df)
            
            render_tier_filters(df)
            
    except Exception as e:
        logger.error(f"Error rendering advanced filters: {str(e)}")

def render_fundamental_filters(df: pd.DataFrame):
    """Render fundamental data filters"""
    try:
        st.markdown("**💰 Fundamental Filters**")
        
        # EPS change filter
        if 'eps_change_pct' in df.columns:
            eps_change_input = st.text_input(
                "Min EPS Change %",
                value=st.session_state.get('min_eps_change', ""),
                placeholder="e.g. -50 or leave empty",
                help="Enter minimum EPS growth percentage",
                key="min_eps_change"
            )
        
        # PE ratio filters
        if 'pe' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                min_pe_input = st.text_input(
                    "Min PE Ratio",
                    value=st.session_state.get('min_pe', ""),
                    placeholder="e.g. 10",
                    key="min_pe"
                )
            
            with col2:
                max_pe_input = st.text_input(
                    "Max PE Ratio",
                    value=st.session_state.get('max_pe', ""),
                    placeholder="e.g. 30",
                    key="max_pe"
                )
        
        # Data completeness requirement
        require_fundamentals = st.checkbox(
            "Only show stocks with PE and EPS data",
            value=st.session_state.get('require_fundamental_data', False),
            help="Filter out stocks missing fundamental data",
            key="require_fundamental_data"
        )
        
    except Exception as e:
        logger.error(f"Error rendering fundamental filters: {str(e)}")

def render_tier_filters(df: pd.DataFrame):
    """Render tier-based filters"""
    try:
        # EPS tier filter
        if 'eps_tier' in df.columns:
            eps_tiers = ProductionFilterEngine.get_unique_values(df, 'eps_tier')
            if eps_tiers:
                selected_eps_tiers = st.multiselect(
                    "EPS Tier",
                    options=eps_tiers,
                    default=st.session_state.get('eps_tier_filter', []),
                    key="eps_tier_filter"
                )
        
        # PE tier filter
        if 'pe_tier' in df.columns:
            pe_tiers = ProductionFilterEngine.get_unique_values(df, 'pe_tier')
            if pe_tiers:
                selected_pe_tiers = st.multiselect(
                    "PE Tier",
                    options=pe_tiers,
                    default=st.session_state.get('pe_tier_filter', []),
                    key="pe_tier_filter"
                )
        
        # Price tier filter
        if 'price_tier' in df.columns:
            price_tiers = ProductionFilterEngine.get_unique_values(df, 'price_tier')
            if price_tiers:
                selected_price_tiers = st.multiselect(
                    "Price Range",
                    options=price_tiers,
                    default=st.session_state.get('price_tier_filter', []),
                    key="price_tier_filter"
                )
                
    except Exception as e:
        logger.error(f"Error rendering tier filters: {str(e)}")

def render_clear_filters_button():
    """Render clear filters button with proper state management"""
    try:
        # Check if triggered from main area
        if st.session_state.get('trigger_clear', False):
            st.session_state['trigger_clear'] = False
            clear_clicked = True
        else:
            clear_clicked = st.button(
                "🗑️ Clear All Filters", 
                use_container_width=True, 
                type="primary"
            )
        
        if clear_clicked:
            clear_all_filters()
            st.success("✅ All filters cleared!")
            st.rerun()
            
    except Exception as e:
        logger.error(f"Error rendering clear filters button: {str(e)}")

def clear_all_filters():
    """Clear all filter states"""
    try:
        # Reset all filter-related session state
        filter_keys = [
            'category_filter', 'sector_filter', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'patterns',
            'min_score', 'trend_filter', 'min_eps_change',
            'min_pe', 'max_pe', 'require_fundamental_data',
            'quick_filter', 'quick_filter_applied'
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if key in ['min_score']:
                    st.session_state[key] = 0
                elif key in ['trend_filter']:
                    st.session_state[key] = 'All Trends'
                elif key in ['require_fundamental_data', 'quick_filter_applied']:
                    st.session_state[key] = False
                elif key in ['quick_filter']:
                    st.session_state[key] = None
                elif key in ['min_eps_change', 'min_pe', 'max_pe']:
                    st.session_state[key] = ""
                else:
                    st.session_state[key] = []
        
        # Reset trend range
        st.session_state['trend_range'] = (0, 100)
        
        # Reset active filter count
        st.session_state['active_filter_count'] = 0
        
    except Exception as e:
        logger.error(f"Error clearing filters: {str(e)}")

def apply_quick_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quick action filters"""
    try:
        st.markdown("### ⚡ Quick Actions")
        qa_cols = st.columns(5)
        
        # Quick action buttons
        with qa_cols[0]:
            if st.button("📈 Top Gainers", use_container_width=True):
                st.session_state['quick_filter'] = 'top_gainers'
                st.session_state['quick_filter_applied'] = True
        
        with qa_cols[1]:
            if st.button("🔥 Volume Surges", use_container_width=True):
                st.session_state['quick_filter'] = 'volume_surges'
                st.session_state['quick_filter_applied'] = True
        
        with qa_cols[2]:
            if st.button("🎯 Breakout Ready", use_container_width=True):
                st.session_state['quick_filter'] = 'breakout_ready'
                st.session_state['quick_filter_applied'] = True
        
        with qa_cols[3]:
            if st.button("💎 Hidden Gems", use_container_width=True):
                st.session_state['quick_filter'] = 'hidden_gems'
                st.session_state['quick_filter_applied'] = True
        
        with qa_cols[4]:
            if st.button("🌊 Show All", use_container_width=True):
                st.session_state['quick_filter'] = None
                st.session_state['quick_filter_applied'] = False
        
        # Apply quick filter if active
        quick_filter = st.session_state.get('quick_filter')
        if quick_filter and st.session_state.get('quick_filter_applied', False):
            filtered_df = QuickFilterEngine.apply_quick_filter(df, quick_filter)
            
            # Show quick filter status
            filter_names = {
                'top_gainers': '📈 Top Gainers',
                'volume_surges': '🔥 Volume Surges',
                'breakout_ready': '🎯 Breakout Ready',
                'hidden_gems': '💎 Hidden Gems'
            }
            filter_name = filter_names.get(quick_filter, quick_filter)
            st.info(f"**Quick Filter Active:** {filter_name} | Showing {len(filtered_df):,} stocks")
            
            return filtered_df
        
        return df
        
    except Exception as e:
        logger.error(f"Error applying quick actions: {str(e)}")
        return df

def get_current_filters() -> Dict[str, Any]:
    """Get current filter state"""
    try:
        filters = {}
        
        # List filters
        list_filters = [
            ('categories', 'category_filter'),
            ('sectors', 'sector_filter'),
            ('eps_tiers', 'eps_tier_filter'),
            ('pe_tiers', 'pe_tier_filter'),
            ('price_tiers', 'price_tier_filter'),
            ('patterns', 'patterns')
        ]
        
        for filter_key, session_key in list_filters:
            if session_key in st.session_state:
                value = st.session_state[session_key]
                if isinstance(value, list) and value:
                    # Extract actual values from formatted strings
                    if filter_key == 'categories' and value:
                        filters[filter_key] = [cat.split(' (')[0] for cat in value]
                    else:
                        filters[filter_key] = value
        
        # Numeric filters
        numeric_filters = ['min_score', 'min_eps_change', 'min_pe', 'max_pe']
        for filter_key in numeric_filters:
            if filter_key in st.session_state:
                value = st.session_state[filter_key]
                if filter_key == 'min_score':
                    if value > 0:
                        filters[filter_key] = value
                elif isinstance(value, str) and value.strip():
                    try:
                        filters[filter_key] = float(value.strip())
                    except ValueError:
                        pass
                elif isinstance(value, (int, float)) and value is not None:
                    filters[filter_key] = value
        
        # Boolean filters
        if st.session_state.get('require_fundamental_data', False):
            filters['require_fundamental_data'] = True
        
        # Trend filter
        trend_filter = st.session_state.get('trend_filter', 'All Trends')
        if trend_filter != 'All Trends':
            filters['trend_filter'] = trend_filter
            filters['trend_range'] = st.session_state.get('trend_range', (0, 100))
        
        return FilterStateManager.validate_filter_state(filters)
        
    except Exception as e:
        logger.error(f"Error getting current filters: {str(e)}")
        return {}

def show_filter_status(original_df: pd.DataFrame, filtered_df: pd.DataFrame, filters: Dict[str, Any]):
    """Show filter status and results"""
    try:
        quick_filter_active = st.session_state.get('quick_filter_applied', False)
        filter_count = len([k for k, v in filters.items() if v])
        
        if quick_filter_active or filter_count > 0:
            col1, col2 = st.columns([5, 1])
            
            with col1:
                status_parts = []
                
                # Quick filter status
                if quick_filter_active:
                    quick_filter = st.session_state.get('quick_filter')
                    filter_names = {
                        'top_gainers': '📈 Top Gainers',
                        'volume_surges': '🔥 Volume Surges',
                        'breakout_ready': '🎯 Breakout Ready',
                        'hidden_gems': '💎 Hidden Gems'
                    }
                    status_parts.append(filter_names.get(quick_filter, 'Quick Filter'))
                
                # Regular filters
                if filter_count > 0:
                    status_parts.append(f"{filter_count} filter{'s' if filter_count > 1 else ''}")
                
                status_text = " + ".join(status_parts)
                st.info(f"**Active:** {status_text} | **{len(filtered_df):,} stocks** shown")
            
            with col2:
                if st.button("Clear Filters", type="secondary"):
                    st.session_state['trigger_clear'] = True
                    st.rerun()
        
        # Update active filter count for sidebar
        total_active = filter_count + (1 if quick_filter_active else 0)
        st.session_state['active_filter_count'] = total_active
        
    except Exception as e:
        logger.error(f"Error showing filter status: {str(e)}")

def render_summary_metrics(filtered_df: pd.DataFrame, original_df: pd.DataFrame):
    """Render summary metrics"""
    try:
        cols = st.columns(6)
        
        with cols[0]:
            total_filtered = len(filtered_df)
            total_original = len(original_df)
            pct_of_all = (total_filtered/total_original*100) if total_original > 0 else 0
            st.metric(
                "Total Stocks",
                f"{total_filtered:,}",
                f"{pct_of_all:.0f}% of {total_original:,}"
            )
        
        with cols[1]:
            if not filtered_df.empty and 'master_score' in filtered_df.columns:
                avg_score = filtered_df['master_score'].mean()
                std_score = filtered_df['master_score'].std()
                st.metric(
                    "Avg Score", 
                    f"{avg_score:.1f}",
                    f"σ={std_score:.1f}"
                )
            else:
                st.metric("Avg Score", "N/A")
        
        with cols[2]:
            show_fundamentals = (
                st.session_state.user_preferences['display_mode'] == 
                "Hybrid (Technical + Fundamentals)"
            )
            
            if show_fundamentals and 'pe' in filtered_df.columns:
                valid_pe = (
                    filtered_df['pe'].notna() & 
                    (filtered_df['pe'] > 0) & 
                    (filtered_df['pe'] < CONFIG.MAX_VALID_PE)
                )
                pe_count = valid_pe.sum()
                
                if pe_count > 0:
                    median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                    pe_pct = (pe_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                    st.metric(
                        "Median PE",
                        f"{median_pe:.1f}x",
                        f"{pe_pct:.0f}% have data"
                    )
                else:
                    st.metric("PE Data", "Limited")
            else:
                if not filtered_df.empty and 'master_score' in filtered_df.columns:
                    min_score = filtered_df['master_score'].min()
                    max_score = filtered_df['master_score'].max()
                    st.metric("Score Range", f"{min_score:.1f}-{max_score:.1f}")
                else:
                    st.metric("Score Range", "N/A")
        
        with cols[3]:
            show_fundamentals = (
                st.session_state.user_preferences['display_mode'] == 
                "Hybrid (Technical + Fundamentals)"
            )
            
            if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
                valid_eps = (
                    filtered_df['eps_change_pct'].notna() & 
                    ~np.isinf(filtered_df['eps_change_pct'])
                )
                positive_growth = valid_eps & (filtered_df['eps_change_pct'] > 0)
                strong_growth = valid_eps & (filtered_df['eps_change_pct'] > 50)
                
                growth_count = positive_growth.sum()
                strong_count = strong_growth.sum()
                
                if growth_count > 0:
                    st.metric(
                        "EPS Growth +ve",
                        f"{growth_count}",
                        f"{strong_count} >50%"
                    )
                else:
                    st.metric("EPS Growth", "No data")
            else:
                if 'acceleration_score' in filtered_df.columns:
                    accelerating = (filtered_df['acceleration_score'] >= 80).sum()
                    st.metric("Accelerating", f"{accelerating}")
                else:
                    st.metric("Accelerating", "0")
        
        with cols[4]:
            if 'rvol' in filtered_df.columns:
                high_rvol = (filtered_df['rvol'] > 2).sum()
                st.metric("High RVOL", f"{high_rvol}")
            else:
                st.metric("High RVOL", "0")
        
        with cols[5]:
            if 'trend_quality' in filtered_df.columns:
                strong_trends = (filtered_df['trend_quality'] >= 80).sum()
                total = len(filtered_df)
                pct = (strong_trends/total*100) if total > 0 else 0
                st.metric(
                    "Strong Trends",
                    f"{strong_trends}",
                    f"{pct:.0f}%"
                )
            else:
                with_patterns = (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0
                st.metric("With Patterns", f"{with_patterns}")
                
    except Exception as e:
        logger.error(f"Error rendering summary metrics: {str(e)}")

def render_main_tabs(filtered_df: pd.DataFrame, original_df: pd.DataFrame):
    """Render main application tabs"""
    try:
        tabs = st.tabs(["🏆 Rankings", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"])
        
        with tabs[0]:
            render_rankings_tab(filtered_df)
        
        with tabs[1]:
            render_analysis_tab(filtered_df)
        
        with tabs[2]:
            render_search_tab(filtered_df)
        
        with tabs[3]:
            render_export_tab(filtered_df)
        
        with tabs[4]:
            render_about_tab()
            
    except Exception as e:
        logger.error(f"Error rendering main tabs: {str(e)}")

def render_rankings_tab(df: pd.DataFrame):
    """Render rankings tab"""
    try:
        st.markdown("### 🏆 Top Ranked Stocks")
        
        if df.empty:
            st.warning("No stocks match the selected filters.")
            return
        
        # Display options
        col1, col2 = st.columns([2, 4])
        
        with col1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(
                    st.session_state.user_preferences['default_top_n']
                )
            )
            st.session_state.user_preferences['default_top_n'] = display_count
        
        with col2:
            sort_options = ['Rank', 'Master Score', 'RVOL', 'Momentum']
            if 'trend_quality' in df.columns:
                sort_options.append('Trend')
            
            sort_by = st.selectbox("Sort by", options=sort_options, index=0)
        
        # Get and sort display data
        display_df = df.head(display_count).copy()
        
        if sort_by == 'Master Score':
            display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL':
            display_df = display_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum':
            display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Trend' and 'trend_quality' in display_df.columns:
            display_df = display_df.sort_values('trend_quality', ascending=False)
        
        # Render data table
        render_stocks_table(display_df)
        
        # Show quick statistics
        render_quick_statistics(df)
        
    except Exception as e:
        logger.error(f"Error rendering rankings tab: {str(e)}")
        st.error("Error displaying rankings. Please try refreshing.")

def render_stocks_table(df: pd.DataFrame):
    """Render stocks data table with proper formatting"""
    try:
        if df.empty:
            return
        
        show_fundamentals = (
            st.session_state.user_preferences['display_mode'] == 
            "Hybrid (Technical + Fundamentals)"
        )
        
        # Prepare display dataframe
        display_df = df.copy()
        
        # Add trend indicator if available
        if 'trend_quality' in display_df.columns:
            display_df['trend_indicator'] = display_df['trend_quality'].apply(
                lambda x: ("🔥" if pd.notna(x) and x >= 80 else
                          "✅" if pd.notna(x) and x >= 60 else
                          "➡️" if pd.notna(x) and x >= 40 else
                          "⚠️" if pd.notna(x) else "➖")
            )
        
        # Select and format columns
        display_cols = ['rank', 'ticker', 'company_name', 'master_score']
        
        if 'trend_indicator' in display_df.columns:
            display_cols.append('trend_indicator')
        
        display_cols.append('price')
        
        # Add fundamental columns if enabled
        if show_fundamentals:
            if 'pe' in display_df.columns:
                display_df['pe_formatted'] = display_df['pe'].apply(UIComponents.format_pe_ratio)
                display_cols.append('pe_formatted')
            
            if 'eps_change_pct' in display_df.columns:
                display_df['eps_change_formatted'] = display_df['eps_change_pct'].apply(
                    UIComponents.format_percentage
                )
                display_cols.append('eps_change_formatted')
        
        # Add remaining columns
        remaining_cols = ['from_low_pct', 'ret_30d', 'rvol', 'patterns', 'category', 'sector']
        for col in remaining_cols:
            if col in display_df.columns:
                display_cols.append(col)
        
        # Format columns
        format_display_columns(display_df, display_cols)
        
        # Rename columns for display
        column_renames = {
            'rank': 'Rank',
            'ticker': 'Ticker',
            'company_name': 'Company',
            'master_score': 'Score',
            'trend_indicator': 'Trend',
            'price': 'Price',
            'pe_formatted': 'PE',
            'eps_change_formatted': 'EPS Δ%',
            'from_low_pct': 'From Low',
            'ret_30d': '30D Ret',
            'rvol': 'RVOL',
            'patterns': 'Patterns',
            'category': 'Category',
            'sector': 'Sector'
        }
        
        final_cols = [col for col in display_cols if col in display_df.columns]
        final_df = display_df[final_cols].copy()
        
        # Rename columns
        final_df.columns = [column_renames.get(col, col) for col in final_df.columns]
        
        # Display table
        st.dataframe(
            final_df,
            use_container_width=True,
            height=min(600, len(final_df) * 35 + 50),
            hide_index=True
        )
        
    except Exception as e:
        logger.error(f"Error rendering stocks table: {str(e)}")
        st.error("Error displaying data table")

def format_display_columns(df: pd.DataFrame, columns: List[str]):
    """Format columns for display"""
    try:
        # Format numeric columns
        format_rules = {
            'master_score': lambda x: f"{SafeMath.safe_float(x, 0):.1f}",
            'price': lambda x: UIComponents.format_currency(x),
            'from_low_pct': lambda x: f"{SafeMath.safe_float(x, 0):.0f}%",
            'ret_30d': lambda x: UIComponents.format_percentage(x),
            'rvol': lambda x: f"{SafeMath.safe_float(x, 1):.1f}x"
        }
        
        for col, formatter in format_rules.items():
            if col in df.columns and col in columns:
                try:
                    df[col] = df[col].apply(formatter)
                except Exception as e:
                    logger.warning(f"Error formatting {col}: {str(e)}")
                    df[col] = df[col].fillna('N/A')
                    
    except Exception as e:
        logger.error(f"Error formatting display columns: {str(e)}")

def render_quick_statistics(df: pd.DataFrame):
    """Render quick statistics below table"""
    try:
        with st.expander("📊 Quick Statistics"):
            stat_cols = st.columns(4)
            
            with stat_cols[0]:
                st.markdown("**Score Distribution**")
                if 'master_score' in df.columns and not df.empty:
                    st.text(f"Max: {df['master_score'].max():.1f}")
                    st.text(f"Min: {df['master_score'].min():.1f}")
                    st.text(f"Std: {df['master_score'].std():.1f}")
                else:
                    st.text("No score data")
            
            with stat_cols[1]:
                st.markdown("**Returns (30D)**")
                if 'ret_30d' in df.columns and not df.empty:
                    st.text(f"Max: {df['ret_30d'].max():.1f}%")
                    st.text(f"Avg: {df['ret_30d'].mean():.1f}%")
                    st.text(f"Positive: {(df['ret_30d'] > 0).sum()}")
                else:
                    st.text("No return data")
            
            with stat_cols[2]:
                st.markdown("**RVOL Stats**")
                if 'rvol' in df.columns and not df.empty:
                    st.text(f"Max: {df['rvol'].max():.1f}x")
                    st.text(f"Avg: {df['rvol'].mean():.1f}x")
                    st.text(f">2x: {(df['rvol'] > 2).sum()}")
                else:
                    st.text("No RVOL data")
            
            with stat_cols[3]:
                st.markdown("**Categories**")
                if 'category' in df.columns and not df.empty:
                    top_categories = df['category'].value_counts().head(3)
                    for cat, count in top_categories.items():
                        st.text(f"{cat}: {count}")
                else:
                    st.text("No category data")
                    
    except Exception as e:
        logger.error(f"Error rendering quick statistics: {str(e)}")

def render_analysis_tab(df: pd.DataFrame):
    """Render analysis tab"""
    try:
        st.markdown("### 📊 Market Analysis")
        
        if df.empty:
            st.info("No data available for analysis.")
            return
        
        # Score distribution and pattern analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = ProductionVisualizer.create_score_distribution(df)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_patterns = ProductionVisualizer.create_pattern_analysis(df)
            st.plotly_chart(fig_patterns, use_container_width=True)
        
        # Master score breakdown
        st.markdown("#### Top Stocks Breakdown")
        fig_breakdown = ProductionVisualizer.create_master_score_breakdown(df, 15)
        st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Sector analysis
        render_sector_analysis(df)
        
        # Category analysis
        render_category_analysis(df)
        
    except Exception as e:
        logger.error(f"Error rendering analysis tab: {str(e)}")
        st.error("Error loading analysis. Please try refreshing.")

def render_sector_analysis(df: pd.DataFrame):
    """Render sector analysis section"""
    try:
        st.markdown("#### Sector Performance")
        
        if 'sector' not in df.columns:
            st.info("Sector data not available.")
            return
        
        sector_df = df.groupby('sector').agg({
            'master_score': ['mean', 'count'],
            'rvol': 'mean',
            'ret_30d': 'mean' if 'ret_30d' in df.columns else 'size'
        }).round(2)
        
        if not sector_df.empty:
            sector_df.columns = ['Avg Score', 'Count', 'Avg RVOL', 'Avg 30D Ret']
            sector_df = sector_df.sort_values('Avg Score', ascending=False)
            
            # Add percentage of total
            sector_df['% of Total'] = (sector_df['Count'] / len(df) * 100).round(1)
            
            st.dataframe(
                sector_df.style.background_gradient(subset=['Avg Score']),
                use_container_width=True
            )
        else:
            st.info("No sector data available for analysis.")
            
    except Exception as e:
        logger.error(f"Error in sector analysis: {str(e)}")
        st.warning("Unable to display sector analysis.")

def render_category_analysis(df: pd.DataFrame):
    """Render category analysis section"""
    try:
        st.markdown("#### Category Performance")
        
        if 'category' not in df.columns:
            st.info("Category data not available.")
            return
        
        category_df = df.groupby('category').agg({
            'master_score': ['mean', 'count'],
            'category_percentile': 'mean' if 'category_percentile' in df.columns else 'size'
        }).round(2)
        
        if not category_df.empty:
            category_df.columns = ['Avg Score', 'Count', 'Avg Cat %ile']
            category_df = category_df.sort_values('Avg Score', ascending=False)
            
            st.dataframe(
                category_df.style.background_gradient(subset=['Avg Score']),
                use_container_width=True
            )
        else:
            st.info("No category data available for analysis.")
            
    except Exception as e:
        logger.error(f"Error in category analysis: {str(e)}")
        st.warning("Unable to display category analysis.")

def render_search_tab(df: pd.DataFrame):
    """Render search tab"""
    try:
        st.markdown("### 🔍 Advanced Stock Search")
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search stocks",
                placeholder="Enter ticker or company name...",
                help="Search by ticker symbol or company name",
                key="search_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        
        # Perform search
        if search_query or search_clicked:
            if search_query.strip():
                search_results = ProductionSearchEngine.search_stocks(df, search_query)
                
                if not search_results.empty:
                    st.success(f"Found {len(search_results)} matching stock(s)")
                    render_search_results(search_results)
                else:
                    st.warning("No stocks found matching your search criteria.")
            else:
                st.info("Please enter a search term.")
                
    except Exception as e:
        logger.error(f"Error rendering search tab: {str(e)}")
        st.error("Error in search functionality.")

def render_search_results(results_df: pd.DataFrame):
    """Render detailed search results with comprehensive stock information"""
    try:
        show_fundamentals = (
            st.session_state.user_preferences['display_mode'] == 
            "Hybrid (Technical + Fundamentals)"
        )
        
        # Display each result in detail
        for idx, stock in results_df.iterrows():
            with st.expander(
                f"📊 {stock.get('ticker', 'N/A')} - {stock.get('company_name', 'N/A')} "
                f"(Rank #{int(SafeMath.safe_float(stock.get('rank', 9999), 9999))})",
                expanded=True
            ):
                # Header metrics
                UIComponents.render_stock_metrics(stock)
                
                # Score breakdown
                render_score_breakdown(stock)
                
                # Additional details
                render_stock_details(stock, show_fundamentals)
                
    except Exception as e:
        logger.error(f"Error rendering search results: {str(e)}")
        st.error("Error displaying search results")

def render_score_breakdown(stock: pd.Series):
    """Render score breakdown for individual stock"""
    try:
        st.markdown("#### 📈 Score Components")
        score_cols = st.columns(6)
        
        components = [
            ("Position", stock.get('position_score', 50), CONFIG.POSITION_WEIGHT),
            ("Volume", stock.get('volume_score', 50), CONFIG.VOLUME_WEIGHT),
            ("Momentum", stock.get('momentum_score', 50), CONFIG.MOMENTUM_WEIGHT),
            ("Acceleration", stock.get('acceleration_score', 50), CONFIG.ACCELERATION_WEIGHT),
            ("Breakout", stock.get('breakout_score', 50), CONFIG.BREAKOUT_WEIGHT),
            ("RVOL", stock.get('rvol_score', 50), CONFIG.RVOL_WEIGHT)
        ]
        
        for i, (name, score, weight) in enumerate(components):
            with score_cols[i]:
                safe_score = SafeMath.safe_float(score, 50)
                
                # Color coding
                if safe_score >= 80:
                    color = "🟢"
                elif safe_score >= 60:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(
                    f"**{name}**<br>"
                    f"{color} {safe_score:.0f}<br>"
                    f"<small>Weight: {weight:.0%}</small>",
                    unsafe_allow_html=True
                )
        
        # Show patterns if available
        patterns = stock.get('patterns', '')
        if patterns and str(patterns).strip():
            st.markdown(f"**🎯 Patterns:** {patterns}")
            
    except Exception as e:
        logger.error(f"Error rendering score breakdown: {str(e)}")

def render_stock_details(stock: pd.Series, show_fundamentals: bool):
    """Render detailed stock information"""
    try:
        detail_cols = st.columns(3)
        
        with detail_cols[0]:
            render_classification_details(stock, show_fundamentals)
        
        with detail_cols[1]:
            render_performance_details(stock)
        
        with detail_cols[2]:
            render_technical_details(stock)
            
    except Exception as e:
        logger.error(f"Error rendering stock details: {str(e)}")

def render_classification_details(stock: pd.Series, show_fundamentals: bool):
    """Render classification and fundamental details"""
    try:
        st.markdown("**📊 Classification**")
        st.text(f"Sector: {stock.get('sector', 'Unknown')}")
        st.text(f"Category: {stock.get('category', 'Unknown')}")
        
        if 'eps_tier' in stock:
            st.text(f"EPS Tier: {stock.get('eps_tier', 'Unknown')}")
        if 'pe_tier' in stock:
            st.text(f"PE Tier: {stock.get('pe_tier', 'Unknown')}")
        
        if show_fundamentals:
            st.markdown("**💰 Fundamentals**")
            
            # PE Ratio
            pe_value = stock.get('pe')
            pe_display = UIComponents.format_pe_ratio(pe_value)
            if pe_display != "N/A":
                pe_color = ("🟢" if pe_display != "Loss" and 
                          SafeMath.safe_float(pe_value, float('inf')) < 20 else "🔴")
                st.text(f"PE Ratio: {pe_color} {pe_display}")
            else:
                st.text("PE Ratio: - (N/A)")
            
            # EPS Current
            eps_current = stock.get('eps_current')
            if pd.notna(eps_current):
                eps_display = UIComponents.format_currency(eps_current)
                st.text(f"EPS: {eps_display}")
            else:
                st.text("EPS: - (N/A)")
            
            # EPS Change
            eps_change = stock.get('eps_change_pct')
            if pd.notna(eps_change):
                eps_display = UIComponents.format_percentage(eps_change)
                
                if SafeMath.safe_float(eps_change, 0) >= 100:
                    eps_emoji = "🚀"
                elif SafeMath.safe_float(eps_change, 0) >= 50:
                    eps_emoji = "🔥"
                elif SafeMath.safe_float(eps_change, 0) >= 20:
                    eps_emoji = "📈"
                elif SafeMath.safe_float(eps_change, 0) >= 0:
                    eps_emoji = "➕"
                else:
                    eps_emoji = "➖"
                
                st.text(f"EPS Growth: {eps_emoji} {eps_display}")
            else:
                st.text("EPS Growth: - (N/A)")
                
    except Exception as e:
        logger.error(f"Error rendering classification details: {str(e)}")

def render_performance_details(stock: pd.Series):
    """Render performance details"""
    try:
        st.markdown("**📈 Performance**")
        
        performance_periods = [
            ("1 Day", 'ret_1d'),
            ("7 Days", 'ret_7d'),
            ("30 Days", 'ret_30d'),
            ("3 Months", 'ret_3m'),
            ("6 Months", 'ret_6m'),
            ("1 Year", 'ret_1y')
        ]
        
        for period, col in performance_periods:
            if col in stock.index:
                value = stock.get(col)
                if pd.notna(value):
                    formatted_value = UIComponents.format_percentage(value)
                    st.text(f"{period}: {formatted_value}")
                    
    except Exception as e:
        logger.error(f"Error rendering performance details: {str(e)}")

def render_technical_details(stock: pd.Series):
    """Render technical analysis details"""
    try:
        st.markdown("**🔍 Technicals**")
        
        # Price levels
        low_52w = SafeMath.safe_float(stock.get('low_52w', 0), 0)
        high_52w = SafeMath.safe_float(stock.get('high_52w', 0), 0)
        from_high_pct = SafeMath.safe_float(stock.get('from_high_pct', 0), 0)
        
        if low_52w > 0:
            st.text(f"52W Low: {UIComponents.format_currency(low_52w)}")
        if high_52w > 0:
            st.text(f"52W High: {UIComponents.format_currency(high_52w)}")
        if abs(from_high_pct) < 1000:
            st.text(f"From High: {from_high_pct:.0f}%")
        
        # SMA analysis
        st.markdown("**📊 SMA Analysis**")
        current_price = SafeMath.safe_float(stock.get('price', 0), 0)
        
        sma_checks = [
            ('sma_20d', '20 DMA'),
            ('sma_50d', '50 DMA'),
            ('sma_200d', '200 DMA')
        ]
        
        trading_above = []
        trading_below = []
        
        for sma_col, sma_label in sma_checks:
            if sma_col in stock.index:
                sma_value = SafeMath.safe_float(stock.get(sma_col, 0), 0)
                if sma_value > 0 and current_price > 0:
                    if current_price > sma_value:
                        pct_above = ((current_price - sma_value) / sma_value) * 100
                        st.text(f"✅ {sma_label}: {UIComponents.format_currency(sma_value)} (↑ {pct_above:.1f}%)")
                        trading_above.append(sma_label)
                    else:
                        pct_below = ((sma_value - current_price) / sma_value) * 100
                        st.text(f"❌ {sma_label}: {UIComponents.format_currency(sma_value)} (↓ {pct_below:.1f}%)")
                        trading_below.append(sma_label)
        
        # Trend quality
        if 'trend_quality' in stock.index:
            tq = SafeMath.safe_float(stock.get('trend_quality', 50), 50)
            if tq >= 80:
                st.text(f"Trend: 💪 Strong ({tq:.0f})")
            elif tq >= 60:
                st.text(f"Trend: 👍 Good ({tq:.0f})")
            else:
                st.text(f"Trend: 👎 Weak ({tq:.0f})")
                
    except Exception as e:
        logger.error(f"Error rendering technical details: {str(e)}")

def render_export_tab(df: pd.DataFrame):
    """Render export tab"""
    try:
        st.markdown("### 📥 Export Data")
        
        if df.empty:
            st.warning("No data available for export.")
            return
        
        # Export template selection
        st.markdown("#### 📋 Export Templates")
        export_template = st.radio(
            "Choose export template:",
            options=[
                "Full Analysis (All Data)",
                "Day Trader Focus",
                "Swing Trader Focus", 
                "Investor Focus"
            ],
            help="Select a template based on your trading style"
        )
        
        template_map = {
            "Full Analysis (All Data)": "full",
            "Day Trader Focus": "day_trader",
            "Swing Trader Focus": "swing_trader",
            "Investor Focus": "investor"
        }
        
        selected_template = template_map[export_template]
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            render_excel_export_section(df, selected_template)
        
        with col2:
            render_csv_export_section(df)
        
        # Export statistics
        render_export_statistics(df)
        
    except Exception as e:
        logger.error(f"Error rendering export tab: {str(e)}")
        st.error("Error loading export functionality.")

def render_excel_export_section(df: pd.DataFrame, template: str):
    """Render Excel export section"""
    try:
        st.markdown("#### 📊 Excel Report")
        st.markdown(
            "Comprehensive multi-sheet report including:\n"
            "- Top 100 stocks with all scores\n"
            "- Complete stock list\n"
            "- Sector analysis\n"
            "- Pattern frequency analysis\n"
            "- Performance summaries\n"
        )
        
        if st.button("Generate Excel Report", type="primary", use_container_width=True):
            if len(df) == 0:
                st.error("No data to export. Please adjust your filters.")
            else:
                with st.spinner("Creating Excel report..."):
                    try:
                        excel_file = ProductionExportEngine.create_excel_report(df, template=template)
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"wave_detection_report_{timestamp}.xlsx"
                        
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_file,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("Excel report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating Excel report: {str(e)}")
                        logger.error(f"Excel export error: {str(e)}", exc_info=True)
                        
    except Exception as e:
        logger.error(f"Error in Excel export section: {str(e)}")

def render_csv_export_section(df: pd.DataFrame):
    """Render CSV export section"""
    try:
        st.markdown("#### 📄 CSV Export")
        st.markdown(
            "Enhanced CSV format with:\n"
            "- All ranking scores\n"
            "- Price and return data\n"
            "- Pattern detections\n"
            "- Category classifications\n"
            "- Technical indicators\n"
        )
        
        if st.button("Generate CSV Export", use_container_width=True):
            if len(df) == 0:
                st.error("No data to export. Please adjust your filters.")
            else:
                try:
                    csv_data = ProductionExportEngine.create_csv_export(df)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"wave_detection_data_{timestamp}.csv"
                    
                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                    
                    st.success("CSV export generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating CSV: {str(e)}")
                    logger.error(f"CSV export error: {str(e)}", exc_info=True)
                    
    except Exception as e:
        logger.error(f"Error in CSV export section: {str(e)}")

def render_export_statistics(df: pd.DataFrame):
    """Render export statistics preview"""
    try:
        st.markdown("---")
        st.markdown("#### 📊 Export Preview")
        
        export_stats = {
            "Total Stocks": len(df),
            "Average Score": f"{df['master_score'].mean():.1f}" if 'master_score' in df.columns and not df.empty else "N/A",
            "Stocks with Patterns": (df['patterns'] != '').sum() if 'patterns' in df.columns else 0,
            "High RVOL (>2x)": (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0,
            "Positive 30D Returns": (df['ret_30d'] > 0).sum() if 'ret_30d' in df.columns else 0,
            "Data Quality": f"{(1 - df['master_score'].isna().sum() / len(df)) * 100:.1f}%" if 'master_score' in df.columns and not df.empty else "N/A"
        }
        
        stat_cols = st.columns(3)
        for i, (label, value) in enumerate(export_stats.items()):
            with stat_cols[i % 3]:
                st.metric(label, value)
                
    except Exception as e:
        logger.error(f"Error rendering export statistics: {str(e)}")

def render_about_tab():
    """Render about tab"""
    try:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 Production-Ready Stock Analysis System
            
            Wave Detection Ultimate 3.0 is a bulletproof, production-grade stock ranking system 
            engineered for zero-downtime performance and institutional-quality analytics.
            
            #### 🎯 Core Engine Features
            
            **Master Score 3.0** - Proprietary ranking algorithm:
            - **Position Analysis (30%)** - 52-week range positioning with smart math
            - **Volume Dynamics (25%)** - Multi-timeframe volume pattern analysis  
            - **Momentum Tracking (15%)** - Advanced momentum calculations
            - **Acceleration Detection (10%)** - Momentum acceleration algorithms
            - **Breakout Probability (10%)** - Technical breakout prediction
            - **RVOL Integration (10%)** - Real-time relative volume analysis
            
            **Production Architecture:**
            - **Zero-bug tolerance** with comprehensive error handling
            - **Smart caching** for sub-2-second response times
            - **Mobile-first** responsive design
            - **Bulletproof data pipeline** with automatic validation
            - **Memory optimized** for cloud deployment
            
            **Advanced Features:**
            - **Smart interconnected filtering** with dynamic updates
            - **Pattern recognition engine** with 15+ detection algorithms
            - **Real-time Google Sheets integration** with fallback mechanisms
            - **Professional export capabilities** (Excel multi-sheet, CSV)
            - **Comprehensive search** with fuzzy matching
            
            #### 💡 How to Use Effectively
            
            1. **Quick Actions** - Use top buttons for instant market insights
            2. **Smart Filters** - Combine multiple filters for precision screening
            3. **Rankings** - Focus on top-ranked stocks with high scores
            4. **Analysis** - Understand market sectors and patterns
            5. **Search** - Deep dive into specific stocks
            6. **Export** - Download data for external analysis
            
            #### 🔧 Pro Tips for Best Results
            
            - **Hybrid Mode** shows both technical and fundamental data
            - **Pattern combinations** indicate strongest opportunities  
            - **Score >80** typically indicates high-quality setups
            - **Multiple filters** help narrow down to best opportunities
            - **Regular exports** enable tracking historical performance
            
            #### 📊 Pattern Recognition Guide
            
            - 🔥 **CAT LEADER** - Top 10% performer in category
            - 💎 **HIDDEN GEM** - Strong in category, undervalued overall  
            - 🚀 **ACCELERATING** - Momentum building rapidly
            - ⚡ **VOL EXPLOSION** - Extreme unusual volume activity
            - 🎯 **BREAKOUT** - Technical breakout setup ready
            - 👑 **MARKET LEADER** - Top 5% overall performer
            - 🌊 **MOMENTUM WAVE** - Sustained momentum with acceleration
            - 💰 **LIQUID LEADER** - High liquidity with strong performance
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Technical Indicators
            
            **Trend Strength:**
            - 🔥 Strong (80-100)
            - ✅ Good (60-79) 
            - ➡️ Neutral (40-59)
            - ⚠️ Weak (0-39)
            
            **Score Quality:**
            - **90-100:** Exceptional
            - **80-89:** Excellent
            - **70-79:** Very Good
            - **60-69:** Good
            - **50-59:** Average
            - **<50:** Below Average
            
            #### ⚡ Performance Stats
            
            - **Real-time processing** with 1-hour intelligent caching
            - **1,790+ stocks** with 41 data points each
            - **Sub-second filtering** on any combination
            - **Mobile optimized** for all devices
            - **Zero-downtime** cloud architecture
            
            #### 🔒 Data & Security
            
            - **Live Google Sheets** integration
            - **Automatic data validation** and cleaning
            - **Error recovery** with cached fallbacks
            - **Memory efficient** processing
            - **Production logging** for monitoring
            
            #### 💬 System Status
            
            **Version:** 3.0.6-PRODUCTION-BULLETPROOF
            **Status:** Production Ready
            **Last Updated:** December 2024
            **Architecture:** 4-file modular design
            **Deployment:** Streamlit Community Cloud optimized
            
            ---
            
            **🏗️ Built with Smart Engineering:**
            - Bulletproof error handling
            - Production-grade caching
            - Mobile-first responsive design  
            - Zero-tolerance bug policy
            - Institutional-quality algorithms
            """)
        
        # System statistics
        render_system_statistics()
        
    except Exception as e:
        logger.error(f"Error rendering about tab: {str(e)}")
        st.error("Error loading about information")

def render_system_statistics():
    """Render current system statistics"""
    try:
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            total_stocks = len(st.session_state.get('ranked_df', pd.DataFrame()))
            st.metric("Total Stocks Loaded", f"{total_stocks:,}")
        
        with stats_cols[1]:
            # Get current filtered count from session
            filtered_count = st.session_state.get('current_filtered_count', 0)
            st.metric("Currently Filtered", f"{filtered_count:,}")
        
        with stats_cols[2]:
            data_quality = st.session_state.get('data_quality', {})
            completeness = data_quality.get('completeness', 0)
            st.metric("Data Quality", f"{completeness:.1f}%")
        
        with stats_cols[3]:
            last_refresh = st.session_state.get('last_refresh', datetime.now())
            cache_age = datetime.now() - last_refresh
            minutes = int(cache_age.total_seconds() / 60)
            cache_status = "Fresh" if minutes < 60 else "Refresh recommended"
            st.metric("Cache Age", f"{minutes} min", cache_status)
        
        # Performance metrics if available
        if 'performance_metrics' in st.session_state:
            with st.expander("⚡ Performance Metrics"):
                perf_metrics = st.session_state.performance_metrics
                if perf_metrics:
                    total_time = sum(perf_metrics.values())
                    st.metric("Total Processing Time", f"{total_time:.2f}s")
                    
                    # Show top 3 slowest operations
                    sorted_metrics = sorted(perf_metrics.items(), key=lambda x: x[1], reverse=True)
                    for func_name, elapsed in sorted_metrics[:3]:
                        st.caption(f"{func_name}: {elapsed:.2f}s")
                        
    except Exception as e:
        logger.error(f"Error rendering system statistics: {str(e)}")

def render_footer():
    """Render application footer"""
    try:
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; padding: 1rem;">
                <strong>Wave Detection Ultimate 3.0</strong> | Production Edition with Zero-Bug Tolerance<br>
                <small>🚀 Real-time momentum detection • 🎯 Bulletproof analytics • 💎 Smart filtering • ⚡ Lightning fast</small><br>
                <small>Built with Smart Engineering • Production-Ready Architecture • Institutional Quality</small>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        logger.error(f"Error rendering footer: {str(e)}")

# ============================================
# APPLICATION ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Critical application failure: {str(e)}")
        st.error("💥 Critical Error: Application failed to start")
        st.code(f"Error: {str(e)}")
        st.info("Please refresh the page or contact support")
