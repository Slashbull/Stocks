"""
Wave Detection Ultimate 3.0 - Main Application
==============================================
Clean, modular main application orchestrating all components.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Import our modular components
from config import CONFIG, UI_CONFIG, logger
from data_processor import cached_load_and_process_data
from ranking_engine import RankingEngine
from ui_components import (
    UIStyler, SidebarManager, MetricsDisplay, DataTableRenderer,
    QuickActionsBar, FilterEngine, ChartRenderer
)
from utils import SessionManager, calculate_data_quality

warnings.filterwarnings('ignore')

class WaveDetectionApp:
    """Main application class orchestrating all components"""
    
    def __init__(self):
        self.ranking_engine = RankingEngine()
        self.session_manager = SessionManager()
        
    def run(self):
        """Main application entry point"""
        # Configure Streamlit page
        st.set_page_config(
            page_title="Wave Detection Ultimate 3.0",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply styling and create header
        UIStyler.apply_custom_css()
        UIStyler.create_header()
        
        # Initialize session state
        self.session_manager.initialize_session_defaults()
        
        try:
            # Load and process data
            ranked_df = self._load_data()
            
            # Render sidebar and get filters
            filters = SidebarManager.render_sidebar(ranked_df)
            
            # Handle quick actions
            quick_action = QuickActionsBar.render_quick_actions()
            
            # Apply filters
            filtered_df = self._apply_filters(ranked_df, filters, quick_action)
            
            # Render summary metrics
            MetricsDisplay.render_summary_metrics(filtered_df, ranked_df)
            
            # Main content tabs
            self._render_main_tabs(filtered_df, ranked_df)
            
            # Footer
            self._render_footer()
            
        except Exception as e:
            st.error(f"❌ Application Error: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                st.info("""
                Common solutions:
                - Check internet connectivity
                - Refresh the page
                - Clear cache and retry
                """)
    
    def _load_data(self) -> pd.DataFrame:
        """Load and process data with caching"""
        # Check if we should use cached data
        cache_age = (datetime.now() - st.session_state.last_refresh).total_seconds()
        use_cache = cache_age < CONFIG.CACHE_TTL
        
        if use_cache and 'ranked_df' in st.session_state:
            st.caption(f"Using cached data from {st.session_state.last_refresh.strftime('%I:%M %p')}")
            return st.session_state.ranked_df
        
        # Load fresh data
        with st.spinner("📥 Loading and processing stock data..."):
            # Use CSV file if available, otherwise Google Sheets
            try:
                # Try to load from local CSV first
                processed_df, timestamp = cached_load_and_process_data(
                    CONFIG.DEFAULT_SHEET_URL, CONFIG.DEFAULT_GID
                )
            except Exception as e:
                # Fallback to local CSV if Google Sheets fails
                logger.warning(f"Google Sheets failed, trying local CSV: {str(e)}")
                from data_processor import load_and_process_data
                processed_df, timestamp = load_and_process_data(csv_file='Stocks.csv')
            
            # Calculate rankings
            ranked_df = self.ranking_engine.calculate_rankings(processed_df)
            
            # Store in session
            st.session_state.ranked_df = ranked_df
            st.session_state.data_timestamp = timestamp
            st.session_state.last_refresh = datetime.now()
            
            # Calculate data quality
            st.session_state.data_quality = calculate_data_quality(ranked_df)
        
        st.success(f"✅ Loaded {len(ranked_df):,} stocks successfully!")
        return ranked_df
    
    def _apply_filters(self, df: pd.DataFrame, filters: dict, quick_action: str = None) -> pd.DataFrame:
        """Apply all filters and quick actions"""
        # Apply quick action first if selected
        if quick_action:
            filtered_df = FilterEngine.apply_quick_action(df, quick_action)
            st.info(f"Quick filter applied: {quick_action.replace('_', ' ').title()}")
        else:
            filtered_df = df
        
        # Apply sidebar filters
        filtered_df = FilterEngine.apply_filters(filtered_df, filters)
        
        return filtered_df
    
    def _render_main_tabs(self, filtered_df: pd.DataFrame, ranked_df: pd.DataFrame):
        """Render the main application tabs"""
        tabs = st.tabs(UI_CONFIG.TAB_NAMES)
        
        # Tab 1: Rankings
        with tabs[0]:
            self._render_rankings_tab(filtered_df)
        
        # Tab 2: Wave Radar
        with tabs[1]:
            self._render_wave_radar_tab(filtered_df)
        
        # Tab 3: Analysis
        with tabs[2]:
            self._render_analysis_tab(filtered_df)
        
        # Tab 4: Search
        with tabs[3]:
            self._render_search_tab(ranked_df)
        
        # Tab 5: Export
        with tabs[4]:
            self._render_export_tab(filtered_df)
        
        # Tab 6: About
        with tabs[5]:
            self._render_about_tab()
    
    def _render_rankings_tab(self, df: pd.DataFrame):
        """Render the rankings tab"""
        st.markdown("### 🏆 Top Ranked Stocks")
        
        # Display options
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(
                    st.session_state.user_preferences.get('default_top_n', CONFIG.DEFAULT_TOP_N)
                )
            )
        
        with col2:
            display_mode = st.selectbox(
                "Display Mode",
                options=["Technical", "Hybrid"],
                index=0 if st.session_state.user_preferences.get('display_mode') == 'Technical' else 1
            )
        
        # Update preferences
        st.session_state.user_preferences['default_top_n'] = display_count
        st.session_state.user_preferences['display_mode'] = display_mode
        
        # Render table
        show_fundamentals = display_mode == "Hybrid"
        DataTableRenderer.render_rankings_table(df, display_count, show_fundamentals)
        
        # Show insights
        if not df.empty:
            self._render_rankings_insights(df.head(display_count))
    
    def _render_rankings_insights(self, df: pd.DataFrame):
        """Render insights from rankings"""
        st.markdown("#### 🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top patterns
            if 'patterns' in df.columns:
                all_patterns = []
                for patterns in df['patterns'].dropna():
                    if patterns:
                        all_patterns.extend(patterns.split(' | '))
                
                if all_patterns:
                    from collections import Counter
                    pattern_counts = Counter(all_patterns)
                    
                    st.markdown("**🎯 Most Common Patterns:**")
                    for pattern, count in pattern_counts.most_common(5):
                        st.write(f"• {pattern}: {count} stocks")
        
        with col2:
            # Sector performance
            if 'sector' in df.columns and 'master_score' in df.columns:
                sector_performance = df.groupby('sector')['master_score'].mean().sort_values(ascending=False)
                
                st.markdown("**📈 Top Performing Sectors:**")
                for sector, avg_score in sector_performance.head(5).items():
                    st.write(f"• {sector[:30]}: {avg_score:.1f}")
    
    def _render_wave_radar_tab(self, df: pd.DataFrame):
        """Render the Wave Radar tab"""
        st.markdown("### 🌊 Wave Radar™ - Early Detection System")
        
        if df.empty:
            st.warning("No data available for Wave Radar analysis.")
            return
        
        # Market sentiment indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Market Momentum")
            
            # Calculate momentum indicators
            high_momentum = (df.get('momentum_score', 0) >= 80).sum()
            total_stocks = len(df)
            momentum_pct = (high_momentum / total_stocks * 100) if total_stocks > 0 else 0
            
            st.metric("High Momentum Stocks", high_momentum, f"{momentum_pct:.1f}% of filtered")
            
            # Volume surge detection
            volume_surge = (df.get('rvol', 0) >= 3).sum()
            st.metric("Volume Surge", volume_surge, "RVOL > 3x")
            
            # Breakout candidates
            breakout_ready = (df.get('breakout_score', 0) >= 80).sum()
            st.metric("Breakout Ready", breakout_ready, "Score > 80")
        
        with col2:
            st.markdown("#### 📊 Pattern Emergence")
            
            # Emerging patterns analysis
            if 'patterns' in df.columns:
                pattern_stocks = df[df['patterns'] != ''].copy()
                
                if not pattern_stocks.empty:
                    st.metric("Stocks with Patterns", len(pattern_stocks))
                    
                    # Top emerging patterns
                    all_patterns = []
                    for patterns in pattern_stocks['patterns']:
                        if patterns:
                            all_patterns.extend(patterns.split(' | '))
                    
                    if all_patterns:
                        from collections import Counter
                        top_patterns = Counter(all_patterns).most_common(3)
                        
                        st.markdown("**🔥 Hot Patterns:**")
                        for pattern, count in top_patterns:
                            st.write(f"• {pattern}: {count}")
                else:
                    st.info("No pattern detections in current filter")
        
        # Radar chart for market segments
        if len(df) > 10:  # Only if sufficient data
            self._render_market_radar(df)
    
    def _render_market_radar(self, df: pd.DataFrame):
        """Render market radar visualization"""
        st.markdown("#### 🎯 Market Segment Radar")
        
        # Analyze by category
        if 'category' in df.columns and 'master_score' in df.columns:
            category_performance = df.groupby('category').agg({
                'master_score': 'mean',
                'momentum_score': 'mean',
                'volume_score': 'mean',
                'rvol': 'mean'
            }).round(1)
            
            # Create radar chart data
            import plotly.graph_objects as go
            
            categories = category_performance.index.tolist()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=category_performance['master_score'].tolist(),
                theta=categories,
                fill='toself',
                name='Master Score',
                line_color='blue'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=category_performance['momentum_score'].tolist(),
                theta=categories,
                fill='toself',
                name='Momentum',
                line_color='red',
                opacity=0.6
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Market Segment Performance Radar",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_analysis_tab(self, df: pd.DataFrame):
        """Render the analysis tab"""
        st.markdown("### 📊 Market Analysis")
        
        if df.empty:
            st.warning("No data available for analysis.")
            return
        
        # Render charts
        col1, col2 = st.columns(2)
        
        with col1:
            ChartRenderer.render_sector_distribution(df)
            ChartRenderer.render_score_distribution(df)
        
        with col2:
            ChartRenderer.render_category_performance(df)
            
            # Additional analysis metrics
            st.markdown("#### 📈 Market Statistics")
            
            if 'master_score' in df.columns:
                scores = df['master_score'].dropna()
                if not scores.empty:
                    st.write(f"**Score Statistics:**")
                    st.write(f"• Mean: {scores.mean():.1f}")
                    st.write(f"• Median: {scores.median():.1f}")
                    st.write(f"• Std Dev: {scores.std():.1f}")
                    st.write(f"• Range: {scores.min():.1f} - {scores.max():.1f}")
    
    def _render_search_tab(self, df: pd.DataFrame):
        """Render the search tab"""
        st.markdown("### 🔍 Stock Search & Analysis")
        
        # Search interface
        search_term = st.text_input(
            "Search stocks by ticker or company name:",
            placeholder="e.g. RELIANCE, TCS, ACC",
            help="Enter ticker symbol or company name"
        )
        
        if search_term:
            # Search logic
            search_results = self._search_stocks(df, search_term)
            
            if not search_results.empty:
                st.markdown(f"#### 📊 Search Results for '{search_term}'")
                
                # Display detailed results
                for _, stock in search_results.iterrows():
                    with st.expander(f"📈 {stock['ticker']} - {stock.get('company_name', 'N/A')}"):
                        self._render_stock_details(stock)
            else:
                st.warning(f"No stocks found matching '{search_term}'")
        else:
            # Show search suggestions
            st.markdown("#### 💡 Search Suggestions")
            
            if not df.empty:
                # Top performers
                if 'master_score' in df.columns:
                    top_performers = df.nlargest(5, 'master_score')['ticker'].tolist()
                    st.write(f"**Top Performers:** {', '.join(top_performers)}")
                
                # High volume
                if 'rvol' in df.columns:
                    high_volume = df.nlargest(5, 'rvol')['ticker'].tolist()
                    st.write(f"**High Volume:** {', '.join(high_volume)}")
    
    def _search_stocks(self, df: pd.DataFrame, search_term: str) -> pd.DataFrame:
        """Search stocks by ticker or company name"""
        if df.empty:
            return pd.DataFrame()
        
        search_term = search_term.upper().strip()
        
        # Search in ticker
        ticker_match = df[df['ticker'].str.contains(search_term, case=False, na=False)]
        
        # Search in company name
        if 'company_name' in df.columns:
            name_match = df[df['company_name'].str.contains(search_term, case=False, na=False)]
            results = pd.concat([ticker_match, name_match]).drop_duplicates()
        else:
            results = ticker_match
        
        return results.sort_values('master_score', ascending=False) if 'master_score' in results.columns else results
    
    def _render_stock_details(self, stock: pd.Series):
        """Render detailed stock information"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Scores:**")
            if 'master_score' in stock:
                st.write(f"Master Score: {stock['master_score']:.1f}")
            if 'momentum_score' in stock:
                st.write(f"Momentum: {stock['momentum_score']:.1f}")
            if 'volume_score' in stock:
                st.write(f"Volume: {stock['volume_score']:.1f}")
        
        with col2:
            st.write("**💰 Fundamentals:**")
            if 'pe' in stock and pd.notna(stock['pe']):
                st.write(f"PE Ratio: {stock['pe']:.1f}x")
            if 'eps_change_pct' in stock and pd.notna(stock['eps_change_pct']):
                st.write(f"EPS Growth: {stock['eps_change_pct']:.1f}%")
        
        with col3:
            st.write("**🎯 Patterns:**")
            if 'patterns' in stock and stock['patterns']:
                patterns = stock['patterns'].split(' | ')
                for pattern in patterns[:3]:  # Show top 3 patterns
                    st.write(f"• {pattern}")
    
    def _render_export_tab(self, df: pd.DataFrame):
        """Render the export tab"""
        st.markdown("### 📥 Export Data")
        
        if df.empty:
            st.warning("No data to export.")
            return
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Export Options")
            
            export_format = st.selectbox(
                "Select format:",
                ["CSV", "Excel", "JSON"]
            )
            
            include_all_columns = st.checkbox(
                "Include all columns",
                value=False,
                help="Include all technical columns, not just display columns"
            )
            
            max_rows = st.number_input(
                "Maximum rows:",
                min_value=10,
                max_value=len(df),
                value=min(100, len(df)),
                step=10
            )
        
        with col2:
            st.markdown("#### 📈 Export Preview")
            
            # Prepare export data
            if include_all_columns:
                export_df = df.head(max_rows)
            else:
                display_cols = [col for col in UI_CONFIG.TECHNICAL_COLUMNS if col in df.columns]
                export_df = df.head(max_rows)[display_cols]
            
            st.write(f"Will export {len(export_df)} rows × {len(export_df.columns)} columns")
            
            # Download button
            if export_format == "CSV":
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                # Create Excel file
                from io import BytesIO
                buffer = BytesIO()
                export_df.to_excel(buffer, index=False, engine='xlsxwriter')
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    def _render_about_tab(self):
        """Render the about tab"""
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 About This System
            
            Wave Detection Ultimate 3.0 is a professional-grade stock ranking system designed to identify 
            momentum opportunities in the Indian stock market. The system analyzes 1,791+ stocks across 
            41 data points to provide comprehensive rankings and pattern detection.
            
            #### 🎯 Master Score 3.0 Components
            
            - **Position Analysis (30%)** - 52-week range positioning analysis
            - **Volume Dynamics (25%)** - Multi-timeframe volume pattern analysis  
            - **Momentum Tracking (15%)** - Price momentum across multiple periods
            - **Acceleration Detection (10%)** - Momentum acceleration signals
            - **Breakout Probability (10%)** - Technical breakout readiness
            - **RVOL Integration (10%)** - Relative volume analysis
            
            #### 🔍 Pattern Detection
            
            The system detects 16+ patterns including technical patterns (CAT LEADER, HIDDEN GEM, 
            ACCELERATING) and fundamental patterns (VALUE MOMENTUM, EARNINGS ROCKET, QUALITY LEADER).
            
            #### 📊 Data Coverage
            
            - **1,791 stocks** across all market caps
            - **41 data points** per stock including price, volume, fundamentals
            - **Daily refresh** with latest market data
            - **99.9% data completeness** on technical indicators
            
            #### 🚀 Key Features
            
            - Real-time ranking calculations
            - Smart interconnected filtering
            - Wave Radar™ early detection system
            - Advanced pattern recognition
            - Professional-grade analytics
            - Export capabilities for further analysis
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Performance Indicators
            
            - 🔥 **Excellent (80-100)** - Strong buy signals
            - ✅ **Good (60-79)** - Positive momentum  
            - ➡️ **Neutral (40-59)** - Mixed signals
            - ⚠️ **Weak (0-39)** - Caution advised
            
            #### 🎨 System Architecture
            
            **Modular Design:**
            - Configuration management
            - Data processing pipeline
            - Ranking engine
            - UI components
            - Utilities & helpers
            
            **Performance Optimized:**
            - Vectorized calculations
            - Smart caching
            - Memory efficient
            - Sub-second response times
            
            #### 📝 Version History
            
            **v3.0.4 (Current)**
            - Refactored modular architecture
            - Enhanced performance
            - Better error handling
            - Improved UI/UX
            
            **v2.x (Legacy)**
            - Original monolithic design
            - Basic pattern detection
            - Limited scalability
            
            #### 💬 Support
            
            For optimal performance:
            - Use filters to narrow results
            - Clear cache if data seems stale  
            - Export data for offline analysis
            - Monitor data quality indicators
            """)
        
        # System statistics
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            if 'ranked_df' in st.session_state:
                st.metric("Stocks Loaded", f"{len(st.session_state.ranked_df):,}")
            else:
                st.metric("Stocks Loaded", "0")
        
        with stats_cols[1]:
            if 'data_quality' in st.session_state:
                completeness = st.session_state.data_quality.get('completeness', 0) * 100
                st.metric("Data Quality", f"{completeness:.1f}%")
            else:
                st.metric("Data Quality", "N/A")
        
        with stats_cols[2]:
            cache_age = (datetime.now() - st.session_state.last_refresh).total_seconds() / 60
            st.metric("Cache Age", f"{cache_age:.0f} min")
        
        with stats_cols[3]:
            if 'performance_metrics' in st.session_state:
                total_time = sum(st.session_state.performance_metrics.values())
                st.metric("Load Time", f"{total_time:.2f}s")
            else:
                st.metric("Load Time", "N/A")
    
    def _render_footer(self):
        """Render application footer"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            Wave Detection Ultimate 3.0 | Professional Edition with Wave Radar™<br>
            <small>Real-time momentum detection • Early entry signals • Smart money flow tracking</small>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Application entry point"""
    app = WaveDetectionApp()
    app.run()

if __name__ == "__main__":
    main()