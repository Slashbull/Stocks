"""
Wave Detection Ultimate 3.0 - Configuration Module
==================================================
Centralized configuration for the stock ranking system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging

@dataclass(frozen=True)
class AppConfig:
    """Application configuration with validated weights and settings"""
    
    # Data source settings
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM/edit?usp=sharing"
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    
    # Master Score 3.0 weights (must sum to 1.0)
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Pattern detection thresholds
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
        "long_strength": 80
    })
    
    # Performance thresholds
    RVOL_MAX_THRESHOLD: float = 20.0
    MIN_VALID_PRICE: float = 0.01
    MIN_DATA_COMPLETENESS: float = 0.8
    STALE_DATA_HOURS: int = 24
    
    # Quick action thresholds
    TOP_GAINER_MOMENTUM: float = 80
    VOLUME_SURGE_RVOL: float = 3.0
    BREAKOUT_READY_SCORE: float = 80
    
    # Tier definitions for categorization
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
            "Negative/NA": (-float('inf'), 0),
            "0-10": (0, 10),
            "10-15": (10, 15),
            "15-20": (15, 20),
            "20-30": (20, 30),
            "30-50": (30, 50),
            "50+": (50, float('inf'))
        },
        "price": {
            "0-100": (0, 100),
            "100-250": (100, 250),
            "250-500": (250, 500),
            "500-1000": (500, 1000),
            "1000-2500": (1000, 2500),
            "2500-5000": (2500, 5000),
            "5000+": (5000, float('inf'))
        }
    })
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate weights sum to 1.0
        total_weight = (
            self.POSITION_WEIGHT + self.VOLUME_WEIGHT + self.MOMENTUM_WEIGHT +
            self.ACCELERATION_WEIGHT + self.BREAKOUT_WEIGHT + self.RVOL_WEIGHT
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

@dataclass
class UIConfig:
    """UI-specific configuration"""
    
    # Column definitions for different display modes
    TECHNICAL_COLUMNS: List[str] = field(default_factory=lambda: [
        'rank', 'ticker', 'company_name', 'price', 'master_score',
        'position_score', 'volume_score', 'momentum_score', 'rvol', 'patterns'
    ])
    
    HYBRID_COLUMNS: List[str] = field(default_factory=lambda: [
        'rank', 'ticker', 'company_name', 'price', 'master_score',
        'pe', 'eps_change_pct', 'momentum_score', 'rvol', 'patterns'
    ])
    
    # Styling configuration
    SCORE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'excellent': '#2ecc71',  # Green for 80+
        'good': '#f39c12',       # Orange for 60-79
        'average': '#95a5a6',    # Gray for 40-59
        'poor': '#e74c3c'        # Red for <40
    })
    
    # Tab configuration
    TAB_NAMES: List[str] = field(default_factory=lambda: [
        "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", 
        "🔍 Search", "📥 Export", "ℹ️ About"
    ])

@dataclass
class DataConfig:
    """Data processing configuration"""
    
    # Required columns for core functionality
    REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price'])
    
    # Numeric columns that need cleaning
    NUMERIC_COLUMNS: List[str] = field(default_factory=lambda: [
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
    ])
    
    # Categorical columns
    CATEGORICAL_COLUMNS: List[str] = field(default_factory=lambda: [
        'ticker', 'company_name', 'category', 'sector'
    ])
    
    # Volume ratio columns for scoring
    VOLUME_RATIO_COLUMNS: Dict[str, float] = field(default_factory=lambda: {
        'vol_ratio_1d_90d': 0.20,
        'vol_ratio_7d_90d': 0.20,
        'vol_ratio_30d_90d': 0.20,
        'vol_ratio_30d_180d': 0.15,
        'vol_ratio_90d_180d': 0.25
    })

# Global configuration instances
CONFIG = AppConfig()
UI_CONFIG = UIConfig()
DATA_CONFIG = DataConfig()

# Logging configuration
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup centralized logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Create logger instance
logger = setup_logging()