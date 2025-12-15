#!/usr/bin/env python
"""
BuildingFM Unified Configuration System

Single source of truth for all scripts. Change ACTIVE_SCHEMA to switch everything.

Usage:
    from buildingfm_config import cfg

    # Access derived values
    cfg.weather_var_ids      # Auto-derived from schema
    cfg.forecast_target_ids  # Auto-derived from schema
    cfg.data_dir             # Auto-generated path
    cfg.patch_size           # User-configured value
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pandas as pd

# =============================================================================
# USER CONFIGURATION - Edit this section to configure your experiments
# =============================================================================

# Schema Selection: 'full' or 'lite'
# - full: 112 variables (hvac_schema.yaml)
# - lite: 24 variables (hvac_schema_lite.yaml)
ACTIVE_SCHEMA = 'lite'

# Patch Size: Independent configuration
# - For lite (24 vars): Can use 16, 32, or 64
# - For full (112 vars): Must use 128 due to MOIRAI's 512 token limit
PATCH_SIZE = 128

# Data Frequency
DATA_FREQ = '15min'

# =============================================================================
# SCHEMA DEFINITIONS - Variable IDs for each schema
# =============================================================================

SCHEMA_CONFIGS = {
    'full': {
        'schema_file': 'hvac_schema.yaml',
        'num_variates': 112,
        'data_suffix': '_15min',
        'variable_groups': {
            'weather': list(range(0, 8)),           # IDs 0-7
            'power_main': [10, 11],
            'power_odu': [12, 13],
            'power_circuits': list(range(14, 28)),  # IDs 14-27
            'power_idu': list(range(30, 34)),       # IDs 30-33
            'idu_supply': list(range(34, 42)),      # IDs 34-41
            'idu_coil': list(range(42, 47)),        # IDs 42-46
            'idu_control': [47],
            'zone_temps': list(range(50, 98)),      # IDs 50-97
            'iaq': list(range(98, 102)),            # IDs 98-101
            'solar_flux': list(range(104, 120)),    # IDs 104-119
        },
        'forecast_targets': [10, 12, 30, 50],  # Main power, ODU, IDU, Zone A
        'fdd_targets': [12, 13],               # ODU power for FDD
    },
    'lite': {
        'schema_file': 'hvac_schema_lite.yaml',
        'num_variates': 24,
        'data_suffix': '_lite_15min',
        'variable_groups': {
            'weather': [0, 1],                      # Temp, RH
            'power_odu': [2],                       # ODU Power
            'power_idu': [3],                       # IDU Power
            'system': list(range(4, 8)),            # IDU system vars
            'zones': list(range(8, 16)),            # Zone temps A-H
            'solar': list(range(16, 24)),           # Solar flux A-H
        },
        'forecast_targets': [2, 8, 9],    # ODU Power, Zone A, Zone B
        'fdd_targets': [2],               # ODU power for FDD
    },
}

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Base paths - suffixes auto-appended based on schema
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_BASE = PROJECT_ROOT / 'data'
OUTPUT_BASE = PROJECT_ROOT / 'outputs'


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class BuildingFMConfig:
    """
    Central configuration object with all derived values.

    All values are derived from ACTIVE_SCHEMA selection.
    """
    schema_name: str
    schema_path: Path
    num_variates: int
    patch_size: int
    data_freq: str

    # Paths
    data_dir: Path
    output_dir: Path
    evaluation_output_dir: Path

    # Variable ID groups (derived from schema)
    weather_var_ids: List[int]
    odu_power_var_ids: List[int]
    idu_power_var_ids: List[int]
    forecast_target_ids: List[int]
    fdd_target_ids: List[int]

    # Full variable groups dict
    variable_groups: Dict[str, List[int]] = field(default_factory=dict)

    # Training parameters
    context_length_days: int = 2
    prediction_length_days: int = 1

    @property
    def steps_per_hour(self) -> int:
        """Time steps per hour based on data frequency"""
        freq_minutes = pd.Timedelta(self.data_freq).total_seconds() / 60
        return int(60 / freq_minutes)

    @property
    def steps_per_day(self) -> int:
        """Time steps per day"""
        return self.steps_per_hour * 24

    @property
    def context_length(self) -> int:
        """Context length in time steps"""
        return self.context_length_days * self.steps_per_day

    @property
    def prediction_length(self) -> int:
        """Prediction length in time steps"""
        return self.prediction_length_days * self.steps_per_day

    @property
    def seasonal_period(self) -> int:
        """Seasonal period for metrics (1 day)"""
        return self.steps_per_day

    def get_train_weather_ids(self) -> Tuple[int, ...]:
        """Return weather IDs as tuple for CausalChainMaskedPrediction"""
        return tuple(self.weather_var_ids)

    def get_train_odu_ids(self) -> Tuple[int, ...]:
        """Return ODU IDs as tuple for CausalChainMaskedPrediction"""
        return tuple(self.odu_power_var_ids)

    def get_zone_temp_ids(self) -> List[int]:
        """Return zone temperature variable IDs"""
        return self.variable_groups.get('zones', self.variable_groups.get('zone_temps', []))


def build_config() -> BuildingFMConfig:
    """
    Build configuration from ACTIVE_SCHEMA selection.

    This is the main entry point for getting configuration.
    """
    if ACTIVE_SCHEMA not in SCHEMA_CONFIGS:
        raise ValueError(f"Unknown schema: {ACTIVE_SCHEMA}. Choose from: {list(SCHEMA_CONFIGS.keys())}")

    schema_cfg = SCHEMA_CONFIGS[ACTIVE_SCHEMA]

    # Build paths
    data_suffix = schema_cfg['data_suffix']
    schema_path = CONFIG_DIR / schema_cfg['schema_file']
    data_dir = DATA_BASE / f'buildingfm_processed{data_suffix}'
    output_dir = OUTPUT_BASE / f'buildingfm{data_suffix}'
    eval_output_dir = OUTPUT_BASE / f'evaluation{data_suffix}'

    # Extract variable groups
    var_groups = schema_cfg['variable_groups']

    return BuildingFMConfig(
        schema_name=ACTIVE_SCHEMA,
        schema_path=schema_path,
        num_variates=schema_cfg['num_variates'],
        patch_size=PATCH_SIZE,
        data_freq=DATA_FREQ,

        # Paths
        data_dir=data_dir,
        output_dir=output_dir,
        evaluation_output_dir=eval_output_dir,

        # Variable IDs
        weather_var_ids=var_groups.get('weather', []),
        odu_power_var_ids=var_groups.get('power_odu', []),
        idu_power_var_ids=var_groups.get('power_idu', []),
        forecast_target_ids=schema_cfg['forecast_targets'],
        fdd_target_ids=schema_cfg['fdd_targets'],

        # Full groups dict
        variable_groups=var_groups,
    )


# =============================================================================
# MODULE-LEVEL CONFIG INSTANCE
# =============================================================================

# Global config instance - import this in other scripts
cfg = build_config()


# =============================================================================
# CONVENIENCE EXPORTS FOR BACKWARD COMPATIBILITY
# =============================================================================

# These allow scripts to import specific values directly:
#   from buildingfm_config import WEATHER_VAR_IDS, DATA_DIR

WEATHER_VAR_IDS = cfg.weather_var_ids
FORECAST_TARGET_IDS = cfg.forecast_target_ids
ODU_POWER_VAR_IDS = cfg.odu_power_var_ids

DATA_DIR = cfg.data_dir
OUTPUT_DIR = cfg.output_dir

CONTEXT_LENGTH = cfg.context_length
PREDICTION_LENGTH = cfg.prediction_length
STEPS_PER_DAY = cfg.steps_per_day
STEPS_PER_HOUR = cfg.steps_per_hour
SEASONAL_PERIOD = cfg.seasonal_period


# =============================================================================
# SELF-TEST / INFO OUTPUT
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("BuildingFM Configuration")
    print("=" * 60)
    print(f"Active Schema: {cfg.schema_name}")
    print(f"Schema Path:   {cfg.schema_path}")
    print(f"Num Variates:  {cfg.num_variates}")
    print(f"Patch Size:    {cfg.patch_size}")
    print(f"Data Freq:     {cfg.data_freq}")
    print()
    print("Paths:")
    print(f"  Data Dir:    {cfg.data_dir}")
    print(f"  Output Dir:  {cfg.output_dir}")
    print(f"  Eval Dir:    {cfg.evaluation_output_dir}")
    print()
    print("Variable IDs:")
    print(f"  Weather:     {cfg.weather_var_ids}")
    print(f"  ODU Power:   {cfg.odu_power_var_ids}")
    print(f"  IDU Power:   {cfg.idu_power_var_ids}")
    zone_ids = cfg.get_zone_temp_ids()
    if len(zone_ids) > 5:
        print(f"  Zone Temps:  {zone_ids[:5]}... ({len(zone_ids)} total)")
    else:
        print(f"  Zone Temps:  {zone_ids}")
    print()
    print("Task Targets:")
    print(f"  Forecast:    {cfg.forecast_target_ids}")
    print(f"  FDD:         {cfg.fdd_target_ids}")
    print()
    print("Time Configuration:")
    print(f"  Steps/Hour:  {cfg.steps_per_hour}")
    print(f"  Steps/Day:   {cfg.steps_per_day}")
    print(f"  Context:     {cfg.context_length} steps ({cfg.context_length_days} days)")
    print(f"  Prediction:  {cfg.prediction_length} steps ({cfg.prediction_length_days} days)")
    print("=" * 60)
