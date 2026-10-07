"""Synthetic data generators."""
from .microseismic import (
    MSConfig,
    generate_window,
    make_station_network,
    ricker_wavelet,
)

__all__ = ["MSConfig", "generate_window", "make_station_network", "ricker_wavelet"]
