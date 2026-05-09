"""
FuelPriceLoader — reads pre-generated CSV from create_fuel_prices.py.

Run once before prepare_full_dataset.py:
    python create_fuel_prices.py

This writes data/external/fuel_prices_20240101_20260508.csv with realistic
historical TTF gas, EUA carbon, Brent oil prices and derived marginal costs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


class FuelPriceLoader:
    """
    Loads fuel and carbon prices from the pre-generated CSV.

    Columns returned:
      timestamp                   UTC, hourly
      gas_ttf_eur_mwh             TTF Natural Gas (EUR/MWh)
      coal_api2_usd_ton           Coal API2 (USD/ton)
      co2_eua_eur_ton             EUA Carbon (EUR/ton)
      oil_brent_usd_barrel        Brent Crude (USD/Bbl)
      eur_usd_rate                EUR/USD exchange rate
      gas_marginal_cost_eur_mwh   Gas-CCGT incl. CO2
      coal_marginal_cost_eur_mwh  Coal incl. CO2
      gas_coal_spread_eur_mwh     Merit-order indicator
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/external")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_fuel_prices(
        self,
        start_date: Union[str, "pd.Timestamp"],
        end_date: Union[str, "pd.Timestamp"],
        use_cache: bool = True,
        resample_to_hourly: bool = True,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date).strftime("%Y%m%d")
        end   = pd.Timestamp(end_date).strftime("%Y%m%d")
        cache_file = self.cache_dir / f"fuel_prices_{start}_{end}.csv"

        if cache_file.exists():
            df = pd.read_csv(cache_file, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if resample_to_hourly and df["timestamp"].diff().dropna().median() > pd.Timedelta("1h"):
                df = self._resample_to_hourly(df)
            return df

        raise FileNotFoundError(
            f"Fuel price cache not found: {cache_file}\n"
            "Run first:  python create_fuel_prices.py"
        )

    def load_fuel_prices_from_csv(
        self,
        csv_path: Union[str, Path],
        date_column: str = "timestamp",
        resample_to_hourly: bool = True,
    ) -> pd.DataFrame:
        df = pd.read_csv(csv_path, parse_dates=[date_column])
        df = df.rename(columns={date_column: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if resample_to_hourly:
            df = self._resample_to_hourly(df)
        return df

    def _resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.set_index("timestamp")
            .resample("h")
            .ffill()
            .reset_index()
        )
