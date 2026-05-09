"""
Neighbor Country Generation Loader
===================================
Downloads generation data for neighboring countries via Energy-Charts API.
No API key required — uses the same free Energy-Charts endpoint.

Key insight:
  DE-LU prices are heavily influenced by FRENCH NUCLEAR generation.
  When France has high nuclear output, it exports surplus to Germany,
  pushing German day-ahead prices down.

  ES prices depend heavily on IBERIAN HYDRO (Spain + Portugal reservoirs).

Data sources (all free, no registration):
  Energy-Charts API   https://api.energy-charts.info/public_power
  SMARD.de            https://www.smard.de  (German physical border flows)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


_BASE_EC  = "https://api.energy-charts.info"

# Map zone → neighbor countries to download + their relevant columns
_NEIGHBOR_CFG = {
    "DE-LU": {
        "fr": ["nuclear", "hydro_water_reservoir", "hydro_run_of_river_and_poundage",
               "fossil_gas", "wind_onshore", "solar"],
        # "ch": ["nuclear", "hydro_water_reservoir"],  # Switzerland nuclear too
    },
    "ES": {
        "pt": ["hydro_water_reservoir", "hydro_run_of_river_and_poundage",
               "wind_onshore", "solar"],
        "fr": ["nuclear", "hydro_water_reservoir"],
    },
}

# Energy-Charts country code → readable prefix for column names
_EC_COUNTRY_CODE = {
    "fr": "fr",
    "ch": "ch",
    "pt": "pt",
}


# ===========================================================================
class NeighborGenerationLoader:
    """
    Downloads neighboring-country generation data from Energy-Charts API
    to use as proxy features for cross-border price pressure.

    Usage
    -----
    loader = NeighborGenerationLoader(cache_dir='data/external/neighbors')
    df = loader.load_all(zone='DE-LU', start_date='2024-01-01', end_date='2026-05-06')
    """

    def __init__(self, cache_dir: str = "data/external/neighbors"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def load_all(
        self,
        zone: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download and merge all neighbor generation features.

        Returns hourly DataFrame with columns like:
          fr_nuclear_mw, fr_hydro_mw, fr_fossil_gas_mw,
          fr_wind_mw, fr_solar_mw, fr_total_mw,
          pt_hydro_mw, pt_wind_mw, ...
        """
        cfg = _NEIGHBOR_CFG.get(zone, {})
        if not cfg:
            print(f"  No neighbor config for zone {zone}")
            return pd.DataFrame()

        start_ts = _utc(start_date)
        end_ts   = _utc(end_date)
        merged   = pd.DataFrame(
            {"timestamp": pd.date_range(start_ts, end_ts + pd.Timedelta(hours=23),
                                        freq="1h", tz="UTC")}
        )

        for country, columns in cfg.items():
            try:
                print(f"  Fetching Energy-Charts: {country.upper()} generation...")
                df_c = self._load_country(
                    country, columns, start_date, end_date, use_cache
                )
                prefix_cols = {c: f"{country}_{c}" for c in df_c.columns if c != "timestamp"}
                df_c = df_c.rename(columns=prefix_cols)
                merged = merged.merge(df_c, on="timestamp", how="left")
                print(f"    OK: {list(prefix_cols.values())}")
            except Exception as exc:
                print(f"  ⚠️   {country.upper()} generation failed: {exc}")

        # Forward-fill hourly gaps (generation data sometimes has sparse timestamps)
        fill_cols = [c for c in merged.columns if c != "timestamp"]
        merged[fill_cols] = merged[fill_cols].ffill().bfill()

        print(f"  Neighbor features shape: {merged.shape}")
        return merged

    # ------------------------------------------------------------------
    def _load_country(
        self,
        country: str,
        want_cols: list[str],
        start_date: str,
        end_date: str,
        use_cache: bool,
    ) -> pd.DataFrame:
        """Download one country's generation data (hourly, MW)."""
        s = start_date.replace("-", "")[:8]
        e = end_date.replace("-", "")[:8]
        cache = self.cache_dir / f"gen_{country}_{s}_{e}.csv"

        if use_cache and cache.exists():
            df = pd.read_csv(cache, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df

        # Query Energy-Charts in annual chunks (API typically limited to ~1 year)
        start_ts = _utc(start_date)
        end_ts   = _utc(end_date)
        chunks   = []
        cur = start_ts
        while cur <= end_ts:
            chunk_end = min(cur + pd.DateOffset(months=11),
                            end_ts + pd.Timedelta(hours=23))
            raw = self._fetch_ec(country, cur, chunk_end)
            if raw is not None and not raw.empty:
                chunks.append(raw)
            cur = chunk_end + pd.Timedelta(hours=1)
            time.sleep(0.5)   # polite rate-limiting

        if not chunks:
            raise RuntimeError(f"No data returned for {country}")

        df = pd.concat(chunks).sort_values("timestamp")
        df = df.drop_duplicates("timestamp").reset_index(drop=True)

        # Keep only requested columns (+ timestamp)
        keep = ["timestamp"] + [c for c in want_cols if c in df.columns]
        df   = df[keep]

        # Resample to hourly (Energy-Charts returns 15-min or 30-min sometimes)
        df = (
            df.set_index("timestamp")
            .resample("1h")
            .mean()
            .reset_index()
        )

        df.to_csv(cache, index=False)
        return df

    # ------------------------------------------------------------------
    def _fetch_ec(
        self,
        country: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """Single Energy-Charts /public_power request."""
        params = {
            "country": country,
            "start":   start.strftime("%Y-%m-%dT%H:%MZ"),
            "end":     end.strftime("%Y-%m-%dT%H:%MZ"),
        }
        resp = requests.get(f"{_BASE_EC}/public_power", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "unix_seconds" not in data or "production_types" not in data:
            return None

        ts  = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
        out = pd.DataFrame({"timestamp": ts})

        for prod in data["production_types"]:
            name = (
                prod.get("name", "")
                .lower()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
                .strip("_")
            )
            vals = np.asarray(prod.get("data", []), dtype=float)
            if len(vals) == len(ts):
                out[name] = vals

        return out


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _utc(date_str: str) -> pd.Timestamp:
    ts = pd.Timestamp(date_str)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
