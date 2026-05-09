"""
ENTSO-E Transparency Platform Data Loader
==========================================
Provides:
  - Cross-border physical flows (hourly MW)     ← most impactful for DE-LU
  - Generation by production type (hourly MW)   ← nuclear, hydro, pumped storage
  - Day-ahead wind & solar forecast (hourly MW) ← forward-looking signal

Getting your free API key (takes ~2 minutes):
  1. https://transparency.entsoe.eu/ → Register
  2. My Account Settings → Web API Security Token → Generate
  3. Windows:  set ENTSOE_API_KEY=your_token_here
     Mac/Linux: export ENTSOE_API_KEY=your_token_here
  4. OR pass api_key= directly to EntsoELoader()

Install:
  pip install entsoe-py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np

try:
    from entsoe import EntsoePandasClient
    _ENTSOE_OK = True
except ImportError:
    _ENTSOE_OK = False
    warnings.warn("entsoe-py not installed — run:  pip install entsoe-py", stacklevel=2)


# ---------------------------------------------------------------------------
# EIC area codes  (entsoe-py accepts these as country_code arguments)
# ---------------------------------------------------------------------------
_EIC = {
    "DE-LU": "10Y1001A1001A82H",   # German-Luxembourg bidding zone
    "ES":    "10YES-REE------0",    # Spain
    "FR":    "10YFR-RTE------C",    # France
    "AT":    "10YAT-APG------L",    # Austria
    "PL":    "10YPL-AREA-----S",    # Poland
    "CZ":    "10YCZ-CEPS-----N",    # Czech Republic
    "CH":    "10YCH-SWISSGRIDZ",    # Switzerland
    "PT":    "10YPT-REN------W",    # Portugal
    "BE":    "10YBE----------2",    # Belgium
    "NL":    "10YNL----------L",    # Netherlands
    "DK1":   "10YDK-1--------W",    # Denmark West
}

_NEIGHBORS = {
    "DE-LU": ["FR", "AT", "CH", "PL", "CZ"],
    "ES":    ["FR", "PT"],
}

# PSR type → human-readable column name (entsoe-py uses string key names)
_PSR_MAP = {
    "Nuclear":                            "nuclear_mw",
    "Hydro Run-of-river and poundage":    "hydro_ror_mw",
    "Hydro Water Reservoir":              "hydro_reservoir_mw",
    "Hydro Pumped Storage":               "pumped_storage_generation_mw",
    "Hydro Pumped Storage_Consumption":   "pumped_storage_consumption_mw",
    "Fossil Gas":                         "gas_mw",
    "Fossil Hard coal":                   "hard_coal_mw",
    "Fossil Brown coal/Lignite":          "lignite_mw",
    "Fossil Oil":                         "oil_mw",
    "Wind Onshore":                       "wind_onshore_entsoe_mw",
    "Wind Offshore":                      "wind_offshore_entsoe_mw",
    "Solar":                              "solar_entsoe_mw",
    "Biomass":                            "biomass_mw",
    "Other renewable":                    "other_renewable_mw",
    "Other":                              "other_mw",
}

_FORECAST_MAP = {
    "Wind Onshore":  "wind_forecast_mw",
    "Wind Offshore": "wind_offshore_forecast_mw",
    "Solar":         "solar_forecast_mw",
}


# ===========================================================================
class EntsoELoader:
    """
    Load ENTSO-E Transparency Platform data with local CSV caching.

    Usage
    -----
    loader = EntsoELoader(api_key="your_token")            # explicit key
    loader = EntsoELoader()                                 # reads ENTSOE_API_KEY env var

    df = loader.load_all_features("DE-LU", "2024-01-01", "2026-05-06")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/external/entsoe",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        api_key = api_key or os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise ValueError(
                "\n  ENTSO-E API key missing!\n"
                "  1. Register free: https://transparency.entsoe.eu/\n"
                "  2. My Account → Web API Security Token → Generate\n"
                "  3. Windows:  set ENTSOE_API_KEY=your_token\n"
                "     Mac/Linux: export ENTSOE_API_KEY=your_token\n"
                "  OR: EntsoELoader(api_key='your_token')"
            )
        if not _ENTSOE_OK:
            raise ImportError("pip install entsoe-py")

        self.client = EntsoePandasClient(api_key=api_key)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def load_all_features(
        self,
        zone: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download everything and return one merged hourly DataFrame.

        Columns (depending on availability):
          net_import_<neighbor>_mw, net_import_total_mw
          nuclear_mw, hydro_total_mw, pumped_storage_net_mw
          gas_mw, hard_coal_mw, lignite_mw, fossil_mix_mw
          wind_forecast_mw, solar_forecast_mw, renewable_forecast_total_mw
          load_forecast_mw
        """
        print(f"\n[ENTSO-E] Loading all features for {zone} ({start_date} → {end_date})")

        start_ts = self._utc(start_date)
        end_ts   = self._utc(end_date)
        merged   = pd.DataFrame(
            {"timestamp": pd.date_range(start_ts, end_ts + pd.Timedelta(hours=23),
                                        freq="1h", tz="UTC")}
        )

        # 1. Cross-border flows
        try:
            print("  [1/5] Cross-border flows ...")
            flows = self.load_cross_border_flows(zone, start_date, end_date, use_cache)
            merged = merged.merge(flows, on="timestamp", how="left")
            print(f"        {len(flows)} hourly rows, {flows.shape[1]-1} flow columns")
        except Exception as exc:
            print(f"  Warning: Cross-border flows failed: {exc}")

        # 2. Generation by type
        try:
            print("  [2/5] Generation by production type ...")
            gen = self.load_generation_by_type(zone, start_date, end_date, use_cache)
            gen_cols = [c for c in gen.columns if c != "timestamp"]
            merged = merged.merge(gen[["timestamp"] + gen_cols], on="timestamp", how="left")
            print(f"        {len(gen)} hourly rows, {len(gen_cols)} type columns")
        except Exception as exc:
            print(f"  Warning: Generation by type failed: {exc}")

        # 3. Wind & solar forecast
        try:
            print("  [3/5] Day-ahead wind & solar forecast ...")
            fc = self.load_generation_forecast(zone, start_date, end_date, use_cache)
            if not fc.empty:
                fc_cols = [c for c in fc.columns if c != "timestamp"]
                merged = merged.merge(fc[["timestamp"] + fc_cols], on="timestamp", how="left")
                print(f"        {len(fc)} hourly rows, {len(fc_cols)} forecast columns")
        except Exception as exc:
            print(f"  Warning: Generation forecast failed: {exc}")

        # 4. Day-ahead load forecast (published D-1 — genuinely available at inference)
        # Note: actual load already comes from Energy-Charts generation data; no need to
        # re-fetch it here (avoids column collision with total_load_mw).
        try:
            print("  [4/4] Day-ahead load forecast ...")
            load_fc = self.load_load_forecast(zone, start_date, end_date, use_cache)
            if not load_fc.empty:
                merged = merged.merge(load_fc, on="timestamp", how="left")
                print(f"        {len(load_fc)} hourly rows")
        except Exception as exc:
            print(f"  Warning: Load forecast failed: {exc}")

        # Derive aggregate columns
        merged = self._add_derived(merged)

        # Forward-fill gaps (weekends / holidays without new data)
        fill_cols = [c for c in merged.columns if c != "timestamp"]
        merged[fill_cols] = merged[fill_cols].ffill().bfill()

        print(f"  ENTSO-E merged shape: {merged.shape}")
        return merged

    # ------------------------------------------------------------------
    def load_cross_border_flows(
        self,
        zone: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Net cross-border physical flows to zone from each neighbor (hourly MW).
        Positive = import into zone, Negative = export from zone.
        """
        cache = self._cache_path("xborder", zone, start_date, end_date)
        if use_cache and cache.exists():
            return self._load_cache(cache)

        start = self._utc(start_date)
        end   = self._utc(end_date) + pd.Timedelta(hours=23)
        zone_eic = _EIC.get(zone, zone)

        all_flows: dict[str, pd.Series] = {}
        for neighbor in _NEIGHBORS.get(zone, []):
            nb_eic = _EIC.get(neighbor, neighbor)
            col    = f"net_import_{neighbor.lower()}_mw"
            # Use default-arg capture to avoid closure-over-variable bug
            try:
                imp = self._fetch_chunks(
                    lambda s, e, _f=nb_eic, _t=zone_eic: self.client.query_crossborder_flows(_f, _t, start=s, end=e),
                    start, end, label=f"import {neighbor}→{zone}",
                )
                exp = self._fetch_chunks(
                    lambda s, e, _f=zone_eic, _t=nb_eic: self.client.query_crossborder_flows(_f, _t, start=s, end=e),
                    start, end, label=f"export {zone}→{neighbor}",
                )
                imp_h = imp.resample("h").mean() if imp is not None else pd.Series(dtype=float)
                exp_h = exp.resample("h").mean() if exp is not None else pd.Series(dtype=float)
                net   = imp_h.subtract(exp_h, fill_value=0)
                all_flows[col] = net
            except Exception as exc:
                print(f"    ⚠️   {neighbor}↔{zone} flows failed: {exc}")

        if not all_flows:
            return pd.DataFrame({"timestamp": pd.date_range(start, end, freq="h", tz="UTC")})

        df = pd.DataFrame(all_flows)
        df.index.name = "timestamp"
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.to_csv(cache, index=False)
        return df

    # ------------------------------------------------------------------
    def load_generation_by_type(
        self,
        zone: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Actual generation by production type (hourly MW)."""
        cache = self._cache_path("gen_type", zone, start_date, end_date)
        if use_cache and cache.exists():
            return self._load_cache(cache)

        start    = self._utc(start_date)
        end      = self._utc(end_date) + pd.Timedelta(hours=23)
        zone_eic = _EIC.get(zone, zone)

        raw = self._fetch_chunks(
            lambda s, e, _z=zone_eic: self.client.query_generation(_z, start=s, end=e),
            start, end, label=f"generation {zone}",
        )
        if raw is None:
            return pd.DataFrame()

        # Resample to hourly
        gen = raw.resample("h").mean()

        # Flatten MultiIndex columns if present
        if isinstance(gen.columns, pd.MultiIndex):
            gen.columns = [
                "_".join(str(c) for c in col if c).rstrip("_")
                for col in gen.columns
            ]

        # Rename using PSR map
        rename = {k: v for k, v in _PSR_MAP.items() if k in gen.columns}
        gen    = gen.rename(columns=rename)

        gen.index.name = "timestamp"
        df = gen.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.to_csv(cache, index=False)
        return df

    # ------------------------------------------------------------------
    def load_generation_forecast(
        self,
        zone: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Day-ahead wind & solar generation forecast (hourly MW)."""
        cache = self._cache_path("gen_forecast", zone, start_date, end_date)
        if use_cache and cache.exists():
            return self._load_cache(cache)

        start    = self._utc(start_date)
        end      = self._utc(end_date) + pd.Timedelta(hours=23)
        zone_eic = _EIC.get(zone, zone)

        raw = self._fetch_chunks(
            lambda s, e, _z=zone_eic: self.client.query_wind_and_solar_forecast(
                _z, start=s, end=e, psr_type=None
            ),
            start, end, label=f"W&S forecast {zone}",
        )
        if raw is None:
            return pd.DataFrame()

        fc = raw.resample("h").mean()
        if isinstance(fc.columns, pd.MultiIndex):
            fc.columns = [
                "_".join(str(c) for c in col if c).rstrip("_")
                for col in fc.columns
            ]

        rename = {k: v for k, v in _FORECAST_MAP.items() if k in fc.columns}
        fc     = fc.rename(columns=rename)

        fc.index.name = "timestamp"
        df = fc.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.to_csv(cache, index=False)
        return df

    # ------------------------------------------------------------------
    def load_actual_load(
        self,
        zone: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Actual system load (total electricity consumption), hourly MW."""
        cache = self._cache_path("load_actual", zone, start_date, end_date)
        if use_cache and cache.exists():
            return self._load_cache(cache)

        start    = self._utc(start_date)
        end      = self._utc(end_date) + pd.Timedelta(hours=23)
        zone_eic = _EIC.get(zone, zone)

        raw = self._fetch_chunks(
            lambda s, e: self.client.query_load(zone_eic, start=s, end=e),
            start, end, label=f"actual load {zone}",
        )
        if raw is None:
            return pd.DataFrame()

        load_h = raw.resample("h").mean()
        load_h.index.name = "timestamp"
        df = load_h.reset_index()
        df.columns = ["timestamp", "total_load_mw"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.to_csv(cache, index=False)
        return df

    # ------------------------------------------------------------------
    def load_load_forecast(
        self,
        zone: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Day-ahead load forecast (MW) — published D-1, no data leakage at inference."""
        cache = self._cache_path("load_forecast", zone, start_date, end_date)
        if use_cache and cache.exists():
            return self._load_cache(cache)

        start    = self._utc(start_date)
        end      = self._utc(end_date) + pd.Timedelta(hours=23)
        zone_eic = _EIC.get(zone, zone)

        raw = self._fetch_chunks(
            lambda s, e: self.client.query_load_forecast(zone_eic, start=s, end=e),
            start, end, label=f"load forecast {zone}",
        )
        if raw is None:
            return pd.DataFrame()

        fc_h = raw.resample("h").mean()
        fc_h.index.name = "timestamp"
        df = fc_h.reset_index()
        df.columns = ["timestamp", "load_forecast_mw"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.to_csv(cache, index=False)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _utc(date_str: str) -> pd.Timestamp:
        ts = pd.Timestamp(date_str)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    def _cache_path(self, name: str, zone: str, start: str, end: str) -> Path:
        z   = zone.lower().replace("-", "")
        s   = start.replace("-", "")[:8]
        e   = end.replace("-", "")[:8]
        return self.cache_dir / f"{name}_{z}_{s}_{e}.csv"

    @staticmethod
    def _load_cache(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # apply(pd.Timestamp) handles each mixed +01:00/+02:00 DST offset individually,
        # then tz_convert to UTC — avoids array_strptime entirely
        df["timestamp"] = df["timestamp"].apply(lambda x: pd.Timestamp(x).tz_convert("UTC"))
        return df

    def _fetch_chunks(self, query_fn, start, end, label="", chunk_months=11):
        """
        Execute query_fn(start, end) in yearly chunks (ENTSO-E API limit ≈ 1 year).
        Returns concatenated Series/DataFrame or None on total failure.
        """
        pieces = []
        current = start
        while current <= end:
            chunk_end = min(current + pd.DateOffset(months=chunk_months), end)
            try:
                result = query_fn(current, chunk_end + pd.Timedelta(hours=1))
                if result is not None and len(result) > 0:
                    pieces.append(result)
                    print(f"      ✓ {label}: {current.date()} → {chunk_end.date()}")
            except Exception as exc:
                print(f"      ⚠️  {label} chunk {current.date()}→{chunk_end.date()}: {exc}")
            current = chunk_end + pd.Timedelta(hours=1)

        if not pieces:
            return None
        result = pd.concat(pieces)
        # Remove duplicate indices
        result = result[~result.index.duplicated(keep="first")]
        return result.sort_index()

    @staticmethod
    def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
        """Compute aggregate columns from raw ENTSO-E columns."""

        # Total net import
        flow_cols = [c for c in df.columns if c.startswith("net_import_") and c != "net_import_total_mw"]
        if flow_cols:
            df["net_import_total_mw"] = df[flow_cols].sum(axis=1)

        # Total hydro
        hydro_cols = [c for c in ["hydro_ror_mw", "hydro_reservoir_mw"] if c in df.columns]
        if hydro_cols:
            df["hydro_total_mw"] = df[hydro_cols].sum(axis=1)

        # Net pumped storage (positive = generating high price; negative = pumping = expecting low)
        gen_col = "pumped_storage_generation_mw"
        con_col = "pumped_storage_consumption_mw"
        if gen_col in df.columns and con_col in df.columns:
            df["pumped_storage_net_mw"] = df[gen_col].fillna(0) - df[con_col].fillna(0)
        elif gen_col in df.columns:
            df["pumped_storage_net_mw"] = df[gen_col]

        # Fossil mix
        fossil_cols = [c for c in ["gas_mw", "hard_coal_mw", "lignite_mw", "oil_mw"] if c in df.columns]
        if fossil_cols:
            df["fossil_mix_mw"] = df[fossil_cols].sum(axis=1)

        # Total renewable forecast
        fc_cols = [c for c in ["wind_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw"]
                   if c in df.columns]
        if fc_cols:
            df["renewable_forecast_total_mw"] = df[fc_cols].sum(axis=1)

        return df
