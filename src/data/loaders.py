"""
Data loading utilities for electricity price forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
from io import StringIO


class EnergyChartsLoader:
    """
    Load historical electricity price and generation data from energy-charts.info
    """
    
    BASE_URL = "https://api.energy-charts.info"
    
    ZONE_MAPPING = {
        'DE-LU': 'de',
        'ES': 'es'
    }

    TIMEZONE = {
        'DE-LU': 'Europe/Berlin',
        'ES': 'Europe/Madrid'
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize loader with optional cache directory.
        
        Parameters
        ----------
        cache_dir : Path, optional
            Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_day_ahead_prices(
        self,
        zone: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load Day Ahead Auction (DAA) prices for a bidding zone.
        
        Parameters
        ----------
        zone : str
            Bidding zone ('DE-LU' or 'ES')
        start_date : str or datetime
            Start date (inclusive) in local timezone
        end_date : str or datetime
            End date (inclusive) in local timezone
        use_cache : bool, default=True
            Use cached data if available
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['timestamp', 'price_eur_mwh']
            Timestamps are in UTC internally
        """
        if zone not in self.ZONE_MAPPING:
            raise ValueError(f"Unknown zone: {zone}. Must be one of {list(self.ZONE_MAPPING.keys())}")
        
        # Convert dates to datetime in local timezone
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
    
        
        # Check cache
        cache_file = self.cache_dir / zone.lower() / f"daa_prices_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file)
            # Parse timestamp with timezone info
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df
        
        # Download data
        print(f"Downloading DAA prices for {zone} from {start_date} to {end_date}")
        df = self._download_daa_prices(zone, start_date, end_date)
        
        # Cache the data with local timezone formatting
        self._save_cache_with_local_tz(df, cache_file, zone)
        print(f"Cached data to {cache_file}")
        
        return df
    
    def _save_cache_with_local_tz(self, df: pd.DataFrame, cache_file: Path, zone: str):
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        df_to_save = df.copy()

        # 👉 Convert UTC → local timezone (CET/CEST)
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.tz_convert(self.TIMEZONE[zone])

        # 👉 Format with explicit offset (+01:00 / +02:00)
        df_to_save['timestamp'] = df_to_save['timestamp'].apply(lambda x: x.isoformat())

        df_to_save.to_csv(cache_file, index=False)


    def _download_daa_prices(
        self,
        zone: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Internal method to download DAA prices using the Energy-Charts API.
        Returns data with UTC timestamps.
        """
        # start_date and end_date are already in UTC
        params = {
            'bzn': zone,
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        }
        response = requests.get(f"{self.BASE_URL}/price", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'unix_seconds' not in data or 'price' not in data:
            raise ValueError('Energy-Charts API returned an unexpected price response')

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['unix_seconds'], unit='s', utc=True),
            'price_eur_mwh': data['price']
        })
        # Keep timestamps in UTC (don't convert to local timezone)
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        return df
    
    def load_generation_data(
        self,
        zone: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load power generation data by source.
        
        Parameters
        ----------
        zone : str
            Bidding zone ('DE-LU' or 'ES')
        start_date : str or datetime
            Start date (inclusive) in local timezone
        end_date : str or datetime
            End date (inclusive) in local timezone
        use_cache : bool, default=True
            Use cached data if available
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['timestamp', 'wind', 'solar', 'nuclear', 'gas', 'coal', 'hydro', 'other']
            All values in MW. Timestamps are in UTC internally.
        """
        if zone not in self.ZONE_MAPPING:
            raise ValueError(f"Unknown zone: {zone}")
        
        # Convert dates to UTC
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache
        cache_file = self.cache_dir / zone.lower() / f"generation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached generation data from {cache_file}")
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        # Download data
        print(f"Downloading generation data for {zone}")
        df = self._download_generation_data(zone, start_date, end_date)
        
        # Cache with local timezone formatting
        self._save_cache_with_local_tz(df, cache_file, zone)
        
        return df
    
    def _download_generation_data(
        self,
        zone: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Internal method to download generation data using the Energy-Charts API.
        Returns data with UTC timestamps and comprehensive generation features.
        
        Features extracted:
        - Individual renewable sources (wind_onshore, wind_offshore, solar, hydro variants)
        - Fossil fuel sources (coal, gas, oil types)
        - Storage (pumped hydro consumption/generation)
        - Cross-border trading
        - Load and residual load
        - Renewable share metrics
        """
        params = {
            'country': self.ZONE_MAPPING[zone],
            'start': start_date.strftime('%Y-%m-%dT%H:%MZ'),
            'end': end_date.strftime('%Y-%m-%dT%H:%MZ')
        }
        response = requests.get(f"{self.BASE_URL}/public_power", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'unix_seconds' not in data or 'production_types' not in data:
            raise ValueError('Energy-Charts API returned an unexpected generation response')

        timestamps = pd.to_datetime(data['unix_seconds'], unit='s', utc=True)
        df = pd.DataFrame({'timestamp': timestamps})

        # Extract all generation sources with their original names
        for prod in data['production_types']:
            name = prod.get('name', '').lower().replace(' ', '_').replace('/', '_')
            values = np.asarray(prod.get('data', []), dtype=float)
            df[name] = values

        # Create aggregated features for price forecasting
        
        # 1. Total renewable generation (key price driver)
        renewable_cols = [col for col in df.columns if any(x in col for x in
                         ['wind', 'solar', 'hydro_run', 'biomass'])]
        if renewable_cols:
            df['total_renewable_mw'] = df[renewable_cols].sum(axis=1)
        
        # 2. Total fossil generation (marginal cost setter)
        fossil_cols = [col for col in df.columns if any(x in col for x in
                      ['fossil', 'coal', 'gas', 'oil']) and 'pumped' not in col]
        if fossil_cols:
            df['total_fossil_mw'] = df[fossil_cols].sum(axis=1)
        
        # 3. Flexible generation (can respond to price signals)
        flexible_cols = [col for col in df.columns if any(x in col for x in
                        ['fossil_gas', 'hydro_water_reservoir', 'hydro_pumped_storage'])]
        if flexible_cols:
            df['total_flexible_mw'] = df[flexible_cols].sum(axis=1)
        
        # 4. Must-run generation (inflexible, low marginal cost)
        mustrun_cols = [col for col in df.columns if any(x in col for x in
                       ['nuclear', 'lignite', 'biomass', 'hydro_run'])]
        if mustrun_cols:
            df['total_mustrun_mw'] = df[mustrun_cols].sum(axis=1)
        
        # 5. Storage dynamics (important for price volatility)
        if 'hydro_pumped_storage_consumption' in df.columns:
            df['pumped_storage_consumption_mw'] = df['hydro_pumped_storage_consumption'].abs()
        else:
            df['pumped_storage_consumption_mw'] = 0.0
            
        if 'hydro_pumped_storage' in df.columns:
            df['pumped_storage_generation_mw'] = df['hydro_pumped_storage']
        else:
            df['pumped_storage_generation_mw'] = 0.0
        
        # Net storage (negative = charging, positive = discharging)
        df['net_storage_mw'] = df['pumped_storage_generation_mw'] - df['pumped_storage_consumption_mw']
        
        # 6. Cross-border flows (import/export affects domestic prices)
        if 'cross_border_electricity_trading' in df.columns:
            df['cross_border_flow_mw'] = df['cross_border_electricity_trading']
        else:
            df['cross_border_flow_mw'] = 0.0
        
        # 7. Load metrics (demand side)
        if 'load' in df.columns:
            df['total_load_mw'] = df['load']
        
        if 'residual_load' in df.columns:
            df['residual_load_mw'] = df['residual_load']
        
        # 8. Renewable penetration (affects price dynamics)
        if 'renewable_share_of_generation' in df.columns:
            df['renewable_share_pct'] = df['renewable_share_of_generation']
        elif 'total_renewable_mw' in df.columns and 'total_load_mw' in df.columns:
            # Calculate if not provided
            total_gen = df['total_renewable_mw'] + df.get('total_fossil_mw', 0)
            df['renewable_share_pct'] = (df['total_renewable_mw'] / total_gen * 100).fillna(0)
        
        # 9. Supply-demand balance indicator
        if 'total_load_mw' in df.columns:
            renewable = df['total_renewable_mw'] if 'total_renewable_mw' in df.columns else 0
            fossil = df['total_fossil_mw'] if 'total_fossil_mw' in df.columns else 0
            cross_border = df['cross_border_flow_mw'] if 'cross_border_flow_mw' in df.columns else 0
            total_supply = renewable + fossil + cross_border
            df['supply_demand_ratio'] = total_supply / df['total_load_mw']
        
        # 10. Merit order position indicators
        # High renewable share pushes expensive fossil plants out of merit order
        if 'residual_load_mw' in df.columns and 'total_load_mw' in df.columns:
            df['residual_load_ratio'] = df['residual_load_mw'] / df['total_load_mw']
        
        # Select final feature columns for modeling
        feature_cols = ['timestamp']
        
        # Keep individual renewable sources (weather-dependent)
        for col in ['wind_onshore', 'wind_offshore', 'solar', 'hydro_run-of-river']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Keep key fossil sources (marginal cost setters)
        for col in ['fossil_gas', 'fossil_hard_coal', 'fossil_brown_coal_lignite',
                   'fossil_oil', 'fossil_coal-derived_gas']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Keep other important sources
        for col in ['nuclear', 'biomass', 'hydro_water_reservoir']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Add aggregated features
        for col in ['total_renewable_mw', 'total_fossil_mw', 'total_flexible_mw',
                   'total_mustrun_mw', 'net_storage_mw', 'cross_border_flow_mw',
                   'total_load_mw', 'residual_load_mw', 'renewable_share_pct',
                   'supply_demand_ratio', 'residual_load_ratio',
                   'pumped_storage_consumption_mw', 'pumped_storage_generation_mw']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Return only the columns that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        return df[available_cols]


class WeatherDataLoader:
    """
    Load weather data from Open-Meteo API (Historical + Forecast)
    
    Intelligently uses:
    - Historical Weather API for past data (training)
    - Forecast API for future data (predictions)
    """

    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    COORDINATES = {
        'DE-LU': {'latitude': 51.1657, 'longitude': 10.4515},
        'ES': {'latitude': 40.4168, 'longitude': -3.7038}
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/external")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _to_utc(timestamp: Union[str, datetime]) -> pd.Timestamp:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            return ts.tz_localize('UTC')
        return ts.tz_convert('UTC')

    def load_weather_data(
        self,
        zone: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load comprehensive weather data for electricity price forecasting.
        
        Parameters
        ----------
        zone : str
            Bidding zone ('DE-LU' or 'ES')
        start_date : str or datetime
            Start date
        end_date : str or datetime
            End date
        use_cache : bool, default=True
            Use cached data if available
            
        Returns
        -------
        pd.DataFrame
            DataFrame with weather features relevant for electricity prices:
            - temperature_2m_c: Temperature at 2m height (affects heating/cooling demand)
            - temperature_80m_c: Temperature at 80m height (wind turbine hub height)
            - wind_speed_10m_ms: Wind speed at 10m (general wind conditions)
            - wind_speed_100m_ms: Wind speed at 100m (wind power generation)
            - wind_direction_10m_deg: Wind direction (affects offshore/onshore generation)
            - solar_irradiance_wm2: Shortwave radiation (solar PV generation)
            - cloud_cover_pct: Cloud cover (affects solar generation)
            - precipitation_mm: Rainfall (affects hydro reservoir levels)
            - relative_humidity_pct: Humidity (affects demand patterns)
            - pressure_hpa: Atmospheric pressure (weather system indicator)
            Timestamps are in UTC internally
        """
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache
        cache_file = self.cache_dir / f"weather_{zone.lower()}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached weather data from {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            return df
        
        # Download data
        print(f"Downloading weather data for {zone}")
        df = self._download_weather_data(zone, start_date, end_date)
        
        # Cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df_to_save = df.copy()

        df_to_save['timestamp'] = df_to_save['timestamp'].dt.tz_convert('Europe/Berlin')
        df_to_save['timestamp'] = df_to_save['timestamp'].apply(lambda x: x.isoformat())

        df_to_save.to_csv(cache_file, index=False)
        
        return df
    
    def _download_weather_data(
        self,
        zone: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Download comprehensive weather data from Open-Meteo API.
        Intelligently uses Historical API for past data and Forecast API for future data.
        """
        if zone not in self.COORDINATES:
            raise ValueError(f"Unknown zone: {zone}")

        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)

        start_ts = self._to_utc(start_date)
        end_ts = self._to_utc(end_date)
        
        # Determine current date (for splitting historical vs forecast)
        now = pd.Timestamp.now(tz='UTC')
        yesterday = now - pd.Timedelta(days=1)
        
        coords = self.COORDINATES[zone]
        
        # Define weather variables to request
        hourly_vars = [
            'temperature_2m', 'temperature_80m', 'windspeed_10m', 'windspeed_100m',
            'winddirection_10m', 'shortwave_radiation', 'cloudcover', 'precipitation',
            'relativehumidity_2m', 'surface_pressure'
        ]
        
        dfs = []
        
        # 1. Historical data (if requested range includes past)
        if start_ts < yesterday:
            hist_end = min(end_ts, yesterday)
            print(f"  Fetching historical weather: {start_ts.date()} to {hist_end.date()}")
            
            params = {
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'hourly': ','.join(hourly_vars),
                'start_date': start_ts.strftime('%Y-%m-%d'),
                'end_date': hist_end.strftime('%Y-%m-%d'),
                'timezone': 'UTC'
            }
            
            response = requests.get(self.HISTORICAL_URL, params=params, timeout=30)
            response.raise_for_status()
            df_hist = self._parse_weather_response(response.json())
            if not df_hist.empty:
                dfs.append(df_hist)
        
        # 2. Forecast data (if requested range includes future)
        if end_ts > yesterday:
            forecast_start = max(start_ts, yesterday)
            print(f"  Fetching forecast weather: {forecast_start.date()} to {end_ts.date()}")
            
            params = {
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'hourly': ','.join(hourly_vars),
                'timezone': 'UTC'
            }
            
            response = requests.get(self.FORECAST_URL, params=params, timeout=30)
            response.raise_for_status()
            df_forecast = self._parse_weather_response(response.json())
            if not df_forecast.empty:
                dfs.append(df_forecast)
        
        # Combine historical and forecast data
        if not dfs:
            raise ValueError("No weather data retrieved")
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Filter to exact requested range
        mask = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
        return df.loc[mask].reset_index(drop=True)
    
    def _parse_weather_response(self, data: dict) -> pd.DataFrame:
        """Parse Open-Meteo API response and create DataFrame with derived features."""
        hourly = data.get('hourly', {})
        if 'time' not in hourly:
            return pd.DataFrame()

        # Create DataFrame with all weather features
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time'], utc=True),
            'temperature_2m_c': hourly.get('temperature_2m', []),
            'temperature_80m_c': hourly.get('temperature_80m', []),
            'wind_speed_10m_ms': hourly.get('windspeed_10m', []),
            'wind_speed_100m_ms': hourly.get('windspeed_100m', []),
            'wind_direction_10m_deg': hourly.get('winddirection_10m', []),
            'solar_irradiance_wm2': hourly.get('shortwave_radiation', []),
            'cloud_cover_pct': hourly.get('cloudcover', []),
            'precipitation_mm': hourly.get('precipitation', []),
            'relative_humidity_pct': hourly.get('relativehumidity_2m', []),
            'pressure_hpa': hourly.get('surface_pressure', [])
        })

        # Add derived features
        df['hdd'] = np.maximum(18 - df['temperature_2m_c'], 0)
        df['cdd'] = np.maximum(df['temperature_2m_c'] - 22, 0)
        df['wind_power_proxy'] = np.power(df['wind_speed_100m_ms'], 3)
        cloud_factor = df['cloud_cover_pct'] / 100.0
        df['solar_generation_proxy'] = df['solar_irradiance_wm2'] * (1 - 0.75 * cloud_factor)
        df['temp_gradient_80m_2m'] = df['temperature_80m_c'] - df['temperature_2m_c']
        
        return df


if __name__ == "__main__":
    # Test the loaders
    print("Testing data loaders...")
    
    # Test energy charts loader
    loader = EnergyChartsLoader(cache_dir="data/raw")
    
    # Load some sample data
    df_prices = loader.load_day_ahead_prices(
        zone='DE-LU',
        start_date='2026-04-02',
        end_date='2026-05-03',
        use_cache=False
    )

    df_generation = loader.load_generation_data(
        zone='DE-LU',
        start_date='2026-04-02',
        end_date='2026-05-03',
        use_cache=False
    )
    

    print("\nSample DAA prices:")
    print(df_prices.head())
    print(f"\nPrice statistics:\n{df_prices['price_eur_mwh'].describe()}")

    print("\nSample generation data:")
    print(df_generation.describe())
    
    # Test weather loader
    weather_loader = WeatherDataLoader(cache_dir="data/external")
    df_weather = weather_loader.load_weather_data(
        zone='DE-LU',
        start_date='2026-04-02',
        end_date='2026-05-10',
        use_cache=False
    )
    
    print("\nSample weather data:")
    print(df_weather.head())
    
    print("\nData loaders test completed successfully!")