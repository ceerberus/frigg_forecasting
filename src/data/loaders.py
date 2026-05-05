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
        Returns data with UTC timestamps.
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

        for prod in data['production_types']:
            name = prod.get('name', '').lower()
            values = np.asarray(prod.get('data', []), dtype=float)
            if name in ('wind_onshore', 'wind_offshore'):
                df['wind'] = df.get('wind', np.zeros(len(timestamps), dtype=float)) + values
            elif name == 'solar':
                df['solar'] = values
            elif name in ('nuclear', 'gas', 'coal', 'hydro', 'other'):
                df[name] = values
            else:
                df[name] = values

        for col in ['wind', 'solar', 'nuclear', 'gas', 'coal', 'hydro', 'other']:
            if col not in df.columns:
                df[col] = np.nan

        return df[['timestamp', 'wind', 'solar', 'nuclear', 'gas', 'coal', 'hydro', 'other']]


class WeatherDataLoader:
    """
    Load weather data from Open-Meteo API or other sources
    """

    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
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
        Load weather data (temperature, wind speed, solar irradiance).
        
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
            DataFrame with columns: ['timestamp', 'temperature_c', 'wind_speed_ms', 'solar_irradiance_wm2']
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
        Download weather data from the Open-Meteo API.
        Returns data with UTC timestamps.
        """
        if zone not in self.COORDINATES:
            raise ValueError(f"Unknown zone: {zone}")

        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)

        start_ts = self._to_utc(start_date)
        end_ts = self._to_utc(end_date)

        coords = self.COORDINATES[zone]
        params = {
            'latitude': coords['latitude'],
            'longitude': coords['longitude'],
            'hourly': 'temperature_2m,windspeed_10m,shortwave_radiation',
            'start_date': start_ts.strftime('%Y-%m-%d'),
            'end_date': end_ts.strftime('%Y-%m-%d'),
            'timezone': 'UTC'
        }

        response = requests.get(self.OPEN_METEO_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hourly = data.get('hourly', {})
        if 'time' not in hourly:
            raise ValueError('Open-Meteo API returned an unexpected weather response')

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time'], utc=True),
            'temperature_c': hourly.get('temperature_2m', []),
            'wind_speed_ms': hourly.get('windspeed_10m', []),
            'solar_irradiance_wm2': hourly.get('shortwave_radiation', [])
        })

        mask = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
        df = df.loc[mask].reset_index(drop=True)
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
        use_cache=True
    )
    
    print("\nSample DAA prices:")
    print(df_prices.head())
    print(f"\nPrice statistics:\n{df_prices['price_eur_mwh'].describe()}")
    
    # Test weather loader
    weather_loader = WeatherDataLoader(cache_dir="data/external")
    df_weather = weather_loader.load_weather_data(
        zone='DE-LU',
        start_date='2026-04-02',
        end_date='2026-05-03',
        use_cache=True
    )
    
    print("\nSample weather data:")
    print(df_weather.head())
    
    print("\nData loaders test completed successfully!")