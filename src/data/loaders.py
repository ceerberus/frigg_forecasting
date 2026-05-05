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
            Start date (inclusive)
        end_date : str or datetime
            End date (inclusive)
        use_cache : bool, default=True
            Use cached data if available
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['timestamp', 'price_eur_mwh']
        """
        if zone not in self.ZONE_MAPPING:
            raise ValueError(f"Unknown zone: {zone}. Must be one of {list(self.ZONE_MAPPING.keys())}")
        
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache
        cache_file = self.cache_dir / zone.lower() / f"daa_prices_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            return df
        
        # Download data
        print(f"Downloading DAA prices for {zone} from {start_date} to {end_date}")
        
        # Note: This is a placeholder - actual API calls would go here
        # For now, we'll create a template structure
        df = self._download_daa_prices(zone, start_date, end_date)
        
        # Cache the data
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"Cached data to {cache_file}")
        
        return df
    
    def _download_daa_prices(
        self,
        zone: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Internal method to download DAA prices.
        
        Note: This is a placeholder implementation.
        In production, this would make actual API calls to energy-charts.info
        """
        # Create hourly timestamp range
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq='H',
            tz='UTC'
        )
        
        # Placeholder: Generate synthetic data for development
        # In production, replace with actual API calls
        np.random.seed(42)
        
        if zone == 'DE-LU':
            # Germany: Higher volatility, occasional negative prices
            base_price = 60
            prices = base_price + np.random.normal(0, 20, len(timestamps))
            # Add some negative prices (5% of the time)
            negative_mask = np.random.random(len(timestamps)) < 0.05
            prices[negative_mask] = np.random.uniform(-50, 0, negative_mask.sum())
        else:  # ES
            # Spain: More stable, fewer negative prices
            base_price = 55
            prices = base_price + np.random.normal(0, 15, len(timestamps))
            # Rare negative prices (1% of the time)
            negative_mask = np.random.random(len(timestamps)) < 0.01
            prices[negative_mask] = np.random.uniform(-20, 0, negative_mask.sum())
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price_eur_mwh': prices
        })
        
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
            Start date (inclusive)
        end_date : str or datetime
            End date (inclusive)
        use_cache : bool, default=True
            Use cached data if available
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['timestamp', 'wind', 'solar', 'nuclear', 'gas', 'coal', 'hydro', 'other']
            All values in MW
        """
        if zone not in self.ZONE_MAPPING:
            raise ValueError(f"Unknown zone: {zone}")
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache
        cache_file = self.cache_dir / zone.lower() / f"generation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached generation data from {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            return df
        
        # Download data
        print(f"Downloading generation data for {zone}")
        df = self._download_generation_data(zone, start_date, end_date)
        
        # Cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        
        return df
    
    def _download_generation_data(
        self,
        zone: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Internal method to download generation data.
        Placeholder implementation - replace with actual API calls.
        """
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H', tz='UTC')
        
        np.random.seed(42)
        n = len(timestamps)
        
        if zone == 'DE-LU':
            # Germany generation mix
            data = {
                'timestamp': timestamps,
                'wind': np.random.uniform(5000, 40000, n),
                'solar': np.random.uniform(0, 35000, n),
                'nuclear': np.random.uniform(6000, 8000, n),
                'gas': np.random.uniform(5000, 15000, n),
                'coal': np.random.uniform(3000, 12000, n),
                'hydro': np.random.uniform(2000, 5000, n),
                'other': np.random.uniform(1000, 3000, n)
            }
        else:  # ES
            # Spain generation mix
            data = {
                'timestamp': timestamps,
                'wind': np.random.uniform(2000, 15000, n),
                'solar': np.random.uniform(0, 12000, n),
                'nuclear': np.random.uniform(6000, 7500, n),
                'gas': np.random.uniform(4000, 12000, n),
                'coal': np.random.uniform(500, 3000, n),
                'hydro': np.random.uniform(3000, 8000, n),
                'other': np.random.uniform(500, 2000, n)
            }
        
        return pd.DataFrame(data)


class WeatherDataLoader:
    """
    Load weather data from Open-Meteo API or other sources
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/external")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
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
        df.to_csv(cache_file, index=False)
        
        return df
    
    def _download_weather_data(
        self,
        zone: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Placeholder for weather data download.
        In production, use Open-Meteo API or ERA5 reanalysis data.
        """
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H', tz='UTC')
        
        np.random.seed(42)
        n = len(timestamps)
        
        # Add seasonal patterns
        hour_of_year = timestamps.dayofyear * 24 + timestamps.hour
        
        if zone == 'DE-LU':
            # Germany: Colder, more wind
            temp_base = 10 + 10 * np.sin(2 * np.pi * hour_of_year / (365 * 24))
            data = {
                'timestamp': timestamps,
                'temperature_c': temp_base + np.random.normal(0, 3, n),
                'wind_speed_ms': np.abs(8 + 4 * np.sin(2 * np.pi * hour_of_year / (365 * 24)) + np.random.normal(0, 3, n)),
                'solar_irradiance_wm2': np.maximum(0, 400 * np.sin(2 * np.pi * hour_of_year / (365 * 24)) + np.random.normal(0, 50, n))
            }
        else:  # ES
            # Spain: Warmer, more solar
            temp_base = 18 + 12 * np.sin(2 * np.pi * hour_of_year / (365 * 24))
            data = {
                'timestamp': timestamps,
                'temperature_c': temp_base + np.random.normal(0, 4, n),
                'wind_speed_ms': np.abs(6 + 3 * np.sin(2 * np.pi * hour_of_year / (365 * 24)) + np.random.normal(0, 2, n)),
                'solar_irradiance_wm2': np.maximum(0, 600 * np.sin(2 * np.pi * hour_of_year / (365 * 24)) + np.random.normal(0, 80, n))
            }
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the loaders
    print("Testing data loaders...")
    
    # Test energy charts loader
    loader = EnergyChartsLoader(cache_dir="data/raw")
    
    # Load some sample data
    df_prices = loader.load_day_ahead_prices(
        zone='DE-LU',
        start_date='2024-01-01',
        end_date='2024-01-07',
        use_cache=False
    )
    
    print("\nSample DAA prices:")
    print(df_prices.head())
    print(f"\nPrice statistics:\n{df_prices['price_eur_mwh'].describe()}")
    
    # Test weather loader
    weather_loader = WeatherDataLoader(cache_dir="data/external")
    df_weather = weather_loader.load_weather_data(
        zone='DE-LU',
        start_date='2024-01-01',
        end_date='2024-01-07',
        use_cache=False
    )
    
    print("\nSample weather data:")
    print(df_weather.head())
    
    print("\nData loaders test completed successfully!")

# Made with Bob
