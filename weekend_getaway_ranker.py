"""
Weekend Getaway Ranker for India
A program to rank travel destinations based on distance, rating, and popularity.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from typing import Optional, List, Tuple


class WeekendGetawayRanker:
    """Main class for ranking weekend getaway destinations."""
    
    def __init__(self, csv_file: str = "india_travel_places.csv"):
        """
        Initialize the ranker with a CSV file.
        
        Args:
            csv_file: Path to the CSV file containing travel destinations
        """
        self.df = pd.read_csv(csv_file)
        self.scaler = MinMaxScaler()
        
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth using Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of first point (in degrees)
            lat2, lon2: Latitude and longitude of second point (in degrees)
            
        Returns:
            Distance in kilometers
        """
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        R = 6371
        
        return R * c
    
    def get_city_coordinates(self, city_name: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a city name from the dataset.
        
        Args:
            city_name: Name of the city
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        city_match = self.df[self.df['City'].str.lower() == city_name.lower()]
        if not city_match.empty:
            return (city_match.iloc[0]['Latitude'], city_match.iloc[0]['Longitude'])
        return None
    
    def calculate_distances(self, source_lat: float, source_lon: float) -> pd.Series:
        """
        Calculate distances from source to all destinations.
        
        Args:
            source_lat: Source latitude
            source_lon: Source longitude
            
        Returns:
            Series of distances in kilometers
        """
        distances = []
        for _, row in self.df.iterrows():
            dist = self.haversine_distance(
                source_lat, source_lon,
                row['Latitude'], row['Longitude']
            )
            distances.append(dist)
        return pd.Series(distances, name='Distance')
    
    def normalize_features(self, distances: pd.Series, ratings: pd.Series, 
                          popularity: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize distance, rating, and popularity to 0-1 scale.
        
        Note: For distance, we invert it (closer = higher score) after normalizing.
        
        Args:
            distances: Series of distances
            ratings: Series of ratings
            popularity: Series of popularity scores
            
        Returns:
            Tuple of normalized arrays (distance_score, rating_norm, popularity_norm)
        """
        # Normalize all features to 0-1 scale
        distance_norm = self.scaler.fit_transform(distances.values.reshape(-1, 1)).flatten()
        # For distance, we want closer = higher, so we invert after normalization
        distance_score = 1 - distance_norm
        
        rating_norm = self.scaler.fit_transform(ratings.values.reshape(-1, 1)).flatten()
        popularity_norm = self.scaler.fit_transform(popularity.values.reshape(-1, 1)).flatten()
        
        return distance_score, rating_norm, popularity_norm
    
    def calculate_ranking_score(self, distance_score: np.ndarray, rating_norm: np.ndarray,
                               popularity_norm: np.ndarray, 
                               distance_weight: float = 0.4, 
                               rating_weight: float = 0.4,
                               popularity_weight: float = 0.2) -> np.ndarray:
        """
        Calculate combined ranking score.
        
        Args:
            distance_score: Normalized distance scores (closer = higher)
            rating_norm: Normalized ratings
            popularity_norm: Normalized popularity scores
            distance_weight: Weight for distance (default 0.4)
            rating_weight: Weight for rating (default 0.4)
            popularity_weight: Weight for popularity (default 0.2)
            
        Returns:
            Array of combined scores
        """
        # Ensure weights sum to 1
        total_weight = distance_weight + rating_weight + popularity_weight
        if abs(total_weight - 1.0) > 0.01:
            distance_weight /= total_weight
            rating_weight /= total_weight
            popularity_weight /= total_weight
        
        scores = (distance_weight * distance_score + 
                 rating_weight * rating_norm + 
                 popularity_weight * popularity_norm)
        
        return scores
    
    def filter_by_distance(self, df: pd.DataFrame, max_distance: float) -> pd.DataFrame:
        """
        Filter destinations by maximum distance.
        
        Args:
            df: DataFrame with Distance column
            max_distance: Maximum distance in kilometers
            
        Returns:
            Filtered DataFrame
        """
        return df[df['Distance'] <= max_distance].copy()
    
    def filter_by_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """
        Filter destinations by category (if Category column exists).
        
        Args:
            df: DataFrame with Category column
            category: Category to filter by (e.g., 'Hills', 'Beach', 'Historical')
            
        Returns:
            Filtered DataFrame
        """
        if 'Category' not in df.columns:
            print("Warning: 'Category' column not found in dataset. Returning unfiltered data.")
            return df
        
        return df[df['Category'].str.lower() == category.lower()].copy()
    
    def rank_destinations(self, source: str, top_n: int = 10, 
                         max_distance: Optional[float] = None,
                         category: Optional[str] = None) -> pd.DataFrame:
        """
        Rank destinations from a source city.
        
        Args:
            source: Source city name or tuple of (latitude, longitude)
            top_n: Number of top destinations to return (default 10)
            max_distance: Optional maximum distance filter in kilometers
            category: Optional category filter (e.g., 'Hills', 'Beach', 'Historical')
            
        Returns:
            DataFrame with ranked destinations
        """
        # Get source coordinates
        if isinstance(source, str):
            coords = self.get_city_coordinates(source)
            if coords is None:
                # Try to parse as coordinates
                try:
                    lat, lon = map(float, source.split(','))
                    source_lat, source_lon = lat, lon
                except:
                    raise ValueError(f"City '{source}' not found in dataset and not valid coordinates.")
            else:
                source_lat, source_lon = coords
        elif isinstance(source, (tuple, list)) and len(source) == 2:
            source_lat, source_lon = float(source[0]), float(source[1])
        else:
            raise ValueError("Source must be a city name (string) or coordinates (tuple/list of 2 floats)")
        
        # Calculate distances
        distances = self.calculate_distances(source_lat, source_lon)
        
        # Create working dataframe
        result_df = self.df.copy()
        result_df['Distance'] = distances
        
        # Apply filters
        if max_distance is not None:
            result_df = self.filter_by_distance(result_df, max_distance)
        
        if category is not None:
            result_df = result_df.copy()  # Ensure we have a copy
            result_df = self.filter_by_category(result_df, category)
        
        if result_df.empty:
            print("No destinations found matching the criteria.")
            return pd.DataFrame()
        
        # Normalize features
        distance_score, rating_norm, popularity_norm = self.normalize_features(
            result_df['Distance'],
            result_df['Rating'],
            result_df['Popularity']
        )
        
        # Calculate ranking scores
        scores = self.calculate_ranking_score(distance_score, rating_norm, popularity_norm)
        result_df['Score'] = scores
        
        # Sort by score (descending) and select top N
        result_df = result_df.sort_values('Score', ascending=False).head(top_n)
        
        # Select and reorder columns for display
        display_columns = ['Place Name', 'City', 'Distance', 'Rating', 'Popularity', 'Score']
        if 'Category' in result_df.columns:
            display_columns.insert(2, 'Category')
        
        result_df = result_df[display_columns].copy()
        
        # Round numeric columns for better display
        result_df['Distance'] = result_df['Distance'].round(2)
        result_df['Rating'] = result_df['Rating'].round(2)
        result_df['Popularity'] = result_df['Popularity'].round(2)
        result_df['Score'] = result_df['Score'].round(4)
        
        return result_df.reset_index(drop=True)


def main():
    """Main function to run the Weekend Getaway Ranker."""
    print("=" * 70)
    print("Weekend Getaway Ranker for India")
    print("=" * 70)
    print()
    
    # Initialize ranker
    try:
        ranker = WeekendGetawayRanker("india_travel_places.csv")
    except FileNotFoundError:
        print("Error: 'india_travel_places.csv' file not found!")
        print("Please ensure the CSV file exists in the current directory.")
        return
    
    # Get source city from user
    print("Enter source city name (e.g., 'Mumbai', 'Delhi') or coordinates (lat,lon):")
    source_input = input("Source: ").strip()
    
    # Optional filters
    print("\nOptional filters (press Enter to skip):")
    max_dist_input = input("Maximum distance in km (e.g., 500): ").strip()
    max_distance = float(max_dist_input) if max_dist_input else None
    
    category_input = input("Category filter (Hills/Beach/Historical): ").strip()
    category = category_input if category_input else None
    
    # Rank destinations
    print("\n" + "=" * 70)
    print("Top 10 Weekend Getaway Destinations:")
    print("=" * 70)
    
    try:
        results = ranker.rank_destinations(
            source=source_input,
            top_n=10,
            max_distance=max_distance,
            category=category
        )
        
        if not results.empty:
            print("\n")
            print(results.to_string(index=False))
            print("\n" + "=" * 70)
            print(f"Total destinations found: {len(results)}")
        else:
            print("No destinations found matching your criteria.")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

