"""
Example usage of the Weekend Getaway Ranker
"""

from weekend_getaway_ranker import WeekendGetawayRanker

# Initialize the ranker
ranker = WeekendGetawayRanker("india_travel_places.csv")

# Example 1: Rank destinations from Mumbai
print("Example 1: Top destinations from Mumbai")
print("=" * 70)
results = ranker.rank_destinations("Mumbai", top_n=10)
print(results)
print()

# Example 2: Rank destinations with distance filter
print("Example 2: Top destinations from Delhi within 500 km")
print("=" * 70)
results = ranker.rank_destinations("Delhi", top_n=10, max_distance=500)
print(results)
print()

# Example 3: Rank destinations by category
print("Example 3: Top Hill stations from Bangalore")
print("=" * 70)
# Using coordinates for Bangalore
results = ranker.rank_destinations((12.9716, 77.5946), top_n=10, category="Hills")
print(results)
print()

# Example 4: Combined filters
print("Example 4: Beach destinations within 300 km from Mumbai")
print("=" * 70)
results = ranker.rank_destinations("Mumbai", top_n=10, max_distance=300, category="Beach")
print(results)

