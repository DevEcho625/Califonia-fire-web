import redis
import json
import hashlib
from typing import Optional, Any, List
from datetime import timedelta
from app.core.config import settings
import geohash

class CacheService:
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.ttl = settings.CACHE_TTL
    
    def _generate_key(self, prefix: str, params: dict) -> str:
        """Generate cache key from parameters"""
        param_str = json.dumps(params, sort_keys=True)
        hash_digest = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = self.redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        if ttl is None:
            ttl = self.ttl
        self.redis_client.setex(
            key, 
            timedelta(seconds=ttl), 
            json.dumps(value)
        )
    
    def get_nearby_keys(self, lat: float, lng: float, radius: float = 1.0) -> List[str]:
        """Get cache keys for nearby areas using geohash"""
        # Generate geohash for current location
        current_hash = geohash.encode(lat, lng, precision=5)
        
        # Get neighboring geohashes
        neighbors = geohash.neighbors(current_hash)
        all_hashes = [current_hash] + list(neighbors)
        
        # Generate pattern keys for each hash
        keys = []
        for hash_val in all_hashes:
            pattern = f"*{hash_val}*"
            keys.extend(self.redis_client.keys(pattern))
        
        return keys
    
    def preload_nearby_areas(self, lat: float, lng: float, data_func):
        """Preload cache for nearby areas"""
        nearby_keys = self.get_nearby_keys(lat, lng)
        if not nearby_keys:  # If no nearby data cached
            # Get data for nearby areas
            nearby_data = data_func(lat, lng)
            for area in nearby_data:
                cache_key = self._generate_key("area", area)
                self.set(cache_key, area)

cache_service = CacheService()