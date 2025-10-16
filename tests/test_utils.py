from wildfire_map.utils import CaliforniaGeoDataLoader

def test_california():
    loader = CaliforniaGeoDataLoader()
    assert loader.is_in_california(37.95, -121.29), "CaliforniaGeoDataLoader is not working"