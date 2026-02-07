def test_fastapi_app_imports():
    # ensures app loads and routes exist
    from app.main import app  # noqa: F401
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/predict" in paths
