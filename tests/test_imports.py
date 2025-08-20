def test_imports():
    import emg_insight
    from emg_insight.datasets.ninapro_db1 import load_recording
    assert hasattr(emg_insight, "__version__")
