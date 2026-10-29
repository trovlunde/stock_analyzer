from stock_analysis.ai.model_manager import ModelManager


def test_constructing_does_not_create_models_dir(tmp_path):
    models_dir = tmp_path / "models"

    ModelManager(models_dir=str(models_dir))

    assert not models_dir.exists()


def test_load_model_on_missing_dir_returns_none_without_raising(tmp_path):
    models_dir = tmp_path / "models"
    manager = ModelManager(models_dir=str(models_dir))

    assert manager.load_model("AAPL_movement", "daily") is None


def test_list_models_on_missing_dir_returns_empty_without_raising(tmp_path):
    models_dir = tmp_path / "models"
    manager = ModelManager(models_dir=str(models_dir))

    assert manager.list_models() == []


def test_list_models_on_missing_dir_does_not_log_an_error(tmp_path, caplog):
    """A not-yet-trained deployment is normal, not an error — don't spam ERROR logs for it."""
    models_dir = tmp_path / "models"
    manager = ModelManager(models_dir=str(models_dir))

    with caplog.at_level("ERROR"):
        manager.list_models()

    assert caplog.records == []


def test_save_model_creates_dir_and_round_trips(tmp_path):
    models_dir = tmp_path / "models"
    manager = ModelManager(models_dir=str(models_dir))

    assert manager.save_model({"dummy": "model"}, "AAPL_movement", "daily") is True
    assert models_dir.exists()
    assert manager.load_model("AAPL_movement", "daily", bypass_cache=True) == {"dummy": "model"}
