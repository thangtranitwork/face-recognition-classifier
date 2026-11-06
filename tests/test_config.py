"""Kiểm thử đơn vị cho module config."""

from pathlib import Path
import pytest

from facecls.config import load_config, AppConfig


def test_load_config_defaults(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("cosine_threshold: 0.42\nengine: insightface\n")

    cfg = load_config(cfg_file)
    assert cfg.cosine_threshold == 0.42
    assert cfg.engine == "insightface"
    assert cfg.allowed_modules == ["detection", "recognition"]
    assert cfg.det_thresh == 0.6
    assert cfg.image_max_size == 1600
    assert cfg.random_seed == 42
    assert cfg["cosine_threshold"] == 0.42


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file_12345.yaml")
