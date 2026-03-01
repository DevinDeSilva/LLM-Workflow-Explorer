import pytest
import yaml
import logging
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.utils.utils import load_config, time_wrapper, regex_add_strings


def test_load_config_success(tmp_path):
    # Create a dummy config file
    config_data = {"key": "value", "nested": {"foo": "bar"}}
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # First try without mock, see if it works (if dycomutils is installed)
    try:
        config = load_config(str(config_file))
        assert config["key"] == "value"
    except (ImportError, AttributeError):
        # Fallback to mock if dycomutils is not behaving as expected in this environment
        with patch("dycomutils.config.ConfigDict", side_effect=lambda x: x):
            config = load_config(str(config_file))
            assert config == config_data

def test_load_config_file_not_found():
    config = load_config("non_existent_file.yaml")
    assert config == {}

def test_load_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        f.write("invalid: yaml: :")

    config = load_config(str(config_file))
    assert config == {}

def test_time_wrapper(caplog):
    caplog.set_level(logging.INFO)
    
    @time_wrapper
    def mock_func(x, y):
        return x + y

    result = mock_func(1, 2)
    assert result == 3
    assert "Function mock_func Took" in caplog.text

def test_regex_add_strings_success():
    template = "Hello {name}, welcome to {place}!"
    result = regex_add_strings(template, name="Alice", place="Wonderland")
    assert result == "Hello Alice, welcome to Wonderland!"

def test_regex_add_strings_partial():
    template = "Hello {name}, welcome to {place}!"
    result = regex_add_strings(template, name="Alice")
    assert result == "Hello Alice, welcome to {place}!"

def test_regex_add_strings_special_chars():
    template = "Value: {special_key}"
    # re.escape(key) should handle characters that might be regex-significant
    result = regex_add_strings(template, **{"special_key": "fixed*value"})
    assert result == "Value: fixed*value"

def test_regex_add_strings_complex_key():
    # If the key itself has special regex characters
    template = "Value: {key.with.dots}"
    result = regex_add_strings(template, **{"key.with.dots": "success"})
    assert result == "Value: success"

def test_regex_add_strings_non_string_value():
    template = "Count: {count}"
    result = regex_add_strings(template, count=10)
    assert result == "Count: 10"

def test_regex_add_strings_error():
    # Trigger an exception (template being None should cause re.sub to fail)
    with patch("src.utils.utils.logger") as mock_logger:
        result = regex_add_strings(None, key="value")
        assert result == ""
        mock_logger.error.assert_called()

