import json
import yaml
from pathlib import Path
from typing import TypeVar, Type
from pydantic import BaseModel

# TypeVar ensures accurate type hinting for subclasses
T = TypeVar('T', bound='BaseConfig')


class BaseConfig(BaseModel):
    
    @classmethod
    def from_yaml(cls: Type[T], file_path: str | Path) -> T:
        """
        Reads a YAML file and validates it directly into the Pydantic model.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            # safe_load parses the YAML into a standard Python dictionary
            yaml_dict = yaml.safe_load(f) or {}

        # model_validate is Pydantic v2's built-in way to load from a dictionary
        return cls.model_validate(yaml_dict)