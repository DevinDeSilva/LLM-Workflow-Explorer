from importlib.metadata import PackageNotFoundError, version

from .manager import ProvOneManager

try:
    __version__ = version("pyexplannotator")
except PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = ["ProvOneManager", "__version__"]
