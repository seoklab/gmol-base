from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gmol-base")
except PackageNotFoundError:
    pass

del PackageNotFoundError, version
