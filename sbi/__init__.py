version_prefix = "0.1.0"
try:
    from sbi._version import __version__

except ImportError:
    __version__ = version_prefix
