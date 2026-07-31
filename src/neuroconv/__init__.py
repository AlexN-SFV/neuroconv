from importlib.metadata import PackageNotFoundError, version

from .basedatainterface import BaseDataInterface
from .baseextractorinterface import BaseExtractorInterface
from .basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from .nwbconverter import ConverterPipe, NWBConverter
from .tools import get_format_summaries
from .tools.yaml_conversion_specification import run_conversion_from_yaml

# SpikeInterface stamps the defining package's version into an extractor's provenance dict and reads
# ``<package>.__version__`` back when reloading it, so any neuroconv-owned extractor — such as the
# recording that decodes logarithmically transformed EDF+ signals — cannot be pickled or reloaded unless
# this attribute exists.
try:
    __version__ = version("neuroconv")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree with no metadata
    __version__ = "0.0.0"
# Deleted so the package's public surface is unchanged: tests/imports.py pins the exact contents of
# neuroconv.__dict__, and leaving these bound would add two names to it.
del version, PackageNotFoundError
