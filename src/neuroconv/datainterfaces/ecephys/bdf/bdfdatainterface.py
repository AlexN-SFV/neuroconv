import numpy as np
from pydantic import FilePath

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....tools import get_package
from ....utils import DeepDict

# Physical-dimension string -> factor converting that unit to microvolts.
_UNIT_TO_MICROVOLTS = {
    "uv": 1.0,
    "µv": 1.0,
    "microvolt": 1.0,
    "microvolts": 1.0,
    "mv": 1e3,
    "millivolt": 1e3,
    "millivolts": 1e3,
    "v": 1e6,
    "volt": 1e6,
    "volts": 1e6,
}


def _unit_to_microvolts(unit: str) -> float:
    """Factor to convert a channel's physical unit to microvolts (assume µV if unrecognized)."""
    return _UNIT_TO_MICROVOLTS.get(str(unit).strip().lower(), 1.0)


def _read_bdf_recording(file_path: FilePath, channels_to_skip: list | None = None):
    """
    Read a BioSemi BDF file into a SpikeInterface recording, using pyedflib directly.

    BDF stores 24-bit samples. neo's ``EDFRawIO`` (used by SpikeInterface's ``read_edf``) reads them as
    16-bit, corrupting the data, so this reads with ``pyedflib`` (which handles 24-bit correctly) and
    wraps the result in a ``NumpyRecording``: the 24-bit digital values are stored as ``int32`` and the
    per-channel gain/offset (converted to microvolts) let the ecephys writer scale to volts.

    Returns
    -------
    tuple
        ``(recording, header)`` where ``header`` is pyedflib's file-level header dict.
    """
    get_package(package_name="pyedflib", excluded_platforms_and_python_versions=dict(darwin=dict(arm=["3.9"])))
    import pyedflib
    from spikeinterface.core import NumpyRecording

    reader = pyedflib.EdfReader(str(file_path))
    try:
        header = reader.getHeader()
        start_datetime = reader.getStartdatetime()
        number_of_signals = reader.signals_in_file

        channel_names = [str(reader.getLabel(index)).strip() for index in range(number_of_signals)]
        sampling_frequencies = [float(reader.getSampleFrequency(index)) for index in range(number_of_signals)]

        keep = [index for index, name in enumerate(channel_names) if not (channels_to_skip and name in channels_to_skip)]
        if not keep:
            raise ValueError("No channels remain after applying channels_to_skip.")

        unique_rates = {sampling_frequencies[index] for index in keep}
        if len(unique_rates) != 1:
            raise ValueError(
                "BDF channels have differing sampling rates "
                f"({sorted(unique_rates)} Hz); only uniform-rate recordings are supported. "
                "Use channels_to_skip to drop off-rate channels (e.g. a status/annotation channel)."
            )
        sampling_frequency = unique_rates.pop()

        gains_to_microvolts = np.empty(len(keep), dtype="float64")
        offsets_to_microvolts = np.empty(len(keep), dtype="float64")
        digital_traces = []
        for position, index in enumerate(keep):
            physical_min, physical_max = reader.getPhysicalMinimum(index), reader.getPhysicalMaximum(index)
            digital_min, digital_max = reader.getDigitalMinimum(index), reader.getDigitalMaximum(index)
            gain = (physical_max - physical_min) / (digital_max - digital_min)
            offset = physical_min - digital_min * gain
            to_microvolts = _unit_to_microvolts(reader.getPhysicalDimension(index))
            gains_to_microvolts[position] = gain * to_microvolts
            offsets_to_microvolts[position] = offset * to_microvolts
            digital_traces.append(reader.readSignal(index, digital=True).astype("int32"))
    finally:
        reader._close()

    kept_names = [channel_names[index] for index in keep]
    traces = np.stack(digital_traces, axis=1)  # (n_samples, n_channels), int32 24-bit digital values

    recording = NumpyRecording(
        traces_list=[traces],
        sampling_frequency=sampling_frequency,
        channel_ids=kept_names,
    )
    # Gains/offsets convert the stored digital values to microvolts; the ecephys writer then scales to volts.
    recording.set_channel_gains(gains=gains_to_microvolts)
    recording.set_channel_offsets(offsets=offsets_to_microvolts)
    recording.set_property(key="channel_name", values=np.asarray(kept_names))

    header = dict(header)
    header["startdate"] = start_datetime
    return recording, header


class BDFRecordingInterface(BaseRecordingExtractorInterface):
    """
    Data interface for converting BioSemi Data Format (BDF) data to NWB.

    BDF is BioSemi's variant of EDF, but with 24-bit samples instead of 16-bit. neo's EDF reader (used
    by SpikeInterface) truncates those to 16-bit, so this interface reads the file with ``pyedflib``
    directly (which handles 24-bit correctly) and wraps it in a ``NumpyRecording``.

    See https://www.biosemi.com/faq/file_format.htm for the format specification.

    Not supported on M1 macs.
    """

    display_name = "BDF Recording"
    keywords = BaseRecordingExtractorInterface.keywords + ("BioSemi Data Format", "EEG")
    associated_suffixes = (".bdf",)
    info = "Interface for BioSemi Data Format (BDF) recording data."

    @classmethod
    def get_source_schema(cls) -> dict:
        source_schema = super().get_source_schema()
        source_schema["properties"]["file_path"]["description"] = "Path to the .bdf file."
        return source_schema

    @staticmethod
    def get_available_channel_ids(file_path: FilePath) -> list:
        """Return all channel names in a BDF file."""
        get_package(package_name="pyedflib", excluded_platforms_and_python_versions=dict(darwin=dict(arm=["3.9"])))
        import pyedflib

        reader = pyedflib.EdfReader(str(file_path))
        try:
            return [str(reader.getLabel(index)).strip() for index in range(reader.signals_in_file)]
        finally:
            reader._close()

    @classmethod
    def get_extractor_class(cls):
        from spikeinterface.core import NumpyRecording

        return NumpyRecording

    def _initialize_extractor(self, interface_kwargs: dict):
        """Read the BDF file into a NumpyRecording (via pyedflib), caching the header."""
        recording, header = _read_bdf_recording(
            file_path=interface_kwargs["file_path"], channels_to_skip=interface_kwargs.get("channels_to_skip")
        )
        self._bdf_header = header
        return recording

    def __init__(
        self,
        file_path: FilePath,
        *,
        verbose: bool = False,
        es_key: str = "ElectricalSeries",
        channels_to_skip: list | None = None,
    ):
        """
        Load and prepare data for BDF.

        Parameters
        ----------
        file_path : str or Path
            Path to the .bdf file.
        verbose : bool, default: False
            Allows verbose.
        es_key : str, default: "ElectricalSeries"
            Key for the ElectricalSeries metadata.
        channels_to_skip : list, default: None
            Channels to skip when adding the data to the nwbfile. Use this to drop non-neural channels
            such as the BioSemi ``Status`` trigger channel, or channels sampled at a different rate.
        """
        super().__init__(
            file_path=file_path, verbose=verbose, es_key=es_key, channels_to_skip=channels_to_skip
        )

    def get_metadata(self) -> DeepDict:
        metadata = super().get_metadata()

        header = getattr(self, "_bdf_header", dict())
        nwbfile_metadata = dict(session_start_time=header.get("startdate"), experimenter=header.get("technician"))
        nwbfile_metadata = {key: value for key, value in nwbfile_metadata.items() if value}
        if "experimenter" in nwbfile_metadata:
            nwbfile_metadata["experimenter"] = [nwbfile_metadata["experimenter"]]
        metadata["NWBFile"].update(nwbfile_metadata)

        subject_metadata = {
            key: value
            for key, value in dict(
                subject_id=header.get("patientcode"), date_of_birth=header.get("birthdate")
            ).items()
            if value
        }
        if subject_metadata:
            # Fill the schema-required fields (subject_id, sex, species) with defaults so the metadata
            # stays valid whenever the BDF header carries any subject info (cf. the EEGLAB interface).
            subject_metadata.setdefault("subject_id", "Unknown")
            subject_metadata.setdefault("species", "Unknown species")
            subject_metadata.setdefault("sex", "U")
            metadata["Subject"] = {**metadata.get("Subject", dict()), **subject_metadata}

        return metadata
