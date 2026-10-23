"""
SpikeInterface recording extractor for MEF 2 recordings.

The parsing lives in :mod:`._mef2_reader` and the decoding in :mod:`._red`, which depend only on NumPy and,
optionally, numba. This module is where SpikeInterface enters, so it must only be imported lazily, as
:meth:`~..mef2datainterface.MEF2RecordingInterface.get_extractor_class` does.

The class is defined at module scope because SpikeInterface records an extractor's class by its import path
in ``to_dict()``; classes built inside a function cannot be pickled or reloaded.
"""

import warnings

import numpy as np
from pydantic import DirectoryPath
from spikeinterface.core import BaseRecording

from ._mef2_reader import MEF2Recording, read_mef2_recording
from ._mef3_extractor import MEF3RecordingSegment
from .._channel_ids import (
    _stacklevel_outside_neuroconv,
    report_renamed_channels,
    set_original_channel_names,
)


def _gains_to_microvolts(recording: MEF2Recording) -> np.ndarray:
    """Each channel's voltage conversion factor, which is in microvolts per count, with the sign it is stored with."""
    gains = np.array([channel.voltage_conversion_factor for channel in recording.channels], dtype="float64")
    # 0 is the header's "no entry".
    missing = [channel.name for channel, gain in zip(recording.channels, gains) if gain == 0.0 or not np.isfinite(gain)]
    if missing:
        shown = ", ".join(repr(name) for name in missing[:5]) + (
            f", ... ({len(missing)} channels)" if len(missing) > 5 else ""
        )
        warnings.warn(
            f"The MEF 2 channels {shown} of '{recording.path.name}' give no voltage conversion factor, so their "
            "samples are written as though one count were one microvolt.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
        gains[[recording.channel_ids.index(name) for name in missing]] = 1.0
    return gains


class MEF2RecordingExtractor(BaseRecording):
    """
    Lazy SpikeInterface recording for the channels of a MEF 2 recording directory sampled at one rate.

    Each uninterrupted run of blocks becomes one segment, placed on the recording's timeline at its first
    block's start time, measured from the recording's header start time (see ``MEF2Recording.start_time``).

    Parameters
    ----------
    folder_path : DirectoryPath
        The directory holding the recording's ``.mef`` files.
    channels_to_skip : list of str, optional
        Channels to leave out.
    stream_name : str, optional
        The sampling rate to read, such as ``"500 Hz"``; required when the files have more than one.
    validate_crc : bool, default: False
        Check every block's CRC as it is decoded.
    """

    extractor_name = "MEF2Recording"
    mode = "folder"
    name = "mef2"

    def __init__(
        self,
        folder_path: DirectoryPath,
        channels_to_skip: list | None = None,
        stream_name: str | None = None,
        validate_crc: bool = False,
    ):
        recording = read_mef2_recording(
            folder_path, channels_to_skip=channels_to_skip, stream_name=stream_name, validate_crc=validate_crc
        )
        channel_ids = recording.channel_ids
        super().__init__(
            sampling_frequency=recording.sampling_frequency, channel_ids=channel_ids, dtype=np.dtype("int32")
        )
        for run_index, run in enumerate(recording.runs):
            self.add_recording_segment(
                MEF3RecordingSegment(
                    channels=recording.channels,
                    run_index=run_index,
                    number_of_samples=run.number_of_samples,
                    sampling_frequency=recording.sampling_frequency,
                    t_start=(run.start_time - recording.start_time) / 1e6,
                )
            )

        # Gains take the stored counts to microvolts; the ecephys writer then scales microvolts to the volts
        # NWB's ElectricalSeries expects. A negative factor means the counts are inverted (MEF 2 specification,
        # "Voltage Conversion Factor"), and is kept, so every recording comes out with the same polarity. MEF
        # has no offset.
        gains = _gains_to_microvolts(recording)
        self.set_channel_gains(gains=gains)
        self.set_channel_offsets(offsets=np.zeros(len(channel_ids)))
        self.set_property(key="channel_name", values=np.asarray(channel_ids))
        self.set_property(key="physical_unit", values=np.asarray(["uV"] * len(channel_ids)))
        self.set_property(key="gain_to_physical_unit", values=gains)
        self.set_property(key="offset_to_physical_unit", values=np.zeros(len(channel_ids)))
        set_original_channel_names(self, labels=recording.channel_labels, ids=channel_ids)
        report_renamed_channels(self, source=f"The MEF 2 recording '{recording.path.name}'")

        self.mef2_recording = recording
        self._kwargs = dict(
            folder_path=str(folder_path),
            channels_to_skip=channels_to_skip,
            stream_name=stream_name,
            validate_crc=validate_crc,
        )
