"""
Unique channel ids for EDF files read through SpikeInterface.

SpikeInterface's EDF reader takes the channel ids from the labels (``use_names_as_ids=True``) and refuses
a file that repeats one within a stream. Such a file is read with neo's positional ids instead and then
renamed to the labels made unique, by the rule in :mod:`.._channel_ids`. A file whose labels are unique
is read exactly as before, so its recording is the same object it always was.
"""

import numpy as np
from pydantic import FilePath

from ._edf_log_transform import _decode_field
from .._channel_ids import (
    ORIGINAL_CHANNEL_NAME_PROPERTY,
    indices_to_keep,
    set_original_channel_names,
    unique_channel_ids,
)

_ANNOTATIONS_SIGNAL_LABEL = "EDF Annotations"
_STATIC_HEADER_SIZE = 256
_NUMBER_OF_SIGNALS_OFFSET = 252
# Widths of the per-signal fields preceding the samples-per-record field: label, transducer, dimension,
# physical/digital minimum and maximum, prefiltering.
_WIDTHS_BEFORE_SAMPLES_PER_RECORD = (16, 80, 8, 8, 8, 8, 8, 80)


def has_repeated_labels(file_path: FilePath) -> bool:
    """
    Whether any two data signals sampled at the same rate share a label, from the header alone.

    Read directly rather than through pyedflib, which reads every annotation in an EDF+ file on opening
    it, so that the check costs nothing for the files that pass it. Labels are compared stripped, which
    is at least as coarse as pyedflib's comparison, so a repeat SpikeInterface would refuse is never missed.
    """
    with open(file_path, "rb") as file:
        file.seek(_NUMBER_OF_SIGNALS_OFFSET)
        try:
            number_of_signals = int(_decode_field(file.read(4)) or 0)
        except ValueError:
            # A malformed header is the reader's to report, in its own words.
            return False
        if number_of_signals <= 0:
            return False
        labels = [_decode_field(file.read(16)) for _ in range(number_of_signals)]
        file.seek(_STATIC_HEADER_SIZE + number_of_signals * sum(_WIDTHS_BEFORE_SAMPLES_PER_RECORD))
        samples_per_record = [_decode_field(file.read(8)) for _ in range(number_of_signals)]

    # Every signal shares the record duration, so the samples per record alone groups them by rate.
    seen = set()
    for label, samples in zip(labels, samples_per_record):
        if label == _ANNOTATIONS_SIGNAL_LABEL:
            continue
        if (samples, label) in seen:
            return True
        seen.add((samples, label))
    return False


def available_channel_ids(file_path: FilePath) -> list[str]:
    """The ids of every data signal in a file read through SpikeInterface, in file order."""
    from pyedflib import EdfReader

    edf_reader = EdfReader(str(file_path))
    try:
        labels = edf_reader.getSignalLabels()
        sampling_frequencies = list(edf_reader.getSampleFrequencies())
    finally:
        # EDFlib refuses to open a file it already has open, so the handle is released here rather than
        # left to garbage collection.
        edf_reader.close()
    return unique_channel_ids(labels=labels, groups=sampling_frequencies)


def neo_labels_and_ids(recording) -> tuple[list[str], list[str]]:
    """
    The label and id of every data signal in the file, as neo numbers them, from a recording neo read.

    neo lists the signals in file order and groups them into one stream per sampling rate, so ids made
    unique within each stream here are the ones :func:`available_channel_ids` gives.
    """
    signal_channels = recording.neo_reader.header["signal_channels"]
    labels = [str(name) for name in signal_channels["name"]]
    return labels, unique_channel_ids(labels=labels, groups=list(signal_channels["stream_id"]))


def rename_to_unique_channel_ids(recording):
    """
    Rename a recording read with neo's positional ids to its labels made unique.

    The caller reports the renaming with :func:`.._channel_ids.report_renamed_channels` once it has
    dropped any channels, so a repeat that is dropped goes unmentioned.

    Parameters
    ----------
    recording : NeoBaseRecordingExtractor
        Read with ``use_names_as_ids=False``, so its ids are neo's signal numbers.
    """
    labels, ids = neo_labels_and_ids(recording)
    signal_channels = recording.neo_reader.header["signal_channels"]
    position_of_neo_id = {str(neo_id): position for position, neo_id in enumerate(signal_channels["id"])}
    positions = [position_of_neo_id[str(neo_id)] for neo_id in recording.get_channel_ids()]
    stream_labels = [labels[position] for position in positions]
    stream_ids = [ids[position] for position in positions]

    renamed = recording.rename_channels(new_channel_ids=stream_ids)
    # SpikeInterface set channel_name to the labels, repeats and all, when it was not using them as ids,
    # and the electrodes table identifies a channel by that name, so it must be the unique id instead.
    renamed.set_property(key="channel_name", values=np.asarray(stream_ids))
    set_original_channel_names(recording=renamed, labels=stream_labels, ids=stream_ids)
    return renamed


def channel_ids_to_remove(recording, channels_to_skip: list) -> list:
    """
    The ids ``channels_to_skip`` removes, matching each entry to an id or to a label.

    A label the file repeats removes every channel carrying it. Entries matching nothing are passed
    through, so SpikeInterface's own error about an unknown channel is unchanged.
    """
    ids = [str(channel_id) for channel_id in recording.get_channel_ids()]
    labels = recording.get_property(ORIGINAL_CHANNEL_NAME_PROPERTY)
    labels = ids if labels is None else [str(label) for label in labels]
    keep = set(indices_to_keep(labels=labels, ids=ids, channels_to_skip=channels_to_skip))
    removed = [recording.get_channel_ids()[index] for index in range(len(ids)) if index not in keep]
    unmatched = [entry for entry in channels_to_skip if entry not in set(labels) | set(ids)]
    return removed + unmatched
