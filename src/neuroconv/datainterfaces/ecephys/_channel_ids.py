"""
Channel ids for the EEG readers whose channels are named by their file (EDF, BDF, BrainVision, EEGLAB).

Those formats do not require channel labels to be unique, and real files repeat them: an exporter that
writes a generic label for every auxiliary input, or a montage that lists one electrode twice. The labels
become SpikeInterface channel ids and the NWB electrodes table's ``channel_name`` column, both of which
must be unique, so a repeated label is renamed rather than failing the conversion.

The rule, shared so every reader names the same channel the same way: the first channel with a label
keeps it, and each later one gets ``_2``, ``_3``, ... in file order, skipping any name already taken. The
labels as the file gives them are kept as the ``original_channel_name`` property, which the writer turns
into an electrodes column. A file whose labels are already unique gets ids identical to its labels and
no extra property, so its output is unchanged.

Once that column is in an electrodes table, every row needs a value: :func:`complete_original_channel_names`
gives the rows of other recordings, whose labels were never renamed, their ``channel_name``.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pydantic

ORIGINAL_CHANNEL_NAME_PROPERTY = "original_channel_name"
ORIGINAL_CHANNEL_NAME_DESCRIPTION = (
    "The channel's label as the source file gives it. It differs from channel_name only where the file "
    "gave more than one channel the same label and the repeat was renamed to keep channel_name unique."
)

# Frames in these directories are skipped when attributing a warning, so it points at the caller's own
# line: neuroconv itself, and pydantic, whose validate_call wraps interface constructors.
_INTERNAL_DIRECTORIES = tuple(
    str(Path(path).resolve()) for path in (Path(__file__).parents[2], Path(pydantic.__file__).parent)
)


def _stacklevel_outside_neuroconv() -> int:
    """
    The ``stacklevel`` that attributes a warning, raised by this function's caller, to the first frame outside neuroconv.

    Computed rather than fixed because the same reader is reached at different depths: through an
    interface's constructor, through its extractor directly, or from a sub-interface.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and str(Path(frame.f_code.co_filename).resolve()).startswith(_INTERNAL_DIRECTORIES):
        frame = frame.f_back
        level += 1
    return level


def unique_channel_ids(labels: list[str], groups: list | None = None) -> list[str]:
    """
    Return one unique id per channel, renaming each repeat of a label after its first occurrence.

    Parameters
    ----------
    labels : list of str
        The channel labels, in file order.
    groups : list, optional
        One key per channel. Ids need only be unique within a group, so a label repeated in two groups
        is left alone. EDF uses this for its streams: SpikeInterface reads one sampling rate at a time
        and has always accepted a label that recurs at another rate, so those files keep their ids.

    Returns
    -------
    list of str
        The ids, aligned with ``labels``. Equal to ``labels`` when no label repeats within a group.
    """
    labels = [str(label) for label in labels]
    groups = [None] * len(labels) if groups is None else list(groups)
    # A new id avoids every label in the file, not just its group's, so a renamed channel cannot take
    # the name of a real channel in another stream, which would merge the two in the electrodes table.
    taken = set(labels)
    seen_in_group = set()
    ids = []
    for label, group in zip(labels, groups):
        if (group, label) not in seen_in_group:
            seen_in_group.add((group, label))
            ids.append(label)
            continue
        suffix = 2
        while f"{label}_{suffix}" in taken:
            suffix += 1
        new_id = f"{label}_{suffix}"
        taken.add(new_id)
        ids.append(new_id)
    return ids


def warn_about_renamed_channels(labels: list[str], ids: list[str], source: str, accepts_channels_to_skip=True) -> None:
    """
    Say which channels were renamed, once per file.

    Parameters
    ----------
    labels, ids : list of str
        The channel labels and the ids chosen for them, aligned.
    source : str
        Names the file in the message, e.g. ``"The EDF file 'x.edf'"``.
    accepts_channels_to_skip : bool, default: True
        Whether the reader takes ``channels_to_skip``, so the message only explains it where it applies.
    """
    renamed = [(label, channel_id) for label, channel_id in zip(labels, ids) if label != channel_id]
    if not renamed:
        return
    shown = ", ".join(f"{label!r} -> {channel_id!r}" for label, channel_id in renamed[:10])
    if len(renamed) > 10:
        shown += f", ... ({len(renamed)} channels)"
    message = (
        f"{source} gives more than one channel the same label. Channel ids must be unique, so the repeats "
        f"are renamed: {shown}. The labels as the file gives them are kept in the "
        f"'{ORIGINAL_CHANNEL_NAME_PROPERTY}' electrodes column."
    )
    if accepts_channels_to_skip:
        message += (
            " channels_to_skip accepts either a renamed id, to drop that channel alone, or a repeated label, "
            "to drop every channel carrying it."
        )
    warnings.warn(message, UserWarning, stacklevel=_stacklevel_outside_neuroconv())


def indices_to_keep(labels: list[str], ids: list[str], channels_to_skip: list | None) -> list[int]:
    """
    Positions of the channels that ``channels_to_skip`` leaves, matching each entry to an id or a label.

    A label matches every channel carrying it, which is what skipping by label did before repeats were
    renamed; a renamed id matches that channel alone. An entry matching nothing is not reported here.
    """
    skip = set(channels_to_skip or [])
    return [index for index, (label, channel_id) in enumerate(zip(labels, ids)) if not ({label, channel_id} & skip)]


def unknown_channels_to_skip(labels: list[str], ids: list[str], channels_to_skip: list | None) -> list:
    """The entries of ``channels_to_skip`` that are neither a channel id nor a label, sorted."""
    return sorted(set(channels_to_skip or []) - set(labels) - set(ids))


def set_original_channel_names(recording, labels: list[str], ids: list[str]) -> None:
    """Record the file's labels on ``recording`` when any id differs from its label; otherwise do nothing."""
    if list(labels) != list(ids):
        recording.set_property(key=ORIGINAL_CHANNEL_NAME_PROPERTY, values=np.asarray(labels))


def report_renamed_channels(recording, source: str, accepts_channels_to_skip: bool = True) -> None:
    """
    Warn about the renamed channels a recording still holds, once any channel selection is done.

    For a reader that renames before channels are dropped: a repeat that was dropped is not reported,
    and when none remain the ``original_channel_name`` property is removed again, so the result matches
    a reader that dropped the channels first.
    """
    labels = recording.get_property(ORIGINAL_CHANNEL_NAME_PROPERTY)
    if labels is None:
        return
    labels = [str(label) for label in labels]
    ids = [str(channel_id) for channel_id in recording.get_channel_ids()]
    if labels == ids:
        recording.delete_property(ORIGINAL_CHANNEL_NAME_PROPERTY)
        return
    warn_about_renamed_channels(
        labels=labels, ids=ids, source=source, accepts_channels_to_skip=accepts_channels_to_skip
    )


def complete_original_channel_names(recording, nwbfile, metadata: dict):
    """
    Keep the ``original_channel_name`` electrodes column filled for every row, before ``recording`` is written.

    Left alone, the writer fills the column with ``""`` for every row whose recording lacks the property,
    which is every recording that renamed nothing: another file in the same pipe, or the half of a file
    whose repeats went to a different interface. Their labels are their channel names, so:

    * a recording without the property, written to a table that has the column, gets it from its channel
      names;
    * a recording with the property, written to a table whose rows predate the column, first has the
      column added with those rows' channel names.

    Parameters
    ----------
    recording : BaseRecording
        The recording about to be written. It is not modified; a view carries the added property.
    nwbfile : NWBFile
    metadata : dict
        Not modified.

    Returns
    -------
    tuple
        ``(recording, metadata)`` to write with. The metadata describes the column when the writer may
        create it, so it is not left as "no description".
    """
    from ...tools.spikeinterface.spikeinterface import _get_channel_name

    has_property = ORIGINAL_CHANNEL_NAME_PROPERTY in recording.get_property_keys()
    electrodes = nwbfile.electrodes
    has_column = electrodes is not None and ORIGINAL_CHANNEL_NAME_PROPERTY in electrodes.colnames

    if has_column and not has_property:
        recording = recording.select_channels(channel_ids=recording.get_channel_ids())
        recording.set_property(key=ORIGINAL_CHANNEL_NAME_PROPERTY, values=np.asarray(_get_channel_name(recording)))
        return recording, metadata

    if not has_property:
        return recording, metadata

    if not has_column and electrodes is not None:
        existing = len(electrodes)
        channel_names = (
            [str(name) for name in electrodes["channel_name"][:]] if "channel_name" in electrodes.colnames else []
        )
        # A row with no channel_name has no label this can recover; blank is all it can say.
        channel_names = channel_names if len(channel_names) == existing else [""] * existing
        nwbfile.add_electrode_column(
            name=ORIGINAL_CHANNEL_NAME_PROPERTY, description=ORIGINAL_CHANNEL_NAME_DESCRIPTION, data=channel_names
        )
        return recording, metadata

    ecephys = dict(metadata.get("Ecephys", {}))
    columns = list(ecephys.get("Electrodes", []))
    if not any(column.get("name") == ORIGINAL_CHANNEL_NAME_PROPERTY for column in columns):
        columns.append(dict(name=ORIGINAL_CHANNEL_NAME_PROPERTY, description=ORIGINAL_CHANNEL_NAME_DESCRIPTION))
    ecephys["Electrodes"] = columns
    return recording, {**metadata, "Ecephys": ecephys}
