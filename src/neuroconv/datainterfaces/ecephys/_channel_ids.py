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
"""

import warnings

import numpy as np

ORIGINAL_CHANNEL_NAME_PROPERTY = "original_channel_name"


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
    warnings.warn(message, UserWarning, stacklevel=3)


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
