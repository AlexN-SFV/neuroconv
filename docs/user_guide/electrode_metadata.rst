Adding Channel Metadata to the Electrodes Table
===============================================

A recording format usually carries little metadata per channel — often just a label and a unit — while
the information you want in the NWB electrodes table (channel type, anatomical location, impedance,
quality flags, coordinates) lives somewhere else: a BIDS ``*_channels.tsv``, a lab spreadsheet, an
acquisition log.

NeuroConv does not read those files, and does not need a reader for each of them. **Anything set as a
channel property on the recording extractor becomes a column of the electrodes table.**

.. code-block:: python

    import numpy as np

    interface = MyRecordingInterface(file_path=file_path)
    recording = interface.recording_extractor

    recording.set_property("channel_type", np.array(["ECOG", "ECOG", "SEEG"]))
    recording.set_property("impedance_ohm", np.array([12000.0, 11400.0, 9800.0]))

Each property becomes one column, with one value per channel in the recording's channel order.

Describing the columns
----------------------

``metadata["Ecephys"]["Electrodes"]`` supplies *descriptions* for those columns — it does not supply
their values. Entries are ``{"name": ..., "description": ...}`` and the name must match a property you
set; a column with no entry is described as ``"no description"``.

.. code-block:: python

    metadata = interface.get_metadata()
    metadata["Ecephys"]["Electrodes"] = [
        dict(name="channel_type", description="electrode type from the lab's channel table"),
        dict(name="impedance_ohm", description="impedance measured at implantation, in ohms"),
    ]

Property names with special meaning
-----------------------------------

A few names are mapped onto the NWB standard rather than written verbatim:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Becomes
   * - ``brain_area``
     - the standard ``location`` column
   * - ``location`` (two-dimensional)
     - ``rel_x``, ``rel_y``, ``rel_z``
   * - ``group``, ``group_name``
     - the electrode group link, handled by the writer
   * - ``channel_name``, ``contact_ids``
     - the channel and electrode identifiers, always written
   * - ``physical_unit``, ``gain_to_uV``, ``offset_to_uV``, ``gain_to_physical_unit``, ``offset_to_physical_unit``
     - the series' unit and scaling — deliberately *not* electrode columns

Setting ``x``, ``y`` and ``z`` writes the standard absolute-coordinate columns.

Worked example: a BIDS sidecar
------------------------------

The only real work is matching the external table's channel names to the recording's, which is rarely
mechanical — acquisition systems prefix their labels, so a sidecar's ``A3`` may be ``POL A3`` in the
file.

.. code-block:: python

    import csv
    import numpy as np

    channels = {
        row["name"]: row
        for row in csv.DictReader(open(channels_tsv, encoding="utf-8"), delimiter="\t")
    }
    channel_ids = [str(channel_id) for channel_id in recording.channel_ids]

    def sidecar_name(channel_id):
        # Adjust to your files; this strips an acquisition prefix such as "POL ".
        return channel_id.split()[-1] if " " in channel_id else channel_id

    matched = sum(1 for channel_id in channel_ids if sidecar_name(channel_id) in channels)
    if matched < len(channel_ids) // 2:
        raise ValueError(
            f"only {matched} of {len(channel_ids)} channels matched the sidecar — "
            f"check the name mapping. Recording: {channel_ids[:3]}, sidecar: {list(channels)[:3]}"
        )

    def column(field):
        return np.array([channels.get(sidecar_name(c), {}).get(field, "n/a") for c in channel_ids])

    recording.set_property("bids_type", column("type"))
    recording.set_property("status", column("status"))
    recording.set_property("brain_area", column("group"))  # -> the standard 'location' column

The match check matters more than it looks. Without it a mismatched mapping does not raise — every
lookup simply misses and the column comes out uniformly ``"n/a"``, which reads like a sidecar with no
data rather than a bug in the name mapping.

Half is a deliberately loose floor, chosen so the example survives a sidecar that legitimately covers
only some channels. Tighten it to ``matched < len(channel_ids)`` when every channel should be present,
which is the usual case for a BIDS sidecar describing the whole recording — a partly-correct mapping
otherwise passes and leaves a half-populated column.

.. note::

    **Units are not metadata.** This happens in practice: some acquisition software writes a channel's
    unit through an ASCII encoding that silently drops the micro sign, so a signal recorded in ``µV`` is
    labelled ``V`` and is then read a million times too large, while the BIDS sidecar beside it still
    says ``µV``.

    ``set_property`` cannot fix that — a unit determines the scaling, so the correction is to the gain:

    .. code-block:: python

        header_units = recording.get_property("physical_unit")  # what the file declared
        mislabelled = np.array(
            [sidecar == "µV" and header.strip() == "V" for sidecar, header in zip(column("units"), header_units)]
        )
        gains = recording.get_channel_gains().copy()
        gains[mislabelled] /= 1e6
        recording.set_channel_gains(gains)

    NeuroConv will not do this for you: inferring a unit the file does not state would be a silent
    change to the data. Check ``physical_unit`` against your external source before converting.
