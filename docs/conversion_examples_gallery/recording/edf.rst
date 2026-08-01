European Data Format (EDF) conversion
-------------------------------------

Install NeuroConv with the additional dependencies necessary for reading EDF data.

.. code-block:: bash

    pip install "neuroconv[edf]"

Converting EDF Electrode Channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~neuroconv.datainterfaces.ecephys.edf.edfdatainterface.EDFRecordingInterface` is designed specifically for electrode recording channels that will be stored as an ElectricalSeries in the NWB file.
Other auxiliary signal channels should be excluded using the ``channels_to_skip`` parameter.

.. code-block:: python

    from datetime import datetime
    from zoneinfo import ZoneInfo
    from pathlib import Path
    from neuroconv.datainterfaces import EDFRecordingInterface

    file_path = f"{ECEPHY_DATA_PATH}/edf/electrode_and_analog_data/electrode_and_analog_data.edf"

    # Get all channel IDs to identify which ones to skip
    all_channels = EDFRecordingInterface.get_available_channel_ids(file_path)
    print(f"Available channels: {all_channels}")

    # Identify non-electrode channels that should be skipped
    # We don't have an automatic way to detect non-electrode channels, user passes this knowledge here
    channels_to_skip = ["TRIG", "OSAT", "PR", "Pleth"]  # Example: trigger and physiological monitoring

    # Create interface with channels to skip
    interface = EDFRecordingInterface(
        file_path=file_path,
        channels_to_skip=channels_to_skip
    )

    # Extract what metadata we can from the source files
    metadata = interface.get_metadata()
    # For data provenance we add the time zone information to the conversion
    session_start_time = metadata["NWBFile"]["session_start_time"].replace(tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    # Add subject information (required for DANDI upload)
    metadata["Subject"] = dict(subject_id="subject1", species="Mus musculus", sex="M", age="P30D")

    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = f"{path_to_save_nwbfile}"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

Handling Channels with Heterogeneous Offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single NWB ``ElectricalSeries`` stores one scalar ``offset`` shared by all of its channels.
Some EDF files contain channels whose physical-to-digital offsets differ from one another (for example
different electrode types digitized over different physical ranges), which cannot be represented in a
single ``ElectricalSeries``. A normal ``run_conversion`` on such a file raises
``ValueError: Recording extractors with heterogeneous offsets are not supported.``

Use :py:meth:`~neuroconv.datainterfaces.ecephys.baserecordingextractorinterface.BaseRecordingExtractorInterface.split_by_offset`
to partition the channels by offset. It returns one sub-interface per distinct offset (each restricted to
the channels sharing that offset, and each with a distinct ``es_key``), using only the offsets in the file
itself — no BIDS or other external metadata required. Combining the sub-interfaces in a ``ConverterPipe``
then writes one ``ElectricalSeries`` per offset into a **single** NWB file, sharing one electrodes table
and without rescaling the data. If the recording has a single offset, ``split_by_offset`` returns just the
original interface.

.. code-block:: python

    from zoneinfo import ZoneInfo
    from neuroconv import ConverterPipe
    from neuroconv.datainterfaces import EDFRecordingInterface

    file_path = f"{ECEPHY_DATA_PATH}/edf/heterogeneous_offsets.edf"

    interface = EDFRecordingInterface(file_path=file_path)

    # One sub-interface per distinct channel offset (each with a distinct es_key)
    sub_interfaces = interface.split_by_offset()
    converter = ConverterPipe(data_interfaces={sub.es_key: sub for sub in sub_interfaces})

    metadata = converter.get_metadata()
    session_start_time = metadata["NWBFile"]["session_start_time"].replace(tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    metadata["Subject"] = dict(subject_id="subject1", species="Mus musculus", sex="M", age="P30D")

    # Writes one NWB file with one ElectricalSeries per distinct offset
    nwbfile_path = f"{path_to_save_nwbfile}"
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

Discontinuous EDF+ (EDF+D) Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EDF+ files declare in their header whether the recording is continuous (``EDF+C``) or discontinuous
(``EDF+D``). In an ``EDF+D`` file the data records need not be contiguous in time, so a record's start
time cannot be derived from its position: the format instead stores it in the mandatory
``EDF Annotations`` signal, whose first annotation in each record is a "time-keeping" annotation giving
that record's offset from the start of the file.

Neither SpikeInterface's reader nor the ``pyedflib`` library beneath it can open these files — ``pyedflib``
refuses them with *"the file is discontinuous and cannot be read"*. ``EDFRecordingInterface`` therefore
detects ``EDF+D`` from the header and reads such files with neuroconv's own reader, which needs only
NumPy. No change to your code is required:

.. code-block:: python

    from zoneinfo import ZoneInfo
    from neuroconv.datainterfaces import EDFRecordingInterface

    # Any .edf path; EDF+D is detected from the file's own header.
    interface = EDFRecordingInterface(file_path=file_path)

    interface.is_discontinuous  # True for an EDF+D file
    interface.number_of_runs    # number of contiguous runs of data records found

    metadata = interface.get_metadata()
    session_start_time = metadata["NWBFile"]["session_start_time"].replace(tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    metadata["Subject"] = dict(subject_id="subject1", species="Homo sapiens", sex="U", age="P30Y")

    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

Each contiguous run of records becomes one ``ElectricalSeries``, placed at its true start time on the
session timeline (via ``starting_time`` and ``rate``) and sharing a single electrodes table with the
others. The series are named with the run index appended — ``ElectricalSeries0``, ``ElectricalSeries1``,
and so on. Two ``TimeIntervals`` tables are written alongside them, both on the same timeline: ``runs``
gives each run's boundaries with a ``run_index`` column, and ``annotations`` holds the annotations
carried in the ``EDF Annotations`` signals. Both follow the recording if it is moved with
``set_aligned_starting_time``.

.. note::

    Many exporters label continuous recordings ``EDF+D`` as a matter of course. When the records turn out
    to be contiguous — which is common — ``number_of_runs`` is 1 and the file is written as a single
    ``ElectricalSeries`` with no ``runs`` table, just as a plain EDF would be. An ``annotations`` table is
    still written if the file carries any annotations.

Set ``write_annotations=False`` or ``write_runs=False`` to suppress either table:

.. code-block:: python

    interface.run_conversion(
        nwbfile_path=nwbfile_path,
        metadata=metadata,
        overwrite=True,
        write_annotations=False,
    )

.. note::

    These two blocks are illustrative rather than doctested, because there is no ``EDF+D`` file in the
    gin test data yet. The tests build them byte by byte instead; see
    ``tests/test_modalities/test_ecephys/test_edfd_interface.py``.

Plain EDF and ``EDF+C`` files are unaffected by any of this and continue to be read by SpikeInterface.

Floating-Point and Long-Integer Data (EDF+ "Filtered" Signals)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EDF stores every sample as a 2-byte integer, which cannot hold the dynamic range of floating-point or
long-integer data. EDF+ addresses that with a logarithmic transformation rather than a wider sample: the
samples stay 2-byte integers, the signal is marked by the physical dimension ``Filtered``, and its
original unit together with the two transformation parameters go in the prefiltering field — for example
``sign*LN[sign*(uV      )/(0.01    )]/(0.002   )``. See
`the specification <https://www.edfplus.info/specs/edffloat.html>`_.

``EDFRecordingInterface`` detects such signals from the header and decodes them, giving back physical
values in the signal's original unit. On a continuous file the decoding happens on top of the usual
reader, which has no notion of the transformation and would otherwise apply the header's *linear* gain
and offset to what are logarithms — returning values wrong by many orders of magnitude with no error and
no warning. A discontinuous file is read by neuroconv's own reader, which decodes natively. Both use one
shared implementation, so a transformed signal means the same thing either way.

.. code-block:: python

    interface = EDFRecordingInterface(file_path=file_path)
    interface.log_transformed_channels  # {} for an ordinary file

Two consequences worth knowing:

* The recording's dtype becomes ``float64`` rather than ``int16``, since decoded values are physical and
  the specification's own accuracy table reaches 1.4e68 — beyond what ``float32`` holds. A recording
  carries one dtype, so a single transformed signal beside many linear ones promotes them all, and the
  dataset is four times the size it would be as ``int16``. Compression recovers much of that; if the
  transformed signal is not needed, ``channels_to_skip`` keeps the recording integer.
* Accuracy is inherently about ``a/2`` in relative terms, ``a`` being the logarithmic step. That is the
  trade the transformation makes, not a loss introduced by the conversion.

A signal marked ``Filtered`` whose prefiltering field does not carry readable parameters is refused
rather than guessed at, because its samples are logarithms and cannot be interpreted without them. Drop
it with ``channels_to_skip`` if you do not need it. If its original unit is not a voltage, it is decoded
and read as microvolts with a warning, since an ``ElectricalSeries`` holds voltages — convert such a
channel with
:py:class:`~neuroconv.datainterfaces.ecephys.edf.edfanaloginterface.EDFAnalogInterface` instead.

When the Header's Unit Is Wrong
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some exporters write the physical dimension through an ASCII encoding that silently drops the micro
sign, so a channel recorded in ``µV`` is labelled ``V`` — and is then read one million times too large.
BESA exports do this: their BIDS ``channels.tsv`` says ``µV`` for channels whose EDF header says ``V``,
and the affected channels give themselves away by having physical ranges no larger than those of the
channels correctly labelled ``uV`` in the same file.

NeuroConv reports the unit the header declared, on continuous and discontinuous files alike, but will
not second-guess it — inferring a unit the file does not state would be a silent change to the data.
Check it against your sidecar before converting:

.. code-block:: python

    import numpy as np
    from neuroconv.datainterfaces import EDFRecordingInterface

    interface = EDFRecordingInterface(file_path=file_path)
    recording = interface.recording_extractor

    header_units = recording.get_property("physical_unit")
    print(set(header_units))  # e.g. {'V', 'uV'} -- a mix is the tell

    mislabelled = np.array([unit.strip() == "V" for unit in header_units])
    gains = recording.get_channel_gains().copy()
    gains[mislabelled] /= 1e6
    recording.set_channel_gains(gains)

For carrying the rest of a BIDS sidecar — channel type, status, anatomical location, coordinates — into
the NWB electrodes table, see
:doc:`channel metadata and the electrodes table </user_guide/electrode_metadata>`. That route is not
EDF-specific: any channel property set on the recording becomes an electrodes column.

Converting Auxiliary EDF Channels as TimeSeries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Auxiliary signals such as physiological monitoring signals, triggers, or auxiliary data should be stored as generic TimeSeries objects
rather than ElectricalSeries. Use the dedicated :py:class:`~neuroconv.datainterfaces.ecephys.edf.edfanaloginterface.EDFAnalogInterface`
to handle these channels.

**Important**: TimeSeries objects in PyNWB require all channels to have the same physical unit.
If your auxiliary channels have different units (e.g., triggers with no unit, OSAT in %, PR in bpm),
you'll need to create separate EDFAnalogInterface instances for each unit type:

.. code-block:: python

    from datetime import datetime
    from zoneinfo import ZoneInfo
    from pathlib import Path
    from neuroconv.datainterfaces import EDFAnalogInterface

    file_path = f"{ECEPHY_DATA_PATH}/edf/electrode_and_analog_data/electrode_and_analog_data.edf"

    # Example: Trigger channels (no unit)
    trigger_channels = ["TRIG"]  # Trigger signals

    interface = EDFAnalogInterface(
        file_path=file_path,
        channels_to_include=trigger_channels
    )

    # Extract metadata and add timezone information
    metadata = interface.get_metadata()
    # For data provenance we add the time zone information to the conversion
    session_start_time = metadata["NWBFile"]["session_start_time"].replace(tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    # Add subject information (required for DANDI upload)
    metadata["Subject"] = dict(subject_id="subject1", species="Mus musculus", sex="M", age="P30D")

    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = f"{path_to_save_nwbfile}"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

Combining Electrode and Auxiliary Channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To convert both electrode and auxiliary channels into a single NWB file, use the ConverterPipe with multiple interfaces.
Remember to group auxiliary channels by their unit types:

.. code-block:: python

    from datetime import datetime
    from zoneinfo import ZoneInfo
    from pathlib import Path
    from neuroconv import ConverterPipe
    from neuroconv.datainterfaces import EDFRecordingInterface, EDFAnalogInterface
    from neuroconv.utils import dict_deep_update

    file_path = f"{ECEPHY_DATA_PATH}/edf/electrode_and_analog_data/electrode_and_analog_data.edf"

    # Define the channels to process
    all_non_electrical_channels = ["TRIG", "OSAT", "PR", "Pleth"]  # All auxiliary channels

    # Create electrode interface (skip all auxiliary channels)
    recording_interface = EDFRecordingInterface(
        file_path=file_path,
        channels_to_skip=all_auxiliary_channels,

    )

    # Create separate analog interfaces for each unit type
    trigger_channels_metadata_key = "time_series_trigger"  # No unit
    trigger_interface = EDFAnalogInterface(
        file_path=file_path,
        channels_to_include=["TRIG"],  # No unit
        metadata_key=trigger_channels_metadata_key
    )

    percent_channels_metadata_key = "time_series_oxygen"  # Percentage unit
    percent_interface = EDFAnalogInterface(
        file_path=file_path,
        channels_to_include=["OSAT"],  # Percentage units
        metadata_key=percent_channels_metadata_key,
    )

    # Combine all interfaces
    converter = ConverterPipe(
        data_interfaces=[recording_interface, trigger_interface, percent_interface],

    )

    # Extract metadata and add timezone information
    metadata = converter.get_metadata()
    # For data provenance we add the time zone information to the conversion
    session_start_time = metadata["NWBFile"]["session_start_time"].replace(tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    # Add subject information (required for DANDI upload)
    metadata["Subject"] = dict(subject_id="subject1", species="Mus musculus", sex="M", age="P30D")

    # REQUIRED: Customize TimeSeries names when using multiple analog interfaces
    user_edited_metadata = {
        "TimeSeries": {
            trigger_channels_metadata_key: {
                "name": "TimeSeriesTrigger",
                "description": "Trigger signals from EDF file"
            },
            percent_channels_metadata_key: {
                "name": "TimeSeriesOxygen",
                "description": "Oxygen saturation monitoring data"
            }
        }
    }

    # The metadata_key parameter ensures each interface creates entries with the correct names
    metadata = dict_deep_update(metadata, user_edited_metadata)

    # Convert all channel types to a single NWB file
    nwbfile_path = f"{path_to_save_nwbfile}"
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
