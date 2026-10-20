BrainVision data conversion
---------------------------

Install NeuroConv with the additional dependencies necessary for reading BrainVision data.

.. code-block:: bash

    pip install "neuroconv[brainvision]"

Convert BrainVision Core Data Format recordings to NWB using
:py:class:`~neuroconv.datainterfaces.ecephys.brainvision.brainvisiondatainterface.BrainVisionRecordingInterface`.

A BrainVision recording is three files sharing a base name: a ``.vhdr`` text header, a ``.vmrk`` text
marker file and a ``.eeg`` binary data file. Point the interface at the ``.vhdr``; the other two are read
from beside it. If the files were renamed without updating the names the header gives for them, as BIDS
conversion commonly leaves them, they are found by the header's own base name instead.

.. code-block:: python

    from datetime import datetime
    from zoneinfo import ZoneInfo
    from neuroconv.datainterfaces import BrainVisionRecordingInterface

    file_path = f"{ECEPHY_DATA_PATH}/brainvision/File_brainvision_1.vhdr"
    interface = BrainVisionRecordingInterface(file_path=file_path)

    # Extract what metadata we can from the source files
    metadata = interface.get_metadata()
    # The recording time comes from the marker file's "New Segment" marker, when it carries a date.
    # Anonymised recordings often do not, so set it if it is missing.
    if "session_start_time" not in metadata["NWBFile"]:
        metadata["NWBFile"].update(session_start_time=datetime(2020, 1, 1, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo")))
    # Add subject information (required for DANDI upload)
    metadata["Subject"] = dict(subject_id="subject1", species="Homo sapiens", sex="M", age="P30Y")

    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = f"{path_to_save_nwbfile}"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

Each channel's resolution and unit from the header become its gain, so ``INT_16``, ``INT_32`` and
``IEEE_FLOAT_32`` data are all written with their physical scale. Channels that are not neural data,
such as a force sensor recorded alongside the EEG, can be left out by name:

.. code-block:: python

    # The names come from the header alone, without reading any samples
    channel_ids = BrainVisionRecordingInterface.get_available_channel_ids(file_path)
    # Here the last channel stands in for an auxiliary one
    interface = BrainVisionRecordingInterface(file_path=file_path, channels_to_skip=channel_ids[-1:])

Markers and segments
^^^^^^^^^^^^^^^^^^^^

The markers in the ``.vmrk`` file are written to a ``TimeIntervals`` table named ``"markers"``, with
the marker's type, its description (in the ``label`` column), its size in samples, and the channel it
refers to. Pass ``write_markers=False`` as a conversion option to leave them out. A marker positioned
outside the recorded samples has no time in the recording and is left out, with a warning. When several
BrainVision recordings are written into one NWB file, as through a ``ConverterPipe``, they share the
table, and its ``source_file`` column names the ``.vhdr`` each marker came from.

A recording that was paused and resumed carries a ``New Segment`` marker at each resumption. Each run
of samples between them is written as its own ``ElectricalSeries``, placed at the time its marker
states and sharing one electrodes table, and the runs are listed in a ``TimeIntervals`` table named
``"segments"``. A recording that was never paused is written as a single ``ElectricalSeries``.

The recording's start time, ``session_start_time``, is the date on the ``New Segment`` marker at the first
sample. A later segment's date places that segment relative to the dated segment before it; if the first
segment has no date, later dates have nothing to be measured from, so those pauses cannot be recovered and
a warning says so.

Only binary, continuous (unsegmented) time-domain data is supported. ASCII data and data segmented
by BrainVision Analyzer are refused with an error.
