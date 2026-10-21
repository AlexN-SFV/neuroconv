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

    >>> from datetime import datetime
    >>> from zoneinfo import ZoneInfo
    >>> from neuroconv.datainterfaces import BrainVisionRecordingInterface
    >>>
    >>> file_path = f"{ECEPHY_DATA_PATH}/brainvision/File_brainvision_4_float32.vhdr"
    >>> interface = BrainVisionRecordingInterface(file_path=file_path)
    >>>
    >>> # Extract what metadata we can from the source files
    >>> metadata = interface.get_metadata()
    >>> # The recording time comes from the marker file's "New Segment" marker, when it carries a date.
    >>> # Anonymised recordings often do not, and this one has no markers at all, so set it if it is missing.
    >>> if "session_start_time" not in metadata["NWBFile"]:
    ...     metadata["NWBFile"].update(session_start_time=datetime(2020, 1, 1, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo")))
    >>> # Add subject information (required for DANDI upload)
    >>> metadata["Subject"] = dict(subject_id="subject1", species="Homo sapiens", sex="M", age="P30Y")
    >>>
    >>> # Choose a path for saving the nwb file and run the conversion
    >>> nwbfile_path = f"{path_to_save_nwbfile}"
    >>> interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

Each channel's resolution and unit from the header become its gain, so ``INT_16``, ``INT_32`` and
``IEEE_FLOAT_32`` data are all written with their physical scale. Channels that are not neural data,
such as a force sensor recorded alongside the EEG, can be left out by name:

.. code-block:: python

    >>> # The names come from the header alone, without reading any samples
    >>> channel_ids = BrainVisionRecordingInterface.get_available_channel_ids(file_path)
    >>> # Here the last channel stands in for an auxiliary one
    >>> interface = BrainVisionRecordingInterface(file_path=file_path, channels_to_skip=channel_ids[-1:])
    >>> len(interface.recording_extractor.get_channel_ids()) == len(channel_ids) - 1
    True

A header that gives two channels the same name has the later one renamed (``Fp1_2``), with a warning,
because channel ids must be unique; the names as the header gives them are written to an
``original_channel_name`` electrodes column. In ``channels_to_skip``, a renamed id skips that channel
alone, and a repeated name skips every channel carrying it.

Markers and segments
^^^^^^^^^^^^^^^^^^^^

The markers in the ``.vmrk`` file are written to a ``TimeIntervals`` table named ``"markers"``, with
the marker's type, its description (in the ``label`` column), its size in samples, and the channel it
refers to. Pass ``write_markers=False`` as a conversion option to leave them out. A marker positioned
outside the recorded samples has no time in the recording and is left out, with a warning. When several
BrainVision recordings are written into one NWB file, as through a ``ConverterPipe``, they share the
table, and its ``source_file`` column names the ``.vhdr`` each marker came from. Interfaces over the
same recording, such as one per channel type built with complementary ``channels_to_skip``, write its
markers only once.

A recording that was paused and resumed carries a ``New Segment`` marker at each resumption. Each run
of samples between them is written as its own ``ElectricalSeries``, placed at the time its marker
states and sharing one electrodes table, and the runs are listed in a ``TimeIntervals`` table named
``"segments"``. A recording that was never paused is written as a single ``ElectricalSeries``.

The recording's start time, ``session_start_time``, is the date on the ``New Segment`` marker at the first
sample. A later segment's date places that segment relative to the dated segment before it; if the first
segment has no date, later dates have nothing to be measured from, so those pauses cannot be recovered and
a warning says so.

Epoched exports
^^^^^^^^^^^^^^^

BrainVision Analyzer can export epochs cut around events (``SegmentationType=MARKERBASED`` in the header).
Such a file holds equal-length epochs back to back, each opened by a ``New Segment`` marker dated with the
epoch's time in the original recording. Each epoch is written as its own ``ElectricalSeries`` at that
time, as for a paused recording, and the ``"segments"`` table marks them with ``is_epoch``. Epochs cut from
overlapping windows overlap in time, which separate series can do. A file whose markers do not open
exactly ``SegmentDataPoints``-long epochs is refused rather than split at the wrong places.

Only binary time-domain data is supported. ASCII data, averages over epochs (``Averaged=YES``) and other
Analyzer segmentation types are refused with an error.
