MEF 2 data conversion
---------------------

Install NeuroConv with the additional dependencies necessary for reading MEF 2 data.

.. code-block:: bash

    pip install "neuroconv[mef]"

Convert MEF 2.0 and 2.1 (Multiscale Electrophysiology Format) recordings to NWB using
:py:class:`~neuroconv.datainterfaces.ecephys.mef.mef2datainterface.MEF2RecordingInterface`.

A MEF 2 recording is a directory of ``.mef`` files, one per channel, as published by IEEG.org and Pennsieve
Discover. Point the interface at the directory. The files are read by NeuroConv's own reader, which decodes
MEF's RED compression as Mayo's mef_lib, the reference implementation, does; no other MEF library is needed.
The ``mef`` extra installs numba, which compiles the decoder; where numba is not available (Intel macOS) a
pure-NumPy decoder is used, which gives the same samples much more slowly.

.. code-block:: python

    from datetime import datetime
    from zoneinfo import ZoneInfo
    from neuroconv.datainterfaces import MEF2RecordingInterface

    folder_path = "sub-01_ses-postimplant_task-clinical_ieeg-mef"

    # One stream per sampling rate; a directory whose files share one rate has a single stream
    stream_names = MEF2RecordingInterface.get_stream_names(folder_path)

    # Leave out channels that are not neural data, such as the EKG
    interface = MEF2RecordingInterface(folder_path, stream_name=stream_names[0], channels_to_skip=["EKG1", "EKG2"])

    # Extract what metadata we can from the source files
    metadata = interface.get_metadata()
    # De-identified recordings are moved to 2000-01-01, with a warning; set the real date where it is known
    metadata["NWBFile"].update(session_start_time=datetime(2020, 1, 1, 12, 0, tzinfo=ZoneInfo("UTC")))
    # Add subject information (required for DANDI upload)
    metadata["Subject"] = dict(subject_id="subject1", species="Homo sapiens", sex="M", age="P30Y")

    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = "sub-01_ses-postimplant_task-clinical_ieeg.nwb"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

The samples are stored as the integer counts the files hold, with each channel's voltage conversion factor,
in microvolts per count, as the series' gain. The factor keeps its sign: MEF 2 marks inverted counts with a
negative factor, and PennEPI's recordings have one, so every recording comes out in microvolts with the same
polarity. All channels of a stream must share the stretches of time they were recorded over; leave the others
out with ``channels_to_skip`` and convert them separately.

Channels sampled at different rates are converted one stream at a time, with ``stream_name``. To write
several streams to one NWB file, give each interface its own ``metadata_key``, and give its series its own
name under that key; otherwise the interfaces collide on the default ``ElectricalSeries`` name:

.. code-block:: python

    from neuroconv import ConverterPipe

    interfaces = []
    for stream_name in MEF2RecordingInterface.get_stream_names(folder_path):
        metadata_key = "mef2_" + stream_name.replace(" ", "_")  # such as "mef2_512_Hz"
        interfaces.append(MEF2RecordingInterface(folder_path, stream_name=stream_name, metadata_key=metadata_key))

    converter = ConverterPipe(data_interfaces=interfaces)
    metadata = converter.get_metadata()
    for interface in interfaces:
        series_name = "ElectricalSeries_" + interface.metadata_key.removeprefix("mef2_")
        metadata["Ecephys"]["ElectricalSeries"][interface.metadata_key]["name"] = series_name
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

The streams share the recording's timeline, and each writes its own rows to the ``runs`` table described
below, with the stream in the ``source_file`` column, such as ``"recording (512 Hz)"``.

The header's subject fields (names, ID, anonymized name), institution and comments are never read, so they
cannot reach the NWB file; set ``metadata["Subject"]`` yourself.

Gaps and timing
^^^^^^^^^^^^^^^

A recording with interruptions is written as one ``ElectricalSeries`` per uninterrupted run of samples, each
placed at its first block's start time and sharing one electrodes table, and the runs are listed in a
``TimeIntervals`` table named ``"runs"``. As for MEF 3, a run breaks where the file flags a block as following
a discontinuity, in MEF 2.1's discontinuity index or, in MEF 2.0, in the block's own header, with these
exceptions:

* a flagged block that starts exactly where the previous block ended, in whole samples, continues the run;
* an unflagged block whose start time is more than a second after the previous block ended starts a new run,
  with a warning, since the writer left a gap unflagged;
* an unflagged block whose start time is more than a second *before* the previous block ended continues the
  run, with a warning. MEF 2 writers set their clock back in such steps while the samples run on unbroken.

Some recordings drop samples often: one PennEPI channel has 959 runs, most of a single block of a few hundred
samples. Each is a series of its own, so that no sample is placed at a time it was not recorded at.

Each series uses the channel's nominal sampling rate, while the block timestamps follow the acquisition
system's clock, which drifts against it. In one Mayo recording that clock ran 50 ppm slow, so towards the end
of a day-long run a sample's time in the NWB file is about 4 s earlier than its block's timestamp.

The session starts at the header's recording start time, which IEEG.org's annotation times count from, so
the NWB times of the series equal IEEG.org's offsets. The one-sample block of value 0 that IEEG.org's writer
puts before the first real block of every MEF 2.1 file is left out, so in those files the first series
starts a couple of seconds into the session.

Annotations, CRCs and encryption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An IEEG.org ``annotations.xml`` in the directory, or given as ``annotations_file_path``, is written to a
``TimeIntervals`` table named ``"annotations"``, with the annotation's type in the ``label`` column and its
layer, annotator and channels. Its times are placed through the block index on the samples they mark, which is
where IEEG.org's viewer showed them. Pass ``write_annotations=False`` as a conversion option to leave them out.

Pass ``validate_crc=True`` to check every block's CRC as it is decoded, at a cost of a few percent of decoding
time; a MEF 2.1 header whose CRC fails then raises instead of warning.

Files with session or data encryption are refused; decrypt them with the MEF tools first. MEF 2.2, whose
timestamps may be shifted by an offset kept in the encrypted part of the header, is not supported.
