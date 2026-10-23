Blackrock data conversion
-------------------------

Install NeuroConv with the additional dependencies necessary for reading Blackrock data.

.. code-block:: bash

    pip install "neuroconv[blackrock]"

Convert Blackrock data to NWB using
:py:class:`~neuroconv.datainterfaces.ecephys.blackrock.blackrockdatainterface.BlackrockRecordingInterface`.

A Blackrock recording is a file set sharing one base name: one ``.ns1`` to ``.ns6`` file per sampling rate,
holding the continuous data, and a ``.nev`` file holding spikes, digital events and comments. Point the
interface at the ``.nsX`` file to convert; the ``.nev`` beside it is found by its base name, which may contain
dots of its own.

.. code-block:: python

    >>> from neuroconv.datainterfaces import BlackrockRecordingInterface
    >>>
    >>> # For this interface we need to pass the location of the ``.ns5`` file
    >>> file_path = f"{ECEPHY_DATA_PATH}/blackrock/FileSpec2.3001.ns5"
    >>> # Change the file_path to the location in your system
    >>> interface = BlackrockRecordingInterface(file_path=file_path, verbose=False)
    >>>
    >>> # Extract what metadata we can from the source files
    >>> metadata = interface.get_metadata()
    >>> # The start time is the .nsX header's TimeOrigin, which the format defines as UTC
    >>> metadata["NWBFile"]["session_start_time"].isoformat()
    '2014-01-06T20:23:05.559000+00:00'
    >>> # Add subject information (required for DANDI upload)
    >>> metadata["Subject"] = dict(subject_id="subject1", species="Mus musculus", sex="M", age="P30D")
    >>>
    >>> # Choose a path for saving the nwb file and run the conversion
    >>> nwbfile_path = f"{path_to_save_nwbfile}"
    >>> interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

When the ``.nev`` file has another base name, pass it as ``nev_file_path``. The samples are stored as the
int16 counts the file holds, with each channel's gain taken from its header range: 0.25 µV per count for the
neural inputs and 0.1526 mV for the analog inputs of a Cerebus or NeuroPort. A FileSpec 2.1 ``.nsX`` header
holds no scaling and no start time; both are read from its ``.nev``, and the file is refused without one.

Timing
^^^^^^

The first sample is at 0 and ``session_start_time`` is the file's ``TimeOrigin``, so the onsets of a BIDS
``events.tsv``, which count from a data file's first sample, line up with the data. Older files' clocks start
with the recording, but FileSpec 3.0 clocks keep counting from when the processor started, and PTP (Gemini)
clocks count nanoseconds of the Unix epoch, which are not UTC either; neither is written as it is.

* A recording paused and resumed is written as one ``ElectricalSeries`` per run, each at its start time, sharing
  one electrodes table, and the runs are listed in a ``TimeIntervals`` table named ``"runs"``.
* A clock reset (Central's ResetCorrect) restarts the clock near zero, so the file does not say when what
  follows was recorded. The run after a reset is placed where the run before it ends, plus its own clock value
  since the reset, which is the earliest it can have started; the ``runs`` table marks it in its
  ``after_clock_reset`` column, and a warning says so.
* PTP files stamp every sample. Their packets are read in timestamp order, since some files store blocks of
  them out of order, and a new run starts wherever a sample is missing (a gap of more than 1.5 sample periods).
  Each run is written at the nominal sampling rate from its first sample.

Events
^^^^^^

The ``.nev`` file's digital and serial input events and its comments are written to a ``TimeIntervals`` table
named ``"nev_events"``, each spanning one sample, with the input word or the comment's text in its ``label``
column. They are placed on the same timeline as the samples, through the samples they fall between for a PTP
file, so they stay on the sample they were recorded with however far its clock drifts from the nominal rate.
Several recordings written into one NWB file share the ``nev_events`` and ``runs`` tables, with a
``source_file`` column. Pass ``write_nev_events=False`` or ``write_runs=False`` to ``run_conversion`` to leave
them out. Spikes are converted with
:py:class:`~neuroconv.datainterfaces.ecephys.blackrock.blackrockdatainterface.BlackrockSortingInterface`, which
places them on the same timeline.

Ripple Neuro records data in this format as well.
