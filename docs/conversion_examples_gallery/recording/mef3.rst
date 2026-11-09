MEF 3 data conversion
---------------------

Install NeuroConv with the additional dependencies necessary for reading MEF 3 data.

.. code-block:: bash

    pip install "neuroconv[mef]"

Convert MEF 3.0 (Multiscale Electrophysiology Format) sessions to NWB using
:py:class:`~neuroconv.datainterfaces.ecephys.mef.mef3datainterface.MEF3RecordingInterface`.

A MEF 3 session is a ``.mefd`` directory holding one ``.timd`` directory per channel. Point the interface at
the ``.mefd`` directory. The files are read by NeuroConv's own reader, which decodes MEF's RED compression
exactly as meflib, the reference implementation, does; neither pymef nor MNE is needed. The ``mef`` extra
installs numba, which compiles the decoder; where numba is not available (Intel macOS) a pure-NumPy decoder is
used, which gives the same samples much more slowly.

.. code-block:: python

    from datetime import datetime
    from zoneinfo import ZoneInfo
    from neuroconv.datainterfaces import MEF3RecordingInterface

    folder_path = "sub-01_ses-01_task-rest_ieeg.mefd"

    # The channel names come from the session's directories, in acquisition channel order
    channel_ids = MEF3RecordingInterface.get_available_channel_ids(folder_path)

    # Leave out channels that are not neural data, such as an EKG
    interface = MEF3RecordingInterface(folder_path=folder_path, channels_to_skip=["EKG"])

    # Extract what metadata we can from the source files
    metadata = interface.get_metadata()
    # De-identified sessions are moved to 1970-01-01, with a warning; set the real date where it is known
    metadata["NWBFile"].update(session_start_time=datetime(2020, 1, 1, 12, 0, tzinfo=ZoneInfo("UTC")))
    # Add subject information (required for DANDI upload)
    metadata["Subject"] = dict(subject_id="subject1", species="Homo sapiens", sex="M", age="P30Y")

    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = "sub-01_ses-01_task-rest_ieeg.nwb"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

The samples are stored as the integer counts the file holds, with each channel's units conversion factor,
in microvolts per count and with its sign, as the series' gain. All channels converted together must share a
sampling rate and the stretches of time they were recorded over; leave the others out with
``channels_to_skip`` and convert them separately. ``MEF3RecordingInterface.get_channel_sampling_frequencies``
gives each channel's rate, reading one metadata file per channel, for a session recorded at several rates.
Skipped channels still count towards the session: its start time is the earliest first block of any channel,
skipped or not, and their records are kept. So one interface per rate, over complementary ``channels_to_skip``
and written through one ``ConverterPipe``, puts every series on one timeline under one ``session_start_time``,
writes the session's records once, and labels each rate's rows of the ``"runs"`` table with the rate, as MEF 2
streams are.

Gaps and timing
^^^^^^^^^^^^^^^

A session recorded with interruptions is written as one ``ElectricalSeries`` per uninterrupted run of
samples, each placed at its first block's start time and sharing one electrodes table, and the runs are
listed in a ``TimeIntervals`` table named ``"runs"``. A run breaks where the file flags a block as a
discontinuity, as meflib, the reference implementation, defines one, with two exceptions decided in whole
samples:

* a flagged block that starts exactly where the previous block ended continues the run, since writers flag
  the first block of every MEF segment whether or not anything is missing, so a session split into several
  segments with nothing missing between them is a single series;
* an unflagged block whose start time is more than a second from where the previous block ended starts a new
  run, with a warning, since no clock drifts that far in one block.

Block timestamps are not otherwise compared with sample counts: writers cut blocks on a fixed grid of times
while the samples in them drift, so a continuous recording's timestamps disagree with its sample counts by a
few samples at many block boundaries.

Each series uses the channel's nominal sampling rate. A recording's clock drifts against it, typically by
tens of milliseconds an hour, so a sample's time in the NWB file can differ from the block timestamps by that
much towards the end of a long run. This is the timing BIDS ``events.tsv`` files assume when they give
onsets as sample counts over the nominal rate.

Records, lossy blocks and encryption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MEF records (``Note``, ``EDFA``, ``Seiz``, ``CSti``, ``ESti``, ``Curs``, ``Epoc``, ...) at session, channel
and segment level are written to a ``TimeIntervals`` table named ``"records"``, with the record's type, its
text in the ``label`` column, its other fields in ``details``, and its channel. Pass ``write_records=False``
as a conversion option to leave them out.

A writer can store RED blocks lossily, scaled by a factor or with a fitted line subtracted. Such blocks are
decoded as meflib decodes them, with a warning saying how many there are and the largest scale factor, since
the precision lost when the file was written cannot be recovered.

A session whose metadata is encrypted is read with its password, passed as ``password``; its level (1 or 2)
is found from the files' password validation fields, and a level 2 password opens both levels. The passwords
can also be given by level, as ``level_1_password`` and ``level_2_password``. A password that does not
validate raises an error naming the file and the level that failed. As in meflib, a password has at most 15
characters and is keyed by the low byte of each character's code point, so non-ASCII passwords match files
meflib wrote. A level 1 password given on its own opens level 1 sections; meflib's own reader does not do this
for an unspecified password, but its writer encrypts level 1 sections with exactly that key. Take the password
from a secret or an environment variable rather than writing it into code:

.. code-block:: python

    import os

    interface = MEF3RecordingInterface(folder_path=folder_path, password=os.environ["MEF3_PASSWORD"])

Section 2 of each metadata file (sampling rate, units conversion factor, descriptions) is decrypted. Of
section 3 only the recording time offset and the GMT offset are decrypted and read; the subject's names, ID and
recording location are never read. The password is left out of the extractor's kwargs, so an encrypted
session's extractor cannot be rebuilt from ``to_dict()``. Encrypted records are left out with a warning, and
encrypted data blocks are refused; decrypt those with the MEF tools first. Decryption needs the
``cryptography`` package, which the ``mef`` extra installs.
