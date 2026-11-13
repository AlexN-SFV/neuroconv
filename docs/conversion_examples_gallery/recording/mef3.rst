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
in microvolts per count and with its sign, as the series' gain.

An ``ElectricalSeries`` has one rate and one set of runs, so the channels are read in groups that share a
sampling rate and the stretches of time they were recorded over, and each group is written as its own series
into the one NWB file, on the session's timeline: a session of iEEG at 32 kHz beside eye-tracking channels at
150 Hz, or of channels with gaps beside a continuously recorded telemetry channel, converts as it is. With
several groups, each series is named ``ElectricalSeries_<rate>Hz``, with ``_<n>`` after it when several groups
share a rate (``ElectricalSeries_1000Hz_0``, ``ElectricalSeries_1000Hz_1``), and a run's index after that when
the group has gaps. Channels at one rate that fall into several groups are warned about.
``interface.split_by_channel_group()`` gives one interface per group, each with its own ``metadata_key``, to
write through a ``ConverterPipe`` or align on their own; ``split_by_offset()`` gives the same groups. A whole
session can be aligned with ``set_aligned_starting_time``; aligning timestamps or single runs needs one group.

Channels can still be converted separately, leaving the others out with ``channels_to_skip``.
``MEF3RecordingInterface.get_channel_sampling_frequencies`` gives each channel's rate, reading one metadata
file per channel. Skipped channels still count towards the session: its start time is the earliest first block of any channel,
skipped or not, and their records are kept. So one interface per rate, over complementary ``channels_to_skip``
and written through one ``ConverterPipe``, puts every series on one timeline under one ``session_start_time``,
writes the session's records once, and labels each rate's rows of the ``"runs"`` table with the rate, as MEF 2
streams are.

A writer may create a channel and never write a sample to it. Such a channel is refused, and with it the session,
unless it is left out: ``MEF3RecordingInterface.get_empty_channel_ids`` lists the channels whose block indices hold
no samples, reading nothing else and needing no password, to pass to ``channels_to_skip``. Like any skipped
channel, an empty one keeps its records. A channel whose index cannot be read is not listed: the call raises,
naming the file.

A segment directory that holds data but is missing its metadata (``.tmet``), index (``.tidx``) or data
(``.tdat``) file cannot be read. It is left out with a warning, and the rest of its channel is kept;
``MEF3RecordingInterface.get_incomplete_segments`` lists such segments and the files they lack. A channel none of
whose samples can then be read is refused like an empty one: ``MEF3RecordingInterface.get_unreadable_channel_ids``
lists them, reading directories and block indices only, to pass to ``channels_to_skip``. A skipped channel with no
metadata file has no rate, so pass the same list to ``get_channel_sampling_frequencies``.

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

A recording that drops a few samples every so often can have thousands of short runs, and thousands of series
take far longer to write, and more room, than the samples in them. ``concatenate_runs_above=n`` writes a channel
group with more than ``n`` runs as one ``ElectricalSeries`` with timestamps instead: the runs back to back, each
at the nominal rate from its own start time, so the gaps are in the timestamps. The ``"runs"`` table lists the
runs either way. Groups with ``n`` runs or fewer are still written one series per run.

A recorder can also drop a few samples on each channel independently, so channels that share nearly every sample
time still differ in their runs and are read as a group each, every one its own series. ``merge_groups_max_fill=f``
writes groups at one rate that share one sample clock as one series instead, over every time any of them has a
sample, as long as no channel then has no sample for more than the fraction ``f`` of the series. A channel's
samples where it has none are written as 0, and every such stretch is listed, with its channel and its number of
samples, in the file's ``invalid_times`` table. A continuous channel beside channels with long gaps stays apart,
since those channels would be missing too much of its time, and so do channels whose runs do not start a whole
number of samples apart.

A segment with no data file and no block in its index, one a writer opened and never wrote to, is left out
with a warning. A segment whose index lists blocks but which has no data file is refused.

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

Some writers store their records offset by the session's start rather than by the recording time offset the
blocks are stored against. Read the MEF 3 way, such records would all fall far outside the session. A record
file whose times all fit the session when read from its start, and none of which fits it when read with the
recording time offset (by more than the session's length), is read from the start, with a warning saying how
many records were. Any other file is read as stored. A record time stored as 0 is the time its file counts
from (meflib stores a time equal to the recording time offset as 0), and a record that states no time
(``UUTC_NO_ENTRY``) is left out of the table with a warning.

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
