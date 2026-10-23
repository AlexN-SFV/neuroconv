Blackrock sorting data conversion
---------------------------------

Install NeuroConv with the additional dependencies necessary for reading Blackrock sorting data.

.. code-block:: bash

    pip install "neuroconv[blackrock]"

Convert Blackrock sorting data to NWB using
:py:class:`~neuroconv.datainterfaces.ecephys.blackrock.blackrockdatainterface.BlackrockSortingInterface`.

.. code-block:: python

    >>> from neuroconv.datainterfaces import BlackrockSortingInterface
    >>>
    >>> file_path = f"{ECEPHY_DATA_PATH}/blackrock/FileSpec2.3001.nev"
    >>> # Change the file_path to the location of the file in your system
    >>> interface = BlackrockSortingInterface(file_path=file_path, verbose=False)
    >>>
    >>> # Extract what metadata we can from the source files
    >>> metadata = interface.get_metadata()
    >>> # The start time is the TimeOrigin of the .ns5 recorded with the .nev, which is UTC
    >>> metadata["NWBFile"]["session_start_time"].isoformat()
    '2014-01-06T20:23:05.559000+00:00'
    >>> # Add subject information (required for DANDI upload)
    >>> metadata["Subject"] = dict(subject_id="subject1", species="Mus musculus", sex="M", age="P30D")
    >>>
    >>> nwbfile_path = f"{path_to_save_nwbfile}" # This should be something like: "./saved_file.nwb"
    >>> interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

The spike times are placed on the timeline
:py:class:`~neuroconv.datainterfaces.ecephys.blackrock.blackrockdatainterface.BlackrockRecordingInterface`
gives the ``.nsX`` files of the same base name, so spikes and samples line up in one NWB file: the set's earliest
first sample at 0, runs after a clock reset moved past the run before, PTP spikes on the samples they were
recorded with. Every stream of a set shares that origin, so any of them gives the spikes the same times; of
several, the smallest file is read. For a PTP set, pass the stream you convert as ``nsx_to_load`` (``6`` for the
``.ns6``) to anchor the spikes to its own samples; ``nsx_to_load=[]`` keeps the ``.nev`` file's own clock.
