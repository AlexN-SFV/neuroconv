BioSemi Data Format (BDF) conversion
------------------------------------

Install NeuroConv with the additional dependencies necessary for reading BDF data.

.. code-block:: bash

    pip install "neuroconv[bdf]"

Convert BioSemi BDF data to NWB using
:py:class:`~neuroconv.datainterfaces.ecephys.bdf.bdfdatainterface.BDFRecordingInterface`.

BDF is BioSemi's variant of :doc:`EDF <edf>`, but with 24-bit samples instead of 16-bit. The interface
reads the file with ``pyedflib`` directly (which handles the 24-bit data correctly) rather than through
the EDF/neo reader, and writes the signal as an ``ElectricalSeries``. Non-neural channels — such as the
BioSemi ``Status`` trigger channel, or any channel sampled at a different rate — can be excluded with
``channels_to_skip``.

.. code-block:: python

    from datetime import datetime
    from zoneinfo import ZoneInfo
    from pathlib import Path
    from neuroconv.datainterfaces import BDFRecordingInterface

    file_path = f"{ECEPHY_DATA_PATH}/bdf/recording.bdf"

    # List channels to identify any non-neural ones to skip (e.g. the Status/trigger channel)
    all_channels = BDFRecordingInterface.get_available_channel_ids(file_path)
    print(f"Available channels: {all_channels}")

    interface = BDFRecordingInterface(file_path=file_path, channels_to_skip=["Status"])

    # Extract what metadata we can from the source file
    metadata = interface.get_metadata()
    # For data provenance we add the time zone information to the conversion
    session_start_time = metadata["NWBFile"]["session_start_time"].replace(tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    # Add subject information (required for DANDI upload)
    metadata["Subject"] = dict(subject_id="subject1", species="Homo sapiens", sex="M", age="P30Y")

    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = f"{path_to_save_nwbfile}"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
