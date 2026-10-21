"""Tests for the rule the EEG readers share for turning channel labels into unique channel ids."""

from neuroconv.datainterfaces.ecephys._channel_ids import (
    indices_to_keep,
    unique_channel_ids,
    unknown_channels_to_skip,
)


def test_unique_labels_are_their_own_ids():
    assert unique_channel_ids(["Fz", "Cz", "Pz"]) == ["Fz", "Cz", "Pz"]


def test_repeats_are_numbered_from_two_in_file_order():
    assert unique_channel_ids(["a", "b", "a", "a"]) == ["a", "b", "a_2", "a_3"]


def test_a_new_id_avoids_every_existing_label():
    """Including a label that comes later in the file, which must not have its name taken first."""
    assert unique_channel_ids(["a", "a", "a_2"]) == ["a", "a_3", "a_2"]


def test_groups_need_only_be_unique_within_themselves():
    assert unique_channel_ids(["a", "a", "a"], groups=[256, 512, 256]) == ["a", "a", "a_2"]


def test_a_new_id_avoids_labels_in_other_groups():
    """Two channels of one name in the electrodes table would merge, whichever stream each came from."""
    assert unique_channel_ids(["a", "a", "a_2"], groups=[1, 1, 2]) == ["a", "a_3", "a_2"]


def test_skipping_matches_a_label_or_an_id():
    labels, ids = ["a", "a", "b"], ["a", "a_2", "b"]
    assert indices_to_keep(labels=labels, ids=ids, channels_to_skip=["a"]) == [2]
    assert indices_to_keep(labels=labels, ids=ids, channels_to_skip=["a_2"]) == [0, 2]
    assert indices_to_keep(labels=labels, ids=ids, channels_to_skip=None) == [0, 1, 2]
    assert unknown_channels_to_skip(labels=labels, ids=ids, channels_to_skip=["a_2", "c"]) == ["c"]
