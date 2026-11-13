"""Tests for reading MEF 3 sessions whose metadata is encrypted.

The encrypted sessions are built here from the synthetic fixtures in ``mef3_fixtures/``: each metadata file
of a copy gets password validation fields for known test passwords and has its sections encrypted the way
meflib encrypts them. ``encrypted.mefd``, which pymef (meflib) itself wrote with the passwords "chair" (level
1) and "table" (level 2), checks the scheme against the reference implementation.
"""

import hashlib
import pickle
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("cryptography")

from test_mef3_interface import read_quietly, remove_segment_file  # noqa: E402

from neuroconv.datainterfaces import MEF3RecordingInterface  # noqa: E402
from neuroconv.datainterfaces.ecephys.mef._mef3_crypto import (  # noqa: E402
    MEF3Passwords,
    aes_ecb_encrypt,
    password_bytes,
)
from neuroconv.datainterfaces.ecephys.mef._mef3_extractor import (  # noqa: E402
    MEF3RecordingExtractor,
)
from neuroconv.datainterfaces.ecephys.mef._mef3_reader import (  # noqa: E402
    read_mef3_session,
)

FIXTURES = Path(__file__).parent / "mef3_fixtures"
GAPPED = FIXTURES / "gapped.mefd"
PYMEF_ENCRYPTED = FIXTURES / "encrypted.mefd"
# Written by meflib with "Straße" (level 1) and "clé-niveau-2" (level 2); see build_mef3_fixtures.py.
PYMEF_NON_ASCII = FIXTURES / "encrypted_non_ascii.mefd"
START_TIME = 946684800000000  # the µUTC start of the synthetic sessions, 2000-01-01T00:00:00 UTC
RECORDING_TIME_OFFSET = START_TIME - 1_000_000  # what build_mef3_fixtures.py gives every session

# Test passwords only, at most 15 characters as meflib requires. The second has a character beyond ASCII, of
# which meflib keeps the low byte of the code point.
LEVEL_1 = "level-one-test"
LEVEL_2 = "Stufe-ß-zwei"

# The fields of metadata section 3 (METADATA_*_OFFSET in meflib.h).
DST_START, DST_END = 13320, 13328
SUBJECT_FIELDS = {"subject_name_1": (13340, 128), "subject_name_2": (13468, 128), "subject_ID": (13596, 128)}
RECORDING_LOCATION = (13724, 512)
# Some writers fill both DST fields with this: 2**47 µs, a misplaced "no entry" that reads as 1974-06-17.
MALFORMED_DST_NO_ENTRY = bytes.fromhex("0000000000800000")
# Markers written into the subject fields; none of them, nor the first four bytes of subject_name_1 (which share
# an AES block with the GMT offset), may reach anything the reader returns.
MARKERS = {
    "subject_name_1": b"QXSUBJNAMEONEQX",
    "subject_name_2": b"QXSUBJNAMETWOQX",
    "subject_ID": b"QXSUBJIDQX",
    "recording_location": b"QXRECLOCATIONQX",
}


def _tmet_paths(session_path: Path) -> list[Path]:
    return sorted(session_path.glob("*.timd/*.segd/*.tmet"))


def encrypt_session(
    session_path: Path,
    *,
    level_1_password: str,
    level_2_password: str | None = None,
    section_2_level: int = 1,
    section_3_level: int = 2,
    edit=None,
):
    """
    Encrypt every metadata file of a session in place, as meflib's ``write_MEF_file`` does.

    ``edit(channel_name, raw)`` may change a file's plaintext first. The CRCs of the universal header are left
    as they are; neither this reader nor meflib's reader in this mode checks them.
    """
    level_1_key = password_bytes(level_1_password)
    level_2_key = None if level_2_password is None else password_bytes(level_2_password)
    keys = {1: level_1_key, 2: level_2_key}
    for path in _tmet_paths(session_path):
        raw = bytearray(path.read_bytes())
        if edit is not None:
            edit(path.parent.parent.name[: -len(".timd")], raw)
        raw[868:884] = hashlib.sha256(level_1_key).digest()[:16]
        if level_2_key is not None:
            raw[884:900] = bytes(a ^ b for a, b in zip(hashlib.sha256(level_2_key).digest()[:16], level_1_key))
        if section_2_level:
            raw[2560:13312] = aes_ecb_encrypt(keys[section_2_level], bytes(raw[2560:13312]))
        if section_3_level:
            raw[13312:16384] = aes_ecb_encrypt(keys[section_3_level], bytes(raw[13312:16384]))
        raw[1024] = section_2_level
        raw[1025] = section_3_level
        path.write_bytes(bytes(raw))


def write_subject_fields(channel_name: str, raw: bytearray):
    for name, (offset, length) in SUBJECT_FIELDS.items():
        raw[offset : offset + length] = MARKERS[name].ljust(length, b"\0")
    offset, length = RECORDING_LOCATION
    raw[offset : offset + length] = MARKERS["recording_location"].ljust(length, b"\0")
    raw[DST_START : DST_START + 8] = MALFORMED_DST_NO_ENTRY
    raw[DST_END : DST_END + 8] = MALFORMED_DST_NO_ENTRY


def summary(session) -> dict:
    """Everything the reader returns about a session, in a form two reads can be compared by."""
    return dict(
        sampling_frequency=session.sampling_frequency,
        gmt_offset=session.gmt_offset,
        start_time=session.start_time,
        runs=session.runs,
        records=session.records,
        lossy_blocks=session.lossy_blocks,
        channels=[
            dict(
                name=channel.name,
                sampling_frequency=channel.sampling_frequency,
                units_conversion_factor=channel.units_conversion_factor,
                units_description=channel.units_description,
                acquisition_channel_number=channel.acquisition_channel_number,
                reference_description=channel.reference_description,
                blocks=channel.blocks.tobytes(),
                samples=channel.read_blocks(0, len(channel.blocks)).tobytes(),
            )
            for channel in session.channels
        ],
    )


@pytest.fixture(scope="module")
def plaintext():
    return summary(read_quietly(GAPPED))


@pytest.fixture
def gapped_copy(tmp_path):
    destination = tmp_path / "gapped.mefd"
    shutil.copytree(GAPPED, destination)
    return destination


class TestPasswords:
    def test_password_bytes_follow_meflib(self):
        """The low byte of each character's code point, zero-padded to 16, as extract_terminal_password_bytes keys it."""
        assert password_bytes("abc") == b"abc" + b"\0" * 13
        assert password_bytes("ß") == bytes([0xDF]) + b"\0" * 15  # U+00DF, not 0x9F, the last byte of C3 9F
        assert password_bytes("é") == bytes([0xE9]) + b"\0" * 15  # U+00E9, not 0xA9
        assert password_bytes("x" * 15) == b"x" * 15 + b"\0"

    @pytest.mark.parametrize("name", ["password", "level_1_password", "level_2_password"])
    def test_more_than_15_characters_are_refused(self, name):
        """meflib's check_password allows MAX_PASSWORD_CHARACTERS = 15; its 16th key byte is always zero."""
        secret = "sixteen-chars-xx"
        with pytest.raises(ValueError, match=f"{name} is longer than 15 characters") as error:
            MEF3Passwords(**{name: secret})
        assert secret not in str(error.value)
        with pytest.raises(ValueError, match="at most 15 characters"):
            password_bytes(secret)
        MEF3Passwords(**{name: "fifteen-chars-x"})

    def test_repr_hides_the_passwords(self):
        for passwords in (
            MEF3Passwords(password=LEVEL_2),
            MEF3Passwords(level_1_password=LEVEL_1, level_2_password=LEVEL_2),
        ):
            assert LEVEL_1 not in repr(passwords) and LEVEL_2 not in repr(passwords)
            assert LEVEL_1 not in str(passwords) and LEVEL_2 not in str(passwords)

    def test_both_ways_at_once_are_refused(self):
        with pytest.raises(ValueError, match="not both ways") as error:
            MEF3Passwords(password=LEVEL_2, level_1_password=LEVEL_1)
        assert LEVEL_1 not in str(error.value) and LEVEL_2 not in str(error.value)

    def test_nothing_given_is_none(self):
        assert MEF3Passwords.from_arguments() is None


class TestPymefWrittenSession:
    """encrypted.mefd was written by meflib itself: section 2 at level 1 ("chair"), section 3 at level 2 ("table")."""

    @pytest.mark.parametrize(
        "passwords",
        [
            pytest.param(MEF3Passwords(password="table"), id="level-2-password"),
            pytest.param(MEF3Passwords(level_1_password="chair", level_2_password="table"), id="pair"),
            pytest.param(MEF3Passwords(level_2_password="table"), id="level-2-by-name"),
        ],
    )
    def test_reads_as_written(self, passwords):
        session = read_quietly(PYMEF_ENCRYPTED, passwords=passwords)
        (channel,) = session.channels
        assert session.sampling_frequency == 1000
        assert channel.units_conversion_factor == 1.0 and channel.units_description == "microvolts"
        assert session.start_time == START_TIME
        assert session.gmt_offset == 3600
        np.testing.assert_array_equal(channel.read_blocks(0, len(channel.blocks)), np.arange(500, dtype=np.int32))

    def test_its_level_1_password_cannot_open_section_3(self):
        with pytest.raises(
            ValueError, match="encrypted at level 2, and the MEF 3 password given is only good for level 1"
        ):
            read_quietly(PYMEF_ENCRYPTED, passwords=MEF3Passwords(password="chair"))


class TestPymefWrittenNonASCIIPasswords:
    """encrypted_non_ascii.mefd: meflib keyed it by code-point low bytes, which only non-ASCII passwords show."""

    @pytest.mark.parametrize(
        "passwords",
        [
            pytest.param(MEF3Passwords(password="clé-niveau-2"), id="level-2-password"),
            pytest.param(MEF3Passwords(level_1_password="Straße", level_2_password="clé-niveau-2"), id="pair"),
        ],
    )
    def test_reads_as_written(self, passwords):
        session = read_quietly(PYMEF_NON_ASCII, passwords=passwords)
        (channel,) = session.channels
        assert session.sampling_frequency == 1000 and channel.units_conversion_factor == 1.0
        assert session.start_time == START_TIME and session.gmt_offset == 3600
        np.testing.assert_array_equal(channel.read_blocks(0, len(channel.blocks)), np.arange(500, dtype=np.int32))

    def test_its_level_1_password_validates_at_level_1(self):
        with pytest.raises(ValueError, match="only good for level 1"):
            read_quietly(PYMEF_NON_ASCII, passwords=MEF3Passwords(password="Straße"))

    def test_last_utf8_byte_keys_are_refused(self):
        """The same characters keyed by the last byte of their UTF-8 encoding: 0x9F for ß, 0xA9 for é."""
        last_utf8_bytes = "".join(chr(character.encode("utf-8")[-1]) for character in "clé-niveau-2")
        with pytest.raises(ValueError, match="matches neither"):
            read_quietly(PYMEF_NON_ASCII, passwords=MEF3Passwords(password=last_utf8_bytes))


class TestEncryptedSessions:
    def test_level_1_only(self, gapped_copy, plaintext):
        """Section 2 at level 1, section 3 in the clear: the level 1 password is enough."""
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, section_3_level=0)
        for passwords in (MEF3Passwords(password=LEVEL_1), MEF3Passwords(level_1_password=LEVEL_1)):
            assert summary(read_quietly(gapped_copy, passwords=passwords)) == plaintext

    def test_level_2(self, gapped_copy, plaintext):
        """Section 2 at level 1 and section 3 at level 2, as meflib writes them: the level 2 password opens both."""
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        for passwords in (
            MEF3Passwords(password=LEVEL_2),
            MEF3Passwords(level_2_password=LEVEL_2),
            MEF3Passwords(level_1_password=LEVEL_1, level_2_password=LEVEL_2),
        ):
            assert summary(read_quietly(gapped_copy, passwords=passwords)) == plaintext

    def test_both_sections_at_level_2(self, gapped_copy, plaintext):
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2, section_2_level=2)
        assert summary(read_quietly(gapped_copy, passwords=MEF3Passwords(password=LEVEL_2))) == plaintext

    def test_level_1_password_for_a_level_2_section(self, gapped_copy):
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        with pytest.raises(
            ValueError, match=r"encrypted at level 2.*only good for level 1\. Give its level 2"
        ) as error:
            read_quietly(gapped_copy, passwords=MEF3Passwords(password=LEVEL_1))
        assert LEVEL_1 not in str(error.value)

    @pytest.mark.parametrize(
        "passwords, message",
        [
            pytest.param(
                MEF3Passwords(password="wrong-password"),
                r"does not open .*: it matches neither the file's level 1 nor its level 2 .* encrypted at level 1 and 2",
                id="single",
            ),
            pytest.param(
                MEF3Passwords(level_1_password="wrong-password", level_2_password=LEVEL_2),
                "level 1 password does not match the level 1 password validation field",
                id="wrong-level-1",
            ),
            pytest.param(
                MEF3Passwords(level_1_password=LEVEL_1, level_2_password="wrong-password"),
                "level 2 password does not match the level 2 password validation field",
                id="wrong-level-2",
            ),
            pytest.param(
                MEF3Passwords(level_2_password=LEVEL_1),
                "level 2 password does not match",
                id="level-1-given-as-level-2",
            ),
        ],
    )
    def test_wrong_password(self, gapped_copy, passwords, message):
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        with pytest.raises(ValueError, match=message) as error:
            read_quietly(gapped_copy, passwords=passwords)
        for secret in (LEVEL_1, LEVEL_2, "wrong-password"):
            assert secret not in str(error.value)

    def test_no_password(self, gapped_copy):
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        with pytest.raises(ValueError, match=r"is encrypted \(section 2 at level 1, section 3 at level 2\)\. Give"):
            read_quietly(gapped_copy)

    def test_unencrypted_session_ignores_a_password(self, plaintext):
        assert summary(read_quietly(GAPPED, passwords=MEF3Passwords(password=LEVEL_2))) == plaintext
        assert summary(read_quietly(GAPPED, passwords=MEF3Passwords(password="wrong-password"))) == plaintext

    def test_each_channel_keeps_its_own_factor(self, gapped_copy):
        """Factors can differ by channel within a session: 1000/32767, 5000/32767 and 50/32767 µV here."""
        factors = {"Cz": 0.030518511310219765, "Fp1": 0.15259255468845367, "EKG": 0.0015259254723787308}

        def set_factor(channel_name, raw):
            raw[8760:8768] = np.float64(factors[channel_name]).tobytes()

        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2, edit=set_factor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recording = MEF3RecordingExtractor(gapped_copy, password=LEVEL_2)
        session = recording.mef3_session
        assert {channel.name: channel.units_conversion_factor for channel in session.channels} == factors
        np.testing.assert_array_equal(
            recording.get_property("gain_to_physical_unit"), [factors[name] for name in recording.get_channel_ids()]
        )
        units = dict(zip(recording.get_channel_ids(), recording.get_property("physical_unit")))
        to_microvolts = {"microvolts": 1.0, "millivolts": 1000.0}
        np.testing.assert_allclose(
            recording.get_channel_gains(),
            [factors[name] * to_microvolts[units[name]] for name in recording.get_channel_ids()],
        )

    def test_un_offset_first_index_entry(self, gapped_copy, plaintext):
        """
        Some writers store the first .tidx entry's start time as a true µUTC time, not offset.

        Times the writer offset are negative; that one is positive, so it is taken as it is, and the session
        reads exactly as when every entry is offset.
        """
        for index_path in gapped_copy.glob("*.timd/*-000000.segd/*.tidx"):
            raw = bytearray(index_path.read_bytes())
            stored = int(np.frombuffer(bytes(raw[1024 + 8 : 1024 + 16]), dtype="<i8")[0])
            assert stored < 0
            raw[1024 + 8 : 1024 + 16] = np.int64(-stored + RECORDING_TIME_OFFSET).tobytes()
            index_path.write_bytes(bytes(raw))
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        session = read_quietly(gapped_copy, passwords=MEF3Passwords(password=LEVEL_2))
        assert session.start_time == START_TIME
        assert summary(session) == plaintext

    def test_encrypted_data_blocks_are_still_refused(self, gapped_copy):
        index_path = gapped_copy / "Cz.timd" / "Cz-000000.segd" / "Cz-000000.tidx"
        raw = bytearray(index_path.read_bytes())
        raw[1024 + 44] |= 0x02  # RED_LEVEL_1_ENCRYPTION_MASK
        index_path.write_bytes(bytes(raw))
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        with pytest.raises(ValueError, match="data blocks are not supported, with or without a password"):
            read_quietly(gapped_copy, passwords=MEF3Passwords(password=LEVEL_2))


class TestSubjectFieldsStayOut:
    def test_no_section_3_subject_field_reaches_the_output(self, gapped_copy, plaintext, tmp_path):
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2, edit=write_subject_fields)
        forbidden = list(MARKERS.values()) + [MARKERS["subject_name_1"][:4]]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            interface = MEF3RecordingInterface(folder_path=gapped_copy, password=LEVEL_2)
            metadata = interface.get_metadata()
            nwbfile_path = tmp_path / "out.nwb"
            interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

        texts = [pickle.dumps(interface.mef3_session), repr(metadata).encode()]
        texts += [str(warning.message).encode() for warning in caught]
        texts.append(nwbfile_path.read_bytes())

        def collect(name, item):
            values = list(item.attrs.values())
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "OSU":
                values.append(item[()])
            for value in values:
                texts.append(repr(value).encode())

        with h5py.File(nwbfile_path, "r") as file:
            file.visititems(collect)
        for text in texts:
            for marker in forbidden:
                assert marker not in text
        # The malformed DST marker changes nothing: the times and the time zone are those of the plaintext.
        assert summary(interface.mef3_session) == plaintext

    def test_passwords_stay_out_of_the_extractor_description(self, gapped_copy):
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface = MEF3RecordingInterface(folder_path=gapped_copy, password=LEVEL_2)
        for description in (
            repr(interface.recording_extractor.to_dict()),
            repr(interface.extractor_kwargs),
            repr(interface.recording_extractor),
        ):
            assert LEVEL_2 not in description
        assert MEF3RecordingInterface.get_available_channel_ids(gapped_copy, password=LEVEL_2) == ["Cz", "Fp1", "EKG"]
        assert MEF3RecordingInterface.get_channel_sampling_frequencies(gapped_copy, password=LEVEL_2) == {
            "Cz": 1000.0,
            "Fp1": 1000.0,
            "EKG": 1000.0,
        }


class TestIncompleteSegmentsOfAnEncryptedSession:
    """Incomplete segments are found from the directories alone, and an encrypted session reads around them as a
    plaintext one does, the password never in what is raised or warned."""

    @staticmethod
    def _incomplete(session_path: Path):
        remove_segment_file(session_path, "Cz", 1, ".tidx")  # a segment of a channel otherwise kept
        remove_segment_file(session_path, "Fp1", 0, ".tmet")  # every segment of a channel
        remove_segment_file(session_path, "Fp1", 1, ".tmet")

    def test_reads_as_the_plaintext_session(self, gapped_copy, tmp_path):
        plaintext_copy = tmp_path / "plaintext.mefd"
        shutil.copytree(GAPPED, plaintext_copy)
        self._incomplete(plaintext_copy)
        encrypt_session(gapped_copy, level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        self._incomplete(gapped_copy)

        # No password needed to find them.
        assert MEF3RecordingInterface.get_unreadable_channel_ids(gapped_copy) == ["Fp1"]
        assert MEF3RecordingInterface.get_incomplete_segments(gapped_copy) == {
            "Cz": {"Cz-000001.segd": [".tidx"]},
            "Fp1": {"Fp1-000000.segd": [".tmet"], "Fp1-000001.segd": [".tmet"]},
        }
        assert MEF3RecordingInterface.get_channel_sampling_frequencies(
            gapped_copy, password=LEVEL_2, channels_to_skip=["Fp1"]
        ) == {"Cz": 1000.0, "EKG": 1000.0}

        passwords = MEF3Passwords(level_1_password=LEVEL_1, level_2_password=LEVEL_2)
        with pytest.raises(ValueError, match="get_unreadable_channel_ids") as error:
            read_quietly(gapped_copy, passwords=passwords)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            session = read_mef3_session(gapped_copy, passwords=passwords, channels_to_skip=["Fp1"])
        expected = read_quietly(plaintext_copy, channels_to_skip=["Fp1"])
        assert [summary(session.select_group(index)) for index in range(len(session.channel_groups))] == [
            summary(expected.select_group(index)) for index in range(len(expected.channel_groups))
        ]
        messages = [str(error.value)] + [str(warning.message) for warning in caught]
        assert any("Cz-000001.segd (no .tidx file)" in message for message in messages)
        for message in messages:
            for secret in (LEVEL_1, LEVEL_2):
                assert secret not in message
