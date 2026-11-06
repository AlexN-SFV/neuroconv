"""
Passwords for MEF 3.0 sessions whose metadata is encrypted.

meflib encrypts section 2 of a metadata file (the channel's technical fields: sampling rate, units conversion
factor, descriptions) and section 3 (the recording time offset, the time zone, and the subject's identity) with
AES-128 in ECB mode, each section at the level its encryption byte gives: level 1 with the level 1 password,
level 2 with the level 2 password. Everything here follows meflib (``meflib.c`` in pymef 1.4.8):

* ``extract_terminal_password_bytes``: a password is the last UTF-8 byte of each of its first 16 characters,
  padded with zero bytes to 16. Those bytes are the AES key.
* ``process_password_data`` and ``check_password``: the universal header holds two validation fields.
  ``level_1_password_validation_field`` is the first 16 bytes of the SHA-256 of the level 1 password, and
  ``level_2_password_validation_field`` is the first 16 bytes of the SHA-256 of the level 2 password, XORed
  with the level 1 password. So a level 2 password also yields the level 1 password, and opens both levels.

The passwords are held only as long as the objects that are given them; they never appear in a ``repr``, a
message or a warning.
"""

import hashlib
from dataclasses import dataclass, field

PASSWORD_BYTES = 16
AES_BLOCK_BYTES = 16
# UNIVERSAL_HEADER_LEVEL_1_PASSWORD_VALIDATION_FIELD_OFFSET and _LEVEL_2_.
_LEVEL_1_VALIDATION_FIELD = slice(868, 884)
_LEVEL_2_VALIDATION_FIELD = slice(884, 900)


def password_bytes(password: str) -> bytes:
    """meflib's ``extract_terminal_password_bytes``: the AES key a password stands for."""
    terminal_bytes = bytes(character.encode("utf-8")[-1] for character in password[:PASSWORD_BYTES])
    return terminal_bytes.ljust(PASSWORD_BYTES, b"\0")


def _sha256_16(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()[:PASSWORD_BYTES]


def _xor(first: bytes, second: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(first, second))


def _level_1_key_if_level_1(key: bytes, universal_header: bytes) -> bytes | None:
    return key if _sha256_16(key) == universal_header[_LEVEL_1_VALIDATION_FIELD] else None


def _level_1_key_if_level_2(key: bytes, universal_header: bytes) -> bytes | None:
    """The level 1 password a level 2 password unlocks, or None when ``key`` is not the level 2 password."""
    level_1_key = _xor(_sha256_16(key), universal_header[_LEVEL_2_VALIDATION_FIELD])
    return level_1_key if _sha256_16(level_1_key) == universal_header[_LEVEL_1_VALIDATION_FIELD] else None


def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    """meflib's ``AES_decrypt``, one 16-byte block at a time, over a whole number of blocks."""
    if len(data) % AES_BLOCK_BYTES:
        raise ValueError(f"AES-ECB decrypts whole 16-byte blocks; got {len(data)} bytes.")
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError:
        raise ImportError(
            "Reading an encrypted MEF 3 session needs the 'cryptography' package. Install it with "
            'pip install "neuroconv[mef]".'
        ) from None
    decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return decryptor.update(data) + decryptor.finalize()


def aes_ecb_encrypt(key: bytes, data: bytes) -> bytes:
    """meflib's ``AES_encrypt``; neuroconv only decrypts, but the tests build encrypted sessions with it."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return encryptor.update(data) + encryptor.finalize()


@dataclass(frozen=True)
class MEF3Passwords:
    """
    The password, or the level 1 and level 2 passwords, of a MEF 3 session.

    Give either ``password``, whose level is found from each file's validation fields, or one or both of
    ``level_1_password`` and ``level_2_password``, each checked at its own level. The values are left out of
    the ``repr``.
    """

    password: str | None = field(default=None, repr=False)
    level_1_password: str | None = field(default=None, repr=False)
    level_2_password: str | None = field(default=None, repr=False)

    def __post_init__(self):
        given = [value for value in (self.password, self.level_1_password, self.level_2_password) if value is not None]
        if not given:
            raise ValueError("MEF3Passwords needs a password, or a level 1 or level 2 password.")
        if self.password is not None and (self.level_1_password is not None or self.level_2_password is not None):
            raise ValueError(
                "Give a MEF 3 password either as password, whose level is found from the files, or as "
                "level_1_password and level_2_password, not both ways."
            )
        if any(not isinstance(value, str) or not value for value in given):
            # The value itself is never shown.
            raise ValueError("A MEF 3 password must be a non-empty string.")

    @classmethod
    def from_arguments(
        cls,
        password: str | None = None,
        level_1_password: str | None = None,
        level_2_password: str | None = None,
    ) -> "MEF3Passwords | None":
        """The passwords, or None when none is given."""
        if password is None and level_1_password is None and level_2_password is None:
            return None
        return cls(password=password, level_1_password=level_1_password, level_2_password=level_2_password)

    def keys(self, universal_header: bytes, path, levels_needed: set[int]) -> dict[int, bytes]:
        """
        The AES key of each level in ``levels_needed`` (1 and/or 2), checked against the file's validation fields.

        Raises a ValueError naming the file and the level that failed, never the password.
        """
        keys = {}
        if self.password is not None:
            key = password_bytes(self.password)
            level_1_key = _level_1_key_if_level_1(key, universal_header)
            if level_1_key is not None:
                keys[1] = level_1_key
            else:
                level_1_key = _level_1_key_if_level_2(key, universal_header)
                if level_1_key is None:
                    raise ValueError(
                        f"The MEF 3 password does not open '{path}': it matches neither the file's level 1 nor its "
                        f"level 2 password validation field, and the file is encrypted at level "
                        f"{' and '.join(str(level) for level in sorted(levels_needed))}."
                    )
                keys[1], keys[2] = level_1_key, key
        else:
            if self.level_1_password is not None:
                level_1_key = _level_1_key_if_level_1(password_bytes(self.level_1_password), universal_header)
                if level_1_key is None:
                    raise ValueError(
                        f"The MEF 3 level 1 password does not match the level 1 password validation field of '{path}'."
                    )
                keys[1] = level_1_key
            if self.level_2_password is not None:
                key = password_bytes(self.level_2_password)
                level_1_key = _level_1_key_if_level_2(key, universal_header)
                if level_1_key is None:
                    raise ValueError(
                        f"The MEF 3 level 2 password does not match the level 2 password validation field of '{path}'."
                    )
                keys[2] = key
                keys.setdefault(1, level_1_key)

        missing = sorted(levels_needed - set(keys))
        if missing:
            raise ValueError(
                f"'{path}' is encrypted at level {missing[0]}, and the MEF 3 password given is only good for level "
                f"{max(keys)}. Give its level {missing[0]} password."
            )
        return {level: keys[level] for level in levels_needed}
