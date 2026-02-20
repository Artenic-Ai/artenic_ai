"""Tests for artenic_ai_platform.config.crypto — 100% coverage."""

from __future__ import annotations

from artenic_ai_platform.config.crypto import SecretManager


class TestSecretManagerInit:
    def test_with_passphrase(self) -> None:
        sm = SecretManager("my-secret-key")
        assert sm._fernet is not None

    def test_without_passphrase_generates_random_key(self) -> None:
        sm = SecretManager("")
        assert sm._fernet is not None

    def test_default_passphrase_is_empty(self) -> None:
        sm = SecretManager()
        assert sm._fernet is not None


class TestEncryptDecrypt:
    def test_round_trip(self) -> None:
        sm = SecretManager("test-key")
        plaintext = "hello world"
        ciphertext = sm.encrypt(plaintext)
        assert ciphertext != plaintext
        assert sm.decrypt(ciphertext) == plaintext

    def test_empty_string_round_trip(self) -> None:
        sm = SecretManager("test-key")
        ciphertext = sm.encrypt("")
        assert sm.decrypt(ciphertext) == ""

    def test_unicode_round_trip(self) -> None:
        sm = SecretManager("test-key")
        plaintext = "Bonjour le monde! \u2603\u2764"
        ciphertext = sm.encrypt(plaintext)
        assert sm.decrypt(ciphertext) == plaintext

    def test_different_passphrases_produce_different_ciphertexts(self) -> None:
        sm1 = SecretManager("key-one")
        sm2 = SecretManager("key-two")
        ct1 = sm1.encrypt("secret")
        ct2 = sm2.encrypt("secret")
        assert ct1 != ct2

    def test_same_passphrase_can_decrypt(self) -> None:
        sm1 = SecretManager("same-key")
        sm2 = SecretManager("same-key")
        ct = sm1.encrypt("secret")
        assert sm2.decrypt(ct) == "secret"

    def test_wrong_passphrase_returns_sentinel(self) -> None:
        sm1 = SecretManager("right-key")
        sm2 = SecretManager("wrong-key")
        ct = sm1.encrypt("secret")
        result = sm2.decrypt(ct)
        assert result == "***DECRYPTION_FAILED***"

    def test_invalid_ciphertext_returns_sentinel(self) -> None:
        sm = SecretManager("test-key")
        result = sm.decrypt("not-valid-base64!!!")
        assert result == "***DECRYPTION_FAILED***"

    def test_corrupted_ciphertext_returns_sentinel(self) -> None:
        sm = SecretManager("test-key")
        ct = sm.encrypt("secret")
        corrupted = ct[:10] + "XXXX" + ct[14:]
        result = sm.decrypt(corrupted)
        assert result == "***DECRYPTION_FAILED***"


class TestMask:
    def test_long_value_shows_last_4(self) -> None:
        assert SecretManager.mask("sk-1234567890abcdef") == "***cdef"

    def test_exactly_visible_chars_returns_masked(self) -> None:
        assert SecretManager.mask("abcd") == "***"

    def test_shorter_than_visible_chars_returns_masked(self) -> None:
        assert SecretManager.mask("ab") == "***"

    def test_empty_string_returns_masked(self) -> None:
        assert SecretManager.mask("") == "***"

    def test_custom_visible_chars(self) -> None:
        assert SecretManager.mask("sk-1234567890", visible_chars=6) == "***567890"

    def test_visible_chars_one(self) -> None:
        assert SecretManager.mask("secret", visible_chars=1) == "***t"

    def test_value_exactly_one_more_than_visible(self) -> None:
        assert SecretManager.mask("abcde", visible_chars=4) == "***bcde"
