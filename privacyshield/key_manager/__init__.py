"""
key_manager/
------------
Encryption and key management for PrivacyShield.

Public API:
  encrypt_token_map()  → saves .privacyshield file, returns key bytes
  decrypt_token_map()  → reads .privacyshield file, returns token map
  generate_key()       → create a new Fernet key
  key_to_string()      → bytes → base64 string (show to user)
  string_to_key()      → base64 string → bytes (from user input)
"""

from .encryptor import encrypt_token_map, generate_key, key_to_string, string_to_key
from .decryptor import decrypt_token_map

__all__ = [
    "encrypt_token_map",
    "decrypt_token_map",
    "generate_key",
    "key_to_string",
    "string_to_key",
]
