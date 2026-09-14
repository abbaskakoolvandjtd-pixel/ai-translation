
import os
import sys

from dotenv import load_dotenv, dotenv_values

# Determine the directory containing this file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Build the full path to the .env file located alongside this script
DOTENV_PATH = os.path.join(BASE_DIR, ".env")

# Load environment variables from the .env file in the code directory
load_dotenv(dotenv_path=DOTENV_PATH)



# 2️⃣ Read the entire .env file into a dictionary.
# You can give an absolute or relative path to the .env file.
ENV_DICT = dotenv_values(dotenv_path=DOTENV_PATH)


# print(ENV_DICT.items())


def _cast_value(value: str):
    """
    Convert a string to an appropriate Python type (int, float, bool).
    If the conversion is not applicable, return the original string.
    """
    # Detect integer values
    if value.isdigit():
        return int(value)

    # Detect floating‑point numbers
    try:
        return float(value)
    except ValueError:
        pass

    # Detect boolean values
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"

    # Fallback: keep as string
    return value


class TranslationConfig:
    """
    All keys present in the .env file become attributes of this class.
    """


# 3️⃣ Populate the class with values from .env (executed at import time)
for key, raw_val in ENV_DICT.items():
    # Convert the raw string to the appropriate Python type
    val = _cast_value(raw_val)
    # print(key, ":", val)
    # If the attribute already exists, overwrite it; otherwise create it dynamically
    setattr(TranslationConfig, key, val)
sys.path.append(TranslationConfig.LOCAL_SHARE_FILE_PATH)
sys.path.append(TranslationConfig.LOCAL_SPMBI_PATH)