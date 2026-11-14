"""Environment-driven settings and shared clients for the backend."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=openai_api_key)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
# On Azure App Service, /tmp is ephemeral and not shared across container instances.
# If DB_PATH is in /home/site/data (persistent shared storage), we redirect OUTPUT_DIR
# to a subfolder there so PDFs are saved persistently and are accessible to all instances.
db_path_env = os.getenv("DB_PATH")
if db_path_env and "/home/site/data" in db_path_env:
    OUTPUT_DIR = "/home/site/data/output"
elif OUTPUT_DIR == "/tmp" and os.path.exists("/home/site/data"):
    OUTPUT_DIR = "/home/site/data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Storage behavior:
# STORE_REPORTS = "disk" (default) -> write to OUTPUT_DIR and serve via /download/<file>
# STORE_REPORTS = "memory" -> do not persist; generate to disk, read once, delete, and offer one-time memory download via /download-temp/<token>
STORE_REPORTS = os.getenv("STORE_REPORTS", "disk").lower()
# SAVE_TX_JSON = "true"/"false" -> whether to persist extracted bank tx JSON artifacts
SAVE_TX_JSON = os.getenv("SAVE_TX_JSON", "false").lower() == "true"
# DETERMINISTIC_RECURRING = "true"/"false" -> compute recurring debits/credits from
# structured transactions (LLM entries only fill non-overlapping gaps). Kill switch
# back to pure-LLM recurring detection if set to "false".
DETERMINISTIC_RECURRING = os.getenv("DETERMINISTIC_RECURRING", "true").lower() == "true"

# Logo shipped with the package (used on report covers and page headers)
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
