from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / ".chroma"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
