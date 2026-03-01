import sys
from pathlib import Path

# Add project root to PYTHONPATH for pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))