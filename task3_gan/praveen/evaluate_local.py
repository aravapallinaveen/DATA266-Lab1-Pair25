"""Repository entry point for Praveen's Task 3 inference."""
from pathlib import Path
import sys

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

from evaluate import main

if __name__ == "__main__":
    main()
