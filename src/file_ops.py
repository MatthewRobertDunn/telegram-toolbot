import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def read_file(filename: str) -> Optional[str]:
    path = Path(__file__).parent / filename
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    logger.info("Read file: %s", path)
    return text