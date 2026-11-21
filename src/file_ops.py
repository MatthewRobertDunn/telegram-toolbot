"""Read a small text file shipped next to the bot (e.g. ./system_prompt)."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def read_file(filename: str) -> Optional[str]:
    path = Path(__file__).parent / filename
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("read_file: %s does not exist",
                       path.resolve())
        return None
    except UnicodeDecodeError as e:
        logger.warning("read_file: %s is not valid UTF-8: %s",
                       path.resolve(), e)
        return None
    except OSError as e:
        logger.warning("read_file: could not read %s: %s",
                       path.resolve(), e)
        return None
    logger.info("Read file: %s", path)
    return text
