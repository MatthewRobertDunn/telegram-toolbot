"""Registry of opencode.ai-compatible skills.

Scans the skills source directory (``config.SKILLS_DIR``) for skill
definitions and exposes them as a list of ``SkillInfo`` entries with
``name``, ``description`` and ``sandbox_path`` attributes.

``sandbox_path`` is the skill file's path as seen inside the sandbox,
where the skills directory is mounted read-only at
``config.SANDBOX_SKILLS_DIR`` (e.g.
``/skills/acceptance-and-commitment/SKILL.md``).

Supported forms, mirroring opencode.ai discovery:

* ``<id>/SKILL.md`` (nesting in grouping directories allowed,
  e.g. ``teams/release/SKILL.md`` has the ID ``release``)
* ``<id>.md`` directly at the source root

The file path determines the skill ID; the frontmatter ``name`` is only
a display label. Hidden files/directories and ``node_modules`` are
skipped, and directories more than five levels below the source root
are not descended into. A directory that yields a skill is not
searched further, so its reference material is not offered as skills.
"""

import logging
import os
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)

SKILL_ENTRY_FILENAME = "SKILL.md"
_MAX_DEPTH = 6
_SKIPPED_DIRS = frozenset({"node_modules"})


@dataclass(frozen=True)
class SkillInfo:
    """A single discovered skill definition.

    ``sandbox_path`` is the skill file's absolute path inside the
    sandbox (POSIX separators), e.g.
    ``/skills/acceptance-and-commitment/SKILL.md``.
    """

    name: str
    description: str
    sandbox_path: str


# Populated by load_skills() at bot startup (see chatbot.post_init).
SKILLS: list[SkillInfo] = []


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse the ``---`` frontmatter block into a flat string map.

    Only the simple ``key: value`` subset used by skill frontmatter
    (``name``, ``description``, ...) is supported. Lines that are
    neither ``key: value`` pairs nor indented continuations are
    reported via a warning and skipped, never fatal. Missing or absent
    frontmatter yields an empty map.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    current: str | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line[:1] in (" ", "\t") and current is not None:
            # Indented continuation of a folded/plain multi-line value.
            fields[current] += " " + line.strip()
            continue
        key, sep, value = line.partition(":")
        if not sep or not key.strip() or key != key.strip():
            if line.strip():
                logger.warning("Malformed frontmatter line ignored: %r", line)
            current = None
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        fields[key] = value
        current = key
    return fields


def _iter_skill_files(root: str):
    """Yield ``(skill_id, file_path)`` for each skill under root.

    Two forms are recognised:

    * ``<id>.md`` directly at the root (flat form; ``SKILL.md`` at the
      root is not a skill — an entry file must live inside a directory).
    * ``**/SKILL.md`` (directory form). A directory containing
      ``SKILL.md`` is one skill: its subtree is not searched further,
      so reference material next to the entry file is never offered as
      skills. The walk does not descend into directories more than
      five levels below the root (``_MAX_DEPTH``).

    Hidden files/directories and ``node_modules`` are skipped here, so
    everything yielded is already visible. Later duplicates of an ID
    are yielded too; the caller keeps the last one so deeper
    definitions override earlier ones.
    """
    # Flat form: <id>.md directly at the source root.
    try:
        entries = sorted(os.listdir(root))
    except OSError as e:
        logger.warning("Could not list skills directory %s: %s", root, e)
        return
    for entry in entries:
        if entry.startswith(".") or not entry.endswith(".md"):
            continue
        if entry == SKILL_ENTRY_FILENAME:
            continue
        full = os.path.join(root, entry)
        if os.path.isfile(full):
            yield os.path.splitext(entry)[0], full
    # Directory form: **/SKILL.md, one skill per directory.
    for current, dirs, files in os.walk(root):
        depth = os.path.relpath(current, root).split(os.sep)
        level = 0 if depth == ["."] else len(depth)
        if level >= _MAX_DEPTH:
            del dirs[:]
            continue
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".") and d not in _SKIPPED_DIRS)
        if SKILL_ENTRY_FILENAME in files:
            full = os.path.join(current, SKILL_ENTRY_FILENAME)
            yield os.path.basename(current), full
            dirs[:] = []


def load_skills(skills_dir: str | None = None) -> list[SkillInfo]:
    """Discover skills and refresh the module-level ``SKILLS`` registry.

    Returns the list of discovered skills sorted by ID. Missing files,
    unreadable files, or a missing source directory yield fewer or no
    entries rather than an error.
    """
    global SKILLS
    root = os.path.abspath(skills_dir or config.SKILLS_DIR)
    found: dict[str, SkillInfo] = {}
    if not os.path.isdir(root):
        logger.warning("Skills directory does not exist: %s", root)
        SKILLS = []
        return SKILLS
    for skill_id, path in _iter_skill_files(root):
        try:
            with open(path, encoding="utf-8") as f:
                frontmatter = _parse_frontmatter(f.read())
        except OSError as e:
            logger.warning("Could not read skill file %s: %s", path, e)
            continue
        relative = os.path.relpath(os.path.abspath(path), root)
        sandbox_path = (config.SANDBOX_SKILLS_DIR.rstrip("/") + "/"
                        + relative.replace(os.sep, "/"))
        found[skill_id] = SkillInfo(
            name=skill_id,
            description=frontmatter.get("description", ""),
            sandbox_path=sandbox_path,
        )
    SKILLS = sorted(found.values(), key=lambda skill: skill.name)
    logger.info("Loaded %d skill(s) from %s", len(SKILLS), root)
    return SKILLS


def get_skill(name: str) -> SkillInfo | None:
    """Return the registry entry with this skill ID, if any."""
    for skill in SKILLS:
        if skill.name == name:
            return skill
    return None


def get_active_skills() -> list[SkillInfo]:
    """Return the skills currently enabled for agents.

    All discovered skills are active for now; per-skill toggling can
    filter this list later without changing callers.
    """
    return list(SKILLS)
