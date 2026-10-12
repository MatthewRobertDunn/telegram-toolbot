"""Edit tool backend for the toolbot sandbox.

Performs exact string replacement in a file, ported from opencode's edit
tool replacer chain (exact, line-trimmed, block-anchor, whitespace-
normalized, indentation-flexible). Reads a JSON payload from stdin:
{"path": ..., "old_string": ..., "new_string": ..., "replace_all": ...}.
Prints {"ok": true, ...} or {"ok": false, "error": "..."} to stdout.
"""
import json
import re
import sys

SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.65
MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.65


def levenshtein(a: str, b: str) -> int:
    if a == "" or b == "":
        return max(len(a), len(b))
    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,
                               matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + cost)
    return matrix[len(a)][len(b)]


def simple_replacer(content: str, find: str):
    yield find


def line_trimmed_replacer(content: str, find: str):
    original_lines = content.split("\n")
    search_lines = find.split("\n")
    if not search_lines:
        return
    if search_lines[-1] == "":
        search_lines.pop()
    for i in range(0, len(original_lines) - len(search_lines) + 1):
        if all(original_lines[i + j].strip() == search_lines[j].strip()
               for j in range(len(search_lines))):
            match_start = sum(len(original_lines[k]) + 1 for k in range(i))
            match_end = match_start + sum(
                len(original_lines[i + k])
                + (1 if k < len(search_lines) - 1 else 0)
                for k in range(len(search_lines)))
            yield content[match_start:match_end]


def block_anchor_replacer(content: str, find: str):
    original_lines = content.split("\n")
    search_lines = find.split("\n")
    if len(search_lines) < 3:
        return
    if search_lines[-1] == "":
        search_lines.pop()
    first_line = search_lines[0].strip()
    last_line = search_lines[-1].strip()
    search_size = len(search_lines)
    max_line_delta = max(1, search_size // 4)

    candidates = []
    for i, line in enumerate(original_lines):
        if line.strip() != first_line:
            continue
        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last_line:
                actual_size = j - i + 1
                if abs(actual_size - search_size) <= max_line_delta:
                    candidates.append((i, j))
                break
    if not candidates:
        return

    def similarity(candidate):
        start, end = candidate
        actual_size = end - start + 1
        lines_to_check = min(search_size - 2, actual_size - 2)
        if lines_to_check <= 0:
            return 1.0
        total = 0.0
        for j in range(1, search_size - 1):
            original = original_lines[start + j].strip()
            search = search_lines[j].strip()
            max_len = max(len(original), len(search))
            if max_len == 0:
                continue
            total += 1 - levenshtein(original, search) / max_len
        return total / lines_to_check

    if len(candidates) == 1:
        if similarity(candidates[0]) >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
            yield _span(content, candidates[0])
        return

    best = max(candidates, key=similarity)
    if similarity(best) >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD:
        yield _span(content, best)


def _span(content: str, candidate: tuple[int, int]) -> str:
    original_lines = content.split("\n")
    start, end = candidate
    match_start = sum(len(original_lines[k]) + 1 for k in range(start))
    match_end = match_start + sum(
        len(original_lines[k]) + (1 if k < end else 0)
        for k in range(start, end + 1))
    return content[match_start:match_end]


def whitespace_normalized_replacer(content: str, find: str):
    def normalize_whitespace(text: str) -> str:
        return " ".join(text.split())

    normalized_find = normalize_whitespace(find)
    lines = content.split("\n")
    for line in lines:
        if normalize_whitespace(line) == normalized_find:
            yield line
        elif normalized_find and normalized_find in normalize_whitespace(line):
            words = find.strip().split()
            if words:
                pattern = r"\s+".join(re.escape(word) for word in words)
                try:
                    match = re.search(pattern, line)
                except re.error:
                    match = None
                if match:
                    yield match.group(0)
    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(0, len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i:i + len(find_lines)])
            if normalize_whitespace(block) == normalized_find:
                yield block


def indentation_flexible_replacer(content: str, find: str):
    def remove_indentation(text: str) -> str:
        lines = text.split("\n")
        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            return text
        min_indent = min(len(m.group(1)) for m in
                         (re.match(r"^(\s*)", line) for line in non_empty))
        return "\n".join(line[min_indent:] if line.strip() else line
                         for line in lines)

    normalized_find = remove_indentation(find)
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(0, len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i:i + len(find_lines)])
        if remove_indentation(block) == normalized_find:
            yield block


def multi_occurrence_replacer(content: str, find: str):
    start = 0
    while True:
        index = content.find(find, start)
        if index == -1:
            break
        yield find
        start = index + len(find)


REPLACERS = (
    simple_replacer,
    line_trimmed_replacer,
    block_anchor_replacer,
    whitespace_normalized_replacer,
    indentation_flexible_replacer,
    multi_occurrence_replacer,
)


def is_disproportionate_match(search: str, old_string: str) -> bool:
    old_lines = old_string.split("\n")
    search_lines = search.split("\n")
    if len(search_lines) >= max(len(old_lines) + 3, len(old_lines) * 2):
        return True
    if len(old_lines) == 1:
        return False
    return (len(search.strip())
            > max(len(old_string.strip()) + 500, len(old_string.strip()) * 4))


def replace(content: str, old_string: str, new_string: str,
            replace_all: bool = False) -> str:
    if old_string == new_string:
        raise ValueError(
            "No changes to apply: oldString and newString are identical.")
    if old_string == "":
        raise ValueError(
            "oldString cannot be empty when editing an existing file. "
            "Provide the exact text to replace, or use write for an "
            "intentional full-file replacement.")

    not_found = True
    for replacer in REPLACERS:
        for search in replacer(content, old_string):
            index = content.find(search)
            if index == -1:
                continue
            not_found = False
            if is_disproportionate_match(search, old_string):
                raise ValueError(
                    "Refusing replacement because the matched span is much "
                    "larger than oldString. Re-read the file and provide the "
                    "full exact oldString for the intended replacement.")
            if replace_all:
                return content.replace(search, new_string)
            last_index = content.rfind(search)
            if index != last_index:
                continue
            return content[:index] + new_string + content[index + len(search):]

    if not_found:
        raise ValueError(
            "Could not find oldString in the file. It must match exactly, "
            "including whitespace, indentation, and line endings.")
    raise ValueError(
        "Found multiple matches for oldString. Provide more surrounding "
        "context to make the match unique.")


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        path = payload["path"]
        old_string = payload["old_string"]
        new_string = payload["new_string"]
        replace_all = bool(payload.get("replace_all", False))

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            raise ValueError(f"Cannot edit binary file: {path}")

        normalized_old = old_string.replace("\r\n", "\n")
        normalized_new = new_string.replace("\r\n", "\n")
        ending = "\r\n" if "\r\n" in content else "\n"
        if ending == "\r\n":
            normalized_old = normalized_old.replace("\n", "\r\n")
            normalized_new = normalized_new.replace("\n", "\r\n")
        updated = replace(content, normalized_old, normalized_new, replace_all)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(updated)
        print(json.dumps({"ok": True, "data": {
            "path": path, "output": "Edit applied successfully."}}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    sys.exit(main())
