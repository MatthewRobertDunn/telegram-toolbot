import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import aiosqlite
import persistence.journal_store as journal_store
import persistence.sqlite_store as sqlite_store
from persistence.journal_store import ReplaceOutcome, StoreValidationError
from tools import journal_tool
from mock_context import mock_ctx
from user_settings import UserSettings


def _counting_executescript(counter: dict):
    original = aiosqlite.Connection.executescript

    def counting(conn, sql, *args, **kwargs):
        counter["calls"] += 1
        return original(conn, sql, *args, **kwargs)

    return counting


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


class TestJournalStore(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "journal.db")
        self.store = journal_store.create_journal_store(self.db)
        await self.store.create_schema()

    async def asyncTearDown(self):
        await self.store.close()
        self._tmp.cleanup()

    async def test_schema_script_runs_once_per_database(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        counter = {"calls": 0}
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        with mock.patch.object(aiosqlite.Connection, "executescript",
                               _counting_executescript(counter)):
            first = journal_store.create_journal_store(path)
            second = journal_store.create_journal_store(path)
            try:
                await first.add_entry(user_id=1, topic="A", entry="one",
                                      created=created,
                                      created_epoch_ms=epoch)
                await first.search_entries(user_id=1)
                # Repeated explicit calls on one instance do nothing.
                await first.create_schema()
                await first.create_schema()
                # A second store over the same file skips the DDL too:
                # readiness is shared per (path, table) for the process,
                # and it still reads the shared database. Any DDL re-run
                # would show up in the counter below.
                self.assertIsNotNone(
                    await second.get_entry(user_id=1, entry_id=1))
            finally:
                await second.close()
                await first.close()
        # One script application for the whole process, not one per store
        # or per operation.
        self.assertEqual(counter["calls"], 1)

    async def test_db_path_resolved_at_call_time(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with mock.patch.object(sqlite_store.config, "sql_db_path", path):
            # No explicit path: the store must pick up the patched
            # config when it connects, not when it was imported.
            store = journal_store.create_journal_store()
            try:
                await store.add_entry(
                    user_id=1, topic="A", entry="x",
                    created="2026-11-14T16:34:00+00:00",
                    created_epoch_ms=_ms("2026-11-14T16:34:00+00:00"))
            finally:
                await store.close()
        self.assertGreater(os.path.getsize(path), 0)

    async def test_validation_raises_typed_error(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        with self.assertRaises(StoreValidationError) as caught:
            await self.store.add_entry(
                user_id=1, topic="   ", entry="x",
                created=created, created_epoch_ms=epoch)
        # The message is the user-facing text, and the error is still a
        # ValueError so generic handling keeps working.
        self.assertEqual(str(caught.exception), "Topic cannot be empty")
        self.assertIsInstance(caught.exception, ValueError)

    async def test_add_and_get_round_trip(self):
        created = "2026-11-14T16:34:00+10:30"
        record = await self.store.add_entry(
            user_id=1, topic="Mood", entry="Felt okay today.",
            created=created, created_epoch_ms=_ms(created))
        self.assertEqual(record.id, 1)
        fetched = await self.store.get_entry(
            user_id=1, entry_id=1)
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched.id, 1)
        self.assertEqual(fetched.user_id, 1)
        self.assertEqual(fetched.created, created)
        self.assertEqual(fetched.created_epoch_ms, _ms(created))
        self.assertEqual(fetched.topic, "mood")
        self.assertEqual(fetched.entry, "Felt okay today.")
        second = await self.store.add_entry(
            user_id=1, topic="Sleep", entry="Slept well.",
            created=created, created_epoch_ms=_ms(created))
        self.assertEqual(second.id, 2)

    async def test_get_entries_batch(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        for topic in ("A", "B", "C"):
            await self.store.add_entry(
                user_id=1, topic=topic, entry=topic,
                created=created, created_epoch_ms=epoch)
        found = await self.store.get_entries(
            user_id=1, entry_ids=[3, 1])
        self.assertEqual([r.id for r in found], [3, 1])
        self.assertEqual(
            await self.store.get_entries(user_id=1, entry_ids=[]), [])
        self.assertEqual(
            await self.store.get_entries(
                user_id=1, entry_ids=[1, 999]), [found[1]])
        self.assertEqual(
            await self.store.get_entries(user_id=2, entry_ids=[1, 2]), [])

    async def test_add_and_search_reject_invalid_inputs(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        for bad_topic in ("", "   "):
            with self.assertRaises(StoreValidationError):
                await self.store.add_entry(
                    user_id=1, topic=bad_topic, entry="x",
                    created=created, created_epoch_ms=epoch)
        for bad_entry in ("", "  \n "):
            with self.assertRaises(StoreValidationError):
                await self.store.add_entry(
                    user_id=1, topic="Mood", entry=bad_entry,
                    created=created, created_epoch_ms=epoch)
        with self.assertRaises(StoreValidationError):
            await self.store.add_entry(
                user_id=1, topic="t" * 129, entry="x",
                created=created, created_epoch_ms=epoch)
        record = await self.store.add_entry(
            user_id=1, topic="t" * 128, entry="x",
            created=created, created_epoch_ms=epoch)
        self.assertEqual(len(record.topic), 128)
        # Search limit bounds fail the same way.
        for bad_limit in (0, 51):
            with self.assertRaises(StoreValidationError):
                await self.store.search_entries(user_id=1, limit=bad_limit)
        self.assertEqual(
            await self.store.get_entry(
                user_id=1, entry_id=999), None)

    async def test_ids_never_reused(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        await self.store.add_entry(
            user_id=1, topic="A", entry="one",
            created=created, created_epoch_ms=epoch)
        await self.store.add_entry(
            user_id=1, topic="A", entry="two",
            created=created, created_epoch_ms=epoch)
        self.assertTrue(await self.store.delete_entry(
            user_id=1, entry_id=1))
        third = await self.store.add_entry(
            user_id=1, topic="A", entry="three",
            created=created, created_epoch_ms=epoch)
        self.assertEqual(third.id, 3)

    async def test_user_isolation(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        record = await self.store.add_entry(
            user_id=1, topic="Mood", entry="user one entry",
            created=created, created_epoch_ms=epoch)
        self.assertIsNone(await self.store.get_entry(
            user_id=2, entry_id=record.id))
        self.assertFalse(await self.store.delete_entry(
            user_id=2, entry_id=record.id))
        self.assertEqual(
            await self.store.replace_in_entry(
                user_id=2, entry_id=record.id,
                old_string="one", new_string="two"),
            ReplaceOutcome.NOT_FOUND)
        self.assertEqual(
            await self.store.search_entries(
                user_id=2, query="one"), [])
        self.assertEqual(
            await self.store.list_topics(user_id=2), [])
        self.assertIsNotNone(await self.store.get_entry(
            user_id=1, entry_id=record.id))

    async def test_replace(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        record = await self.store.add_entry(
            user_id=1, topic="Mood", entry="Felt anxious and tired.",
            created=created, created_epoch_ms=epoch)
        with self.assertRaises(ValueError):
            await self.store.replace_in_entry(
                user_id=1, entry_id=record.id, old_string="",
                new_string="x")
        # A no-op replacement (old == new) is rejected outright, even
        # when the text is present in the entry.
        with self.assertRaises(ValueError):
            await self.store.replace_in_entry(
                user_id=1, entry_id=record.id, old_string="anxious",
                new_string="anxious")
        self.assertEqual(
            await self.store.replace_in_entry(
                user_id=1, entry_id=999, old_string="x",
                new_string="y"),
            ReplaceOutcome.NOT_FOUND)
        self.assertEqual(
            await self.store.replace_in_entry(
                user_id=1, entry_id=record.id, old_string="elated",
                new_string="okay"),
            ReplaceOutcome.TEXT_NOT_FOUND)
        self.assertEqual(
            await self.store.replace_in_entry(
                user_id=1, entry_id=record.id, old_string="anxious",
                new_string="calm"),
            ReplaceOutcome.UPDATED)
        fetched = await self.store.get_entry(
            user_id=1, entry_id=record.id)
        assert fetched is not None
        self.assertEqual(fetched.entry, "Felt calm and tired.")
        self.assertEqual(fetched.created, created)
        self.assertEqual(fetched.created_epoch_ms, epoch)
        # FTS index followed the edit.
        self.assertEqual(
            await self.store.search_entries(
                user_id=1, query="anxious"), [])
        found = await self.store.search_entries(
            user_id=1, query="calm")
        self.assertEqual([r.id for r in found], [record.id])
        # First occurrence only.
        await self.store.replace_in_entry(
            user_id=1, entry_id=record.id, old_string="and",
            new_string="AND")
        await self.store.replace_in_entry(
            user_id=1, entry_id=record.id, old_string="AND",
            new_string="")
        fetched = await self.store.get_entry(
            user_id=1, entry_id=record.id)
        assert fetched is not None
        self.assertEqual(fetched.entry, "Felt calm  tired.")
        self.assertNotIn("AND", fetched.entry)

    async def test_replace_multiple_matches(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        record = await self.store.add_entry(
            user_id=1, topic="Log", entry="ha ha ha",
            created=created, created_epoch_ms=epoch)
        self.assertEqual(
            await self.store.replace_in_entry(
                user_id=1, entry_id=record.id, old_string="ha",
                new_string="he"),
            ReplaceOutcome.MULTIPLE_MATCHES)
        fetched = await self.store.get_entry(
            user_id=1, entry_id=record.id)
        assert fetched is not None
        self.assertEqual(fetched.entry, "ha ha ha")
        self.assertEqual(
            await self.store.replace_in_entry(
                user_id=1, entry_id=record.id, old_string="ha",
                new_string="he", replace_all=True),
            ReplaceOutcome.UPDATED)
        fetched = await self.store.get_entry(
            user_id=1, entry_id=record.id)
        assert fetched is not None
        self.assertEqual(fetched.entry, "he he he")

    async def test_update_entry_topic(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        record = await self.store.add_entry(
            user_id=1, topic="Mood", entry="x",
            created=created, created_epoch_ms=epoch)
        self.assertEqual(
            await self.store.update_entry_topic(
                user_id=1, entry_id=999,
                old_topic="Mood", new_topic="Sleep"),
            ReplaceOutcome.NOT_FOUND)
        self.assertEqual(
            await self.store.update_entry_topic(
                user_id=1, entry_id=record.id,
                old_topic="Wrong", new_topic="Sleep"),
            ReplaceOutcome.TEXT_NOT_FOUND)
        with self.assertRaises(ValueError):
            await self.store.update_entry_topic(
                user_id=1, entry_id=record.id,
                old_topic="Mood", new_topic="Mood")
        with self.assertRaises(ValueError):
            await self.store.update_entry_topic(
                user_id=1, entry_id=record.id,
                old_topic="Mood", new_topic="  ")
        with self.assertRaises(ValueError):
            await self.store.update_entry_topic(
                user_id=1, entry_id=record.id,
                old_topic="Mood", new_topic="t" * 129)
        self.assertEqual(
            await self.store.update_entry_topic(
                user_id=1, entry_id=record.id,
                old_topic="Mood", new_topic="Sleep"),
            ReplaceOutcome.UPDATED)
        fetched = await self.store.get_entry(
            user_id=1, entry_id=record.id)
        assert fetched is not None
        self.assertEqual(fetched.topic, "sleep")
        self.assertEqual(
            await self.store.update_entry_topic(
                user_id=2, entry_id=record.id,
                old_topic="Sleep", new_topic="Mood"),
            ReplaceOutcome.NOT_FOUND)

    async def test_search_text_semantics(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        await self.store.add_entry(
            user_id=1, topic="Mood", entry="Felt anxious all afternoon.",
            created=created, created_epoch_ms=epoch)
        await self.store.add_entry(
            user_id=1, topic="Sleep", entry="Slept badly, anxious dreams.",
            created=created, created_epoch_ms=epoch)
        await self.store.add_entry(
            user_id=1, topic="Mood", entry="Calm morning walk.",
            created=created, created_epoch_ms=epoch)
        both = await self.store.search_entries(
            user_id=1, query="anxious")
        self.assertEqual(len(both), 2)
        # Multiple terms require all to match.
        narrow = await self.store.search_entries(
            user_id=1, query="anxious dreams")
        self.assertEqual(len(narrow), 1)
        self.assertEqual(narrow[0].topic, "sleep")
        phrase = await self.store.search_entries(
            user_id=1, query='"anxious dreams"')
        self.assertEqual(len(phrase), 1)
        prefix = await self.store.search_entries(
            user_id=1, query="anx")
        self.assertEqual(len(prefix), 2)
        upper = await self.store.search_entries(
            user_id=1, query="ANXIOUS")
        self.assertEqual(len(upper), 2)
        # Gibberish operators cannot break the query builder.
        with self.assertRaises(ValueError):
            await self.store.search_entries(
                user_id=1, query="***")

    async def test_search_topic_filter_case_insensitive(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        await self.store.add_entry(
            user_id=1, topic="Mood", entry="one",
            created=created, created_epoch_ms=epoch)
        await self.store.add_entry(
            user_id=1, topic="Sleep", entry="two",
            created=created, created_epoch_ms=epoch)
        found = await self.store.search_entries(
            user_id=1, topic="mood")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].entry, "one")

    async def test_search_start_window_and_order(self):
        stamps = [
            "2026-11-10T09:00:00+00:00",
            "2026-11-12T09:00:00+00:00",
            "2026-11-14T09:00:00+00:00",
        ]
        for i, stamp in enumerate(stamps):
            await self.store.add_entry(
                user_id=1, topic="Log", entry=f"entry {i}",
                created=stamp, created_epoch_ms=_ms(stamp))
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, start_ms=_ms("2026-11-15T00:00:00+00:00"))],
            ["entry 2", "entry 1", "entry 0"])
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, sort_descending=False)],
            ["entry 0", "entry 1", "entry 2"])
        limited = await self.store.search_entries(user_id=1, limit=1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0].entry, "entry 2")
        # start is inclusive in both directions: the entry sitting
        # exactly on the bound is in, and one millisecond either side
        # excludes it from that direction.
        start = _ms("2026-11-12T09:00:00+00:00")
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, start_ms=start)], ["entry 1", "entry 0"])
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, start_ms=start - 1)], ["entry 0"])
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, start_ms=start, sort_descending=False)],
            ["entry 1", "entry 2"])
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, start_ms=start + 1, sort_descending=False)],
            ["entry 2"])

    async def test_search_start_combines_with_query_filter(self):
        # start positions the scan; the text filter still decides what
        # matches inside it.
        for topic, entry, stamp in (
            ("Mood", "anxious morning", "2026-11-10T09:00:00+00:00"),
            ("Mood", "calm afternoon", "2026-11-12T09:00:00+00:00"),
            ("Sleep", "anxious night", "2026-11-14T09:00:00+00:00"),
        ):
            await self.store.add_entry(
                user_id=1, topic=topic, entry=entry,
                created=stamp, created_epoch_ms=_ms(stamp))
        start = _ms("2026-11-12T09:00:00+00:00")
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, start_ms=start, query="anxious")],
            ["anxious morning"])
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, start_ms=start, query="anxious",
            sort_descending=False)],
            ["anxious night"])
        self.assertEqual([r.entry for r in await self.store.search_entries(
            user_id=1, query="anxious")],
            ["anxious night", "anxious morning"])

    async def test_search_page_hints(self):
        stamps = [
            "2026-11-10T09:00:00+00:00",
            "2026-11-12T09:00:00+00:00",
            "2026-11-14T09:00:00+00:00",
        ]
        for i, stamp in enumerate(stamps):
            await self.store.add_entry(
                user_id=1, topic="Log", entry=f"entry {i}",
                created=stamp, created_epoch_ms=_ms(stamp))
        page = await self.store.search_page(user_id=1, cursor_id=3, limit=1)
        self.assertEqual([r.entry for r in page.entries], ["entry 1"])
        self.assertTrue(page.has_more)
        # The cursor is the page edge's id, computed once by the store
        # for the tool to report.
        self.assertEqual(page.next_cursor, 2)
        self.assertIsInstance(page.next_cursor, int)
        first = await self.store.search_page(user_id=1, limit=1)
        self.assertEqual([r.entry for r in first.entries], ["entry 2"])
        self.assertTrue(first.has_more)
        self.assertEqual(first.next_cursor, 3)
        last = await self.store.search_page(user_id=1, cursor_id=2)
        self.assertEqual([r.entry for r in last.entries], ["entry 0"])
        self.assertFalse(last.has_more)
        self.assertIsNone(last.next_cursor)
        asc_first = await self.store.search_page(
            user_id=1, limit=1, sort_descending=False)
        self.assertEqual([r.entry for r in asc_first.entries], ["entry 0"])
        self.assertTrue(asc_first.has_more)
        self.assertEqual(asc_first.next_cursor, 1)
        asc_last = await self.store.search_page(user_id=1, cursor_id=2,
                                                sort_descending=False)
        self.assertEqual([r.entry for r in asc_last.entries], ["entry 2"])
        self.assertFalse(asc_last.has_more)
        self.assertIsNone(asc_last.next_cursor)
        full = await self.store.search_page(user_id=1)
        self.assertEqual(len(full.entries), 3)
        self.assertFalse(full.has_more)
        self.assertIsNone(full.next_cursor)
        # A start before every entry has nothing at or before it to return.
        empty = await self.store.search_page(
            user_id=1, start_ms=_ms("2026-11-10T08:59:59+00:00"))
        self.assertEqual(empty.entries, [])
        self.assertFalse(empty.has_more)
        self.assertIsNone(empty.next_cursor)
        await self.store.add_entry(
            user_id=1, topic="Mood", entry="mood entry",
            created="2026-11-09T09:00:00+00:00",
            created_epoch_ms=_ms("2026-11-09T09:00:00+00:00"))
        topic_page = await self.store.search_page(
            user_id=1, topic="Log", limit=1)
        self.assertEqual([r.entry for r in topic_page.entries], ["entry 2"])
        self.assertTrue(topic_page.has_more)
        self.assertEqual(topic_page.next_cursor, 3)
        topic_full = await self.store.search_page(
            user_id=1, topic="Mood")
        self.assertEqual(len(topic_full.entries), 1)
        self.assertFalse(topic_full.has_more)
        self.assertIsNone(topic_full.next_cursor)

    async def test_search_page_pages_by_timestamp_with_backdated_rows(self):
        # A later id can carry an earlier timestamp (import, manual db
        # fix, a future tool passing explicit created): ordering must
        # follow created_epoch_ms, and the id-only cursor must still
        # resume at the right position. An id-ordered implementation
        # would start with "oldest" and fail this test.
        base = _ms("2026-11-14T09:00:00+00:00")
        newest = await self.store.add_entry(
            user_id=1, topic="A", entry="newest",
            created="2026-11-14T09:02:00+00:00",
            created_epoch_ms=base + 120_000)
        middle = await self.store.add_entry(
            user_id=1, topic="B", entry="middle",
            created="2026-11-14T09:00:00+00:00", created_epoch_ms=base)
        oldest = await self.store.add_entry(
            user_id=1, topic="C", entry="oldest",
            created="2026-11-14T08:58:00+00:00",
            created_epoch_ms=base - 120_000)
        self.assertEqual((newest.id, middle.id, oldest.id), (1, 2, 3))
        self.assertLess(oldest.created_epoch_ms, newest.created_epoch_ms)

        page = await self.store.search_page(user_id=1, limit=1)
        self.assertEqual([r.entry for r in page.entries], ["newest"])
        self.assertEqual(page.next_cursor, 1)
        page = await self.store.search_page(
            user_id=1, cursor_id=1, limit=1)
        self.assertEqual([r.entry for r in page.entries], ["middle"])
        self.assertEqual(page.next_cursor, 2)
        page = await self.store.search_page(
            user_id=1, cursor_id=2, limit=1)
        self.assertEqual([r.entry for r in page.entries], ["oldest"])
        self.assertFalse(page.has_more)
        self.assertIsNone(page.next_cursor)
        page = await self.store.search_page(user_id=1, limit=1,
                                            sort_descending=False)
        self.assertEqual([r.entry for r in page.entries], ["oldest"])
        page = await self.store.search_page(user_id=1, cursor_id=3, limit=1,
                                            sort_descending=False)
        self.assertEqual([r.entry for r in page.entries], ["middle"])
        self.assertEqual(page.next_cursor, 2)
        page = await self.store.search_page(user_id=1, cursor_id=2, limit=1,
                                            sort_descending=False)
        self.assertEqual([r.entry for r in page.entries], ["newest"])
        self.assertFalse(page.has_more)

    async def test_search_page_pages_inside_tied_milliseconds(self):
        # Three entries sharing one millisecond: the id cursor resumes
        # inside the tie, so no entry is skipped or repeated.
        created = "2026-11-14T09:00:00+00:00"
        epoch = _ms(created)
        for topic in ("A", "B", "C"):
            await self.store.add_entry(
                user_id=1, topic=topic, entry=topic,
                created=created, created_epoch_ms=epoch)
        page = await self.store.search_page(user_id=1, limit=2)
        self.assertEqual([r.id for r in page.entries], [3, 2])
        self.assertTrue(page.has_more)
        self.assertEqual(page.next_cursor, 2)
        page = await self.store.search_page(user_id=1, cursor_id=2, limit=2)
        self.assertEqual([r.id for r in page.entries], [1])
        self.assertFalse(page.has_more)
        self.assertIsNone(page.next_cursor)
        page = await self.store.search_page(user_id=1, limit=2,
                                            sort_descending=False)
        self.assertEqual([r.id for r in page.entries], [1, 2])
        self.assertTrue(page.has_more)
        self.assertEqual(page.next_cursor, 2)
        page = await self.store.search_page(user_id=1, cursor_id=2, limit=2,
                                            sort_descending=False)
        self.assertEqual([r.id for r in page.entries], [3])
        self.assertFalse(page.has_more)
        self.assertIsNone(page.next_cursor)

    async def test_search_page_rejects_stale_cursor(self):
        created = "2026-11-14T09:00:00+00:00"
        epoch = _ms(created)
        for topic in ("A", "B", "C"):
            await self.store.add_entry(
                user_id=1, topic=topic, entry=topic,
                created=created, created_epoch_ms=epoch)
        page = await self.store.search_page(user_id=1, limit=2)
        self.assertEqual(page.next_cursor, 2)
        # The edge row is gone: the id cannot be resolved to a position,
        # so paging fails loudly instead of guessing one.
        self.assertTrue(await self.store.delete_entry(user_id=1, entry_id=2))
        with self.assertRaises(StoreValidationError) as caught:
            await self.store.search_page(user_id=1, cursor_id=2, limit=2)
        self.assertEqual(str(caught.exception),
                         "Cursor entry no longer exists; restart paging.")
        with self.assertRaises(StoreValidationError):
            await self.store.search_page(user_id=2, cursor_id=3, limit=2)

    async def test_search_page_start_positions_then_cursor_resumes(self):
        # start places the first page inclusively - entries sitting on
        # that instant are returned, not skipped - and the cursor id
        # handed back resumes the scan from that exact row without a
        # timestamp of its own.
        created = "2026-11-14T09:00:00+00:00"
        epoch = _ms(created)
        for topic in ("A", "B", "C"):
            await self.store.add_entry(
                user_id=1, topic=topic, entry=topic,
                created=created, created_epoch_ms=epoch)
        await self.store.add_entry(
            user_id=1, topic="D", entry="older",
            created="2026-11-14T08:59:59+00:00",
            created_epoch_ms=epoch - 1000)
        await self.store.add_entry(
            user_id=1, topic="E", entry="newer",
            created="2026-11-14T09:00:01+00:00",
            created_epoch_ms=epoch + 1000)
        # ids 1..3 share the millisecond; "newer" is above start and
        # "older" is below it.
        page = await self.store.search_page(user_id=1, start_ms=epoch,
                                            limit=2)
        self.assertEqual([r.entry for r in page.entries], ["C", "B"])
        self.assertTrue(page.has_more)
        self.assertEqual(page.next_cursor, 2)
        page = await self.store.search_page(user_id=1,
                                            cursor_id=page.next_cursor,
                                            limit=2)
        self.assertEqual([r.entry for r in page.entries], ["A", "older"])
        self.assertFalse(page.has_more)
        self.assertIsNone(page.next_cursor)
        page = await self.store.search_page(user_id=1, start_ms=epoch,
                                            limit=2, sort_descending=False)
        self.assertEqual([r.entry for r in page.entries], ["A", "B"])
        self.assertTrue(page.has_more)
        self.assertEqual(page.next_cursor, 2)
        page = await self.store.search_page(user_id=1,
                                            cursor_id=page.next_cursor,
                                            limit=2, sort_descending=False)
        self.assertEqual([r.entry for r in page.entries], ["C", "newer"])
        self.assertFalse(page.has_more)
        self.assertIsNone(page.next_cursor)

    async def test_search_sorts_by_epoch_not_text(self):
        # Lexically the +11:00 stamp sorts after the Z stamp, but its
        # instant is earlier; ordering must follow created_epoch_ms, so a
        # lexical ORDER BY created DESC would return the opposite order.
        early = "2026-11-14T16:00:00+00:00"
        late = "2026-11-14T16:30:00+11:00"
        self.assertLess(early, late)  # lexical order disagrees with time
        await self.store.add_entry(
            user_id=1, topic="Log", entry="utc entry",
            created=early, created_epoch_ms=_ms(early))
        await self.store.add_entry(
            user_id=1, topic="Log", entry="sydney entry",
            created=late, created_epoch_ms=_ms(late))
        found = await self.store.search_entries(
            user_id=1)
        self.assertEqual([r.entry for r in found],
                         ["utc entry", "sydney entry"])

    async def test_delete_removes_fts_record(self):
        created = "2026-11-14T16:34:00+00:00"
        record = await self.store.add_entry(
            user_id=1, topic="Mood", entry="uniquewordxyz",
            created=created, created_epoch_ms=_ms(created))
        self.assertEqual(len(await self.store.search_entries(
            user_id=1, query="uniquewordxyz")), 1)
        self.assertTrue(await self.store.delete_entry(
            user_id=1, entry_id=record.id))
        self.assertFalse(await self.store.delete_entry(
            user_id=1, entry_id=record.id))
        self.assertEqual(await self.store.search_entries(
            user_id=1, query="uniquewordxyz"), [])

    async def test_list_topics(self):
        self.assertEqual(
            await self.store.list_topics(user_id=1), [])
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        await self.store.add_entry(
            user_id=1, topic="Sleep", entry="a",
            created=created, created_epoch_ms=epoch)
        await self.store.add_entry(
            user_id=1, topic="Mood", entry="b",
            created=created, created_epoch_ms=epoch)
        await self.store.add_entry(
            user_id=1, topic="mood", entry="c",
            created=created, created_epoch_ms=epoch)
        # Case-insensitive uniqueness, alphabetical.
        self.assertEqual(
            await self.store.list_topics(user_id=1),
            ["mood", "sleep"])
        # Topics vanish once unused: the sleep entry is the only one
        # with its topic, while "mood" survives its other entry.
        await self.store.delete_entry(
            user_id=1, entry_id=1)
        self.assertEqual(
            await self.store.list_topics(user_id=1),
            ["mood"])


class TestJournalTools(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "journal.db")
        patcher = mock.patch.object(
            sqlite_store.config, "sql_db_path", self.db)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.store = journal_store.create_journal_store(self.db)

    async def asyncTearDown(self):
        await self.store.close()

    def _ctx(self, user_id=1, timezone_name=None):
        settings = UserSettings.default()
        if timezone_name is not None:
            settings.timezone = timezone_name
        return mock_ctx(user_id=user_id, conversation=[],
                        settings=settings)

    async def test_add_uses_configured_timezone(self):
        # The entry's timestamp follows the user's configured timezone
        # (Sydney, +11:00-ish); a context with no settings loaded at all
        # falls back to UTC.
        offset = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%z")
        cases = [
            (self._ctx(timezone_name="Australia/Sydney"), offset),
            (mock_ctx(user_id=1, conversation=[]), "+0000"),
        ]
        for entry_id, (ctx, expected_suffix) in enumerate(cases, 1):
            result = await journal_tool.journal_add(
                ctx, topic="Mood", entry="Testing timezones.")
            self.assertTrue(
                result.startswith(f"Created journal entry id:{entry_id} "))
            record = await self.store.get_entry(
                user_id=1, entry_id=entry_id)
            assert record is not None
            expected_suffix = expected_suffix[:3] + ":" + expected_suffix[3:]
            self.assertTrue(record.created.endswith(expected_suffix))
            parsed = datetime.fromisoformat(record.created)
            self.assertAlmostEqual(parsed.timestamp() * 1000,
                                   record.created_epoch_ms, delta=1000)

    async def test_add_validation(self):
        ctx = self._ctx()
        self.assertEqual(
            await journal_tool.journal_add(ctx, topic="  ", entry="x"),
            "Error: Topic cannot be empty")
        self.assertEqual(
            await journal_tool.journal_add(ctx, topic="Mood", entry="  "),
            "Error: Entry cannot be empty")
        self.assertEqual(
            await journal_tool.journal_add(ctx, topic="t" * 129, entry="x"),
            "Error: Topic must be at most 128 characters")
        # Non-string input is validated by the store too; the tool
        # translates the typed error verbatim.
        self.assertEqual(
            await journal_tool.journal_add(ctx, topic=123, entry="x"),  # type: ignore[arg-type]
            "Error: Topic cannot be empty")

    async def test_get_and_delete(self):
        ctx = self._ctx()
        await journal_tool.journal_add(ctx, topic="Mood", entry="Hi.")
        out = await journal_tool.journal_get(ctx, ids=[1])
        self.assertIn("ID: 1", out)
        self.assertIn("Topic: mood", out)
        self.assertIn("Entry: Hi.", out)
        self.assertIn("Created: ", out)
        self.assertEqual(
            await journal_tool.journal_get(ctx, ids=[999]),
            "Journal entry 999 not found")
        self.assertEqual(await journal_tool.journal_get(ctx, ids=[]),
                         "Error: ids cannot be empty.")
        self.assertEqual(await journal_tool.journal_get(ctx, ids="nope"),  # type: ignore[arg-type]
                         "Error: ids must be an array of entry IDs.")
        self.assertEqual(await journal_tool.journal_delete(ctx, id=1),
                         "Journal entry 1 deleted")
        self.assertEqual(await journal_tool.journal_delete(ctx, id=1),
                         "Journal entry 1 not found")

    async def test_cross_user_tools_see_nothing(self):
        await journal_tool.journal_add(
            self._ctx(user_id=1), topic="Mood", entry="private")
        other = self._ctx(user_id=2)
        self.assertEqual(await journal_tool.journal_get(other, ids=[1]),
                         "Journal entry 1 not found")
        self.assertEqual(await journal_tool.journal_delete(other, id=1),
                         "Journal entry 1 not found")
        self.assertEqual(
            await journal_tool.journal_edit(
                other, id=1, oldString="private", newString="x"),
            "Journal entry 1 not found")
        self.assertEqual(await journal_tool.journal_query(other, query="private"),
                         "No journal entries found.")
        self.assertEqual(await journal_tool.journal_list_topics(other),
                         "No journal topics yet.")

    async def test_replace_tool(self):
        ctx = self._ctx()
        await journal_tool.journal_add(ctx, topic="Mood", entry="Felt bad.")
        # One store-raised validation surfaced verbatim, two outcomes the
        # tool translates itself; the full branch sweep is at store level.
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=1, oldString="bad", newString="bad"),
            "Error: No changes to apply: "
            "oldString and newString are identical.")
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=1, oldString="great", newString="okay"),
            "Text not found in journal entry 1")
        # An id that exists for nobody but this user still gets this
        # user's wording, not a raw store error.
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=404, oldString="bad", newString="okay"),
            "Journal entry 404 not found")
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=1, oldString="bad", newString="okay"),
            "Journal entry 1 updated")
        self.assertIn("Entry: Felt okay.",
                      await journal_tool.journal_get(ctx, ids=[1]))

    async def test_get_multiple_ids(self):
        ctx = self._ctx()
        await journal_tool.journal_add(ctx, topic="Mood", entry="one")
        await journal_tool.journal_add(ctx, topic="Sleep", entry="two")
        await journal_tool.journal_add(ctx, topic="Log", entry="three")
        out = await journal_tool.journal_get(ctx, ids=[1, 3])
        self.assertIn("Found 2 journal entries:", out)
        self.assertIn("ID: 1", out)
        self.assertIn("ID: 3", out)
        self.assertNotIn("ID: 2\n", out)
        self.assertLess(out.index("ID: 1"), out.index("ID: 3"))
        flipped = await journal_tool.journal_get(ctx, ids=[3, 1])
        self.assertLess(flipped.index("ID: 3"), flipped.index("ID: 1"))
        partial = await journal_tool.journal_get(ctx, ids=[1, 999])
        self.assertIn("ID: 1", partial)
        self.assertIn("Journal entries not found: 999", partial)
        self.assertEqual(
            await journal_tool.journal_get(ctx, ids=[998, 999]),
            "Journal entries not found: 998, 999")
        dupes = await journal_tool.journal_get(ctx, ids=[1, 1])
        self.assertIn("ID: 1", dupes)
        self.assertNotIn("Found", dupes)

    async def test_edit_tool_multiple_matches(self):
        ctx = self._ctx()
        await journal_tool.journal_add(
            ctx, topic="Log", entry="ha ha ha")
        out = await journal_tool.journal_edit(
            ctx, id=1, oldString="ha", newString="he")
        self.assertTrue(out.startswith("Error: Found multiple matches"))
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=1, oldString="ha", newString="he",
                replaceAll=True),
            "Journal entry 1 updated")
        self.assertIn("Entry: he he he",
                      await journal_tool.journal_get(ctx, ids=[1]))

    async def test_topic_edit_tool(self):
        ctx = self._ctx()
        await journal_tool.journal_add(
            ctx, topic="Mood", entry="Feeling okay.")
        # One store-raised validation surfaced verbatim, one outcome the
        # tool translates itself; the full branch sweep is at store level.
        self.assertEqual(
            await journal_tool.journal_topic_edit(
                ctx, id=1, oldTopic="Mood", newTopic="Mood"),
            "Error: No changes to apply: "
            "oldTopic and newTopic are identical.")
        self.assertEqual(
            await journal_tool.journal_topic_edit(
                ctx, id=1, oldTopic="Wrong", newTopic="Sleep"),
            "Topic does not match journal entry 1")
        # An id that belongs to nobody still gets this user's wording.
        self.assertEqual(
            await journal_tool.journal_topic_edit(
                ctx, id=404, oldTopic="Mood", newTopic="Sleep"),
            "Journal entry 404 not found")
        self.assertEqual(
            await journal_tool.journal_topic_edit(
                ctx, id=1, oldTopic="Mood", newTopic="Sleep"),
            "Journal entry 1 topic updated")
        self.assertIn("Topic: sleep",
                      await journal_tool.journal_get(ctx, ids=[1]))

    async def test_search_tool_validation_and_format(self):
        ctx = self._ctx()
        await journal_tool.journal_add(ctx, topic="Mood", entry="Calm day.")
        # start is the datetime parameter: it must parse as ISO 8601
        # with a timezone.
        self.assertEqual(
            await journal_tool.journal_query(ctx, start="2026-11-01"),
            "Error: start timestamp must include an explicit timezone "
            "or UTC indicator, e.g. '2026-11-01T00:00:00+10:30'.")
        self.assertEqual(
            await journal_tool.journal_query(ctx, start="not a date"),
            "Error: Invalid start timestamp 'not a date'. "
            "Use ISO 8601, e.g. '2026-11-01T00:00:00+10:30'.")
        # cursor and start both position the same scan: the caller must
        # pick one.
        self.assertEqual(
            await journal_tool.journal_query(
                ctx, cursor=1, start="2026-11-01T00:00:00+10:30"),
            "Error: Provide either cursor or start, not both: cursor "
            "resumes paging from an entry id, start positions the "
            "first page by timestamp.")
        # limit stays a plain int with its range check; anything else is
        # a boundary type error, not a coerced value.
        self.assertEqual(
            await journal_tool.journal_query(ctx, limit=0),
            "Error: limit must be an integer between 1 and 50.")
        self.assertEqual(
            await journal_tool.journal_query(ctx, limit=51),
            "Error: limit must be an integer between 1 and 50.")
        self.assertEqual(
            await journal_tool.journal_query(ctx, query="nothing matching"),
            "No journal entries found.")
        out = await journal_tool.journal_query(ctx, query="calm")
        self.assertIn("Found 1 journal entry:", out)
        self.assertIn("ID: 1", out)

    async def test_search_tool_start_filter_with_offsets(self):
        ctx = self._ctx(timezone_name="Australia/Adelaide")
        await journal_tool.journal_add(ctx, topic="Log", entry="sydney entry")
        record = await self.store.get_entry(
            user_id=1, entry_id=1)
        assert record is not None
        inside = await journal_tool.journal_query(
            ctx, start="2030-01-01T00:00:00+10:30")
        self.assertIn("sydney entry", inside)
        day = record.created[:10]
        outside = await journal_tool.journal_query(
            ctx, start=f"{day}T00:00:00+10:30")
        self.assertEqual(outside, "No journal entries found.")
        # Same start oldest-first finds it (entries at or after start).
        future = await journal_tool.journal_query(
            ctx, start=f"{day}T00:00:00+10:30", sortDescending=False)
        self.assertIn("sydney entry", future)
        # start is inclusive: exactly the entry's own timestamp returns
        # it in both directions.
        self.assertIn(
            "sydney entry",
            await journal_tool.journal_query(ctx, start=record.created))
        self.assertIn(
            "sydney entry",
            await journal_tool.journal_query(ctx, start=record.created,
                                             sortDescending=False))

    async def test_query_tool_paging(self):
        ctx = self._ctx()
        stamps = [
            "2026-11-10T09:00:00+00:00",
            "2026-11-12T09:00:00+00:00",
            "2026-11-14T09:00:00+00:00",
        ]
        for i, stamp in enumerate(stamps):
            await self.store.add_entry(
                user_id=1, topic="Log", entry=f"entry {i}",
                created=stamp, created_epoch_ms=_ms(stamp))
        # start positions the first page inclusively: the entry sitting
        # on it comes back, and nothing more exists in that direction.
        out = await journal_tool.journal_query(
            ctx, start="2026-11-12T09:00:00+00:00")
        self.assertIn("Found 2 journal entries:", out)
        self.assertIn("entry 1", out)
        self.assertIn("entry 0", out)
        self.assertNotIn("More journal entries exist", out)
        # Truncated start page hints with the edge entry's id, as a
        # number the caller passes back as cursor.
        trunc = await journal_tool.journal_query(
            ctx, start="2026-11-14T09:00:00+00:00", limit=1)
        self.assertIn("entry 2", trunc)
        self.assertIn("More journal entries exist.", trunc)
        self.assertIn("cursor=3", trunc)
        full = await journal_tool.journal_query(ctx)
        self.assertIn("Found 3 journal entries:", full)
        self.assertNotIn("More journal entries exist", full)
        first = await journal_tool.journal_query(ctx, limit=1)
        self.assertIn("entry 2", first)
        self.assertIn("More journal entries exist.", first)
        self.assertIn("cursor=3", first)
        # Feeding the hint back resumes after the page edge: no repeat,
        # no skip.
        second = await journal_tool.journal_query(ctx, cursor=3, limit=1)
        self.assertIn("entry 1", second)
        self.assertNotIn("entry 2", second)
        self.assertIn("More journal entries exist.", second)
        self.assertIn("cursor=2", second)
        third = await journal_tool.journal_query(ctx, cursor=2, limit=1)
        self.assertIn("entry 0", third)
        self.assertNotIn("More journal entries exist", third)
        oldest = await journal_tool.journal_query(
            ctx, limit=1, sortDescending=False)
        self.assertIn("entry 0", oldest)
        self.assertIn("More journal entries exist.", oldest)
        self.assertIn("cursor=1", oldest)
        oldest_next = await journal_tool.journal_query(
            ctx, cursor=1, sortDescending=False, limit=1)
        self.assertIn("entry 1", oldest_next)
        self.assertIn("cursor=2", oldest_next)

    async def test_query_tool_reports_stale_cursor(self):
        ctx = self._ctx()
        await journal_tool.journal_add(ctx, topic="Mood", entry="one")
        first = await journal_tool.journal_query(ctx, limit=1)
        self.assertIn("ID: 1", first)
        await journal_tool.journal_delete(ctx, id=1)
        # The id names a deleted entry: the store refuses to guess a
        # position and the tool surfaces the message verbatim.
        self.assertEqual(
            await journal_tool.journal_query(ctx, cursor=1, limit=1),
            "Error: Cursor entry no longer exists; restart paging.")
        # An id that was never issued behaves the same way.
        self.assertEqual(
            await journal_tool.journal_query(ctx, cursor=99, limit=1),
            "Error: Cursor entry no longer exists; restart paging.")

    async def test_list_topics_tool(self):
        ctx = self._ctx()
        self.assertEqual(await journal_tool.journal_list_topics(ctx),
                         "No journal topics yet.")
        await journal_tool.journal_add(ctx, topic="Sleep", entry="a")
        await journal_tool.journal_add(ctx, topic="Mood", entry="b")
        self.assertEqual(await journal_tool.journal_list_topics(ctx),
                         "mood\nsleep")

    def test_all_seven_tools_are_declared(self):
        from tool_registry import BOTH, all_tools
        declared = {t.name: t for t in all_tools()
                    if t.name.startswith('journal_')}
        self.assertEqual(sorted(declared), [
            "journal_add", "journal_delete", "journal_edit", "journal_get",
            "journal_list_topics", "journal_query", "journal_topic_edit"])
        for name, tool in declared.items():
            with self.subTest(tool=name):
                self.assertEqual(tool.scope, BOTH)
                self.assertTrue(getattr(tool.func, "_needs_ctx", False))
        # journal_query's paging parameters: cursor is an id, an integer
        # the model never has to quote, while start is a datetime and
        # stays text.
        query = declared["journal_query"]
        self.assertEqual(query.properties["cursor"]["type"], "integer")
        self.assertEqual(query.properties["start"]["type"], "string")


if __name__ == '__main__':
    unittest.main()
