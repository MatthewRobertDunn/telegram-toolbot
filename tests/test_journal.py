import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import journal_store
from journal_store import ReplaceOutcome
from tools import journal_tool
from mock_context import mock_ctx
from user_settings import UserSettings


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
        self.assertEqual(fetched.topic, "Mood")
        self.assertEqual(fetched.entry, "Felt okay today.")
        second = await self.store.add_entry(
            user_id=1, topic="Sleep", entry="Slept well.",
            created=created, created_epoch_ms=_ms(created))
        self.assertEqual(second.id, 2)

    async def test_add_rejects_empty_and_long(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        for bad_topic in ("", "   "):
            with self.assertRaises(ValueError):
                await self.store.add_entry(
                    user_id=1, topic=bad_topic, entry="x",
                    created=created, created_epoch_ms=epoch)
        for bad_entry in ("", "  \n "):
            with self.assertRaises(ValueError):
                await self.store.add_entry(
                    user_id=1, topic="Mood", entry=bad_entry,
                    created=created, created_epoch_ms=epoch)
        with self.assertRaises(ValueError):
            await self.store.add_entry(
                user_id=1, topic="t" * 129, entry="x",
                created=created, created_epoch_ms=epoch)
        record = await self.store.add_entry(
            user_id=1, topic="t" * 128, entry="x",
            created=created, created_epoch_ms=epoch)
        self.assertEqual(len(record.topic), 128)
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
        # Owner still sees everything.
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
        # Creation timestamp is untouched.
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

    async def test_replace_identical_rejected(self):
        created = "2026-11-14T16:34:00+00:00"
        epoch = _ms(created)
        record = await self.store.add_entry(
            user_id=1, topic="Mood", entry="Same text.",
            created=created, created_epoch_ms=epoch)
        with self.assertRaises(ValueError):
            await self.store.replace_in_entry(
                user_id=1, entry_id=record.id, old_string="Same",
                new_string="Same")

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
        # Entry unchanged after ambiguous edit.
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
        self.assertEqual(narrow[0].topic, "Sleep")
        # Quoted phrase.
        phrase = await self.store.search_entries(
            user_id=1, query='"anxious dreams"')
        self.assertEqual(len(phrase), 1)
        # Prefix search.
        prefix = await self.store.search_entries(
            user_id=1, query="anx")
        self.assertEqual(len(prefix), 2)
        # Case-insensitive.
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

    async def test_search_date_range_and_order(self):
        stamps = [
            "2026-11-10T09:00:00+00:00",
            "2026-11-12T09:00:00+00:00",
            "2026-11-14T09:00:00+00:00",
        ]
        for i, stamp in enumerate(stamps):
            await self.store.add_entry(
                user_id=1, topic="Log", entry=f"entry {i}",
                created=stamp, created_epoch_ms=_ms(stamp))
        window = await self.store.search_entries(
            user_id=1, start_ms=_ms("2026-11-11T00:00:00+00:00"),
            end_ms=_ms("2026-11-15T00:00:00+00:00"))
        # Newest first.
        self.assertEqual([r.entry for r in window], ["entry 2", "entry 1"])
        # Start inclusive, end exclusive.
        self.assertEqual(len(await self.store.search_entries(
            user_id=1, start_ms=_ms("2026-11-12T09:00:00+00:00"))), 2)
        self.assertEqual(len(await self.store.search_entries(
            user_id=1, end_ms=_ms("2026-11-12T09:00:00+00:00"))), 1)
        with self.assertRaises(ValueError):
            await self.store.search_entries(
                user_id=1, start_ms=10, end_ms=10)
        limited = await self.store.search_entries(
            user_id=1, limit=1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0].entry, "entry 2")

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
        first = await self.store.add_entry(
            user_id=1, topic="Mood", entry="b",
            created=created, created_epoch_ms=epoch)
        await self.store.add_entry(
            user_id=1, topic="mood", entry="c",
            created=created, created_epoch_ms=epoch)
        # Case-insensitive uniqueness, alphabetical.
        self.assertEqual(
            await self.store.list_topics(user_id=1),
            ["Mood", "Sleep"])
        # Topics vanish once unused.
        await self.store.delete_entry(
            user_id=1, entry_id=first.id)
        self.assertEqual(
            await self.store.list_topics(user_id=1),
            ["mood", "Sleep"])


class TestJournalTools(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "journal.db")
        patcher = mock.patch.object(
            journal_store.config, "memory_db_path", self.db)
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
        ctx = self._ctx(timezone_name="Australia/Sydney")
        result = await journal_tool.journal_add(
            ctx, topic="Mood", entry="Testing timezones.")
        self.assertTrue(result.startswith("Created journal entry id:1 "))
        record = await self.store.get_entry(
            user_id=1, entry_id=1)
        assert record is not None
        expected_offset = datetime.now(
            ZoneInfo("Australia/Sydney")).strftime("%z")
        self.assertTrue(record.created.endswith(
            expected_offset[:3] + ":" + expected_offset[3:]))
        parsed = datetime.fromisoformat(record.created)
        self.assertAlmostEqual(parsed.timestamp() * 1000,
                               record.created_epoch_ms, delta=1000)

    async def test_add_defaults_to_utc_without_settings(self):
        ctx = mock_ctx(user_id=1, conversation=[])
        result = await journal_tool.journal_add(
            ctx, topic="Mood", entry="No settings loaded.")
        self.assertTrue(result.startswith("Created journal entry id:1 "))
        record = await self.store.get_entry(
            user_id=1, entry_id=1)
        assert record is not None
        self.assertTrue(record.created.endswith("+00:00"))

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

    async def test_get_and_delete(self):
        ctx = self._ctx()
        await journal_tool.journal_add(ctx, topic="Mood", entry="Hi.")
        out = await journal_tool.journal_get(ctx, id=1)
        self.assertIn("ID: 1", out)
        self.assertIn("Topic: Mood", out)
        self.assertIn("Entry: Hi.", out)
        self.assertIn("Created: ", out)
        self.assertEqual(
            await journal_tool.journal_get(ctx, id=999),
            "Journal entry 999 not found")
        self.assertEqual(await journal_tool.journal_get(ctx, id="nope"),
                         "Error: Invalid journal entry id.")
        self.assertEqual(await journal_tool.journal_delete(ctx, id=1),
                         "Journal entry 1 deleted")
        self.assertEqual(await journal_tool.journal_delete(ctx, id=1),
                         "Journal entry 1 not found")

    async def test_cross_user_tools_see_nothing(self):
        await journal_tool.journal_add(
            self._ctx(user_id=1), topic="Mood", entry="private")
        other = self._ctx(user_id=2)
        self.assertEqual(await journal_tool.journal_get(other, id=1),
                         "Journal entry 1 not found")
        self.assertEqual(await journal_tool.journal_delete(other, id=1),
                         "Journal entry 1 not found")
        self.assertEqual(
            await journal_tool.journal_edit(
                other, id=1, oldString="private", newString="x"),
            "Journal entry 1 not found")
        self.assertEqual(await journal_tool.journal_search(other, query="private"),
                         "No journal entries found.")
        self.assertEqual(await journal_tool.journal_list_topics(other),
                         "No journal topics yet.")

    async def test_replace_tool(self):
        ctx = self._ctx()
        await journal_tool.journal_add(ctx, topic="Mood", entry="Felt bad.")
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=1, oldString="", newString="x"),
            "Error: oldString cannot be empty.")
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=1, oldString="bad", newString="bad"),
            "Error: No changes to apply: "
            "oldString and newString are identical.")
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=999, oldString="a", newString="b"),
            "Journal entry 999 not found")
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=1, oldString="great", newString="okay"),
            "Text not found in journal entry 1")
        self.assertEqual(
            await journal_tool.journal_edit(
                ctx, id=1, oldString="bad", newString="okay"),
            "Journal entry 1 updated")
        self.assertIn("Entry: Felt okay.",
                      await journal_tool.journal_get(ctx, id=1))

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
                      await journal_tool.journal_get(ctx, id=1))

    async def test_search_tool_validation_and_format(self):
        ctx = self._ctx()
        await journal_tool.journal_add(ctx, topic="Mood", entry="Calm day.")
        self.assertEqual(
            await journal_tool.journal_search(ctx, start="2026-11-01"),
            "Error: start timestamp must include an explicit timezone "
            "or UTC indicator, e.g. '2026-11-01T00:00:00+10:30'.")
        self.assertEqual(
            await journal_tool.journal_search(ctx, start="not a date"),
            "Error: Invalid start timestamp 'not a date'. "
            "Use ISO 8601, e.g. '2026-11-01T00:00:00+10:30'.")
        self.assertEqual(
            await journal_tool.journal_search(
                ctx, start="2026-11-15T00:00:00+00:00",
                end="2026-11-01T00:00:00+00:00"),
            "Error: Start timestamp must be earlier than end timestamp")
        self.assertEqual(
            await journal_tool.journal_search(ctx, query="nothing matching"),
            "No journal entries found.")
        out = await journal_tool.journal_search(ctx, query="calm")
        self.assertIn("Found 1 journal entry:", out)
        self.assertIn("ID: 1", out)

    async def test_search_tool_date_filter_with_offsets(self):
        ctx = self._ctx(timezone_name="Australia/Adelaide")
        await journal_tool.journal_add(ctx, topic="Log", entry="sydney entry")
        record = await self.store.get_entry(
            user_id=1, entry_id=1)
        assert record is not None
        day = record.created[:10]
        inside = await journal_tool.journal_search(
            ctx, start=f"{day}T00:00:00+10:30",
            end="2030-01-01T00:00:00+10:30")
        self.assertIn("sydney entry", inside)
        outside = await journal_tool.journal_search(
            ctx, start="2030-01-01T00:00:00+10:30")
        self.assertEqual(outside, "No journal entries found.")

    async def test_list_topics_tool(self):
        ctx = self._ctx()
        self.assertEqual(await journal_tool.journal_list_topics(ctx),
                         "No journal topics yet.")
        await journal_tool.journal_add(ctx, topic="Sleep", entry="a")
        await journal_tool.journal_add(ctx, topic="Mood", entry="b")
        self.assertEqual(await journal_tool.journal_list_topics(ctx),
                         "Mood\nSleep")

    async def test_register_all_six_tools(self):
        definitions: list[dict] = []
        tool_map: dict = {}
        journal_tool.register(definitions, tool_map)
        names = [d["function"]["name"] for d in definitions]
        self.assertEqual(names, ["journal_add", "journal_get",
                                 "journal_delete", "journal_edit",
                                 "journal_search", "journal_list_topics"])
        for name in names:
            self.assertIn(name, tool_map)
            self.assertTrue(getattr(tool_map[name], "_requires_context",
                                    False))


if __name__ == '__main__':
    unittest.main()
