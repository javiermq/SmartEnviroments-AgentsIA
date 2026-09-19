import csv
import json
import unittest
from datetime import datetime, timedelta

from smart_home_agent.context import DATA_DIR, build_context, system_prompt
from smart_home_agent.conversation_backend import ConversationBackend
from smart_home_agent.conversation_demo import _system_prompt
from smart_home_agent.sensor_queries import query_aggregation


class ContextTests(unittest.TestCase):
    def test_snapshot_profile_and_current_activity(self):
        context = build_context("2026-09-16T18:56:30+02:00")
        self.assertEqual(context["user"]["nombre"], "Mariola")
        self.assertEqual(context["sensors_at_t0"]["timestamp"], "2026-09-16T18:56:00+02:00")
        self.assertEqual(context["current_activities"], [{"name": "cooking", "start_time": "2026-09-16T18:40:00+02:00"}])
        self.assertNotIn("uwb_distance_kitchen_m", context["sensors_at_t0"])

    def test_boundary_switches_activity_without_future_leak(self):
        context = build_context("2026-09-16T19:10:00+02:00")
        self.assertEqual(context["current_activities"][0]["name"], "resting")
        self.assertEqual(context["activities_last_12h"][-1]["name"], "cooking")
        for activity in context["activities_last_12h"]:
            self.assertLessEqual(activity["end_time"], context["t0"])
        self.assertNotIn("end_time", context["current_activities"][0])

    def test_midnight_history_and_aggregation(self):
        context = build_context("2026-09-16T06:00:00+02:00")
        history = context["activities_last_12h"]
        self.assertEqual(history[0]["start_time"], "2026-09-15T18:00:00+02:00")
        self.assertEqual(history[-1]["end_time"], context["t0"])
        self.assertTrue(history[-1]["ongoing_at_t0"])
        result = query_aggregation("watch.sleep", "2026-09-15T22:15:00+02:00", context["t0"], "total")
        self.assertEqual(result["samples"], 465)
        self.assertEqual(result["result"], 465)

    def test_missing_data_and_timezone(self):
        context = build_context("2026-09-18T06:00:00+02:00")
        self.assertIsNone(context["sensors_at_t0"])
        self.assertEqual(context["current_activities"], [])
        self.assertEqual(context["activities_last_12h"], [])
        equivalent = build_context("2026-09-16T16:56:00Z")
        self.assertEqual(equivalent["current_activities"][0]["name"], "cooking")
        with self.assertRaises(ValueError):
            build_context("2026-09-16T18:56:00")

    def test_proximity_values_and_missing_readings(self):
        with (DATA_DIR / "simulated_sensor_data_2026-09-16.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 1440)
        self.assertEqual(float(rows[0]["cercania_distance_bedroom"]), 0.881)
        missing = []
        for row in rows:
            self.assertFalse(any(key.startswith("uwb_distance_") for key in row))
            for key, value in row.items():
                if key.startswith("cercania_distance_"):
                    if value:
                        self.assertTrue(0 <= float(value) <= 1)
                    else:
                        missing.append((row["timestamp"], key))
        self.assertTrue(missing)
        timestamp, key = missing[0]
        self.assertIsNone(build_context(timestamp)["sensors_at_t0"][key])

    def test_previous_day_complete(self):
        for kind, timestamp in (("sensor_data", "timestamp"), ("activities", "start_time")):
            records = []
            for day in (15, 16):
                with (DATA_DIR / f"simulated_{kind}_2026-09-{day}.tsv").open() as handle:
                    records.append(list(csv.DictReader(handle, delimiter="\t")))
            self.assertEqual(len(records[0]), len(records[1]))
            for previous, current in zip(*records):
                self.assertEqual(datetime.fromisoformat(current[timestamp]) - datetime.fromisoformat(previous[timestamp]), timedelta(days=1))

    def test_both_entrypoints_include_context(self):
        t0 = "2026-09-16T18:56:00+02:00"
        prompt = system_prompt(t0)
        self.assertEqual(_system_prompt(t0), prompt)
        backend = ConversationBackend(t0)
        self.assertEqual(backend.messages[0]["content"], prompt)
        context = json.loads(prompt.split("CONTEXTO_JSON:\n", 1)[1])
        self.assertTrue(context["activities_last_12h"])

    def test_short_time_uses_session_day(self):
        backend = ConversationBackend("2026-09-15T10:00:00+02:00")
        responses = iter([
            {"content": "", "tool_calls": [{"function": {"name": "query_aggregation", "arguments": {
                "sensor_type": "watch.steps", "time_init": "09:00", "time_end": "09:40", "aggregation": "total"}}}]},
            {"content": "Respuesta"},
        ])
        backend._request_ollama = lambda: next(responses)
        backend.respond("Pasos")
        result = json.loads(next(m["content"] for m in backend.messages if m.get("role") == "tool"))
        self.assertTrue(result["time_init"].startswith("2026-09-15"))
