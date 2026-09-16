import unittest

from smart_home_agent.sensor_queries import query_aggregation


class QueryAggregationTests(unittest.TestCase):
    def test_sleep_total_for_full_day(self):
        result = query_aggregation("watch.sleep", "00:00", "24:00", "total")
        self.assertEqual(result["result"], 525.0)
        self.assertEqual(result["samples"], 1440)
        self.assertEqual(result["unit"], "minutes")

    def test_sport_steps(self):
        result = query_aggregation("watch.steps", "09:00", "09:40", "total")
        self.assertEqual(result["samples"], 40)
        self.assertGreater(result["result"], 3000)

    def test_invalid_aggregation_is_rejected(self):
        with self.assertRaises(ValueError):
            query_aggregation("watch.steps", "09:00", "09:40", "median")


if __name__ == "__main__":
    unittest.main()
