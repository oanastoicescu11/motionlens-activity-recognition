import unittest

from app.backend.core.ingest_parser import (
    extract_accelerometer_points,
    parse_ingest_batch,
)


def _valid_body() -> dict:
    return {
        "source": "web-devicemotion",
        "messageId": 11,
        "sessionId": "sess-1",
        "deviceId": "dev-1",
        "samples": [
            {
                "timestampMs": 1698501144401,
                "acc": {"x": 0.11, "y": -0.02, "z": -0.31},
            },
            {
                "timestampMs": 1698501144421,
                "acc": {"x": 0.21, "y": 0.01, "z": -0.11},
            },
        ],
    }


class IngestSchemaTests(unittest.TestCase):
    def test_parses_valid_batch(self) -> None:
        parsed = parse_ingest_batch(_valid_body())
        self.assertEqual(parsed.message_id, 11)
        self.assertEqual(parsed.session_id, "sess-1")
        self.assertEqual(len(parsed.samples), 2)

    def test_accepts_optional_placement_label(self) -> None:
        body = _valid_body()
        body["placementLabel"] = "front_pocket"
        parsed = parse_ingest_batch(body)
        self.assertEqual(parsed.device_id, "dev-1")

    def test_parses_optional_gyro_payload(self) -> None:
        body = _valid_body()
        body["samples"][0]["gyro"] = {"alpha": 1.1, "beta": -2.2, "gamma": 3.3}
        parsed = parse_ingest_batch(body)
        self.assertIsNotNone(parsed.samples[0].gyro)
        assert parsed.samples[0].gyro is not None
        self.assertAlmostEqual(parsed.samples[0].gyro.alpha, 1.1)
        self.assertAlmostEqual(parsed.samples[0].gyro.beta, -2.2)
        self.assertAlmostEqual(parsed.samples[0].gyro.gamma, 3.3)

    def test_extracts_accelerometer_only(self) -> None:
        parsed = parse_ingest_batch(_valid_body())
        points = extract_accelerometer_points(parsed)
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0][0], 1698501144401000000)

    def test_rejects_missing_samples(self) -> None:
        body = _valid_body()
        del body["samples"]
        with self.assertRaises(ValueError):
            parse_ingest_batch(body)

    def test_parse_ingest_batch_rejects_unknown_shape(self) -> None:
        with self.assertRaises(ValueError):
            parse_ingest_batch({"messageId": 1, "sessionId": "s", "deviceId": "d"})


if __name__ == "__main__":
    unittest.main()
