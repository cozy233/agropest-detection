"""Regression checks for annotation conversions and invalid input handling."""

import unittest

from labels import parse_label


class AnnotationTests(unittest.TestCase):
    def test_normalized_box_and_background_offset(self):
        label, box = parse_label("0 0.5 0.5 0.2 0.4", 100, 50)
        self.assertEqual(label, 1)
        for actual, expected in zip(box, [40, 15, 60, 35]):
            self.assertAlmostEqual(actual, expected)

    def test_border_box_is_clipped(self):
        self.assertEqual(parse_label("11 0 1 0.2 0.2", 100, 100), (12, [0, 90, 10, 100]))

    def test_empty_line(self):
        self.assertIsNone(parse_label("  ", 100, 50))

    def test_bad_annotations_are_rejected(self):
        for line in ["12 .5 .5 .2 .2", "1.5 .5 .5 .2 .2", "0 nan .5 .2 .2",
                     "0 .5 .5 0 .2", "0 .5 .5 .2", "0 1.5 .5 .2 .2"]:
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_label(line, 100, 50)


if __name__ == "__main__":
    unittest.main()
