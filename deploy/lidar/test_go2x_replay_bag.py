import unittest

from .go2x_replay_bag import nearest_pairs


class Go2XReplayBagTests(unittest.TestCase):
    def test_closest_odom_on_either_side_and_at_boundaries(self):
        self.assertEqual(nearest_pairs([9, 12, 18, 21], [10, 20], 5),
                         [(0, 0, 1), (1, 0, 2), (2, 1, 2), (3, 1, 1)])

    def test_rejects_samples_at_or_outside_tolerance(self):
        self.assertEqual(nearest_pairs([5, 10, 30], [10, 20], 5), [(1, 0, 0)])

    def test_rejects_invalid_time_series(self):
        for invalid in ([], [0, 2], [1, 1], [2, 1]):
            with self.subTest(stamps=invalid), self.assertRaises(ValueError):
                nearest_pairs(invalid, [1, 2])
            with self.subTest(stamps=invalid), self.assertRaises(ValueError):
                nearest_pairs([1, 2], invalid)
