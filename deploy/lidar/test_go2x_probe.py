import unittest
from types import SimpleNamespace

from .go2x_probe import CloudProbe


def cloud(stamp, frame='lidar', width=1, fields=None, data=None):
    if fields is None:
        fields = [SimpleNamespace(name=axis, offset=4 * index, datatype=7, count=1)
                  for index, axis in enumerate('xyz')]
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=1, nanosec=stamp)),
        width=width, height=1, point_step=12, row_step=12 * width, fields=fields,
        data=bytes(12 * width) if data is None else data,
    )


class Go2XProbeTests(unittest.TestCase):
    def test_no_or_only_one_message_is_inconclusive(self):
        probe = CloudProbe()
        self.assertEqual(probe.report(10.)['issues'], {'insufficient_samples': 1})
        probe.observe(cloud(10), 2_000_000_000)
        self.assertEqual(probe.report(10.)['status'], 'check_failed')

    def test_accepts_variable_point_count(self):
        probe = CloudProbe()
        probe.observe(cloud(10), 2_000_000_000)
        probe.observe(cloud(20, width=3), 2_000_000_000)
        report = probe.report(1.)
        self.assertEqual(report['status'], 'cloud_metadata_ok')
        self.assertEqual(report['frames'], ['lidar'])
        self.assertEqual(report['last_layout']['data_bytes'], 36)
        self.assertAlmostEqual(report['last_header_age_s'], 0.99999998)

    def test_rejects_mixed_frames_and_duplicate_or_regressing_stamps(self):
        probe = CloudProbe()
        for message in (cloud(10), cloud(10, frame='other'), cloud(9)):
            probe.observe(message, 2_000_000_000)
        self.assertEqual(probe.report(1.)['issues'], {
            'multiple_frames': 1, 'non_increasing_stamp': 2,
        })

    def test_rejects_invalid_metadata(self):
        probe = CloudProbe()
        message = cloud(0, frame='', width=0, fields=[], data=b'invalid')
        message.header.stamp.sec = 0
        probe.observe(message, 2_000_000_000)
        self.assertEqual(probe.report(1.)['issues'], {
            'empty_frame': 1, 'zero_stamp': 1, 'empty_cloud': 1,
            'invalid_data_layout': 1, 'invalid_xyz_fields': 1, 'insufficient_samples': 1,
        })

    def test_rejects_xyz_outside_point_record_and_changed_fields(self):
        probe = CloudProbe()
        probe.observe(cloud(1), 2_000_000_000)
        message = cloud(2)
        message.fields[2].offset = 12
        probe.observe(message, 2_000_000_000)
        self.assertEqual(probe.report(1.)['issues'], {
            'changed_fields': 1, 'invalid_xyz_fields': 1,
        })
