"""Read-only Go2 X PointCloud2 probe for an external ROS 2 computer."""

import argparse
from collections import Counter
import json
import math
import time


class CloudProbe:
    def __init__(self):
        self.received = 0
        self.frames = set()
        self.issues = Counter()
        self.first_stamp_ns = None
        self.last_stamp_ns = None
        self.last_layout = None
        self.last_header_age_s = None

    def observe(self, message, receive_ros_ns):
        self.received += 1
        self.frames.add(message.header.frame_id)
        if not message.header.frame_id.strip():
            self.issues['empty_frame'] += 1

        stamp = message.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns == 0:
            self.issues['zero_stamp'] += 1
        else:
            if self.first_stamp_ns is None:
                self.first_stamp_ns = stamp_ns
            if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
                self.issues['non_increasing_stamp'] += 1
            self.last_stamp_ns = stamp_ns
            self.last_header_age_s = (receive_ros_ns - stamp_ns) / 1e9

        previous_shape = None if self.last_layout is None else self.last_layout['fields']
        fields = sorted((field.name, field.offset, field.datatype, field.count)
                        for field in message.fields)
        self.last_layout = {
            'width': message.width, 'height': message.height,
            'point_step': message.point_step, 'row_step': message.row_step,
            'data_bytes': len(message.data), 'fields': fields,
        }
        if previous_shape is not None and previous_shape != fields:
            self.issues['changed_fields'] += 1
        if message.width == 0 or message.height == 0:
            self.issues['empty_cloud'] += 1
        if (message.point_step == 0 or message.row_step < message.width * message.point_step
                or len(message.data) != message.row_step * message.height):
            self.issues['invalid_data_layout'] += 1

        for axis in ('x', 'y', 'z'):
            matches = [field for field in fields if field[0] == axis]
            if len(matches) != 1:
                self.issues['invalid_xyz_fields'] += 1
                break
            _, offset, datatype, count = matches[0]
            size = {7: 4, 8: 8}.get(datatype, 0)
            if not size or count != 1 or offset + size > message.point_step:
                self.issues['invalid_xyz_fields'] += 1
                break

    def report(self, elapsed):
        issues = self.issues.copy()
        if self.received < 2:
            issues['insufficient_samples'] = 1
        if len(self.frames) > 1:
            issues['multiple_frames'] = 1
        return {
            'status': 'check_failed' if issues else 'cloud_metadata_ok',
            'received': self.received,
            'window_seconds': round(elapsed, 3),
            'frames': sorted(self.frames),
            'first_stamp_ns': self.first_stamp_ns,
            'last_stamp_ns': self.last_stamp_ns,
            'last_header_age_s': self.last_header_age_s,
            'last_layout': self.last_layout,
            'issues': dict(issues),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud-topic', default='/utlidar/cloud')
    parser.add_argument('--seconds', type=float, default=10.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 120:
        parser.error('--seconds must be in (0, 120]')

    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
    except ImportError as error:
        parser.exit(1, f'Cannot import ROS 2: source your ROS 2 environment first ({error})\n')

    probe = CloudProbe()
    rclpy.init(args=[])
    node = None
    try:
        node = rclpy.create_node('go2x_cloud_probe', enable_rosout=False,
                                 start_parameter_services=False)
        subscription = node.create_subscription(
            PointCloud2, args.cloud_topic,
            lambda message: probe.observe(message, node.get_clock().now().nanoseconds),
            qos_profile_sensor_data,
        )
        start = time.monotonic()
        while (remaining := start + args.seconds - time.monotonic()) > 0:
            rclpy.spin_once(node, timeout_sec=min(remaining, 0.25))
        result = probe.report(time.monotonic() - start)
        result['cloud_topic'] = subscription.topic_name
        result['matched_publishers'] = subscription.get_publisher_count()
        result['discovered_sensor_topics'] = {
            name: sorted(types) for name, types in node.get_topic_names_and_types()
            if name != subscription.topic_name and node.count_publishers(name) and
            set(types).intersection({'sensor_msgs/msg/PointCloud2', 'sensor_msgs/msg/Imu',
                                     'nav_msgs/msg/Odometry', 'sensor_msgs/msg/LaserScan'})
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result['status'] == 'cloud_metadata_ok' else 2
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
