"""Create a sensor-only Go2 X replay bag ordered by original header time."""

import argparse
from bisect import bisect_left
from pathlib import Path
from statistics import median


CLOUD = '/utlidar/cloud_base'
ODOM = '/utlidar/robot_odom'
REPLAY_TOPICS = {
    CLOUD: ('/go2x_replay/cloud_base', 'sensor_msgs/msg/PointCloud2'),
    ODOM: ('/go2x_replay/odom', 'nav_msgs/msg/Odometry'),
}


def nearest_pairs(cloud_stamps, odom_stamps, tolerance_ns=50_000_000):
    """Match strictly ordered source stamps without stretching the time limit."""
    for stamps in (cloud_stamps, odom_stamps):
        if not stamps or stamps[0] <= 0 or any(a >= b for a, b in zip(stamps, stamps[1:])):
            raise ValueError('Missing, zero, repeated, or regressing source stamps')
    pairs = []
    for cloud_index, stamp in enumerate(cloud_stamps):
        index = bisect_left(odom_stamps, stamp)
        candidates = range(max(0, index - 1), min(index + 1, len(odom_stamps)))
        closest = min(candidates, key=lambda candidate: abs(odom_stamps[candidate] - stamp))
        delta = abs(stamp - odom_stamps[closest])
        if delta < tolerance_ns:
            pairs.append((cloud_index, closest, delta))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    if not (args.input / 'metadata.yaml').is_file() or args.output.exists():
        parser.error('Use an existing input bag and a new output directory')

    import json
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.input)), rosbag2_py.ConverterOptions('', ''))
    types = {field.name: field.type for field in reader.get_all_topics_and_types()}
    if any(types.get(name) != kind for name, (_, kind) in REPLAY_TOPICS.items()):
        parser.error('Missing required Go2 X sensor topics or incorrect message types')
    classes = {name: get_message(kind) for name, (_, kind) in REPLAY_TOPICS.items()}
    rows = {name: [] for name in REPLAY_TOPICS}
    payload_bytes = 0
    while reader.has_next():
        name, payload, _receive_stamp = reader.read_next()
        if name not in rows:
            continue
        payload_bytes += len(payload)
        if payload_bytes > 256 * 1024**2:
            parser.error('Use a short baseline bag (selected sensors exceed 256 MiB)')
        message = deserialize_message(payload, classes[name])
        header = message.header
        if name == CLOUD and header.frame_id != 'base_link':
            parser.error('Cloud must already be in base_link')
        if name == ODOM and (header.frame_id, message.child_frame_id) != ('odom', 'base_link'):
            parser.error('Expected odom → base_link odometry')
        rows[name].append((header.stamp.sec * 10**9 + header.stamp.nanosec, payload))

    cloud, odom = rows[CLOUD], rows[ODOM]
    pairs = nearest_pairs([entry[0] for entry in cloud], [entry[0] for entry in odom])
    if len(pairs) < 2:
        parser.error('Not enough synchronized scan/odometry samples')
    odom_indexes = {odom_index for _, odom_index, _ in pairs}
    replay = [(cloud[index][0], CLOUD, cloud[index][1]) for index, _, _ in pairs]
    replay.extend((odom[index][0], ODOM, odom[index][1]) for index in odom_indexes)

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.output), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    try:
        for topic_id, (topic, kind) in enumerate(REPLAY_TOPICS.values()):
            writer.create_topic(rosbag2_py.TopicMetadata(id=topic_id, name=topic, type=kind,
                                                         serialization_format='cdr'))
        for stamp, original_topic, payload in sorted(replay, key=lambda row: (row[0], row[1])):
            writer.write(REPLAY_TOPICS[original_topic][0], payload, stamp)
    finally:
        writer.close()

    print(json.dumps({'input_clouds': len(cloud), 'matched_clouds': len(pairs),
                      'input_odometry': len(odom), 'replay_odometry': len(odom_indexes),
                      'median_pair_delta_ms': median(pair[2] for pair in pairs) / 1e6,
                      'max_pair_delta_ms': max(pair[2] for pair in pairs) / 1e6,
                      'header_timestamps_preserved': True}, indent=2))


if __name__ == '__main__':
    main()
