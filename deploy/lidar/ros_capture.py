"""Read-only ROS 2 sensor capture and OccupancyGrid export.

No publishers, service clients, Move/StopMove or robot state changes. Run on
an external ROS workstation first. Never launch this against a real robot
without the operator deliberately selecting its ROS domain/network.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from .geometry import decode_xyz,project_cloud,quaternion_matrix,sensor_stamp,ClockTranslator
from .grid_map import SavedGrid
from .recording import EventWriter
from .session import AnalysisSession


def stamp(msg):return float(msg.sec)+float(msg.nanosec)*1e-9

def pose(transform):
    t,q=transform.translation,transform.rotation
    rotation=quaternion_matrix((q.x,q.y,q.z,q.w))
    roll=math.atan2(rotation[2,1],rotation[2,2])
    pitch=math.asin(float(np.clip(-rotation[2,0],-1,1)))
    yaw=math.atan2(rotation[1,0],rotation[0,0])
    xyz=(float(t.x),float(t.y),float(t.z))
    if not all(math.isfinite(x) for x in xyz):raise ValueError('Nonfinite transform')
    return (xyz[0],xyz[1],yaw),max(abs(roll),abs(pitch))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    cap=sub.add_parser('capture')
    cap.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/go2_builtin.json')
    cap.add_argument('--output',type=Path,required=True)
    cap.add_argument('--seconds',type=float,default=60.)
    cap.add_argument('--map',type=Path,help='Optional saved-grid live analysis; never sends robot commands')
    cap.add_argument('--search-center',type=float,nargs=2)
    cap.add_argument('--search-radius',type=float,default=1.)
    cap.add_argument('--evaluate-goal',type=float,nargs=2,help='Memory-only command analysis')
    cap.add_argument('--analysis-output',type=Path,help='Required with --map; new JSONL file')
    save=sub.add_parser('save-map')
    save.add_argument('--topic',default='/go2_lidar/map')
    save.add_argument('--output',type=Path,required=True)
    save.add_argument('--seconds',type=float,default=15.)
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or not 0<args.seconds<=3600:parser.error('seconds must be 0..3600')
    if args.output.exists():parser.error('Output already exists; choose a new path')
    analysis = None
    analysis_stream = None
    if args.command == 'capture':
        if args.map:
            if args.search_center is None or args.analysis_output is None:
                parser.error('--map requires --search-center and --analysis-output')
            if args.analysis_output.exists() or args.analysis_output.resolve() == args.output.resolve():
                parser.error('Analysis output must be a different, new file')
            analysis = AnalysisSession(SavedGrid.load(args.map), args.search_center,
                                       args.search_radius, args.evaluate_goal)
        elif args.analysis_output or args.evaluate_goal or args.search_center:
            parser.error('Live analysis options require --map')
    # Lazy imports keep --help and all offline workflows independent of ROS.
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2,Imu
    from nav_msgs.msg import OccupancyGrid,Odometry
    from tf2_ros import Buffer,TransformListener
    rclpy.init(args=[])
    node=Node('go2_lidar_readonly_capture')
    started=time.monotonic();written=[0];writer=None
    last_valid_time=[0.];timeout_reported=[False]
    try:
        if analysis is not None:
            analysis_stream = args.analysis_output.open('x', encoding='utf-8')
        def emit(event):
            writer.write(event)
            if event['type']=='frame':
                last_valid_time[0]=event['scan_stamp']
                timeout_reported[0]=False
            else:
                timeout_reported[0]=True
            if analysis is not None:
                result = analysis.process(event)
                result['hardware_verified'] = False
                result['transport'] = 'analysis-only-memory-sink'
                analysis_stream.write(json.dumps(result, allow_nan=False)+'\n')
                analysis_stream.flush()
        if args.command=='save-map':
            def on_map(msg):
                if written[0]:return
                try:
                    m=SavedGrid.from_message(msg);m.save(args.output)
                    written[0]=1;print('Saved '+m.map_id,flush=True)
                except ValueError as exc:node.get_logger().warning(str(exc))
            node.create_subscription(OccupancyGrid,args.topic,on_map,
                                     QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        else:
            config=json.loads(args.config.read_text())
            required=('cloud_topic','base_frame','odom_frame','cloud_frame')
            if any(not isinstance(config.get(k),str) or not config[k] for k in required):
                raise ValueError('Explicit topic and frame names are required')
            # Construct the local TF listener only; no extrinsics are broadcast.
            buffer=Buffer(cache_time=Duration(seconds=5.));listener=TransformListener(buffer,node)
            writer=EventWriter(args.output,dict(provenance='ros2-sensor-capture',config=config,
                                                hardware_verified=False,wall_started_unix=time.time()))
            previous=[None];latest_imu=[None];odometry=[];pending=[];translator=ClockTranslator()
            def on_odom(msg):
                if msg.header.frame_id!=config['odom_frame'] or msg.child_frame_id!=config['base_frame']:
                    return
                odometry.append(msg)
                del odometry[:-30]
            if config.get('odom_topic'):
                node.create_subscription(Odometry,config['odom_topic'],on_odom,qos_profile_sensor_data)
            def on_imu(msg):
                vals=[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z,
                      msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z]
                if all(math.isfinite(v) for v in vals):
                    latest_imu[0]=dict(source_stamp=stamp(msg.header.stamp),frame_id=msg.header.frame_id,
                                       gyro=vals[:3],acceleration=vals[3:])
                else:latest_imu[0]=None
            if config.get('imu_topic'):
                node.create_subscription(Imu,config['imu_topic'],on_imu,qos_profile_sensor_data)
            def process_cloud(msg):
                t=time.monotonic()-started
                try:
                    source=stamp(msg.header.stamp)
                    ros_now=node.get_clock().now().nanoseconds*1e-9
                    translated=translator.convert(source,ros_now,t,config.get('max_sensor_age',.20))
                    if msg.header.frame_id!=config['cloud_frame']:
                        raise ValueError('Unexpected cloud frame; check profile')
                    query=Time.from_msg(msg.header.stamp)
                    extrinsic=buffer.lookup_transform(config['base_frame'],msg.header.frame_id,query,
                                                      timeout=Duration(seconds=0.)).transform
                    if config.get('odom_topic'):
                        if not odometry:raise ValueError('No odometry messages')
                        nearest=min(odometry,key=lambda m:abs(stamp(m.header.stamp)-source))
                        if abs(stamp(nearest.header.stamp)-source)>.05:
                            raise ValueError('Odometry/scan not synchronized')
                        from types import SimpleNamespace
                        odom_transform=SimpleNamespace(translation=nearest.pose.pose.position,
                                                       rotation=nearest.pose.pose.orientation)
                        odom_source_stamp=stamp(nearest.header.stamp)
                    else:
                        odom=buffer.lookup_transform(config['odom_frame'],config['base_frame'],query,
                                                     timeout=Duration(seconds=0.))
                        odom_transform=odom.transform
                        odom_source_stamp=stamp(odom.header.stamp)
                    odom_xy,tilt=pose(odom_transform)
                    if tilt>math.radians(config.get('max_tilt_degrees',10.)):
                        raise ValueError('Planar projection tilt limit exceeded')
                    odom_time=sensor_stamp(odom_source_stamp,ros_now,t,.20)
                    if abs(odom_time-translated)>.05:raise ValueError('Odometry/scan not synchronized')
                    q=extrinsic.rotation;tr=extrinsic.translation
                    angles,ranges=project_cloud(decode_xyz(msg),(tr.x,tr.y,tr.z),(q.x,q.y,q.z,q.w),
                                               **{k:config[k] for k in ('bins','min_z','max_z','min_range','max_range')})
                    imu=latest_imu[0]
                    if imu is not None:
                        if not 0<=source-imu['source_stamp']<=.20:
                            imu=None
                        else:
                            imu=dict(imu,stamp=sensor_stamp(imu['source_stamp'],ros_now,t,.4))
                    emit(dict(type='frame',t=time.monotonic()-started,scan_stamp=translated,odom_stamp=odom_time,
                                      source_stamp=source,angles=angles.tolist(),ranges=ranges.tolist(),
                                      max_range=config['max_range'],odom_xy_yaw=list(odom_xy),imu=imu,
                                      observed_bins=int(np.count_nonzero(ranges)),
                                      cloud_fields=[dict(name=f.name,offset=f.offset,datatype=f.datatype,count=f.count)
                                                    for f in msg.fields]))
                    written[0]+=1
                except Exception as exc:
                    emit(dict(type='fault',t=time.monotonic()-started,reason=type(exc).__name__+': '+str(exc)))
            def on_cloud(msg):
                source=stamp(msg.header.stamp)
                if previous[0] is not None and source<=previous[0]:
                    emit(dict(type='fault',t=time.monotonic()-started,reason='cloud_timestamp_repeated_or_regressed'))
                    return
                previous[0]=source
                if len(pending)>=4:
                    pending.pop(0)
                    emit(dict(type='fault',t=time.monotonic()-started,reason='capture_queue_overflow'))
                pending.append(msg)
            def drain():
                now=time.monotonic()-started
                if now-last_valid_time[0]>.20 and not timeout_reported[0]:
                    emit(dict(type='fault',t=now,reason='sensor_stream_timeout'))
                if not pending:return
                msg=pending[0];source=stamp(msg.header.stamp)
                age=node.get_clock().now().nanoseconds*1e-9-source
                ready=buffer.can_transform(config['base_frame'],msg.header.frame_id,Time.from_msg(msg.header.stamp))
                if config.get('odom_topic'):
                    ready=ready and any(abs(stamp(m.header.stamp)-source)<=.05 for m in odometry)
                else:
                    ready=ready and buffer.can_transform(config['odom_frame'],config['base_frame'],Time.from_msg(msg.header.stamp))
                if ready or age<0 or age>config.get('max_sensor_age',.20):
                    pending.pop(0);process_cloud(msg)
            node.create_timer(.01,drain)
            node.create_subscription(PointCloud2,config['cloud_topic'],on_cloud,qos_profile_sensor_data)
        while rclpy.ok() and time.monotonic()-started<args.seconds:
            rclpy.spin_once(node,timeout_sec=.05)
            if args.command=='save-map' and written[0]:break
    except KeyboardInterrupt:pass
    finally:
        if writer is not None:
            emit(dict(type='fault',t=time.monotonic()-started,reason='capture_ended'))
        if analysis is not None:analysis.close()
        if analysis_stream is not None:analysis_stream.close()
        if writer is not None:writer.close()
        node.destroy_node();rclpy.shutdown()
    print(json.dumps(dict(valid_samples=written[0],output=str(args.output),hardware_verified=False)))
    return 0 if written[0] else 2


if __name__=='__main__':raise SystemExit(main())
