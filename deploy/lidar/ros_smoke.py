"""Actual ROS/RTAB-Map smoke test using synthetic 3D measurements, no robot.

The sensor generator knows truth only to cast measurements and score results.
No odometry or map pose is supplied to ICP. An isolated ROS domain is required.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--domain',type=int,default=87)
    args=parser.parse_args()
    if not 1<=args.domain<=100:parser.error('Use an isolated domain 1..100')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['ROS_DOMAIN_ID']=str(args.domain)
    os.environ['ROS_LOCALHOST_ONLY']='1'
    os.environ['RMW_IMPLEMENTATION']='rmw_fastrtps_cpp'
    os.environ['ROS_LOG_DIR']=str(out/'ros_logs')
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid,Odometry
    from sensor_msgs.msg import PointCloud2,PointField,Imu
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
    from tf2_ros import StaticTransformBroadcaster
    rclpy.init(args=[]);node=rclpy.create_node('go2_lidar_synthetic_test_sensor')
    broadcaster=StaticTransformBroadcaster(node)
    tf=TransformStamped();tf.header.frame_id='base_link';tf.child_frame_id='utlidar_lidar';tf.transform.rotation.w=1.
    tf.header.stamp=node.get_clock().now().to_msg();broadcaster.sendTransform(tf)
    pub=node.create_publisher(PointCloud2,'/go2_lidar_test/cloud',qos_profile_sensor_data)
    imu_pub=node.create_publisher(Imu,'/go2_lidar_test/imu',qos_profile_sensor_data)
    received={'odometry':0,'maps':0,'last_position':None,'map_free_cells':0}
    def odom(msg):
        q=msg.pose.pose.orientation
        if q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z<.5:return
        received['odometry']+=1
        p=msg.pose.pose.position;received['last_position']=[p.x,p.y,p.z]
    def grid(msg):
        received['maps']+=1;received['map_free_cells']=sum(v==0 for v in msg.data)
    node.create_subscription(Odometry,'/go2_lidar/odom',odom,qos_profile_sensor_data)
    node.create_subscription(OccupancyGrid,'/go2_lidar/map',grid,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    launch=Path(__file__).with_name('go2_lidar_mapping.launch.py')
    config=json.loads(Path(__file__).with_name('configs').joinpath('go2_workstation_icp.json').read_text())
    config['cloud_topic']='/go2_lidar_test/cloud'
    config['imu_topic']='/go2_lidar_test/imu'
    profile=out/'profile.json';profile.write_text(json.dumps(config))
    processes=[];handles=[]
    def start(command,name):
        stream=(out/(name+'.log')).open('w');handles.append(stream)
        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc);return proc
    mapping=start(['ros2','launch',str(launch),'database:='+str(out/'mapping.db'),
                   'cloud_topic:=/go2_lidar_test/cloud','publish_odom_tf:=true'],'mapping')
    capture=start([sys.executable,'-m','deploy.lidar.ros_capture','capture','--config',str(profile),
                   '--output',str(out/'capture.jsonl'),'--seconds','16'],'capture')
    save=None;observe=None
    # Ray intersections against the synthetic room boundary and fixed boxes.
    rectangles=[(-4.,4.,-3.,3.),(-2.8,-2.,-2.4,-1.3),(1.5,2.4,1.,2.2)]
    fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(('x','y','z'))]
    initial=time.monotonic();sent=0;truth=(0.,0.,0.)
    try:
        while time.monotonic()-initial<18:
            elapsed=time.monotonic()-initial
            if elapsed>8 and save is None:
                save=start([sys.executable,'-m','deploy.lidar.ros_capture','save-map','--output',str(out/'map.npz'),'--seconds','8'],'save_map')
            if (out/'map.npz').exists() and observe is None:
                observe=start([sys.executable,'-m','deploy.lidar.ros_capture','capture','--config',str(profile),
                               '--output',str(out/'live-capture.jsonl'),'--seconds','8',
                               '--map',str(out/'map.npz'),'--search-center','0','0','--search-radius','.5',
                               '--evaluate-goal','1','0','--analysis-output',str(out/'live-analysis.jsonl')],'live_analysis')
            # Initial stationary scans let all subscribers discover each other.
            x=min(.30,max(0.,elapsed-5)*.03);yaw=min(.10,max(0.,elapsed-9)*.02)
            truth=(x,0.,yaw)
            points=[]
            for angle in np.linspace(-math.pi,math.pi,240,endpoint=False):
                dx,dy=math.cos(angle+yaw),math.sin(angle+yaw)
                hits=[]
                for x0,x1,y0,y1 in rectangles:
                    if abs(dx)>1e-6:
                        for wall in (x0,x1):
                            t=(wall-x)/dx
                            if t>.1 and y0<=t*dy<=y1:hits.append(t)
                    if abs(dy)>1e-6:
                        for wall in (y0,y1):
                            t=wall/dy
                            if t>.1 and x0<=x+t*dx<=x1:hits.append(t)
                r=min(hits) if hits else 8.
                for z in np.linspace(-.3,1.2,14):points.append((r*math.cos(angle),r*math.sin(angle),z))
                for radius in np.arange(.4,r,.25):points.append((radius*math.cos(angle),radius*math.sin(angle),-.32))
            header=Header();header.frame_id='utlidar_lidar';header.stamp=node.get_clock().now().to_msg()
            imu=Imu();imu.header.frame_id='base_link';imu.header.stamp=header.stamp
            imu.orientation.w=math.cos(yaw/2);imu.orientation.z=math.sin(yaw/2)
            imu.angular_velocity.z=.02 if 9<elapsed<14 else 0.
            imu.linear_acceleration.z=9.81;imu_pub.publish(imu)
            msg=point_cloud2.create_cloud(header,fields,np.asarray(points,dtype=np.float32))
            # Deliberately interrupt sensor traffic after live analysis starts.
            # The subscriber must stop its analysis command and never resume it.
            if not 13.0<elapsed<13.6:
                pub.publish(msg);sent+=1
            end=time.monotonic()+.10
            while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.01)
            if mapping.poll() is not None:break
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        for proc in processes:
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for stream in handles:stream.close()
        node.destroy_node();rclpy.shutdown()
    error=None if received['last_position'] is None else float(np.linalg.norm(np.array(received['last_position'][:2])-truth[:2]))
    valid=0;faults=0;imu_frames=0
    if (out/'capture.jsonl').exists():
        for line in (out/'capture.jsonl').read_text().splitlines():
            event=json.loads(line);valid+=event.get('type')=='frame';faults+=event.get('type')=='fault'
            imu_frames+=event.get('imu') is not None
    live=[]
    if (out/'live-analysis.jsonl').exists():
        live=[json.loads(line) for line in (out/'live-analysis.jsonl').read_text().splitlines()]
    live_matched=sum(r['reason']=='matched' for r in live)
    live_commands=sum(r['decision']['source']=='autonomy' for r in live)
    moving=False;stopped_after_motion=False;recovered_pose_after_stop=False;resumed=False
    for record in live:
        if record['decision']['source']=='autonomy':
            if stopped_after_motion:resumed=True
            moving=True
        elif moving and record['reason']=='sensor_stream_timeout':
            stopped_after_motion=True
        if stopped_after_motion and record['reason']=='matched':recovered_pose_after_stop=True
    from .recording import replay
    from .grid_map import SavedGrid
    replay_matched=0
    if (out/'map.npz').exists():
        analysis=replay(out/'capture.jsonl',SavedGrid.load(out/'map.npz'),(0,0),.5)
        (out/'replay.json').write_text(json.dumps(analysis,indent=2)+'\n')
        replay_matched=sum(r['reason']=='matched' for r in analysis['records'])
    result=dict(test='actual-ros2-rtabmap-with-synthetic-rays',hardware_verified=False,
                imu_fusion_verified=False,closed_loop_navigation_verified=False,
                sensor_frames=sent,received=received,capture_valid_frames=valid,capture_faults=faults,imu_frames=imu_frames,
                live_matched_frames=live_matched,live_analysis_commands=live_commands,replay_matched_frames=replay_matched,
                dropout_stopped_analysis=stopped_after_motion,pose_recovered_after_dropout=recovered_pose_after_stop,
                resumed_without_rearm=resumed,
                final_truth=list(truth),final_xy_error_m=error,
                saved_map=(out/'map.npz').is_file())
    result['passed']=(received['odometry']>=20 and received['maps']>=2 and received['map_free_cells']>100
                      and result['saved_map'] and valid>=10 and error is not None and error<.15
                      and imu_frames>=10 and live_matched>=10 and live_commands>=1 and replay_matched>=valid*.90
                      and stopped_after_motion and recovered_pose_after_stop and not resumed)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
