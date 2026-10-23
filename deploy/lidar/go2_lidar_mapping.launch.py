"""Camera-free RTAB-Map ICP mapping on a ROS workstation; no motion nodes.

Run: ros2 launch <absolute path to this file> database:=<new absolute .db path>
Requires a measured base-to-cloud TF. Existing maps are never deleted.
"""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def build(context):
    def value(name):return LaunchConfiguration(name).perform(context)
    def boolean(name):
        v=value(name).lower()
        if v not in ('true','false'):raise ValueError(name+' must be true or false')
        return v=='true'
    database=Path(value('database')).expanduser()
    if not database.is_absolute() or database.suffix!='.db':
        raise ValueError('database must be an absolute .db path')
    mode=value('mode')
    if mode not in ('mapping','localization'):raise ValueError('Unknown mode')
    if mode=='localization' and not database.is_file():raise ValueError('Saved database missing')
    if mode=='mapping' and database.exists():raise ValueError('Refusing to overwrite/resume a map; choose a new database')
    database.parent.mkdir(parents=True,exist_ok=True)
    sim=boolean('use_sim_time')
    base,odom_frame=value('base_frame'),value('odom_frame')
    cloud=value('cloud_topic');external=value('external_odom_topic')
    output_odom=external or '/go2_lidar/odom'
    parameters={
        'use_sim_time':sim,'frame_id':base,'map_frame_id':'map',
        'database_path':str(database),'subscribe_depth':False,'subscribe_rgb':False,
        'subscribe_scan_cloud':True,'approx_sync':bool(external),'approx_sync_max_interval':.05,
        'topic_queue_size':10,'sync_queue_size':10,'qos_scan':2,'qos_odom':2,
        'publish_tf':True,'odom_sensor_sync':bool(external),'wait_for_transform':0.0,
        'Reg/Strategy':'1','Reg/Force3DoF':'true',
        'Icp/VoxelSize':'0.05','Icp/PointToPlane':'true',
        'Icp/MaxCorrespondenceDistance':'0.3','Icp/CorrespondenceRatio':'0.3',
        'RGBD/ProximityBySpace':'true','RGBD/ProximityByTime':'false',
        'RGBD/ProximityMaxPaths':'0','RGBD/ProximityAngle':'180',
        'RGBD/OptimizeFromGraphEnd':'false','RGBD/OptimizeMaxError':'3.0',
        'Grid/Sensor':'0','Grid/3D':'false','Grid/CellSize':'0.05',
        'Grid/RayTracing':'true','Grid/MinClusterSize':'3',
        'Grid/RangeMax':'8.0','Grid/NormalsSegmentation':'false','Grid/MapFrameProjection':'false',
        'Grid/MaxObstacleHeight':'1.5','Grid/MaxGroundHeight':value('ground_height'),
        'Mem/IncrementalMemory':'true' if mode=='mapping' else 'false',
        'Mem/InitWMWithAllNodes':'true' if mode=='localization' else 'false',
        'Mem/LocalizationReadOnly':'true' if mode=='localization' else 'false',
    }
    nodes=[]
    if not external:
        nodes.append(Node(package='rtabmap_odom',executable='icp_odometry',name='icp_odometry',
                          namespace='go2_lidar',output='screen',parameters=[{
                              'use_sim_time':sim,'frame_id':base,'odom_frame_id':odom_frame,
                              'publish_tf':boolean('publish_odom_tf'),'qos':2,
                              'Icp/VoxelSize':'0.05','Icp/PointToPlane':'true',
                              'Icp/MaxCorrespondenceDistance':'0.3','Icp/CorrespondenceRatio':'0.3',
                              'Odom/ResetCountdown':'0','Reg/Force3DoF':'true',
                          }],remappings=[('scan_cloud',cloud),('odom',output_odom)]))
    nodes.append(Node(package='rtabmap_slam',executable='rtabmap',name='rtabmap',
                      namespace='go2_lidar',output='screen',parameters=[parameters],
                      remappings=[('scan_cloud',cloud),('odom',output_odom),('map','/go2_lidar/map')]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('database',description='New mapping DB or existing localization DB; never deleted'),
        DeclareLaunchArgument('mode',default_value='mapping'),
        DeclareLaunchArgument('ground_height',default_value='-0.27',description='Ground cutoff relative to base; measure body height before real use'),
        DeclareLaunchArgument('cloud_topic',default_value='/utlidar/cloud'),
        DeclareLaunchArgument('base_frame',default_value='base_link'),
        DeclareLaunchArgument('odom_frame',default_value='go2_lidar_odom'),
        DeclareLaunchArgument('external_odom_topic',default_value='',description='Optional validated external LIO odometry'),
        DeclareLaunchArgument('publish_odom_tf',default_value='false',description='Enable only when this is the sole odom TF owner'),
        DeclareLaunchArgument('use_sim_time',default_value='false'),
        OpaqueFunction(function=build),
    ])
