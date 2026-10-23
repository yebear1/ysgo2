"""Offline data-layout, map identity, heading recovery and fail-closed tests."""
from dataclasses import replace
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from .fixtures import room, ray_scan
from .geometry import decode_xyz,project_cloud,sensor_stamp,wrap,ClockTranslator
from .grid_map import SavedGrid
from .localization import GridLocalizer
from .recording import EventWriter,events,replay
from .session import AnalysisSession
from ..navigation.runtime import PlanarSafety


class SensorTests(unittest.TestCase):
    def test_command_help_without_initializing_ros(self):
        for module in ('cli', 'ros_capture', 'ros_smoke'):
            run=subprocess.run([sys.executable, '-m', 'deploy.lidar.'+module, '--help'],
                               capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('usage:',run.stdout)

    def test_cloud_padding_endian_offsets_and_nans(self):
        for big in (False,True):
            prefix='>' if big else '<'
            rows=[]
            for y in range(2):
                row=b''.join(struct.pack(prefix+'ffff',99.,x+y,2.,float('nan') if x==1 else 3.) for x in range(2))
                rows.append(row+b'padding!')
            msg=NS(width=2,height=2,point_step=16,row_step=40,data=b''.join(rows),is_bigendian=big,
                   fields=[NS(name=n,offset=o,datatype=7,count=1) for n,o in [('x',4),('y',8),('z',12)]])
            xyz=decode_xyz(msg)
            np.testing.assert_equal(xyz[:,0],[0,1,1,2])
            self.assertTrue(np.isnan(xyz[1,2]))
            msg.row_step=31
            with self.assertRaises(ValueError):decode_xyz(msg)

    def test_overlapping_cloud_fields_rejected(self):
        msg=NS(width=1,height=1,point_step=12,row_step=12,data=bytes(12),is_bigendian=False,
               fields=[NS(name=n,offset=0,datatype=7,count=1) for n in ('x','y','z')])
        with self.assertRaises(ValueError):decode_xyz(msg)

    def test_projection_nearest_transform_and_unknown(self):
        a,r=project_cloud([[1,0,0],[2,0,0],[.5,0,10],[float('nan'),0,0]],(0,0,0),(0,0,0,1))
        self.assertEqual(np.count_nonzero(r),1)
        self.assertAlmostEqual(max(r),1)
        a,r=project_cloud([[1,0,0]],(0,1,0),(0,0,math.sqrt(.5),math.sqrt(.5)))
        self.assertAlmostEqual(max(r),2)
        self.assertAlmostEqual(a[np.argmax(r)],math.pi/2)

    def test_invalid_transform_rejected(self):
        with self.assertRaises(ValueError):project_cloud([[1,0,0]],(0,0,0),(0,0,0,0))

    def test_live_clock_jump_latches(self):
        clock=ClockTranslator()
        self.assertAlmostEqual(clock.convert(100.,100.1,10.),9.9)
        self.assertAlmostEqual(clock.convert(100.1,100.2,10.1),10.)
        with self.assertRaises(ValueError):clock.convert(110.2,110.3,10.2)
        with self.assertRaises(ValueError):clock.convert(100.2,100.3,10.2)

    def test_sensor_age_preserved(self):
        self.assertAlmostEqual(sensor_stamp(100.,100.1,10.),9.9)
        for source in (101.,99.,float('nan')):
            with self.assertRaises(ValueError):sensor_stamp(source,100.,10.)


class MapTests(unittest.TestCase):
    def test_map_hash_roundtrip_and_tampering(self):
        m=room()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'map.npz'; m.save(path)
            self.assertEqual(SavedGrid.load(path).map_id,m.map_id)
            with self.assertRaises(FileExistsError):m.save(path)
            np.savez(path,data=m.data,resolution=.1,origin=m.origin,frame_id='map',map_id=m.map_id)
            with self.assertRaises(ValueError):SavedGrid.load(path)

    def test_rotated_and_nonfinite_map_origins_rejected(self):
        m=room()
        origin=NS(position=NS(x=-4.,y=-3.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))
        message=NS(header=NS(frame_id='map'),info=NS(origin=origin,width=161,height=121,resolution=.05),
                   data=m.data.reshape(-1))
        self.assertEqual(SavedGrid.from_message(message).map_id,m.map_id)
        origin.position.z=float('nan')
        with self.assertRaises(ValueError):SavedGrid.from_message(message)
        origin.position.z=0.;origin.orientation.z=.1
        with self.assertRaises(ValueError):SavedGrid.from_message(message)

    def test_unknown_is_not_free_and_invalid_values(self):
        m=room();v,d=m.values([[-100.,0.]])
        self.assertEqual(v[0],-1);self.assertTrue(math.isinf(d[0]))
        bad=m.data.copy().astype(float);bad[0,0]=1.5
        with self.assertRaises(ValueError):SavedGrid(bad,.05,(-4.,-3.))


class LocalizationTests(unittest.TestCase):
    def test_all_starting_headings_and_duplicate_not_confirmation(self):
        m=room()
        for deg in (-30,30,60,180):
            with self.subTest(heading=deg):
                p=(.20,-.15,math.radians(deg));loc=GridLocalizer(m,(0,0),.5)
                scan=ray_scan(m,p)
                first=loc.update(scan,(0,0,0),0.)
                self.assertEqual(first.reason,'confirming')
                again=loc.update(scan,(0,0,0),.01)
                self.assertEqual(again.confirmations,1)
                for i in (1,2):result=loc.update(replace(scan,stamp=i*.1),(0,0,0),i*.1)
                self.assertTrue(result.pose.matched)
                self.assertLess(np.linalg.norm(np.array(result.pose.xy_yaw[:2])-p[:2]),.10)
                self.assertLess(abs(wrap(result.pose.xy_yaw[2]-p[2])),math.radians(3))

    def test_symmetric_room_refuses_heading(self):
        m=room(True);loc=GridLocalizer(m,(0,0),.5)
        result=loc.update(ray_scan(m,(0,0,0)),(0,0,0),0)
        self.assertEqual(result.reason,'map_match_ambiguous')
        self.assertIsNone(result.pose)

    def test_vo_or_lio_alone_never_matches(self):
        m=room();scan=ray_scan(m,(0,0,0));loc=GridLocalizer(m,(0,0),.5)
        for i in range(4):
            result=loc.update(replace(scan,stamp=i*.1,ranges=tuple([0.]*72)),(i*.01,0,0),i*.1)
            self.assertIsNone(result.pose)

    def test_wrong_map_stale_future_and_discontinuity(self):
        m=room();scan=ray_scan(m,(0,0,0));loc=GridLocalizer(m,(0,0),.5)
        for i in range(3):loc.update(replace(scan,stamp=i*.1),(0,0,0),i*.1)
        self.assertTrue(loc.verified)
        result=loc.update(replace(scan,stamp=.3),(3,0,0),.3)
        self.assertEqual(result.reason,'odometry_discontinuity');self.assertFalse(loc.verified)
        for stamp,now in ((.4,.8),(1.,.8),(-1.,0.)):
            self.assertIsNone(loc.update(replace(scan,stamp=stamp),(0,0,0),now).pose)
        other=GridLocalizer(room(True),(0,0),.5)
        self.assertIsNone(other.update(scan,(0,0,0),0.).pose)

    def test_tracking_uses_relative_odom_not_odom_origin(self):
        m=room();loc=GridLocalizer(m,(0,0),.5)
        for i in range(3):loc.update(ray_scan(m,(0,0,math.pi/2),stamp=i*.1),(100,20,0),i*.1)
        result=loc.update(ray_scan(m,(0,.05,math.pi/2),stamp=.3),(100.05,20,0),.3)
        self.assertTrue(result.pose.matched)
        self.assertLess(np.linalg.norm(np.array(result.pose.xy_yaw[:2])-[0,.05]),.10)


class ReplayTests(unittest.TestCase):
    def make_record(self,path,m,fault=False,stale=False):
        writer=EventWriter(path,dict(provenance='unit-test-synthetic',map_id=m.map_id))
        scan=ray_scan(m,(0,0,0))
        for i in range(8):
            if fault and i==3:
                writer.write(dict(type='fault',t=i*.1,reason='test_dropout'));continue
            writer.write(dict(type='frame',t=i*.1,scan_stamp=i*.1-(1 if stale else 0),
                              odom_stamp=i*.1,angles=list(scan.angles),ranges=list(scan.ranges),
                              max_range=8.,odom_xy_yaw=[0,0,0]))
        writer.close()

    def test_fault_recovery_does_not_resume_commands(self):
        m=room()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'record.jsonl';self.make_record(path,m,fault=True)
            result=replay(path,m,(0,0),.5,goal=(1,0))
            self.assertTrue(result['ever_armed'])
            self.assertTrue(any(r['decision']['source']=='autonomy' for r in result['records'][:3]))
            self.assertTrue(all(r['decision']['source']=='stop' for r in result['records'][3:]))
            self.assertFalse(result['hardware_verified'])

    def test_stale_scans_never_arm(self):
        m=room()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'record.jsonl';self.make_record(path,m,stale=True)
            result=replay(path,m,(0,0),.5,goal=(1,0))
            self.assertFalse(result['ever_armed'])
            self.assertTrue(all(r['decision']['source']=='stop' for r in result['records']))

    def test_map_identity_mismatch_and_time_reversal(self):
        m=room()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'record.jsonl';self.make_record(path,m)
            with self.assertRaises(ValueError):replay(path,room(True),(0,0),.5)
            writer=EventWriter(Path(d)/'bad.jsonl',{})
            writer.write(dict(type='fault',t=1));writer.write(dict(type='fault',t=0));writer.close()
            with self.assertRaises(ValueError):list(events(Path(d)/'bad.jsonl'))

    def test_compute_overrun_cannot_issue_a_fresh_command(self):
        m=room();session=AnalysisSession(m,(0,0),.5,goal=(1,0))
        scan=ray_scan(m,(0,0,0))
        event=dict(type='frame',t=0.,scan_stamp=0.,odom_stamp=0.,angles=list(scan.angles),
                   ranges=list(scan.ranges),max_range=8.,odom_xy_yaw=[0,0,0])
        for i in range(2):
            session.process(dict(event,t=i*.1,scan_stamp=i*.1,odom_stamp=i*.1))
        with patch('deploy.lidar.session.time.perf_counter',side_effect=[10.,10.3]):
            result=session.process(dict(event,t=.2,scan_stamp=.2,odom_stamp=.2))
        self.assertEqual(result['decision']['source'],'stop')
        self.assertFalse(session.ever_armed)
        self.assertAlmostEqual(session.now,.5)
        later=session.process(dict(event,t=.3,scan_stamp=.3,odom_stamp=.3))
        self.assertGreaterEqual(session.now,.5)
        self.assertEqual(later['decision']['source'],'stop')

    def test_empty_cloud_sector_blocks_motion(self):
        scan=ray_scan(room(),(0,0,0))
        guard=PlanarSafety()
        self.assertTrue(guard((.1,0,0),scan))
        ranges=list(scan.ranges);ranges[0]=0
        self.assertFalse(guard((.1,0,0),replace(scan,ranges=tuple(ranges))))


if __name__=='__main__':unittest.main()
