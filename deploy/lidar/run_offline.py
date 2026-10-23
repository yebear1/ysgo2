"""Fixed synthetic LiDAR localization + navigation matrix, with raw traces.

Truth is used only by the measurement generator, collision checker and scorer.
The localizer receives scans and relative odometry; no truth/map pose fallback.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .fixtures import room,ray_scan
from .geometry import compose,wrap
from .localization import GridLocalizer
from .recording import AnalysisSink,EventWriter
from ..navigation.contracts import NavigationFrame,RobotState
from ..navigation.runtime import NavigationRuntime


def run_case(grid,heading,output):
    p=np.array([0.,0.,math.radians(heading)])
    odom=np.zeros(3);localizer=GridLocalizer(grid,(0,0),.5)
    clock=[0.];sink=AnalysisSink();runtime=NavigationRuntime(sink,lambda:clock[0],clock_id='simulation')
    runtime.update_map(grid.map_id,grid.grid)
    goals=[(1.,0.),(0.,0.)];completed=0;active=False;ever_started=False
    collisions=0;match_at=None;latencies=[];minimum_progress=0.;last_progress=0.;last_p=p.copy()
    writer=EventWriter(output,dict(provenance='synthetic-2d-raycast',map_id=grid.map_id,
                                  heading_degrees=heading,hardware_verified=False))
    failure=None
    try:
        for i in range(700):
            t=i*.1;clock[0]=t
            scan=ray_scan(grid,p,t,noise=.004,seed=i)
            start=time.perf_counter();matched=localizer.update(scan,odom,t)
            elapsed=time.perf_counter()-start;latencies.append(elapsed)
            # Sensor time remains the original measurement time; account for
            # actual CPU time so expensive global searches cannot emit motion.
            clock[0]=t+elapsed
            frame=NavigationFrame(RobotState(t,i,True,True),matched.pose,scan,(0.,0.,0.),'simulation')
            if matched.pose is not None and matched.pose.matched and match_at is None:match_at=t
            if not active and matched.pose is not None and matched.pose.matched:
                try:
                    runtime.set_goal(goals[completed],frame);runtime.arm(frame)
                    active=True;ever_started=True
                except ValueError:pass
            decision=runtime.step(frame)
            writer.write(dict(type='frame',t=t,scan_stamp=t,odom_stamp=t,angles=list(scan.angles),
                              ranges=list(scan.ranges),max_range=scan.max_range,odom_xy_yaw=odom.tolist(),
                              localization=asdict(matched),decision=asdict(decision),truth=p.tolist()))
            if active and decision.source=='stop':
                if decision.reason=='goal_reached':
                    completed+=1;active=False
                    if completed==len(goals):break
                else:
                    failure=decision.reason;break # Never silently rearm after a fault.
            if np.linalg.norm(p[:2]-last_p[:2])>=.03 or abs(wrap(p[2]-last_p[2]))>=.08:
                minimum_progress=max(minimum_progress,t-last_progress);last_progress=t;last_p=p.copy()
            velocity=decision.velocity
            step=(velocity[0]*.1,velocity[1]*.1,velocity[2]*.1)
            p=np.array(compose(p,step));odom=np.array(compose(odom,step))
            # Evaluate occupied footprint independently of the navigation planner.
            corners=np.array([[x,y] for x in (-.39,.39) for y in (-.20,.20)])
            c,s=math.cos(p[2]),math.sin(p[2]);points=corners@np.array([[c,s],[-s,c]])+p[:2]
            values,_=grid.values(points)
            if np.any(values>=65):collisions+=1;failure='fixture_collision';break
        else:failure='simulation_timeout'
    finally:writer.close();runtime.cancel('test_complete')
    return dict(heading=heading,passed=bool(completed==2 and collisions==0 and np.linalg.norm(p[:2])<=.20),
                completed_goals=completed,collisions=collisions,ever_started=ever_started,
                startup_match_seconds=match_at,simulation_seconds=t,final_distance=float(np.linalg.norm(p[:2])),
                longest_progress_interval_seconds=minimum_progress,failure=failure,
                processing_p95_seconds=float(np.percentile(latencies,95)),processing_max_seconds=max(latencies))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    grid=room();grid.save(out/'map.npz')
    results=[]
    for heading in (0,-30,30,60,180):
        result=run_case(grid,heading,out/('heading_'+str(heading)+'.jsonl'))
        results.append(result);print(json.dumps(result),flush=True)
    sources=sorted(Path(__file__).parent.glob('*.py'))
    report=dict(passed=all(x['passed'] for x in results),backend='synthetic-2d-lidar-with-relative-odometry',
                hardware_verified=False,tx1_native_verified=False,imu_fusion_verified=False,
                external_slam_verified=False,criteria=dict(final_distance_m=.20,max_seconds=70,collisions=0),
                cases=results,source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sources})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
