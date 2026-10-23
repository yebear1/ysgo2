"""Bounded 2-D saved-grid relocalization, independent of odometry origins.

An experimental flat-indoor matcher, not a 3-D LIO/SLAM implementation. Startup
searches all headings inside a configured position region. Ambiguous matches
are rejected. Odometry only predicts scan matching; it never verifies a map.
"""
from dataclasses import dataclass
import math
import numpy as np
from .geometry import compose, inverse, wrap
from ..navigation.contracts import PoseEstimate


@dataclass(frozen=True)
class MatchResult:
    pose: object
    reason: str
    score: float = 0.
    inlier_ratio: float = 0.
    ambiguity_gap: float = 0.
    confirmations: int = 0


class GridLocalizer:
    def __init__(self, saved_map, search_center, search_radius=1., confirmations=3):
        center=np.asarray(search_center,dtype=float)
        if (center.shape != (2,) or not np.isfinite(center).all() or
                not math.isfinite(search_radius) or not 0 <= search_radius <= 3. or
                type(confirmations) is not int or confirmations < 2):
            raise ValueError('A bounded starting region and >=2 confirmations are required')
        self.map=saved_map
        self.center=center
        self.radius=search_radius
        self.required=confirmations
        self.last_stamp=None
        self.last_odom=None
        self.candidate=None
        self.verified=False
        self.count=0
        self.last_result=MatchResult(None,'waiting_for_scan')

    def reset(self, reason='reset'):
        self.last_odom=self.candidate=None
        self.verified=False
        self.count=0
        self.last_result=MatchResult(None,reason)
        return self.last_result

    def _score(self, poses, points):
        poses=np.asarray(poses)
        c,s=np.cos(poses[:,2]),np.sin(poses[:,2])
        endpoints=np.stack((c[:,None]*points[None,:,0]-s[:,None]*points[None,:,1],
                            s[:,None]*points[None,:,0]+c[:,None]*points[None,:,1]),axis=-1)
        v,d=self.map.values(endpoints+poses[:,None,:2])
        # Endpoints score distance to an OBSERVED occupied surface. A noisy
        # return may land just behind a one-cell wall in an unknown cell. That
        # is valid surface evidence, not evidence that the unknown cell is free.
        # Unknown regions with no nearby observed surface score zero; base and
        # ray prefixes below must still be explicitly observed free.
        likelihood=np.where(d<=.30,np.exp(-.5*(d/.12)**2),0.)
        support=np.mean(d <= .18,axis=1)
        # Check several points along each ray: endpoints alone can match walls
        # through another room. Unknown/occupied prefixes reduce the score.
        free=[]
        for fraction in (.25,.50,.75,.90):
            fv,_=self.map.values(endpoints*fraction+poses[:,None,:2])
            free.append(np.mean((fv>=0)&(fv<35),axis=1))
        free=np.mean(free,axis=0)
        base,_=self.map.values(poses[:,:2])
        scores=np.mean(likelihood,axis=1)*free
        scores[(base<0)|(base>=35)]=0
        return scores,support

    def _rank(self, poses, points):
        scores=[]; support=[]
        for chunk in np.array_split(poses,max(1,int(math.ceil(len(poses)/512)))):
            a,b=self._score(chunk,points); scores.extend(a); support.extend(b)
        return np.asarray(scores),np.asarray(support)

    @staticmethod
    def _distinct(a,b,xy=.35,yaw=math.radians(20)):
        return np.linalg.norm(a[:2]-b[:2]) > xy or abs(wrap(a[2]-b[2])) > yaw

    def _search(self, points, prediction):
        if prediction is None:
            offsets=np.arange(-self.radius,self.radius+1e-8,.20) if self.radius else np.array([0.])
            # Include the actual configured center even for a nonmultiple radius.
            offsets=np.unique(np.r_[offsets,0.])
            xs=self.center[0]+offsets; ys=self.center[1]+offsets
            yaws=np.arange(-math.pi,math.pi,math.radians(10))
        else:
            xs=prediction[0]+np.arange(-.15,.151,.05)
            ys=prediction[1]+np.arange(-.15,.151,.05)
            yaws=prediction[2]+np.arange(-8,8.1,2)*math.pi/180
        poses=np.stack(np.meshgrid(xs,ys,yaws,indexing='ij'),axis=-1).reshape(-1,3)
        scores,support=self._rank(poses,points)
        seeds=[]
        for i in np.argsort(-scores):
            if all(self._distinct(poses[i],p,xy=.25,yaw=math.radians(15)) for p in seeds):
                seeds.append(poses[i])
            if len(seeds)==8: break
        refined=[]
        for p in seeds:
            offsets=np.stack(np.meshgrid(np.arange(-.10,.101,.05),np.arange(-.10,.101,.05),
                                         np.arange(-5,5.1,1)*math.pi/180,indexing='ij'),axis=-1).reshape(-1,3)
            refined.append(offsets+p)
        poses=np.concatenate(refined)
        if prediction is None:
            poses=poses[np.all(np.abs(poses[:,:2]-self.center)<=self.radius+1e-8,axis=1)]
        scores,support=self._rank(poses,points)
        order=np.argsort(-scores); best=order[0]
        second=next((scores[i] for i in order if self._distinct(poses[i],poses[best])),0.)
        pose=poses[best].copy(); pose[2]=wrap(pose[2])
        return pose,float(scores[best]),float(support[best]),float(scores[best]-second)

    def update(self, scan, odom_xy_yaw, now):
        # A duplicate may keep the old result, never increment confirmations or
        # refresh its source stamp. Reverse time invalidates the whole session.
        try:
            stamp=float(scan.stamp); odom=np.asarray(odom_xy_yaw,dtype=float)
            if not math.isfinite(now) or not math.isfinite(stamp) or not 0 <= now-stamp <= .25:
                return self.reset('scan_stale_or_future')
            if self.last_stamp is not None and stamp < self.last_stamp:
                return self.reset('scan_clock_regressed')
            if self.last_stamp == stamp:
                return self.last_result
            self.last_stamp=stamp
            angles=np.asarray(scan.angles,dtype=float); ranges=np.asarray(scan.ranges,dtype=float)
            if (scan.frame_id != 'base_link' or odom.shape != (3,) or not np.isfinite(odom).all() or
                    angles.ndim != 1 or angles.shape != ranges.shape or len(angles)>720 or
                    not np.isfinite(angles).all() or not np.isfinite(ranges).all() or
                    not math.isfinite(scan.max_range) or scan.max_range <= 0 or
                    np.any(ranges < 0) or np.any(ranges > scan.max_range)):
                return self.reset('invalid_scan_or_odometry')
            valid=(ranges>.10)&(ranges<scan.max_range*.995)
            if valid.sum()<32:
                return self.reset('insufficient_returns')
            ordered=np.sort(angles[valid]%(2*math.pi))
            gaps=np.diff(np.r_[ordered,ordered[0]+2*math.pi])
            if gaps.min()<=1e-6 or gaps.max()>math.radians(35):
                return self.reset('insufficient_angular_coverage')
            points=np.stack((np.cos(angles[valid])*ranges[valid],np.sin(angles[valid])*ranges[valid]),axis=1)
            # Fixed upper bound on work; uniformly sample angular returns.
            if len(points)>144: points=points[np.linspace(0,len(points)-1,144).astype(int)]
            prediction=None
            if self.candidate is not None and self.last_odom is not None:
                delta=compose(inverse(self.last_odom),odom)
                if np.linalg.norm(delta[:2])>.4 or abs(delta[2])>.5:
                    return self.reset('odometry_discontinuity')
                prediction=np.asarray(compose(self.candidate,delta))
            # Unverified candidates continue global search to retain alternatives.
            best,score,inliers,gap=self._search(points,prediction if self.verified else None)
            if score<.70 or inliers<.80:
                return self.reset('map_match_rejected')
            if not self.verified and gap<.06:
                return self.reset('map_match_ambiguous')
            if prediction is not None and (np.linalg.norm(best[:2]-prediction[:2])>.20 or
                                           abs(wrap(best[2]-prediction[2]))>math.radians(12)):
                return self.reset('map_match_inconsistent')
            self.count+=1; self.candidate=best; self.last_odom=odom
            self.verified=self.count>=self.required
            pose=PoseEstimate(stamp,tuple(float(x) for x in best),self.map.map_id,self.verified)
            self.last_result=MatchResult(pose,'matched' if self.verified else 'confirming',score,inliers,gap,self.count)
            return self.last_result
        except (ValueError,TypeError,OverflowError,IndexError):
            return self.reset('invalid_localization_input')
