"""Synthetic fixtures ONLY; no hardware/model fidelity or SLAM claims."""
import math
import numpy as np
from .grid_map import SavedGrid
from ..navigation.contracts import RangeScan


def room(symmetric=False):
    data=np.zeros((121,161),dtype=np.int8)
    data[:2,:]=100; data[-2:,:]=100; data[:,:2]=100; data[:,-2:]=100
    if not symmetric:
        data[12:37,25:41]=100
        data[76:103,110:130]=100
        data[25:29,115:150]=100
        data[88:92,18:73]=100
    return SavedGrid(data,.05,(-4.,-3.))


def ray_scan(grid, pose, stamp=0., bins=72, noise=0., seed=0):
    rng=np.random.default_rng(seed)
    angles=np.linspace(-math.pi,math.pi,bins,endpoint=False)
    ranges=[]
    origin=np.asarray(pose[:2]); steps=np.arange(.10,8.,.02)
    for angle in angles+pose[2]:
        points=origin+steps[:,None]*np.array([math.cos(angle),math.sin(angle)])
        values,_=grid.values(points)
        hits=np.flatnonzero((values>=65)|(values<0))
        distance=steps[hits[0]] if len(hits) else 8.
        ranges.append(float(np.clip(distance+rng.normal(0,noise),.10,8.)))
    return RangeScan(stamp,tuple(angles),tuple(ranges),8.)
