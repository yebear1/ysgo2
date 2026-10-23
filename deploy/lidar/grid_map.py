"""Versioned occupancy map artifacts with content-derived identity."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


class SavedGrid:
    def __init__(self, data, resolution, origin, frame_id='map'):
        data=np.asarray(data)
        origin=np.asarray(origin,dtype=float)
        if (data.ndim != 2 or min(data.shape) < 3 or data.size > 4000000 or
                not np.isfinite(data).all() or np.any((data < -1)|(data > 100)) or
                np.any(data != np.floor(data)) or not np.isfinite(resolution) or resolution <= 0 or
                origin.shape != (2,) or not np.isfinite(origin).all() or frame_id != 'map'):
            raise ValueError('Invalid axis-aligned map grid')
        self.data=data.astype(np.int8).copy()
        self.resolution=float(resolution)
        self.origin=tuple(float(x) for x in origin)
        self.frame_id=frame_id
        if not np.any(self.data >= 65) or not np.any(self.data == 0):
            raise ValueError('Map needs observed free space and occupied structure')
        meta=json.dumps([self.resolution,self.origin,list(data.shape),frame_id],separators=(',',':'))
        self.map_id='sha256:'+hashlib.sha256(meta.encode()+self.data.tobytes()).hexdigest()
        self.distance=cv2.distanceTransform((self.data < 65).astype(np.uint8),cv2.DIST_L2,5)*self.resolution
        self.data.setflags(write=False)
        self.distance.setflags(write=False)

    @property
    def grid(self):
        h,w=self.data.shape
        return dict(data=self.data.copy(),resolution=self.resolution,origin=self.origin,width=w,height=h)

    def cells(self, xy):
        return np.floor((np.asarray(xy)-self.origin)/self.resolution).astype(int)

    def values(self, xy):
        c=self.cells(xy)
        inside=(c[...,0]>=0)&(c[...,1]>=0)&(c[...,0]<self.data.shape[1])&(c[...,1]<self.data.shape[0])
        x=np.clip(c[...,0],0,self.data.shape[1]-1); y=np.clip(c[...,1],0,self.data.shape[0]-1)
        return np.where(inside,self.data[y,x],-1), np.where(inside,self.distance[y,x],np.inf)

    def save(self, path):
        path=Path(path)
        # np.savez appends .npz; explicitly require it so identity/path remain clear.
        if path.suffix != '.npz': raise ValueError('Map path must end in .npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream,data=self.data,resolution=self.resolution,
                                origin=self.origin,frame_id=self.frame_id,map_id=self.map_id)

    @classmethod
    def load(cls, path):
        with np.load(path,allow_pickle=False) as saved:
            result=cls(saved['data'],float(saved['resolution']),saved['origin'],str(saved['frame_id']))
            if str(saved['map_id']) != result.map_id: raise ValueError('Map content hash mismatch')
            return result

    @classmethod
    def from_message(cls, msg):
        o=msg.info.origin
        # Rotated grids must be reprojected upstream; do not silently ignore yaw.
        q=np.array([o.orientation.x,o.orientation.y,o.orientation.z,o.orientation.w])
        if (msg.header.frame_id != 'map' or not np.isfinite(q).all() or
                np.linalg.norm(q[:3]) > 1e-6 or abs(abs(q[3])-1) > 1e-6
                or not np.isfinite(o.position.z) or abs(o.position.z)>1e-6):
            raise ValueError('Expected axis-aligned OccupancyGrid in map frame')
        w,h=int(msg.info.width),int(msg.info.height)
        if w <= 0 or h <= 0 or w*h > 4000000 or len(msg.data) != w*h:
            raise ValueError('Invalid OccupancyGrid dimensions')
        return cls(np.asarray(msg.data).reshape(h,w),msg.info.resolution,(o.position.x,o.position.y))
