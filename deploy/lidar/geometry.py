"""Strict ROS-independent point cloud decoding and planar geometry."""
import math
import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0]+c*b[0]-s*b[1], a[1]+s*b[0]+c*b[1], wrap(a[2]+b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c*a[0]-s*a[1], s*a[0]-c*a[1], -a[2])


def quaternion_matrix(xyzw):
    q = np.asarray(xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .01:
        raise ValueError('Quaternion must be finite and normalized')
    x,y,z,w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def decode_xyz(message, max_points=500000):
    """Decode PointCloud2, honoring row padding, field offsets and byte order.

    Only documented XYZ float fields are consumed; no assumption about the
    vendor's intensity/ring/per-point time layout or motion compensation.
    """
    w,h,step,row = (int(message.width), int(message.height),
                    int(message.point_step), int(message.row_step))
    if min(w,h,step) <= 0 or w*h > max_points or row < w*step or len(message.data) != row*h:
        raise ValueError('Malformed or oversized PointCloud2 layout')
    fields = {f.name:f for f in message.fields}
    if len(fields) != len(message.fields):
        raise ValueError('Duplicate PointCloud2 fields')
    result = []
    spans = []
    for name in ('x','y','z'):
        f = fields.get(name)
        if f is None or f.datatype not in (7,8) or f.count != 1:
            raise ValueError('PointCloud2 requires scalar float XYZ fields')
        size = 4 if f.datatype == 7 else 8
        if f.offset < 0 or f.offset+size > step:
            raise ValueError('PointCloud2 field outside point_step')
        if any(f.offset < end and f.offset+size > start for start,end in spans):
            raise ValueError('Overlapping PointCloud2 XYZ fields')
        spans.append((f.offset,f.offset+size))
        dtype = ('>' if message.is_bigendian else '<')+('f4' if size == 4 else 'f8')
        result.append(np.ndarray((h,w), dtype=dtype, buffer=bytes(message.data),
                                 offset=f.offset, strides=(row,step)).reshape(-1))
    return np.stack(result, axis=1).astype(float)


def project_cloud(points, translation, quaternion, *, bins=72, min_z=-.20,
                  max_z=.50, min_range=.10, max_range=8.):
    """Return nearest obstacle rays in base_link; empty bins stay unknown (0).

    Heights are relative to base_link, NOT the floor. Configure using measured
    sensor mounting/body height. No ground segmentation or deskew is invented.
    Adjacent missing beams are never filled by interpolation or max range.
    """
    if (not isinstance(bins,int) or bins < 24 or bins > 720 or
            not all(math.isfinite(v) for v in (min_z,max_z,min_range,max_range)) or
            min_z >= max_z or not 0 < min_range < max_range):
        raise ValueError('Invalid cloud projection configuration')
    p, t = np.asarray(points,dtype=float), np.asarray(translation,dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError('Invalid cloud or transform')
    p = p[np.isfinite(p).all(axis=1)] @ quaternion_matrix(quaternion).T + t
    radius = np.linalg.norm(p[:,:2],axis=1)
    keep = ((p[:,2] >= min_z)&(p[:,2] <= max_z)&(radius >= min_range)&(radius <= max_range))
    p, radius = p[keep], radius[keep]
    # Nearest endpoint at its measured angle, not the center of a coarse bin.
    indices = np.floor((np.arctan2(p[:,1],p[:,0])+math.pi)/(2*math.pi)*bins).astype(int)%bins
    angles = -math.pi+(np.arange(bins)+.5)*2*math.pi/bins
    ranges = np.zeros(bins)
    for i in np.argsort(radius)[::-1]:
        k=indices[i]
        ranges[k]=radius[i]
        angles[k]=math.atan2(p[i,1],p[i,0])
    return angles, ranges


def sensor_stamp(source, ros_now, monotonic_now, max_age=.25):
    """Preserve source age; receipt time may never make stale data fresh.

    Requires source and ros_now in the same synchronized ROS time domain.
    A device clock offset must be calibrated upstream, not inferred per packet.
    """
    if not all(math.isfinite(v) for v in (source,ros_now,monotonic_now,max_age)):
        raise ValueError('Nonfinite timestamp')
    age=ros_now-source
    if age < 0 or age > max_age or max_age <= 0:
        raise ValueError('Future or stale sensor timestamp / clocks unsynchronized')
    return monotonic_now-age


class ClockTranslator:
    """Anchor a live ROS/system clock to monotonic time; latch clock jumps.

    For live, synchronized clocks only. Accelerated /clock replay must use the
    portable event replay instead of pretending that it is live wall time.
    """
    def __init__(self, tolerance=.05):
        self.offset = None
        self.tolerance = tolerance
        self.failed = False

    def convert(self, source, ros_now, monotonic_now, max_age=.20):
        if self.failed:
            raise ValueError('Clock discontinuity latched; start a new capture session')
        translated = sensor_stamp(source, ros_now, monotonic_now, max_age)
        offset = ros_now-monotonic_now
        if self.offset is None:
            self.offset = offset
        elif abs(offset-self.offset) > self.tolerance:
            self.failed = True
            raise ValueError('ROS/system clock jumped relative to monotonic time')
        return translated
