"""Portable JSONL replay format. No ROS, DDS, sockets or hardware control."""
import json
import math
from pathlib import Path
from .session import AnalysisSink, AnalysisSession

SCHEMA='go2-lidar-replay-v1'


class EventWriter:
    def __init__(self,path,metadata):
        self.stream=Path(path).open('x',encoding='utf-8')
        self.write(dict(type='header',schema=SCHEMA,**metadata))
    def write(self,event):
        self.stream.write(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n')
        self.stream.flush()
    def close(self):self.stream.close()


def events(path):
    with Path(path).open(encoding='utf-8') as stream:
        first=stream.readline()
        header=json.loads(first)
        if header.get('type')!='header' or header.get('schema')!=SCHEMA:
            raise ValueError('Unsupported recording schema')
        yield header
        previous=-math.inf
        for line in stream:
            if len(line)>1024*1024:raise ValueError('Oversized event')
            event=json.loads(line)
            t=event.get('t')
            if type(t) not in (int,float) or not math.isfinite(t) or t<previous:
                raise ValueError('Recording time must be finite and monotonic')
            previous=t
            yield event


def replay(path,saved_map,search_center,search_radius=1.,goal=None):
    """Evaluate captured poses/commands; this is NOT closed-loop robot success.

    Replay uses recorded sensor age and adds actual matching compute time before
    dispatch. No sleeping, synthesized sensor refresh or autonomous rearming.
    """
    session = AnalysisSession(saved_map, search_center, search_radius, goal)
    source = iter(events(path))
    header = next(source)
    if header.get('map_id') not in (None, saved_map.map_id):
        raise ValueError('Recording is tied to a different saved map')
    records = []
    try:
        for event in source:
            records.append(session.process(event))
    finally:
        final_decision=session.close()
    return dict(schema=SCHEMA, source=header, map_id=saved_map.map_id, frames=session.sequence,
                hardware_verified=False, closed_loop_navigation_verified=False,
                transport='analysis-only-memory-sink', ever_armed=session.ever_armed,
                final_decision=final_decision.__dict__, records=records)
