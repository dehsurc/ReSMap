from typing import List, Optional

import numpy as np
from shapely import ops
from shapely.geometry import LinearRing, LineString, Polygon, box
from shapely.geometry.base import BaseGeometry


def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    ''' Split Multi-geoms to list and check is valid or is empty.
        
    Args:
        geom (BaseGeometry): geoms to be split or validate.
    
    Returns:
        geometries (List): list of geometries.
    '''
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon', 
        'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom,]
        else:
            return []


def get_ped_crossing_contour(polygon: Polygon, 
                             local_patch: box) -> Optional[LineString]:
    ''' Extract ped crossing contours to get a closed polyline.

    Args:
        polygon (Polygon): ped crossing polygon to be extracted.
        local_patch (tuple): local patch params
    
    Returns:
        line (LineString): a closed line
    '''

    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = ext.intersection(local_patch)
    if lines.type != 'LineString':
        # remove points in intersection results
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)
        
        # same instance but not connected.
        if lines.type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))
            
            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)

    if not lines.is_empty:
        start = list(lines.coords[0])
        end = list(lines.coords[-1])
        if not np.allclose(start, end, atol=1e-3):
            new_line = list(lines.coords)
            new_line.append(start)
            lines = LineString(new_line) # make ped cross closed
        return lines
    
    return None
