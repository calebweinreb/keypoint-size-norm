from matplotlib import collections
from matplotlib import patches
import numpy as np

from kpsn.util.keypt_io import keypt_parents

def plot_mouse(ax, keypt_frame, xaxis, yaxis, scatter_kw = {}, line_kw = {}):
    
    for i, parent in enumerate(keypt_parents):
        if i == 0:
            continue
        curr_child = keypt_frame[i]
        curr_parent = keypt_frame[parent]
        ax.plot((curr_child[xaxis], curr_parent[xaxis]),
                (curr_child[yaxis], curr_parent[yaxis]),
                **{'color':'black', **line_kw})
        
    ax.scatter(keypt_frame[..., xaxis], keypt_frame[..., yaxis],
        **{'s': 3, **scatter_kw})
    
    ax.set_aspect(1.)
    

# def apply_transform_x(trasnsform)

# def bar_label(fig, ax, text, bar_width)


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    # https://gist.github.com/pv/8036995

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



def region_polygons(regions, vertices, **kws):
    """
    Returns:
    polys: matplotlib.patches.Polygon
    """
    coll = collections.PatchCollection(
        [patches.Polygon(vertices[reg]) for reg in regions]
    )
    coll.set(**kws)
    return coll



def _adjust_bounds(ax, points, incl_origin = False):
    if incl_origin:
        points = np.concatenate([points, [[0, 0]]])
    margin = 0.1 * np.ptp(points, axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])
