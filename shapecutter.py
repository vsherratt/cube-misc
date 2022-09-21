from functools import partial, lru_cache
import itertools

import numpy as np
import shapely.affinity
import shapely.ops
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


def load_naturalearth(country="GBR"):
    """
    Load country borders from Natural Earth

    Arguments:
        country: 3-character country ISO code.

    Returns:
        Shapely geometry representing the country border
    """
    # Read all country borders
    filename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(filename)

    # Extract the requested country
    geometries = [
        record.geometry
        for record in reader.records()
        if record.attributes["ISO_A3"] == country
    ]
    geometry = shapely.ops.unary_union(geometries)
    if geometry.is_empty:
        raise ValueError(f"country not found: {country}")

    return geometry


@lru_cache()
def get_transformer(source, target):
    """
    Create a function to transform shapes from one CRS to another

    Arguments:
        source: Current CRS of shapes.
        target: CRS to transform to.

    Returns:
        Transformation function (Geometry -> Geometry)
    """
    if source is None or target is None or source == target:
        return lambda geom: geom

    def transform(xs, ys, zs=None):
        xs = np.array(xs)
        ys = np.array(ys)
        if zs is not None:
            zs = np.array(zs)
        tfpoints = target.transform_points(source, xs, ys, zs)

        if zs is None:
            tfpoints = tfpoints[:, 0:2]
        return tfpoints.T

    return partial(shapely.ops.transform, transform)


def match_cube_coords(geometry, cube, crs=None):
    """
    Transform a shape to the same CRS and modulus as a cube

    Arguments:
        geometry: Shape to transform.
        cube: Cube to match.
        crs: CRS of the shape, if different to the cube's

    Returns:
        Transformed shape.
    """
    # Ensure coordinate systems match
    if crs is not None:
        cube_crs = cube.coord_system().as_cartopy_crs()
        tf = get_transformer(crs, cube_crs)
        geometry = tf(geometry)

    # Translate by the modulus as necessary to match the range of
    # x-coordinates used in the cube
    xcoord, _ = get_xy_coords(cube)
    modulus = xcoord.units.modulus
    if modulus is not None:
        geom_min, _, geom_max, _ = geometry.bounds  # = (xmin, ymin, xmax, ymax)
        cube_min = xcoord.points.min()
        cube_max = xcoord.points.max()

        # Add modulus as much as needed
        while geom_max < cube_min:
            geometry = shapely.affinity.translate(geometry, modulus)
            geom_min, _, geom_max, _ = geometry.bounds

        # Subtract modulus as much as needed
        while geom_min > cube_max:
            geometry = shapely.affinity.translate(geometry, -modulus)
            geom_min, _, geom_max, _ = geometry.bounds

    return geometry


def get_xy_coords(cube):
    """
    Find a cube's X and Y dimension coords

    Arguments:
        cube: Cube to get coords of.

    Returns:
        DimCoords in the X and Y directions.
    """
    xcoord = cube.coord(axis="x", dim_coords=True)
    ycoord = cube.coord(axis="y", dim_coords=True)
    return xcoord, ycoord


def get_intersection_weights(cube, geometry, match_cube_dims=False):
    """
    Calculate what proportion of each grid cell intersects a given shape

    Arguments:
        cube: Cube defining a grid.
        geometry: Shape to intersect.
        match_cube_dims:
            Whether to match cube shape or not:

            - If False (the default), the returned array will have shape (x, y)
            - If True, its shape will be compatible with the cube

    Returns:
        Intersection weights.
    """
    # Determine output shape
    xcoord, ycoord = get_xy_coords(cube)
    ndim = 2
    xdim = 0
    ydim = 1
    if match_cube_dims:
        # Make broadcastable to cube shape
        ndim = cube.ndim
        xdim = cube.coord_dims(xcoord)[0]
        ydim = cube.coord_dims(ycoord)[0]
    shape = [1] * ndim
    shape[xdim] = len(xcoord.points)
    shape[ydim] = len(ycoord.points)

    # Ensure coords have bounds
    if not xcoord.has_bounds():
        xcoord.guess_bounds()
    if not ycoord.has_bounds():
        ycoord.guess_bounds()

    # Calculate the weights
    weights = np.zeros(shape)
    indices = [range(n) for n in shape]
    for i in itertools.product(*indices):
        x0, x1 = xcoord.bounds[i[xdim]]
        y0, y1 = ycoord.bounds[i[ydim]]
        cell = sgeom.box(x0, y0, x1, y1)
        weight = cell.intersection(geometry).area / cell.area
        weights[i] = weight

    return weights


def mask_shape(cube, geometry, crs=None, threshold=0):
    """
    Mask a cube according to a shape

    Arguments:
        cube: Cube to mask.
        geometry: Shape to use.
        crs: CRS of the shape, if different to the cube's.
        threshold: Minimum proportion of cell that must intersect the shape.

    Returns:
        Cube with cells outside the shape masked.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be in range 0-1")

    geometry = match_cube_coords(geometry, cube, crs=crs)

    # Mask points below the specified threshold
    weights = get_intersection_weights(cube, geometry, True)
    if threshold == 0:
        mask2d = weights == 0
    else:
        mask2d = weights < threshold

    # Apply mask to the cube
    mask = np.broadcast_to(mask2d, cube.shape)
    data = np.ma.array(cube.data, mask=mask)
    return cube.copy(data=data)


def mask_country(cube, country="GBR", threshold=0):
    """
    Mask a cube according to Natural Earth country borders

    Arguments:
        cube: Cube to mask.
        country: 3-character country ISO code.
        threshold: Minimum proportion of cell that must intersect the shape.

    Returns:
        Cube with cells outside the country masked.
    """
    geometry = load_naturalearth(country)
    crs = ccrs.Geodetic()
    return mask_shape(cube, geometry, crs=crs, threshold=threshold)
