# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
"""
Fast geometry operations for nuPlan feature extraction.
Optimized Cython implementations of hot path functions.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport cos, sin, atan2, sqrt, M_PI

np.import_array()

# Type definitions
ctypedef np.float64_t DTYPE_f64
ctypedef np.float32_t DTYPE_f32
ctypedef np.int32_t DTYPE_i32


cdef inline double _normalize_angle(double angle) nogil:
    """Normalize angle to [-pi, pi]."""
    while angle > M_PI:
        angle -= 2.0 * M_PI
    while angle < -M_PI:
        angle += 2.0 * M_PI
    return angle


cdef inline void _transform_point(
    double x, double y,
    double ego_x, double ego_y,
    double cos_h, double sin_h,
    double* out_x, double* out_y
) nogil:
    """Transform a single point to ego frame."""
    cdef double dx = x - ego_x
    cdef double dy = y - ego_y
    out_x[0] = cos_h * dx - sin_h * dy
    out_y[0] = sin_h * dx + cos_h * dy


def transform_points_to_ego_frame(
    np.ndarray[DTYPE_f64, ndim=2] points,
    double ego_x,
    double ego_y,
    double ego_heading
):
    """
    Vectorized coordinate transformation from world to ego frame.
    
    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Array of (x, y) coordinates in world frame
    ego_x, ego_y : float
        Ego vehicle position in world frame
    ego_heading : float
        Ego vehicle heading in radians
    
    Returns
    -------
    np.ndarray, shape (N, 2)
        Transformed points in ego frame
    """
    cdef Py_ssize_t n = points.shape[0]
    cdef np.ndarray[DTYPE_f64, ndim=2] result = np.empty((n, 2), dtype=np.float64)
    
    cdef double cos_h = cos(-ego_heading)
    cdef double sin_h = sin(-ego_heading)
    
    cdef Py_ssize_t i
    cdef double x, y
    
    for i in range(n):
        x = points[i, 0]
        y = points[i, 1]
        _transform_point(x, y, ego_x, ego_y, cos_h, sin_h, &result[i, 0], &result[i, 1])
    
    return result


def transform_points_to_ego_frame_2d(
    np.ndarray[DTYPE_f64, ndim=1] x,
    np.ndarray[DTYPE_f64, ndim=1] y,
    double ego_x,
    double ego_y,
    double ego_heading
):
    """
    Transform 1D arrays of x, y coordinates to ego frame.
    
    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Coordinate arrays in world frame
    ego_x, ego_y : float
        Ego vehicle position
    ego_heading : float
        Ego vehicle heading in radians
    
    Returns
    -------
    tuple of (local_x, local_y) arrays
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray[DTYPE_f64, ndim=1] local_x = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_f64, ndim=1] local_y = np.empty(n, dtype=np.float64)
    
    cdef double cos_h = cos(-ego_heading)
    cdef double sin_h = sin(-ego_heading)
    
    cdef Py_ssize_t i
    
    for i in range(n):
        _transform_point(x[i], y[i], ego_x, ego_y, cos_h, sin_h, &local_x[i], &local_y[i])
    
    return local_x, local_y


def interpolate_lane_points(
    np.ndarray[DTYPE_f64, ndim=2] coords,
    Py_ssize_t num_points
):
    """
    Fast lane point interpolation using linear sampling.
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Lane coordinates
    num_points : int
        Number of points to interpolate
    
    Returns
    -------
    np.ndarray, shape (num_points, 2)
        Interpolated points
    """
    cdef np.ndarray[DTYPE_f64, ndim=2] result = np.zeros((num_points, 2), dtype=np.float64)
    
    cdef Py_ssize_t n = coords.shape[0]
    if n < 2:
        return result
    
    # Compute cumulative distances along the polyline
    cdef Py_ssize_t i
    cdef double total_length = 0.0
    cdef double dx, dy
    
    # Simple linear interpolation along the polyline
    cdef double segment_length
    cdef double target_dist, current_dist
    cdef Py_ssize_t current_idx
    cdef double t
    
    # Calculate total length
    for i in range(n - 1):
        dx = coords[i + 1, 0] - coords[i, 0]
        dy = coords[i + 1, 1] - coords[i, 1]
        total_length += sqrt(dx * dx + dy * dy)
    
    if total_length == 0:
        return result
    
    cdef double step = total_length / (num_points - 1) if num_points > 1 else 0
    
    # Generate interpolated points
    cdef Py_ssize_t j
    current_dist = 0.0
    current_idx = 0
    
    for j in range(num_points):
        target_dist = j * step
        
        # Find the segment containing target_dist
        while current_idx < n - 1:
            dx = coords[current_idx + 1, 0] - coords[current_idx, 0]
            dy = coords[current_idx + 1, 1] - coords[current_idx, 1]
            segment_length = sqrt(dx * dx + dy * dy)
            
            if current_dist + segment_length >= target_dist - 1e-9:
                break
            
            current_dist += segment_length
            current_idx += 1
        
        if current_idx >= n - 1:
            # Last point
            result[j, 0] = coords[n - 1, 0]
            result[j, 1] = coords[n - 1, 1]
        else:
            # Interpolate within segment
            dx = coords[current_idx + 1, 0] - coords[current_idx, 0]
            dy = coords[current_idx + 1, 1] - coords[current_idx, 1]
            segment_length = sqrt(dx * dx + dy * dy)
            
            if segment_length > 1e-9:
                t = (target_dist - current_dist) / segment_length
                if t < 0:
                    t = 0
                if t > 1:
                    t = 1
                result[j, 0] = coords[current_idx, 0] + t * dx
                result[j, 1] = coords[current_idx, 1] + t * dy
            else:
                result[j, 0] = coords[current_idx, 0]
                result[j, 1] = coords[current_idx, 1]
    
    return result


def batch_transform_neighbors(
    np.ndarray[DTYPE_f64, ndim=2] world_coords,
    double ego_x,
    double ego_y,
    double ego_heading,
    np.ndarray[DTYPE_f64, ndim=1] headings
):
    """
    Batch transform neighbor agent coordinates and headings to ego frame.
    
    Parameters
    ----------
    world_coords : np.ndarray, shape (N, 2)
        World coordinates of neighbor agents
    ego_x, ego_y : float
        Ego position
    ego_heading : float
        Ego heading in radians
    headings : np.ndarray, shape (N,)
        Headings of neighbor agents in world frame
    
    Returns
    -------
    tuple of (local_coords, local_headings)
        local_coords: np.ndarray, shape (N, 2)
        local_headings: np.ndarray, shape (N,)
    """
    cdef Py_ssize_t n = world_coords.shape[0]
    cdef np.ndarray[DTYPE_f64, ndim=2] local_coords = np.empty((n, 2), dtype=np.float64)
    cdef np.ndarray[DTYPE_f64, ndim=1] local_headings = np.empty(n, dtype=np.float64)
    
    cdef double cos_h = cos(-ego_heading)
    cdef double sin_h = sin(-ego_heading)
    
    cdef Py_ssize_t i
    cdef double dx, dy
    
    for i in range(n):
        dx = world_coords[i, 0] - ego_x
        dy = world_coords[i, 1] - ego_y
        local_coords[i, 0] = cos_h * dx - sin_h * dy
        local_coords[i, 1] = sin_h * dx + cos_h * dy
        local_headings[i] = _normalize_angle(headings[i] - ego_heading)
    
    return local_coords, local_headings


def fast_stack_features(
    list arrays,
    tuple target_shape,
    DTYPE_f32 fill_value = 0.0
):
    """
    Fast array stacking with pre-allocated output.
    
    Parameters
    ----------
    arrays : list of np.ndarray
        List of arrays to stack
    target_shape : tuple
        Target shape for the output array
    fill_value : float
        Value to fill remaining elements
    
    Returns
    -------
    np.ndarray
        Stacked array with target shape
    """
    cdef np.ndarray[DTYPE_f32, ndim=3] result = np.full(target_shape, fill_value, dtype=np.float32)
    
    cdef Py_ssize_t n_arrays = len(arrays)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t d0, d1, d2
    cdef np.ndarray[DTYPE_f32, ndim=2] arr
    
    cdef Py_ssize_t max_i = target_shape[0]
    cdef Py_ssize_t max_j = target_shape[1]
    cdef Py_ssize_t max_k = target_shape[2] if len(target_shape) > 2 else 1
    
    for idx in range(n_arrays):
        if idx >= max_i:
            break
        arr = arrays[idx]
        d0 = arr.shape[0]
        d1 = arr.shape[1] if arr.ndim > 1 else 1
        
        for i in range(d0):
            if i >= max_j:
                break
            for j in range(d1):
                if j >= max_k:
                    break
                result[idx, i, j] = arr[i, j]
    
    return result


def quaternion_to_heading_batch(
    np.ndarray[DTYPE_f64, ndim=1] qw,
    np.ndarray[DTYPE_f64, ndim=1] qx,
    np.ndarray[DTYPE_f64, ndim=1] qy,
    np.ndarray[DTYPE_f64, ndim=1] qz
):
    """
    Vectorized quaternion to heading conversion.
    
    Parameters
    ----------
    qw, qx, qy, qz : np.ndarray, shape (N,)
        Quaternion components
    
    Returns
    -------
    np.ndarray, shape (N,)
        Heading angles in radians
    """
    cdef Py_ssize_t n = qw.shape[0]
    cdef np.ndarray[DTYPE_f64, ndim=1] headings = np.empty(n, dtype=np.float64)
    
    cdef Py_ssize_t i
    cdef double siny_cosp, cosy_cosp
    
    for i in range(n):
        siny_cosp = 2.0 * (qw[i] * qz[i] + qx[i] * qy[i])
        cosy_cosp = 1.0 - 2.0 * (qy[i] * qy[i] + qz[i] * qz[i])
        headings[i] = atan2(siny_cosp, cosy_cosp)
    
    return headings


def compute_rotation_matrix(double heading):
    """
    Compute 2D rotation matrix.
    
    Parameters
    ----------
    heading : float
        Heading angle in radians
    
    Returns
    -------
    np.ndarray, shape (2, 2)
        Rotation matrix
    """
    cdef double cos_h = cos(heading)
    cdef double sin_h = sin(heading)
    
    cdef np.ndarray[DTYPE_f64, ndim=2] R = np.empty((2, 2), dtype=np.float64)
    R[0, 0] = cos_h
    R[0, 1] = -sin_h
    R[1, 0] = sin_h
    R[1, 1] = cos_h
    
    return R


def rotate_vectors(
    np.ndarray[DTYPE_f64, ndim=2] vectors,
    double heading
):
    """
    Rotate 2D vectors by heading angle.
    
    Parameters
    ----------
    vectors : np.ndarray, shape (N, 2)
        Vectors to rotate
    heading : float
        Rotation angle in radians
    
    Returns
    -------
    np.ndarray, shape (N, 2)
        Rotated vectors
    """
    cdef Py_ssize_t n = vectors.shape[0]
    cdef np.ndarray[DTYPE_f64, ndim=2] result = np.empty((n, 2), dtype=np.float64)
    
    cdef double cos_h = cos(heading)
    cdef double sin_h = sin(heading)
    
    cdef Py_ssize_t i
    cdef double vx, vy
    
    for i in range(n):
        vx = vectors[i, 0]
        vy = vectors[i, 1]
        result[i, 0] = cos_h * vx - sin_h * vy
        result[i, 1] = sin_h * vx + cos_h * vy
    
    return result


def compute_lane_vectors(
    np.ndarray[DTYPE_f64, ndim=2] sampled_points
):
    """
    Compute direction vectors between consecutive lane points.
    
    Parameters
    ----------
    sampled_points : np.ndarray, shape (N, 2)
        Sampled lane centerline points
    
    Returns
    -------
    np.ndarray, shape (N-1, 2)
        Direction vectors (normalized)
    """
    cdef Py_ssize_t n = sampled_points.shape[0]
    cdef np.ndarray[DTYPE_f64, ndim=2] vectors = np.zeros((n, 2), dtype=np.float64)
    
    cdef Py_ssize_t i
    cdef double dx, dy, vec_len
    
    for i in range(n - 1):
        dx = sampled_points[i + 1, 0] - sampled_points[i, 0]
        dy = sampled_points[i + 1, 1] - sampled_points[i, 1]
        vec_len = sqrt(dx * dx + dy * dy)
        if vec_len > 0:
            vectors[i, 0] = dx / vec_len
            vectors[i, 1] = dy / vec_len
    
    return vectors


def normalize_angles(np.ndarray[DTYPE_f64, ndim=1] angles):
    """
    Normalize array of angles to [-pi, pi].
    
    Parameters
    ----------
    angles : np.ndarray, shape (N,)
        Angles in radians
    
    Returns
    -------
    np.ndarray, shape (N,)
        Normalized angles
    """
    cdef Py_ssize_t n = angles.shape[0]
    cdef np.ndarray[DTYPE_f64, ndim=1] result = np.empty(n, dtype=np.float64)
    
    cdef Py_ssize_t i
    cdef double angle
    
    for i in range(n):
        angle = angles[i]
        while angle > M_PI:
            angle -= 2.0 * M_PI
        while angle < -M_PI:
            angle += 2.0 * M_PI
        result[i] = angle
    
    return result
