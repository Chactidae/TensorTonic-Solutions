import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.array(T)
    points = np.array(points)
    single = (points.ndim == 1)
    if single:
        points = points.reshape(1, 3)

    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])

    transformed_hom = points_hom @ T.T

    transformed = transformed_hom[:, :3]
    if single:
        return transformed.ravel()
    return transformed