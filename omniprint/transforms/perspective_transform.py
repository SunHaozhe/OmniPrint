import random
import numpy as np 
from PIL import Image 


_plan_square = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
_middle_point_coordinates = np.array([0.5, 0.5], dtype=np.float64)

def compute_perspective_params(plan_1, plan_2, width_, height_):
    """
    Given the coordinates of the four corners of the first quadrilateral 
    and the coordinates of the four corners of the second quadrilateral, 
    compute the perspective transform that maps a new point in the first 
    quadrilateral onto the appropriate position on the second quadrilateral. 
    
    https://web.archive.org/web/20150222120106/xenia.media.mit.edu/~cwren/interpolator/ 
    
    plan_1, plan_2:
        list of 4 tuples, each tuple contains two float 
        plan_src is the first quadrilateral
        plan_target is the second quadrilateral
    p1, p2:
        tuple of two float (corner coordinates normalized to [0, 1] range)
        p1 corresponds to coordinates in the source plan (x, y)
        p2 corresponds to coordinates in the target plan (X, Y) 
    """
    A = []
    plan_1 = np.array(plan_1, copy=True)
    plan_2 = np.array(plan_2, copy=True)
    plan_1[:, 0] = plan_1[:, 0] * width_
    plan_1[:, 1] = plan_1[:, 1] * height_
    plan_2[:, 0] = plan_2[:, 0] * width_
    plan_2[:, 1] = plan_2[:, 1] * height_
    
    for p1, p2 in zip(plan_1, plan_2):
        A.append([p1[0], p1[1], 1, 0, 0, 0, - p2[0] * p1[0], - p2[0] * p1[1]])
        A.append([0, 0, 0, p1[0], p1[1], 1, - p2[1] * p1[0], - p2[1] * p1[1]])
    A = np.array(A, dtype=np.float)
    b = np.array(plan_2).reshape(8)
    try:
        res = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    except:
        res = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(b)
    return np.array(res).reshape(8)


def is_convex_quadrilateral(quadrilateral):
    len_ = len(quadrilateral)
    indices = [[1, 0, 2], [0, 2, 3], [2, 3, 1], [3, 1, 0]]
    for idx_1, idx_2, idx_3 in indices:
        edge_1 = quadrilateral[idx_2] - quadrilateral[idx_1]
        edge_2 = quadrilateral[idx_3] - quadrilateral[idx_2] 
        # 2D cross product 
        if edge_1[0] * edge_2[1] - edge_2[0] * edge_1[1] >= 0:
            return False
    return True


def random_quadrilateral(gaussian_std=0.05, max_nb_trials=5): 
    for _ in range(max_nb_trials):
        random_2d_points = []
        
        for i in range(8):
            random_2d_points.append(random.gauss(0, gaussian_std))
        random_2d_points = np.array(random_2d_points).reshape((4, 2))
        
        # top right corner
        random_2d_points[1, 0] *= - 1
        # bottom left corner
        random_2d_points[2, 1] *= - 1
        # bottom right corner
        random_2d_points[3, 0] *= - 1
        random_2d_points[3, 1] *= - 1
        
        quadrilateral = random_2d_points + _plan_square 
        if is_convex_quadrilateral(quadrilateral):
            return quadrilateral
    return _plan_square


def transform(img, mask, quadrilateral=None, gaussian_std=0.05, 
              return_perspective_params=False):
    """
    img.mode must be "RGB"
    mask.mode must be "L"
    
    quadrilateral and gaussian_std cannot be both None. 

    If both are provided, gaussian_std will be used, 
    which means that random quadrilateral will be generated. 
    """ 
    if quadrilateral is not None:
        assert is_convex_quadrilateral(quadrilateral), \
        "Corner points do not constitute a convex quadrilateral."
    if gaussian_std is not None:
        quadrilateral = random_quadrilateral(gaussian_std=gaussian_std)
    perspective_params = compute_perspective_params(quadrilateral, _plan_square, 
                                                    img.size[0], img.size[1])
    img = img.transform(img.size, Image.PERSPECTIVE, 
                        perspective_params, Image.BICUBIC, fillcolor=(255, 255, 255))
    mask = mask.transform(mask.size, Image.PERSPECTIVE, 
                        perspective_params, Image.BICUBIC, fillcolor=0)
    if not return_perspective_params:
        return img, mask
    else:
        return img, mask, perspective_params 









