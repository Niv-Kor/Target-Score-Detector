import ContourClassifier as cntr
import Geometry2D as geo2D
import numpy as np
import cv2

def create_group_polygon(img, hits):
    '''
    Calculate the polygon that contours a group of hits.

    Parameters:
        {Numpy.array} img - The image in which the group appears
        {List} hits - [
                         {HitsManager.Hit} A hit in the group
                         ...
                      ]

    Returns:
        {Numpy.array} A strict contour around the group of hits,
                      or None if the quantity of the hits is too low to form a group.
    '''

    blank_img = np.zeros(img.shape, dtype=img.dtype)
    blank_img = cv2.cvtColor(blank_img, cv2.COLOR_RGB2GRAY)
    
    # draw lines between all hits
    for h1 in hits:
        for h2 in hits:
            if h1 != h2:
                cv2.line(blank_img, h1.point, h2.point, (0xff,0xff,0xff), 3)
    
    # find external contour
    contours = cv2.findContours(blank_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    blank_img = cv2.cvtColor(blank_img, cv2.COLOR_GRAY2RGB)
    
    if len(contours[0]) > 0:
        return contours[0][0]
    else:
        return None

def measure_grouping_diameter(contour):
    '''
    Calculate the diameter of a grouping contour.

    Parameters:
        {Numpy.array} contour - The contour to measure

    Returns:
        {Number} The diameter of the grouping contour.
    '''

    # find two furthest points in the polygon
    contPts = [(contour[m][0][0],contour[m][0][1]) for m in range(len(contour))]
    point_A = contPts[0] # random point
    point_B = cntr.contour_distances_from(contPts, point_A)[::-1][0]
    point_A = cntr.contour_distances_from(contPts, point_B)[::-1][0]
    
    # find their distance for each other
    point_A = (point_A[0], point_A[1])
    point_B = (point_B[0], point_B[1])
    return geo2D.euclidean_dist(point_A, point_B)