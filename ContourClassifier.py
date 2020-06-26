import Geometry2D as geo2D
import numpy as np
import cv2

def contour_distances_from(contourPts, point):
    pts = [[p[0], p[1], 0] for p in contourPts]
    
    for i in range(len(pts)):
        p = pts[i]
        xy = (p[0],p[1])
        p[2] = geo2D.euclidean_dist(xy, point)
    
    return sorted(pts, key=lambda x: x[2])

def extend_contour_line(img, contour, bullseye, length):
    def normalize(vector):
        square_sum = 0
        for x in vector:
            square_sum += x ** 2

        magnitude = square_sum ** .5
        return np.array(vector) / magnitude

    # find a rectangle that strictly bounds the contour
    bounding_rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(bounding_rect)
    box = np.int0(box)
    A = box[0]
    B = box[1]
    C = box[2]
    D = box[3]

    # find the two shorter edges
    AB = geo2D.euclidean_dist(A, B)
    BC = geo2D.euclidean_dist(B, C)

    if AB < BC:
        edge_1_pts = (A,B)
        edge_2_pts = (C,D)
    else:
        edge_1_pts = (B,C)
        edge_2_pts = (A,D)

    # calculate the middle points of the two edges
    alpha = (int((edge_1_pts[0][0] + edge_1_pts[1][0]) / 2),int((edge_1_pts[0][1] + edge_1_pts[1][1]) / 2))
    beta = (int((edge_2_pts[0][0] + edge_2_pts[1][0]) / 2),int((edge_2_pts[0][1] + edge_2_pts[1][1]) / 2))

    # decide which edge is closer to the target's bulls'eye point
    alpha_dist = geo2D.euclidean_dist(alpha, bullseye)
    beta_dist = geo2D.euclidean_dist(beta, bullseye)
    front_point = alpha if alpha_dist < beta_dist else beta
    rear_point = beta if alpha_dist < beta_dist else alpha

    # calculate the estimated point of the projectile's back
    direction = normalize(np.array(rear_point) - np.array(front_point))
    end_point = np.array(front_point) + direction * length
    end_point = end_point.tolist()
    end_point = tuple([int(x) for x in end_point])

    # extend the line
    cv2.line(img, front_point, end_point, (0xff,0x0,0xff), 4)

def is_contour_rect(cont, A, B, samples):
    x_distance = B[0] - A[0]
    y_distance = B[1] - A[1]
    x_step = x_distance / samples
    y_step = y_distance / samples

    # contour is a very small square
    if (x_step == 0 or y_step == 0):
        return False

    x_vals = np.arange(A[0], B[0], x_step)
    y_vals = np.arange(A[1], B[1], y_step)
    points = [(x,y) for x, y in zip(x_vals, y_vals)]

    for p in points:
        # check if point is outside the contour
        if cv2.pointPolygonTest(cont, p, False) < 0:
            return False

    return True

def filter_convex_contours(contours):
    filtered = []

    for cont in contours:
        contPts = [(cont[m][0][0],cont[m][0][1]) for m in range(len(cont))]
        point_A = contPts[0] # some random point on the contour

        # find the two furthest points on the contour
        point_B = contour_distances_from(contPts, point_A)[::-1][0]
        point_A = contour_distances_from(contPts, point_B)[::-1][0]

        # calculate the point between the two
        point_C = ((point_A[0] + point_B[0]) / 2, (point_A[1] + point_B[1]) / 2)

        # if this point is outside the contour, it's convex,
        # if it's inside it, the contour is relatively straight
        if is_contour_rect(cont, point_A, point_B, 5):
            filtered.append(cont)

    return filtered