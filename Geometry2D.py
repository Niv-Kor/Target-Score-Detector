import numpy as np
import cv2

def euclidean_dist(p1, p2):
    '''
    Parameters:
        {Tuple} p1 - (
                        {Number} x coordinate of the first point,
                        {Number} y coordinate of the first point
                     )
        {Tuple} p2 - (
                        {Number} x coordinate of the second point,
                        {Number} y coordinate of the second point
                     )

    Returns:
        {Number} The euclidean distance between the two points.
    '''

    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** .5

def zero_pad_as(img, paddingShape):
    '''
    Apply an image with zero padding, up to a given size.
    The image is placed in the center of the new created one.

    Parameters:
        {Numpy.array} img - The image to which the zero padding should be applied
        {Tuple} paddingShape - (
                                  {Number} The desired new height of the image,
                                  {Number} The desired new width of the image
                               )

    Returns:
        {Tuple} (
                   {Numpy.array} An array consisting 5 of the small image's anchor points
                                 (in the center of the larger padded image).
                                 E.g: A ----------- B
                                      |             |
                                      |      E      |
                                      |             |
                                      D ----------- C

                                [
                                   (
                                      {Number} x coordinates of the point,
                                      {Number} y coordinates of the point
                                   ),
                                   ...
                                ],
                   {Numpy.array} The newly created padded image (with the orignal image in the center)
                )
    '''

    img_h, img_w, _ = img.shape
    p_h, p_w, _ = paddingShape
    vertical = int((p_h - img_h) / 2)
    horizontal = int((p_w - img_w) / 2)
    a = (horizontal,vertical)
    b = (horizontal + img_w,vertical)
    c = (horizontal + img_w,vertical + img_h)
    d = (horizontal,vertical + img_h)
    e = (int(horizontal + img_w / 2),int(vertical + img_h / 2))
    pad_img = cv2.copyMakeBorder(img, vertical, vertical, horizontal, horizontal, cv2.BORDER_CONSTANT)
    anchor_points = [a, b, c, d, e]

    return anchor_points, pad_img

def calc_model_scale(edges, modelShape):
    '''
    Calculate the scale of the warped homography transformation relative
    to the actual model's shape.

    Parameters:
        {tuple} edges - The AB, BC, CD and DA edges of the transformation
        {tuple} modelShape - (
                                 {Number} The height of the target model image that this object processes,
                                 {Number} The width of the target model image that this object processes
                             )

    Returns:
        {tuple} (
                    {Number} The average size of the horizontal edges divided by
                            the average size of the vertical edges (width / height ratio),
                    {Number} The average size of the vertical edges divided by
                            the average size of the horizontal edges (height / width ratio),
                    {Number} The estimated size of the homography transformation
                            divided by the estimated size of the target model
                            (transformed size / actual size ratio)
                )
    '''

    horizontal_edge = (edges[0] + edges[2]) / 2
    vertical_edge = (edges[1] + edges[3]) / 2
    hor_percent = horizontal_edge / vertical_edge
    ver_percent = vertical_edge / horizontal_edge
    hor_scale = horizontal_edge / modelShape[1]
    ver_scale = vertical_edge / modelShape[0]
    scale_percent = (hor_scale + ver_scale) / 2

    return hor_percent, ver_percent, scale_percent

def calc_vertices_and_edges(transform):
    '''
    Take a prespective transformation and extract the position of its vertices
    and the lengths of its edges.

    Parameters:
        {Numpy.array} transform - The prespective transform product of an image

    Returns:
        {Tuple} (
                   {Tuple} A, B, C, D, E vertices (respectively) of the transformation.
                           E.g: A ----------- B
                                |             |
                                |      E      |
                                |             |
                                D ----------- C
                (
                   {Tuple} (
                              {Number} x coordinates of point A,
                              {Number} y coordinates of point A
                           ),
                   ...,
                ),
                   {Tuple} (
                              {Number} The length of AB edge,
                              {Number} The length of BC edge,
                              {Number} The length of CD edge,
                              {Number} The length of DA edge
                           )
                )
    '''

    vertices = [transform[m][0] for m in range(len(transform))]
    A = vertices[0]
    B = vertices[1]
    C = vertices[2]
    D = vertices[3]
    ab = euclidean_dist(A, B)
    bc = euclidean_dist(B, C)
    cd = euclidean_dist(C, D)
    da = euclidean_dist(D, A)

    return vertices, (ab, bc, cd, da)

def calc_distances_from(matSize, point):
    '''
    Create a matrix of distances, where each value is the distance from a given point.

    Parameters:
        {Tuple} matSize - (
                             {Number} The height of the matrix [px],
                             {Number} The width of the matrix [px]
                          )
        {Tuple} point - (
                           {Number} x coordinate of the parameter point,
                           {Number} y coordinate of the parameter point,
                        )
    
    Returns:
        {Numpy.array} A matrix of distances.
    '''

    dx = np.arange(matSize[1])
    dy = np.arange(matSize[0])
    x, y = point[0], point[1]
    mat_X, mat_Y = np.meshgrid(dx, dy)
    distances = ((mat_X, mat_Y), euclidean_dist((mat_X,mat_Y), (x,y)))
    return distances