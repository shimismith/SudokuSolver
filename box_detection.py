import cv2
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def kmeans_cluster(lines):
    scaler = StandardScaler()
    scaler.fit(lines)
    lines = scaler.transform(lines)

    kmeans = KMeans(n_clusters=20).fit(lines)  # cluster based on rho and theta to combine lines
    lines = scaler.inverse_transform(kmeans.cluster_centers_)

    # cluster the lines by angle so we don't attempt to find intersections between parallel lines
    # lines with angles near 0 and 180 are both verical
    # we multiply by 2 to get 0 to 360 and plot it on the unit circle and then do kmeans with these coordinates
    angles = np.concatenate((np.cos(2*lines[:, 1]).reshape(lines.shape[0],1), np.sin(2*lines[:, 1]).reshape(lines.shape[0],1)), axis=1)
    kmeans_angles = KMeans(n_clusters=2).fit(angles)

    # sort the lines by the angle clusters
    sorted_lines = np.concatenate((lines[np.where(kmeans_angles.labels_ == 0)], lines[np.where(kmeans_angles.labels_ == 1)]))

    return sorted_lines


def intersections(lines):
    '''
    This function solves a system of linear equations for each intersection.
    x*cos(θ_1) + y*sin(θ_1) = r1
    x*cos(θ_2) + y*sin(θ_2) = r2
    '''

    augmented = np.array([np.cos(lines[:, 1]), np.sin(lines[:, 1]), lines[:, 0]]).T  # the augmented matrix representing the system

    i = np.arange(20)
    indices = np.array(np.meshgrid(i[:10],i[10:])).reshape(2,100).T
    augmented = augmented[indices]

    return np.linalg.solve(augmented[:, :, 0:2], augmented[:, :, 2])




def for_cropped(img, img_bw):
    edges = cv2.Canny(img_bw, 0, 0)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    lines = np.reshape(lines, (lines.shape[0], 2))

    print(len(lines))

    lines = kmeans_cluster(lines)

    points = intersections(lines)
    print(points.shape)

    for i in range(0, points.shape[0]):
        point = points[i]
        cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), thickness=-1)

    cv2.imwrite('hough.jpg', img)


if __name__ == '__main__':
    img = cv2.imread('cropped.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # this thresholds the image to make it black and white
    # we use THRESH_BINARY_INV to invert the image aswell
    (thresh, img_bw) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    for_cropped(img, img_bw)