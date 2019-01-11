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


def for_cropped(img, img_bw):
    edges = cv2.Canny(img_bw, 0, 0)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    lines = np.reshape(lines, (lines.shape[0], 2))

    print(len(lines))

    lines = kmeans_cluster(lines)

    for i in range(0, lines.shape[0]):
        line = lines[i]

        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if i < 10:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        else:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


    cv2.imwrite('hough.jpg', img)


if __name__ == '__main__':
    img = cv2.imread('cropped.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # this thresholds the image to make it black and white
    # we use THRESH_BINARY_INV to invert the image aswell
    (thresh, img_bw) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    for_cropped(img, img_bw)