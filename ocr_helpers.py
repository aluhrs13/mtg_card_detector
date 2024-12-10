import cv2
import numpy as np
import math
from PIL import Image

def resize(img):
    height, width = img.shape[:2]
    new_height = 800
    new_width = int((new_height / height) * width)
    return cv2.resize(img, (new_width, new_height))

def find_card(img, thresh_c=5, kernel_size=(3, 3), size_thresh=10000):
    """
    Find contour of the largest card in the image.
    :param img: source image
    :param thresh_c: value of the constant C for adaptive thresholding
    :param kernel_size: dimension of the kernel used for dilation and erosion
    :param size_thresh: threshold for size (in pixel) of the contour to be a candidate
    :return: largest card contour
    """
    debug=True
    # Typical pre-processing - grayscale, blurring, thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 1)

    cv2.imshow('Threshold', resize(img_thresh))

    # Dilute the image, then erode them to remove minor noises
    kernel = np.ones(kernel_size, np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    # Find the contour
    cnts, _ = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        print('no contours')
        return None

    # Display the image with bounding boxes
    # Convert the image back to BGR color space
    img_erode_bgr = cv2.cvtColor(img_erode, cv2.COLOR_GRAY2BGR)
    # Draw bounding boxes around all contours
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_erode_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if debug:
        cv2.imshow('Bounding Boxes', resize(img_erode_bgr))

    # Filter contours to find quadrilaterals above size threshold
    cnts_rect = []
    for cnt in cnts:
        size = cv2.contourArea(cnt)
        if size < size_thresh:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            cnts_rect.append(approx)

    if not cnts_rect:
        return []
        return [np.array([[[0, 0]], [[img.shape[1] - 1, 0]], [[img.shape[1] - 1, img.shape[0] - 1]], [[0, img.shape[0] - 1]]])]

    # Find the largest quadrilateral contour
    largest_card = max(cnts_rect, key=cv2.contourArea)
    return [largest_card]


def find_card_canny(img, thresh1=50, thresh2=150, kernel_size=(10, 10), size_thresh=30000):
    """
    Find contours of all cards in the image using Canny edge detection
    :param img: source image
    :param thresh1: first threshold for the hysteresis procedure in Canny edge detector
    :param thresh2: second threshold for the hysteresis procedure in Canny edge detector
    :param kernel_size: dimension of the kernel used for dilation and erosion
    :param size_thresh: threshold for size (in pixel) of the contour to be a candidate
    :return: list of candidate contours
    """
    debug = True
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display grayscale image
    if debug:
        cv2.imshow('Grayscale Image', resize(img_gray))
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, thresh1, thresh2)
    # Display edges
    if debug:
        cv2.imshow('Canny Edges', resize(edges))
    
    # Dilate and erode to close gaps in edges
    kernel = np.ones(kernel_size, np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)
    # Display dilated and eroded edges
    if debug:
        cv2.imshow('Dilated and Eroded Edges', resize(edges_eroded))
    
    # Find contours from edges
    cnts, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        print('no contours')
        return []
    
    # Optional: Draw bounding boxes around detected contours
    img_contours = img.copy()
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if debug:
        cv2.imshow('Contours', resize(img_contours))
    
    # Filter contours to find rectangles of appropriate size
    cnts_rect = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            cnts_rect.append(approx)
    if cnts_rect:
        cnts_rect = [max(cnts_rect, key=cv2.contourArea)]

        if cv2.contourArea(cnts_rect[0]) < size_thresh:
            cnts_rect = []

    if not cnts_rect:
        return [np.array([[[0, 0]], [[img.shape[1] - 1, 0]], [[img.shape[1] - 1, img.shape[0] - 1]], [[0, img.shape[0] - 1]]])]
        cnts_rect = find_card(img, size_thresh=size_thresh)

    return cnts_rect

# www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    """
    initialzie a list of coordinates that will be ordered such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
    :param pts: array containing 4 points
    :return: ordered list of 4 points
    """
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    """
    Transform a quadrilateral section of an image into a rectangular area
    From: www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    :param image: source image
    :param pts: 4 corners of the quadrilateral
    :return: rectangular image of the specified area
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    mat = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, mat, (maxWidth, maxHeight))

    # If the image is horizontally long, rotate it by 90
    if maxWidth > maxHeight:
        center = (maxHeight / 2, maxHeight / 2)
        mat_rot = cv2.getRotationMatrix2D(center, 270, 1.0)
        warped = cv2.warpAffine(warped, mat_rot, (maxHeight, maxWidth))

    # return the warped image
    return warped


def remove_glare(img):
    """
    Reduce the effect of glaring in the image
    Inspired from:
    http://www.amphident.de/en/blog/preprocessing-for-automatic-pattern-identification-in-wildlife-removing-glare.html
    The idea is to find area that has low saturation but high value, which is what a glare usually look like.
    :param img: source image
    :return: corrected image with glaring smoothened out
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(img_hsv)
    non_sat = (s < 32) * 255  # Find all pixels that are not very saturated

    # Slightly decrease the area of the non-satuared pixels by a erosion operation.
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_sat = cv2.erode(non_sat.astype(np.uint8), disk)

    # Set all brightness values, where the pixels are still saturated to 0.
    v[non_sat == 0] = 0
    # filter out very bright pixels.
    glare = (v > 200) * 255

    # Slightly increase the area for each pixel
    glare = cv2.dilate(glare.astype(np.uint8), disk)
    glare_reduced = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 200
    glare = cv2.cvtColor(glare, cv2.COLOR_GRAY2BGR)
    corrected = np.where(glare, glare_reduced, img)
    return corrected