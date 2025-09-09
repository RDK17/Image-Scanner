import cv2 as cv
import numpy as np
import networkx as nx
import tools as t
from collections import deque
import graphs

def take_picture(outline = False, avg_contours = False): #take a picture
    cap = cv.VideoCapture(0)
    img_shape = cap.read()[1].shape
    img = np.zeros(img_shape)
    tot_contours = 0
    n = 0
    frame_num = 0
    quads = deque()
    hd = cv.createHausdorffDistanceExtractor()
    d_H = lambda x, y: t.safe_hd(hd,x,y)

    while True:

        ret, frame = cap.read()
        quad = get_box(frame, ret_type = "points").reshape(-1,1,2)
        quads.append(quad)
        if len(quads) > 5:#add memory param w default 5?
            quads.popleft()
       
        if outline:
            picture = img_of_contours(frame)
            cv.imshow("contours", picture)
        else:
            G = t.cnts2ENG(quads, 10, t.my_hd)
            box = t.get_voted_cnt(G)
            box = t.tup2cnt(box)
            frame_c = frame.copy()
            cv.drawContours(frame_c, [box], 0 ,(255,0,0), 2)
            frame_num +=1
            frame_box = frame_c
            if frame_num%15 == 0:
                None
                #print(f"quad info: quad = {quad}, size = {quad.size}, shape = {quad.shape}")
            cv.imshow("scanning...", frame_box)
            

        if avg_contours:
            tot_contours += len(get_contours(frame))
            n+=1
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('c'):
            img = frame
            cv.imshow("img", img)
            cv.waitKey(2000)
            cv.destroyWindow("img")

        if key == ord('v'):
            view_contours(frame,5, show_min_rect = True)

        if key == ord('s'):
            _, current_frame = cap.read()
            scan(current_frame)

    if avg_contours:
        print(f"average number of contours detected per frame: {int(tot_contours/n)}")
    cap.release()
    cv.destroyAllWindows()


def get_contours(img):
    img_c = img.copy()
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    mid = np.median(gray)
    gray = cv.GaussianBlur(gray,(5,5),0)
    edged = cv.Canny(gray, .66*mid, 1.33*mid)
    contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours


def img_of_contours(img, f = get_contours, all_cnt = True, show_min_rect = False, show_poly = False, view = 0):
    if all_cnt:    
        contours = f(img)
    else:
        contours = sorted(f(img), key = lambda c: c.size, reverse = True)[view]
    if len(img.shape)==3:
        width, height, _ = img.shape
    else:
        width, height = img.shape
    black_img = np.zeros((width, height, 3), dtype = np.uint8)
    cv.drawContours(black_img, contours,-1,(0,255,0),2)
    if show_min_rect:
        rect = cv.minAreaRect(contours)
        box = cv.boxPoints(rect)
        box = np.intp(box)
        cv.drawContours(black_img, [box], 0 ,(255,0,0), 2)
    if show_poly:
        epsilon = 0.05*cv.arcLength(contours, True)
        poly = cv.approxPolyDP(contours, epsilon, True)
        print(f"number of vertices: {len(poly)}")
        cv.drawContours(black_img, [poly], -1, (0,255,0), 3)
    return black_img


def view_contours(img, n, show_min_rect = False, show_poly = False):
        imgs = [img_of_contours(img, all_cnt = False, show_min_rect = show_min_rect, show_poly = show_poly, view = i) for i in range(n)]
        for i in range(n):
            cv.imshow("cnt", imgs[i])
            cv.waitKey(1000)
            cv.destroyWindow("cnt")


def get_box(img, ret_type = "image"):
    contours = get_contours(img)
    if contours:
        cnt = max(contours, key = lambda c: cv.arcLength(c,True))
    else:
        return img
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.intp(box)
    if ret_type == "image":
        cv.drawContours(img, [box], 0 ,(255,0,0), 2)
        return img
    elif ret_type == "points":
        return box


def get_quad(img, ret_type ="image"):#BROKEN
    box = get_box(img, "points")

    if box is None or len(box)< 4:
        print("[WARN] get_box() retuned less than 4 points")
        return None

    try:
        s = box.sum(axis = 1)
        diff = np.diff(box, axis=1)
        tl = box[np.argmin(s)]
        br = box[np.argmax(s)]
        tr = box[np.argmin(diff)]
        bl = box[np.argmax(diff)]
        pts1 = np.array([tl, tr, br, bl], dtype = np.float32)
        M, width, height = t.get_fit_isometry(pts1, scale = 15, get_dims = True)
        new_img = cv.warpPerspective(img,M,(int(width),int(height)))
        dst = cv.cvtColor(new_img,cv.COLOR_BGR2GRAY)
        dst = cv.GaussianBlur(dst,(5,5),0)
        ret, th = cv.threshold(dst,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        contours = get_contours(th)
        contours = sorted(contours, key = lambda c: c.size, reverse = True)
        if contours != []:
            epsilon = 0.05*cv.arcLength(contours[0], True)
            poly = cv.approxPolyDP(contours[0], epsilon, True)
            cv.drawContours(new_img,[poly], 0 ,(255,0,0),2)
        else:
            return None
        if ret_type == "image":
            return new_img
        if ret_type == "points":
            M_inv = np.linalg.inv(M)
            poly = poly.astype(np.float32)
            return cv.perspectiveTransform(poly,M_inv)
    except Exception as e:
        print("[ERROR] in get_quad:", e)
        return None


def scan(img, wait=2, save = False):
    pts1 = get_box(img, ret_type = "points")
    if t.get_fit_isometry(pts1) is not None:
        M, width, height = t.get_fit_isometry(pts1, scale = -10, get_dims = True)
        print(M, width, height)
        new_img = cv.warpPerspective(img,M,(int(width),int(height)))
    else:
        print("couldn't fit isometry")
        new_img = img
    gray_img = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
    scanned_img = cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    cv.imshow("scanned", scanned_img)
    cv.waitKey(wait*1000)
    cv.destroyWindow("scanned")
    if save:
        cv.imwrite("test_unscanned.jpg",scanned_img)


if __name__ == "__main__":
    take_picture(False, False)

