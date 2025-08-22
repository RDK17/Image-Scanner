import cv2 as cv
import numpy as np

def take_picture(avg_contours = False):
    cap = cv.VideoCapture(0)
    img_shape = cap.read()[1].shape
    img = np.zeros(img_shape)
    tot_contours = 0
    n = 0
    while True:

        ret, frame = cap.read()
        contour_img = img_of_contours(frame)
        if avg_contours:
            tot_contours += len(get_contours(contour_img))
            n+=1
        cv.imshow("contours", contour_img)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            img = contour_img
            cv.imshow("img", img)
            cv.waitKey(2000)
            cv.destroyWindow("img")
        if key == ord('v'):
            view_contours(frame,5)
    if avg_contours:
        print(f"average number of contours detected per frame: {int(tot_contours/n)}")
    cap.release()
    cv.destroyAllWindows()

def get_contours(img):
    img_c = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),0)
    edged = cv.Canny(gray, 75, 200)
    contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours

def img_of_contours(img, f = get_contours, all_cnt = True, view = 0):
    if all_cnt:    
        contours = f(img)
    else:
        contours = sorted(f(img), key = lambda c: cv.arcLength(c, True), reverse = True)[view]
    width, height, _ = img.shape
    black_img = np.zeros((width, height, 3), dtype = np.uint8)
    cv.drawContours(black_img, contours,-1,(0,255,0),2)
    return black_img

def view_contours(img, n):
        imgs = [img_of_contours(img, all_cnt = False, view = i) for i in range(n)]
        for i in range(n):
            cv.imshow("cnt", imgs[i])
            cv.waitKey(1000)
            cv.destroyWindow("cnt")

take_picture(True)
