import numpy as np
import cv2 as cv

def set_d(A,B):
    d = lambda x,y: np.linalg.norm(x-y)
    dst = d(A[0],B[0])
    pts = (0,0)
    for a in A:
        for b in B:
            curr_dst = d(a,b)
            if curr_dst < dst:
                dst = curr_dst
                pts = (a, b)
    return dst, pts

def get_contours(img):
    img_c = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mid = np.median(gray)
    gray = cv.GaussianBlur(gray,(5,5),0)
    edged = cv.Canny(gray, .66*mid, 1.33*mid)
    contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours

img = cv.imread("GOBRUINSW.jpeg")
contours = get_contours(img)

n1, n2 = np.random.randint(0,len(contours),size = 2)
cnt1 = contours[n1]
cnt2 = contours[n2]

min_pts = set_d(cnt1,cnt2)[1]
min_dst = set_d(cnt1, cnt2)[0]
print(f"distance between contours: {min_dst}")

black_img = np.zeros(img.shape)
cv.drawContours(black_img, [cnt1,cnt2] , -1, (0,255,0), 4)
cv.line(black_img, min_pts[0][0], min_pts[1][0], (0,0,255), 2)
cv.imshow("contours", black_img)
cv.waitKey(5000)
cv.destroyAllWindows()
