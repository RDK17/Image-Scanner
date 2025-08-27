import cv2 as cv
import numpy as np
import networkx as nx

def take_picture(outline = False, avg_contours = False): #take a picture
    cap = cv.VideoCapture(0)
    img_shape = cap.read()[1].shape
    img = np.zeros(img_shape)
    tot_contours = 0
    n = 0
    while True:

        ret, frame = cap.read()
        if outline:
            picture = img_of_contours(frame)
            cv.imshow("contours", picture)
        else:
            picture = frame
            picture_box = get_box(picture)
            cv.imshow("to be scanned", picture_box)
        if avg_contours:
            tot_contours += len(get_contours(picture))
            n+=1
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            img = picture
            cv.imshow("img", img)
            cv.waitKey(2000)
            cv.destroyWindow("img")
        if key == ord('v'):
            view_contours(frame,5, show_min_rect = True)
        if key == ord('s'):
            box = get_box(frame, ret_type = "points")
            s = box.sum(axis = 1)
            diff = np.diff(box, axis=1)
            tl = box[np.argmin(s)]
            br = box[np.argmax(s)]
            tr = box[np.argmin(diff)]
            bl = box[np.argmax(diff)]
            pts1 = np.array([tl, tr, br, bl], dtype = np.float32)
            height = np.linalg.norm(tl - bl)
            width = np.linalg.norm(tl - tr)
            pts2 = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = np.float32)
            M = cv.getPerspectiveTransform(pts1,pts2)
            dst = cv.warpPerspective(frame,M,(int(width),int(height)))
            cv.imshow("scanned", dst)
            cv.waitKey(2000)
            cv.destroyWindow("scanned")



            
        
    if avg_contours:
        print(f"average number of contours detected per frame: {int(tot_contours/n)}")
    cap.release()
    cv.destroyAllWindows()

def get_contours(img):
    img_c = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mid = np.median(gray)
    #gray = cv.GaussianBlur(gray,(5,5),0)
    edged = cv.Canny(gray, .66*mid, 1.33*mid)
    contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours

def img_of_contours(img, f = get_contours, all_cnt = True, show_min_rect = False, show_poly = False, view = 0):
    if all_cnt:    
        contours = f(img)
    else:
        contours = sorted(f(img), key = lambda c: c.size, reverse = True)[view]
    width, height, _ = img.shape
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

##def get_close_contours(contours, epsilon):
#    hausdorff = cv.createHausdorffDistanceExtractor()
#    G = nx.Graph()
#    n = len(contours)
#    G.add_nodes_from(range(n))
#    for i in range(n):
#        for j in range(i+1,n):
#            d = hausdorff.computeDistance(contours[i], contours[j])
#            if d < epsilon:
#                G.add_edge(i,j)
#    
#    close_contours = list(nx.connected_components(G))
#    print(close_contours)
#    new_contours = []
#    for component in close_contours:
#        new_cnt = np.vstack(tuple(cnt for cnt in component))
#        if new_cnt.shape[0] >= 2:
#            new_contours.append(new_cnt.reshape(-1,1,2))
#    return new_contours
#
#g = lambda x: get_close_contours(get_contours(x),5.0)

take_picture(False, False)

