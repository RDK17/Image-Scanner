import numpy as np
import cv2 as cv
import time
import graphs

def end_d(A,B, dst_only = True):
    d = lambda x,y: np.linalg.norm(x-y)
    ends = [[A[-1,0],B[-1,0]],[A[-1,0],B[0,0]],[A[0,0],B[-1,0]],[A[0,0],B[0,0]]]
    get_d = lambda p: d(p[0],p[1])
    if dst_only:
        ds = [get_d(p) for p in ends]
        dst = min(ds)
        return dst
    ends = sorted(ends, key = get_d)
    pts = ends[0]
    dst = d(pts[0],pts[1])
    return dst, pts


def is_hashable(dt):
    try:
        hash(dt)
    except TypeError:
        return False
    return True


hd = cv.createHausdorffDistanceExtractor()


def safeHD(x, y, hd=hd): #BROKEN
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return float("inf")
    if type(x) == tuple:
        x = tup2cnt(x)
    if type(y) == tuple:
        y = tup2cnt(y)
    return hd.computeDistance(x, y)

def my_hd(cnt1, cnt2):
    if type(cnt1) == tuple:
        cnt1 = tup2cnt(cnt1)
    if type(cnt2) == tuple:
        cnt2 = tup2cnt(cnt2)
    A = cnt1.reshape(-1,2).astype(np.float32)
    B = cnt2.reshape(-1,2).astype(np.float32)
    dists = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    min_A_2_B = dists.min(axis=1)
    min_B_2_A = dists.min(axis=0)
    return max(min_A_2_B.max(), min_B_2_A.max())  


def get_contours(img):
    img_c = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mid = np.median(gray)
    gray = cv.GaussianBlur(gray,(5,5),0)
    edged = cv.Canny(gray, .66*mid, 1.33*mid)
    contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours


def adj_box(pts,c):
    dtl = np.array([-c,-c])
    dtr = np.array([c,-c])
    dbl = np.array([-c,c])
    dbr = np.array([c,c])
    new_pts = [pts[0] + dtl, pts[1] + dtr, pts[2] + dbr, pts[3] + dbl]
    return np.array(new_pts, dtype = np.float32)


def cnts2ENG(nodes, epsilon, d): #pending
    if not is_hashable(nodes[0]):
        nodes = [cnt2tup(node) for node in nodes]
        G = graphs.ENG(epsilon, d)
    for node in nodes:
        G.add_node(node)
    return G


def cnt2tup(cnt):
    return tuple(tuple(pt.ravel()) for pt in cnt)


def tup2cnt(tup):
    cnt = np.array(tup, dtype = np.int32)
    cnt = cnt.reshape(-1,1,2)
    return cnt


def get_voted_cnt(G): #pending
    cmpts = G.connected_components()
    voted_party = max(cmpts, key = lambda cmpt: len(cmpt))
    for node in G.nodes:
        if node in voted_party:
            return node


def get_fit_isometry(pts1, scale = 0, get_dims=False):
    if len(pts1) != 4:
        print(f"get_fit_isometry: pts1 has only {len(pts1)} points")
        return None
    pts1 = pts1.astype(np.float32)
    tl, tr, br, bl = pts1
    pts1 = adj_box(pts1, scale)
    height = np.linalg.norm(tl - bl)
    width = np.linalg.norm(tl - tr)
    pts2 = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = np.float32)
    M = cv.getPerspectiveTransform(pts1,pts2)
    if get_dims:
        return M, width, height
    return M


if __name__ == "__main__":
    None
    #img = cv.imread("GOBRUINSW.jpeg")
    #contours = get_contours(img)
    #cnts = sorted(contours, key = lambda cnt: len(cnt), reverse = True)[:5]
    #G = cnts2ENG(cnts,10,safeHD)
    #winner = get_voted_cnt(G)

    
