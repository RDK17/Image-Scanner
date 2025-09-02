import numpy as np
import cv2 as cv
import networkx as nx 

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

def get_contours(img):
    img_c = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mid = np.median(gray)
    gray = cv.GaussianBlur(gray,(5,5),0)
    edged = cv.Canny(gray, .66*mid, 1.33*mid)
    contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours

def get_density_graph(contours, d, epsilon):
    #brute force find edges in contours where an edge between x and y must have d(x,y) < epsilon
    #return visited cnts(stored as a set) as an array
    G = nx.Graph()
    n = len(contours)
    G.add_nodes_from(range(n)) 
    for i in range(n):
        for j in range(n):
            #print(i,j)
            if i!=j and d(contours[i], contours[j]) < epsilon:
                G.add_edge(i,j)
    return G


def partition(img, d, epsilon):
    contours = get_contours(img)
    G = get_density_graph(contours,d,epsilon)
    prtn = list(nx.connected_components(G))
    return prtn

def adj_box(pts,c):
    dtl = np.array([-c,-c])
    dtr = np.array([c,-c])
    dbl = np.array([-c,c])
    dbr = np.array([c,c])
    new_pts = [pts[0] + dtl, pts[1] + dtr, pts[2] + dbr, pts[3] + dbl]
    return np.array(new_pts, dtype = np.float32)

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



img = cv.imread("GOBRUINSW.jpeg")

##contours = get_contours(img)
##
##n1, n2 = np.random.randint(0,len(contours),size = 2)
##cnt1 = contours[n1]
##cnt2 = contours[n2]
##
##min_pts = set_d(cnt1,cnt2)[1]
##min_dst = set_d(cnt1, cnt2)[0]
##print(f"distance between contours: {min_dst}")
##
##black_img = np.zeros(img.shape)
##cv.drawContours(black_img, [cnt1,cnt2] , -1, (0,255,0), 4)
##cv.line(black_img, min_pts[0][0], min_pts[1][0], (0,0,255), 2)
##cv.imshow("contours", black_img)
##cv.waitKey(5000)
##cv.destroyAllWindows()

#print(len(get_contours(img)))
#partition(img, end_d, 8.0)
