# level_A1a
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
global good, matchesMask, kp1, kp2


root = Tk()
path1 = filedialog.askopenfilename(initialdir="C:/", title="choose your image", filetypes=(("jpeg files","jpg"),("all files","*.*")))
img1 = cv2.imread(path1,0)
path2 = filedialog.askopenfilename(initialdir="C:/", title="choose your image", filetypes=(("jpeg files","jpg"),("all files","*.*")))
img2 = cv2.imread(path2,0)
root.withdraw()


#img1 = cv2.imread('1-2.jpg',0) # query Img # 2-1, 2-2
#img2 = cv2.imread('2-1.jpg',0) # train Img 왼쪽이나 위쪽 이미지, 변형없이 그대로 들어가는 이미지임

def detect(right, left): 
    global good, matchesMask, kp1, kp2
    kp1, des1 = sift.detectAndCompute(right,None)
    kp2, des2 = sift.detectAndCompute(left,None) # left = img2

    # FLANN=Fast Library for Approximate Nearest Neighbors 큰 이미지에서 특성을 매칭할 때 성능을 위해 최적화된 라이브러리 모음
    FLANN_INDEX_KDTREE = 1
    # FlannBasedMatcher는 사전 자료형 인자 indexParams, searchParams가 필요
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) #key=value
    search_params = dict(checks= 50) #특성 매칭을 위한 반복 횟수->높을 수록 정확도 상승, 속도 하락

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2) # k값만큼의 가까운 후보들을 뽑아준다

    # store all the good matches as per Lowe's ratio test.
    # knn으로 뽑은 1위(m)가 0.9*2위(n)보다 가까우면 [good]에 들어감
    good=[]
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
        
    MIN_MATCH_COUNT = 4 # 최소한 4개는 뽑겠다는 의미
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2) #source
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2) #destination
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = img1.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2) # 차원,행,열
        dst = cv2.perspectiveTransform(pts,M)
        #img2 = cv2.polylines(left,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    
    else: # good list의 요소 개수가 적어 homography 불가능, ratio가 너무 빠듯하거나, 이미지의 문제 
        print("Not enugh matches are found - %d%d" %(len(good),MIN_MATCH_COUNT))
        matchesMask=(None)
    
    return M

text=(50,100)
color=(0,255,0)
size = 1
thick = 2

# SIFT 추출기 생성
sift = cv2.SIFT_create() 
matrix = detect(img1, img2)
round_M = matrix.astype('int32')
#print(round_M)
if round_M[0,2]>0 and round_M[1,2] in range(-10,10):  
    img1 = cv2.putText(img1, "img1 = right", text, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA) 
    img2 = cv2.putText(img2, "img2 = left", text, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)   
    print("img1 = 우, img2=좌")
elif round_M[0,2] in range(-10,10) and round_M[1,2]>0: 
    img1 = cv2.putText(img1, "img1 = down", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA) 
    img2 = cv2.putText(img2, "img2 = up", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA)   
    print("img1 = 하, img2=상")
elif round_M[0,2]<0 and round_M[1,2] in range(-10,10):
    img1 = cv2.putText(img1, "img1 = left", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA) 
    img2 = cv2.putText(img2, "img2 = right", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA)   
    print("img1 = 좌, img2=우")
elif round_M[0,2] in range(-10,10) and round_M[1,2]<0:
    img1 = cv2.putText(img1, "img1 = up", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA) 
    img2 = cv2.putText(img2, "img2 = down", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA)   
    print("img1 = 상, img2=하")
    
elif round_M[0,2]>0 and round_M[1,2]>0: 
    img1 = cv2.putText(img1, "img1 = right/down", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA) 
    img2 = cv2.putText(img2, "img2 = left/up", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA)   
    print("img1 = 우하, img2=좌상") 
elif round_M[0,2]<0 and round_M[1,2]<0:  
    img1 = cv2.putText(img1, "img1 = left/up", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA) 
    img2 = cv2.putText(img2, "img2 = right/down", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA)   
    print("img1 = 좌상, img2=우하") 
elif round_M[0,2]>0 and round_M[1,2]<0: 
    img1 = cv2.putText(img1, "img1 = right/up", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA) 
    img2 = cv2.putText(img2, "img2 = left/down", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA)   
    print("img1 = 우상, img2=좌하")    
elif round_M[0,2]<0 and round_M[1,2]>0:
    img1 = cv2.putText(img1, "img1 = left/down", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA) 
    img2 = cv2.putText(img2, "img2 = right/up", text, cv2.FONT_HERSHEY_SIMPLEX, size, (0,255,0), thick, cv2.LINE_AA)   
    print("img1 = 좌하, img2=우상")   
else:
    print("err")
cv2.imshow("img1",img1)
cv2.imshow("img2",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()