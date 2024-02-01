from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

root = Tk()
path1 = filedialog.askopenfilename(initialdir="C:/", title="choose your image", filetypes=(("jpeg files","jpg"),("all files","*.*")))
img1 = cv2.imread(path1,0) # reference!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
path2 = filedialog.askopenfilename(initialdir="C:/", title="choose your image", filetypes=(("jpeg files","jpg"),("all files","*.*")))
img2 = cv2.imread(path2,0) # time
root.withdraw()

def change(i1, i2):
    temp = i1
    i1 = i2
    i2 = temp
    return i1, i2

def detect(right, left): 
    global good, matchesMask, kp1, kp2, w, h
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
        if m.distance < 0.4*n.distance:
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
        img2 = cv2.polylines(left,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    
    else: # good list의 요소 개수가 적어 homography 불가능, ratio가 너무 빠듯하거나, 이미지의 문제 
        print("Not enugh matches are found - %d%d" %(len(good),MIN_MATCH_COUNT))
        matchesMask=(None)
    
    draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        
    img3 = cv2.resize(img3, [1702, 478])
    cv2.imshow('gray', img3)
    
    return M
######################################################################################################

img=[img1, img2] 
sift = cv2.SIFT_create() 
I_1=img[0]
for i in range(len(img)-1): #i=0,1,2

    I0=img[i+1]
    #print(" ")
    #00침이 n분에 갈 때 얼마나 돌아야 하는 지에 대한 정보를 행렬로 받는다.
    matrix = detect(I_1, I0)
    cos = matrix[0,0] 
    sin = matrix[1,0]
    theta = math.degrees(math.atan2(sin,cos))
    
    if theta<0: #theta가 음수라면 360을 더해준다
        theta=360+theta
    else:
        theta = theta
   
    t_minute = theta/360*60
    print("시간 =",np.int8(t_minute),"분")
    
cv2.waitKey(0)
cv2.destroyAllWindows()