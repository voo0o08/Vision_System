# level_A2 이미지 순서 없이 random
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

global good, matchesMask, kp1, kp2, h, w

root = Tk()
path1 = filedialog.askopenfilename(initialdir="C:/", title="choose your image", filetypes=(("jpeg files","jpg"),("all files","*.*")))
img1 = cv2.imread(path1)
path2 = filedialog.askopenfilename(initialdir="C:/", title="choose your image", filetypes=(("jpeg files","jpg"),("all files","*.*")))
img2 = cv2.imread(path2)
path3 = filedialog.askopenfilename(initialdir="C:/", title="choose your image", filetypes=(("jpeg files","jpg"),("all files","*.*")))
img3 = cv2.imread(path3)
path4 = filedialog.askopenfilename(initialdir="C:/", title="choose your image", filetypes=(("jpeg files","jpg"),("all files","*.*")))
img4 = cv2.imread(path4)
root.withdraw()

# 경우의 수 set
# 제출한 코드에는 set1만 돌아갑니다.

set1 = [[img1, img2, img3, img4],
       #[img1, img2, img4, img3],   
       #[img1, img3, img2, img4],
       #[img1, img3, img4, img2],
       #[img1, img4, img2, img3],
       #[img1, img4, img3, img2]
       ]

       
set2 = [[img2, img1, img3, img4],
       [img2, img1, img4, img3],
       [img2, img3, img1, img4],
       [img2, img3, img4, img1], 
       [img2, img4, img3, img1],
       [img2, img4, img1, img3]
       ] 
       
set3 = [[img3, img1, img4, img2],
       [img3, img1, img2, img4],
       [img3, img2, img4, img1],
       [img3, img2, img1, img4],
       [img3, img4, img1, img2],
       [img3, img4, img2, img1]
       ]
       
set4 = [[img4, img1, img2, img3],
       [img4, img1, img3, img2],
       [img4, img2, img1, img3],
       [img4, img2, img3, img1],
       [img4, img3, img1, img2],
       [img4, img3, img2, img1]]

# 이미지의 이동값이 음수가 떴을 때, 두 이미지를 바꾸어주는 함수
def change(i1, i2):
    temp = i1
    i1 = i2
    i2 = temp
    return i1, i2

# 이전과제의 코드를 함수화함
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
        if m.distance < 0.5*n.distance:
            good.append(m)
        
    MIN_MATCH_COUNT = 4 # 최소한 4개는 뽑겠다는 의미
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2) #source
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2) #destination
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w,c = img1.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2) # 차원,행,열
        dst = cv2.perspectiveTransform(pts,M)
    
    else: # good list의 요소 개수가 적어 homography 불가능, ratio가 너무 빠듯하거나, 이미지의 문제 
        print("Not enugh matches are found - %d%d" %(len(good),MIN_MATCH_COUNT))
        matchesMask=(None)
    
    return M
######################################################################################################

k = 0

for img in set1:
    I_1=img[0]
    total_row=I_1.shape[0]
    total_col=I_1.shape[1]
    mov_row=0
    mov_col=0

    for i in range(len(img)-1): #i=0,1,2
        plus_y=0
        plus_x=0
        I0=img[i+1]
        #print(" ")
        R_h,R_w,c = I0.shape
        L_h,L_w,c = I_1.shape

        # SIFT 추출기 생성
        sift = cv2.SIFT_create() 
        matrix = detect(I0,I_1)
        round_M = matrix.astype('int32') # float -> int로 바꾸어줌 이미지의 크기에따라 int8도 무관
        #print(round_M)
        right = I0
        left = I_1  
        #if round_M[0,2]>0 and round_M[1,2] in range(-20,20):  
            #print("img1 = 우, img2=좌")
        #elif round_M[0,2] in range(-20,20) and round_M[1,2]>0: 
            #print("img1 = 하, img2=상")
        if round_M[0,2]+10<0 and round_M[1,2] in range(-20,20):
            #print("img1 = 좌, img2=우")
            right,left = change(I0,I_1)
            matrix = detect(right,left)
        elif round_M[0,2] in range(-20,20) and round_M[1,2]+10<0:
            #print("img1 = 상, img2=하")
            right,left = change(I0,I_1)
            matrix = detect(right,left)
        #elif round_M[0,2]>0 and round_M[1,2]>0: 
            #print("img1 = 우하, img2=좌상") 
        elif round_M[0,2]+10<0 and round_M[1,2]+10<0:  
            #print("img1 = 좌상, img2=우하") 
            right,left = change(I0,I_1)
            matrix = detect(right,left)
        elif round_M[0,2]>0 and round_M[1,2]+10<0: 
            #print("img1 = 우상, img2=좌하")   
            zero_mat = np.zeros((abs(round_M[1,2]),L_w,3)) # left 이미지 위로 zero padding 이미지 씹힘 방지
            left = np.uint8(np.concatenate((zero_mat,left),axis=0)) #3차원 배열을 합치기 위함 axis=0은 row방향
            #left=np.uint8(np.r_[zero_mat, left]) # 흑백일 때
            if mov_row<abs(round_M[1,2])-10:
                mov_row=abs(round_M[1,2])
                total_row=total_row+abs(round_M[1,2])
            matrix = detect(right,left)
        elif round_M[0,2]+10<0 and round_M[1,2]>0: 
            #print("img1 = 좌하, img2=우상")   
            zero_mat = np.zeros((L_h,abs(round_M[0,2]),3))
            #print(zero_mat.shape, left.shape)
            left = np.uint8(np.concatenate((zero_mat,left),axis=1)) #3차원 배열을 합치기 위함 axis=1은 col방향
            #left=np.uint8(np.c_[zero_mat,left]) #흑백일 때
            if mov_col<abs(round_M[0,2])-10:
                mov_col=abs(round_M[0,2])
                total_col=total_col+abs(round_M[0,2])
            matrix = detect(right,left)
        #else:
            #print("1,1이미지입니당!!")
        round_M = matrix.astype('int32')
        #print(round_M)

        draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        re = cv2.drawMatches(right, kp1, left, kp2, good, None, **draw_params)
       
        if abs(round_M[1,2])-mov_row-10>0: # 지금까지 제일 많이 움직인 거리(mov_row)보다 현재 움직이려는 거리가 더 크면 stitching 결과의 크기가 달라짐
            total_row=total_row+abs(round_M[1,2])-mov_row
            mov_row=round_M[1,2]

        if abs(round_M[0,2])-mov_col-10>0:
            total_col=total_col+abs(round_M[0,2])-mov_col
            mov_col=round_M[0,2]
        
        dst=cv2.warpPerspective(right, matrix, (total_col, total_row), borderValue=0)
        
        left_pad = np.uint8(np.zeros((dst.shape[0], dst.shape[1], dst.shape[2])))
        left_pad[:left.shape[0], :left.shape[1], :left.shape[2]] = left
        result = np.maximum(dst, left_pad)
        result = result[:total_row,:total_col]
        if i == 2:
            cv2.imshow("stitchig %d" %k, result)
        #cv2.imshow("stitchig %d" %i, result)
        I_1 = result
    #print(k)
    k = k+1
cv2.waitKey(0)
cv2.destroyAllWindows()