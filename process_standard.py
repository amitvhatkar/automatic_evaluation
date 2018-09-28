import imutils
import threading
import os
import glob
import numpy as np
import cv2
import csv
import time
import sys
from color_markers import *

sys.stdout = open('temp_log', 'w')


width = 5
img = np.zeros((500,500),dtype=np.uint8)
img_with_circles = np.zeros((500,500),dtype=np.uint8)
img_thick = np.zeros((500,500),dtype=np.uint8)
first_warped_frame = np.zeros((500,500),dtype=np.uint8)

list_circles = []
list_export = []
features_result=[]
name_akaze = "perfect_trajectory_akaze.png"
name_circle = "perfect_trajectory_circle.png"
name_thick = "perfect_trajectory_thick.png"
csv_file_name = "results_perfect.csv"
result_dic = []


circle_step = 25;
list_white = []
img_plot = []
adj_x = 0 #this variable is used to adjust the offset while evaluating the trajectory
adj_y = 0 #this variable is used to adjust the offset while evaluating the trajectory
flag_first = True


blue = (255,0,0)


def evaluation():
    # load the image and convert it to grayscale
    im1 = cv2.imread("perfect_trajectory_akaze.png")
    im2 = cv2.imread("perfect_trajectory_akaze.png")

    # load the image and convert it to grayscale
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    akaze = cv2.AKAZE_create()
    brisk = cv2.BRISK_create()
    orb = cv2.ORB_create()

    (akazekps1, akazedescs1) = akaze.detectAndCompute(gray1, None)
    (akazekps2, akazedescs2) = akaze.detectAndCompute(gray2, None)

    (siftkps1, siftdescs1) = sift.detectAndCompute(gray1, None)
    (siftkps2, siftdescs2) = sift.detectAndCompute(gray2, None)

    (surfkps1, surfdescs1) = surf.detectAndCompute(gray1, None)
    (surfkps2, surfdescs2) = surf.detectAndCompute(gray2, None)

    (briskkps1, briskdescs1) = brisk.detectAndCompute(gray1, None)
    (briskkps2, briskdescs2) = brisk.detectAndCompute(gray2, None)

    (orbkps1, orbdescs1) = orb.detectAndCompute(gray1, None)
    (orbkps2, orbdescs2) = orb.detectAndCompute(gray2, None)

    #print("No of KeyPoints:")
    #print("akazekeypoints1: {}, akazedescriptors1: {}".format(len(akazekps1), akazedescs1.shape))
    #print("akazekeypoints2: {}, akazedescriptors2: {}".format(len(akazekps2), akazedescs2.shape))
    #print("siftkeypoints1: {}, siftdescriptors1: {}".format(len(siftkps1), siftdescs1.shape))
    #print("siftkeypoints2: {}, siftdescriptors2: {}".format(len(siftkps2), siftdescs2.shape))
    #print("surfkeypoints1: {}, surfdescriptors1: {}".format(len(surfkps1), surfdescs1.shape))
    #print("surfkeypoints2: {}, surfdescriptors2: {}".format(len(surfkps2), surfdescs2.shape))
    #print("briskkeypoints1: {}, briskdescriptors1: {}".format(len(briskkps1), briskdescs1.shape))
    #print("briskkeypoints2: {}, briskdescriptors2: {}".format(len(briskkps2), briskdescs2.shape))
    #print("orbkeypoints1: {}, orbdescriptors1: {}".format(len(orbkps1), orbdescs1.shape))
    #print("orbkeypoints2: {}, orbdescriptors2: {}".format(len(orbkps2), orbdescs2.shape))

    # Match the features
    bfakaze = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    akazematches = bfakaze.knnMatch(akazedescs1, akazedescs2, k=2)
    siftmatches = bf.knnMatch(siftdescs1, siftdescs2, k=2)
    surfmatches = bf.knnMatch(surfdescs1, surfdescs2, k=2)
    briskmatches = bf.knnMatch(briskdescs1, briskdescs2, k=2)
    orbmatches = bf.knnMatch(orbdescs1, orbdescs2, k=2)

    # Apply ratio test on AKAZE matches
    goodakaze = []
    for m, n in akazematches:
        if m.distance < 0.9 * n.distance:
            goodakaze.append([m])

    im3akaze = cv2.drawMatchesKnn(im1, akazekps1, im2, akazekps2, goodakaze[:], None, flags=2)
    #cv2.imshow("AKAZE matching", im3akaze)
    goodakaze = np.asarray(goodakaze)

    features_result.append(goodakaze.shape[0])

    # Apply ratio test on SIFT matches
    goodsift = []
    for m, n in siftmatches:
        if m.distance < 0.9 * n.distance:
            goodsift.append([m])

    im3sift = cv2.drawMatchesKnn(im1, siftkps1, im2, siftkps2, goodsift[:], None, flags=2)
    #cv2.imshow("SIFT matching", im3sift)
    goodsift = np.asarray(goodsift)
    features_result.append(goodsift.shape[0])


    # Apply ratio test on SURF matches
    goodsurf = []
    for m, n in surfmatches:
        if m.distance < 0.9 * n.distance:
            goodsurf.append([m])

    im3surf = cv2.drawMatchesKnn(im1, surfkps1, im2, surfkps2, goodsurf[:], None, flags=2)
    #cv2.imshow("SURF matching", im3surf)
    goodsurf = np.asarray(goodsurf)
    features_result.append(goodsurf.shape[0])


    # Apply ratio test on ORB matches
    goodorb = []
    for m, n in orbmatches:
        if m.distance < 0.9 * n.distance:
            goodorb.append([m])
    im3orb = cv2.drawMatchesKnn(im1, orbkps1, im2, orbkps2, goodorb[:], None, flags=2)
    #cv2.imshow("ORB matching", im3orb)
    goodorb = np.asarray(goodorb)
    features_result.append(goodorb.shape[0])


    # Apply ratio test on BRISK matches
    goodbrisk = []
    for m, n in briskmatches:
        if m.distance < 0.9 * n.distance:
            goodbrisk.append([m])

    im3brisk = cv2.drawMatchesKnn(im1, briskkps1, im2, briskkps2, goodbrisk[:], None, flags=2)
    #cv2.imshow("BRISK matching", im3brisk)
    goodbrisk = np.asarray(goodbrisk)
    features_result.append(goodbrisk.shape[0])


    return features_result

def warping(image,contours):
    #print("I am in warping")


    x1 = contours[0][0][0]
    y1 = contours[0][0][1]
    x2 = contours[1][0][0]
    y2 = contours[1][0][1]
    x3 = contours[2][0][0]
    y3 = contours[2][0][1]
    x4 = contours[3][0][0]
    y4 = contours[3][0][1]

    s1 = x1 + y1
    s2 = x2 + y2
    s3 = x3 + y3
    s4 = x4 + y4

    t = max(s1, s2, s3, s4)
    if t == s1:
        x2_main = x1
        y2_main = y1
        x1 = 0
        y1 = 0

    elif t == s2:
        x2_main = x2
        y2_main = y2
        x2 = 0
        y2 = 0

    elif t == s3:
        x2_main = x3
        y2_main = y3
        x3 = 0
        y3 = 0

    else:
        x2_main = x4
        y2_main = y4
        x4 = 0
        y4 = 0

    # print(x2_main, y2_main)

    t = min(s1, s2, s3, s4)
    if t == s1:
        x4_main = x1
        y4_main = y1
        x1 = 0
        y1 = 0

    elif t == s2:
        x4_main = x2
        y4_main = y2
        x2 = 0
        y2 = 0

    elif t == s3:
        x4_main = x3
        y4_main = y3
        x3 = 0
        y3 = 0

    else:
        x4_main = x4
        y4_main = y4
        x4 = 0
        y4 = 0
    # print(x4_main, y4_main)

    t = max(x1, x2, x3, x4)
    x3_main = t
    index_min = np.argmax([x1, x2, x3, x4])
    if index_min == 0:
        x1 = 0
    elif index_min == 1:
        x2 = 0
    elif index_min == 2:
        x3 = 0
    else:
        x4 = 0

    t = max(x1, x2, x3, x4)
    x1_main = t

    t = max(y1, y2, y3, y4)
    y1_main = t
    index_min = np.argmax([y1, y2, y3, y4])
    if index_min == 0:
        y1 = 0
    elif index_min == 1:
        y2 = 0
    elif index_min == 2:
        y3 = 0
    else:
        y4 = 0

    t = max(y1, y2, y3, y4)
    y3_main = t


    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[(x1_main, y1_main), (x2_main, y2_main), (x3_main, y3_main), (x4_main, y4_main)]], dtype=np.int32)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    pts1 = np.float32([(x3_main, y3_main), (x2_main, y2_main), (x4_main, y4_main), (x1_main, y1_main)])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(masked_image, M, (500, 500))
    
    cv2.imshow("Warped", dst)
    cv2.waitKey(1)

    return dst


def filter_top_of_robot(isStandard, frame, x_first = 0, y_first = 0):
    #print("I am in filter_top_of_robot")
    global img_plot, flag_first, adj_x, adj_y, list_white, img
    if not isStandard:
    	img_plot = cv2.imread(name_thick)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([140,  150, 150])
    upper_red = np.array([170, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)


    #cv2.imshow('Mask',mask)
    #cv2.imshow("Result",res)
    #cv2.imshow('Frame',frame)
    #print("Thick plot shape:", img_thick.shape)
    #cv2.waitKey()

    gray = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    (_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    if contours.__len__() != 0:
        cnt = contours[0]
        (x, y), radius = cv2.minEnclosingCircle(cnt)     # Making Minimum Enclosing Circle around the contour to get the coordinates of the centre
        center = (int(x), int(y))
        # coordinates.append([x,y])
        radius = int(radius)
        cv2.circle(res, center, radius, (0, 255, 0), 2)
        #cv2.imshow("Res", res)
        #cv2.waitKey(1)

        #print(center, radius)
        if (3.14) * (radius * radius) < 400: # This will filter out small contours which are found
            # print("small circle")
            x = 0
            y = 0

        if flag_first==True and (x,y)!=(0,0): # This if will be executed if it is the first pixel in the trajectory
            adj_x = x_first - x
            adj_y = y_first - y
            # print(adj_x,adj_y)
            flag_first = False

        if int(x) != 0 and int(y) != 0:
            x = x + adj_x
            y = y + adj_y
            #print("Image plot: ", img_plot[int(x),int(y),0]==255, center)
            if img_plot[int(x),int(y),0]==255: # Check if the pixel is plotted on White Foreground or Black Background
                list_white.append(1)
            else:
                list_white.append(0)
            
            cv2.line(img, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 1)
        else:
            cv2.line(img, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 20)
    '''   
    else:
        #cv2.imshow('Frame',frame)
        #print("Thick plot shape:", img_thick.shape)
        #cv2.waitKey(100)
        #print("no contour found bro")

    
    cv2.imshow('Original',frame)
    cv2.imshow("Perfect Trajectory",img)
    cv2.imshow('Thick Trajectory ',img_thick)
    cv2.waitKey(1)
    '''


def deleteframes(isStandard, file_name, contours, team_id = 0, x_first = 0, y_first = 0):

    global list_white, img_plot, adj_y,adj_x, flag_first, list_circles, img
    isFirstFrameCaptured = False
    cap = cv2.VideoCapture(file_name)
    while(cap.isOpened()):
    	#print("I am in deleteframes")
    	ret, image = cap.read()
    	if ret == False:
    		break

    	elif ret == True:
            warped_frame = warping(image,contours)
            filter_top_of_robot(isStandard, warped_frame, x_first, y_first)

        #cv2.imshow("Original", image)
        #cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    #print("line 374", list_white.__len__())
    plot_path = "team_id_" + str(team_id) + ".png"
    cv2.imwrite(plot_path, img)
    listlen = list_white.__len__()
    list_ones = list_white.count(1)
    followaccuracy = (list_ones/listlen)*100
    #print("ACUURACY MEASURE",followaccuracy)
    #print("inside ps line 379:", list_ones, listlen)


    result = {'Team_ID': team_id, 'Plot Path':plot_path , "Follow Accuracy":followaccuracy}

    list_white = []
    img = np.zeros((500,500),dtype=np.uint8)
    adj_x = 0 #this variable is used to adjust the offset while evaluating the trajectory
    adj_y = 0 #this variable is used to adjust the offset while evaluating the trajectory
    flag_first = True

    return result

    '''
    if not isStandard:
    	
    
    list_step = int(list_circles.__len__()/circle_step)
    print("List step = " + str(list_step))
    print("list circle len = " + str(list_circles.__len__()))
    print("image thick", img_thick.shape)
    
    print("*****************")
    print(list_circles)
    print("*****************")
    

    print("Lenght of circle list: ",list_circles.__len__())
    img_with_circles = np.zeros((500,500),dtype=np.uint8)
    list_circles = list(set(list(list_circles)))
    print("Lenght of unique circle list: ",list_circles.__len__())

    for i in range(0,list_circles.__len__(),200):
        list_export.append(list_circles[i])
        print((int(list_circles[i][0]), int(list_circles[i][1])))
        cv2.circle(img_with_circles, (int(list_circles[i][0]), int(list_circles[i][1])), 3, blue, -1)

        #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) 



    cv2.imwrite("first_warped_frame.png", first_warped_frame);
    cv2.imwrite(name_akaze,img)
    cv2.imwrite(name_circle,img_with_circles)
    cv2.imwrite(name_thick,img_thick)

    return (list_export)
    '''

def getContours(frame):

	area=(frame.shape)
	frame_area =area[0]*area[1]
	
	#print(frame_area)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	ret,thresh = cv2.threshold(gray,127,255,0)
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	#temp =  cv2.drawContours(gray, contours, 2, (0,255,0), 30)
	#cv2.imshow("All Counter", temp)
	#cv2.waitKey(100000)
	peri = cv2.arcLength(contours[2], True)
	approx = cv2.approxPolyDP(contours[2], 0.01 * peri, True)
	#3rd element in contours corrosponds to outter fram of arena
	return approx#contours[2]	
	
	#return contours

def standard_feature(file_name):
	#file_name = "output_correct_orientation.avi"#"new_video_final_9_ref.mov" #"output_correct_orientation.avi"

    cap = cv2.VideoCapture(file_name) # Capture video from camera
    ret, first_frame = cap.read()
    #cv2.imwrite("frame0.jpg" , frame)

    (coordinates) = getContours(first_frame)
    #print(coordinates ,"------------------")
    warped_frame = warping(first_frame, coordinates)
    cv2.imwrite("warped_first.jpg", warped_frame)
    (list_export) = getStandardFeatures(warped_frame)#deleteframes(isStandard = True,file_name = file_name, contours = coordinates)

    features_result = evaluation()

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(1, length - 100)
    ret, last_frame = cap.read()
    cv2.imwrite("standard_last.jpg", last_frame)
    #img = cv2.imread('arena_correct_placed.jpg', 1)
    red_corners, green_corners, blue_corners= getColorPoints(last_frame, isCenter = False)

    result = {'plot_akaze': name_akaze,'plot_circle':name_circle,'plot_thick':name_thick,\
    'list_circles':list_export,'features_result':features_result,\
    'red_corners': red_corners, 'green_corners' : green_corners,\
    'blue_corners' : blue_corners
    }
    result_dic.append(result.copy())




    fields = ['plot_akaze', 'plot_circle', 'plot_thick', 'list_circles',\
     'features_result', 'red_corners', 'green_corners', 'blue_corners']
    with open(csv_file_name, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(result_dic)

    fields = ['Team_ID', 'Plot Path', "Follow Accuracy", 'Red Matches','Green Matches', 'Blue Matches', 'Feature Matching', 'HOG', "Points Travelled"]
    with open("Results/Results.csv", 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
    print("Compleated!!!")
    # Our operations on the frame come here


    # Display the resulting frame
    #cv2.imshow('frame',gray)
    #cv2.waitKey(5000)
    return csv_file_name
