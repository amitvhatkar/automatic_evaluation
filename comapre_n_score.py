import cv2
import numpy as np
import os
import csv
import ast
import PyQt5
'''

* Project:           e-Yantra Automatic Evaluation of Videos
* Author List:       Saim Shaikh, Siddharth Aggarwal
* Filename:          evaluation_modular.py
* Functions:         eval, programmatic, feature_matching, HOG_correlation
* Global Variables:  perfect_values,path_to_perfect_image,feature_list,feature_list,coordinates
                     img_perfect,csv_file_name,result_dic
'''
perfect_values = []
path_to_perfect_image=""
feature_list=[]
coordinates = []

red_corners = []
blue_corners = []
green_corners = []

path_to_plot =""
path_to_circle=""

'''
    Function
    Name: programmatic
    Input: plot
    with the programatic circles
    Output: result
    of
    programmatic
    analysis in percentage

    Logic: In this function we accept a plot with the programmatic circles plotted according to the
    checkpoint coordinates, we extract the roi of these circles and check whether the trajectory is passing
    through the circles.Based on this we calculate the score.
'''
result = []
def write_to_csv(row):
    #fields = ['Team_ID','Plot Path','Circle Path', 'Handling Count', "Physical Marker 1 Time", "Physical Marker 2 Time", "Follow Accuracy",'prog','feature','hog']
    with open('Results/Results.csv', 'a') as csvfile:
        #print("hi i am in write")
        # creating a csv dict writer object
        writer = csv.writer(csvfile)

        # writing headers (field names)
#        writer.writeheader()

        # writing data rows
        #print(result_dic)
        #print("Writing row:",row)
        writer.writerow(row)

def start_scoring(csv_perfect):
    global path_to_plot, path_to_perfect_image, feature_list, coordinates, result, red_corners, green_corners, blue_corners

    with open(csv_perfect,'r') as csvfile:

        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        # extracting field names through first row
        fields = next(csvreader)
        # extracting each data row one by one
        for row in csvreader:
            # print(row)
            perfect_values.append(row)

    path_to_perfect_image = perfect_values[0][0] #extracting perfect trajectory for evaluation
    feature_list= ast.literal_eval(perfect_values[0][4]) #akaze values of the perfect trajectory
    red_corners = list(ast.literal_eval(perfect_values[0][5]))
    green_corners =list(ast.literal_eval(perfect_values[0][6]))
    blue_corners = list(ast.literal_eval(perfect_values[0][7]))

    coordinates = ast.literal_eval(perfect_values[0][3]) #co-ordinates of checkpoints for programmatic analysis
    #print(coordinates)
    perfect_values.clear()

    with open("Results/intermediate_results.csv",'r') as csvfile:

        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        # extracting field names through first row
        fields = next(csvreader)
        # extracting each data row one by one
        for row in csvreader:

            print("Shape of row------->",row)
            if row[1] == "Exception":
                row.append("Exception") #Feature matching
                row.append("Exception") #HOG corelation
                write_to_csv(row)
                continue    
            print("Printing Row : ",row[3],row[4], row[5])
            
        #     perfect_values.append(row)
        # print(perfect_values)
        # for i in range(0,perfect_values.__len__()):
            path_to_plot =row[1] #extracting percentagerfect trajectory for evaluation
            #print(path_to_plot)
            red_string = row[3];
            green_string = row[4];
            blue_string = row[5];
            #result = []

            try:
                row[3], row[4], row[5] = colorPositionChecker(red_center = list(ast.literal_eval(red_string)),\
                green_center = list(ast.literal_eval(green_string)),blue_center = list(ast.literal_eval(blue_string)))
            except:
                row[3] = "Exception"
                row[4] = "Exception"
                row[5] = "Exception"
            try:
                features_result = feature_match()
                #print("Printing feature result: ", features_result)
                row.append(str(features_result))
            except:
                row.append("Exception")

            try:
                hog_result = HOG_correlation()
                #print(hog_result)
                row.append(str(hog_result))
            except:
                row.append("Exception")

            #print("path to plot", path_to_plot)
            try:
                travelled_points_result = programmatic(path_to_plot)
                row.append(str(travelled_points_result))
            except:
                row.append("Exception")

            #print("Before writing line 113 in cns:",row)
            write_to_csv(row)


def chkCenterPosition(points, corner_list):
    restult = 0
    
    for point in points:
        
        for corners in corner_list:
            #print("x,",corners[0] , point[0] , corners[2])
            #print("y",corners[1] , point[1] , corners[3])
            #print((corners[0] <= point[0] <= corners[2] and \                corners[1] <= point[1] <= corners[3]))
            #print("-------------------------------------")
            if(corners[0] <= point[0] <= corners[2] and \
                corners[1] <= point[1] <= corners[3]):
                restult += 1
                break;

    return restult;

def colorPositionChecker(red_center, green_center, blue_center):
    global red_corners, green_corners, blue_corners

    red_matches = chkCenterPosition(red_center, red_corners)
    green_matches = chkCenterPosition(green_center, green_corners)
    blue_matches = chkCenterPosition(blue_center, blue_corners)

    return red_matches, green_matches, blue_matches


def programmatic(path_to_circle):
    img_circle = cv2.imread(path_to_circle)


    #cv2.imshow("Path plot",img_circle)
    #cv2.waitKey()
    # programmatic checkpoints
    circle_radius = 8 #raduis of the programmatic circle
    check_list = [] #list of circles passed by the trajectory
    check_counter = 0 #count of checkpoints crossed

    #iterating through the co-ordinates
    #print("length of co-ordinates",coordinates)
    for i in coordinates:
        #print(i)
        a = int(i[0])
        b = int(i[1])

        #extracting the roi of the plotted circle
        roi = img_circle[b - (3 * circle_radius): b + (3 * circle_radius),
              a - (3 * circle_radius): a + (3 * circle_radius)]
        roi = roi.reshape(int(roi.size / 3), 3)

        #checking whether the trajectory is passing through the roi
        if [255, 255, 255] in roi.tolist():
            check_list.append(1)
            check_counter += 1

        else:
            check_list.append(0)

    #checking result
    check_result = ((check_counter / check_list.__len__()) * 100)

    #print("Programmatic Analysis Result = ", check_counter, check_list.__len__())
    #print(check_result)
    return check_result
    # load the image and convert it to grayscale



'''
Function
Name: feature_match
Input: plot
with the trajectory
Output: feature matching
results

Logic: In this function various feature matching algorithms like AKAZE BRISK are used to extract and match features between
the perfect and imperfect trajectories.Based on the number of correct matches a score is calculated
'''
####################################### ANALYSIS USING FEATURE MATCHING ##################################################
def feature_match():
    # load the image and convert it to grayscale
    im1 = cv2.imread(path_to_perfect_image)
    im2 = cv2.imread(path_to_plot)

    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE,BRISK descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    akaze = cv2.AKAZE_create()
    brisk = cv2.BRISK_create()
    orb = cv2.ORB_create()

    #compute the descriptors and keypoints using AKAZE BRISK ORB SIFT and SURF
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

    # Match the fezatures using the Brute Force Matcher
    bfakaze = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    #Refine the Brute Force Matches using the KNN Matcher
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
    goodakaze = np.asarray(goodakaze)
    #print(feature_list)
    #print(goodakaze.shape[0])
    #calculate the Akaze core using the number of good matches
    #print(goodakaze.shape[0])
    #print(feature_list[0])
    similarity_akaze = (goodakaze.shape[0]/feature_list[0])*100

    # Apply ratio test on SIFT matches
    goodsift = []
    for m, n in siftmatches:
        if m.distance < 0.9 * n.distance:
            goodsift.append([m])
    #im3sift = cv2.drawMatchesKnn(img_perfect, siftkps1, img_akaze, siftkps2, goodsift[:], None, flags=2)
    goodsift = np.asarray(goodsift)
    similarity_sift = (goodsift.shape[0] / feature_list[1]) * 100

    # Apply ratio test on SURF matches
    goodsurf = []
    for m, n in surfmatches:
        if m.distance < 0.9 * n.distance:
            goodsurf.append([m])
    goodsurf = np.asarray(goodsurf)
    similarity_surf = (goodsurf.shape[0] / feature_list[2]) * 100

    # Apply ratio test on ORB matches
    goodorb = []
    for m, n in orbmatches:
        if m.distance < 0.9 * n.distance:
            goodorb.append([m])
    goodorb = np.asarray(goodorb)
    similarity_orb = (goodorb.shape[0] / feature_list[3]) * 100

    # Apply ratio test on BRISK matches
    goodbrisk = []
    for m, n in briskmatches:
        if m.distance < 0.9 * n.distance:
            goodbrisk.append([m])
    goodbrisk = np.asarray(goodbrisk)

    #Calculating the Similarity using the BRISK algorithm
    similarity_brisk = (goodbrisk.shape[0] / feature_list[4]) * 100
    features_result = (similarity_akaze+similarity_brisk+similarity_orb+similarity_sift+similarity_surf)/5

   #calculating overall similarity by aggregating the results of various feature actching algorithms
    #print("Overall similarity using features: ")
    #print(features_result)
    return features_result



    ######################################### HOG CORRELATION ###############################################
xdef HOG_correlation():
    bin_n = 16

    img = cv2.imread(path_to_perfect_image)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)

    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)

    bins = np.int32(bin_n * ang / (2 * np.pi))

    # Divide to 4 sub-squares

    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]

    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]

    hist1 = np.hstack(hists)

    img = cv2.imread(path_to_plot)
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    img = cv2.warpAffine(img, M, (cols, rows))

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)

    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)

    bins = np.int32(bin_n * ang / (2 * np.pi))

    # Divide to 4 sub-squares

    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]

    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]

    hist2 = np.hstack(hists)

    # print(np.corrcoef(hist1,hist2))

    hog_result = ((np.corrcoef(hist1, hist2)[0][1]) * 100)
    #print("HOG CORRELATION RESULT = ")
    #print(hog_result)
    return hog_result


#evaluate('/Users/siddharth/Desktop/EYSIP/NEW VIDS & RESULTS/Results/results_perfect.csv')#,'/Users/siddharth/Desktop/EYSIP/NEW VIDS & RESULTS/Results/intermediate_results.csv')