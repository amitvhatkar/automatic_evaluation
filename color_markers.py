import numpy as np
import cv2


offset =  20

blue_center = []
green_center = []
red_center = []
list_circles = []

def getPoints(masked, isCenter = True):

	gray = cv2.bilateralFilter(masked, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)

	(_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)	
	contours = sorted(contours, key=cv2.contourArea, reverse=True)

	points = []
	for cnt in contours:
		#cnt1 = contours[0]
		(x1, y1), _ = cv2.minEnclosingCircle(cnt)
		x1 = int(x1)
		y1 = int(y1)
		isAway = True

		if(len(points) != 0):
			for x, y in points:
				if abs(x - x1) < offset and  abs(y - y1) < offset:
					isAway = False
					break;
					#centers.remove((x, y))
			if(isAway):
				points.append((x1, y1))
		else:
			points.append((x1, y1))

	if not isCenter:
		result = list(map(lambda x: (x[0] - offset, x[1] - offset, x[0] + offset, x[1] + offset), points))
		return result

	
	#print(points)
	#cv2.imshow("Masked....", masked)
	#cv2.waitKey()
	
	return points
	

	#print("Result: ",result)

	'''
	cv2.rectangle(masked, (centers[0][0] - offset, centers[0][1] - offset), (centers[0][0] + offset, centers[0][1] + offset), (255,0,0), -1)
	cv2.rectangle(masked, (centers[1][0] - offset, centers[1][1] - offset), (centers[1][0] + offset, centers[1][1] + offset), (255,0,0), -1)
	cv2.imshow("lalala", masked)
	cv2.waitKey()
	
	return result
	'''
def getPinkPoints(hsv, isCenter):
	lower_pink = np.array([140,  150, 150])
	upper_pink = np.array([170, 255, 255])
	mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
	
	#cv2.imshow("Bot", mask_pink)
	#cv2.waitKey()

	return getPoints(mask_pink, isCenter);

def getRedPoints(hsv, isCenter):
	lower_red = np.array([0,230,230])
	upper_red = np.array([10,255,255])	
	mask_red = cv2.inRange(hsv, lower_red, upper_red)
	
	return getPoints(mask_red, isCenter);

def getBluePoints(hsv, isCenter):
	lower_blue = np.array([110,  40, 40])
	upper_blue = np.array([130, 255, 255])	
	mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
	
	return getPoints(mask_blue, isCenter);

def getGreenPoints(hsv, isCenter):
	lower_green = np.array([60,  40, 40])
	upper_green= np.array([70, 255, 255])
	mask_green = cv2.inRange(hsv, lower_green, upper_green)
	
	return getPoints(mask_green, isCenter);


def getColorPoints(img, isCenter):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	red_points = getRedPoints(hsv, isCenter)
	green_points = getGreenPoints(hsv, isCenter)
	blue_points = getBluePoints(hsv, isCenter)
	

	return red_points, green_points, blue_points


def getStandardFeatures(img):

	#cv2.imshow("sent image", img)
	#cv2.waitKey()

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	lower_black = np.array([0,0,0])
	upper_black= np.array([85,85,180])

	mask_black = cv2.inRange(hsv, lower_black, upper_black)
	
	img_with_circles = np.zeros((500,500),dtype=np.uint8)
	img_thick = np.zeros((500,500),dtype=np.uint8)
	img_akaze = np.zeros((500,500),dtype=np.uint8)

	(x_first,y_first) = getPinkPoints(hsv, True)[0]
	#print("Initial points: ",(x_first,y_first))
	list_circles.append((x_first,y_first))
	step = 50;
	k = 0
	for i in range(0, 500):
		for j in range(0, 500):
			#print(mask_black[i,j])
			if mask_black[i,j] == 255:
				#(x, y), radius = cv2.minEnclosingCircle(cnt)
				#print(x, y, contours.__len__(), cnt)
				if k% step == 0:
					list_circles.append((j,i))
				k+= 1;	
				cv2.circle(img_thick, (j, i), 5	, [255,255,255], -1)
				cv2.circle(img_akaze, (j, i), 1	, [255,255,255], -1)
	#cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) 


	#(_, contours, _) = cv2.findContours(img_with_circles.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)	
	#contours = sorted(contours, key=cv2.contourArea, reverse=True)
	'''
	print(list_circles)
	print(len(list_circles))


	for (i,j) in list_circles:
		cv2.circle(img_with_circles, (i, j), 2	, [255,255,255], -1)	
	'''
	
	#cv2.imshow("img_akaze", img_akaze)
	#cv2.waitKey()

	cv2.imwrite("perfect_trajectory_thick.png",img_thick)
	cv2.imwrite("perfect_trajectory_akaze.png",img_akaze)
	return (list_circles)

'''
if __name__== "__main__":
	img = cv2.imread('warped_first.jpg', 1)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	pink_points = getPinkPoints(hsv, True)
	#print(getColorPoints(img, isCenter = True))
	print(pink_points)
'''