import os
import cv2
import csv
import ast
from process_standard import *

#from process_standard import deleteframes

img_plot = [] # Reading the image with thick line
img_with_circles = [] # Reading the image with circles

perfect_values= [] # This list will be formed once in the program. It will store the values from the csv file of perfect run
path_to_circle_image = "" #parsing through the list and giving the path of the image with circles
path_to_thick_line = ""  #parsing through the list and giving the path of the image with thick line
# perfect_values[0][3] = [] #parsing through the list and giving the first coordinate of perfect trajectory
x_first = 0
y_first = 0 # x_first & y_first store integer value of the x & y coordinates of the first pixel of perfect trajectory
result_dic = []
fields = ['Team_ID','Plot Path',"Follow Accuracy",'Red Points', 'Green Points', 'Blue Points']

def getStandardValues(csvpath):
	global perfect_values,path_to_circle_image,path_to_thick_line,x_first,y_first,img_plot,img_with_circles
	with open(csvpath, 'r') as csvfile:  # Opening the csv file
		# creating a csv reader object
		csvreader = csv.reader(csvfile)

		# extracting field names through first row
		fields = next(csvreader)

		# extracting each data row one by one
		for row in csvreader:
		    perfect_values.append(row)

	path_to_circle_image = perfect_values[0][1]  # parsing through the list and giving the path of the image with circles
	path_to_thick_line = perfect_values[0][2]  # parsing through the list and giving the path of the image with thick line
	perfect_values[0][3] = ast.literal_eval(perfect_values[0][3])  # parsing through the list and giving the first coordinate of perfect trajectory
	(x_first, y_first) = perfect_values[0][3][0]  # x_first & y_first store integer value of the x & y coor
	#img_plot = cv2.imread(path_to_thick_line)  # Reading the image with thick line
	#img_with_circles = cv2.imread(path_to_circle_image)  # Reading the image with circles

def writeIntermediateResult():
	csv_file_name = os.path.join(os.getcwd(), 'Results', 'intermediate_results.csv')

	with open(csv_file_name, 'w') as csvfile:
        # creating a csv dict writer object
		writer = csv.DictWriter(csvfile, fieldnames=fields)
		writer.writeheader()
		writer.writerows(result_dic)

def start_evaluating(csv_file_name, folder_path):
	
	getStandardValues(csv_file_name)

	
	#print("Starting evaluation")


	
	for file in os.listdir(folder_path):

		filename = os.fsdecode(file)
		result = {}
		team_id = filename.split(".")[0];
		#print(folder_path+""+filename)
		try:
			cap = cv2.VideoCapture(folder_path+filename) # Capture video from camera
			ret, frame = cap.read()
			#cv2.imwrite("frame0.jpg" , frame)

			(coordinates) = getContours(frame)
			print("--------Inside Evaluate for: ----------", team_id)
			(result) = deleteframes(False, folder_path+filename, coordinates, team_id, x_first, y_first)

			length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			cap.set(1, length - 100)
			ret, last_frmae = cap.read()
		    #cv2.imwrite("standard_last.jpg", img)

			#img = cv2.imread('arena_misplaced.jpg', 1)
			red_center, green_center, blue_center = getColorPoints(last_frmae, isCenter = True)
			result['Red Points'] = red_center
			result['Green Points'] = green_center
			result['Blue Points'] = blue_center
			
			#print("inside start evaluating line 85: ...", result, team_id)

			result_dic.append(result.copy())
		
		except:
			print("************Exception*****************", team_id)
			for value in fields:
				result[value] = "Exception" 	
			
			result['Team_ID'] = team_id
			result_dic.append(result.copy())
		
	print(result_dic)
	writeIntermediateResult()
	
	