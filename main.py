#from evaluate_final import evaluate
#from standardization import standard
#from threaded_final import mainplot
from process_standard import standard_feature
from evaluate import start_evaluating
from comapre_n_score import start_scoring

import cv2


videos_folder_path = "videos/"

if __name__== "__main__":


    csv_file_name = standard_feature("perfect_place.mp4")

    #csv_file_name = "results_perfect.csv"
    
    start_evaluating(csv_file_name, videos_folder_path)

    start_scoring(csv_file_name)