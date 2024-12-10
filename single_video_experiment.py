import cv2
from config import pinecone_client, neo4j_driver, pinecone_index
from vlm import VLM_EMB
import os
from tqdm import tqdm
from mian_pp import process_video
from retrival import input_preprocess,retrieve_method
from util import extract_frame,calculate_video_duration

import os
from vlm import VLM_EMB
# from config import neo4j_driver,pinecone_index

if __name__ == "__main__":
    local_path = os.path.dirname(os.path.abspath(__file__))

    video_path = os.path.join(local_path,"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4")
    CC_json_path = os.path.join(local_path,"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.json")


    vlm = VLM_EMB()
    

    tmp_frame_path = os.path.join(local_path,"frames/tmp/")

    
    duration =  calculate_video_duration(video_path)


    total_results = retrieve_method(video_path = video_path,total_time=duration)


    process_video(video_path=video_path, CC_json_path=CC_json_path,vlm_endpoint=vlm)

    print(total_results)
    print("="*100)
    print(f"Best range is at start: {total_results[0]['start']}s, end {total_results[0]['end']}s, in video name: {total_results[0]['name']}")

    test_range = range(0,duration,50)

    for _S in range(1,4):
         # Initialize counters
        correct = 0
        total_count = 0

        TP = 0  # True Positives
    
        FN = 0  # False Negatives
        # Define a tolerance 
        strength = 1
        for _sec in tqdm(test_range, desc=f"testing strength={_S}"):
            if _sec % 100 == 0:
                continue
                
            total_results = retrieve_method(video_path=video_path, strength=_S, target_sec=float(_sec),total_time=duration)
            
            # Prediction: Check if the predicted range includes the target second
            predicted_start = total_results[0]['start']
            predicted_end = total_results[0]['end']
            prediction = predicted_start <= _sec <= predicted_end

            # Ground Truth
            event_occurs = True

            # Update counts based on prediction and ground truth
            if prediction:
                if event_occurs:
                    TP += 1  # True Positive: Correctly predicted event occurrence
            else:
                if event_occurs:
                    FN += 1  # False Negative: Missed actual event occurrence

            total_count += 1
    
        recall = TP / total_count if total_count else 0       # True Positive Rate
        print(f"Recall strength={_S} at (True Positive Rate): {round(recall * 100, 2)}%")