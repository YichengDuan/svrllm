import cv2
from config import pinecone_client, neo4j_driver, pinecone_index
from vlm import VLM_EMB
import os
from tqdm import tqdm

from retrival import extract_frame,input_preprocess




def retrieve_method(video_path,strength=1,target_sec=1250.0,total_time=3600.0):
    output_image_path = "./frames/target_img.jpg"

    extract_frame(video_path=video_path,time_sec=target_sec,output_image_path=output_image_path)
    result = input_preprocess(img_path=output_image_path,strength=strength)
    # print(result)
    total_results = []
    for _hit in result:
        if len(_hit['next']) == 0:
            tmp_end = total_time
        else:
            tmp_end = _hit['next'][-1]['timestamp']
        if len(_hit['previous']) == 0:
            tmp_start = 0
        else:
            tmp_start = _hit['previous'][-1]['timestamp']
        video_name = _hit['namespace']
        total_results.append({'start':tmp_start, 'end':tmp_end, 'name':video_name})
    return total_results






if __name__ == "__main__":
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4")
    print("test on 1250s")
    
    cap = cv2.VideoCapture(video_path,cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    total_results = retrieve_method(video_path = video_path,total_time=duration)
    print(total_results)
    print("="*100)
    print(f"Best range is at start: {total_results[0]['start']}s, end {total_results[0]['end']}s, in video name: {total_results[0]['name']}")

    test_range = range(0,3500,50)

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
                
            total_results = retrieve_method(video_path=video_path,strength=_S, target_sec=float(_sec),total_time=duration)
            
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