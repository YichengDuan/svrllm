# Licensed under the MIT License

# Video preprocessing with multiprocessing 
import cv2
import numpy as np
import requests
import os
import pinecone  # Import the Pinecone client
import json
from neo4j import GraphDatabase
from numpy.matlib import empty


def extract_frames(video_path, output_dir, interval=300):
    """
    Extract frames from a video every 'interval' seconds.
    
    :param video_path: Path to the input video file.
    :param output_dir: Directory to save extracted frames.
    :param interval: Time interval in seconds to extract frames (default is 300 seconds).
    :return: List of paths to the extracted frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    extracted_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            extracted_frames.append(frame_filename)
        
        frame_count += 1

    cap.release()
    return extracted_frames

def send_to_vlm(frame, cc:dict, background:dict, retrun_vec:bool):
    ## take the frame, take the background, take the CC 
    # prompt enhancement
    # prompt = "" img + text
    #   messages = [
        # {
        #     "role": "user",
        #     "content": [
        #         {
        #             "type": "image",
        #             "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        #         },
        #         {"type": "text", "text": "Is there a dog? Only anwser yes or no"},
        #         # 可改
        #     ],
        # }
    # ]

    # send to vlm model and get the vector

    vec = [0.0,0.0,0.0,0.0,0.0,0]
    vlm_answer = ""
    if retrun_vec : 
        return vec
    else:
        return vlm_answer
    


# def store_vector_in_pinecone(vector, vector_id):
#     """
#     Store the vector in the Pinecone index.
    
#     :param vector: The vector to store.
#     :param vector_id: Unique ID for the vector.
#     """
#     try:
#         pinecone_index.upsert([(vector_id, vector)])
#         print(f"Vector stored in Pinecone with ID: {vector_id}")
#     except Exception as e:
#         print(f"Error storing vector in Pinecone: {e}")


def video_pair_generation(video_path,CC_sequence,invterval:300,tag:bool)->tuple[list]:
    # time frame alignment need 
    # with the give interval ................
    # cc "PrimaryTag" can use as segmentation
    # diveide if known cc tag ... 
    # can do video img size change ...

    # index alignment need !!!!!!!
    vid_frames = []
    cc_seq_data = []
    

    return vid_frames,cc_seq_data

def extract_cc(CC_json_path:str)-> dict:
    """
    Extract CC data from CC_json_path.
    
    :param CC_json_path: Path to the CC JSON file.
    :return: dict of CC data.
    """
    cc_data = {"background":{},"CC":[]}
    temp_text = []  # store the uncompleted subtitles
    temp_start_time = None  # store the start time

    with open(CC_json_path, 'r', encoding='utf-8') as file:
        # each line for the json
        for line in file:
            try:
                # load the json file
                tmp_line = json.loads(line.strip())
                # if segmentation
                if "PrimaryTag" in tmp_line and tmp_line["PrimaryTag"].startswith("SEG"):
                    cc_data["CC"].append({
                        "StartTime": tmp_line.get("StartTime"),
                        "EndTime": tmp_line.get("EndTime"),
                        "PrimaryTag": tmp_line.get("PrimaryTag", ""),
                        "Type": tmp_line.get("Type", ""),
                    })
                # according to the key to determine is background or not ( here we use the existence of start time and end time)
                elif "StartTime" in tmp_line and "EndTime" in tmp_line:
                    text = tmp_line.get("Text", "")
                    start_time = tmp_line.get("StartTime")
                    end_time = tmp_line.get("EndTime")
                    primaryTag = tmp_line.get("PrimaryTag")
                    if temp_text:
                        # temp_text is not empty which means that there are uncompleted subtitles
                        temp_text.append(text)
                        # check if the cc is completed according ".", "!", "?"
                        if text.endswith((".", "!", "?","♪",")")):
                            cc_data["CC"].append({
                                "StartTime": temp_start_time,
                                "EndTime": end_time,
                                "PrimaryTag": primaryTag,
                                "Text": " ".join(temp_text),
                            })
                            temp_text = []
                            temp_start_time = None
                    else:
                        # if the subtitle is single
                        if text.endswith((".", "!", "?")):
                            cc_data["CC"].append({
                                "StartTime": start_time,
                                "EndTime": end_time,
                                "PrimaryTag": primaryTag,
                                "Text": text,
                            })
                        else:
                            # uncompleted, a new start
                            temp_text.append(text)
                            temp_start_time = start_time
                else:
                    # background
                    for key, value in tmp_line.items():
                        cc_data["background"][key] = value
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}, error: {e}")
    return cc_data
    
def store_data():
    # node <-> vec pair


    return

def process_video(video_path,CC_json_path, vlm_endpoint,pinecone_index:pinecone.Index, neo4j_driver:GraphDatabase.driver):
    """
    Process the video to extract frames, send them to VLM, 
    and store vectors in Pinecone, neo4j for graph construction
    """
    CC_sequent_raw = extract_cc(CC_json_path)

    extracted_frames,CC_sequent = video_pair_generation(video_path, CC_sequent_raw)
    
    # write an loop that batch of zip  extracted_frames,CC_sequent 

    result = send_to_vlm()

    # send to database:
    # store in pinecone
    # store in neo4j

    store_data()
    # for idx, frame_path in enumerate(extracted_frames):
    #     vector = send_to_vlm(frame_path, vlm_endpoint)
    #     if vector:
    #         store_vector_in_pinecone(vector, f"frame-{idx}")
    #         print(f"Processed {frame_path}, vector stored in Pinecone.")

# Example usage
if __name__ == '__main__':
    # video_path = 'your_video.mp4'
    # vlm_endpoint = 'http://your-vlm-endpoint.com/api'  # Replace with your VLM endpoint
    # process_video(video_path, vlm_endpoint)

    process_video()