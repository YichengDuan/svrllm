# Licensed under the MIT License

# Video preprocessing with multiprocessing 
import cv2
import numpy as np
import requests
import os
import pinecone  # Import the Pinecone client
import json
from neo4j import GraphDatabase

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

def extract_cc(CC_json_path:str)-> list:
    """
    Extract CC data from CC_json_path.
    
    :param CC_json_path: Path to the CC JSON file.
    :return: List of CC data.
    """
    cc_data = {"background":{},"CC":{}}
    # format

    with open(CC_json_path, 'r') as file:
        tmp_data = []
        # load by line from CC_json_path
        for line in file.readlines():
            tmp_line = json.loads(line)
            print(tmp_line)
            tmp_data.append()
            


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