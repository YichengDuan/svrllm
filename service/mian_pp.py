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

def send_to_vlm(image_path, vlm_endpoint):
    """
    Send an image to the VLM and get the vector.
    
    :param image_path: Path to the image file.
    :param vlm_endpoint: Endpoint URL of the VLM service.
    :return: The vector response from the VLM.
    """
    with open(image_path, 'rb') as img_file:
        response = requests.post(vlm_endpoint, files={'file': img_file})
    
    if response.status_code == 200:
        return response.json()  # Assuming the VLM returns JSON with the vector
    else:
        raise Exception(f"Error sending image to VLM: {response.status_code}, {response.text}")

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


def video_pair_generation(video_path,CC_sequence):
    # time frame alignment need 
    
    return

def extract_cc(CC_json_path:str)-> list:
    """
    Extract CC data from CC_json_path.
    
    :param CC_json_path: Path to the CC JSON file.
    :return: List of CC data.
    """
    cc_data = {}
    with open(CC_json_path, 'r') as file:
        tmp_data = []
        # load by line from CC_json_path
        for line in file.readlines():
            tmp_line = json.loads(line)
            print(tmp_line)
            tmp_data.append()



    return cc_data
    


def process_video(video_path,CC_json_path, vlm_endpoint,pinecone_index:pinecone.Index, neo4j_driver:GraphDatabase.driver):
    """
    Process the video to extract frames, send them to VLM, 
    and store vectors in Pinecone, neo4j for graph construction
    """
    CC_sequent_raw = extract_cc(CC_json_path)

    extracted_frames,CC_sequent = video_pair_generation(video_path, CC_sequent_raw)
    
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