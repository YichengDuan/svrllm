# Licensed under the MIT

# video perpreocssing with mutiprocsssing 


import cv2
import numpy as np
import requests
import os

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

def store_vector_in_vdb(vector, vdb_endpoint):
    """
    Store the vector in the VDB.
    
    :param vector: The vector to store.
    :param vdb_endpoint: Endpoint URL of the VDB service.
    """
    response = requests.post(vdb_endpoint, json={'vector': vector})
    
    if response.status_code != 200:
        raise Exception(f"Error storing vector in VDB: {response.status_code}, {response.text}")

def process_video(video_path, vlm_endpoint, vdb_endpoint, output_dir='frames'):
    """
    Process the video to extract frames, send them to VLM, and store vectors in VDB.
    
    :param video_path: Path to the input video file.
    :param vlm_endpoint: Endpoint URL of the VLM service.
    :param vdb_endpoint: Endpoint URL of the VDB service.
    :param output_dir: Directory to save extracted frames.
    """
    extracted_frames = extract_frames(video_path, output_dir)
    
    for frame_path in extracted_frames:
        vector = send_to_vlm(frame_path, vlm_endpoint)
        store_vector_in_vdb(vector, vdb_endpoint)
        print(f"Processed {frame_path}, vector stored in VDB.")

# Example usage
video_path = 'your_video.mp4'
vlm_endpoint = 'http://your-vlm-endpoint.com/api'  # Replace with your VLM endpoint
vdb_endpoint = 'http://your-vdb-endpoint.com/api'  # Replace with your VDB endpoint

process_video(video_path, vlm_endpoint, vdb_endpoint)