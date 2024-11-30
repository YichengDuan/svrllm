# Licensed under the MIT License

# Video preprocessing with multiprocessing 
import cv2
import numpy as np
import requests
import os
import pinecone  # Import the Pinecone client
import json
from neo4j import GraphDatabase
from datetime import datetime
from numpy.matlib import empty
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
from PIL import Image


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
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_seconds = timestamp_ms / 1000.0
            frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            extracted_frames.append({
                "frame_path": frame_filename,
                "timestamp": timestamp_seconds
            })
        
        frame_count += 1

    cap.release()
    return extracted_frames


model_path = "./model"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
device = torch.device("mps")
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map=None)
model = model.to(device)

def send_to_vlm(frames: list[dict], ccs: list[dict], background: dict, retrun_vec: bool):
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

    if len(frames) != len(ccs):
        raise ValueError("The length of frames and ccs must match.")
    
    images = []
    prompts = []
    for frame, cc in zip(frames, ccs):
        image = cv2.imread(frame["frame_path"])
        if image is None:
            raise FileNotFoundError(f"Cannot load image from path: {frame['frame_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        images.append(image_pil)
        
        cc_text = cc.get("cc_text", "No subtitle provided.")
        selected_background = {
            key: background.get(key, "N/A") 
            for key in ["Program_Title", "Comment", "Duration"]
        }
        background_info = "; ".join([f"{key}: {value}" for key, value in selected_background.items()])
        prompt = (
            f"Analyze the image at timestamp {frame['timestamp']} seconds in a video.\n"
            f"Subtitle: {cc_text}\n"
            f"Background Info: {background_info}\n"
            f"Provide summary."
        )
        prompts.append(prompt)

    for image in images:
        if not isinstance(image, Image.Image):
            raise ValueError("One of the images is not a valid PIL Image.")

    if not all(isinstance(prompt, str) and len(prompt) > 0 for prompt in prompts):
        raise ValueError("Invalid prompts: Ensure all prompts are non-empty strings.")
    
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        if retrun_vec:
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            vectors = last_hidden_state.mean(dim=1).cpu().numpy()
            results = [{"frame_path": frame["frame_path"], "vector": vector} 
                       for frame, vector in zip(frames, vectors)]
        else:
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            results = [{"frame_path": frame["frame_path"], "text": text} 
                       for frame, text in zip(frames, output_texts)]
    
    return results
    

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


def parse_time(time_str) -> datetime:
    time_format = "%Y%m%d%H%M%S.%f"
    return datetime.strptime(time_str, time_format)


def video_pair_generation(video_path, CC_sequence, interval:int = 300, tag:bool = True) -> tuple[list]:
    # time frame alignment need 
    # with the give interval ................
    # cc "PrimaryTag" can use as segmentation
    # diveide if known cc tag ... 
    # can do video img size change ...

    # index alignment need !!!!!!!
    
    output_dir = "./frames"
    vid_frames = extract_frames(video_path, output_dir, interval=interval)
    print(f"[Finished]: extract_frames. Length of vid_frames = {len(vid_frames)}")

    cc_seq_data = []
    raw_cc_data = CC_sequence["CC"]
    video_start_time = parse_time(raw_cc_data[0]["StartTime"])
    cc_index = 0  # Start from the first CC entry

    for frame in vid_frames:
        frame_time = frame["timestamp"]

        # Update the CC index to the closest matching CC
        while cc_index < len(raw_cc_data):
            cc_end_time = parse_time(raw_cc_data[cc_index]["EndTime"])
            cc_primary_tag = raw_cc_data[cc_index]["PrimaryTag"]
            if frame_time > (cc_end_time - video_start_time).total_seconds() or "SEG" in cc_primary_tag:
                cc_index += 1
            else:
                break

        # Check if the frame falls into the current CC range
        cc_start_time = parse_time(raw_cc_data[cc_index]["StartTime"])
        cc_end_time = parse_time(raw_cc_data[cc_index]["EndTime"])
        matching_cc = (
            raw_cc_data[cc_index]
            if cc_index < len(raw_cc_data) and (cc_start_time - video_start_time).total_seconds() <= frame_time <= (cc_end_time - video_start_time).total_seconds()
            else None
        )

        cc_seq_data.append({
            "cc_text": matching_cc["Text"] if matching_cc else None,
            "primary_tag": matching_cc["PrimaryTag"] if matching_cc and tag else None
        })

    return vid_frames, cc_seq_data

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

def process_video(video_path,CC_json_path, vlm_endpoint=None,pinecone_index:pinecone.Index=None, neo4j_driver:GraphDatabase.driver=None):
    """
    Process the video to extract frames, send them to VLM, 
    and store vectors in Pinecone, neo4j for graph construction
    """
    CC_sequent_raw = extract_cc(CC_json_path)
    print(f"[Finished]: extract_cc. Length of CC in CC_sequent_raw = {len(CC_sequent_raw['CC'])}")

    extracted_frames, CC_sequent = video_pair_generation(video_path, CC_sequent_raw)
    print(f"[Finished]: video_pair_generation. Length of extracted_frames = {len(extracted_frames)}")
    
    # write an loop that batch of zip  extracted_frames,CC_sequent 

    result = send_to_vlm(extracted_frames, CC_sequent, CC_sequent_raw["background"], True)
    print(f"[Finished]: send_to_vlm. Length of result = {len(result)}")

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
    video_path = './test.mp4'
    CC_json_path = './test.json'
    # vlm_endpoint = 'http://your-vlm-endpoint.com/api'  # Replace with your VLM endpoint
    # process_video(video_path, vlm_endpoint)

    process_video(video_path=video_path, CC_json_path=CC_json_path)