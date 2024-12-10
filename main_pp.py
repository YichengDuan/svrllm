# Licensed under the MIT License

# Video preprocessing with multiprocessing 
import cv2
import os
import pinecone  # Import the Pinecone client
import json
from neo4j import GraphDatabase,Driver

from datetime import datetime
from util import create_neo4j_node, store_vector_in_pinecone, extract_frames_opencv
from vlm import VLM_EMB
from tqdm import tqdm
from config import pinecone_index,neo4j_driver

def send_to_vlm(frames: list[dict], ccs: list[dict],vlm:VLM_EMB, background: dict, retrun_vec: bool,batch_size=2):
    ## take the frame, take the background, take the CC 
    # send to vlm model and get the vector

    if len(frames) != len(ccs):
        raise ValueError("The length of frames and ccs must match.")
    
    image_path_list = []
    text_prompt_list = []
    for frame, cc in zip(frames, ccs):
        image_path = frame["frame_path"]
        if image_path is None:
            raise FileNotFoundError(f"Cannot load image from path: {frame['frame_path']}")
        image_path_list.append(image_path)
        
        cc_text = cc.get("cc_text", "No subtitle provided.")
        selected_background = {
            key: background.get(key, "N/A") 
            for key in ["Program_Title", "Comment", "Duration"]
        }
        # background_info = "; ".join([f"{key}: {value}" for key, value in selected_background.items()])
        # text_prompt = (
        #     f"Analyze the image at timestamp {frame['timestamp']} seconds in a video.\n"
        #     f"Subtitle: {cc_text}\n"
        #     f"Background Info: {background_info}\n"
        #     f"Provide summary."
        # )
        text_prompt = (
            f"This is a news frame extracted from a program. \n"
            f"Program Title: {selected_background['Program_Title']}\n"
            f"Comment: {selected_background['Comment']}\n"
            f"Program Duration: {selected_background['Duration']}\n"

            f"Frame Timestamp: {frame['timestamp']} s\n"
            f"Subtitle at this timestamp: '{cc_text}'\n"
            f"Task: Summarize the key event depicted in this frame based on the subtitle and the program background. Also include the frame timestamp in your response.\n"
        )
        text_prompt_list.append(text_prompt)
    
    result = []
    l = len(list(zip(image_path_list, text_prompt_list)))
    for ndx in tqdm(range(0, l, batch_size), desc="vlm embedding/generation..."):
        batch_image_path_list = image_path_list[ndx:min(ndx + batch_size, l)]
        batch_text_prompt_list = text_prompt_list[ndx:min(ndx + batch_size, l)]
        if retrun_vec:
            result.extend(vlm.generate_dense_vector(batch_image_path_list, batch_text_prompt_list))
        else:
            result.extend(vlm.generate_text(batch_image_path_list, batch_text_prompt_list))
    return result
        

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
    
    
    # get video name from video_path
    video_name = CC_sequence["background"]["FileName"]
    output_dir = f"./frames/{video_name}"
    vid_frames = extract_frames_opencv(video_path, output_dir, interval=interval)
    
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
    
def store_data(frame_list,pinecone_index:pinecone,neo4j_driver:Driver,cc_list,vector_list,name_spaces=None):
    """
    Store data in Pinecone and Neo4j.
    each element in vector_list is a dictionary including data and vectors.
    fo each running, the function will get the data of one single vector.
    """
    result = 0
    # initiate the Neo4j session
    neo4j_session = neo4j_driver.session()
    last_node_id = None
    for index in range(len(frame_list)):
        # store the frame_path, timestamp (from frame_list), and cc_text, primary_tag (from cc_list) into neo4j
        data = {"frame_path": frame_list[index]["frame_path"], "timestamp": frame_list[index]["timestamp"], "cc_text": cc_list[index]["cc_text"], "primary_tag": cc_list[index]["primary_tag"]}
        current_node_id = create_neo4j_node(neo4j_session, data=data, last_node_id=last_node_id)
        # store the vector into Pinecone
        store_vector_in_pinecone(pinecone_index,vector_list[index], current_node_id,name_spaces)
        last_node_id = current_node_id  # update the last_node_id for the next iteration
        result += 1
    return result


def process_video(video_path,
                  CC_json_path, 
                  vlm_endpoint:VLM_EMB,
                  video_extract_interval=100,
                  pinecone_index:pinecone.Index=pinecone_index, 
                  neo4j_driver:Driver=neo4j_driver):
    """
    Process the video to extract frames, send them to VLM, 
    and store vectors in Pinecone, neo4j for graph construction
    """
    CC_sequent_raw = extract_cc(CC_json_path)
    print(f"[Finished]: extract_cc. Length of CC in CC_sequent_raw = {len(CC_sequent_raw['CC'])}")


    extracted_frames, CC_sequent = video_pair_generation(video_path= video_path, CC_sequence= CC_sequent_raw, interval=video_extract_interval)
    print(f"[Finished]: video_pair_generation. Length of extracted_frames = {len(extracted_frames)}")


    result = send_to_vlm(extracted_frames, CC_sequent, vlm_endpoint, CC_sequent_raw["background"], True)
    print(f"[Finished]: send_to_vlm. Length of result = {len(result)}")

    
    # Store data in Neo4j and Pinecone.
    video_name = CC_sequent_raw["background"]["FileName"]
    db_return_result = store_data(extracted_frames,pinecone_index,neo4j_driver,CC_sequent,result, name_spaces=video_name)
    print(f"[Finished]: store_data. Length of result = {db_return_result}")
