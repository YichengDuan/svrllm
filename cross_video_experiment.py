import os
import shutil

import numpy as np

from main_pp import process_video
from retrieval import retrieve_method
from util import extract_random_frame, get_video_path
from vlm import VLM_EMB

# rewrite



def read_multi_video_into_database(parent_dir = "data/more_videos"):
    # load the VLM model
    vlm = VLM_EMB()
    for folder_name in sorted(os.listdir(parent_dir), key=lambda x: int(x)):
        folder_path = os.path.join(parent_dir, folder_name)
        if os.path.isdir(folder_path):
            video_path = None
            cc_json_path = None

            for file in os.listdir(folder_path):
                if file.endswith(".mp4"):
                    video_path = os.path.join(folder_path, file)
                elif file.endswith(".json"):
                    cc_json_path = os.path.join(folder_path, file)

            # if found both video and JSON files, process the video and store in the database
            if video_path and cc_json_path:
                process_video(video_path=video_path, CC_json_path=cc_json_path, vlm_endpoint=vlm)
            else:
                print(f"Warning: Missing video or JSON file in folder {folder_name}!")


def retrieve_cross_using_existing_frame(search_range = 50):
    output_dir = "frames/targets/"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # extract a frame from random video in the search range
    videos_path = get_video_path("data/10_videos", search_range=search_range)
    match_count = 0
    match_count2 = 0
    match_count3 = 0
    total_count = 0
    mrr_list = []

    for video in videos_path:
        video_name = video["video_name"]
        video_path = video["video_path"]
        print(f"start to evaluate on {video_name}")
        frames = extract_random_frame(video_path = video_path, output_dir =output_dir,extract_num=1 )
        for frame in frames:
            total_count += 1
            ground_truth = {"video_name": video_name, "time": frame[0]}
            # retrieve and get the result
            prediction = retrieve_method(image_path=frame[1],top_k=5)
            # top 1
            prediction2 = prediction[:1]
            # top 3
            prediction3 = prediction[:3]
            # evaluate the result
            match, mrr = evaluate_single_frame(prediction, ground_truth)
            match2, _ = evaluate_single_frame(prediction2, ground_truth)
            match3, _ = evaluate_single_frame(prediction3, ground_truth)
            if match:
                match_count += 1
                mrr_list.append(mrr)
            if match2:
                match_count2 += 1
            if match3:
                match_count3 += 1
    # calculate the metrics
    mmr_array = np.array(mrr_list)
    total_mrr = mmr_array.mean()
    hit_rate = match_count / total_count # top 5
    hit_rate2 = match_count2 / total_count # top 1
    hit_rate3 = match_count3 / total_count # top 3
    print(f"Hit@1:{hit_rate2},Hit@3: {hit_rate3:.3f}, Hit@5:{hit_rate}, Mean MRR: {total_mrr:.3f}"  )
    return 1

def retrieve_cross_using_internet_frame(image_path,strength=1,  total_time=0):

    retrieve_method(image_path)
    return 1

def evaluate_single_frame(prediction, ground_truth, including_time = False):
    """
    return: Precision, Recall, F1 Score
    """
    gt_video = ground_truth["video_name"]
    gt_time = ground_truth["time"]

    matched = False
    rank = 0
    for n,pre in enumerate(prediction):
        rank = n + 1
        if including_time:
            if pre["name"] == gt_video and pre["start"] <= gt_time <= pre["end"]:
                matched = True
                break
        else:
            if pre["name"] == gt_video :
                matched = True
                break
    if matched:
        mrr = 1/rank
    else:
        mrr = 0
    return matched,mrr

retrieve_cross_using_existing_frame()