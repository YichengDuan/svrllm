import cv2
from config import pinecone_client, neo4j_driver, pinecone_index
from vlm import VLM_EMB
import os
from tqdm import tqdm
from mian_pp import process_video
from retrival import retrieve_method
from util import extract_frame,calculate_video_duration,extract_frames_opencv, perdict_result, cleanup_db

import os
from vlm import VLM_EMB
# from config import neo4j_driver,pinecone_index

if __name__ == "__main__":
    local_path = os.path.dirname(os.path.abspath(__file__))

    video_path = os.path.join(local_path,"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4")
    CC_json_path = os.path.join(local_path,"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.json")

    extract_interval = [2,5,10,20,50,100]
    method_list = ["simple_mean", "max_pool", "mean_pool", "concat_layers"]
    strength_list = [1,2,3]
    top_k_list = [1,2,3]

    vlm = VLM_EMB()
    

    
    duration = calculate_video_duration(video_path)


    tmp_frame_path = os.path.join(local_path,"frames/tmp/")



    process_video(video_path=video_path, CC_json_path=CC_json_path,vlm_endpoint=vlm)

    test_frames = extract_frames_opencv(video_path=video_path,output_dir=tmp_frame_path,interval=300)

    total_results = retrieve_method(image_path = video_path,strength=1,top_k=1,total_time=duration,summary_output=False)

   

    def sv_test(extract_interval,method,strength):
        return
