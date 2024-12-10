import os
from vlm import VLM_EMB
# from config import neo4j_driver,pinecone_index

video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4")
CC_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.json")


from main_pp import process_video


vlm = VLM_EMB()
process_video(video_path=video_path, CC_json_path=CC_json_path,vlm_endpoint=vlm)