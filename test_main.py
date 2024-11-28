import os
# from config import neo4j_driver,pinecone_index

video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4")
CC_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.json")


# 实现 从 。/services 里调取 视频处理模块

from mian_pp import process_video

process_video(video_path,CC_json_path)