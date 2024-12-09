from mian_pp import process_video
from retrival import retrieve_method
from vlm import VLM_EMB

# rewrite



def read_multi_video_into_database(video_paths):
    vlm = VLM_EMB()
    # for .... to process each video into the database
    process_video(video_path=video_path, CC_json_path=CC_json_path, vlm_endpoint=vlm)

def retrieve_cross_using_existing_frame():


def retrieve_cross_using_internet_frame(image_path,strength=1,  total_time=0):

    retrieve_method(image_path)
