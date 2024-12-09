### provide a img with user prompt.

# get relevant data from video

# top k 
import cv2
from config import pinecone_client, neo4j_driver, pinecone_index
from vlm import VLM_EMB
import os
from tqdm import tqdm

VLM_BACKEND = VLM_EMB()

def input_preprocess(img_path='./frames/frame_1200s.jpg', prompt="", strength = 1, top_k = 3):
    """
    :param img_path: path to the image
    :param prompt: user prompt for image description
    :param strength: number of previous and next frames to consider
    :param top_k: number of top relevant vectors to consider
    :return: list of top relevant vectors and their corresponding data from neo4j
             the format: a list of k_top parts(dict) containing three dictionaries:
             self(the target result itself), previous(pre n_th node), and next(next n_th node) where n_th is defined by strength
    """
    res_vec = VLM_BACKEND.generate_dense_vector([img_path],[prompt])[0]
    # get all the namespaces in pinecone index
    stats = pinecone_index.describe_index_stats()
    namespaces = list(stats["namespaces"].keys())
    # get top 10 vectors from pinecone
    # REMINDER: TO RUN THIS METHOD, YOU NEED TO DOWNLOAD THE LATEST PINECONE SDK BY USING THE COMMAND: pip install "pinecone[grpc]"
    res = pinecone_index.query_namespaces(vector=res_vec,top_k=top_k,include_values=False,namespaces=namespaces)
    # use the id to find the corresponding data from neo4j
    results = []
    neo4j_session = neo4j_driver.session()
    for match in res["matches"]:
        vec_id = match["id"]
        # # find the relevant data for each result, extend the result
        query1 = f"""
                    MATCH (v:data)
                    WHERE v.node_id = $node_id
                    RETURN v AS node
                 """
        self_result = neo4j_session.run(query1, node_id=vec_id).data()
        query2 = f"""        
                    MATCH (v:data)<-[:CONNECTED_TO*1..{strength}]-(n:data)
                    WHERE v.node_id = $node_id
                    RETURN n AS node
                    ORDER BY n.timestamp DESC
                    LIMIT {strength}
                """
        pre_result = neo4j_session.run(query2, node_id=vec_id).data()
        pre_result = [item["node"] for item in pre_result]
        query3 = f"""  
                MATCH (v:data)-[:CONNECTED_TO*1..{strength}]->(m:data)
                WHERE v.node_id = $node_id
                RETURN m AS node
                ORDER BY m.timestamp ASC
                LIMIT {strength};
             """
        next_result = neo4j_session.run(query3, node_id=vec_id).data()
        next_result = [item["node"] for item in next_result]
        result = {"self": self_result[0]["node"], "previous": pre_result, "next": next_result,"namespace":match["namespace"] }
        results.append(result)
    return results

def extract_frame(video_path, time_sec, output_image_path):
    """
    Extracts a frame from the video at the specified time and saves it as an image.

    Parameters:
        video_path (str): Path to the input video file.
        time_sec (float): Time in seconds at which to extract the frame.
        output_image_path (str): Path to save the extracted image.

    Returns:
        bool: True if frame extraction was successful, False otherwise.
    """
    # Open the video file
    vidcap = cv2.VideoCapture(video_path,cv2.CAP_AVFOUNDATION)
    if not vidcap.isOpened():
        print("Error: Cannot open video file.")
        return False

    # Get frames per second (fps) to calculate frame number
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Cannot retrieve FPS from video.")
        vidcap.release()
        return False

    # Calculate frame number corresponding to the given time
    frame_number = int(fps * time_sec)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number >= total_frames:
        print("Error: Time exceeds video duration.")
        vidcap.release()
        return False

    # Set the video position to the desired frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, image = vidcap.read()
    if not success:
        print("Error: Cannot read frame.")
        vidcap.release()
        return False

    # Save the frame as an image file
    cv2.imwrite(output_image_path, image)
    # Release the video capture object
    vidcap.release()
    return True

def retrieve_method(video_path,strength=1,target_sec=1250.0,total_time=3600.0):
    output_image_path = "./frames/target_img.jpg"

    extract_frame(video_path=video_path,time_sec=target_sec,output_image_path=output_image_path)
    result = input_preprocess(img_path=output_image_path,strength=strength)
    # print(result)
    total_results = []
    for _hit in result:
        if len(_hit['next']) == 0:
            tmp_end = total_time
        else:
            tmp_end = _hit['next'][-1]['timestamp']
        if len(_hit['previous']) == 0:
            tmp_start = 0
        else:
            tmp_start = _hit['previous'][-1]['timestamp']
        video_name = _hit['namespace']
        total_results.append({'start':tmp_start, 'end':tmp_end, 'name':video_name})
    return total_results

if __name__ == "__main__":
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4")
    print("test on 1250s")
    
    cap = cv2.VideoCapture(video_path,cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    total_results = retrieve_method(video_path = video_path,total_time=duration)
    print(total_results)
    print("="*100)
    print(f"Best range is at start: {total_results[0]['start']}s, end {total_results[0]['end']}s, in video name: {total_results[0]['name']}")

    test_range = range(0,3500,50)

    for _S in range(1,4):
         # Initialize counters
        correct = 0
        total_count = 0

        TP = 0  # True Positives
    
        FN = 0  # False Negatives
        # Define a tolerance 
        strength = 1
        for _sec in tqdm(test_range, desc=f"testing strength={_S}"):
            if _sec % 100 == 0:
                continue
                
            total_results = retrieve_method(video_path=video_path,strength=_S, target_sec=float(_sec),total_time=duration)
            
            # Prediction: Check if the predicted range includes the target second
            predicted_start = total_results[0]['start']
            predicted_end = total_results[0]['end']
            prediction = predicted_start <= _sec <= predicted_end

            # Ground Truth
            event_occurs = True

            # Update counts based on prediction and ground truth
            if prediction:
                if event_occurs:
                    TP += 1  # True Positive: Correctly predicted event occurrence
            else:
                if event_occurs:
                    FN += 1  # False Negative: Missed actual event occurrence

            total_count += 1
    
        recall = TP / total_count if total_count else 0       # True Positive Rate
        print(f"Recall strength={_S} at (True Positive Rate): {round(recall * 100, 2)}%")


    
