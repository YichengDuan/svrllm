### provide a img with user prompt.

# get relevant data from video

# top k 
import cv2
from config import pinecone_client, neo4j_driver, pinecone_index
from vlm import VLM_EMB

my_vlm = VLM_EMB()

def input_preprocess(img_path='./frames/frame_1200s.jpg', prompt="what is this story about?", strength = 5, top_k = 3):
    """
    :param img_path: path to the image
    :param prompt: user prompt for image description
    :param strength: number of previous and next frames to consider
    :param top_k: number of top relevant vectors to consider
    :return: list of top relevant vectors and their corresponding data from neo4j
             the format: a list of k_top parts(dict) containing three dictionaries:
             self(the target result itself), previous(pre n_th node), and next(next n_th node) where n_th is defined by strength
    """
    res_vec = my_vlm.generate_dense_vector([img_path],[""])[0]
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
        query = f"""
                    MATCH (v:data)
                    WHERE v.node_id = $node_id
                    RETURN 'self' AS direction, v AS node
                
                    UNION
                
                    MATCH (v:data)<-[:CONNECTED_TO*1..{strength}]-(n:data)
                    WHERE v.node_id = $node_id
                    RETURN 'previous' AS direction, n AS node
                    ORDER BY n.timestamp DESC
                    LIMIT {strength}
                
                    UNION
                
                    MATCH (v:data)-[:CONNECTED_TO*1..{strength}]->(m:data)
                    WHERE v.node_id = $node_id
                    RETURN 'next' AS direction, m AS node
                    ORDER BY m.timestamp ASC
                    LIMIT {strength};
                 """
        result = neo4j_session.run(query, node_id=vec_id).data()
        results.append(result)
    return results

print(input_preprocess())

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
    vidcap = cv2.VideoCapture(video_path)
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
    print(f"Frame saved at {output_image_path}")

    # Release the video capture object
    vidcap.release()
    return True

def retrieve_method(target_sec=1200.0):
    output_image_path = "./frames/target_img"
    extract_frame(video_path="",time_sec=target_sec,output_image_path=output_image_path)
    result = input_preprocess(img_path=output_image_path)
    return