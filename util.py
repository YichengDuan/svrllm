import cv2
import os

def create_neo4j_node(session, data:dict , last_node_id=None):
    """
    :param session: Neo4j session
    :param data: the data dictionary
    :param last_node_id:    The id of the last node
    """
    # Unpacking the data
    frame_path, timestamp, cc_text, primary_tag = data['frame_path'], data['timestamp'], data['cc_text'], data['primary_tag']
    # create the current node
    query = """
        CREATE (v:data {
            frame_path: $frame_path,
            timestamp: $timestamp,
            cc_text: $cc_text,
            primary_tag: $primary_tag,
            node_id: randomUUID()  })
            RETURN v.node_id AS node_id 
             
        """
    result = session.run(query, frame_path=frame_path, timestamp=timestamp, cc_text=cc_text, primary_tag=primary_tag)
    current_node_id = result.single()["node_id"]
    if last_node_id is not None:
        query = """
            MATCH (v1:data {node_id: $id1}), (v2:data {node_id: $id2})
              CREATE (v1)-[:CONNECTED_TO]->(v2)
                """
        session.run(query, id1=last_node_id, id2=current_node_id)
    return current_node_id

def store_vector_in_pinecone(pinecone_index,vector, vector_id,name_space):
    """
    Store the vector in the Pinecone index.
    :param pinecone_index: The Pinecone index.
    :param vector: The vector to store.
    :param vector_id: Unique ID for the vector.
    :param name_space: The Pinecone name space.
    """
    # process the vector_id
    vector_id_str = str(vector_id)
    try:
        pinecone_index.upsert([(vector_id_str, vector)],namespace=name_space)
    except Exception as e:
        print(f"Error storing vector in Pinecone: {e}")


def extract_frame(video_path:str, time_sec:float, output_image_path:str):
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

def extract_frames_opencv(video_path:str, output_dir:str, interval=300):
    """
    Extract frames from a video every 'interval' seconds using OpenCV with frame seeking.
    
    :param video_path: Path to the input video file.
    :param output_dir: Directory to save extracted frames.
    :param interval: Time interval in seconds to extract frames (default is 300 seconds).
    :return: List of dictionaries containing frame paths and their corresponding timestamps.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path,cv2.CAP_ANY)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    extracted_frames = []
    current_time = 0.0

    while current_time < duration:
        # Set the position in milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame at {current_time} seconds not found.")
            current_time += interval
            continue

        frame_filename = os.path.join(output_dir, f'frame_{int(current_time)}s.jpg')
        cv2.imwrite(frame_filename, frame)
        extracted_frames.append({
            "frame_path": frame_filename,
            "timestamp": current_time
        })
        current_time += interval

    cap.release()
    return extracted_frames

def calculate_video_duration(video_path) -> float:
    """
    Calculate the duration of a video in seconds.

    Parameters:
        video_path (str): Path to the input video file.
    Returns:
        float: Duration of the video in seconds.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = float(total_frames / fps)
    cap.release()

    return duration


def cleanup_db() -> bool:
    """
    Cleans up all data from Neo4j and Pinecone databases.

    :return: True if cleanup is successful, False otherwise.
    """
    try:
        from config import neo4j_driver, pinecone_index

        # ----- Clean Pinecone Index -----
        # Deletes all vectors from the Pinecone index.
        print("Initiating Pinecone index cleanup...")
        pinecone_index.delete(delete_all=True)
        print("Pinecone index successfully cleaned.")

        # ----- Clean Neo4j Database -----
        # Deletes all nodes and relationships in the Neo4j graph database.
        print("Initiating Neo4j database cleanup...")
        with neo4j_driver.session() as session:
            # Cypher query to match and delete all nodes and relationships.
            cypher_query = "MATCH (n) DETACH DELETE n"
            session.run(cypher_query)
        print("Neo4j database successfully cleaned.")

        return True

    except Exception as e:
        # Logs the exception message for debugging purposes.
        print(f"Error during database cleanup: {e}")
        return False


def perdict_result(total_results:dict,k:int,true_time:float)-> bool:
    """
    Prediction: Check if the predicted range includes the true second
    :param total_results: dict
    :param k: int
    :param true_time: float

    :return: bool
    """
    predicted_start = total_results[k]['start']
    predicted_end = total_results[k]['end']
    prediction = predicted_start <= true_time <= predicted_end

    return prediction