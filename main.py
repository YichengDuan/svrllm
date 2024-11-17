from flask import Flask, request, jsonify, send_file
import os
import yaml
from neo4j import GraphDatabase

app = Flask(__name__)

# Load configuration from config.yaml
config = {}
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(config)

# Directory to store uploaded and processed files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Neo4j Aura connection details (ensure these values are set in config.yaml)
neo4j_uri = config.get('neo4j_uri', 'bolt://localhost:7687')
neo4j_user = config.get('neo4j_user', 'neo4j')
neo4j_password = config.get('neo4j_password', 'password')

# Establish Neo4j connection
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    # Placeholder for video processing and further pipeline
    result = process_video(video_path)
    return jsonify({"message": "Video processed", "result": result})

def process_video(video_path):
    # Placeholder function for video pre-processing
    print(f"Processing video at: {video_path}")
    
    # Simulated VLM and VectorDB interaction
    processed_result = {
        "processed_video_path": video_path,
        "retrieved_data": retrieve_related_content(video_path)
    }

    # Insert processed video information into Neo4j
    insert_data_into_neo4j(video_path, processed_result["retrieved_data"])

    return processed_result

def retrieve_related_content(video_path):
    # Placeholder function to simulate retrieval logic
    print(f"Retrieving content for video: {video_path}")
    # Return mock data as a placeholder
    return {"related_videos": ["video1.mp4", "video2.mp4"]}

def insert_data_into_neo4j(video_path, related_data):
    """
    Inserts video information and related content into Neo4j.
    """
    with neo4j_driver.session() as session:
        session.write_transaction(create_video_node, video_path, related_data)

def create_video_node(tx, video_path, related_data):
    """
    Creates a node for the video and its related data in Neo4j.
    """
    query = (
        "MERGE (v:Video {path: $video_path}) "
        "SET v.timestamp = timestamp() "
        "WITH v "
        "UNWIND $related_videos AS related "
        "MERGE (r:Video {path: related}) "
        "MERGE (v)-[:RELATED_TO]->(r)"
    )
    tx.run(query, video_path=video_path, related_videos=related_data.get('related_videos', []))

@app.route('/read_videos', methods=['GET'])
def read_videos():
    """
    Read video nodes and their relationships from Neo4j.
    """
    with neo4j_driver.session() as session:
        result = session.read_transaction(get_video_nodes)
        return jsonify(result)

def get_video_nodes(tx):
    """
    Retrieves video nodes and their related content.
    """
    query = (
        "MATCH (v:Video)-[:RELATED_TO]->(r:Video) "
        "RETURN v.path AS video_path, collect(r.path) AS related_videos"
    )
    results = tx.run(query)
    return [{"video_path": record["video_path"], "related_videos": record["related_videos"]} for record in results]

@app.route('/generate_summary', methods=['GET'])
def generate_summary():
    # Simulated summary generation
    summary = "This is a summary of the retrieved content."
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)