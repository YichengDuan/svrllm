from flask import Flask, request, jsonify, send_file
import os

from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import bcrypt

app = Flask(__name__)



# JWT Configuration
app.config['JWT_SECRET_KEY'] = config.get('jwt_secret_key', 'your_jwt_secret')  # Set your JWT secret key
jwt = JWTManager(app)

# Directory to store uploaded and processed files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/register', methods=['POST'])
def register():
    """
    Register a new user.
    """
    username = request.json.get('username')
    password = request.json.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    # Check if the user already exists
    if mongo_users_collection.find_one({"username": username}):
        return jsonify({"error": "User already exists"}), 409

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Store the user in MongoDB
    mongo_users_collection.insert_one({
        "username": username,
        "password": hashed_password
    })

    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    """
    Login an existing user and return a JWT token.
    """
    username = request.json.get('username')
    password = request.json.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    # Find the user in MongoDB
    user = mongo_users_collection.find_one({"username": username})
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({"error": "Invalid username or password"}), 401

    # Create a JWT token
    access_token = create_access_token(identity=username)
    return jsonify({"access_token": access_token}), 200

@app.route('/upload_video', methods=['POST'])
@jwt_required()
def upload_video():
    """
    Upload a video (requires authentication).
    """
    current_user = get_jwt_identity()  # Get the current user from the token
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

    # Insert processed data into MongoDB
    insert_data_into_mongodb(video_path, processed_result["retrieved_data"])

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
        session.execute_write(create_video_node, video_path, related_data)

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

def insert_data_into_mongodb(video_path, related_data):
    """
    Inserts video information and related content into MongoDB.
    """
    document = {
        "video_path": video_path,
        "related_videos": related_data.get('related_videos', [])
    }
    result = mongo_collection.insert_one(document)
    print(f"Inserted document with ID: {result.inserted_id}")

@app.route('/read_videos', methods=['GET'])
@jwt_required()
def read_videos():
    """
    Read video nodes and their relationships from Neo4j (requires authentication).
    """
    with neo4j_driver.session() as session:
        result = session.execute_write(get_video_nodes)
        return jsonify(result)

@app.route('/read_mongodb', methods=['GET'])
@jwt_required()
def read_mongodb():
    """
    Read video data from MongoDB (requires authentication).
    """
    documents = list(mongo_collection.find({}, {'_id': 0}))  # Exclude MongoDB ObjectID from the response
    return jsonify(documents)

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
@jwt_required()
def generate_summary():
    # Simulated summary generation (requires authentication)

    # 拿到相似度，通过图去往下找，知道节点相似度低于一个值， 

    summary = "This is a summary of the retrieved content."
    return jsonify({"summary": summary})






if __name__ == '__main__':
    app.run(debug=True)