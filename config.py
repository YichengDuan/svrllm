import yaml
from neo4j import GraphDatabase
from pinecone import Pinecone, ServerlessSpec
import platform
# Load configuration from config.yaml
config = {}
try:
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError as e:
    print(f"Error, No config file: {e}")
    exit(1)

print(f"loaded config: {str(config)}")


# Neo4j Aura connection details (ensure these values are set in config.yaml)
neo4j_uri = config.get('neo4j_uri', 'bolt://localhost:7687')
neo4j_user = config.get('neo4j_user', 'neo4j')
neo4j_password = config.get('neo4j_password', 'password')
# Establish Neo4j connection
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
    driver.verify_connectivity()

pinecone_client = Pinecone(
    api_key=config.get('pinecone_api_key',"")
)

if 'video' not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name='video', 
        dimension=1536, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
pinecone_index = pinecone_client.Index("video")

def select_video_backend():
    os_type = platform.system()
    if os_type == 'Darwin':
        # macOS
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_ANY  # Alternatively, CAP_FFMPEG

    return backend

CV2_BACKEND = select_video_backend()