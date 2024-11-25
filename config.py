import yaml
from neo4j import GraphDatabase
import pinecone  # Import the Pinecone client

# Load configuration from config.yaml
config = {}
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(config)


# Neo4j Aura connection details (ensure these values are set in config.yaml)
neo4j_uri = config.get('neo4j_uri', 'bolt://localhost:7687')
neo4j_user = config.get('neo4j_user', 'neo4j')
neo4j_password = config.get('neo4j_password', 'password')
# Establish Neo4j connection
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


# Initialize Pinecone with your API key
PINECONE_API_KEY = config.get('pinecone_api_key',"")  # Replace with your Pinecone API key
pinecone.init(api_key=PINECONE_API_KEY)

# Create or connect to a Pinecone index
index_name = "video"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=512,  # Adjust dimension based on your vector size
        metric="cosine"  # Replace with your preferred metric, e.g., 'euclidean'
    )
pinecone_index = pinecone.Index(index_name)