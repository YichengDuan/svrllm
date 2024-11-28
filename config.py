import yaml
from neo4j import GraphDatabase
from pinecone import Pinecone, ServerlessSpec
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
with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
    driver.verify_connectivity()


pinecone_client = Pinecone(
    api_key=config.get('pinecone_api_key',"")
)

# usage Now do stuff
# if 'video' not in pc.list_indexes().names():
#     pc.create_index(
#         name='video', 
#         dimension=1536, 
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-west-2'
#         )
#     )
