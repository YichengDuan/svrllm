
def create_neo4j_node(session, data , last_node_id=None):
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