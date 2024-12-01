### provide a img with user prompt.

# get relevant data from video

# top k 

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

def retrieve_method():

    return

input_preprocess()