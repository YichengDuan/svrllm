### provide a img with user prompt.

# get relevant data from video

# top k 

from config import pinecone_client, neo4j_driver, pinecone_index
from vlm import VLM_EMB

my_vlm = VLM_EMB()

def input_preprocess(img_path='./frames/frame_1200s.jpg', prompt="what is this story about?", strength = 2, top_k = 3):
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

def retrieve_method():

    return

input_preprocess()