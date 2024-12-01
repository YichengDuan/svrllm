### provide a img with user prompt.

# get relevant data from video

# top k 

from config import pinecone_client, neo4j_driver, pinecone_index
from vlm import VLM_EMB

my_vlm = VLM_EMB()

def input_preprocess(img_path='./frames/frame_1200s.jpg', prompt="what is this story about?", strength = 5, top_k = 3):
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
        query1 = """
                    MATCH (v:data)
                    WHERE v.node_id = $node_id
                    RETURN v;
                 """
        target_result = neo4j_session.run(query1, node_id=vec_id).data()
        # find the relevant data for each result, extend the result
        query2 = """
                    MATCH (v:data)
                    WHERE v.node_id = $node_id
                    WITH v 
                    
                    MATCH (v)-[:CONNECTED_TO]->(n:data)  
                    RETURN 'previous' AS direction, n AS node
                    ORDER BY n.timestamp DESC 
                    LIMIT $strength
                    
                    UNION
                    
                    MATCH (v)<-[:CONNECTED_TO]-(m:data)  
                    RETURN 'next' AS direction, m AS node
                    ORDER BY m.timestamp ASC 
                    LIMIT $strength;
                 """
        extend_result = neo4j_session.run(query2, node_id=vec_id,strength = strength).data()
        result = {"target_result": target_result, "extend_result": extend_result}
        results.append(result)
    return results

def retrieve_method():

    return

input_preprocess()