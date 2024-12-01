### provide a img with user prompt.

# get relevant data from video

# top k 

from config import pinecone_client, neo4j_driver, pinecone_index
from vlm import VLM_EMB

my_vlm = VLM_EMB()

def input_perprocess(img_path='./frames/frame_1200s.jpg', prompt="what is this story about?"):
    res_vec = my_vlm.generate_dense_vector([img_path],[""])[0]
    res = pinecone_index.query(vector=res_vec,
                        top_k=10,
                        include_values=False)
    
    print(res)
    return

def retrieve_method():

    return

input_perprocess()