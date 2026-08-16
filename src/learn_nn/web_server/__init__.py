import os
import torch
from bottle import route, run, response, static_file
from learn_nn.models.mini_llm.base_component import Embedding, PosEmbedding

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@route("/api/cross_product_pos_embedding", method="GET")
def cross_product_pos_embedding():
    src_vocab = [0,1,2,3,4,5]
    zero_embedding = Embedding(src_vocab)
    zero_embedding._test_fill_weights(0)
    embedding = Embedding(src_vocab)
    
    # 输入[B, T], 输出[B,T,C]
    embedding_input = torch.ones(1,500,dtype=torch.long)
    embedding_output = embedding(embedding_input)
    zero_embedding_output = zero_embedding(embedding_input)

    print(f"embedding_output size: {embedding_output.shape}")

    pos_embedding = PosEmbedding()
    pos_zero_output = pos_embedding(zero_embedding_output)
    pos_embedding_output = pos_embedding(embedding_output)

    print(f"pos_embedding_output size: {pos_embedding_output.shape}")    
    embed = embedding_output.detach().tolist()
    pos_zero = pos_zero_output.detach().tolist()
    pos_embed = pos_embedding_output.detach().tolist()
    
    return {"embed": embed, "pos_zero": pos_zero, "pos_embed": pos_embed}


@route("/static/<filepath:path>")
def serve_static(filepath):
    return static_file(filepath, root=STATIC_DIR)

@route("/")
def index():
    return static_file("index.html", root=STATIC_DIR)


def main():
    run(host="localhost", port=3003)


if __name__ == '__main__':
    main()