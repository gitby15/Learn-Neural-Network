import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_compile_cache"
os.makedirs("/tmp/torch_compile_cache", exist_ok=True)
from learn_nn.models.mini_llm.model_code.model import MiniLLM
from learn_nn.models.mini_llm.train.pretrain import PreTrainWorker
from learn_nn.models.mini_llm.dataset.minimind import get_dataset
import random
import torch
import torch.nn as nn
import numpy as np

_SEED_ = 88
random.seed(_SEED_)
torch.manual_seed(_SEED_)
torch.cuda.manual_seed_all(_SEED_)
np.random.seed(_SEED_)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_device(DEVICE)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
print(f"using device: {DEVICE}")

def main():
    batchs, tokenizer = get_dataset(100000, 128)
    model = MiniLLM(len(tokenizer))
    model = torch.compile(model)
    train_worker = PreTrainWorker(model, epochs=100)
    train_worker.train(batchs)
    input_str = ['今天天气', '番茄炒']

    result = train_worker.inference(input_str)
    print("result: ", result)



if __name__ == "__main__":
    main()
