

from learn_nn.models.mini_llm.model_code.model import MiniLLM
from learn_nn.models.mini_llm.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_nn.models.mini_llm.train.pretrain import PreTrainWorker
from learn_nn.models.mini_llm.dataset.minimind import get_dataset

def main():
    batchs, tokenizer = get_dataset(100000, 128)
    model = MiniLLM(len(tokenizer))
    train_worker = PreTrainWorker(model)
    train_worker.train(batchs)
    


if __name__ == "__main__":
    main()