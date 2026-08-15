import torch
import tqdm

DEFAULT_BATCH_SIZE = 32
from learn_nn.train_framework.dataset.get_tatoeba import (
    EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX,
    get_dataset,
    tokenize_source, tokenize_target,
    idx_list_to_token_source, idx_list_to_token_target,
)

# Todo: 减少padding的数量
class TensorHandlerBTC:
    
    @staticmethod
    def padding(idx_list_batch: list[list[int]]) -> list[list[int]]:
        # 找出最长的list
        max_len = max(len(idx) for idx in idx_list_batch)
        # 对每一个list进行填充
        padded: list[list[int]] = []
        # print(f"max_len is: {max_len}, arrlist is: {[len(idx) for idx in idx_list_batch]}")
        
        for idx_list in idx_list_batch:
            padding_size = max_len - len(idx_list)
            # if padding_size > 0:
                # 大部分情况应该不用padding，打印需要padding的场景
                # print(f"padding_size is: {padding_size}", idx_list)
                # pass
            idx_list += [PAD_IDX] * padding_size
            padded.append(idx_list)
        return padded
    
    @staticmethod
    def pairs_to_train_batches(
        pairs: list[tuple[str,str]],
        batch_size: int = DEFAULT_BATCH_SIZE
        # output shape is (src, tgt_input, tgt_output) as [B, T]
    ) -> list[tuple[torch.Tensor,torch.Tensor, torch.Tensor]]:

        # 1) str → idx
        tensor_pairs = []
        
        progress = tqdm.tqdm(pairs, desc="Processing pairs")
        
        src_batch = []
        tgt_input_batch = []
        tgt_output_batch = []

        for index, pair in enumerate(progress):
            s,t = pair
            src_batch.append([SOS_IDX]+tokenize_source(s)+[EOS_IDX])
            tgt_input_batch.append([SOS_IDX]+tokenize_target(t))
            tgt_output_batch.append(tokenize_target(t)+[EOS_IDX])
            i = index + 1
            if i%batch_size == 0 or i == len(pairs):
                src_batch = TensorHandlerBTC.padding(src_batch)
                tgt_input_batch = TensorHandlerBTC.padding(tgt_input_batch)
                tgt_output_batch = TensorHandlerBTC.padding(tgt_output_batch)

                tensor_pairs.append((
                    torch.tensor(src_batch),
                    torch.tensor(tgt_input_batch), 
                    torch.tensor(tgt_output_batch)
                ))
                src_batch = []
                tgt_input_batch = []
                tgt_output_batch = []
        return tensor_pairs

    @staticmethod
    def src_str_to_tensor(src: str) -> torch.Tensor:
        # 1) str → idx
        src_batch = [SOS_IDX]+tokenize_source(src)+[EOS_IDX]
        # 2) idx → tensor, 形状 [1, T] (BTC, B=1)
        src_tensor = torch.tensor(src_batch)
        return src_tensor.unsqueeze(0)
    
           
if __name__ == "__main__":
    
    pairs, test_pairs, src_vocab, tgt_vocab = get_dataset(min_len=0, max_len=10)
    epoch_batches = TensorHandlerBTC.pairs_to_train_batches(pairs, 5)
    for index, (src_tensor, tgt_input_tensor, tgt_output_tensor) in enumerate(epoch_batches):
        print(f"batch: {index}")
        print(f"src: {src_tensor.shape}, tgt_in: {tgt_input_tensor.shape}, tgt_out: {tgt_output_tensor.shape}")