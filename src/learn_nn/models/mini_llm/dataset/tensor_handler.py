class TensorHandler:
    @staticmethod
    def padding(idx_list_batch: list[list[int]], pad_idx:int) -> list[list[int]]:
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
            idx_list += [pad_idx] * padding_size
            padded.append(idx_list)
        return padded