import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from modelscope.hub.file_download import dataset_file_download
from learn_nn.models.mini_llm.dataset.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_nn.models.mini_llm.dataset.tensor_handler import TensorHandler
MAX_LENGTH = 512

def _download_data():
    file_path = dataset_file_download(
        dataset_id='gongjy/minimind_dataset',
        file_path='pretrain_t2t_mini.jsonl'
    )
    print(f"预训练文件保存路径: {file_path}")
    return file_path



class PretrainDataset(Dataset):
    def __init__(self, tokenizer, max_length=MAX_LENGTH):
        super().__init__()
        data_path = _download_data()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')
    def __len__(self):
        return len(self.samples)

    def __getitem_raw__(self, index):
        sample = self.samples[index]
        text = str(sample.get('text', sample.get('content', '')))
        return text

    def _fill_special_token(self, token_list):
        return [self.tokenizer.bos_token_id] + token_list + [self.tokenizer.eos_token_id]
    
    def __getitem__(self, index):
        text = self.__getitem_raw__(index)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        )
        tokens = encoded["input_ids"]
        input_ids = self._fill_special_token(tokens)
        
        # Todo: 为什么输出的pad要是-100？
        labels = [-100 if x == self.tokenizer.pad_token_id else x for x in input_ids]
        
        return input_ids, labels

    def input_str_to_batch(self, input_str_list: list[list[str]]):
        input_idx_list = [self.tokenizer(input_str)['input_ids'] for input_str in input_str_list]
        input_idx_list = TensorHandler.align_batch_idx(input_idx_list, self.tokenizer.pad_token_id)
        input_idx_batch_tensor = torch.tensor(input_idx_list, dtype=torch.long)

        return input_idx_batch_tensor

    def get_train_batchs(self,row_count:int, batch_size: int):

        # 返回的input_idx和label都是[B, T结构]
        batchs = []
        input_batch = []
        label_batch = []
        for i in range(row_count):
            idx_pair = self.__getitem__(i)
            input_batch.append(idx_pair[0])
            label_batch.append(idx_pair[1])
            count = i+1
            if count%batch_size==0 or count == row_count:
                padded_input_batch = TensorHandler.align_batch_idx(input_batch, self.tokenizer.pad_token_id)
                padded_label_batch = TensorHandler.align_batch_idx(label_batch, -100)

                batchs.append((torch.tensor(padded_input_batch, dtype=torch.long), torch.tensor(padded_label_batch, dtype=torch.long)))
                input_batch = []
                label_batch = []
            
        return batchs


def get_dataset(data_count:int=1000, batch_size=32):
    tokenizer_factory = MinimindTokenizer()
    tokenizer = tokenizer_factory.get_tokenizer()
    ds = PretrainDataset(tokenizer=tokenizer)
    batchs = ds.get_train_batchs(data_count, batch_size)
    return batchs, tokenizer

if __name__ == "__main__":
    tokenizer_factory = MinimindTokenizer()
    tokenizer = tokenizer_factory.get_tokenizer()
    dataset = PretrainDataset(tokenizer=tokenizer)
    data = dataset.__getitem__(1);
    print(f"len: ", dataset.__len__())