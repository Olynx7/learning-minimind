import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class PretrainDataset(Dataset):
	def __init__(self, data_path, tokenizer, max_length=512) -> None:
		super().__init__()
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.samples = load_dataset('json', data_files=data_path, split='train')

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		sample = self.samples[index]

		# 构建输入文本
		encoding = self.tokenizer(
			str(sample['text']),
			max_length=self.max_length,
			padding='max_length',
			truncation=True,
			return_tensors='pt',
		)
		input_ids = encoding['input_ids'].squeeze()
		loss_mask = (input_ids != self.tokenizer.pad_token_id).long()

		X = torch.tensor(input_ids[:-1], dtype=torch.long)
		Y = torch.tensor(input_ids[1:], dtype=torch.long)
		loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

		return X, Y, loss_mask
