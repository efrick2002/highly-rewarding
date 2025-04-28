from utils import get_registry_decorator, log_on_main
from abc import ABC, abstractmethod
from datasets import Dataset, load_dataset, load_from_disk
from typing import Dict
from transformers import PreTrainedTokenizer
import torch
import os

RANK = int(os.environ.get("RANK", -1))

class BaseDataset(ABC):
    def get_dataset(train_dataset_path: str, split="train", from_disk=False, shuffle=False, seed=42) -> Dataset:
        pass

class BaseDataCollator(ABC):

    def __init__(self, tokenizer, max_length):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = max_length
        self.first = True
    
    @abstractmethod
    def __call__(self, data):
        pass

REGISTERED_DATASET_CLASSES: Dict[str, BaseDataset] = {}
register_dataset = get_registry_decorator(REGISTERED_DATASET_CLASSES)

REGISTERED_DATASET_COLLATORS: Dict[str, BaseDataCollator] = {}
register_collator = get_registry_decorator(REGISTERED_DATASET_COLLATORS)

@register_dataset("singleturn-sft")
class SingleTurnSFTDataset(BaseDataset):
    def get_dataset(train_dataset_path: str, split="train", from_disk=False, shuffle=False, seed=42) -> Dataset:
        # Load formatted dataset from huggingface.
        
        if from_disk:
            train_dataset = load_from_disk(train_dataset_path)[split]
        else:
            train_dataset = load_dataset(train_dataset_path, split=split)
        
        assert all([
            "messages" in train_dataset.features,
        ]), f"""Missing dataset columns, {train_dataset.features} missing one or more required columns: "messages."""
        
        if shuffle:
            train_dataset = train_dataset.shuffle(seed=seed)
        
        return train_dataset.select_columns([
            "messages"
        ])

@register_dataset("pairwise-reward")
class PairwiseRewardDataset(BaseDataset):
    def get_dataset(train_dataset_path: str, split="train", from_disk=False, shuffle=False, seed=42) -> Dataset:
        # Load formatted dataset from huggingface.
        
        if from_disk:
            train_dataset = load_from_disk(train_dataset_path)[split]
        else:
            train_dataset = load_dataset(train_dataset_path, split=split)
        
        if shuffle:
            train_dataset = train_dataset.shuffle(seed=seed)

        assert all([
            "messages" in train_dataset.features,
            "labels" in train_dataset.features,
        ]), f"""Missing dataset columns, {train_dataset.features} missing one or more required columns: "messages, labels."""
        
        return train_dataset.select_columns([
            "messages", 
            "labels",
        ])
        
@register_collator("singleturn-sft")
class SingleTurnSFTDataCollator(BaseDataCollator):
    
    def __call__(self, data):
        
        prompts = []
        responses = []

        for row in data:
            messages = row['messages']
            prompts.append(messages[:-1])
            responses.append(messages)

        prompt_ids = self.tokenizer.apply_chat_template(prompts, add_generation_prompt=True)

        seq_ids = self.tokenizer.apply_chat_template(responses, add_generation_prompt=False)

        response_ids = []

        for seq, prompt_id in zip(seq_ids, prompt_ids):
            response_ids.append(
                seq[len(prompt_id):]
            )

        max_len = max([len(prompt_id) + len(response_id) for prompt_id, response_id in zip(prompt_ids, response_ids)])

        labels = []
        attention_mask = []
        input_ids = []

        for prompt_id, response_id in zip(prompt_ids, response_ids):

            prompt_length = len(prompt_id)
            response_length = len(response_id)
            padding_length = max_len - prompt_length - response_length

            padding = [self.tokenizer.pad_token_id]*padding_length

            input_ids.append(
                (prompt_id + response_id + padding)[:self.max_length]
            )
            labels.append(
                ([-100]*len(prompt_id) + response_id + [-100]*padding_length)[:self.max_length]
            )
            attention_mask.append(
                ([1]*(prompt_length + response_length) + [0]*padding_length)[:self.max_length]
            )

        return dict(
            input_ids=torch.tensor(input_ids),
            labels=torch.tensor(labels),
            attention_mask=torch.tensor(attention_mask),
        )

@register_collator("pairwise-reward")
class PairwiseRewardDataCollator(BaseDataCollator):
    def __call__(self, data):
        
        messages = [message for row in data for message in row['messages']] # Unroll pairs.
        labels = [row['labels'] for row in data]

        # Format messages with chat template. Do not tokenize.
        formatted_messages = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        # We want to remove the cls token if it happens to be in the formatted messages.
        formatted_messages = [msg.replace(self.tokenizer.cls_token, "<cls>") for msg in formatted_messages]

        # Now formatted_messages is a list of strings
        # Now we need to add the cls token to the end of each string
        formatted_messages = [msg + self.tokenizer.cls_token for msg in formatted_messages]
        
        if self.first:
            print(f"Rank {RANK}: First batch of data:")
            print(f"Rank {RANK}: Messages: {messages}")
            print(f"Rank {RANK}: Formatted messages: {formatted_messages}")
            print(f"Rank {RANK}: Labels: {labels}")

            self.first = False

        # Tokenize the formatted messages
        tokenized_messages = self.tokenizer(
            formatted_messages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        return dict(
            input_ids=tokenized_messages['input_ids'],
            attention_mask=tokenized_messages['attention_mask'],
            labels=torch.vstack(labels),
        )
        
@register_collator("double-cls-pairwise-reward")
class DoubleClsPairwiseRewardDataCollator(BaseDataCollator):
    def __call__(self, data):
        
        messages = [message for row in data for message in row['messages']] # Unroll pairs.
        labels = [row['labels'] for row in data]

        # Format messages with chat template. Do not tokenize.
        formatted_messages = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        # We want to remove the cls token if it happens to be in the formatted messages.
        formatted_messages = [msg.replace(self.tokenizer.cls_mean, "<cls_mean>").replace(self.tokenizer.cls_logvar, "<cls_logvar>") for msg in formatted_messages]

        # Now formatted_messages is a list of strings
        # Now we need to add the cls token to the end of each string
        formatted_messages = [msg + self.tokenizer.cls_mean +  self.tokenizer.cls_logvar for msg in formatted_messages]
        
        if self.first:
            print(f"Rank {RANK}: First batch of data:")
            print(f"Rank {RANK}: Messages: {messages}")
            print(f"Rank {RANK}: Formatted messages: {formatted_messages}")
            print(f"Rank {RANK}: Labels: {labels}")

            self.first = False

        # Tokenize the formatted messages
        tokenized_messages = self.tokenizer(
            formatted_messages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        return dict(
            input_ids=tokenized_messages['input_ids'],
            attention_mask=tokenized_messages['attention_mask'],
            labels=torch.vstack(labels),
        )
        
        
        
        

        