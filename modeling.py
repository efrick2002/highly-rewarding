from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
from transformers.utils import ModelOutput
from dataclasses import dataclass
from model_type_registry import MODEL_TYPE_REGISTRY
from torch import nn
import torch
from typing import Dict, Callable, List
from utils import get_registry_decorator, log_on_main
import os

RANK = int(os.environ.get("RANK", -1))

REGISTERED_INITS: Dict[str, Callable] = {}
REGISTERED_MODEL_CLASSES: Dict[str, Callable] = {}

register_init = get_registry_decorator(REGISTERED_INITS)
register_model_class = get_registry_decorator(REGISTERED_MODEL_CLASSES)

@register_init("reset_params")
def reset_params_init(module):
    return module.reset_parameters()


@register_init("he_unif")
def he_unif_init(module):
    return nn.init.kaiming_uniform_(module.weight, nonlinearity="sigmoid")


@register_init("xavier_unif")
def xavier_unif_init(module):
    return nn.init.xavier_uniform_(module.weight)


@register_init("tiny_normal")
def tiny_normal_init(module):
    return nn.init.kaiming_normal_(module.weight)

@dataclass
class BTRewardOutputs(ModelOutput):
    rewards: torch.FloatTensor = None

class ThurstoneRewardOutputs(ModelOutput):
    means: torch.FloatTensor = None
    logvars: torch.FloatTensor = None

def get_model_tokenizer(base_model_name: str, pad_token_if_none: str | None, chat_template: str | None, new_special_tokens: List[str], cls_token: str | None, truncation_side: str = "left") -> PreTrainedTokenizer:

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token == None:

        assert pad_token_if_none != None, "Set a pad token, since tokenizer doesn't have one."

        tokenizer.add_special_tokens({"pad_token": pad_token_if_none})

        log_on_main(f"Pad token set to: {pad_token_if_none}")

    if cls_token != None:

        if tokenizer.cls_token == None:

            assert tokenizer.cls_token != "<cls>", "Cannot set CLS token to <cls>, which is the placeholder cls token."

            tokenizer.add_special_tokens({"cls_token": cls_token})

            log_on_main(f"CLS token set to: {tokenizer.cls_token}")
        
        else:

            log_on_main(f"WARNING: CLS token already set to: {tokenizer.cls_token}, ignoring cls_token argument.")
    
    if chat_template != None:

        tokenizer.chat_template = chat_template

    if new_special_tokens:

        tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

        log_on_main(f"Added {len(new_special_tokens)} new special tokens: {new_special_tokens}")

    tokenizer.truncation_side = truncation_side

    return tokenizer

@register_model_class("bt-reward-model")
def get_bt_reward_model_class(model_type: str, tokenizer: PreTrainedTokenizer, init_type: str = "reset_params") -> PreTrainedModel:
    # Should construct and return the model class such that the trainer can call .from_pretrained on it.
    
    transformer_model_cls, pretrained_model_cls = MODEL_TYPE_REGISTRY[model_type]

    init_func = REGISTERED_INITS[init_type]

    cls_token = tokenizer.cls_token_id

    class RewardPretrainedModel(pretrained_model_cls):

        def _init_weights(self, module):
            std = self.config.initializer_range
            if isinstance(module, nn.Linear):
                init_func(module)  # was reset params
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    class RewardModel(RewardPretrainedModel):

        def __init__(
                self,
                config,
                **kwargs,
        ):
            super().__init__(config)

            self.model = transformer_model_cls(config)

            self.head = nn.Linear(
                in_features=config.hidden_size,
                out_features=1,
            )

            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def forward(self, input_ids, attention_mask):

            hidden_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            ).last_hidden_state  # (bs, num_token, embed_dim)

            print(f"Rank {RANK}: hidden_outputs shape: {hidden_outputs.shape}")

            print(f"Rank {RANK}: input_ids: {input_ids}")

            print(f"Rank {RANK}: cls_token: {cls_token}")

            print(f"Rank {RANK}: DEBUG: {input_ids == cls_token}")

            cls_mask = input_ids == cls_token

            print(f"Rank {RANK}: cls_mask shape: {cls_mask.shape}")

            cls_hidden_dim = hidden_outputs[cls_mask]

            print(f"Rank {RANK}: cls_hidden_dim shape: {cls_hidden_dim.shape}")

            # assert cls_hidden_dim.shape[0] == input_ids.shape[0], f"CLS hidden dim shape: {cls_hidden_dim.shape}, input_ids shape: {input_ids.shape}"

            rewards = self.head(cls_hidden_dim) # (bs, 1)

            print(f"Rank {RANK}: rewards shape: {rewards.shape}")

            
            # The pairwise rewards are flattened, so we need to unflatten them. For now, we will assume it is always pairwise.
            rewards = rewards.view(-1, 2)
            
            assert rewards.shape[0]*2 == input_ids.shape[0] and rewards.shape[1] == 2, f"Rewards shape: {rewards.shape}, input_ids shape: {input_ids.shape}"

            return BTRewardOutputs(
                rewards=rewards
            )

    return RewardModel


@register_model_class("thurstone-reward-model")
def get_thurstone_reward_model_class(model_type: str, tokenizer: PreTrainedTokenizer, init_type: str = "reset_params") -> PreTrainedModel:
    # Should construct and return the model class such that the trainer can call .from_pretrained on it.
    
    transformer_model_cls, pretrained_model_cls = MODEL_TYPE_REGISTRY[model_type]

    init_func = REGISTERED_INITS[init_type]

    class RewardPretrainedModel(pretrained_model_cls):

        def _init_weights(self, module):
            std = self.config.initializer_range
            if isinstance(module, nn.Linear):
                init_func(module)  # was reset params
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    class RewardModel(RewardPretrainedModel):

        def __init__(
                self,
                config,
                **kwargs,
        ):
            super().__init__(config)

            self.model = transformer_model_cls(config)

            self.head = nn.Linear(
                in_features=config.hidden_size,
                out_features=2,
            )

            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def forward(self, input_ids, attention_mask):

            hidden_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            ).last_hidden_state  # (bs, num_token, embed_dim)

            head_output = self.head(hidden_outputs) # (bs, num_token, 2)

            cls_token = tokenizer.cls_token_id

            # Since we know there is exactly one CLS token per sequence, we can use argmax
            cls_token_indices = (input_ids == cls_token).argmax(dim=1)
            means_and_logvars = head_output[torch.arange(head_output.shape[0]), cls_token_indices]

            # The pairwise rewards are flattened, so we need to unflatten them. For now, we will assume it is always pairwise.
            means = means_and_logvars[:, 0]
            logvars = means_and_logvars[:, 1]
            
            # The pairwise rewards are flattened, so we need to unflatten them. For now, we will assume it is always pairwise.
            means = means.view(-1, 2)
            logvars = logvars.view(-1, 2)
            
            assert means.shape[0]*2 == input_ids.shape[0] and means.shape[1] == 2, f"Means shape: {means.shape}, input_ids shape: {input_ids.shape}"
            assert logvars.shape == means.shape, f"Logvars shape: {logvars.shape}, means shape: {means.shape}"

            return ThurstoneRewardOutputs(
                means=means,
                logvars=logvars
            )

    return RewardModel