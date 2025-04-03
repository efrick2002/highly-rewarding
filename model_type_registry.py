from transformers import (
    Qwen2PreTrainedModel,
    Qwen2Model,
    LlamaModel,
    LlamaPreTrainedModel,
    MistralModel,
    MistralPreTrainedModel
)

# Add models here for training support
MODEL_TYPE_REGISTRY = {
    "llama": (LlamaModel, LlamaPreTrainedModel),
    "qwen2": (Qwen2Model, Qwen2PreTrainedModel),
    "mistral": (MistralModel, MistralPreTrainedModel),
}