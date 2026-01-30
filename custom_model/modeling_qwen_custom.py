import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

# If your environment provides Qwen3 specific classes, use them here:
try:
    from transformers import Qwen3ForCausalLM, Qwen3Model
except ImportError:
    # Fallback to Qwen2 if Qwen3 is not explicitly in the library but used as a type
    Qwen3ForCausalLM = Qwen2ForCausalLM 
    Qwen3Model = Qwen2Model

from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_qwen_custom import Qwen3ReasonerConfig

class GatedMLP(nn.Module):
    def __init__(self, dim, ffn_size, depth=1, steps=1, dropout=0.1):
        super().__init__()
        self.steps = steps
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, ffn_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_size, dim),
                nn.Dropout(dropout)
            ))
        
        # Gating mechanism
        self.gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

        # Stable Initialization:
        # 1. Negative bias makes sigmoid(gate(x)) close to 0 initially.
        nn.init.constant_(self.gate.bias, -4.0)
        # 2. Small weight gain ensuring minimal interference during warmup.
        nn.init.xavier_uniform_(self.gate.weight, gain=0.01)

    def forward(self, x):
        h = x
        # Outer Loop: Recurrently apply the same layers
        for _ in range(self.steps):
            # Inner Loop: Sequential layers within each step
            for layer in self.layers:
                h = layer(h) + h  # Skip connection
        
        # Learned gate to decide how much info to keep from the multi-step reasoning
        g = self.sigmoid(self.gate(x))
        return g * h + (1 - g) * x

class Qwen3ForCausalLMWithReasoner(Qwen3ForCausalLM):
    config_class = Qwen3ReasonerConfig

    def __init__(self, config):
        super().__init__(config)
        
        # Replace Transformer with Gated MLP for per-token refinement
        self.reasoner = GatedMLP(
            dim=config.hidden_size,
            ffn_size=getattr(config, "reasoner_ffn_size", 4 * config.hidden_size),
            depth=getattr(config, "reasoner_mlp_depth", 2),
            steps=getattr(config, "reasoner_steps", 1),
            dropout=getattr(config, "reasoner_dropout", 0.1)
        )
        
        # Stratgy: Layer-wise Injection via Hook
        # This ensures compatibility with rotary embeddings, KV cache, and gradient checkpointing.
        self.inject_layer = getattr(config, "reasoner_injection_layer", 24)
        self.reasoning_token_id = 176245
        
        # Register hook on the specific layer
        if self.inject_layer < len(self.model.layers):
            self.model.layers[self.inject_layer].register_forward_pre_hook(self._reasoner_pre_hook)
        
        self.post_init()

    def _reasoner_pre_hook(self, module, args):
        # args[0] is hidden_states
        hidden_states = args[0]
        input_ids = getattr(self, "_current_input_ids", None)
        
        if input_ids is not None:
            mask = (input_ids == self.reasoning_token_id)
            if mask.any():
                # Apply Reasoner refinement
                # We must handle the case where hidden_states might be (Batch, Seq, Dim)
                # but mask might be (Batch, Seq). 
                # PyTorch indexing handles this: hidden_states[mask] returns (N, Dim)
                tokens_hidden = hidden_states[mask]
                reasoned_hidden_states = self.reasoner(tokens_hidden)
                
                # Update in-place to ensure the layer receives modified inputs
                # Note: We use .clone() logic if we want to be safest, but in-place is usually fine for pre-hooks 
                # unless gradient checkpointing is extremely strict.
                hidden_states[mask] = reasoned_hidden_states
        
        return args

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Temporarily store input_ids for the hook to use
        self._current_input_ids = input_ids
        
        # Call the standard forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        # Cleanup
        self._current_input_ids = None
        
        return outputs
