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

class GatedMoE(nn.Module):
    def __init__(self, dim, ffn_size, num_domains=10, depth=1, steps=1, dropout=0.1):
        super().__init__()
        self.shared_expert = GatedMLP(dim, ffn_size, depth, steps, dropout)
        self.domain_experts = nn.ModuleList([
            GatedMLP(dim, ffn_size, depth, steps, dropout) for _ in range(num_domains)
        ])
        
    def forward(self, x, domain_ids):
        # x: (N, D), domain_ids: (N,)
        shared_out = self.shared_expert(x)
        
        # Sum shared expert and domain-specific expert output
        final_out = shared_out
        
        if domain_ids is not None:
            unique_domains = domain_ids.unique()
            for d_id in unique_domains:
                if 0 <= d_id < len(self.domain_experts):
                    mask = (domain_ids == d_id)
                    # Add domain specific contribution
                    final_out[mask] = final_out[mask] + self.domain_experts[d_id](x[mask])
        
        return final_out

class Qwen3ForCausalLMWithReasoner(Qwen3ForCausalLM):
    config_class = Qwen3ReasonerConfig

    def __init__(self, config):
        super().__init__(config)
        
        # Replace single Gated MLP with MoE Reasoner
        self.reasoner = GatedMoE(
            dim=config.hidden_size,
            ffn_size=getattr(config, "reasoner_ffn_size", 4 * config.hidden_size),
            num_domains=getattr(config, "num_domains", 10),
            depth=getattr(config, "reasoner_mlp_depth", 2),
            steps=getattr(config, "reasoner_steps", 1),
            dropout=getattr(config, "reasoner_dropout", 0.1)
        )
        
        # Stratgy: Layer-wise Injection via Hook
        self.inject_layer = getattr(config, "reasoner_injection_layer", 24)
        self.reasoning_token_id = 176245
        
        if self.inject_layer < len(self.model.layers):
            self.model.layers[self.inject_layer].register_forward_pre_hook(self._reasoner_pre_hook)
        
        self.post_init()

    def _reasoner_pre_hook(self, module, args):
        hidden_states = args[0]
        input_ids = getattr(self, "_current_input_ids", None)
        domain_ids = getattr(self, "_current_domain_ids", None)
        
        if input_ids is not None:
            mask = (input_ids == self.reasoning_token_id)
            if mask.any():
                tokens_hidden = hidden_states[mask]
                
                # Prepare domain_ids for the reasoned tokens
                reasoning_domain_ids = None
                if domain_ids is not None:
                    # Case 1: domain_ids is (Batch, Seq) - token level
                    if domain_ids.dim() == 2:
                        reasoning_domain_ids = domain_ids[mask]
                    # Case 2: domain_ids is (Batch,) or (Batch, 1) - sequence level
                    elif domain_ids.dim() == 1:
                        # Expand to match token mask (Batch, Seq) then index
                        expanded_domain_ids = domain_ids.unsqueeze(1).expand_as(mask)
                        reasoning_domain_ids = expanded_domain_ids[mask]
                    elif domain_ids.dim() == 0:
                        # Single scalar (Batch=1 or global)
                        reasoning_domain_ids = domain_ids.expand(tokens_hidden.size(0))

                reasoned_hidden_states = self.reasoner(tokens_hidden, reasoning_domain_ids)
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
        domain_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Store input_ids and domain_ids for the hook
        self._current_input_ids = input_ids
        self._current_domain_ids = domain_ids
        
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
        
        self._current_input_ids = None
        self._current_domain_ids = None
        
        return outputs
