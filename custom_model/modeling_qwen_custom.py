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
        
        self.post_init()

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # IMPORTANT: To avoid deep recursion or breaking internal logic, we manually execute the layers 
        # to perform injection at a specific middle layer.
        
        # 1. Standard Pre-processing (Embeddings)
        # Note: We rely on the internal self.model for standard utilities
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        # 2. Preparation (similar to Qwen3Model.forward)
        # We try to use the model's internal setup if possible, or just call manually.
        # For simplicity and stability, we execute the layers loop here.
        
        # Handle position_ids and attention_mask
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(0, hidden_states.shape[1], device=device).unsqueeze(0)
        
        # 3. Layer Loop with Injection
        reasoning_token_id = 176245
        # We can configure which layer to inject at. Default to middle-deep (e.g., 24/32)
        inject_layer = getattr(self.config, "reasoner_injection_layer", 24)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Perform Injection at the start of chosen layer or during hidden state flow
            if i == inject_layer and input_ids is not None:
                mask = (input_ids == reasoning_token_id)
                if mask.any():
                    # We inject AFTER the previous layer's output but BEFORE the next layer starts
                    # This way, layer[inject_layer] and onwards "see" the refined state.
                    tokens_hidden = hidden_states[mask]
                    reasoned_hidden_states = self.reasoner(tokens_hidden)
                    hidden_states = hidden_states.clone() # Avoid in-place issues
                    hidden_states[mask] = reasoned_hidden_states

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                all_self_attns += (layer_outputs[2 if use_cache else 1],)

        hidden_states = self.model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + (next_decoder_cache, all_hidden_states, all_self_attns)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
