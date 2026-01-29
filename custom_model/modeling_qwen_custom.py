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

class Qwen3ForCausalLMWithReasoner(Qwen3ForCausalLM):
    config_class = Qwen3ReasonerConfig

    def __init__(self, config):
        # We call the super init which sets up the base model and lm_head
        super().__init__(config)
        
        # Add the reasoner component
        nhead = getattr(config, "reasoner_nhead", None) or getattr(config, "num_attention_heads", 16)
        ffn_size = getattr(config, "reasoner_ffn_size", 4096)
        
        self.reasoner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=nhead,
                dim_feedforward=ffn_size,
                dropout=getattr(config, "reasoner_dropout", 0.1),
                batch_first=True,
            ),
            num_layers=getattr(config, "reasoner_layers", 1),
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Modified forward to include reasoner logic.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (last_hidden_state, past_key_values, all_hidden_states, all_attentions)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        
        # --- Reasoner Injection ---
        # Logic: Only apply reasoning to positions where input_id is 176245
        reasoning_token_id = 176245
        combined_hidden_states = hidden_states
        
        if input_ids is not None:
            mask = (input_ids == reasoning_token_id)
            if mask.any():
                # Pass through reasoner (contextual)
                # Use attention_mask to avoid attending to padding tokens
                src_key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
                reasoned_hidden_states = self.reasoner(hidden_states, src_key_padding_mask=src_key_padding_mask)
                
                # Clone to avoid in-place mod and facilitate gradient flow
                combined_hidden_states = hidden_states.clone()
                # Average original and reasoned hidden states at specific positions
                combined_hidden_states[mask] = (hidden_states[mask] + reasoned_hidden_states[mask]) / 2.0
        # --------------------------

        logits = self.lm_head(combined_hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
