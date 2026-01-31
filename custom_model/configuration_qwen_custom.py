from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

try:
    from transformers import Qwen3Config
except ImportError:
    Qwen3Config = Qwen2Config

class Qwen3ReasonerConfig(Qwen3Config):
    model_type = "qwen3"
    
    def __init__(
        self,
        reasoner_mlp_depth=2,
        reasoner_steps=4,
        reasoner_ffn_size=4096,
        reasoner_dropout=0.1,
        num_domains=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reasoner_mlp_depth = reasoner_mlp_depth
        self.reasoner_ffn_size = reasoner_ffn_size
        self.reasoner_dropout = reasoner_dropout
        self.num_domains = num_domains
