from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

try:
    from transformers import Qwen3Config
except ImportError:
    Qwen3Config = Qwen2Config

class Qwen3ReasonerConfig(Qwen3Config):
    model_type = "qwen3"
    
    def __init__(
        self,
        reasoner_nhead=12,
        reasoner_ffn_size=4096,
        reasoner_layers=1,
        reasoner_dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reasoner_nhead = reasoner_nhead
        self.reasoner_ffn_size = reasoner_ffn_size
        self.reasoner_layers = reasoner_layers
        self.reasoner_dropout = reasoner_dropout
