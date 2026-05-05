"""MSA (Memory Sparse Attention) Configuration"""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


DEFAULT_MSA_CONFIG = {
    "router_layer_idx": "all",
    "aux_loss": False,
    "lmloss_weight": 1.0,
    "auxloss_weight": 0.1,
    "recloss_weight": 0.0,
    "ansloss_weight": 1.0,
    "aux_loss_method": "INFONCE",
    "decouple_router": False,
    "rewrite_position": False,
    "top_k_docs": 16,
    "pooling_kernel_size": 64,
    "infonce_loss_temp": 0.1,
    "head_reduce_method": "max",
    "query_reduce_method": "max",
    "chunk_reduce_method": "max",
    "decouple_pooling_mode": "mean",
}


class DotDict(dict):
    """支持点号访问的字典类"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


class MSAConfig(Qwen3Config):
    """
    MSA 模型配置类，继承自 Qwen3Config。
    确保 config.msa_config 支持点号访问，例如 config.msa_config.top_k_docs。
    """

    model_type = "msa"

    def __init__(self, msa_config=None, **kwargs):
        super().__init__(**kwargs)

        merged_msa_config = dict(DEFAULT_MSA_CONFIG)

        if msa_config is not None:
            merged_msa_config.update(dict(msa_config))
        elif isinstance(getattr(self, "msa_config", None), dict):
            merged_msa_config.update(dict(getattr(self, "msa_config")))

        self.msa_config = DotDict(merged_msa_config)

    def __setattr__(self, name, value):
        if name == "msa_config" and isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        super().__setattr__(name, value)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        output = super().from_dict(config_dict, **kwargs)

        if isinstance(output, tuple):
            config, unused_kwargs = output
            return_tuple = True
        else:
            config = output
            unused_kwargs = None
            return_tuple = False

        merged_msa_config = dict(DEFAULT_MSA_CONFIG)

        if hasattr(config, "msa_config") and isinstance(config.msa_config, dict):
            merged_msa_config.update(dict(config.msa_config))

        if "msa_config" in config_dict and isinstance(config_dict["msa_config"], dict):
            merged_msa_config.update(dict(config_dict["msa_config"]))

        config.msa_config = DotDict(merged_msa_config)

        if return_tuple:
            return config, unused_kwargs

        return config
