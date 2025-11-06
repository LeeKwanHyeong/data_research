from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.Titan.common.configs import TitanConfig


def build_patch_mixer_base(cfg: PatchMixerConfig):
    from modeling_module.models.PatchMixer.PatchMixer import BaseModel
    return BaseModel(cfg)

def build_patch_mixer_quantile(cfg: PatchMixerConfig):
    from modeling_module.models.PatchMixer.PatchMixer import QuantileModel
    return QuantileModel(cfg)

# --- Titan builders (Titans.py 리팩토링 버전: config 기반) ---
def build_titan_base(cfg: TitanConfig):
    from modeling_module.models.Titan.Titans import TitanBaseModel
    # 리팩토링된 Titans는 from_config 지원
    return TitanBaseModel.from_config(cfg)

def build_titan_lmm(cfg: TitanConfig):
    from modeling_module.models.Titan.Titans import TitanLMMModel
    return TitanLMMModel.from_config(cfg)

def build_titan_seq2seq(cfg: TitanConfig):
    from modeling_module.models.Titan.Titans import TitanSeq2SeqModel
    return TitanSeq2SeqModel.from_config(cfg)


def build_patchTST_base(cfg: PatchTSTConfig):
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTPointModel
    return PatchTSTPointModel.from_config(cfg)

def build_patchTST_quantile(cfg: PatchTSTConfig):
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTQuantileModel
    return PatchTSTQuantileModel.from_config(cfg)



