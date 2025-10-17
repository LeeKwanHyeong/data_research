from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.Titan.common.configs import TitanConfig


def build_patch_mixer_base(cfg: PatchMixerConfig):
    from modeling_module.models.PatchMixer.PatchMixer import BaseModel
    return BaseModel(cfg)

def build_patch_mixer_feature(cfg: PatchMixerConfig):
    from modeling_module.models.PatchMixer.PatchMixer import FeatureModel
    return FeatureModel(cfg)

def build_patch_mixer_quantile(cfg: PatchMixerConfig):
    from modeling_module.models.PatchMixer.PatchMixer import QuantileModel
    return QuantileModel(cfg)

def build_titan_base(cfg: TitanConfig):
    from modeling_module.models.Titan.Titans import Model
    return Model(cfg)

def build_titan_lmm(cfg: TitanConfig):
    from modeling_module.models.Titan.Titans import LMMModel
    return LMMModel(cfg)

def build_titan_seq2seq(cfg: TitanConfig):
    from modeling_module.models.Titan.Titans import LMMSeq2SeqModel
    return LMMSeq2SeqModel(cfg)

def build_titan_patch(cfg: TitanConfig):
    from modeling_module.models.Titan.Titans import PatchLMMModel
    return PatchLMMModel(cfg)

def build_titan_feature(cfg: TitanConfig):
    from modeling_module.models.Titan.Titans import FeatureModel
    return FeatureModel(cfg)

def build_patchTST_base(cfg: PatchTSTConfig):
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTPointModel
    return PatchTSTPointModel(cfg)

def build_patchTST_quantile(cfg: PatchTSTConfig):
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTQuantileModel
    return PatchTSTQuantileModel(cfg)



