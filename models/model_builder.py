import torch

from models.PatchMixer.common.configs import PatchMixerConfig
from models.PatchTST.common.configs import PatchTSTConfig
from models.Titan.common.configs import TitanConfig


def build_patch_mixer_base(cfg: PatchMixerConfig):
    from models.PatchMixer.PatchMixer import BaseModel
    return BaseModel(cfg)

def build_patch_mixer_feature(cfg: PatchMixerConfig):
    from models.PatchMixer.PatchMixer import FeatureModel
    return FeatureModel(cfg)

def build_patch_mixer_quantile(cfg: PatchMixerConfig):
    from models.PatchMixer.PatchMixer import QuantileModel
    return QuantileModel(cfg)

def build_titan_base(cfg: TitanConfig):
    from models.Titan.Titans import Model
    return Model(cfg)

def build_titan_lmm(cfg: TitanConfig):
    from models.Titan.Titans import LMMModel
    return LMMModel(cfg)

def build_titan_seq2seq(cfg: TitanConfig):
    from models.Titan.Titans import LMMSeq2SeqModel
    return LMMSeq2SeqModel(cfg)

def build_titan_feature(cfg: TitanConfig):
    from models.Titan.Titans import FeatureModel
    return FeatureModel(cfg)

def build_patchTST_base(cfg: PatchTSTConfig):
    from models.PatchTST.supervised.PatchTST import BaseModel
    return BaseModel(cfg)

def build_patchTST_quantile(cfg: PatchTSTConfig):
    from models.PatchTST.supervised.PatchTST import QuantileModel
    return QuantileModel(cfg)



