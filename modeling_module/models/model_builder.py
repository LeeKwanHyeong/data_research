from typing import Union, Any

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.Titan.common.configs import TitanConfig


# -----------------------------
# PatchMixer: dict → PatchMixerConfig
# -----------------------------
def _ensure_patchmixer_config(cfg: Union[PatchMixerConfig, dict, Any]) -> PatchMixerConfig:
    """
    구(旧) 체크포인트에서는 config가 dict로만 저장되어 있어서,
    dict가 넘어오면 PatchMixerConfig로 감싸주는 호환 함수.
    """
    if isinstance(cfg, PatchMixerConfig):
        return cfg
    if isinstance(cfg, dict):
        return PatchMixerConfig(**cfg)
    # 예: argparse.Namespace 등일 경우
    if hasattr(cfg, "__dict__"):
        return PatchMixerConfig(**cfg.__dict__)
    raise TypeError(f"Unsupported config type for PatchMixer: {type(cfg)}")


def build_patch_mixer_base(cfg):
    from modeling_module.models.PatchMixer.PatchMixer import BaseModel
    cfg = _ensure_patchmixer_config(cfg)
    return BaseModel(cfg)


def build_patch_mixer_quantile(cfg):
    from modeling_module.models.PatchMixer.PatchMixer import QuantileModel
    cfg = _ensure_patchmixer_config(cfg)
    return QuantileModel(cfg)


# -----------------------------
# Titan: dict/Namespace → TitanConfig
# -----------------------------
def _ensure_titan_config(cfg: Union[TitanConfig, dict, Any]) -> TitanConfig:
    """
    Titan 구(旧) 체크포인트 호환용:
    - dict 또는 Namespace로 저장된 config를 TitanConfig dataclass로 변환.
    """
    if isinstance(cfg, TitanConfig):
        return cfg
    if isinstance(cfg, dict):
        return TitanConfig(**cfg)
    if hasattr(cfg, "__dict__"):
        return TitanConfig(**cfg.__dict__)
    raise TypeError(f"Unsupported config type for Titan: {type(cfg)}")


def build_titan_base(cfg):
    from modeling_module.models.Titan.Titans import TitanBaseModel
    cfg = _ensure_titan_config(cfg)
    return TitanBaseModel.from_config(cfg)


def build_titan_lmm(cfg):
    from modeling_module.models.Titan.Titans import TitanLMMModel
    cfg = _ensure_titan_config(cfg)
    return TitanLMMModel.from_config(cfg)


def build_titan_seq2seq(cfg):
    from modeling_module.models.Titan.Titans import TitanSeq2SeqModel
    cfg = _ensure_titan_config(cfg)
    return TitanSeq2SeqModel.from_config(cfg)


# -----------------------------
# PatchTST: dict/Namespace → PatchTSTConfig
# -----------------------------
def _ensure_patchtst_config(cfg: Union[PatchTSTConfig, dict, Any]) -> PatchTSTConfig:
    """
    PatchTST 구(旧) 체크포인트 호환용:
    - dict 또는 Namespace로 저장된 config를 PatchTSTConfig dataclass로 변환.
    - (attn/head/decomp 가 dict로 들어있으면 가능한 한 Config 클래스로 감싸줌)
    """
    if isinstance(cfg, PatchTSTConfig):
        return cfg

    if isinstance(cfg, dict):
        cfgd = dict(cfg)
        # nested config가 dict로 들어있을 수 있으니, 가능한 경우 변환
        try:
            from modeling_module.models.PatchTST.common.configs import (
                AttentionConfig,
                HeadConfig,
                DecompositionConfig,
            )

            if "attn" in cfgd and isinstance(cfgd["attn"], dict):
                cfgd["attn"] = AttentionConfig(**cfgd["attn"])
            if "head" in cfgd and isinstance(cfgd["head"], dict):
                cfgd["head"] = HeadConfig(**cfgd["head"])
            if "decomp" in cfgd and isinstance(cfgd["decomp"], dict):
                cfgd["decomp"] = DecompositionConfig(**cfgd["decomp"])
        except Exception:
            # 해당 Config 클래스가 없거나, 구조가 달라도 일단 PatchTSTConfig(**cfgd) 시도
            pass

        return PatchTSTConfig(**cfgd)

    if hasattr(cfg, "__dict__"):
        return PatchTSTConfig(**cfg.__dict__)

    raise TypeError(f"Unsupported config type for PatchTST: {type(cfg)}")


def build_patchTST_base(cfg):
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTPointModel
    cfg = _ensure_patchtst_config(cfg)
    return PatchTSTPointModel.from_config(cfg)


def build_patchTST_quantile(cfg):
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTQuantileModel
    cfg = _ensure_patchtst_config(cfg)
    return PatchTSTQuantileModel.from_config(cfg)