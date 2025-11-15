# model_io.py
import os
import json
import torch
from dataclasses import asdict, is_dataclass

from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfigMonthly,
    PatchMixerConfigWeekly,
)
from modeling_module.models.PatchTST.common.configs import (
    PatchTSTConfigMonthly,
    PatchTSTConfig,
    HeadConfig,
    AttentionConfig,
)
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.training.config import DecompositionConfig


# ------------------------------------------------------------------
# 0. (선택) 옛 포맷 지원용: config dict → config 객체 복원 함수들
#    기존에 save_model_dict로 저장했던 ckpt를 살리고 싶을 때만 사용
# ------------------------------------------------------------------
def _rebuild_patchtst(cfgd: dict):
    cfgd = dict(cfgd)
    if "attn" in cfgd and isinstance(cfgd["attn"], dict):
        cfgd["attn"] = AttentionConfig(**cfgd["attn"])
    if "head" in cfgd and isinstance(cfgd["head"], dict):
        cfgd["head"] = HeadConfig(**cfgd["head"])
    if "decomp" in cfgd and isinstance(cfgd["decomp"], dict):
        cfgd["decomp"] = DecompositionConfig(**cfgd["decomp"])
    return PatchTSTConfig(**cfgd)


def _rebuild_patchmixer_monthly(cfgd: dict):
    return PatchMixerConfigMonthly(**cfgd)


def _rebuild_patchmixer_weekly(cfgd: dict):
    return PatchMixerConfigWeekly(**cfgd)


def _rebuild_titan(cfgd: dict):
    return TitanConfig(**cfgd)


# ------------------------------------------------------------------
# 1. 새로운 저장 유틸 (훈련 시 반드시 이걸로 저장)
# ------------------------------------------------------------------
def save_model(model, cfg, path: str):
    """
    단일 모델을 저장하는 유틸.

    ckpt 포맷:
      {
        "cfg":       cfg 객체 (그대로 pickle),
        "cfg_state": dict(asdict(cfg)) or cfg.__dict__,
        "cfg_cls":   cfg 클래스 이름(str),
        "state_dict": model.state_dict()
      }
    """
    if is_dataclass(cfg):
        cfg_state = asdict(cfg)
    else:
        cfg_state = getattr(cfg, "__dict__", None)

    ckpt = {
        "cfg": cfg,  # 그대로 pickle (dataclass면 문제 없음)
        "cfg_state": cfg_state,
        "cfg_cls": type(cfg).__name__,
        "state_dict": model.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"[save] model saved to: {path}")


def save_json_config(cfg, path: str):
    """
    config를 json으로 별도 저장하고 싶을 때 사용 (옵션)
    """
    if is_dataclass(cfg):
        data = asdict(cfg)
    else:
        data = getattr(cfg, "__dict__", None)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[save] config saved to: {path}")


# ------------------------------------------------------------------
# 2. 새로운 로더 (Jupyter / inference 에서 사용)
#    - builders: {"titan_base": lambda cfg: build_titan_base(cfg), ...}
#    - strict=True 로 설정하면, 키/shape 틀리면 바로 RuntimeError
# ------------------------------------------------------------------
import os
import torch
from typing import Dict

def load_model_dict(save_dir: str, builders: Dict[str, callable], device="cpu", strict: bool = False):
    """
    - save_dir: 각 모델이 `{name}.pt` 형식으로 저장된 디렉터리
    - builders: {"titan_base": build_titan_base, "patchmixer_base": build_patch_mixer_base, ...}
      각 value는 (cfg) -> nn.Module 을 반환하는 함수여야 함.
    """
    models = {}

    for name, build_fn in builders.items():
        path = os.path.join(save_dir, f"{name}.pt")
        if not os.path.exists(path):
            print(f"[warn] checkpoint not found: {path}")
            continue

        print(f"[load] {name} ← {path}")
        ckpt = torch.load(path, map_location="cpu")

        # ★ 여기서 config 꺼내서 builder에 넘겨줌
        cfg_obj = ckpt.get("cfg", None)

        # ★ 꼭 build_fn(cfg_obj)로 '호출'해야 함
        model = build_fn(cfg_obj)

        # 혹시라도 builder가 모델이 아닌 걸 반환하면 방어
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"builder for '{name}' must return nn.Module, got {type(model)}. "
                f"확인: build_fn={build_fn}"
            )

        # state_dict 로드
        sd = ckpt["state_dict"]
        try:
            model.load_state_dict(sd, strict=strict)
        except RuntimeError as e:
            print(f"[warn] {name}: strict load failed → partial load로 재시도")
            own = model.state_dict()
            filtered = {}
            skipped = []
            for k, v in sd.items():
                if k not in own:
                    continue
                if own[k].shape != v.shape:
                    skipped.append((k, v.shape, own[k].shape))
                    continue
                filtered[k] = v
            if skipped:
                print("[load partial] shape-mismatch skipped:")
                for k, s_ckpt, s_model in skipped:
                    print(f"  - {k}: ckpt={s_ckpt}, model={s_model}")
            model.load_state_dict(filtered, strict=False)

        model.to(device).eval()
        models[name] = model

    return models