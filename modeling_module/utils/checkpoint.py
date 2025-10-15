# utils/checkpoint.py
import json
import os, torch

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfigMonthly
from modeling_module.models.PatchTST.common.configs import PatchTSTConfigMonthly, HeadConfig, AttentionConfig
from modeling_module.models.Titan.common.configs import TitanConfigMonthly
from modeling_module.training.config import DecompositionConfig


def save_model_dict(model_dict, save_dir, cfg_by_name=None, builder_key_by_name=None):
    """
    model_dict: {"PatchMixer Base": model, ...}
    cfg_by_name: 각 모델 이름에 해당하는 config 객체(선택)
    """
    os.makedirs(save_dir, exist_ok=True)
    index = {}
    for name, model in model_dict.items():
        ckpt = {
            "name": name,
            "state_dict": model.state_dict(),
            "arch_cls": model.__class__.__name__,  # 예: "LMMSeq2SeqModel"
            "builder_key": builder_key_by_name.get(name) if builder_key_by_name else None,
        }
        if cfg_by_name and name in cfg_by_name:
            cfg = cfg_by_name[name]
            from dataclasses import asdict, is_dataclass
            ckpt["config"] = asdict(cfg) if is_dataclass(cfg) else getattr(cfg, "__dict__", None)
            ckpt["config_cls"] = type(cfg).__name__
        path = os.path.join(save_dir, f"{name.replace(' ', '_')}.pt")
        torch.save(ckpt, path)
        index[name] = path
    torch.save(index, os.path.join(save_dir, "_index.pt"))
    return index

def _assert_horizon(model, cfg_obj):
    # 모델별로 horizon 읽는 방법이 다르면 조건 분기
    exp = getattr(cfg_obj, "horizon", None)
    if exp is None:
        return
    # Titan 예시: 최종 head out_features == horizon
    head = getattr(model, "output_proj", None)
    if isinstance(head, torch.nn.Linear):
        out = head.out_features
        assert out == exp, f"horizon mismatch: ckpt {exp} vs model {out}"

def _rebuild_patchtst(cfgd: dict):
    cfgd = dict(cfgd)
    if "attn" in cfgd and isinstance(cfgd["attn"], dict):
        cfgd["attn"] = AttentionConfig(**cfgd["attn"])
    if "head" in cfgd and isinstance(cfgd["head"], dict):
        cfgd["head"] = HeadConfig(**cfgd["head"])
    if "decomp" in cfgd and isinstance(cfgd["decomp"], dict):
        cfgd["decomp"] = DecompositionConfig(**cfgd["decomp"])
    return PatchTSTConfigMonthly(**cfgd)

def _rebuild_titan(cfgd: dict):
    return TitanConfigMonthly(**cfgd)

def _rebuild_patchmixer(cfgd: dict):
    return PatchMixerConfigMonthly(**cfgd)

def _is_quantile_state(state_dict: dict) -> bool:
    return any(k.startswith("q_head.") for k in state_dict.keys()) \
        or  any(k.startswith("delta_head.") for k in state_dict.keys())

def _pick_builder_key_safely(name: str, bkey_in_ckpt: str | None, state_dict: dict) -> str:
    # 1) ckpt가 빌더키를 갖고 있으면 그걸 우선
    if bkey_in_ckpt:
        return bkey_in_ckpt
    # 2) decoder.* 존재 여부로 Titan 분기
    has_decoder = any(k.startswith("decoder.") for k in state_dict.keys())
    if "Titan" in name:
        if "Seq2" in name or "Seq2seq" in name:
            return "titan_seq2seq" if has_decoder else "titan_lmm"
        else:
            return "titan_lmm" if not has_decoder else "titan_seq2seq"
    # 3) 나머지 힌트
    if 'PatchMixer' in name:
        if 'Quantile' in name:
            return 'patchmixer_quantile'
        else:
            return 'patchmixer_base'

    if 'PatchTST' in name:
        if 'Quantile' in name:
            return 'patchtst_quantile'
        return 'patchtst_quantile' if _is_quantile_state(state_dict) else 'patchtst_base'

    return 'patchtst_base'

# def load_model_dict(save_dir, builders: dict, device="cpu", strict=True):
#     index = torch.load(os.path.join(save_dir, "_index.pt"), map_location="cpu")
#     loaded = {}
#     for name, path in index.items():
#         ckpt = torch.load(path, map_location="cpu")
#         cfgd = ckpt.get("config")
#         clsname = ckpt.get("config_cls", "")
#         # --- config 복원 ---
#         cfg_obj = None
#         if cfgd:
#             if "PatchTST" in clsname:
#                 cfg_obj = _rebuild_patchtst(cfgd)
#             elif "Titan" in clsname:
#                 cfg_obj = _rebuild_titan(cfgd)
#             elif "PatchMixer" in clsname:
#                 cfg_obj = _rebuild_patchmixer(cfgd)
#
#         # --- 빌더키 안전 선택(이름 vs 내용 불일치 자동 보정) ---
#         bkey = _pick_builder_key_safely(name, ckpt.get("builder_key"), ckpt["state_dict"])
#         if bkey not in builders:
#             raise KeyError(f"builder for '{name}' (key={bkey}) not provided")
#
#         model = builders[bkey](cfg_obj)  # 저장 당시 cfg로 재빌드 → horizon/target_dim 일치
#         try:
#             model.load_state_dict(ckpt["state_dict"], strict=strict)
#         except RuntimeError as e:
#             # --- PointHead 구조 변경 (proj → net.*) 자동 보정 ---
#             if any(k.startswith("head.proj") for k in sd_keys) and any(k.startswith("head.net") for k in own.keys()):
#                 fixed = {}
#                 for k, v in ckpt["state_dict"].items():
#                     if k.startswith("head.proj"):
#                         newk = k.replace("head.proj", "head.net.2")  # Dense layer 위치에 매핑
#                         fixed[newk] = v
#                     else:
#                         fixed[k] = v
#                 ok = {k: v for k, v in fixed.items() if k in own and own[k].shape == v.shape}
#                 model.load_state_dict(ok, strict=False)
#                 print(f"[load warning] {name}: converted head.proj.* → head.net.2.* automatically.")
#                 continue
#
#             # QuantileModel인데 q_head.*만 없음 → 부분 로드 허용
#             sd_keys = set(ckpt["state_dict"].keys())
#             own = model.state_dict()
#
#             def _partial_load_with_msg(msg: str):
#                 ok = {k: v for k, v in ckpt["state_dict"].items()
#                       if (k in own and own[k].shape == v.shape)}
#                 model.load_state_dict(ok, strict=False)
#                 print(f"[load warning] {name}: {msg}  partial={len(ok)}/{len(own)} keys; "
#                       f"missing params randomly initialized.")
#
#             # --- Quantile 전용 헤드 결측 허용 ---
#             need_q = any(k.startswith("q_head.") for k in own.keys())
#             miss_q = not any(k.startswith("q_head.") for k in sd_keys)
#             need_d = any(k.startswith("delta_head.") for k in own.keys())
#             miss_d = not any(k.startswith("delta_head.") for k in sd_keys)
#
#             if need_q and miss_q:
#                 _partial_load_with_msg("checkpoint lacks q_head.*")
#             elif need_d and miss_d:
#                 _partial_load_with_msg("checkpoint lacks delta_head.*")
#             else:
#                 if strict:
#                     raise
#                 _partial_load_with_msg("shape/key mismatch")
#
#         model.to(device).eval()
#         loaded[name] = model
#     return loaded

def load_model_dict(save_dir, builders, device="cpu", strict=False):
    """
    여러 모델을 한꺼번에 로드하는 통합 함수.
    - builders: {"patchtst_base": lambda cfg: model_class(cfg), ...}
    - 자동으로 head 구조 변환 지원 (Point ↔ Quantile)
    """
    models = {}

    for name, build_fn in builders.items():
        path = os.path.join(save_dir, f"{name}.pt")
        if not os.path.exists(path):
            print(f"[warn] checkpoint not found: {path}")
            continue

        print(f"[load] {name} ← {path}")
        ckpt = torch.load(path, map_location=device)
        cfg_obj = ckpt.get("cfg", None)
        model = build_fn(cfg_obj)
        model_name = model.__class__.__name__

        try:
            model.load_state_dict(ckpt["state_dict"], strict=strict)
        except RuntimeError as e:
            print(f"[warn] {model_name}: strict load failed → attempting adaptive remap")
            sd_keys = set(ckpt["state_dict"].keys())
            own = model.state_dict()

            # -----------------------------
            # (1) Quantile ckpt → PointModel 변환
            # -----------------------------
            if any(k.startswith("head.net") for k in sd_keys) and any(k.startswith("head.proj") for k in own.keys()):
                print(f"[load remap] {name}: Quantile checkpoint detected → converting to PointHead.")
                fixed = {}
                for k, v in ckpt["state_dict"].items():
                    if k.startswith("head.net.2.weight"):
                        fixed["head.proj.weight"] = v.mean(dim=0, keepdim=True)
                    elif k.startswith("head.net.2.bias"):
                        fixed["head.proj.bias"] = v.mean().unsqueeze(0)
                ckpt["state_dict"].update(fixed)
                ok = {k: v for k, v in ckpt["state_dict"].items() if k in own and own[k].shape == v.shape}
                model.load_state_dict(ok, strict=False)
                print(f"Converted Quantile → PointHead successfully.")
                models[name] = model.to(device)
                continue

            # -----------------------------
            # (2) Point ckpt → QuantileModel 변환
            # -----------------------------
            if any(k.startswith("head.proj") for k in sd_keys) and any(k.startswith("head.net") for k in own.keys()):
                print(f"[load remap] {name}: Point checkpoint detected → expanding to QuantileHead layers.")
                q_list = getattr(cfg_obj, "quantiles", [0.1, 0.5, 0.9])
                fixed = {}
                for k, v in ckpt["state_dict"].items():
                    if k.startswith("head.proj.weight"):
                        v_expanded = v.unsqueeze(0).expand(len(q_list), -1, -1)
                        fixed["head.net.2.weight"] = v_expanded.mean(dim=0)
                    elif k.startswith("head.proj.bias"):
                        fixed["head.net.2.bias"] = v
                ckpt["state_dict"].update(fixed)
                ok = {k: v for k, v in ckpt["state_dict"].items() if k in own and own[k].shape == v.shape}
                model.load_state_dict(ok, strict=False)
                print(f"Converted Point → QuantileHead successfully.")
                models[name] = model.to(device)
                continue

            # -----------------------------
            # (3) delta_head 누락 시 무시 (Titan / PatchMixer)
            # -----------------------------
            missing_delta = [k for k in own.keys() if k.startswith("delta_head")]
            if missing_delta and not any(k.startswith("delta_head") for k in sd_keys):
                print(f"[load partial] {name}: skipping delta_head.* (missing in checkpoint)")
                ok = {k: v for k, v in ckpt["state_dict"].items() if k in own and own[k].shape == v.shape}
                model.load_state_dict(ok, strict=False)
                models[name] = model.to(device)
                continue

            # -----------------------------
            # (4) 기타 일반적인 mismatch
            # -----------------------------
            print(f"[load partial] {name}: non-critical mismatch → partial load.")
            ok = {k: v for k, v in ckpt["state_dict"].items() if k in own and own[k].shape == v.shape}
            model.load_state_dict(ok, strict=False)

        models[name] = model.to(device)

    return models


# ---------------------------------------
# Save utility (for completeness)
# ---------------------------------------
def save_model(model, cfg, path):
    ckpt = {
        "cfg": cfg,
        "state_dict": model.state_dict()
    }
    torch.save(ckpt, path)
    print(f"[save] model saved to: {path}")


def save_json_config(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2, ensure_ascii=False)
    print(f"[save] config saved to: {path}")