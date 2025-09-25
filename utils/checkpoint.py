# utils/checkpoint.py
import os, torch
from dataclasses import asdict

from models.PatchMixer.common.configs import PatchMixerConfigMonthly
from models.PatchTST.common.configs import PatchTSTConfigMonthly, HeadConfig, AttentionConfig
from models.Titan.common.configs import TitanConfigMonthly
from training.config import DecompositionConfig


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

def _pick_builder_key_safely(name: str, bkey_in_ckpt: str | None, state_dict: dict) -> str:
    # 1) ckpt가 빌더키를 갖고 있으면 그걸 우선
    if bkey_in_ckpt:
        return bkey_in_ckpt
    # 2) decoder.* 존재 여부로 Titan 분기
    has_decoder = any(k.startswith("decoder.") for k in state_dict.keys())
    if "Titan" in name:
        if "Seq2" in name or "Seq2seq" in name:
            return "titan_seq2seq" if has_decoder else "titan_lmm"  # ✅ 내용 기준으로 보정
        else:
            return "titan_lmm" if not has_decoder else "titan_seq2seq"
    # 3) 나머지 힌트
    if "PatchMixer" in name and "Quantile" in name:
        return "patchmixer_quantile"
    if "PatchMixer" in name:
        return "patchmixer_base"
    if "PatchTST" in name:
        return "patchtst_base"
    # 4) 최후의 보루
    return "patchtst_base"

def load_model_dict(save_dir, builders: dict, device="cpu", strict=True):
    index = torch.load(os.path.join(save_dir, "_index.pt"), map_location="cpu")
    loaded = {}
    for name, path in index.items():
        ckpt = torch.load(path, map_location="cpu")
        cfgd = ckpt.get("config")
        clsname = ckpt.get("config_cls", "")
        # --- config 복원 ---
        cfg_obj = None
        if cfgd:
            if "PatchTST" in clsname:
                cfg_obj = _rebuild_patchtst(cfgd)
            elif "Titan" in clsname:
                cfg_obj = _rebuild_titan(cfgd)
            elif "PatchMixer" in clsname:
                cfg_obj = _rebuild_patchmixer(cfgd)

        # --- 빌더키 안전 선택(이름 vs 내용 불일치 자동 보정) ---
        bkey = _pick_builder_key_safely(name, ckpt.get("builder_key"), ckpt["state_dict"])
        if bkey not in builders:
            raise KeyError(f"builder for '{name}' (key={bkey}) not provided")

        model = builders[bkey](cfg_obj)  # 저장 당시 cfg로 재빌드 → horizon/target_dim 일치
        try:
            model = builders[bkey](cfg_obj)  # ← 복원된 cfg로 반드시 빌드
            _assert_horizon(model, cfg_obj)  # ← 여기서 검증 (틀리면 즉시 에러로 원인 명확)
            model.load_state_dict(ckpt["state_dict"], strict=strict)
        except RuntimeError as e:
            if strict:
                raise
            # strict=False일 때는 모양 맞는 키만 부분 로드
            own = model.state_dict()
            ok = {k: v for k, v in ckpt["state_dict"].items() if (k in own and own[k].shape == v.shape)}
            model.load_state_dict(ok, strict=False)
            print(f"[load warning] {name}: partial load, loaded={len(ok)}/{len(own)} keys")

        model.to(device).eval()
        loaded[name] = model
    return loaded