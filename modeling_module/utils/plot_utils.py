import os
import inspect
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt

from modeling_module.training.forecaster import DMSForecaster

# ==============================
# Global DEBUG
# ==============================
DEBUG_FCAST = True


# ==============================
# Small helpers
# ==============================
def _log_var(tag, arr):  # arr: (H,) 또는 (H,1)
    arr = np.asarray(arr).reshape(-1)
    print(f"[PLOT-DBG] {tag}: var(H)={np.nanvar(arr):.6g}, "
          f"mean(H)={np.nanmean(arr):.6g}, unique(H)={np.unique(np.round(arr, 6)).size}")


def extract_first_step_q(out):
    """
    out에서 첫 타임스텝의 q10, q50, q90을 추출.
    지원 형태:
      - Tensor: (B, Q, H) 또는 (B, H, Q)
      - dict:   {"q": Tensor(...)} 또는 {"q10","q50","q90"} 각각 텐서
      - tuple/list: 내부에 Tensor가 들어있는 경우
    """
    import torch

    # -------------------------
    # 1) dict 출력 처리
    # -------------------------
    if isinstance(out, dict):
        # case 1: 개별 quantile 키가 있는 경우
        if all(k in out for k in ("q10", "q50", "q90")):
            def _first_step(t):
                if not torch.is_tensor(t):
                    raise AssertionError(f"q tensor must be torch.Tensor, got {type(t)}")
                if t.dim() == 3:  # (B,H,1) 또는 (B,1,H) 등
                    # 일단 첫 시점 기준으로 사용
                    return t[:, 0, 0]
                elif t.dim() == 2:  # (B,H)
                    return t[:, 0]
                elif t.dim() == 1:  # (B,)
                    return t
                else:
                    raise AssertionError(f"unexpected q tensor dim: {t.dim()}")

            return (
                _first_step(out["q10"]),
                _first_step(out["q50"]),
                _first_step(out["q90"]),
            )

        # case 2: 단일 "q" 텐서만 있는 경우 (B,Q,H) 또는 (B,H,Q)
        if "q" in out and torch.is_tensor(out["q"]):
            q = out["q"]
            if q.dim() == 3:
                B, A, C = q.shape
                # (B,Q,H) 형태라고 가정할 수 있는 경우
                if A in (3, 5, 9):
                    # Q축이 dim=1
                    q10_t = q[:, 0, 0]
                    q50_t = q[:, 1, 0]
                    q90_t = q[:, 2, 0]
                # (B,H,Q) 형태라고 가정할 수 있는 경우
                elif C in (3, 5, 9):
                    # Q축이 dim=2
                    q10_t = q[:, 0, 0]
                    q50_t = q[:, 0, 1]
                    q90_t = q[:, 0, 2]
                else:
                    # 모호하면 일단 (B,Q,H)로 취급
                    q10_t = q[:, 0, 0]
                    q50_t = q[:, 1, 0]
                    q90_t = q[:, 2, 0]
                return q10_t, q50_t, q90_t

            elif q.dim() == 2 and q.shape[1] >= 3:  # (B,Q)
                return q[:, 0], q[:, 1], q[:, 2]

            else:
                raise AssertionError(f"unexpected 'q' tensor shape: {q.shape}")

        # dict지만 위 두 케이스 모두 아닌 경우
        raise AssertionError(
            f"cannot extract quantiles from dict output; keys={list(out.keys())}"
        )

    # -------------------------
    # 2) tuple / list 출력 처리
    # -------------------------
    if isinstance(out, (tuple, list)):
        # 안에서 텐서만 골라서 사용
        tensor_candidates = [t for t in out if torch.is_tensor(t)]
        if not tensor_candidates:
            raise AssertionError(
                f"quantile model output tuple/list has no tensor values: "
                f"types={[type(t) for t in out]}"
            )
        out = tensor_candidates[0]

    # -------------------------
    # 3) 나머지는 텐서라고 가정하고 처리
    # -------------------------
    if not torch.is_tensor(out):
        raise AssertionError(f"expect tensor, got {type(out)}")

    if out.dim() != 3:
        raise AssertionError(
            f"expect 3D tensor (B,Q,H or B,H,Q), got shape={getattr(out, 'shape', None)}"
        )

    B, A, C = out.shape
    # (B,Q,H) 형태
    if A in (3, 5, 9):
        return out[:, 0, 0], out[:, 1, 0], out[:, 2, 0]
    # (B,H,Q) 형태
    if C in (3, 5, 9):
        return out[:, 0, 0], out[:, 0, 1], out[:, 0, 2]

    # 마지막 fallback: 앞 3개를 Q축으로 보고 사용
    return out[:, 0, 0], out[:, 1, 0], out[:, 2, 0]

@torch.no_grad()
def _to_1d_history(x: torch.Tensor) -> np.ndarray:
    """
    단일 샘플 텐서에서 lookback 구간의 1D 시계열을 추출한다.
    기대 입력: x.shape in {(1,L), (1,L,1), (1,L,C), (1,C,L)}
    """
    x = x.squeeze(0)
    if x.dim() == 1:             # (L,)
        return x.detach().cpu().numpy()
    if x.dim() == 2:
        h, w = x.shape
        if h >= w:               # (L, C) 가정 → 첫 채널
            return x[:, 0].detach().cpu().numpy()
        else:                    # (C, L) 가정 → 첫 채널의 시계열
            return x[0, :].detach().cpu().numpy()
    return np.array([])


def _supports_kw(model: torch.nn.Module, kw: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
        return kw in sig.parameters
    except Exception:
        return False


def _safe_call(model, x, **kwargs):
    """
    모델이 실제로 받는 키워드만 걸러서 한 번에 호출.
    실패 시, (x) → (x, future_exo) 백업 시도.
    """
    try:
        sig = inspect.signature(model.forward)
        allowed = {k: v for k, v in kwargs.items()
                   if (v is not None and k in sig.parameters)}
        return model(x, **allowed)
    except Exception:
        fe = kwargs.get("future_exo", None)
        try:
            return model(x) if fe is None else model(x, future_exo=fe)
        except Exception:
            # 최후의 수단: mode만 제거하고 재시도
            allowed = {k: v for k, v in kwargs.items()
                       if (v is not None and k in ("future_exo",))}
            return model(x, **allowed)


@torch.no_grad()
def _infer_horizon(model, default=120):
    for k in ("horizon", "output_horizon", "H", "Hm"):
        if hasattr(model, k):
            try:
                return int(getattr(model, k))
            except Exception:
                pass
    return default


@torch.no_grad()
def _probe_output(model, x1, device='cuda' if torch.cuda.is_available() else 'cpu',
                  future_exo_cb=None, future_exo_batch=None,
                  part_ids=None, past_exo_cont=None, past_exo_cat=None):
    """
    모델을 한 번 호출해 '형태 파악용' 출력을 가져온다.
    - 반환은 Tensor 또는 dict( {'point': Tensor, 'q': Tensor|dict} )일 수 있다.
    - 배치 제공 외생/ID가 있으면 함께 전달(지원되는 키만).
    """
    model = model.to(device).eval()
    Hm = _infer_horizon(model, default=120)

    # future_exo 우선순위: 배치 제공 > 콜백 생성 > None
    exo = None
    if isinstance(future_exo_batch, torch.Tensor):
        exo = future_exo_batch
        if exo.dim() == 2:  # (H,E) → (1,H,E)
            exo = exo.unsqueeze(0)
        exo = exo.to(device)
    elif future_exo_cb is not None:
        ex = future_exo_cb(0, Hm, device=device)  # (H,D)
        exo = ex.unsqueeze(0).expand(x1.size(0), -1, -1)

    out = _safe_call(
        model, x1.to(device),
        future_exo=exo,
        part_ids=(part_ids.to(device) if isinstance(part_ids, torch.Tensor) else None),
        past_exo_cont=(past_exo_cont.to(device) if isinstance(past_exo_cont, torch.Tensor) else None),
        past_exo_cat=(past_exo_cat.to(device) if isinstance(past_exo_cat, torch.Tensor) else None),
        mode="eval",
    )

    # tuple/list면 첫 텐서나 dict를 꺼낸다
    if isinstance(out, (tuple, list)):
        for t in out:
            if torch.is_tensor(t) or isinstance(t, dict):
                out = t
                break
    return out  # Tensor or dict


# ==============================
# Quantile rolling (IMS)
# ==============================
@torch.no_grad()
def _roll_quantile_ims(
    model,
    x_init,
    horizon: int,
    *,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    future_exo_cb=None,
    future_exo_batch: Optional[torch.Tensor] = None,  # (1,Hm,E) or (B,Hm,E) or (Hm,E)
    part_ids: Optional[torch.Tensor] = None,
    past_exo_cont: Optional[torch.Tensor] = None,
    past_exo_cat: Optional[torch.Tensor] = None,
    target_channel: int = 0,
    fill_mode: str = "copy_last",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantile 모델(출력 (B,Q,H) 또는 (B,H,Q))을 IMS로 굴려 길이=horizon의
    q10/q50/q90 시퀀스를 만든다. 반환: (q10, q50, q90) 각 (B,H).
    """
    model = model.to(device).eval()
    x = x_init.to(device).float().clone()
    if x.dim() == 2:
        x = x.unsqueeze(-1)  # (B,L)->(B,L,1)
    B = x.size(0)

    # probe로 모양 파악
    out_probe = _probe_output(
        model, x[:1], device=device,
        future_exo_cb=future_exo_cb, future_exo_batch=future_exo_batch,
        part_ids=(part_ids[:1] if isinstance(part_ids, torch.Tensor) else None),
        past_exo_cont=(past_exo_cont[:1] if isinstance(past_exo_cont, torch.Tensor) else None),
        past_exo_cat=(past_exo_cat[:1] if isinstance(past_exo_cat, torch.Tensor) else None),
    )
    # if isinstance(out_probe, (tuple, list)):
    #     out_probe = next(t for t in out_probe if torch.is_tensor(t))
    # assert torch.is_tensor(out_probe) and out_probe.dim() == 3, \
    #     f"expect 3D output for quantile model, got {type(out_probe)} {getattr(out_probe, 'shape', None)}"

    # ---- NEW: dict / tuple / list 래핑된 출력 처리 ----
    if isinstance(out_probe, dict):
        # 1) dict 안에서 텐서만 모으고
        tensor_candidates = [
            v for v in out_probe.values()
            if torch.is_tensor(v)
        ]
        # 2) 그 중에서 dim == 3 인 텐서를 우선 사용 (B, Q, H)나 (B, H, Q)일 가능성 높음
        three_d = [v for v in tensor_candidates if v.dim() == 3]
        if three_d:
            out_probe = three_d[0]
        elif tensor_candidates:
            # 3D 텐서가 없으면 일단 첫 텐서를 사용(추후 필요시 로직 보완)
            out_probe = tensor_candidates[0]
        else:
            raise AssertionError(
                f"quantile model output dict has no tensor values: keys={list(out_probe.keys())}"
            )

    elif isinstance(out_probe, (tuple, list)):
        tensor_candidates = [t for t in out_probe if torch.is_tensor(t)]
        if not tensor_candidates:
            raise AssertionError(
                f"quantile model output tuple/list has no tensor values: "
                f"types={[type(t) for t in out_probe]}"
            )
        three_d = [t for t in tensor_candidates if t.dim() == 3]
        out_probe = three_d[0] if three_d else tensor_candidates[0]

    # ---- 기존 assert 유지 (3차원 텐서를 기대) ----
    assert torch.is_tensor(out_probe) and out_probe.dim() == 3, \
        f"expect 3D output for quantile model, got {type(out_probe)} {getattr(out_probe, 'shape', None)}"

    # 축 자동 감지
    if out_probe.shape[1] in (3, 5, 9):          # (B,Q,Hm)
        Hm = out_probe.shape[2]
        # def extract_first_step_q(out):
        #     if isinstance(out, (tuple, list)):
        #         out = next(tt for tt in out if torch.is_tensor(tt))
        #     return out[:, 0, 0], out[:, 1, 0], out[:, 2, 0]
    elif out_probe.shape[2] in (3, 5, 9):        # (B,Hm,Q)
        Hm = out_probe.shape[1]
        # def extract_first_step_q(out):
        #     if isinstance(out, (tuple, list)):
        #         out = next(tt for tt in out if torch.is_tensor(tt))
        #     return out[:, 0, 0], out[:, 0, 1], out[:, 0, 2]
    else:
        raise RuntimeError(f"cannot infer quantile axis from shape {tuple(out_probe.shape)}")

    q10_seq, q50_seq, q90_seq = [], [], []

    def _prepare_next_input(x, y_step, *, target_channel=0, fill_mode="copy_last"):
        B_, L, C = x.shape
        y_step = y_step.reshape(B_, 1, 1)
        if C == 1:
            new_tok = y_step
        else:
            last = x[:, -1:, :].clone()
            new_tok = torch.zeros_like(last) if fill_mode == "zeros" else last
            new_tok[:, 0, target_channel] = y_step[:, 0, 0]
        return torch.cat([x[:, 1:, :], new_tok], dim=1)

    for t in range(int(horizon)):
        # step별 future exo 준비 (길이는 항상 Hm)
        exo = None
        if isinstance(future_exo_batch, torch.Tensor):
            ex = future_exo_batch
            if ex.dim() == 2:    # (Hm,E) -> (1,Hm,E)
                ex = ex.unsqueeze(0)
            # 배치 길이 보정
            exo = ex.expand(B, -1, -1).to(device)
        elif future_exo_cb is not None:
            ex = future_exo_cb(t, Hm, device=device)   # (Hm, D)
            exo = ex.unsqueeze(0).expand(B, -1, -1)    # (B,Hm,D)

        out = _safe_call(
            model, x,
            future_exo=exo,
            part_ids=(part_ids.to(device) if isinstance(part_ids, torch.Tensor) else None),
            past_exo_cont=(past_exo_cont.to(device) if isinstance(past_exo_cont, torch.Tensor) else None),
            past_exo_cat=(past_exo_cat.to(device) if isinstance(past_exo_cat, torch.Tensor) else None),
            mode="eval",
        )
        q10_t, q50_t, q90_t = extract_first_step_q(out)

        if DEBUG_FCAST and t < 5:
            nm = str(getattr(model, "model_name", "Unknown"))
            print(f"[Q-IMS][{nm}] t={t} q10={float(q10_t[0]):.6g} q50={float(q50_t[0]):.6g} q90={float(q90_t[0]):.6g}")

        q10_seq.append(q10_t.unsqueeze(1))
        q50_seq.append(q50_t.unsqueeze(1))
        q90_seq.append(q90_t.unsqueeze(1))

        # q50으로 다음 입력 윈도우 갱신
        x = _prepare_next_input(x, q50_t, target_channel=target_channel, fill_mode=fill_mode)

    q10 = torch.cat(q10_seq, dim=1)  # (B,H)
    q50 = torch.cat(q50_seq, dim=1)
    q90 = torch.cat(q90_seq, dim=1)
    return q10, q50, q90


def _align_len(yhat: np.ndarray, H: int):
    yhat = np.asarray(yhat).reshape(-1)
    if yhat.size == H:
        return yhat, None
    if yhat.size == 1:
        return np.repeat(yhat, H), "[rep]"
    if yhat.size > H:
        return yhat[:H], "[cut]"
    pad = np.full(H - yhat.size, np.nan)
    return np.concatenate([yhat, pad], axis=0), "[pad]"


# ==============================
# Common predictor for any model (DICT 출력 우선 처리 포함)
# ==============================
@torch.no_grad()
def _predict_any(
    model,
    x1,
    horizon: int,
    device: str,
    future_exo_cb=None,
    *,
    part_ids: Optional[torch.Tensor] = None,
    future_exo_batch: Optional[torch.Tensor] = None,  # (1,Hm,E) or (B,Hm,E) or (Hm,E)
    past_exo_cont: Optional[torch.Tensor] = None,
    past_exo_cat: Optional[torch.Tensor] = None,
):
    """
    - 학습과 동일하게: 배치 제공 외생/part_ids를 모델 forward로 전달(지원 시)
    - Quantile 모델이 Hm < horizon이면 IMS 롤링으로 확장해 120개월 등 생성
    - Point 모델의 Hm < horizon인 경우는 기존 DMSForecaster 사용(필요시 point-IMS 함수 분리 가능)
    """
    was_training = model.training
    model.eval()  # ★ 추론은 항상 eval
    try:
        # def _build_future_exo(Hm: int):
        #     if isinstance(future_exo_batch, torch.Tensor):
        #         ex = future_exo_batch
        #         if ex.dim() == 2:
        #             ex = ex.unsqueeze(0)  # (H,E)->(1,H,E)
        #         return ex.to(device)
        #     if future_exo_cb is None:
        #         return None
        #     ex = future_exo_cb(0, Hm, device=device)   # (H,D)
        #     return ex.unsqueeze(0)  # (1,H,D)

        def _build_future_exo(
                Hm: int
        ):
            """
            future_exo_batch / future_exo_cb 두 소스를 조합해서
            (1, Hm, E) 형태의 exogenous 텐서를 만들어 반환.
            - future_exo_batch가 실질적인 feature(마지막 차원 E>0)를 가지고 있으면 그걸 우선 사용
            - 그렇지 않으면(빈 텐서, E=0) 콜백(future_exo_cb)을 사용
            """

            # 1) 배치에 들어온 exo가 '실제'로 존재하는 경우만 사용
            if isinstance(future_exo_batch, torch.Tensor):
                ex = future_exo_batch
                # ex: (H, E) or (B, H, E) or (1, H, E) 형태일 수 있음

                # (a) 완전히 비어있거나, feature 차원이 0이면 의미 없는 exo → 무시
                if ex.numel() == 0 or ex.size(-1) == 0:
                    ex = None
                else:
                    # (b) (H, E) 이면 (1, H, E) 로 승격
                    if ex.dim() == 2:
                        ex = ex.unsqueeze(0)
                    # (c) (B, H, E) 인 경우는 그대로 사용
                    return ex.to(device)

            # 2) 여기까지 왔다는 것은:
            #   - future_exo_batch 가 None 이었거나
            #   - 텐서이긴 한데 numel==0 또는 E==0 이라서 무시된 상황
            if future_exo_cb is None:
                # exogenous 없이도 동작하도록 None 반환
                return None

            # 3) 콜백 기반 future exo 생성
            ex = future_exo_cb(0, Hm, device=device)  # (H, E) or (1, H, E) 예상
            if ex is None:
                return None
            if isinstance(ex, torch.Tensor):
                if ex.dim() == 2:
                    ex = ex.unsqueeze(0)  # (1, H, E)
                return ex.to(device)

            # 혹시 다른 형식이면 그냥 버린다
            return None

        Hm = int(getattr(model, "horizon", horizon)) or horizon
        future_exo = _build_future_exo(Hm)
        x1_dev = x1.to(device)

        # 1) 1차 호출
        with torch.no_grad():
            out = _safe_call(
                model, x1_dev,
                future_exo=(future_exo.expand(x1.size(0), -1, -1) if isinstance(future_exo, torch.Tensor) else None),
                part_ids=(part_ids.to(device) if isinstance(part_ids, torch.Tensor) else None),
                past_exo_cont=(past_exo_cont.to(device) if isinstance(past_exo_cont, torch.Tensor) else None),
                past_exo_cat=(past_exo_cat.to(device) if isinstance(past_exo_cat, torch.Tensor) else None),
                mode="eval",
            )

        # 1.5) dict 출력 우선 처리 (Quantile/Point 혼합도 안전 처리)
        if isinstance(out, dict):
            # (a) 'q'가 있으면 분위수 처리
            if "q" in out and torch.is_tensor(out["q"]):
                q = out["q"]  # (B,Q,Hm) or (B,Hm,Q)
                if q.dim() != 3:
                    raise RuntimeError(f"expect 3D tensor in out['q'], got {tuple(q.shape)}")

                B, A, Bdim = q.shape
                if Bdim in (3, 5, 9):             # (B,Hm,Q)
                    out_hq = q
                elif A in (3, 5, 9):              # (B,Q,Hm)
                    out_hq = q.transpose(1, 2)    # -> (B,Hm,Q)
                else:
                    # 축을 못 찾으면 중앙값만 포인트화
                    q50 = q.mean(dim=1).squeeze(0).detach().cpu().numpy()
                    q50 = q50[:horizon] if q50.size >= horizon else np.pad(np.asarray(q50, dtype=float), (0, horizon - q50.size), mode="constant", constant_values=np.nan)
                    return {"point": q50}

                Qn = out_hq.size(-1)
                def _slice_or_pad(v):
                    v = np.asarray(v, dtype=float)
                    return v[:horizon] if v.size >= horizon else np.pad(v, (0, horizon - v.size), mode="constant", constant_values=np.nan)

                # Hm == horizon 이면 바로 반환
                if out_hq.shape[1] >= horizon:
                    if Qn == 3:
                        i10, i50, i90 = (0, 1, 2)
                    elif Qn == 5:
                        i10, i50, i90 = (1, 2, 3)
                    elif Qn == 9:
                        i10, i50, i90 = (1, 4, 7)
                    else:
                        i10 = i50 = i90 = None

                    if i50 is not None:
                        q10 = out_hq[0, :, i10].detach().cpu().numpy()
                        q50 = out_hq[0, :, i50].detach().cpu().numpy()
                        q90 = out_hq[0, :, i90].detach().cpu().numpy()
                        return {"point": _slice_or_pad(q50),
                                "q": {"q10": _slice_or_pad(q10), "q50": _slice_or_pad(q50), "q90": _slice_or_pad(q90)}}
                    else:
                        q50 = out_hq[0].median(dim=-1).values.detach().cpu().numpy()
                        return {"point": _slice_or_pad(q50)}

                # Hm < horizon 이면 IMS로 확장
                q10, q50, q90 = _roll_quantile_ims(
                    model, x1, horizon=horizon, device=device,
                    future_exo_cb=future_exo_cb, future_exo_batch=future_exo_batch,
                    part_ids=part_ids, past_exo_cont=past_exo_cont, past_exo_cat=past_exo_cat,
                    target_channel=0, fill_mode="copy_last",
                )
                q10 = q10.squeeze(0).detach().cpu().numpy()
                q50 = q50.squeeze(0).detach().cpu().numpy()
                q90 = q90.squeeze(0).detach().cpu().numpy()
                return {"point": q50, "q": {"q10": q10, "q50": q50, "q90": q90}}

            # (b) 'point'만 있는 dict
            if "point" in out and torch.is_tensor(out["point"]):
                point = out["point"].squeeze(0).detach().cpu().numpy().reshape(-1)
                if point.size >= horizon:
                    return {"point": point[:horizon]}
                # 부족하면 IMS/DMS 확장 (현재 DMSForecaster는 future_exo_cb만 사용)
                f = DMSForecaster(model, target_channel=0, fill_mode="copy_last",
                                  lmm_mode="eval", predict_fn=None, ttm=None, future_exo_cb=future_exo_cb)
                with torch.no_grad():
                    y_hat = f.forecast_DMS_to_IMS(
                        x_init=x1_dev, horizon=horizon, device=device,
                        extend="ims", context_policy="once"
                    )
                return {"point": y_hat.squeeze(0).detach().cpu().numpy()}

        # 2) 3D(분위수) — Tensor 직접 반환 케이스
        if torch.is_tensor(out) and out.dim() == 3:
            B, A, Bdim = out.shape
            if Bdim in (3, 5, 9):
                out_hq = out  # (B,Hm,Q)
            elif A in (3, 5, 9):
                out_hq = out.transpose(1, 2)  # (B,Hm,Q)
            else:
                q50 = out.mean(dim=1).squeeze(0).detach().cpu().numpy()
                q50 = q50[:horizon] if q50.size >= horizon else np.pad(np.asarray(q50, dtype=float), (0, horizon - q50.size), mode="constant", constant_values=np.nan)
                return {"point": q50}

            Hm = out_hq.shape[1]
            def _slice_or_pad(v):
                v = np.asarray(v, dtype=float)
                return v[:horizon] if v.size >= horizon else np.pad(v, (0, horizon - v.size), mode="constant", constant_values=np.nan)

            if Hm >= horizon:
                Qn = out_hq.size(-1)
                if Qn == 3:
                    i10, i50, i90 = (0, 1, 2)
                elif Qn == 5:
                    i10, i50, i90 = (1, 2, 3)
                elif Qn == 9:
                    i10, i50, i90 = (1, 4, 7)
                else:
                    i10 = i50 = i90 = None

                if i50 is not None:
                    q10 = out_hq[0, :, i10].detach().cpu().numpy()
                    q50 = out_hq[0, :, i50].detach().cpu().numpy()
                    q90 = out_hq[0, :, i90].detach().cpu().numpy()
                    return {"point": _slice_or_pad(q50),
                            "q": {"q10": _slice_or_pad(q10), "q50": _slice_or_pad(q50), "q90": _slice_or_pad(q90)}}
                else:
                    q50 = out_hq[0].median(dim=-1).values.detach().cpu().numpy()
                    return {"point": _slice_or_pad(q50)}

            # Hm < horizon → IMS 롤링
            q10, q50, q90 = _roll_quantile_ims(
                model, x1, horizon=horizon, device=device,
                future_exo_cb=future_exo_cb, future_exo_batch=future_exo_batch,
                part_ids=part_ids, past_exo_cont=past_exo_cont, past_exo_cat=past_exo_cat,
                target_channel=0, fill_mode="copy_last",
            )
            q10 = q10.squeeze(0).detach().cpu().numpy()
            q50 = q50.squeeze(0).detach().cpu().numpy()
            q90 = q90.squeeze(0).detach().cpu().numpy()
            return {"point": q50, "q": {"q10": q10, "q50": q50, "q90": q90}}

        # 3) 2D(Direct 포인트)
        if torch.is_tensor(out) and out.dim() == 2:
            point = out.squeeze(0).detach().cpu().numpy()
            if point.size >= horizon:
                return {"point": point[:horizon]}
            # IMS로 연장 (DMSForecaster: 현재는 future_exo_cb만 사용)
            f = DMSForecaster(model, target_channel=0, fill_mode="copy_last",
                              lmm_mode="eval", predict_fn=None, ttm=None, future_exo_cb=future_exo_cb)
            with torch.no_grad():
                y_hat = f.forecast_DMS_to_IMS(
                    x_init=x1_dev, horizon=horizon, device=device,
                    extend="ims", context_policy="once"
                )
            return {"point": y_hat.squeeze(0).detach().cpu().numpy()}

        # 4) 기타 출력 → IMS 경로
        f = DMSForecaster(model, target_channel=0, fill_mode="copy_last",
                          lmm_mode="eval", predict_fn=None, ttm=None, future_exo_cb=future_exo_cb)
        with torch.no_grad():
            y_hat = f.forecast_DMS_to_IMS(
                x_init=x1_dev, horizon=horizon, device=device,
                extend="ims", context_policy="once"
            )
        return {"point": y_hat.squeeze(0).detach().cpu().numpy()}

    finally:
        if was_training:
            model.train()


# ==============================
# Core plotting
# ==============================
def _plot_single_series(
    *,
    hist: Optional[np.ndarray],
    y_true: Optional[np.ndarray],
    preds_point: Dict[str, np.ndarray],
    preds_q10: Dict[str, np.ndarray],
    preds_q50: Dict[str, np.ndarray],
    preds_q90: Dict[str, np.ndarray],
    horizon: int,
    title: str,
    out_path: Optional[str],
    show: bool,
    zoom_future: bool = False,
    zoom_len: Optional[int] = None,
):
    """
    한 파트에 대해 히스토리 + 여러 모델의 예측을 그린다.
    - horizon 길이를 기준으로 x-축(1..H)을 맞춘다.
    - zoom_future=True 이면 미래 구간 일부만(예: 27) 확대하여 그린다.
    """
    t_hist = np.arange(-len(hist) + 1, 1) if (hist is not None and hist.size > 0) else None
    t_fut = np.arange(1, horizon + 1)

    plt.figure(figsize=(12, 5))

    # history
    if hist is not None and hist.size > 0:
        plt.plot(t_hist, hist, label="History", linewidth=2, alpha=0.8)

    # ground truth (있다면)
    if y_true is not None:
        yt = np.asarray(y_true, float).reshape(-1)
        if yt.size > horizon:
            yt = yt[:horizon]
        elif yt.size < horizon:
            yt = np.concatenate([yt, np.full(horizon - yt.size, np.nan)])
        if zoom_future:
            zL = int(zoom_len or horizon)
            zL = max(1, min(zL, horizon))
            plt.plot(t_fut[:zL], yt[:zL], label="True", linewidth=2)
        else:
            plt.plot(t_fut, yt, label="True", linewidth=2)

    # quantile (있다면)
    for nm in list(preds_q50.keys()):
        q10 = np.asarray(preds_q10.get(nm))
        q50 = np.asarray(preds_q50.get(nm))
        q90 = np.asarray(preds_q90.get(nm))
        if q10 is None or q50 is None or q90 is None:
            continue

        def _fit(a):
            a = a.reshape(-1)
            if a.size > horizon:
                return a[:horizon]
            if a.size < horizon:
                return np.concatenate([a, np.full(horizon - a.size, np.nan)])
            return a

        q10, q50, q90 = _fit(q10), _fit(q50), _fit(q90)
        if zoom_future:
            zL = int(zoom_len or horizon); zL = max(1, min(zL, horizon))
            plt.fill_between(t_fut[:zL], q10[:zL], q90[:zL], alpha=0.15, label=f"{nm} P10–P90")
            plt.plot(t_fut[:zL], q50[:zL], linewidth=1.8, alpha=0.95, label=f"{nm} P50")
        else:
            plt.fill_between(t_fut, q10, q90, alpha=0.15, label=f"{nm} P10–P90")
            plt.plot(t_fut, q50, linewidth=1.8, alpha=0.95, label=f"{nm} P50")

    # point-only models
    for nm, yhat in preds_point.items():
        if nm in preds_q50:  # 중앙선 중복 회피
            continue
        a = np.asarray(yhat).reshape(-1)
        if a.size > horizon:
            a = a[:horizon]
        elif a.size < horizon:
            a = np.concatenate([a, np.full(horizon - a.size, np.nan)])
        if zoom_future:
            zL = int(zoom_len or horizon); zL = max(1, min(zL, horizon))
            plt.plot(t_fut[:zL], a[:zL], label=nm, alpha=0.9)
        else:
            plt.plot(t_fut, a, label=nm, alpha=0.9)

    # 간단 앙상블 (q90 기반)
    stack = []
    for nm in preds_point.keys():
        base = preds_q90.get(nm, preds_point[nm])
        base = np.asarray(base).reshape(-1)
        if base.size > horizon:
            base = base[:horizon]
        elif base.size < horizon:
            base = np.concatenate([base, np.full(horizon - base.size, np.nan)])
        stack.append(base)
    if stack:
        M = np.vstack(stack)
        ens_q90 = np.nanmean(M, axis=0)
        if zoom_future:
            zL = int(zoom_len or horizon); zL = max(1, min(zL, horizon))
            plt.plot(t_fut[:zL], ens_q90[:zL], linewidth=2.8, alpha=0.95, label="Ensemble (q90-based)")
        else:
            plt.plot(t_fut, ens_q90, linewidth=2.8, alpha=0.95, label="Ensemble (q90-based)")

    plt.axvline(0, color="gray", linewidth=1, alpha=0.6)
    plt.title(title)
    plt.xlabel("Time (history ≤ 0 < future)")
    plt.ylabel("Demand")
    plt.legend(ncol=2)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


# ==============================
# Batch unpack helper (VAL / INFER 공용)
# ==============================
def _unpack_plot_batch(batch, mode: str):
    """
    Returns: xb, yb, part_ids, fe_cont, pe_cont, pe_cat
    지원 형식:
      - VAL  : (x,y), (x,y,part), (x,y,part,fe,peC,peK), (x,y,part,fe,peC)
      - INFER: (x,part), (x,part,fe,peC,peK)
    """
    xb = yb = part_ids = fe_cont = pe_cont = pe_cat = None
    if mode == "val":
        if len(batch) == 2:
            xb, yb = batch
        elif len(batch) == 3:
            xb, yb, part_ids = batch
        elif len(batch) == 6:
            xb, yb, part_ids, fe_cont, pe_cont, pe_cat = batch
        elif len(batch) == 5:
            xb, yb, part_ids, fe_cont, pe_cont = batch
        else:
            raise ValueError(f"val batch unsupported shape: {len(batch)}")
    else:
        if len(batch) == 2:
            xb, part_ids = batch
        elif len(batch) == 5:
            xb, part_ids, fe_cont, pe_cont, pe_cat = batch
        else:
            raise ValueError(f"infer batch unsupported shape: {len(batch)}")
    return xb, yb, part_ids, fe_cont, pe_cont, pe_cat


# ==============================
# Unified executors (VAL / INFER)
# ==============================
@torch.no_grad()
def _run_and_plot_many(
    *,
    models: Dict[str, torch.nn.Module],
    loader,
    device: str = "cuda" if torch.cuda.is_available() else 'cpu',
    horizon: int,
    mode: str,                      # 'val' | 'infer'
    plan_dt: Optional[int] = None,  # anchor label (YYYYMM or YYYYWW)
    granularity: str = "month",     # 'month' | 'week'
    max_plots: int = 100,
    out_dir: Optional[str] = None,
    show: bool = True,
    future_exo_cb=None,
    truth_cb: Optional[Callable[[str, int, int, str], Optional[np.ndarray]]] = None,
    zoom_future: bool = False,
    zoom_len: Optional[int] = None,
):
    """
    단일 엔진:
      - mode='val'  : (xb, yb[, ...]) 배치에서 y_true와 함께 플롯
      - mode='infer': (xb, part_ids[, ...]) 배치에서 히스토리 + 예측만 플롯
    plan_dt가 있으면 타이틀에 앵커 표기.
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plotted = 0
    for batch in loader:
        xb, yb, part_ids, fe_cont, pe_cont, pe_cat = _unpack_plot_batch(batch, mode)

        if xb.dim() == 2:
            xb = xb.unsqueeze(-1)  # (B,L)->(B,L,1)

        B = xb.size(0)
        for i in range(B):
            if plotted >= max_plots:
                return
            x1  = xb[i:i+1].to(device)
            pid = (part_ids[i] if (part_ids is not None and i < len(part_ids))
                   else f"idx{i}")

            # 단일 샘플 외생 슬라이스
            fe1 = fe_cont[i:i+1] if (isinstance(fe_cont, torch.Tensor)) else None
            pc1 = pe_cont[i:i+1] if (isinstance(pe_cont, torch.Tensor)) else None
            pk1 = pe_cat[i:i+1]  if (isinstance(pe_cat,  torch.Tensor)) else None
            pid1 = None
            if isinstance(pid, torch.Tensor):
                pid1 = pid.unsqueeze(0) if pid.dim() == 0 else pid
            elif isinstance(part_ids, torch.Tensor):
                pid1 = part_ids[i:i+1]

            # y_true 준비
            y_true = None
            if mode == "val" and (yb is not None):
                y_true = yb[i:i+1].detach().cpu().numpy().reshape(-1)
                if y_true.size > horizon:
                    y_true = y_true[:horizon]
            elif (mode == "infer") and (truth_cb is not None) and (plan_dt is not None):
                y_true = truth_cb(pid, plan_dt, horizon, granularity)
                if y_true is not None:
                    y_true = np.asarray(y_true, float).reshape(-1)
                    if y_true.size > horizon:
                        y_true = y_true[:horizon]
                    elif y_true.size < horizon:
                        y_true = np.concatenate([y_true, np.full(horizon - y_true.size, np.nan)])

            # 각 모델 예측 수집
            preds_point, preds_q10, preds_q50, preds_q90 = {}, {}, {}, {}
            for name, mdl in models.items():

                p = _predict_any(
                    mdl, x1, device=device, future_exo_cb=future_exo_cb, horizon=horizon,
                    part_ids=pid1,
                    future_exo_batch=fe1,
                    past_exo_cont=pc1,
                    past_exo_cat=pk1,
                )
                preds_point[name] = p["point"]
                if "q" in p:
                    preds_q10[name] = p["q"].get("q10")
                    preds_q50[name] = p["q"].get("q50")
                    preds_q90[name] = p["q"].get("q90")

            # DEBUG
            if DEBUG_FCAST and horizon == 27:
                for nm, yhat in preds_point.items():
                    _log_var(f"{nm} point(H=27)", yhat[:27])

            # 타이틀
            if plan_dt is not None:
                title = f"[{mode.upper()}:{granularity}] H={horizon} from {plan_dt} – part: {pid}"
            else:
                title = f"[{mode.upper()}] H={horizon} – part: {pid}"

            # 플롯
            hist = _to_1d_history(x1)
            out_path = (os.path.join(out_dir, f"{mode}_{granularity}_H{horizon}_{pid}.png")
                        if out_dir else None)
            _plot_single_series(
                hist=hist,
                y_true=y_true,
                preds_point=preds_point,
                preds_q10=preds_q10,
                preds_q50=preds_q50,
                preds_q90=preds_q90,
                horizon=horizon,
                title=title,
                out_path=out_path,
                show=show,
                zoom_future=zoom_future,
                zoom_len=zoom_len,
            )
            plotted += 1


# ==============================
# Public API (27-week / 120-month)
# ==============================
@torch.no_grad()
def plot_27w(
    models: Dict[str, torch.nn.Module],
    loader,
    *,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    mode: str = "val",                # 'val' | 'infer'
    plan_yyyyww: Optional[int] = None,
    max_plots: int = 100,
    out_dir: Optional[str] = None,
    show: bool = True,
    future_exo_cb=None,
    truth_cb: Optional[Callable[[str, int, int, str], Optional[np.ndarray]]] = None,
):
    """
    27주 예측 전용 플로터.
      - mode='val'이면 (xb, yb[, part_ids, fe, peC, peK]) 배치에서 y_true와 함께 그림.
      - mode='infer'이면 (xb, part_ids[, fe, peC, peK]) 배치에서 히스토리 + 예측만 표시
    """
    _run_and_plot_many(
        models=models,
        loader=loader,
        device=device,
        horizon=27,
        mode=mode,
        plan_dt=plan_yyyyww,
        granularity="week",
        max_plots=max_plots,
        out_dir=out_dir,
        show=show,
        future_exo_cb=future_exo_cb,
        truth_cb=truth_cb,
        zoom_future=True,
        zoom_len=27,
    )


@torch.no_grad()
def plot_120m(
    models: Dict[str, torch.nn.Module],
    loader,
    *,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    mode: str = "val",                # 'val' | 'infer'
    plan_yyyymm: Optional[int] = None,
    max_plots: int = 100,
    out_dir: Optional[str] = None,
    show: bool = True,
    future_exo_cb=None,
    truth_cb: Optional[Callable[[str, int, int, str], Optional[np.ndarray]]] = None,
):
    """
    120개월 예측 전용 플로터.
      - mode='val'이면 (xb, yb[, part_ids, fe, peC, peK]) 배치에서 y_true와 함께 그림.
      - mode='infer'이면 (xb, part_ids[, fe, peC, peK]) 배치에서 히스토리 + 예측만 표시
    """
    _run_and_plot_many(
        models=models,
        loader=loader,
        device=device,
        horizon=120,
        mode=mode,
        plan_dt=plan_yyyymm,
        granularity="month",
        max_plots=max_plots,
        out_dir=out_dir,
        show=show,
        future_exo_cb=future_exo_cb,
        truth_cb=truth_cb,
        zoom_future=False,   # 월 120은 전체 보기 기본
        zoom_len=None,
    )