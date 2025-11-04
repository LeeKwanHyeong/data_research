from typing import Dict, Optional
import numpy as np
import torch.nn as nn
from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfigMonthly, PatchMixerConfigWeekly
)
from modeling_module.models.PatchTST.common.configs import PatchTSTConfigMonthly
from modeling_module.models.Titan.common.configs import TitanConfigMonthly, TitanConfigPatchMonthly, \
    TitanConfigPatchWeekly
from modeling_module.models.model_builder import (
    build_patch_mixer_base, build_patch_mixer_quantile,
    build_titan_base, build_titan_lmm, build_patchTST_base, build_titan_seq2seq, build_titan_patch
)
from modeling_module.training.metrics import quantile_metrics
from modeling_module.training.model_trainers.patchmixer_train import train_patchmixer
from modeling_module.training.model_trainers.patchtst_train import train_patchtst
from modeling_module.training.model_trainers.titan_train import train_titan
from modeling_module.utils.metrics import mae, rmse, smape

from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb


# --- (중요) 모델이 외생(feature)을 실제 사용할 수 있게 보장하는 헬퍼 ---
def _ensure_exo_head(model, exo_dim: int = 2):
    """
    모델 내부에 exo_head / exo_dim 이 없거나 exo_dim==0 이면 간단한 linear head를 부착.
    (model_builder.py는 그대로 두고 여기서만 보강)
    """
    # 이미 지원하면 끝
    if getattr(model, "exo_dim", 0) >= exo_dim and getattr(model, "exo_head", None) is not None:
        return model

    # 붙일 수 있는지 확인
    if hasattr(model, "exo_dim"):
        model.exo_dim = int(exo_dim)
    else:
        # 속성 추가(파이토치 모듈에 동적 부착 OK)
        setattr(model, "exo_dim", int(exo_dim))

    if getattr(model, "exo_head", None) is None:
        # 간단한 2-layer MLP (B,H,D) -> (B,H,1)
        model.exo_head = nn.Sequential(
            nn.Linear(exo_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    return model


def _make_calendar_cb_from_cfg(cfg) -> Optional[callable]:
    """
    cfg.date_type ('M'|'W' 등)에 맞춰 sin/cos 캘린더 외생 콜백 생성.
    - 주간: 52 주기, 월간: 12 주기 등은 compose_exo_calendar_cb 내부에서 처리.
    """
    try:
        return compose_exo_calendar_cb(date_type=cfg.date_type)
    except Exception:
        # date_type이 없거나 에러면 None 반환(외생 비사용)
        return None



def run_total_train_weekly(train_loader, val_loader, device='cuda', *, lookback, horizon):
    results: Dict[str, Dict] = {}

    # ---------------- PatchMixer ----------------
    # pm_base_config = PatchMixerConfigWeekly(
    #     lookback=lookback,
    #     horizon=horizon,
    #     device=device,
    #     loss_mode='point',
    #     point_loss='mae'
    # )
    # pm_quantile_config = PatchMixerConfigWeekly(
    #     lookback=lookback,
    #     horizon=horizon,
    #     device=device,
    #     loss_mode='quantile',
    #     quantiles=(0.1, 0.5, 0.9)
    # )
    #
    # # 외생 콜백(주간: 52 주기 sin/cos)
    # future_exo_cb = _make_calendar_cb_from_cfg(pm_base_config)
    #
    # # 모델 생성 (model_builder는 그대로 유지)
    # pm_base_model = build_patch_mixer_base(pm_base_config)
    # pm_quantile_model = build_patch_mixer_quantile(pm_quantile_config)
    #
    # # 모델이 외생을 실제로 쓸 수 있게 exo_head를 보강(필요 시)
    # if future_exo_cb is not None:
    #     _ensure_exo_head(pm_base_model, exo_dim=2)
    #     _ensure_exo_head(pm_quantile_model, exo_dim=2)
    #
    # print(f"[EXO] base exo_dim={getattr(pm_base_model, 'exo_dim', 0)} "
    #       f"exo_head? {hasattr(pm_base_model, 'exo_head') and pm_base_model.exo_head is not None}")
    # print(f"[EXO] qmdl exo_dim={getattr(pm_quantile_model, 'exo_dim', 0)} "
    #       f"exo_head? {hasattr(pm_quantile_model, 'exo_head') and pm_quantile_model.exo_head is not None}")
    #
    # print('PatchMixer Base (Weekly)')
    # best_pm_base = train_patchmixer(
    #     pm_base_model,
    #     train_loader, val_loader,
    #     lr=1e-3,
    #     loss_mode='point',
    #     point_loss='mae',
    #     quantiles=(0.1, 0.5, 0.9),
    #     use_intermittent=True,
    #     future_exo_cb=future_exo_cb,   # ← 트레이너로 콜백 전달
    #     exo_is_normalized=True         # RevIN 공간에서 가산하는 구조라면 True
    # )
    # results['PatchMixer Base'] = best_pm_base
    #
    # print('PatchMixer Quantile (Weekly)')
    # best_pm_quantile = train_patchmixer(
    #     pm_quantile_model,
    #     train_loader, val_loader,
    #     lr=1e-3,
    #     loss_mode='quantile',
    #     quantiles=(0.1, 0.5, 0.9),
    #     use_intermittent=True,
    #     future_exo_cb=future_exo_cb,
    #     exo_is_normalized=True
    # )
    # results['PatchMixer Quantile'] = best_pm_quantile

    # ---------------- Titan (point + TTA) ----------------
    ti_config = TitanConfigPatchWeekly(
        device = device,
        lookback = lookback,
        horizon = horizon,
        loss_mode = 'point',
        point_loss = 'huber'
    )

    ti_patch_config = TitanConfigPatchWeekly(
        device = device,
        lookback = lookback,
        horizon = horizon,
        loss_mode = 'point',
        point_loss = 'huber'
    )

    ti_base = build_titan_base(ti_config)
    ti_lmm = build_titan_lmm(ti_config)
    ti_seq2seq = build_titan_seq2seq(ti_config)
    ti_patch = build_titan_patch(ti_patch_config)

    print('Titan Base')
    best_ti_base = train_titan(
        ti_base,
        train_loader, val_loader,
        lr = 1e-3, loss_mode = 'point', tta_steps = 3
    )
    results['Titan Base'] = best_ti_base

    print('Titan LMM')
    best_ti_lmm = train_titan(
        ti_lmm,
        train_loader, val_loader,
        lr = 1e-3, loss_mode = 'point', tta_steps = 3
    )
    results['Titan LMM'] = best_ti_lmm

    print('Titan Seq2Seq')
    best_ti_seq2seq = train_titan(
        ti_seq2seq,
        train_loader, val_loader,
        lr = 1e-3, loss_mode = 'point', tta_steps = 3
    )
    results['Titan Seq2Seq'] = best_ti_seq2seq

    print('Titan Patch')
    best_ti_patch = train_titan(
        ti_patch,
        train_loader, val_loader,
        lr = 1e-3, loss_mode = 'point', tta_steps = 3
    )
    results['Titan Patch'] = best_ti_patch


    return results


def run_total_train_monthly(train_loader, val_loader, device='cuda', *, lookback, horizon):
    results = {}

    # ---------------- PatchMixer ----------------
    pm_base_config = PatchMixerConfigMonthly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='point',
        point_loss='mae'
    )
    pm_quantile_config = PatchMixerConfigMonthly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9)
    )

    # 외생 콜백(월간: 12 주기 sin/cos)
    future_exo_cb = _make_calendar_cb_from_cfg(pm_base_config)

    pm_base_model = build_patch_mixer_base(pm_base_config)
    pm_quantile_model = build_patch_mixer_quantile(pm_quantile_config)

    if future_exo_cb is not None:
        _ensure_exo_head(pm_base_model, exo_dim=2)
        _ensure_exo_head(pm_quantile_model, exo_dim=2)

    print('PatchMixer Base (Monthly)')
    best_pm_base = train_patchmixer(
        pm_base_model,
        train_loader, val_loader,
        lr=1e-3,
        loss_mode='point',
        point_loss='mae',
        quantiles=(0.1, 0.5, 0.9),
        use_intermittent=True,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=True
    )
    results['PatchMixer Base'] = best_pm_base

    print('PatchMixer Quantile (Monthly)')
    best_pm_quantile = train_patchmixer(
        pm_quantile_model,
        train_loader, val_loader,
        lr=1e-3,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9),
        use_intermittent=True,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=True
    )
    results['PatchMixer Quantile'] = best_pm_quantile

    # # ---------------- Titan (point + TTA) ----------------
    # ti_config = TitanConfigMonthly(
    #     device = device,
    #     lookback = lookback,
    #     horizon = horizon,
    #     loss_mode = 'point',
    #     point_loss = 'huber'
    # )
    #
    # ti_patch_config = TitanConfigPatchMonthly(
    #     device = device,
    #     lookback = lookback,
    #     horizon = horizon,
    #     loss_mode = 'point',
    #     point_loss = 'huber'
    # )
    #
    # ti_base = build_titan_base(ti_config)
    # ti_lmm = build_titan_lmm(ti_config)
    # ti_seq2seq = build_titan_seq2seq(ti_config)
    # ti_patch = build_titan_patch(ti_patch_config)
    #
    # print('Titan Base')
    # best_ti_base = train_titan(
    #     ti_base,
    #     train_loader, val_loader,
    #     lr = 1e-3, loss_mode = 'point', tta_steps = 3
    # )
    # results['Titan Base'] = best_ti_base
    #
    # print('Titan LMM')
    # best_ti_lmm = train_titan(
    #     ti_lmm,
    #     train_loader, val_loader,
    #     lr = 1e-3, loss_mode = 'point', tta_steps = 3
    # )
    # results['Titan LMM'] = best_ti_lmm
    #
    # print('Titan Seq2Seq')
    # best_ti_seq2seq = train_titan(
    #     ti_seq2seq,
    #     train_loader, val_loader,
    #     lr = 1e-3, loss_mode = 'point', tta_steps = 3
    # )
    # results['Titan Seq2Seq'] = best_ti_seq2seq
    #
    # print('Titan Patch')
    # best_ti_patch = train_titan(
    #     ti_patch,
    #     train_loader, val_loader,
    #     lr = 1e-3, loss_mode = 'point', tta_steps = 3
    # )
    # results['Titan Patch'] = best_ti_patch
    #
    # # ---------------- PatchTST(Quantile + point) ----------------
    # pt_config = PatchTSTConfigMonthly(
    #     device = device,
    #     lookback=lookback,
    #     horizon=horizon,
    #     loss_mode = 'auto',
    #     quantiles = (0.1, 0.5, 0.9)
    # )
    #
    # pt_base = build_patchTST_base(pt_config)
    #
    # print('PatchTST Base')
    # best_pt_base = train_patchtst(
    #     pt_base,
    #     train_loader, val_loader,
    #     lr = 1e-3, loss_mode = 'auto', use_intermittent = True
    # )
    # results['PatchTST Base'] = best_pt_base
    #
    #
    # print('PatchTST Quantile')
    # best_pt_quantile = train_patchtst(
    #     pt_base,
    #     train_loader, val_loader,
    #     lr = 1e-3, loss_mode = 'quantile', use_intermittent = True
    # )
    # results['PatchTST Quantile'] = best_pt_quantile

    return results


def summarize_metrics(results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    table = {}
    for name, res in results.items():
        y = res['y_true'].reshape(-1)
        yhat = res['y_pred'].reshape(-1)

        row = {
            'MAE': mae(y, yhat),
            'RMSE': rmse(y, yhat),
            'SMAPE': smape(y, yhat),
        }

        # q_pred가 dict 형태 {0.1: ..., 0.5: ..., 0.9: ...}일 때만 구간지표 계산
        if res.get('q_pred') is not None and 0.1 in res['q_pred'] and 0.9 in res['q_pred']:
            result = quantile_metrics(y, yhat)
            row['converage_per_q'] = result['coverage_per_q']
            row['i80_cov'] = result['i80_cov']
            row['i80_wid'] = result['i80_wid']

        table[name] = row

    return table
