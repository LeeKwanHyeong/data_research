from typing import Dict, Tuple, Optional
from pathlib import Path
from dataclasses import asdict, is_dataclass, replace

import numpy as np
import torch

from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfigMonthly,
    PatchMixerConfig,
    PatchMixerConfigWeekly,
)
# PatchMixerConfigDaily, Hourly는 별도 클래스 없이 PatchMixerConfig + kwargs로 처리

from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.models.model_builder import (
    build_titan_base,
    build_titan_lmm,
    build_titan_seq2seq,
    build_patch_mixer_base,
    build_patch_mixer_quantile,
    build_patchTST_base,
    build_patchTST_quantile,
)
from modeling_module.training.config import SpikeLossConfig, TrainingConfig, StageConfig
from modeling_module.training.metrics import quantile_metrics
from modeling_module.training.model_trainers.patchmixer_train import train_patchmixer
from modeling_module.training.model_trainers.patchtst_train import train_patchtst
from modeling_module.training.model_trainers.titan_train import train_titan
from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb
from modeling_module.utils.metrics import smape, rmse, mae


# ===================== 공통 유틸 =====================

def _get_part_vocab_size_from_loader(loader) -> int:
    try:
        return len(getattr(loader.dataset, "part_vocab", {}))
    except Exception:
        return 0


def save_model(model: torch.nn.Module, cfg, path: str) -> None:
    path = str(path)
    state = {
        "model_state": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    if cfg is not None:
        if is_dataclass(cfg):
            state["config"] = asdict(cfg)
        else:
            cfg_dict = getattr(cfg, "__dict__", None)
            if cfg_dict is not None:
                state["config"] = dict(cfg_dict)
            else:
                state["config"] = cfg
    torch.save(state, path)


def _make_ckpt_path(
        save_dir: Path,
        freq: str,
        model_name: str,
        lookback: int,
        horizon: int,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{freq}_{model_name}_L{lookback}_H{horizon}.pt"
    return save_dir / fname


def _build_common_train_configs(
        *,
        device: str,
        lookback: int,
        horizon: int,
        freq: str = 'weekly'
) -> Tuple[TrainingConfig, TrainingConfig, SpikeLossConfig, Tuple[StageConfig, StageConfig]]:
    # 주기별 학습률/에폭 튜닝
    if freq == 'hourly':
        base_lr = 1e-4
        warmup_epochs = 20
        spike_epochs = 50
    elif freq == 'daily':
        base_lr = 2e-4
        warmup_epochs = 30
        spike_epochs = 70
    else:  # weekly, monthly
        base_lr = 3e-4
        warmup_epochs = 50
        spike_epochs = 100

    # 1) 2-stage 학습 스케줄
    stg_warmup = StageConfig(
        epochs=warmup_epochs,
        spike_enabled=False,
        lr=base_lr,
        use_horizon_decay=False,
    )

    stg_spike = StageConfig(
        epochs=spike_epochs,
        spike_enabled=True,
        lr=base_lr / 3,
        use_horizon_decay=True,
        tau_h=0.85,
    )
    stages = (stg_warmup, stg_spike)

    # 2) Spike loss 공통 설정
    spike_cfg = SpikeLossConfig(
        enabled=True,
        strategy='mix',
        huber_delta=0.6,
        asym_up_weight=1.0,
        asym_down_weight=8.0,
        mad_k=1.5,
        w_spike=32.0,
        w_norm=1.0,
        alpha_huber=0.6,
        beta_asym=0.4,
        mix_with_baseline=False,
        gamma_baseline=0.0,
    )

    # 3) Point용 공통 TrainingConfig
    point_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        lr=base_lr,
        weight_decay=1e-3,
        t_max=40,
        patience=100,
        loss_mode='point',
        point_loss='huber',
        huber_delta=0.8,
        use_intermittent=True,
        alpha_zero=3.0,
        alpha_pos=1.0,
        gamma_run=0.3,
        use_horizon_decay=False,
        tau_h=0.85,
        val_use_weights=False,
        spike_loss=spike_cfg,
        max_grad_norm=30.0,
    )

    # 4) Quantile용 공통 TrainingConfig
    quantile_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        lr=base_lr,
        weight_decay=1e-4,
        t_max=40,
        patience=20,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9),
        use_intermittent=True,
        alpha_zero=1.2,
        alpha_pos=1.0,
        gamma_run=0.6,
        use_horizon_decay=False,
        tau_h=0.85,
        val_use_weights=False,
        spike_loss=spike_cfg,
        max_grad_norm=30.0,
    )

    return point_train_cfg, quantile_train_cfg, spike_cfg, stages


# ===================== GENERIC RUNNER (통합) =====================

def _run_total_train_generic(
        train_loader,
        val_loader,
        device: str,
        lookback: int,
        horizon: int,
        freq: str,  # 'monthly', 'weekly', 'daily', 'hourly'
        save_dir: Optional[str]
) -> Dict[str, Dict]:
    save_root = Path(save_dir) if save_dir is not None else None

    # 1. Config & Exo Callback 설정
    point_train_cfg, quantile_train_cfg, spike_cfg, stages = _build_common_train_configs(
        device=device, lookback=lookback, horizon=horizon, freq=freq
    )

    # date_type 매핑
    date_type_map = {'weekly': 'W', 'monthly': 'M', 'daily': 'D', 'hourly': 'H'}
    dt_char = date_type_map.get(freq, 'W')
    future_exo_cb = compose_exo_calendar_cb(date_type=dt_char)

    # 2. Exo Dimension 결정 (Sin/Cos 인코딩 기준)
    # Monthly/Weekly: period=1 (sin, cos) -> 2
    # Daily: period=2 (dow, doy) -> 4
    # Hourly: period=2 (hour, hour_of_week) -> 4
    if freq in ('daily', 'hourly'):
        exo_dim = 4
    else:
        exo_dim = 2

    results: Dict[str, Dict] = {}

    # --------------------------------------------------
    # 1) PatchMixer
    # --------------------------------------------------
    # 주기별 Patch Len / Stride 추천값
    if freq == 'hourly':
        patch_len, stride = 24, 12
        season_period = 24
    elif freq == 'daily':
        patch_len, stride = 14, 7
        season_period = 7
    elif freq == 'weekly':
        patch_len, stride = 12, 8
        season_period = 52
    else:  # monthly
        patch_len, stride = 6, 3
        season_period = 12

    # Config kwargs 구성
    pm_kwargs = dict(
        lookback=lookback,
        horizon=horizon,
        device=device,
        enc_in=1,
        d_model=64,
        e_layers=3,
        patch_len=patch_len,
        stride=stride,
        f_out=128,
        head_hidden=128,
        exo_dim=exo_dim,
        use_part_embedding=True,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,
        final_nonneg=True,
        use_eol_prior=False,
        exo_is_normalized_default=(freq != 'monthly'),  # 월간만 False 권장
        expander_season_period=season_period,
        expander_n_harmonics=min(season_period // 2, 16),
    )

    # Base
    pm_base_cfg = PatchMixerConfig(**pm_kwargs, loss_mode='point', point_loss='huber', head_dropout=0.05)
    pm_base_model = build_patch_mixer_base(pm_base_cfg)

    print(f'PatchMixer Base ({freq.capitalize()})')
    best_pm_base = train_patchmixer(
        pm_base_model, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_base_cfg.exo_is_normalized_default,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixerBase", lookback, horizon)
        save_model(pm_base_model, pm_base_cfg, ckpt_path)
        best_pm_base["ckpt_path"] = str(ckpt_path)
    results['PatchMixer Base'] = best_pm_base

    # Quantile
    pm_q_cfg = PatchMixerConfig(**pm_kwargs, loss_mode='quantile', quantiles=(0.1, 0.5, 0.9), head_dropout=0.02)
    pm_q_model = build_patch_mixer_quantile(pm_q_cfg)

    print(f'PatchMixer Quantile ({freq.capitalize()})')
    best_pm_q = train_patchmixer(
        pm_q_model, train_loader, val_loader,
        train_cfg=quantile_train_cfg, stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_q_cfg.exo_is_normalized_default,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixerQuantile", lookback, horizon)
        save_model(pm_q_model, pm_q_cfg, ckpt_path)
        best_pm_q["ckpt_path"] = str(ckpt_path)
    results['PatchMixer Quantile'] = best_pm_q

    # --------------------------------------------------
    # 2) Titan
    # --------------------------------------------------
    ti_config = TitanConfig(
        lookback=lookback,
        horizon=horizon,
        input_dim=1,
        d_model=256,
        n_layers=3,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        contextual_mem_size=256,
        persistent_mem_size=64,
        use_exogenous=True,
        exo_dim=exo_dim,
        final_clamp_nonneg=True,
        use_revin=True,
        revin_use_std=True,
        revin_subtract_last=False,
        revin_affine=True,
        use_lmm=False
    )
    if freq == 'hourly':
        ti_config.contextual_mem_size = 512

    # Titan Base
    ti_base = build_titan_base(ti_config)
    print(f'Titan Base ({freq.capitalize()})')
    best_ti_base = train_titan(
        ti_base, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanBase", lookback, horizon)
        save_model(ti_base, ti_config, ckpt_path)
        best_ti_base["ckpt_path"] = str(ckpt_path)
    results['Titan Base'] = best_ti_base

    # Titan LMM
    ti_config_lmm = replace(ti_config, use_lmm=True)
    ti_lmm = build_titan_lmm(ti_config_lmm)
    print(f'Titan LMM ({freq.capitalize()})')
    best_ti_lmm = train_titan(
        ti_lmm, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanLMM", lookback, horizon)
        save_model(ti_lmm, ti_config_lmm, ckpt_path)
        best_ti_lmm["ckpt_path"] = str(ckpt_path)
    results['Titan LMM'] = best_ti_lmm

    # Titan Seq2Seq
    ti_seq2seq = build_titan_seq2seq(ti_config)
    print(f'Titan Seq2Seq ({freq.capitalize()})')
    best_ti_s2s = train_titan(
        ti_seq2seq, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanSeq2Seq", lookback, horizon)
        save_model(ti_seq2seq, ti_config, ckpt_path)
        best_ti_s2s["ckpt_path"] = str(ckpt_path)
    results['Titan Seq2Seq'] = best_ti_s2s

    # --------------------------------------------------
    # 3) PatchTST
    # --------------------------------------------------
    pt_kwargs = dict(
        device=device,
        lookback=lookback,
        horizon=horizon,
        c_in=1,
        d_model=256,
        n_layers=3,
        patch_len=patch_len,
        stride=stride,
    )

    pt_base_cfg = PatchTSTConfig(**pt_kwargs, loss_mode='point', point_loss='huber')
    pt_base = build_patchTST_base(pt_base_cfg)

    print(f'PatchTST Base ({freq.capitalize()})')
    best_pt_base = train_patchtst(
        pt_base, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchTSTBase", lookback, horizon)
        save_model(pt_base, pt_base_cfg, ckpt_path)
        best_pt_base["ckpt_path"] = str(ckpt_path)
    results['PatchTST Base'] = best_pt_base

    pt_q_cfg = PatchTSTConfig(**pt_kwargs, loss_mode='quantile', quantiles=(0.1, 0.5, 0.9))
    pt_q = build_patchTST_quantile(pt_q_cfg)

    print(f'PatchTST Quantile ({freq.capitalize()})')
    best_pt_q = train_patchtst(
        pt_q, train_loader, val_loader,
        train_cfg=quantile_train_cfg, stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=True,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchTSTQuantile", lookback, horizon)
        save_model(pt_q, pt_q_cfg, ckpt_path)
        best_pt_q["ckpt_path"] = str(ckpt_path)
    results['PatchTST Quantile'] = best_pt_q

    return results


# ===================== EXPORTED FUNCTIONS =====================

def run_total_train_weekly(train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu', *, lookback,
                           horizon, save_dir=None):
    return _run_total_train_generic(train_loader, val_loader, device, lookback, horizon, 'weekly', save_dir)


def run_total_train_monthly(train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu', *,
                            lookback, horizon, save_dir=None):
    return _run_total_train_generic(train_loader, val_loader, device, lookback, horizon, 'monthly', save_dir)


def run_total_train_daily(train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu', *, lookback,
                          horizon, save_dir=None):
    return _run_total_train_generic(train_loader, val_loader, device, lookback, horizon, 'daily', save_dir)


def run_total_train_hourly(train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu', *, lookback,
                           horizon, save_dir=None):
    return _run_total_train_generic(train_loader, val_loader, device, lookback, horizon, 'hourly', save_dir)


def summarize_metrics(results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for name, res in results.items():
        y = res['y_true'].reshape(-1)
        yhat = res['y_pred'].reshape(-1)

        row: Dict[str, float] = {
            'MAE': mae(y, yhat),
            'RMSE': rmse(y, yhat),
            'SMAPE': smape(y, yhat),
        }
        if res.get('q_pred') is not None and 0.1 in res['q_pred'] and 0.9 in res['q_pred']:
            result = quantile_metrics(y, yhat)
            row['converage_per_q'] = result['coverage_per_q']
            row['i80_cov'] = result['i80_cov']
            row['i80_wid'] = result['i80_wid']
        table[name] = row
    return table