from typing import Dict, Tuple, Optional
from pathlib import Path
from dataclasses import asdict, is_dataclass

import numpy as np
import torch

from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfigMonthly,
    PatchMixerConfig,
    PatchMixerConfigWeekly,
)
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
    """
    통합 ckpt 저장 헬퍼.
    - model.state_dict()
    - config(dataclass면 asdict, 아니면 __dict__)
    - model class name
    """
    path = str(path)
    state = {
        "model_state": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    # config 직렬화
    if cfg is not None:
        if is_dataclass(cfg):
            state["config"] = asdict(cfg)
        else:
            # dataclass가 아니면, 최대한 안전하게 뽑기
            cfg_dict = getattr(cfg, "__dict__", None)
            if cfg_dict is not None:
                state["config"] = dict(cfg_dict)
            else:
                state["config"] = cfg
    torch.save(state, path)


def _make_ckpt_path(
    save_dir: Path,
    freq: str,          # 'weekly' or 'monthly'
    model_name: str,    # 'PatchMixerBase', 'TitanLMM', ...
    lookback: int,
    horizon: int,
) -> Path:
    """
    모델별 ckpt 파일명 생성 헬퍼.
    예:  <save_dir>/weekly_PatchMixerBase_L72_H36.pt
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{freq}_{model_name}_L{lookback}_H{horizon}.pt"
    return save_dir / fname


def _build_common_train_configs(
    *,
    device: str,
    lookback: int,
    horizon: int,
) -> Tuple[TrainingConfig, TrainingConfig, SpikeLossConfig, Tuple[StageConfig, StageConfig]]:
    """
    주간/월간 공통으로 쓰는 Stage, SpikeLoss, TrainingConfig를 생성하는 헬퍼.
    필요하면 주간/월간에서 override.
    """
    # 2-stage 학습 스케줄
    stg_warmup = StageConfig(
        epochs=1,
        spike_enabled=False,
        lr=3e-4,
        use_horizon_decay=False,
    )
    stg_spike = StageConfig(
        epochs=1,
        spike_enabled=True,
        lr=1e-4,
        use_horizon_decay=True,
        tau_h=0.85,
    )
    stages = (stg_warmup, stg_spike)

    # Spike loss 공통 설정
    spike_cfg = SpikeLossConfig(
        enabled=True,
        strategy='mix',
        huber_delta=0.6,
        asym_up_weight=1.0,
        asym_down_weight=8.0,  # 언더예측 페널티
        mad_k=1.5,
        w_spike=32.0,
        w_norm=1.0,
        alpha_huber=0.6,
        beta_asym=0.4,
        mix_with_baseline=False,
        gamma_baseline=0.0,
        # w_cap=12.0,  # 필요 시 사용
    )

    point_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        lr=3e-4,
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

    quantile_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        lr=3e-4,
        weight_decay=1e-4,
        t_max=40,
        patience=10,
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

# ===================== WEEKLY =====================

def run_total_train_weekly(
        train_loader,
        val_loader,
        device: str = 'cuda' if torch.cuda.is_available() else 'mps',
        *,
        lookback: int,
        horizon: int,
        save_dir: Optional[str] = None,   # ← ckpt 저장 루트 디렉토리
) -> Dict[str, Dict]:
    """
    주간 전체 모델(PatchMixer, Titan, PatchTST)을 학습시키고
    각 결과를 반환. save_dir가 주어지면 ckpt도 함께 저장.
    """
    save_root = Path(save_dir) if save_dir is not None else None

    # 공통 학습 설정
    point_train_cfg, quantile_train_cfg, spike_cfg, stages = _build_common_train_configs(
        device=device, lookback=lookback, horizon=horizon
    )

    # 주간 캘린더 exogenous (W)
    future_exo_cb = compose_exo_calendar_cb(date_type='W')

    results: Dict[str, Dict] = {}

    # --------------------------------------------------
    # PatchMixer (Weekly)
    # --------------------------------------------------
    weekly_exo_dim = 4
    use_eol = False

    pm_base_config = PatchMixerConfigWeekly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='point',
        point_loss='huber',

        enc_in=1,
        d_model=64,
        e_layers=3,
        patch_len=12,
        stride=8,
        f_out=128,
        head_hidden=128,
        head_dropout=0.05,

        exo_dim=weekly_exo_dim,
        use_part_embedding=True,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,

        final_nonneg=True,
        use_eol_prior=use_eol,
        eol_feature_index=0,

        exo_is_normalized_default=True,

        expander_season_period=52,
        expander_n_harmonics=16,
    )

    pm_quantile_config = PatchMixerConfigWeekly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9),

        enc_in=1,
        d_model=64,
        e_layers=3,
        patch_len=12,
        stride=8,
        f_out=128,
        head_hidden=128,
        head_dropout=0.02,

        exo_dim=weekly_exo_dim,
        use_part_embedding=True,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,

        final_nonneg=True,
        use_eol_prior=use_eol,
        eol_feature_index=0,

        exo_is_normalized_default=True,

        expander_season_period=52,
        expander_n_harmonics=8,
    )

    pm_base_model = build_patch_mixer_base(pm_base_config)
    pm_quantile_model = build_patch_mixer_quantile(pm_quantile_config)

    print('PatchMixer Base (Weekly)')
    best_pm_base = train_patchmixer(
        pm_base_model,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_base_config.exo_is_normalized_default,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "weekly", "PatchMixerBase", lookback, horizon)
        save_model(pm_base_model, pm_base_config, ckpt_path)
        best_pm_base["ckpt_path"] = str(ckpt_path)
    results['PatchMixer Base'] = best_pm_base

    print('PatchMixer Quantile (Weekly)')
    best_pm_quantile = train_patchmixer(
        pm_quantile_model,
        train_loader,
        val_loader,
        train_cfg=quantile_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_quantile_config.exo_is_normalized_default,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "weekly", "PatchMixerQuantile", lookback, horizon)
        save_model(pm_quantile_model, pm_quantile_config, ckpt_path)
        best_pm_quantile["ckpt_path"] = str(ckpt_path)
    results['PatchMixer Quantile'] = best_pm_quantile

    # --------------------------------------------------
    # Titan (Weekly)
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
        exo_dim=2,  # 주간 calendar sin/cos
        final_clamp_nonneg=True,
    )

    ti_base = build_titan_base(ti_config)
    ti_lmm = build_titan_lmm(ti_config)
    ti_seq2seq = build_titan_seq2seq(ti_config)

    print('Titan Base (Weekly)')
    best_ti_base = train_titan(
        ti_base,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "weekly", "TitanBase", lookback, horizon)
        save_model(ti_base, ti_config, ckpt_path)
        best_ti_base["ckpt_path"] = str(ckpt_path)
    results['Titan Base'] = best_ti_base

    print('Titan LMM (Weekly)')
    best_ti_lmm = train_titan(
        ti_lmm,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "weekly", "TitanLMM", lookback, horizon)
        save_model(ti_lmm, ti_config, ckpt_path)
        best_ti_lmm["ckpt_path"] = str(ckpt_path)
    results['Titan LMM'] = best_ti_lmm

    print('Titan Seq2Seq (Weekly)')
    best_ti_seq2seq = train_titan(
        ti_seq2seq,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "weekly", "TitanSeq2Seq", lookback, horizon)
        save_model(ti_seq2seq, ti_config, ckpt_path)
        best_ti_seq2seq["ckpt_path"] = str(ckpt_path)
    results['Titan Seq2Seq'] = best_ti_seq2seq

    # --------------------------------------------------
    # PatchTST (Weekly)
    # --------------------------------------------------
    print('PatchTST Base (Weekly)')
    pt_point_config = PatchTSTConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        loss_mode='point',
        point_loss='huber',
        c_in=1,
        d_model=256,
        n_layers=3,
        patch_len=16,
        stride=8,
    )
    pt_base = build_patchTST_base(pt_point_config)

    pt_quantile_config = PatchTSTConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9),
        c_in=1,
        d_model=256,
        n_layers=3,
        patch_len=16,
        stride=8,
    )
    pt_quantile = build_patchTST_quantile(pt_quantile_config)

    best_pt_base = train_patchtst(
        pt_base,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "weekly", "PatchTSTBase", lookback, horizon)
        save_model(pt_base, pt_point_config, ckpt_path)
        best_pt_base["ckpt_path"] = str(ckpt_path)
    results['PatchTST Base'] = best_pt_base

    print('PatchTST Quantile (Weekly)')
    best_pt_quantile = train_patchtst(
        pt_quantile,
        train_loader,
        val_loader,
        train_cfg=quantile_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=True,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "weekly", "PatchTSTQuantile", lookback, horizon)
        save_model(pt_quantile, pt_quantile_config, ckpt_path)
        best_pt_quantile["ckpt_path"] = str(ckpt_path)
    results['PatchTST Quantile'] = best_pt_quantile

    return results


# ===================== MONTHLY =====================

def run_total_train_monthly(
        train_loader,
        val_loader,
        device: str = 'cuda' if torch.cuda.is_available() else 'mps',
        *,
        lookback: int,
        horizon: int,
        save_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    월간 전체 모델(PatchMixer, Titan, PatchTST)을 학습시키고 결과 반환.
    save_dir가 주어지면 ckpt 저장.
    """
    save_root = Path(save_dir) if save_dir is not None else None

    # 공통 학습 설정
    point_train_cfg, quantile_train_cfg, spike_cfg, stages = _build_common_train_configs(
        device=device, lookback=lookback, horizon=horizon
    )

    # 월간 캘린더 exogenous (M)
    future_exo_cb = compose_exo_calendar_cb(date_type='M')

    results: Dict[str, Dict] = {}

    # --------------------------------------------------
    # PatchMixer (Monthly)
    # --------------------------------------------------
    monthly_exo_dim = 2
    use_eol = False

    pm_base_config = PatchMixerConfigMonthly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='point',
        point_loss='mae',  # 월간은 scale 변동이 커서 MAE가 안정적인 편

        enc_in=1,
        d_model=64,
        e_layers=3,
        patch_len=6,
        stride=3,
        f_out=128,
        head_hidden=128,
        head_dropout=0.02,

        exo_dim=monthly_exo_dim,
        use_part_embedding=True,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,

        final_nonneg=True,
        use_eol_prior=use_eol,
        eol_feature_index=0,

        exo_is_normalized_default=False,  # 월간은 역정규화 후 가산 권장

        expander_season_period=12,
        expander_n_harmonics=6,
    )

    pm_quantile_config = PatchMixerConfigMonthly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9),

        enc_in=1,
        d_model=64,
        e_layers=3,
        patch_len=6,
        stride=3,
        f_out=128,
        head_hidden=128,
        head_dropout=0.02,

        exo_dim=monthly_exo_dim,
        use_part_embedding=True,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,

        final_nonneg=True,
        use_eol_prior=use_eol,
        eol_feature_index=0,

        exo_is_normalized_default=False,

        expander_season_period=12,
        expander_n_harmonics=6,
    )

    pm_base_model = build_patch_mixer_base(pm_base_config)
    pm_quantile_model = build_patch_mixer_quantile(pm_quantile_config)

    print('PatchMixer Base (Monthly)')
    best_pm_base = train_patchmixer(
        pm_base_model,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_base_config.exo_is_normalized_default,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "monthly", "PatchMixerBase", lookback, horizon)
        save_model(pm_base_model, pm_base_config, ckpt_path)
        best_pm_base["ckpt_path"] = str(ckpt_path)
    results['PatchMixer Base'] = best_pm_base

    print('PatchMixer Quantile (Monthly)')
    best_pm_quantile = train_patchmixer(
        pm_quantile_model,
        train_loader,
        val_loader,
        train_cfg=quantile_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_quantile_config.exo_is_normalized_default,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "monthly", "PatchMixerQuantile", lookback, horizon)
        save_model(pm_quantile_model, pm_quantile_config, ckpt_path)
        best_pm_quantile["ckpt_path"] = str(ckpt_path)
    results['PatchMixer Quantile'] = best_pm_quantile

    # --------------------------------------------------
    # Titan (Monthly)
    # --------------------------------------------------
    ti_m_config = TitanConfig(
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
        exo_dim=2,  # 월간 calendar sin/cos
        final_clamp_nonneg=True,
    )

    ti_base_m = build_titan_base(ti_m_config)
    ti_lmm_m = build_titan_lmm(ti_m_config)
    ti_seq2seq_m = build_titan_seq2seq(ti_m_config)

    print('Titan Base (Monthly)')
    best_ti_base_m = train_titan(
        ti_base_m,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "monthly", "TitanBase", lookback, horizon)
        save_model(ti_base_m, ti_m_config, ckpt_path)
        best_ti_base_m["ckpt_path"] = str(ckpt_path)
    results['Titan Base'] = best_ti_base_m

    print('Titan LMM (Monthly)')
    best_ti_lmm_m = train_titan(
        ti_lmm_m,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "monthly", "TitanLMM", lookback, horizon)
        save_model(ti_lmm_m, ti_m_config, ckpt_path)
        best_ti_lmm_m["ckpt_path"] = str(ckpt_path)
    results['Titan LMM'] = best_ti_lmm_m

    print('Titan Seq2Seq (Monthly)')
    best_ti_seq2seq_m = train_titan(
        ti_seq2seq_m,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "monthly", "TitanSeq2Seq", lookback, horizon)
        save_model(ti_seq2seq_m, ti_m_config, ckpt_path)
        best_ti_seq2seq_m["ckpt_path"] = str(ckpt_path)
    results['Titan Seq2Seq'] = best_ti_seq2seq_m

    # --------------------------------------------------
    # PatchTST (Monthly)
    # --------------------------------------------------
    print('PatchTST Base (Monthly)')
    pt_point_config_m = PatchTSTConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        loss_mode='point',
        point_loss='huber',
        c_in=1,
        d_model=256,
        n_layers=3,
        patch_len=6,
        stride=3,
    )
    pt_base_m = build_patchTST_base(pt_point_config_m)

    pt_quantile_config_m = PatchTSTConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9),
        c_in=1,
        d_model=256,
        n_layers=3,
        patch_len=6,
        stride=3,
    )
    pt_quantile_m = build_patchTST_quantile(pt_quantile_config_m)

    best_pt_base_m = train_patchtst(
        pt_base_m,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "monthly", "PatchTSTBase", lookback, horizon)
        save_model(pt_base_m, pt_point_config_m, ckpt_path)
        best_pt_base_m["ckpt_path"] = str(ckpt_path)
    results['PatchTST Base'] = best_pt_base_m

    print('PatchTST Quantile (Monthly)')
    best_pt_quantile_m = train_patchtst(
        pt_quantile_m,
        train_loader,
        val_loader,
        train_cfg=quantile_train_cfg,
        stages=list(stages),
        future_exo_cb=future_exo_cb,
        exo_is_normalized=True,
    )
    if save_root is not None:
        ckpt_path = _make_ckpt_path(save_root, "monthly", "PatchTSTQuantile", lookback, horizon)
        save_model(pt_quantile_m, pt_quantile_config_m, ckpt_path)
        best_pt_quantile_m["ckpt_path"] = str(ckpt_path)
    results['PatchTST Quantile'] = best_pt_quantile_m

    return results



# ===================== METRIC SUMMARY =====================

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

        # q_pred가 dict 형태 {0.1: ..., 0.5: ..., 0.9: ...}일 때만 구간지표 계산
        if res.get('q_pred') is not None and 0.1 in res['q_pred'] and 0.9 in res['q_pred']:
            result = quantile_metrics(y, yhat)
            row['converage_per_q'] = result['coverage_per_q']
            row['i80_cov'] = result['i80_cov']
            row['i80_wid'] = result['i80_wid']

        table[name] = row

    return table