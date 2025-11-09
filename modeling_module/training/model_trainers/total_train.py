from typing import Dict
import numpy as np

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfigMonthly, PatchMixerConfig, \
    PatchMixerConfigWeekly
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.Titan.Titans import TitanBaseModel, TitanLMMModel, TitanSeq2SeqModel
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.models.model_builder import build_titan_base, build_titan_lmm, build_titan_seq2seq, \
    build_patch_mixer_base, build_patch_mixer_quantile, build_patchTST_base, build_patchTST_quantile
from modeling_module.training.config import SpikeLossConfig, TrainingConfig, StageConfig
from modeling_module.training.metrics import quantile_metrics
from modeling_module.training.model_trainers.patchmixer_train import train_patchmixer
from modeling_module.training.model_trainers.patchtst_train import train_patchtst
from modeling_module.training.model_trainers.titan_train import train_titan
from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb
from modeling_module.utils.metrics import smape, rmse, mae


def _get_part_vocab_size_from_loader(loader):
    try:
        return len(getattr(loader.dataset, "part_vocab", {}))
    except Exception:
        return 0

def run_total_train_weekly(
        train_loader,
        val_loader,
        device='cuda',
        *,
        lookback,
        horizon,
):
    stg_warmup = StageConfig(
        epochs=10,
        spike_enabled=False,
        lr=3e-4,  # warm-up은 살짝 높여도 ok
        use_horizon_decay=False  # 초기는 스케일/시즌 맞추기에 집중
    )

    # ② Spike-boost: spike ON, lr ↓, 30~50 epoch
    stg_spike = StageConfig(
        epochs=50,
        spike_enabled=True,
        lr=1e-4,
        use_horizon_decay=True,  # 필요 시 ON
        tau_h=0.85  # 근미래 가중 재가동
    )

    stages = [stg_warmup, stg_spike]

    spike_cfg = SpikeLossConfig(
        enabled=True,
        strategy='mix',
        huber_delta=0.6,
        asym_up_weight=1.0,
        asym_down_weight=8.0,  # 언더예측 벌점
        mad_k=1.5,
        w_spike=32.0,
        w_norm=1.0,
        alpha_huber=0.6,
        beta_asym=0.4,
        mix_with_baseline=False,
        gamma_baseline=0.0,
        # # NEW (optional cap)
        # w_cap=12.0,
    )
    point_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        lr=3e-4,               # 1e-3(부스트) → 3e-4(안정)
        weight_decay=1e-3,
        t_max=40,
        patience=100,
        loss_mode='point',  # Titan은 현재 포인트 출력
        point_loss='huber',  # base는 huber, mix에서 huber/asym이 더해짐
        huber_delta = 0.8,
        use_intermittent=True,  # 간헐수요 가중
        alpha_zero=3.0, # ← 2.2 → 3.0 (0수요 편향 억제)
        alpha_pos=1.0,
        gamma_run=0.3,  # ← 드리프트 억제
        use_horizon_decay=False, tau_h=0.85,  # 근미래 가중
        val_use_weights=False,  # 검증은 공정평가
        spike_loss=spike_cfg,  # ★ 여기만 켜면 losses/engine이 자동 분기
        max_grad_norm = 30.0
    )
    quantile_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        lr=3e-4,  # 1e-3(부스트) → 3e-4(안정)
        weight_decay=1e-4,
        t_max=40,
        patience=10,
        loss_mode='quantile',  # Titan은 현재 포인트 출력
        quantiles = (0.1, 0.5, 0.9),
        use_intermittent=True,  # 간헐수요 가중
        alpha_zero=1.2, alpha_pos=1.0, gamma_run=0.6,
        use_horizon_decay=False, tau_h=0.85,  # 근미래 가중
        val_use_weights=False,  # 검증은 공정평가
        spike_loss=spike_cfg,  # ★ 여기만 켜면 losses/engine이 자동 분기
        max_grad_norm=30.0
    )

    future_exo_cb = compose_exo_calendar_cb(date_type = 'W')


    results: Dict[str, Dict] = {}

    # ---------------- PatchMixer ----------------
    weekly_exo_dim = 4  # compose_exo_calendar_cb('W')라면 보통 sin/cos 2개
    use_eol = False  # weekly에서 sequence(EOL proxy)를 future_exo에 포함했다면 True로

    pm_base_config = PatchMixerConfigWeekly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='point',
        point_loss='huber',

        # ---- LTB 권장 파라미터 ----
        enc_in=1,
        d_model=64, e_layers=3,
        patch_len=12, stride=8,
        f_out=128,
        head_hidden=128,
        head_dropout=0.05,

        # 외생/파트 임베딩
        exo_dim=weekly_exo_dim,  # 주간 캘린더(woY sin/cos)
        use_part_embedding=True,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,

        # 출력 제약/EOL
        final_nonneg=True,
        use_eol_prior=use_eol,  # sequence를 exo에 포함했다면 True
        eol_feature_index=0,  # future_exo[:, :, 0]이 sequence라면 0

        # 정규화 후/원단위 가산 방식
        exo_is_normalized_default=True,  # 주간은 정규화 공간에서 가산 권장

        # 계절 파라미터
        expander_season_period=52,
        expander_n_harmonics=16,
    )

    pm_quantile_config = PatchMixerConfigWeekly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9),

        # ---- LTB 권장 파라미터(동일 용량) ----
        enc_in=1,
        d_model=64, e_layers=3,
        patch_len=12, stride=8,
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
        train_loader, val_loader,
        train_cfg=point_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
        # ▶ configs의 기본값을 그대로 사용
        exo_is_normalized=pm_base_config.exo_is_normalized_default,
    )
    results['PatchMixer Base'] = best_pm_base

    print('PatchMixer Quantile (Weekly)')
    best_pm_quantile = train_patchmixer(
        pm_quantile_model,
        train_loader, val_loader,
        train_cfg=quantile_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_quantile_config.exo_is_normalized_default,
    )
    results['PatchMixer Quantile'] = best_pm_quantile

    # # ---------------- Titan (Quantile + TTA) ----------------
    ti_config = TitanConfig(
        lookback=lookback,
        horizon=horizon,
        # 아래는 Titans.py에서 사용하는 공통 옵션들(필요 시 설정)
        input_dim=1,  # 데이터로더 입력 채널 수에 맞춰 조정
        d_model=256,
        n_layers=3,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        contextual_mem_size=256,
        persistent_mem_size=64,
        use_exogenous=True, exo_dim=2,  # 캘린더 sin/cos 자동 주입 조건
        final_clamp_nonneg=True,

    )


    # (선택) Patch 변형 호출부 호환: 현재는 Seq2Seq로 매핑됨

    ti_base = build_titan_base(ti_config)
    ti_lmm = build_titan_lmm(ti_config)
    ti_seq2seq = build_titan_seq2seq(ti_config)


    print('Titan Base')
    best_ti_base = train_titan(
        ti_base, train_loader, val_loader,
        train_cfg=point_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
    )
    results['Titan Base'] = best_ti_base

    print('Titan LMM')
    best_ti_lmm = train_titan(
        ti_lmm, train_loader, val_loader,
        train_cfg=point_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
    )
    results['Titan LMM'] = best_ti_lmm

    print('Titan Seq2Seq')
    best_ti_seq2seq = train_titan(
        ti_seq2seq, train_loader, val_loader,
        train_cfg=point_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
    )
    results['Titan Seq2Seq'] = best_ti_seq2seq

    # ---------------- PatchTST ----------------

    print('PatchTST Base')
    pt_point_config = PatchTSTConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        loss_mode='point',
        point_loss='huber',  # 권장
        c_in=1,  # 입력 채널
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

    print('PatchTST Base (Weekly)')
    best_pt_base = train_patchtst(
        pt_base,
        train_loader, val_loader,
        train_cfg=point_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
    )
    results['PatchTST Base'] = best_pt_base

    print('PatchTST Quantile (Weekly)')
    best_pt_quantile = train_patchtst(
        pt_quantile,
        train_loader, val_loader,
        train_cfg=quantile_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=True
    )
    results['PatchTST Quantile'] = best_pt_quantile


    return results

def run_total_train_monthly(
        train_loader,
        val_loader,
        device='cuda',
        *,
        lookback,
        horizon,
):
    # ---------------- 2-Stage 스케줄 (월간도 주간과 동일) ----------------
    stg_warmup = StageConfig(
        epochs=1,
        spike_enabled=False,
        lr=3e-4,                 # warm-up은 약간 높게
        use_horizon_decay=False
    )
    stg_spike = StageConfig(
        epochs=1,
        spike_enabled=True,      # 스파이크 강화 단계
        lr=1e-4,
        use_horizon_decay=True,
        tau_h=0.85
    )
    stages = [stg_warmup, stg_spike]

    # ---------------- 손실/스파이크 구성 (LTB 권장) ----------------
    spike_cfg = SpikeLossConfig(
        enabled=True,
        strategy='mix',
        huber_delta=0.6,
        asym_up_weight=1.0,
        asym_down_weight=8.0,   # 언더예측 강벌
        mad_k=1.5,
        w_spike=32.0,
        w_norm=1.0,
        alpha_huber=0.6,
        beta_asym=0.4,
        mix_with_baseline=False,
        gamma_baseline=0.0,
        # w_cap=12.0,            # 필요 시 상한 적용
    )

    # 포인트(평균/중앙 추정용) — Titan/PM Base에 사용
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
        use_intermittent=True,   # 0 수요/간헐 가중
        alpha_zero=3.0,          # 0 수요 편향 억제
        alpha_pos=1.0,
        gamma_run=0.3,           # 드리프트 억제
        use_horizon_decay=False, tau_h=0.85,
        val_use_weights=False,
        spike_loss=spike_cfg,
        max_grad_norm=30.0
    )

    # 분위(구간 예측용) — PM/PatchTST Quantile에 사용
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
        alpha_zero=1.2, alpha_pos=1.0, gamma_run=0.6,
        use_horizon_decay=False, tau_h=0.85,
        val_use_weights=False,
        spike_loss=spike_cfg,
        max_grad_norm=30.0
    )

    # ---------------- 월간 외생(달 주기) 콜백 ----------------
    # 월간은 month sin/cos(주기 12) 중심. compose_exo_calendar_cb('M') 사용.
    future_exo_cb = compose_exo_calendar_cb(date_type='M')

    results: Dict[str, Dict] = {}

    # ==================== PatchMixer ====================
    # Base(Point)

    monthly_exo_dim = 2  # compose_exo_calendar_cb('M') → month sin/cos
    use_eol = False  # 월간에도 sequence를 exo에 넣었다면 True로 전환

    pm_base_config = PatchMixerConfigMonthly(
        lookback=lookback,
        horizon=horizon,
        device=device,
        loss_mode='point',
        point_loss='mae',  # 월간은 scale 변동이 커서 MAE가 안정적인 편

        # ---- LTB 권장 파라미터 ----
        enc_in=1,
        d_model=64, e_layers=3,
        patch_len=6, stride=3,  # 월단위 patch 권장(6~8 / 3~4)
        f_out=128,
        head_hidden=128,
        head_dropout=0.02,

        exo_dim=monthly_exo_dim,  # month sin/cos
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

        # ---- LTB 권장 파라미터(동일 용량) ----
        enc_in=1,
        d_model=64, e_layers=3,
        patch_len=6, stride=3,
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
        train_loader, val_loader,
        train_cfg=point_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_base_config.exo_is_normalized_default,
    )
    results['PatchMixer Base'] = best_pm_base

    print('PatchMixer Quantile (Monthly)')
    best_pm_quantile = train_patchmixer(
        pm_quantile_model,
        train_loader, val_loader,
        train_cfg=quantile_train_cfg,
        stages=stages,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=pm_quantile_config.exo_is_normalized_default,
    )
    results['PatchMixer Quantile'] = best_pm_quantile

    # # ==================== Titan ====================
    # # 월간은 장기 트렌드/메모리 효과가 중요 → 모델 용량을 주간 대비 소폭 상향
    # ti_config = TitanConfig(
    #     lookback=lookback,
    #     horizon=horizon,
    #     input_dim=1,
    #     d_model=256,
    #     n_layers=3,
    #     n_heads=4,
    #     d_ff=512,
    #     dropout=0.1,
    #     contextual_mem_size=256,
    #     persistent_mem_size=64,
    #     use_exogenous=True, exo_dim=2,     # month sin/cos
    #     final_clamp_nonneg=True,
    # )
    # ti_base = build_titan_base(ti_config)
    # ti_lmm  = build_titan_lmm(ti_config)
    # ti_seq2seq = build_titan_seq2seq(ti_config)
    #
    # print('Titan Base (Monthly)')
    # best_ti_base = train_titan(
    #     ti_base, train_loader, val_loader,
    #     train_cfg=point_train_cfg,
    #     stages=stages,
    #     future_exo_cb=future_exo_cb,
    # )
    # results['Titan Base'] = best_ti_base
    #
    # print('Titan LMM (Monthly)')
    # best_ti_lmm = train_titan(
    #     ti_lmm, train_loader, val_loader,
    #     train_cfg=point_train_cfg,
    #     stages=stages,
    #     future_exo_cb=future_exo_cb,
    # )
    # results['Titan LMM'] = best_ti_lmm
    #
    # print('Titan Seq2Seq (Monthly)')
    # best_ti_seq2seq = train_titan(
    #     ti_seq2seq, train_loader, val_loader,
    #     train_cfg=point_train_cfg,
    #     stages=stages,
    #     future_exo_cb=future_exo_cb,
    # )
    # results['Titan Seq2Seq'] = best_ti_seq2seq
    #
    # # ==================== PatchTST ====================
    # # 월간 패치 길이는 8~16 권장(월별 데이터가 적으므로 지나친 분절은 피함)
    # pt_point_config = PatchTSTConfig(
    #     device=device,
    #     lookback=lookback,
    #     horizon=horizon,
    #     loss_mode='point',
    #     point_loss='huber',
    #     c_in=1,
    #     d_model=256,
    #     n_layers=3,
    #     patch_len=12,   # 월간 권장: 12
    #     stride=6,
    # )
    # pt_base = build_patchTST_base(pt_point_config)
    #
    # pt_quantile_config = PatchTSTConfig(
    #     device=device,
    #     lookback=lookback,
    #     horizon=horizon,
    #     loss_mode='quantile',
    #     quantiles=(0.1, 0.5, 0.9),
    #     c_in=1,
    #     d_model=256,
    #     n_layers=3,
    #     patch_len=12,
    #     stride=6,
    # )
    # pt_quantile = build_patchTST_quantile(pt_quantile_config)
    #
    # print('PatchTST Base (Monthly)')
    # best_pt_base = train_patchtst(
    #     pt_base,
    #     train_loader, val_loader,
    #     train_cfg=point_train_cfg,
    #     stages=stages,
    #     future_exo_cb=future_exo_cb,
    # )
    # results['PatchTST Base'] = best_pt_base
    #
    # print('PatchTST Quantile (Monthly)')
    # best_pt_quantile = train_patchtst(
    #     pt_quantile,
    #     train_loader, val_loader,
    #     train_cfg=quantile_train_cfg,
    #     stages=stages,
    #     future_exo_cb=future_exo_cb,
    #     exo_is_normalized=True
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
