from typing import Dict

import numpy as np

from models.PatchMixer.common.configs import PatchMixerConfigMonthly
from models.PatchTST.common.configs import PatchTSTConfigMonthly
from models.Titan.common.configs import TitanConfigMonthly
from models.model_builder import build_patch_mixer_base, build_patch_mixer_quantile, build_titan_base, build_titan_lmm, \
    build_patchTST_base, build_titan_seq2seq
from training.metrics import quantile_metrics
from training.model_trainers.patchmixer_train import train_patchmixer
from training.model_trainers.patchtst_train import train_patchtst
from training.model_trainers.titan_train import train_titan
from utils.metrics import mae, rmse, smape


def run_total_train_monthly(train_loader, val_loader, device = 'cuda'):
    results = {}

    # ---------------- PatchMixer ----------------
    pm_config = PatchMixerConfigMonthly(
        device = device,
        loss_mode = 'quantile',
        quantiles = (0.1, 0.5, 0.9)
    )

    pm_base_model = build_patch_mixer_base(pm_config)
    pm_quantile_model = build_patch_mixer_quantile(pm_config)

    print('PatchMixer Base')
    best_pm_base = train_patchmixer(
        pm_base_model,
        train_loader, val_loader,
        lr = 1e-3, loss_mode = 'quantile',
        quantiles = (0.1, 0.5, 0.9), use_intermittent = True,
    )
    results['PatchMixer Base'] = best_pm_base

    print('PatchMixer Quantile')
    best_pm_quantile = train_patchmixer(
        pm_quantile_model,
        train_loader, val_loader,
        lr = 1e-3, loss_mode = 'quantile',
        quantiles = (0.1, 0.5, 0.9), use_intermittent = True
    )
    results['PatchMixer Quantile'] = best_pm_quantile

    # ---------------- Titan (point + TTA) ----------------
    ti_config = TitanConfigMonthly(
        device = device,
        loss_mode = 'point',
        point_loss = 'huber'
    )

    ti_base = build_titan_base(ti_config)
    ti_lmm = build_titan_lmm(ti_config)
    ti_seq2seq = build_titan_seq2seq(ti_config)

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

    # ---------------- PatchTST(Quantile + point) ----------------
    pt_config = PatchTSTConfigMonthly(
        device = device,
        loss_mode = 'auto',
        quantiles = (0.1, 0.5, 0.9)
    )

    pt_base = build_patchTST_base(pt_config)

    print('PatchTST Base')
    best_pt_base = train_patchtst(
        pt_base,
        train_loader, val_loader,
        lr = 1e-3, loss_mode = 'auto', use_intermittent = True
    )
    results['PatchTST Base'] = best_pt_base


    print('PatchTST Quantile')
    best_pt_quantile = train_patchtst(
        pt_base,
        train_loader, val_loader,
        lr = 1e-3, loss_mode = 'auto', use_intermittent = True
    )
    results['PatchTST Quantile'] = best_pt_quantile

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

        if res.get('q_pred') is not None and 0.1 in res['q_pred'] and 0.9 in res['q_pred']:
            result = quantile_metrics(y, yhat)
            coverage_per_q = result['coverage_per_q']
            i80_cov = result['i80_cov']
            i80_wid = result['i80_wid']

            row['converage_per_q'] = coverage_per_q
            row['i80_cov'] = i80_cov
            row['i80_wid'] = i80_wid

        table[name] = row

    return table