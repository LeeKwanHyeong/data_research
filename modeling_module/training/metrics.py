from modeling_module.utils.helper import quantile_coverage, interval_coverage_width

def quantile_metrics(pred, y, quantiles=(0.1,0.5,0.9)):
    if pred.dim() != 3: return {}
    cov = quantile_coverage(pred, y, quantiles, reduce="overall")
    i80 = interval_coverage_width(pred, y, lower_q=0.1, upper_q=0.9, reduce="overall")
    return {
        "coverage_per_q": cov["coverage_per_q"].detach().cpu().numpy(),
        "i80_cov": float(i80["interval_cov"].detach().cpu()),
        "i80_wid": float(i80["interval_wid"].detach().cpu()),
    }