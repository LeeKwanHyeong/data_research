import torch


def _prepare_next_input(
        x: torch.Tensor,
        y_step: torch.Tensor,
        target_channel: int = 0,
        fill_mode: str = 'copy_last',
) -> torch.Tensor:
    '''
    현재 윈도우 x에 대해 한 스텝 예측값 y_step을 마지막 시점으로 붙여 넣고,
    가장 오래된 시점을 버려 윈도우를 한 칸 전진.
    :param x: [B, L, C] 입력 윈도우 (Batch, Lookback, Channel)
    :param y_step: 이번 스텝에 예측된 target value.
    :param target_channel: int
        target 변수가 위치한 채널 인덱스 (기본 0)
    :param fill_mode: {'copy_last', 'zeros'}
        다변량 (C>1)일 때, 타깃 외 채널을 새 시점에 채우는 방식
        - 'copy_last': 직전 시점 값을 복사
        - 'zeros': 0으로 채움
    :return:
    x_next: [B, L, C] 한 칸 전진된 다음 입력 윈도우
    '''
    assert x.dim() == 3, f"x must be [B, L, C], got {x.shape}"
    B,L,C = x.shape
    y_step = y_step.reshape(B, 1, 1) # -> [B, 1, 1]

    if C == 1:
        new_token = y_step # 단변량이면 예측값만 넣으면 됨

    else:
        last = x[:, -1:, :].clone() # 직전 시점 값
        if fill_mode == 'zeros':
            new_token = torch.zeros_like(last)
        else:
            new_token = last # copy_last

        new_token[:, 0, target_channel] = y_step[:, 0, 0]

    # Window slide + new series
    x_next = torch.cat([x[:, 1:, :], new_token], dim = 1)
    return x_next

def _winsorize_clamp(
    hist: torch.Tensor,          # [B, L] (target 채널 히스토리)
    y_step: torch.Tensor,        # [B]
    nonneg: bool = True,
    clip_q: tuple[float, float] = (0.05, 0.95),
    clip_mul: float = 4.0,       # 상위 분위수 배수 상한
    max_growth: float = 0.05      # 직전값 대비 최대 성장배율 max_growth=2.0 ~ 2.5
) -> torch.Tensor:
    """
    히스토리 분위수/직전값 대비 성장률로 y_step을 winsorize + clamp.
    (nan_to_num 대신 torch.where로 텐서별 대체)
    """
    # 안전하게 float로
    hist = hist.float()
    y = y_step.float()

    B, L = hist.shape
    last = hist[:, -1]  # [B]

    # 히스토리 내 비유한/비유효값을 직전값으로 대체하여 분위수 계산 안정화
    hist_safe = torch.where(torch.isfinite(hist), hist, last.unsqueeze(1))

    q_lo = torch.quantile(hist_safe, clip_q[0], dim=1)  # [B]
    q_hi = torch.quantile(hist_safe, clip_q[1], dim=1)  # [B]

    min_cap = torch.zeros_like(q_lo) if nonneg else q_lo
    cap_quant  = q_hi * clip_mul
    cap_growth = torch.where(last > 0, last * max_growth, cap_quant)
    max_cap = torch.minimum(cap_quant, cap_growth)  # [B]

    # 요소별로 nan/±inf를 안전 값으로 치환
    y = torch.where(torch.isnan(y),      last,    y)
    y = torch.where(torch.isposinf(y),   max_cap, y)
    y = torch.where(torch.isneginf(y),   min_cap, y)

    # 최종 클램프
    y = torch.clamp(y, min=min_cap, max=max_cap)
    return y


def _dampen_to_last(
    last: torch.Tensor,  # [B]
    y_step: torch.Tensor,
    damp: float = 0.1    # 0(무효)~1(그대로), 0.3~0.7 권장
) -> torch.Tensor:
    """
    예측값을 직전 관측과 혼합하여 급변을 완화.
    """
    if damp <= 0.0:
        return y_step
    return (1.0 - damp) * last + damp * y_step

def _guard_multiplicative(
    last: torch.Tensor,          # [B]
    y_raw: torch.Tensor,         # [B]
    max_step_up: float = 0.10,   # 스텝당 최대 상승비율 (예: +10%)
    max_step_down: float = 0.20  # 스텝당 최대 하락비율 (예: -20%)
) -> torch.Tensor:
    """
    직전값 대비 예측 비율을 log 도메인에서 클램프.
    last=0 보호를 위해 eps 사용.
    """
    eps = 1e-6
    last_safe = torch.clamp(last, min=eps)
    y_safe = torch.clamp(y_raw, min=eps)  # 비율/로그 계산 안정화

    ratio = y_safe / last_safe                     # [B]
    log_ratio = torch.log(ratio)                   # [B]

    min_ratio = 1.0 - max_step_down                # 예: 0.8
    max_ratio = 1.0 + max_step_up                  # 예: 1.1
    log_min = torch.log(torch.tensor(min_ratio, device=last.device))
    log_max = torch.log(torch.tensor(max_ratio, device=last.device))

    log_ratio = torch.clamp(log_ratio, min=log_min, max=log_max)
    y_guard = last_safe * torch.exp(log_ratio)
    return y_guard