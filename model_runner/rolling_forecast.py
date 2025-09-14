import copy
import torch
import polars as pl
from torch import nn, optim

'''
    모든 파라미터를 우선 동결(required_grad = False)한 후,
    - output head(=output_proj)
    - LayerNorm(각 블록 내부)
    만 미세 조정 가능하도록 연다.
    * 옵션으로 train_blocks = True면 블록 전체를 열 수 있다.
'''
# option: 업데이트 대상 좁히기
def set_ttm_trainable_params(model, train_head = True, train_layer_norm = True, train_blocks = False):
    for p in model.parameters():
        p.requires_grad = False

    if train_blocks:
        for layer in model.encoder.layers:
            for p in layer.parameters():
                p.requires_grad = True

    if train_layer_norm:
        for layer in model.encoder.layers:
            for m in layer.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

    if train_head and hasattr(model, 'output_proj'):
        for p in model.output_proj.parameters():
            p.requires_grad = True

# contextual memory reset (part 바뀔 때)
def reset_contextual_memory(model):
    for layer in model.encoder.layers:
        layer.attn.contextual_memory = None

# 한 파트/한 컷 오프에서의 예측 + (옵션) 보정
@torch.no_grad()
def predict_horizon(model, x_recent, device, use_lmm_eval = True):
    model.eval()
    x_recent = x_recent.to(device).float()
    # LMMModel이면 mode = 'eval'로 LMM 비활성, 원하면 True/False 토글
    pred = model(x_recent, mode = 'eval' if use_lmm_eval else 'train')
    return pred.squeeze(0).detach().cpu() # (H,)

def ttm_adapt_once(model, x_recent, y_recent, steps = 1, lr = 1e-4):
    # 필요한 파라미터만 requires_grad = True로 셋팅되어 있다고 가정
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(steps):
        pred = model(x_recent) # (1, H)
        loss = loss_fn(pred, y_recent) # 참조 라벨은 과거 구간
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm = 50)
        optimizer.step()
    return float(loss.item())

def rolling_forecast_with_ttm(
        model,
        part_series, # numpy or list of demand qty(time-sorted)
        cut_dates,  # cut-off yyyymm list (각 컷오프에서 예측)
        lookback, horizon,
        device = 'cuda',
        ttm_lr = 5e-5, ttm_steps = 1,
        use_weight_rollback = True
):
    '''
    part_series: ex) [q_201901, q_201902, ...] (실제 수요)
    cut_dates  : ex) [idx_202201, idx_202202, ...] index 처리 (날짜 -> 인덱스 매핑 필요)
    '''
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    set_ttm_trainable_params(model, train_head = True, train_layer_norm = True, train_blocks = False)
    results = []
    base_state = copy.deepcopy(model.state_dict()) # Rollback snapshot

    for cut_idx in cut_dates:
        # 1) input_window 확보
        # X: [cut_idx - lookback, ... , cut_idx - 1]
        if cut_idx < lookback:
            continue
        x_win = part_series[cut_idx - lookback : cut_idx] # length L
        x_tensor = torch.tensor(x_win, dtype = torch.float32).view(1, lookback, 1)

        # 2) Context Memory Injection (Label 없어도 OK)
        reset_contextual_memory(model) # part/cut-off 단위로 리셋 권장
        with torch.no_grad():
            embedded_ctx = model.encoder.input_proj(x_tensor.to(device))
        for layer in model.encoder.layers:
            layer.attn.update_contextual_memory(embedded_ctx.detach())

        # 3) forecast T+1, ..., T+H
        y_hat = predict_horizon(model, x_tensor, device = device) # (H,)

        # 4) 보정(adapt): 반드시 '이미 발생한 과거 라벨'로만 적용
        # 예: cut_idx 시점에 ttm을 할때, 이전 컷오프의 정답(y)로만 적용
        # 여기서는 예시로 cut_idx - 1구간의 라벨을 가정(옵션)
        if cut_idx - horizon >= 0:
            y_recent = part_series[cut_idx - horizon : cut_idx] # 과거 H 구간
            y_t = torch.tensor(y_recent, dtype = torch.float32).view(1, horizon).to(device)
            loss_after = ttm_adapt_once(model, x_tensor.to(device), y_t, steps = ttm_steps, lr = ttm_lr)
        else:
            loss_after = None

        # 5) 결과 저장
        results.append({
            'cut_index': int(cut_idx),
            'forecast': y_hat.numpy().tolist(),
            'adapt_loss': loss_after
        })

        # 6) 가중치 롤백: 파트/컷오프별 오염 방지
        if use_weight_rollback:
            model.load_state_dict(base_state)
    return results