import math
import torch


def run_experiment(alpha, seq_len, hidden_size=8):
    """
    只研究梯度沿时间传播的动力学。

    h_t = tanh(h_{t-1} @ W_hh)
    W_hh = alpha * I
    loss = sum(h_T)
    """

    dtype = torch.float64

    # 很小的初始状态使 tanh 初期接近线性区
    h_0 = torch.full(
        (1, hidden_size),
        fill_value=1e-6,
        dtype=dtype,
        requires_grad=True,
    )

    W_hh = alpha * torch.eye(
        hidden_size,
        dtype=dtype,
    )

    h_t = h_0

    for _ in range(seq_len):
        h_t = torch.tanh(h_t @ W_hh)

    # 损失只依赖最后一个时间步
    loss = h_t.sum()
    loss.backward()

    final_hidden_norm = h_t.detach().norm().item()
    initial_grad_norm = h_0.grad.norm().item()

    # 在线性近似下：
    # loss=sum(h_T)，所以最终梯度基准为 sqrt(H)
    linear_approx = math.sqrt(hidden_size) * (
        abs(alpha) ** seq_len
    )

    return {
        "final_hidden_norm": final_hidden_norm,
        "initial_grad_norm": initial_grad_norm,
        "linear_approx": linear_approx,
    }


alphas = [0.5, 1.0, 1.5]
sequence_lengths = [1, 5, 10, 20, 30, 40]

for alpha in alphas:
    print()
    print("=" * 72)
    print(f"alpha = {alpha}")
    print("=" * 72)

    print(
        f"{'T':>4} "
        f"{'||h_T||':>16} "
        f"{'||dL/dh_0||':>18} "
        f"{'线性近似':>18}"
    )

    for seq_len in sequence_lengths:
        result = run_experiment(
            alpha=alpha,
            seq_len=seq_len,
        )

        print(
            f"{seq_len:4d} "
            f"{result['final_hidden_norm']:16.6e} "
            f"{result['initial_grad_norm']:18.6e} "
            f"{result['linear_approx']:18.6e}"
        )

        assert math.isfinite(
            result["initial_grad_norm"]
        )