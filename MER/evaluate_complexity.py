import torch
import time
import torch.nn as nn
import copy
from thop import profile, clever_format

# 导入你的模型
from models import MDNet


# ==========================================
# 1. 模拟你的 DataConfig 以初始化模型
# ==========================================
class DummyConfig:
    dataset_name = 'MER'
    hidden_size = 64
    num_classes = 2
    dropout = 0.5
    batch_size = 1
    subject_num = 73

    @staticmethod
    def activation():
        return nn.ReLU()


class MDNetWrapper(nn.Module):
    def __init__(self, model, lengths, subject_labels, groups):
        super().__init__()
        self.model = model
        self.lengths = lengths
        self.subject_labels = subject_labels
        self.groups = groups

    def forward(self, eeg, mod2, mod3):
        return self.model(eeg, mod2, mod3, self.lengths, self.subject_labels, self.groups)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DummyConfig()
    params = {"group_num": 4}

    print(f"[*] Initializing MDNet on {device}...")
    model = MDNet(config, params).to(device)
    model.eval()

    # 构造虚拟输入
    eeg = torch.randn(1, 18, 800).to(device)
    gsr = torch.randn(1, 1, 800).to(device)
    ppg = torch.randn(1, 1, 800).to(device)
    lengths = torch.tensor([800]).to(device)
    subject_labels = torch.tensor([1]).to(device)
    groups = [[1, 2], [3, 4], [5, 6], [7, 8]]

    # [关键步骤] 预跑一次 forward 触发动态层实例化
    print("[*] Running dry-run to instantiate dynamic layers...")
    with torch.no_grad():
        _ = model(eeg, gsr, ppg, lengths, subject_labels, groups)

    # 封装模型
    wrapper_model = MDNetWrapper(model, lengths, subject_labels, groups)

    # ==========================================
    # 步骤 1: 先测推理效率 (此时模型没有被 thop 修改)
    # ==========================================
    print("\n" + "=" * 40)
    print(" 1. 推理效率评估 (Efficiency)")
    print("=" * 40)

    iterations = 300
    warmup = 50

    print(f"[*] Warming up GPU for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = wrapper_model(eeg, gsr, ppg)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[*] Testing latency for {iterations} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = wrapper_model(eeg, gsr, ppg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_latency_ms
    print(f"Avg Latency/Sample : {avg_latency_ms:.2f} ms")
    print(f"Throughput (FPS)   : {fps:.2f} samples/second")

    # ==========================================
    # 步骤 2: 再测计算复杂度 (使用 thop)
    # ==========================================
    print("\n" + "=" * 40)
    print(" 2. 计算复杂度评估 (Complexity)")
    print("=" * 40)

    # 使用深拷贝防止 thop 的钩子污染原始测试模型（虽然这里已经是最后一步了，但这是好习惯）
    profile_model = copy.deepcopy(wrapper_model)

    macs, params_count = profile(profile_model, inputs=(eeg, gsr, ppg), verbose=False)
    flops = macs * 2
    formatted_flops, formatted_params = clever_format([flops, params_count], "%.3f")

    print(f"Total Parameters : {formatted_params}")
    print(f"FLOPs            : {formatted_flops}")
    print("=" * 40)


if __name__ == "__main__":
    main()