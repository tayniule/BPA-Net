import torch
import time
import torch.nn as nn
import copy
from thop import profile, clever_format

# 1. 从新的 models_refed.py 导入模型
from models import MDNet


# ==========================================
# 1. 模拟 REFED 数据集的 DataConfig
# ==========================================
class DummyConfig:
    dataset_name = 'REFED'
    hidden_size = 64
    num_classes = 2
    dropout = 0.5
    batch_size = 1
    subject_num = 32  # REFED 数据集包含 32 名被试

    @staticmethod
    def activation():
        return nn.ReLU()


# ==========================================
# 2. 编写适配 REFED 模态的 Wrapper
# ==========================================
class MDNetRefedWrapper(nn.Module):
    def __init__(self, model, lengths, subject_labels, groups):
        super().__init__()
        self.model = model
        self.lengths = lengths
        self.subject_labels = subject_labels
        self.groups = groups

    def forward(self, eeg, fnirs):
        # REFED 版本只有 eeg 和 fnirs 两个数据输入
        return self.model(eeg, fnirs, self.lengths, self.subject_labels, self.groups)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DummyConfig()
    params = {"group_num": 4}  # 假设分组数为 4

    print(f"[*] Initializing MDNet (REFED version) on {device}...")
    model = MDNet(config, params).to(device)
    model.eval()

    # ==========================================
    # 3. 构造 REFED 数据虚拟输入
    # ==========================================
    # EEGNet 通常需要 (Batch, 1, Channels, Time)
    # 假设 REFED 为 64 通道 EEG, 2 通道 fNIRS，长度均为 800
    eeg = torch.randn(1, 1, 64, 800).to(device)
    fnirs = torch.randn(1, 1, 51, 800).to(device)

    lengths = torch.tensor([800]).to(device)
    subject_labels = torch.tensor([1]).to(device)
    # 构造对应 32 个被试的虚拟分组映射
    groups = [list(range(0, 8)), list(range(8, 16)), list(range(16, 24)), list(range(24, 32))]

    # [关键步骤] 预跑一次触发动态层实例化
    print("[*] Running dry-run to instantiate dynamic layers (leaner, etc.)...")
    with torch.no_grad():
        _ = model(eeg, fnirs, lengths, subject_labels, groups)

    # 封装模型
    wrapper_model = MDNetRefedWrapper(model, lengths, subject_labels, groups)

    # ==========================================
    # 步骤 1: 推理效率评估 (Latency)
    # ==========================================
    print("\n" + "=" * 40)
    print(" 1. REFED 模型效率评估 (Efficiency)")
    print("=" * 40)

    iterations = 300
    warmup = 50

    print(f"[*] Warming up GPU for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = wrapper_model(eeg, fnirs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[*] Testing latency for {iterations} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = wrapper_model(eeg, fnirs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_latency_ms
    print(f"Avg Latency/Sample : {avg_latency_ms:.2f} ms")
    print(f"Throughput (FPS)   : {fps:.2f} samples/second")

    # ==========================================
    # 步骤 2: 计算复杂度评估 (FLOPs)
    # ==========================================
    print("\n" + "=" * 40)
    print(" 2. REFED 模型复杂度评估 (Complexity)")
    print("=" * 40)

    profile_model = copy.deepcopy(wrapper_model)
    macs, params_count = profile(profile_model, inputs=(eeg, fnirs), verbose=False)
    flops = macs * 2
    formatted_flops, formatted_params = clever_format([flops, params_count], "%.3f")

    print(f"Total Parameters : {formatted_params}")
    print(f"FLOPs            : {formatted_flops}")
    print("=" * 40)


if __name__ == "__main__":
    main()