# TSADNET
云端跑实验
如果在纯 Linux 系统（如 Ubuntu 20.04/22.04）上配置，过程会比 WSL2 更加顺畅，因为你是直接与硬件和内核交互，不存在文件系统性能损耗或网络映射问题。

以下是直接在 Linux 上从零搭建 **Mamba + TSAD-Net** 环境的专业步骤：

---

### 1. 系统底层检查 (GPU & Driver)

首先确保你的 Linux 已经安装了 NVIDIA 驱动。

```bash
# 检查驱动版本
nvidia-smi

```

**注意**：只要你的 `Driver Version` 大于 520，你就可以安装 CUDA 11.8 或 12.1。

---

### 2. 安装 CUDA Toolkit (开发环境)

即便你系统里有驱动，你通常还需要 `nvcc` 编译器来编译 Mamba 的 C++/CUDA 算子。

建议安装 **CUDA 11.8**（目前 Mamba 兼容性最稳的版本）：

1. 去 [CUDA 下载页面](https://developer.nvidia.com/cuda-11-8-0-download-archive) 选择 `Linux` -> `x86_64` -> `Ubuntu` -> `deb (local)`。
2. 按照官方给出的 4-5 行指令运行。
3. **配置环境变量**（非常重要，决定了 `pip` 能否找到编译器）：
```bash
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

```



---

### 3. 使用 Conda 构建 Python 环境

```bash
# 创建 3.10 环境 (为了更好的 NumPy/PyTorch 2.0 兼容性)
conda create -n mamba_linux python=3.10 -y
conda activate mamba_linux

# 1. 先装 PyTorch (针对 CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 TSAD-Net 原有的依赖 (略微提升 NumPy 版本避免之前的报错)
pip install transformers==4.33.1 numpy==1.23.5 pandas==1.5.3 scipy==1.9.1 scikit-learn==1.0.2 easydict==1.13 pynvml==12.0.0

```

---

### 4. 安装 Mamba 核心库

在 Linux 下，`pip` 通常会自动下载预编译的二进制包。如果下载失败，它会自动调用你刚装好的 `nvcc` 进行本地编译。

```bash
# 安装构建依赖
pip install packaging ninja

# 安装 Mamba 底层算子 (因果卷积)
pip install causal-conv1d>=1.2.0

# 安装 Mamba 主程序
pip install mamba-ssm

```

---

### 5. 验证安装 (Linux 命令行测试)

在终端输入 `python` 并运行：

```python
import torch
from mamba_ssm import Mamba

# 检查 CUDA
print(f"CUDA: {torch.cuda.is_available()}")

# 测试 Mamba 算子
model = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).to("cuda")
input_data = torch.randn(2, 64, 128).to("cuda")
output = model(input_data)

print(f"Success! Output shape: {output.shape}")

```

---

### Linux 环境下的特有优势

1. **显存回收更高效**：相比 Windows，Linux 能够更精准地释放碎片显存。
2. **多进程支持 (NumWorkers)**：在 Linux 上，你可以放心地在 `DataLoader` 中把 `num_workers` 设为 8 或 16。而在 Windows 上，多进程经常会导致内存溢出或报错。
3. **支持原生 Triton**：Mamba 的高性能很大程度上归功于 Triton 实现的融合核（Fused Kernels）。在原生 Linux 上，Triton 可以直接通过 GPU 生成优化的机器码，这是 Mamba 跑得快的根本原因。

---

### 下一步：代码改进

环境配好后，针对你的 `TSAD_Net.py`，改进的重点在于**模态特征提取阶段**。

你原来的音频和视频处理是：
`Linear/Conv1d` -> `Mean Pooling` -> `Tensor`

**改进后的逻辑应当是：**
`Linear/Conv1d` -> **`Mamba Encoder`** -> `Mean/Max Pooling` -> `Tensor`

**这样做的好处：**
原本的 `Mean Pooling` 会把音频中“先开心后悲伤”的情绪平均掉，变成“无情绪”；而 Mamba 可以记住这种情绪的动态变化，从而提取出更有区分度的特征。

**你需要我为你写一段针对音频/视频分支集成 Mamba 的完整 `TSAD_Net` 修改代码吗？**
