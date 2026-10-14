# 雷达脉内调制样式识别

Python 实现的单脉冲识别流程：复数 IQ 信号 → STFT 功率谱 → STFT-CNN → 调制类别。
已在当前目录创建 `.venv`，实际完成 CPU 训练、独立测试和新信号推理。

## 本地 Web 工作台

安装 `requirements.txt` 后启动中文界面：

```bash
python3 -B -E -m aimc.web --port 8080
```

浏览器打开 `http://127.0.0.1:8080`。工作台提供样例信号生成、IQ 文件上传、
CPU 模型预测、功率时频图、IQ 波形、实验评估、模型参数和 JSON 结果导出。
上传支持 `.npy`、`.npz` 和无表头两列 `I,Q` 的 `.csv`，每次最多 64 条脉冲、8 MB。
合成模型要求 1024 点 / 1 MHz，官方演示模型要求 1000 点 / 50 MHz。
识别记录只在当前浏览器保存预测元数据，不保存上传的原始 IQ。
界面使用已有 `outputs/` 权重和实验结果，不触发训练或数据下载。
若复制项目后 `.venv` 的 Python 链接失效，需使用可用的 Python 环境或重建虚拟环境。

## 最新简化版：官方 30 类数据

官方数据可以下载。已获取并校验新版 AIMC-Spec 的全部 30 个测试类别文件，
每类 1000 条原始脉冲。数据来源是作者公开发布的
[AIMC-Spec-v2](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2)，版本 1。
[新版论文](https://arxiv.org/abs/2601.08265v2)明确写 30 类，原仓库 README 的 33 类描述与之不一致。

按用户要求采用简单、低负载方案：停止完整训练集下载和 CUDA 安装，保留现有 CPU 环境。
从已下载的官方测试文件中，每类选 100 条训练、25 条验证、50 条测试，按原始脉冲重新划分。
排除全零、非有限值和重复原始波形，每个脉冲的 5 档 SNR 样本全部保留在同一集合。
加噪后分别为 15000、3750、7500 条，覆盖全部 30 类。
这是使用官方信号的小规模演示，**不是官方完整训练集实验，也不是官方固定测试协议的基准结果**。

运行默认 20 轮训练并喂入留出脉冲：

```bash
nice -n 10 .venv/bin/python -E -m aimc official-demo --epochs 20
```

该命令默认仅使用 1 个 CPU 线程，计算占空比 25%，进程优先级 nice 10，不使用 GPU。
官方简化版使用原 STFT-CNN 的 6 次方差预处理，学习率为 0.0006，输入图像为 96×96。
`outputs/official_demo/resources.json` 和 `resources.jsonl` 记录计算时间、休息时间和内存占用。
占空比限制的是本训练进程的计算/休息时间，不是整台电脑的固定利用率。
原始数据在 `data/official/raw/AIMC_Spec_v2_test/`，划分及排除清单在 `data/official_demo/manifest.json`。
解码优先使用 `hdf5plugin`；本机未安装该插件时，通过 HDF5 原始块接口调用已有 `libblosc`，没有重新合成官方干净信号。

训练权重、测试指标、曲线、混淆矩阵及留出脉冲预测写入 `outputs/official_demo/`。
本次已实际完成 20 轮：7500 条留出测试样本准确率 **7.61%**，Macro-F1 **0.0654**；
喂入每类一条 6 dB 留出信号，30 条中仅 **2 条正确**。
训练约 18 分钟，计算占空比约 24.6%，包括测试在内的进程峰值内存约 1.9 GB。
这个小样本模型的识别效果很差，权重仅适合展示流程，不能当作可靠的调制识别结果。
该模型的 IQ 窗口是 **1000 点、50 MHz 采样率**，与下文早期 8 类模型的配置不同。

```bash
.venv/bin/python -E -m aimc predict --checkpoint outputs/official_demo/best.pt --input data/official_demo/heldout_pulses.npz --threads 1
```

安装依赖使用 `requirements.txt`；若系统无 `libblosc`，另安装 `hdf5plugin`。
未下载完成的压缩包及已经解压的少量官方训练文件保留在 `data/official/`，当前没有下载或训练后台任务。

## 最初的 8 类合成演示

以下说明记录最初实现的 8 类实验；其数据、权重和结果保留不变。

## 与开源项目的关系

直接导入 [seb-cocks/AIMC-image-classification](https://github.com/seb-cocks/AIMC-image-classification) 的两个源码模块，没有重写其 CNN：

- `src/STFT_CNN/STFT_CNN.py` 中的 `MainModel`。
- `src/common/spectrogram.py` 中的 `batched_iq_to_power_spectrogram`。

上游放在 `third_party/AIMC-image-classification/`，固定提交为
`6d50291fd3ad29389dff872b3ce73c06fb6f5c7c`，原始源码保留。

最初演示的数据是自行生成的 8 类单脉冲 IQ，**不是官方 AIMC-Spec 数据，也不是论文指标复现**。
采用原函数支持的每时刻峰值参考、40 dB 动态范围归一化，双线性缩放到 96×96，灰度重复为三通道。
通过原模型参数 `use_paper_preprocess=False` 关闭迭代绝对值/方差预处理，CNN 主干保持原样。
原项目导出器默认使用 224×224 和另一种噪声归一化，本地配置与其有所区别。
任务为单标签分类，训练使用交叉熵；不处理多发射源重叠的多标签识别。

## 类别与信号

| 标签         | 调制样式       |
| ---------- | ---------- |
| `CW`       | 未调制单频脉冲    |
| `LFM_UP`   | 线性上扫频      |
| `LFM_DOWN` | 线性下扫频      |
| `NLFM`     | 幂函数型非线性上扫频 |
| `BPSK`     | 二相编码       |
| `BFSK`     | 二频移键控      |
| `4FSK`     | 四频移键控      |
| `SFM`      | 正弦调频       |

默认每条 IQ 有 1024 点，采样率为 1 MHz。载频、带宽、初相、幅度、码元数等随机变化；
SNR 为 `[-6, -3, 0, 3, 6] dB`，定义为脉冲平均信号功率与复高斯噪声期望功率之比。
训练、验证、测试分别使用随机种子 42、43、44，独立生成。训练前还会检查跨集合完全重复的 IQ。
这些是简化信号模型，不覆盖实际接收机失真、未知类别和完整工程信号条件。

## 已完成的训练

| 项目            | 本次运行                                |
| ------------- | ----------------------------------- |
| 环境            | Python 3.13，PyTorch CPU，4 线程        |
| 数据量           | 训练 2048，验证 512，测试 800               |
| 训练            | 30 轮，batch 64，Adam，初始学习率 0.003，余弦衰减 |
| 最佳权重          | 按验证损失选取第 27 轮                       |
| 独立测试准确率       | **71.00%**                          |
| 独立测试 Macro-F1 | **0.7058**                          |
| 新生成信号         | 6 dB，独立种子 98765，16 条中 **13 条正确**    |

测试集没有用于早停或选择权重。上述数值仅针对本地合成分布。
当前模型对 CW/BPSK、BFSK/4FSK 的区分仍较弱，具体见分类指标和混淆矩阵。
softmax 分数未做置信度校准，高分也可能预测错误；`--min-confidence` 仅用于标记低分结果，不能可靠检测未知调制。

结果文件：

- `outputs/stft_cnn/best.pt`：权重、类别顺序及预处理配置。
- `outputs/stft_cnn/config.json`：训练配置及上游版本。
- `outputs/stft_cnn/history.json`、`history.csv`：每轮训练/验证记录。
- `outputs/stft_cnn/metrics.json`：测试准确率、Macro-F1、分类别和分 SNR 指标。
- `outputs/stft_cnn/fresh_predictions.json`：16 条新信号的预测、真实标签和分数。
- `outputs/stft_cnn/single_prediction.json`：单条新信号的预测结果。

![训练曲线](outputs/stft_cnn/training_curves.png)
![混淆矩阵](outputs/stft_cnn/confusion_matrix.png)
![各类功率时频图](outputs/stft_cnn/spectrograms.png)

## 运行

在当前目录执行。现有 `.venv` 已安装依赖，`-E` 可避免本机 ROS 的 `PYTHONPATH` 干扰。

一条命令重新生成数据、训练、测试、喂入新信号，会更新默认数据和结果目录：

```bash
.venv/bin/python -E -m aimc demo --epochs 30 --device cpu
```

分步执行：

```bash
.venv/bin/python -E -m aimc generate
.venv/bin/python -E -m aimc train --epochs 30 --device cpu
.venv/bin/python -E -m aimc evaluate --input data/synthetic/test.npz
```

生成并喂入一条新信号：

```bash
.venv/bin/python -E -m aimc sample --class LFM_UP --snr 6 --output data/new_pulse.npy
.venv/bin/python -E -m aimc predict --input data/new_pulse.npy --sample-rate 1000000 --output outputs/stft_cnn/single_prediction.json
```

喂入自己的 IQ：

```bash
.venv/bin/python -E -m aimc predict --input your_iq.npy --sample-rate 1000000 --min-confidence 0.7
```

支持的文件格式：

- `.npy`：复数 `(N,)` 或 `(B,N)`；实数 `(2,N)`、`(N,2)`、`(B,2,N)` 或 `(B,N,2)`。
- `.npz`：`iq` 字段使用上述形状；可附带标量 `sample_rate_hz`，程序会核对采样率。
- `.csv`：无表头，每行两列 `I,Q`，表示一个脉冲。

默认权重要求每个脉冲恰好 1024 点，训练和推理应采用相同的采样率及脉冲分割方式。
程序会拒绝 NaN、Inf、全零脉冲、长度不匹配和已知采样率不匹配的数据。
没有采样率元数据的文件需调用方保证采样率正确。这里不自动裁剪、补零或重采样。

## 用自己的标注数据训练

准备 `data/real/train.npz`、`val.npz`、`test.npz`。三个集合须独立、类别顺序一致。
实际数据应按采集批次/发射源划分，避免同一脉冲或同源近重复样本跨集合。
每个文件的字段如下：

```python
import numpy as np

# iq: complex64，形状 (B,1024)，或上述实数 IQ 形状。
# labels: int64，形状 (B,)，类别索引从 0 开始。
# classes: 字符串数组，形状 (C,)，顺序就是类别索引顺序。
np.savez_compressed(
    "data/real/train.npz",
    iq=iq,
    labels=labels,
    classes=np.array(["CW", "LFM_UP", "LFM_DOWN", "NLFM", "BPSK", "BFSK", "4FSK", "SFM"]),
    sample_rate_hz=1_000_000,
    # snr_db=snr_db,  # 可选，每条样本一个值，用于分 SNR 评估。
)
```

```bash
.venv/bin/python -E -m aimc train --data-dir data/real --out-dir outputs/real --epochs 50
.venv/bin/python -E -m aimc predict --checkpoint outputs/real/best.pt --input your_iq.npy --sample-rate 1000000
```

自己的训练数据可以使用任意不少于两类的类别名称；合成器限定为上表 8 类。
采样率、脉冲长度可分别用 `train --sample-rate`、`--samples` 配置，推理自动读取权重内的预处理参数。
完整 AIMC-Spec 的 HDF5/多标签 memmap 不能直接当作本地 NPZ 使用；其导出和基准训练见上游 README。

## 环境与测试

新环境推荐 Python 3.10+：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install pytest
.venv/bin/python -E -m pytest -q tests
```

`requirements-lock.txt` 记录本次 Python 3.13 CPU 环境的具体版本。
GPU 需自行安装适配驱动的 PyTorch，再使用 `--device cuda`；本机此次使用 CPU。
测试覆盖独立划分、SNR 定义、IQ 格式校验、谱图批处理一致性、上游模型反向传播、
指标计算、权重重载和文件推理一致性。

## 来源

上游项目关联论文：Sebastian L. Cocks 等，AIMC-Spec，arXiv:2601.08265。
参考 https://arxiv.org/abs/2601.08265 和原项目 `README.md`、`ref.bib`。
此次下载的上游仓库未包含单独的 LICENSE 文件，保留来源与作者信息，分发或其他使用前应确认原作者许可。
