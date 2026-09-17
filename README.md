
<p align="center">
  <a href="https://explorerglobal.cn/">
    <img src="assets/logo.png" alt="知天下 AI — 公司官网" width="280">
  </a>
</p>

<!-- <h1 align="center">Explorer</h1> -->
##

<p align="center"><strong>基于三维重建条件的图像修复与新视角合成</strong></p>

<p align="center">
  <a href="https://huggingface.co/XplorerAI/ExplorerAI-v0.1-1.3b">
    <img src="https://img.shields.io/badge/Hugging%20Face-Weights-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face 权重下载">
  </a>
  <a href="https://explorerglobal.cn/">
    <img src="https://img.shields.io/badge/Website-explorerglobal.cn-073363" alt="知天下 AI 公司官网">
  </a>
</p>

Explorer 以 NVIDIA ArtiFixer 1.3B 为初始化进行进一步微调。给定参考照片、相机参数和目标视角轨迹，本文所述流程先训练粗高斯模型，再结合其 RGB 渲染、不透明度和场景尺度信息，使用 Explorer 生成目标视角的修复图像。

## 可视化结果展示

### TowerShipModel

https://github.com/user-attachments/assets/d32b91c5-5aa9-49eb-915c-9ee43752be4b

<!-- <video src="https://github.com/eXplorerAI-CN/Explorer/tree/main/assets/demo/assets/demo/sailing-ship-comparison.mp4" controls muted playsinline preload="metadata" width="100%"></video> -->


<!-- ### XJTLUGateOfWisdomStoneGate

<video src="assets/demo/stone-gate-comparison.mp4" controls muted playsinline preload="metadata" width="100%"></video> -->

### JinshanlingGreatWall

https://github.com/user-attachments/assets/2d012e02-63de-4faf-b28a-48288353369f

<!-- <video src="https://github.com/eXplorerAI-CN/Explorer/tree/main/assets/demo/assets/demo/great-wall-comparison.mp4" controls muted playsinline preload="metadata" width="100%"></video> -->

### SuzhouOfficeBuilding

<!-- <video src="https://github.com/eXplorerAI-CN/Explorer/tree/main/assets/demo/assets/demo/urban-buildings-comparison.mp4" controls muted playsinline preload="metadata" width="100%"></video> -->

https://github.com/user-attachments/assets/e54f5ea0-c936-44e6-99b8-425783c25379



## 测试数据

**测试数据下载：** [阿里云 OSS](https://pubres.explorerglobal.cn/ai/explorer_inference_demo.zip) · [Google Drive（谷歌网盘）](https://drive.google.com/file/d/1iG8mYy2K4dsNDKPML9rxNjeN39XK5L88/view?usp=sharing)

<!-- 发布前，将下方两个占位符替换为实际下载地址，并移除“链接待补充”。 -->
[dataset-oss]: <ALIYUN_OSS_DATASET_URL>
[dataset-google-drive]: <GOOGLE_DRIVE_DATASET_URL>

测试数据包含以下五个场景。每个场景由 `colmap_reference/` 和 `colmap_trajectory/` 两部分组成，分别提供参考视角数据、目标轨迹数据以及目标视角下真实图像。两者均包含 `images/` 与 `sparse/` 目录。

### 数据目录

下载并解压后，将包含这五个场景文件夹的目录设为 `DATA_ROOT`：

```text
<DATA_ROOT>/
├── JinshanlingGreatWall/
│   ├── colmap_reference/
│   │   ├── images/
│   │   └── sparse/
│   └── colmap_trajectory/
│       ├── images/
│       └── sparse/
├── QiandaoLakeVisitorCenter/..
├── SuzhouOfficeBuilding/..
├── TowerShipModel/..
└── XJTLUGateOfWisdomStoneGate/..
```

`SCENE_NAME` 必须与上述场景文件夹名称完全一致，区分大小写。参考相机和目标相机需使用同一场景坐标系。

### 数据与推理入口的对应关系

准备阶段将两个 COLMAP 目录分别传给 `prepare_no_caption_coarse.py`：

| 数据 | 命令参数 | 说明 |
| --- | --- | --- |
| `<SCENE_NAME>/colmap_reference/` | `--reference-colmap` | 参考图像及其 COLMAP 数据，用于准备参考输入和训练粗高斯模型。 |
| `<SCENE_NAME>/colmap_trajectory/` | `--trajectory-colmap` | 目标视角的 COLMAP 数据，用于准备渲染和推理所需的相机轨迹。 |

这两个参数均指向包含 `images/` 和 `sparse/` 的父目录。准备阶段生成后续渲染使用的 `inputs/trajectory.json`。

## 推理流程

```text
准备数据与模型权重 → 构建镜像 → 启动容器
    → 准备参考图像与目标相机
    → 训练粗高斯模型 → 渲染 RGB / 不透明度
    → MoGe3 尺度估计 → Explorer 图像修复
```

示例沿用 **12 张参考图、81 个目标视角、1.3B Stage1 双向配置、50 步采样、无 caption**。其中，高斯训练用于重建当前场景，不会微调 Explorer 权重。参考图数量和轨迹长度是本示例配置，不代表所有发布场景均采用相同数量。

### 1. 环境与模型准备

运行环境需要 Linux、NVIDIA GPU、兼容的 NVIDIA 驱动、Docker、已配置 GPU 支持的 NVIDIA Container Toolkit，以及 Git。

本文使用 `Dockerfile.3090-unified`。文件名不代表实测显存要求；支持的 GPU、峰值显存和耗时需以最终发布配置的实测结果为准。

**Explorer 权重下载：** [🤗 XplorerAI/ExplorerAI-v0.1-1.3b](https://huggingface.co/XplorerAI/ExplorerAI-v0.1-1.3b)。

下载后，将 Explorer checkpoint 放到下方约定的本地位置，或相应修改推理命令中的 `--checkpoint`。其余依赖模型的下载地址和版本将在发布时补齐。

在本地准备以下模型文件：

```text
<MODEL_ROOT>/
├── Wan-AI/Wan2.1-T2V-1.3B-Diffusers/   # 完整的本地 Diffusers 模型目录
├── Ruicheng/moge-3-vitl/model.pt
└── eXplorerAI-CN/Explorer/Explorer-1.3b.pt
```

| 组件 | 用途 |
| --- | --- |
| Wan 2.1 1.3B Diffusers 目录 | 通过 `--model-id` 加载模型配置及所需组件。 |
| MoGe3 权重 | 为场景尺度估计提供几何估计。 |
| Explorer 权重 | 通过 `--checkpoint` 加载进一步微调后的权重。 |

`default_negative_prompt.pt` 也需位于发布代码预期的位置。无 caption 表示无需用户提供文本提示词，不代表可以直接删除管线初始化所需的资源文件。

### 2. 构建 Docker 镜像

在**宿主机**上进入已获取的 Explorer 仓库：

```bash
cd /path/to/Explorer
git submodule update --init --recursive

docker build -f Dockerfile.3090-unified -t explorer-unified:3090-cu128-v1 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PYTORCH_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PYTORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu128/ .
```

以上软件源沿用团队提供的构建配置。如需替换，请确保新源提供 Dockerfile 要求的软件包和 CUDA 版本。构建镜像需要联网。

### 3. 设置路径并启动容器

在**宿主机的同一个 Bash 会话**中执行，将示例路径替换为本机绝对路径：

```bash
CODE='/path/to/Explorer'               # 代码目录
DATA_ROOT='/path/to/datasets'          # 测试数据解压后的根目录
MODEL_ROOT='/path/to/models'           # 模型文件根目录
SCENE_NAME='XJTLUGateOfWisdomStoneGate' # 可替换为上述任一场景文件夹名称
OUTPUT='/path/to/results/run-001'      # 本次运行的结果目录
GPU_ID=0

mkdir -p "$OUTPUT"

docker run --rm -it --gpus "device=${GPU_ID}" --ipc=host --network none \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,exec,size=16g,mode=1777 \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e DATA_ROOT=/data -e MODEL_ROOT=/models \
  -e SCENE_NAME="$SCENE_NAME" -e OUTPUT=/output \
  --mount "type=bind,src=$CODE,dst=/workspace/Explorer,readonly" \
  --mount "type=bind,src=$DATA_ROOT,dst=/data,readonly" \
  --mount "type=bind,src=$MODEL_ROOT,dst=/models,readonly" \
  --mount "type=bind,src=$OUTPUT,dst=/output" \
  --workdir /workspace/Explorer \
  explorer-unified:3090-cu128-v1
```

每次运行请使用新的空输出目录。`mkdir -p` 仅创建目录，不会检查已有目录是否为空。宿主机用户需有输入与模型的读取权限，以及输出目录的写入权限。

| 宿主机路径 | 容器内路径 | 权限 |
| --- | --- | --- |
| `$CODE` | `/workspace/Explorer` | 只读 |
| `$DATA_ROOT` | `/data` | 只读 |
| `$MODEL_ROOT` | `/models` | 只读 |
| `$OUTPUT` | `/output` | 读写 |

容器内的 `$OUTPUT` 为 `/output`，结果会保存到宿主机配置的输出目录。`/tmp` 随容器删除，其 16 GB 限额是内存文件系统上限，不是 GPU 显存。

容器采用 `--network none`，因此启动前需备齐全部模型文件及其他运行资源。镜像名称和挂载目录不要求宿主机存在 `/data16t`。

**以下命令均在该容器内执行。**

### 4. 准备场景输入

根据 `DATA_ROOT` 和 `SCENE_NAME` 设置参考数据与目标轨迹目录：

```bash
REFERENCE_COLMAP="${DATA_ROOT}/${SCENE_NAME}/colmap_reference"
TRAJECTORY_COLMAP="${DATA_ROOT}/${SCENE_NAME}/colmap_trajectory"

python model_eval/prepare_no_caption_coarse.py \
  --stage prepare \
  --reference-colmap "$REFERENCE_COLMAP" \
  --trajectory-colmap "$TRAJECTORY_COLMAP" \
  --output-dir "$OUTPUT"
```

该步骤准备以下产物，后续阶段统一使用这些路径：

- `inputs/reference/`：参考照片副本与 COLMAP 相机数据。
- `inputs/trajectory/`：目标相机的 COLMAP 数据。
- `inputs/trajectory.json`：供粗模型渲染器使用的目标相机轨迹。

相机转换需与图像几何保持一致；对于含畸变的输入，需要正确处理图像及其标定参数。

### 5. 训练粗高斯模型

使用本示例的全部 12 张参考照片训练当前场景：

```bash
python -u model_eval/prepare_no_caption_coarse.py \
  --stage train \
  --reference-colmap "$OUTPUT/inputs/reference" \
  --output-dir "$OUTPUT/coarse-model" \
  --training-steps 10000 \
  --workers 4 \
  --seed 42
```

预期产物包括 `coarse_model.ply` 和 3DGUT 训练 checkpoint。实际路径以 `coarse-model/COARSE_TRAIN.json` 为准。

### 6. 渲染粗 RGB 与不透明度

使用**上一步得到的 3DGUT checkpoint** 渲染目标相机：

```bash
python -u model_eval/prepare_no_caption_coarse.py \
  --stage render \
  --reference-colmap "$OUTPUT/inputs/reference" \
  --trajectory-json "$OUTPUT/inputs/trajectory.json" \
  --checkpoint "$OUTPUT/coarse-model/training/reference/ours_10000/ckpt_10000.pt" \
  --output-dir "$OUTPUT/coarse-render"
```

如果修改训练步数或保存位置，请根据 `COARSE_TRAIN.json` 同步修改 checkpoint 路径。这里使用的是场景重建 checkpoint；Explorer 权重在第 8 步加载。

对于 81 个目标视角，输出编号为 `00000.png` 至 `00080.png`：

- `coarse-render/coarse-rgb/`：粗模型 RGB 渲染图。
- `coarse-render/coarse-mask/`：同次渲染的累计不透明度，按 `round(alpha × 255)` 保存为灰度图。0 表示透明，255 表示不透明，中间值保留。

请保持 RGB、不透明度图与目标相机顺序一一对应；不要将连续不透明度图阈值化为二值 mask。

### 7. 使用 MoGe3 估计场景尺度

```bash
python -u model_eval/estimate_no_caption_scale.py \
  --reference-images "$OUTPUT/inputs/reference/images" \
  --reference-colmap "$OUTPUT/inputs/reference/sparse/0" \
  --moge-model "$MODEL_ROOT/Ruicheng/moge-3-vitl/model.pt" \
  --output-dir "$OUTPUT/scale"
```

后续推理命令读取 `scale/SCALE.json` 中的尺度报告。

### 8. 运行 Explorer

```bash
python -u model_eval/run_inference-no_caption-from_colmap.py \
  --reference-images "$OUTPUT/inputs/reference/images" \
  --reference-colmap "$OUTPUT/inputs/reference/sparse/0" \
  --trajectory-colmap "$OUTPUT/inputs/trajectory/sparse/0" \
  --rgb-dir "$OUTPUT/coarse-render/coarse-rgb" \
  --mask-dir "$OUTPUT/coarse-render/coarse-mask" \
  --scale-json "$OUTPUT/scale/SCALE.json" \
  --model-id "$MODEL_ROOT/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --checkpoint "$MODEL_ROOT/eXplorerAI-CN/Explorer/Explorer-1.3b.pt" \
  --steps 50 \
  --output-dir "$OUTPUT/Explorer"
```

该入口不要求用户输入文本提示词。请配套使用 1.3B 模型目录和 Explorer 权重；内部文本特征构造和注意力设置由发布脚本控制。

### 输出结果

本示例的预期输出结构如下：

```text
<OUTPUT>/
├── inputs/
│   ├── reference/                  # 准备后的参考照片与 COLMAP 数据
│   ├── trajectory/                 # 准备后的目标相机 COLMAP 数据
│   └── trajectory.json             # 供粗模型渲染器使用的目标相机
├── COARSE_PREPARE.json
├── coarse-model/
│   ├── coarse_model.ply
│   ├── training/reference/ours_10000/ckpt_10000.pt
│   └── COARSE_TRAIN.json
├── coarse-render/
│   ├── coarse-rgb/                 # 81 张粗 RGB 图像
│   ├── coarse-mask/                # 81 张灰度不透明度图
│   └── COARSE_RENDER.json
├── scale/
│   └── SCALE.json
└── Explorer/
    ├── INPUT_FRAME_MAP.tsv         # 目标相机与 RGB/mask 的对应关系
    ├── RUN.json
    ├── SUCCESS.json               # 成功完成后生成
    ├── pred/                      # 00000.png ... 00080.png
    └── final/                     # 按目标轨迹命名的图像
        └── SHA256SUMS
```

退出容器后，结果仍保存在宿主机。输出图像采用 PNG 格式。

## 与 ArtiFixer 的关系及差异分析

Explorer 使用 ArtiFixer 1.3B 权重初始化，并在此基础上进一步微调。下面分别说明已测量的权重变化。

### 权重更新幅度

我们对指定的初始化 checkpoint 与进一步训练后的 checkpoint 进行了逐张量比较。两者的 **1,095 个张量名称及形状一致，全部张量均有数值变化**。这表明该 checkpoint 在现有参数结构上进行了更新；不能据此推断具体优化器或训练方式。

设原始权重为 $W_0$，微调后权重为 $W_1$，更新量为 $\Delta W=W_1-W_0$。相对更新幅度定义为：

$$
r=\frac{\|\Delta W\|_F}{\|W_0\|_F}.
$$

<!-- 朴素理解：先把所有参数的变化量平方求和再开方，得到“总共改动了多少”；再除以原始权重的大小，便于比较不同模块的相对改动。整体指标使用全部对齐参数；模块指标先在对应矩阵组内合并平方和，再计算比值。 -->

| 权重空间指标 | 实测结果 | 含义或参数作用 |
| --- | --- | --- |
| 整体相对 Frobenius | **1.61%** | 衡量全部参数相对于原始权重范数的总体更新幅度 |
| 邻视图 V 投影矩阵 | **47.52%** | 将邻视图内容映射为交叉注意力的值特征，影响纹理和外观信息如何参与目标视角生成。 |
| 相机条件矩阵 | **21.69%** | 将相机射线几何映射为内部特征，影响模型如何利用视点和观察方向信息。 |
| 不透明度条件矩阵 | **14.31%** | 编码重建渲染的不透明度，为模型提供几何覆盖程度的线索，参与修复与补全。 |

后三项在全部 30 个 Transformer block 的对应权重矩阵上汇总，不含偏置。原始矩阵范数较小时，相对更新百分比也可能较大。

从相对更新幅度看，邻视图信息和几何条件相关矩阵变化较突出；从更新能量总量看，FFN 与自注意力模块合计约占 **66.48%**，说明主干也有广泛更新。


## 许可

Explorer 的初始化来源为 [ArtiFixer 1.3B](https://huggingface.co/nvidia/ArtiFixer)，衍生权重的使用需遵循上游 [NVIDIA 许可证](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf)的非商业研究与评估限制。具体条款及 NVIDIA 与其关联公司的例外以许可证全文为准。Wan 基础模型的 Apache 2.0 许可不会替代 ArtiFixer 权重的限制。

代码许可证与测试数据许可将在本次发布时明确；第三方代码、权重和数据保留各自的许可及版权声明。

## 致谢

感谢以下项目及其作者开放代码、模型和研究成果：

- [ArtiFixer](https://github.com/nv-tlabs/ArtiFixer)：提供模型初始化及上游图像修复管线，是本工作的直接基础。
- [MoGe](https://github.com/microsoft/MoGe)：提供单目几何估计，用于本流程的场景尺度估计。
- [3DGRUT](https://github.com/nv-tlabs/3dgrut)：提供高斯场景重建与渲染。请使用 Explorer 仓库包含的兼容依赖版本。
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1)：提供底层模型系列。

## 加入我们

我们关注三维重建、新视角合成与生成式模型，欢迎对这些方向感兴趣的研究者和工程师加入 Explorer，一起探索三维场景的理解与生成。

如果你有相关研究或工程经验，欢迎将简历发送至 **[cv@explorer.global](mailto:cv@explorer.global)**，并附上能够展示你工作的 GitHub 项目、论文或 Demo。

邮件主题建议使用：`加入 Explorer－姓名－研究或技术方向`。
