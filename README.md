# 推理流程

按顺序执行：**构建镜像 → 启动容器 → 准备、训练和渲染 3DGS → MoGe3 → Explorer**。

## 1 用 Dockerfile 构建镜像

```bash
cd Explorer
git submodule update --init --recursive

docker build -f Dockerfile.3090-unified -t Explorer-unified:3090-cu128-v1 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PYTORCH_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PYTORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu128/ .
```


## 2 设置路径并启动容器

在 **3090 宿主机**设置数据集、场景名和结果目录。第一次运行时，`OUTPUT` 使用空目录。

后面的命令均在 **容器内**执行；三个路径变量已通过 `-e` 传入。各阶段依次运行，进程退出后释放显存。首次训练会编译官方 CUDA/Slang 扩展，需要几分钟。

```bash
CODE='/data16t/code/Explorer/'
DATA_ROOT='/data16t/生成式模型测试数据集/release_to_github' # 数据集根目录
SCENE_NAME='西浦智慧之门石门'                             # 本次场景名
OUTPUT="${DATA_ROOT}/${SCENE_NAME}/006-result"        # 保存目录，可手动修改
mkdir -p "$OUTPUT"

docker run --rm -it --gpus device=0 --ipc=host --network none \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,exec,size=16g,mode=1777 \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e DATA_ROOT="$DATA_ROOT" -e SCENE_NAME="$SCENE_NAME" -e OUTPUT="$OUTPUT" \
  --mount "type=bind,src=$CODE,dst=/workspace/Explorer,readonly" \
  --mount type=bind,src=/data16t,dst=/data16t,readonly \
  --mount "type=bind,src=$OUTPUT,dst=$OUTPUT" \
  --workdir /workspace/Explorer \
  Explorer-unified:3090-cu128-v1
```


## 3 从 12 张参考图训练 3DGS，并渲染 coarse RGB / mask
### 3.1 准备适用于 3dgut 格式的数据

- 原始数据保持只读。内部相机转成等价的零畸变 OPENCV 表示，保留内参和世界坐标；后续步骤都使用这些准备好的路径。
- `$OUTPUT/inputs/reference/`：12 张参考照片副本及 `sparse/0/` 的 COLMAP BIN/TXT 模型。
- `$OUTPUT/inputs/trajectory/`：规范化后的目标相机 COLMAP，供 Explorer 使用。
- `$OUTPUT/inputs/trajectory.json`：81 个目标相机，供官方渲染器使用。

```bash
python model_eval/prepare_no_caption_coarse.py \
  --stage prepare \
  --reference-colmap "${DATA_ROOT}/${SCENE_NAME}/reference_view_12images" \
  --trajectory-colmap "${DATA_ROOT}/${SCENE_NAME}/trajectory_81_poses" \
  --output-dir "$OUTPUT"
```
### 3.2 用 3dgut 训练粗高斯模型

全部 12 张参考照片用于训练，得到 `coarse_model.ply` 和官方 checkpoint。实际路径记录在 `$OUTPUT/coarse-model/COARSE_TRAIN.json`。

```bash
python -u model_eval/prepare_no_caption_coarse.py \
  --stage train \
  --reference-colmap "$OUTPUT/inputs/reference" \
  --output-dir "$OUTPUT/coarse-model" \
  --training-steps 10000 \
  --workers 4 \
  --seed 42
```
### 3.3 沿 81 个目标相机渲染

渲染使用上一步训练的 **粗模型 checkpoint**，保留完整 3DGUT 配置。若修改训练步数，请按 `COARSE_TRAIN.json` 更新 checkpoint 路径。结果自动保存为各 81 张 PNG，编号 `00000.png … 00080.png`：
- `coarse-render/coarse-rgb/`：粗模型颜色渲染。
- `coarse-render/coarse-mask/`：同次渲染的累计不透明度 alpha，按 `round(alpha × 255)` 保存为灰度图。0 表示透明/缺失，255 表示不透明，中间灰度保留。

```bash
python -u model_eval/prepare_no_caption_coarse.py \
  --stage render \
  --reference-colmap "$OUTPUT/inputs/reference" \
  --trajectory-json "$OUTPUT/inputs/trajectory.json" \
  --checkpoint "$OUTPUT/coarse-model/training/reference/ours_10000/ckpt_10000.pt" \
  --output-dir "$OUTPUT/coarse-render"
```


## 4 运行 MoGe3

```bash
python -u model_eval/estimate_no_caption_scale.py \
  --reference-images "$OUTPUT/inputs/reference/images" \
  --reference-colmap "$OUTPUT/inputs/reference/sparse/0" \
  --moge-model '/data16t/huggingface/Ruicheng/moge-3-vitl/model.pt' \
  --output-dir "$OUTPUT/scale"
```


## 5 运行 Explorer

- 使用 **1.3B、Stage1 bidirectional、12 张参考、50 步、无 caption**。
- `default_negative_prompt.pt` 是官方初始化依赖；实际文本特征为 `[1,512,4096]` 的 BF16 全零张量，`text_guidance_scale=1.0`，负提示特征不参与预测。
- `--checkpoint` Explorer 1.3B 模型权重路径。
- `--output-dir` 结果保存路径。

```bash
python -u model_eval/run_inference-no_caption-from_colmap.py \
  --reference-images "$OUTPUT/inputs/reference/images" \
  --reference-colmap "$OUTPUT/inputs/reference/sparse/0" \
  --trajectory-colmap "$OUTPUT/inputs/trajectory/sparse/0" \
  --rgb-dir "$OUTPUT/coarse-render/coarse-rgb" \
  --mask-dir "$OUTPUT/coarse-render/coarse-mask" \
  --scale-json "$OUTPUT/scale/SCALE.json" \
  --model-id '/data16t/huggingface/Wan-AI/Wan2.1-T2V-1.3B-Diffusers' \
  --checkpoint '/data16t/huggingface/eXplorerAI-CN/Explorer/Explorer-1.3b.pt' \
  --steps 50 \
  --output-dir "$OUTPUT/Explorer"
```


## 6 输出结果目录结构

所有文件保存在宿主机 `$OUTPUT` 下，退出容器后仍保留。各阶段只创建自己的新目录，不覆盖已有结果。

```text
$OUTPUT/                            # 默认：${DATA_ROOT}/${SCENE_NAME}/006-result
├── inputs/                         # 规范化 COLMAP、参考照片副本、trajectory.json
├── COARSE_PREPARE.json             # 数据准备状态和输入记录
├── coarse-model/
│   ├── coarse_model.ply            # 训练得到的粗高斯模型
│   ├── training/reference/ours_10000/ckpt_10000.pt
│   └── COARSE_TRAIN.json           # 训练产物路径
├── coarse-render/
│   ├── coarse-rgb/                 # 81 张粗 RGB
│   ├── coarse-mask/                # 81 张灰度不透明度 mask
│   └── COARSE_RENDER.json          # 渲染状态
├── scale/SCALE.json                # MoGe3 尺度报告
└── Explorer/
    ├── INPUT_FRAME_MAP.tsv         # 轨迹与 RGB/mask 的对应关系
    ├── RUN.json / SUCCESS.json     # 推理状态；成功完成才有 SUCCESS
    ├── pred/00000.png … 00080.png  # 81 张无损修复图
    └── final/                      # 按目标轨迹文件名保存的修复图
        └── SHA256SUMS
```



# acknowledge

https://github.com/nv-tlabs/artifixer

https://github.com/microsoft/MoGe







