#!/usr/bin/env python3
"""Explorer：分三次命令准备数据、训练官方粗高斯、渲染 coarse RGB / mask。

prepare：从原始 12 图和 81 相机 COLMAP 生成内部模型及相机 JSON。
train：读取准备好的参考模型，保存高斯 PLY 和官方 checkpoint。
render：读取同次 checkpoint 和相机 JSON，生成 81 张 RGB 和 mask。
每条命令只执行所选阶段；MoGe3 和 Explorer 图像修复另行运行，不需要 caption。
目标照片始终不读取。使用说明见仓库根目录 README.md。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from model_eval.no_caption_inputs import data_lines, read_image, read_model, sha256, transform_frame


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def new_output(path: Path) -> None:
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
            f"输出目录必须为空，避免覆盖已有结果：{path}")
    path.mkdir(parents=True, exist_ok=True)


def unpack(stream, fmt):
    """COLMAP BIN 使用小端序；截断文件要报错，不能悄悄少读相机。"""
    size = struct.calcsize('<' + fmt)
    data = stream.read(size)
    require(len(data) == size, f'COLMAP BIN 文件被截断：{stream.name}')
    return struct.unpack('<' + fmt, data)


def binary_to_text(source, destination, need_points):
    """只转换 COLMAP 序列化格式，保留 pose、内参、点 ID 与 track。

    独立读取标准 BIN，避免为了转换文件加载 CUDA/训练模块。
    转换后交给已有 TXT 校验器检查相机和图片名。
    """
    models = {0: ('SIMPLE_PINHOLE', 3), 1: ('PINHOLE', 4), 4: ('OPENCV', 8)}
    lines = []
    with (source / 'cameras.bin').open('rb') as stream:
        for _ in range(unpack(stream, 'Q')[0]):
            cid, model, w, h = unpack(stream, 'iiQQ')
            require(model in models, f'请先去畸变；不支持 COLMAP camera model ID={model}')
            name, count = models[model]
            values = unpack(stream, 'd' * count)
            lines.append(' '.join(map(str, [cid, name, w, h, *values])) + '\n')
    (destination / 'cameras.txt').write_text(''.join(lines))
    lines = []
    with (source / 'images.bin').open('rb') as stream:
        for _ in range(unpack(stream, 'Q')[0]):
            values = unpack(stream, 'idddddddi')
            name = bytearray()
            while True:
                byte = stream.read(1)
                require(byte, 'images.bin 中图像名没有结束符')
                if byte == b'\0':
                    break
                name.extend(byte)
            count = unpack(stream, 'Q')[0]
            points = unpack(stream, 'ddq' * count)
            lines += [' '.join(map(str, values)) + ' ' + name.decode('utf-8') + '\n',
                      ' '.join(map(str, points)) + '\n']
    (destination / 'images.txt').write_text(''.join(lines), encoding='utf-8')
    if need_points:
        lines = []
        with (source / 'points3D.bin').open('rb') as stream:
            for _ in range(unpack(stream, 'Q')[0]):
                values = unpack(stream, 'QdddBBBd')
                count = unpack(stream, 'Q')[0]
                tracks = unpack(stream, 'ii' * count)
                lines.append(' '.join(map(str, (*values, *tracks))) + '\n')
        (destination / 'points3D.txt').write_text(''.join(lines))


def read_points(path):
    points = {}
    for line in data_lines(path):
        values = line.split()
        require(len(values) >= 8 and (len(values) - 8) % 2 == 0, f'无效稀疏点：{path}')
        pid = int(values[0])
        require(pid not in points, f'重复 POINT3D_ID：{pid}')
        xyz = list(map(float, values[1:4]))
        rgb, error = list(map(int, values[4:7])), float(values[7])
        require(all(math.isfinite(v) for v in [*xyz, error]) and all(0 <= v <= 255 for v in rgb),
                f'稀疏点值无效：{pid}')
        tracks = [(int(values[i]), int(values[i + 1])) for i in range(8, len(values), 2)]
        points[pid] = dict(xyz=xyz, rgb=rgb, error=error, tracks=tracks)
    require(points, '参考 COLMAP 需要非空 points3D，用于初始化粗模型及估计尺度')
    return points


def read_colmap(root, need_points=True):
    """接受 sparse/0 或 sparse；同一层 BIN/TXT 都有时明确优先 TXT。"""
    root = Path(root).expanduser().resolve()
    names = ['cameras', 'images'] + (['points3D'] if need_points else [])
    for directory in (root / 'sparse/0', root / 'sparse'):
        for extension in ('txt', 'bin'):
            if not all((directory / f'{name}.{extension}').is_file() for name in names):
                continue
            with tempfile.TemporaryDirectory(prefix='explorer-colmap-') as temporary:
                source = directory
                if extension == 'bin':
                    source = Path(temporary)
                    binary_to_text(directory, source, need_points)
                _, cameras, frames = read_model(source)
                points = read_points(source / 'points3D.txt') if need_points else {}
            # 官方 COLMAP loader 也按图像名排序。新生成的 RGB/mask 使用此顺序。
            frames.sort(key=lambda f: f['name'])
            return dict(source_dir=directory, cameras=cameras, frames=frames, points=points,
                        source_files=[directory / f'{name}.{extension}' for name in names])
    raise ValueError(f'找不到完整 COLMAP 模型：{root}/sparse[/0]，需要 {names} 的 TXT 或 BIN')


def check_inputs(reference_root, trajectory_root):
    reference_root, trajectory_root = [Path(p).expanduser().resolve() for p in (reference_root, trajectory_root)]
    reference = read_colmap(reference_root)
    target = read_colmap(trajectory_root, need_points=False)
    require(len(reference['frames']) == 12, '参考 COLMAP 必须注册恰好 12 张照片')
    require(len(target['frames']) == 81, '目标 COLMAP 必须注册恰好 81 个相机位姿')
    sizes = {(m['cameras'][f['camera_id']]['w'], m['cameras'][f['camera_id']]['h'])
             for m in (reference, target) for f in m['frames']}
    require(len(sizes) == 1, '参考与目标相机必须使用同一画布尺寸')
    require(len({f['camera_id'] for f in target['frames']}) == 1, '81 个目标位姿须共用一个相机内参')
    size = sizes.pop()
    reference_paths = [reference_root / 'images' / f['name'] for f in reference['frames']]
    for path in reference_paths:
        read_image(path, size, 'reference').close()
    # 不列出、不打开 trajectory_root/images 中的文件。
    return dict(reference_model=reference, trajectory_model=target, targets=target['frames'],
                size=size, reference_paths=reference_paths)


def write_model(destination, model):
    """写标准 TXT+BIN；使用等价 OPENCV 零畸变相机，保留真实主点。"""
    destination.mkdir(parents=True)
    frames, points = model['frames'], model['points']
    used_camera_ids = {frame['camera_id'] for frame in frames}
    cameras = {cid: camera for cid, camera in model['cameras'].items() if cid in used_camera_ids}
    camera_lines, image_lines, point_lines = [], [], []
    with (destination / 'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(cameras)))
        for cid, camera in cameras.items():
            values = [camera[k] for k in ('fl_x', 'fl_y', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2')]
            stream.write(struct.pack('<iiQQ8d', cid, 4, camera['w'], camera['h'], *values))
            camera_lines.append(' '.join(map(str, [cid, 'OPENCV', camera['w'], camera['h'], *values])) + '\n')
    with (destination / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(frames)))
        for frame in frames:
            values = [frame['id'], *frame['q'], *frame['t'], frame['camera_id']]
            stream.write(struct.pack('<idddddddi', *values))
            stream.write(frame['name'].encode('utf-8') + b'\0')
            stream.write(struct.pack('<Q', len(frame['points2d'])))
            observations = []
            for x, y, pid in frame['points2d']:
                stream.write(struct.pack('<ddq', x, y, pid))
                observations += [x, y, pid]
            image_lines += [' '.join(map(str, values)) + ' ' + frame['name'] + '\n',
                            ' '.join(map(str, observations)) + '\n']
    with (destination / 'points3D.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(points)))
        for pid, point in points.items():
            values = [pid, *point['xyz'], *point['rgb'], point['error']]
            stream.write(struct.pack('<QdddBBBd', *values))
            stream.write(struct.pack('<Q', len(point['tracks'])))
            tracks = []
            for image_id, point_index in point['tracks']:
                stream.write(struct.pack('<ii', image_id, point_index))
                tracks += [image_id, point_index]
            point_lines.append(' '.join(map(str, [*values, *tracks])) + '\n')
    for name, lines in [('cameras', camera_lines), ('images', image_lines), ('points3D', point_lines)]:
        (destination / f'{name}.txt').write_text(''.join(lines), encoding='utf-8')


def prepare_inputs(reference_root, trajectory_root, output):
    info = check_inputs(reference_root, trajectory_root)
    output = Path(output).expanduser().resolve()
    for root in (reference_root, trajectory_root):
        root = Path(root).expanduser().resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output), '输出不能与输入目录重叠')
    destination = output / 'inputs'
    require(not destination.exists(), f'输入副本已存在，拒绝覆盖：{destination}')
    reference = destination / 'reference'
    target = destination / 'trajectory'
    model = copy.deepcopy(info['reference_model'])
    frames_by_id = {f['id']: f for f in model['frames']}
    # 子集 COLMAP 有时残留其他图的点轨迹。粗模型只用 12 张参考真正观测的点。
    points = {}
    for pid, point in model['points'].items():
        tracks = [(iid, index) for iid, index in point['tracks']
                  if iid in frames_by_id and 0 <= index < len(frames_by_id[iid]['points2d'])
                  and frames_by_id[iid]['points2d'][index][2] == pid]
        if tracks:
            points[pid] = {**point, 'tracks': tracks}
    require(points, '没有与 12 张参考照片双向关联的稀疏点')
    model['points'] = points
    names = []
    for index, frame in enumerate(model['frames']):
        original = frame['name']
        # 官方 BIN reader 对中文文件名逐字节解码；内部使用 ASCII 别名绕开此限制。
        if not original.isascii() or 'images' in original:
            frame['name'] = f'__reference_{index:05d}{Path(original).suffix.lower()}'
        names.append((original, frame['name']))
        frame['points2d'] = [(x, y, pid if pid in points and (frame['id'], i) in points[pid]['tracks'] else -1)
                             for i, (x, y, pid) in enumerate(frame['points2d'])]
    require(len({name for _, name in names}) == 12, '参考图内部别名与现有文件名冲突')
    write_model(reference / 'sparse/0', model)
    # 目标副本只保存相机；points3D 为空时也清空二维观测，避免悬空点关联。
    target_model = {**info['trajectory_model'],
                    'frames': [{**frame, 'points2d': []} for frame in info['targets']]}
    write_model(target / 'sparse/0', target_model)
    (reference / 'images').mkdir()
    for source, (_, name) in zip(info['reference_paths'], names):
        shutil.copyfile(source, reference / 'images' / name)
    write_json(destination / 'reference_names.json', dict(names))
    # 这里不复制目标照片；只有 81 个相机的参数。
    frames = [transform_frame(f, info['trajectory_model']['cameras'][f['camera_id']]) for f in info['targets']]
    trajectory_json = destination / 'trajectory.json'
    write_json(trajectory_json, {**info['trajectory_model']['cameras'][info['targets'][0]['camera_id']], 'frames': frames})
    return dict(reference_root=reference, trajectory_root=target, trajectory_json=trajectory_json,
                size=info['size'], targets=info['targets'])


def prepare(args: argparse.Namespace) -> dict | None:
    """只准备内部副本，原始模型和参考照片在准备前后校验哈希。"""
    require(args.trajectory_colmap is not None, "prepare 阶段必须提供原始 --trajectory-colmap 根目录")
    info = check_inputs(args.reference_colmap, args.trajectory_colmap)
    print(f"INPUT_OK：12 张参考，81 个目标相机，尺寸 {info['size']}；目标照片未读取", flush=True)
    if args.check_only:
        return None
    for root in (args.reference_colmap, args.trajectory_colmap):
        require(not args.output_dir.is_relative_to(root) and not root.is_relative_to(args.output_dir),
                "输出不能与输入目录重叠")
    input_files = info['reference_model']['source_files'] + info['trajectory_model']['source_files'] + info['reference_paths']
    receipt = dict(status='RUNNING', reference_colmap=str(args.reference_colmap),
                   trajectory_colmap=str(args.trajectory_colmap), output_dir=str(args.output_dir),
                   inputs_sha256={str(path): sha256(path) for path in input_files},
                   reference_images=12, target_frames=81, image_size=list(info['size']))
    new_output(args.output_dir)
    report = args.output_dir / 'COARSE_PREPARE.json'
    # 独占创建状态文件，避免同时运行两次 prepare 时互相覆盖。
    with report.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    try:
        prepared = prepare_inputs(args.reference_colmap, args.trajectory_colmap, args.output_dir)
        require(all(sha256(Path(path)) == digest for path, digest in receipt['inputs_sha256'].items()),
                '准备期间原始输入文件发生变化')
        receipt.update(status='SUCCESS', prepared_reference_colmap=str(prepared['reference_root']),
                       prepared_trajectory_colmap=str(prepared['trajectory_root']),
                       trajectory_json=str(prepared['trajectory_json']))
    except BaseException as error:
        receipt.update(status='FAILED', error=str(error))
        raise
    finally:
        write_json(report, receipt)
    return receipt


def check_reference(root: Path) -> None:
    """检查官方训练器所需的内部布局；原始数据检查由 prepare 阶段完成。"""
    import numpy as np
    from PIL import Image
    from threedgrut.datasets.utils import read_colmap_extrinsics_binary, read_colmap_intrinsics_binary

    sparse = root / "sparse/0"
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        require((sparse / name).is_file(), f"缺少规范化的 COLMAP 文件：{sparse / name}")
    frames = read_colmap_extrinsics_binary(sparse / "images.bin")
    cameras = read_colmap_intrinsics_binary(sparse / "cameras.bin")
    require(len(frames) == 12, f"训练需要恰好 12 张参考照片，当前为 {len(frames)}")
    for frame in frames:
        camera = cameras[frame.camera_id]
        # 官方 PINHOLE 分支会把主点设成图像中心。使用等价的零畸变
        # OPENCV 表示，才能保留输入 COLMAP 的 fx、fy、cx、cy。
        require(camera.model == "OPENCV" and len(camera.params) == 8,
                "内部训练相机必须先规范成 OPENCV：fx fy cx cy 0 0 0 0")
        require(np.isfinite(camera.params).all() and np.equal(camera.params[4:], 0).all(),
                "此无 caption 流程只接受已去畸变的针孔相机")
        with Image.open(root / "images" / frame.name) as image:
            require(image.size == (camera.width, camera.height), f"参考照片与相机尺寸不一致：{frame.name}")


def train(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from plyfile import PlyData
    from data_processing.threedgrut_training import DEFAULT_THREEDGRUT_CONFIG_DIR, train_3dgrut

    check_reference(args.reference_colmap)
    require(args.training_steps > 0 and args.workers >= 0, "训练步数必须为正数，workers 不能为负数")
    new_output(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ply_path = args.output_dir / "coarse_model.ply"
    training_root = args.output_dir / "training"
    # 只调整输入/输出、迭代数及数据划分。模型、损失、MCMC 增密与
    # 3DGUT 渲染参数，使用官方 sparse MCMC 配置。
    # 12 张全部用于训练，不抽出验证图；否则 selected_indices=[0..11]
    # 会创建空验证集，官方数据加载器无法初始化它的包围盒。
    overrides = [
        f"path={json.dumps(str(args.reference_colmap), ensure_ascii=False)}",
        f"out_dir={json.dumps(str(training_root), ensure_ascii=False)}",
        'experiment_name=""',
        "selected_indices_file=null",
        "train_test_split_file=null",
        "image_path_override=null",
        "dataset.test_split_interval=0",
        "dataset.downsample_factor=1",
        f"n_iterations={args.training_steps}",
        f"checkpoint.iterations=[{args.training_steps}]",
        f"num_workers={args.workers}",
        f"val_frequency={args.training_steps + 1}",
        "validate_first=False",
        "test_last=False",
        "export_ingp.enabled=False",
        "export_ply.enabled=True",
        f"export_ply.path={json.dumps(str(ply_path), ensure_ascii=False)}",
    ]
    train_3dgrut("apps/colmap_3dgut_sparse_mcmc", overrides, DEFAULT_THREEDGRUT_CONFIG_DIR)

    checkpoint = training_root / args.reference_colmap.stem / f"ours_{args.training_steps}" / f"ckpt_{args.training_steps}.pt"
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"训练未生成 checkpoint：{checkpoint}")
    require(ply_path.is_file(), f"训练未导出 PLY：{ply_path}")
    vertex = PlyData.read(ply_path)["vertex"].data
    require(len(vertex) > 0, "导出的 PLY 不含高斯点")
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    require(required.issubset(vertex.dtype.names or ()), "PLY 缺少标准高斯属性")
    require(all(np.isfinite(vertex[name]).all() for name in vertex.dtype.names), "PLY 高斯属性出现 NaN/Inf")
    return {"status": "SUCCESS", "method": "official 3DGUT sparse MCMC", "reference_colmap": str(args.reference_colmap),
            "training_images": 12, "training_steps": args.training_steps, "seed": args.seed,
            "checkpoint": str(checkpoint), "ply": str(ply_path), "gaussians": len(vertex)}


def render(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from threedgrut.render import Renderer
    from data_processing.camera_trajectories import camera_intrinsics_for_frame, read_camera_trajectory

    require(args.checkpoint is not None and args.checkpoint.is_file(), "render 阶段必须提供有效的 --checkpoint")
    require(args.trajectory_json is not None and args.trajectory_json.is_file(), "render 阶段必须提供 --trajectory-json")
    check_reference(args.reference_colmap)
    trajectory = read_camera_trajectory(args.trajectory_json)
    require(len(trajectory["frames"]) == 81, "目标轨迹必须恰好包含 81 个相机")
    sizes = set()
    for frame in trajectory["frames"]:
        camera = camera_intrinsics_for_frame(trajectory, frame)
        require(np.isfinite(np.asarray(frame["transform_matrix"])).all(), "目标位姿包含 NaN/Inf")
        require(all(np.isfinite(camera[key]) for key in ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2")),
                "目标内参包含 NaN/Inf")
        sizes.add((camera["w"], camera["h"]))
    require(len(sizes) == 1, "81 帧必须使用同一输出尺寸")
    size = next(iter(sizes))
    new_output(args.output_dir)

    # 删除 file_path，仅把相机交给官方渲染器，目标 RGB 绝不参与训练或渲染。
    # read_camera_trajectory 已统一 applied_transform，保存一次供官方直接读取。
    for frame in trajectory["frames"]:
        frame.pop("file_path", None)
    trajectory_path = args.output_dir / "trajectory.json"
    write_json(trajectory_path, trajectory)
    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        out_dir=str(args.output_dir / "official-render"),
        path=str(args.reference_colmap),
        save_gt=False,
        computes_extra_metrics=False,
        config_overrides={"path": str(args.reference_colmap), "experiment_name": "",
                          "selected_indices_file": None, "train_test_split_file": None,
                          "image_path_override": None, "dataset.test_split_interval": 0,
                          "dataset.downsample_factor": 1},
    )
    if renderer.writer is not None:
        renderer.writer.close()
        renderer.writer = None

    checked_frames = 0

    def check_render_outputs(_model, _inputs, outputs):
        """在官方转成 8-bit PNG 前验证浮点输出，避免 NaN 被转成黑像素。"""
        nonlocal checked_frames
        for key in ("pred_rgb", "pred_opacity"):
            require(torch.isfinite(outputs[key]).all().item(), f"第 {checked_frames} 帧 {key} 含 NaN/Inf")
        opacity = outputs["pred_opacity"]
        require(opacity.min().item() >= -1e-5 and opacity.max().item() <= 1 + 1e-5,
                f"第 {checked_frames} 帧 opacity 超出 [0,1]")
        checked_frames += 1

    hook = renderer.model.register_forward_hook(check_render_outputs)
    try:
        renderer.render_from_file(trajectory_path, output_subdir="")
    finally:
        hook.remove()
    require(checked_frames == 81, f"实际只渲染了 {checked_frames} 帧")

    official_output = Path(renderer.out_dir) / f"ours_{int(renderer.global_step)}"
    for source_name, destination_name in (("renders", "coarse-rgb"), ("opacity", "coarse-mask")):
        source = official_output / source_name
        expected = [f"{index:05d}.png" for index in range(81)]
        require(sorted(path.name for path in source.glob("*.png")) == expected, f"官方输出帧不完整：{source}")
        for name in expected:
            with Image.open(source / name) as image:
                image.load()
                require(image.format == "PNG" and image.size == size, f"输出 PNG 格式或尺寸错误：{source / name}")
                require(image.mode == ("RGB" if source_name == "renders" else "L"), f"输出颜色模式错误：{source / name}")
        # 原样移动官方 PNG：mask 是累计 opacity，黑透明、白不透明，保留灰度。
        source.rename(args.output_dir / destination_name)
    return {"status": "SUCCESS", "checkpoint": str(args.checkpoint), "frames": 81, "size": list(size),
            "coarse_rgb": str(args.output_dir / "coarse-rgb"), "coarse_mask": str(args.output_dir / "coarse-mask"),
            "official_render_output": str(official_output)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "render"), required=True)
    parser.add_argument("--reference-colmap", type=Path, required=True,
                        help="prepare：原始 12 图 COLMAP 根目录；train/render：准备好的 inputs/reference")
    parser.add_argument("--trajectory-colmap", type=Path, help="prepare：原始 81 个目标相机的 COLMAP 根目录")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="prepare：整个空结果目录，如 006-result；train/render：各自独立空子目录")
    parser.add_argument("--check-only", action="store_true", help="仅用于 prepare：只读检查原始数据，不写输出")
    parser.add_argument("--training-steps", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectory-json", type=Path, help="render：prepare 生成的 inputs/trajectory.json")
    parser.add_argument("--checkpoint", type=Path, help="render：官方训练生成的 .pt checkpoint")
    args = parser.parse_args(argv)
    if args.check_only and args.stage != 'prepare':
        parser.error('--check-only 仅用于 --stage prepare')
    if args.stage == 'prepare' and args.trajectory_colmap is None:
        parser.error('--stage prepare 需要 --trajectory-colmap')
    for name in ("reference_colmap", "trajectory_colmap", "output_dir", "trajectory_json", "checkpoint"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    if args.stage == 'prepare':
        result = prepare(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    result = train(args) if args.stage == "train" else render(args)
    report = args.output_dir / f"COARSE_{args.stage.upper()}.json"
    write_json(report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
