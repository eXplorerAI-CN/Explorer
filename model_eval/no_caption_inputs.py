# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""读取发布数据：COLMAP TXT、参考照片、外部 RGB 和 opacity。

这个文件只整理输入，不运行模型。TXT 的记录顺序就是轨迹顺序；
图片按文件名字典序排列，第 n 张 RGB/mask 对应第 n 条轨迹。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_model(root):
    """兼容 sparse/0、sparse 和直接传入模型目录；明确使用 TXT。"""
    root = Path(root)
    for directory in (root / "sparse/0", root / "sparse", root / "0", root):
        if all((directory / name).is_file() for name in ("cameras.txt", "images.txt")):
            return directory
    raise ValueError(f"找不到 cameras.txt + images.txt：{root}（只有 BIN 时请先导出 TXT）")


def data_lines(path):
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            yield line


def read_model(root):
    directory = find_model(root)
    cameras = {}
    for line in data_lines(directory / "cameras.txt"):
        fields = line.split()
        camera_id, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        values = list(map(float, fields[4:]))
        require(model in {"PINHOLE", "SIMPLE_PINHOLE", "OPENCV"}, f"请提供去畸变针孔相机：{model}")
        require(len(values) == {"PINHOLE": 4, "SIMPLE_PINHOLE": 3, "OPENCV": 8}[model],
                f"相机参数数量错误：{line}")
        if model == "OPENCV":
            require(all(v == 0 for v in values[4:]), "请先去畸变：OPENCV 的 k1/k2/p1/p2 必须为 0")
        fx, fy, cx, cy = (values[0], values[0], *values[1:]) if model == "SIMPLE_PINHOLE" else values[:4]
        require(width >= 16 and height >= 16 and fx > 0 and fy > 0
                and all(math.isfinite(x) for x in values), f"无效相机：{line}")
        require(camera_id not in cameras, f"重复 CAMERA_ID：{camera_id}")
        cameras[camera_id] = dict(camera_model="OPENCV", w=width, h=height,
                                  fl_x=fx, fl_y=fy, cx=cx, cy=cy, k1=0., k2=0., p1=0., p2=0.)

    # images.txt 每张图占两行。第二行可以为空，不能先过滤空行。
    lines = iter((directory / "images.txt").read_text(encoding="utf-8-sig").splitlines())
    frames = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        require(len(fields) == 10, f"无效 images.txt 记录：{line}")
        image_id, camera_id, name = int(fields[0]), int(fields[8]), fields[9]
        require(Path(name).name == name and "\\" not in name
                and not any(c in name for c in "\t\r\n")
                and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}, f"不支持的图像名：{name}")
        pose = list(map(float, fields[1:8]))
        require(all(math.isfinite(x) for x in pose)
                and math.isclose(math.hypot(*pose[:4]), 1., abs_tol=1e-5, rel_tol=0), f"无效 pose：{name}")
        require(camera_id in cameras, f"{name} 的 CAMERA_ID 不存在")
        observations = next(lines, None)
        require(observations is not None, f"{name} 缺少 POINTS2D 行（无关联时也应保留空行）")
        fields2 = observations.split()
        require(len(fields2) % 3 == 0, f"{name} 的 POINTS2D 格式错误")
        points2d = [(float(fields2[i]), float(fields2[i + 1]), int(fields2[i + 2]))
                    for i in range(0, len(fields2), 3)]
        frames.append(dict(id=image_id, name=name, camera_id=camera_id, q=pose[:4], t=pose[4:], points2d=points2d))
    require(frames, f"空 COLMAP 模型：{directory}")
    require(len({f['id'] for f in frames}) == len(frames), f"重复 IMAGE_ID：{directory}")
    require(len({f['name'].casefold() for f in frames}) == len(frames), f"重复图像名：{directory}")
    return directory, cameras, frames


def rotation(q):
    """COLMAP 的四元数顺序为 w,x,y,z；返回 world-to-camera 旋转。"""
    w, x, y, z = q
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]


def transform_frame(frame, camera):
    import numpy as np
    from data_processing.camera_trajectories import opencv_w2c_to_opengl_c2w

    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = rotation(frame["q"]), frame["t"]
    return {**camera, "transform_matrix": opencv_w2c_to_opengl_c2w(w2c).tolist()}


def png_sequence(directory):
    directory = Path(directory)
    require(directory.is_dir(), f"图片目录不存在：{directory}")
    paths = sorted((p for p in directory.iterdir() if not p.name.startswith(".")
                    and not p.is_dir() and p.suffix.lower() == ".png"), key=lambda p: p.name)
    require(len(paths) == 81, f"{directory} 应有 81 张 PNG，实际 {len(paths)} 张")
    return paths


def read_image(path, size, kind):
    """RGB 返回 RGB；mask 返回 L。保留 0..255，不反色、不二值化。"""
    from PIL import Image, ImageChops

    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"需要普通图像文件：{path}")
    with Image.open(path) as image:
        require(image.size == tuple(size), f"尺寸不一致：{path} 为 {image.size}，应为 {tuple(size)}")
        require(getattr(image, "n_frames", 1) == 1, f"不支持动画：{path}")
        require(image.getexif().get(274, 1) == 1, f"请先将 EXIF 方向应用到像素：{path}")
        require("transparency" not in image.info, f"请先合成透明通道：{path}")
        require(image.mode in ({"1", "L", "RGB"} if kind == "mask" else {"L", "RGB"}),
                f"不支持 {kind} 模式 {image.mode}：{path}")
        if kind != "reference":
            require(image.format == "PNG", f"粗图和 mask 必须是真正的 PNG：{path}")
        if image.format == "PNG":
            with path.open("rb") as stream:
                header = stream.read(25)
            require(header[24] <= 8, f"不支持高位深 PNG：{path}")
        image.load()
        if kind == "mask" and image.mode == "RGB":
            r, g, b = image.split()
            require(not ImageChops.difference(r, g).getbbox() and not ImageChops.difference(r, b).getbbox(),
                    f"mask 三通道必须相同：{path}")
        return image.convert("L" if kind == "mask" else "RGB")


def source_signature(source_dir, references):
    """尺度与参考图/相机绑定。整体搬迁数据目录后签名仍相同。"""
    paths = [source_dir / "cameras.txt", source_dir / "images.txt", source_dir / "points3D.txt", *references]
    return {f"{i}:{path.name}": sha256(path) for i, path in enumerate(paths)}


def add_reference_args(parser):
    parser.add_argument('--reference-images', type=Path, required=True, help='12 张真实参考照片所在的 images 目录')
    parser.add_argument('--reference-colmap', type=Path, required=True, help='参考相机的 COLMAP TXT 模型目录')


def inspect_references(reference_images, reference_colmap):
    """只读取明确指定的参考照片和模型；尺度阶段不需要轨迹或 coarse 图。"""
    reference_images = Path(reference_images).expanduser().resolve()
    source_dir, source_cameras, references = read_model(Path(reference_colmap).expanduser().resolve())
    require(len(references) == 12, f"应有 12 张注册参考图，实际 {len(references)} 张")
    canvas = {(source_cameras[f['camera_id']]['w'], source_cameras[f['camera_id']]['h']) for f in references}
    require(len(canvas) == 1, f"参考图的画布尺寸不同：{canvas}")
    size = canvas.pop()
    reference_paths = [reference_images / f['name'] for f in references]
    for path in reference_paths:
        read_image(path, size, 'reference').close()
    return dict(source_dir=source_dir, source_cameras=source_cameras, references=references, size=size,
                reference_images_dir=reference_images, reference_paths=reference_paths,
                input_dirs=[reference_images, source_dir])


def inspect_inputs(reference_images, reference_colmap, trajectory, rgb_dir, mask_dir):
    """五项输入路径各自指定，不推测场景根目录或兄弟目录的名字。"""
    scene = inspect_references(reference_images, reference_colmap)
    target_dir, target_cameras, targets = read_model(Path(trajectory).expanduser().resolve())
    require(len(targets) == 81, f"应有 81 条轨迹，实际 {len(targets)} 条")
    require([f["name"] for f in targets] == sorted(f["name"] for f in targets), "轨迹记录名称必须已按字典序排列，不自动重排 pose")
    require(len({f['camera_id'] for f in targets}) == 1, "81 帧目标轨迹必须共用一个相机")
    size = scene['size']
    require(all((target_cameras[f['camera_id']]['w'], target_cameras[f['camera_id']]['h']) == size for f in targets),
            "参考与轨迹的画布尺寸不同")
    rgb_dir, mask_dir = (Path(p).expanduser().resolve() for p in (rgb_dir, mask_dir))
    require(rgb_dir != mask_dir, 'RGB 和 mask 必须分别指定不同目录')
    rgb_paths, mask_paths = png_sequence(rgb_dir), png_sequence(mask_dir)
    for kind, paths in [("rgb", rgb_paths), ("mask", mask_paths)]:
        for path in paths:
            read_image(path, size, kind).close()
    scene.update(target_dir=target_dir, target_cameras=target_cameras, targets=targets,
                 rgb_dir=rgb_dir, mask_dir=mask_dir, rgb_paths=rgb_paths, mask_paths=mask_paths)
    scene['input_dirs'] += [target_dir, rgb_dir, mask_dir]
    return scene


def frame_map(scene):
    return "index\ttrajectory_name\trgb_input\tmask_input\n" + "".join(
        f"{i}\t{frame['name']}\t{rgb}\t{mask}\n"
        for i, (frame, rgb, mask) in enumerate(zip(scene['targets'], scene['rgb_paths'], scene['mask_paths'])))


def resolve_scale(scene, camera_scale=None, scale_json=None):
    if scale_json is not None:
        require(Path(scale_json).is_file(), f"找不到尺度报告：{scale_json}；请先运行 estimate_no_caption_scale.py，或明确指定 --camera-scale")
        receipt = json.loads(Path(scale_json).read_text(encoding="utf-8"))
        require(receipt.get("status") == "READY", "尺度报告未通过质量检查")
        require(receipt.get("source_signature") == source_signature(scene['source_dir'], scene['reference_paths']),
                "尺度报告与当前参考图/COLMAP 不匹配，请重新估计")
        camera_scale = float(receipt["camera_scale"])
        require(math.isclose(camera_scale, float(receipt['metric_scale']) * .01, rel_tol=1e-12), "尺度报告的倍率不正确")
    require(camera_scale is not None and math.isfinite(camera_scale) and camera_scale > 0,
            "请提供 --scale-json，或明确提供正数 --camera-scale；没有默认尺度")
    return camera_scale


def new_output(path, scene, *, allow_scale=False):
    """接受预先 mkdir 的空目录；推理根目录还允许保留 scale/，不会覆盖旧结果。"""
    path = Path(path).expanduser().resolve()
    for source in scene['input_dirs']:
        require(not path.is_relative_to(source) and not source.is_relative_to(path), "输出目录不能与输入目录重叠")
    if path.exists():
        require(path.is_dir(), f"输出路径不是目录：{path}")
        leftovers = [p for p in path.iterdir() if not (allow_scale and p.name == 'scale' and p.is_dir() and not p.is_symlink())]
        require(not leftovers, f"输出目录已有文件，拒绝覆盖，请指定新目录：{path}")
    else:
        path.mkdir(parents=True)
    return path


def resize_to_multiple_of_16(tensor):
    """与官方 data/utils.py 一致：就近调整到 16 的倍数。"""
    import torch.nn.functional as F
    h, w = tensor.shape[-2:]
    size = (round(h / 16) * 16, round(w / 16) * 16)
    return F.interpolate(tensor, size, mode='bilinear') if size != (h, w) else tensor


def compute_camera_rays(transforms, frame_indices, neighbor_indices, scale, image_shape):
    """官方相机条件计算的无畸变针孔版本。

    沿用 model_training/data/utils.py 的坐标系、归一化内参、4 帧分组和
    Plücker 射线公式。直接计算针孔射线，避免导入整个 3DGRUT 数据集包。
    本入口在 read_model 中已拒绝带畸变的相机。
    """
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from model_training.utils.pose_utils import invert_SE3

    frames = transforms['frames']
    height, width = image_shape

    def poses(indices):
        matrices = torch.tensor([frames[i]['transform_matrix'] for i in indices], dtype=torch.float32)
        matrices[:, :3, 3] *= scale
        return matrices

    def intrinsics(indices):
        return torch.tensor([[frames[i]['fl_x'] / frames[i]['w'], frames[i]['fl_y'] / frames[i]['h'],
                              frames[i]['cx'] / frames[i]['w'], frames[i]['cy'] / frames[i]['h']]
                             for i in indices], dtype=torch.float32)

    def Ks(values):
        result = torch.zeros(len(values), 3, 3)
        result[:, 0, 0], result[:, 1, 1] = values[:, 0], values[:, 1]
        result[:, 0, 2], result[:, 1, 2] = values[:, 2] - .5, values[:, 3] - .5
        result[:, 2, 2] = 1.
        return result

    targets, neighbors = poses(frame_indices), poses(neighbor_indices)
    ref_w2c = invert_SE3(targets[0])
    targets, neighbors = ref_w2c @ targets, ref_w2c @ neighbors
    target_intrinsics = intrinsics(frame_indices)
    averaged_poses, averaged_intrinsics = [targets[0]], [target_intrinsics[0]]
    # Wan VAE：首帧单独编码，随后每 4 帧压缩为一个 latent 帧。81 → 21。
    for start in range(1, len(targets), 4):
        group = targets[start:start + 4]
        require(len(group) == 4, '目标帧数必须满足 1 + 4*n')
        matrix = torch.eye(4)
        matrix[:3, :3] = torch.tensor(Rotation.from_matrix(group[:, :3, :3].numpy()).mean().as_matrix(), dtype=torch.float32)
        matrix[:3, 3] = group[:, :3, 3].mean(dim=0)
        averaged_poses.append(matrix)
        averaged_intrinsics.append(target_intrinsics[start:start + 4].mean(dim=0))
    target_poses = torch.stack(averaged_poses)
    values = torch.stack(averaged_intrinsics)

    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    pixels = torch.stack([xx.flatten(), yy.flatten()], dim=1).float() + .5
    directions = []
    for fx, fy, cx, cy in values.tolist():
        # float32 参数与官方相机类一致；像素位置取像素中心 +0.5。
        focal = torch.from_numpy(np.array([fx * width, fy * height], dtype=np.float32))
        centre = torch.from_numpy(np.array([cx * width, cy * height], dtype=np.float32))
        xy = (pixels - centre) / focal
        xyz = torch.cat([xy, torch.ones_like(xy[:, :1])], dim=1)
        directions.append(xyz / torch.linalg.norm(xyz, dim=1, keepdim=True))
    directions = torch.bmm(torch.stack(directions), target_poses[:, :3, :3].transpose(1, 2))
    origins = target_poses[:, :3, 3].unsqueeze(1).expand_as(directions)
    rays = torch.cat([torch.linalg.cross(origins, directions), directions], dim=-1)
    return dict(w2cs=invert_SE3(target_poses), Ks=Ks(values), neighbor_w2cs=invert_SE3(neighbors),
                neighbor_Ks=Ks(intrinsics(neighbor_indices)), camera_rays=rays.reshape(-1, height, width, 6))
