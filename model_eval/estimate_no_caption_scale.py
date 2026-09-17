"""Explorer 的 MoGe3 相机尺度估计；独立执行后释放显存。

使用说明见仓库根目录 README.md。

比较同一稀疏点的 COLMAP 相机 Z 深度与 MoGe3 米制 Z 深度。
每张参考图独立估计一个比例，再取稳健中值，避免点多的图主导结果。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_eval.no_caption_inputs import (
    add_reference_args, data_lines, inspect_references, new_output, read_image, require, rotation,
    sha256, source_signature, write_json,
)


def sparse_pairs(scene):
    """只使用同时存在于 POINTS2D 和 points3D track 的有效关联。"""
    points = {}
    for line in data_lines(scene['source_dir'] / 'points3D.txt'):
        fields = line.split()
        require(len(fields) >= 8 and (len(fields) - 8) % 2 == 0, 'points3D.txt 格式错误')
        pid = int(fields[0])
        require(pid not in points, f'重复 POINT3D_ID：{pid}')
        xyz = list(map(float, fields[1:4]))
        require(all(math.isfinite(v) for v in xyz), f'无效稀疏点：{pid}')
        tracks = {(int(fields[i]), int(fields[i + 1])) for i in range(8, len(fields), 2)}
        points[pid] = xyz, tracks
    rows = []
    for frame in scene['references']:
        camera = scene['source_cameras'][frame['camera_id']]
        z_axis = rotation(frame['q'])[2]
        pairs, seen = [], set()
        for index, (x, y, pid) in enumerate(frame['points2d']):
            if pid not in points or pid in seen or not (0 <= x < camera['w'] and 0 <= y < camera['h']):
                continue
            xyz, tracks = points[pid]
            if (frame['id'], index) not in tracks:
                continue
            z = sum(a * b for a, b in zip(z_axis, xyz)) + frame['t'][2]
            if z > 0 and math.isfinite(z):
                pairs.append((pid, x, y, z))
                seen.add(pid)
        rows.append(pairs)
    eligible = [pairs for pairs in rows if len(pairs) >= 30]
    require(len(eligible) >= 8 and sum(map(len, eligible)) >= 400
            and len({p[0] for pairs in eligible for p in pairs}) >= 100,
            '可用稀疏点关联不足：至少 8 张参考图各 30 对，全局 400 对、100 个不同点')
    return rows


def fit_scale(rows):
    """rows[i] 是第 i 张图的 (point_id, log(metric_z / colmap_z))。"""
    centres = [statistics.median(v for _, v in row) if row else None for row in rows]
    active = [i for i, row in enumerate(rows) if len(row) >= 30]
    candidate = None
    while active:
        candidate = statistics.median(centres[i] for i in active)
        retained = [i for i in active
                    if math.log(.7) - 1e-12 <= centres[i] - candidate <= math.log(1.3) + 1e-12
                    and statistics.median(abs(math.expm1(max(-745., min(709., candidate - v))))
                                          for _, v in rows[i]) <= .3 + 1e-12]
        if retained == active:
            break
        active = retained
    total = sum(len(rows[i]) for i in active)
    unique = len({pid for i in active for pid, _ in rows[i]})
    ready = len(active) >= 8 and total >= 400 and unique >= 100
    metric = math.exp(candidate) if ready else None
    return dict(status='READY' if ready else 'FAILED', metric_scale=metric,
                camera_scale=metric * .01 if ready else None,
                retained_images=active, retained_pairs=total, unique_points=unique,
                per_image_valid_pairs=list(map(len, rows)),
                per_image_metric_scale=[math.exp(v) if v is not None else None for v in centres],
                method='median of per-image median log-depth ratios',
                thresholds=dict(minimum_images=8, minimum_pairs_per_image=30, minimum_total_pairs=400,
                                minimum_unique_points=100, maximum_relative_scale_deviation=.3,
                                maximum_median_relative_depth_residual=.3))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_args(parser)
    parser.add_argument('--moge-model', type=Path, help='本地 MoGe3 model.pt')
    parser.add_argument('--output-dir', type=Path, required=True, help='手动指定尺度输出目录，例如 /path/006-result/scale')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--check-only', action='store_true', help='只检查数据和稀疏点，不加载 MoGe3')
    args = parser.parse_args(argv)
    scene = inspect_references(args.reference_images, args.reference_colmap)
    pairs = sparse_pairs(scene)
    print('12 张参考图的可用稀疏点数量：', list(map(len, pairs)), flush=True)
    if args.check_only:
        print('CHECK_OK：数据及稀疏点关联通过；实际尺度仍需 MoGe3 预测。')
        return
    require(args.moge_model and args.moge_model.is_file(), '请提供本地 --moge-model 文件')
    output = new_output(args.output_dir, scene)
    receipt = dict(status='RUNNING', source_signature=source_signature(scene['source_dir'], scene['reference_paths']),
                   reference_images=str(scene['reference_images_dir']), reference_colmap=str(scene['source_dir']),
                   moge_model=str(args.moge_model), moge_sha256=sha256(args.moge_model),
                   source_names=[f['name'] for f in scene['references']])
    write_json(output / 'SCALE.json', receipt, exclusive=True)
    try:
        import numpy as np
        import torch
        from moge.model.v3 import MoGeModel

        require(torch.cuda.is_available(), 'MoGe3 尺度预测需要 CUDA GPU')
        # 显式构造 v3，避免误用官方准备流程中的 MoGe v2。
        checkpoint = torch.load(args.moge_model, map_location='cpu', weights_only=True, mmap=True)
        model = MoGeModel(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        model = model.to(args.device).eval()
        rows = []
        for i, (frame, path, observations) in enumerate(zip(scene['references'], scene['reference_paths'], pairs)):
            print(f"MoGe3 [{i + 1}/12] {frame['name']}", flush=True)
            with read_image(path, scene['size'], 'reference') as image:
                rgb = np.array(image, dtype=np.float32) / 255.
            camera = scene['source_cameras'][frame['camera_id']]
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(args.device)
            fov = math.degrees(2 * math.atan(camera['w'] / (2 * camera['fl_x'])))
            with torch.inference_mode():
                prediction = model.infer(tensor, fov_x=fov, resolution_level=9, refine_steps=3,
                                         use_fp16=True, apply_mask=True)
            # infer 返回的 depth 已是米制 Z 深度，无需再乘 prediction['metric_scale']。
            depth = prediction['depth'].float().cpu().numpy()
            mask = prediction['mask'].cpu().numpy()
            require(depth.ndim == 2 and mask.shape == depth.shape, 'MoGe3 深度/mask 形状异常')
            height, width = depth.shape
            row = []
            for pid, x, y, z in observations:
                u, v = math.floor(x * width / camera['w']), math.floor(y * height / camera['h'])
                metric_z = float(depth[v, u])
                if mask[v, u] > 0 and math.isfinite(metric_z) and metric_z > 0:
                    row.append((pid, math.log(metric_z) - math.log(z)))
            rows.append(row)
            del prediction, tensor
        receipt.update(fit_scale(rows))
        require(receipt['status'] == 'READY', 'MoGe3 尺度一致性检查未通过，详见 SCALE.json；没有回退尺度')
        require(source_signature(scene['source_dir'], scene['reference_paths']) == receipt['source_signature'],
                '尺度估计期间参考数据发生变化')
        write_json(output / 'SCALE.json', receipt)
        print(f"camera_scale={receipt['camera_scale']:.17g}\n尺度报告：{output / 'SCALE.json'}")
    except Exception as error:
        receipt.update(status='FAILED', metric_scale=None, camera_scale=None, error=str(error))
        write_json(output / 'SCALE.json', receipt)
        raise


if __name__ == '__main__':
    main()
