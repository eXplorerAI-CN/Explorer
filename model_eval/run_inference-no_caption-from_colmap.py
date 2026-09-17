#!/usr/bin/env python3
"""Explorer：读取参考相机、coarse RGB/mask 和尺度报告，执行无 caption Stage1。

先运行 prepare_no_caption_coarse.py 准备、训练和渲染，再独立运行
estimate_no_caption_scale.py 得到 SCALE.json。最后运行本文件进行 Explorer 图像修复。
每项输入路径明确指定；模型默认为 1.3B、50 步、全零文本特征。
使用说明见仓库根目录 README.md。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_eval.no_caption_inputs import (
    add_reference_args, frame_map, inspect_inputs, new_output, read_image, require, resolve_scale,
    sha256, transform_frame, write_json,
)


def load_tensors(scene, camera_scale):
    import numpy as np
    import torch
    from model_eval.no_caption_inputs import compute_camera_rays, resize_to_multiple_of_16

    def images(paths, kind):
        arrays = []
        for path in paths:
            with read_image(path, scene['size'], kind) as image:
                arrays.append(np.array(image))
        tensor = torch.from_numpy(np.stack(arrays)).float() / 255.
        tensor = tensor.unsqueeze(1) if kind == 'mask' else tensor.permute(0, 3, 1, 2)
        return resize_to_multiple_of_16(tensor)

    rendered = images(scene['rgb_paths'], 'rgb')
    opacity = images(scene['mask_paths'], 'mask').squeeze(1)
    neighbors = images(scene['reference_paths'], 'reference')
    # 两组相机分别读取各自的内外参；同名参考和目标也保持独立。
    frames = [transform_frame(f, scene['target_cameras'][f['camera_id']]) for f in scene['targets']]
    frames += [transform_frame(f, scene['source_cameras'][f['camera_id']]) for f in scene['references']]
    transforms = {**scene['target_cameras'][scene['targets'][0]['camera_id']], 'frames': frames}
    cameras = compute_camera_rays(transforms, list(range(81)), list(range(81, 93)),
                                  camera_scale, tuple(rendered.shape[-2:]))
    return rendered, opacity, neighbors, cameras


def run_inference(scene, args, camera_scale, output):
    import torch
    import torch.nn.functional as F
    from accelerate.utils import set_seed
    from PIL import Image
    from model_eval.checkpoint_loading import load_model_weights_from_pt
    from model_training.constants import MAX_SEQUENCE_LENGTH
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
    from model_training.pipeline.pipeline import ArtifixerPipeline

    require(torch.cuda.is_available(), '推理需要 CUDA GPU；Mac 可编辑代码，但不能运行这条 CUDA 推理流程')
    device = torch.device(args.device)
    require(device.type == 'cuda', '--device 必须是 CUDA 设备，例如 cuda:0')
    torch.cuda.set_device(device)
    set_seed(args.seed)  # 在构建模型前设种子，保持和已验证的 3090 入口相同的顺序。
    # 与官方 get_pipe 的模型构建方式相同，直接导入推理类，避免引入训练数据集/3DGRUT。
    model_id = str(args.model_id)
    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder='scheduler', torch_dtype=torch.bfloat16)
    transformer = WanTransformer3DModel.from_config(model_id, subfolder='transformer', torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder='vae', torch_dtype=torch.bfloat16).to(device)
    pipe = ArtifixerPipeline(
        vae=vae, scheduler=scheduler, transformer=transformer,
        tokenizer=None, text_encoder=None, default_negative_prompt_path=args.negative_prompt,
        frames_per_block=None, gradient_checkpointing=False, checkpoint_every_n_blocks=1, attention_backend='native',
    )
    pipe = pipe.to(torch.bfloat16).to(device)
    load_model_weights_from_pt(pipe.transformer, args.checkpoint)
    pipe.transformer.eval()
    rendered, opacity, neighbors, cameras = load_tensors(scene, camera_scale)

    # 真正的无 caption：直接给全零特征；不要把空字符串送入 T5。
    prompt = torch.zeros(1, MAX_SEQUENCE_LENGTH, 4096, dtype=torch.bfloat16, device=device)
    camera_kwargs = {key: value.unsqueeze(0).to(device=device,
                     dtype=torch.bfloat16 if key == 'camera_rays' else value.dtype)
                     for key, value in cameras.items()}
    with torch.inference_mode():
        prediction = pipe.forward_inference(
            rendered_rgb=rendered.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
            rendered_opacity=opacity.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
            neighbors=neighbors.unsqueeze(0),  # 留在 CPU，VAE 每次只取 1 张，12 张全部参与。
            prompt=prompt, text_guidance_scale=1.0,
            num_inference_steps=args.steps, max_neighbors_per_encode=1,
            show_progress=True, **camera_kwargs,
        )[0].cpu()
    require(len(prediction) == 81, f'模型返回 {len(prediction)} 帧，应为 81 帧')
    require(bool(torch.isfinite(prediction).all()), '模型输出出现 NaN/Inf，未发布成功结果')

    # 保留一份无损 PNG；final/ 沿用轨迹原名称和扩展名，便于下游使用。
    (output / 'pred').mkdir()
    (output / 'final').mkdir()
    width, height = scene['size']
    checksums = []
    for i, frame in enumerate(prediction):
        frame = frame.float().clamp(0, 1)
        if frame.shape[-2:] != (height, width):
            frame = F.interpolate(frame.unsqueeze(0), (height, width), mode='bilinear', align_corners=False)[0]
        pixels = (frame.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()
        image = Image.fromarray(pixels)
        image.save(output / 'pred' / f'{i:05d}.png')
        final = output / 'final' / scene['targets'][i]['name']
        image.save(final, **({'quality': 95, 'subsampling': 0} if final.suffix.lower() in {'.jpg', '.jpeg'} else {}))
        checksums.append(f'{sha256(final)}  {final.name}\n')
    (output / 'final' / 'SHA256SUMS').write_text(''.join(checksums))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_args(parser)
    parser.add_argument('--trajectory-colmap', type=Path, required=True, help='81 个目标相机的 COLMAP TXT 模型目录')
    parser.add_argument('--rgb-dir', type=Path, required=True, help='81 张 coarse RGB PNG 的目录')
    parser.add_argument('--mask-dir', type=Path, required=True, help='81 张 coarse opacity/mask PNG 的目录')
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument('--scale-json', type=Path, help='手动指定 MoGe3 的 SCALE.json，不自动寻找')
    scale.add_argument('--camera-scale', type=float, help='已知的相机倍率；等于米/COLMAP单位 × 0.01，无默认值')
    parser.add_argument('--checkpoint', type=Path, help='Explorer 权重，须兼容 ArtiFixer / Wan2.1 1.3B transformer 结构')
    parser.add_argument('--model-id', type=Path, help='本地 Wan2.1-T2V-1.3B-Diffusers 目录')
    parser.add_argument('--output-dir', type=Path, required=True, help='单独的空推理结果目录，例如 /path/006-result/Explorer；其中保存 pred/ 和 final/')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=50, help='正式推理保持 50；1 仅用于运行验证')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--negative-prompt', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'default_negative_prompt.pt')
    parser.add_argument('--check-only', action='store_true', help='只读检查 12+81+81 输入图片，不加载模型')
    parser.add_argument('--print-frame-map', action='store_true', help='打印全部 81 条 RGB/mask/轨迹对应关系')
    args = parser.parse_args(argv)
    for name in ('reference_images', 'reference_colmap', 'trajectory_colmap', 'rgb_dir', 'mask_dir',
                 'scale_json', 'checkpoint', 'model_id', 'output_dir', 'negative_prompt'):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    scene = inspect_inputs(args.reference_images, args.reference_colmap, args.trajectory_colmap, args.rgb_dir, args.mask_dir)
    print(f"INPUT_OK：12 张参考 + 81 张 RGB + 81 张 mask，宽×高={scene['size']}", flush=True)
    print(f"参考图片：{scene['reference_images_dir']}\n参考 COLMAP：{scene['source_dir']}\n轨迹：{scene['target_dir']}\n"
          f"RGB：{scene['rgb_dir']}\nmask：{scene['mask_dir']}\n输出：{args.output_dir}", flush=True)
    if args.print_frame_map:
        print(frame_map(scene))
    camera_scale = None
    if args.scale_json is not None or args.camera_scale is not None:
        camera_scale = resolve_scale(scene, args.camera_scale, args.scale_json)
        print(f'camera_scale={camera_scale:.17g}', flush=True)
    if args.check_only:
        return
    require(camera_scale is not None, '请提供 --scale-json 或 --camera-scale')
    require(args.checkpoint and args.checkpoint.is_file(), '请提供 --checkpoint 文件')
    require(args.model_id and (args.model_id / 'transformer/config.json').is_file(), '请提供本地 --model-id 目录')
    import json
    config = json.loads((args.model_id / 'transformer/config.json').read_text())
    require(config.get('num_layers') == 30 and config.get('num_attention_heads') == 12
            and config.get('attention_head_dim') == 128, '--model-id 必须是 Wan2.1-T2V-1.3B-Diffusers')
    require(args.negative_prompt.is_file(), f'缺少 {args.negative_prompt}')
    require(args.steps > 0, '--steps 必须为正整数')
    # 输入、尺度报告和粗渲染都已经由前面的独立步骤生成；本入口只读取它们。
    input_files = scene['reference_paths'] + scene['rgb_paths'] + scene['mask_paths']
    input_files += [directory / name for directory in (scene['source_dir'], scene['target_dir'])
                    for name in ('cameras.txt', 'images.txt')]
    if args.scale_json is not None:
        input_files.append(args.scale_json)
    input_hashes = {str(path): sha256(path) for path in input_files}
    output = new_output(args.output_dir, scene)
    receipt = dict(status='RUNNING', inputs_sha256=input_hashes,
                   inputs={key: str(scene[key]) for key in ['reference_images_dir', 'source_dir', 'target_dir', 'rgb_dir', 'mask_dir']},
                   output_dir=str(output), scale_json=str(args.scale_json.resolve()) if args.scale_json else None,
                   camera_scale=camera_scale,
                   checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                   model_id=str(args.model_id.resolve()), num_references=12, num_frames=81,
                   image_size=list(scene['size']), pipeline='bidirectional', steps=args.steps,
                   seed=args.seed, prompt_mode='zero', text_guidance_scale=1.0, attention_backend='native')
    write_json(output / 'RUN.json', receipt, exclusive=True)
    try:
        (output / 'INPUT_FRAME_MAP.tsv').write_text(frame_map(scene), encoding='utf-8')
        run_inference(scene, args, camera_scale, output)
        require(all(sha256(Path(path)) == digest for path, digest in input_hashes.items()),
                '推理期间输入数据发生变化，未发布成功结果')
        receipt.update(status='SUCCESS')
        write_json(output / 'SUCCESS.json', receipt)
        print(f"完成：{output / 'final'}（无损预测另存于 pred/）")
    except BaseException as error:
        receipt.update(status='FAILED', error=str(error))
        raise
    finally:
        write_json(output / 'RUN.json', receipt)


if __name__ == '__main__':
    main()
