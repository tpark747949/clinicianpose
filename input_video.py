import os
import cv2
import json
import mmcv
import numpy as np
from tqdm import tqdm
from mmcv.runner import load_checkpoint
from mmpose.apis import inference_top_down_pose_model, process_mmdet_results
from mmpose.datasets import DatasetInfo
from models import build_posenet
from mmdet.apis import inference_detector, init_detector
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def init_pose_model(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def vis_pose_on_frame(frame, pose_results, style, thickness=2):
    fig = plt.figure(figsize=(frame.shape[1]/100, frame.shape[0]/100), dpi=100)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(frame[:,:,::-1])
    bk.set_zorder(-1)
    for dt in pose_results:
        dt_joints = np.array(dt['keypoints']).reshape(17,-1)
        # Fix: Unpack x, y, score for each keypoint
        joints_dict = {i: (int(x), int(y)) for i, (x, y, _) in enumerate(dt_joints)}
        # Draw skeleton
        for k, link_pair in enumerate(style.link_pairs):
            line = mlines.Line2D(
                [joints_dict[link_pair[0]][0], joints_dict[link_pair[1]][0]],
                [joints_dict[link_pair[0]][1], joints_dict[link_pair[1]][1]],
                ls='-', lw=thickness, alpha=1, color=link_pair[2])
            line.set_zorder(0)
            ax.add_line(line)
        # Draw keypoints
        for k in range(dt_joints.shape[0]):
            circle = mpatches.Circle(tuple(dt_joints[k,:2]), 
                                     radius=thickness*2, 
                                     ec='black', 
                                     fc=style.ring_color[k], 
                                     alpha=1, 
                                     linewidth=1)
            circle.set_zorder(1)
            ax.add_patch(circle)
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    fig.canvas.draw()
    annotated = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated = annotated.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = [pair + [tuple(np.array(color[i])/255.)] for i, pair in enumerate(link_pairs)]
        self.point_color = point_color
        self.ring_color = [tuple(np.array(c)/255.) for c in point_color]

# Chunhua style (from your script)
color2 = [(252,176,243),(252,176,243),(252,176,243),
    (0,176,240), (0,176,240), (0,176,240),
    (255,255,0), (255,255,0),(169, 209, 142),
    (169, 209, 142),(169, 209, 142),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127)]
link_pairs2 = [
    [15, 13], [13, 11], [11, 5], 
    [12, 14], [14, 16], [12, 6], 
    [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
    [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
]
point_color2 = [(240,2,127),(240,2,127),(240,2,127), 
    (240,2,127), (240,2,127), 
    (255,255,0),(169, 209, 142),
    (255,255,0),(169, 209, 142),
    (255,255,0),(169, 209, 142),
    (252,176,243),(0,176,240),(252,176,243),
    (0,176,240),(252,176,243),(0,176,240),
    (255,255,0),(169, 209, 142),
    (255,255,0),(169, 209, 142),
    (255,255,0),(169, 209, 142)]
chunhua_style = ColorStyle(color2, link_pairs2, point_color2)

def main(
    video_path,
    det_config,
    det_checkpoint,
    pose_config,
    pose_checkpoint,
    output_json,
    output_video=None,
    device='cuda:0',
    det_cat_id=1,
    bbox_thr=0.3,
    thickness=2
):
    # Initialize models
    det_model = init_detector(det_config, det_checkpoint, device=device)
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is not None:
        dataset_info = DatasetInfo(dataset_info)

    # Video IO
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    else:
        out_vid = None

    all_results = []
    for frame_idx in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame to temp file for detector (or adapt for array input)
        temp_img = f'temp_frame_{frame_idx}.jpg'
        cv2.imwrite(temp_img, frame)
        mmdet_results = inference_detector(det_model, temp_img)
        person_results = process_mmdet_results(mmdet_results, det_cat_id)
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            temp_img,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None
        )
        # Collect keypoints for JSON
        frame_result = {
            "frame_index": frame_idx,
            "people": [
                {
                    "bbox": person['bbox'],
                    "keypoints": person['keypoints']  # [x1, y1, score1, ...]
                }
                for person in pose_results
            ]
        }
        all_results.append(frame_result)
        # Annotate frame if needed
        if out_vid:
            annotated = vis_pose_on_frame(frame, pose_results, chunhua_style, thickness)
            out_vid.write(annotated)
        os.remove(temp_img)
    cap.release()
    if out_vid:
        out_vid.release()
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved keypoints to {output_json}")
    if output_video:
        print(f"Saved annotated video to {output_video}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video pose estimation with PCT")
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--det-config', required=True, help='Detection config')
    parser.add_argument('--det-ckpt', required=True, help='Detection checkpoint')
    parser.add_argument('--pose-config', required=True, help='Pose config')
    parser.add_argument('--pose-ckpt', required=True, help='Pose checkpoint')
    parser.add_argument('--out-json', required=True, help='Output JSON file')
    parser.add_argument('--out-video', default=None, help='Output annotated video (optional)')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    args = parser.parse_args()
    main(
        video_path=args.video,
        det_config=args.det_config,
        det_checkpoint=args.det_ckpt,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_ckpt,
        output_json=args.out_json,
        output_video=args.out_video,
        device=args.device
    )