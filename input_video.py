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
    # Draw directly on a copy of the frame using OpenCV primitives
    annotated = frame.copy()
    for dt in pose_results:
        dt_joints = np.array(dt['keypoints']).reshape(17, -1)
        # Draw skeleton
        for k, link_pair in enumerate(style.link_pairs):
            i, j, color = link_pair
            x1, y1, s1 = dt_joints[i]
            x2, y2, s2 = dt_joints[j]
            # Only draw if both endpoints are inside the frame
            if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                cv2.line(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    tuple(int(255*c) for c in color),
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
        # Draw keypoints
        for k in range(dt_joints.shape[0]):
            x, y, s = dt_joints[k]
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                color = tuple(int(255*c) for c in style.ring_color[k])
                cv2.circle(
                    annotated,
                    (int(x), int(y)),
                    radius=thickness*2,
                    color=color,
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )
                cv2.circle(
                    annotated,
                    (int(x), int(y)),
                    radius=thickness*2,
                    color=(0,0,0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
    return annotated

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
                    "bbox": person['bbox'].tolist() if isinstance(person['bbox'], np.ndarray) else person['bbox'],
                    "keypoints": person['keypoints'].tolist() if isinstance(person['keypoints'], np.ndarray) else person['keypoints']
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
    # Ensure output directory exists before saving JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
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