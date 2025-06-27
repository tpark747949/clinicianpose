Using input_video.py:

python your_script.py \
  --video input.mp4 \
  --det-config path/to/det_config.py \
  --det-ckpt path/to/det_ckpt.pth \
  --pose-config path/to/pct_config.py \
  --pose-ckpt path/to/pct_ckpt.pth \
  --out-json output_keypoints.json \
  --out-video output_annotated.mp4

Output format:

[
  {
    "frame_index": 0,
    "people": [
      {
        "bbox": [x1, y1, x2, y2, score],
        "keypoints": [x1, y1, score1, x2, y2, score2, ..., x17, y17, score17]
      },
      ...
    ]
  },
  {
    "frame_index": 1,
    "people": [
      {
        "bbox": [x1, y1, x2, y2, score],
        "keypoints": [x1, y1, score1, x2, y2, score2, ..., x17, y17, score17]
      },
      ...
    ]
  },
  ...
]