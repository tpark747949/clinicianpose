Using input_video.py:

(PCT) tpark747949@snuh-medisc-2:~/clinicianpose$ PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python vis_tools/demo_img_with_mmdet.py vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth configs/pct_base_classifier.py weights/pct/swin_base.pth --img-root images/ --out-img-root output/ --thickness 2 --img image8.jpg


python input_video.py \
  --video input/video/hanging.mp4 \
  --det-config vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py \
  --det-ckpt https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth \
  --pose-config configs/pct_base_classifier.py \
  --pose-ckpt weights/pct/swin_base.pth \
  --out-json output/video/hanging_annotated.json \
  --out-video output/video/hanging_annotated.mp4

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