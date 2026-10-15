# Third-party components

| Component | Use | Licence |
|---|---|---|
| [basketball-player-detection-3](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo) dataset (Roboflow Universe) | training data of the default detector | CC BY 4.0 |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | detector architecture / training | AGPL-3.0 |
| [BoxMOT](https://github.com/mikel-brostrom/boxmot) (BoT-SORT, OSNet) | tracking + re-identification | AGPL-3.0 |
| [trackers](https://github.com/roboflow/trackers) (McByte) | tracking | Apache-2.0 |
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | single-player mask tracking | Apache-2.0 |
| [McByte++](https://github.com/tstanczyk95/McBytePlusPlus) | EdgeTAM MPS fix (contiguous reshape) | Apache-2.0 |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (MLX 4-bit) | number rechecks, touch checks | Apache-2.0 |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (MLX 4-bit) | jersey number reading | Qwen Research License (non-commercial) |
| [DINOv2](https://github.com/facebookresearch/dinov2) | appearance features | Apache-2.0 |

The default detector (`detector-v1` release) was trained by this project with YOLO11s on the dataset above
(attribution: Roboflow Universe, "basketball-player-detection-3", CC BY 4.0).
