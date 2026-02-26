import cv2
import time
import cv2
import numpy as np
import openvino as ov
import utils

from tracker.fasttracker import Fasttracker

# ----- 모델 불러옴 -----
core = ov.Core()
compiled_model = core.compile_model("./output/rfdetr_n.onnx", device_name="GPU")
infer_request = compiled_model.create_infer_request()

#tracking

config = {
    "track_thresh": 0.7,
    "track_buffer": 30,
    "match_thresh": 0.7,
    "min_box_area": 100,
    "reset_velocity_offset_occ": 5,
    "reset_pos_offset_occ": 3,
    "enlarge_bbox_occ": 1.2,
    "dampen_motion_occ": 0.85,
    "active_occ_to_lost_thresh": 15,
    "init_iou_suppress": 0.8,
    "roi_repair_max_gap": 15,
    "dir_window_N": 10,
    "dir_margin_deg": 2.0
}

class Args:
    mot20 = False

tracker = Fasttracker(config=config, args=Args(), frame_rate=30)

# ----- 영상 불러옴 -----
video = cv2.VideoCapture("./videos/test1.mp4")  # 원본 따로 보관
orig_h, orig_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

# ----- 시각화 -----
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
]
SCORE_THRESHOLD = 0.7

# ----- 추론 -----
while True:
    ret, img_orig = video.read()
    if not ret:
        break
    img = utils.preprocess_img(img_orig.copy(), 384, 384)
    input_tensor = ov.Tensor(array=img, shared_memory=True)

    start = time.perf_counter()
    infer_request.set_input_tensor(input_tensor)
    infer_request.start_async()
    infer_request.wait()
    print("detection:", time.perf_counter() - start)

    # ----- 후처리 -----
    boxes_cxcywh_norm = infer_request.get_output_tensor(0).data[0]
    logits = infer_request.get_output_tensor(1).data[0]
    boxes_xyxy, scores, classes = utils.postprocess(boxes_cxcywh_norm, logits, orig_w, orig_h)

    dets = np.concatenate([boxes_xyxy, scores[:, None], classes[:, None]], axis=1).astype(np.float32)

    # tracking
    start = time.perf_counter()
    tracks = tracker.update(dets, [orig_h, orig_w], (orig_h, orig_w))
    print("tracking:", time.perf_counter() - start)
    # print("듀아아ㅏ")
    result = [t.to_dict() for t in tracks]
    # for r in result:
    #     print(r)

    # print("max score:", scores.max())
    # print(boxes_xyxy[scores > SCORE_THRESHOLD])
    # print(scores[scores > SCORE_THRESHOLD])
    # print(classes[scores > SCORE_THRESHOLD])

    mask = scores > SCORE_THRESHOLD
    filtered_boxes   = boxes_xyxy[mask]
    filtered_scores  = scores[mask]
    filtered_classes = classes[mask]

    result_img = img_orig.copy()
    for track in tracks:
        d = track.to_dict()
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        tid = d["track_id"]
        cls = d["cls"]
        score = d["score"]
        color = COLORS[tid % len(COLORS)]
        cls_name = utils.COCO_CLASSES.get(cls, str(cls))
        label = f"{cls_name}: {score:.2f} (ID:{tid})"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(result_img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("result", result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break