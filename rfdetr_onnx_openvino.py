import time
import cv2
import numpy as np
import openvino as ov
import utils

# ----- 모델 불러옴 -----
core = ov.Core()
compiled_model = core.compile_model("./output/rfdetr_s.onnx", device_name="GPU")
infer_request = compiled_model.create_infer_request()

# ----- 이미지 불러옴 -----
img_orig = cv2.imread("./videos/test2_2.png")  # 원본 따로 보관
orig_h, orig_w = img_orig.shape[:2]
img = utils.preprocess_img(img_orig.copy(), 512, 512)
input_tensor = ov.Tensor(array=img, shared_memory=True)

# ----- 추론 -----
start = time.perf_counter()
for i in range(0,30):
    infer_request.set_input_tensor(input_tensor)
    infer_request.start_async()
    infer_request.wait()
print(time.perf_counter() - start)
# ----- 후처리 -----
boxes_cxcywh_norm = infer_request.get_output_tensor(0).data[0]
logits = infer_request.get_output_tensor(1).data[0]
boxes_xywh, scores, classes = utils.postprocess(boxes_cxcywh_norm, logits, orig_w, orig_h)
print("max score:", scores.max())
print(boxes_xywh[scores > 0.55])
print(scores[scores > 0.55])
print(classes[scores > 0.55])

# ----- 시각화 -----
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
]
SCORE_THRESHOLD = 0.55

mask = scores > SCORE_THRESHOLD
filtered_boxes   = boxes_xywh[mask]
filtered_scores  = scores[mask]
filtered_classes = classes[mask]

result_img = img_orig.copy() 

for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS[int(cls) % len(COLORS)]
    label = f"{utils.COCO_CLASSES.get(int(cls), str(cls))}: {score:.2f}"

    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(result_img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(result_img, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite("./videos/test2_results.png", result_img)