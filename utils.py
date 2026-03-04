import cv2
import numpy as np

COCO_CLASSES = {
    # ------------------------------------------------------------------------
    # RF-DETR
    # Copyright (c) 2025 Roboflow. All Rights Reserved.
    # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    # ------------------------------------------------------------------------
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

def preprocess_img(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """전처리입니당

    Args:
        img (np.ndarray): opencv로 불러온 이미지
        w (int): 너비
        h (int): 높이

    Returns:
        np.ndarray: 전처리된 이미지
    """
    img = cv2.resize(img, (w,h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1)).copy()
    img = np.expand_dims(img, 0)
    return img

def postprocess(
    boxes: np.ndarray,
    logits: np.ndarray,
    orig_w: int,
    orig_h: int,
    conf:float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, w, h= boxes[:, 0], boxes[:, 1], boxes [:, 2], boxes[:, 3]

    x1 = (cx - w/2) * orig_w
    y1 = (cy - h/2) * orig_h
    x2 = (cx + w/2) * orig_w
    y2 = (cy + h/2) * orig_h

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # clamp
    boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, orig_w)
    boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, orig_h)
    
    # softmax(몰루)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    fg_probs = probs[:, 1:] # 배경없앰
    scores = fg_probs.max(axis=1)
    classes = fg_probs.argmax(axis=1) + 1 

    keep = scores > conf

    return boxes_xyxy[keep], scores[keep], classes[keep]
