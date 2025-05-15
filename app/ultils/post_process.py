import cv2
import numpy as np


def filter_boxes(boxes, box_confidences, box_class_probs,obj_thresh=0.25):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= obj_thresh)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores, nms_thresh=0.45):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def dfl(position):
    # Distribution Focal Loss (DFL)

    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.array(range(mc), dtype=np.float32).reshape(1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position,image_size=(640, 640)):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([image_size[1] // grid_h, image_size[0] // grid_w]).reshape(1, 2, 1, 1)

    if position.shape[1] == 4:
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    else:
        position = dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data,image_size=(640, 640), nms_thresh=0.45, obj_thresh=0.25):
    """
    Xử lý kết quả đầu ra của mô hình phát hiện đối tượng và áp dụng NMS
    """
    # Khởi tạo biến
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    # Xử lý dữ liệu từ đầu vào (tối ưu: tránh sử dụng vòng lặp khi có thể)
    for i in range(default_branch):
        idx = pair_per_branch * i
        boxes.append(box_process(input_data[idx],image_size))
        classes_conf.append(input_data[idx + 1])
        # Sử dụng phương thức numpy trực tiếp thay vì tạo mảng mới
        score_shape = input_data[idx + 1][:, :1, :, :].shape
        scores.append(np.ones(score_shape, dtype=np.float32))

    # Tối ưu hàm flatten
    def sp_flatten(tensor):
        ch = tensor.shape[1]
        # Sửa lỗi cú pháp
        tensor_t = tensor.transpose(0, 2, 3, 1)
        return tensor_t.reshape(-1, ch)

    # Tối ưu: kết hợp các bước chuyển đổi để giảm số lần lặp
    flattened_boxes = np.concatenate([sp_flatten(v) for v in boxes])
    flattened_classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    flattened_scores = np.concatenate([sp_flatten(v) for v in scores])

    # Lọc hộp theo ngưỡng
    boxes, classes, scores = filter_boxes(flattened_boxes, flattened_scores, flattened_classes_conf,obj_thresh)

    # Tối ưu NMS bằng cách xử lý theo lô (batch) thay vì từng lớp
    unique_classes = np.unique(classes)

    # Nếu không có kết quả sau khi lọc, trả về None
    if len(unique_classes) == 0:
        return None, None, None

    # Tối ưu: Dự phân bổ mảng để tránh resize nhiều lần
    max_boxes = len(boxes)  # max possible boxes
    all_indices = np.arange(max_boxes)

    result_boxes = []
    result_classes = []
    result_scores = []

    # Áp dụng NMS cho từng lớp
    for c in unique_classes:
        inds = np.where(classes == c)[0]
        b = boxes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, nms_thresh)

        if len(keep) > 0:
            result_boxes.append(b[keep])
            result_classes.append(np.full(len(keep), c))
            result_scores.append(s[keep])

    # Kiểm tra nếu không có kết quả
    if not result_boxes:
        return None, None, None

    # Ghép kết quả
    final_boxes = np.concatenate(result_boxes)
    final_classes = np.concatenate(result_classes)
    final_scores = np.concatenate(result_scores)

    return final_boxes, final_classes, final_scores


def draw(image, boxes, scores, classes,CLASSES):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def setup_model(args):
    model_path = args.get('model_path')
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from app.ultils.pytorch_executor import Torch_model_container
        model = Torch_model_container(model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from app.ultils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(model_path, args.get("target"), args.get("device_id"), args.get("stt"))
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from app.ultils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform


def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False