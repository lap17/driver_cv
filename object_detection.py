import numpy as np
import cv2
import copy


SELECTED_CLASSES  = { 
    "MAIN": { 67: 'cell phone', 0: 'person'},
    "CUSTOM": { 0: 'seatbelt'}
}

DRAW_PARAMS = { 
    "MAIN": { 67: (55, 0, 255), 0: (255, 0, 55)},
    "CUSTOM": { 0: (0, 255, 55)}
}

MODEL_CONFIG = {
    "MAIN": "models/yolov5l.onnx",
    "CUSTOM": "models/seatbelt_v0.onnx" 
}

CONFIDENCE_VALUE = 0.6


def yolo_predictions(detect_img, draw_img, net, model):
    input_image, detections = get_detections(detect_img, net)
    boxes_np, confidences_np, class_ids, index = non_maximum_supression(input_image, detections, model)
    result_img, result_data = drawing_boxes(draw_img, boxes_np, confidences_np, class_ids, index, model)
    return result_img, result_data


def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3),dtype=np.uint8)
    input_image[0:row,0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (640,640), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections


def non_maximum_supression(input_image, detections, model):
    boxes = []
    confidences = []
    class_ids = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/640
    y_factor = image_h/640
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > CONFIDENCE_VALUE:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > 0.25:
                if class_id in SELECTED_CLASSES[model]:
                    class_ids.append(class_id)
                    cx, cy , w, h = row[0:4]
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy-0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)
                    box = np.array([left,top, width, height])
                    confidences.append(confidence)
                    boxes.append(box)
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    return boxes_np, confidences_np, class_ids, index


def drawing_boxes(image, boxes_np, confidences_np, class_ids, index, model):
    if model == "MAIN":
        result_data = {
            'cell phone': [],
            'person': []
        }
    else:
        result_data = {
            'seatbelt': []
        }
    for ind in index:
        class_label = SELECTED_CLASSES[model][class_ids[ind]]
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = '{}: {:.0f}%'.format(class_label, bb_conf*100)
        cv2.rectangle(image, (x, y), (x+w, y+h), DRAW_PARAMS[model][class_ids[ind]], 2)
        cv2.putText(image, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        boxes = result_data[class_label]
        boxes.append(boxes_np[ind])
        result_data[class_label] = boxes
    return image, result_data


def yolo_detection_by_model(detect_img, draw_img, model):
    net = cv2.dnn.readNetFromONNX(MODEL_CONFIG[model])
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    draw_img, result_data = yolo_predictions(detect_img, draw_img, net, model)
    return result_data


def yolo_detection(detect_img, draw_img):
    result_data = {}
    for model in MODEL_CONFIG:
        result_data_by_model = yolo_detection_by_model(detect_img, draw_img, model)
        result_data.update(result_data_by_model)
    detect_alerts(draw_img, result_data)


def detect_alerts(draw_img, result_data):
    if 'person' not in result_data:
        return
    boxes_person = result_data["person"]
    boxes_seatbelt = result_data["seatbelt"]
    boxes_cell_phone = result_data["cell phone"]
    alert_cell_phone = False
    count_seatbelt = 0
    for b_p in boxes_person:
        for b_s in boxes_seatbelt:
            if is_box_inside_box(b_p, b_s):
                count_seatbelt += 1
        for b_c in boxes_cell_phone:
            if is_box_inside_box(b_p, b_c):
                alert_cell_phone = True
                break
    alert_seatbelt = count_seatbelt < len(boxes_person)
    drawing_alerts(draw_img, alert_seatbelt, alert_cell_phone)
    return


def is_box_inside_box(external_box, inner_box):
    centroidx, centroidy = get_centroids(inner_box)
    x, y, w, h =  external_box
    x1, x2, y1, y2 = x, x+w, y, y+h 
    return x1 < centroidx < x2 and y1 < centroidy < y2


def get_centroids(box):
    x, y, w, h =  box
    centroidx = x + w/2
    centroidy = y + h/2
    return centroidx, centroidy


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


def drawing_alerts(img, alert_seatbelt, alert_cell_phone):
    img_h, img_w, _ = img.shape
    SEATBELT_txt_pos = (10, int(img_h // 2 * 1.55))
    CELL_PHONE_txt_pos = (10, int(img_h // 2 * 1.4))
    if alert_seatbelt:
        plot_text(img, 'SEATBELT IS PUT: NO', SEATBELT_txt_pos, (0, 0, 255))
    else:
        plot_text(img, 'SEATBELT IS PUT: YES', SEATBELT_txt_pos, (0, 255, 0))
    if alert_cell_phone:
        plot_text(img, 'CELL PHONE IS USED: YES', CELL_PHONE_txt_pos, (0, 0, 255))
    else:
        plot_text(img, 'CELL PHONE IS USED: NO', CELL_PHONE_txt_pos, (0, 255, 0))
