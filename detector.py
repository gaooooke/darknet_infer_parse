import darknet
import numpy as np
import os
import cv2


lbs = {"nomask":"SP1001",
       "vests": "SP1003",
       "mask":"SP1009"}

def Predictor(image, network, class_names, class_colors,confidence_thr=0.3):
    width = image.shape[1]
    height = image.shape[0]
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    predictor = darknet.detect_image(network, class_names, darknet_image, thresh=confidence_thr)
    darknet.free_image(darknet_image)
    
    return predictor,image,image_resized

weights = "./models/yolo-obj_final.weights"
config_file = "./models/yolo-obj.cfg"
data_file = "./models/obj.data"

network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )

def parse_detector(img):
    labels = []
    scores = []
    bboxes = []

    predictor,image,image_resized = Predictor(img,network,class_names,class_colors)
    
    for i in range(len(predictor)):
        label = predictor[i][0]    
        score = predictor[i][1]
        bbox = predictor[i][2]

        labels.append(label)
        scores.append(score)
        bboxes.append(bbox)

    return bboxes,labels ,scores,image,image_resized,predictor

def compute_iou(box1, box2, box1_area, box2_area):
    # Calculate intersection areas
    x1 = np.maximum(box1[0], box2[0])
    x2 = np.minimum(box1[2], box2[2])
    y1 = np.maximum(box1[1], box2[1])
    y2 = np.minimum(box1[3], box2[3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box1_area + box2_area - intersection
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    """计算两个框的IoU重合度.
    boxes1, boxes2: [x1, y1, x2, y2].
    """
    # 计算两个框的面积
    area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    # 计算两个框的重合度
    overlaps = compute_iou(boxes1, boxes2, area1, area2)
    return overlaps

def shift_box(mboxes):   # XYWH -> XYXY
    box = []
    # x1,y1,x2,y2 = mboxes             # XYWH

    # # left top
    # x1 = x1-x2/2
    # y1 = y1-y2/2

    # # right bottom
    # x2 = x2 + x1
    # y2 = y2 + y1
    x, y, w, h = mboxes
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    
    box.append([xmin, ymin, xmax, ymax])
    return box[0]


def get_max_iou(boxes,bboxes,scores):

    rboxs = []
    if boxes is not None and len(boxes) > 0:                        # me
        if bboxes is not None and len(bboxes) > 0:                  # liu
            for _,lbox in enumerate(bboxes):
                mboxs = []
                ious = []
                for _,mbox in enumerate(boxes):
                    rslt = compute_overlaps(shift_box(list(mbox)), lbox)
                    if rslt > 0:     # IOU > 0
                        ious.append(rslt)
                        mboxs.append(shift_box(list(mbox))) 

                rboxs.append([]) if len(ious) == 0 or len(mboxs) == 0 else rboxs.append(list(mboxs[np.argmin(ious)]))
                    
    return rboxs[0]
    

def Detector(img,bboxes):  
    boxes,classes,scores,image,image_resized,predictor = parse_detector(img)

    type1 = []
    type2 = []
    type3 = []
    for i,clss in enumerate(classes):
        if clss == "vests":
            type1.append(boxes[i])
        elif clss == "nomask":
            type2.append(boxes[i])
        # elif clss == "mask":
        #     type3.append(boxes[i])

    
    box1 = get_max_iou(type1,bboxes,scores) if len(type1) != 0 else []
    box2 = get_max_iou(type2,bboxes,scores) if len(type2) != 0 else []
    # box3 = get_max_iou(type3,bboxes,scores) if len(type3) != 0 else []

    return {"vests":box1,
            "nomask":box2}, image, image_resized,predictor


def box_float2int(x1,y1,x2,y2):
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    return x1,y1,x2,y2


def drawbox(img,bboxes):

    rdict,image,image_resized,predictor = Detector(img,bboxes)
    types = []
    if len(rdict["vests"]) > 0:
        x11,y11,x21,y21 = rdict["vests"]
        x11,y11,x21,y21 = box_float2int(x11,y11,x21,y21)
        types.append(["SP1003"])

    if len(rdict["nomask"]) > 0 :
        x12,y12,x22,y22 = rdict["nomask"]
        x12,y12,x22,y22 = box_float2int(x12,y12,x22,y22)
        types.append(["SP1001"])

    # if len(rdict["mask"]) > 0 :
    #     x13,y13,x23,y23 = rdict["mask"]
    #     x13 = int(x13)
    #     y13 = int(y13)
    #     x23 = int(x23)
    #     y23 = int(y23)  
    #     cv2.rectangle(image, (x13, y13), (x23, y23), (0, 0, 255), 2)
    #     types.append(["SP1009"])

    image = darknet.draw_boxes(predictor, image_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image,types


if __name__ == "__main__":
    # --------------------------TEST crop_person--------------------------------
    img = '/media/ps/D/TeamWork/2020-11-18/11.jpg'
    img = cv2.imread(img)

    # bboxes = [[910,290,1270,1050]]  # 1.jpg
    bboxes = [[888,282,1167,955],[1284,311,1486,960]]  # 2
    # bboxes = [[890,290,1280,1050]]  # 3
    # bboxes = [[900,290,1280,1050]]  # 4
    # bboxes = [[1070,350,1430,1070]] # 5
    # bboxes = [[1060,350,1450,1070]] # 6

    rimg,types = drawbox(img,bboxes)
    # cv2.rectangle(rimg,(bboxes[0][0],bboxes[0][1]),(bboxes[0][2],bboxes[0][3]),(255,255,155),2)
    # cv2.rectangle(rimg,(bboxes[1][0],bboxes[1][1]),(bboxes[1][2],bboxes[1][3]),(255,255,155),2)

    cv2.imshow("demo",rimg)
    cv2.waitKey(0)
