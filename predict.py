import darknet
import cv2
from imutils.video import VideoStream

def Predictor(image,network, class_names, class_colors,confidence_thr=0.4):
    """
    Input:
         image: input 
         network: yolov4 model,the output of load_network
         class_names: the name of class,the output of load_network
         class_color: box color,the output of load_network
         confidence_thr: confidence ,default 0.4
    Output:
         return : predictor
    """
    width = image.shape[1]
    height = image.shape[0]
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    predictor = darknet.detect_image(network, class_names, darknet_image, thresh=0.4)
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

if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"]="1"
    ######################################################
    ####          TEST PASS
    #####################################################
    vid = "rtsp://admin:qwe,asd.@10.10.15.153:554/h264/ch1/main/av_stream"
    vs = VideoStream(src=vid).start()
    i = 0
    while True:
        i = i+1
        frame =vs.read()
        if frame is not None:
            predictor,img,image_resized = Predictor(frame,network,class_names,class_colors)
            image = darknet.draw_boxes(predictor, image_resized, class_colors)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("./result/"+str(i)+".jpg",img)
            # print(detections)
            cv2.imshow("demo",img)
            cv2.waitKey(1)

