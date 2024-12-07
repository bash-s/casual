import onnxruntime
import onnx
import time
import cv2
import numpy as np
onnx_model = onnx.load("./weights/best.onnx")
onnx.checker.check_model(onnx_model)
img_path = "./qr/1.png"
img = cv2.imread(img_path)
img = cv2.resize(img,(640,640)).astype(np.float32).reshape(1,3,640,640)/255
epochs = 1
total_time = 0
session = onnxruntime.InferenceSession("./weights/best.onnx",providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name
for i in range(epochs):
    t1 = time.time()
    outputs = session.run(None, {input_name: img})
    print(outputs[0].shape)
    t2 = time.time()
    print(f"onnx inference time:{(t2-t1)*1000:.2f}ms")
    total_time += t2-t1
print(f"average onnx inference time:{total_time/epochs*1000:.2f}ms")

class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.7):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        input_tensor,ratio = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs,ratio)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize图片不要直接使用resize，需要按比例缩放，空白区域填空纯色即可
        input_img,ratio = self.ratioresize(input_img)

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor,ratio


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output,ratio):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions,ratio)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions,ratio):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        # boxes = self.rescale_boxes(boxes)
        boxes *= ratio

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    #等比例缩放图片
    def ratioresize(self,im,  color=114):
        shape = im.shape[:2] 
        new_h, new_w = self.input_height, self.input_width
        padded_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * color
    
        # Scale ratio (new / old)
        r = min(new_h / shape[0], new_w / shape[1])
    
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        padded_img[: new_unpad[1], : new_unpad[0]] = im
        padded_img = np.ascontiguousarray(padded_img)
        return padded_img, 1 / r

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
    
        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
    
            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
    
            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]
    
            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]
    
        return keep_boxes
       
    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
    
        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    
        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area
    
        # Compute IoU
        iou = intersection_area / union_area
    
        return iou
    
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

if __name__ == "__main__":
    yolov8_detector = YOLOv8("./weights/best.onnx", conf_thres=0.1, iou_thres=0.1)
    image = cv2.imread("./qr/1.png")
    boxes, scores, class_ids = yolov8_detector(image)
    # print(boxes, scores, class_ids)
    