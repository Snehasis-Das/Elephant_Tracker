# src/detection/detector.py

from ultralytics import YOLO

class ElephantDetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.5):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame):
        """
        Run YOLO detection on a frame.

        Args:
            frame (np.ndarray): input frame (BGR)

        Returns:
            list: [[x1, y1, x2, y2, conf, class_name], ...]
        """
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]  # use YOLO class label

            # ✅ no filter → keep everything
            detections.append([x1, y1, x2, y2, conf, class_name])

        return detections