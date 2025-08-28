# src/detection/tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class ElephantTracker:
    def __init__(self, max_age=30, n_init=3):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections, frame):
        """
        Args:
            detections (list): [[x1,y1,x2,y2,conf,class_name], ...]
            frame (np.ndarray): frame
        Returns:
            list: [(track_id, [x1,y1,x2,y2], conf, class_name), ...]
        """
        input_dets = [
            ([x1, y1, x2, y2], conf, class_name)
            for (x1, y1, x2, y2, conf, class_name) in detections
        ]

        tracks = self.tracker.update_tracks(input_dets, frame=frame)

        output = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            output.append((track_id, [l, t, r, b], track.det_conf, track.det_class))
        return output