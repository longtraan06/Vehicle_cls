import cv2
import numpy as np
import ncnn
import time
import os
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import queue
from threading import Thread
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from threading import Thread, Event

def convert_bbox_to_z(bbox):
    """
    Chuyển đổi hộp giới hạn [x1, y1, x2, y2] thành dạng phép đo [cx, cy, a, r]
    trong đó cx, cy là tâm, a là diện tích, r là tỷ lệ khung hình.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + 0.5 * w
    cy = bbox[1] + 0.5 * h
    a = w * h
    r = w / float(h)
    return np.array([cx, cy, a, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    """
    Chuyển đổi vector trạng thái thành hộp giới hạn.
    Phiên bản "bất tử": Xử lý cả trường hợp vector trạng thái đầu vào chứa NaN.
    """
    # Lớp phòng vệ 1: Kiểm tra toàn bộ vector trạng thái đầu vào
    if np.any(np.isnan(x)):
        return np.array([[0, 0, 0, 0]]) # Trả về hộp vô hiệu nếu trạng thái hỏng

    s = x[2]
    r = x[3]

    # Lớp phòng vệ 2: Kẹp các giá trị để đảm bảo tính vật lý
    s = max(0, s)
    r = max(1e-4, r)

    w = np.sqrt(s * r)
    h = s / w if w > 1e-4 else 0

    # Lớp phòng vệ 3: Kiểm tra kết quả tính toán cuối cùng
    if np.isnan(w) or np.isnan(h) or np.isinf(w) or np.isinf(h):
        return np.array([[0, 0, 0, 0]])

    return np.array([
        x[0] - 0.5 * w,
        x[1] - 0.5 * h,
        x[0] + 0.5 * w,
        x[1] + 0.5 * h
    ]).reshape((1, 4))


class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.class_name = None
        self.color = None
        self.class_counts = defaultdict(int)
        self.best_image_info = {'score': 0.0, 'image': None}

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        self.kf.predict()
        if self.kf.x[2] < 0:
            self.kf.x[2] = 0
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)
    
class CameraGrabber(Thread):
    """
    Một luồng chuyên dụng để đọc frame từ một nguồn video.
    PHIÊN BẢN NÂNG CẤP: Tự động phát hiện nguồn là video file hay live stream.
    - Nếu là video file: Sẽ hãm tốc độ đọc để giả lập FPS thật, tránh làm sập hệ thống.
    - Nếu là live stream: Sẽ đọc nhanh nhất có thể và có cơ chế tự kết nối lại.
    """
    def __init__(self, camera_id, source, input_queue, stop_event, target_fps_for_video=30, processing_width=None):
        super().__init__()
        self.daemon = True
        self.camera_id = camera_id
        self.source = source
        self.input_queue = input_queue
        self.stop_event = stop_event
        self.target_fps_for_video = target_fps_for_video
        self.processing_width = processing_width
        print(f"[INFO] CameraGrabber {self.camera_id} đang khởi tạo cho nguồn: {self.source}")

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Không thể mở camera {self.camera_id} tại nguồn: {self.source}")
            return

        # --- TỰ ĐỘNG PHÁT HIỆN LOẠI NGUỒN ---
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        is_live_stream = frame_count < 1 # Video files có frame_count > 0

        if is_live_stream:
            print(f"[INFO] CameraGrabber {self.camera_id}: Đã phát hiện LIVE STREAM.")
        else:
            print(f"[INFO] CameraGrabber {self.camera_id}: Đã phát hiện VIDEO FILE. Sẽ hãm tốc độ về ~{self.target_fps_for_video} FPS.")
            frame_delay = 1.0 / self.target_fps_for_video

        while not self.stop_event.is_set():
            # Chỉ bắt đầu đếm giờ nếu là video file
            if not is_live_stream:
                start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                # Logic xử lý khi kết thúc stream
                if is_live_stream:
                    print(f"[WARNING] Mất kết nối hoặc kết thúc stream từ camera {self.camera_id}. Thử kết nối lại sau 5s...")
                    cap.release()
                    time.sleep(5)
                    cap = cv2.VideoCapture(self.source)
                    continue
                else:
                    print(f"[INFO] CameraGrabber {self.camera_id}: Đã xử lý hết video file. Dừng luồng.")
                    break # Thoát vòng lặp nếu là video file
            if self.processing_width is not None and frame.shape[1] > self.processing_width:
                h, w, _ = frame.shape
                ratio = self.processing_width / w
                new_h = int(h * ratio)
                frame = cv2.resize(frame, (self.processing_width, new_h), interpolation=cv2.INTER_LINEAR)
            # Đẩy frame vào hàng đợi một cách an toàn
            try:
                if self.input_queue.full():
                    self.input_queue.get_nowait() # Bỏ frame cũ nhất nếu hàng đợi đầy
                self.input_queue.put((self.camera_id, frame), timeout=1)
            except queue.Full:
                continue
            
            # Chỉ áp dụng delay hãm tốc nếu là video file
            if not is_live_stream:
                elapsed_time = time.time() - start_time
                sleep_time = frame_delay - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        cap.release()
        print(f"[INFO] CameraGrabber {self.camera_id} đã dừng.")

# Dán đoạn mã này để thay thế hoàn toàn lớp DisplayWorker cũ

class DisplayWorker(Thread):
    """
    Một luồng chuyên dụng để hiển thị frame (nếu UI được bật) VÀ
    luôn tính toán FPS thông lượng cuối cùng.
    PHIÊN BẢN NÂNG CẤP: Tự vẽ các đối tượng được theo dõi.
    """
    def __init__(self, display_queue, stop_event, fps_stats, ui_enabled=True, min_hits_to_display=3, video_writer_queue=None):
        super().__init__()
        self.daemon = True
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.fps_stats = fps_stats
        self.ui_enabled = ui_enabled
        self.min_hits_to_display = min_hits_to_display # <<< THÊM MỚI: Nhận cấu hình min_hits
        self.video_writer_queue = video_writer_queue
        self.last_finalized_images = {} 
    def run(self):
        if self.ui_enabled:
            print("[INFO] DisplayWorker đang chạy ở chế độ UI. Nhấn 'q' trên cửa sổ video để thoát.")
        else:
            print("[INFO] DisplayWorker đang chạy ở chế độ Headless (chỉ tính FPS).")

        frame_counts = defaultdict(int)
        start_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # <<< THAY ĐỔI: Nhận gói dữ liệu mới, có thêm 'finalized_vehicles' >>>
                cam_id, frame, trackers_to_draw, proc_fps, finalized_vehicles = self.display_queue.get(timeout=1)

                # <<< LOGIC MỚI: Cập nhật ảnh cuối cùng nếu có >>>
                if finalized_vehicles:
                    # Lấy phương tiện cuối cùng trong danh sách vừa được hoàn tất
                    last_vehicle = finalized_vehicles[-1] 
                    if 'image' in last_vehicle and last_vehicle['image'] is not None:
                        # Lưu ảnh này vào bộ nhớ của worker
                        self.last_finalized_images[cam_id] = last_vehicle['image']

                frame_counts[cam_id] += 1
                
                # ... (phần tính toán FPS giữ nguyên) ...
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 1.0:
                    for cid, count in frame_counts.items():
                        self.fps_stats[cid] = count / elapsed_time
                    frame_counts.clear()
                    start_time = current_time
                
                display_fps = self.fps_stats.get(cam_id, 0.0)

                # Vẽ các tracker đang hoạt động (giữ nguyên)
                for tracker in trackers_to_draw:
                    box = tracker.get_state()[0]
                    if tracker.hits >= self.min_hits_to_display and not np.any(np.isnan(box)):
                        x1, y1, x2, y2 = box.astype(int)
                        label = f"ID {tracker.id}: {tracker.class_name}"
                        pt1, pt2 = (x1, y1), (x2, y2)
                        color = tracker.color if tracker.color is not None else (0, 0, 255)
                        cv2.rectangle(frame, pt1, pt2, color, 2)
                        cv2.putText(frame, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Vẽ thông tin FPS (giữ nguyên)
                cv2.putText(frame, f"Proc FPS: {proc_fps:.2f} (Tracking)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"System FPS: {display_fps:.2f} (Display)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # <<< LOGIC MỚI: Vẽ ảnh thumbnail của phương tiện đã lưu >>>
                if cam_id in self.last_finalized_images:
                    saved_img = self.last_finalized_images[cam_id]
                    
                    # Kích thước thumbnail
                    thumb_h, thumb_w = 100, 100
                    try:
                        thumbnail = cv2.resize(saved_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                        
                        # Vị trí để vẽ thumbnail (bên dưới thông tin FPS)
                        y_offset = 90
                        x_offset = 10
                        
                        # Đảm bảo không vẽ ra ngoài khung hình
                        if y_offset + thumb_h < frame.shape[0] and x_offset + thumb_w < frame.shape[1]:
                            # Vẽ tiêu đề
                            cv2.putText(frame, "Last Vehicle Saved:", (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            # Đặt thumbnail vào frame
                            frame[y_offset:y_offset + thumb_h, x_offset:x_offset + thumb_w] = thumbnail
                    except cv2.error:
                        # Bỏ qua nếu ảnh bị lỗi (ví dụ: kích thước bằng 0)
                        pass

                # Gửi frame cho VideoWriter (giữ nguyên)
                if self.video_writer_queue is not None:
                    try:
                        self.video_writer_queue.put((cam_id, frame.copy()), block=False)
                    except queue.Full:
                        pass
                
                # Hiển thị UI (giữ nguyên)
                if self.ui_enabled:
                    cv2.imshow(f"Camera {cam_id}", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_event.set()
                        break
            except queue.Empty:
                if self.ui_enabled and cv2.waitKey(1) & 0xFF == ord('q'):
                     self.stop_event.set()
                     break
                continue
        
        if self.ui_enabled:
            cv2.destroyAllWindows()
        print("[INFO] DisplayWorker đã dừng.")

class InferenceWorker(Thread):
    """
    Một luồng duy nhất nhận các frame từ hàng đợi, thực hiện suy luận theo lô,
    và đưa kết quả vào hàng đợi đầu ra.
    """
    def __init__(self, model, input_queue, output_queue, stop_event, batch_size=2):
        super().__init__()
        self.daemon = True
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.batch_size = batch_size
        print(f"[INFO] InferenceWorker khởi tạo với batch size = {self.batch_size}")

    def run(self):
        while not self.stop_event.is_set():
            batch_data = []
            try:
                # Cố gắng lấy một lô dữ liệu
                for _ in range(self.batch_size):
                    # Lấy item đầu tiên với timeout dài hơn một chút
                    # Các item sau lấy với timeout ngắn để xử lý lô không hoàn chỉnh
                    timeout = 0.5 if len(batch_data) > 0 else 1.0
                    data = self.input_queue.get(timeout=timeout)
                    batch_data.append(data)
            except queue.Empty:
                # Nếu không có dữ liệu sau timeout, tiếp tục vòng lặp
                if not batch_data:
                    continue

            # Tách id và frame ra khỏi lô
            cam_ids, frames = zip(*batch_data)

            # Thực hiện suy luận trên toàn bộ lô
            # Đây là điểm mấu chốt: 1 lần gọi cho nhiều ảnh
            all_detections = self.model.detect_batch(list(frames))

            # Đẩy kết quả vào hàng đợi đầu ra
            for i in range(len(batch_data)):
                try:
                    self.output_queue.put((cam_ids[i], frames[i], all_detections[i]), timeout=1)
                except queue.Full:
                    continue

        print("[INFO] InferenceWorker đã dừng.")



class StatsPrinter(Thread):
    """
    Một luồng siêu nhẹ chuyên dụng để in tóm tắt thống kê FPS ra console
    một cách định kỳ, phù hợp cho môi trường headless.
    """
    def __init__(self, fps_stats, stop_event, print_interval=2.0):
        super().__init__()
        self.daemon = True
        self.fps_stats = fps_stats
        self.stop_event = stop_event
        self.print_interval = print_interval # In thống kê sau mỗi 2 giây

    def run(self):
        print("[INFO] StatsPrinter bắt đầu chạy.")
        while not self.stop_event.is_set():
            # Đợi cho đến kỳ in tiếp theo
            if self.stop_event.wait(self.print_interval):
                # Nếu nhận được tín hiệu dừng trong lúc đợi, thoát ngay
                break

            # Tạo chuỗi tóm tắt FPS
            summary_parts = []
            # Sắp xếp theo cam_id để thứ tự in ra luôn ổn định
            sorted_cam_ids = sorted(self.fps_stats.keys())

            for cam_id in sorted_cam_ids:
                fps = self.fps_stats.get(cam_id, 0.0)
                summary_parts.append(f"Cam {cam_id}: {fps:.2f} FPS")
            
            summary_string = " | ".join(summary_parts)
            
            # In ra console và sử dụng '\r' để ghi đè lên dòng cũ
            # flush=True đảm bảo log được ghi ra ngay lập tức trong container
            if summary_string:
                print(f"\r[STATS] {summary_string}", end="", flush=True)

        print("\n[INFO] StatsPrinter đã dừng.")

class VideoWriterWorker(Thread):
    """
    PHIÊN BẢN NÂNG CẤP: Logic dừng luồng mạnh mẽ hơn.
    """
    def __init__(self, writer_queue, stop_event, output_dir="output_videos", fps=20.0):
        super().__init__()
        self.daemon = True
        self.writer_queue = writer_queue
        self.stop_event = stop_event
        self.output_dir = output_dir
        self.fps = fps
        self.video_writers = {}
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] VideoWriterWorker đã sẵn sàng, video sẽ được lưu tại: {self.output_dir}")

    def run(self):
        try:
            while True: # <<< THAY ĐỔI: Vòng lặp vô tận
                try:
                    # Chờ dữ liệu, nhưng không timeout quá lâu để có thể kiểm tra stop_event
                    data = self.writer_queue.get(timeout=0.5) 
                    
                    # <<< LOGIC MỚI: Kiểm tra "Poison Pill" >>>
                    if data is None:
                        # Nhận được tín hiệu dừng, thoát khỏi vòng lặp
                        break

                    cam_id, frame = data
                    
                    if cam_id not in self.video_writers:
                        h, w, _ = frame.shape
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        file_path = os.path.join(self.output_dir, f"output_cam_{cam_id}.avi")
                        print(f"[INFO] Tạo file video cho Camera {cam_id} tại: {file_path}")
                        self.video_writers[cam_id] = cv2.VideoWriter(file_path, fourcc, self.fps, (w, h))
                    
                    self.video_writers[cam_id].write(frame)
                    self.writer_queue.task_done() # Báo cho queue biết item đã được xử lý

                except queue.Empty:
                    # Nếu không có frame, kiểm tra xem có nên dừng không
                    if self.stop_event.is_set():
                        break
                    continue
        finally:
            # Dọn dẹp: Đóng tất cả các file video khi luồng dừng
            print("[INFO] VideoWriterWorker đang đóng các file video...")
            for writer in self.video_writers.values():
                writer.release()
            print("[INFO] VideoWriterWorker đã dừng và đóng file thành công.")

    def close(self):
        # Có thể gọi hàm này từ bên ngoài để chủ động đóng
        self.stop_event.set()

# Lớp YOLOv8NCNN giữ nguyên, không cần thay đổi
class YOLOv8NCNN:
    def __init__(self, param_path, bin_path, conf_threshold=0.25, iou_threshold=0.45, per_class_conf=None):
        self.input_size = 640 # <<< QUAN TRỌNG: Đảm bảo kích thước này khớp với lúc bạn export model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.per_class_conf = per_class_conf if per_class_conf is not None else {}
        self.net = ncnn.Net()

        # Enable optimization for ARM CPU
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 4  # Adjust based on your RPi5 cores
        
        # Load model
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        np.random.seed(112006) # Đảm bảo màu sắc không thay đổi mỗi lần chạy
        self.colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(self.class_names))]
    
    def preprocess(self, img):
        """Preprocess image for YOLOv8"""
        # Resize with letterbox
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded, scale, pad_h, pad_w
    
    def postprocess(self, output, scale, pad_h, pad_w, orig_shape):
        """Postprocess YOLOv8 output (Optimized Version)"""
        predictions = output[0].T
        
        boxes = predictions[:, :4]  # cx, cy, w, h
        scores = predictions[:, 4:]
        
        # 1. Lọc bằng ngưỡng tin cậy (Vectorized)
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Lọc những detection có confidence dưới ngưỡng chung
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if boxes.shape[0] == 0:
            return []

        # 2. Chuyển đổi hộp từ (cx, cy, w, h) sang (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # 3. Scale lại tọa độ về ảnh gốc
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip tọa độ
        x1 = np.clip(x1, 0, orig_shape[1])
        y1 = np.clip(y1, 0, orig_shape[0])
        x2 = np.clip(x2, 0, orig_shape[1])
        y2 = np.clip(y2, 0, orig_shape[0])
        
        # Chuẩn bị dữ liệu cho cv2.dnn.NMSBoxes
        # Chuyển (x1, y1, x2, y2) thành (x, y, w, h)
        boxes_for_nms = np.column_stack([x1, y1, x2 - x1, y2 - y1]).astype(int)

        # 4. SỬ DỤNG OPENCV NMS TỐI ƯU
        # Chú ý: cv2.dnn.NMSBoxes cần boxes ở định dạng (x, y, w, h)
        keep_indices = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)
        
        # Nếu keep_indices không rỗng, nó là một mảng 2D, cần flatten
        if isinstance(keep_indices, np.ndarray):
            keep_indices = keep_indices.flatten()
        else: # Có thể trả về tuple rỗng
            return []

        # 5. Lọc và tạo kết quả cuối cùng
        allowed_classes = {'bicycle', 'car', 'motorcycle', 'bus', 'truck'} # Dùng set để truy vấn nhanh hơn
        
        results = []
        for idx in keep_indices:
            class_id = class_ids[idx]
            class_name = self.class_names[class_id]
            
            # Lọc class và lọc lại theo ngưỡng riêng (nếu có)
            final_threshold = self.per_class_conf.get(class_name, self.conf_threshold)

            if class_name in allowed_classes and confidences[idx] >= final_threshold:
                box = np.array([x1[idx], y1[idx], x2[idx], y2[idx]])
                results.append({
                    'box': box.astype(int),
                    'confidence': float(confidences[idx]),
                    'class_id': int(class_id),
                    'class_name': class_name
                })
                
        return results

    def detect(self, img, debug=False):
        """Run detection on an image"""
        orig_shape = img.shape[:2]
        
        preprocessed, scale, pad_h, pad_w = self.preprocess(img)
        
        mat_in = ncnn.Mat.from_pixels(
            preprocessed,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.input_size,
            self.input_size
        )
        
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])
        
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        
        ret, mat_out = ex.extract("out0")
        
        output = np.array(mat_out)
        
        if len(output.shape) == 1:
            output = output.reshape(84, -1)[np.newaxis, :, :]
        elif len(output.shape) == 2:
            output = output[np.newaxis, :, :]
            
        detections = self.postprocess(output, scale, pad_h, pad_w, orig_shape)
        
        return detections
    def detect_batch(self, images):
        """
        Thực hiện phát hiện trên một lô (danh sách) các ảnh.
        Trả về một danh sách các kết quả detection.
        """
        results = []
        for img in images:
            # Gọi hàm detect cho từng ảnh
            detections = self.detect(img)
            results.append(detections)
        return results

class VideoWriterAsync:
    def __init__(self, name, fourcc, fps, frame_size):
        self.q = queue.Queue()
        self.writer = cv2.VideoWriter(name, fourcc, fps, frame_size)
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            frame = self.q.get()
            if frame is None: # Tín hiệu kết thúc
                break
            self.writer.write(frame)
        self.writer.release()

    def write(self, frame):
        self.q.put(frame)

    def release(self):
        self.q.put(None) # Gửi tín hiệu kết thúc
        self.thread.join() # Chờ thread kết thúc


def calculate_iou(boxA, boxB):
    """
    Tính toán chỉ số Intersection over Union (IoU) giữa hai hộp bao.
    Args:
        boxA, boxB: Tọa độ hộp dạng [x1, y1, x2, y2]
    Returns:
        Giá trị IoU (float)
    """
    # Xác định tọa độ của vùng giao nhau (intersection)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích vùng giao nhau
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Tính diện tích của từng hộp
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Tính diện tích vùng hợp nhất (union)
    unionArea = float(boxAArea + boxBArea - interArea)

    # Tính IoU
    iou = interArea / unionArea if unionArea > 0 else 0
    return iou

def calculate_iou_matrix(boxesA, boxesB):
    if boxesA.size == 0 or boxesB.size == 0:
        return np.empty((boxesA.shape[0], boxesB.shape[0]))

    # Mở rộng chiều để tận dụng NumPy broadcasting
    # boxesA sẽ có shape (N, 1, 4)
    # boxesB sẽ có shape (1, M, 4)
    # NumPy sẽ so sánh mỗi hộp trong A với mỗi hộp trong B
    boxesA = np.expand_dims(boxesA, axis=1)
    boxesB = np.expand_dims(boxesB, axis=0)
    
    # Xác định tọa độ của vùng giao nhau (intersection)
    # Kết quả sẽ là một ma trận shape (N, M)
    xA = np.maximum(boxesA[..., 0], boxesB[..., 0])
    yA = np.maximum(boxesA[..., 1], boxesB[..., 1])
    xB = np.minimum(boxesA[..., 2], boxesB[..., 2])
    yB = np.minimum(boxesA[..., 3], boxesB[..., 3])

    # Tính diện tích vùng giao nhau
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Tính diện tích của từng hộp
    boxAArea = (boxesA[..., 2] - boxesA[..., 0]) * (boxesA[..., 3] - boxesA[..., 1])
    boxBArea = (boxesB[..., 2] - boxesB[..., 0]) * (boxesB[..., 3] - boxesB[..., 1])
    
    # Tính IoU
    unionArea = boxAArea + boxBArea - interArea
    iou = interArea / unionArea
    
    # Xử lý trường hợp unionArea = 0 (tránh chia cho 0)
    iou[unionArea == 0] = 0
    
    return iou

def associate_detections_to_trackers(tracked_boxes, detections_boxes, iou_threshold=0.3):
    """
    Liên kết detections với trackers bằng cách sử dụng ma trận IoU vector hóa và thuật toán Hungarian.
    Phiên bản này được "bọc thép" để chống lại các giá trị đầu vào không hợp lệ.
    """
    tracked_boxes = np.asarray(tracked_boxes)
    detections_boxes = np.asarray(detections_boxes)

    if tracked_boxes.size == 0 or detections_boxes.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections_boxes)),
            np.arange(len(tracked_boxes)),
        )

    iou_matrix = calculate_iou_matrix(tracked_boxes, detections_boxes)
    cost_matrix = 1 - iou_matrix
    
    # === LỚP PHÒNG VỆ QUAN TRỌNG NHẤT ===
    # Nếu một tracker tạo ra dự đoán không hợp lệ (NaN), cost_matrix sẽ chứa NaN.
    # Ta thay thế các giá trị NaN bằng 1.0 (chi phí tối đa), đảm bảo chúng không bao giờ được khớp.
    cost_matrix[np.isnan(cost_matrix)] = 1.0

    # Bây giờ thuật toán Hungarian sẽ luôn an toàn để chạy
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    matched_trackers_indices = set()
    matched_detections_indices = set()

    for r, c in zip(row_ind, col_ind):
        # Chỉ giữ lại các cặp ghép có IoU > ngưỡng VÀ chi phí hợp lệ
        if iou_matrix[r, c] >= iou_threshold:
            matches.append([r, c])
            matched_trackers_indices.add(r)
            matched_detections_indices.add(c)
    
    all_trackers_indices = set(range(tracked_boxes.shape[0]))
    all_detections_indices = set(range(detections_boxes.shape[0]))
    
    unmatched_trackers = np.array(list(all_trackers_indices - matched_trackers_indices))
    unmatched_detections = np.array(list(all_detections_indices - matched_detections_indices))

    return np.array(matches, dtype=int), unmatched_detections, unmatched_trackers

def calculate_quality_score(frame, box, confidence):
    """
    Tính điểm chất lượng cho một vùng ảnh được phát hiện dựa trên nhiều yếu tố.
    """
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return 0.0

    # 1. Trọng số cho độ tin cậy (quan trọng nhất)
    score = confidence * 1.5

    # 2. Trọng số cho diện tích
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame.shape[0] * frame.shape[1]
    normalized_area = box_area / frame_area
    score += normalized_area * 0.5

    # 3. Trọng số cho vị trí trung tâm (càng gần trung tâm càng tốt)
    frame_center_x, frame_center_y = frame.shape[1] / 2, frame.shape[0] / 2
    box_center_x, box_center_y = (x1 + x2) / 2, (y1 + y2) / 2
    dist_from_center = np.sqrt((frame_center_x - box_center_x)**2 + (frame_center_y - box_center_y)**2)
    max_dist = np.sqrt(frame_center_x**2 + frame_center_y**2)
    centrality_score = 1.0 - (dist_from_center / max_dist)
    score += centrality_score * 1.0

    # 4. (Nâng cao) Trọng số cho độ sắc nét (chống mờ)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_sharpness = min(sharpness / 2000.0, 1.0) # Cần tinh chỉnh ngưỡng 2000
        score += normalized_sharpness * 0.7

    return score

def finalize_vehicle_data_from_tracker(tracker):
    # Logic tương tự, nhưng lấy dữ liệu từ thuộc tính của tracker
    if not tracker.class_counts:
        return None
    final_class = max(tracker.class_counts, key=tracker.class_counts.get)
    best_image = tracker.best_image_info['image']
    if best_image is not None:
        return {
            'id': tracker.id,
            'final_class': final_class,
            'class_details': dict(tracker.class_counts),
            'image': best_image,
            'last_box': tracker.get_state()[0].astype(int)
        }
    return None

def save_finalized_results(finalized_vehicles, output_dir="finalized_vehicles"):
    """
    Lưu tất cả hình ảnh và thông tin đã được phân loại cuối cùng.
    """
    if not finalized_vehicles:
        return
        
    # print(f"\nSaving results for {len(finalized_vehicles)} finalized vehicles...")
    os.makedirs(output_dir, exist_ok=True)
    
    for vehicle in finalized_vehicles:
        file_name = f"ID_{vehicle['id']}_{vehicle['final_class']}.jpg"
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, vehicle['image'])
        # print(f"  - Saved: {file_name} | Details: {vehicle['class_details']}")

def main(ENABLE_UI=True, SAVE_VIDEO=False,
        STATS_PRINT_INTERVAL=1.0,  PARAM_PATH=str, 
        BIN_PATH=str, CAMERA_SOURCES=None, PROCESSING_WIDTH=800,
        DETECTION_INTERVAL=4, IOU_THRESHOLD_TRACKING=0.3,
        MAX_AGE=15, MIN_HITS_TO_DISPLAY = 2, MIN_HITS_FOR_CLASSIFICATION = 10,
        DEFAULT_CONF_THRESHOLD=0.3, PER_CLASS_THRESHOLDS=None):

    print("[INFO] loading YOLO NCNN...")
    model = YOLOv8NCNN(PARAM_PATH, BIN_PATH, conf_threshold=DEFAULT_CONF_THRESHOLD, iou_threshold=0.5, per_class_conf=PER_CLASS_THRESHOLDS)
    print("[INFO] model loaded.")

    raw_frame_queue = queue.Queue(maxsize=len(CAMERA_SOURCES) * 2)
    inference_input_queue = queue.Queue(maxsize=2)
    inference_output_queue = queue.Queue(maxsize=2)
    display_queue = queue.Queue(maxsize=len(CAMERA_SOURCES) * 2)
    video_writer_queue = queue.Queue(maxsize=len(CAMERA_SOURCES) * 5) if SAVE_VIDEO else None

    stop_event = Event()
    threads = []
    fps_stats = {} 

    
    # 1. Các luồng CameraGrabber
    for i, source in enumerate(CAMERA_SOURCES):
        grabber = CameraGrabber(i, source, raw_frame_queue, stop_event, processing_width=PROCESSING_WIDTH)
        threads.append(grabber)

    # 2. Một luồng InferenceWorker
    inference_worker = InferenceWorker(model, inference_input_queue, inference_output_queue, stop_event, batch_size=1)
    threads.append(inference_worker)

    display_worker = DisplayWorker(
        display_queue, stop_event, fps_stats, 
        ui_enabled=ENABLE_UI, 
        min_hits_to_display=MIN_HITS_TO_DISPLAY,
        video_writer_queue=video_writer_queue  # <<< KẾT NỐI
    )
    threads.append(display_worker)

    stats_printer = StatsPrinter(fps_stats, stop_event, print_interval=STATS_PRINT_INTERVAL)
    threads.append(stats_printer)

    if SAVE_VIDEO:
        video_writer_worker = VideoWriterWorker(video_writer_queue, stop_event, fps=20.0)
        threads.append(video_writer_worker)

    print(f"[INFO] run {len(threads)} thread...")
    for t in threads:
        t.start()

    trackers_per_camera = {i: [] for i in range(len(CAMERA_SOURCES))}
    frame_counters = {i: 0 for i in range(len(CAMERA_SOURCES))}
    KalmanBoxTracker.count = 0

    all_finalized_vehicles = []

    try:
        while not stop_event.is_set():
            try:
                cam_id, frame = raw_frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            finalized_this_frame = []

            start_time = time.time()
            current_trackers = trackers_per_camera[cam_id]
            
            predicted_boxes = [tracker.predict()[0] for tracker in current_trackers]
            
            if frame_counters[cam_id] % DETECTION_INTERVAL == 0:
                # Lấy kết quả detection
                inference_input_queue.put((cam_id, frame))
                processed_cam_id, _, detections = inference_output_queue.get(timeout=2)
                
                detection_boxes = [d['box'] for d in detections]
                
                matches, unmatched_dets, unmatched_trks = \
                    associate_detections_to_trackers(np.array(predicted_boxes), np.array(detection_boxes), iou_threshold=IOU_THRESHOLD_TRACKING)

                for match in matches:
                    tracker_idx, detection_idx = match[0], match[1]
                    det = detections[detection_idx]
                    tracker = current_trackers[tracker_idx]
                    
                    tracker.update(det['box'])
                    tracker.class_name = det['class_name']
                    tracker.color = model.colors[det['class_id']]
                    
                    tracker.class_counts[det['class_name']] += 1
                    
                    # 2. Tính điểm chất lượng và cập nhật ảnh tốt nhất
                    box = det['box'].astype(int)
                    score = calculate_quality_score(frame, box, det['confidence'])
                    
                    if score > tracker.best_image_info['score']:
                        tracker.best_image_info['score'] = score
                        # Crop và lưu lại hình ảnh
                        x1, y1, x2, y2 = box
                        tracker.best_image_info['image'] = frame[y1:y2, x1:x2].copy()

                for d_idx in unmatched_dets:
                    det = detections[d_idx]
                    new_tracker = KalmanBoxTracker(det['box'])
                    new_tracker.class_name = det['class_name']
                    new_tracker.color = model.colors[det['class_id']]
                    # Cập nhật lần đầu cho tracker mới
                    new_tracker.class_counts[det['class_name']] += 1
                    current_trackers.append(new_tracker)
                
                next_frame_trackers = []
                
                for i, t in enumerate(current_trackers):
                    is_unmatched_and_old = i in unmatched_trks and t.time_since_update > MAX_AGE
                    
                    if not is_unmatched_and_old:
                        next_frame_trackers.append(t)
                    else:
                        if t.hits >= MIN_HITS_FOR_CLASSIFICATION:
                            final_data = finalize_vehicle_data_from_tracker(t)
                            if final_data:
                                finalized_this_frame.append(final_data)
                                all_finalized_vehicles.append(final_data)
                trackers_per_camera[cam_id] = next_frame_trackers
                if finalized_this_frame:
                    save_finalized_results(finalized_this_frame)
            
            processing_time = time.time() - start_time
            proc_fps = 1.0 / processing_time if processing_time > 0 else 0
            
            try:
                display_queue.put((cam_id, frame, trackers_per_camera[cam_id], proc_fps, finalized_this_frame), timeout=1)
            except queue.Full:
                pass

            frame_counters[cam_id] += 1

    except KeyboardInterrupt:
        print("\n[INFO] catch (Ctrl+C)...")
    finally:
        print("[INFO] Đang dừng các luồng, vui lòng đợi...")
        if not stop_event.is_set():
            stop_event.set()
        time.sleep(1) 
        if SAVE_VIDEO:
            if not video_writer_queue.full():
                 video_writer_queue.put(None)
        
        for t in threads:
            t.join(timeout=5.0)
        
        print("\n--- TỔNG KẾT ---")
        print(f"Tổng số phương tiện đã được theo dõi và hoàn tất: {len(all_finalized_vehicles)}")
        save_finalized_results(all_finalized_vehicles)

        print("[INFO] Hệ thống đã dừng hoàn toàn.")

if __name__ == "__main__":

    # --- Cấu hình Chung ---
    ENABLE_UI = True
    STATS_PRINT_INTERVAL = 1.0
    SAVE_VIDEO = False
    # --- Cấu hình Mô hình và Nguồn Camera ---
    PARAM_PATH = "yolo11n_ncnn_model/model.ncnn.param"
    BIN_PATH = "yolo11n_ncnn_model/model.ncnn.bin"
    CAMERA_SOURCES = [
        "tvid_shorter.mp4",
        "4.mp4"
    ]
    
    # --- Cấu hình Hiệu Năng và Tracking ---
    PROCESSING_WIDTH = 800
    DETECTION_INTERVAL = 4
    IOU_THRESHOLD_TRACKING = 0.3    
    MAX_AGE = 15           
    MIN_HITS_TO_DISPLAY = 2     
    MIN_HITS_FOR_CLASSIFICATION = 10
    # --- Cấu hình Ngưỡng Tin Cậy của Model ---
    DEFAULT_CONF_THRESHOLD = 0.3
    PER_CLASS_THRESHOLDS = {
        'bicycle': 0.55, 'car': 0.35, 'motorcycle': 0.15,
        'bus': 0.5, 'truck': 0.4, 'person':0.1
    }
    main(ENABLE_UI, SAVE_VIDEO,
        STATS_PRINT_INTERVAL, PARAM_PATH, 
        BIN_PATH, CAMERA_SOURCES, PROCESSING_WIDTH,
        DETECTION_INTERVAL, IOU_THRESHOLD_TRACKING,
        MAX_AGE, MIN_HITS_TO_DISPLAY, MIN_HITS_FOR_CLASSIFICATION,
        DEFAULT_CONF_THRESHOLD, PER_CLASS_THRESHOLDS)