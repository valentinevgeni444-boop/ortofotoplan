# engine.py
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds, from_gcps
from affine import Affine
import logging
import json
from pathlib import Path
import time
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    id: int
    area_m2: float
    perimeter_m: float
    centroid_x: float
    centroid_y: float
    centroid_geo_x: float
    centroid_geo_y: float
    bbox: Tuple[int, int, int, int]
    major_axis_m: float
    minor_axis_m: float
    compactness: float
    class_name: str = ""


@dataclass
class ProcessingParams:
    min_object_area_m2: float = 3.0
    distance_threshold: float = 80.0
    overlap_threshold: float = 0.3
    confidence: float = 0.12
    tile_size: int = 960
    weight_ssim: float = 0.15
    weight_histogram: float = 0.10
    weight_texture: float = 0.20
    weight_hsv: float = 0.20
    weight_orb: float = 0.35
    change_threshold: float = 0.30
    max_align_size: int = 15000
    align_offset_threshold: float = 5.0


@dataclass
class ProcessingResult:
    change_mask: np.ndarray
    change_probability: np.ndarray
    objects: List[DetectedObject]
    processing_time: float = 0.0


class ComparisonMetrics:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def structural_similarity(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2
        mean1, std1 = gray1.mean(), gray1.std()
        mean2, std2 = gray2.mean(), gray2.std()
        if std1 < 1e-6 or std2 < 1e-6:
            return 0.0 if abs(mean1 - mean2) > 10 else 1.0
        cov = np.mean((gray1 - mean1) * (gray2 - mean2))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim_val = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
                   ((mean1**2 + mean2**2 + c1) * (std1**2 + std2**2 + c2))
        return 1.0 - ssim_val
    
    def histogram_correlation(self, img1, img2):
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist1, hist1); cv2.normalize(hist2, hist2)
        return 1.0 - max(0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    
    def texture_difference(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2
        k = 9
        var1 = cv2.GaussianBlur(gray1.astype(np.float32)**2, (k,k), 0) - cv2.GaussianBlur(gray1.astype(np.float32), (k,k), 0)**2
        var2 = cv2.GaussianBlur(gray2.astype(np.float32)**2, (k,k), 0) - cv2.GaussianBlur(gray2.astype(np.float32), (k,k), 0)**2
        return min(np.mean(np.abs(var1 - var2)) / 255.0, 1.0)
    
    def hsv_color_difference(self, img1, img2):
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
        diff_h = np.abs(hsv1[:,:,0].astype(np.float32) - hsv2[:,:,0].astype(np.float32))
        diff_h = np.minimum(diff_h, 180 - diff_h) / 180.0
        diff_s = np.abs(hsv1[:,:,1].astype(np.float32) - hsv2[:,:,1].astype(np.float32)) / 255.0
        return np.mean(diff_h) * 0.6 + np.mean(diff_s) * 0.4
    
    def orb_feature_difference(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.5
        try:
            matches = sorted(self.bf.match(des1, des2), key=lambda x: x.distance)
            good = [m for m in matches if m.distance < 50]
            return 1.0 - len(good) / max(len(kp1), len(kp2))
        except:
            return 0.5
    
    def combined_metric(self, img1, img2, weights=None):
        if weights is None:
            weights = {'ssim': 0.15, 'histogram': 0.10, 'texture': 0.20, 'hsv': 0.20, 'orb': 0.35}
        if img1.dtype != np.uint8:
            img1 = np.clip(img1, 0, 255).astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = np.clip(img2, 0, 255).astype(np.uint8)
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        m = {
            'ssim': self.structural_similarity(img1, img2),
            'histogram': self.histogram_correlation(img1, img2),
            'texture': self.texture_difference(img1, img2),
            'hsv': self.hsv_color_difference(img1, img2),
            'orb': self.orb_feature_difference(img1, img2),
        }
        return sum(m[k] * weights[k] for k in weights), m


class OrthoChangeEngine:
    
    def __init__(self, params=None, progress_callback=None):
        self.params = params or ProcessingParams()
        self.progress_callback = progress_callback
        self.img_old = None
        self.img_new = None
        self.profile = None
        self.pixel_size = None
        self.crs = None
        self.transform = None
        self.transform_old = None
        self.transform_new = None
        self.img_old_bounds = None
        self.img_new_bounds = None
        self.old_path = ""
        self.new_path = ""
        self.model = None
        self.metrics = ComparisonMetrics()
        self.is_aligned = False
        self._alignment_offset = (0, 0)
    
    def _report(self, pct, msg=""):
        if self.progress_callback:
            self.progress_callback(pct, msg)
    
    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO('yolov8m.pt')
            logger.info("YOLOv8m загружена")
    
    def _build_transform_from_gcps(self, gcps, width, height):
        try:
            return from_gcps(gcps)
        except:
            if len(gcps) >= 2:
                gcp0, gcp1 = gcps[0], gcps[-1]
                dx_px = gcp1.col - gcp0.col
                dy_px = gcp1.row - gcp0.row
                dx_geo = gcp1.x - gcp0.x
                dy_geo = gcp1.y - gcp0.y
                res_x = abs(dx_geo / dx_px) if dx_px != 0 else 1.0
                res_y = abs(dy_geo / dy_px) if dy_px != 0 else 1.0
                left = gcp0.x - gcp0.col * res_x
                top = gcp0.y + gcp0.row * res_y
                return Affine(res_x, 0, left, 0, -res_y, top)
        return None
    
    def load_images(self, old_path, new_path):
        self.old_path = old_path
        self.new_path = new_path
        
        logger.info(f"Загрузка: {old_path}")
        with rasterio.open(old_path) as src:
            self.img_old = src.read([1, 2, 3]).astype(np.float32)
            self.profile = src.profile
            self.pixel_size = src.res[0]
            self.transform_old = src.transform
            self.img_old_bounds = src.bounds
            
            if src.crs:
                self.crs = src.crs
            elif src.gcps and len(src.gcps) > 1:
                self.crs = src.gcps[1]
            
            if (self.transform_old is None or getattr(self.transform_old, 'is_identity', True)) and src.gcps:
                self.transform_old = self._build_transform_from_gcps(src.gcps[0], src.width, src.height)
                if self.transform_old:
                    self.img_old_bounds = rasterio.transform.array_bounds(src.height, src.width, self.transform_old)
                    logger.info("Transform старого снимка построен из GCP")
        
        logger.info(f"Загрузка: {new_path}")
        with rasterio.open(new_path) as src:
            self.img_new = src.read([1, 2, 3]).astype(np.float32)
            self.transform_new = src.transform
            self.img_new_bounds = src.bounds
            
            if not self.crs and src.gcps and len(src.gcps) > 1:
                self.crs = src.gcps[1]
            
            if (self.transform_new is None or getattr(self.transform_new, 'is_identity', True)) and src.gcps:
                self.transform_new = self._build_transform_from_gcps(src.gcps[0], src.width, src.height)
                if self.transform_new:
                    self.img_new_bounds = rasterio.transform.array_bounds(src.height, src.width, self.transform_new)
                    logger.info("Transform нового снимка построен из GCP")
            
            self.pixel_size = min(self.pixel_size, src.res[0])
    
    def extract_gcp(self):
        gcp_list = []
        for path in [self.old_path, self.new_path]:
            if not path:
                continue
            try:
                with rasterio.open(path) as src:
                    if src.gcps and src.gcps[0]:
                        for gcp in src.gcps[0]:
                            gcp_list.append({
                                'pixel': (gcp.col, gcp.row),
                                'geo': (gcp.x, gcp.y),
                                'info': gcp.info if gcp.info else '',
                                'file': os.path.basename(path)
                            })
            except:
                pass
        return gcp_list
    
    def align_images(self):
        """Привязка снимков по координатам из GCP"""
        has_old = self.transform_old is not None and not getattr(self.transform_old, 'is_identity', True)
        has_new = self.transform_new is not None and not getattr(self.transform_new, 'is_identity', True)
        
        if has_old and has_new:
            ox, oy = self.transform_old * (0, 0)
            nx, ny = self.transform_new * (0, 0)
            dx, dy = abs(nx - ox), abs(ny - oy)
            self._alignment_offset = (dx, dy)
            logger.info(f"Смещение: ΔX={dx:.2f}, ΔY={dy:.2f}")
            
            if (dx > self.params.align_offset_threshold or dy > self.params.align_offset_threshold) and self.crs:
                logger.info("Смещение выше порога — выполняю привязку...")
                self._reproject_to_common_grid()
                self.is_aligned = True
                return
        
        if self.img_old.shape != self.img_new.shape:
            h = min(self.img_old.shape[1], self.img_new.shape[1])
            w = min(self.img_old.shape[2], self.img_new.shape[2])
            self.img_old = self.img_old[:, :h, :w]
            self.img_new = self.img_new[:, :h, :w]
        
        self.transform = self.transform_old if has_old else (self.transform_new if has_new else None)
        self.is_aligned = True
    
    def _reproject_to_common_grid(self):
        left = min(self.img_old_bounds.left, self.img_new_bounds.left)
        bottom = min(self.img_old_bounds.bottom, self.img_new_bounds.bottom)
        right = max(self.img_old_bounds.right, self.img_new_bounds.right)
        top = max(self.img_old_bounds.top, self.img_new_bounds.top)
        
        resolution = self.pixel_size
        if resolution < 0.00001:
            resolution = 0.0001
        
        width = min(int((right - left) / resolution), self.params.max_align_size)
        height = min(int((top - bottom) / resolution), self.params.max_align_size)
        
        dst_transform = from_bounds(left, bottom, right, top, width, height)
        
        old_aligned = np.zeros((3, height, width), dtype=np.float32)
        new_aligned = np.zeros((3, height, width), dtype=np.float32)
        
        for i in range(3):
            reproject(self.img_old[i], old_aligned[i],
                     src_transform=self.transform_old, dst_transform=dst_transform,
                     src_crs=self.crs, dst_crs=self.crs, resampling=Resampling.bilinear)
            reproject(self.img_new[i], new_aligned[i],
                     src_transform=self.transform_new, dst_transform=dst_transform,
                     src_crs=self.crs, dst_crs=self.crs, resampling=Resampling.bilinear)
        
        self.img_old = old_aligned
        self.img_new = new_aligned
        self.transform = dst_transform
        self.pixel_size = resolution
        self.profile['transform'] = dst_transform
        self.profile['width'] = width
        self.profile['height'] = height
        logger.info(f"Перепроецировано: {width}x{height}")
    
    def pixel_to_geo(self, col, row):
        if self.transform is None:
            return (col, row)
        return self.transform * (col, row)
    
    def process(self):
        t0 = time.time()
        
        self._report(1, "Выравнивание...")
        self.align_images()
        
        self._report(3, "Загрузка модели...")
        self._load_model()
        
        self._report(10, "YOLO: снимок 1...")
        old_objects = self._detect_objects(self.img_old)
        logger.info(f"Объектов на снимке 1: {len(old_objects)}")
        
        self._report(40, "YOLO: снимок 2...")
        new_objects = self._detect_objects(self.img_new)
        logger.info(f"Объектов на снимке 2: {len(new_objects)}")
        
        self._report(65, "Сравнение...")
        appeared, disappeared = self._compare_objects(old_objects, new_objects)
        candidates = appeared + disappeared
        logger.info(f"Кандидатов: {len(candidates)}")
        
        self._report(80, "Метрики...")
        confirmed = []
        for obj in candidates:
            y1, x1, y2, x2 = obj['bbox']
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(self.img_old.shape[1], y2), min(self.img_old.shape[2], x2)
            if y2 <= y1 or x2 <= x1:
                continue
            
            p_old = self.img_old[:, y1:y2, x1:x2].transpose(1,2,0)
            p_new = self.img_new[:, y1:y2, x1:x2].transpose(1,2,0)
            if p_old.size == 0 or p_new.size == 0:
                continue
            
            combined, _ = self.metrics.combined_metric(
                np.clip(p_old, 0, 255).astype(np.uint8),
                np.clip(p_new, 0, 255).astype(np.uint8),
                weights={'ssim': self.params.weight_ssim, 'histogram': self.params.weight_histogram,
                        'texture': self.params.weight_texture, 'hsv': self.params.weight_hsv,
                        'orb': self.params.weight_orb}
            )
            if combined > self.params.change_threshold:
                confirmed.append(obj)
        
        logger.info(f"Подтверждено: {len(confirmed)}")
        
        self._report(95, "Формирование результата...")
        mask = np.zeros_like(self.img_old[0], dtype=bool)
        objects = []
        
        for i, obj in enumerate(confirmed):
            y1, x1, y2, x2 = obj['bbox']
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(mask.shape[0], y2), min(mask.shape[1], x2)
            mask[y1:y2, x1:x2] = True
            cx, cy = obj['cx'], obj['cy']
            gx, gy = self.pixel_to_geo(cx, cy)
            objects.append(DetectedObject(
                id=i+1,
                area_m2=round(obj['area_m2'], 2),
                perimeter_m=0,
                centroid_x=round(cx * self.pixel_size, 2),
                centroid_y=round(cy * self.pixel_size, 2),
                centroid_geo_x=round(gx, 2),
                centroid_geo_y=round(gy, 2),
                bbox=obj['bbox'],
                major_axis_m=0, minor_axis_m=0, compactness=0,
                class_name=obj.get('class_name', '')
            ))
        
        self._report(100, f"Готово: {len(objects)}")
        return ProcessingResult(change_mask=mask, change_probability=mask.astype(float),
                               objects=objects, processing_time=time.time()-t0)
    
    def _detect_objects(self, img):
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8).transpose(1,2,0)
        h, w = img_uint8.shape[:2]
        all_boxes, all_classes = [], []
        ts = self.params.tile_size
        overlap = ts // 3
        
        for y in range(0, h, ts - overlap):
            for x in range(0, w, ts - overlap):
                y2, x2 = min(y+ts, h), min(x+ts, w)
                tile = img_uint8[y:y2, x:x2]
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue
                results = self.model(tile, conf=self.params.confidence, iou=0.5, verbose=False)
                if results[0].boxes is not None:
                    for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(),
                                       results[0].boxes.cls.cpu().numpy()):
                        box[0]+=x; box[1]+=y; box[2]+=x; box[3]+=y
                        all_boxes.append(box); all_classes.append(cls)
        
        if not all_boxes:
            return []
        
        all_boxes = np.array(all_boxes); all_classes = np.array(all_classes)
        keep = self._nms(all_boxes, 0.4)
        all_boxes = all_boxes[keep]; all_classes = all_classes[keep]
        
        objects = []
        for box, cls in zip(all_boxes, all_classes):
            x1, y1, x2, y2 = box.astype(int)
            area_m2 = (x2-x1)*(y2-y1)*self.pixel_size**2
            if area_m2 < self.params.min_object_area_m2:
                continue
            
            objects.append({
                'bbox': (y1, x1, y2, x2),
                'cx': (x1+x2)/2, 'cy': (y1+y2)/2,
                'area_m2': area_m2,
                'class_id': int(cls),
                'class_name': self.model.names[int(cls)]
            })
        
        return objects
    
    def _nms(self, boxes, threshold=0.5):
        if len(boxes) == 0:
            return []
        x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        order = areas.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]; keep.append(i)
            if len(order) == 1: break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= threshold)[0] + 1]
        return keep
    
    def _compare_objects(self, old_objects, new_objects):
        appeared, disappeared = [], []
        for new_obj in new_objects:
            if not any(self._is_same_place(new_obj, old_obj) for old_obj in old_objects):
                appeared.append(new_obj)
        for old_obj in old_objects:
            if not any(self._is_same_place(old_obj, new_obj) for new_obj in new_objects):
                disappeared.append(old_obj)
        return appeared, disappeared
    
    def _is_same_place(self, a, b):
        y1_a,x1_a,y2_a,x2_a = a['bbox']
        y1_b,x1_b,y2_b,x2_b = b['bbox']
        inter = max(0, min(y2_a,y2_b)-max(y1_a,y1_b)) * max(0, min(x2_a,x2_b)-max(x1_a,x1_b))
        area_a = (y2_a-y1_a)*(x2_a-x1_a)
        area_b = (y2_b-y1_b)*(x2_b-x1_b)
        if (inter/area_a if area_a else 0) > self.params.overlap_threshold:
            return True
        if (inter/area_b if area_b else 0) > self.params.overlap_threshold:
            return True
        return np.sqrt((a['cx']-b['cx'])**2 + (a['cy']-b['cy'])**2) < self.params.distance_threshold*0.5
    
    def save_results(self, result, output_dir, prefix="result"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prof = self.profile.copy()
        prof.update(dtype=np.uint8, count=1)
        with rasterio.open(str(out/f"{prefix}_mask.tif"), 'w', **prof) as d:
            d.write(result.change_mask.astype(np.uint8)*255, 1)
        with open(str(out/f"{prefix}_report.json"), 'w', encoding='utf-8') as f:
            json.dump({'time_sec':round(result.processing_time,2), 'total_objects':len(result.objects),
                      'objects':[{'id':o.id,'area_m2':o.area_m2,'class':o.class_name,
                                 'geo':[o.centroid_geo_x,o.centroid_geo_y]} for o in result.objects]},
                      f, indent=2, ensure_ascii=False)
        return {'mask':str(out/f"{prefix}_mask.tif"), 'report':str(out/f"{prefix}_report.json")}