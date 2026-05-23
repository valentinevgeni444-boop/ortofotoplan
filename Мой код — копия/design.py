# design.py
import sys
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QCheckBox,
    QFileDialog, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import (
    QPainter, QPixmap, QImage, QPen, QColor, QFont, QPolygonF
)
import rasterio
from skimage import measure


class CompassWidget(QWidget):
    """Компас в левом нижнем углу"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.angle = 0
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        cx, cy = self.width() / 2, self.height() / 2
        r = 20
        
        painter.setPen(QPen(QColor(255, 255, 255, 150), 1.5))
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.angle)
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FF3333"))
        north = QPolygonF([QPointF(0, -r+2), QPointF(-4, 0), QPointF(4, 0)])
        painter.drawPolygon(north)
        
        painter.setBrush(QColor(150, 150, 150))
        south = QPolygonF([QPointF(0, r-2), QPointF(-4, 0), QPointF(4, 0)])
        painter.drawPolygon(south)
        
        painter.restore()
        
        painter.setPen(QPen(QColor("white"), 1))
        font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QPointF(cx - 4, cy - r - 2), "N")


class CanvasWidget(QWidget):
    """Холст с быстрым зумом, перетаскиванием, шторкой, сеткой и компасом"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.pixmap1 = None
        self.pixmap2 = None
        self.result_mask = None
        self.result_contours = []
        
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.min_scale = 0.05
        self.max_scale = 50.0
        
        self.dragging = False
        self.last_mouse_pos = None
        
        self.swipe_mode = False
        self.swipe_position = 0.5
        self.swipe_dragging = False
        
        self.current_geo_x = 0.0
        self.current_geo_y = 0.0
        self.transform = None
        self.crs = None
        self.img_width = 0
        self.img_height = 0
        
        self.show_grid = False
        self.grid_step = 100
        
        self.compass = CompassWidget(self)
        self.compass.move(10, self.height() - 70)
        
        self._redraw_timer = QTimer()
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self.update)
    
    def set_image1(self, path):
        try:
            with rasterio.open(path) as src:
                data = src.read([1,2,3]).astype(np.float32).transpose(1,2,0)
                data = np.clip(data / 255.0, 0, 1)
                self.img_width = src.width
                self.img_height = src.height
                self.transform = src.transform
                self.crs = src.crs
            
            data = (data * 255).astype(np.uint8)
            h, w, _ = data.shape
            qimg = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
            self.pixmap1 = QPixmap.fromImage(qimg)
            self._fit_to_window()
            self.update()
        except Exception as e:
            print(f"Ошибка загрузки снимка 1: {e}")
    
    def set_image2(self, path):
        try:
            with rasterio.open(path) as src:
                data = src.read([1,2,3]).astype(np.float32).transpose(1,2,0)
                data = np.clip(data / 255.0, 0, 1)
                self.img_width = max(self.img_width, src.width)
                self.img_height = max(self.img_height, src.height)
                if self.transform is None:
                    self.transform = src.transform
                    self.crs = src.crs
            
            data = (data * 255).astype(np.uint8)
            h, w, _ = data.shape
            qimg = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
            self.pixmap2 = QPixmap.fromImage(qimg)
            self._fit_to_window()
            self.update()
        except Exception as e:
            print(f"Ошибка загрузки снимка 2: {e}")
    
    def set_result(self, mask):
        self.result_mask = mask
        if mask is not None and np.any(mask):
            self.result_contours = measure.find_contours(mask.astype(float), 0.5)
        else:
            self.result_contours = []
        self.update()
    
    def set_swipe_mode(self, enabled):
        self.swipe_mode = enabled
        self.update()
    
    def set_grid(self, show, step=100):
        self.show_grid = show
        self.grid_step = step
        self.update()
    
    def _fit_to_window(self):
        if self.img_width == 0 or self.img_height == 0:
            return
        w = self.width()
        h = self.height()
        self.scale = min(w / self.img_width, h / self.img_height) * 0.9
        self.offset_x = (w - self.img_width * self.scale) / 2
        self.offset_y = (h - self.img_height * self.scale) / 2
    
    def _img_to_widget(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y
    
    def _widget_to_img(self, x, y):
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale
    
    def zoom(self, factor, cx=None, cy=None):
        if cx is None: cx = self.width() / 2
        if cy is None: cy = self.height() / 2
        img_x, img_y = self._widget_to_img(cx, cy)
        self.scale *= factor
        self.scale = max(self.min_scale, min(self.scale, self.max_scale))
        self.offset_x = cx - img_x * self.scale
        self.offset_y = cy - img_y * self.scale
        self.update()
    
    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.zoom(factor, event.position().x(), event.position().y())
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.swipe_mode:
                self.swipe_dragging = True
                self._update_swipe(event.position())
            else:
                self.dragging = True
                self.last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._fit_to_window()
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.swipe_dragging:
            self._update_swipe(event.position())
        elif self.dragging and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.position()
            self.update()
        
        img_x, img_y = self._widget_to_img(event.position().x(), event.position().y())
        if self.transform and 0 <= img_x < self.img_width and 0 <= img_y < self.img_height:
            self.current_geo_x, self.current_geo_y = self.transform * (img_x, img_y)
    
    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.swipe_dragging = False
        self.last_mouse_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def _update_swipe(self, pos):
        rect = QRectF(self.offset_x, self.offset_y,
                     self.img_width * self.scale, self.img_height * self.scale)
        self.swipe_position = max(0, min(1, (pos.x() - rect.x()) / rect.width()))
        self.update()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.compass.move(10, self.height() - 70)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        
        if self.pixmap1 is None and self.pixmap2 is None:
            painter.setPen(QColor("#888888"))
            font = QFont("Segoe UI", 14)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                           "Загрузите ортофотопланы\nи нажмите «Сравнить»")
            return
        
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.scale, self.scale)
        
        if self.swipe_mode and self.pixmap1 and self.pixmap2:
            split = int(self.img_width * self.swipe_position)
            painter.save()
            painter.setClipRect(0, 0, split, self.img_height)
            painter.drawPixmap(0, 0, self.pixmap1)
            painter.restore()
            painter.save()
            painter.setClipRect(split, 0, self.img_width - split, self.img_height)
            painter.drawPixmap(0, 0, self.pixmap2)
            painter.restore()
            painter.setPen(QPen(QColor("#00d4ff"), 2))
            painter.drawLine(split, 0, split, self.img_height)
        else:
            if self.pixmap2:
                painter.drawPixmap(0, 0, self.pixmap2)
            elif self.pixmap1:
                painter.drawPixmap(0, 0, self.pixmap1)
            
            if self.result_contours:
                painter.setPen(QPen(QColor("#FF3333"), 2))
                for contour in self.result_contours:
                    if len(contour) > 2:
                        for i in range(len(contour) - 1):
                            painter.drawLine(
                                QPointF(contour[i][1], contour[i][0]),
                                QPointF(contour[i+1][1], contour[i+1][0])
                            )
        
        if self.show_grid and self.transform:
            self._draw_coord_grid(painter)
        
        painter.restore()
    
    def _draw_coord_grid(self, painter):
        view_left, view_top = self._widget_to_img(0, 0)
        view_right, view_bottom = self._widget_to_img(self.width(), self.height())
        view_width = view_right - view_left
        
        if view_width > 5000: px_step = 1000
        elif view_width > 2000: px_step = 500
        elif view_width > 500: px_step = 100
        else: px_step = 50
        
        start_x = int(view_left // px_step) * px_step
        for x in range(start_x, int(view_right) + px_step, px_step):
            if x < 0 or x > self.img_width: continue
            painter.setPen(QPen(QColor(0, 229, 255, 40), 0.5))
            painter.drawLine(x, int(view_top), x, int(view_bottom))
            if self.transform:
                geo_x, _ = self.transform * (x, self.img_height // 2)
                painter.setPen(QPen(QColor(0, 229, 255, 120), 1))
                painter.setFont(QFont("Consolas", 7))
                painter.drawText(QPointF(x + 3, 12), f"{geo_x:.0f}")
        
        start_y = int(view_top // px_step) * px_step
        for y in range(start_y, int(view_bottom) + px_step, px_step):
            if y < 0 or y > self.img_height: continue
            painter.setPen(QPen(QColor(0, 229, 255, 40), 0.5))
            painter.drawLine(int(view_left), y, int(view_right), y)
            if self.transform:
                _, geo_y = self.transform * (self.img_width // 2, y)
                painter.setPen(QPen(QColor(0, 229, 255, 120), 1))
                painter.setFont(QFont("Consolas", 7))
                painter.drawText(QPointF(3, y - 3), f"{geo_y:.0f}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OrthoDetect Pro — Обнаружение изменений")
        self.resize(1400, 850)
        self.setStyleSheet("background-color: #1a1a1a;")
        
        self.old_path = ""
        self.new_path = ""
        self._old_transform = None
        
        self.on_select_old = None
        self.on_select_new = None
        self.on_start = None
        self.on_save = None
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self._create_top_panel(layout)
        self.canvas = CanvasWidget()
        layout.addWidget(self.canvas, 1)
        self._create_bottom_panel(layout)
        
        self._coord_timer = QTimer()
        self._coord_timer.timeout.connect(self._update_coords)
        self._coord_timer.start(100)
    
    def _create_top_panel(self, layout):
        panel = QFrame()
        panel.setFixedHeight(50)
        panel.setStyleSheet("background-color: #252525; border-bottom: 1px solid #333;")
        
        hbox = QHBoxLayout(panel)
        hbox.setContentsMargins(15, 8, 15, 8)
        hbox.setSpacing(10)
        
        title = QLabel("OrthoDetect Pro")
        title.setStyleSheet("color: #00d4ff; font-size: 14px; font-weight: bold;")
        hbox.addWidget(title)
        
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #444;")
        hbox.addWidget(sep)
        
        self.btn_old = QPushButton("📷 Снимок 1")
        self.btn_old.setStyleSheet(self._btn_style())
        self.btn_old.clicked.connect(self._on_old)
        hbox.addWidget(self.btn_old)
        
        self.lbl_old = QLabel("—")
        self.lbl_old.setStyleSheet("color: #888; font-size: 11px;")
        hbox.addWidget(self.lbl_old)
        
        self.btn_new = QPushButton("📷 Снимок 2")
        self.btn_new.setStyleSheet(self._btn_style())
        self.btn_new.clicked.connect(self._on_new)
        hbox.addWidget(self.btn_new)
        
        self.lbl_new = QLabel("—")
        self.lbl_new.setStyleSheet("color: #888; font-size: 11px;")
        hbox.addWidget(self.lbl_new)
        
        hbox.addStretch()
        
        self.progress = QProgressBar()
        self.progress.setFixedWidth(150)
        self.progress.setFixedHeight(20)
        self.progress.setStyleSheet("""
            QProgressBar { background-color: #333; border: 1px solid #444;
                border-radius: 3px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 2px; }
        """)
        self.progress.setVisible(False)
        hbox.addWidget(self.progress)
        
        self.lbl_percent = QLabel("")
        self.lbl_percent.setStyleSheet("color: #00d4ff; font-size: 12px; font-weight: bold;")
        hbox.addWidget(self.lbl_percent)
        
        self.btn_swipe = QPushButton("↔ Шторка")
        self.btn_swipe.setStyleSheet(self._btn_style("#1e3a5f"))
        self.btn_swipe.setCheckable(True)
        self.btn_swipe.clicked.connect(self._on_swipe)
        hbox.addWidget(self.btn_swipe)
        
        self.chk_grid = QCheckBox("Сетка")
        self.chk_grid.setStyleSheet("color: #888;")
        self.chk_grid.toggled.connect(lambda v: self.canvas.set_grid(v))
        hbox.addWidget(self.chk_grid)
        
        self.btn_start = QPushButton("▶ СРАВНИТЬ")
        self.btn_start.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; border: none;
                border-radius: 4px; padding: 8px 20px; font-weight: bold; font-size: 13px; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #666; }
        """)
        self.btn_start.clicked.connect(self._on_start)
        hbox.addWidget(self.btn_start)
        
        self.btn_save = QPushButton("💾")
        self.btn_save.setStyleSheet(self._btn_style())
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        hbox.addWidget(self.btn_save)
        
        self.lbl_result = QLabel("")
        self.lbl_result.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
        hbox.addWidget(self.lbl_result)
        
        layout.addWidget(panel)
    
    def _create_bottom_panel(self, layout):
        panel = QFrame()
        panel.setFixedHeight(50)
        panel.setStyleSheet("background-color: #252525; border-top: 1px solid #333;")
        
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(15, 3, 15, 3)
        vbox.setSpacing(2)
        
        geo_row = QHBoxLayout()
        
        self.lbl_geo_status = QLabel("🔴 Без привязки")
        self.lbl_geo_status.setStyleSheet("color: #F44336; font-size: 11px; font-weight: bold;")
        geo_row.addWidget(self.lbl_geo_status)
        
        self.lbl_crs = QLabel("")
        self.lbl_crs.setStyleSheet("color: #888; font-size: 10px;")
        geo_row.addWidget(self.lbl_crs)
        
        geo_row.addStretch()
        
        self.lbl_offset = QLabel("")
        self.lbl_offset.setStyleSheet("color: #888; font-size: 10px;")
        geo_row.addWidget(self.lbl_offset)
        
        self.lbl_gcp = QLabel("")
        self.lbl_gcp.setStyleSheet("color: #888; font-size: 10px;")
        geo_row.addWidget(self.lbl_gcp)
        
        vbox.addLayout(geo_row)
        
        status_row = QHBoxLayout()
        
        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        status_row.addWidget(self.status_label)
        
        status_row.addStretch()
        
        self.coord_label = QLabel("X: —  Y: —")
        self.coord_label.setStyleSheet("color: #00E5FF; font-family: Consolas; font-size: 11px;")
        status_row.addWidget(self.coord_label)
        
        vbox.addLayout(status_row)
        
        layout.addWidget(panel)
    
    def _btn_style(self, bg="#333"):
        return f"""
            QPushButton {{ background-color: {bg}; color: #E0E0E0; border: none;
                border-radius: 4px; padding: 6px 12px; font-size: 11px; }}
            QPushButton:hover {{ background-color: #444; }}
        """
    
    def _update_coords(self):
        x, y = self.canvas.current_geo_x, self.canvas.current_geo_y
        self.coord_label.setText(f"X: {x:.2f}  Y: {y:.2f}")
    
    def _on_old(self):
        f, _ = QFileDialog.getOpenFileName(self, "Снимок 1", "", "GeoTIFF (*.tif *.tiff);;Все (*.*)")
        if f:
            self.old_path = f
            self.lbl_old.setText(os.path.basename(f))
            self.canvas.set_image1(f)
            if self.on_select_old:
                self.on_select_old(f)
    
    def _on_new(self):
        f, _ = QFileDialog.getOpenFileName(self, "Снимок 2", "", "GeoTIFF (*.tif *.tiff);;Все (*.*)")
        if f:
            self.new_path = f
            self.lbl_new.setText(os.path.basename(f))
            self.canvas.set_image2(f)
            if self.on_select_new:
                self.on_select_new(f)
    
    def _on_swipe(self, checked):
        self.canvas.set_swipe_mode(checked)
    
    def _on_start(self):
        if not self.old_path or not self.new_path:
            QMessageBox.warning(self, "Предупреждение", "Загрузите оба снимка!")
            return
        if self.on_start:
            self.on_start()
    
    def _on_save(self):
        if self.on_save:
            self.on_save()
    
    def set_processing(self, active):
        self.btn_start.setEnabled(not active)
        self.btn_save.setEnabled(False)
        self.progress.setVisible(active)
        if active:
            self.btn_start.setText("⏳ ОБРАБОТКА...")
            self.progress.setValue(0)
            self.lbl_percent.setText("0%")
        else:
            self.btn_start.setText("▶ СРАВНИТЬ")
    
    def set_progress(self, pct):
        self.progress.setValue(pct)
        self.lbl_percent.setText(f"{pct}%")
    
    def set_status(self, text):
        self.status_label.setText(text)
    
    def set_result_text(self, text):
        self.lbl_result.setText(text)
    
    def show_result(self, mask, count):
        self.btn_save.setEnabled(True)
        self.canvas.set_result(mask)
        self.set_status(f"Готово. Объектов: {count}")
    
    def set_image_size(self, w, h):
        self.canvas.img_width = w
        self.canvas.img_height = h
    
    def set_georeference_old(self, transform, crs, pixel_size):
        print(f"DEBUG: set_georeference_old | crs={crs}")
        self.canvas.transform = transform
        self.canvas.crs = crs
        self._old_transform = transform
        self._update_georef(crs)
    
    def set_georeference_new(self, transform, crs, pixel_size):
        print(f"DEBUG: set_georeference_new | crs={crs}")
        self.canvas.transform = transform
        self.canvas.crs = crs
        if self._old_transform:
            ox, oy = self._old_transform * (0, 0)
            nx, ny = transform * (0, 0)
            dx, dy = abs(nx - ox), abs(ny - oy)
            if dx > 0.01 or dy > 0.01:
                self.lbl_offset.setText(f"ΔX={dx:.2f} ΔY={dy:.2f} м")
                self.lbl_offset.setStyleSheet("color: #FF9800; font-size: 10px;")
            else:
                self.lbl_offset.setText("Смещение: < 0.01 м")
                self.lbl_offset.setStyleSheet("color: #4CAF50; font-size: 10px;")
        self._update_georef(crs)
    
    def _update_georef(self, crs):
        print(f"DEBUG: _update_georef | crs={crs}")
        if crs:
            epsg = crs.to_epsg() if crs.to_epsg() else None
            self.lbl_geo_status.setText("🟢 С привязкой")
            self.lbl_geo_status.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            self.lbl_crs.setText(f"EPSG:{epsg}" if epsg else "Проекция")
        else:
            self.lbl_geo_status.setText("🔴 Без привязки")
            self.lbl_geo_status.setStyleSheet("color: #F44336; font-size: 11px; font-weight: bold;")