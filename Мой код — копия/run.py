# run.py
import sys
import numpy as np
import threading
from PySide6.QtWidgets import QApplication, QFileDialog
from engine import OrthoChangeEngine, ProcessingParams
from design import MainWindow


class App:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.design = MainWindow()
        self.engine = None
        self.result = None
        
        self.design.on_select_old = self.load_old
        self.design.on_select_new = self.load_new
        self.design.on_start = self.start
        self.design.on_save = self.save
    
    def load_old(self, path):
        import rasterio
        try:
            with rasterio.open(path) as src:
                img = src.read([1,2,3]).astype(np.float32).transpose(1,2,0) / 255.0
                self.design.set_image_size(src.width, src.height)
                gcp_count = len(src.gcps[0]) if src.gcps else 0
                self.design.set_georeference_old(
                    src.transform if not getattr(src.transform, 'is_identity', True) else None,
                    src.crs, src.res[0], gcp_count
                )
            self.design.set_old_image(img)
            self.design.set_status("Снимок 1 загружен")
        except Exception as ex:
            self.design.set_status(f"Ошибка: {ex}")
    
    def load_new(self, path):
        import rasterio
        try:
            with rasterio.open(path) as src:
                img = src.read([1,2,3]).astype(np.float32).transpose(1,2,0) / 255.0
                self.design.set_image_size(src.width, src.height)
                gcp_count = len(src.gcps[0]) if src.gcps else 0
                self.design.set_georeference_new(
                    src.transform if not getattr(src.transform, 'is_identity', True) else None,
                    src.crs, src.res[0], gcp_count
                )
            self.design.set_new_image(img)
            self.design.set_status("Снимок 2 загружен")
        except Exception as ex:
            self.design.set_status(f"Ошибка: {ex}")
    
    def start(self):
        self.design.set_processing(True)
        self.design.set_status("Запуск...")
        threading.Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        try:
            self.engine = OrthoChangeEngine(
                ProcessingParams(min_object_area_m2=3.0, distance_threshold=80.0,
                                overlap_threshold=0.3, confidence=0.12, tile_size=960,
                                change_threshold=0.30, align_offset_threshold=5.0),
                progress_callback=lambda p, t=None: self.design.set_progress(p)
            )
            self.engine.load_images(self.design.old_path, self.design.new_path)
            self.result = self.engine.process()
            
            gcp = self.engine.extract_gcp()
            self.design.set_gcp_info(len(gcp))
            
            self._done()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            self.design.set_processing(False)
    
    def _done(self):
        self.design.set_processing(False)
        r = self.result
        if r.objects:
            total = sum(o.area_m2 for o in r.objects)
            self.design.set_result_text(f"Найдено: {len(r.objects)} объектов, {total:.1f} м²")
        else:
            self.design.set_result_text("Изменений не найдено")
        self.design.set_progress(100)
        self.design.show_result(r.change_mask, len(r.objects))
    
    def save(self):
        d = QFileDialog.getExistingDirectory(self.design, "Папка для сохранения")
        if d and self.engine and self.result:
            self.engine.save_results(self.result, d)
            self.design.set_status(f"Сохранено в: {d}")
    
    def run(self):
        self.design.show()
        self.app.exec()


if __name__ == "__main__":
    App().run()