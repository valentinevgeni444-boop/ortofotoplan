# comparison.py
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OrthoComparator:
    """Сравнение методом разницы пикселей."""
    
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
    
    def _report(self, pct, msg=""):
        if self.progress_callback:
            self.progress_callback(pct, msg)
    
    def compare(self, old, new):
        """
        Простая разница пикселей.
        Возвращает: probability_map, binary_mask
        """
        self._report(20, "Вычисление разницы...")
        
        # Разница по каждому каналу
        diff = np.abs(new - old)
        
        # Усреднение по каналам
        prob = np.mean(diff, axis=0)
        
        self._report(50, "Нормализация...")
        
        # Нормализация в [0, 1]
        mn, mx = prob.min(), prob.max()
        if mx - mn > 1e-8:
            prob = (prob - mn) / (mx - mn)
        
        self._report(70, "Бинаризация...")
        
        # Порог: всё что выше 95-го перцентиля
        thresh = np.percentile(prob, 95)
        mask = prob > max(thresh, 0.05)
        
        self._report(85, "Готово")
        
        return prob, mask