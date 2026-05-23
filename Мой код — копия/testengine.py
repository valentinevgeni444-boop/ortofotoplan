# test_engine.py — проверочный модуль, не трогает основную программу
import time
import numpy as np
import rasterio

print("=" * 50)
print("ТЕСТ ДВИЖКА")
print("=" * 50)

# Укажите пути к вашим файлам
OLD = r"C:\Users\Owner\Desktop\Новая папка\проект 1\ortho__2.tif"
NEW = r"C:\Users\Owner\Desktop\Новая папка\проект 2\ortho__1.tif"

print("\n1. Загрузка старого снимка...")
t = time.time()
with rasterio.open(OLD) as src:
    img_old = src.read([1, 2, 3]).astype(np.float32)
    print(f"   Размер: {img_old.shape[1]}x{img_old.shape[2]}, время: {time.time()-t:.2f} сек")

print("\n2. Загрузка нового снимка...")
t = time.time()
with rasterio.open(NEW) as src:
    img_new = src.read([1, 2, 3]).astype(np.float32)
    print(f"   Размер: {img_new.shape[1]}x{img_new.shape[2]}, время: {time.time()-t:.2f} сек")

print("\n3. Нормализация...")
t = time.time()
old = img_old / 255.0
new = img_new / 255.0
print(f"   Время: {time.time()-t:.2f} сек")

print("\n4. Разница...")
t = time.time()
diff = np.abs(new - old)
print(f"   Время: {time.time()-t:.2f} сек")

print("\n5. Усреднение по каналам...")
t = time.time()
prob = np.mean(diff, axis=0)
print(f"   Время: {time.time()-t:.2f} сек")

print("\n6. Нормализация prob...")
t = time.time()
mn, mx = prob.min(), prob.max()
prob = (prob - mn) / (mx - mn)
print(f"   Время: {time.time()-t:.2f} сек")

print("\n7. Порог (95-й перцентиль)...")
t = time.time()
thresh = np.percentile(prob, 95)
mask = prob > max(thresh, 0.05)
print(f"   Порог: {thresh:.4f}, пикселей: {np.sum(mask)}, время: {time.time()-t:.2f} сек")

print("\n8. Заливка дыр...")
t = time.time()
from scipy.ndimage import binary_fill_holes
mask = binary_fill_holes(mask)
print(f"   Время: {time.time()-t:.2f} сек")

print("\n9. Разметка объектов...")
t = time.time()
from scipy import ndimage
labeled, n = ndimage.label(mask)
print(f"   Регионов: {n}, время: {time.time()-t:.2f} сек")

print("\n10. Извлечение свойств...")
t = time.time()
from skimage import measure
regions = measure.regionprops(labeled)
print(f"   Время: {time.time()-t:.2f} сек")

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТ")
print("=" * 50)
print(f"Всего регионов: {n}")
print(f"Объектов > 5 м²: {sum(1 for r in regions if r.area * 0.25 >= 5)}")

if regions:
    print("\nТоп-5 крупнейших:")
    for r in sorted(regions, key=lambda x: x.area, reverse=True)[:5]:
        print(f"  Площадь: {r.area} пикс, Центр: ({r.centroid[1]:.0f}, {r.centroid[0]:.0f})")

print("\nГОТОВО")