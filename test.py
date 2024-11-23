from ultralytics import YOLO
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr

# Завантажуємо зображення
image = load_image_bgr("./a29a3c83c61be9ee5ea03525e4c2ac17_jpeg.rf.94995cbecb4f6b16651a766670a52132.jpg")

# Завантажуємо натреновану модель
model_path = "./runs/detect/military_detection4/weights/best.pt"  # Шлях до вашої моделі
model = YOLO(model_path)

# Виконуємо інференс
results = model.predict(image, save=False)

# Перевіряємо структуру результатів
predictions = results[0].boxes  # Отримуємо координати боксів і класи

# Перетворюємо в формат Supervision
detections = sv.Detections(
    xyxy=predictions.xyxy.cpu().numpy(),  # Координати меж (bounding boxes)
    confidence=predictions.conf.cpu().numpy(),  # Ймовірності
    class_id=predictions.cls.cpu().numpy().astype(int)  # Класи об'єктів
)

# Анотуємо результати
annotator = sv.BoxAnnotator(thickness=4)
annotated_image = annotator.annotate(image, detections)
label_annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
annotated_image = label_annotator.annotate(annotated_image, detections)

# Відображаємо зображення з анотаціями
sv.plot_image(annotated_image)
