from ultralytics import YOLO

# Навчання YOLOv8
def train_model():
    # Завантажуємо базову модель YOLOv8n
    model = YOLO('yolov8n.pt')

    # Запускаємо навчання
    model.train(
        data='./datasets/data.yaml',  # Шлях до конфігураційного файлу
        epochs=50,         # Кількість епох
        imgsz=640,         # Розмір зображень
        batch=16,          # Розмір батчу
        name='military_detection',  # Ім'я проекту
        workers=4          # Кількість потоків для завантаження даних
    )

if __name__ == "__main__":
    train_model()
