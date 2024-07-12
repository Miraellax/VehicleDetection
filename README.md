# Технологии
- Python
- PyQT (интерфейс)
- PyTorch, TorchVision
- Numpy
- Ultralytics (YOLO)

# Описание  
  Приложение разрабатывалось для детекции спецтранспорта и подачи визуального и аудио сигнала оператору камер наблюдения и ограждений территории. 
  
  Приложение позволяет захватывать часть экрана и с помощью моделей YOLO обнаруживать присутствие спецтранспорта - пожарные машины, скорая помощь, реанимация, службы правопорядка + обычный транспорт.
  ![изображение](https://github.com/user-attachments/assets/f3052b09-8515-45ba-8f8a-e7bd8400b74b)

  Для отделения работы интерфейса от работы модели детекции используются потоки. Модель перемещается в новый [поток](https://github.com/Miraellax/VehicleDetection/blob/dc3ca44e8a466daa769092dd8b8947abea77f9c3/Main/MainWindow.py#L335) (QThread), получает данные из основного потока и передает обратно обработанные данные для отображения. Для работы модели создан класс [worker'а](https://github.com/Miraellax/VehicleDetection/blob/dc3ca44e8a466daa769092dd8b8947abea77f9c3/Main/MainWindow.py#L382), ответственный за вызов функций [модели](https://github.com/Miraellax/VehicleDetection/blob/dc3ca44e8a466daa769092dd8b8947abea77f9c3/Main/ECModel.py#L26).


# Подготовка данных
Данные для обучения были собраны вручную и размечены с помощью сервиса **Roboflow**. К изображениям применены аугментации **TorchVision** для расширения обучающей выборки.

Размеченные и подготовленные данные хранятся в папках Emergency Vehicles Russia.v3i.yolov8 / .voc.

# Обучение моделей
  Для работы приложения обучены несколько моделей **YOLO** разной версии и размера - YOLO v5, v8.

  Обучение происходило в [файле](https://github.com/Miraellax/VehicleDetection/blob/dc3ca44e8a466daa769092dd8b8947abea77f9c3/Model_training_final.ipynb) формата **Jupyter Notebook**. Использовался сервис **Wandb** для отслеживания процесса и качества обучения моделей.

  Сравнение качества и скорости работы обученных моделей:
  ![изображение](https://github.com/user-attachments/assets/e736ebaa-c70b-4082-a986-72d6eebb6bc0)
