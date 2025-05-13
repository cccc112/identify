import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# 設置數據路徑
data_dir = "C:/hand/augmented_images/augmented_images"

# 創建數據生成器
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 創建訓練和驗證數據生成器
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),  # 增加圖片尺寸
    color_mode='rgb',
    class_mode='sparse',
    batch_size=32,
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    color_mode='rgb',
    class_mode='sparse',
    batch_size=32,
    subset='validation',
    shuffle=False
)

# 計算類別權重
class_weights = compute_class_weight('balanced', classes=np.unique(train_data.classes), y=train_data.classes)
class_weights = dict(enumerate(class_weights))

# 創建增強版模型
model = Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),  # 增加Dropout比例
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

# 顯示模型摘要
model.summary()

# 編譯模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 定義回調函數
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# 訓練模型
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stop]
)

# 繪製訓練過程圖
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 保存模型
model.save("C:/hand/augmented_model.h5")

# 輸出最終的訓練和驗證准確率
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

# 在完整的驗證集上評估模型
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy on full set: {val_accuracy:.4f}")
