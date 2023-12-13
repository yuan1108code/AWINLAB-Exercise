from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# 加載數據集
df = pd.read_csv("dogs.csv")
print('資料的形式:', df.shape)
print('遺漏的數量:', df.isnull().sum().sum())
df.head()

# 建立 CNN 模型
num_classes = 15  # 根據您的類別數量調整這個值
model = Sequential()


# Convolutional layers
# activation = ‘relu’：利用此激活函數，處理非線性、稀疏、避免梯度消失等問題，且其計算效率非常簡單，因此效率好
# activation = ‘Softmax’：利用此激活函數，讓輸出概率分佈且輸出所有元素之和為1，而且有平滑性，讓反向傳播計算簡單
# kernel_initializer = ‘Adam’：可以自動調整學習率，而且有良好收斂性，在此算法中適合處理大規模問題，不需要手動調整學習率
# kernel_initializer = ‘uniform’：令權重均勻分佈，而且簡單且快速具有隨機性，通常適用於淺層網路中

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Dense layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# 編譯模型
# loss=’sparse_categorical_crossentropy’：適用於多分類問題，而且進行one-hot編碼，能夠處理不平衡問題，適用單標籤分類
# optimizer = ‘Adam’：能夠自動調整學習，具有低內存需求，能支援稀疏梯度，適用於大多數問題，具有動量項可以加速模型收斂
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam', 
              metrics=['accuracy'])
# 訓練模型
# epochs = 200：從觀察中發現，用大規模的模擬，可以提高學習能量，能夠處理複雜的問題和大數據集
# batch_size = 30：樣本中有720筆數據，因此我拆分成 720 / 30次來訓練
# validation_split = 0.2：訓練集佔有80%，測試集佔其中20%
# verbose = 2：詳細模式，來輸出更詳細的訊息
# train_history = model.fit(X_train, y_train, epochs=200, 
#                           batch_size=30, validation_split = 0.2, verbose=2)

model.summary()


