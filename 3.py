import numpy as np
import pandas as pd
import os
import glob

#from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt



# Функция для создания изображений графиков
def Plot(file_path):
    df = pd.read_excel(file_path, engine='openpyxl', skiprows=1)
    # Проверяем, есть ли столбцы 'Lх_мм' и 'Нагр_кг'
    if 'Lх_мм' in df.columns and 'Нагр_кг' in df.columns:
        # Проверяем, есть ли данные в таблице
        if not df.empty:
            # Проверяем, есть ли значения в столбцах 'Lх_мм' и 'Нагр_кг'
            plt.figure()
            # Строим точечный график
            plt.scatter(df['Lх_мм'], df['Нагр_кг'])
            plt.xlabel('Lх_мм')
            plt.ylabel('Нагр_кг')
            #plt.title('Точечный график')
            
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.savefig(f'{file_path}.png', dpi=300, bbox_inches='tight', pad_inches=0)
            #plt.show()

            return
    print('Ошибка: неверный формат файла')

#Отображает в окне график
def PlotShow(file_path):
    df = pd.read_excel(file_path, engine='openpyxl', skiprows=1)
    # Проверяем, есть ли столбцы 'Lх_мм' и 'Нагр_кг'
    if 'Lх_мм' in df.columns and 'Нагр_кг' in df.columns:
        # Проверяем, есть ли данные в таблице
        if not df.empty:
            # Проверяем, есть ли значения в столбцах 'Lх_мм' и 'Нагр_кг'
            plt.figure()
            # Строим точечный график
            plt.scatter(df['Lх_мм'], df['Нагр_кг'])
            plt.xlabel('Lх_мм')
            plt.ylabel('Нагр_кг')
            plt.title('Точечный график')
            plt.show()

            return
    print('Ошибка: неверный формат файла')

def get_filenames():
    return glob.glob('*.XLSX')
    

def convertToImg():
    for item in get_filenames():
        Plot(item)


# Путь к папке с изображениями
path_good = '.\\good'
path_bad = '.\\bad'

# Создаем список файлов в папке
files_good = os.listdir(path_good)
files_bad = os.listdir(path_bad)

# Создаем список меток для каждого изображения
labels_good = [0] * len(files_good)
labels_bad = [1] * len(files_bad)
labels = labels_good + labels_bad

# Создаем список изображений
images = []
for file in files_good:
    img = tf.keras.preprocessing.image.load_img(os.path.join(path_good, file), target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    images.append(img_array)
for file in files_bad:
    img = tf.keras.preprocessing.image.load_img(os.path.join(path_bad, file), target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    images.append(img_array)

# Преобразуем списки в формат, который может использовать модель
images = tf.stack(images)
labels = tf.keras.utils.to_categorical(labels)

# Создаем модель
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
model.trainable = False

# Добавляем слои
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x)

# Создаем модель
model = tf.keras.models.Model(inputs=model.input, outputs=output)

# Компилируем модель
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
model.fit(images, labels, epochs=10, batch_size=32)

images = []
files = glob.glob('*.png')

for file in files:
    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    images.append(img_array)
    
predicted_result = model.predict(tf.stack(images))

'''
df_data = []

for index in range(len(files)):
    df_data.extend([{"Название файла": files[index], 'Состояние:': 'Исправен' if predicted_result[index][0] > 0.5 else 'Неправильный', 'Точность': predicted_result[index][0]}])
    print(files[index], ':', 'Исправен' if predicted_result[index][0] > 0.5 else 'Неправильный', ':', predicted_result[index][0])

pd.DataFrame(df_data).to_excel('result.xlsx', index=False)
'''


# Создаем DataFrame
df_data = []
for index in range(len(files)):
    df_data.extend([{"Название файла": files[index], 'Состояние:': 'Исправен' if predicted_result[index][0] > 0.5 else 'Неправильный', 'Точность': predicted_result[index][0]}])
    print(files[index], ':', 'Исправен' if predicted_result[index][0] > 0.5 else 'Неправильный', ':', predicted_result[index][0])
df = pd.DataFrame(df_data)

# Создаем новый столбец с гиперссылками
def create_hyperlink(file_name):
    path = f"{file_name}"
    return f'=HYPERLINK("{path}","{file_name}")'

df['Название файла (ссылка)'] = df['Название файла'].apply(create_hyperlink)

# Удаляем старый столбец
df.drop(columns=['Название файла'], inplace=True)

# Сохраняем DataFrame в Excel файл
df.to_excel('result.xlsx', index=False)

#Показывает график файла. Нужно написать название файла без расширения    
while True :
    Name = input("Введите название файла: ")
    FileName = Name + ".xlsx"
    PlotShow(FileName)
   
