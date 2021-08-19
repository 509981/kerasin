# kerasin
Kerasin это рабочее название проекта оптимизации нейронных сетей написанных на фреймворке keras. 

Подбор сети происходит с помощью генетического алгоритма. Для этого, kerasin умеет генерировать сети самостоятельно. Топология сети случайным образом может образовывать пробросы и ветвления. С начала образуется каркас - ориентированный граф с одним входом и одним выходом. Он гарантировано не имеет циклов. Затем на этот скелет наращивается "мясо" в виде случайного набора слоев. Параметры слоев также меняются случайно, варьируясь в пределах своих допусков.

Как в приличном генетическом алгорите, каждый бот(keras-модель) описывается своим хромосомным набором. Будем называть его геномом. Вот фрагмент такого генома:
```
1.0 MaxPooling1D: name=max_pooling1d_6
1.0 MaxPooling1D: trainable=True
1.0 MaxPooling1D: dtype=float32
1.0 MaxPooling1D: strides=(1,)
1.0 MaxPooling1D: pool_size=(2,)
1.0 MaxPooling1D: inbound_layers=[0]
1.1 Dropout: name=dropout_23
1.1 Dropout: trainable=True
1.1 Dropout: dtype=float32
1.1 Dropout: rate=0.2891937916654299
2.0 Concatenate: name=concatenate_40
2.0 Concatenate: axis=-1
2.0 Concatenate: inbound_layers=[0, 1]
```
Мы можем как считать генотип с керас модели(секвенирование), так и наоборот - синтезировать по геному модель.

Класс поддерживает кроссовер из фрагментов сети своих родителей. Геном родителей режется на две части и одна из них формирует геном потомка. 
По геному можно провести мутацию. Мутации могут касаться как простой смены параметров слоя так и смены типа слоя или их соединения.

Для предотвращения близкородственного скрещивания и вырождения, введено понятие фамилии бота. В кросовере участвуют только боты с разной фамилией. Новая фамилия присваивается только случайно сгенерированым сетям. Она сохраняется при мутации. При кросовере бот берет фамилию ~~жены~~,ой, ...сети с самой лучшей оценкой.

kerasin называет бота натурально по имени, фамилии и эпохе рождения: **bot_EE.NNN(FFF)**

где: EE - эпоха рождения; NNN - порядковый номер рождения в эпохе; FFF - фамилия. Например: bot_02.008(013) 

---

### МЕТОДЫ КЛАССА

#### Конструктор класса
**kerasin(complexity = 1, nPopul=10, maxi_goal=False)**
   
  - *complexity* - сложность сети 1 - на 5 слоев, 2-на 10 и т.п. Количество слоев при генерации ботов точно не соблюдается.
  - *nPopul* - размер популяции
  - *maxi_goal* - целевая функция метрики. True,- если максимизация(accuracy), False - минимизация(loss)

Конструкция создает экземпляр класса 
  
---
#### Подмешивание в популяцию внешних моделей
**add_model(self,model,name='')**
 - *model* - керас модель
 - *name* - имя бота

Для ускорения сходимости или оптимизации конкретной модели, описываем в Sequential или функциональной форме keras и добавляем ее в популяцию. Следите чтобы имя бота было уникальным.

---
#### Генерация случайных ботов.  
**generate(nPopul=1,epoch=0)**
  - *nPopul* - количество ботов для генерации
  - *epoch* - Текущая эпоха. Задание параметра epoch важно, для правильного именования ботов.

Генерация nPopul случайных ботов.  

---
#### Описание входа моделей. 
**add_input(shape,isSequence = False)**
  - *shape* - форма входа(Без batch размерности) для mnist это может быть (28,28) или (784,)
  - *isSequence* - указание на то, что в подаваемых данных важна последовательность

Добавить описание входа моделей. 

---
#### Описание выхода моделей. 
**add_output(shape,layer)**
  - *shape* - форма выхода. 
  - *layer* - выходной слой керас

Добавить описание выхода моделей. 

---
#### Установка параметров обучения популяции
**compile(optimizer="adam", loss=None, metrics=None )**

Описываем параметры обучения популяции. Назначение параметров совпадает с их назначением в keras описанных [здесь](https://keras.io/api/models/model_training_apis/)

---
#### Запуск эволюции
**fit(ga_epochs=1, x=None, y=None, batch_size=None,  epochs=1,  verbose="auto",  validation_split=0.0,  x_val=None,  y_val=None, rescore = False )**
  - *ga_epochs* - количество эпох генетики, Не путать с *epoch* - количеством эпох обучения модели
  - *rescore* - принудительная перетренировка ранее тренированных моделей.

Запуск эволюции. Назначение остальных параметров совпадает с их назначением в keras описанных [здесь](https://keras.io/api/models/model_training_apis/)

---
#### Запуск эволюции c генератором
**fit_generator(ga_epochs=1, train_gen = None, batch_size=None, epochs=1, verbose="auto", validation_gen = None, rescore = False )**
  - *ga_epochs* - количество эпох генетики. Не путать с *epoch* - количеством эпох обучения модели
  - *rescore* - принудительная тренировка моделей которые уже прошли оценку.

Запуск Эволюции на ga_epochs генетических эпох c генератором. Назначение остальных параметров совпадает с их назначением в keras. Все описано [здесь](https://keras.io/api/models/model_training_apis/)

---
#### Получить керас модель  
**get_model(idx=0)**
  - *idx* - индекс бота в популяции

Получить керас модель от idx бота популяции. Все боты пронумерованы в порядке убывания оценки т.е. победитель имеет индекс - 0

---
#### Отчет по эпохе
**report(scoreboard=True,best_detail=False)**
  - *scoreboard* - Вывод сводной таблицы чемпионов
  - *best_detail* - добавляем подробное описание чемпиона

Отчет по эпохе

---
#### Получить индекс бота по имени
**get_index(bot_name)**
  - *bot_name* - имя бота

Получаем индекс бота по имени в популяции. -1 если такого бота в текущей популяции нет
Так-же не забываем, что индекс это место на скамеейке победителей

---
#### Получить имя бота по индексу
**get_bot_name(bot_idx = 0)**
  - *bot_idx* - индекс бота в популяции

Возвращает имя бота по индексу в текущей популяции. Если параметр не задан, получаем имя чемпиона

---
=======
###  Пример
```
!pip install git+https://github.com/509981/Kerasin.git

import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

import kerasin as ks

# Создаем класс декомпозитора на 5 слоев
G=ks.kerasin(complexity = 1, nPopul=100, maxi_goal=True)

# Описываем параметры модели
G.addInput(shape = (28,28,1))
G.addOutput(shape = (10,),layer = Dense(10, activation="softmax"))

# Описываем параметры обучения
G.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# Запуск эволюции
G.fit( ga_epochs = 10, x = x_train, y = y_train, batch_size=125, epochs=20, verbose=0, x_val=x_test, y_val = y_test)

# Выводим результат
G.report(False,True)
# Обученные боты расположены по убыванию val_accuracy. 
utils.plot_model(G.get_model(0), dpi=60, show_shapes=True)
```
