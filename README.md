## kerasin
Kerasin это рабочее название проекта оптимизации нейронных сетей написанных на фреймворке keras. Керасин это вещество ставшее мостом от эпохи угольных энергоносителей к нефтяным. А еще, керас и керасин где-то рядом... )))

Подбор сети происходит с помощью генетического алгоритма. Для этого, Керасин умеет генерировать сети самостоятельно. Топология сети случайным образом может образовывать пробросы и ветвления. С начала образуется каркас - ориентированный граф с одним входом и одним выходом. Он гарантировано не имеет циклов. Затем на этот скелет наращивается мясо в виде случайного набора слоев. Параметры слоев также меняются случайно, варьируясь в пределах своих возможностей.

Класс поддерживает кроссовер из фрагментов сети своих родителей. Сеть родителей режется пополам и одна из частей формирует сеть потомка. 
Есть возможность проведения мутации. Мутации могут касаться как простой смены параметров слоев так и смены типа слоя или их соединения.

---

# МЕТОДЫ КЛАССА

### Конструктор класса
**kerasin(complexity = 1, nPopul=10, maxi_goal=False)**
   
  - *complexity* - сложность сети 1 - на 5 слоев, 2-на 10 и т.п. Количество слоев при генерации ботов точно не соблюдается.
  - *nPopul* - размер популяции
  - *maxi_goal* - целевая функция метрики. True,- если максимизация(accuracy), False - минимизация(loss)

Конструкция создает экземпляр класса 
  
---
### Генерация случайных ботов.  
**generate(nPopul=1,epoch=0)**
  - *nPopul* - количество ботов для генерации
  - *epoch* - Текущая эпоха. Задание параметра epoch важно, для правильного именования ботов.

Генерация nPopul случайных ботов.  

---
### Описание входа моделей. 
**addInput(shape,isSequence = False)**
  - *shape* - форма входа(Без batch размерности) для mnist это может быть (28,28) или (784,)
  - *isSequence* - указание на то, что в подаваемых данных важна последовательность

Добавить описание входа моделей. 

---
### Описание выхода моделей. 
**addOutput(shape,layer)**
  - *shape* - форма выхода. 
  - *layer* - выходной слой керас

Добавить описание выхода моделей. 

---
### Установка параметров обучения популяции
**compile(optimizer="adam", loss=None, metrics=None )**

Описываем параметры обучения популяции. Назначение параметров совпадает с их назначением в keras описанных [здесь](https://keras.io/api/models/model_training_apis/)

---
### Запуск эволюции
**fit(ga_epochs=1, x=None, y=None, batch_size=None,  epochs=1,  verbose="auto",  validation_split=0.0,  x_val=None,  y_val=None, rescore = False )**
  - *ga_epochs* - количество эпох генетики, Не путать с *epoch* - количеством эпох обучения модели
  - *rescore* - принудительная перетренировка ранее тренированных моделей.

Запуск эволюции. Назначение остальных параметров совпадает с их назначением в keras описанных [здесь](https://keras.io/api/models/model_training_apis/)

---
### Запуск эволюции c генератором
**fit_generator(ga_epochs=1, train_gen = None, batch_size=None, epochs=1, verbose="auto", validation_gen = None, rescore = False )**
  - *ga_epochs* - количество эпох генетики. Не путать с *epoch* - количеством эпох обучения модели
  - *rescore* - принудительная перетренировка ранее тренированых моделей.

Запуск Эволюции на ga_epochs генетических эпох c генератором. Назначение остальных параметров совпадает с их назначением в keras. Все описано [здесь](https://keras.io/api/models/model_training_apis/)

---
### Получить керас модель  
**get_model(idx=0)**
  - *idx* - индекс бота в популяции

Получить керас модель от idx бота популяции

---
### Отчет по эпохе
**report(scoreboard=True,best_detail=False)**
  - *best_detail* - добавляем подробное описание чемпиона

Отчет по эпохе

---
### Задание профиля  
**set_profile(profile_name,path='')**
  - *profile_name* - имя профиля
  - *path* - путь размещения профилей

Задаем имя профиля, когда хотим чтобы в каталог с этим именем выгружались боты после генерации
Полный путь формируется объединением path/profile_name

---
### Вывести список ботов профиля
**show_profile(profile_name=None,nSurv=0,include_unscored=True)**
  - *profile_name* - имя профиля
  - *nSurv* - взять nSurv лучших, (0 - всех)
  - *include_unscored* - включать всех у кого нет оценки
  
Вывести список ботов профиля. 

---
#  Пример
```

# Создаем класс декомпозитора на 5 слоев
G=ks.kerasin(complexity = 1, nPopul=10, maxi_goal=True)
# Описываем параметры модели
G.addInput(shape = input_shape)
G.addOutput(shape = (10,),layer = Dense(10, activation="softmax"))
# Описываем параметры обучения
G.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
# Запуск эволюции
G.fit( ga_epochs = 5, x = x_train, y = y_train, batch_size=125, epochs=2, verbose=0, x_val=x_test, y_val = y_test)
# Выводим результат
G.report(False,True)
# Обученные боты расположены по убыванию val_accuracy. 
utils.plot_model(G.get_model(0), dpi=60, show_shapes=True)
```
