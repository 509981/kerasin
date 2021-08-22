#Поиск оптимальной модели нейросети с применением генетического алгоритма
#Применяется класс который способен генерировать случайным образом модели совместимые с фреймворком keras. Проводить операции кроссовера и мутации над ними.
#Версия 1.2 от 19 августа 2021 г.`
#Автор: Утенков Дмитрий Владимирович
#e-mail: 509981@gmail.com 
#Тел:   +7-908-440-9981

#Библиотека

#import networkx as nx
#from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from copy import deepcopy

from tensorflow.keras.layers import Dense,Dropout,Input,concatenate,BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.layers import LSTM,Embedding,Reshape,GaussianNoise,Activation
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D,Flatten,LSTM,LeakyReLU
from tensorflow.keras.models import Model,clone_model
from tensorflow.keras import utils
import keras.backend as K


trace = False

def prn(*arg):
  if trace: print(arg)

## Орграф


#///////////////////////////////////////////////////////////////////////////////
# Генерация нециклического орграфа c одним выходом и 1-4 входами
#///////////////////////////////////////////////////////////////////////////////

class neuro_graph:
  def __init__(self):
    self.edjes = list()
    self.n_in = 1
  # Количество вершин графа
  def nNodes(self):
    ret=0
    for edje in self.edjes:
      ret = max(ret,edje[0],edje[1])
    return ret

  # Добавить ребро
  def addEdje(self,node_from,node_to):
    self.edjes.append((node_from,node_to))
    
  # Сгенерировать граф
  def generate(self,maxnodes,n_in = 1):
    self.clear()
    is_cycle = True
    while(is_cycle):
      self.clear()
      self.__generate_subnet__(5,maxnodes,n_in,0)
      is_cycle = self.__renumerate__()

  # Очистить граф
  def clear(self):
    self.edjes = []

  # Функция генерации орграфа с одним выходом и n_in входов
  # не рекомендуется более 5 вершин из-за появления циклов
  # n - количество вершин
  # n_in - количество входов
  def __generate_graph__(self,n,n_in = 1):
    self.n_in=n_in
    # Вероятностный список количества входящих в вершину ребер
    nodes_in_list = [1,1,1,1,1,1,1,1,2]
    # Создаем матрицу инцидентности
    graph = np.zeros((n,n))
    # Проходим матрицу построчно из конца к началу
    for i in range(n-1,n_in-1,-1):
      # Определяем количество входящих ребер nodes_in в вершину i
      nodes_in = nodes_in_list[random.randint(0,len(nodes_in_list)-1)]
      # Создаем список возможных ребер node_area
      # убираем последний столбец тк не может получать ребро от out node
      node_area = np.arange(0,n-1)  
      if i<n: 
        node_area = np.setdiff1d(node_area,i) # Не может ссылаться на себя
        # node не может получить вход от node который получил вход от него
        for node in node_area:
          if graph[node][i] > 0: 
            node_area = np.setdiff1d(node_area,node)
            
      if i==n-1: 
        node_area = np.setdiff1d(node_area,np.arange(n_in))  #Нельзя связывать вход и выход напрямую

      # Предотвращение висячих нодов 
      # исключим из nodes_in все заполненые колонки если дело близко к концу
      # для этого соберем ноды имеющие входы в filled_cols
      filled_cols = []
      for j in range(n-n_in):
        if np.sum(graph[:,j])>0: filled_cols.append(j)
      # Исключим из списка возможных вершин те которуе уже имеют вершину в пользу висячих 
      if n-1-len(filled_cols)>=i: 
        node_area = np.setdiff1d(node_area,filled_cols)
        #if len(node_area)==0: node_area = np.array(filled_cols)
      if len(node_area)==0:
        #  Если после этого вариантов не осталось и нод остался висящим 
        # соединяем его входу
        graph[i][0] = 1
        #node_area = np.array(filled_cols)
      # Проследим чтобы требуемое количество нодов не привысило кол-во вариантов
      nodes_in = min(nodes_in,node_area.shape[0])
      # Выбираем входящие ноды из оставшихся
      list_nodes = random.sample(list(node_area),nodes_in)
      # Делаем отметку в матрице инцидентности
      for node in list_nodes:
        graph[i][node] = 1
    # Финальная проверка висячих строк
    # Если находим, замыкаем на out node
    for j in range(0,n-1):
      if np.sum(graph[:,j])==0: 
          graph[n-1][j] = 1
    return graph    

  # Рекурсивная функция генерации орграфа путем развертки 
  # случайного ребра на вложеный орграф
  #
  # nodes - количество вершин на итерации
  # maxnodes - максимальное количество вершин графа
  # n_in - кол-во входных вершин(только для 1 итерации, для остальных=1)
  # start_node = номер первого нода итерации
  def __generate_subnet__(self,nodes,maxnodes,n_in = 1,start_node=0):
    assert n_in < nodes # Количество входов должно быть меньше чем количество вершин
    # Генерируем орграф с матрицей смежности MAJ
    MAJ = self.__generate_graph__(nodes,n_in)
    for node_to in range(nodes):
      for node_from in range(nodes):
        # Находим ребро в матрице смежности
        if MAJ[node_to][node_from]>0:
          if(( (node_from < n_in and node_to==nodes-1) # Если вход напрямую соединенс выходом
            or random.random()>0.3)               # или вмешивается случай
            and start_node+nodes<maxnodes):       #и количество вершин не превышает лимит
            # ребро разворачиваем в подсеть с 5 вершинами максимум.
            subnodes=min(5,maxnodes-start_node-nodes)
            # Соединяем подсеть с вершинами разорваного ребра
            self.edjes.append((start_node+node_from, start_node+nodes))
            self.edjes.append((start_node+subnodes+nodes-1, start_node+node_to))
            # Только сеть первой итерации может иметь несколько входов
            # поэтому далее указываем явно n_in=1
            self.__generate_subnet__(subnodes,maxnodes,1,start_node+nodes)          
          else:
            # Сохраняем ребро из матрицы инцидентности
            self.edjes.append((start_node+node_from,start_node+node_to))
    #prn(MAJ)
    return 

  #   Перенумеровать вершины в порядке построения:т.е. не одна вершина не будет собрана
  # пока не будут собраны входящие в нее 
  # Это необходимо для удобного изменения входящих дуг и кросовера
  def __renumerate__(self):
    num_nodes = self.nNodes()
    node_done = set(range(self.n_in))
    skiped_layer = True
    connectors_in = []
    n_con_done = 0
    new_idx = self.n_in
    trans = {a:a for a in range(num_nodes)}
    while(skiped_layer):
      #prn('--------------')
      skiped_layer = False
      if n_con_done > 1:
        #prn('Err: граф цикличен')
        return True
      n_con_done += 1

      for idx in range(self.n_in,num_nodes+1):
        if not idx in node_done:
          
          skiped_layer = True
          unfill_ins = False
          list_in = [x[0] for x in self.edjes if x[1] == idx]#[edjes[1]==idx for len(edjes)]
          #prn(idx,list_in)
          for in_node in list_in:
            #prn(' - ',in_node)
            if not in_node in node_done:
              #prn(' - ',in_node,'пуст, пропускаем')
              unfill_ins = True
              break
            #else:
              #prn(' - ',in_node,'done')
          if unfill_ins: continue
          node_done.add(idx)
          trans[idx]=new_idx
          #prn(idx,' слой определен как ',new_idx)
          new_idx += 1
          n_con_done = 0
          #prn(trans)
    new_edjes = []
    trans[0]=0
    for edje in self.edjes:
      #new_node_from = [x[1] for x in trans if x[0] == edje[0]]
      #new_node_to = [x[1] for x in trans if x[0] == edje[1]]
      #prn(new_node_from,new_node_to,edje)
      new_edjes.append((trans[edje[0]],trans[edje[1]]))
    self.edjes = new_edjes
    return False


## Класс гена нейросети

GUnknown, GInput, GMain, GExt  = range(4)
#Список основных типов слоев содержит правильные названия их классов и частоту
types_list = ['Dense', 'Dense', 'Conv2D', 'Conv2D','Conv1D','Conv1D',\
                 'MaxPooling1D','MaxPooling2D','LSTM','GaussianNoise','Flatten']

#types_list = ['Conv2D', 'Conv2D','MaxPooling2D','GaussianNoise','Dense']
#ext_types_list = ['Flatten','concatenate','Dropout','BatchNormalization']                 


# типы активации
type_activations = ['linear','relu', 'elu','tanh','softmax','sigmoid', 'selu']


#///////////////////////////////////////////////////////////////////////////////
# Класс гена нейросети 
#///////////////////////////////////////////////////////////////////////////////

class gen(object):
  def __init__(self,layer_idx=0, sublayer_idx=0, block_name='', var_name='', value=0):
    self.layer_idx = layer_idx  # Номер слоя
    self.sublayer_idx = sublayer_idx  # Номер вложеннного слоя
    self.name = block_name  # Имя типа слоя из types_list
    self.var_name = var_name  # Имя параметра слоя керас
    self.value = value  # Значение параметра слоя керас
    self.changed = False  # Признак измененного значения

  # Получить строку представления гена
  def get(self):
    return ""+str(self.layer_idx)+"."+ str(self.sublayer_idx)+" "+self.name+ ": "+\
    self.var_name +"="+ str(self.value)

  # Получить строку представления гена
  def load_csv(self,str):
    lst = str.split(';')
    self.layer_idx = int(lst[0])
    self.sublayer_idx = int(lst[1])
    self.name = lst[2]
    self.var_name = lst[3]
    self.value = lst[4]
    try:
      if self.value == '':
        pass
      elif '[' == self.value[0]: # значение в виде списка
        self.value = [int(x) for x in self.value.strip('[]').split(',')]
      elif '(' == self.value[0]: # значение в виде кортежа
        self.value = tuple([None if x == 'None' else int(x) for x in self.value.strip('()').split(',') if x != ''])
      elif 'L1L2' in self.value:
        self.value = 'l1_l2'
      elif 'L1' in self.value:
        self.value = 'l1'
      elif 'L2' in self.value:
        self.value = 'l2'
      elif self.value=='None':
        self.value = None
      elif self.value=='True':
        self.value = True
      elif self.value=='False':
        self.value = False
      elif self.value.isdigit():
        self.value = int(self.value)
      elif self.value.replace('.','').isdigit():
        self.value = float(self.value)
      elif 'Wrapper' in self.value:
        return False
    except:
      print('Не смог загрузить параметр ',str,lst,self.value)
      return False
    self.changed = False
    return True

  # Получить строку представления гена
  def save_csv(self):
    return ""+str(self.layer_idx)+";"+ str(self.sublayer_idx)+";"+self.name+ ";"+\
    self.var_name +";"+ str(self.value)+'\n'


  # Изменяем параметр генома случайным способом
  # arg - словарь аргументов для проверки на взамоисключающее сочетание параметров
  def mutate(self):
    self.changed = True
    if self.var_name == 'name': return 2 # Изменение типа слоя
    elif 'Concatenate' == self.name and self.var_name == 'inbound_layers': return 3 # Изменение связи
    #elif 'concat' in self.name: return 3 # Изменение связи
    if self.name == 'InputLayer': return 0 # Игнор
    elif self.var_name == 'units': self.value=2**random.randint(2,10)
    elif self.var_name == 'filters': 
      self.value=2**random.randint(2,8)
      # Conv1D:The number of filters must be evenly divisible by the number of groups. Received: groups=3, filters=16
      '''
      try:
        gr = arg['groups']
        while True:
          flt=2**random.randint(2,8)
          if flt % gr == 0: 
            self.value = flt / gr
            break       
      except:
        self.value = 1
      #prn("f!!!",gr,flt,self.value)
      '''  
      #self.value=2**random.randint(2,8)
    elif self.var_name == 'activation': self.value = type_activations[random.randint(0,len(type_activations)-1)]
    elif self.var_name == 'use_bias': self.value = (random.random() > .3)
    elif self.var_name == 'scale' and 'BatchNormalization' == self.name: self.value = (random.random() > .3)
    elif self.var_name == 'center' and 'BatchNormalization' ==self.name: self.value = (random.random() > .3)
    elif 'regularizer' in self.var_name: self.value = random.sample([None,None,None,None,'l1','l2','l1_l2'],1)[0] #tf.keras.regularizers.l2(0.01)
    elif self.var_name == 'rate' and 'Dropout' == self.name: self.value=random.random()*.5
    elif 'dropout' in str.lower(self.var_name): self.value=random.random()*.5 #LSTM
    elif self.var_name == 'stddev': self.value=random.random()*.5 # Noise
    elif self.var_name == 'go_backwards': self.value = (random.random() < .3) #LSTM
    elif self.var_name == 'unit_forget_bias': self.value = (random.random() > .3) #LSTM
    elif self.var_name == 'unroll': self.value = (random.random() < .3) #LSTM
    elif self.var_name == 'stateful': self.value = (random.random() < .3) #LSTM
    elif self.var_name == 'pool_size': self.value = random.randint(2,4) #MaxPooling
    elif self.var_name == 'alpha': self.value = random.random() #LeakyReLU
    elif self.var_name == 'kernel_size': 
      winsize = random.randint(2,7)
      if '1D' in self.name: self.value = (winsize,)
      elif '2D' in self.name: self.value = (winsize,winsize)
      elif '3D' in self.name: self.value = (winsize,winsize,winsize)
    elif self.var_name == 'strides': 
      val = random.randint(1,4)
      if '1D' in self.name: self.value = (val,)
      elif '2D' in self.name: self.value = (val,val)
      elif '3D' in self.name: self.value = (val,val,val)
    elif self.var_name == 'padding':
      self.value=random.sample(['valid','same'],1)[0]
    else:
      # Список неизменяемых параметров
      if self.var_name == 'dilation_rate': pass
        #  В настоящее время указание любого dilation_rateзначения! = 1 несовместимо с указанием любого значения шага! = 1.'
      elif self.var_name == 'groups': 
        self.value = 1
        # Conv1D:The number of filters must be evenly divisible by the number of groups. Received: groups=3, filters=16
        '''
        The number of input channels must be evenly divisible by the number of groups. Received groups=8.0, but the input has 28 channels (full input shape is (None, 28, 28))
        try:
          flt = arg['filters']
          while True:
            gr=random.randint(1,4)
            if flt % gr == 0: 
              self.value = flt / gr
              break       
        except:
            self.value = 1
        prn("!!!",gr,flt,self.value)
        '''
      # Эти параметры обычно нет смысла менять
      elif 'constraint' in self.var_name: pass
      elif self.var_name == 'dtype': pass
      elif self.var_name == 'return_sequences': pass
      elif self.var_name == 'seed': pass
      elif self.var_name == 'trainable': pass
      elif self.var_name == 'data_format': pass
      elif self.var_name == 'axis': self.value = -1
      elif self.var_name == 'ragged': pass  # Input
      elif self.var_name == 'sparse': pass  # Input
      elif self.var_name == 'batch_input_shape': pass  # Input
      # Разобраться позже
      elif self.var_name == 'inbound_layers': pass # return 3 # Изменение связи не 'Concatenate' == self.name
      elif self.var_name == 'momentum': pass  # BatchNorm
      elif self.var_name == 'epsilon': pass  # BatchNorm
      elif self.var_name == 'scale': pass  # BatchNorm
      elif self.var_name == 'noise_shape': pass  # Dropout
      elif self.var_name == 'implementation': pass  # lstm
      elif self.var_name == 'time_major': pass  # lstm
      elif self.var_name == 'return_state': pass  # lstm
      elif self.var_name == 'recurrent_activation': pass  # lstm
      else:
        prn('пропущено:',self.name,'-',self.get())
      return 0

    return 1
  
  # Копирование гена
  def copy(self):
    clone = deepcopy(self)
    #clone._b = some_op(clone._b)
    return clone


## Класс слоя нейросети

def remove_batch_dim(shape):
  s = list(shape)
  s.pop(0)
  return tuple(s)

#///////////////////////////////////////////////////////////////////////////////
# Класс слоя
#///////////////////////////////////////////////////////////////////////////////

class gen_layer(object):
  def __init__(self):
    self.connector = None # Выходной тензор keras f api
    self.__sublayers__ = list()  # Список вложенных ссылок на слой керас
    self.genom = list()  # Геном(хромосома) слоя

    #Свойства топологии
    self.list_in = list()         # Список входов
    self.list_out = list()        # Список выходов
    # Свойства нейрона
    self.neurotype = GUnknown  # Тип слоя
    self.shape_out = ()        # Форма на выходе
    self.data = None  # Любые данные к слою

  '''
  def copy(self):
    clone = copy.deepcopy(self)
    #clone._b = some_op(clone._b)
    return clone


  def __deepcopy__(self, memo): # memo is a dict of id's to copies
    id_self = id(self)        # memoization avoids unnecesary recursion
    _copy = memo.get(id_self)
    if _copy is None:
       #_copy = type(self)(
         #deepcopy(self.layers, memo), 
         #deepcopy(self.b, memo))
         memo[id_self] = _copy 
    return _copy

  def __deepcopy__(self, memo):
    prn('deep copying layers...') 
    clone = type(self)()
    memo[id(self)] = clone
    clone.list_in = deepcopy(self.list_in, memo)  
    clone.list_out = deepcopy(self.list_out, memo)  
    clone._sublayers_ = deepcopy(self._sublayers_, memo)  
    clone.connector = deepcopy(self.connector, memo)  
    return clone  
  '''
  # Получить геном списком
  def get_genom(self,mutable_only=True,changed_only=False): 
    if len(self.genom) == 0:# and (not changed_only):
      print('Генотип не заполнен. Вызовите sequence()',self.connector)
      return []
      #self.sequence(0)
    lst = list()
    for l in self.genom:
      if not changed_only or l.changed:
        if mutable_only:
          # Тестируем ген на изменчивость 
          cpy = l.copy()
          if cpy.mutate()==0: continue
        lst.append(l)
    return lst


  # Извлечь генотип из словаря параметров слоя get_config()
  def __cfg_to_genom__(self,cfg,layer_name='xxx', layer_idx = -1, sublayer_idx=0):
    genom = []
    for key, value in cfg.items():
      if not ('initializer' in key):
        genom.append( gen( layer_idx, sublayer_idx, layer_name, key, value) )

    return genom

  # Перевести генотип в словарь аргументов совместимых с слоем керас
  def __genom_to_cfg__(self,genom=None,cfg = None):
    if genom == None: genom = self.genom
    if cfg == None: cfg = dict()
    for g in genom:
      if g.var_name != 'inbound_layers':
        cfg[g.var_name] = g.value
    return cfg
  # Тест Неактивности слоя. Неактивный слой не имеет входа/выхода
  def IsInactiveLayer(self):
    return len(self.list_in)+len(self.list_out) == 0

  # Прочитать генотип из фенотипа(модели керас)
  def sequence(self,idx_layer):
    #prn(len(self.sublayer)) 
    if self.connector == None: 
      print(idx_layer,' - нет слоя для сиквенции',self.list_in,self.list_out)
      assert not self.IsInactiveLayer() # Не пустой слой причина?
      return None    
    self.genom.clear()
    for idx in range(len(self.__sublayers__)):
      layer = self.__sublayers__[idx]
      l_type = layer.__class__.__name__
      self.genom.extend( self.__cfg_to_genom__(layer.get_config(),l_type, idx_layer, idx ) )
      if 'Concat' in l_type or l_type in types_list:
        # Для главного слоя указываем входящие
        self.genom.append( gen( idx_layer, idx, l_type, 'inbound_layers', self.list_in) )

    return self.genom
  # Добавить слой в список
  def addLayer(self,layer):
    self.__sublayers__.append(layer)

  # Очистить слой от признаков фенотипа и топологии
  def clear(self): 
    self.list_in.clear()
    self.list_out.clear()
    self.connector = None
    self.__sublayers__.clear()
    #self.genom.clear()

  # Тестирует является ли слой выходом
  def isOut(self): 
    return (len(self.list_out)==0 and len(self.list_in)>0)
  
  # Получить форму выходного тензора слоя
  def __get_shape__(self):
    if self.connector == None: return None
    return self.connector.shape   
  # Количество входящих связей
  def NumConnectorIn(self): return len(self.list_in)
  # Количество исходящих связей
  def NumConnectorOut(self): return len(self.list_out)
  # Добавить Начальный слой
  def addConnectionIn(self,connector_id): 
    if not connector_id in self.list_in: self.list_in.append(connector_id)
  # Добавить Конечный слой
  def addConnectionOut(self,connector_id): 
    if not connector_id in self.list_out: self.list_out.append(connector_id)
  # Удалить Входящую связь
  def delConnectionIn(self,connector_id): 
    self.list_in.remove(connector_id)
  # Удалить Исходящую связь
  def delConnectionOut(self,connector_id): 
    self.list_out.remove(connector_id)
  # Установить тип слоя
  #def SetNeuro(self,neurotype): self.neurotype = neurotype

  def get_idx(self,name):
    for idx in range(len(self.__sublayers__)):      
      if self.__sublayers__[idx].name == name: 
        #print(self.layers[idx].__sublayers__[0].name, idx)
        return idx
    return -1


  # Подбираем слой под входящую конфигурацию
  def __suggest_layer__(self,inbound_layers,is_sequence=False):
    if len(inbound_layers) == 0:
      return 'InputLayer'
    elif len(inbound_layers) > 1:
      return 'concatenate'
    else:
      shape_dim = len(inbound_layers[0].shape)-1
      layer_in_name = inbound_layers[0]._keras_history.layer.__class__.__name__

      while True:
        neurotype = random.sample(types_list,1)[0]
        #prn(neurotype,layer_in_name) 
        if neurotype == 'Flatten'  and random.random()<.1 and shape_dim>1 and neurotype != layer_in_name: break; # and layer_in != 'Flatten':
        elif neurotype == 'MaxPooling2D' and shape_dim==3: break
        elif neurotype == 'Conv2D' and shape_dim==3: break
        elif neurotype == 'MaxPooling1D' and shape_dim==2: break
        elif neurotype == 'Conv1D' and shape_dim>1: break
        elif neurotype == 'LSTM' and shape_dim==2 and is_sequence: break
        elif neurotype == 'GaussianNoise' and random.random()<.4 and neurotype != layer_in_name: break
        elif neurotype == 'Dense': break
    return neurotype



  # Получаем образец набора параметров слоя
  def __get_layer_params__(self,neurotype):
    if neurotype == 'Dense':
      x = Dense(10)
    elif neurotype == 'concatenate':
      return dict()
    elif neurotype == 'Flatten':
      x = Flatten()
    elif neurotype == 'MaxPooling2D':
      x = MaxPooling2D()
    elif neurotype == 'Conv2D':
      x = Conv2D(10,1)
    elif neurotype == 'MaxPooling1D':
      x = MaxPooling1D()
    elif neurotype == 'Conv1D':
      x = Conv1D(10,1)
    elif neurotype == 'LSTM':
      x = LSTM(10)
    elif neurotype == 'GaussianNoise':
      x = GaussianNoise(.5)
    elif neurotype == 'Dropout':
      x = Dropout(.5)
    elif neurotype == 'BatchNormalization':
      x = BatchNormalization()
    elif neurotype == 'LeakyReLU':
      x = LeakyReLU()
    elif neurotype == 'Activation':      
      x = Activation('relu')
    else:
      print('недопустимый слой:',neurotype)
      assert False
    return x.get_config()

  # Создаем слой
  def __create_layer__(self,neurotype,conn_in = None,cfg=None):
    #prn('попытка сборки слоя - ',neurotype)
    if neurotype == 'InputLayer':
        # Создаем Вх. слой только если он не существует Пересоздание не допустимо(?)
        assert self.connector == None
        x = Input(self.shape_out)
        self.addLayer(x._keras_history.layer)
        #self.genom.clear()
        return x
    #assert x != None and cfg != None

      #elif neurotype == 'Embedding':
      #  x = Embedding(self.shape_out,2**random.randint(5,9))
    elif str.lower(neurotype) == 'concatenate':
      # Несколько входов - значит коннектор
      assert len(conn_in)>1
      #self.SetNeuro(GConcat)
      try:
        x = concatenate(conn_in)
        self.addLayer(x._keras_history.layer)
      except:
        prn('прямая конкатенация невозможна. Применяю Flatten.')
        flate_connector = []
        for conn in conn_in:
          if len(conn.shape) == 2: 
            flate_connector.append(conn)
          else:
            x = Flatten()(conn)
            self.addLayer(x._keras_history.layer)
            flate_connector.append(x)
        try:        
          x = concatenate(flate_connector)
          self.addLayer(x._keras_history.layer)
        except:
          self.__sublayers__.clear()
          print('ошибка конкатенации слоев с Flatten',self.__get_shape__())
          return None
      return x
    try:
      #if True:
      x = conn_in[0]
      if neurotype == 'Dense':
        x = Dense.from_config(cfg)(x)
      elif neurotype == 'Flatten':
        x = Flatten.from_config(cfg)(x)
      elif neurotype == 'MaxPooling2D':
        x = MaxPooling2D.from_config(cfg)(x)
      elif neurotype == 'Conv2D':
        x = Conv2D.from_config(cfg)(x)
      elif neurotype == 'MaxPooling1D':
        x = MaxPooling1D.from_config(cfg)(x)
      elif neurotype == 'Conv1D':
        x = Conv1D.from_config(cfg)(x)
      elif neurotype == 'LSTM':
        x = LSTM.from_config(cfg)(x)
      elif neurotype == 'GaussianNoise':
        x = GaussianNoise.from_config(cfg)(x)
      elif neurotype == 'Dropout':
        x = Dropout.from_config(cfg)(x)
      elif neurotype == 'BatchNormalization':
        x = BatchNormalization.from_config(cfg)(x)
      elif neurotype == 'LeakyReLU':
        x = LeakyReLU.from_config(cfg)(x)
      elif neurotype == 'Activation':
        x = Activation.from_config(cfg)(x)
      else:
        print(neurotype,' - неописаный слой')
        return None  
      self.addLayer(x._keras_history.layer)
      
    except Exception as e:
      #except ValueError:
      prn(neurotype,' - слой с входом',conn_in,'не собрался: '+str(e))
      #prn('Слой активен?:',not self.IsInactiveLayer())
      return None
    return x


  # Добавляем вспомогательные слои
  def __create_extralayers__(self,x):
    layer_in_name = x._keras_history.layer.__class__.__name__     
    if random.random()>.7:
      #prn('сборка слоя - Dropout')
      arg = self.__get_layer_params__('Dropout')
      cfg = self.__mutate__('Dropout',arg)
      x = self.__create_layer__('Dropout',[x],cfg)
    if random.random()>.7:
      #prn('сборка слоя - BatchNormalization')
      arg = self.__get_layer_params__('BatchNormalization')
      cfg = self.__mutate__('BatchNormalization',arg)
      x = self.__create_layer__('BatchNormalization',[x],cfg)
    if random.random()>.85: #1/8 layer_in_name == 'Dense' 
      arg = self.__get_layer_params__('LeakyReLU')
      cfg = self.__mutate__('LeakyReLU',arg)
      x = self.__create_layer__('LeakyReLU',[x],cfg)
    return x

  # Проводим мутацию аргументов слоя без сохранения
  def __mutate__(self,layer_name,arg):
    genom = self.__cfg_to_genom__(arg,layer_name)
    #arg_to_verify = arg.copy()
    #assert len(genom)>0
    for g in genom:
      if g.var_name != 'name':
        g.mutate()#arg_to_verify)
        #arg_to_verify[g.var_name] = g.value # Актуализируем данные для проверки непротиворечивости
    return self.__genom_to_cfg__(genom, arg)

  # Создаем слой случайно или из генотипа
  def build_layer(self,connectors_in,is_sequence=False):
    #if self.neurotype == GInput:
    if len(self.list_in) == 0:
      if self.connector == None:
        self.connector = self.__create_layer__('InputLayer')
      return self.connector
    #changed_gens = self.getChangedGens()
    changed_gens = [g for g in self.genom if g.changed]
    #prn(changed_gens,self.genom)
    change_type = [g for g in changed_gens if g.var_name == 'name']
    #prn('=====================',change_type)
    self.connector = None
    self.__sublayers__.clear() # !!! __del__ надо?
    if len(change_type)>0 or len(self.genom)==0:
      # Требуется замена типа слоя или создание
      #self.__sublayers__.clear() # !!! __del__ надо?
      while(self.connector == None):
        layer_type = self.__suggest_layer__(connectors_in,is_sequence)
        arg = self.__get_layer_params__(layer_type)
        # Для случайного задания параметров нового слоя проводим мутацию всего генома слоя
        cfg = self.__mutate__(layer_type,arg)
        # Создаем слой
        self.connector = self.__create_layer__(layer_type, connectors_in, cfg)
        prn('создание',layer_type,cfg)

      self.connector = self.__create_extralayers__(self.connector)
      #layers.deserialize(({'class_name': layer_type, 'config': arg})
    else:
      assert len(self.genom)>0
      # Переформируем слой на основе генома
      if len(connectors_in)>1:
        self.connector = self.__create_layer__('concatenate', connectors_in)
      else:
        sublayer_idx = -1
        x = connectors_in[0]
        for idx in range(10):
          gens = [g for g in self.genom  if g.sublayer_idx == idx and g.var_name != "inbound_layers"]
          if len(gens)==0: break
          if str.lower(gens[0].name) == 'concatenate':
            #assert len(connectors_in)==1
            print('Ошибка: Обнаружена конкатенация с одним входом!',connectors_in)
            return None

          cfg = self.__genom_to_cfg__(gens)
          #layer_type = cfg['name']
          prn('замена',gens[0].name,cfg)
          x = self.__create_layer__(gens[0].name, [x], cfg)
        self.connector = x

    return self.connector

  # Обработчик финального слоя
  def finish(self,layer_in):
    if len(self.genom) and self.data==None:
      #prn('add flat build') 
      self.connector = self.build_layer([layer_in])
    if self.data:
      cfg = self.data.get_config()
      name = self.data.__class__.__name__
    elif len(self.__sublayers__)>0:
      #Если нет информации о финальном слое попробуем ее 
      #взять из последнего слоя
      cfg = self.__sublayers__[-1].get_config()
      name = self.__sublayers__[-1].__class__.__name__
    else:
      print('Нет данных о финальном слое')
      return None
    #prn('add flat build',len(fl.shape_out),fl.data,name,cfg) 
    assert len(self.shape_out)>0  # Должна быть определена форма выхода. Используйте addOutput()
    if len(self.shape_out)==1 and len(layer_in.shape)>2:# and len(fl.__sublayers__)<2:
      #prn('add flat',len(self.__final_layer__.shape_out),len(out_con.shape)) 
      self.__sublayers__.clear()
      layer_in = Flatten()(layer_in)
      self.addLayer(layer_in._keras_history.layer)
      self.connector = None

    if self.connector == None:
      #prn('add flat creat') 
      self.connector = self.__create_layer__(name, [layer_in], cfg)
    return self.connector



##Класс сети


#///////////////////////////////////////////////////////////////////////////////
# Класс генотипированной сети керас
#///////////////////////////////////////////////////////////////////////////////

class gen_net(object):
  # кол-во попыток autocrossover()
  max_crossover_loop = 25

  def __init__(self,name,family):
    
    self.name = name
    self.description = ""
    # Слои сети
    self.layers = list()#[ gen_layer() for i in range(nodes) ]
    # Количество слоев Input
    self.nodes_in = 1
    # Слой выхода
    self.__final_layer__ = gen_layer()
    # Керас модель
    self.model = None
    # Признак учета последовательности во входных данных
    self.is_sequence = False
    # Результат работы fit 
    self.hist = None
    # Оценка точности сети.Как правило loss. -1 если оценки не было
    self.score = -1
    # family присваивается случайно сгенерированым сетям. сохраняется при мутации
    # при кросовере берет фамилию жены, ой, сети с самой лучшей оценкой.
    # служит для оценки родственности сетей
    self.__family__ = family
    self.train_duration = 0 # Длительность обучения сек.

  def __len__(self):
    return len(self.layers)

  def __getitem__(self, key):
    return self.layers[key]

  def get_family(self):
    return self.__family__

  def set_family(self,family):
    self.__family__ = family

  '''
  def __deepcopy__(self, memo): # memo is a dict of id's to copies
        id_self = id(self)        # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.layers, memo), 
                #deepcopy(self.b, memo))
            memo[id_self] = _copy 
        return _copy


  def __deepcopy__(self, memo):
    prn('deep copying ...') 
    clone = type(self)()
    memo[id(self)] = clone
    clone.model = clone_model(self.model)
    clone.layers = deepcopy(self.layers, memo)  
    return clone
  '''
  

  def save(self,path):
    if not os.path.exists(path):
      os.mkdir(path)
    file = open(path+self.name+'.gn', 'w')
    gt = self.sequence()
    for g in gt:
      file.write(g.save_csv())

    file.write( gen(9999,0,'','output_shape',self.__final_layer__.shape_out).save_csv())
    file.write( gen(9999,0,'','description',self.description).save_csv())
    file.write( gen(9999,0,'','family',self.__family__).save_csv())
    file.write( gen(9999,0,'','score',self.score).save_csv())
    file.write( gen(9999,0,'','train_duration',self.train_duration).save_csv())

    trainable_count = int(np.sum([K.count_params(p) for p in self.model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in self.model.non_trainable_weights]))

    file.write( gen(9999,0,'','trainable_count',trainable_count).save_csv())
    file.write( gen(9999,0,'','non_trainable_count',non_trainable_count).save_csv())


    hist = self.hist
    if hist != None:
      for i in range(len(hist['loss'])):
        #print(hist['loss'],i)
        file.write( gen(9999,i,'','loss',hist['loss'][i]).save_csv())
        if 'val_loss' in hist:
          file.write( gen(9999,i,'','val_loss',hist['val_loss'][i]).save_csv())
        if 'accuracy' in hist:
          file.write( gen(9999,i,'','accuracy',hist['accuracy'][i]).save_csv())
        if 'val_accuracy' in hist:
          file.write( gen(9999,i,'','val_accuracy',hist['val_accuracy'][i]).save_csv())

    file.close()


  def load(self,name,path):
    full_genom=[]
    try:
      file = open(path+name+'.gn', 'r')
      #new = gen_net(self.name,self.get_family())
      lines = file.readlines()
      if len(lines) == 0: return False

      from collections import defaultdict
      hist = defaultdict(lambda: list())
      for line in lines:
        g = gen()
        if not g.load_csv(line.strip()): continue
        if g.layer_idx == 9999:
          if g.var_name == 'output_shape':
            self.__final_layer__.shape_out = tuple(g.value)
          elif g.var_name == 'description':
            self.description = str(g.value)
          elif g.var_name == 'family':
            self.__family__ = int(g.value)
          elif g.var_name == 'score':
            self.score = float(g.value)
          elif g.var_name == 'train_duration':
            self.train_duration = int(g.value)
          elif g.var_name in 'val_loss val_accuracy':
            hist[g.var_name].append(float(g.value))
          elif g.var_name == 'non_trainable_count':
            pass
          elif g.var_name == 'trainable_count':
            pass
        else:
          full_genom.append(g)
      file.close()
      #print(self.print_genom(full_genom))
      
      #G = gen_net(name,15)
      self.name = name
      self.hist = hist
      if not self.load_genom(full_genom):
        print('не смог инициализировать модель по загруженому геному',path+name+'.gn')
        return False

      if self.synthesis() == None:
        print('не смог синтезировать загруженую модель',path+name+'.gn')
        return False

    except:
      print('не смог загрузить',path+name+'.gn')
      return False
    return True

  # Копирование экземпляра класса
  def copy(self):
    #prn('deep copying ...')     
    self.sequence()
    gn = self.get_genom(False,False)
    clone = gen_net(self.name+' clone',self.get_family())
    clone.nodes_in = self.nodes_in
    clone.is_sequence = self.is_sequence
    #clone.family = self.family
    clone.__final_layer__.shape_out = self.__final_layer__.shape_out
    clone.__final_layer__.data = self.__final_layer__.data
    clone.load_genom(gn)
    if clone.synthesis(): return clone
    print('Клонирование неудачно...')    
    return None

  
  #
  #   Методы работы с топологией
  #

  #   Добавить удалить связь между слоями True в случае успеха
  def reConnection(self,id_from_new,id_from,id_to):
    assert id_from_new < id_to
    assert id_from < id_to
    res = self.layers[id_to].delConnectionIn(id_from)
    res = self.layers[id_to].addConnectionIn(id_from_new)
    self.layers[id_from].delConnectionOut(id_to)
    self.layers[id_from_new].addConnectionOut(id_to)
    return False
  #   Добавить удалить связь между слоями True в случае успеха
  def delConnection(self,id_from,id_to):
    maxnode = max(id_from,id_to)
    assert maxnode<len(self.layers)
    res = self.layers[id_to].delConnectionIn(id_from)
    if res:
      return self.layers[id_from].delConnectionOut(id_to)
    return False
  #   Добавить проброс между слоями
  def addConnection(self,id_from,id_to):
    maxnode = max(id_from,id_to)
    assert maxnode<100
    # Если таких слоев еще нет - создаем
    for idx in range(len(self.layers)-1,maxnode):
      self.layers.append( gen_layer() )

    self.layers[id_to].addConnectionIn(id_from)
    self.layers[id_from].addConnectionOut(id_to)
  #   Очистить слои
  def clearModel(self):
    self.model = None
    for l in self.layers:
      l.clear()
    self.layers.clear()
    if self.__final_layer__:
      # Не чистим форму и слой-образец(data)
      self.__final_layer__.__sublayers__.clear()
      self.__final_layer__.connector = None
    '''
    self.layers
    for i in range(len(self.layers)-self.nodes_in):
      self.layers.pop(self.nodes_in)
    for l in self.layers:
      l.clear()
    #prn(len(self.layers),self.nodes_in)
    assert len(self.layers)==self.nodes_in
    '''

  #
  #   Методы работы с внешними слоями
  #
  #   Добавить стандартный вход 
  # is_sequence - Да еслина вход подается последовательность данных
  def addInput(self, shape, is_sequence=False):    
    self.layers.append( gen_layer() )
    id = len(self.layers)-1
    assert id < self.nodes_in
    self.layers[id].shape_out = shape
    self.layers[id].neurotype = GInput
    self.is_sequence = (is_sequence or self.is_sequence)
  #   Добавить Embedding вход
  def addInputEmbedding(self,id, len_dict, is_sequence=False):
    assert id < self.nodes_in
    self.is_sequence = is_sequence
    self.layers[id].shape_out = len_dict
    self.layers[id].neurotype = GEmbedding
  #   Определить параметы выходного слоя
  #  out_layer - Выходной слой keras 
  def addOutput(self, shape, out_layer):
    self.__final_layer__.data = out_layer
    self.__final_layer__.shape_out = shape
  #
  #   Сохранение/чтение модели
  #
  #   Сохранить модель в файл
  def save_model_to_file(self,fname):
    json_string = self.model.to_json()
    try:
      f = open(fname, 'w')
      f.write(json_string)
    except:
      print('Ошибка записи файла!')
    finally:
      f.close()
  #   Прочитать модель из файла
  def load_model_from_file(self,fname):
    try:
      f = open(fname)
      json_string = f.read()
      self.model = model_from_json(json_string)
    except:
      print('Ошибка чтения файла!')
    finally:
      f.close()
  #
  #   Методы генерации модели
  #
  # Сборка слоя
  def __build_layer__(self,idx):
    # Соберем входящие тензоры
    conn_in= list()
    layer = self.layers[idx]
    assert not layer.IsInactiveLayer()
    for layer_in in layer.list_in:
        l = self.layers[layer_in]
        assert (not l.IsInactiveLayer())
        if l.connector == None:
          print('Входящие слои должны быть определены ранее: для ',idx,'не определен',layer_in)
          print('Неактивен: ',l.IsInactiveLayer(),layer.list_in,layer)
          return None
        conn_in.append(l.connector)

    return layer.build_layer(conn_in, self.is_sequence)


  #   Собрать сеть из генома
  def synthesis(self):

    self.model = None
    out_idx = 0
    model_in = []
    # Перебор слоев и сборка
    for idx in range(len(self.layers)):      
      layer = self.layers[idx]
      if layer.IsInactiveLayer(): continue
      connector = self.__build_layer__(idx)
      if connector == None:
          prn('Сборка слоя',idx,'завершена с ошибкой.Синтез неудачен.')
          return None

      if len(layer.list_in)==0: model_in.append(connector)
      #if layer.isOut(): 
      out_con = connector
      out_idx = idx
    # Ищем выход и присоединяем его к финалу
    if self.layers[out_idx].NumConnectorOut()>0:
      # Случай когда между выходом и финалом неактивный слой
      self.layers[out_idx].list_out.clear()

    #prn(self.__final_layer__.data.shape)
    try:
      self.__final_layer__.finish(out_con)
      self.__final_layer__.list_in.clear()
      self.__final_layer__.list_in.append(out_idx)
      self.model = Model(model_in, self.__final_layer__.connector)
    except:
      prn('Err: ошибка сборки модели')
    self.get_shape_out() # Проверка выходного слоя
    return self.model


  def get_shape_out(self):
    if self.__final_layer__.connector != None:
      #print(self.__final_layer__.shape_out)
      #print(remove_batch_dim(self.__final_layer__.connector.shape))
      assert remove_batch_dim(self.__final_layer__.connector.shape) == self.__final_layer__.shape_out,'Не совпадают формы модели с заявленой формой add_output()'
    #assert remove_batch_dim(self.__final_layer__.data.shape) == self.__final_layer__.shape_out,'Не совпадают форма выхода и форма слоя заявленые в add_output()'
    return self.__final_layer__.shape_out


  #   Сгенерировать случайную сеть
  def generate(self,max_layers,nodes_in):
    self.nodes_in = nodes_in
    G = neuro_graph()  
    G.generate(max_layers,nodes_in)
    #self.clear()
    for edje in G.edjes:
      self.addConnection(edje[0],edje[1])

    for i in range(len(self.layers)):
      if len(self.layers[i].list_out) == 0:
        self.__final_layer__.list_in.clear()
        self.__final_layer__.list_in.append(i)
    self.synthesis()
    return self.model

  # Извлечь геном из модели
  def sequence(self):
    if self.model == None: 
      print('Нет модели для сиквенции!')
      return []    

    lst = list()
    for idx in range(len(self.layers)):
      #prn('sa',self.layers[idx],idx,self.layers[idx].IsInactiveLayer())
      if not self.layers[idx].IsInactiveLayer():
        lst.extend(self.layers[idx].sequence(idx))
      #if s: lst.extend(s)
    lst.extend(self.__final_layer__.sequence(idx+1))
    return lst

  # Очистить генотип
  def clear_genom(self):
    for l in self.layers:
      l.genom.clear()
    if self.__final_layer__:  
      self.__final_layer__.genom.clear()
  
  # Получить граф из связи слоев
  def get_graph(self):
    edjes = []
    for idx in range(len(self.layers)):
      l = self.layers[idx]
      for i in l.list_in:
        edjes.append((i,idx))        
    if self.__final_layer__:
      edjes.append((self.__final_layer__.list_in[0],idx+1))
    return edjes


  # Извлечь из генома фрагмент от слоя layer_from до layer_to
  #######################? layer_from = -1 значит без input ???
  # changed_only - Фильтрует только измененный геном
  # concat_in_begin - Разрешает конкатенации в начале списка!!! см __link_graph__()
  def __get_part_of_genom__(self, genom, layer_from = 0, layer_to = 0,concat_in_begin = True):
    if layer_to==0: layer_to = 10000
    #if layer_from == -1: layer_from = self.nodes_in
    lst = []#gen for gen in genom if gen.layer_idx >= layer_from and gen.layer_idx < layer_to]
    for gen in genom:
      if gen.layer_idx >= layer_from and gen.layer_idx < layer_to:
        # Предотвращаем появление конкатенаций в начале списка
        if (gen.name == 'Concatenate' or gen.name == 'Flatten') and not concat_in_begin: continue
        else: concat_in_begin = True
        lst.append( gen )
    return lst

  # Получить геном списком
  # mutable_only - вывод только изменяемых параметров
  def get_genom(self,mutable_only=True, changed_only=False):
    lst = []
    for idx in range(len(self.layers)):
      if not self.layers[idx].IsInactiveLayer():
        lst.extend(self.layers[idx].get_genom(mutable_only,changed_only))
    if not mutable_only:
      lst.extend(self.__final_layer__.get_genom(False,changed_only))
    return lst

  # Получить  представление всего генома строкой
  def print_genom(self,genom=None,changed_only=False):
    strn = ""
    if genom == None:
      genom = self.get_genom(False,changed_only)
    for l in genom:
      if not changed_only or l.changed: 
        strn += l.get()+'\n'
    strn += 'Итого признаков: '+str(len(genom))
    return strn

  # Провести мутацию генома
  # proc - процент
  # power - сила мутаций: множитель на вероятностнось смены типа слоя(0.5) и смены связи(0.2)
  def mutate(self,proc,power=1):
#    if len(self.genom) == 0 and (not changed_only):
#      prn('Генотип не заполнен. Вызовите sequence()') 
    self.sequence()
    backup = deepcopy(self.get_genom(False))
    while True:
      try:
        lst = self.get_genom(True)
        rebuild_nodes = set()
        relink_nodes = set()
        #prn(len(lst))
        assert len(lst)>0
        
        #lst_to_mutate = [g for g in lst if 'Concatenate' == g.name and g.var_name == 'inbound_layers']
        lst_to_mutate = random.sample(lst,int(proc*len(lst)))

        for g in lst_to_mutate:
          ret = g.mutate()
          prn('Мутация: Попытка изменения параметра слоя',g.get())
          #layer = self.layers[gen.layer_idx]
          if ret == 2 and random.random()<power*0.5: # Изменение типа слоя
            rebuild_nodes.add(g.layer_idx)
          elif ret == 3 and random.random()<power*0.2: # Изменение связи
            relink_nodes.add(g.layer_idx)
            pass
        lst_to_report = lst_to_mutate.copy()
        for node in rebuild_nodes:
          for i in range(3): 
            ret = self.__build_layer__(node)
            if(ret != None ): break
          if(ret == None): 
            print('ошибка при смене типа слоя:',node,print_genom(lst))
            #prn(self.print_genom(lst_to_report))
            raise
          self.layers[node].sequence(node)
          prn('Мутация: Попытка изменения типа слоя',node,'на',ret)
        #if(ret == None ): continue # Мутация типа не удалась !!!!
        if len(relink_nodes):
          #prn('$$$$$$$$$$$$$$$$$$$$$$$$$$',relink_nodes)
          lst = self.get_genom(False,False)
          nodes_in,nodes_out,nodes_v,edjes = self.__genom_to_graph__(lst)
          #удалим нач и конец из массивов вх/выхода
          nodes = ({edje[0] for edje in edjes} | {edje[1] for edje in edjes})
          #if len(nodes)==0: return None,None
          nodes_in.discard(min(nodes))
          nodes_out.discard(max(nodes))

          #prn('2>',nodes_in,nodes_out,nodes_v,edjes)
          # Выберем и удалим связь
          dismiss_link = []
          for node in relink_nodes:
            if not node in nodes_v:
              print('Нод',node,'не входит в',nodes_v,edjes)
              print(self.print_genom(lst_to_mutate))
              assert False
            node_in = random.sample(self.layers[node].list_in,1)[0]
            dismiss_link.append((node_in,node))
            edjes.remove((node_in,node))
            prn('Мутация: Попытка изменения связи из ',node_in,'в',node)
            #if not node_in in nodes_v
            nodes_out.add(node_in)
            #nodes_in.add()
          #prn('3>',nodes_in,nodes_out,nodes_v,edjes)
          # Перелинкуем граф с удаленной связью
          edjes = self.__link_graph__(edjes,nodes_in,nodes_out,nodes_v,dismiss_link)

          # запишем новую связь в генотип
          for node in relink_nodes:
            gn = [g for g in lst if g.var_name == "inbound_layers" and g.layer_idx==node][0]
            #prn('gen1',gn.get())
            assert gn.var_name == "inbound_layers"
            gn.value = list({edje[0] for edje in edjes if edje[1] == node})
            #prn('gen2',gn.get())
          # Применим новую топологию в слоях
          for idx in range(len(self.layers)):
            layer = self.layers[idx]
            layer.list_in = [edje[0] for edje in edjes if edje[1] == idx]
            layer.list_out = [edje[1] for edje in edjes if edje[0] == idx]
            #prn('layer.list_in',layer.list_in,layer.list_out)
        # Синтезируем новую модель
        #notReady = (self.synthesis() == None)
        if self.synthesis() == None: raise
        else: break
      except:
        # не получилось, откатим мутации и повторим цикл
        #print('backup синтез')
        #lst = deepcopy(backup) 
        self.load_genom(deepcopy(backup))
        #print(self.print_genom(deepcopy(backup)))
        #print('синтез bk')
        ret = self.synthesis()
        if ret==None:
          print(self.print_genom(deepcopy(backup)))
          print('Не смог восстановить исходную сеть после неудачной мутации!!!')
          assert False
        prn('Мутация не удачна. Еще попытка.')

    prn('Мутация завершена успешно.')
    self.sequence()
    return lst_to_report
    
  
  # Создание слоев layers по полному генотипу genom
  def load_genom(self, full_genom):
    if len(full_genom)<=2:
      print('Генотип отсутствует или мал')
      return False
    
    # Сохраним доп инфу
    shape_out = self.__final_layer__.shape_out
    data = self.__final_layer__.data
    # Очистка
    self.model = None
    self.clearModel()
    self.clear_genom()
    self.nodes_in = 0
    # Создание топологии
    layer_idx = -1
    for gen in full_genom:
      if gen.layer_idx != layer_idx:
        '''
        while gen.layer_idx != layer_idx:
          layer = gen_layer()
          self.layers.append(layer)
          if gen.layer_idx - layer_idx>1:
            prn('Warning: добавлен неактивный слой',layer_idx,gen.layer_idx)
          layer_idx += 1 
          

        '''
        layer = gen_layer()
        self.layers.append(layer)
        if len(self.layers) != gen.layer_idx+1:
          prn('Ошибка: Нарушение нумерации генома ',gen.layer_idx)
          prn(self.print_genom(full_genom))
          layer_idx += 1
          continue
          #return False
        layer_idx = gen.layer_idx

        
      if gen.var_name == "inbound_layers":
          edjes = gen.value
          for conn_in in edjes:
            if conn_in>=layer_idx:
              print('Входящая вершина должна уже существовать. ее индекс ',conn_in,' больше индекса текущего слоя ',layer_idx)
              return False
            self.addConnection(conn_in,layer_idx)
      elif gen.var_name == "batch_input_shape":
          layer.shape_out = remove_batch_dim(gen.value)
          layer.genom.append( gen )
      else:
          # Перенос признаков
          layer.genom.append( gen )
      if gen.name == 'InputLayer':
        self.nodes_in += 1
    self.layers[layer_idx-1].list_out.clear()
    assert len(self.__final_layer__.shape_out)>0
    #self.__final_layer__ = self.layers[layer_idx]
    self.layers.pop(layer_idx)#!!!! совместить с предыдущ строкой
    self.__final_layer__.list_in.clear()
    self.__final_layer__.list_in.append(layer_idx-1)    
    # Восстановим доп. инфу
    self.__final_layer__.shape_out = shape_out
    self.__final_layer__.data = data
    #print('asgas',self.print_genom(self.get_genom()))
    return True

  # Извлечение графа из генотипа
  def __genom_to_graph__(self, genom):
    
    if genom==None or len(genom)<=2:

      print('Генотип отсутствует или мал',genom)
      print(len(genom),self.print_genom(genom))
      return None
    # Создание топологии
    valid_nodes = set()
    for gen in genom:
      valid_nodes.add(gen.layer_idx)
    nodes_in = valid_nodes.copy()
    nodes_out = valid_nodes.copy()
    nodes_v = set()
    #prn(nodes_in,nodes_out,valid_nodes)
    edjes = []
    for gen in genom:
      if gen.var_name == "inbound_layers":
          for conn_in in gen.value:
            if conn_in >= gen.layer_idx:
              print('Входящая вершина должна уже существовать. ее индекс ',conn_in,' больше индекса текущего слоя ',layer_idx)
              return None
            if conn_in in valid_nodes and gen.layer_idx in valid_nodes:
              # Добавляем ребро если еще не существует
              if not (conn_in,gen.layer_idx) in edjes:
                edjes.append((conn_in,gen.layer_idx))
              if len(gen.value)>1: nodes_v.add(gen.layer_idx)
              nodes_in.discard(gen.layer_idx)
              nodes_out.discard(conn_in)

    #prn(nodes_in,nodes_out,valid_nodes)
    return nodes_in,nodes_out,nodes_v,edjes

  # Конкатенация родительских графов с перелинковкой
  #   parents - список структур родительских графов от __genom_to_graph__():
  #   (nodes_in,nodes_out,nodes_v,edjes)
  #   Порядок следования графов будет учитываться при построении
  def __join_graph__(self,parents):
    step = 0
    full_graph = list()
    all_node_in = set()
    all_node_out = set()
    all_node_v = set()
    nodedict = dict()
    for idx in range(len(parents)):
      node_in,node_out,node_v,edjes = parents[idx]
      nodes = ({edje[0] for edje in edjes} | {edje[1] for edje in edjes})
      if len(nodes)==0: return None,None
      #состыкуем края и удалим стыки из массивов вх/выхода
      end_graph = max(nodes)
      node_in.discard(min(nodes))
      node_out.discard(end_graph)
      if( idx < len(parents)-1 ): edjes.append((end_graph,end_graph+1))
      #prn('до ',node_in,node_out,node_v,edjes)
      shift = step-min(nodes)
      # Преобразуем вершины
      node_in = [node+shift for node in node_in]
      node_out = [node+shift for node in node_out]
      node_v = [node+shift for node in node_v]
      edjes = [(edje[0]+shift,edje[1]+shift) for edje in edjes]
      #Запомним ссылки на первоначальные вершины
      nodedict.update({(idx,node):node+shift for node in nodes})
      all_node_in.update(node_in)
      all_node_out.update(node_out)
      all_node_v.update(node_v)
      full_graph.extend(edjes)
      #prn('>> ',node_in,node_out,edjes)
      #prn(nodedict)
      step = max(nodes)+shift+1

    #prn(all_node_in,all_node_out,all_node_v)
    full_graph = self.__link_graph__(full_graph,all_node_in,all_node_out,all_node_v)
    #return all_node_in,all_node_out,all_node_v,full_graph,nodedict
    return full_graph,nodedict


  # Замыкание висящих входов/выходов графа
  #   full_graph - список ребер
  #   all_node_in - множество висящих входов
  #   all_node_out - множество висящих выходов
  #   all_node_v - множество вершин-конкатенаций
  #   dismiss_link - список ребер которые нельзя проводить
  def __link_graph__(self,full_graph,all_node_in,all_node_out,all_node_v,dismiss_link = None):
    # Замкнем выходы
    sorted_node_out = list(all_node_out)
    sorted_node_out.sort(reverse=True)
    #prn('замыкание out',sorted_node_out)
    for n_out in sorted_node_out:
      # Замкнем выходы на свободные входы
      valid_in = [n for n in all_node_in if n>n_out]
      if len(valid_in):
        n_in = random.sample(valid_in,1)[0]
        full_graph.append((n_out,n_in))
        #prn('замыкание 1',(n_out,n_in))
        all_node_in.discard(n_in)
      else:
        #Если не получилось подключим к конкатенации
        dismiss_node = []
        if dismiss_link:
          dismiss_node = [n[1] for n in dismiss_link if n[0]==n_out]
          #prn('>>',dismiss_node)
        valid_v = [n for n in all_node_v if n>n_out and not n in dismiss_node]
        #prn('valid_v>>',valid_v)
        if len(valid_v)>0:
          n_in = random.sample(valid_v,1)[0]
          full_graph.append((n_out,n_in))
          #prn('замыкание 3',(n_out,n_in))
        else:
          #ойойой добавить в конец конкат или высушить выход
          # или отказаться
          node = n_out
          FirstSkip = True
          while True: #not (node in all_node_v or node in all_node_in):
            nodes_out = [edje[1] for edje in full_graph if edje[0] == node]
            if len(nodes_out)>0: break
            nodes_in = [edje[0] for edje in full_graph if edje[1] == node]
            if len(nodes_in)==0: break
            elif len(nodes_in)==1:
              if random.random()>.5 or FirstSkip:
                full_graph.remove((nodes_in[0],node))
                #prn('сушим ',(nodes_in[0],node))
                FirstSkip = False
              else:
                valid_v = [n for n in all_node_v if n>n_out]
                if len(valid_in)==0: continue
                n_in = random.sample(valid_v,1)[0]
                full_graph.append((node,n_in))
                break 
            node = nodes_in[0] 

    # Оставшиеся входы замкнем куда попало
    sorted_node_in = list(all_node_in)
    sorted_node_in.sort(reverse=False)
    #prn('замыкание in',sorted_node_in)
    for n_in in sorted_node_in:
        try:
          n_out = random.randint(0,n_in-1)
        except:
          return None
        full_graph.append((n_out,n_in))
        #prn('замыкание 2',(n_out,n_in))
        all_node_in.discard(n_in)
        #all_node_out.discard(n_out)
    
    # Проверим есть ли concatenate с одним входом
    for node in all_node_v:
      nodes_in = [edje[0] for edje in full_graph if edje[1] == node ]
      if len(nodes_in)==1:
        #Есть. Проводим ребро транзитом
        full_graph.remove((nodes_in[0],node))
        nodes_out = [edje[1] for edje in full_graph if edje[0] == node ]
        prn('#Линкер: Обход одиночной конкатенации',node,'в',full_graph)
        for node_out in nodes_out:
          #if node_out in all_node_v:
          #  prn('#Линкер: Конкатенация ссылается на следующую конкатенацию')
          #  prn(full_graph)
          #  return None  !!! это не проблемма
          full_graph.remove((node,node_out))
          edje = (nodes_in[0],node_out)
          if not full_graph.count(edje): full_graph.append(edje)

        #all_node_v.remove(node)!!!!!!
        
      elif len(nodes_in)==0:
        # Конкатенация оказалась первой, не отработал get_genom()?
        prn(full_graph)
        return None
        

    #prn(full_graph)
    return full_graph
  '''
  #Количество слоев в генотипе(без учета __sublayers__)
  #Не подходит для работы с фрагментами сетей так-как не учитывает неактивные ноды
  def getNumLayers(self,genom):
    nodes = set() 
    for gen in genom:
      nodes.add(gen.layer_idx)
    return len(nodes)

  '''
  # Кроссовер до создания удачной модели или достижения лимита кроссовера - публичная версия
  def crossover(self,parent_genom,min_len=.3,min_parents=2):
    #Количество слоев в генотипе(без учета __sublayers__) не учитывает неактивные ноды
    def get_num_layers(genom):
      nodes = set() 
      for gen in genom:
        nodes.add(gen.layer_idx)
      return len(nodes)    
    gt = list()
    res = None
    n_loop = 0
    while res == None:
      gt.clear()
      for idx in range(len(parent_genom)):
        parent = parent_genom[idx]
        n = get_num_layers(parent)
        width = random.randint(int(min_len*n),n-1)
        if idx==0: st = 0
        elif idx+1 == len(parent_genom): st = n-width
        else: st = random.randint(0,n-width)
        prn(idx,'. Выделяем родительский фрагмент:',st,st+width,' из ',n)
        addgt = self.__get_part_of_genom__(parent,st,st+width,False)
        if len(addgt)==0: break
        gt.append( addgt )
      if len(addgt)>0:
        res = self.__crossover__(gt)
      n_loop += 1

      if n_loop > self.max_crossover_loop: 
        prn('Достигнут лимит кросовера',self.max_crossover_loop)
        return None
    return res
  
  # Кросовер из родительского генотипа
  def __crossover__(self,parent_genom):
    Graphs = list()
    # Формируем массив родительских графов
    for parent in parent_genom:
      gn = self.__genom_to_graph__(parent)
      if gn == None: 
        prn('Проблема с генотипом. Кроссовер прошел неудачно')
        return None
      Graphs.append(gn)

    # Конкатенируем родительские графы в порядке следования
    #all_node_in,all_node_out,all_node_v,full_graph 
    full_graph,node_dict = self.__join_graph__(Graphs)
    if full_graph == None:
      prn('Линковка не удалась. Кроссовер прошел неудачно')
      return None

    children_genom = list()
    
    # Проходим по родителям
    for idx in range(len(parent_genom)):
      # Проходим по генам родителя
      for g in parent_genom[idx]:
        # Переводим индекс слоя в его индекс в новом графе
        try:
          layer_idx = node_dict[(idx,g.layer_idx)]
        except:
          print('В словаре вершин ',node_dict,'нет',(idx,g.layer_idx),':',full_graph)#self.print_genom(parent_genom))
          return None
        value = g.value
        # Устанавливаем новые входящие узлы
        if g.var_name == "inbound_layers":
            value = [edje[0] for edje in full_graph if edje[1]==layer_idx]
            #prn(value)
        # Сохраняем
        children_genom.append( gen(layer_idx, g.sublayer_idx, g.name, g.var_name, value) )
    
    self.load_genom(children_genom)
    model = self.synthesis()
    #if model
    #self.genom = children_genom
    return model

  #   Прочитать модель из файла
  def load_model(self,model):
      
    self.clearModel()
    self.model = model
    if len(self.description)==0:
      self.description = 'Загружено из keras модели '+ model.name
    nodes = dict()
    for layer in model.layers:
      ltype = layer.__class__.__name__
      layer_name = layer.name
      #print('qf',type(layer.get_input_at(0)))
      #layer_in_name  = layer.get_input_at(0).name.split('/')[0]
      #print('main',layer_name)
      if 'module_wrapper' in layer_name:
        print('Не могу работать с module_wrapper !!!')
        return False
      if isinstance(layer._inbound_nodes[0].inbound_layers,list):
        list_in = layer._inbound_nodes[0].inbound_layers
        layer_in_name = ''
      else:
        list_in = [layer._inbound_nodes[0].inbound_layers]
        layer_in_name = layer._inbound_nodes[0].inbound_layers.name
      '''
      if isinstance(layer.get_input_at(0),list):
        list_in = layer.get_input_at(0)
        layer_in_name = ''
      else:
        list_in = [layer.get_input_at(0)]
        layer_in_name = layer.get_input_at(0).name
      '''
      if '_input' in layer_in_name:
        newlayer = gen_layer()
        newlayer.shape_out = remove_batch_dim(layer.input_shape)
        newlayer.build_layer([])
        #newlayer.addLayer( layer )
        #newlayer.connector = layer.get_output_at(0)
        self.layers.append( newlayer )
        nodes[newlayer.__sublayers__[0].name] = newlayer
        #print(newlayer.__sublayers__[0].name)

      #print( '>',layer_name )
      if ltype in types_list or ltype == 'InputLayer' or ltype == 'Concatenate':
        newlayer = gen_layer()
        newlayer.addLayer( layer )
        newlayer.connector = layer.get_output_at(0)
        if len(layer.outbound_nodes) == 0: #Финальный слой
          self.__final_layer__ =  newlayer
          newlayer.shape_out = remove_batch_dim(layer.output_shape)
          newlayer.data = layer
        else:
          self.layers.append( newlayer )

        nodes[layer_name] = newlayer

        for idx in range(len(list_in)):
          #print('qf',layer.get_input_at(idx))
          layer_cor_name  = list_in[idx].name.split('/')[0]
          if self.__get_idx__( layer_cor_name )==-1:
            print(layer_name,': Не могу найти входящий слой',layer_cor_name)
            return False

          if '_input' in layer_cor_name:  # Input в модели Sequential
            self.addConnection(0,len(self.layers)-1)
            #print(0,'to ! ',len(self.layers)-1)
          elif 'input' in layer_cor_name and 'input' in layer_name:  # Input в функциональной модели 
            newlayer.shape_out = remove_batch_dim(layer.input_shape)
          else:
            i = self.__get_idx__( layer_cor_name )
            if i != -1: 
              if len(layer.outbound_nodes) == 0:
                self.__final_layer__.addConnectionIn( i )
              else:
                self.addConnection(i,len(self.layers)-1)
                #print(i,'to',len(self.layers)-1)
            #newlayer.addConnectionIn( i )

        '''
        for idx in range(len(layer.outbound_nodes)):
          layer_cor_name  = layer.get_output_at(idx).name.split('/')[0]
          i = self.__get_idx__( layer_cor_name )
          if i != -1: newlayer.addConnectionOut( i )
        '''
      elif ltype == 'Dropout' or ltype == 'BatchNormalization' or ltype == 'Activation':
        #print('drop',layer_in_name)
        layer_in_name  = layer_in_name.split('/')[0]
        if self.__get_idx__( layer_in_name )==-1:
            print(layer_name,': Не могу найти входящий слой',layer_in_name)
            return False        
        nodes[layer_in_name].addLayer( layer )
        nodes[layer_in_name].connector = layer.get_output_at(0)
        '''
        i = self.__get_idx__( layer_in_name )
        if i != -1: 
          self.addConnection(i,len(self.layers)-1)
        '''
      else:
        print('Нет обработчика для слоя ',ltype)
        return False
    #print(nodes)
    return True

  def __get_idx__(self,name):
    for idx in range(len(self.layers)):
      if self.layers[idx].get_idx(name) > -1:
        return idx
    return -1


## Kerasin

#///////////////////////////////////////////////////////////////////////////////
# Класс Генетического алгоритма для поиска оптимальной модели нееросети
#///////////////////////////////////////////////////////////////////////////////

class kerasin:  
  # complexity - сложность сети 1 - на 5 слоев, 2-на 10 и т.п. Количество слоев при генерации ботов точно не соблюдается. max_layer = x5 complexity
  # nPopul - размер популяции
  # maxi_goal - целевая функция метрики. True если максимизация(accuracy), False - минимизация(loss)
  def __init__(self,complexity=1, nPopul=10, maxi_goal = False):
    self.popul = list() # Список популяции
    self.nPopul = nPopul
    self.nFamily = 0
    self.shape_in = None # форма входа(без батч-размерности)
    self.shape_out = None
    self.is_sequence = False  # Поданые данные являются последовательностью?
    self.output_layer = None
    # Ссылки на данные обучения валидации и тестирования
    self.x_train=None
    self.y_train=None
    self.x_val=None
    self.y_val=None
    self.x_test=None
    self.y_test=None
    self.maxi_goal = maxi_goal   # Это задача максимизации?
    self.max_layers = 5*complexity  # Примерное количество генерируемых слоев
    self.pSurv = .5 #Процент победителей в общей популяции
    self.train_generator = None #Ссылка на генератор для fit_generator
    self.profile = None # Имя профиля для записи ботов
    # Управление Генетическим алгоритмом
    # 1. Распределение популяции 'popul_distribution' кортеж: (доля оставляемых чемпионов, доля ботов полученых кросовером,
    #                                доля ботов полученых мутацией, доля случайных ботов). По умолчанию (5,25,25,45)
    # Все доли будут нормированы и приведены к 100%
    self.ga_control = {'popul_distribution': (5,25,25,45)}

  # Генератор имени бота
  def __botname__(self,epoch,num,family):
    return "bot_"+str(epoch).zfill(2)+'.'+str(num).zfill(3)+'('+str(family).zfill(3)+')'
  
  # Сенерировать nPopul случайных ботов
  def generate(self,nPopul=1,epoch=0):
    for idx in range(nPopul):
      self.nFamily += 1
      G=gen_net( self.__botname__(epoch,len(self.popul)+1,self.nFamily), self.nFamily )
      G.addInput(self.shape_in, self.is_sequence)
      G.addOutput( self.shape_out,self.output_layer )
      G.generate(self.max_layers,1)
      self.popul.append( G )
      if self.profile != None:
        G.save(self.profile+'/')
      #print('Создан бот',G.name)

  # Добавить бота из модели
  # model - керас модель
  # name - имя бота
  def add_model(self,model,name=''):
    #self.nFamily += 1
    if name == '': name = model.name
    G=gen_net(name,-1)#self.nFamily)
    G.addOutput(self.shape_out,self.output_layer)
    if G.load_model(model):
      #assert self.shape_in == G.get_shape_in(), 'Форма входа загружаемой модели не совпадает с профилем'
      assert self.shape_out == G.get_shape_out(), 'Форма выхода загружаемой модели не совпадает с профилем'
      # Проверить соответствие  shape !!!!
      self.popul.append( G )
      return True
    return False
    

  # Добавить описание входа моделей
  # shape - форма входа(Без batch размерности) для mnist это может быть (28,28) или (784,)
  # isSequence - указание на то, что в подаваемых данных важна последовательность
  def add_input(self,shape,isSequence = False):
    self.shape_in = shape
    self.is_sequence = isSequence
  # Добавить описание выхода моделей
  # shape - форма выхода. layer = выходной слой керас
  def add_output(self,shape,layer):
    self.output_layer = layer
    self.shape_out = shape
  # Описываем параметры обучения популяции
  def compile(self, optimizer="adam", loss=None, metrics=None ):
    self.loss=loss
    self.optimizer=optimizer
    self.metrics=metrics
    return
  # Определить параметры обучения популяции
  def set_param(self, epochs, batch_size, maxi_goal=False, loss="mse",optimizer="adam",metrics=["accuracy"],verbose=0):
    self.fit_epochs=epochs
    self.batch_size=batch_size
    self.verbose=verbose
    self.loss=loss
    self.optimizer=optimizer
    self.metrics=metrics
    self.maxi_goal = maxi_goal
    return
  # Определить данные для обучения и валидации
  def set_train(self, x_train, y_train, x_val = None, y_val = None):
    self.x_train=x_train
    self.y_train=y_train
    self.x_val=x_val
    self.y_val=y_val
  # Определить данные для тестирования моделей
  def set_test(self, x_test, y_test):
    self.x_test=x_test
    self.y_test=y_test
  # Запуск Эволюции на nEpochs генетических эпох
  # Не путать!:  ga_epochs - количество эпох генетики, 
  #              epoch - количество эпох обучения модели
  def fit( self, 
            ga_epochs=1,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose="auto",
            validation_split=0.0,
            x_val=None,
            y_val=None,
            rescore = False
        ):  
    self.x_train=x
    self.y_train=y
    self.x_val=x_val
    self.y_val=y_val
    self.fit_epochs=epochs
    self.batch_size=batch_size
    self.verbose=verbose

    if self.output_layer == None:
      print('Error: Нет данных о финальном слое. Используйте addOutput()')
      return False
    #if self.x_train == None or self.y_train==None:
    #  print('Нет данных для обучения')
    #  return False
    #self.val_data = None
    if self.profile == None: start_epoch = 0
    else: start_epoch = self.__get_epochs_in_profile__()
    
    for epoch in range(start_epoch,start_epoch+ga_epochs): 
      print('================',epoch+1,'EPOCH OF GA ================')
      if self.__epoch__(epoch+1,(epoch+1-start_epoch)/ga_epochs,rescore): return False
    self.report(True)
    return True

  # Запуск Эволюции на nEpochs генетических эпох c генератором
  # Не путать!:  ga_epochs - количество эпох генетики, 
  #              epoch - количество эпох обучения модели
  def fit_generator( self, 
            ga_epochs=1,
            train_gen = None,
            batch_size=None,
            epochs=1,
            verbose="auto",
            validation_gen = None,
            rescore = False
        ):  
    if train_gen == None:
      print('определите генератор')
      return False
    self.train_generator  = train_gen
    self.validation_generator = validation_gen
    self.fit_epochs=epochs
    self.batch_size=batch_size
    self.verbose=verbose

    if self.output_layer == None:
      print('Error: Нет данных о финальном слое. Используйте addOutput()')
      return False
    
    if self.profile == None: start_epoch = 0
    else: start_epoch = self.__get_epochs_in_profile__()
    
    for epoch in range(start_epoch,start_epoch+ga_epochs): 
      print('================',epoch+1,'EPOCH OF GA ================')
      if self.__epoch__(epoch+1,(epoch+1-start_epoch)/ga_epochs,rescore): return False
    self.report(True)
    return True

  # Задаем имя профиля когда хотим чтобы в каталог с этим именем 
  # выгружались боты после генерации
  def set_profile(self,profile_name,path=''):
    assert self.shape_in != None and self.shape_out != None, 'Сначала определите формы входа и выхода - add_input()/add_output() !'
    self.profile = path+profile_name+str(self.shape_in)+'-'+str(self.shape_out)

  # Задаем имя профиля когда хотим чтобы в каталог с этим именем 
  # выгружались боты после генерации
  # nSurv - взять nSurv лучших, (0 - всех)
  # include_unscored - включать всех у кого нет оценки
  def show_profile(self,profile_name=None,nSurv=0,include_unscored=True):
    if profile_name == None:profile_name = self.profile
    if profile_name == None:
      print('Профиль не задан')
      return None
    if not ')-(' in profile_name: profile_name += str(self.shape_in)+'-'+str(self.shape_out)
    g = gen()
    bot_scored_list = []
    bot_unscored_list = []
    for root, dirs, files in os.walk(self.profile+'/'):
      for filename in files:
        if not '.gn' in filename: continue
        #print(filename)
        file = open(self.profile+'/'+filename, 'r')
        lines = file.readlines()
        
        lst = [filename.replace('.gn',''),self.profile+'/'+filename,0,0,'',None,None,None]
        for line in lines:
          if not g.load_csv(line.strip()): continue
          if g.layer_idx == 9999:
            if g.var_name == 'description':              
              lst[4] = str(g.value)
            elif g.var_name == 'family':
              lst[3] = int(g.value)
            elif g.var_name == 'score':
              lst[2] = float(g.value)
        if lst[2]>0:
          bot_scored_list.append(tuple(lst))
        elif include_unscored:
          bot_unscored_list.append(tuple(lst))
        file.close()
    bot_scored_list.sort( key=lambda x:x[2], reverse=self.maxi_goal)   # сортируем по оценке
    if nSurv>0:
      while len(bot_scored_list) >= nSurv: bot_scored_list.pop()
    bot_scored_list.extend(bot_unscored_list)
    return bot_scored_list

  # Считаем количество пройденых эпох по ботам в профиле
  def __get_epochs_in_profile__(self,profile_name=None):
    if profile_name == None:profile_name = self.profile+'/'
    if profile_name == None:
      print('Профиль не задан')
      return None
    if not ')-(' in profile_name: profile_name += str(self.shape_in)+'-'+str(self.shape_out)
    max_epoch = 0
    max_family = 0
    for root, dirs, files in os.walk(self.profile+'/'):
      for filename in files:
        if not '.gn' in filename: continue
        #print('FA',filename[4:6])
        try:
          max_epoch = max(max_epoch,int(filename[4:6]))
          max_family = max(max_family,int(filename[11:14]))
        except:
          pass # вероятно имя задано вручную
    print('Обнаружено GA эпох',max_epoch,'. Фамилий:',max_family)
    self.nFamily = max_family
    return max_epoch

  # Загрузить бота с именем bot_name из профиля в популяцию
  # Возвращает индекс бота в популяции
  def load_bot(self, bot_name):
    # Проверим, возможно бот уже в популяции
    idx = self.get_index(bot_name)
    if idx>-1: 
      print('Бот с именем',bot_name,'- уже в популяции. Не загружаю.')
      if self.popul[idx].get_family() == -1:
        print('Предотвращена попытка загрузки посевного бота с одинаковым именем',bot_name)
        return None
      return idx
    bot = gen_net('',0)
    bot.addOutput(self.shape_out,self.output_layer)
    if bot.load(bot_name,self.profile+'/'):
      self.popul.append(bot)
      #print('Загружен бот ',bot_name,'с оценкой',bot_score)
      return len(self.popul)-1
    return None

  # Возвращает индекс бота в популяции
  # -1 если бота в текущей популяции нет
  def get_index(self, bot_name):
    for idx in range(len(self.popul)):
      if self.popul[idx].name == bot_name: return idx
    return -1

  # Возвращает имя бота по индексу
  def get_bot_name(self, bot_idx = 0):
    if bot_idx >= len(self.popul): raise KeyError('Индекс больше размера популяции')
    return self.popul[bot_idx].name

  # Процедура одной генетической эпохи
  def __epoch__(self,epoch,progress,rescore = False):
    def getBotInExpovariate(lmb,excluse_bots={}):
      # Исключаем близкородственные скрещивания
      excluse_family = set()
      for i in excluse_bots:
        excluse_family.add(self.popul[i].get_family())
      for i in range(100):
        nBot = int(random.expovariate(lmb))
        if not nBot in excluse_bots and nBot<len(self.popul) and (not self.popul[nBot].get_family() in excluse_family): return nBot
      # Достигнут лимит
      return 0
    def get_qty_of_popul(idx):
      qty = self.ga_control['popul_distribution'][idx]/sum(self.ga_control['popul_distribution'])*self.nPopul
      if qty>0 and int(qty)==0: return 1
      return int(qty)

    # Это первый проход?
    # В первом проходе в популяции могут быть только боты посева
    first_pass = sum([1 for b in self.popul if b.get_family()!=-1])==0

    nSurv = self.pSurv*self.nPopul    
    #print(self.show_profile(self.profile+'/'))
    if first_pass and self.profile != None:
      # Из фонда получаем nSurv лучших ботов плюс всех не прошедших оценку
      for bot_name,bot_file,bot_score,_,_,_,_,_ in self.show_profile(self.profile+'/',nSurv,True):
        if self.load_bot(bot_name) != None:
          print('Загружен бот ',bot_name,'с оценкой',bot_score)
        else:
          return True
      print('Загружено ',len(self.popul),'ботов профиля',self.profile)

    # На этом этапе боты посева имеют фамилию -1. Исправим это
    for bot in self.popul:
      if bot.get_family() == -1:
        self.nFamily += 1
        bot.set_family(self.nFamily)

    # Пополнение популяции случайными представителями
    nRndBots = self.nPopul-len(self.popul)
    if nRndBots>0: 
      self.generate(nRndBots,epoch)
      print('Сгенерированы ',nRndBots,'случайных ботов')
    
    # Тестируем популяцию
    new_popul = []
    for idx in range(len(self.popul)):
      bot = self.popul[idx]
      if bot.score == -1 or rescore:
        model = bot.model
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)
        start_time = time.time()
        if self.train_generator:
          try:
            train_steps = self.train_generator.samples // self.batch_size
            val_steps = self.validation_generator.samples // self.batch_size
          except:
            train_steps = None
            val_steps = None
          history = model.fit_generator( self.train_generator, steps_per_epoch = train_steps, validation_data = self.validation_generator, validation_steps = val_steps, epochs=self.fit_epochs, verbose=self.verbose)
        else:
          history = model.fit( self.x_train, self.y_train , batch_size=self.batch_size, epochs=self.fit_epochs, validation_data=(self.x_val,self.y_val),verbose=self.verbose)
        bot.hist = history.history
        bot.train_duration = int(time.time()-start_time)
        if self.maxi_goal:
          bot.score = max(bot.hist['val_accuracy'])
        else:
          bot.score = min(bot.hist['val_loss'])
        if self.x_test!=None and self.y_test!=None:
          bot.score = model.evaluate(self.x_test,self.y_test,verbose=0)
        if self.profile != None: bot.save(self.profile+'/')
        #bot.sequence()
      print(str(idx+1).zfill(2),bot.name,' Оценка =',round(bot.score,7),'за',bot.train_duration,'cек.',bot.description)
      #bot.SetScore(scores,hist)
    
    
    #for bot in self.popul:
        

    # Отбираем победителей
    self.popul.sort( key=lambda bot: bot.score, reverse=self.maxi_goal)   # сортируем по оценке
    self.report()
    if progress == 1: return True
    # Оставляем лучших
    for idx in range(get_qty_of_popul(0)):
      new_popul.append(self.popul[idx])
    print('Оставляем',len(new_popul),'лучших ботов предыдущей популяции')
    # Кроссовер 
    for idx in range(get_qty_of_popul(1)):
      res=None
      while res==None:
        # Выбираем родителей по экспотенциальному распределению. #(параметр - чуть выше вероятности выбора 0-ого элемента, остальные по нисходящей)
        parents = []
        desc = ''
        #family_idx = 10000000
        for i in range(2):#2-родителей   random.randint())
          parent = getBotInExpovariate(.2,parents)
          parents.append(parent)
          self.popul[parent].sequence()
          desc += self.popul[parent].name+','
          #family_idx = min(parent,family_idx)
        family = self.popul[min(parents)].get_family()
        bot=gen_net( self.__botname__(epoch+1,len(new_popul)+1,family), family )
        #bot.addInput(self.shape_in, self.is_sequence)
        bot.addOutput(self.shape_out,self.output_layer) 
        res = bot.crossover([self.popul[parent].get_genom(False) for parent in parents])
        if not res: print('неудачный кросовер от',parents)
        # Если кросовер не удачен с этими родителями, идем за следующими

      #bot.mutate(random.random()*.2)
      new_popul.append(bot)
      bot.description = "Потомок "+desc+';'+ bot.description
      print('Бот',bot.name,'потомок ботов',parents)
    # Мутации другой половины победителей
    for i in range(get_qty_of_popul(2)):
      nBot = getBotInExpovariate(.2)
      #print('dzshae',nBot)
      bot = self.popul[nBot].copy()
      if bot:
        bot.name = self.__botname__(epoch+1,len(new_popul)+1, bot.get_family())
        bot.description = "Мутант от "+self.popul[nBot].name+';'+ bot.description
        bot.mutate(random.random()*.5)
        new_popul.append(bot)
        print('Бот',bot.name,'мутировал из бота',nBot)
      else:
        print('От бота',self.popul[nBot].name,'мутация не удалась')
    # Удаляем с аутсайдеров
    #for idx in range(nPopul-nSurv): self.popul.pop()
    self.popul = new_popul
    return False
    
  # Отчет по эпохе
  # best_detail - добавляем подробное описание чемпиона
  def report(self,scoreboard=True,best_detail=False):
    if scoreboard:
      nSurv = int(self.pSurv*len(self.popul))
      print('--------------------------')
      for idx in range(nSurv):
        bot = self.popul[idx]
        print(idx+1,':',bot.name,'-',round(bot.score,5),bot.description)
      print('--------------------------')
    if best_detail:
      best = self.popul[0]
      print('*************** BEST MODEL ****************')
      print('Бот:',best.name,':',best.description)
      best.model.summary()
      #print(best.hist)
      print('------------ Геном -----------')
      gn = best.sequence()
      print(best.print_genom(gn))
      plt.title('Loss Plot')
      plt.plot(best.hist['loss'],label='LOSS')
      plt.plot(best.hist['val_loss'],label='VAL_LOSS')
      plt.xlabel('Эпохи')
      plt.ylabel('Значение')
      plt.legend()
      plt.show()
      if 'accuracy' in best.hist: 
        plt.title('Accuracy Plot')
        plt.plot(best.hist['accuracy'],label='ACCURACY')
        plt.plot(best.hist['val_accuracy'],label='VAL_ACCURACY')
        plt.xlabel('Эпохи')
        plt.ylabel('Значение')
        plt.legend()
        plt.show()
   
  # Получить керас модель от idx бота популяции
  def get_model(self,idx=0):
    return self.popul[idx].model

  # Оценка качества текущей популяции на устойчивость к объему данных, 
  # переобученность и зависимость от количества эпох 
  def evaluate(self):
    pass
