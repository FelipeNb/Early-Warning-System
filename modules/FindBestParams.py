import pandas as pd
import csv
import numpy as np
from datetime import datetime
import random
import time as tm
from multiprocessing import Pool
import requests
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from math import sqrt
import sys


def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))


def mae(y_actual, y_predicted):
    return mean_absolute_error(y_actual, y_predicted)


def cm(y_actual, y_predicted):
    tn, fp, fn, tp = confusion_matrix(y_actual, y_predicted).ravel()
    return tn, fp, fn, tp


def precision(y_actual, y_predicted):
    tn, fp, fn, tp = cm(y_actual, y_predicted)
    if((tp + fp) == 0):
        return 0.0
    return float(tp)/(tp+fp)


def recall(y_actual, y_predicted):
    tn, fp, fn, tp = cm(y_actual, y_predicted)
    if((tp+fn) == 0):
        return 0.0
    return float(tp)/(tp+fn)


def fmeasure(y_actual, y_predicted):
    if precision(y_actual, y_predicted) == 0.0 or recall(y_actual, y_predicted) == 0.0:
        return 0.0
    return 2 * ((precision(y_actual, y_predicted)*recall(y_actual, y_predicted))/(precision(y_actual, y_predicted)+recall(y_actual, y_predicted)))


central = {'central': [], 'other': []}
right = []
single = {
    'decision_tree': {'decision_tree': [], 'other': []},
    'deep_learning': {'deep_learning': [], 'other': []},
    'logistic_regression': {'logistic_regression': [], 'other': []},
    'svm': {'svm': [], 'other': []},
    'knn': {'knn': [], 'other': []},
    'random_forest': {'random_forest': [], 'other': []},
}


port = str(sys.argv[1])
date = str(sys.argv[2])

df = pd.read_csv('Datasets/test/test_SBIE_'+date+'.csv')

dt = []
names = []
for j in range(len(df.index)):
    row = (str(df['class'][j]), {
        'total': str(df['total'][j]),
        'algum conteudo foi publicado': str(df['algum conteudo foi publicado'][j]),
        'assinatura criada': str(df['assinatura criada'][j]),
        'assinatura de discussao criada': str(df['assinatura de discussao criada'][j]),
        'comentario apagado': str(df['comentario apagado'][j]),
        'comentario criado': str(df['comentario criado'][j]),
        'curso visto': str(df['curso visto'][j]),
        'discussao visualizada': str(df['discussao visualizada'][j]),
        'discussao criada': str(df['discussao criada'][j]),
        'formulario de confirmacao de submissao visualizado': str(df['formulario de confirmacao de submissao visualizado'][j]),
        'formulario de submissao visualizado': str(df['formulario de submissao visualizado'][j]),
        'lista de instancias de modulos de cursos visualizados': str(df['lista de instancias de modulos de cursos visualizados'][j]),
        'lista de usuarios vistos': str(df['lista de usuarios vistos'][j]),
        'modulo do curso visualizado': str(df['modulo do curso visualizado'][j]),
        'o status da submissao foi visualizado': str(df['o status da submissao foi visualizado'][j]),
        'o usuario aceitou o acordo da tarefa': str(df['o usuario aceitou o acordo da tarefa'][j]),
        'o usuario salvou um envio': str(df['o usuario salvou um envio'][j]),
        'perfil do usuario visto': str(df['perfil do usuario visto'][j]),
        'post atualizado': str(df['post atualizado'][j]),
        'post criado': str(df['post criado'][j]),
        'relatorio de notas do usuario visualizado': str(df['relatorio de notas do usuario visualizado'][j]),
        'relatorio de um utilizador do curso visualizado': str(df['relatorio de um utilizador do curso visualizado'][j]),
        'submissao criada': str(df['submissao criada'][j]),
        'um arquivo foi enviado': str(df['um arquivo foi enviado'][j]),
        'um envio foi submetido': str(df['um envio foi submetido'][j]),
        'visualizado relatorio de usuario': str(df['visualizado relatorio de usuario'][j]),
    }
    )
    dt.append(row)
    names.append(df['name'][j])

# 0 - true negative
# 1 - true positive
# 2 - false negative
# 3 - false positive
with open('TestResults/who_Will_be_recommended.csv', mode='a') as write_file:
    write_writer = csv.writer(write_file, delimiter=',')
    for idx, i in enumerate(dt):
        r = requests.post('http://127.0.0.1:' + port +
                        '/Agent_central', data=i[1])  # .json()
        r = r.json()
        central['other'].append(1 if r['recommend'] == 'inactive' else 0)
        for method, arr in r['lista'].items():
            single[str(method)]['other'].append(1 if arr['class'] == 'inactive' else 0)
        right.append(1 if i[0] == 'inactive' else 0)
        write_writer.writerow([names[idx],'notify' if i[0] == 'inactive' else 'do nothing',date])


with open('TestResults/test_result_SBIE_' + date + '.csv', mode='w') as write_file:
    write_writer = csv.writer(write_file, delimiter=',')
    write_writer.writerow([
        'Method',
        'precision',
        'recall',
        'fmeasure',
        'rmse',
        'mae',
    ])
    write_writer.writerow([
        'central',
        precision(right, central['other']),
        recall(right, central['other']),
        fmeasure(right, central['other']),
        rmse(right, central['other']),
        mae(right, central['other']),
    ])
    for method, lista in single.items():
        write_writer.writerow([
            method,
            precision(right, lista['other']),
            recall(right, lista['other']),
            fmeasure(right, lista['other']),
            rmse(right, lista['other']),
            mae(right, lista['other']),
        ])

print(date)
print('DONE!!!!')
