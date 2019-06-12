import pandas as pd
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from numpy import *
import itertools


base = '../Datasets/train/'

files_FINAL = [
    'SBIE_2018-12-30.csv',
]

files = [
    'train_SBIE_2018-12-30.csv'
]

dic = {
    'location': files,
    'delete_columns': ['name'],
    'change_to_int': [''],
    'change_to_float': [''],
    'what_is_exp': ['total-algum conteudo foi publicado-assinatura criada-assinatura de discussao criada-comentario apagado-comentario criado-curso visto-discussao visualizada-discussao criada-formulario de confirmacao de submissao visualizado-formulario de submissao visualizado-lista de instancias de modulos de cursos visualizados-lista de usuarios vistos-modulo do curso visualizado-o status da submissao foi visualizado-o usuario aceitou o acordo da tarefa-o usuario salvou um envio-perfil do usuario visto-post atualizado-post criado-relatorio de notas do usuario visualizado-relatorio de um utilizador do curso visualizado-submissao criada-um arquivo foi enviado-um envio foi submetido-visualizado relatorio de usuario'],
    'classificatory_data': ['class'],
    'regularization_log_reg': [1],
    'solver_log_reg': [
         'lbfgs',
        'liblinear',
    ],
    'multiclass_log_reg': [
         'ovr',
         'multinomial',
         'auto'
    ],
    'random_state_log_reg': ['none'],
    'regularization_svm': [1],
    'kernel_svm': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed', 'callable'],
    'gamma_svm': range(1,101, 10),
    'random_state_svm': ['none'],
    'criterion_decision_tree': [
        'entropy'
    ],
    'max_depth_decision_tree': ['none'],
    'random_state_decision_tree': ['none'],
    'criterion_random_forest': [
        'entropy'
    ],
    'n_estimators_random_forest': [60],
    'n_processors_random_forest': [-1],
    'random_state_random_forest': ['none'],
    'type_distance_knn': [1],
    'n_neighbours_knn': [3],
    'metric_knn': [
        'euclidean',
    ],
    'loss_deep': [
        'binary_crossentropy',
    ],
    'metric_deep': [
        'accuracy',
    ],
    'optmizer_deep': [
        'Adam',
         'Nadam',
         'Optimizer',
        'SGD',
    ],
    'activation_deep': [
        'elu',
         'linear',
        'relu',
        'sigmoid',
        'softmax',
    ],
    'activation_output_deep': [
        'elu',
         'linear',
        'relu',
        'sigmoid',
        'softmax',
    ],
}
with open('TestFitBestParams.csv', mode='w') as write_file:
	write_writer = csv.writer(write_file, delimiter=',')
	write_writer.writerow([
		'location',
		'delete_columns',
		'change_to_int',
		'change_to_float',
		'what_is_exp',
		'classificatory_data',
		'regularization_log_reg',
		'solver_log_reg',
		'multiclass_log_reg',
		'random_state_log_reg',
		'regularization_svm',
		'kernel_svm',
		'gamma_svm',
		'random_state_svm',
		'criterion_decision_tree',
		'max_depth_decision_tree',
		'random_state_decision_tree',
		'criterion_random_forest',
		'n_estimators_random_forest',
		'n_processors_random_forest',
		'random_state_random_forest',
		'type_distance_knn',
		'n_neighbours_knn',
		'metric_knn',
		'loss_deep',
		'metric_deep',
		'optmizer_deep',
		'activation_deep',
		'activation_output_deep'
])
	for x in itertools.product(
		dic['location'],
		dic['delete_columns'],
		dic['change_to_int'],
		dic['change_to_float'],
		dic['what_is_exp'],
		dic['classificatory_data'],
		dic['regularization_log_reg'],
		dic['solver_log_reg'],
		dic['multiclass_log_reg'],
		dic['random_state_log_reg'],
		dic['regularization_svm'],
		dic['kernel_svm'],
		dic['gamma_svm'],
		dic['random_state_svm'],
		dic['criterion_decision_tree'],
		dic['max_depth_decision_tree'],
		dic['random_state_decision_tree'],
		dic['criterion_random_forest'],
		dic['n_estimators_random_forest'],
		dic['n_processors_random_forest'],
		dic['random_state_random_forest'],
		dic['type_distance_knn'],
		dic['n_neighbours_knn'],
		dic['metric_knn'],
		dic['loss_deep'],
		dic['metric_deep'],
		dic['optmizer_deep'],
		dic['activation_deep'],
		dic['activation_output_deep']
		):
			write_writer.writerow([
				base + x[0],
				x[1],
				x[2],
				x[3],
				x[4],
				x[5],
				x[6],
				x[7],
				x[8],
				x[9],
				x[10],
				x[11],
				x[12],
				x[13],
				x[14],
				x[15],
				x[16],
				x[17],
				x[18],
				x[19],
				x[20],
				x[21],
				x[22],
				x[23],
				x[24],
				x[25],
				x[26],
				x[27],
				x[28]
			])

