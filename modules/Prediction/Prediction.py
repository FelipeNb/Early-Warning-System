####################
# AUTOR: Felipe Neves Braz
####################
from numpy.random import seed
import numpy as np
import pandas as pd
import keras
# import matplotlib.pyplot as plt
from SBS import SBS
from io import StringIO
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD 
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# from matplotlib.colors import ListedColormap
from settings import *
from imblearn.over_sampling import SMOTE

# location = '../DatasetsTest/chronic-kidney-disease-full.csv'
# delete_columns = ['ex']
# change_to_int = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class']
# rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane,class
# change_to_float = ['sc','sg','hemo', 'pot', 'rbcc']
# sc,sg,hemo,pot,rbcc
# what_is_exp = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane']
# age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wbcc,rbcc,htn,dm,cad,appet,pe,ane
# classificatory_data = 'class'


location = None
delete_columns = None
change_to_int = None
change_to_float = None
what_is_exp = None
classificatory_data = None


regularization_log_reg = None
solver_log_reg = None
multiclass_log_reg = None
random_state_log_reg = None

regularization_svm = None
kernel_svm = None
gamma_svm = None
random_state_svm = None

criterion_decision_tree = None
max_depth_decision_tree = None
random_state_decision_tree = None

criterion_random_forest = None
n_estimators_random_forest = None
n_processors_random_forest = None
random_state_random_forest = None

type_distance_knn = None
n_neighbours_knn = None
metric_knn = None

loss_deep = None
metric_deep = None
optmizer_deep = None
activation_deep = None
activation_output_deep = None

percentage = None
labelencoder_X_1 = None

def initialize_all_parameters(file):
	df_settings = pd.read_csv('../Prediction/' + str(file))
	df_settings = df_settings.fillna(-9999)

	training_settings = getTrainingSettings(df_settings)
	configuration_models = getConfigurationModels(df_settings)

	global location
	global delete_columns
	global change_to_int
	global change_to_float
	global what_is_exp
	global classificatory_data
	global regularization_log_reg
	global solver_log_reg
	global multiclass_log_reg
	global random_state_log_reg
	global regularization_svm
	global kernel_svm
	global gamma_svm
	global random_state_svm
	global criterion_decision_tree
	global max_depth_decision_tree
	global random_state_decision_tree
	global criterion_random_forest
	global n_estimators_random_forest
	global n_processors_random_forest
	global random_state_random_forest
	global type_distance_knn
	global n_neighbours_knn
	global metric_knn
	global loss_deep
	global metric_deep
	global optmizer_deep
	global activation_deep
	global activation_output_deep
	global percentage

	location = training_settings['location']
	delete_columns = training_settings['delete_columns']
	change_to_int = training_settings['change_to_int']
	change_to_float = training_settings['change_to_float']
	what_is_exp = training_settings['what_is_exp']
	classificatory_data = training_settings['classificatory_data']
	regularization_log_reg = configuration_models['regularization_log_reg']
	solver_log_reg = configuration_models['solver_log_reg']
	multiclass_log_reg = configuration_models['multiclass_log_reg']
	random_state_log_reg = configuration_models['random_state_log_reg']
	regularization_svm = configuration_models['regularization_svm']
	kernel_svm = configuration_models['kernel_svm']
	gamma_svm = configuration_models['gamma_svm']
	random_state_svm = configuration_models['random_state_svm']
	criterion_decision_tree = configuration_models['criterion_decision_tree']
	max_depth_decision_tree = configuration_models['max_depth_decision_tree']
	random_state_decision_tree = configuration_models['random_state_decision_tree']
	criterion_random_forest = configuration_models['criterion_random_forest']
	n_estimators_random_forest = configuration_models['n_estimators_random_forest']
	n_processors_random_forest = configuration_models['n_processors_random_forest']
	random_state_random_forest = configuration_models['random_state_random_forest']
	type_distance_knn = configuration_models['type_distance_knn']
	n_neighbours_knn = configuration_models['n_neighbours_knn']
	metric_knn = configuration_models['metric_knn']
	loss_deep = configuration_models['loss_deep']
	metric_deep = configuration_models['metric_deep']
	optmizer_deep = configuration_models['optmizer_deep']
	activation_deep = configuration_models['activation_deep']
	activation_output_deep = configuration_models['activation_output_deep']
	percentage = 0.10


def df_conj_x_y_expected(deep_learning = None):
	df = pd.read_csv(location)
	for i in delete_columns:
		df.drop(i,1,inplace=True)
	type_str =  change_to_int
	df, conj = df, {}
	df, conj = handle_data(df, tc= type_str, tc_float = change_to_float)
	what_is_expected = what_is_exp
	# print(df)
	if deep_learning == True:
		x, y = np.array(df.drop([classificatory_data], 1).values),np.array(df[classificatory_data].values)
	else:
		x, y = np.array(df.drop([classificatory_data], 1)),np.array(df[classificatory_data])
	global labelencoder_X_1
	return df, conj, x, y, what_is_expected, type_str, labelencoder_X_1

def split_data(x, y, percent=percentage):
	# x,y = SMOTE().fit_resample(x, y)
	x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=percent)
	return x_train, x_test,y_train, y_test

def convert_to_numbers(df, to_change=[]):
	conj = {}
	for i in to_change:
		global labelencoder_X_1
		labelencoder_X_1=LabelEncoder()
		df[i]=labelencoder_X_1.fit_transform(df[i])
		t = list(labelencoder_X_1.classes_)
		s = labelencoder_X_1.transform(t)
		conj[i] = dict(zip(t,s))
	return df, conj

def handle_data(df, tc=[], tc_float=[]):
	df, conj = convert_to_numbers(df,to_change= tc)
	df.replace('?',-99999, inplace=True)
	df.fillna(df.mean())
	if tc_float != []:
		df[tc_float].astype(np.float64)
	return df, conj

def print_importance_features(df, x_train,y_train):
	feat_labels = df.columns[:]
	forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
	forest.fit(x_train, y_train)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]
	print("Importances:")
	for f in range(x_train.shape[1]):
		print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))

#####################   Stardatization   #######################

def stardatization(trainData, trainTest):
	sc = StandardScaler()
	sc.fit(trainData)
	x_train_std = sc.transform(trainData)
	x_test_std = sc.transform(trainTest)
	return x_train_std, x_test_std

def Logistic_Regression_Stardatization(x_train, y_train,x_test, y_test, C=1000.0, multi_class='auto', random_state=0, plot=False, xLabel='', yLabel='', position= 'upper left'):

	x_train_std, x_test_std = stardatization(x_train, x_test)
	
	lr = LogisticRegression(C=regularization_log_reg,solver='liblinear', multi_class=multi_class, random_state=random_state)
	lr.fit(x_train_std, y_train)
	
	x_combined_std = np.vstack((x_train_std, x_test_std))
	y_combined = np.hstack((y_train, y_test))

	if plot:
		plot_with_regions(x= x_combined_std,
						  y= y_combined,
						  classifier= lr,
						  test_idx= range(105,150),
						  xLabel=xLabel,
						  yLabel=yLabel,
						  position=position)
	y_pred = lr.predict(x_test_std)
	accuracy = accuracy_score(y_test, y_pred)
	return lr, accuracy

def Perceptron_Stardatization(x_train, y_train,x_test, y_test, max_iter=40, eta0= 0.0001, random_state=0, plot=False, xLabel='', yLabel='', position= 'upper left'):

	x_train_std, x_test_std = stardatization(x_train, x_test)

	ppn = Perceptron(max_iter=40, eta0=0.0001, random_state=0) # n_iter is deprecated
	ppn.fit(x_train_std, y_train)

	x_combined_std = np.vstack((x_train_std, x_test_std))
	y_combined = np.hstack((y_train, y_test))
	
	if plot:
		plot_with_regions(x= x_combined_std,
						  y= y_combined,
						  classifier= ppn,
						  test_idx= range(105,150),
						  xLabel=xLabel,
						  yLabel=yLabel,
						  position=position)

	y_pred = ppn.predict(x_test_std)
	accuracy = accuracy_score(y_test, y_pred)
	return ppn, accuracy

#####################   Stardatization   #######################
######################################################
#####################   Normal   ######################

def func_Logistic_Regression(x_train, y_train,x_test, y_test, C=1000.0, multi_class='auto', random_state=0, plot=False, xLabel='', yLabel='', position= 'upper left'):
	lr = LogisticRegression(C=regularization_log_reg,solver=solver_log_reg, multi_class=multiclass_log_reg, random_state=random_state_log_reg)
	lr.fit(x_train, y_train)
	
	if plot:
		plot_with_regions(x= x_train,
						  y= y_train,
						  classifier= lr,
						  test_idx= range(105,150),
						  xLabel=xLabel,
						  yLabel=yLabel,
						  position=position)
	y_pred = lr.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	return lr, accuracy

def func_SVM(x_train, y_train,x_test, y_test, type= 'sigmoid',gamma=100, C=1.0, random_state=0, plot=False, xLabel='', yLabel='', position='upper left'):
	# type can be:
	# 	linear
	# 	rbf
	# 	sigmoid
	# 	poli
	# 	precomputed
	# 	callable

	svm = SVC(kernel=str(kernel_svm),gamma=gamma_svm, C=regularization_svm, random_state=random_state_svm,probability=True)
	svm.fit(x_train, y_train)

	if plot:
		plot_with_regions(x= x_train,
						  y= y_train,
						  classifier= svm,
						  test_idx= range(105,150),
						  xLabel=xLabel,
						  yLabel=yLabel,
						  position=position)

	y_pred = svm.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	return svm, accuracy

def func_Decision_Tree(x_train, y_train,x_test, y_test, type='entropy', max_depth=3, random_state=0, plot=False, xLabel='', yLabel='', position= 'upper left'):	
	tree = DecisionTreeClassifier(criterion=criterion_decision_tree, max_depth=max_depth_decision_tree, random_state=random_state_decision_tree)
	tree.fit(x_train, y_train)

	if plot:
		x_combined = np.vstack((x_train, x_test))
		y_combined = np.hstack((y_train, y_test))
		plot_with_regions(x= x_combined,
						  y= y_combined,
						  classifier= tree,
						  test_idx= range(105,150),
						  xLabel=xLabel,
						  yLabel=yLabel,
						  position=position)

	y_pred = tree.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	return tree, accuracy

	# export_graphviz(tree,out_file='tree.dot',feature_names=[xLabel,yLabel])

	#dot -Tpng tree.dot -o tree.png
	#in command line on terminal

def func_Random_Forest(x_train, y_train,x_test, y_test,type='entropy', n_estimators=10, random_state=0, n_jobs=2, plot=False, xLabel='', yLabel='', position= 'upper left'):
	forest = RandomForestClassifier(criterion=criterion_random_forest, n_estimators=n_estimators_random_forest, random_state=random_state_random_forest, n_jobs=n_processors_random_forest)
	forest.fit(x_train, y_train)

	if plot:
		plot_with_regions(x= np.vstack((x_train, x_test)),
						  y= np.vstack((y_train, y_test)),
						  classifier= forest,
						  test_idx= range(105,150),
						  xLabel=xLabel,
						  yLabel=yLabel,
						  position=position)

	y_pred = forest.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	return forest, accuracy

def func_Knn(x_train, y_train,x_test, y_test,n_neighbors=5, p=2,type='minkowski', plot=False, xLabel='', yLabel='', position= 'upper left'):
	knn = KNeighborsClassifier(n_neighbors=n_neighbours_knn, p=type_distance_knn,metric=metric_knn)
	knn.fit(x_train, y_train)
	
	if plot:
		x_combined = np.vstack((x_train, x_test))
		y_combined = np.hstack((y_train, y_test))
		plot_with_regions(x= x_combined,
						  y= y_combined,
						  classifier= knn,
						  test_idx= range(105,150),
						  xLabel=xLabel,
						  yLabel=yLabel,
						  position=position)

	y_pred = knn.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	return knn, accuracy

#####################   Normal   #######################
######################################################
#####################   with SGD   #######################

def func_SGDClassifier(x_train, y_train,x_test, y_test, type='log'):
	# type can be:
	# 	perceptron
	# 	log
	# 	hinge

	cl = SGDClassifier(loss='perceptron')
	cl.fit(x_train, y_train)

	return cl, accuracy_score(y_test, cl.predict(x_test))

#####################   with SGD   #######################


#####################   Deep Learning   #######################

def construct_model(dic, classifier):

	# set the input layer
	classifier.add(Dense(dic['n_nodes']['n_nodes_hl1'], activation=dic['activation_hidden'], input_shape= (dic['n_nodes_input'],)))
	
	# construct hidden layers
	for i in range(len(dic['n_nodes'])-1):
		classifier.add(Dense(dic['n_nodes']['n_nodes_hl'+str(i+2)], activation=dic['activation_hidden']))
	
	# set the output layer
	classifier.add(Dense(dic['n_nodes_output'], activation=dic['activation_output']))

	classifier.compile(optimizer= dic['optimizer'], loss=dic['loss'], metrics=[dic['metrics']])
	return classifier

def train_test_model(x_train, y_train,x_test, y_test,classifier, batch_size= 10, epochs= 50):
	classifier.fit(x_train, y_train, batch_size= batch_size, epochs= epochs, verbose=0)

	y_pred, y_classes = classifier.predict(x_test), classifier.predict_classes(x_test)
	t = []
	n_col = 0
	try:
		n_col = y_test.shape[1]
	except IndexError:
		n_col = 0		

	if n_col != 0:
		for i in y_pred:
			k = []
			for j in i:
				if j > 0.5:
					k.append(1)
				else:
					k.append(0)
			t.append(k)
	else:
		for i in zip(y_classes,y_pred):
			k = []
			if i[1][0] > 0.5:
				k.append(i[0][0])
			else:
				k.append( 0 if i[0][0] == 1 else 1)
			t.append(k)
	y_pred = np.array(t)
	acc = accuracy_score(y_test,y_pred)
	return classifier, acc

def func_DeepLearning(x_train, y_train,x_test, y_test, n_nodes_output, hidden_nodes_obj={'n_nodes_hl1':6}, batch_size=10, epochs=50):
	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.transform(x_test)

	classifier = Sequential()
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	dic = {}
	
	dic['loss'] = loss_deep
	dic['optimizer'] = optmizer_deep
	dic['activation_output'] = activation_output_deep

	dic['metrics'] = metric_deep
	dic['activation_hidden'] = activation_deep
	dic['n_nodes_input'] = x_train.shape[1]
	dic['n_nodes_output'] = n_nodes_output
	dic['n_nodes'] = hidden_nodes_obj	

	classifier =  construct_model(dic, classifier)
	classifier, acc = train_test_model(x_train, y_train,x_test, y_test, batch_size= batch_size, epochs= epochs, classifier=classifier)
	return classifier, acc, sc

#####################   Deep Learning   #######################
