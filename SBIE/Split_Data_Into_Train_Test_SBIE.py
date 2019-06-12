import pandas as pd
import json
from datetime import datetime

df = pd.read_csv('Logs_2018.3.csv')
df_notas = pd.read_csv('notas_2018.3.csv')
df_notas.set_index('name', inplace=True)

dic = {}

columns = [
    'class',
    'name',
    'total',
    'algum conteudo foi publicado',
    'assinatura criada',
    'assinatura de discussao criada',
    'comentario apagado',
    'comentario criado',
    'curso visto',
    'discussao visualizada',
    'discussao criada',
    'formulario de confirmacao de submissao visualizado',
    'formulario de submissao visualizado',
    'lista de instancias de modulos de cursos visualizados',
    'lista de usuarios vistos',
    'modulo do curso visualizado',
    'o status da submissao foi visualizado',
    'o usuario aceitou o acordo da tarefa',
    'o usuario salvou um envio',
    'perfil do usuario visto',
    'post atualizado',
    'post criado',
    'relatorio de notas do usuario visualizado',
    'relatorio de um utilizador do curso visualizado',
    'submissao criada',
    'um arquivo foi enviado',
    'um envio foi submetido',
    'visualizado relatorio de usuario',
]

dates = [
    [
        '2018-10-22',
        '../modules/Datasets/train/train_SBIE_2018-10-22.csv',
        '../modules/Datasets/test/test_SBIE_2018-10-22.csv'
    ],
    [
        '2018-10-29',
        '../modules/Datasets/train/train_SBIE_2018-10-29.csv',
        '../modules/Datasets/test/test_SBIE_2018-10-29.csv'
    ],
    [
        '2018-11-05',
        '../modules/Datasets/train/train_SBIE_2018-11-05.csv',
        '../modules/Datasets/test/test_SBIE_2018-11-05.csv'
    ],
    [
        '2018-11-12',
        '../modules/Datasets/train/train_SBIE_2018-11-12.csv',
        '../modules/Datasets/test/test_SBIE_2018-11-12.csv'
    ],
    [
        '2018-12-01',
        '../modules/Datasets/train/train_SBIE_2018-12-01.csv',
        '../modules/Datasets/test/test_SBIE_2018-12-01.csv'
    ],
    [
        '2018-12-30',
        '../modules/Datasets/train/train_SBIE_2018-12-30.csv',
        '../modules/Datasets/test/test_SBIE_2018-12-30.csv'
    ],
]

df_dic = df.to_dict()  # TRANSFORM DATAFRAME INTO A DICTIONARY

# FORMAT DATE TIME INTO DD/MM/YYYY
for row in df_dic['time']:
    df_dic['time'][row] = datetime.strptime(
        str(df_dic['time'][row]).split(" ")[0], '%d/%m/%Y')


# TRANSFORM DICTIONARY TO DATAFRAME AGAIN
df_dict = pd.DataFrame.from_dict(df_dic)


for index, date in enumerate(dates):
    # INITIALIZE COLUMNS INTO THE DATAFRAME
    for column in columns:
        dic[column] = {}
    # INITIALIZE ROWS VALUES TO THE USERS THAT HAS TEST GRADE
    for index1, row in df_dict[df_dict['time'] < date[0]].iterrows():
        if str(row['name']) not in dic[str(row['event'])] and str(row['name']) in df_notas.index:
            for event in columns:
                dic[event][str(row['name'])] = 0
            dic['name'][str(row['name'])] = str(row['name'])

    # POPULATE THE COLUMNS WITH THE RIGHT VALUES
    for index1, row in df_dict[df_dict['time'] < date[0]].iterrows():
        if(str(row['name']) in dic[str(row['event'])] and str(row['name']) in df_notas.index):
            dic[str(row['event'])][str(row['name'])] += 1
            dic['class'][str(row['name'])] = 'active' if int(
                df_notas['total'][str(row['name'])]) > 60 else 'inactive'

            # AGGREGATE TEST GRADES INTO TOTAL
            if(index == 0):
                dic['total'][str(row['name'])] = int(
                    df_notas['atv1'][str(row['name'])])
            elif(index == 1):
                dic['total'][str(row['name'])] = sum(
                    [int(df_notas['atv1'][str(row['name'])]), int(df_notas['atv2'][str(row['name'])])])
            elif(index == 2):
                dic['total'][str(row['name'])] = sum([int(df_notas['atv1'][str(row['name'])]), int(
                    df_notas['atv2'][str(row['name'])]), int(df_notas['atv3'][str(row['name'])])])
            elif(index == 3):
                dic['total'][str(row['name'])] = sum([int(df_notas['atv1'][str(row['name'])]), int(df_notas['atv2'][str(
                    row['name'])]), int(df_notas['atv3'][str(row['name'])]), int(df_notas['atv4'][str(row['name'])])])
            elif(index == 4):
                dic['total'][str(row['name'])] = sum([int(df_notas['atv1'][str(row['name'])]), int(df_notas['atv2'][str(row['name'])]), int(
                    df_notas['atv3'][str(row['name'])]), int(df_notas['atv4'][str(row['name'])]), int(df_notas['atv5'][str(row['name'])])])
            elif(index == 5):
                dic['total'][str(row['name'])] = sum([int(df_notas['atv1'][str(row['name'])]), int(df_notas['atv2'][str(row['name'])]), int(df_notas['atv3'][str(
                    row['name'])]), int(df_notas['atv4'][str(row['name'])]), int(df_notas['atv5'][str(row['name'])]), int(df_notas['atv6'][str(row['name'])])])

    data_frameCount = pd.DataFrame.from_dict(dic)
    data_frameCount = data_frameCount.sort_values(by='class', ascending=False)

    test_size_1 = int(
        len(data_frameCount[data_frameCount['class'] == 'active']) * 0.20)
    test_size_0 = int(
        len(data_frameCount[data_frameCount['class'] == 'inactive']) * 0.20)

    train_data_1 = data_frameCount[data_frameCount['class'] ==
                                   'active'].loc[data_frameCount[data_frameCount['class'] == 'active'].index[:-test_size_1]]
    train_data_0 = data_frameCount[data_frameCount['class'] ==
                                   'inactive'].loc[data_frameCount[data_frameCount['class'] == 'inactive'].index[:-test_size_0]]
    test_data_1 = data_frameCount[data_frameCount['class'] ==
                                  'active'].loc[data_frameCount[data_frameCount['class'] == 'active'].index[-test_size_1:]]
    test_data_0 = data_frameCount[data_frameCount['class'] ==
                                  'inactive'].loc[data_frameCount[data_frameCount['class'] == 'inactive'].index[-test_size_0:]]

    final_train_df = train_data_1.append(train_data_0, ignore_index=True)
    final_test_df = test_data_1.append(test_data_0, ignore_index=True)

    # WRITE THE DATAFRAME INTO A CSV FILE
    final_train_df.to_csv(date[1], index=None, header=True, columns=columns)
    final_test_df.to_csv(date[2], index=None, header=True, columns=columns)
