import pandas as pd
import json
from datetime import datetime

df = pd.read_csv('Logs_2019.1.csv')
df_notas = pd.read_csv('notas_2019.1.csv')
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
    ['2019-03-26', '../modules/Datasets/test/test_SBIE_2019-03-26.csv'],
    ['2019-04-02', '../modules/Datasets/test/test_SBIE_2019-04-02.csv'],
    ['2019-04-09', '../modules/Datasets/test/test_SBIE_2019-04-09.csv'],
    ['2019-04-16', '../modules/Datasets/test/test_SBIE_2019-04-16.csv'],
    ['2019-05-04', '../modules/Datasets/test/test_SBIE_2019-05-04.csv'],
    ['2019-05-30', '../modules/Datasets/test/test_SBIE_2019-05-30.csv'],
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
    # INITIALIZE VALUES OF ROWS TO THE USERS THAT HAS TEST GRADE
    for index1, row in df_dict[df_dict['time'] < date[0]].iterrows():
        if str(row['event']) in columns and str(row['name']) not in dic[str(row['event'])] and str(row['name']) in df_notas.index:
            for event in columns:
                dic[event][str(row['name'])] = 0
            dic['name'][str(row['name'])] = str(row['name'])

    # POPULATE THE COLUMNS WITH THE RIGHT VALUES
    for index1, row in df_dict[df_dict['time'] < date[0]].iterrows():
        if(str(row['event']) in columns and str(row['name']) in dic[str(row['event'])] and str(row['name']) in df_notas.index):
            dic[str(row['event'])][str(row['name'])] += 1
            dic['class'][str(row['name'])] = 'active' if int(
                df_notas['total'][str(row['name'])]) >= 60 else 'inactive'

            # AGGREGATE TEST GRADES INTO TOTAL
            if(index == 0):
                dic['total'][str(row['name'])] = int(
                    df_notas['atv1'][str(row['name'])])
            elif(index == 1):
                dic['total'][str(row['name'])] = sum(
                    [int(df_notas['atv1'][str(row['name'])]), int(df_notas['atv2'][str(row['name'])])])
            elif(index == 2):
                dic['total'][str(row['name'])] = sum(
                    [int(df_notas['atv1'][str(row['name'])]), int(df_notas['atv2'][str(row['name'])]), int(df_notas['atv3'][str(row['name'])])])
            elif(index == 3):
                dic['total'][str(row['name'])] = sum(
                    [int(df_notas['atv1'][str(row['name'])]), int(df_notas['atv2'][str(row['name'])]), int(df_notas['atv3'][str(row['name'])]), int(df_notas['atv4'][str(row['name'])])])
            elif(index == 4):
                dic['total'][str(row['name'])] = sum([int(df_notas['atv1'][str(row['name'])]), int(
                    df_notas['atv2'][str(row['name'])]), int(df_notas['atv3'][str(row['name'])]), int(df_notas['atv4'][str(row['name'])]), int(df_notas['atv5'][str(row['name'])])])
            elif(index == 5):
                dic['total'][str(row['name'])] = sum([int(df_notas['atv1'][str(row['name'])]), int(df_notas['atv2'][str(row['name'])]), int(
                    df_notas['atv3'][str(row['name'])]), int(df_notas['atv4'][str(row['name'])]), int(df_notas['atv5'][str(row['name'])]), int(df_notas['atv6'][str(row['name'])])])

    data_frameCount = pd.DataFrame.from_dict(dic)
    data_frameCount = data_frameCount.sort_values(by='class', ascending=False)

    # WRITE THE DATAFRAME INTO A CSV FILE
    data_frameCount.to_csv(date[1], index=None, header=True, columns=columns)
