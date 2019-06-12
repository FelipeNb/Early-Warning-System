####################
# AUTOR: Felipe Neves Braz
####################
import gc
import web
import time
import requests
import json
import urllib
import sys
sys.path.append('../Prediction')
from Prediction import *
import threading
#import procname
#procname.setprocname(str(sys.argv[1])+'Agent_central')
# from database import Database

ipAddress = '127.0.0.1'
port = int(sys.argv[1])
Agent_decision_tree = str(sys.argv[2])
Agent_deep_learning = str(sys.argv[3])
Agent_knn = str(sys.argv[4])
Agent_logistic_regression = str(sys.argv[5])
Agent_random_forest = str(sys.argv[6])
Agent_svm = str(sys.argv[7])

urls = (
    '/Agent_central', 'Main',
)

app = web.application(urls, globals())

agents = [
        ('http://' + ipAddress +':'+ Agent_decision_tree +'/Agent_decision_tree','decision_tree'),
        ('http://' + ipAddress +':'+ Agent_deep_learning +'/Agent_deep_learning','deep_learning'),
        ('http://' + ipAddress +':'+ Agent_knn +'/Agent_knn','knn'),
        ('http://' + ipAddress +':'+ Agent_logistic_regression +'/Agent_logistic_regression','logistic_regression'),
        ('http://' + ipAddress +':'+ Agent_random_forest +'/Agent_random_forest','random_forest'),
        ('http://' + ipAddress +':'+ Agent_svm +'/Agent_svm','svm'),
    ]

############  functions  ##############

def dispatch_data_agent(agent, aux, url='', dt={}):
    rt_bool = False
    try:
        rt = requests.post(url, data = dt).json()
        rt_bool = True
    except:
        rt_bool = False

    if rt_bool:
        t = getattr(aux, 'returns')
        t.append((agent, rt))
        setattr(aux, 'returns', t)
    setattr(aux, 'n_threads', getattr(aux, 'n_threads') -1)

def handle_missing_data(user=None, df={}):
    dt_frame = get_recent_data_from_user(user)
    if df != {}:
        for i in dt_frame:
            df[i] = df[i] if df[i] != '?' else dt_frame[i]
    return df

def should_recommend(decision_tree,knn,deep_learing,logistic_regression,random_forest,svm):
    '''
        Main code of central agent.
        This file can see which class and what they are.
        Here we consider that each agent is an object that looks like:
            {
                'class':0,
                'accuracy': 0.95
            }
    '''
    recommend = 'active'
    intensity = 0.0
    count = 0
    lista = [decision_tree,knn,deep_learing,logistic_regression,random_forest,svm]
    newlist = sorted(lista, key=lambda k: k['accuracy'],reverse=True)
    somatorio = 0.0
    positives = []

    for i in newlist:
        if i['class'] == 'inactive':
            positives.append(float(i['accuracy']))
            somatorio += float(i['accuracy'])
        else:
            somatorio -= float(i['accuracy'])
        # count+= int(i['class'])

    if somatorio > 0:
        recommend = 'inactive'
        intensity = np.mean(np.array(positives))

    return recommend, intensity

class Vars:
    pass

############  functions  ##############

class Main:
    def POST(self):
        param = web.input(_method='post')
        aux = Vars()
        setattr(aux, 'n_threads', len(agents))
        setattr(aux, 'choosenOne', '')
        setattr(aux, 'returns', [])
        if param:
            # param = handle_missing_data(param.get('user'))
            for url_agent, agent in agents:
                t1 = threading.Thread(name=str(agent), 
                                      target=dispatch_data_agent,
                                      args=(str(agent), aux, url_agent, param))
                t1.start()
            while getattr(aux, 'n_threads') > 0:
                True

            dic = dict(getattr(aux, 'returns'))
            recommend, intensity = should_recommend(
                            decision_tree = dic['decision_tree'] if 'decision_tree' in dic else {'class': 1, 'accuracy': 1.0},
                            knn = dic['knn'] if 'knn' in dic else {'class': 1, 'accuracy': 1.0},
                            deep_learing = dic['deep_learing'] if 'deep_learing' in dic else {'class': 1, 'accuracy': 1.0},
                            logistic_regression = dic['logistic_regression'] if 'logistic_regression' in dic else {'class': 1, 'accuracy': 1.0},
                            random_forest = dic['random_forest'] if 'random_forest' in dic else {'class': 1, 'accuracy': 1.0},
                            svm = dic['svm'] if 'svm' in dic else {'class': 1, 'accuracy': 1.0}
                        )
        return json.dumps({'recommend': recommend, 'intensity':intensity, 'lista': dic}, separators=(',', ':'), indent=2)

class MainServer(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, (ipAddress, port))

if __name__ == "__main__":
    app = MainServer(urls, globals())
    app.run(port=port)

