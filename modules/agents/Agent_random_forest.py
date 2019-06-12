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
#import procname
#procname.setprocname(str(sys.argv[1])+'Agent_random_forest')

ipAddress = '127.0.0.1'
port = int(sys.argv[1])

urls = (
    '/Agent_random_forest', 'Main',
)

app = web.application(urls, globals())

############  INIT VARIABLES  ##############
initialize_all_parameters(str(sys.argv[2]))
df, conj, x, y, what_is_expected, type_str, labelEncoder = df_conj_x_y_expected()
x_train,x_test,y_train, y_test = split_data(x, y)
n_param = len(x_train[0])
i = time.time()
classifier, accuracy = func_Random_Forest(x_train,y_train,x_test,y_test,plot=False)
f = time.time()
# print('Random_Forest' + str(f-i))
############  INIT VARIABLES  ##############

class Main:
    def POST(self):
        param = web.input(_method='post')
        obj = {}
        if param:
            x_pred = []
            for parameter in what_is_expected:
                parameter_data = param.get(str(parameter))
                if not parameter_data or parameter_data == '?':
                    x_pred.append(-9999)
                elif str(parameter) in type_str:
                    x_pred.append(conj[str(parameter)][str(parameter_data)])
                else:
                    x_pred.append(parameter_data)
            x_pred = np.array(x_pred)
            x_pred = x_pred.astype(np.float64)
            y_pred = classifier.predict([x_pred])
            y_pred_proba = classifier.predict_proba([x_pred])
            obj = {"class": str(labelEncoder.inverse_transform([y_pred[0]])[
                                0]), 'accuracy':str(y_pred_proba[0][y_pred[0]]), "code": 200}
        else:    
            obj["error"] = [{"message": "Sorry, bad request, params missing (HAVE TO BE "+ n_param + "PARAMETERS)", "code": 400}]
        gc.collect()
        return json.dumps(obj, separators=(',', ':'), indent=2)
            

class MainServer(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, (ipAddress, port))


if __name__ == "__main__":
    app = MainServer(urls, globals())
    app.run(port=port)
