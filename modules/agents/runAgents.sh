#!/bin/sh

python  Agent_decision_tree.py $1 $8 &
python  Agent_deep_learning.py $2 $8 &
python  Agent_knn.py $3 $8 &
python  Agent_logistic_regression.py $4 $8 &
python  Agent_random_forest.py $5 $8 &
python  Agent_svm.py $6 $8 &
python  Agent_central.py $7 $1 $2 $3 $4 $5 $6 

# 8880 8881 8882 8883 8884 8885 8886 Configuration_SBIE_2018-10-22.csv