#!/bin/bash
cabecalho="location,delete_columns,change_to_int,change_to_float,what_is_exp,classificatory_data,regularization_log_reg,solver_log_reg,multiclass_log_reg,random_state_log_reg,regularization_svm,kernel_svm,gamma_svm,random_state_svm,criterion_decision_tree,max_depth_decision_tree,random_state_decision_tree,criterion_random_forest,n_estimators_random_forest,n_processors_random_forest,random_state_random_forest,type_distance_knn,n_neighbours_knn,metric_knn,loss_deep,metric_deep,optmizer_deep,activation_deep,activation_output_deep"
num=$1
port=$4
name="runAgent_$port"
find="Find$name"
find_best="central_fmeasure,knn_fmeasure,svm_fmeasure,decision_tree_fmeasure,logistic_regression_fmeasure,random_forest_fmeasure,deep_learning_fmeasure,central_mae,knn_mae,svm_mae,decision_tree_mae,logistic_regression_mae,random_forest_mae,deep_learning_mae,location,delete_columns,change_to_int,change_to_float,what_is_exp,classificatory_data,regularization_log_reg,solver_log_reg,multiclass_log_reg,random_state_log_reg,regularization_svm,kernel_svm,gamma_svm,random_state_svm,criterion_decision_tree,max_depth_decision_tree,random_state_decision_tree,criterion_random_forest,n_estimators_random_forest,n_processors_random_forest,random_state_random_forest,type_distance_knn,n_neighbours_knn,metric_knn,loss_deep,metric_deep,optmizer_deep,activation_deep,activation_output_deep"
echo $find_best > ../modules/Configutarion_$3.csv
awk "NR >= $1 && NR < $2" 'TestFitBestParams.csv' |
while read row
do
    echo $cabecalho > ../modules/Prediction/SystemSBIE_$3.csv
    echo $row >> ../modules/Prediction/SystemSBIE_$3.csv
    cd ../modules/agents/
    ./runAgents.sh $num $port $((port +1)) $((port +2)) $((port +3)) $((port +4)) $((port +5)) $((port +6)) SystemSBIE_$3.csv & 
    sleep 6;cd ..;python FindBestParams.py $num Configutarion_$3.csv $((port +6)) $find;
    echo $(pwd)
    cd ../SBIE
    pkill -f $name
    pkill -f $find
    pkill -f Agent_decision_tree$num
    pkill -f Agent_deep_learning$num
    pkill -f Agent_knn$num
    pkill -f Agent_logistic_regression$num
    pkill -f Agent_random_forest$num
    pkill -f Agent_svm$num
    pkill -f Agent_central$num
    pkill -f $num'Agent_central'
    pkill -f $num'Agent_decision_tree'
    pkill -f $num'Agent_deep_learning'
    pkill -f $num'Agent_knn'
    pkill -f $num'Agent_logistic_regression'
    pkill -f $num'Agent_random_forest'
    pkill -f $num'Agent_svm'
    num=$((num+1))
done
