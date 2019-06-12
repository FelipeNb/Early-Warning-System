#!/bin/bash
cabecalho="location,delete_columns,change_to_int,change_to_float,what_is_exp,classificatory_data,regularization_log_reg,solver_log_reg,multiclass_log_reg,random_state_log_reg,regularization_svm,kernel_svm,gamma_svm,random_state_svm,criterion_decision_tree,max_depth_decision_tree,random_state_decision_tree,criterion_random_forest,n_estimators_random_forest,n_processors_random_forest,random_state_random_forest,type_distance_knn,n_neighbours_knn,metric_knn,loss_deep,metric_deep,optmizer_deep,activation_deep,activation_output_deep"
lines=$(wc -l 'TestFitBestParams.csv' | awk '{ print $1 }')
dividein=4
port=8880

#novo
eachGroup=$lines/$dividein
commands=" ::: "
for i in $(seq 0 $(($dividein -1)))
do 
    from=$(($i * $eachGroup))
    to=$(( (($i + 1) * $eachGroup) + 1 ))
    commands="$commands $from $to $i $port"
    port=$((port+7))
done
echo $commands
parallel -N4 ./runTestSBIE.sh $commands
