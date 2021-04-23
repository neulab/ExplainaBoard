# /usr2/home/pliu3/data/InterpretEval/task-panx/genPreComputed

function tensorEval(){

  # Task forms
  task_type="chunk"
  # Model Name or team's name, e.g., nyu
  model_name=$1


  path_aspect_conf="conf.ner-aspects"
  path_json_output=./output/$model_name
  path_json_input=./template.json

  path_data=../data/trainDevTest/$task_type/
  path_label=../data/labels/$task_type/
  path_submission=../data/testPred/$task_type/$model_name

  echo
  echo "##################################################################################"
  echo "                     Tensor Evaluation Start for Team: " $1
  echo "##################################################################################"
  echo "#task_type: " $task_type
  echo "#submission team: " $model_name
  echo "#path_json_output: " $path_aspect_conf
  echo "#path_json_input: " $path_json_input
  echo "#path_data: " $path_data
  echo "#path_label: " $path_label
  echo "#path_submission: " $path_submission
  echo



  if [ ! -d $path_json_output  ];then
    mkdir $path_json_output
  else
    rm -fr $path_json_output/*
    echo dir exist
  fi


  n_json=0
  for TEST in $(ls ${path_submission}/test-*.tsv); do

    replace=""
    find="test-"
    FILENAME=$(basename $TEST .tsv)
    corpus_type=${FILENAME//$find/$replace}


    echo "#Evaluating Language: " $corpus_type


    path_json_file=$path_json_output/$corpus_type.json

    python3 tensorEvaluation.py \
      --text  $path_submission/test-$corpus_type.tsv \
      --path_aspect_conf $path_aspect_conf \
      --task_type $task_type  \
      --corpus_type $corpus_type\
      --model_name $model_name \
      --path_json_input $path_json_input \
      --fn_write_json $path_json_file

    n_json=$(($n_json+1))


  done

  echo
  echo "##################################################################################"
  echo "                     Tensor Evaluation Finished for Team: " $1
  echo "##################################################################################"
  echo "#the number of analyzed json files: " $n_json
  echo "#corpus_type: " $corpus_type
  echo "#submission team: " $model_name
  echo "#path_json_output: " $path_aspect_conf
  echo




}



if [ ! -d ./output/  ];then
  mkdir ./output/
else
  rm -fr ./output/*
fi


path_submission=../data/testPred/chunk
for model in $(ls ${path_submission}); do

  #echo $team
  tensorEval $model

done





