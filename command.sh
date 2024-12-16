MODEL="roberta-large"
corpus="strong_awry_utterances"
bs=1
bert_forecast="/home/sqt2/myExperiment/cs6742-trama/trama_bert_forecast.py"
mode="full"
for seed in 11 12 13 14 15 42 81 93 188 830
do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python ${bert_forecast}\
        --convokit_path "/home/sqt2/ConvoKit"\
        --model_name_or_path "/reef/sqt2/ConvoKitCGA-Redo/cmv/${MODEL}/seed-${seed}"\
        --context_mode ${mode}\
        --do_eval True\
        --corpus_name ${corpus}\
        --random_seed ${seed}\
        --output_dir "/home/sqt2/myExperiment/cs6742-trama/predictions/nonsense-prompt/${corpus}/${seed}"
done
