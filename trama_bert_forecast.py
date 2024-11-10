import sys 
import os
import json
from utils import *
args = parseargs([['-convokit', '--convokit_path', 'path_to_convokit', str],
                    ['-model', '--model_name_or_path', 'model_name', str, 'roberta-large'],
                    ['-mode', '--context_mode', 'context_mode', str, 'full'],
                     ['-corpus_name', '--corpus_name', 'corpus_name', str],
                     ['-train', '--do_train', 'train_before_evaluate', bool],
                     ['-eval', '--do_eval', 'evaluate_or_not', bool],
                     ['-lr', '--learning_rate', 'learning_rate', float, 2e-5],
                     ['-bs', '--per_device_batch_size', 'number_of_samples_on_each_GPU', int, 12],
                     ['-epoch', '--num_train_epochs', 'num_train_epochs', int, 4],
                     ['-output', '--output_dir', 'output_directory', str],
                     ['-seed', '--random_seed', 'random_seed', int, 42],
                     ])
# Installable Convokit is not updated till now
if args.convokit_path:
    print("Using custom convokit at {}".format(args.convokit_path))
    sys.path.insert(1, args.convokit_path)
from functools import partial
from BERTModel import BERTModelCGA
from convokit import Forecaster, download, Corpus, Utterance, Conversation, Speaker

def load_trama(corpus_name):
    corpus = Corpus(utterances=[])
    speaker = Speaker()
    with open('{}.json'.format(corpus_name), 'r') as file:
        data = json.load(file)
    with open('chatgpt_{}.txt'.format(corpus_name), 'r') as file:
        all_lines = file.readlines()
    prompts = {}
    for line in all_lines:
        id, prompt = line.strip().split(":=:=:=:=")
        prompts[id] = prompt
    all_new_utterances = []
    for conv_id in data:
        utt_id = conv_id.split("_")[1]
        if prompts[conv_id]:
            new_convo = Conversation(owner=corpus, id=conv_id, meta={'has_removed_comment':False,
                                                                     'split':'test'})
            prompt_utterance = Utterance(id=conv_id,
                                speaker=speaker,
                                conversation_id=conv_id,
                                reply_to=None,
                                text=prompts[conv_id],
                                timestamp=0,
                                )
            reply_utterance = Utterance(id=utt_id,
                                speaker=speaker,
                                conversation_id=conv_id,
                                reply_to=conv_id,
                                text=data[conv_id][-1]['text'],
                                timestamp=1,
                                )
            all_new_utterances.append(prompt_utterance)
            all_new_utterances.append(reply_utterance)
    corpus.add_utterances(all_new_utterances)
    return corpus
def load_corpus(corpus_name):
    if corpus_name == "wikiconv":
        corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
        label_metadata = "conversation_has_personal_attack"

    else:
        if corpus_name == "cmv":
            cmv_dir = "/reef/lyt5_cga_cmv"
            corpus = Corpus(cmv_dir)
            label_metadata = "has_removed_comment"
        elif "awry_utterances" in corpus_name:
            corpus = load_trama(corpus_name)
            label_metadata = "has_removed_comment"
            
        # Add one dummy utterance to the end of all conversations 
        # to standardize the evaluation framework
        all_new_utterances = []
        for convo in corpus.iter_conversations():
            last_utterance_id = convo.get_chronological_utterance_list()[-1].id
            random_speaker = convo.get_chronological_speaker_list()[0]
            convo_id = convo.id
            dummy_id = convo.id + "_dummy_reply"
            new_utterance = Utterance(id=dummy_id,
                                    speaker=random_speaker,
                                    conversation_id=convo_id,
                                    reply_to=last_utterance_id,
                                    text="This#is#a#dummy#reply.",
                                    timestamp=1672516520,
                                    )
            all_new_utterances.append(new_utterance)
        corpus.add_utterances(all_new_utterances)
    return corpus, label_metadata

def main(args):
    
    corpus, label_metadata = load_corpus(args.corpus_name)

        
    config_dict = {
                    "output_dir": args.output_dir, 
                    "per_device_batch_size": args.per_device_batch_size, 
                    "num_train_epochs": args.num_train_epochs, 
                    "learning_rate": args.learning_rate, 
                    "random_seed": args.random_seed,
                    "context_mode": args.context_mode,
                    "device": "cuda"
                    }
    
    if args.do_train:
        # Initialize BERTForecaster from scratch.
        bert = BERTModelCGA(args.model_name_or_path, config=config_dict)
        bert_forecaster = Forecaster(bert, label_metadata)

        # Train, Select best model checkpoint, and tune for best threshold.
        if args.corpus_name == "delta_cmv":
            bert_forecaster.fit(corpus, 
                        partial(new_delta_train_selector, split="train"),
                        val_context_selector=partial(full_selector, split="val"))
        # Will delete this if else soon
        else:
            bert_forecaster.fit(corpus, 
                        partial(last_only_selector, split="train"),
                        val_context_selector=partial(full_selector, split="val"))
    else:
        # Load Pretrained BERTForecaster from checkpoint and dev_config.json
        config_file = os.path.join(args.model_name_or_path, "dev_config.json")
        with open(config_file, 'r') as file:
            model_config = json.load(file)
        bert = BERTModelCGA(os.path.join(args.model_name_or_path, model_config['best_checkpoint']), config=config_dict)
        bert.best_threshold = model_config['best_threshold']
        bert_forecaster = Forecaster(bert, label_metadata)
        # Create the output_dir for saving predictions and results
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"Created directory: {args.output_dir}")
        else:
            print(f"Directory already exists: {args.output_dir}")
    
    if args.do_eval:
        corpus = bert_forecaster.transform(corpus, partial(full_selector, split="test"))
        
        # _, metrics = bert_forecaster.summarize(corpus, lambda c: c.meta['split'] == "test")
        # result_file = os.path.join(args.output_dir, "test_results.json")
        # with open(result_file, 'w') as outfile:
        #     json_object = json.dumps(metrics, indent=4)
        #     outfile.write(json_object)
    return

                     # type_context
print(f'ARGPARSE OPTIONS {args}')
main(args)