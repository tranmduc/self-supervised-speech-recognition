from stt import Transcriber
import sys
import configparser
import glob
import os
import pandas as pd

def dirwalk(dir, bag, wildcards):
    bag.extend(glob.glob(path.join(dir, wildcards)))
    for f in os.listdir(dir):
        fullpath = os.path.join(dir, f)
        if os.path.isdir(fullpath) and not os.path.islink(fullpath):
            dirwalk(fullpath, bag, wildcards)

def speech2text():
	config = configparser.ConfigParser()
	config.read("config.txt")
	transcriber = Transcriber(pretrain_model = config["TRANSCRIBER"]["pretrain_model"],
							finetune_model = config["TRANSCRIBER"]["finetune_model"],
							dictionary = config["TRANSCRIBER"]["dictionary"],
							lm_type = 'kenlm',
							lm_lexicon = config["TRANSCRIBER"]["lm_lexicon"],
							lm_model = config["TRANSCRIBER"]["lm_model"],
							lm_weight = config["TRANSCRIBER"]["lm_weight"],
							word_score = config["TRANSCRIBER"]["word_score"],
							beam_size = config["TRANSCRIBER"]["beam_size"])

	audioList = []
	# audioList = glob.glob(os.path.join(config["TRANSCRIBER"]["wav_folder"], '*.wav'))
	dirwalk(config["TRANSCRIBER"]["wav_folder"], audioList, "*.wav")
	print(audioList)

	hypos = transcriber.transcribe(audioList)
	print(hypos)

	result_path = 'result.csv'
	if 'result_path' in config["TRANSCRIBER"]:
		result_path = config["TRANSCRIBER"]["result_path"]

	dict_df = {'path': audioList, 'hypos': hypos}		
	result_df = pd.DataFrame(dict_df)
	result_df.to_csv(result_path, index=False)

if __name__ == '__main__':
	speech2text()