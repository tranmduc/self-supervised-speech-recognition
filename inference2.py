from stt import Transcriber
import configparser
import pandas as pd

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
	audioList.append(config["TRANSCRIBER"]["audio_file"])

	hypos = transcriber.transcribe(audioList)
	print(hypos)

if __name__ == '__main__':
	speech2text()