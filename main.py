from stt import Transcriber
import configparser

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

def getPrediction(filename):
	audioList = []
	audioList.append(filename)
    hypos = transcriber.transcribe(audioList)
	print(hypos)
    return hypos