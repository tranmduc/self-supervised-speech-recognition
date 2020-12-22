## Self-supervised speech recognition with limited amount of labeled data


This is a wrapper version of [wav2vec 2.0 framework](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec), which attempts to build an accurate speech recognition models with small amount of transcribed data (eg. 1 hour)


Transfer learning is still the main technique:
 - Transfer from self-supervised models (pretrain on unlabeled data)
 - Transfer from multilingual models (pretrain on multilingual data)

## Required resources

#### 1. Labeled data, which is pairs of (audio, transcript)
The more you have, the better the model is. Prepare at least 1 hour if you have a large amount of  unlabeled data. Otherwise, at least 50 hours is recommended.

#### 2. Text data for building language models. 
This should includes both well-written text and conversational text, which can easily collected from news/forums websties. At least 1 GB of text is recommended.

#### 3. Unlabeled data (audios without transcriptions) of your own language. 
This is optional but very crucial. A good amount of unlabeled audios (eg. 500 hours) will significantly reduce the amount of labeled data needed, and also boost up the model performance. Youtube/Podcast is a great place to collect the data for your own language

## Install instruction
Please follow this [instruction](https://github.com/mailong25/self-supervised-speech-recognition/blob/master/Dependencies.md)

## Steps to build an accurate speech recognition model for your language

### 1. Train a self-supervised model on unlabeled data (Pretrain)

#### 1.1 Prepare unlabeled audios
Collect unlabel audios and put them all together in a single directory. Audio format requirements:\
Format: wav, PCM 16 bit, single channel\
Sampling_rate: 16000\
Length: 5 to 30 seconds\
Content: silence should be removed from the audio. Also, each audio should contain only one person speaking.\
Please look at unlabel_audio directory for examples.

#### 1.2 Download an initial model
Instead of training from scratch, we download and use english wav2vec model for weight initialization. This pratice can be apply to all languages.
```
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
```

#### 1.3 Run Pre-training
```
python3 pretrain.py --fairseq_path path/to/libs/fairseq --audio_path path/to/audio_directory --init_model path/to/wav2vec_small.pt
```
Logs and checkpoints will be stored at outputs directory\
Log_file path: outputs/date_time/exp_id/hydra_train.log.  You should check the loss value to decide when to stop the training process.\
Best_checkpoint path: outputs/date_time/exp_id/checkpoints/checkpoint_best.pt\
In my casse, it took ~ 2 days for the model to converge, train on 100 hours of data using 2 NVIDIA Tesla V100.

### 2. Finetune the self-supervised model on the labeled data

---------------- prepare labeled data  ---------------- \
Transcript file\
Text format should be the same as step 2 (Train a language model).\
One trainng sample per line with format "audio_name \tab transcript"\
Example of a transcript file:
```
1.wav AND IT WAS A MATTER OF COURSE THAT IN THE MIDDLE AGES WHEN THE CRAFTSMEN
2.wav AND WAS IN FACT THE KIND OF LETTER USED IN THE MANY SPLENDID MISSALS PSALTERS PRODUCED BY PRINTING IN THE FIFTEENTH CENTURY
3.wav JOHN OF SPIRES AND HIS BROTHER VINDELIN FOLLOWED BY NICHOLAS JENSON BEGAN TO PRINT IN THAT CITY
4.wav BEING THIN TOUGH AND OPAQUE
```

Audio format: wav, PCM 16 bit, single channel, Sampling_rate: 16000.\
Silence should be removed from the audio.\
Also, each audio should contain only one person speaking.\

---------------- Generate dictionary file ----------------
```
python3 gen_dict.py --transcript_file path/to/transcript.txt --save_dir path/to/save_dir
```
The dictionary file will be stored at save_dir/dict.ltr.txt. Use the file for fine-tuning and inference.\

---------------- Finetune the Model ----------------
```
python3 finetune.py --transcript_file path/to/transcript.txt --audio_dir path/to/audio_directory --pretrain_model path/to/pretrain_checkpoint_best.pt --dict_file path/to/dict.ltr.txt
```
Logs and checkpoints will be stored at outputs directory\
Log_file path: outputs/date_time/exp_id/hydra_train.log. You should check the loss value to decide when to stop the training process.\
Best_checkpoint path: outputs/date_time/exp_id/checkpoints/checkpoint_best.pt\
In my casse, it took ~ 12 hours for the model to converge, train on 100 hours of data using 2 NVIDIA Tesla V100.\

### 3. Train a language model
---------------- Prepare text file ---------------- \
Collect all texts and put them all together in a single file. \
To avoid vocabulary mismatch, the text must include all transcripts from the labeled data (Required resources #1)\
Text file format:
- One sentence per line
- Upper case
- All numbers should be transformed into verbal form.
- All special characters (eg. punctuation) should be removed. The final text should contain words only
- Words in a sentence must be separated by whitespace character 

Example of a text file for English case:
```
AND IT WAS A MATTER OF COURSE THAT IN THE MIDDLE AGES WHEN THE CRAFTSMEN
AND WAS IN FACT THE KIND OF LETTER USED IN THE MANY SPLENDID MISSALS PSALTERS PRODUCED BY PRINTING IN THE FIFTEENTH CENTURY
JOHN OF SPIRES AND HIS BROTHER VINDELIN FOLLOWED BY NICHOLAS JENSON BEGAN TO PRINT IN THAT CITY
BEING THIN TOUGH AND OPAQUE
...
```
Example of a text file for Chinese case:
```
每 个 人 都 有 他 的 作 战 策 略 直 到 脸 上 中 了 一 拳
这 是 我 年 轻 时 候 住 的 房 子 。
这 首 歌 使 我 想 起 了 我 年 轻 的 时 候 。
...
```

---------------- Train Language Model ----------------
```
python3 train_lm.py --kenlm_path path/to/libs/kenlm --text_file path/to/text_file.txt --output_path ./lm
```
The LM model and the lexicon file will be stored at output_path

### 4. Make prediction on single audio

```
from stt import Transcriber
transcriber = Transcriber(w2vec = 'path/to/finetune.pt', w2vec_dict = 'path/to/dict.ltr.txt',
                          lm_lexicon = 'path/to/lm/lexicon.txt', lm_model = 'path/to/lm/lexicon.txt',
                          lm_weight = 1.5, word_score = -1, beam_size = 50)
hypos = transcriber.transcribe(['path/to/wavs/0_1.wav','path/to/wavs/0_2.wav'])
print(hypos)
```
Note that the first prediction call will take a lot of time (model loading)

## Older version on Vietnamese speech recognition: 
https://github.com/mailong25/self-supervised-speech-recognition/tree/vietnamese
