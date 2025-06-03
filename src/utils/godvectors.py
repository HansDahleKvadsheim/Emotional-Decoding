import numpy as np
import pandas as pd

def create_godvectors_csv(sentiment_path, save_path):

    file = open(sentiment_path, 'r')
    content = file.read()
    split_content = content.split("\n")
    emotion_value_pairs = []

    for i in range(len(split_content)):
        line = split_content[i]
        if line[0:4] == 'Line' or line == '':
            pass
        else:
            emotion_value_pair = line.split(": ")
            print(emotion_value_pair)
            emotion_value_pairs.append(emotion_value_pair)


    emotions = []
    sentence_emotions = []
    for i in range(len(emotion_value_pairs)):
        print((sentence_emotions))
        sentence_emotions.append(emotion_value_pairs[i][-1])
        if i % 8 == 7:
            emotions.append(np.array(sentence_emotions))
            sentence_emotions = []

    emotions_array = np.array(emotions)

    god_vectors = pd.DataFrame(data=emotions_array, columns=['Anticipation', 'joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger'])

    god_vectors.to_csv(path_or_buf=save_path)


def get_godvectors_from_csv(path):
    return pd.read_csv(path)