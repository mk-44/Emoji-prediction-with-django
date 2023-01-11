from keras.models import model_from_json
import emoji
import numpy as np

emoji_dictionary = {
    "0": "\u2764\uFE0F",
    "1": ":baseball:",
    "2": ":grinning_face_with_big_eyes:",
    "3": ":disappointed_face:",
    "4": ":fork_and_knife:",
}


with open('services/emojifier/model_json.json', 'r') as f:
	model = model_from_json(f.read())	

model.load_weights('services/emojifier/best_model.h5')


def get_word_vectors():
   	word_vectors = {}

   	with open('services/emojifier/glove.6B.50d.txt', 'r', encoding = 'utf-8') as f:
   		for line in f:
   			word = line.split()[0]
   			vector = line.split()[1:]
   			word_vectors[word] = vector
   	return word_vectors


word_vectors = get_word_vectors()


def word_embedding(lines, dims = 50, max_len = 10):
    cnt = 0
    m = len(lines)
    ans = np.zeros((m, max_len, dims))
    
    for i, line in enumerate(lines):
        words = line.split()
        for j, word in enumerate(words):
            try:
                ans[i, j] = word_vectors[word.lower()]
                cnt += 1
            except:
                pass
    return ans



def predict(X):
	X_em = word_embedding(X)
	Yp = model.predict(X_em).argmax(axis = 1)[0]
	print(Yp)
	pred_emoji = emoji.emojize(emoji_dictionary.get(str(Yp)))
	return pred_emoji
