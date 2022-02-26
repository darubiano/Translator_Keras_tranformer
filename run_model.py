import pickle
from tensorflow import keras
from keras_transformer import get_custom_objects,decode

# Cargar modelo preentrenado
loaded_mod = keras.models.load_model('translatorTransformer.h5',
                                         custom_objects=get_custom_objects())
# Cargar diccionario inverso de español
target_token_dict_inv = pickle.load(open('target_token_dict_inv.pkl','rb'))
# Cargar diccionario de ingles a numeros
source_token_dict = pickle.load(open('source_token_dict.pkl','rb'))
# Cargar diccionario de español a numeros
target_token_dict = pickle.load(open('target_token_dict.pkl','rb'))

#### Pruebas, con modelo propio reemplazar loaded_mod por model
# Funcion para traducir texto
def translate(sentence):
    sentence = sentence.lower()
    sentence_tokens = [tokens + ['<END>', '<PAD>'] for tokens in [sentence.split(' ')]]
    print(sentence_tokens)
    tr_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in sentence_tokens][0]
    print(tr_input)
    decoded = decode(
        loaded_mod, 
        tr_input, 
        start_token = target_token_dict['<START>'],
        end_token = target_token_dict['<END>'],
        pad_token = target_token_dict['<PAD>'],
        max_len=100,
    )
    print(decoded)
    print('Frase original: {}'.format(sentence))
    print('Traducción: {}'.format(' '.join(map(lambda x: target_token_dict_inv[x], decoded[1:-1]))))

# Nota es necesario cargar los diccionarios de palabras
translate("The best way to predict the future is to invent it")