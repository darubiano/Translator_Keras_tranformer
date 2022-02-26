# -*- coding: utf-8 -*-
"""
pip install keras-transformer
@author: darub
"""
import numpy as np
import pickle
from keras_transformer import get_custom_objects, get_model, decode
# obtener simpre los mismo resultados en el entraniento del modelo
np.random.seed(0)

# Leer los datos de entranamiento, english-Español
dataset = pickle.load(open('english-spanish.pkl', 'rb'))
# Ver traducion
print(dataset[120000,0])
print(dataset[120000,1])

### Preprocesamiento de datos creacion de tokens ###
source_tokens = []
# sentences in english
for sentence in dataset[:,0]:
    source_tokens.append(sentence.split(' '))
print(source_tokens[120000])

# oraciones en español
target_tokens = []
for oracion in dataset[:,1]:
    target_tokens.append(oracion.split(' '))
print(target_tokens[120000])

# Crear diccionario de equivalencias de palabras
def build_token_dict(token_list):
    # Diccionario para remplazar los valores <PAD> rellenar, <START> inicio oracion, <END> fin de la oracion
    token_dict = {'<PAD>':0,'<START>':1,'<END>':2}
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict

# Generar dicionario que represente cada palabra
source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)
# Diccionario inverso
target_token_dict_inv = {v:k for k,v in target_token_dict.items()}
print(source_token_dict)
print(target_token_dict)
# Diccionario que se va usar en la traducion final
print(target_token_dict_inv)

#
# Agregar start, end y pad a cada frase del set de entramiento
encoder_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
decoder_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
print(encoder_tokens[120000])
print(decoder_tokens[120000])
# frase en español con el final
output_tokens = [tokens+['<END>'] for tokens in target_tokens]
print(output_tokens[120000])
# frase mas larga en ingles
source_max_len = max(map(len, encoder_tokens))
# frase mas larga en español
target_max_len = max(map(len, decoder_tokens))

# rellenar con PAD segun la frase mas larga
encoder_tokens = [tokens + ['<PAD>']*(source_max_len-len(tokens)) for tokens in encoder_tokens]
decoder_tokens = [tokens + ['<PAD>']*(target_max_len-len(tokens)) for tokens in decoder_tokens]
print(encoder_tokens[120000])
print(decoder_tokens[120000])
# rellenar con PAD segun la frase mas larga
output_tokens = [tokens + ['<PAD>']*(target_max_len-len(tokens)) for tokens in output_tokens]
print(output_tokens[120000])

# Transformar los tokens "palabras" a numeros
encoder_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encoder_tokens]
decoder_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decoder_tokens]
print(encoder_input[120000])
print(encoder_input[120000])
#
output_decoded = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]
print(output_decoded[120000])

### CREAR la red transformer ###
# token_num numero maximo de palabras del diccionario de ingles y español 25269
# embed_dim emdeding de entrada 32
# encoder_num numero de encoders 6
# decoder_num numero de decodificadores 6
# head_num bloques atencionales busca las relaciones entre frases
# hidden_dim capa oculta de 128 neuronas
# dropout_rate desactivar 0.05 neuronas aleatoriamente para evitar overfitting
model = get_model(
        token_num= max(len(source_token_dict),len(target_token_dict)),
        embed_dim = 32,
        encoder_num = 2,
        decoder_num = 2,
        head_num = 8,
        hidden_dim=128,
        dropout_rate = 0.05,
        use_same_embed = False,
    )
model.compile('adam', 'sparse_categorical_crossentropy')
# resumen del modelo
model.summary()

# Entrenamiento del modelo
x = [np.array(encoder_input),np.array(decoder_input)]
y = np.array(output_decoded)
# Entrenar 30 epocas con grupos de 32 frases
#model.fit(x,y, epochs=30, batch_size=32)

# Guardar modelo
#model.save('translatorTransformer.h5')

# Guardar diccionario de palabras inverso numeros a palabras
#with open('target_token_dict_inv.pkl','wb') as fp:
#    pickle.dump(target_token_dict_inv,fp)

# Guardar diccionario de ingles a numeros
#with open('source_token_dict.pkl','wb') as fp:
#    pickle.dump(source_token_dict,fp)

# Guardar diccionario de español a numeros
#with open('target_token_dict.pkl','wb') as fp:
#    pickle.dump(target_token_dict,fp)


