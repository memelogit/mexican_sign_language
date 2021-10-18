#from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, MaxPooling2D, TimeDistributed, Conv2D, Flatten
from keras.layers.core import Dense
from tensorflow.keras.optimizers import Adagrad, Adadelta, RMSprop, Adam
from tensorflow.keras.callbacks import TensorBoard
from train_lib import VideoSamples
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Creamos el objeto VideoSamples
total_clases = 10
total_samples = 100
videos = VideoSamples('videos', color=True, total_samples=total_samples)

# Feature extraction model
# Agregamos dos capas de conv2D adicionales para reducir el número de parámetros
model = Sequential()                                                 #     (time-step, width, height, features)
model.add(TimeDistributed(Conv2D(4, (2,2), activation='relu'), input_shape=(None,      128,   72,    3))) # 1280 x 720
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='max_pool')))
model.add(TimeDistributed(Conv2D(8, (2,2), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='max_pool')))
model.add(TimeDistributed(Conv2D(12, (2,2), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='max_pool')))
model.add(TimeDistributed(Flatten(name='flat')))

# Hasta el momento tenemos un vector uni-dimensional que es una representación de la imagen.
model.add(LSTM(50, activation='relu'))
model.add(Dense(total_clases, activation='softmax')) # <-- Número de clases

# Optimizadores
# -------------------------------------------------------------------------------------------------
# *** Gradient Descendient ***
# En GD, SGD y mini-batch SGD el Larning Rate (lr) es siempre el mismo
# wt = wt-1 - (lr x DL/DW)
#
# *** AdaGrad ***
# En Adaptive Gradient Optimizer (AdaGrad), se tienen diferentes lr para cada neurona en cada iteración
# wt = wt-1 - (lrT x DL/DW)
# lrT = lr/sqr(at + E)^2
#    at = alpha de t = sumatoria i=0 hasta t (DL/DW_i)^2
#    E  = épsilon = un número muy pequeño solo para evitar que sea cero el denomidador del lr
# entonces... a medida que incrementa el at decrementa el lr
#
# *** Adadelta / RMSprop (Root Mean Square Propagation) ***
# El problema con el AdaGrad es que al incrementar alpha_t, el lr se vulve muy pequeño lo que provoca
# un estancamiento en wt. Entonces hacemos un pequeño cambio al lr
# lrT = lr/sqr(w_avg_t + E)^2
#    w_avg_t = gama x w_avg_t-1 + ((1 - gama) x (DL/DW)^2)
#    gama = 0.95 por lo general
# entonces... el lr decrementa poco a poco
#
# *** Adam (AdaGrad + RMSprop) Adaptive momentum ***
# mt = beta_1 x mt-1 + (1-beta_1) x gt     -> AdaGrad, beta_1 = 0.9
# vt = beta_2 x vt-1 + (1-beta_2) x gt^2   -> RMSprop, beta_2 = 0.99999
#    gt = 10 (default)
# ajuste de bias
#    mt_^ = mt / (1 - beta_1)
#    vt_^ = vt / (1 - beta_2)
# actualizamos el paso
#    wt = wt-1 - alpha(mt_^ / sqr(vt_^ + epsilon))
#    alpha = lr = 0.0001 (default)
# entonces... el lr da saltos rápidos y luego se van reduciendo
# -------------------------------------------------------------------------------------------------
# Adam()                         -> converge a 0.002 pero después se va hacia el cielo
# Adagrad(learning_rate=0.8)     -> se estanca el loss
# RMSprop()                      -> disminuye el loss muy despacio
# RMSprop(learning_rate=0.01)    -> se fue al infinito
# RMSprop(learning_rate=0.005)   -> loss demasiado alto
# RMSprop(learning_rate=0.001)   -> aún no converge tan bien, vámonos más depacio
# RMSprop(learning_rate=0.0001)  -> funciona de maravilla
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.0001),
    metrics=['accuracy']
)
print(model.summary())

# Best model callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='color/best_model.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='auto',
    save_best_only=True,
    verbose=1
)
tensorboard_callback = TensorBoard(log_dir='color', histogram_freq=1)

# Generamos los samples a 55 frames
print('-I- Generating video samples. It will take a while...')
X, y = videos.load_data(0.1, max_frames=55) # Crop video duration to 55 frames
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Entrenamos el modelo
model.fit(
    X_train, y_train,
    batch_size=10,
    epochs=100,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=[model_checkpoint_callback, tensorboard_callback]
)

# Evaluación del modelo
# videos = VideoSamples('videos', color=True, total_samples=20)
# X, y = videos.load_data(0.1)
# loss, acc = model.evaluate(X, y, verbose=1)
# print(f'loss: {loss:.6f}, acc: {acc:.2f}')

# TENSORBOARD
# Para ver los detalles del entrenamiento
# % tensorboard --logdir color