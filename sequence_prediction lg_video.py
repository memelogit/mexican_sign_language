import os
from keras.models import load_model
from train_lib import VideoSequence

###############################################################################
# LOAD MODEL

model = load_model('color/best_model.h5')

###############################################################################
# PREDICTIONS

video = VideoSequence(
    os.path.join('secuencias', 'saltar_andy.mp4'),
    #os.path.join('secuencias', 'gustar.mp4'),
    #os.path.join('secuencias', 'oler.mp4'),
    color=True,
    classes=['abrir', 'cerrar', 'comer', 'conocer', 'descansar', 'gustar', 'ir', 'oler', 'salir', 'saltar']
)

video.preview()

video_length = 55

# Z, y = video.load_data(0.1, max_frames=video_length, seek=100)
# predicted = video.index_to_class(model.predict(Z))
# print(f'-I- predicted: {predicted}')

print(f'-I- Total frames in video: {video.total_frames}')
for seek in range(0, video.total_frames - video_length, 10):
    Z, y = video.load_data(0.1, max_frames=video_length, seek=seek)
    acc, predicted = video.index_to_class(model.predict(Z))
    if acc >= 0.4:
        print(f'-I- predicted: {predicted} at {seek}-{seek + video_length} frame, acc:{acc}')