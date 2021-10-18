from keras.models import load_model
import numpy as np
from train_lib import VideoSamples

###############################################################################
# LOAD MODEL

model = load_model('color/best_model.h5')

###############################################################################
# PREDICTIONS

sources = []
correct = 0
incorrect = 0

for _ in range(100):
    videos = VideoSamples('videos', color=True, total_samples=1, random=True)

    if videos.samples[0].source in sources:
        continue
    else:
        sources.append(videos.samples[0].source) 

    Z, y = videos.load_data(0.1, max_frames=55)

    expected = videos.samples[0].class_name
    predicted = videos.index_to_class(model.predict(Z))

    #print('-' * 80)
    #print(f'-I- path: {videos.samples[0].source}')
    #videos.samples[0].preview()
    print(f'-I- Expected: {expected}, predicted: {predicted}')

    if expected == predicted:
        correct += 1
    else:
        incorrect += 1

print('-' * 80)
print(f'-I- Correct:   {correct}')
print(f'-I- Incorrect:   {incorrect}')