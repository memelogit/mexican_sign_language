import cv2
import numpy as np
from glob import glob
from os import path
from pathlib import Path
from random import choice
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class Video:
    def __init__(self, source:str, color:bool = True):
        self._source = source
        self._capture = cv2.VideoCapture(source)
        assert self._capture.isOpened(), 'Cannot load video. Please verify the source path'
        self._color = color
        self._size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def source(self):
        return self._source
    
    @property
    def size(self):
        return self._size
    
    @property
    def total_frames(self):
        return self._frame_count
    
    @property
    def class_name(self):
        return Path(self._source).parts[1]
    
    def resize(self, *args:list):
        ''' Resize the video. Arguments can be: (width, heigh) or percentage
        '''
        if len(args) == 1:
            self._size = (int(self._size[0] * args[0]), int(self._size[1] * args[0]))
        elif len(args) == 2:
            self._size = (args[0], args[1])
        else:
            SyntaxError('Invalid number of arguments')
    
    def __resize_frame(self, frame):
        return cv2.resize(frame, self._size, interpolation = cv2.INTER_AREA) 

    def preview(self):
        self._capture.set(1, 1)
        while self._capture.isOpened():
            ret, frame = self._capture.read()
            if ret:
                if not self._color:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Video Preview', self.__resize_frame(frame))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    
    def get_batch(self, max_frames:int = 100, seek:int = 0) -> np.array:
        self._capture.set(1, 1)
        time_steps = max_frames if self._frame_count > max_frames else self._frame_count
        #print('-I- time_steps:', time_steps)
        frames = np.empty(
            (time_steps, self._size[1], self._size[0], 1 if not self._color else 3),
            np.dtype(float)
        )
        frame_counter = 0
        
        seek_counter = 0
        while self._capture.isOpened():
            ret, frame = self._capture.read()
            if ret:
                # seek
                if seek_counter < seek:
                    seek_counter += 1
                    continue

                frame = self.__resize_frame(frame)

                if not self._color:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = frame.reshape(self._size[1], self._size[0], 1)
                
                frames[frame_counter] = frame/255.0
                frame_counter += 1
                if frame_counter >= max_frames:
                    break
            else:
                break
        #print(f'-I- seek started at {seek_counter}')
        #print(f'-I- total frames {len(frames)}')
        
        return frames

    def __del__(self):
        self._capture.release()
    
    def __len__(self):
        return self._frame_count

class VideoSamples:
    def __init__(self, repository:str, total_samples:int = 1, color:bool = True, random:bool = False):
        self._samples = []
        self._video_files = []
        self._repo = repository
        self._color = color
        self._total_samples = total_samples

        for video_file in glob(path.join(self._repo, '*/*.mp4')):
            self._video_files.append(video_file)
        
        for i in range(total_samples):
            if random:
                video = self.get_random_video()
            else:
                video = Video(self._video_files[i], self._color)
            self._samples.append(video)
    
    def get_random_video(self) -> Video:
        return Video(choice(self._video_files), self._color)
    
    @property
    def classes(self):
        y = []
        label_encoder = LabelEncoder()
        for i, sample in enumerate(self._samples):
            y.append(sample.class_name)
        return to_categorical(label_encoder.fit_transform(y))

    @property
    def samples(self):
        return self._samples
    
    def load_data(self, size, max_frames:int = 100):
        label_encoder = LabelEncoder()
        X, y = [], []
        for i, sample in enumerate(self._samples):
            sample.resize(size)
            width, height = sample.size[0], sample.size[1]
            X.append(sample.get_batch(max_frames))
            y.append(sample.class_name)
            #print(f'\r{i+1:04} {sample.class_name}')
        features = 1 if self._color == False else 3
        # [samples, time-step, width, height, features]
        X = np.array(X).reshape(self._total_samples, max_frames, height, width, features)
        y = np.array(y).reshape(self._total_samples, 1)
        y = to_categorical(label_encoder.fit_transform(y))
        return X, y
    
    def index_to_class(self, output_array:np.array):
        index = (np.argmax(output_array, axis=-1)[0]).astype(int)
        class_list = list(set(Path(video).parts[1] for video in self._video_files))
        class_list.sort()
        #print(class_list)
        return class_list[index]

class VideoSequence(Video):

    def __init__(self, source: str, color: bool = True, classes = []):
        super().__init__(source, color=color)
        self.class_list = classes
    
    def load_data(self, size, max_frames:int = 100, seek:int = 0):
        label_encoder = LabelEncoder()
        X, y = [], []

        # we only need to resize the first time
        if seek == 0: self.resize(size)

        width, height = self.size[0], self.size[1]
        X.append(self.get_batch(max_frames, seek=seek))
        y.append(self.class_name)
        
        features = 1 if self._color == False else 3
        X = np.array(X).reshape(1, max_frames, height, width, features)
        y = np.array(y).reshape(1, 1)
        y = to_categorical(label_encoder.fit_transform(y))
        return X, y
    
    def index_to_class(self, output_array:np.array):
        index = (np.argmax(output_array, axis=-1)[0]).astype(int)
        index_value = max(output_array[0])
        #print(f'-I- index max: {index_value}')
        return index_value, self.class_list[index]

if __name__ == '__main__':
    video = Video('videos/abrir/IMG_2401.mp4')
    print(video.class_name)
    video.resize(0.1)
    video.preview()

    videos = VideoSamples('videos', color=True, total_samples=1)
    print(videos.load_data(0.1, max_frames=60))