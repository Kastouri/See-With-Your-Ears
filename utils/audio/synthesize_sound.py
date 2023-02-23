import glob
import numpy as np
import os
import re
import sys
import soundfile as sf
import pydub 
from pydub import AudioSegment
from pydub.playback import play as play_audio
from time import sleep
from threading import Timer, Thread, Event
import threading
import librosa
import pygame
import wave

SOUND_GENERATION_INTERVAL = .1


def setInterval(interval):
    """Decorator that calls repeats functions call every 'interval' Period from:
    https://stackoverflow.com/questions/12435211/threading-timer-repeat-function-every-n-seconds"""
    def decorator(function):
        def wrapper(*args, **kwargs):
            stopped = threading.Event()

            def loop(): # executed in another thread
                while not stopped.wait(interval): # until stopped
                    function(*args, **kwargs)

            t = threading.Thread(target=loop)
            t.daemon = True # stop if the program exits
            t.start()
            return stopped
        return wrapper
    return decorator


class SoudSynthesizer:
    def __init__(self, img_size, sound_source='hrir/diffuse', sample_rate=44100, regenerate_sound_interval=0.010, save_audio=True) -> None:
        self.sound_sources = sound_source
        self.img_size = img_size
        # initialize pygame sound player
        self.sample_rate = sample_rate
        self.mixer_initialized = False

        # initialize sound players for each object
        self.object_sounds = {}
        # initialize detections
        self.detections = {}

        self.channel = None

        # config TODO: define sounds for more classes
        self.config = { 
                    'classes': 
                        {
                            'cup': {'sound': 'hi_hat'},
                            'keyboard': {'sound': 'snare'},
                            'mouse': {'sound': 'tom'}
                        },
                    'save_audio': save_audio,
                    'save_audio_period': 50,
                    'audio_output': 'data/audio_outputs/'
                    }
        if self.config['save_audio']:
            self.audio_output_i = 0
            self.audio_output_np = None
            
        # load datasets
        self.load_sources()

        self.load_sounds()

    def interpolate_hrir(self, elevation, azimuth):
        # index of the closest hrir sample 
        distances = self.angles_quantization - np.array([elevation, azimuth])
        idx = np.argmin(np.linalg.norm(distances, axis=1))
        elevation, azimuth = self.angles_quantization[idx]

        
        return self.hrir_data[(int(elevation), int(azimuth))]

    def load_sounds(self):
        for class_id in self.config['classes'].keys():
            self.sounds[class_id] = self.load_track(f"data/sounds/{self.config['classes'][class_id]['sound']}.wav")[0]
            
        
    def load_track(self, path):
        [sound, fs_s] = sf.read(path)
        sound = self.signal_to_int(sound).T
        sound = np.column_stack((sound, sound))
        return sound, fs_s
    
    def load_sources(self):
        """loads HIRIR dataset and saves it to a dictionary that can be accessed using the angles
        """
        if self.sound_sources != 'hrir/diffuse':
            raise NotImplementedError
        hrir_dir = "data/hrir/diffuse/*/*"
        hrir_src = glob.glob(hrir_dir)

        self.hrir_data = {}
        self.sounds = {}
        self.angles_quantization = []
        for hrir_ir_p in hrir_src:
            # hrir_ir_p.split('.wav')[0].split('H').split()
            hrir_ir_file_n = hrir_ir_p.split('/')[-1].split('.wav')[0]
            elevation = int(re.findall(r'H(.*?)e', hrir_ir_file_n)[0])
            azimuth = int(re.findall(r'e(.*?)a', hrir_ir_file_n)[0])
            [hrir, fs_h] = sf.read(hrir_ir_p) 
            self.hrir_data[(elevation, azimuth)] = [hrir, fs_h]
            self.hrir_data[(elevation, -azimuth)] = [hrir[:, [1,0]], fs_h]
            # save quantization levels
            self.angles_quantization.extend([[elevation, azimuth], [elevation, -azimuth]])
                
        self.angles_quantization = np.array(self.angles_quantization)
        
    def play_sound(self, elevation, azimuth):

        hrir, _ = self.interpolate_hrir(elevation, azimuth)
        # convolve with example sound
        if self.sounds is None:
            [sound, fs_s] = sf.read("data/sounds/hi_hat.WAV")
            sound = self.signal_to_int(sound).T
            sound = np.column_stack((sound, sound))
            self.sounds = sound, fs_s
        else:
            sound, fs_s = self.sounds 

        sound_3d = self.compute_hr_response(sound, hrir)
        sound_player = pygame.sndarray.make_sound(sound_3d)

    def compute_hr_response(self, sound_singnal, hrir_signal):
        sound_singnal = self.signal_to_float(sound_singnal)
        if len(sound_singnal.shape)>1:
            sound_mono = np.mean(sound_singnal, axis=1)
        else:
            sound_mono = sound_singnal
        s_l = np.convolve(sound_mono, hrir_signal[:, 0])
        s_r = np.convolve(sound_mono, hrir_signal[:, 1])
        s_spatial = np.column_stack((s_l, s_r))

        return self.signal_to_int(s_spatial)
        
    def signal_to_int(self, signal):
        signal = (32768 * signal).astype(np.int16)
        return signal

    def signal_to_float(self, signal):
        signal = (signal.astype(np.float16) / 32768.)
        return signal

    def run(self):
        self.synthesize_3d_sound()
    
    @setInterval(SOUND_GENERATION_INTERVAL)
    def synthesize_3d_sound(self):
        detected_objects = False
        # copy detection 
        detections = self.detections.copy()
        
        if not self.mixer_initialized:
            pygame.mixer.pre_init(self.sample_rate, size=-16, channels=2, buffer=2048)
            pygame.mixer.init() 
            self.mixer_initialized = True

        # for every detection
        sound_total = np.zeros((int(SOUND_GENERATION_INTERVAL * self.sample_rate), 2), dtype=np.int16)

        for id, obj_dict in detections.items():
            # TODO: Use mutex or copy value at the beginning to avoid race conditions
            if obj_dict['class_id'] not in self.config['classes'].keys():
                continue
            obj_class = obj_dict['class_id']
            elevation = obj_dict['elevation'] 
            azimuth = obj_dict['azimuth'] 
            area = (obj_dict['u_right'] * obj_dict['v_bott']) 
            area /= (self.img_size[0]*self.img_size[1])
            area = min(area, 0.5)
            pause = round(1/area**2-3)  # experimental
            
            # if detection is new create new sound player
            if id not in self.object_sounds.keys():
                # if detection is old, compute convolution from last reached point
                self.object_sounds[id] = {'sound': self.sounds[obj_class],
                                          'current_i': 0, 'pause': 0, 'sound_3d': None}

            # skip object if the sound is pause:
            if self.object_sounds[id]['pause'] > 0.:
                self.object_sounds[id]['pause'] -= SOUND_GENERATION_INTERVAL
                self.object_sounds[id]['pause'] = max(self.object_sounds[id]['pause'], 0.)
                continue
                        
            # convert audio to 3d audio
            obj_audio_segment = self.object_sounds[id]['sound']
            hrir, _ = self.interpolate_hrir(elevation, azimuth)

            if self.object_sounds[id]['sound_3d'] is None:
                self.object_sounds[id]['sound_3d'] = self.compute_hr_response(obj_audio_segment, hrir) 
            interval_num_samples = int(SOUND_GENERATION_INTERVAL * self.sample_rate)
            i_end = min(interval_num_samples, len(self.object_sounds[id]['sound_3d']))
            if i_end < interval_num_samples:
                self.object_sounds[id]['sound_3d'] = None
                self.object_sounds[id]['pause'] = pause * SOUND_GENERATION_INTERVAL # pause for one second
                continue
            sound_3d = self.object_sounds[id]['sound_3d'][:i_end]
            self.object_sounds[id]['sound_3d'] = self.object_sounds[id]['sound_3d'][i_end:]

            sound_total += sound_3d   
            detected_objects = True 

        if detected_objects:
            sound_player = pygame.sndarray.make_sound(sound_total)    
            if self.channel is None:
                self.channel = sound_player.play()
            else:
                self.channel.queue(sound_player)

        if self.config['save_audio']:
            
            if self.audio_output_np is None:
                self.audio_output_np = sound_total
            else:
                self.audio_output_np = np.concatenate((self.audio_output_np, sound_total), axis=0)
            if self.audio_output_i == round(self.config['save_audio_period']/SOUND_GENERATION_INTERVAL):
                sound_output = pygame.mixer.Sound(self.audio_output_np)

                audio_out_f = wave.open(self.config['audio_output'] + 'output.wav', 'w')
                audio_out_f.setframerate(self.sample_rate)
                audio_out_f.setnchannels(2)
                audio_out_f.setsampwidth(2)
                audio_out_f.writeframesraw(sound_output.get_raw())
                exit(0)
            
            self.audio_output_i += 1

    def update_synthesizer(self, detections):
        self.detections = detections


def signal_to_int(signal):
    signal = (32768 * signal).astype(np.int16)
    return signal


def test_loop():
    """Test to find out the best way to play sound in a loop, on portion at a time
    """
    period = 0.01

    [sound, fs_s] = sf.read("data/sounds/hi_hat.WAV")
    sample_rate = fs_s
    pygame.mixer.pre_init(sample_rate, size=-16, channels=2, buffer=1024)
    pygame.mixer.init() 
    # load sound 

    sound = signal_to_int(sound).T
    sound = np.column_stack((sound, sound))
    sound_player_old = pygame.sndarray.make_sound(sound[0:int(sample_rate*period)])   
    channel_old = sound_player_old.play()
    sound_player_old.stop()

    # play segments of the sound
    for i in range(0, len(sound), int(sample_rate*period)):
        #sound_player.stop()
        channel_is_busy = channel_old.get_busy()

        sound_player_neu = pygame.sndarray.make_sound(sound[i:i+int(sample_rate*period)])    
        channel_neu = channel_old.queue(sound_player_neu)


if __name__ == "__main__":
    # Test audio synthesizer
    # sd = SoudSynthesizer()
    i = 1
    while True:
        test_loop()
        print(i)
        i += 1
