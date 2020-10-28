#-*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import sys

from pydub import AudioSegment 
from pydub.silence import split_on_silence
import youtube_dl

class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

class Youtube_to_Wav():
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': './test.wav',
            'ignoreerrors' : True,
            'logger' : MyLogger(),
            # 'postprocessors': [{
            #     'key' : 'FFmpegExtractAudio',
            #     'preferredcodec' : 'wav',
            #     'preferredquality' : '0'
            #     }],
            'progress_hooks' : [self.my_hook],
        }
        self.link = ''
        self.youtube_code = ''

    def my_hook(self, d):
        if d['status'] == 'finished':
            print('Download finished, now converting...')

    def to_wav(self, link):
        self.link = link
        self.youtube_code = link.split('/')[-1]
        try:
            os.mkdir(self.youtube_code)
        except(FileExistsError):
            pass
        self.ydl_opts['outtmpl'] = './'+self.youtube_code+'/'+'{}.wav'.format(self.youtube_code)
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([link])

    def split(self, min_silence=600, silence_threshold = -16):
        input_path = './'+self.youtube_code
        output_path = os.path.join(self.youtube_code, 'chunks')
        # make output directory
        try:
            os.mkdir(output_path)
        except(FileExistsError):
            pass

        # find wav list and split each wav file
        file_lst = [x for x in os.listdir(input_path) if x.endswith('.wav')]
        for f in file_lst:
            self.long_to_short(os.path.join(input_path, f), output_path, min_silence, silence_threshold)


    def long_to_short(self, file, output_path, min_silence, silnece_threshold):
        # load sound file
        sound = AudioSegment.from_file(file)
        dBFS = sound.dBFS
        # split on silence
        chunks = split_on_silence(sound, 
            min_silence_len = min_silence,
            silence_thresh = dBFS + silnece_threshold,
            keep_silence = min_silence*0.8
        )

        # merge chunk files if chunk length is shorter than 2 seconds
        target_length = 2 * 1000
        output_chunks = [chunks[0]]
        for chunk in chunks[1:]:
            if len(output_chunks[-1]) < target_length:
                output_chunks[-1] += chunk
            else:
                # if the last output chunk is longer than the target length,
                # we can start a new one
                output_chunks.append(chunk)

        # save each chunk file
        filename = file.split('/')[-1][:-4]
        for i, chunk in enumerate(output_chunks):
            chunk_name = filename+"_chunk_{0}.wav".format(i)
            print("saving {}".format(chunk_name)) 
            # specify the bitrate to be 192 k 
            chunk.export(output_path + '/' + chunk_name, bitrate ='192k', format ="wav") 


def link_to_chunks(link):
    ytw = Youtube_to_Wav()
    os.chdir(os.path.dirname(__file__)+'/data')
    ytw.to_wav(link)
    ytw.split()
    

if __name__ == "__main__":
    test_link = 'https://youtu.be/N547u4ottA4'
    link_to_chunks(test_link)