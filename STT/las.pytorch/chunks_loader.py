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

    def split(self, min_silence=400, silence_threshold = -18):
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
            keep_silence = min_silence*1.2
        )

        # merge chunk files if chunk length is shorter than 2 seconds
        target_length = 3 * 1000
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
            chunk_name = filename+"_chunk_{0:0>3}.wav".format(i)
            print("saving {}".format(chunk_name)) 
            # specify the bitrate to be 192 k
            file_name = output_path + '/' + chunk_name
            chunk.export(file_name, bitrate ='192k', format ="wav")
            self.wav_to_pcm(file_name[:-4])
            os.remove(file_name)
    
    def wav_to_pcm(self, file_name):
        print(os.getcwd())
        print('converting {}'.format(file_name))
        os.system("ffmpeg -i {0}.wav -f s16le -ac 1 -ar 16000 -acodec pcm_s16le {0}.pcm".format(file_name))

    def link_to_chunks(self, link):
        print(os.getcwd())
        os.chdir(os.path.dirname(__file__))
        os.chdir('../data/youtube')
        self.to_wav(link)
        self.split()
    

if __name__ == "__main__":
    ytw = Youtube_to_Wav()
    # import pandas as pd
    # csv_file = pd.read_csv('/home/ubuntu/workspace/STT_project_github/STT_project/STT/ytn_news.csv', encoding='utf-8')
    # urls = csv_file['url']
    # print(urls)
    # for idx, url in enumerate(urls):
    #     link_to_chunks(url)
    ytw.link_to_chunks('https://youtu.be/watch?v=OBdMbLa_-Ic')