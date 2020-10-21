from __future__ import unicode_literals
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
    def my_hook(self, d):
        if d['status'] == 'finished':
            print('Download finished, now converting...')

    def to_wav(self, link):
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([link])

if __name__ == '__main__':
    test_link = 'https://youtu.be/GJMNjLRhjnU'
    youtube_to_wav = Youtube_to_Wav()
    youtube_to_wav.to_wav(test_link)