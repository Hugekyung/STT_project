from STT.las_pytorch import YTT
from TS.Summarize import Textrank

# process
# youtube(link) to speech(wav)
# speech(wav) to text
# text summarize -> result

class YoutubetoSummary():
    def __init__(self):
        self.textrank = Textrank()
        self.ytt = YTT

    def youtube_to_text(self, link):
        self.ytt.youtube_to_wav(link)
        result = self.ytt.wav_to_text()
        return result

    def text_to_summary(self, text, option):
        return self.textrank.summarizer(text, option)
    
    def youtube_to_summary(self, link, ts = 'tr', tr_option='krwordrank'):
        text_result = self.youtube_to_text(link)
        for x in text_result:
            if ts == 'tr':
                summarize_result = self.text_to_summary(x, tr_option)
            # bertsum 추가 예정
            else:
                pass
            for summarize in summarize_result:
                print(summarize)

if __name__ == '__main__':
    test_link = 'https://youtu.be/3IBycan2RS8'
    yts = YoutubetoSummary()
    yts.youtube_to_summary(test_link)