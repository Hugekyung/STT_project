from STT.las_pytorch import YTT
from TS.Summarize import Textrank
from BertSum.src.final import data_create, do_BertSum

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

    def textrank_summary(self, text, option):
        return self.textrank.summarizer(text, option)
    
    def bert_summary(self, text):
        data_create(text)
        summary = do_BertSum()
        return summary
    
    def youtube_to_summary(self, link, ts = 'tr', tr_option='krwordrank'):
        text_result = self.youtube_to_text(link)
        for x in text_result:
            if ts == 'tr':
                summarize_result = self.textrank_summary(x, tr_option)
            # bertsum 추가 예정
            elif ts == 'bs':
                summarize_result = self.bert_summary(x)
            print('-'*50+'요약 결과'+'-'*50)
            for summarize in summarize_result:
                print(summarize)
            print('-'*109)

if __name__ == '__main__':
    # 'https://youtu.be/3IBycan2RS8'
    test_link = input('유튜브 링크를 입력해 주세요: ')
    yts = YoutubetoSummary()
    yts.youtube_to_summary(test_link, ts='bs')