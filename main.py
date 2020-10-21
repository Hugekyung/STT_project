# from STT import SpeechtoText
from TS.Summarize import Textrank
from Youtube_to_wav import Youtube_to_Wav

# process
# youtube(link) to speech(wav)
# speech(wav) to text
# text summarize -> result

class SpeechtoSummarize():
    def __init__(self):
        # self.stt_model = SpeechtoText()
        self.ts_model = Textrank()
        self.youtube_to_wav = Youtube_to_Wav()

    def yts(self, link, wav_path='./test.wav'):
        self.youtube_to_wav.to_wav(link)

    def stt(self, wav_path):
        # text = self.stt_model.to_text(wav_path)
        text = '''안녕하세요. 저희 팀은 STS 모델을 제작하고 있습니다.
        STT 모델과 TS 모델을 합쳐서 만드는 모델입니다. 현재 TS 모델은 Textrank로 구현하였습니다. 
        STT 모델은 아직 미완성입니다. STT 모델이 완성되면 추가하도록 하겠습니다. 
        TS 모델의 딥러닝 모델도 구현 중입니다. 완전체가 갖춰질 때까지 기다려주세요.
        곧 찾아뵙겠습니다. 감사합니다.'''
        return text

    def ts(self, text, model='gensim'):
        return self.ts_model.summarizer(text, model)
    
    def sts(self, link, wav_path='./test.wav', textrank = 'gensim'):
        self.yts(link, wav_path)
        print('youtube 영상을 wav 파일로 저장하였습니다(경로: {})'.format(wav_path))
        text = self.stt(wav_path)
        print(text)
        summarize_result = self.ts(text, textrank)
        return summarize_result

if __name__ == '__main__':
    test_link = 'https://www.youtube.com/watch?v=tS_rgotOLe4'
    model = SpeechtoSummarize()
    test_result = model.sts(test_link, './test.wav','gensim')
    print(test_result)