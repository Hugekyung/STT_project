# from STT import SpeechtoText
from TS.Summarize import Textrank

class SpeechtoSummarize():
    def __init__(self):
        # self.stt_model = SpeechtoText()
        self.ts_model = Textrank()

    def stt(self, file):
        # text = self.stt_model.to_text()
        text = '''안녕하세요. 저희 팀은 STS 모델을 제작하고 있습니다.
        STT 모델과 TS 모델을 합쳐서 만드는 모델입니다. 현재 TS 모델은 Textrank로 구현하였습니다. 
        STT 모델은 아직 미완성입니다. STT 모델이 완성되면 추가하도록 하겠습니다. 
        TS 모델의 딥러닝 모델도 구현 중입니다. 완전체가 갖춰질 때까지 기다려주세요.
        곧 찾아뵙겠습니다. 감사합니다.'''
        return text

    def ts(self, text, model='gensim'):
        return self.ts_model.summarizer(text, model)
    
    def sts(self, file, textrank = 'gensim'):
        text = self.stt(file)
        print(text)
        summarize_result = self.ts(text, textrank)
        return summarize_result

if __name__ == '__main__':
    test_file = r'D:\git\STT_project\STT\data\Youtube_test\youtube_test_short2.wav'
    model = SpeechtoSummarize()
    test_result = model.sts(test_file, 'gensim')
    print(test_result)