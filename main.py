from STT.las_pytorch import YTT
from TextRank.Summarize import Textrank
# from BertSum.src.final import data_create, do_BertSum

from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
from krwordrank.word import summarize_with_keywords


# process
# youtube(link) to speech(wav)
# speech(wav) to text
# text summarize -> result

class YoutubetoSummary():
    def __init__(self):
        self.textrank = Textrank()
        self.ytt = YTT
        self.okt=Okt()
        self.link = ''

    def youtube_to_text(self, link):
        self.ytt.youtube_to_wav(link)
        result = self.ytt.wav_to_text()
        return result

    def textrank_summary(self, text, option):
        return self.textrank.summarizer(text, option)
    
    # def bert_summary(self, text):
    #     data_create(text)
    #     summary = do_BertSum()
    #     return summary

    def wordcloud(self, count_na):
        words_na=dict(count_na.most_common())
        # matplotlib.rc("font",family="Malgun Gothic")
        # set_matplotlib_formats("retina")
        # matplotlib.rc("axes",unicode_minus=False)
        font_path = '/home/ubuntu/workspace/STT_project_github/STT_project/malgunbd.ttf'
        wordcloud = WordCloud(font_path = font_path, background_color="white",
            width = 2000,
            height = 1500
        )
        wordcloud=wordcloud.generate_from_frequencies(words_na)
        plt.imshow(wordcloud)
        plt.axis("off")
        print('{} 영상에 대한 워드 클라우드 추출 결과입니다.'.format(self.link))
        plt.show()

    def count(self, text):
        ls="<q>".join(text.split(". ")).replace(".","").split("<q>")
        result_ls=[]
        for sen in ls:
            result_ls.extend(self.okt.pos(sen))
        na = []
        for x in result_ls:
            if x[1] == 'Noun':
                na.append(x[0])
        count_na=Counter(na)
        return count_na


    def youtube_to_summary(self, link, ts = 'tr', tr_option='krwordrank'):
        self.link = link
        text_result = self.youtube_to_text(link)
        for x in text_result:
            if ts == 'tr':
                summarize_result = self.textrank_summary(x, tr_option)
            # elif ts == 'bs':
            #     summarize_result = self.bert_summary(x)
            print('-'*109)
            # print(' ')
            print('{} 영상에 대한 요약 결과입니다.'.format(link))
            # print('-'*40+'요약 결과'+'-'*40)
            for summarize in summarize_result:
                print('> ' + summarize)
                # self.speak(summarize)
            # print(' ')
            print('-'*109)
            # print(' ')
            count_na = self.count(x)
            # keyword = '#' + ' #'.join([x for x, y in count_na.most_common(6)])
            keyword_dic = summarize_with_keywords(x.split(". "), num_keywords=5, min_count =3)
            keyword = '#' + ' #'.join([x for x in list(keyword_dic.keys())])
            print('{} 영상의 키워드: {}'.format(link, keyword))
            # print(' ')
            print('-'*109)
            # print(' ')
            self.wordcloud(count_na)
            print('-'*109)
            # print(' ')
            print('{} 영상의 전체 텍스트입니다.'.format(link))
            for sent in x.split('. '):
                print(sent+'.')

if __name__ == '__main__':
    # 'https://youtu.be/3IBycan2RS8'
    test_link = input('유튜브 링크를 입력해 주세요: ')
    yts = YoutubetoSummary()
    yts.youtube_to_summary(test_link, ts='tr')