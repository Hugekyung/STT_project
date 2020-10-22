#-*- coding: utf-8 -*-

from krwordrank.sentence import summarize_with_sentences
from gensim.summarization.summarizer import summarize
# from textrank import KeysentenceSummarizer
from konlpy.tag import Kkma

class Textrank():

    def __init__(self):
        self.kkma = Kkma()

    def _sentence_tokenizer(self, text):
        sent_lst = self.kkma.sentences(text)
        return sent_lst
    
    def summarizer(self, text, option = 'krwordrank'): # 'gensim', 'textrank', 'krwordrank'
        sent_lst = self._sentence_tokenizer(text)
        if option == 'krwordrank':
            result = summarize_with_sentences(sent_lst, num_keysents=3)[1]
        elif option == 'gensim':
            result = summarize(text, ratio=0.3).split("\n")
        # else:
        #     ks_summarizer = KeysentenceSummarizer(tokenize = self.kkma.morphs)
        #     result = list(zip(*ks_summarizer.summarize(sents=sent_lst,topk=3)))[2]
        return result

if __name__ == '__main__':
    text = '[사진=게티이미지뱅크] [사진=게티이미지뱅크]\n\n요즘은 "오래사세요"(장수)보다 "건강하게 오래사세요"(건강수명)가 화두가 된 것 같다. 100세를 살아도 병으로 오래 누워 지내면 본인은 물론 자식에게 부담이 될 수 있다. 부모의 치료비와 간병비를 대기 위해 집을 팔았다는 자녀의 얘기는 우리를 우울하게 만든다. 어떻게 하면 건강수명을 유지할 수 있을까?◆ 우리 몸의 근육, 왜 중요할까건강수명을 누리려면 먼저 치매, 만성질환, 암 등 치료가 어렵고 투병기간이 긴 질병부터 예방해야 한다. 이를 위해서는 내 몸의 근육부터 지키고, 더욱 키우는 노력이 필요하다. 근력이 이런 질병들을 예방하고 빨리 치유하는 효과가 있기 때문이다.운동이 치매 예방에 도움이 된다는 사실은 의학적으로 검증이 됐다. 허벅지 근육이 탄탄하면 당뇨병 등 만성질환 예방에 좋다. 암에 걸려도 근육이 충분하면 그렇지 않은 사람보다 회복 속도가 빠르다. 예기치 않은 사고로 입원해도 근육이 부실한 사람보다 빨리 퇴원할 수 있다.◆ 근육, 생명유지 위해 꼭 필요하다건강한 사람도 40세가 넘으면 자연스럽게 근육이 줄어든다. 심하면 해 마다 1%씩 감소하는 사람도 있다. 특히 근육의 양 뿐만 아니라 근육 기능의 저하가 동시에 나타나면 건강에 적신호가 켜진다. 근육의 질까지 함께 나빠지면 근감소증의 징후인 것이다.우리 몸에서 소비되는 열량의 60-70%는 기초 대사량, 즉 아무런 활동이 없더라도 생명유지를 위해 쓰인다. 이 기초 대사량은 몸의 근육량에 의해 크게 좌우되므로 이를 유지하고 늘리는 것이 중요하다. 기초 대사량의 유지와 증가를 위해서 유산소운동과 근력운동을 병행하는 것이 좋다(국립암센터 자료).◆ 근감소증, 왜 무서운가근육이 크게 감소하면 먼저 걸음부터 느려진다. 앉았다 일어날 때도 평소보다 힘이 더 들고 관절의 통증도 심해진다. 근감소증까지 이어지면 기운이 없고 쉽게 피곤해지며 휴식 후에도 피로감이 남는다. 결국 자주 눕게 되어 증상 악화를 부채질하게 된다.근육이 급격히 줄면 어지러운 경우가 많고 골다공증도 쉽게 생긴다. 자주 넘어져 골절이 되고 뇌출혈로 연결된다. 질병에 걸렸을 때 투병기간이 길어지고 합병증이 잘 올 수 있다. 결국 지팡이, 휠체어를 빨리 사용하는 원인이 되며 요양시설 입원, 사망까지 이어지게 된다. 근육감소로 투병하는 노인들은 이런 과정들을 거치게 된다.◆ \'근육 저축\' 하고 계시나요?근육을 저축하고 키우기 위해서는 근력운동과 단백질 섭취, 비타민 D 섭취 등을 동시에 하는 것이 가장 좋다. 젊을 때부터 근력운동을 통해 근육의 힘을 많이 키워놓으면 크게 도움이 된다. 하지만 중년, 노년이 되어서도 근력운동을 하면 효과를 볼 수 있다.걷기가 안전한 운동이지만 근력을 키우는 효과가 떨어진다. 따라서 유산소운동과 함께 근력운동을 반드시 병행해야한다. 일상 속에서 빠르게 걷기와 비탈길, 계단 오르기를 같이 하면 좋다. 무릎에 이상이 없으면 스쿼트를 하는 게 도움이 된다. 아파트에서 생활할 경우 올라갈 때는 계단을 이용하고 내려 올 때는 엘리베이터를 타는 게 관절보호에 좋다.◆ 쉽게 구하고 저렴한 단백질 섭취법은?근육 보강을 위해서는 단백질이 포함된 적절한 영양섭취가 중요하다. 단백질은 육류 등 동물성 단백질이 좋은데, 나이가 들면 고기 섭취가 어렵고 피하게 되는 경향이 있다. 이럴 때 계란을 자주 먹으면 근육 손실을 막는데 도움이 된다.계란에는 필수아미노산인 루신이 많고 가격도 저렴해 단백질 보충에 제격이다. 전날 계란을 5개 정도 삶아두었다가 매일 아침 2개씩 먹으면 간편하게 단백질을 섭취할 수 있다. 식사 때 두부 등 콩 음식, 버섯 등을 곁들이면 더욱 좋다. 비싼 단백질 보충제를 사지 않아도 일상의 우리 음식을 통해서 단백질을 충분히 섭취할 수 있다.김용 기자 (ecok@kormedi.com)저작권ⓒ \'건강을 위한 정직한 지식\' 코메디닷컴( kormedi.com ) / 무단전재-재배포 금지'
    Textrank = Textrank()
    print(Textrank.summarizer(text, option='gensim'))