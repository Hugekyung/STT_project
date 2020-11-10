# -*- coding: utf-8 -*-
import scrapy
import re
from datetime import datetime
import pandas as pd
import time
import json
import requests
# 네이버 뉴스 '사회' 카테고리
# 기사 수 TOP3 연합뉴스, 뉴시스, 뉴스1
# 연합뉴스 https://news.naver.com/main/list.nhn?mode=LPOD&mid=sec&oid=001&date=20201023
# 뉴시스 https://news.naver.com/main/list.nhn?mode=LPOD&mid=sec&oid=003&date=20201023
# 뉴스1 https://news.naver.com/main/list.nhn?mode=LPOD&mid=sec&oid=421&date=20201023

class CrawlNews(scrapy.Spider):
    name = "news"
    allowed_domains = ["news.naver.com", "tts.news.naver.com"]
    url_format = "https://news.naver.com/main/list.nhn?mode=LPOD&mid=sec&oid=001&date={}"


    def __init__(self):
        start = input("시작 날짜를 입력하세요(ex.2020-09-30): ")
        end = input("마지막 날짜를 입력하세요(ex.2020-09-30): ")
        startdate =  datetime.strptime(start, "%Y-%m-%d")
        enddate =  datetime.strptime(end, "%Y-%m-%d")

        self.start_urls= []
        for cur_date in pd.date_range(startdate, enddate):
            self.start_urls.append(self.url_format.format(cur_date.strftime("%Y%m%d")))


    def start_requests(self):
        for start_url in self.start_urls: # 가져올 페이지를 리스트에 저장
            # timer = round((time.time() - self.starttime) / 60)
            # percentage = round(self.request_count / len(self.inner_id_ls) * 100, 2)
            # print("{}분 경과, {}개 중 {}개 request 완료({}%)".format(timer, self.request_count, len(self.inner_id_ls), percentage))
            yield scrapy.Request(url=start_url, callback=self.parse, meta={"start_url" : start_url, "page_num": 1})
            time.sleep(5)


    def parse(self, response):
        page_num = response.meta["page_num"]
        if int(response.css("div.paging strong::text").get()) != page_num:
            return # 함수 종료
        else:
            for item in response.css("ul.type06_headline li"):
                url = item.css("a::attr(href)").get()
                yield scrapy.Request(url, callback=self.parse_detail)

            for item2 in response.css("ul.type06 li"):
                url2 = item2.css("a::attr(href)").get()
                yield scrapy.Request(url2, callback=self.parse_detail)
                
        # 다음 페이지 url
        page_num += 1
        start_url = response.meta["start_url"]
        next_page = start_url + "&page={}".format(page_num)
        print("다음 페이지============", next_page)
        yield scrapy.Request(next_page, callback=self.parse, meta={"start_url" : start_url, "page_num" : page_num})
    

    def parse_detail(self, response):
        try:
            if response.css("div.media_end_head_autosummary._auto_summary_wrapper > a::text").get() == '요약봇':
                upload_date = response.css("div.sponsor span.t11::text").get()
                media = response.css("div.press_logo img::attr(title)").get()
                title = response.css("h3#articleTitle::text").get()
                text = ''.join(response.css("div#articleBodyContents::text").getall()).replace("\n","").strip()

                # text 전처리
                temp_text = text
                temp_text = re.sub('\(([A-Za-z가-힣]*=[가-힣]*\))','',str(temp_text)) # (서울=연합뉴스) or (AP=연합뉴스) 형식 제거
                temp_text = re.sub('.*[가-힣] 기자 =','',temp_text) # 기자이름 제거
                temp_text = re.sub('.*[가-힣] 특파원 =','',temp_text)
                temp_text = re.sub('.*[가-힣] 기자=','',temp_text)
                temp_text = re.sub('    ','',temp_text)
                temp_text = re.sub('\'','',temp_text)
                temp_text = re.sub('.*[가-힣]기자','',temp_text) # 기자이름 제거
                temp_text = re.sub('\\xa0','',temp_text) # .*\\xa0.* 포함된 거 제거
                temp_text = re.sub('[A-Za-z]+@yna.co.kr.*','',temp_text) # 이메일 제거
                temp_text = re.sub('[A-Za-z]+@yonhapnews.co.kr.*','',temp_text) # 이메일 제거
                temp_text = re.sub('([0-9a-zA-Z]{1,100}\.[0-9a-zA-Z]{0,100}%?)','',temp_text) # 00.00형태 삭제
                # temp_text = re.sub('[가-힣]{1}\.','',temp_text) # .제거
                temp_text.strip()
                text = temp_text

                # 뉴스 원문 url
                # https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=102&oid=032&aid=0003037750
                # oid: 언론사 구분 번호, aid: 해당 기사 할당 번호
                
                news_url = response.css("div.article_btns_right > a::attr(href)").get()
                oid = re.search('oid=([0-9]..)', news_url).group()
                oid = re.search('[0-9]+', oid).group()
                aid = re.search('(\d+)(?!.*\d)', news_url).group()
                # print('aid는 이거~~~~~~', aid)
                summary_url = "https://tts.news.naver.com/article/{}/{}/summary?callback=window".format(oid, aid)
                yield scrapy.Request(summary_url, callback=self.parse_summary_bot, meta={"summary_url" : summary_url, "upload_date" : upload_date, "media" : media, "title" : title, "text" : text})
        except:
            pass


    def parse_summary_bot(self, response):
        print('요약봇 response 잘 받앗음!!!!!!', response)
        upload_date = response.meta["upload_date"]
        media = response.meta["media"]
        title = response.meta["title"]
        text = response.meta["text"]
        
        try:
            window = response.text # json형식이 아니다
            window = re.sub('.+("summary":)', '', window)
            sent_lst = window.split('<br/><br/>')
            summary_text = [re.sub('[\\\\"});]+', '', x) for x in sent_lst]
            # print(summary_text)
        except:
            print('error!')
        
        yield {
                    "upload_date": upload_date,
                    "media": media,
                    "title": title,
                    "text": text,
                    "summary_text": summary_text,
            }