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
    url_format = "https://news.naver.com/main/list.nhn?mode=LPOD&mid=sec&oid=421&date={}"


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
            
                # 뉴스 원문 url
                # https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=102&oid=032&aid=0003037750
                # oid: 언론사 구분 번호, aid: 해당 기사 할당 번호
                
                # news_url = response.css("div.article_btns_right > a::attr(href)").get()
                # oid = re.search('oid=([0-9]..)', news_url).group()
                # oid = re.search('[0-9]+', oid).group()
                # aid = re.search('(\d+)(?!.*\d)', news_url).group()
                # print('aid는 이거~~~~~~', aid)
                # summary_url = "https://tts.news.naver.com/article/{}/{}/summary?callback=window".format(oid, aid)
                # yield scrapy.Request(summary_url, callback=self.parse_summary_bot, meta={"summary_url" : summary_url})

            yield {
                    "upload_date": upload_date,
                    "media": media,
                    "title": title,
                    "text": text,
                    # "summary_text": summary_text,
            }
        except:
            pass


    # def parse_summary_bot(self, response):
    #     print('요약봇 response 잘 받앗음!!!!!!', response)
    #     # # window = response.json()
    #     # # response_json = response_json.decode('utf-8').replace('\0', '')
    #     # form = response.decode('utf-8')
    #     # window = requests.get(response.meta["summary_url"]).json(form)
    #     # summary_text = window[0]['summary']
    #     # # print(window)
        
    #     summary_text = response.xpath("/html/body/div[1]/div[6]/div[1]/div/div/div/div[5]/div[9]/pre/span/span[2]").get() # 뭔가 문제가 있다...
    #     # summary_text = response.css("span.cm-string::text").get()
    #     # summary_text = response.css("div:nth-child(9) > pre > span > span::text").get()
    #     return summary_text