# -*- coding: utf-8 -*-
import requests
import bs4
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from flask import Flask, render_template, request
import sys
import time
import pickle

application = Flask(__name__)

# # 그러면 allcontentlist_kanjidic = [{'한자':3,'한자':1},{'한자':1, '한자':2},...]
# # allcontentlist_kanjidic에서 각 딕셔너리중에서 가장 값이 높은것의 한자를 찾아내서 kd의 요소로서 초기화
# for x in range(len(allcontentlist_kanjidic)):
#     allcontentlist_kanjidic[x] = max(allcontentlist_kanjidic[x], key=allcontentlist_kanjidic[x].get)
# # alltitlelist 앞에 [키워드] 식으로 추가해주기
# for a in range(len(alltitlelist)):
#     alltitlelist[a] = '《' + allcontentlist_kanjidic[a] + '》' + alltitlelist[a]

with open('final_result.pkl', 'rb') as f:
    final_result_loaded = pickle.load(f)

with open('allcontentlist.pkl', 'rb') as f:
    allcontentlist_loaded = pickle.load(f)

with open('alltitlelist.pkl', 'rb') as f:
    alltitlelist_loaded = pickle.load(f)

alltitlelist_loaded_important = []
allcontentlist_loaded_important = []
alltitlelist_loaded_normal = []
allcontentlist_loaded_normal = []

for a in range(len(alltitlelist_loaded)):
    if final_result_loaded[a] == '중요':
        alltitlelist_loaded_important.append(alltitlelist_loaded[a])
        allcontentlist_loaded_important.append(allcontentlist_loaded[a])
    if final_result_loaded[a] == '기타':
        alltitlelist_loaded_normal.append(alltitlelist_loaded[a])
        allcontentlist_loaded_normal.append(allcontentlist_loaded[a])


# 리스트의 모든값을 html에 넣고 위에서부터 차례로 나열
@application.route("/")
def mainlist():
    usrid = request.args.get('usrid')
    pw = request.args.get('pw')
    return render_template("title.html", alltitlelist_loaded_important=alltitlelist_loaded_important, alltitlelist_loaded_normal=alltitlelist_loaded_normal, usrid=usrid, pw=pw)


@application.route("/important_admin_pw1122/<int:numoftitlelist>")
def showcontent_important(numoftitlelist):
    return render_template("content.html", thatnumcontent=allcontentlist_loaded_important[numoftitlelist])


@application.route("/normal_admin_pw1122/<int:numoftitlelist>")
def showcontent_normal(numoftitlelist):
    return render_template("content.html", thatnumcontent=allcontentlist_loaded_normal[numoftitlelist])


if __name__ == "__main__":
    application.run(host='0.0.0.0')