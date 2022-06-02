# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pickle

application = Flask(__name__)

with open('final_result.pkl', 'rb') as f:
    final_result_loaded = pickle.load(f)

with open('alltitlelist.pkl', 'rb') as f:
    alltitlelist_loaded = pickle.load(f)

with open('allcontentlist.pkl', 'rb') as f:
    allcontentlist_loaded = pickle.load(f)

alltitlelist_loaded_important = []
allcontentlist_loaded_important = []
alltitlelist_loaded_normal = []
allcontentlist_loaded_normal = []


for a in range(len(alltitlelist_loaded)):
    if final_result_loaded[a] == 'imp':
        alltitlelist_loaded_important.append(alltitlelist_loaded[a])
        allcontentlist_loaded_important.append(allcontentlist_loaded[a])
    if final_result_loaded[a] == 'oth':
        alltitlelist_loaded_normal.append(alltitlelist_loaded[a])
        allcontentlist_loaded_normal.append(allcontentlist_loaded[a])


# insert all the titles and contents to html
@application.route("/")
def mainlist():
    usrid = request.args.get('usrid')
    pw = request.args.get('pw')
    return render_template("title.html", alltitlelist_loaded_important=alltitlelist_loaded_important, alltitlelist_loaded_normal=alltitlelist_loaded_normal, usrid=usrid, pw=pw)


@application.route("/important_admin_pw1122/<int:numoftitlelist>")
def showcontent_important(numoftitlelist):
    return render_template("content.html", thatnumtitle=alltitlelist_loaded_important[numoftitlelist], thatnumcontent=allcontentlist_loaded_important[numoftitlelist])


@application.route("/normal_admin_pw1122/<int:numoftitlelist>")
def showcontent_normal(numoftitlelist):
    return render_template("content.html", thatnumtitle=alltitlelist_loaded_normal[numoftitlelist], thatnumcontent=allcontentlist_loaded_normal[numoftitlelist])


if __name__ == "__main__":
    application.run(host='0.0.0.0')


