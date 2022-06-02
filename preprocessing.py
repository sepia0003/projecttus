from selenium import webdriver
import time
import pickle


# function which scraps all online titles & contents
def getonebulletin(bulletinnum, alltitlelist, allcontentlist):
    # saving number of titles per bulletin
    subcontamut = browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:' + str(bulletinnum) + ':htmlDisplayOfAll:0:htmlCountCol21702"]').text
    subcontamut = subcontamut.rstrip("件")
    subcontamut = subcontamut.lstrip("全")
    subcontamut = int(subcontamut)

    # checking "more" button. if there is, click it. if not, scrap as it is.
    showall = 1
    try:
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:' + str(bulletinnum) + ':htmlDisplayOfAll:0:htmlCountCol217"]').click()
    except:
        for i in range(subcontamut):
            elem = browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:' + str(bulletinnum) + ':htmlDetailTbl:' + str(i) + ':htmlTitleCol1"]')
            alltitlelist.append(elem.text)

            # clicking titles, changing the window and saving contents in allcontentlist
            browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:'+str(bulletinnum)+':htmlDetailTbl:'+str(i)+':htmlTitleCol1"]').click()
            time.sleep(1)
            browser.switch_to.window(browser.window_handles[1])
            elem = browser.find_element_by_xpath('//*[@id="form1:htmlMain"]')
            allcontentlist.append(elem.text)
            browser.close()
            browser.switch_to.window(browser.window_handles[0])
        showall = 0

    # if clicked "more" button, scrap all titles and contents and save them to alltitlelist & allcontentlist. Then, go back to main bulletin menu.
    for j in range(subcontamut):
        if showall == 0:
            break
        if j == 50:
            browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:htmlDetailTbl2:deluxe1__pagerNext"]').click()
            browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:htmlDetailTbl2:deluxe1__pagerNext"]').click()
        elem = browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlDetailTbl2:' + str(j) + ':htmlTitleCol3"]')
        alltitlelist.append(elem.text)

        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlDetailTbl2:' + str(j) + ':htmlTitleCol3"]').click()
        time.sleep(1)
        browser.switch_to.window(browser.window_handles[1])
        elem = browser.find_element_by_xpath('//*[@id="form1:htmlMain"]')
        allcontentlist.append(elem.text)
        browser.close()
        browser.switch_to.window(browser.window_handles[0])

    if showall == 1 and bulletinnum == 2:
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()
    elif showall == 1 and bulletinnum == 4:
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()
    elif showall == 1 and bulletinnum == 6:
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()
    elif showall == 1 and bulletinnum == 7:
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()
    elif showall == 1:
        browser.find_element_by_xpath('//*[@id="form1:Poa00201A:htmlParentTable:0:htmlHeaderTbl:0:retrurn"]').click()


alltitlelist = []
allcontentlist = []
# ==========================================================================
print("input 'Y' to renew scrapped TUS online bulletin board.\ninput 'N' to use previously scrapped one.\ninput 'S' to skip TUS online bulletin board.\n")
onlineinput = input()

if onlineinput == 'Y':
    print("please input your id\n")
    tusid = input()
    print("please input your password\n")
    tusps = input()

    # accessing login page and inputing id & pw
    browser = webdriver.Chrome("/usr/local/bin/chromedriver")
    browser.get("https://class.admin.tus.ac.jp/up/faces/login/Com00501A.jsp")
    browser.find_element_by_id("form1:htmlUserId").send_keys(tusid)  # id
    browser.find_element_by_id("form1:htmlPassword").send_keys(tusps)  # ps
    browser.find_element_by_id("form1:login").click()

    # executing scraping
    for i in range(12):  # number of bulletin is 0~11
        getonebulletin(i, alltitlelist, allcontentlist)

    with open('previous_online_title.pkl', 'wb') as f:
        pickle.dump(alltitlelist, f)
    with open('previous_online_content.pkl', 'wb') as f:
        pickle.dump(allcontentlist, f)

if onlineinput == 'N':
    with open('previous_online_title.pkl', 'rb') as f:
        previous_online_title = pickle.load(f)
    with open('previous_online_content.pkl', 'rb') as f:
        previous_online_content = pickle.load(f)

    alltitlelist = alltitlelist + previous_online_title
    allcontentlist = allcontentlist + previous_online_content

# ==========================================================================
print("input 'Y' to renew saved TUS offline bulletin board.\ninput 'N' to use previously saved one.\ninput 'S' to skip TUS offline bulletin board.\n")
offlineinput = input()

if offlineinput == 'Y':
    print("please input title of TUS offline bulletin board. if there is no more title to add, input 'N'.\ntitle (input 'N' to exit):\n")
    titleinput = input()

    previous_offline_title = []
    previous_offline_content = []
    while titleinput != 'N':
        alltitlelist.append(titleinput)
        previous_offline_title.append(titleinput)
        print("content:\n")
        contentinput = input()
        allcontentlist.append(contentinput)
        previous_offline_content.append(contentinput)

        print("title (input 'N' to exit):\n")
        titleinput = input()

    with open('previous_offline_title.pkl', 'wb') as f:
        pickle.dump(previous_offline_title, f)
    with open('previous_offline_content.pkl', 'wb') as f:
        pickle.dump(previous_offline_content, f)

if offlineinput == 'N':
    with open('previous_offline_title.pkl', 'rb') as f:
        previous_offline_title = pickle.load(f)
    with open('previous_offline_content.pkl', 'rb') as f:
        previous_offline_content = pickle.load(f)

    alltitlelist = alltitlelist + previous_offline_title
    allcontentlist = allcontentlist + previous_offline_content

    print("please input title of TUS offline bulletin board. if there is no more title to add, input 'N'.\ntitle (input 'N' to exit):\n")
    titleinput = input()

    while titleinput != 'N':
        alltitlelist.append(titleinput)
        previous_offline_title.append(titleinput)
        print("content:\n")
        contentinput = input()
        allcontentlist.append(contentinput)
        previous_offline_content.append(contentinput)

        print("title (input 'N' to exit):\n")
        titleinput = input()

    with open('previous_offline_title.pkl', 'wb') as f:
        pickle.dump(previous_offline_title, f)
    with open('previous_offline_content.pkl', 'wb') as f:
        pickle.dump(previous_offline_content, f)


# making title-content dict
titlecontentdic = {}
for l in range(len(alltitlelist)):
    titlecontentdic[l] = [alltitlelist[l], allcontentlist[l]]

# constituting vacab =========================================================
# copying allcontentlist to make vocabulary
allcontentlist_copied = allcontentlist.copy()
allcontentlist_kanjidic = []

# allcontentlist_copied = [content1, content2, ...]
# changing all hiragana to "|" for all contents
for k in range(len(allcontentlist_copied)):
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("あ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("い", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("う", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("え", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("お", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("か", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("き", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("く", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("け", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("こ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("さ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("し", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("す", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("せ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("そ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("た", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ち", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("つ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("て", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("と", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("な", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("に", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぬ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ね", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("の", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("は", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ひ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ふ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("へ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ほ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ま", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("み", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("む", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("め", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("も", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("や", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ゆ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("よ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ら", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("り", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("る", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("れ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ろ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("わ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("を", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ん", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("が", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぎ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぐ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("げ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ご", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ざ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("じ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ず", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぜ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぞ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("だ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("づ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("で", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ど", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ば", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("び", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぶ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("べ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぼ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぱ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぴ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぷ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぺ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ぽ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ゃ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ゅ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("ょ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("っ", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("・", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("。", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("、", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace(",", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("\n", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("※", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("！", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("？", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("「", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("」", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("【", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("】", "|")
    allcontentlist_copied[k] = allcontentlist_copied[k].replace("＝", "|")

# allcontentlist_copied = [co|ten|1, c|||nt2, ...]
k = 0
for q in range(len(allcontentlist)):
    allcontentlist_temp = allcontentlist_copied[q].split("|") # for each content, cut the content with '|'. the result will be: allcontentlist_temp = ['漢字', ' ', ' ', '漢字', '漢字', ...]
    allcontentlist_temp = ' '.join(allcontentlist_temp).split() # to erase ' ', join it and split it. the result will be: allcontentlist_temp = ['漢字', '漢字', '漢字', ...]

    for s in range(len(allcontentlist_temp)):
        if len(allcontentlist_temp[s]) == 1:
            allcontentlist_temp[s] = allcontentlist_temp[s].replace(allcontentlist_temp[s], "") # if the 漢字 is just one word(like 漢 or 字), erase it.
    allcontentlist_temp = ' '.join(allcontentlist_temp).split()

    # append one dict{} to allcontentlist_kanjidic[], count same 漢字 and erase it while there is no 漢字 in allcontentlist_temp
    allcontentlist_kanjidic.append({})
    # appending key:value to allcontentlist_kanjidic from the first element of allcontentlist_temp
    while len(allcontentlist_temp) != 0:
        allcontentlist_kanjidic[k][allcontentlist_temp[0]] = 0
        # adding 1, if there are same letters
        for i in allcontentlist_temp:
            if i == allcontentlist_temp[0]:
                allcontentlist_kanjidic[k][allcontentlist_temp[0]] += 1
        # erasing all same letters.
        z = allcontentlist_temp[0]
        while z in allcontentlist_temp:
            allcontentlist_temp.remove(allcontentlist_temp[0])
    k += 1

# allcontentlist_kanjidic = [{'漢字':3,'漢字':1},{'漢字':1, '漢字':2},...]
letterlist_temp = []
for i in range(len(allcontentlist_kanjidic)):
    a = list(allcontentlist_kanjidic[i].keys()) # copying all keys of allcontentlist_kanjidic
    letterlist_temp = letterlist_temp + a

words = list(dict.fromkeys(letterlist_temp)) # erase same letters between dicts.

# giving words id. word_to_id={漢字0:0, 漢字1:1, ...}
# giving id words. id_to_word={0:漢字0, 1:漢字1, ...}
print("input 'Y' to renew vocab with collected data.\ninput 'N' to use previous vocab.\n")
vocabinput = input()

if vocabinput == 'Y':
    word_to_id = {"[PAD]": 0, "[UNK]": 1}
    for w in words:
        word_to_id[w] = len(word_to_id)

    id_to_word = {i: w for w, i in word_to_id.items()}

if vocabinput != 'Y':
    with open('word_to_id.pkl', 'rb') as f:
        word_to_id = pickle.load(f)
    with open('id_to_word.pkl', 'rb') as f:
        id_to_word = pickle.load(f)

# ============================================================================

# make raw data ==============================================================
# raw data for learning -> 0 ~ ALLDATA / 2
# raw data for verifying -> ALLDATA / 2 ~ ALLDATA
forpred_all_raw_inputs = []
raw_inputs = []
raw_inputs_valid = []

for i in range(len(allcontentlist_kanjidic)):
    a = ' '.join(list(allcontentlist_kanjidic[i].keys()))
    forpred_all_raw_inputs.append(a)

for i in range(len(allcontentlist_kanjidic)//2):
    a = ' '.join(list(allcontentlist_kanjidic[i].keys()))
    raw_inputs.append(a)

for i in range(len(allcontentlist_kanjidic)//2, len(allcontentlist_kanjidic)):
    a = ' '.join(list(allcontentlist_kanjidic[i].keys()))
    raw_inputs_valid.append(a)

raw_labels = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
              1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
              1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
              0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
              1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
              0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
              1, 0, 0]

raw_labels_valid = [0, 0, 0, 0, 0, 0, 0, 1,
                    1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                    1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                    0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 0, 0]
# ============================================================================

with open('alltitlelist.pkl', 'wb') as f:
    pickle.dump(alltitlelist, f)

with open('allcontentlist.pkl', 'wb') as f:
    pickle.dump(allcontentlist, f)

with open('word_to_id.pkl', 'wb') as f:
    pickle.dump(word_to_id, f)

with open('id_to_word.pkl', 'wb') as f:
    pickle.dump(id_to_word, f)


with open('forpred_all_raw_inputs.pkl', 'wb') as f:
    pickle.dump(forpred_all_raw_inputs, f)

with open('raw_inputs.pkl', 'wb') as f:
    pickle.dump(raw_inputs, f)

with open('raw_inputs_valid.pkl', 'wb') as f:
    pickle.dump(raw_inputs_valid, f)

with open('raw_labels.pkl', 'wb') as f:
    pickle.dump(raw_labels, f)

with open('raw_labels_valid.pkl', 'wb') as f:
    pickle.dump(raw_labels_valid, f)

print("done\n")
