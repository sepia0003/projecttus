# -*- coding: utf-8 -*-
from selenium import webdriver
import argparse
import random
import numpy as np
import torch
import torch.utils.data
import time
import pickle


args = {
    "seed": 1234,
    "n_epoch": 200,
    "n_batch": 2,
    "lr": 0.001,
    "save_path": "importance_result.pth",
    "device": torch.device("cude" if torch.cuda.is_available() else "cpu")
}
args = argparse.Namespace(**args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class ArrangedData(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        assert len(self.inputs) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, index):
        return(
            torch.tensor(self.inputs[index]),
            torch.tensor(self.labels[index]),
        )

    def collate_fn(self, batch):
        inputs, labels = list(zip(*batch))

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        batch = [
            inputs,
            labels,
        ]

        return batch


# implementing class for importance prediction
class PredictImportance(torch.nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.embed = torch.nn.Embedding(n_vocab, 4)
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, inputs):
        hidden = self.embed(inputs)
        hidden, _ = torch.max(hidden, dim=1)
        logits = self.linear(hidden)
        return logits


# function for estimating accuracy
def cal_accuracy(logits, labels):
    _, indices = logits.max(-1)
    mat = torch.eq(indices, labels).cpu().numpy()
    total = np.ones_like(mat)
    result = np.sum(mat) / max(1, np.sum(total))
    return result


# function for proceeding learning
def learn_per_ep(args, model, loader, loss_fn, optimizer):
    model.train()
    losses, access = [], []
    for batch in loader:
        optimizer.zero_grad()
        inputs, labels = map(lambda v: v.to(args.device), batch)
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        losses.append(loss_val)
        acc_val = cal_accuracy(logits, labels)
        access.append(acc_val)

    return np.mean(losses), np.mean(access)


# function for evaluating
def estimate_ep(args, model, loader, loss_fn):
    model.eval()
    losses, access = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = map(lambda v: v.to(args.device), batch)
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss_val = loss.item()
            losses.append(loss_val)
            acc_val = cal_accuracy(logits, labels)
            access.append(acc_val)

    return np.mean(losses), np.mean(access)


def execute_prediction(word_to_id, model, string):
    token = [word_to_id[w] for w in string.strip().split()]

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([token]).to(args.device)
        logits = model(inputs)
        _, indices = logits.max(-1)
        y_pred = indices[0].numpy()
    result = "중요" if y_pred == 1 else "기타"
    return result


# ======= implementing vocabulary ========
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

alltitlelist = []
allcontentlist = []
titlecontentdic = {}


# function which scraps all titles & contents
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


# executing scraping
for i in range(12):  # number of bulletin is 0~11
    getonebulletin(i, alltitlelist, allcontentlist)

# making dict
for l in range(len(alltitlelist)):
    titlecontentdic[l] = [alltitlelist[l], allcontentlist[l]]

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

# allcontentlist_copied = [co|ten|1, c|t|nt2, ...]
# cutting contents basing on "|" then make the result to list named allcontentlist_temp
# allcontentlist_temp = ['漢字', '漢字', '漢字', ...]
k = 0
for q in range(len(allcontentlist)):
    allcontentlist_temp = allcontentlist_copied[q].split("|")
    allcontentlist_temp = ' '.join(allcontentlist_temp).split()

    for s in range(len(allcontentlist_temp)):
        if len(allcontentlist_temp[s]) == 1:
            allcontentlist_temp[s] = allcontentlist_temp[s].replace(allcontentlist_temp[s], "")
    allcontentlist_temp = ' '.join(allcontentlist_temp).split()

    # making allcontentlist_kanjidic = [{},{},...]
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
# copying all keys of allcontentlist_kanjidic, saving them to letterlist_temp, erase same letters between dicts.
letterlist_temp = []
for i in range(len(allcontentlist_kanjidic)):
    a = list(allcontentlist_kanjidic[i].keys())
    letterlist_temp = letterlist_temp + a

words = list(dict.fromkeys(letterlist_temp))

# giving words id. word_to_id={漢字0:0, 漢字1:1, ...}
# giving id words. id_to_word={0:漢字0, 1:漢字1, ...}
word_to_id = {"[PAD]": 0, "[UNK]": 1}
for w in words:
    word_to_id[w] = len(word_to_id)

id_to_word = {i: w for w, i in word_to_id.items()}
# ======= end of implementing vocabulary ========

# making lists for machine learning which is named raw_inputs and raw_labels
# for raw_inputs, get all keys from half of allcontentlist_kanjidic and make them into "漢字 漢字 漢字"
# and append it to raw_inputs
raw_inputs = []
forpred_all_raw_inputs = []
for i in range(len(allcontentlist_kanjidic)//2):
    a = ' '.join(list(allcontentlist_kanjidic[i].keys()))
    raw_inputs.append(a)

for i in range(len(allcontentlist_kanjidic)):
    a = ' '.join(list(allcontentlist_kanjidic[i].keys()))
    forpred_all_raw_inputs.append(a)

# for raw_labels, give 1 for important contents and 2 for normal contents
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

inputs = []
for s in raw_inputs:
    inputs.append([word_to_id[w] for w in s.split()])

labels = raw_labels


# data for learning
dataset = ArrangedData(inputs, labels)
sampler = torch.utils.data.RandomSampler(dataset)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_batch, sampler=sampler, collate_fn=dataset.collate_fn)

# data for verifying
dataset = ArrangedData(inputs, labels)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_batch, sampler=None, collate_fn=dataset.collate_fn)

# data for testing
dataset = ArrangedData(inputs, labels)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_batch, sampler=None, collate_fn=dataset.collate_fn)


# making model for learning
model = PredictImportance(len(word_to_id))
model.to(args.device)

# setting loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# recording process in history, saving the best accuracy in max_acc
records = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
max_acc = 0


# starting learning
for e in range(args.n_epoch):
    train_loss, train_acc = learn_per_ep(args, model, train_loader, loss_fn, optimizer)
    valid_loss, valid_acc = estimate_ep(args, model, valid_loader, loss_fn)

    records["train_loss"].append(train_loss)
    records["train_acc"].append(train_acc)
    records["valid_loss"].append(valid_loss)
    records["valid_acc"].append(valid_acc)

    if max_acc < valid_acc:
        max_acc = valid_acc
        torch.save(
            {"state_dict": model.state_dict(), "valid_acc": valid_acc},
            args.save_path,
        )


# make model for release
model = PredictImportance(len(word_to_id))
model.to(args.device)

save_dict = torch.load(args.save_path)
model.load_state_dict(save_dict['state_dict'])


# saving final results to pickle for create.py
final_result = []
for i in forpred_all_raw_inputs:
    final_result.append(execute_prediction(word_to_id, model, i))

with open('final_result.pkl', 'wb') as f:
    pickle.dump(final_result, f)

with open('allcontentlist.pkl', 'wb') as f:
    pickle.dump(allcontentlist, f)

with open('alltitlelist.pkl', 'wb') as f:
    pickle.dump(alltitlelist, f)


print("done\n")
