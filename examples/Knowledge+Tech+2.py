
# coding: utf-8

# In[2]:


# from sklearn import datasets
# iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn import metrics, tree, svm
from sklearn.feature_selection import chi2, SelectKBest
import arff
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# In[3]:

import numpy as np
import pandas as pd
import os, re
from ast import literal_eval
os.getcwd()


# In[4]:

###### Extracting data ###### For feature engineering
trainTwt = []
with open("./2017S1-KTproj2-data/train-tweets.txt") as f:
    for line in f:
        trainTwt.append(line)


# In[5]:

trainLabel = []
with open("./2017S1-KTproj2-data/train-labels.txt") as f:
    for line in f:
        trainLabel.append(line)


# In[6]:

twtId = []
twtTxt = []
twtLabel = []
for i in range(len(trainTwt)):
    twtId.append(trainTwt[i].split("\t")[0])
    twtTxt.append(trainTwt[i].split("\t")[1].rstrip())
    twtLabel.append(trainLabel[i].split("\t")[1][0:-1])


# In[7]:

# this cannot replicate the data provided by the course (might be trivial differences in string format)
# sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("a"), "Mary has a lamb"))


# In[ ]:




# In[8]:

# using pandas, read directly from csv which is converted from ARFF


# In[9]:

twtDf = pd.read_csv("/home/tchanchokpong/Dropbox/MIT/KnowledgeTechnology/Proj2/2017S1-KTproj2-data/train.csv")


# In[10]:

sentiment = twtDf['sentiment']
twtDf.drop('sentiment', axis = 1, inplace = True)
twtDf.drop('id', axis = 1, inplace = True)
twtDf.head()


# In[11]:

sentiment.value_counts()


# In[12]:

devel = pd.read_csv("./2017S1-KTproj2-data/dev.csv")


# In[13]:

clf = GaussianNB() # assume normal distribution
clf.fit(twtDf,sentiment)


# In[14]:

devForPred = devel[devel.columns[0:-1]]
devSent = devel[devel.columns[-1]]


# In[15]:

predNaiveDevel = clf.predict(devForPred)


# In[16]:

NaiveCM = confusion_matrix(devSent, predNaiveDevel, labels = ["positive","negative","neutral"])
NaiveCM


# In[17]:

pd.Series(predNaiveDevel).value_counts()


# In[18]:

print(classification_report(devSent,predNaiveDevel, labels = ["positive","negative","neutral"]))


# In[19]:

accuracy_score(devSent,predNaiveDevel)


# In[20]:

# Multinomial Bayes
clfMulti = MultinomialNB()
clfMulti.fit(twtDf, sentiment)


# In[21]:

predMultiNB = clfMulti.predict(devForPred)


# In[22]:

MultiNBCM = confusion_matrix(devSent,predMultiNB, labels = ["positive","negative","neutral"])
MultiNBCM


# In[23]:

pd.Series(predMultiNB).value_counts()


# In[24]:

pd.Series(devSent).value_counts()


# In[25]:

print(classification_report(devSent,predMultiNB, labels = ["positive","negative","neutral"]))


# In[26]:

accuracy_score(devSent,predMultiNB)
# we see marked improvedment (why?)


# In[27]:

0.58181079983759643 - 0.5544051969143321 # 2.74% improvements


# In[28]:

### Decision trees 


# In[29]:

treeClf = tree.DecisionTreeClassifier()
treeClf.fit(twtDf, sentiment) # this could overfit the data 


# In[30]:

# check for overfitting by predicting on itself and see how it does with development
predTrainTree = treeClf.predict(twtDf) 


# In[31]:

print(classification_report(sentiment,predTrainTree,labels = ["positive","negative","neutral"]))


# In[32]:

predDevTree = treeClf.predict(devForPred)


# In[33]:

confusion_matrix(devSent, predDevTree,labels = ["positive","negative","neutral"])


# In[34]:

print(classification_report(devSent,predDevTree,labels = ["positive","negative","neutral"]))


# In[35]:

## possible overfitting problems / 
accuracy_score(devSent,predDevTree)


# In[36]:

# Retrain with maximum depth
treeClfM = tree.DecisionTreeClassifier(max_depth = 32) # from 46 features, set maximum at around 30
treeClfM.fit(twtDf,sentiment)


# In[37]:

predDevTreeM = treeClfM.predict(devForPred)


# In[38]:

confusion_matrix(devSent, predDevTreeM,labels = ["positive","negative","neutral"])


# In[39]:

print(classification_report(devSent,predDevTreeM,labels = ["positive","negative","neutral"]))


# In[40]:

accuracy_score(devSent, predDevTreeM) # higher, avoiding overfit, outperformed by NaiveBayes Multinomial


# In[41]:

RFclf = RandomForestClassifier(max_depth = 32)
RFclf.fit(twtDf, sentiment)


# In[42]:

predRF = RFclf.predict(devForPred)


# In[43]:

print(classification_report(devSent,predRF, labels = ["positive","negative","neutral"])) # better recall 


# In[44]:

confusion_matrix(devSent, predRF,labels = ["positive","negative","neutral"])


# In[45]:

accuracy_score(devSent,predRF) # slight improvement from lower biases


# In[46]:

# Export Decision Trees


# In[47]:

with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(treeClfM, out_file=f)


# In[48]:

############ SVM ###############


# In[49]:

svmClf = svm.LinearSVC()
svmClf.fit(twtDf,sentiment)


# In[50]:

predLSVC = svmClf.predict(devForPred)


# In[51]:

confusion_matrix(devSent, predLSVC,labels = ["positive","negative","neutral"])


# In[52]:

print(classification_report(devSent,predLSVC,labels = ["positive","negative","neutral"]))


# In[53]:

confusion_matrix(devSent,predLSVC, labels = ["positive","negative","neutral"])


# In[54]:

accuracy_score(devSent,predLSVC)


# In[55]:

# svmClfRBF = svm.SVC()
# svmClfRBF.fit(twtDf,sentiment)


# In[56]:

# predSVCRBF = svmClfRBF.predict(devForPred)


# In[57]:

# print(classification_report(devSent,predLSVC,labels = ["positive","negative","neutral"]))


# In[58]:

# accuracy_score(devSent,predSVCRBF) # better off with linear (perhaps)


# In[59]:

######################################### Some sanity Checks ##################################################


# In[60]:

twtTxt[2223]


# In[61]:

literal_eval(twtTxt[2223]) # there aren't many and should not really affect the tagging


# In[ ]:




# In[62]:

twtTxtliteral = []
error = []
for i in range(len(twtTxt)):
    try:
        twtTxtliteral.append(literal_eval(twtTxt[i]))
    except (EOFError,SyntaxError,ValueError) :
        error.append(i)
#        print(i)
        twtTxtliteral.append(twtTxt[i])


# In[63]:

len(twtTxtliteral)


# In[64]:

# features


# In[65]:

######################################### Features Engineering ##################################################


# In[66]:

# Part of speech tagging (POS tagging) using tree tagger
import treetaggerwrapper


# In[67]:

# test
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')


# In[68]:

treetaggerwrapper.make_tags(tagger.tag_text(twtTxt[0]))


# In[69]:

twtTxt[14355]


# In[70]:

### use subjective and objective tags to help the prediction


# In[71]:

subjective = []
for i in range(len(twtTxt)):
    count = 0
    for tag in treetaggerwrapper.make_tags(tagger.tag_text(twtTxt[i])):
        try:
            if tag[1] == "PP" or tag[1] == "PP$" or tag[1] == "UH" or             tag[1] == "RB" or tag[1] == "RBR" or tag[1] == "RBS" or             tag[1] == "MD" or tag[1] == "VB":
                count += 1
        except IndexError:
            continue
    subjective.append(count)


# In[72]:

len(subjective)


# In[73]:

objective = []
for i in range(len(twtTxt)):
    count = 0
    for tag in treetaggerwrapper.make_tags(tagger.tag_text(twtTxt[i])):
        try:
            if tag[1] == "NP" or tag[1] == "NPS" or tag[1] == "NNS" or             tag[1] == "WP$" or tag[1] == "JJR":
                count += 1
        except IndexError:
            continue
    objective.append(count)


# In[74]:

len(objective)


# In[75]:

f1Df = pd.concat([twtDf, pd.DataFrame({"subjective":subjective, "objective":objective})], axis = 1)


# In[76]:

f1Df  # FOR Training with new features


# In[77]:

devTwt = []
with open("./2017S1-KTproj2-data/dev-tweets.txt") as f:
    for line in f:
        devTwt.append(line.split("\t")[1])


# In[78]:

subjectiveDev = []
for i in range(len(devTwt)):
    count = 0
    for tag in treetaggerwrapper.make_tags(tagger.tag_text(devTwt[i])):
        try:
            if tag[1] == "PP" or tag[1] == "PP$" or tag[1] == "UH" or             tag[1] == "RB" or tag[1] == "RBR" or tag[1] == "RBS" or             tag[1] == "MD" or tag[1] == "VB":
                count += 1
        except IndexError:
            continue
    subjectiveDev.append(count)


# In[79]:

sum(subjectiveDev)


# In[80]:

objectiveDev = []
for i in range(len(devTwt)):
    count = 0
    for tag in treetaggerwrapper.make_tags(tagger.tag_text(devTwt[i])):
        try:
            if tag[1] == "NP" or tag[1] == "NPS" or tag[1] == "NNS" or             tag[1] == "WP$" or tag[1] == "JJR":
                count += 1
        except IndexError:
            continue
    objectiveDev.append(count)


# In[81]:

sum(objectiveDev) # same type of distribution


# In[82]:

devEngDf = pd.concat([devForPred, pd.DataFrame({"subjective":subjectiveDev, "objective":objectiveDev})], axis = 1)


# In[83]:

devEngDf.head()


# In[84]:

clfEng = GaussianNB() # assume normal distribution
clfEng.fit(f1Df,sentiment)


# In[85]:

predNaiveDevelEng = clfEng.predict(devEngDf) 
NaiveEngCM = confusion_matrix(devSent, predNaiveDevelEng, labels = ["positive","negative","neutral"])
NaiveEngCM


# In[86]:

print(classification_report(devSent,predNaiveDevelEng,labels = ["positive","negative","neutral"]))


# In[87]:

accuracy_score(devSent,predNaiveDevelEng) # 2% improvements


# In[88]:

clfEngMult = MultinomialNB()
clfEngMult.fit(f1Df, sentiment)


# In[89]:

predNBEngDevelMult = clfEngMult.predict(devEngDf)
NBDevelMultCM = confusion_matrix(devSent,predNBEngDevelMult,labels = ["positive","negative","neutral"])
NBDevelMultCM


# In[90]:

print(classification_report(devSent,predNBEngDevelMult,labels = ["positive","negative","neutral"])) # slightly worse performance


# In[91]:

accuracy_score(devSent,predNBEngDevelMult)


# In[92]:

svmClfEng = svm.LinearSVC()
predSvmEng = svmClfEng.fit(f1Df,sentiment).predict(devEngDf)


# In[93]:

svmEngCM = confusion_matrix(devSent,predSvmEng,labels = ["positive","negative","neutral"])
svmEngCM


# In[94]:

print(classification_report(devSent,predSvmEng,labels = ["positive","negative","neutral"])) # slightly worse performance or no effect


# In[95]:

accuracy_score(devSent,predSvmEng)


# In[96]:

######## use likelihood in tag and type of sentiment ######


# In[97]:

negative = []
for i in range(len(twtTxt)):
    count = 0
    for tag in treetaggerwrapper.make_tags(tagger.tag_text(twtTxt[i])):
        try:
            if tag[1] == "WP$" or tag[1] == "POS" or tag[1] == "RBS" or tag[1] == "JJS":
                count += 1
        except IndexError:
            continue
    negative.append(count)


# In[98]:

positive = []
for i in range(len(twtTxt)):
    count = 0
    for tag in treetaggerwrapper.make_tags(tagger.tag_text(twtTxt[i])):
        try:
            if tag[1] == "RB" or tag[1] == "WRB" or tag[1] == "VBD" or tag[1] == "VBN":
                count += 1
        except IndexError:
            continue
    positive.append(count)


# In[99]:

sum(negative)


# In[100]:

sum(positive)


# In[101]:

negativeDev = []
for i in range(len(devTwt)):
    count = 0
    for tag in treetaggerwrapper.make_tags(tagger.tag_text(devTwt[i])):
        try:
            if tag[1] == "WP$" or tag[1] == "POS" or tag[1] == "RBS" or tag[1] == "JJS":
                count += 1
        except IndexError:
            continue
    negativeDev.append(count)


# In[102]:

sum(negativeDev)


# In[103]:

positiveDev = []
for i in range(len(devTwt)):
    count = 0
    for tag in treetaggerwrapper.make_tags(tagger.tag_text(devTwt[i])):
        try:
            if tag[1] == "RB" or tag[1] == "WRB" or tag[1] == "VBD" or tag[1] == "VBN":
                count += 1
        except IndexError:
            continue
    positiveDev.append(count)


# In[104]:

sum(positiveDev)


# In[105]:

f2Df = featureEngDf = pd.concat([twtDf, pd.DataFrame({"positivity":positive, "negativity":negative})], axis = 1)
f2Df


# In[106]:

devEngDf2 = pd.concat([devForPred, pd.DataFrame({"positivity":positiveDev, "negativity":negativeDev})], axis = 1)
devEngDf2


# In[107]:

clfNBEng2 =  GaussianNB() # test 
predNBEng2 = clfNBEng2.fit(f2Df,sentiment).predict(devEngDf2)


# In[108]:

print(classification_report(devSent,predNBEng2,labels = ["positive","negative","neutral"]))


# In[109]:

accuracy_score(devSent,predNBEng2) # very slight improvements


# In[110]:

clfNBMultEng2 =  MultinomialNB() # test Multinomial
predNBMultEng2 = clfNBMultEng2.fit(f2Df,sentiment).predict(devEngDf2)


# In[111]:

print(classification_report(devSent,predNBMultEng2,labels = ["positive","negative","neutral"]))


# In[112]:

accuracy_score(devSent,predNBMultEng2)


# In[113]:

svmClfEng2 = svm.LinearSVC()
predSvmEng2 = svmClfEng2.fit(f2Df,sentiment).predict(devEngDf2)


# In[114]:

print(classification_report(devSent,predSvmEng2,labels = ["positive","negative","neutral"]))


# In[115]:

accuracy_score(devSent,predSvmEng2)


# In[116]:

f3Df = featureEngDf = pd.concat([twtDf, pd.DataFrame({
    "positivity":positive, "negativity":negative, "subjective":subjective, "objective":objective})], axis = 1)
f3Df


# In[117]:

devEngDf3 = pd.concat([devForPred, pd.DataFrame({
    "positivity":positiveDev, "negativity":negativeDev,
    "subjective":subjectiveDev, "objective":objectiveDev})], axis = 1)
devEngDf3


# In[118]:

clfNBEng3 =  GaussianNB() # test of Multinomial
predNBEng3 = clfNBEng3.fit(f3Df,sentiment).predict(devEngDf3)


# In[119]:

print(classification_report(devSent,predNBEng3,labels = ["positive","negative","neutral"])) # same type of improvements


# In[120]:

accuracy_score(devSent,predNBEng3)


# In[121]:

clfNBMultEng3 =  MultinomialNB() # test of Multinomial
predNBMultEng3 = clfNBMultEng3.fit(f3Df,sentiment).predict(devEngDf3)


# In[122]:

print(classification_report(devSent,predNBMultEng3,labels = ["positive","negative","neutral"]))


# In[123]:

accuracy_score(devSent,predNBMultEng3)


# In[124]:

svmClfEng3 = svm.LinearSVC()
predSvmEng3 = svmClfEng3.fit(f3Df,sentiment).predict(devEngDf3)


# In[125]:

print(classification_report(devSent,predSvmEng3,labels = ["positive","negative","neutral"])) # literally no effect no SVM


# In[126]:

accuracy_score(devSent,predSvmEng3) 


# In[ ]:




# In[127]:

################## C parameter tuning (i mean why not..) ###############


# In[128]:

svmClftune = svm.LinearSVC(C = 5.0)
svmTune = svmClftune.fit(twtDf,sentiment).predict(devForPred)


# In[129]:

print(classification_report(devSent,svmTune,labels = ["positive","negative","neutral"])) # essentially no effect


# In[130]:

accuracy_score(devSent,svmTune)


# In[131]:

print(classification_report(devSent,predLSVC,labels = ["positive","negative","neutral"])) # from above


# In[132]:

accuracy_score(devSent,predLSVC)


# In[133]:

devSent.value_counts()


# In[134]:

2400/len(devSent)


# In[135]:

weightSvm = svm.LinearSVC(class_weight = {"positive":0.25,"negative":0.25,"neutral":0.5})
predSvmEng4 = weightSvm.fit(f3Df,sentiment).predict(devEngDf3)


# In[136]:

print(classification_report(devSent,predSvmEng4,labels = ["positive","negative","neutral"])) 


# In[137]:

accuracy_score(devSent,predSvmEng4)


# In[138]:

predSvmWeight = weightSvm.fit(twtDf,sentiment).predict(devForPred)


# In[139]:

print(classification_report(devSent,predSvmWeight,labels = ["positive","negative","neutral"])) # from above


# In[140]:

accuracy_score(devSent,predSvmWeight)


# In[141]:

weightSVM = confusion_matrix(devSent,predSvmWeight,labels = ["positive","negative","neutral"])
weightSVM


# In[142]:

balanceSvm = svm.LinearSVC(class_weight = "balanced")
predbalance = balanceSvm.fit(twtDf,sentiment).predict(devForPred)


# In[143]:

balanceCM = confusion_matrix(devSent,predbalance,labels = ["positive","negative","neutral"])
balanceCM


# In[144]:

print(classification_report(devSent,predbalance,labels = ["positive","negative","neutral"])) # from above


# In[145]:

testDf = pd.read_csv("./2017S1-KTproj2-data/test.csv")
testDf.drop('id', axis =1 ,inplace =True)
testDf.drop('sentiment', axis = 1 , inplace = True)
testDf


# In[146]:

predTest = weightSvm.predict(testDf)


# In[147]:

#with open('predTest.txt', 'w') as f:
#    read_data = f.write(str(list(predTest)))


# In[148]:

# tweets for examples


# In[154]:

twtTxt[7918]


# In[156]:

sentiment[7918]


# In[164]:

twtTxt[9090]


# In[165]:

list(twtDf.columns)

