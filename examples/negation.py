
### create "not" with something (actually create all of them
notFeatures = []
for i in features:
    notFeatures.append("not " + i)

notFeatures # filter only those which are

negationCount = [0]*len(twtTxtliteral)
negationDict = dict(zip(notFeatures,negationCount))
for f in range(len(notFeatures)):
    nfeaturesCount = []
    for i in twtTxtliteral:
        try:
            nfeaturesCount.append(sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(notFeatures[f]), i)))
        except TypeError:
            nfeaturesCount.append(0)
        negationDict[notFeatures[f]] = nfeaturesCount


twtEngDf = pd.concat([twtDf,pd.DataFrame(negationDict)], axis = 1)

twtEngDf # new training set

devTwt = []
with open("./2017S1-KTproj2-data/dev-tweets.txt") as f:
    for line in f:
        devTwt.append(line.split("\t")[1].rstrip())

devTwtEngDf = pd.concat([devForPred,pd.DataFrame(devNegationDict)], axis = 1)
devTwtEngDf.head()
