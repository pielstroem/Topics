from collections import Counter

#path = "mycorpusfile.txt"

def makeCounter(path):
    with open(path, 'r', encoding='utf-8') as f:
        count = Counter(f.read().split())
    return count

#mycounter = makeCounter(path)

def makeRemoveLists(mycounter):

    countersum = 0.01 * sum(mycounter.values())
    #
    #NOTES
    #
    #generator statt list oder set comprehension
    #Counter als Series
    #Zweiteilen
    hapax = set([hapx for hapx, value in mycounter.items() if value == 1])

    stopwords = set([stopw for stopw, value in mycounter.items() if value > countersum])

    #print(len(hapax))
    #print(len(stopwords))
    return hapax, stopwords

def removestuff(inpath, outpath):
    
    mycounter = makeCounter(inpath)
    
    hapax, stopwords = makeRemoveLists(mycounter)

    with open(inpath, 'r', encoding="utf-8") as tmp:
        last = len(list(tmp)) -2 
    with open(inpath, 'r', encoding='utf-8') as g:
        with open(outpath, 'w', encoding='utf-8') as f:
            for i, line in enumerate(g):
                if i != last:
                    print("working on ...  ", i)
                    f.write(' '.join([word for word in line.split() if word not in hapax or stopwords]) + "\n")
                
                else:
                    print("working on ...  ", i)
                    f.write(' '.join([word for word in line.split() if word not in hapax or stopwords]))
                    print("\nFinished\n")
                    break

#removestuff(path, "mycorpusremoved.txt", hapax, stopwords)