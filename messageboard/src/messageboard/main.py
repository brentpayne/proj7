'''
Created on May 24, 2009

@author: Oso
'''
import nltk.data
import nltk
import re

def main():
    text = ""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    
    file = open('S:\\Coding\\datamining\\proj7\\data\\messageboard','r')
    
    textsets=[]
    authors=[]
    text = ''
    d=184
    skip=False
    for line in file.readlines():
        line = line.lower()
        if(re.search('\s'+str(d)+'$',line)):
            #print line,
            d = d-1
            #print text
            textsets.append(text)
            authors.append(line[0:7])
            text=''
            skip=True
        elif(skip):
            pass
            skip=False
        else:
            #print line,
            text+=line
            #print "***",text,
    #return
    
    vocabulary=[]
    vocab_check={}
    count_list=[]
    last_token = ""
    for text in textsets:
        #print text
        tokens = nltk.wordpunct_tokenize(text.strip())
        text_counts = {}
        for token in tokens:
            bigram = last_token+"|"+token
            last_token = token
            if text_counts.has_key(bigram):
                text_counts[bigram] +=1
            else:
                text_counts[bigram] = 1
                if not vocab_check.has_key(bigram):
                    vocabulary.append(bigram)
                    vocab_check[bigram]=True
        count_list.append(text_counts)
    file.close()
    
    
    ofile = open("cvs4matlab3",'w')
    matrix=[]
    for counts in count_list:
        row = []
        for token in vocabulary:
            if counts.has_key(token):
                row.append(counts[token])
            else:
                row.append(0)
        matrix.append(row)
        outstr = ",".join([str(value) for value in row])
        #print outstr
        ofile.write(outstr+"\n")
    ofile.close()
        
    owl = open("wordlist3",'w')
    matrix=[]
    for token in vocabulary:
        owl.write(token+"\n")
    owl.close()

        
    labelfile = open("labelfile3",'w')
    matrix=[]
    for author in authors:
        if(author=="charles"):
            labelfile.write("2\n")
        else:
            labelfile.write("1\n")
            
    labelfile.close()

    import numpy
    
    nmparray = numpy.array(matrix)

    
    print "it worked"

if __name__ == '__main__':
    main()