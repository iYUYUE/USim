#!/usr/bin/env python

import codecs, os, sys, re
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize

def parseSnippets(xmlfile): 
  
    # create element tree object 
    tree = ET.parse(xmlfile) 
  
    # get root element 
    root = tree.getroot() 
  
    # create empty dict for news items 
    dictionary = {} 

    # iterate news items 
    for item in root.findall('./{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}snippets/{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}snippet'): 
        # print(item.text)
        dictionary[item.attrib['id']] = item.text
      
    # return news items list 
    return dictionary

# print the gold alignments for both partitions
def main(argv):

    xmlfile = 'P4P.xml'
    
    snippets = parseSnippets(xmlfile)

    # create element tree object 
    tree = ET.parse(xmlfile) 
  
    # get root element 
    root = tree.getroot() 
  
    triples = [] 

    # iterate news items 
    for item in root.findall('./{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}paraphrase_candidates'): 
        ids = item.findall('{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}snippet')
        str1 = snippets[ids[0].attrib['id']]
        str2 = snippets[ids[1].attrib['id']]
        align = ''
        for phenomenon in item.find('{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}annotation'):
            if(phenomenon.attrib['type'] not in ['addition_deletion', 'syn_diathesis']):
                aligns = phenomenon.findall('{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}snippet')
                # print(aligns[0].attrib['id'])
                offset1 = int(aligns[0].find('{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}scope').attrib['offset'])
                # print(offset1)
                offset2 = int(aligns[1].find('{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}scope').attrib['offset'])
                length1 = int(aligns[0].find('{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}scope').attrib['length'])
                length2 = int(aligns[1].find('{http://clic.ub.edu/mbertran/formats/paraphrase-corpus}scope').attrib['length'])
                wordList1 = [str(i) for i in range(len(word_tokenize(str1[0:offset1]))+1, len(word_tokenize(str1[0:offset1+length1]))+1)]
                align = align + '_'.join(wordList1) + '-'
                wordList2 = [str(i) for i in range(len(word_tokenize(str2[0:offset2]))+1, len(word_tokenize(str2[0:offset2+length2]))+1)]
                align = align + '_'.join(wordList2) + '-' + phenomenon.attrib['type'] + ','
        triples.append((str1, str2, align))
    

    outf = open('p4p_source.txt', 'wb')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw = streamWriter(outf)

    for t in triples:
        outw.write(' '.join(word_tokenize(t[0])) + '\n')

    outf.flush()

    outf = open('p4p_reference.txt', 'wb')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw = streamWriter(outf)

    for t in triples:
        outw.write(' '.join(word_tokenize(t[1])) + '\n')

    outf.flush()

    outf = open('p4p_alignment.txt', 'wb')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw = streamWriter(outf)

    for t in triples:
        outw.write(t[2] + '\n')

    outf.flush()


if __name__ == '__main__' : main(sys.argv)
