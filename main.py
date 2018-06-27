# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:14:28 2018

@author: Administrator
"""
from nltk.parse import DependencyGraph, DependencyEvaluator
from nltk.parse.transitionparser import TransitionParser



def DGload(filename, zero_based=False, cell_separator=None, top_relation_label='ROOT'):
        """
        :param filename: a name of a file in Malt-TAB format
        :param zero_based: nodes in the input file are numbered starting from 0
        rather than 1 (as produced by, e.g., zpar)
        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.
        :param str top_relation_label: the label by which the top relation is
        identified, for examlple, `ROOT`, `null` or `TOP`.

        :return: a list of DependencyGraphs

        """
        with open(filename,encoding="utf8") as infile:
            return [
                DependencyGraph(
                    tree_str,
                    zero_based=zero_based,
                    cell_separator=cell_separator,
                    top_relation_label=top_relation_label,
                )
                for tree_str in infile.read().split('\n\n')
            ]

def all_score(parsed_sent,gold_sent):
    l=0
    u=0
    a=0
    for i in range(len(gold_sent)-1):
        de = DependencyEvaluator([parsed_sent[i]],[gold_sent[i]])
        las,uas=de.eval()
        l=l+las*len(gold_sent[i].nodes)
        u=u+uas*len(gold_sent[i].nodes)
        a=a+len(gold_sent[i].nodes)
    return (l/a,u/a)

def TPprocess(datapath):
    DGtrain=DGload(datapath+r"\train.conll")
    DGtest=DGload(datapath+r"\test.conll")
    TP=TransitionParser('arc-standard')
    modelfile=datapath+r"\model"
    TP.train(DGtrain,modelfile, verbose=True)
    parsed_sent=TP.parse(DGtest,modelfile)
    las,uas=all_score(parsed_sent,DGtest)
    return las,uas

if __name__ == '__main__' :
    datapath1=r"D:\My Undergraduate\graduate paper\result\Lin2"
    datapath2=r"D:\My Undergraduate\graduate paper\result\3"
    las1,uas1=TPprocess(datapath1)
    print("las1=",las1)
    print("uas1=",uas1)
    las2,uas2=TPprocess(datapath2)
    print("las2=",las2)
    print("uas2=",uas2)


