#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:25:25 2021

@author: amadeu
"""



import random
import numpy as np
import time
from generalNAS_tools.genotypes import OPS_cnn, OPS_rnn



# erzeugt glaube ich einfach nur eine random adj matrix 11x11
def generate_adj():
    #mat = np.zeros([11, 11]) # erzeugt eine 11x11 Matrix "mat" nur mit 0en (die ersten 2 spalten für cell_t-1 und cell_t-2, 8 weitere spalten, weil jeder Node 2 Inputs bekommt)
    mat = np.zeros([19, 19]) # erzeugt eine 19x19 Matrix "mat" nur mit 0en (die ersten 2 spalten für cell_t-1 und cell_t-2, 8 weitere spalten, weil 4 CNN Nodes die jeweils 2 Inputs bekommen
    # und 8 weiter RHN Nodes, welche jeder nur einen Input bekommt

    #mat[:, 10] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0] # jetzt ist 10te/letzte/11te Spalte von "mat" "[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]" anstatt nur die 0en
    # er hat überall 1en wo es Verbindungen gibt: die 8 Zeilen zu denen es 8 Inputs gibt (4 Nodes á 2 Inputs): wrsl brauch er das für diesen global Node
    # ist einfach nur der OutputNode, welcher Input von allen 4 Nodes erhält
    #mat[:, 10] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,0,0,0,0,0,0,0,0] # jetzt ist 10te/letzte/11te Spalte von "mat" "[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]" anstatt nur die 0en

    ## Random choices for CNN ##
    a = random.choice([0, 1]) # 1 random Wert der entweder 0 oder 1 ist
    b = random.choice([0, 1])
    c = random.choice([0, 1, [2,3]]) # random choice, aber diesmal entweder 0, 1 oder [2,3]
    d = random.choice([0, 1, [2,3]])
    e = random.choice([0, 1, [2,3], [4,5]])  # random choice, aber diesmal entweder 0, 1 oder [2,3] oder [4,5]
    f = random.choice([0, 1, [2,3], [4,5]])
    g = random.choice([0, 1, [2,3], [4,5],[6,7]])  # random choice, aber diesmal entweder 0, 1 oder [2,3] oder [4,5] oder [6,7], also 5 inputmöglichkeiten. Wobei er ja bei transform_genotypes 
    # ja dann die 2,4,,6,8te Zeile löscht
    h = random.choice([0, 1, [2,3], [4,5], [6,7]])
    
    ## Random choices for RHN ## 
    aa = random.choice([[8,9], 10]) # bekommt immer 1ten Input
    bb = random.choice([[8,9], 10, 11]) 
    cc = random.choice([[8,9], 10, 11, 12]) # random choice, aber diesmal entweder 0, 1 oder [2,3]
    dd = random.choice([[8,9], 10, 11, 12, 13])
    ee = random.choice([[8,9], 10, 11, 12, 13, 14])  # random choice, aber diesmal entweder 0, 1 oder [2,3] oder [4,5]
    ff = random.choice([[8,9], 10, 11, 12, 13, 14, 15])
    gg = random.choice([[8,9], 10, 11, 12, 13, 14, 15, 16])  # random choice, aber diesmal entweder 0, 1 oder [2,3] oder [4,5] oder [6,7], also 5 inputmöglichkeiten. Wobei er ja bei transform_genotypes 
    # ja dann die 2,4,,6,8te Zeile löscht
    #hh = random.choice([8,9], 10, 11, 12, 13, 14, 15, 16, 17)
        
    
    # gemäß, den randomchoices von oben werden jetzt 1en an bestimmten stellen für die 0en ausgetauscht/eingesetzt
    mat[a, 2] = 1  # nur spalte 2 weil spalte 0 und 1 steht für cell_t-1 und cell_t-2 und können nichts empfangen (deswegen auch ersten 2 spalten nur 0en), außerdem
    # kann spalte 2 (weil Node 0) nur von 1 und 2 empfangen
    mat[b, 3] = 1 
    mat[c, 4] = 1
    mat[d, 5] = 1
    mat[e, 6] = 1
    mat[f, 7] = 1
    mat[g, 8] = 1
    mat[h, 9] = 1
    
    mat[8, 10] = 1  # nur spalte 2 weil spalte 0 und 1 steht für cell_t-1 und cell_t-2 und können nichts empfangen (deswegen auch ersten 2 spalten nur 0en), außerdem
    # kann spalte 2 (weil Node 0) nur von 1 und 2 empfangen
    mat[9, 10] = 1 
    mat[aa, 11] = 1
    mat[bb, 12] = 1 
    mat[cc, 13] = 1
    mat[dd, 14] = 1
    mat[ee, 15] = 1
    mat[ff, 16] = 1
    mat[gg, 17] = 1
    
    # output Node receives Input from each 
    mat[:, 18] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    
    return mat

#test = generate_adj() # ergibt 11x11 matrix

# genereriert eine random feature matrix 11x10
def generate_ops():
    op_num_cnn = len(OPS_cnn)
    op_num_rnn = len(OPS_rnn)
    op_matrix = np.zeros((19, op_num_cnn+op_num_rnn)) # rows stand for nodes/edges
    ## first 2 rows receive 'input' as operation
    op_matrix[0][0] = 1 # 1th operation is 'input', cell_t-1 
    op_matrix[1][0] = 1 # 2th operation is 'input', weil cell_t-2
    # loop over 16 rows; 17th row can not do any operation as it is the last node of rnn and 18th node is the global node
    for i in range(8+8): # 0,1,2,...7; 8,9,10,11,12,13,14,15
        if i<8: # 0,1,2,3,4,5,6,7
            # choose randomly one of the 6 possible operation
            idx_cnn = random.choice(list(range(1, op_num_cnn))) # random choice except 0 which stands for 'input' operation
            # fängt bei 1 und nicht 0 an, weil 0 ist Inputoperation und die darf nicht gewählt werden
            op_matrix[i + 2][idx_cnn] = 1 # füge an spalten-stelle idx eine 1 ein, und Zeile ist 2,3,4,5,6,7,8 ()
        # Zeilen 10-18 bekommen random rnn operationen
        if i>=8: # 8,9,10,11,12,13,14,15
            # i=14
            idx_rnn = random.choice(list(range(0, op_num_rnn-1))) # random choice except 4 which stands for 'output' operation

            op_matrix[i + 2][idx_rnn+5] = 1 # füge an spalten-stelle idx eine 1 ein, und Zeile ist 2,3,4,5,6,7,8 ()

    #op_matrix[10][6] = 1 # weil letzte Zeile von CNN ist natürlich die Operation None: deshalb
    # müssen wir an letzter Zeile (10) von CNN zugreifen weil für output Node steht und dann eben an letzter spalte von dieser Zeile eine 1 setzen, 
    # weil letzte Spalte steht für operation output
    op_matrix[18,9] = 1 # letzte Zeile steht für output Node und dieser erhält ja operation output_rnn
    return op_matrix

# test2 = generate_ops() # ergibt 11x10 matrix

# generate_num = 10000 
# generate_num = 5 
def generate_archs(generate_num):
    archs = []
    archs_hash = []
    cnt = 0
    # es werden zuerst random adj und ops matritzen erzeugt und diese
    # werden dann einfach als dictionary eingespeichert und jedes dictionary bildet ein element einer liste archs
    while cnt < generate_num:
        adj = generate_adj()
        ops = generate_ops()
        if is_valid(adj, ops): # dieser Teil macht meiner meinung nach keinen Sinn
            arch = {"adjacency_matrix":adj, "operations":ops}
            arch_hash = str(hash(str(arch)))
            if arch_hash not in archs_hash:
                archs.append(arch)
                archs_hash.append(arch_hash)
                cnt += 1
    return archs

def is_valid(adj, op, step=4):
    for i in range(step):
        # i=0
        if (adj[:, 2 * i + 2] == adj[:, 2 * i + 3]).all() and (op[2 * i + 2, :] == op[2 * i + 3, :]).all(): # Inputs von [2,3] müssen nicht gleich sein / Spalten von adj (was aber immer True sein wird)
        # und 2te Bedinung ist meistens False, weil operationen gleich sein müssen
            return 0
    return 1

if __name__ =='__main__':
    t1 = time.time()
    arch = generate_archs(10000)
