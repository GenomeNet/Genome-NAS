import numpy as np
from opendomain_utils.genotypes import Genotype, PRIMITIVES
# from GenomNet_MA.genomicNAS_algorithms.train_genomicBONAS import OPS_cnn, OPS_rnn
from generalNAS_tools.genotypes import OPS_cnn, OPS_rnn


#data_point['adjacency_matrix'], data_point['operations']) for data_point in
#                     data_points

#data_points = init_samples

#adjs, opss = [], []

#for data_point in data_points:
#   adjs.append(data_point['adjacency_matrix'])
#   opss.append(data_point['operations'])
    
#adj, ops = adjs[0], opss[0]

def transform_Genotype(adj, ops):
    # 2,3,4 und 5te zeile werden gelöscht. adj matrix nicht mehr 11x11 sonder 7x11 matrix (weil jede cell 4 Nodes hat, brauchen wir 8 edges,)
    adj = np.delete(adj, 2, axis=0) # axis=0 bedueted zeilenweise, und 2 beduetet 2te zeile wird gelöscht
    adj = np.delete(adj, 3, axis=0) # zeile 3 wird gelöscht, aber von jetztiger matrix wo schon zeile 2 gelöscht wurde, d.h. die zeile 4 von ursprünglicher matrix wird gelöscht
    adj = np.delete(adj, 4, axis=0)
    adj = np.delete(adj, 5, axis=0)
    cell_cnn = [ 
        # ops hat shape 11x6, und die ersten 2 lässt er aus, weil die ja schon die 2 Inputs aus vorheriger cell bekommen
        # dann greift er von zeile 2 bis 9 zu weil 8 edges
        (OPS_cnn[np.nonzero(ops[2])[0][0]], np.nonzero(adj[:, 2])[0][0]), 
        ## np.nonzero(ops[2])[0][0]] zeigt einfach nur von 2ter Zeile von ops an, an welcher Stelle die 1 steht
           # an der Stelle wo die 1 steht, an der Stelle ziehen wir uns OPS (deshalb hat ops 6 spalten, weil wir auch 6 Elemente in OPS haben)
           # -> somit bestimmem wir unsere operation
           # output z.B.: 'dil_conv_3x3'
        
        ## np.nonzero(adj[:, 2])[0][0] zeigt einfach nur von 2ter Spalte von adj an, an welcher Stelle die 1 steht
            # gibt dann einfach die Zahl zurück
            # somit bestimmen wir von welchem Node er die Edge bekommt
            # output z.B.: 1
            
        ## wird dann beides einfach zusammengefügt: hätten dann also als ersten Wert in cell: ('dil_conv_3x3', 1)
        (OPS_cnn[np.nonzero(ops[3])[0][0]], np.nonzero(adj[:, 3])[0][0]),
        (OPS_cnn[np.nonzero(ops[4])[0][0]], np.nonzero(adj[:, 4])[0][0]),
        (OPS_cnn[np.nonzero(ops[5])[0][0]], np.nonzero(adj[:, 5])[0][0]),
        (OPS_cnn[np.nonzero(ops[6])[0][0]], np.nonzero(adj[:, 6])[0][0]),
        (OPS_cnn[np.nonzero(ops[7])[0][0]], np.nonzero(adj[:, 7])[0][0]),
        (OPS_cnn[np.nonzero(ops[8])[0][0]], np.nonzero(adj[:, 8])[0][0]),
        (OPS_cnn[np.nonzero(ops[9])[0][0]], np.nonzero(adj[:, 9])[0][0]),
    ]
    
    cell_rnn = [ 
        # ops hat shape 11x6, und die ersten 2 lässt er aus, weil die ja schon die 2 Inputs aus vorheriger cell bekommen
        # dann greift er von zeile 2 bis 9 zu weil 8 edges
        # überspringe zeile 10 so wie ich ja auch bei CNN Zeile 0 und 1 überspringe (weil Input operation)
        (OPS_rnn[np.nonzero(ops[10])[0][0]-5], np.nonzero(adj[:, 10])[0][0]-5), 
        ## np.nonzero(ops[10])[0][0]] zeigt einfach nur 10te Zeile von ops an, an welcher Stelle die 1 steht
           # an der Stelle wo die 1 steht, an der Stelle ziehen wir uns OPS (deshalb hat ops 6 spalten, weil wir auch 6 Elemente in OPS haben)
           # -> somit bestimmem wir unsere operation
           # output z.B.: 'dil_conv_3x3'
        
        ## np.nonzero(adj[:, 2])[0][0] zeigt einfach nur von 2ter Spalte von adj an, an welcher Stelle die 1 steht
            # gibt dann einfach die Zahl zurück
            # somit bestimmen wir von welchem Node er die Edge bekommt
            # output z.B.: 1
            
        ## wird dann beides einfach zusammengefügt: hätten dann also als ersten Wert in cell: ('dil_conv_3x3', 1)
        
        (OPS_rnn[np.nonzero(ops[11])[0][0]-5], np.nonzero(adj[:, 11])[0][0]-5), # bei ops 5 abziehen, weil es 5 CNN ops gibt (spalten 0-4); bei input 5 abziehen, weil es 5 nodes gibt (0-4 zeilen von adj) cellt-1, cellt-2, Node0, Node1, Node2 und von Node3 bekommt RHN ja schon input
        (OPS_rnn[np.nonzero(ops[12])[0][0]-5], np.nonzero(adj[:, 12])[0][0]-5),
        (OPS_rnn[np.nonzero(ops[13])[0][0]-5], np.nonzero(adj[:, 13])[0][0]-5),
        (OPS_rnn[np.nonzero(ops[14])[0][0]-5], np.nonzero(adj[:, 14])[0][0]-5),
        (OPS_rnn[np.nonzero(ops[15])[0][0]-5], np.nonzero(adj[:, 15])[0][0]-5),
        (OPS_rnn[np.nonzero(ops[16])[0][0]-5], np.nonzero(adj[:, 16])[0][0]-5),
        (OPS_rnn[np.nonzero(ops[17])[0][0]-5], np.nonzero(adj[:, 17])[0][0]-5),
    ]
    
    ft_model = Genotype(
        normal=cell_cnn,
        normal_concat=[2, 3, 4, 5],
        reduce=cell_cnn,
        reduce_concat=[2, 3, 4, 5],
        rnn = cell_rnn,
        rnn_concat = range(1,9)
    )
    return ft_model

def transform_matrix(genotype):
    normal = genotype.normal
    node_num = len(normal)+3
    adj = np.zeros((node_num, node_num))
    ops = np.zeros((node_num, len(OPS)))
    for i in range(len(normal)):
        op, connect = normal[i]
        if connect == 0 or connect==1:
            adj[connect][i+2] = 1
        else:
            adj[(connect-2)*2+2][i+2] = 1
            adj[(connect-2)*2+3][i+2] = 1
        ops[i+2][OPS.index(op)] = 1
    adj[2:-1, -1] = 1
    ops[0:2, 0] = 1
    ops[-1][-1] = 1
    return adj, ops

def geno_to_archs(genotypes, ei_scores=None):
    # print(genotypes)
    archs = []
    for i in range(len(genotypes)):
        if isinstance(genotypes, str):
            adj, op = transform_matrix(eval(genotypes[i]))
        else:
            adj, op = transform_matrix(genotypes[i])
        if ei_scores:
            datapoint = {'adjacency_matrix': adj, 'operations': op, 'metrics': ei_scores[i]}
        else:
            datapoint = {'adjacency_matrix': adj, 'operations': op}
        archs.append(datapoint)
    return archs

def geno2mask(genotype):
    des = -1
    mask = np.zeros((14, 8))
    
    op_names, indices = zip(*genotype.normal)
    for cnt, (name, index) in enumerate(zip(op_names, indices)):
        if cnt % 2 == 0:
            des += 1
            total_state = sum(i+2 for i in range(des))
        op_idx = PRIMITIVES.index(name)
        node_idx = index + total_state
        mask[node_idx, op_idx] = 1
    print(mask)
    return mask

if __name__ == '__main__':
    A = np.array([
        [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    ops = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
    ])
    geno = transform_Genotype(A, ops)
    print(geno.normal)
    # print(geno.ops)
    # print(transform_matrix(geno) - A)
    print(transform_matrix(geno)[1] - ops)
