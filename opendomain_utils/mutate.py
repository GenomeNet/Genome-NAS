import numpy as np
import copy
import random

# matrix = new_adj

def is_full_dag(matrix):
  """Full DAG == all vertices on a path from vert 0 to (V-1).

  i.e. no disconnected or "hanging" vertices.

  It is sufficient to check for:
    1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
    2) no cols of 0 except for col 0 (only input vertex has no in-edges)

  Args:
    matrix: V x V upper-triangular adjacency matrix

  Returns:
    True if the there are no dangling vertices.
  """
  # matrix = mutated['adjacency_matrix_cnn']
  shape = np.shape(matrix)

  rows = matrix[:shape[0]-1, :] == 0 # zuerst wird matrix neu gebildet, indem einfach letzte Zeile gekickt wird; true if 0, false if 1
  rows = np.all(rows, axis=1)     # Any row with all 0 will be True
  rows_bad = np.any(rows) # falls es ein element in rows gibt, was true ist (also es eine zeile gibt mit nur 0en)

  cols = matrix[:, 2:] == 0 # neue matrix, indem einfach erste spalte gekickt wird, 
  cols = np.all(cols, axis=0)     # Any col with all 0 will be True
  cols_bad = np.any(cols)

  return (not rows_bad) and (not cols_bad)


def mutate_adj(old_adj):
    # old_adj = old_arch['adjacency_matrix_cnn']
    new_adj = copy.deepcopy(old_adj)
    # define new connection
    # node_to_connect = random.choice(list(range(0,5))) # min=0 ; max=4 ; 3
    all_possible = list([2, 3, 4, 5, 6, 7, 8, 9])
    
    stop = True
    while stop:
        #####
        node_to_connect = random.choice([[0,0], [1,1], [2,3], [4,5], [6,7]])
    
        up = max(node_to_connect)
        
        possible_connects = [x for x in all_possible if x > up]
        
        exclude = np.nonzero(new_adj[node_to_connect[0],:-1])[0]
        
        possible_connects = [x for x in possible_connects if x not in exclude]
        
        if len(possible_connects) != 0:
            stop=False

    to_connect = random.choice(possible_connects)
        
    node_to_disconnect = np.nonzero(new_adj[:,to_connect])
    
    new_adj[node_to_disconnect, to_connect] = 0
    
    new_adj[node_to_connect, to_connect] = 1
    # print("node to connect:{}, to connect:{}, node to disconnect:{}, to disconnect:{}".format(node_to_connect,to_connect,node_to_disconnect,to_disconnect))
    return new_adj


def mutate_ops(old_ops):
    # old_ops = old_arch['operations_cnn']
    new_ops = copy.deepcopy(old_ops)
    node_to_mutate = random.choice(list(range(2,10)))
    
    op_to_disc = np.nonzero(old_ops[node_to_mutate,:])

    # op_to_set = random.choice(list(range(0,3)))
    new_ops[node_to_mutate,op_to_disc] = 0
    
    possible_ops = [1,2,3,4] 

    possible_ops = [x for x in possible_ops if x != int(np.array(op_to_disc))]

    op_to_add = random.choice(possible_ops)
    new_ops[node_to_mutate, op_to_add] = 1
    return new_ops


def mutate_adj_rhn(old_adj):
    # old_adj = init_samples[0]['adjacency_matrix_rhn']
    # old_adj = old_arch['adjacency_matrix_rhn']
    new_adj = copy.deepcopy(old_adj)
    # define new connection
    # node_to_connect = random.choice(list(range(0,5))) # min=0 ; max=4 ; 3
    all_possible = list([3, 4, 5, 6, 7, 8, 9, 10])
        
    stop = True
    while stop:
        node_to_connect = random.choice([2,3,4,5,6,7,8,9])

        possible_connects = [x for x in all_possible if x > node_to_connect]
        
        exclude = np.nonzero(new_adj[node_to_connect,:-1])[0]
        
        possible_connects = [x for x in possible_connects if x not in exclude]
        
        if len(possible_connects) != 0:
            stop=False
        
    to_connect = random.choice(possible_connects)
        
    node_to_disconnect = np.nonzero(new_adj[:,to_connect])
    
    new_adj[node_to_disconnect, to_connect] = 0
    
    new_adj[node_to_connect, to_connect] = 1
    # print("node to connect:{}, to connect:{}, node to disconnect:{}, to disconnect:{}".format(node_to_connect,to_connect,node_to_disconnect,to_disconnect))
    return new_adj


def mutate_ops_rhn(old_ops):
    # old_ops = old_arch['operations_rhn']
    new_ops = copy.deepcopy(old_ops)
    node_to_mutate = random.choice(list(range(3,11)))
    
    op_to_disc = np.nonzero(old_ops[node_to_mutate,:])

    # op_to_set = random.choice(list(range(0,3)))
    new_ops[node_to_mutate,op_to_disc] = 0
    
    possible_ops = [2,3,4,5] 

    possible_ops = [x for x in possible_ops if x != int(np.array(op_to_disc))]

    op_to_add = random.choice(possible_ops)
    new_ops[node_to_mutate, op_to_add] = 1
    return new_ops


def mutate_arch(old_arch):
    
    # old_arch = parent
    # parent['adjacency_matrix_cnn']==samples[2]['adjacency_matrix_cnn']
    change = random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12])
    
    if change==0: # chance adj_cnn
        
        new_arch = {'adjacency_matrix_cnn': mutate_adj(old_arch['adjacency_matrix_cnn']), "operations_cnn": old_arch['operations_cnn'], 'adjacency_matrix_rhn': old_arch['adjacency_matrix_rhn'], "operations_rhn": old_arch['operations_rhn']}
    
    elif change==1: # change ops_cnn
        
        new_arch = {'adjacency_matrix_cnn': old_arch['adjacency_matrix_cnn'], "operations_cnn": mutate_ops(old_arch['operations_cnn']), 'adjacency_matrix_rhn': old_arch['adjacency_matrix_rhn'], "operations_rhn": old_arch['operations_rhn']}
        
    elif change==2: # change adj_rhn
        
        new_arch = {'adjacency_matrix_cnn': old_arch['adjacency_matrix_cnn'], "operations_cnn": old_arch['operations_cnn'], 'adjacency_matrix_rhn': mutate_adj_rhn(old_arch['adjacency_matrix_rhn']), "operations_rhn": old_arch['operations_rhn']}
        
    elif change==3: # change ops_rhn
        
        new_arch = {'adjacency_matrix_cnn': old_arch['adjacency_matrix_cnn'], "operations_cnn": old_arch['operations_cnn'], 'adjacency_matrix_rhn': old_arch['adjacency_matrix_rhn'], "operations_rhn": mutate_ops_rhn(old_arch['operations_rhn'])}
        
    elif change==4: # change adj_cnn and adj_rhn
        
        new_arch = {'adjacency_matrix_cnn': mutate_adj(old_arch['adjacency_matrix_cnn']), "operations_cnn": old_arch['operations_cnn'], 'adjacency_matrix_rhn': mutate_adj_rhn(old_arch['adjacency_matrix_rhn']), "operations_rhn": old_arch['operations_rhn']}
        
    elif change==5: # change ops_cnn and ops_rhn
        
        new_arch = {'adjacency_matrix_cnn': old_arch['adjacency_matrix_cnn'], "operations_cnn": mutate_ops(old_arch['operations_cnn']), 'adjacency_matrix_rhn': old_arch['adjacency_matrix_rhn'], "operations_rhn": mutate_ops_rhn(old_arch['operations_rhn'])}
        
    elif change==6: # change adj_cnn and ops_rhn
        
        new_arch = {'adjacency_matrix_cnn': mutate_adj(old_arch['adjacency_matrix_cnn']), "operations_cnn": old_arch['operations_cnn'], 'adjacency_matrix_rhn': old_arch['adjacency_matrix_rhn'], "operations_rhn": mutate_ops_rhn(old_arch['operations_rhn'])}
        
    elif change==7: # change adj_rhn and ops_cnn
        
        new_arch = {'adjacency_matrix_cnn': old_arch['adjacency_matrix_cnn'], "operations_cnn": mutate_ops(old_arch['operations_cnn']), 'adjacency_matrix_rhn': mutate_adj_rhn(old_arch['adjacency_matrix_rhn']), "operations_rhn": old_arch['operations_rhn']}
        
    elif change==8: # change adj_cnn, adj_rhn and ops_rhn
        
        new_arch = {'adjacency_matrix_cnn': mutate_adj(old_arch['adjacency_matrix_cnn']), "operations_cnn": old_arch['operations_cnn'], 'adjacency_matrix_rhn': mutate_adj_rhn(old_arch['adjacency_matrix_rhn']), "operations_rhn": mutate_ops_rhn(old_arch['operations_rhn'])}
        
    elif change==9: # change adj_cnn, adj_rhn and ops_cnn
        
        new_arch = {'adjacency_matrix_cnn': mutate_adj(old_arch['adjacency_matrix_cnn']), "operations_cnn": mutate_ops(old_arch['operations_cnn']), 'adjacency_matrix_rhn': mutate_adj_rhn(old_arch['adjacency_matrix_rhn']), "operations_rhn": old_arch['operations_rhn']}
        
    elif change==10: # change adj_cnn, ops_cnn and ops_rhn
        
        new_arch = {'adjacency_matrix_cnn': mutate_adj(old_arch['adjacency_matrix_cnn']), "operations_cnn": mutate_ops(old_arch['operations_cnn']), 'adjacency_matrix_rhn': old_arch['adjacency_matrix_rhn'], "operations_rhn": mutate_ops_rhn(old_arch['operations_rhn'])}
        
    elif change==11: # change adj_rhn and ops_cnn and ops_rhn
        
        new_arch = {'adjacency_matrix_cnn': old_arch['adjacency_matrix_cnn'], "operations_cnn": mutate_ops(old_arch['operations_cnn']), 'adjacency_matrix_rhn': mutate_adj_rhn(old_arch['adjacency_matrix_rhn']), "operations_rhn": mutate_ops_rhn(old_arch['operations_rhn'])}
    
    else:  # change adj_cnn, adj_rhn, ops_cnn and ops_rhn
        new_arch = {'adjacency_matrix_cnn': mutate_adj(old_arch['adjacency_matrix_cnn']), "operations_cnn": mutate_ops(old_arch['operations_cnn']), 'adjacency_matrix_rhn': mutate_adj_rhn(old_arch['adjacency_matrix_rhn']), "operations_rhn": mutate_ops_rhn(old_arch['operations_rhn'])}

    return new_arch

# old_arch = parent
# previous_data = dataset_x
def all_mutates(candidates, num_samples, previous_data):
    
    new_archs = []

    strs = []
    
    for previous in previous_data:
        strs.append(str(previous))
        
    no_stop = True
    cnt = 0
    all_samples = []
    
    while no_stop:
        parent = candidates[cnt]
        print('parent')
        print(parent)

        cnt += 1
        
        for i in range(30000):
            mutated = mutate_arch(old_arch=parent)
            if is_full_dag(mutated['adjacency_matrix_cnn']) and str(mutated) not in strs:
                # str(mutated) == str(previous_data)
                # strs.append(str(previous_data[1]))
                strs.append(str(mutated))
    
                new_archs.append(mutated)

        
        if len(new_archs) > num_samples:
            no_stop=False
            
    new_archs = random.sample(new_archs, num_samples)
            
    return new_archs




