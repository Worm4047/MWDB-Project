import numpy as np

def sort_print_n_return(npArray):
    ind = np.argsort(-npArray, axis=1)
    rows = ind.shape[0]
    cols = ind.shape[1]
    twpair = np.zeros((cols,), dtype='i,f')
    for x in range(0, rows):
        l = []
        for y in range(0, cols):
            l.append((ind[x, y], npArray[x,ind[x,y]]))
        twpair = np.vstack([twpair,  np.array(l, dtype='i,f')])
    #need to delete the initialized all zero row
    twpair = np.delete(twpair, (0), axis=0)
    return twpair