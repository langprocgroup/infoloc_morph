import itertools

import numpy as np
import scipy.special
import pandas as pd
import rfutils

def random_code(M, S, l):
    """ 
    Input:
    M: Number of messages.
    S: Number of distinct signals.
    l: Signal length.

    Output:
    An M x S^l array e, where e[m,:] = the length-l code for m.
    """
    return np.random.randint(S, size=(M, l))

def encode_contiguous(ms, code):
    return np.hstack(code[ms,:])

def ints_to_str(ints):
    return "".join(chr(65 + i) for i in ints)

def is_contiguous(k, l, perm):
    canonical_order = range(k*l)
    breaks = [l*k_ for k_ in range(k)]
    words = rfutils.segments(canonical_order, breaks)
    index_sets = {frozenset(word) for word in words}
    return all(
        frozenset(perm_segment) in index_sets
        for perm_segment in rfutils.segments(perm, breaks)
    )

def word_probabilities(M, k, code):
    """
    Input:
    code: Array of size k x S^lk, 

    An array y of shape S^l, where y[*s] is the probability of word s.

    p(s) = \sum_m p(s|m) p(m).
    
    """
    p_Mk = scipy.special.softmax(np.random.randn(*(M,)*k))
    def gen():
        for i, mk in enumerate(itertools.product(*[range(M)]*k)):
            yield ints_to_str(encode_contiguous(mk, code)), p_Mk[mk]
    df = pd.DataFrame(gen())
    df.columns = ['form', 'probability']
    df['form'] = '#' + df['form'] + '#'
    return df.groupby(['form']).sum().reset_index()
    
    

