import heapq

BOTTOM = float('-inf')

def huffman(weights, n=2):
    """ N-ary Huffman code

    Inputs: 
    weights: An iterable of weights for symbols.
    n: An integer giving the base for the encoding (by default 2).
    
    Output: 
    A sequence of codes (tuples of integers in range(n)), 
    where the i'th element of the sequence is the Huffman code for
    the i'th element of the input weights.
    """
    # n-ary huffman code, from Cover & Thomas (2006: 118--119)
    assert n >= 2
    heap = [(weight, [i]) for i, weight in enumerate(weights)]
    codebook = [[] for _ in range(len(heap))]
    if n > 2:
        while len(heap) % (n-1) != 1:
            heap.append((BOTTOM, [None]))
    heapq.heapify(heap)
    while len(heap) > 1:
        new_supersymbol = []
        new_weight = 0
        for j in range(n): # lower prob = higher integer
            weight, supersymbol = heapq.heappop(heap)
            for symbol in supersymbol:
                if symbol is not None:
                    codebook[symbol].append(j)
                    new_supersymbol.append(symbol)
            if weight != BOTTOM:
                new_weight += weight
        heapq.heappush(heap, (new_weight, new_supersymbol))
    return [tuple(reversed(code)) for code in codebook]


def test_huffman():
    code = huffman([1, 1, 1, 1], 2)
    assert set(code) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    code = huffman([1, 1, 1, 1], 4)
    assert set(code) == {(0,), (1,), (2,), (3,)}

    code = huffman([1/2, 1/4, 1/8, 1/8], 2)
    assert (list(code) == [(0,), (1, 0), (1, 1, 0), (1, 1, 1)]
            or list(code) == [(0,), (1, 0), (1, 1, 1), (1, 1, 0)])

    # Example from Cover & Thomas (2006: 119)
    code = huffman([.25, .25, .2, .1, .1, .1], 3)
    assert set(code) == {(0,), (1,), (2, 0), (2, 1), (2, 2, 1), (2, 2, 2)}

if __name__ == '__main__':
    import nose
    nose.runmodule()

    
    
