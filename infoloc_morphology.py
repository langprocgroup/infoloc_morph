import os
import sys
import random
import functools
import itertools
import operator
from math import log, exp
from collections import Counter, defaultdict, deque

import rfutils
import rfutils.ordering
import cliqs.corpora
import numpy as np
import pandas as pd
import unimorph

import huffman
import anipa_to_ipa

# NOTE: All logarithms here are natural logarithms!

DELIMITER = '#'
EPSILON = 10 ** -5
safelog = np.log

PROTEINS_PATH = "/Users/canjo/data/genome/"
PROTEINS_FILENAME = "GRCh37_latest_protein.faa"
UNIMORPH_PATH = "/Users/canjo/data/unimorph/"
WOLEX_PATH = "/Users/canjo/data/wolex/"

def extract_manner(anipa_phone):
    if anipa_phone.startswith("C"):
        return anipa_phone[3]
    elif anipa_phone.startswith("V"):
        return "V"
    else:
        return anipa_phone # word or morpheme boundary

def extract_cv(anipa_phone):
    if anipa_phone:
        return anipa_phone[0]
    else:
        return ""

def restorer(xs):
    if isinstance(xs, str):
        return "".join
    else:
        return tuple

def sequence_transformer(f):
    def wrapped(s, *a, **k):
        restore = restorer(s)
        result = f(s, *a, **k)
        return restore(result)
    return wrapped

def delimited_sequence_transformer(f):
    def wrapped(s, *a, **k):
        restore = restorer(s)
        s = list(strip(s, DELIMITER))
        r = f(s, *a, **k)
        r.insert(0, DELIMITER)
        r.append(DELIMITER)
        return restore(r)
    return wrapped

def shuffle_by_skeleton(xs, skeleton):
    """ Shuffle xs while retaining the invariant described by skeleton. """
    # The skeleton is assumed to contain any delimiters
    assert len(skeleton) == len(xs)
    # For example, xs = "static", skeleton = "stVtVt"
    reordering = [None] * len(xs)
    for y in set(skeleton): # iterating through {t, s, V}
        old_indices = [i for i, y_ in enumerate(skeleton) if y_ == y] # first pass, [1, 3, 5]
        new_indices = old_indices.copy()
        random.shuffle(new_indices) # first pass, [3, 5, 1]
        for old_index, new_index in zip(old_indices, new_indices):
            reordering[old_index] = new_index # first pass, [None, 3, None, 5, None, 1]
    assert all(i is not None for i in reordering)
    return rfutils.ordering.reorder(xs, reordering)

def read_faa(filename):
    def gen():
        so_far = []
        code = None
        protein = None
        with open(filename) as infile:
            for line in infile:
                if line.startswith(">"):
                    if code is not None:
                        yield {
                            'protein': protein,
                            'code': code,
                            'form': DELIMITER + "".join(so_far) + DELIMITER,
                        }
                    so_far.clear()
                    code, protein = line.strip(">").split(" ", 1)
                else:
                    so_far.append(line.strip())
    return pd.DataFrame(gen())

def read_wolex(filename):
    df = pd.read_csv(filename)
    df['form'] = df['word'].map(lambda x: tuple(anipa_to_ipa.segment(x)))
    df['ipa_form'] = df['word'].map(anipa_to_ipa.convert_word)
    return df

def reorder_manner(anipa_form):
    skeleton = list(map(extract_manner, anipa_form))
    return shuffle_by_skeleton(anipa_form, skeleton)

def reorder_cv(anipa_form):
    skeleton = list(map(extract_cv, anipa_form))
    return shuffle_by_skeleton(anipa_form, skeleton)

def read_unimorph(filename, with_delimiters=True):
    with open(filename) as infile:
        lines = [line.strip().split("\t") for line in infile if line.strip()]
    lemmas, forms, features = zip(*lines)
    if not lines:
        raise FileNotFoundError
    result = pd.DataFrame({'lemma': lemmas, 'form': forms, 'features': features})
    result['lemma'] = result['lemma'].map(str.casefold)
    result['form'] = result['form'].map(str.casefold)    
    if with_delimiters:
        result['form'] = DELIMITER + result['form'] + DELIMITER
        result['lemma'] = DELIMITER + result['lemma'] + DELIMITER
    return result

def parse_infl(s):
    return sorted(s.split("|"))

def read_ud(lang):
    def gen():
        for s in cliqs.corpora.ud_corpora[lang].sentences(fix_content_head=False):
            for n in s.nodes():
                if n != 0:
                    yield {
                        'word': s.node[n]['word'],
                        'lemma': s.node[n]['lemma'],
                        'pos': s.node[n]['pos'],
                        'infl': tuple(parse_infl(s.node[n]['infl'])),
                    }
    df = pd.DataFrame(gen())
    lemma_infl = []
    word_infl = []
    for infl, lemma, word in zip(df['infl'], df['lemma'], df['word']):
        lemma_infl.append(infl + ("Lemma="+lemma,))
        word_infl.append(infl + ("Word="+word,))
    df['word_infl'] = word_infl        
    df['lemma_infl'] = lemma_infl
    return df

def genome_comparison(**kwds):
    ds = DeterministicScramble(seed=0)
    scrambles = {'even_odd': even_odd, 'shuffled': ds.shuffle}
    return comparison(read_faa, PROTEINS_PATH, [PROTEINS_FILENAME], scrambles, **kwds)

def wolex_comparison(**kwds):
    wolex_filenames = [filename for filename in os.listdir(WOLEX_PATH) if filename.endswith(".Parsed.CSV-utf8")]
    ds = DeterministicScramble(seed=0)
    scrambles = {'even_odd': even_odd, 'manner': reorder_manner, 'shuffled': ds.shuffle, 'cv': reorder_cv}
    return comparison(read_wolex, WOLEX_PATH, wolex_filenames, scrambles, **kwds)

def unimorph_comparison(**kwds):
    ds = DeterministicScramble(seed=0)
    scrambles = {'even_odd': even_odd, 'shuffled': ds.shuffle}
    return comparison(read_unimorph, UNIMORPH_PATH, unimorph.get_list_of_datasets(), scrambles, **kwds)

def ud_morpheme_order_scores(lang, with_lemma=True):
    df = read_ud(lang)
    if with_lemma:
        data = df['lemma_infl']
    else:
        data = df['infl']    
    orders = list(morpheme_orders(data))
    random.shuffle(orders)
    for order in orders:
        yield total_order_score(il.ms_auc, data, order), total_order_score(il.ee_lower_bound, data, order), order
        
def interruptible(iter, breaker=KeyboardInterrupt):
    while True:
        try:
            yield next(iter)
        except breaker:
            break
        except StopIteration:
            break

def is_monotonic(comparator, sequence, epsilon=EPSILON):
    def conditions():
        for x1, x2 in rfutils.sliding(sequence, 2):
            yield comparator(x1, x2) or comparator(x1, x2+epsilon) or comparator(x1, x2-epsilon)
    return all(conditions())

def is_monotonically_decreasing(sequence, epsilon=EPSILON):
    return is_monotonic(operator.ge, sequence, epsilon=epsilon)

def is_nonnegative(x, epsilon=EPSILON):
    return x + epsilon >= 0

def comparison(read, path, langs, scrambles, maxlen=10, seed=0):
    for lang in langs:
        filename = os.path.join(path, lang)
        print("Analyzing", filename, file=sys.stderr)

        try:
            wordforms = read(filename)['form']
        except FileNotFoundError:
            print("File not found", file=sys.stderr)
            continue
        n = len(wordforms)

        if 'count' in wordforms.columns:
            weights = wordforms['count']
        else:
            weights = None

        ht_real = curves_from_sequences(wordforms, maxlen=maxlen, weights=weights)
        ht_real['real'] = 'real'
        ht_real['lang'] = lang
        ht_real['n'] = n
        yield ht_real

        for scramble_name, scramble_fn in scrambles.items():
            ht = curves_from_sequences(wordforms.map(scramble_fn), maxlen=maxlen, weights=weights)
            ht['real'] = scramble_name
            ht['lang'] = lang
            ht['n'] = n
            yield ht

def strip(xs, y):
    result = xs
    if xs[0] == y:
        result = result[1:]
    if result[-1] == y:
        result = result[:-1]
    return result

class DeterministicScramble:
    def __init__(self, seed=0):
        self.seed = seed

    def shuffle(self, s):
        restore = restorer(s)
        r = list(strip(s, DELIMITER))
        np.random.RandomState(self.seed).shuffle(r)
        r.insert(0, DELIMITER)
        r.append(DELIMITER)
        return restore(r)


@delimited_sequence_transformer
def scramble_form(s):
    random.shuffle(s)
    return s

@delimited_sequence_transformer
def reorder_form(s, order):
    return rfutils.ordering.reorder(s, order)

@delimited_sequence_transformer
def reorder_total(s, total_order):
    return sorted(s, key=lambda x: total_order.index(x.split("=")[0]))

@delimited_sequence_transformer
def even_odd(s):
    return s[::2] + s[1::2]

@delimited_sequence_transformer
def outward_in(s):
    s = deque(s)
    r = []
    while True:
        if s:
            first = s.popleft()
            r.append(first)
        else:
            break
        if s:
            last = s.pop()
            r.append(last)
        else:
            break
    return r

def int_to_char(k, offset=ord(DELIMITER)+1): # make sure that the delimiter is not produced by accident
    return chr(k+offset)

# How many "effective phonemes" are there in a language?
# This is given by e^h.
# For example for English orthography (unimorph), h = 1.148 and H = 2.9538
# so the effective grapheme inventory given contextual redundancy is 3.15,
# and the effective grapheme inventory given unigram grapheme frequencies is 19.18
# So English writing could get away with using only 4 symbols.
# Stenographers' keyboards have ~20... similar to the unigram perplexity.
# 

def huffman_lexicon(forms, weights, n):
    codebook = huffman.huffman(weights, n=n)
    return [DELIMITER + "".join(map(int_to_char, code)) + DELIMITER for code in codebook]

def uniform_huffman_lexicon(forms, n):
    N = len(forms)
    weights = np.ones(N)/N
    return huffman_lexicon(forms, weights, n)

# The goal is to get entropy rate at time t, h_t.
# h_t = \sum_{c,x} p(c, x | t) \log p(x|c, t)
#     = \sum_{c,x} p(c, x | t) \log p(c, x | t) / p(c | t)
# So, we need to estimate two distributions:
# p(c, x | t ) and p(c | t). Or, p(c, x | t) and p(x | c, t).
# 

def curves_from_sequences(xs, weights=None, maxlen=None):
    counts = counts_from_sequences(xs, maxlen=maxlen, weights=None)
    return mle_curves_from_counts(counts['count'], counts['x_{<t}'])

def mle_curves_from_counts(counts, context, *labels):
    """ 
    Input: counts, a dataframe with columns 'x_{<t}' and 'count',
    where 'x_{<t}' gives a context, and count gives a weight or count
    for an item in that context.
    """ 
    t = context.map(len)
    joint_logp = conditional_logp_mle(counts, t, *labels)
    conditional_logp = conditional_logp_mle(counts, context, *labels)
    return curves(t, joint_logp, conditional_logp) # TODO how to labels fit in here?

def counts_from_sequences(xs, weights=None, labels=None, maxlen=None):
    if maxlen is None:
        xs = list(xs)
        maxlen = max(map(len, xs))
    if weights is None:
        weights = itertools.repeat(1)
    if labels is None:
        labels = {}
        
    counts = Counter()
    for x, w, *l in zip(xs, weights, *labels.values()):
        for context, x_t in thing_in_context(x):
            for subcontext in padded_subcontexts(context, maxlen):
                counts[(subcontext, x_t, *l)] += w
                
    def rows():
        for (subcontext, x_t, *l), count in counts.items():
            yield (*l, subcontext, x_t, count)
            
    df = pd.DataFrame(rows())
    df.columns = list(labels.keys()) + ['x_{<t}', 'x_t', 'count']
    return df

def logp_mle(counts):
    Z = counts.sum()
    return np.log(counts) - np.log(Z)

def logp_laplace(counts, alpha, V):
    Z = counts.sum()
    return np.log(counts + alpha) - np.log(Z + V*alpha)

def conditional_logp_mle(counts, *contexts):
    df =  pd.DataFrame({'count': counts})
    if contexts:
        for i, context in enumerate(contexts):
            df[i] = context
        context_cols = list(range(len(contexts)))
        Z_context = df.groupby(context_cols).sum().reset_index()
        Z_context.columns = [*context_cols, 'Z']
        df = df.join(Z_context.set_index(context_cols), on=context_cols) # preserve order
        return np.log(df['count']) - np.log(df['Z'])
    else:
        Z = counts.sum()
        return np.log(counts) - np.log(Z)

def curves(t, joint_logp, conditional_logp):
    """ 
    Input:
    t: A vector of dimension D giving time indices for observations.
    joint_logp: A vector of dimension D giving joint probabilities for observations of x and context c.
    conditional_logp: A vector of dimension D giving conditional probabiltiies for observations of x given c.

    Output:
    A dataframe of dimension max(t)+1, with columns t, h_t, I_t, and H_M_lower_bound.
    """
    plogp = np.exp(joint_logp) * conditional_logp
    h_t = -plogp.groupby(t).sum()
    assert is_monotonically_decreasing(h_t)
    I_t = -h_t.diff()
    H_M_lower_bound = np.cumsum(I_t * I_t.index)
    H_M_lower_bound[0] = 0
    df = pd.DataFrame({
        't': np.arange(len(h_t)),
        'h_t': np.array(h_t),
        'I_t': np.array(I_t),
        'H_M_lower_bound': np.array(H_M_lower_bound),
    })
    return df

def ee_lower_bound(curves):
    h = curves['h_t'].min()
    d_t = curves['h_t'] - h
    return np.trapz(y=d_t, x=curves['t'])

def ms_auc(curves):
    """ Area under the memory--surprisal trade-off curve. """
    h = curves['h_t'].min()
    d_t = curves['h_t'] - h
    return np.trapz(y=d_t, x=curves['H_M_lower_bound'])

def morpheme_orders(infl):
    features = list({fv.split("=")[0] for s in infl for fv in s} - {'Lemma'})
    n = len(features)
    for order in itertools.permutations(range(n)):
        yield ['Lemma'] + list(rfutils.ordering.reorder(features, order))

def total_order_scores(J, infl, order, weights=None): # Make sure to include lemma!
    new_infl = infl.map(lambda x: reorder_total(x, order))
    counts = counts_from_sequences(new_infl, weights)
    curves = mle_curves_from_counts(counts['count'], counts['x_{<t}'])
    return J(curves)

def permutation_scores(J, forms, weights, perms=None):
    # should take ~3hr to do a sweep over 9! permutations
    # of (3*3)!=362,880 permutations, 1296 are 3-3-contiguous (~3.6%)
    if perms is None:
        l = rfutils.the_only(forms.map(len).unique()) - 2 # 2 delimiters
        perms = itertools.permutations(range(l))
    for perm in perms:
        new_forms = forms.map(lambda s: reorder_form(s, perm))
        counts = counts_from_sequences(new_forms, weights)
        curves = mle_curves_from_counts(counts['count'], counts['x_{<t}'])        
        yield (J(curves), perm)

def rjust(xs, length, value):
    if isinstance(xs, str):
        return xs.rjust(length, value)
    else:
        num_needed = length - len(xs)
        if num_needed <= 0:
            return xs
        else:
            r = [value] * num_needed
            r.extend(xs)
            return type(xs)(r)

def test_rjust():
    assert rjust("auc", 10, "#") == "#######auc"
    assert rjust("auc", 1, "#") == "auc"
    assert rjust(tuple("auc"), 10, '#') == tuple("#######auc")

def padded_subcontext(context, length):
    return rjust(context[-length:], length, DELIMITER)

def padded_subcontexts(context, maxlen):
    yield ""
    for length in range(1, maxlen + 1):
        yield padded_subcontext(context, length)
        
def thing_in_context(xs):
    restore = restorer(xs)
    if xs[0] is DELIMITER:
        context = [DELIMITER]
        xs = xs[1:]
    else:
        context = []
    for x in xs:
        yield restore(context), x
        context.append(x)

def write_dfs(file, dfs):
    def gen():
        for df in dfs:
            for _, row in df.iterrows():
                yield dict(row)
    rfutils.write_dicts(file, gen())

def main(arg):
    if arg == 'unimorph':
        write_dfs(sys.stdout, unimorph_comparison(maxlen=10))
        return 0
    elif arg == 'wolex':
        write_dfs(sys.stdout, wolex_comparison(maxlen=10))
        return 0
    elif arg == 'genome':
        write_dfs(sys.stdout, genome_comparison(maxlen=10))
    else:
        print("Give me argument in {wolex, unimorph, genome}", file=sys.stderr)
        return 1
        
if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
