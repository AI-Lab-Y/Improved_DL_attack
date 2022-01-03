import numpy as np
import speck as sp
import random
from os import urandom


def make_target_diff_samples(n=64, nr=11, diff=(0x211, 0xa04), flag=1):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    if flag == 1:
        p1l = p0l ^ diff[0]
        p1r = p0r ^ diff[1]
    else:
        p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(key, nr)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)
    return c0l, c0r, c1l, c1r, ks[nr-1][0]


# make a plaintext structure from a random plaintext pair
# diff: difference of the plaintext pair
# neutral_bits is used to generate the plaintext structure
def make_plaintext_structure(nr=11, diff_in=(0x211, 0xa04), diff_out=(0x0040, 0x0), neutral_bits=None, flag=1):
    key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(key, nr)
    ks_v = sp.expand_key(key, 3)
    if flag == 1:
        while 1:
            p0l = np.frombuffer(urandom(2), dtype=np.uint16)
            p0r = np.frombuffer(urandom(2), dtype=np.uint16)
            for i in neutral_bits:
                if isinstance(i, int):
                    i = [i]
                d0 = 0
                d1 = 0
                for j in i:
                    d = 1 << j
                    d0 |= d >> 16
                    d1 |= d & 0xffff
                p0l = np.concatenate([p0l, p0l ^ d0])
                p0r = np.concatenate([p0r, p0r ^ d1])
            p1l = p0l ^ diff_in[0]
            p1r = p0r ^ diff_in[1]
            p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
            p1l, p1r = sp.dec_one_round((p1l, p1r), 0)

            t0l, t0r = sp.encrypt((p0l, p0r), ks_v)
            t1l, t1r = sp.encrypt((p1l, p1r), ks_v)
            if np.sum(t0l ^ t1l ^ diff_out[0]) == 2 ** len(neutral_bits) and np.sum(t0r ^ t1r ^ diff_out[1]) == 2 ** len(neutral_bits):
                c0l, c0r = sp.encrypt((p0l, p0r), ks)
                c1l, c1r = sp.encrypt((p1l, p1r), ks)
                return c0l, c0r, c1l, c1r, ks[nr-1][0]
    else:
        while 1:
            p0l = np.frombuffer(urandom(2), dtype=np.uint16)
            p0r = np.frombuffer(urandom(2), dtype=np.uint16)
            for i in neutral_bits:
                if isinstance(i, int):
                    i = [i]
                d0 = 0
                d1 = 0
                for j in i:
                    d = 1 << j
                    d0 |= d >> 16
                    d1 |= d & 0xffff
                p0l = np.concatenate([p0l, p0l ^ d0])
                p0r = np.concatenate([p0r, p0r ^ d1])
            p1l = p0l ^ diff_in[0]
            p1r = p0r ^ diff_in[1]
            p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
            p1l, p1r = sp.dec_one_round((p1l, p1r), 0)

            t0l, t0r = sp.encrypt((p0l, p0r), ks_v)
            t1l, t1r = sp.encrypt((p1l, p1r), ks_v)
            if np.sum(t0l ^ t1l ^ diff_out[0]) < 2 ** len(neutral_bits) or np.sum(t0r ^ t1r ^ diff_out[1]) < 2 ** len(neutral_bits):
                c0l, c0r = sp.encrypt((p0l, p0r), ks)
                c1l, c1r = sp.encrypt((p1l, p1r), ks)
                return c0l, c0r, c1l, c1r, ks[nr-1][0]


# x0l shape: [n]
# output shape: [n]
def extrac_bits_to_uint(x0l, x0r, x1l, x1r, bits=None):
    n = len(x0l)
    m = len(bits)
    r0l = np.zeros(n, dtype=np.uint32)
    r0r = np.zeros(n, dtype=np.uint32)
    r1l = np.zeros(n, dtype=np.uint32)
    r1r = np.zeros(n, dtype=np.uint32)
    for i in range(m):
        index = bits[i]
        offset = m - 1 - i
        r0l = r0l + (((x0l >> index) & 1) << offset)
        r0r = r0r + (((x0r >> index) & 1) << offset)
        r1l = r1l + (((x1l >> index) & 1) << offset)
        r1r = r1r + (((x1r >> index) & 1) << offset)
    res = (r0l << (3*m)) + (r0r << (2*m)) + (r1l << m) + r1r
    return res


def test_rank_score(t=100, n=64, nr=7, diff=(0x0040, 0x0), nd=None, bits=None, c=20):
    acc_t = 0
    acc_w = 0
    nd = np.load(nd)
    print('the results of true key are: ')
    for i in range(t):
        c0l, c0r, c1l, c1r, sk = make_target_diff_samples(n=n, nr=nr+1, diff=diff, flag=1)
        d0l, d0r = sp.dec_one_round((c0l, c0r), sk)
        d1l, d1r = sp.dec_one_round((c1l, c1r), sk)
        x = extrac_bits_to_uint(d0l, d0r, d1l, d1r, bits=bits)
        z = nd[x]
        s = np.sum(np.log2(z / (1 - z)))
        print('score under true key is ', s)
        if s > c:
            acc_t = acc_t + 1

    print('the results of wrong keys are: ')
    for i in range(t):
        c0l, c0r, c1l, c1r, sk = make_target_diff_samples(n=n, nr=nr+1, diff=diff, flag=0)
        wk = random.randint(0, 2 ** 16 - 1)
        d0l, d0r = sp.dec_one_round((c0l, c0r), wk)
        d1l, d1r = sp.dec_one_round((c1l, c1r), wk)
        x = extrac_bits_to_uint(d0l, d0r, d1l, d1r, bits=bits)
        z = nd[x]
        s = np.sum(np.log2(z / (1 - z)))
        print('score under wrong key is ', s)

        if s <= c:
            acc_w = acc_w + 1

    print('acc_t is ', acc_t / t, ' acc_w is ', acc_w / t)


def test_rank_score_v2(t=100, n=64, nr=7, diff=(0x0040, 0x0), nd=None, bits=None, NBs=None):
    nd = np.load(nd)
    print('the results of true key are: ')
    for i in range(t):
        c0l, c0r, c1l, c1r, sk = make_plaintext_structure(nr=nr+1, diff_in=diff, diff_out=(0x0040, 0), neutral_bits=NBs,
                                                      flag=1)
        # c0l, c0r, c1l, c1r, sk = make_target_diff_samples(n=n, nr=nr+1, diff=diff, flag=1)
        d0l, d0r = sp.dec_one_round((c0l, c0r), sk)
        d1l, d1r = sp.dec_one_round((c1l, c1r), sk)
        x = extrac_bits_to_uint(d0l, d0r, d1l, d1r, bits=bits)
        z = nd[x]
        s = np.sum(np.log2(z / (1 - z)))
        print('score under true key is ', s)

    print('the results of wrong keys are: ')
    for i in range(t):
        c0l, c0r, c1l, c1r, sk = make_plaintext_structure(nr=nr + 1, diff_in=diff, diff_out=(0x0040, 0), neutral_bits=NBs,
                                                      flag=0)
        # c0l, c0r, c1l, c1r, sk = make_target_diff_samples(n=n, nr=nr+1, diff=diff, flag=0)
        wk = random.randint(0, 2**16 - 1)
        d0l, d0r = sp.dec_one_round((c0l, c0r), wk)
        d1l, d1r = sp.dec_one_round((c1l, c1r), wk)
        x = extrac_bits_to_uint(d0l, d0r, d1l, d1r, bits=bits)
        z = nd[x]
        s = np.sum(np.log2(z / (1 - z)))
        print('score under wrong key is ', s)


if __name__ == '__main__':
    bits_for_ND7 = [12, 11, 10, 9, 8, 7]
    neutral_bits = [20, 21, 22, 14, 15, 23]
    # neutral_bits = [20, 21, 22, [9, 16], [2, 11, 25], 14, 15, [6, 29], 23, 30, 7]
    table_1 = './saved_model/student/0x0040-0x0/0.5416_12_7_nd7.npy'

    test_rank_score(t=10000, n=2**13, nr=7, diff=(0x0040, 0x0), nd=table_1, bits=bits_for_ND7, c=200)
