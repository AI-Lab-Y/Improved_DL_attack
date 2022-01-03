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
# neutral_bits is used to form the plaintext structure
def make_homogeneous_set(diff=(0x211, 0xa04), neutral_bits=None):
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
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r


def make_target_diff_samples_v2(nr=11, diff_in=(0x211, 0xa04), diff_out=(0x40, 0), nr_help=3, flag=1, neutral_bits=None):
    if nr == 12:
        while 1:
            key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
            ks = sp.expand_key(key, nr)
            ks_help = sp.expand_key(key, nr_help)
            if (ks[2][0] >> 12) & 1 != (ks[2][0] >> 11) & 1:
                break
    else:
        key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(key, nr)
        ks_help = sp.expand_key(key, nr_help)
    print('the number of neutral bit sets is ', len(neutral_bits))

    num = 0
    while 1:
        p0l, p0r, p1l, p1r = make_homogeneous_set(diff=diff_in, neutral_bits=neutral_bits)
        p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
        p1l, p1r = sp.dec_one_round((p1l, p1r), 0)

        num = num + 1
        # print('num is ', num)

        c0l_help, c0r_help = sp.encrypt((p0l, p0r), ks_help)
        c1l_help, c1r_help = sp.encrypt((p1l, p1r), ks_help)
        if np.sum((c0l_help ^ c1l_help ^ diff_out[0]) == 0) > 0 and np.sum((c0r_help ^ c1r_help ^ diff_out[1]) == 0) >0:
            print('num is ', num)
            print(np.sum((c0l_help ^ c1l_help ^ diff_out[0]) == 0), ' ', np.sum((c0r_help ^ c1r_help ^ diff_out[1]) == 0))

        if flag == 1:
            if np.sum((c0l_help ^ c1l_help ^ diff_out[0]) == 0) == 2**len(neutral_bits):
                if np.sum((c0r_help ^ c1r_help ^ diff_out[1]) == 0) == 2 ** len(neutral_bits):
                    c0l, c0r = sp.encrypt((p0l, p0r), ks)
                    c1l, c1r = sp.encrypt((p1l, p1r), ks)
                    print('num is ', num)
                    return c0l, c0r, c1l, c1r, ks[nr - 1][0]
        else:
            if np.sum((c0l_help ^ c1l_help ^ diff_out[0]) == 0) != 2**len(neutral_bits):
                c0l, c0r = sp.encrypt((p0l, p0r), ks)
                c1l, c1r = sp.encrypt((p1l, p1r), ks)
                print('num is ', num)
                return c0l, c0r, c1l, c1r, ks[nr - 1][0]
            if np.sum((c0r_help ^ c1r_help ^ diff_out[1]) == 0) != 2 ** len(neutral_bits):
                c0l, c0r = sp.encrypt((p0l, p0r), ks)
                c1l, c1r = sp.encrypt((p1l, p1r), ks)
                print('num is ', num)
                return c0l, c0r, c1l, c1r, ks[nr - 1][0]


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


def naive_attack(t=100, nr=7, diff=(0x40, 0x0), nd=None, bits=None, flag=1, n=2**10, c=30):
    nd1 = np.load(nd[0])
    nd2 = np.load(nd[1])
    score = np.zeros(2**16)
    res = np.zeros((t, 2))

    acc = 0
    for i in range(t):
        # print('cur t is ', i)
        c0l, c0r, c1l, c1r, sk = make_target_diff_samples(n=n, nr=nr+1, diff=diff, flag=flag)

        local_flag = 0
        for kg in range(2**6):
            d0l, d0r = sp.dec_one_round((c0l, c0r), kg)
            d1l, d1r = sp.dec_one_round((c1l, c1r), kg)
            x = extrac_bits_to_uint(d0l, d0r, d1l, d1r, bits=bits[0])
            z = nd1[x]
            s = np.sum(np.log2(z / (1 - z)))
            score[kg] = s

            if kg == (sk & 0x3f):
                v1 = s

            if s > c:
                local_flag = 1
        if local_flag == flag:
            acc = acc + 1

        kg_L = sk & 0x3f
        for kg_H in range(1, 2 ** 9):
            kg = (kg_H << 6) + kg_L
            d0l, d0r = sp.dec_one_round((c0l, c0r), kg)
            d1l, d1r = sp.dec_one_round((c1l, c1r), kg)
            x = extrac_bits_to_uint(d0l, d0r, d1l, d1r, bits=bits[1])
            z = nd2[x]
            s = np.sum(np.log2(z / (1 - z)))
            score[kg] = s

            if kg == (sk & 0x7fff):
                v2 = s

        res[i][0], res[i][1] = v1, v2
        # print('the scores of true keys are ', (score[sk & 0x3f], score[sk & 0x7fff]))

    print('recognition accuracy is ', acc / t)
    # np.save('./true_key_rank_score/' + str(flag) + '_key_rank_score_res.npy', res)


def naive_attack_v2(t=100, nr=7, nr_help=3, diff_in=(0x211, 0xa04), diff_out=(0x40, 0x0), nd=None, bits=None, flag=1, NBs=None):
    nd1 = np.load(nd[0])
    nd2 = np.load(nd[1])
    score = np.zeros(2**16)
    res = np.zeros((t, 2))
    for i in range(t):
        print('cur t is ', i)
        c0l, c0r, c1l, c1r, sk = make_target_diff_samples_v2(nr=nr+nr_help+1, nr_help=nr_help, diff_in=diff_in,
                                                             diff_out=diff_out, flag=flag, neutral_bits=NBs)
        for kg in range(2**6):
            d0l, d0r = sp.dec_one_round((c0l, c0r), kg)
            d1l, d1r = sp.dec_one_round((c1l, c1r), kg)
            x = extrac_bits_to_uint(d0l, d0r, d1l, d1r, bits=bits[0])
            z = nd1[x]
            s = np.sum(np.log2(z / (1 - z)))
            score[kg] = s

            if kg == (sk & 0x3f):
                v1 = s

        # sur_key = [hex(kg ^ (sk & 0x3f)) for kg in range(2**6) if score[kg] > 20]
        # print('surviving keys are ', sur_key)

        kg_L = sk & 0x3f
        for kg_H in range(1, 2 ** 9):
            kg = (kg_H << 6) + kg_L
            d0l, d0r = sp.dec_one_round((c0l, c0r), kg)
            d1l, d1r = sp.dec_one_round((c1l, c1r), kg)
            x = extrac_bits_to_uint(d0l, d0r, d1l, d1r, bits=bits[1])
            z = nd2[x]
            s = np.sum(np.log2(z / (1 - z)))
            score[kg] = s

            if kg == (sk & 0x7fff):
                v2 = s
        # sur_key = [hex(kg ^ (sk & 0x7fff)) for kg in range(2 ** 15) if score[kg] > c]
        # print('the number of surviving subkeys is ', len(sur_key))
        # print('surviving keys are ', sur_key)
        # if score[sk & 0x7fff] > c:
        #     acc = acc + 1

        res[i][0], res[i][1] = v1, v2
        print('the scores of true keys are ', (score[sk & 0x3f], score[sk & 0x7fff]))

    # np.save('./key_rank_score/' + str(flag) + '_key_rank_score_res.npy', res)
    np.save('./' + str(flag) + '_key_rank_score_res.npy', res)


if __name__ == '__main__':
    # for 11-round attack
    bits_for_ND7_1 = [12, 11, 10, 9, 8, 7]
    bits_for_ND7_2 = [14, 13, 12, 11, 5, 4]
    bits = [bits_for_ND7_1, bits_for_ND7_2]
    table_1_1 = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/0.5416_12_7_nd7.npy'
    table_1_2 = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/0.5722_14_11_5_4_nd7.npy'
    tables = [table_1_1, table_1_2]
    # neutral_bits = [20, 21, 22, [9, 16], [2, 11, 25], 14, 15, [6, 29], 23, 30]
    neutral_bits = [20, 21, 22, [9, 16], [2, 11, 25], 14, 15, [6, 29]]
    naive_attack_v2(t=1000, nr=7, nr_help=3, diff_in=(0x211, 0xa04), diff_out=(0x40, 0x0), nd=tables,
                    bits=bits, flag=1, NBs=neutral_bits)

    naive_attack_v2(t=1000, nr=7, nr_help=3, diff_in=(0x211, 0xa04), diff_out=(0x40, 0x0), nd=tables,
                    bits=bits, flag=0, NBs=neutral_bits)

    # test the recognition accuracy of new processing strategy
    # naive_attack(t=1000, nr=7, diff=(0x40, 0x0), nd=tables, bits=bits, flag=1, n=2 ** 6, c=7)
    # naive_attack(t=1000, nr=7, diff=(0x40, 0x0), nd=tables, bits=bits, flag=0, n=2 ** 6, c=7)
    #
    # naive_attack(t=1000, nr=7, diff=(0x40, 0x0), nd=tables, bits=bits, flag=1, n=2 ** 8, c=10)
    # naive_attack(t=1000, nr=7, diff=(0x40, 0x0), nd=tables, bits=bits, flag=0, n=2 ** 8, c=10)
    #
    # naive_attack(t=1000, nr=7, diff=(0x40, 0x0), nd=tables, bits=bits, flag=1, n=2 ** 10, c=30)
    # naive_attack(t=1000, nr=7, diff=(0x40, 0x0), nd=tables, bits=bits, flag=0, n=2 ** 10, c=30)

    # for 12-round attack
    # bits_for_ND7_1 = [12, 11, 10, 9, 8, 7]
    # bits_for_ND7_2 = [14, 13, 12, 11, 5, 4]
    # bits = [bits_for_ND7_1, bits_for_ND7_2]
    # table_1_1 = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/0.5416_12_7_nd7.npy'
    # table_1_2 = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/0.5722_14_11_5_4_nd7.npy'
    # tables = [table_1_1, table_1_2]
    # neutral_bits = [22, 20, 13, [12, 19], [14, 21], [6, 29], 30, [0, 8, 31]]
    # naive_attack_v2(t=1000, nr=7, nr_help=4, diff_in=(0x8060, 0x4101), diff_out=(0x40, 0x0), nd=tables,
    #                 bits=bits, flag=1, NBs=neutral_bits)
    #
    # naive_attack_v2(t=1000, nr=7, nr_help=4, diff_in=(0x8060, 0x4101), diff_out=(0x40, 0x0), nd=tables,
    #                 bits=bits, flag=0, NBs=neutral_bits)
