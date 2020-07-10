import logging

logging.basicConfig(level=logging.ERROR)

trigram_c2n = {}
trigram_n2c = {}

turn_map = {'|': '/', '\\': '/', '~': '-', '"': '\'', '`': '\''}
for i in range(65, 91):
    turn_map[chr(i)] = chr(i+32)

cha_list = []
num_list = []
num = 0
for i in range(32, 127):
    cha = chr(i)
    if cha in turn_map.keys():
        continue
    cha_list.append(cha)
    num_list.append(num)
    num += 1

trigram_c2n = dict(zip(cha_list, num_list))
trigram_n2c = dict(zip(num_list, cha_list))


def replace_with_dict(s, m):
    for k, v in m.items():
        s = s.replace(k, v)
    return s


def reg_str(s):
    s = s.lower()
    s = replace_with_dict(s, turn_map)
    if len(s) % 2 == 1:
        s += ' '
    return s


def turn_trigram(si):
    left, right = si[0], si[1]
    left_str = oct(trigram_c2n[left])
    right_str = oct(trigram_c2n[right])
    logging.info(left_str)
    logging.info(right_str)
    left_str = left_str[2:]
    if len(left_str) == 1:
        left_str = '0' + left_str
    right_str = right_str[2:]
    if len(right_str) == 1:
        right_str = '0' + right_str
    left_up, left_down = left_str[0], left_str[1]
    right_up, right_down = right_str[0], right_str[1]
    up_str = '0o' + left_up + right_up
    down_str = '0o' + left_down + right_down
    up = trigram_n2c[int(up_str, 8)]
    logging.info(up_str+' '+down_str)
    down = trigram_n2c[int(down_str, 8)]
    return up+down


def trans(s):
    s = reg_str(s)
    out_list = []
    for ii in range(int(len(s)/2)):
        si = s[ii*2:ii*2+2]
        out_list.append(turn_trigram(si))
    return ''.join(out_list)


print(reg_str('aSd+|'))
print(trans('aSd+|'))
print(trans('cc]w)m'))

