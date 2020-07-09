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
    return s


print(reg_str('aSd+-'))


