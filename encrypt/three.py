three_dict = {
    'A': '111',
    'O': '110',
    'E': '112',
    'B': '101',
    'P': '100',
    'M': '102',
    'F': '121',
    'D': '120',
    'T': '122',
    'N': '011',
    'L': '010',
    'R': '012',
    'Y': '001',
    '_': '000',
    'W': '002',
    'G': '021',
    'K': '020',
    'H': '022',
    'J': '211',
    'Q': '210',
    'X': '212',
    'Z': '201',
    'C': '200',
    'S': '202',
    'I': '221',
    'U': '220',
    'V': '222',
}

three_dict_reverse = {
    '111': 'A',
    '110': 'O',
    '112': 'E',
    '101': 'B',
    '100': 'P',
    '102': 'M',
    '121': 'F',
    '120': 'D',
    '122': 'T',
    '011': 'N',
    '010': 'L',
    '012': 'R',
    '001': 'Y',
    '000': '_',
    '002': 'W',
    '021': 'G',
    '020': 'K',
    '022': 'H',
    '211': 'J',
    '210': 'Q',
    '212': 'X',
    '201': 'Z',
    '200': 'C',
    '202': 'S',
    '221': 'I',
    '220': 'U',
    '222': 'V',
}


def reg_str(s):
    s = s.upper()
    if len(s) % 3 == 2:
        s += '_'
    elif len(s) % 3 == 1:
        s += '__'
    else:
        pass
    return s


def find_in_dict(c):
    if c in three_dict:
        return three_dict[c]
    else:
        return '000'


def turn_element(s):
    a = find_in_dict(s[0])
    b = find_in_dict(s[1])
    c = find_in_dict(s[2])
    ra = three_dict_reverse[a[0]+b[0]+c[0]]
    rb = three_dict_reverse[a[1]+b[1]+c[1]]
    rc = three_dict_reverse[a[2]+b[2]+c[2]]
    return ra + rb + rc


def trans(s):
    s = reg_str(s)
    out_list = []
    for i in range(int(len(s)/3)):
        si = s[i*3:i*3+3]
        out_list.append(turn_element(si))
    print(''.join(out_list))


trans('bob')
