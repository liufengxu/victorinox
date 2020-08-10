import pandas as pd


def zhong(dx):
    di = {}
    for i in dx:
        if i in di:
            di[i] += 1
        else:
            di[i] = 1
    m = 0
    n = -1
    for i in di:
        if m < di[i]:
            m = di[i]
            n = i
    return n


def db(x):
    return 2 * x


def percent(dx):
    s = dx.sum()
    return dx / s


def label_percent(dx):
    di = {}
    for i in dx:
        if i in di:
            di[i] += 1
        else:
            di[i] = 1
    return pd.Series([di[i]/len(dx) for i in dx])


def label_percent_groupby(grp):
    # print(grp)
    # print(type(grp))
    return grp.apply(label_percent)


def label_distibute(dx):
    di = {}
    for i in dx:
        if i in di:
            di[i] += 1.0 / len(dx)
        else:
            di[i] = 1.0 / len(dx)
    return pd.Series(sorted(di.items(), key=lambda x: x[1], reverse=True))


def sigma_num(dx):
    m = dx.mean()
    s = dx.std()
    return pd.Series([abs(x-m)/s for x in dx])


# df = pd.DataFrame({'a':[1,1,1,1,1,10],'b':[3,4,5,6,6,6]})
df = pd.DataFrame({'a':[1,1,1,1,1,10],'tag_b':[3,4,5,6,6,6],'tag_c':[2,2,2,2,100,2],'tag_d':[9,9,9,9,9,9]})
for r in df.columns:
    print(r, type(r))
print(df[[i for i in df.columns.values.tolist() if i[:4]=='tag_']])
print(df.idxmax(axis=1).to_dict().items())
# li = []
# for x, y in df.idxmax(axis=1).to_dict().items():
#     li.append(df.loc[x, y])
# print(pd.Series(li))
print(pd.Series([df.loc[x, y] for x, y in df.idxmax(axis=1).to_dict().items()]))
# print(df.apply(zhong))
# print(df.mode())
# print(df['a'].apply(db))
# print(df.apply(percent))
# print(df.apply(label_percent)['a'])
# print(type(df[['a', 'b']]))
# print(df.apply(label_distibute))
# print(df.b.mean())
# print(df.b.std())
# # print(df.a.mode())

# print(df.groupby('a').apply(sum))
# print(df.groupby('a').apply(label_percent))
#
# for name, grp in df.groupby('a'):
#     # print(name)
#     # print(grp.reset_index())
#     print(grp.reset_index().apply(label_percent))
#
# print('-'*32)
# print(df.groupby('a').apply(label_percent_groupby))
# print(type(df.groupby('a').apply(label_percent_groupby)))
# print(df.groupby('a').apply(lambda grp: grp.apply(label_percent)))

df2 = pd.DataFrame({'a':[1,1,None,1,1,10],'tag_b':[3,4,5,6,6,6],'tag_c':[2,2,2,2,100,2],'tag_d':[9,9,9,9,9,9]})
df2.fillna('-')
print(df2)
# df2.a.apply(pd.to_numeric, errors="coerce")
# print(df2)
print(df2.groupby(df2['tag_d']).transform(sigma_num))
df2.loc[df2['tag_c']==2, 'tag_d'] = 0
print(df2)
