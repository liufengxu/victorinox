import faiss
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data_path = "/Users/liufengxu/Downloads/a.csv"
df_ori = pd.read_csv(data_path)
df = df_ori.drop(['item'], axis=1)
ids = df_ori['item'].values
print(ids)

print(df.shape)
cols = df.columns.values.tolist()
t = df[0:1].copy()
for col in cols:
    # t.loc[0, col] = df.loc[:3, col].mean()
    t.loc[0, col] = df.loc[1496:1498, col].mean()
print(t)

# r = [t, df]
# tt = pd.concat(r)
# print(tt)
# tt = tt.reset_index(drop=True)
# print(tt.head())
# test_df_similarity = cosine_similarity(tt)
# print(test_df_similarity.shape)
# rs = test_df_similarity[0:1].copy()
# rs[0, 0] = 0
# print(rs)
# print(rs.max(axis=1), np.argmax(rs, axis=1))
# print(rs[0, 4083], rs[0, 4070], rs[0, 3944])
# exit()

test_df = df.head()
print(test_df)
test_df_similarity = cosine_similarity(test_df)
print(test_df_similarity)

nb, d = df.shape
print(nb, d)

xb = df.values
xb = xb.astype('float32')
print(xb.dtype)
xb = np.ascontiguousarray(xb)
# index = faiss.IndexFlatL2(d)  # 建立索引
index = faiss.IndexFlatIP(d)  # 建立索引
print(index.is_trained)  # 输出true
# index.add(xb)  # 索引中添加向量
# print(index.ntotal)
# ids = np.arange(nb) + 10000
index2 = faiss.IndexIDMap(index)
index2.add_with_ids(xb, ids)
print(index2.ntotal)

# xq = df[0:4].values
# xq = t.values
query_path = "/Users/liufengxu/Downloads/b.csv"
df_q_ori = pd.read_csv(query_path)
# query_path = "/Users/liufengxu/Downloads/fetch.out"
#mdf_q_ori = pd.read_csv(query_path, sep='\t')
df_q = df_q_ori.drop(['item', 'next_item', 'dt'], axis=1)
label = df_q_ori['next_item']
xq = df_q.values
xq = xq.astype('float32')
xq = np.ascontiguousarray(xq)
k = 10  # 返回每个查询向量的近邻个数
# D, I = index.search(xb[:5], k)
# # 检索check
# print(I)
# print(D)
D, I = index2.search(xq, k)
print('result:')
# xq检索结果
# print(I[:5])
# 前五个检索结果展示
# print(I[-5:])
# 最后五个检索结果展示

print(I.shape)
print(label.shape)


def recall_rate(pred_top, true_label, top_n=6):
    all_cnt = len(true_label)
    shot = 0
    for i in range(len(pred_top)):
        pred = pred_top[i][:top_n]
        ytrue = true_label[i]
        if ytrue in pred:
            shot += 1
    return all_cnt, shot, all_cnt, shot / all_cnt


print(recall_rate(I, label))

