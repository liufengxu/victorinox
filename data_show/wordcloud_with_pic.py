import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

# 保证汉字正常展示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

name = 'draw.xlsx'
data_path = "/Users/liufengxu/Downloads/{}".format(name)
df = pd.read_excel(data_path)
df['uv'] = df['uv']  # // 100
token = []

for i in range(df.shape[0]):
    query = str(df[i:i+1]['query'].values[0])
    cnt = int(df[i:i+1]['uv'].values[0])
    # print(query, type(query), cnt, type(cnt))
    tmp = [query] * cnt
    token += tmp
print(type(token))
wl_split = ' '.join(token)
mask = np.array(Image.open("/Users/liufengxu/Downloads/car.jpeg"))
mywc = WordCloud(mask=mask, font_path='/Users/liufengxu/Public/MSYH.TTC',
                 collocations=False, mode='RGBA', background_color=None).generate(wl_split)

plt.imshow(mywc)
plt.axis("off")
plt.show()

