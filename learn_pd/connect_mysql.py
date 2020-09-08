import pymysql
import pandas as pd
from pandas.util.testing import assert_frame_equal

db = pymysql.connect("localhost", "root", "dnilqa0320", "light_web")
sql = "select * from zong limit 10"

# cursor 方式
cursor = db.cursor()
cursor.execute(sql)
res = cursor.fetchall()  # 返回tuple类型
db.commit()
cursor.close()

# pandas 方式
df = pd.read_sql(sql, db)

db.close()

print(type(res))
print(df)

df2 = pd.DataFrame(list(res)
                   , columns=['id', 'family_name', 'province', 'city', 'years_from', 'years_to', 'other', 'y_tag'])
print(df2)
print(assert_frame_equal(df, df2))  # 比较两个dataframe，None表示相等
df['id'] = df['id'] * 2 - 1
print(assert_frame_equal(df, df2))  # 比较两个dataframe，None表示相等
