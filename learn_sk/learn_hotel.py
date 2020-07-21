import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

df = pd.read_csv('hotel_bookings.csv')
print(df.info())

print(df.children.value_counts())
df.children = df.children.fillna(0)
print(df.children.value_counts())
df = df.drop(['company'], axis=1)
df = df.dropna(axis=0)
print(df.info())

df.loc[df.hotel == 'City Hotel', 'hotel'] = 0
df.loc[df.hotel == 'Resort Hotel', 'hotel'] = 1
print(df.hotel.value_counts())

df.loc[df.arrival_date_month == 'January', 'arrival_date_month'] = 1
df.loc[df.arrival_date_month == 'February', 'arrival_date_month'] = 2
df.loc[df.arrival_date_month == 'March', 'arrival_date_month'] = 3
df.loc[df.arrival_date_month == 'April', 'arrival_date_month'] = 4
df.loc[df.arrival_date_month == 'May', 'arrival_date_month'] = 5
df.loc[df.arrival_date_month == 'June', 'arrival_date_month'] = 6
df.loc[df.arrival_date_month == 'July', 'arrival_date_month'] = 7
df.loc[df.arrival_date_month == 'August', 'arrival_date_month'] = 8
df.loc[df.arrival_date_month == 'September', 'arrival_date_month'] = 9
df.loc[df.arrival_date_month == 'October', 'arrival_date_month'] = 10
df.loc[df.arrival_date_month == 'November', 'arrival_date_month'] = 11
df.loc[df.arrival_date_month == 'December', 'arrival_date_month'] = 12
print(df.arrival_date_month.value_counts())

df.loc[df.meal == 'Undefined', 'meal'] = 0
df.loc[df.meal == 'BB', 'meal'] = 1
df.loc[df.meal == 'FB', 'meal'] = 2
df.loc[df.meal == 'HB', 'meal'] = 3
df.loc[df.meal == 'SC', 'meal'] = 4
print(df.meal.value_counts())

df['customer_type'] = label_encoder.fit_transform(df['customer_type'])
df['assigned_room_type'] = label_encoder.fit_transform(df['assigned_room_type'])
df['deposit_type'] = label_encoder.fit_transform(df['deposit_type'])
df['reservation_status'] = label_encoder.fit_transform(df['reservation_status'])
df['country'] = label_encoder.fit_transform(df['country'])
df['distribution_channel'] = label_encoder.fit_transform(df['distribution_channel'])
df['market_segment'] = label_encoder.fit_transform(df['market_segment'])
df['reserved_room_type'] = label_encoder.fit_transform(df['reserved_room_type'])
df['reservation_status_date'] = label_encoder.fit_transform(df['reservation_status_date'])

y = df['is_canceled'].copy()  # 标签
X = df.drop(['is_canceled', 'reservation_status'], axis=1)  # 特征
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=43)

# 决策树算法
classifier1 = DecisionTreeClassifier(max_depth=100, random_state=43)
classifier1.fit(X_train, y_train)
predictions1 = classifier1.predict(X_test)
print(accuracy_score(y_test, predictions1))

# 随机森林
classifier2 = RandomForestClassifier(n_estimators=600, criterion='entropy', max_depth=5, min_samples_split=1.0,
                                     min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False,
                                     n_jobs=1, random_state=0,
                                     verbose=0)
classifier2.fit(X_train, y_train)
predictions2 = classifier2.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=predictions2))

