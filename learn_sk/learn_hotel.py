import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

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
# 为什么用random_state？
# 决策树的生成过程中，往往引入随机数，这是为了得到更好的分类间隔。
# 如使用的经典鸢尾花数据，它本身的特征是连续的（姑且看做连续特征），所以，计算分割点就需要随机。
classifier1.fit(X_train, y_train)
predictions1 = classifier1.predict(X_test)
score = accuracy_score(y_test, predictions1)
print(score)

# 随机森林
classifier2 = RandomForestClassifier(n_estimators=600, criterion='entropy', max_depth=5, min_samples_split=1.0,
                                     min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False,
                                     n_jobs=1, random_state=0,
                                     verbose=0)
classifier2.fit(X_train, y_train)
predictions2 = classifier2.predict(X_test)
score = accuracy_score(y_true=y_test, y_pred=predictions2)
print(score)
test_pred_rfc = classifier2.predict_proba(X_test)[:, 1]  # 预测为1的可能性
fpr_rfc, tpr_rfc, threshold = metrics.roc_curve(y_test, test_pred_rfc)
auc = metrics.auc(fpr_rfc, tpr_rfc)
score = metrics.accuracy_score(y_test, classifier2.predict(X_test))  # 输入真实值和预测值
print([score, auc])  # 准确率、AUC面积

# LR
# 消除警告方法，设置solver参数或者使用下面两行
# import warnings
# warnings.filterwarnings("ignore")
classifier3 = LogisticRegression(solver='liblinear')
classifier3.fit(X_train, y_train)
predictions3 = classifier3.predict(X_test)
score = accuracy_score(y_true=y_test, y_pred=predictions3)
print(score)

# 投票法
prediciton_vote = pd.DataFrame({'Vote': predictions1.astype(int) + predictions2.astype(int) + predictions3.astype(int)})
vote = {0: False, 1: False, 2: True, 3: True}
prediciton_vote['is_canceled'] = prediciton_vote['Vote'].map(vote)
score = accuracy_score(y_test, prediciton_vote['is_canceled'].values)
print(score)

# 随机森林2
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)  # 训练模型
test_pred_rfc = rfc.predict_proba(X_test)[:, 1]  # 预测为1的可能性
fpr_rfc, tpr_rfc, threshold = metrics.roc_curve(y_test, test_pred_rfc)
auc = metrics.auc(fpr_rfc, tpr_rfc)
score = metrics.accuracy_score(y_test, rfc.predict(X_test))  # 输入真实值和预测值
print([score, auc])  # 准确率、AUC面积
precision_rfc, recall_rfc, thresholds = precision_recall_curve(y_test, test_pred_rfc)
pr_rfc = pd.DataFrame({"precision": precision_rfc, "recall": recall_rfc})
prc_rfc = pr_rfc[pr_rfc.precision >= 0.97].recall.max()
print(prc_rfc)  # 精确度≥0.97条件下的最大召回率

importance = rfc.feature_importances_
indices = np.argsort(importance)[::-1]  # np.argsort()返回数值升序排列的索引，[::-1]表示倒序
features = X_train.columns
for f in range(X_train.shape[1]):
    print("%2d) %3d %20s (%.4f)" % (f + 1, indices[f], features[indices[f]], importance[indices[f]]))
# 作图
plt.figure(figsize=(15, 8))
plt.title('Feature importance')
plt.bar(range(X_train.shape[1]), importance[indices], color='blue')
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# xgboost

# GridSearchCV 可以用来寻找最优参数，但运行时间太长 

# param_test1 = {
#     'max_depth': range(3, 10, 2),
#     'min_child_weight': range(1, 6, 2)}
# param_test2 = {
#     'gamma': [i / 10.0 for i in range(0, 5)]}
# param_test3 = {
#     'subsample': [i / 10.0 for i in range(6, 10)],
#     'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
#
# gsearch = GridSearchCV(
#     estimator=XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0,
#                             subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=1,
#                             scale_pos_weight=1, seed=27), param_grid=param_test1, scoring='roc_auc', n_jobs=1,
#     iid=False, cv=5)
# gsearch.fit(X_train, y_train)
# means = gsearch.cv_results_['mean_test_score']
# params = gsearch.cv_results_['params']
# print(means, params)
# # 模型最好的分数、模型最好的参数、模型最好的评估器
# print(gsearch.best_score_, gsearch.best_params_, gsearch.best_estimator_)

model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=0,
                      max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27, silent=True,
                      subsample=0.8)
model.fit(X_train, y_train)  # 训练模型
test_pred_xgb = model.predict_proba(X_test)[:, 1]  # 预测为1的可能性
fpr_xgb, tpr_xgb, threshold = metrics.roc_curve(y_test, test_pred_xgb)
auc = metrics.auc(fpr_xgb, tpr_xgb)
score = metrics.accuracy_score(y_test, model.predict(X_test))  # 输入真实值和预测值
print([score, auc])  # 准确率、AUC面积
precision_xgb, recall_xgb, thresholds = precision_recall_curve(y_test, test_pred_xgb)
pr_xgb = pd.DataFrame({"precision": precision_xgb, "recall": recall_xgb})
prc_xgb = pr_xgb[pr_xgb.precision >= 0.97].recall.max()
print(prc_xgb)  # 精确度≥0.97条件下的最大召回率

importance = model.feature_importances_
indices = np.argsort(importance)[::-1]  # np.argsort()返回数值升序排列的索引，[::-1]表示倒序
features = X_train.columns
for f in range(X_train.shape[1]):
    print("%2d) %3d %20s (%.4f)" % (f + 1, indices[f], features[indices[f]], importance[indices[f]]))
# 作图
plt.figure(figsize=(15, 8))
plt.title('Feature importance')
plt.bar(range(X_train.shape[1]), importance[indices], color='blue')
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

plt.figure()
plt.grid()
plt.title('Roc xgb')
plt.xlabel('FPR')
plt.ylabel('TPR')
# 对比
plt.plot(fpr_xgb, tpr_xgb, label='roc_xgb(AUC=%0.2f)' % auc)
plt.plot(fpr_rfc, tpr_rfc, label='roc_rf(AUC=%0.2f)' % auc)
plt.legend()
plt.show()

# 对比
plt.plot(recall_xgb, precision_xgb, label='xgb_PR')
plt.plot(recall_rfc, precision_rfc, label='rf_PR')
plt.legend()
plt.show()
