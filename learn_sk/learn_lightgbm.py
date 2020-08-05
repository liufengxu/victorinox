import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


def train(df_train, df_dev, features):
    X = df_train[features]
    y = df_train['label']
    lgb_train = lgb.Dataset(X, label=y)
    X_dev = df_dev[features]
    y_dev = df_dev['label']
    lgb_dev = lgb.Dataset(X_dev, label=y_dev)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_dev, early_stopping_rounds=5)
    return model


def test(df_test, model, features):
    X_test = df_test[features]
    y_test = df_test['label']
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    print("test auc:", roc_auc_score(y_test, y_pred))
    # print("test gauc:",cal_group_auc(y_test, y_pred, df_test['phone_encrypt']))


def display_feature_importance(model):
    print(pd.DataFrame({
        'column': model.feature_name(),
        'importance': model.feature_importance(),
    }).sort_values(by='importance'))


def load_model_train(df_train, df_dev, df_test, features, model_path):
    model = train(df_train, df_dev, features)
    model.save_model(model_path)
    display_feature_importance(model)
    test(df_test, model, features)


df = pd.read_csv('/Users/liufengxu/Downloads/xxxx_train.csv')
df_c2c_rank_train_tmp = df.fillna(0)
features_short = [
    'fea_1',
    'fea_2',
    'fea_...',
    'fea_n'
]
train_data = df_c2c_rank_train_tmp[df_c2c_rank_train_tmp['dt'] < '2020-04-06']
dev_data = df_c2c_rank_train_tmp[df_c2c_rank_train_tmp['dt'] == '2020-04-06']
test_data = df_c2c_rank_train_tmp[df_c2c_rank_train_tmp['dt'] == '2020-04-07']
load_model_train(train_data, dev_data, test_data, features_short, './base.model')

