import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""

    print('*' * 50)
    labels = labels.tolist()
    # preds=preds.reset_index()
    user_id_list = user_id_list.tolist()
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc


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
    print("test gauc:",cal_group_auc(y_test, y_pred, df_test['guid']))


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

