import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score


def train(df_train, features, vaild=False):
    X = df_train[features]
    y = df_train['label']
    xgb_train = xgb.DMatrix(X, label=y)
    watchlist = [(xgb_train, 'train'), ]
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.01,
        'max_depth': 5,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 0,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
        'eval_metric': ['auc', 'logloss']
    }
    num_rounds = 100  # 迭代次数
    model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=50)
    return model


def test(df_test, model, features):
    X_test = xgb.DMatrix(df_test[features])
    y_test = df_test['label']
    y_pred = model.predict(X_test)
    print("test auc:", roc_auc_score(y_test, y_pred))
    # print("test gauc:",cal_group_auc(y_test, y_pred, df_test['phone_encrypt']))


def display_feature_importance(model):
    feature_weight = sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)
    print(feature_weight)


def load_model_train(df_train, df_test, features, model_path):
    model = train(df_train, features)
    model.save_model(model_path)
    display_feature_importance(model)
    test(df_test, model, features)


def load_model(df_test, features, model_path):
    model = xgb.Booster(model_file=model_path)
    display_feature_importance(model)
    test(df_test, model, features)


df = pd.read_csv('/Users/liufengxu/Downloads/xxx_train.csv')
df_c2c_rank_train_tmp = df.fillna(0)
features = [
    , 'fea_1'
    , 'fea_2'
    , 'fea_...'
    , 'fea_n'
]
train_data = df_c2c_rank_train_tmp[df_c2c_rank_train_tmp['dt'] < '2020-04-07']
test_data = df_c2c_rank_train_tmp[df_c2c_rank_train_tmp['dt'] == '2020-04-07']
load_model_train(train_data, test_data, features, './base.model')

