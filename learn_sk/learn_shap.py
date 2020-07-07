import shap
import xgboost

shap.initjs()
X, y = shap.datasets.boston()
print(type(X))
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值

# shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)
shap.summary_plot(shap_values, X)

y_base = explainer.expected_value
print(y_base)

pred = model.predict(xgboost.DMatrix(X))
print(pred.mean())
