import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, auc
import numpy as np
import os
import time

print("[INFO] Carregando o dataset...")
df = pd.read_csv("data/dados_consolidados.csv")
print(f"[INFO] Dataset carregado com {df.shape[0]} amostras e {df.shape[1]} colunas.")

X = df.drop("label", axis=1)
y = df["label"]

print("[INFO] Aplicando VarianceThreshold para redução de features...")
selector = VarianceThreshold(threshold=0.01)
X_reduzido_np = selector.fit_transform(X)
X_reduzido = pd.DataFrame(X_reduzido_np, columns=X.columns[selector.get_support()])
print(f"[INFO] Redução concluída: {X_reduzido.shape[1]} features selecionadas.")

X_train, X_test, y_train, y_test = train_test_split(X_reduzido, y, test_size=0.2, random_state=42)
print(f"[INFO] Dados divididos: {X_train.shape[0]} treino / {X_test.shape[0]} teste")

h2o.init()

train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

target = 'label'
features = [col for col in train.columns if col != target]

train[target] = train[target].asfactor()
test[target] = test[target].asfactor()

config = {'max_runtime_secs': 3600, 'seed': 1, 'keep_cross_validation_models': True, 'keep_cross_validation_predictions': True, 'keep_cross_validation_fold_assignment': True}
print("[INFO] Configuração utilizada no H2O AutoML: ", config)

inicio = time.time()
aml = H2OAutoML(nfolds=5, **config)
aml.train(x=features, y=target, training_frame=train)
fim = time.time()

leader_model = aml.leader

print("[INFO] Avaliando no conjunto de teste...")
inicio_predict = time.time()
preds = leader_model.predict(test).as_data_frame()
fim_predict = time.time()
print(preds.columns)
y_pred = preds['predict']

y_proba = preds['p1']
y_test_np = y_test.to_numpy()

acc_final = accuracy_score(y_test_np, y_pred)
prec_final = precision_score(y_test_np, y_pred)
rec_final = recall_score(y_test_np, y_pred)
f1_final = f1_score(y_test_np, y_pred)
auc_final = roc_auc_score(y_test_np, y_proba)

print(f"[⏱] Tempo total de execução: {fim - inicio:.2f} segundos")
print(f"[⏱] Tempo total da predição: {fim_predict - inicio_predict:.2f} segundos")

print("\n== Resultados no Conjunto de Teste Final ==")
print(f"Acurácia (teste): {acc_final:.4f}")
print(f"Precisão (teste): {prec_final:.4f}")
print(f"Recall (teste):   {rec_final:.4f}")
print(f"F1-score (teste): {f1_final:.4f}")
print(f"AUC ROC (teste):  {auc_final:.4f}")

fpr, tpr, _ = roc_curve(y_test_np, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
plt.title("Curva ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_h2o.png")
plt.show()

matriz_conf = confusion_matrix(y_test_np, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_conf)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.savefig("confusao_h2o.png")
plt.show()

lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

print(f"\n[INFO] Modelo líder selecionado: {leader_model.algo.upper()}")
print(leader_model.params)

explain_figs = h2o.explain(leader_model, test)

h2o.shutdown(prompt=False)