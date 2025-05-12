if __name__ == "__main__":

    from tpot import TPOTClassifier
    import pandas as pd
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, learning_curve
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
        roc_curve, auc, ConfusionMatrixDisplay
    )
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import shap
    import lime
    import lime.lime_tabular
    from flaml import AutoML
    import eli5
    from eli5.sklearn import PermutationImportance
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

    config = {"time_budget": 3600, "task": "classification", "metric": "accuracy", "log_file_name": "flaml.log", "split_type": "stratified", "n_splits": 5, "verbose": 2}
    print("[INFO] Configuração utilizada no FLAML: ", config)

    try:
        model = AutoML()
        inicio = time.time()
        model.fit(X_train=X_train, y_train=y_train, **config)
        fim = time.time()
        print("[✓] Treinamento concluído com sucesso.")
        print("[✓] Pipeline final:")
        print(model.model)
    except Exception as e:
        print(f"[ERRO] Erro durante o fit do AuloML_FLAML: {e}")

    # 4. Métricas de Validação Cruzada
    print("\n[INFO] Calculando métricas detalhadas...")
    pipeline_final = model.model

    inicio_predict = time.time()
    y_pred = model.predict(X_test)
    fim_predict = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"[⏱] Tempo total de execução: {fim - inicio:.2f} segundos")
    print(f"[⏱] Tempo total da predição: {fim_predict - inicio_predict:.2f} segundos")

    acc_final = accuracy_score(y_test, y_pred)
    prec_final = precision_score(y_test, y_pred)
    rec_final = recall_score(y_test, y_pred)
    f1_final = f1_score(y_test, y_pred)
    auc_final = roc_auc_score(y_test, y_proba)


    resultado_txt = os.path.join(f"resultado_FLAML_Pipeline.txt")
    with open(resultado_txt, "w", encoding="utf-8") as f:
        f.write("\n== Resultados no Conjunto de Teste Final ==\n")
        f.write(f"Acurácia (teste): {acc_final:.4f}\n")
        f.write(f"Precisão (teste): {prec_final:.4f}\n")
        f.write(f"Recall (teste):   {rec_final:.4f}\n")
        f.write(f"F1-score (teste): {f1_final:.4f}\n")
        f.write(f"AUC ROC (teste):  {auc_final:.4f}\n")

    nome = "FLAML Pipeline Final"

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.title("Curva ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_flaml.png")
    plt.show()

    matriz_conf = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_conf)
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig("confusao_flaml.png")
    plt.show()

    train_sizes, train_scores, val_scores = learning_curve(
    model.model, X_train, y_train,
    cv=5, scoring='f1', n_jobs=1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    shuffle=True, random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Treinamento')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='orange', label='Validação')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std, alpha=0.1, color='orange')
    plt.title("Curva de Aprendizado")
    plt.xlabel("Tamanho do Conjunto de Treinamento")
    plt.ylabel("F1-score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"curva_aprendizado_flaml.png")
    plt.show()

    df_learning = pd.DataFrame({
        'train_size': train_sizes,
        'train_score_mean': train_scores_mean,
        'train_score_std': train_scores_std,
        'val_score_mean': val_scores_mean,
        'val_score_std': val_scores_std
    })
    df_learning.to_csv(f"curva_aprendizado_flaml.csv", index=False)

    shap_config = {}
    os.makedirs("results/shap", exist_ok=True)
    X_subset = pd.DataFrame(X_test[:100], columns=X_test.columns)
    explainer = shap.TreeExplainer(model.model.model)
    shap_values = explainer.shap_values(X_subset)
    
    plt.figure()
    shap.summary_plot(shap_values, X_subset, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig("results/shap/shap_flaml_dotplot.png")
    plt.close()
    print(f"[✓] Gráfico SHAP salvo em: results/shap/shap_flaml.png")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        class_names=["legit", "phish"],
        mode="classification"
    )

    instance = X_test.iloc[0].to_numpy()
    exp = explainer.explain_instance(instance, model.predict_proba, num_features=20)

    os.makedirs("results/lime", exist_ok=True)
    output_path = f"results/lime/lime_flaml_2.html"
    exp.save_to_file(output_path)
    print(f"[✓] Explicação LIME salva em: {output_path}")

    perm = PermutationImportance(model.model.model, random_state=42).fit(X_test, y_test)

    with open("results/eli5/eli5_flaml.html", "w", encoding="utf-8") as f:
        f.write(eli5.format_as_html(eli5.explain_weights(perm, feature_names=X_test.columns.tolist())))

    print("[✓] Explicação ELI5 salva em: results/eli5/eli5_flaml.html")