# Uma Avaliação de AutoML e Técnicas XAI para Detecção Transparente de Ataques Phishing

Este repositório acompanha o artigo _"Uma Avaliação de AutoML e Técnicas XAI para Detecção Transparente de Ataques Phishing"_, e tem como objetivo compartilhar a base de dados utilizada, os códigos dos experimentos realizados, bem como gráficos e explicações geradas pelas técnicas de interpretabilidade aplicadas.

---

## Estrutura do Repositório

```
.
├── data/
│   └── phishing_dataset.zip            # Conjunto de dados rotulado
├── features/
│   └── feature_description.md          # Descrição das features utilizadas
├── plots/
│   ├── shap_summary_plot.png           # Gráfico resumo do SHAP
│   ├── lime_example_.html              # Exemplo de explicação LIME
│   └── confusion_matrices/             # Matrizes de confusão dos modelos
│   └── learning_curve/                 # Curva de aprendizado
│   └── curve_roc/                      # Curva ROC
├── src/
│   ├── automl/
│   │   ├── run_tpot.py                 # Execução com TPOT
│   │   ├── run_flaml.py                # Execução com FLAML
│   │   ├── run_autosklearn.py          # Execução com Auto-sklearn
│   │   └── run_h2o.py                  # Execução com H2O AutoML
│   ├── xai/
│   │   ├── run_shap.py                 # Geração das explicações com SHAP
│   │   └── run_lime.py                 # Geração das explicações com LIME
```

---

## Técnicas Avaliadas

### Arquiteturas AutoML

- [TPOT](https://epistasislab.github.io/tpot/)
- [FLAML](https://microsoft.github.io/FLAML/)
- [Auto-sklearn](https://automl.github.io/auto-sklearn/)
- [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

### Técnicas de XAI

- [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap)
- [LIME (Local Interpretable Model-agnostic Explanations)](https://github.com/marcotcr/lime)

---

## Metodologia

O objetivo deste estudo foi investigar a eficácia de diferentes pipelines AutoML na detecção de ataques phishing e avaliar a interpretabilidade dos modelos por meio de técnicas de XAI.

Foram aplicadas:
- Validação cruzada com k-fold = 5
- Geração de curvas de aprendizado
- Matrizes de confusão
- Curvas ROC
- Visualizações SHAP e LIME

---

## Métricas Utilizadas

- Acurácia
- Precisão
- Recall
- F1-Score
- Área sob a curva ROC
- Visualizações XAI para avaliação qualitativa

---

## Base de Dados

O dataset está disponível em `data/phishing_dataset.csv` e contém instâncias rotuladas como _phishing_ (1) ou _legítimo_ (0), com um total de **131** features extraídas exclusivamente das URLs, como:

- Comprimento da URL
- Presença de caracteres especiais
- Informações de WHOIS
- Estrutura do domínio
- Análise de redirecionamentos

Uma descrição completa está disponível em [`features/feature_description.md`](features/feature_description.md).
