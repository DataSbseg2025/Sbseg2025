# Uma Avaliação de AutoML e Técnicas XAI para Detecção Transparente de Ataques Phishing

Este repositório acompanha o artigo _"Uma Avaliação de AutoML e Técnicas XAI para Detecção Transparente de Ataques Phishing"_, e tem como objetivo compartilhar a base de dados utilizada, os códigos dos experimentos realizados, bem como gráficos e explicações geradas pelas técnicas de interpretabilidade aplicadas.

---

## Estrutura do Repositório

- `src/`: Código-fonte para pré-processamento, execução dos experimentos AutoML e aplicação das técnicas de XAI.
- `data/`: Conjunto de dados utilizados no estudo (`phishing_dataset.zip`).
- `plots/`: Gráficos gerados durante os experimentos, incluindo curvas ROC, curvas de aprendizado, explicações SHAP e LIME, e matrizes de confusão.
- `README.md`: Visão geral do projeto e instruções de uso.

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

O dataset está disponível em `data/phishing_dataset.csv` e contém instâncias rotuladas como _phishing_ (1) ou _legítimo_ (0), com um total de **131** features extraídas exclusivamente das URLs, das quais, foram selecionadas 67 apos o pre-processamento. Essas fortam:

| Nº | Nome                         | Descrição                                                                                                 | Formato  |
|----|------------------------------|-----------------------------------------------------------------------------------------------------------|----------|
| 1  | qty_dot_url                  | Quantidade de "." na URL                                                                                  | Integer  |
| 2  | qty_hyphen_url               | Quantidade de "-" na URL                                                                                  | Integer  |
| 3  | qty_underlin_url             | Quantidade de "_" na URL                                                                                  | Integer  |
| 4  | qty_slash_url                | Quantidade de "/" na URL                                                                                  | Integer  |
| 5  | qty_questionmark_url         | Quantidade de "?" na URL                                                                                  | Integer  |
| 6  | qty_equal_url                | Quantidade de "=" na URL                                                                                  | Integer  |
| 7  | qty_and_url                  | Quantidade de "&" na URL                                                                                  | Integer  |
| 8  | qty_tilde_url                | Quantidade de "~" na URL                                                                                  | Integer  |
| 9  | qty_comma_url                | Quantidade de "," na URL                                                                                  | Integer  |
| 10 | qty_plus_url                 | Quantidade de "+" na URL                                                                                  | Integer  |
| 11 | qty_asterisk_url             | Quantidade de "*" na URL                                                                                  | Integer  |
| 12 | qty_percent_url              | Quantidade de "%" na URL                                                                                  | Integer  |
| 13 | qty_colon_url                | Quantidade de ":" na URL                                                                                  | Integer  |
| 14 | qty_semicolon_url            | Quantidade de ";" na URL                                                                                  | Integer  |
| 15 | qty_www_url                  | Quantidade de "www" na URL                                                                                | Integer  |
| 16 | qty_com_url                  | Quantidade de ".com" na URL                                                                               | Integer  |
| 17 | url_length                   | Quantidade de caracteres na URL                                                                           | Integer  |
| 18 | tld_length                   | Quantidade de caracteres no TLD                                                                           | Integer  |
| 19 | is_https_scheme              | Utiliza protocolo HTTPS                                                                                   | Boolean  |
| 20 | qty_words                    | Quantidade de palavras na URL                                                                             | Integer  |
| 21 | qty_repeated_characters      | Quantidade de caracteres repetidos na URL                                                                 | Integer  |
| 22 | shortest_word_size           | Tamanho da palavra mais curta da URL                                                                      | Integer  |
| 23 | longest_word_size            | Tamanho da palavra mais longa da URL                                                                      | Integer  |
| 24 | average_word_length          | Comprimento médio das palavras na URL                                                                     | Float    |
| 25 | keyword_occurrences          | Quantidade de ocorrências de palavras-chave (wp, login, admin etc.)                                       | Integer  |
| 26 | has_random_sequence          | Possui sequência aleatória                                                                                | Boolean  |
| 27 | qty_dot_domain               | Quantidade de "." no domínio                                                                              | Integer  |
| 28 | qty_hyphen_domain            | Quantidade de "-" no domínio                                                                              | Integer  |
| 29 | qty_vowels_domain            | Quantidade de vogais no domínio                                                                           | Integer  |
| 30 | length_domain                | Quantidade de caracteres no domínio                                                                       | Integer  |
| 31 | random_domains               | O domínio é formado por sequência aleatória                                                               | Boolean  |
| 32 | shorter_words_host           | Tamanho da palavra mais curta do host                                                                     | Integer  |
| 33 | longest_words_host           | Tamanho da palavra mais longa do host                                                                     | Integer  |
| 34 | medium_length_hHost          | Comprimento médio das palavras do host                                                                    | Float    |
| 35 | qty_dot_path                 | Quantidade de "." no caminho da URL                                                                       | Integer  |
| 36 | qty_hyphen_path              | Quantidade de "-" no caminho da URL                                                                       | Integer  |
| 37 | qty_underlin_path            | Quantidade de "_" no caminho da URL                                                                       | Integer  |
| 38 | qty_slash_path               | Quantidade de "/" no caminho da URL                                                                       | Integer  |
| 39 | qty_equal_path               | Quantidade de "=" no caminho da URL                                                                       | Integer  |
| 40 | qty_tilde_path               | Quantidade de "~" no caminho da URL                                                                       | Integer  |
| 41 | qty_plus_path                | Quantidade de "+" no caminho da URL                                                                       | Integer  |
| 42 | qty_percent_path             | Quantidade de "%" no caminho da URL                                                                       | Integer  |
| 43 | directory_length             | Comprimento total do diretório na URL                                                                     | Integer  |
| 44 | medium_length_path           | Comprimento médio dos caracteres no caminho                                                               | Float    |
| 45 | longest_words_path           | Comprimento da palavra mais longa no caminho                                                              | Integer  |
| 46 | shorter_words_path           | Comprimento da palavra mais curta no caminho                                                              | Integer  |
| 47 | tld_in_subdomain             | TLD presente nos subdomínios                                                                              | Boolean  |
| 48 | qty_subdomains               | Quantidade de subdomínios                                                                                 | Integer  |
| 49 | qty_dot_file                 | Quantidade de "." no nome do arquivo                                                                      | Integer  |
| 50 | qty_hyphen_file              | Quantidade de "-" no nome do arquivo                                                                      | Integer  |
| 51 | qty_underlin_file            | Quantidade de "_" no nome do arquivo                                                                      | Integer  |
| 52 | qty_plus_file                | Quantidade de "+" no nome do arquivo                                                                      | Integer  |
| 53 | qty_percent_file             | Quantidade de "%" no nome do arquivo                                                                      | Integer  |
| 54 | file_length                  | Comprimento do nome do arquivo                                                                            | Integer  |
| 55 | qty_dot_query                | Quantidade de "." na query string                                                                         | Integer  |
| 56 | qty_hyphen_query             | Quantidade de "-" na query string                                                                         | Integer  |
| 57 | qty_underlin_query           | Quantidade de "_" na query string                                                                         | Integer  |
| 58 | qty_slash_query              | Quantidade de "/" na query string                                                                         | Integer  |
| 59 | qty_equal_query              | Quantidade de "=" na query string                                                                         | Integer  |
| 60 | qty_and_query                | Quantidade de "&" na query string                                                                         | Integer  |
| 61 | qty_comma_query              | Quantidade de "," na query string                                                                         | Integer  |
| 62 | qty_plus_query               | Quantidade de "+" na query string                                                                         | Integer  |
| 63 | qty_asterisk_query           | Quantidade de "*" na query string                                                                         | Integer  |
| 64 | qty_percent_query            | Quantidade de "%" na query string                                                                         | Integer  |
| 65 | query_length                 | Comprimento total da query string                                                                         | Integer  |
| 66 | qty_params                   | Quantidade de parâmetros                                                                                  | Integer  |
| 67 | params_length                | Comprimento total dos parâmetros                                                                          | Integer  |
