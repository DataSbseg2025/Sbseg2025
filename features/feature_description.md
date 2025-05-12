# 📄 Descrição das Features

Este documento lista e descreve as **131 features** utilizadas para a detecção de ataques phishing a partir de URLs. As features estão organizadas por categoria conforme a natureza das propriedades analisadas.

---

## 🔗 Features Baseadas em Propriedades da URL

| Nome da Feature        | Descrição                                          | Tipo     |
|------------------------|----------------------------------------------------|----------|
| qty_dot_url            | Quantidade de pontos '.'                           | Numérica |
| qty_hyphen_url         | Quantidade de hífens '-'                           | Numérica |
| qty_underline_url      | Quantidade de underlines '_'                       | Numérica |
| qty_slash_url          | Quantidade de barras '/'                           | Numérica |
| qty_questionmark_url   | Quantidade de interrogações '?'                    | Numérica |
| qty_equal_url          | Quantidade de sinais de igualdade '='              | Numérica |
| qty_at_url             | Quantidade de arrobas '@'                          | Numérica |
| qty_and_url            | Quantidade de '&'                                  | Numérica |
| qty_exclamation_url    | Quantidade de '!'                                  | Numérica |
| qty_space_url          | Quantidade de espaços                              | Numérica |
| qty_tilde_url          | Quantidade de tildes '~'                           | Numérica |
| qty_comma_url          | Quantidade de vírgulas ','                         | Numérica |
| qty_plus_url           | Quantidade de '+'                                  | Numérica |
| qty_asterisk_url       | Quantidade de '*'                                  | Numérica |
| qty_hashtag_url        | Quantidade de '#'                                  | Numérica |
| qty_dollar_url         | Quantidade de '$'                                  | Numérica |
| qty_percent_url        | Quantidade de '%'                                  | Numérica |
| qty_pipe_url           | Quantidade de '|'                                  | Numérica |
| qty_Colon_url          | Quantidade de ':'                                  | Numérica |
| qty_Semicolon_url      | Quantidade de ';'                                  | Numérica |
| qty_www_url            | Ocorrências de 'www'                               | Numérica |
| qty_.com_url           | Ocorrências de '.com'                              | Numérica |
| qty_http_url           | Ocorrências de 'http'                              | Numérica |
| qty_//_url             | Ocorrências de '//'                                | Numérica |
| length_url             | Comprimento total da URL                           | Numérica |
| qty_words              | Número de palavras na URL                          | Numérica |
| shorter_words_URL      | Comprimento da menor palavra                       | Numérica |
| longest_words_URL      | Comprimento da maior palavra                       | Numérica |
| Medium_length_URL      | Comprimento médio das palavras                     | Float    |
| Number_word_URL        | Frequência de palavras típicas (admin, login etc.) | Numérica |
| email_in_url           | Indica presença de email                           | Booleano |
| ip                     | Indica presença de IP                              | Booleano |
| https_in_url           | Indica uso de HTTPS                                | Booleano |
| punycode               | Indica presença de punycode                        | Booleano |
| port_number            | Porta explícita na URL                             | Booleano |
| random_string          | Presença de sequência aleatória                    | Booleano |
| protocol_count         | Quantidade de protocolos                           | Numérica |
| suspecious_tld         | TLD considerado suspeito                           | Booleano |
| ip_block_list          | IP em blacklist                                    | Booleano |

---

## 🌐 Features Baseadas em Domínio/Host

| Nome da Feature        | Descrição                                          | Tipo     |
|------------------------|----------------------------------------------------|----------|
| qty_vowels_domain      | Número de vogais no domínio                        | Numérica |
| domain_length          | Comprimento do domínio                             | Numérica |
| domain_inv_ip          | Domínio representado como IP                       | Booleano |
| server_client_domain   | Contém 'server' ou 'client' no domínio             | Booleano |
| random_domains         | Domínio formado por caracteres aleatórios          | Booleano |
| shorter_words_host     | Palavra mais curta no domínio                      | Numérica |
| longest_words_host     | Palavra mais longa no domínio                      | Numérica |
| Medium_length_Host     | Comprimento médio das palavras                     | Float    |
| Brand_in_domain        | Nome de marca presente no domínio                  | Booleano |

---

## 🗂️ Features Baseadas em Diretório

| Nome da Feature         | Descrição                                         | Tipo     |
|-------------------------|---------------------------------------------------|----------|
| directory_length        | Comprimento total do diretório                    | Numérica |
| Brand_in_path           | Nome de marca presente no caminho (path)          | Booleano |
| Medium_length_Path      | Comprimento médio das palavras                    | Float    |
| longest_words_path      | Palavra mais longa no caminho                     | Numérica |
| shorter_words_path      | Palavra mais curta no caminho                     | Numérica |
| Path_extension          | Extensão do arquivo (txt, exe, js)                | Booleano |
| tld_in_path             | TLD presente no caminho                           | Booleano |
| tld_in_subdomain        | TLD presente em subdomínio                        | Booleano |
| qty_subdomains          | Quantidade de subdomínios                         | Numérica |
| Brand_in_subdomain      | Nome de marca presente no subdomínio              | Booleano |
| Domain_Subdomains       | Domínio aparece nos subdomínios                   | Booleano |

---

## 📄 Features Baseadas em Arquivo

| Nome da Feature      | Descrição                            | Tipo     |
|----------------------|---------------------------------------|----------|
| file_length          | Comprimento do nome do arquivo        | Numérica |

---

## 🧾 Features Baseadas em Parâmetros

| Nome da Feature        | Descrição                                | Tipo     |
|------------------------|-------------------------------------------|----------|
| params_length          | Comprimento total dos parâmetros          | Numérica |
| tld_present_params     | TLD presente nos parâmetros               | Booleano |
| qty_params             | Quantidade de parâmetros                  | Numérica |

---

## 📌 Observações

- Todas as features foram extraídas automaticamente das URLs coletadas.
- As features estão normalizadas ou codificadas em formato binário quando aplicável.
- Para detalhes sobre a geração dos dados e pipeline de extração, consulte o diretório `src/` e o arquivo `README.md`.