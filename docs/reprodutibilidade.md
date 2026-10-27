# Disponibilidade e limites de reprodutibilidade

O material público original é o notebook [`BRIGHTER_PTBR_Experiment.ipynb`](../BRIGHTER_PTBR_Experiment.ipynb), com narrativa, código de leitura de artefatos e resultados incorporados. A árvore do commit [`3ea7669fd4782afccd097e934fc1ab2f4cf2ab81`](https://github.com/Altaeir-13/Brighter-Track-A-PTBR-Experiments/tree/3ea7669fd4782afccd097e934fc1ab2f4cf2ab81), na branch `main`, contém somente esse arquivo. O notebook tem 55 células, sendo 30 Markdown e 25 de código, com 41 outputs. Este documento acompanha uma atualização documental preparada posteriormente; os demais componentes do workspace experimental não são automaticamente disponibilizados por essa atualização.

**O notebook permite inspecionar resultados incorporados, mas o repositório não constitui um pacote autossuficiente para reproduzir o experimento completo.** Treinamento, inferência e avaliação oficial não foram repetidos para preparar esta documentação.

## Como consultar

Abra o notebook no [GitHub](https://github.com/Altaeir-13/Brighter-Track-A-PTBR-Experiments/blob/3ea7669fd4782afccd097e934fc1ab2f4cf2ab81/BRIGHTER_PTBR_Experiment.ipynb) para ler a narrativa e as tabelas salvas. Também é possível obter a árvore histórica para inspeção local:

```bash
git clone https://github.com/Altaeir-13/Brighter-Track-A-PTBR-Experiments.git
cd Brighter-Track-A-PTBR-Experiments
git checkout --detach 3ea7669fd4782afccd097e934fc1ab2f4cf2ab81
```

Essa referência fixa o material original de um arquivo; a documentação posterior pode ser consultada na branch em que foi publicada. Abra o notebook em um visualizador para consultar os outputs preservados, sem executar as células. A execução não é necessária para ler os resultados e não será bem-sucedida em uma cópia que não contenha seus insumos externos.

O notebook não é um roteiro integral de treinamento. Seus metadados declaram `frozen_artifacts_only`, sem carregamento de modelos, treinamento, inferência ou leitura do teste bruto. As opções `RUN_FINAL_REPRODUCTION` e `ALLOW_TEST_ACCESS` estão desabilitadas. A seção opcional de retreinamento depende de código e insumos externos ausentes na árvore original e não foi executada para gerar os resultados incorporados. Alterar essas opções não supre os arquivos faltantes.

## O que existe e o que falta

O workspace preservado dos experimentos contém componentes adicionais. A tabela distingue essa existência local da disponibilidade no commit público original; arquivos mencionados abaixo não devem ser interpretados como links de download ou como material já publicado.

| Componente | Workspace experimental preservado | Commit público original |
|---|---|---|
| Notebook com outputs | Presente | Presente |
| Implementação e interfaces (`src/`, `scripts/`) | Presentes | Ausentes |
| Configurações (`configs/`, 15 YAML) | Presentes | Ausentes |
| Partições externas e internas, manifests e hashes (`metadata/`) | Presentes | Ausentes |
| Dependências (`pyproject.toml`) | Presente, com intervalos de versões | Ausente |
| Lockfile ou imagem completa do ambiente | Não identificado | Ausente |
| Snapshot BRIGHTER e Track A curado (`data/`) | Presentes | Ausentes |
| Pesos dos modelos finais e tokenizadores | Presentes | Ausentes |
| OOF e predições finais congeladas | Presentes | Ausentes |
| Relatórios e resultados JSON/CSV independentes | Presentes | Ausentes; tabelas selecionadas incorporadas no notebook |
| Licença do código | Nenhum arquivo de licença identificado | Nenhum arquivo de licença identificado |

A documentação desta atualização descreve esses componentes e suas limitações; sua publicação não implica publicação do código, dos dados, dos pesos ou dos resultados em arquivos separados.

### Insumos externos do notebook

A primeira célula de código procura `reports/brighter_ptbr/paper/master_results.csv` para localizar a raiz do projeto. Se não o encontrar, lança `FileNotFoundError`. A leitura estática das células identificou os seguintes 14 caminhos literais, presentes no workspace experimental, mas ausentes no commit público original:

```text
reports/brighter_ptbr/paper/master_results.csv
metadata/brighter_ptbr/paper/paper_evidence_manifest.json
metadata/brighter_ptbr/eda/eda_summary.json
metadata/brighter_ptbr/experiments/baseline_summary.json
metadata/brighter_ptbr/experiments/imbalance_loss_summary.json
metadata/brighter_ptbr/experiments/encoder_summary.json
metadata/brighter_ptbr/experiments/threshold_calibration_summary.json
metadata/brighter_ptbr/experiments/ensemble_summary.json
reports/brighter_ptbr/final/final_test_results.json
metadata/brighter_ptbr/final/generalization_gap_summary.json
metadata/brighter_ptbr/final/final_experiment_manifest.json
metadata/brighter_ptbr/final/experiment_state.json
reports/brighter_ptbr/final/generalization_prevalence.csv
reports/brighter_ptbr/final/generalization_threshold_crossing.csv
```

Essa lista descreve dependências literais da leitura dos relatórios; não é uma garantia de suficiência para rotinas opcionais ou para o treinamento completo. A célula de apresentação importa NumPy, pandas, Matplotlib e `IPython.display`. O arquivo local de dependências não fixa todas as versões e não declara IPython/Jupyter explicitamente. Não se oferece aqui uma instalação supostamente validada a partir apenas do notebook.

## Dados e revisões

O manifesto experimental registra a fonte [BRIGHTER Emotion Categories](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories/tree/419566ac8f46f951b51584be36c13e5363008144), configuração `ptbr`, revisão `419566ac8f46f951b51584be36c13e5363008144`. O README do snapshot preservado declara licença **CC-BY 4.0**; essa é a licença registrada na revisão usada. A documentação não redistribui os dados nem atribui essa licença ao código do projeto.

O procedimento local de aquisição resolve e fixa uma revisão antes do download e registra hashes. Usar a revisão mais recente da fonte em vez do SHA acima não garante recuperar os mesmos dados históricos. A disponibilidade atual de todos os insumos externos e sua compatibilidade de instalação não foram verificadas por nova execução.

O payload adquirido continha 2.226/400/4.452 linhas em treino/desenvolvimento/teste. A seleção explícita dos IDs da Track A, separando os registros da Track C, resultou em 2.226/200/2.226. Não se usou deduplicação genérica para essa seleção. Treino e desenvolvimento curados formaram o conjunto de 2.426 exemplos empregado no desenvolvimento experimental; o teste oficial permaneceu reservado até a avaliação final. A sobreposição textual conhecida entre um exemplo de treino e um de teste foi mantida para preservar as partições oficiais.

Os manifests dos componentes finais registram as revisões dos modelos base:

| Componente | Modelo base | Revisão registrada |
|---|---|---|
| CT2 | `melll-uff/bertweetbr` | `7b95f2097a9e37046cca243cc1aeca45b0849fce` |
| CT3 | `neuralmind/bert-large-portuguese-cased` | `aa302f6ea73b759f7df9cad58bd272127b67ec28` |

Informar os nomes dos modelos sem os SHAs não basta para reconstruir uma revisão histórica. Os pesos finais ajustados não estão disponíveis no commit público original; sua ausência também impede reproduzir a inferência final a partir desse material.

## Ambiente e resultados congelados

Os detalhes registrados do ambiente e a distinção entre medidas, declarações e etapas estão em [Ambiente computacional](ambiente.md). Os manifests ajudam a rastrear a execução histórica, mas não substituem um ambiente completamente fixado nem asseguram resultados bit a bit em outro hardware.

O protocolo e as condições dos resultados estão descritos em [Metodologia](metodologia.md) e [Resultados](resultados.md). Os principais valores preservados são EN2 Macro-F1 0,5691 no desenvolvimento aninhado, EN2 0,2852 no teste oficial e CT2 0,1569 no teste. A seleção anterior de modelos e perdas não foi integralmente aninhada; a documentação não apresenta a busca completa como tal.

O estado experimental final registra `TEST_EVALUATED`, espaço de busca fechado e proibição de novos ajustes. Um incidente de geração de relatório após a inferência foi recuperado em modo de relatório, sem nova inferência ou alteração de parâmetros. Essa recuperação não corresponde a uma segunda avaliação experimental.

## O que seria necessário para ampliar a reprodução

Qualquer ampliação da distribuição deve passar por revisão e autorização dos responsáveis, com inventário concreto dos arquivos que serão publicados. As etapas pendentes incluem:

1. Disponibilizar o código e as configurações efetivamente usados, com licença definida e revisão imutável.
2. Publicar uma especificação compatível de dependências e documentar separadamente diferenças de versão entre aquisição, treinamento e geração do notebook.
3. Fornecer instruções de obtenção dos dados na revisão fixa e os manifests da seleção Track A, respeitando a licença e evitando publicar dados privados.
4. Disponibilizar os insumos de relatório necessários ao notebook, folds e parâmetros congelados, após revisão de conteúdo e licenças.
5. Definir uma distribuição apropriada para pesos e artefatos volumosos, se autorizada, sem incorporá-los indiscriminadamente ao Git.
6. Verificar o pacote completo em uma área separada, preservando o teste oficial encerrado e os artefatos originais.

Esses passos descrevem lacunas; não foram executados nesta atualização. Enquanto os insumos necessários permanecerem ausentes, a disponibilidade pública deve ser descrita como notebook de consulta e documentação científica, e não como reprodução completa verificada.
