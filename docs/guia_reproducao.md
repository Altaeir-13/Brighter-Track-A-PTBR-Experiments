# Guia de reprodução metodológica

Este guia distingue a consulta e regeneração de resultados **já registrados** de um novo treinamento de desenvolvimento. Os códigos e configurações históricos permanecem preservados; os dois auxiliares de isolamento abaixo foram acrescentados para a distribuição. Nenhum passo autoriza reavaliar o teste oficial encerrado ou selecionar parâmetros a partir dele.

## 1. Obter e conferir a distribuição

```bash
git clone https://github.com/Altaeir-13/Brighter-Track-A-PTBR-Experiments.git
cd Brighter-Track-A-PTBR-Experiments
git rev-parse HEAD
```

Registre o commit obtido e confira os manifests/hashes disponibilizados com esta distribuição. A publicação original do notebook está no commit `3ea7669fd4782afccd097e934fc1ab2f4cf2ab81`; a ampliação do pipeline possui commit posterior. O notebook histórico deve ser consultado com seus outputs preservados. Não é necessário executá-lo para ler suas tabelas.

Consulte [Ambiente](ambiente.md), [Disponibilidade e limites](reprodutibilidade.md) e [Metodologia](metodologia.md). O ambiente histórico registrado não é um lockfile completo. As instruções de instalação reconstruída e seu relatório de verificação devem ser usados como tal, sem apresentar novas versões como as versões dos treinos históricos. A biblioteca original exige Python >=3.11 e usa restrições de dependências em `pyproject.toml`; siga a versão de Python e as dependências efetivamente verificadas descritas na documentação do ambiente. Um ambiente CPU valida imports, funções e relatórios, mas os entrypoints originais de treinamento/inferência Transformer exigem CUDA.

Use os comandos completos e a sequência de instalação em [Instalação](instalacao.md). Depois de ativar um ambiente compatível, a interface padrão do projeto é:

```bash
python -m pip install -e ".[test]"
```

Não usar um `pip freeze` reconstruído como prova do ambiente histórico. A leitura do notebook pode exigir IPython/Jupyter além do projeto original; a instalação desses componentes é separada da execução do pipeline de treinamento.

## 2. Regerar tabelas a partir dos resultados congelados

Para uma reprodução inicial sem carregar modelos, dados brutos ou predições por instância, use uma pasta nova fora do checkout:

```bash
python scripts/rebuild_frozen_reports.py --destination ../brighter-relatorios-reproduzidos
```

O auxiliar copia os relatórios/metadados publicados para o destino, exclui arquivos de autorização e chama `brighter.paper_evidence.generate_paper_package` nessa cópia. Ele gera `reports/brighter_ptbr/paper/master_results.csv`, tabelas, figuras de síntese, manifest de evidências e `report_rebuild_receipt.json`. O destino deve não existir; o pai deve existir. A fonte e suas evidências não são sobrescritas.

Esse procedimento reconstrói apresentações a partir de agregados já calculados. Não executa inferência, treinamento, avaliação oficial ou busca de hiperparâmetros. Algumas linhas históricas usam média de F1 entre folds, enquanto o artigo apresenta F1 das OOF concatenadas em determinadas comparações; conserve essas definições. A igualdade de todas as tabelas, bytes ou figuras não é presumida: compare o conteúdo numérico e os campos de protocolo e consulte o recibo gerado. Diferenças de redação dos relatórios históricos não substituem as ressalvas do artigo final sobre a busca anterior não totalmente aninhada.

Para recalcular Macro-F1 e Micro-F1 das decisões externas de desenvolvimento, sem selecionar novos parâmetros e sem abrir qualquer artefato de teste:

```bash
python scripts/verify_frozen_development.py --output ../metricas-oof-reproduzidas.json
```

O destino deve ser novo e externo ao checkout. CT1–CT3 e EN1/EN2 são conferidos nas mesmas 2.426 OOF. Essa etapa não aplica a regra final de implantação às OOF: lê as decisões internas/externas já preservadas, mantendo a condição da avaliação aninhada.

O notebook lê arquivos relativos de `reports/` e `metadata/`. A disponibilidade desses insumos deve ser conferida na distribuição efetivamente obtida; a presença do notebook por si só nunca demonstra suficiência do pacote.

## 3. Preparar uma nova árvore para treinar

```bash
python scripts/prepare_reproduction_workspace.py --destination ../brighter-execucao-nova
```

O auxiliar copia código, testes, configurações, licença/avisos disponíveis, especificações de dependências reconstruídas, revisões de modelos e atribuições de folds para um destino novo fora do checkout. Também conserva os manifests históricos de dados/folds/congelamento como referências em `metadata/publication/reference_*.json`, separadas do estado operacional. Não copia dados, pesos, predições, relatórios históricos, estado final ou autorização do teste. Um recibo registra os hashes das cópias. Não altera arquivos-fonte nem inicia uma execução experimental.

Os scripts históricos calculam a raiz a partir do próprio arquivo e não têm um parâmetro geral `--output-root`. Portanto, execute **as cópias dos scripts dentro dessa árvore nova**. Mudar somente o diretório atual, mantendo o caminho do script original, não isola suas saídas. Essa separação também impede confundir o reaproveitamento de caches históricos com um novo treinamento.

```bash
cd ../brighter-execucao-nova
python -m pip install -e ".[test]"
```

Use um ambiente dedicado e registre suas versões e hardware. Os resultados novos são resultados de uma reprodução, não substitutos dos resultados publicados.

## 4. Obter o BRIGHTER na revisão correta

Fonte: [BRIGHTER Emotion Categories na revisão experimental](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories/tree/419566ac8f46f951b51584be36c13e5363008144), configuração `ptbr`, revisão fixa `419566ac8f46f951b51584be36c13e5363008144`. Consulte as condições de uso e atribuições descritas na documentação de licenças desta distribuição. Não substitua a revisão fixa por `main`.

Na nova árvore de execução preparada na etapa anterior, use o auxiliar restrito a desenvolvimento:

```bash
python scripts/acquire_development.py
```

Ele exige o marcador `reproduction_workspace.json`, recusa o checkout de evidências e árvores com estado/autorização finais, e baixa exatamente o README e os dois shards `ptbr/train-00000-of-00001.parquet` e `ptbr/dev-00000-of-00001.parquet` na revisão fixa. Confere os hashes brutos históricos antes de materializar as duas partições locais. A curação reutiliza a implementação original, verifica a fidelidade por coluna e produz somente treino/dev da Track A. Nenhum shard de teste é baixado ou aberto.

Os scripts históricos `download_brighter.py --revision <SHA>` e `curate_track_a.py --revision <SHA>` continuam disponíveis para rastreabilidade; seu procedimento integral também inspeciona teste e não deve substituir esse comando na verificação restrita. A aquisição/auditoria de uma partição não autoriza avaliá-la ou ajustar parâmetros nela.

O payload completo possui 2.226/400/4.452 registros em treino/dev/teste, contendo Track A e C. A seleção explícita de IDs da Track A produz 2.226/200/2.226. O conjunto de desenvolvimento experimental contém os 2.426 registros de treino+dev. Não remover por deduplicação genérica o overlap oficial preservado nem alterar texto/rótulos.

Os hashes de serialização Parquet podem depender das versões usadas na aquisição. Por padrão, se os Parquets curados não forem idênticos em bytes aos históricos, o auxiliar registra `BLOCKED_BEFORE_TRAINING` e recusa adaptar os folds. Isso não altera os manifests de referência. Para uma reprodução metodológica que aceite a nova serialização verificada, prepare outra árvore nova e use explicitamente:

```bash
python scripts/acquire_development.py --accept-reencoded-parquet
```

Essa opção só é aceita depois de verificar os hashes brutos fixos, a curação original e os mesmos IDs na mesma ordem. Ela mantém integralmente as atribuições dos cinco folds e cria fingerprints **operacionais reconstruídos**, documentados no manifest e no recibo. Os hashes e manifests históricos continuam intactos em `metadata/publication/reference_*.json`; não se alega identidade bit a bit dos Parquets reconstruídos. Essa adaptação documenta serialização, sem escolher novos folds ou hiperparâmetros. O recibo `metadata/publication/development_acquisition_receipt.json` registra versões, hashes, fidelidade e ausência de acesso ao teste.

O auxiliar também cria `dataset_manifest.json`, `track_a_manifest.json` e o marcador de snapshot nos caminhos operacionais usados pelos relatórios originais. Esses documentos são marcados `RECONSTRUCTED_DEVELOPMENT_ONLY`, contêm somente treino/dev e hashes da execução nova, e mantêm a referência aos manifests históricos completos. Isso permite que a auditoria `_dataset_integrity` dos relatórios de limiares/fusão e do congelamento confira os seis arquivos de treino/dev (bruto, intermediário e curado) sem abrir o teste. Esses documentos parciais não devem ser apresentados como uma aquisição completa de todas as partições.

## 5. Executar o desenvolvimento (opcional, com custo de treinamento)

Os comandos seguintes pertencem somente à nova árvore de execução, após a instalação e aquisição apropriadas. Requerem CUDA no protocolo Transformer original. Eles não foram executados integralmente como verificação desta publicação.

```bash
python scripts/run_baselines.py
python scripts/run_experiment.py --config configs/bertimbau_base_weighted_bce.yaml
python scripts/run_experiment.py --config configs/bertimbau_base_focal.yaml
python scripts/run_experiment.py --config configs/bertimbau_base_weighted_focal.yaml
python scripts/run_experiment.py --config configs/bertimbau_base_asl.yaml
python scripts/report_imbalance_losses.py
python scripts/run_experiment.py --config configs/bertweetbr_bce.yaml
python scripts/run_experiment.py --config configs/bertweetbr_weighted_focal.yaml
python scripts/run_experiment.py --config configs/bertweetbr_asl.yaml
python scripts/run_experiment.py --config configs/albertina100m_bce.yaml
python scripts/run_experiment.py --config configs/albertina100m_weighted_focal.yaml
python scripts/run_experiment.py --config configs/albertina100m_asl.yaml
python scripts/report_encoders.py
```

O primeiro comando executa/reutiliza as três baselines (TF-IDF, BERTimbau Base+BCE e Large+BCE). As dez configurações seguintes completam as 13 comparações iniciais. O conjunto `configs/` contém também duas YAML de implantação final, totalizando 15 arquivos históricos.

As probabilidades externas cobrem cinco folds. Os parâmetros de treinamento e as revisões dos modelos vêm dos arquivos preservados; não usar nomes de modelos sem seus SHAs. A busca anterior de modelos/perdas não foi integralmente aninhada.

## 6. Selecionar limiares e fusão no desenvolvimento

```bash
python scripts/run_threshold_calibration.py --system T1
python scripts/run_threshold_calibration.py --system T2
python scripts/run_threshold_calibration.py --system T3
python scripts/report_threshold_calibration.py
python scripts/run_ensemble.py
python scripts/report_ensemble.py
```

CT1/T1 corresponde a BERTweetBR+focal ponderada, CT2/T2 a BERTweetBR+ASL (E3) e CT3/T3 a BERTimbau Large+BCE (B2). Para cada treino externo, a calibração utiliza três folds internos e exclui o fold externo da seleção. EN1 fixa os pesos em 0,50/0,50; EN2 seleciona peso e seis limiares nas OOF internas. As predições externas concatenadas sustentam a estimativa EN2 0,5691 publicada. Executar novamente a seleção na área nova pode gerar parâmetros novos; nunca sobrescrever os congelados para adaptá-los ao teste.

## 7. Retreinar os componentes finais em treino+dev

Na árvore nova, depois da seleção de desenvolvimento:

```bash
python scripts/prepare_final_freeze.py
python scripts/train_final_models.py --system T2
python scripts/train_final_models.py --system T3
python scripts/finalize_pretest_freeze.py
```

O primeiro comando **escreve** a regra de implantação, configs finais e estado da nova execução. Os dois comandos de treino usam os 2.426 exemplos e quatro épocas; o último realiza um dry-run em duas frases artificiais e gera o congelamento pré-teste. Esses comandos exigem a área isolada e os requisitos de GPU; não devem ser executados no checkout de evidências históricas.

A regra histórica publicada é EN2 com pesos CT2/CT3 0,70/0,30 e limiares 0,67; 0,36; 0,43; 0,65; 0,58; 0,49 na ordem `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`. O escore 0,5897 do ajuste final nas mesmas OOF não é uma estimativa independente nem substitui 0,5691. Os parâmetros originais devem continuar preservados mesmo se a nova execução divergir.

O módulo histórico `finalization.py` conserva constantes das estimativas publicadas (EN2 0,5691254630769899 e CT2 0,5585138660403824) ao escrever alguns campos dos manifests finais. Esses campos não calculam automaticamente a estimativa de um novo treinamento. Para descrever uma execução nova, use as métricas realmente obtidas em seus summaries aninhados e de fusão, identifique separadamente essas referências históricas e não interprete a repetição de uma constante como reprodução experimental do escore. Os fontes originais foram preservados, inclusive essa limitação.

**A reprodução descrita aqui termina antes de qualquer autorização ou avaliação oficial do teste.** `authorize_final_test.py` e `run_final_test.py` são fontes históricos da avaliação encerrada, não passos a executar novamente. Não reinicialize o estado `TEST_EVALUATED` ou remova suas guardas para contornar essa restrição.

## 8. Inferência e resultados finais

A inferência final depende dos pesos ajustados e tokenizadores dos dois componentes, além das revisões, ordem dos rótulos e regra congelada. Consulte a documentação de artefatos/licenças para saber quais arquivos foram efetivamente distribuídos e onde obtê-los. Os nomes dos modelos base não substituem os checkpoints ajustados.

Enquanto a distribuição dos pesos finais não estiver resolvida e verificada, a inferência histórica completa permanece limitada. Treinar novamente produz outros checkpoints; isso não prova identidade dos pesos originais. O código original de inferência usa CUDA, FP16 e arquivos locais. Não utilizar o teste oficial para validar a instalação: os testes de software podem usar strings artificiais e a validação inicial pode limitar-se a arrays sintéticos.

O auxiliar adicional `scripts/predict_unlabelled.py` permite inferência em textos novos sem rótulos quando os diretórios finais externos estiverem disponíveis. Ele confere todos os hashes do manifest antes de carregar modelos, reutiliza a função de predição e fusão originais, e exige saída nova fora do checkout. Cada linha JSONL de entrada contém somente `id` e `text`; não são aceitos rótulos ou IDs oficiais de teste. Exemplo de uso futuro, depois de obter os pesos e tokenizadores corretos:

```bash
python scripts/predict_unlabelled.py --t2-directory ../modelos/t2_bertweetbr_asl --t3-directory ../modelos/t3_bertimbau_large_bce --input ../textos-novos.jsonl --output ../predicoes-novas.jsonl
```

Esse comando não foi executado com modelos nesta auditoria; os diretórios são caminhos de exemplo, não downloads já disponibilizados. Consulte [Artefatos externos](artefatos_externos.md) para a situação real.

Os resultados finais documentados são EN2 Macro-F1 0,2852 e CT2 0,1569, na avaliação única de 2.226 exemplos. As tabelas podem ser regeradas dos registros congelados pela etapa 2 sem repetir essa avaliação. Nenhum procedimento desta ampliação altera o manuscrito LaTeX ou o PDF submetido.

## 9. Verificação e limites

O relatório de verificação desta distribuição deve registrar o ambiente reconstruído, importações, validação das 15 configurações, testes sintéticos selecionados, preparação das áreas isoladas e regeneração de relatórios efetivamente executados. Os testes de integração que leem dados do teste, carregam checkpoints ou regravam artefatos históricos não integram uma execução indiscriminada de `pytest tests`.

Passar importações/testes unitários e reproduzir tabelas agregadas não comprova um novo treinamento completo, reprodução bit a bit em CUDA ou recuperação dos pesos finais. Essas limitações, a situação das licenças e os artefatos externos pendentes devem continuar explícitos no [README](../README.md) e na [documentação de reprodutibilidade](reprodutibilidade.md).
