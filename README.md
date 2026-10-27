# Do Desenvolvimento ao Teste: Perfil de Erros no Reconhecimento Multirrótulo de Emoções em PT-BR

Documentação científica do artigo aceito no **ERCEMAPI 2026**, de **Randerson Sousa de Sá Nunes e Rafael Torres Anchiêta**. O estudo examina a transferência do desempenho de desenvolvimento para o teste oficial no reconhecimento multirrótulo de emoções em português brasileiro.

## Material disponível

O [notebook BRIGHTER_PTBR_Experiment.ipynb](BRIGHTER_PTBR_Experiment.ipynb) contém a narrativa experimental, código de leitura de artefatos e resultados incorporados: 55 células e 41 outputs preservados. Sua publicação original está fixada no commit [`3ea7669fd4782afccd097e934fc1ab2f4cf2ab81`](https://github.com/Altaeir-13/Brighter-Track-A-PTBR-Experiments/commit/3ea7669fd4782afccd097e934fc1ab2f4cf2ab81). Esta ampliação acrescenta o pipeline e sua documentação, preservando o notebook histórico.

Esta distribuição também disponibiliza o **código original do pipeline, 15 configurações YAML, partições externas e internas, manifests, resultados congelados, 245 artefatos de predição sem texto e figuras agregadas**. Os arquivos científicos foram preservados por hash, com três redações de privacidade documentadas. Os textos do dataset são obtidos da fonte oficial na revisão fixa; os pesos finais ainda dependem de distribuição externa.

Comece pela [instalação verificada em CPU](docs/instalacao.md), pelo [guia de reprodução](docs/guia_reproducao.md) e pela [verificação executada](docs/verificacao_executada.md). **A inferência histórica a partir dos mesmos pesos e o treinamento completo em CUDA não foram reproduzidos nesta ampliação.** Consulte os [limites e a disponibilidade](docs/reprodutibilidade.md).

## Contexto, dados e objetivo

Cada texto pode receber zero, um ou vários dos seis rótulos: raiva (`anger`), nojo (`disgust`), medo (`fear`), alegria (`joy`), tristeza (`sadness`) e surpresa (`surprise`). O sistema produz seis probabilidades por texto, uma por emoção, posteriormente convertidas em decisões binárias por limiares. Essa formulação não pressupõe independência estatística entre as emoções. Macro-F1 é a métrica primária.

Usamos o conjunto [BRIGHTER Emotion Categories](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories/tree/419566ac8f46f951b51584be36c13e5363008144), Track A, configuração `ptbr`, revisão `419566ac8f46f951b51584be36c13e5363008144`. A seleção explícita dos IDs da Track A preserva as partições oficiais: 2.226 textos de treino, 200 de desenvolvimento e 2.226 de teste. Treino e desenvolvimento foram reunidos em 2.426 exemplos. A seleção separa Track A de Track C, sem deduplicação genérica.

## Metodologia

A comparação incluiu TF-IDF com regressão logística, BERTimbau Base e Large, BERTweet.BR e Albertina 100M PT-BR. Foram examinadas BCE, BCE ponderada, focal, focal ponderada e Asymmetric Loss (ASL), nas combinações efetivamente executadas descritas em [Metodologia](docs/metodologia.md).

As configurações pré-selecionadas CT1/T1, CT2/T2 e CT3/T3 são BERTweet.BR com focal ponderada, BERTweet.BR com ASL e BERTimbau Large com BCE, respectivamente. Cinco partições externas e três internas por treino externo separam a seleção de limiares e pesos da avaliação externa. **A seleção anterior de modelos e perdas não foi integralmente aninhada.**

EN1 combina probabilidades CT2/CT3 com pesos iguais e seleciona limiares por emoção. EN2 seleciona conjuntamente o peso da fusão e os limiares nas OOF internas. Após essa avaliação, as OOF dos 2.426 exemplos definiram a regra final: pesos CT2/CT3 de 0,70/0,30 e limiares EN2 de 0,67; 0,36; 0,43; 0,65; 0,58; 0,49, na ordem dos rótulos acima. Os componentes foram retreinados em treino+desenvolvimento e o teste oficial foi avaliado sob protocolo congelado.

## Resultados centrais

| Sistema | Condição | Macro-F1 |
|---|---|---:|
| EN2 | Desenvolvimento: 2.426 OOF externas, seleção interna de limiares e pesos | **0,5691** |
| EN2 | Sistema primário final, teste oficial de 2.226 textos | **0,2852** |
| CT2 | Sistema secundário final, mesmo teste oficial | **0,1569** |

A cardinalidade predita pelo EN2 diminuiu de 1,0639 para 0,3459. Nojo, medo e tristeza apresentaram baixa recuperação no teste. Macro-AP passou de 0,6002 para 0,5042; Brier e ECE agregados diminuíram, embora F1 e ordenação tenham piorado. Qualidade de decisão, ordenação e erro probabilístico descrevem propriedades distintas.

Os resultados não isolaram uma causa única para a queda. Retreinamento, exemplos avaliados e parâmetros de decisão mudam entre as condições. A reutilização do desenvolvimento na seleção anterior, a semente final única, o pequeno suporte de algumas emoções, a sobreposição textual oficial preservada e a ausência de CT3 autônoma no teste restringem as conclusões. Consulte [Resultados e limitações](docs/resultados.md). O escore de ajuste final de 0,5897 não substitui a estimativa aninhada de 0,5691.

## Ambiente e consulta

Os registros históricos de hardware, Python, PyTorch, CUDA, dependências e configurações de treinamento estão em [Ambiente computacional](docs/ambiente.md). A documentação distingue telemetria, campos declarados e diferenças entre aquisição, treinamento e apresentação do notebook; não oferece um lockfile integral reconstruído como se fosse histórico.

O código está em `src/brighter/`, suas interfaces em `scripts/` e as configurações em `configs/`. Use uma árvore nova preparada pelo auxiliar do guia para treinar: mudar somente o diretório atual não isola as saídas dos scripts originais. Para inspecionar as tabelas salvas do notebook, não é necessário executá-lo. O teste oficial está encerrado; não se propõe nova avaliação nem novo ajuste sobre ele.

## Estrutura desta distribuição

```text
BRIGHTER_PTBR_Experiment.ipynb    # notebook histórico inalterado
src/brighter/                   # implementação original
scripts/                        # interfaces originais e auxiliares novos identificados
configs/                        # 13 configurações de treino + 2 finais
tests/                          # testes originais e guardas dos auxiliares
metadata/brighter_ptbr/          # folds, versões, parâmetros e manifests históricos
metadata/publication/            # hashes da distribuição, redações e verificação nova
reports/brighter_ptbr/           # resultados e tabelas congelados
artifacts/predictions/           # OOF internas/externas e saídas finais sem texto
artifacts/models/final/          # somente manifests/configuração de treino/rótulos
figures/brighter_ptbr/           # figuras agregadas originais
requirements/                   # perfil reconstruído CPU, separado do histórico
docs/                           # método, resultados, instalação, licenças e reprodução
LICENSE                         # MIT para o código original
```

`data/` e checkpoints não são distribuídos no Git. O auxiliar de aquisição obtém somente treino/dev oficiais e os seleciona por IDs da Track A na revisão fixa; a aquisição do teste não integra a verificação desta distribuição. [Procedência e hashes](docs/proveniencia.md) identificam o material original e as cópias públicas. [Artefatos externos](docs/artefatos_externos.md) registra tamanhos, hashes e pendências dos dois checkpoints, sem prometer URLs de download ainda inexistentes.

## Verificação sem repetir o teste

```bash
python scripts/check_installation.py
python scripts/verify_frozen_development.py --output ../oof-verificadas.json
python scripts/rebuild_frozen_reports.py --destination ../relatorios-reproduzidos
```

Os dois destinos devem ser novos e ficar fora do checkout. As verificações executadas incluem imports dos 20 módulos, validação das 15 configurações e dos folds, testes sintéticos selecionados, recálculo do F1 das OOF de desenvolvimento e regeneração de 20 linhas de sistemas e 369 evidências a partir dos agregados congelados. Não houve inferência, retreinamento ou nova seleção no teste. Detalhes e limites estão no [registro de verificação](docs/verificacao_executada.md).

## Referência ao artigo

NUNES, Randerson Sousa de Sá; ANCHIÊTA, Rafael Torres. **Do Desenvolvimento ao Teste: Perfil de Erros no Reconhecimento Multirrótulo de Emoções em PT-BR**. ERCEMAPI, 2026. Artigo aceito; dados bibliográficos finais ainda não informados nesta documentação.

DOI, paginação e endereço dos anais serão acrescentados quando confirmados. O código original está sob [MIT](LICENSE), conforme autorização do responsável. IDs e rótulos derivados do BRIGHTER conservam as obrigações CC-BY-4.0; modelos-base possuem licenças próprias. Consulte [licenças e atribuições](docs/licencas_dados_modelos.md). A licença do código não substitui essas condições.
