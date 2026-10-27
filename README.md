# Do Desenvolvimento ao Teste: Perfil de Erros no Reconhecimento Multirrótulo de Emoções em PT-BR

Documentação científica do artigo aceito no **ERCEMAPI 2026**, de **Randerson Sousa de Sá Nunes e Rafael Torres Anchiêta**. O estudo examina a transferência do desempenho de desenvolvimento para o teste oficial no reconhecimento multirrótulo de emoções em português brasileiro.

## Material disponível

O [notebook BRIGHTER_PTBR_Experiment.ipynb](BRIGHTER_PTBR_Experiment.ipynb) contém a narrativa experimental, código de leitura de artefatos e resultados incorporados: 55 células e 41 outputs preservados. Sua publicação original está fixada no commit [`3ea7669fd4782afccd097e934fc1ab2f4cf2ab81`](https://github.com/Altaeir-13/Brighter-Track-A-PTBR-Experiments/commit/3ea7669fd4782afccd097e934fc1ab2f4cf2ab81). Esta atualização acrescenta documentação; não modifica o notebook histórico.

**A distribuição não é autossuficiente para reproduzir o experimento completo.** Código de treinamento, dados, pesos finais, folds e os arquivos externos lidos pelo notebook não estão integralmente publicados. Os outputs podem ser consultados sem executar células. A [auditoria de disponibilidade](docs/reprodutibilidade.md) discrimina o que existe no workspace preservado e o que falta na distribuição pública.

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

Para consultar o código hoje publicado, abra o notebook no GitHub. O código completo do pipeline permanece no workspace experimental e não acompanha esta atualização documental. Não é necessário executar o notebook para inspecionar as tabelas salvas. O teste oficial está encerrado; não se propõe nova avaliação nem novo ajuste sobre ele.

## Estrutura desta distribuição

```text
BRIGHTER_PTBR_Experiment.ipynb    # notebook histórico, com outputs preservados
README.md                       # apresentação e referência ao artigo
docs/
  ambiente.md                   # ambiente histórico e hiperparâmetros
  metodologia.md                # protocolo efetivamente executado
  resultados.md                 # resultados e limites de interpretação
  reprodutibilidade.md          # disponibilidade e arquivos ausentes
  proveniencia.md               # identidade e rastreabilidade documental
  evidencias_documentais.json   # transcrição selecionada e hashes das fontes
```

Diretórios do projeto completo mencionados no notebook, como `src/`, `configs/`, `data/`, `metadata/` e `artifacts/`, não devem ser presumidos presentes nesta árvore. [Procedência e hashes](docs/proveniencia.md) documentam a correspondência do notebook com o material originalmente anonimizado.

## Referência ao artigo

NUNES, Randerson Sousa de Sá; ANCHIÊTA, Rafael Torres. **Do Desenvolvimento ao Teste: Perfil de Erros no Reconhecimento Multirrótulo de Emoções em PT-BR**. ERCEMAPI, 2026. Artigo aceito; dados bibliográficos finais ainda não informados nesta documentação.

DOI, paginação e endereço dos anais serão acrescentados quando confirmados. A licença declarada pelo snapshot BRIGHTER não se estende automaticamente ao código ou ao notebook; nenhum arquivo de licença do projeto foi identificado no commit original.
