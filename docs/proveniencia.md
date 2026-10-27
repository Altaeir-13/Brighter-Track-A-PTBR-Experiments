# Procedência do material público

O repositório original da distribuição usada na revisão é [Altaeir-13/Brighter-Track-A-PTBR-Experiments](https://github.com/Altaeir-13/Brighter-Track-A-PTBR-Experiments), público, de propriedade da conta `Altaeir-13`, com branch principal `main`. A identificação foi conferida em 26/10/2026 usando metadados do GitHub, histórico, objeto Git do notebook e comparação integral com o material anonimizado e a cópia experimental preservada.

## Referência histórica

| Objeto | Identificador |
|---|---|
| Commit original | `3ea7669fd4782afccd097e934fc1ab2f4cf2ab81` |
| Mensagem | `Add reproducibility notebook` |
| Data do commit | `2026-09-04T22:57:32Z` |
| Árvore Git | `fecffc4533e988ca3cd4b92b3a8c645377430b04` |
| Blob do notebook | `718f6e18f3fe83ed74acd7a0fe8d136818e05d66` |
| SHA-256 do notebook publicado, LF | `c6048c163c7af753e31b75fd209df0c037cab63956a1cd8cf7e9b0433309789e` |
| Tamanho publicado | 105.599 bytes |

A [árvore original](https://github.com/Altaeir-13/Brighter-Track-A-PTBR-Experiments/tree/3ea7669fd4782afccd097e934fc1ab2f4cf2ab81) contém apenas `BRIGHTER_PTBR_Experiment.ipynb`. No momento da consulta, havia uma branch (`main`), um commit sem pais e nenhuma referência de tag. O autor e committer registrados são `Anonymous Author`; essa identidade do commit não comprova a autoria do treinamento.

Os relatórios históricos locais `repository_notebook_publish.md`, `anonymous_repository_audit.md` e `anonymous_repository_insertion.md` registram a publicação e a utilização do espelho [Brighter-Track-A-PTBR-Experiments-E2F8](https://anonymous.4open.science/r/Brighter-Track-A-PTBR-Experiments-E2F8/). Sua relação não foi deduzida da semelhança de nomes: o commit e o hash publicados coincidem com o registro histórico e o conteúdo integral do notebook.

A cópia local preservada tem 108.591 bytes e SHA-256 `f0b318c51c7cbe4b2233649c4dcd23f08a4be86704f9f36d72335d5ad585fcb6`. Normalizar somente CRLF para LF reproduz exatamente os bytes e o blob Git do arquivo publicado. O JSON visível no espelho anônimo, obtido pelo link público de arquivo bruto, é integralmente igual ao JSON do GitHub e da cópia local: células, fontes, outputs, contadores e metadados. São 30 células Markdown, 25 de código e 41 outputs.

Não foi observada diferença de conteúdo científico entre os dois serviços nessa consulta. A interface anônima não expõe um commit de origem verificável; a associação ao commit vem da cadeia documental e da igualdade do conteúdo, não de um identificador de snapshot apresentado pelo serviço. A árvore do espelho também mostrou somente o notebook, com expiração indicada em 04/03/2027. Datas de anonimização ou disponibilidade não foram usadas como datas dos experimentos.

## Rastreabilidade dos fatos documentados

[evidencias_documentais.json](evidencias_documentais.json) contém uma transcrição selecionada de números e parâmetros já existentes, com SHA-256 dos arquivos-fonte inspecionados. Não é uma nova avaliação, nem substitui os arquivos originais. Os caminhos listados nesse JSON são referências ao workspace experimental preservado; eles não implicam que o arquivo esteja publicado ou que possa ser baixado deste repositório.

As fontes incluem relatórios de execução, manifests de congelamento, parâmetros decisórios, métricas finais, diagnósticos sobre probabilidades congeladas, configurações e implementação. A documentação conserva a diferença entre a seleção aninhada de limiares/pesos e a seleção anterior de modelos/perdas, entre desenvolvimento e teste, e entre o ambiente de treinamento e o ambiente de apresentação do notebook.

O repositório original é confirmado como origem do material anonimizado. O commit inicial do notebook não contém o código completo de treinamento nem uma cronologia Git das execuções anteriores à publicação; não se atribui a ele essa garantia histórica.

## Ampliação do pipeline

A nova distribuição acrescenta o material científico preservado posteriormente, sem reescrever o histórico. [source_file_manifest.json](../metadata/publication/source_file_manifest.json) registra 652 arquivos copiados, hashes de origem/cópia e as três redações mínimas de privacidade. Também registra exclusões, como dados brutos, citações literais do corpus e pesos volumosos. `.gitattributes` preserva os bytes e finais de linha originais.

Os auxiliares novos de isolamento, aquisição restrita e verificação estão identificados nos docstrings e no [guia](guia_reproducao.md); não são apresentados como código executado nos experimentos históricos. O [registro de verificação](verificacao_executada.md) descreve execuções novas limitadas, sem substituir resultados, parâmetros ou manifests históricos. Alguns caminhos da transcrição documental anterior agora estão publicados; outros, como o manuscrito, continuam apenas referências locais de procedência.
