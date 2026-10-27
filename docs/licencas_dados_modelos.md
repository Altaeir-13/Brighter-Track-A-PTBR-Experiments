# Licenças, atribuições e limites de distribuição

O código original deste projeto é disponibilizado sob a [licença MIT](../LICENSE), conforme autorização dos responsáveis. Essa licença não substitui as licenças do BRIGHTER, dos modelos-base, dos tokenizadores ou de outros materiais de terceiros. As referências abaixo foram conferidas em 26/10/2026 nas revisões exatas registradas pelos experimentos.

## BRIGHTER: aquisição na fonte e atribuição

O conjunto `brighter-dataset/BRIGHTER-emotion-categories`, configuração `ptbr`, declara **CC-BY-4.0** na revisão `419566ac8f46f951b51584be36c13e5363008144`. O [README dessa revisão](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories/blob/419566ac8f46f951b51584be36c13e5363008144/README.md) e a [API oficial da revisão](https://huggingface.co/api/datasets/brighter-dataset/BRIGHTER-emotion-categories/revision/419566ac8f46f951b51584be36c13e5363008144) concordam. A fonte estava pública e sem gating na consulta.

Os textos brutos não são redistribuídos neste GitHub. Use o procedimento de [aquisição e curação no guia de reprodução](guia_reproducao.md), que emprega os scripts originais e a revisão fixa. A seleção de Track A pelos IDs originais e a combinação de treino com desenvolvimento são transformações documentadas; não equivalem a deduplicação genérica nem à criação de uma nova anotação.

A [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) permite compartilhar e adaptar sob suas condições, incluindo atribuição, ligação à licença e indicação de alterações. Partições por IDs, rótulos oficiais e predições derivadas devem conservar a procedência do BRIGHTER. Isso não significa aplicar CC-BY ao código original ou conceder endosso dos autores do dataset. A licença também não elimina direitos de privacidade de terceiros.

Os resultados/predições disponibilizados para auditoria usam identificadores do corpus e valores numéricos, sem incorporar os textos brutos nesses arquivos de predição. A revisão de privacidade e o inventário da publicação distinguem esses arquivos dos dados de origem e dos relatórios qualitativos. A licença não autoriza acrescentar indiscriminadamente textos ou informações pessoais a novos artefatos.

O card do dataset solicita a citação destes trabalhos, sem que seja necessário inventar DOI ou paginação:

- Muhammad, Shamsuddeen Hassan et al. (2025). **BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages**. [arXiv:2502.11926](https://arxiv.org/abs/2502.11926).
- Muhammad, Shamsuddeen Hassan et al. (2025). **SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection**. [arXiv:2503.07269](https://arxiv.org/abs/2503.07269).

## Modelos-base e tokenizadores

| Modelo | Revisão experimental | Licença declarada na revisão |
|---|---|---|
| BERTweetBR: `melll-uff/bertweetbr` | `7b95f2097a9e37046cca243cc1aeca45b0849fce` | [Apache-2.0, card fixo](https://huggingface.co/melll-uff/bertweetbr/blob/7b95f2097a9e37046cca243cc1aeca45b0849fce/README.md) |
| BERTimbau Large: `neuralmind/bert-large-portuguese-cased` | `aa302f6ea73b759f7df9cad58bd272127b67ec28` | [MIT, card fixo](https://huggingface.co/neuralmind/bert-large-portuguese-cased/blob/aa302f6ea73b759f7df9cad58bd272127b67ec28/README.md) |
| BERTimbau Base: `neuralmind/bert-base-portuguese-cased` | `94d69c95f98f7d5b2a8700c420230ae10def0baa` | [MIT, card fixo](https://huggingface.co/neuralmind/bert-base-portuguese-cased/blob/94d69c95f98f7d5b2a8700c420230ae10def0baa/README.md) |
| ALBERTina 100M PT-BR: `PORTULAN/albertina-100m-portuguese-ptbr-encoder` | `d2a39d3d82408cc2a2618beb57a4fe908ee44b27` | [MIT, card fixo](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder/blob/d2a39d3d82408cc2a2618beb57a4fe908ee44b27/README.md) |

As quatro revisões foram resolvidas pela API oficial do Hugging Face, sem gating ou indicação de repositório privado. Os metadados e os READMEs concordam sobre as licenças. Nenhuma dessas árvores apresentava arquivo separado com nome LICENSE, NOTICE ou COPYING; a licença está declarada nos cards. O card ALBERTina também aponta para as licenças MIT da [família ALBERTina](https://huggingface.co/PORTULAN/albertina-ptpt/blob/main/LICENSE) e de [DeBERTa](https://github.com/microsoft/DeBERTa/blob/master/LICENSE). O projeto BERTimbau fornece o [aviso MIT da NeuralMind](https://github.com/neuralmind-ai/portuguese-bert/blob/master/LICENSE).

Os modelos-base e seus vocabulários devem ser obtidos das fontes upstream nas revisões fixas. Esta publicação não replica seus pesos ou tokenizadores. Seus nomes, licenças e SHAs permitem rastrear os componentes, mas não substituem os pesos finais ajustados para reproduzir a inferência histórica.

## Condições para uma futura distribuição dos pesos ajustados

CT2/T2 é derivada de BERTweetBR+ASL; CT3/T3 é derivada de BERTimbau Large+BCE. A [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) de BERTweetBR requer a licença e os avisos aplicáveis, preservação das atribuições e indicação das modificações. Se houver NOTICE upstream aplicável na cadeia de distribuição, ele deve acompanhar o derivado. A licença MIT de BERTimbau exige conservar os avisos de copyright e permissão aplicáveis.

Uma futura distribuição deve identificar o ajuste supervisionado em BRIGHTER PT-BR, a cabeça multirrótulo e a procedência dos tokenizadores. A decisão MIT para o código original não é apresentada como uma decisão automática de relicenciamento dos pesos derivados. Os avisos necessários e a licença de apresentação do pacote de modelos precisam ser preparados antes de seu upload.

Os checkpoints finais permanecem fora desta publicação. Não há pesos finais publicados no Hugging Face ou via Git LFS por esta atualização, nem URL fictícia de download. Consulte [artefatos externos](artefatos_externos.md) e o [catálogo verificável](../metadata/publication/external_artifacts.json) para tamanhos, hashes e pendências.
