# Verificação executada na ampliação de 26/10/2026

Esta é uma verificação nova de software e reconstrução limitada, separada dos experimentos históricos. O [registro estruturado](../metadata/publication/verification.json) conserva versões, contagens, hashes e recibos sem textos do corpus. O manuscrito e seus PDFs não foram alterados.

| Verificação realizada | Resultado |
|---|---|
| Venv nova no Windows, Python 3.12.14, PyTorch 2.13.0+cpu | Instalação concluída; `pip check` sem dependências quebradas |
| Imports dos módulos científicos | 20 módulos importados, sem carregar modelos |
| Configurações | 13 de treino e 2 finais validadas |
| Partições publicadas | 2.426 atribuições externas e 15 manifests internos com cobertura/exclusão verificadas |
| Testes selecionados | **64 passaram**: 27 de validação/aquisição sintética/EDA/perdas, 22 de protocolo e 15 dos auxiliares de distribuição |
| F1 das OOF congeladas | CT1–CT3 e EN1/EN2 recalculados nas decisões de desenvolvimento; EN2 0,5691254630769899 |
| Regeneração de relatórios | 20 linhas de sistemas e 369 evidências; todas as células do CSV principal coincidiram com o original |
| Aquisição real restrita | Somente README, treino e dev no SHA fixo; hashes brutos coincidiram |
| Seleção Track A | 2.226 treino + 200 dev; todos os valores e a ordem das linhas coincidiram com o acervo original |
| Folds na reconstrução | Atribuições externas preservadas; cinco conjuntos de partições internas regenerados coincidiram com os registros |
| Integridade da distribuição | 652 arquivos-fonte conferidos e três redações de privacidade declaradas |
| Guarda do teste oficial | Bloqueou antes de abrir dados, no estado histórico `TEST_EVALUATED` |

Os testes usam arrays/textos artificiais, diretórios temporários e configurações. As operações backward das perdas são verificações matemáticas em tensores sintéticos, sem treinar um encoder ou gerar checkpoint. Os testes originais que acessam dados brutos do teste, carregam modelos ou regravam resultados não foram executados indiscriminadamente. Houve quatro avisos esperados do scikit-learn sobre classes sem positivos em exemplos sintéticos; não houve falhas.

## Diferença real de serialização

Apesar de os bytes dos shards oficiais coincidirem com os hashes históricos, os Parquets materializados pelo ambiente reconstruído tiveram hashes diferentes dos arquivos históricos. O padrão do novo auxiliar recusou prosseguir. Em uma **terceira árvore nova**, a opção explícita `--accept-reencoded-parquet` registrou fingerprints operacionais novos, preservando os quatro manifests históricos de referência e exatamente os mesmos IDs, ordem e atribuições de folds.

Os dados curados reconstruídos foram comparados com os arquivos históricos somente de treino/dev: todas as colunas, valores e ordem coincidiram. Isso confirma identidade semântica nesses splits, não igualdade byte a byte da serialização. Nenhum hash histórico foi substituído. Antes disso, uma incompatibilidade do namespace de caminhos longos do Windows retornado pelo Hugging Face foi corrigida no auxiliar novo e coberta por três regressões. As árvores das tentativas anteriores foram preservadas fora da distribuição.

## Comandos das verificações unitárias

No perfil da [instalação](instalacao.md):

```bash
python -m pytest -q tests/test_validation.py tests/test_download.py tests/test_eda.py tests/test_losses.py
python -m pytest -q tests/test_public_reproduction.py tests/test_unlabelled_interface.py
python -m pytest -q tests/test_curation.py tests/test_experiments.py tests/test_threshold_calibration.py tests/test_ensemble.py tests/test_encoders.py -k "identifies_track or duplicate_groups_compare or exact_row_duplicates or curation_is_deterministic or thresholding_is_fixed or metrics_have_known_values or zero_label_metrics_known_values or oof_completeness_order or config_loading_and_prohibited or final_test_guard_fails_before or threshold_search_is_bounded or ap_is_invariant or alpha_grid_is_exact or probability_fusion_and_endpoints or alpha_tie_break or threshold_tie_break or encoder_configs_keep_frozen_protocol"
python scripts/check_installation.py
```

O terceiro comando seleciona os 22 testes dos módulos de curação, experimentos, limiares, fusão e encoders por funções sintéticas/configurações, excluindo integração com dados/modelos históricos. O [registro estruturado](../metadata/publication/verification.json) identifica as contagens; os testes completos permanecem publicados para inspeção. Não usar `pytest tests` como sinônimo desse escopo seguro.

Para reconstruir relatórios e conferir as decisões OOF, use os comandos do [guia](guia_reproducao.md), com destinos novos fora do checkout. Para obter o desenvolvimento, execute as cópias do auxiliar dentro da árvore preparada. Nenhum comando de autorização/reavaliação do teste integra essa verificação.

## O que não foi verificado

Não foram executados o treinamento completo em CUDA, a inferência com os checkpoints finais, uma nova seleção sobre o teste ou o notebook em um kernel Jupyter. Não foi reconstruída uma imagem histórica de sistema/driver. O êxito dos testes, da aquisição e das tabelas não deve ser apresentado como reprodução integral bit a bit dos modelos e de todo o protocolo.

O código original de finalização conserva constantes dos escores históricos; elas são referências, não novas medições de uma execução reconstruída. Algumas figuras históricas usam a legenda abreviada “leakage-safe development”: a interpretação continua limitada à seleção de limiares/pesos de configurações pré-selecionadas; a busca completa de modelos/perdas não foi inteiramente aninhada.

Os dois pesos finais foram re-hasheados e coincidem com os manifests. Seus arquivos continuam fora do GitHub, como registra o [catálogo externo](artefatos_externos.md). A decisão sobre hospedagem e os avisos dos modelos derivados permanece pendente.
