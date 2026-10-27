# Metodologia executada

Esta documentação acompanha o notebook `BRIGHTER_PTBR_Experiment.ipynb` e descreve o experimento finalizado do artigo **Do Desenvolvimento ao Teste: Perfil de Erros no Reconhecimento Multirrótulo de Emoções em PT-BR**, aceito no ERCEMAPI 2026. Os nomes CT1–CT3 do artigo correspondem aos identificadores T1–T3 nos artefatos e no notebook.

## Dados e partições

O estudo usa BRIGHTER, Track A, configuração `ptbr`, na revisão `419566ac8f46f951b51584be36c13e5363008144` do conjunto `brighter-dataset/BRIGHTER-emotion-categories`. Cada texto recebe uma combinação binária de seis rótulos, nesta ordem: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`. Textos sem emoções anotadas são representados pelo vetor de seis zeros (`[0, 0, 0, 0, 0, 0]`), sem constituir uma sétima classe neutra. A expressão "caso neutro", nas métricas correspondentes, designa somente essa ausência dos seis rótulos.

O payload de origem contém 2.226 linhas de treino, 400 de desenvolvimento e 4.452 de teste. Desenvolvimento e teste incluem registros dos Tracks A e C. A seleção do Track A usa o componente explícito do identificador `ptbr_<split>_track_a_<number>`, preservando texto e rótulos; não usa deduplicação textual para decidir o track. As partições oficiais do Track A têm 2.226, 200 e 2.226 textos, respectivamente. Uma sobreposição textual oficial entre treino e teste foi documentada e mantida.

Treino e desenvolvimento oficiais foram reunidos em 2.426 exemplos. Cinco partições externas, por estratificação multirrótulo iterativa e semente 42, definiram as probabilidades out-of-fold (OOF): cada exemplo foi predito por um modelo que o excluiu de seu treino. As partições externas de validação têm 485 ou 486 exemplos. O teste oficial foi reservado para o sistema final congelado.

Não foram usados pré-processamento textual adicional, reamostragem, aumento de dados, traduções, dados sintéticos ou externos, nem Track B. A tokenização usa os tokenizadores dos modelos e truncamento a 128 tokens.

## Modelos e perdas

Foi avaliada uma referência lexical com TF-IDF de palavras/caracteres e regressão logística um-contra-todos. A comparação de Transformers incluiu BERTimbau Base, BERTimbau Large, BERTweet.BR e Albertina 100M PT-BR. A matriz efetivamente avaliada foi:

| Modelo | Perdas |
|---|---|
| BERTimbau Base | BCE, BCE ponderada, focal, focal ponderada, ASL |
| BERTimbau Large | BCE |
| BERTweet.BR | BCE, focal ponderada, ASL |
| Albertina 100M PT-BR | BCE, focal ponderada, ASL |

A comparação inicial usa limiar 0,50. A BCE opera independentemente por emoção; as variantes ponderadas calculam os pesos com o treino da respectiva partição. A focal usa `gamma=2`; a ASL usa `gamma_pos=0`, `gamma_neg=4` e clipping negativo `0.05`.

As configurações pré-selecionadas para a etapa aninhada foram:

| Artigo | Artefatos | Modelo e perda |
|---|---|---|
| CT1 | T1 | `melll-uff/bertweetbr`, focal ponderada |
| CT2 | T2 | `melll-uff/bertweetbr`, ASL |
| CT3 | T3 | `neuralmind/bert-large-portuguese-cased`, BCE |

Os Transformers usaram quatro épocas, AdamW, taxa inicial `2e-5`, decaimento de pesos `0.01`, aquecimento em 10% dos passos e decaimento linear, sem parada antecipada. O ambiente e os lotes estão detalhados na documentação computacional.

## Seleção aninhada de limiares e pesos

Para cada treino externo de 1.940 ou 1.941 exemplos, três partições internas por estratificação multirrótulo produziram OOF internas cobrindo todo esse treino externo. O fold externo de validação ficou excluído dos treinos internos e da seleção. As OOF internas concatenadas selecionaram um limiar por emoção na grade de 0,01 a 0,99, passo 0,01, maximizando F1; empates, com tolerância numérica de `1e-12`, favorecem o maior limiar. A decisão é `p >= limiar`. Selecionar limiares modifica decisões, sem alterar as probabilidades ou sua ordenação.

A fusão usa probabilidades alinhadas pelo identificador do texto, e não votação de rótulos. Para cada emoção, `p = alpha * p_CT2 + (1 - alpha) * p_CT3`.

- **EN1:** fixa `alpha=0.50` e seleciona os seis limiares nas OOF internas.
- **EN2:** percorre `alpha=0.00,...,1.00`, passo 0,05. Para cada peso, seleciona os seis limiares; escolhe a combinação com maior Macro-F1 interna. Empates de peso favorecem o valor mais próximo de 0,50 e depois o maior peso da CT2.

As regras selecionadas são avaliadas no fold externo, com as probabilidades externas previamente preservadas. As métricas principais de desenvolvimento são calculadas uma vez sobre as 2.426 OOF externas concatenadas, não como média aritmética dos cinco F1. Os pesos da CT2 escolhidos nos cinco treinos externos foram `0.90, 0.90, 0.75, 0.85, 0.95`. O critério de adoção da fusão exigia ganho mínimo de 0,005 em Macro-F1 sobre a CT2 no desenvolvimento.

**Limite do aninhamento:** a seleção anterior de modelos, perdas e componentes reutilizou o desenvolvimento e não foi integralmente encapsulada nas partições externas. O aninhamento separa seleção e avaliação de limiares e pesos das configurações previamente escolhidas. O valor 0,5691 não é uma estimativa livre de todo o otimismo da busca completa.

## Sistema final e teste reservado

Depois da avaliação externa, as OOF dos componentes CT2 e CT3, pareadas nos 2.426 exemplos, selecionaram a regra final. Essa seleção usa todo o conjunto OOF para definir parâmetros de implantação; não substitui a estimativa aninhada.

| Parâmetro | Valor final |
|---|---|
| Sistema primário | EN2 |
| Pesos CT2 / CT3 | 0,70 / 0,30 |
| Limiares EN2, na ordem dos rótulos acima | 0,67; 0,36; 0,43; 0,65; 0,58; 0,49 |
| Sistema secundário pré-registrado | CT2 autônoma |
| Limiares CT2 autônoma | 0,66; 0,53; 0,52; 0,75; 0,65; 0,57 |

O peso final 0,70 está fora da faixa 0,75–0,95 dos pesos escolhidos internamente; o limiar final de nojo 0,36 também fica abaixo da faixa interna 0,39–0,49. Isso documenta sensibilidade dos parâmetros à amostra, sem demonstrar causa da queda no teste. O Macro-F1 de 0,5897 obtido no próprio conjunto OOF usado para selecionar a implantação é um escore de ajuste da regra final, e **não uma estimativa independente de desempenho**.

CT2 e CT3 foram retreinadas em todos os 2.426 exemplos por quatro épocas, semente 42, sem validação ou parada antecipada. Pesos, limiares, ordem dos rótulos e papéis primário/secundário foram congelados antes do teste. CT3 participou apenas como componente: não foi avaliada autonomamente como terceiro sistema final.

A avaliação oficial ocorreu uma vez. Um incidente posterior na geração de figuras foi recuperado a partir de outputs já preservados, sem repetir inferência. As análises posteriores reutilizam predições e probabilidades congeladas; não selecionam novos limiares no teste.

## Métricas e interpretação

A métrica primária é Macro-F1, média dos F1 das seis emoções. Também são reportados Micro-F1, precisão, revocação, perda de Hamming, Jaccard por exemplo (`zero_division=0`), acurácia exata, cardinalidade média, acurácia exata dos casos sem rótulo e proporção desses casos com algum falso positivo.

Macro-AP avalia ordenação sem limiar fixo. Brier mede erro quadrático probabilístico; ECE usa dez intervalos uniformes ponderados pelo número de exemplos. As médias são entre as seis emoções. Brier não isola calibração; mudanças nesses indicadores não demonstram causalidade. Os intervalos de bootstrap pareado por instância usam 10.000 repetições, semente 42 e percentis; descrevem a diferença entre as predições avaliadas, não a variabilidade de todo o processo de seleção e treinamento.

Os diagnósticos de cruzamento de limiares aplicam a regra final também às probabilidades OOF dos componentes para comparar escalas. Eles não devem ser confundidos com as decisões do desenvolvimento aninhado, que empregam parâmetros próprios de cada partição externa.

As conclusões são limitadas ao Track A em PT-BR, às classes raras, a uma semente final e à reutilização do desenvolvimento. Retreinamento, conjunto avaliado e regra de decisão mudam simultaneamente entre as duas condições; os artefatos existentes não isolam uma causa única para a queda.
