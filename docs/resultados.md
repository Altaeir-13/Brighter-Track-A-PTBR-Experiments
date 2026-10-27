# Resultados congelados e limites das conclusões

Os números abaixo foram transcritos dos registros existentes e conferidos com o artigo **Do Desenvolvimento ao Teste: Perfil de Erros no Reconhecimento Multirrótulo de Emoções em PT-BR**, aceito no ERCEMAPI 2026. Esta atualização documental não executou treinamentos nem avaliações. CT1, CT2 e CT3 correspondem a T1, T2 e T3 nos identificadores históricos do notebook e dos artefatos.

## Desenvolvimento e teste oficial

| Sistema e condição | Macro-F1 | Micro-F1 |
|---|---:|---:|
| CT1, limiares selecionados internamente, 2.426 OOF externas | 0,5152 | 0,6197 |
| CT2, limiares selecionados internamente, 2.426 OOF externas | 0,5585 | 0,6750 |
| CT3, limiares selecionados internamente, 2.426 OOF externas | 0,5023 | 0,6248 |
| EN1, pesos iguais e limiares internos, 2.426 OOF externas | 0,5548 | 0,6657 |
| EN2, pesos e limiares internos, 2.426 OOF externas | **0,5691** | 0,6846 |
| EN2 primário final, 2.226 exemplos do teste oficial | **0,2852** | 0,4221 |
| CT2 secundária final, 2.226 exemplos do teste oficial | **0,1569** | 0,2405 |

As métricas de desenvolvimento são calculadas nas predições externas concatenadas. A avaliação é aninhada para limiares e pesos de configurações pré-selecionadas; a busca completa de modelos e perdas não foi inteiramente aninhada. EN2 final usa CT2/CT3 com pesos 0,70/0,30 e os limiares congelados descritos em [metodologia](metodologia.md).

No desenvolvimento, EN2 superou CT2 em +0,0106 de Macro-F1, com intervalo bootstrap pareado percentil de 95% [+0,0011; +0,0202]. Esse intervalo descreve as predições avaliadas e não constitui teste confirmatório de superioridade geral. EN1 ficou abaixo da CT2, mostrando que pesos iguais não produziram o mesmo ganho observado com a seleção conjunta.

Entre desenvolvimento e teste, EN2 passou de 0,5691 para 0,2852 (diferença −0,2839); CT2 passou de 0,5585 para 0,1569 (−0,4016). CT3 não possui resultado autônomo no teste. Portanto, a comparação final mostra vantagem de EN2 sobre CT2 na condição avaliada, sem estabelecer superioridade sobre ambos os componentes ou quantificar isoladamente a contribuição da CT3.

O escore 0,5897 do ajuste final sobre todas as OOF não é uma estimativa independente e não substitui 0,5691.

## Cardinalidade e perfil por emoção

A cardinalidade real permaneceu próxima: 0,8821 no desenvolvimento e 0,8823 no teste. A cardinalidade predita pelo EN2 caiu de **1,0639 para 0,3459**. No teste, sua precisão macro foi 0,6325 e sua revocação macro, 0,2277: o perfil é de numerosas omissões.

| Emoção EN2 | F1 desenvolvimento aninhado | F1 teste | Revocação desenvolvimento | Revocação teste |
|---|---:|---:|---:|---:|
| Raiva | 0,7882 | 0,3883 | 0,7938 | 0,2575 |
| Nojo | 0,2780 | 0,0270 | 0,4500 | 0,0141 |
| Medo | 0,3846 | 0,1401 | 0,5556 | 0,0791 |
| Alegria | 0,8025 | 0,6791 | 0,8025 | 0,5586 |
| Tristeza | 0,6625 | 0,0342 | 0,7514 | 0,0175 |
| Surpresa | 0,4990 | 0,4423 | 0,6897 | 0,4395 |

Nojo, medo e tristeza tiveram baixa recuperação; tristeza mostra que a queda não se limita às classes previamente tratadas como minoritárias. Alegria e surpresa conservaram melhores F1 relativos. Nos 616 exemplos de teste sem rótulo, a acurácia exata do EN2 foi 0,9058 e a proporção com algum falso positivo foi 0,0942. Muitos verdadeiros negativos refletem também a frequência de ausências, e não demonstram, isoladamente, bom reconhecimento de emoções.

## Decisão, ordenação e probabilidades

| Indicador EN2 | Desenvolvimento aninhado | Teste final |
|---|---:|---:|
| Macro-F1: qualidade da decisão | 0,5691 | 0,2852 |
| Macro-AP: ordenação sem limiar fixo | 0,6002 | 0,5042 |
| Brier macro: erro quadrático probabilístico | 0,1275 | 0,1215 |
| ECE: dez intervalos uniformes | 0,2496 | 0,1965 |

F1 e AP caíram, enquanto Brier e ECE agregados diminuíram. Não se sustenta uma piora probabilística generalizada; tampouco a redução desses dois indicadores garante calibração adequada para cada emoção. Os indicadores descrevem propriedades distintas. AP, Brier e ECE de teste são diagnósticos posteriores calculados a partir de probabilidades preservadas, sem modificar o sistema final.

Os cruzamentos dos limiares finais diminuíram para a maioria das emoções quando a regra final foi aplicada às OOF dos componentes e às probabilidades finais. Essa análise descritiva de transferência de escores é diferente da avaliação aninhada. Não demonstra que os limiares foram a única causa da queda, já que a ordenação também se deteriorou.

## Conclusão e limitações

Os ganhos de desenvolvimento não se transferiram integralmente para o teste após o retreinamento. O estudo documenta queda de decisão e ordenação, redução da cardinalidade predita e omissões por emoção. As prevalências dos rótulos mudaram modestamente, mas isso não isola nem elimina mecanismos causais.

Mudaram simultaneamente os modelos por partição para os modelos finais retreinados, os exemplos avaliados e os parâmetros internos para os parâmetros finais de decisão. Uma semente final, o pequeno suporte de algumas emoções, a seleção anterior não totalmente aninhada, a sobreposição oficial preservada e a ausência de CT3 autônoma no teste restringem as conclusões. Não há inferência pareada de modelos por fold e modelos finais nos mesmos textos reservados que permita separar efeito do retreinamento e mudança do conjunto avaliado. **Não foi identificada uma causa única para a queda**, e os resultados não autorizam extrapolação para outros idiomas, domínios ou tracks.
