# Ambiente computacional e configurações de execução

Esta documentação foi preparada em 26/10/2026 a partir da inspeção dos registros históricos dos experimentos BRIGHTER Track A PT-BR. Não houve novo treinamento, inferência, ajuste de limiares ou avaliação no teste oficial para produzi-la. As versões abaixo são transcrições dos registros, sem substituição por versões atuais.

**Escopo das fontes:** os caminhos `reports/`, `metadata/`, `artifacts/`, `configs/` e `src/` citados abaixo identificam o acervo local auditado, ainda não publicado como parte desta atualização. Eles permitem localizar a origem de cada informação no projeto completo, mas não indicam que esses arquivos acompanhem o notebook público. A disponibilização do notebook e desta documentação não equivale à distribuição do pipeline completo.

## Hardware e sistema operacional

Os objetos `environment` de 13 relatórios locais `reports/brighter_ptbr/experiments/*_results.json` concordam nos campos de Python, plataforma, PyTorch, GPU e capacidade de memória. Um exemplo é `bertweetbr_asl_results.json`, chaves `environment.*`.

| Recurso | Valor registrado | Evidência |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti | `environment.gpu` |
| Memória da GPU | 8.589.279.232 bytes, aproximadamente 8 GiB | `environment.gpu_total_memory_bytes` |
| Memória principal | 34.262.110.208 bytes, aproximadamente 31,91 GiB; capacidade nominal de 32 GB | `environment.ram_bytes` |
| Sistema operacional | `Windows-11-10.0.26200-SP0` | `environment.platform` |
| CPU | `AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD`; 6 núcleos físicos e 12 lógicos | `environment.cpu`, `physical_cpu_cores`, `logical_cpu_cores` |

O código local `src/brighter/experiments.py`, função `_environment_metadata`, consulta propriedades CUDA e `psutil` para registrar a memória. Já a função `_gpu_metadata` de `src/brighter/finalization.py` grava RAM de 32 GB e VRAM nominal de 8 GB como constantes declaradas. A confirmação dessas capacidades vem da telemetria dos relatórios, não apenas dos campos declarados nos manifests finais. Não há nome comercial da CPU nem versão do driver NVIDIA suficientemente documentados nas fontes examinadas.

## Software registrado

| Componente | Versão | Fase/fonte local |
|---|---|---|
| Python | 3.12.13 | Relatórios de execução e manifests finais |
| PyTorch | `2.13.0+cu130` | Relatórios de execução e manifests finais |
| CUDA reportada pelo PyTorch | `13.0` | `environment.cuda`, coletada de `torch.version.cuda` |
| cuDNN | `92000` | Valor inteiro de `torch.backends.cudnn.version()` |
| transformers | 5.16.1 | Relatórios de execução e manifests finais |
| tokenizers | 0.23.1 | Manifests finais |
| numpy | 1.26.4 | Manifests finais |
| pandas | 2.3.3 | Manifests finais |
| scikit-learn | 1.9.0 | Relatórios de execução e manifests finais |
| pyarrow | 23.0.1 | Manifests finais |
| huggingface-hub | 1.28.0 | Manifests finais |
| datasets | 4.8.5 | Relatórios de execução e manifesto de aquisição |
| iterative-stratification | 0.1.9 | Relatórios de execução |

Fontes do congelamento: `metadata/brighter_ptbr/final/final_experiment_manifest.json`, chave `software_versions`, e `artifacts/models/final/{t2_bertweetbr_asl,t3_bertimbau_large_bce}/final_training_manifest.json`. A versão CUDA acima caracteriza o build do PyTorch; não comprova a instalação separada de um toolkit CUDA. O valor cuDNN é preservado na forma registrada, sem conversão inferida.

Há diferenças entre fases que impedem tratar essa tabela como um lockfile integral:

- O manifesto de aquisição `metadata/brighter_ptbr/dataset_manifest.json` registra pandas 3.0.5. Os manifests de finalização registram pandas 2.3.3. Não foi encontrado um histórico suficiente para reconstruir todas as transições entre instalações.
- O `pyproject.toml` local define intervalos amplos, incluindo `pandas>=2.1,<3`, `torch>=2.4,<3` e `transformers>=4.45,<6`. Essas restrições não fixam o ambiente histórico e o intervalo de pandas não abrange a versão registrada na aquisição.
- O notebook público registra `metadata.language_info.version=3.14.5` e modo `frozen_artifacts_only`. Esse metadado do notebook de apresentação não substitui o Python 3.12.13 dos registros de treino.
- Não foi localizado um lockfile completo das dependências transitivas nem uma imagem histórica do ambiente. As versões registradas não foram validadas mediante reinstalação nesta auditoria. Um arquivo de versões transcritas deve ser tratado como inventário parcial, não como ambiente reconstruído ou instalação garantida.

## Treinamento dos encoders

As configurações locais `configs/*.yaml`, os `training_provenance.json` dos folds e o código de treino registram o seguinte protocolo. Os manifests finais corroboram quatro épocas, semente e composição dos lotes dos dois componentes do EN2.

| Parâmetro comum | Valor |
|---|---|
| Otimizador | AdamW (`fused=True` no código) |
| Taxa de aprendizado | `2e-5` |
| Weight decay | 0,01 |
| Escalonamento | Linear; warmup de 10% dos passos, arredondado com `round` |
| Épocas | 4 |
| Comprimento máximo | 128 tokens |
| Lote efetivo | 16 |
| Padding | Dinâmico |
| Gradient checkpointing | Habilitado |
| Clipping de gradientes | Norma máxima 1,0 |
| Semente | 42 |

| Encoder | Lote físico | Acumulação | Lote de avaliação no desenvolvimento | Precisão |
|---|---:|---:|---:|---|
| BERTimbau Base | 8 | 2 | 16 | FP16 |
| BERTimbau Large | 1 | 16 | 2 | FP16 |
| BERTweetBR | 8 | 2 | 16 | FP16 |
| ALBERTina PT-BR 100M | 8 | 2 | 16 | FP32 |

As três configurações ALBERTina registram uma exceção técnica por overflow da máscara de atenção DeBERTa sob FP16/BF16. Portanto, a precisão mista FP16 não descreve todos os encoders avaliados. O código usa `GradScaler` no FP16, `num_workers=0` e `pin_memory=True`.

Os componentes finais do EN2 são CT2/T2, BERTweetBR com ASL (`gamma_pos=0`, `gamma_neg=4`, `clip=0.05`), e CT3/T3, BERTimbau Large com `BCEWithLogitsLoss`. As revisões de origem registradas são `7b95f2097a9e37046cca243cc1aeca45b0849fce` e `aa302f6ea73b759f7df9cad58bd272127b67ec28`, respectivamente. O retreino final usa todos os 2.426 exemplos de treino+dev por quatro épocas, sem nova divisão de validação nem early stopping nessa etapa. Os parâmetros finais de fusão e limiares são tratados na documentação metodológica; não foram novamente selecionados para este documento.

Para a baseline TF-IDF + regressão logística, os parâmetros AdamW e de precisão dos encoders não se aplicam. `configs/baseline_tfidf_logreg.yaml` registra palavras com 1–2 gramas e caracteres com 3–5 gramas, máximos de 50.000 e 75.000 features, `min_df=2`, preservação de caixa e acentos, One-vs-Rest, `C=1.0`, solver `liblinear`, máximo de 2.000 iterações e `class_weight=None`.

## Papel dos recursos e limites

Lotes físicos pequenos, acumulação de gradientes e gradient checkpointing permitem usar lote efetivo 16 sob a capacidade nominal de 8 GB da GPU. Os manifests dos retreinos finais registram picos de VRAM de 2.472.825.344 bytes para BERTweetBR+ASL e 5.988.242.944 bytes para BERTimbau Large+BCE. As durações registradas são 138,01396040001418 s e 2.173,4687891000067 s, respectivamente; são observações dessas execuções, não estimativas transferíveis para qualquer máquina.

O código fixa sementes de Python, NumPy e PyTorch, solicita algoritmos determinísticos com `warn_only=True`, habilita cuDNN determinístico e desabilita `benchmark`. Os próprios relatórios alertam que igualdade bit a bit não é garantida entre hardwares, drivers, CUDA, cuDNN ou bibliotecas diferentes. Uma única semente também limita a avaliação de variabilidade do treinamento.

O acervo local conserva predições em Parquet, métricas/manifests/hashes em JSON, tabelas em CSV e relatórios em Markdown. A existência local de código, modelos e dados não comprova sua disponibilidade pública. O notebook contém outputs salvos para inspeção, mas depende de arquivos externos do projeto para reexecutar suas células; a documentação do ambiente não remove essa limitação de reprodução.
