# Checkpoints finais e artefatos externos

Os pesos ajustados finais de CT2/T2 e CT3/T3 permanecem preservados no acervo experimental e **não foram enviados ao Hugging Face, Git LFS, GitHub Releases ou Git convencional nesta atualização**. O [catálogo JSON](../metadata/publication/external_artifacts.json) registra os arquivos necessários, seus tamanhos e SHA-256, com URLs de distribuição nulas. Não interprete os endereços dos modelos-base como download dos checkpoints ajustados.

## Componentes necessários à inferência histórica

| Componente | Checkpoint final | Bytes | SHA-256 |
|---|---|---:|---|
| CT2/T2: BERTweetBR+ASL | `t2_bertweetbr_asl/model.safetensors` | 539.654.760 | `5b845ab4988c02ed8bbf1d1fee7551cd64cdbdac213c19f5ebd24823e79a612a` |
| CT3/T3: BERTimbau Large+BCE | `t3_bertimbau_large_bce/model.safetensors` | 1.337.657.272 | `aa25ca2517e262281ab60fee0fd161a48b57fc46967c0bba7955e03c98f4255d` |

Os dois diretórios finais completos, incluindo pesos e 14 arquivos menores, somam **1.879.672.746 bytes**. Os bytes dos 16 arquivos foram re-hasheados em 26/10/2026: os hashes dos pesos e arquivos com referência histórica concordam com os manifests; os demais arquivos menores têm seu hash documental registrado no catálogo. Essa verificação lê bytes e não carrega modelos ou executa inferência.

O pacote completo de cada componente precisa incluir os arquivos listados no catálogo: configurações, ordem dos rótulos, tokenizador/vocabulário, manifesto e configuração do treinamento, além do safetensors. Não publicar apenas os pesos deixando implicitamente dependentes os arquivos de configuração. A futura instalação deve recuperar esses arquivos nos caminhos relativos esperados, sob `artifacts/models/final/`, em uma cópia de reprodução, conferindo SHA-256 antes do uso.

A inferência EN2 requer os dois componentes. A regra final congelada usa pesos CT2/CT3 de 0,70/0,30 e os limiares 0,67; 0,36; 0,43; 0,65; 0,58; 0,49, na ordem anger/disgust/fear/joy/sadness/surprise. As configurações e manifests publicados documentam essa regra; eles não contêm os parâmetros neurais ajustados.

## Modelos-base já acessíveis

As revisões dos modelos-base permanecem acessíveis nas fontes originais:

- [BERTweetBR, revisão usada por CT2](https://huggingface.co/melll-uff/bertweetbr/tree/7b95f2097a9e37046cca243cc1aeca45b0849fce).
- [BERTimbau Large, revisão usada por CT3](https://huggingface.co/neuralmind/bert-large-portuguese-cased/tree/aa302f6ea73b759f7df9cad58bd272127b67ec28).
- [BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased/tree/94d69c95f98f7d5b2a8700c420230ae10def0baa).
- [ALBERTina 100M PT-BR](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder/tree/d2a39d3d82408cc2a2618beb57a4fe908ee44b27).

Usar a revisão atual ou o modelo-base sem ajuste não recupera os checkpoints históricos finais. Um novo treinamento pode reconstruir metodologicamente o procedimento, mas não é apresentado como garantia de identidade bit a bit com os pesos históricos.

## Opções de hospedagem, sujeitas à decisão dos responsáveis

O [GitHub bloqueia arquivos maiores que 100 MiB no Git convencional](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github); ambos os checkpoints excedem esse limite. As alternativas são:

| Opção | Organização possível | Verificações necessárias |
|---|---|---|
| Hugging Face Model Hub | Dois repositórios de modelos ou um repositório com dois componentes; card com protocolo e limitações; referência imutável por commit | Conta/organização e nomes aprovados; licenças/avisos; quota e regras vigentes; download e SHA-256 depois do upload |
| Git LFS | Somente os artefatos volumosos explicitamente aprovados no repositório GitHub existente | Limites e custos do plano; LFS configurado; instruções de obtenção; licenças/avisos e hashes |
| GitHub Releases | Pacotes de CT2 e CT3 separados, associados a uma versão e acompanhados de manifests | Limites de assets vigentes; release versionada; licenças/avisos; URLs e hashes reais |

O [Model Hub oferece versionamento e cards para modelos](https://huggingface.co/docs/hub/en/models-uploading), sendo uma opção adequada para os checkpoints científicos. Nenhum destino foi criado ou escolhido automaticamente. Custos e quotas precisam ser conferidos na conta responsável antes da escolha. As [condições de licença](licencas_dados_modelos.md) continuam aplicáveis a qualquer alternativa.

Após uma publicação externa autorizada, substituir no catálogo as URLs nulas por links reais fixos, registrar o commit/versão externa, fornecer um comando de aquisição que não grave sobre o acervo original e conferir cada arquivo baixado contra os hashes históricos. Essa conferência de integridade não exige nem autoriza repetir o teste oficial.

## Limitação atual

A ausência dos checkpoints finais impede repetir a inferência histórica com os mesmos pesos a partir apenas deste GitHub. Os resultados e predições congelados podem ser inspecionados e utilizados conforme as instruções autorizadas; a disponibilidade de código, partições, configurações e aquisição dos dados permite reconstruir o protocolo. Nenhuma dessas disponibilidades deve ser apresentada como distribuição já concluída dos pesos finais ou como nova avaliação oficial de teste.
