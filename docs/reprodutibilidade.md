# Disponibilidade, reprodução e limites

A distribuição inicial era apenas o notebook. Esta ampliação publica o código científico original, configurações, partições, manifests, agregados, predições sem texto e figuras. O notebook permanece inalterado. Textos BRIGHTER são obtidos da fonte autorizada; checkpoints finais continuam pendentes de hospedagem externa.

## Componentes disponíveis

| Etapa | Componentes publicados |
|---|---|
| Aquisição e Track A | `download.py`, `curation.py`, interfaces originais e novo `scripts/acquire_development.py`, restrito a treino/dev |
| Treinamento inicial | `experiments.py`, `losses.py`, 13 YAML de treino, baselines e interfaces |
| Validação | `cv_folds.json`, 15 manifests internos, OOF externas/internas e registros de exclusão dos IDs externos |
| Limiares e fusão | `threshold_calibration.py`, `ensemble.py`, relatórios CT1–CT3/EN1/EN2 e parâmetros congelados |
| Retreino final | `finalization.py`, duas YAML finais, manifests de treino e rótulos; pesos binários não incluídos |
| Inferência | Código original em `final_test.py` e novo `predict_unlabelled.py`; execução exige pesos/tokenizadores externos e CUDA |
| Resultados | `reports/brighter_ptbr/`, `metadata/brighter_ptbr/`, `artifacts/predictions/` e `figures/brighter_ptbr/` |
| Verificação | Testes originais, guardas dos auxiliares, instalação reconstruída CPU e recibos de verificação |

São 245 arquivos de predição: IDs oficiais, rótulos, probabilidades, decisões, folds e limiares, sem texto bruto ou identificadores de usuários. Os 14 CSV/JSON externos lidos pelo notebook estão presentes. Isso resolve sua ausência inicial, mas não torna o notebook um pipeline autônomo de treinamento. A execução em kernel não foi presumida nem seus outputs substituídos.

## Percursos de reprodução

Para reconstruir os resultados existentes, use a [instalação](instalacao.md), `scripts/check_installation.py`, `scripts/verify_frozen_development.py` e `scripts/rebuild_frozen_reports.py`. O recálculo de F1 lê apenas decisões OOF de desenvolvimento. A regeneração de tabelas usa agregados congelados, inclusive as métricas finais registradas. Não há inferência ou nova avaliação do teste nesse percurso.

Para novo treinamento metodológico, use `scripts/prepare_reproduction_workspace.py`, a aquisição restrita e as etapas do [guia](guia_reproducao.md), sempre na árvore nova. Os scripts históricos resolvem a raiz pelo próprio arquivo: trocar apenas o diretório atual não isola suas saídas. Resultados novos não substituem os originais.

O auxiliar fixa `419566ac8f46f951b51584be36c13e5363008144`, confere os hashes dos shards oficiais de treino/dev e usa as funções originais de seleção Track A. São 2.226 textos de treino e 200 de desenvolvimento, sem aquisição do teste. O procedimento histórico completo permanece disponível, mas inclui todas as partições e não integra a verificação restrita.

Se uma nova serialização Parquet alterar bytes, o padrão é bloquear. Aceitar explicitamente recodificação preserva IDs, ordem e atribuições dos folds e registra fingerprints operacionais novos, retendo os manifests históricos como referências. Não se substituem hashes históricos nem se afirma igualdade bit a bit nesse caso.

## Integridade e privacidade

[source_file_manifest.json](../metadata/publication/source_file_manifest.json) registra hashes de origem/cópia de 652 arquivos. As três diferenças declaradas são dois caminhos pessoais em narrativas de incidente e dois exemplos textuais de um schema. Algoritmos, números científicos, YAML, predições e avaliações foram preservados. Arquivos com textos reais do corpus, material editorial e dados brutos ficaram de fora. IDs/rótulos derivados mantêm atribuição CC-BY-4.0.

`.gitattributes` evita normalizar finais de linha dos arquivos congelados. Há arquivos históricos CRLF e LF; seus bytes devem permanecer compatíveis com os hashes. O notebook mantém o blob `718f6e18f3fe83ed74acd7a0fe8d136818e05d66` e o commit inicial no histórico. Nem todos os backups e checkpoints do workspace integram a publicação.

## Verificação efetivamente executada

O [registro](verificacao_executada.md) discrimina venv, imports, configurações, folds, testes, aquisição restrita, métricas OOF e tabelas realmente verificados. Versões históricas e reconstruídas estão separadas em [Ambiente](ambiente.md) e [Instalação](instalacao.md). Passar essas verificações não equivale a repetir o treinamento Transformer completo ou a inferência final.

## Limitações restantes

1. Os checkpoints CT2/CT3 e tokenizadores/configurações completos precisam de distribuição externa aprovada e conferida. O [catálogo](artefatos_externos.md) registra hashes reais e URLs finais nulas. Pesos-base não substituem pesos ajustados.
2. Não existe imagem comprovada de todo o ambiente histórico e driver. A venv verificada usa Python 3.12.14 e PyTorch CPU, diferentes do runtime histórico de treinamento em GPU.
3. Não foram revalidados treinamento completo em CUDA, duração ou inferência dos checkpoints. Novos resultados podem divergir por runtime, hardware e determinismo parcial; não se promete identidade bit a bit.
4. A finalização original contém constantes de desempenho histórico. Em nova execução, ler os resultados efetivos dos novos relatórios; constantes históricas não são novas medições. O guia explicita isso.
5. A seleção anterior de modelos/perdas não foi integralmente aninhada. Publicar o pipeline não elimina essa limitação, a semente final única ou o suporte pequeno de algumas emoções.
6. A avaliação oficial está encerrada. Estado `TEST_EVALUATED`, autorizações históricas e resultados documentam a execução anterior; não autorizam repeti-la ou ajustar parâmetros nela. A verificação não executa esses comandos.

Código original: [MIT](../LICENSE), autorizado pelo responsável. Dados e modelos: [licenças e atribuições próprias](licencas_dados_modelos.md). Nenhum fonte LaTeX ou PDF do manuscrito foi alterado nesta etapa.
