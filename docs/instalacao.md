# Instalação e distinção entre ambientes

O ambiente de verificação desta distribuição foi instalado em uma venv nova, com **Python 3.12.14 no Windows e PyTorch 2.13.0+cpu**. Ele foi usado para imports, validação de configurações, testes sintéticos, leitura de OOF de desenvolvimento e geração de tabelas a partir de agregados congelados. Não é o ambiente histórico de GPU, não contém os pesos treinados e não foi usado para repetir o teste oficial.

## Ambiente reconstruído verificado

Na raiz de um clone novo:

```bash
python -m venv .venv
```

Use Python 3.12 para esse perfil. Ative a venv conforme o sistema:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS: adaptação não verificada nesta auditoria
source .venv/bin/activate
```

Depois:

```bash
python -m pip install -r requirements/torch-cpu.txt
python -m pip install -r requirements/verified-cpu-py312-windows.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/check_installation.py
```

O arquivo [verified-cpu-py312-windows.txt](../requirements/verified-cpu-py312-windows.txt) registra as versões instaladas na reconstrução, excluindo o caminho da instalação editable. [torch-cpu.txt](../requirements/torch-cpu.txt) usa o índice oficial CPU do PyTorch. Esse perfil foi verificado no Windows; não se afirma que o mesmo conjunto de wheels esteja disponível ou tenha sido testado em todos os sistemas.

[reconstructed-cpu.in](../requirements/reconstructed-cpu.in) registra as entradas usadas na resolução inicial: versões dos registros finais onde comprovadas e intervalos originais para dependências restantes. Ele não é a fixação transitiva completa. O arquivo com as versões efetivamente instaladas é um retrato da **verificação nova de 26/10/2026**, não um lockfile histórico dos experimentos.

## Ambiente histórico

Os registros de treino informam Python 3.12.13, PyTorch 2.13.0+cu130, CUDA reportada pelo build 13.0, GPU RTX 3060 Ti e outras versões descritas em [Ambiente](ambiente.md). A aquisição registra pandas 3.0.5 e a finalização, pandas 2.3.3. Não há um freeze transitivo histórico único demonstrado. O metadado Python 3.14.5 do notebook é da apresentação dos artefatos.

O perfil reconstruído conserva versões comprovadas de NumPy 1.26.4, pandas 2.3.3, PyArrow 23.0.1, Transformers 5.16.1, tokenizers 0.23.1, scikit-learn 1.9.0, Hugging Face Hub 1.28.0, datasets 4.8.5 e iterative-stratification 0.1.9. A versão de Python e o build CPU de PyTorch diferem dos treinos. Dependências adicionais, como pytest, SciPy, Matplotlib e bibliotecas de apresentação, estão fixadas somente como parte da reconstrução testada.

## Treinamento e CUDA

Os entrypoints Transformers históricos exigem CUDA e usam AdamW fused, FP16 nos BERTimbau/BERTweetBR e FP32 no ALBERTina. O perfil CPU acima não habilita nem valida esse treinamento. Para uma nova execução de desenvolvimento, prepare outra venv compatível com sua GPU e consulte a [instalação oficial do PyTorch](https://pytorch.org/get-started/locally/); registre o build e as versões efetivas. As dependências de projeto continuam em [pyproject.toml](../pyproject.toml), preservado do original.

Não foi reconstruída uma imagem histórica de driver/toolkit, nem testado o treinamento completo em CUDA. Não se promete identidade bit a bit entre um novo treinamento e os pesos históricos. Use as árvores isoladas e os passos do [guia](guia_reproducao.md), preservando os resultados publicados.

## Notebook

IPython, nbformat e nbclient estão incluídos no perfil reconstruído para suporte à leitura e às ferramentas de apresentação. Não há obrigação de executar o notebook para consultar seus outputs salvos. Sua primeira célula requer os metadados e relatórios agora distribuídos; o notebook histórico permanece inalterado. A execução de um kernel Jupyter é uma capacidade adicional e não é presumida como verificada apenas porque nbclient foi instalado.
