# PLN — Processamento de Linguagem Natural

Projeto da disciplina de **Inteligência Artificial** (5º semestre — UDESC), voltado a estudos e atividades de **Processamento de Linguagem Natural (PLN)**.

## Tecnologias

- **Python 3.13**
- [NLTK](https://www.nltk.org/) — toolkit clássico de PLN
- [spaCy](https://spacy.io/) — PLN industrial (tokenização, POS, NER)
- [scikit-learn](https://scikit-learn.org/) — machine learning
- [sentence-transformers](https://www.sbert.net/) — embeddings de sentenças (modelos multilíngues)
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — manipulação de dados
- [matplotlib](https://matplotlib.org/) — visualização
- [Jupyter](https://jupyter.org/) — notebooks

## Como iniciar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/anaalthoff/pln.git
cd pln
```

### 2. Criar o ambiente virtual (venv)

O **venv** isola as dependências do projeto, evitando conflitos com outros projetos ou com o Python global da máquina. A pasta `.venv/` **não é versionada** (está no `.gitignore`) — cada pessoa recria a sua localmente.

```bash
python -m venv .venv
```

### 3. Ativar o ambiente virtual

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

> Se o PowerShell bloquear a execução do script, libere para a sessão atual:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

**Windows (CMD):**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

Quando ativado, aparece `(.venv)` no início da linha do terminal. Para sair, use `deactivate`.

### 4. Instalar as dependências

Com o venv **ativado**, instale tudo que está listado no [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

### 5. Recursos adicionais

Algumas bibliotecas baixam dados de idioma separadamente:

**Modelo de português do spaCy:**
```bash
python -m spacy download pt_core_news_sm
```

**Recursos do NLTK** (executar dentro do código Python conforme a necessidade):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Sobre o `requirements.txt`

O arquivo lista as dependências do projeto. Para **adicionar** uma nova biblioteca:

```bash
pip install nome-da-biblioteca
```

E então registre-a no arquivo (manualmente ou congelando as versões instaladas):

```bash
pip freeze > requirements.txt
```

Assim qualquer pessoa que clonar o projeto consegue reproduzir o mesmo ambiente com `pip install -r requirements.txt`.

## Estrutura

```
pln/
├── .gitignore                       # arquivos ignorados pelo Git (inclui .venv/)
├── README.md                        # este arquivo
├── requirements.txt                 # dependências do projeto
├── tokenizacao.py                   # tokenização com NLTK e spaCy
├── tarefas_basicas.py               # tarefas básicas de PLN (POS, NER, similaridade)
├── stop_words_e_normalizacao.py     # stop words, normalização e embeddings
└── sistema_de_busca.py              # sistema de busca semântica (similaridade + t-SNE)
```
