# Breast Cancer Classification with Perceptron

Esse código utiliza o algoritmo **Perceptron** da biblioteca `scikit-learn` para realizar a classificação de dados referentes ao diagnóstico de câncer de mama, com base em um dataset chamado `breast-cancer-wisconsin.data`

## Requisitos

- Python 3.11+
- Acesso ao Github
- Docker

## Instalação

#### Caso esteja instalando em um container, o arquivo Dockerfile funcional está na raiz do projeto:

1. Clone o repositório:

   git clone https://github.com/joao-tdias/dti-challenge.git
2. Crie a Imagem:

   docker build -t dti-challenge .
3. Rode o container:

   docker run dti-challenge

#### Caso esteja instalando em uma VM:

1. Clone o repositório:

   git clone https://github.com/joao-tdias/dti-challenge.git
2. Crie um ambiente virtual:

   python -m venv venv
3. Ative o ambiente virtual:

   ###### MAC/linux>

   source venv/bin/activate

   ###### Windows>

   venv\Scripts\activate
4. Instale as dependências:

   pip install -r requirements.txt


## Testes

Para fazer os testes automatizados nesse código foi utilizada a biblioteca **pytest.** Para executar os testes, use o seguinte comando:

pytest -v

## Uso

Inicializando a Aplicação:

- Para inicializar a aplicação, use o seguinte comando:

  python main.py

## Logs

Para observabilidade desse código foi utilizada a biblioteca ***loguru***. Você pode encontrar os logs da aplicação no diretório app/logs/logs.log em nível INFO e também ao executar o programa você verá os logs no terminal em nível DEBUG.
