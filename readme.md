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

   docker build -t dti .
3. Rode o container:

   docker run --name dti-challenge --env=PYTHON_VERSION=3.11 --env=PYTHON_PIP_VERSION=23.2.1 --env=PYTHON_GET_PIP_URL=https://raw.githubusercontent.com/pypa/get-pip/049c52c665e8c5fd1751f942316e0a5c777d304f/public/get-pip.py --workdir=/app -p 8080:5000 --runtime=runc -d dti:latest
4. Acesse a aplicação:

   O docker está rodando na porta 8080 da sua máquina, portanto, basta acessar [127.0.0.1:8080/](http://127.0.0.1:8080/run) no seu navegador

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

  python flask_app.py

## Logs

Para observabilidade desse código foi utilizada a biblioteca ***loguru***. Você pode encontrar os logs da aplicação no diretório app/logs/logs.log em nível INFO e também ao executar o programa você verá os logs no terminal em nível DEBUG.


## Rotas

Essa aplicação utiliza 3 rotas, run, train e predict

#### run

A rota run executa o código base sem a necessidade da passagem de nenhum parâmetro. 

Basta acessar [127.0.0.1:8080/](http://127.0.0.1:8080/run)run e o resultado da execução do código com os parâmetros base irá ser calculado

#### train

A rota train executa a rotina de treinar o modelo, ela recebe os parâmetros test_size e eta0 como body da seguinte forma: {"test_size": 0.3, "eta0": 0.1}.

#### predict

A rota predict executa a predição utilizando o modelo já treinado, essa rota recebe os parâmetros em array dentro de um dicionario data, como body da seguinte forma: {"data": [[999999, 5, 10, 10, 3, 7, 3, 8, 10, 2]]}
