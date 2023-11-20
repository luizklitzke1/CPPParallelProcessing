# Benchmark de processamento paralelo em C++

Alunos: Arthur B. Pinotti, Gustavo B. Bruder, Kaue Reblin, Luiz G. Klitzke, Rodrigo K. Franco.

Esse repositório apresenta uma aplicação simples que busca apresentar implementações de diferentes métodos de resolver o processamento de uma multiplicação de matriz NxN por outra matriz NxN - *(N Sendo um número inteiro > 0 e MULTIPLO DE 10*

Sendo eles:

- Processamento linear através de 3 laços de repetição simples encadeados.
- Processamento com paralelização em threads da CPU utilizando recursos da biblioteca padrão *Concurrency* do C++.
- Processamento paralelizado com lógica otimizada para utilização de CUDA Cores de um device compatível.

## Resultados da execução
As configurações selecionadas, juntamente com detalhes específicos logados sobre cada tipo de execução ficam disponíveis em um arquivo chamado "results.txt", criado e atualizado no mesmo diretório aonde está o executável da aplicação: "CPP Parallel Processing.exe".

### Configuração da função de kernel para GPU

Ao executar o processamento com CUDA cores, é logada a configuração do device encontrado, ou uma mensagem de erro caso o computador não possua nenhuma GPU compatível.

Dependendo da massa de dados informada, é montada também o tamanho da grid de blocos e a configuração das threads de cada bloco.

```
[CUDA CORES - INÍCIO]

Device "NVIDIA GeForce GTX 1070 Ti" selecionado.
CUDA cores: 2432	| Multiprocessadores: 19	| Warp size: 32
Max Blocks Per MultiProcessor: 32	| Max Threads per block: 1024
Block Dim : 10 - 10 - 1
Grid  Dim: 100 - 100 - 1

Tempo apenas de processamento com CUDA cores: 3.545800ms
Tempo total de processamento e alocação com CUDA cores : 7.247700ms

[CUDA CORES - FIM]
```

### Tempo de execução

Ao final, podem ser consultados e comparados os tempos de execução dos diferentes métodos, como por exemplo:

```
[RESULTADOS]

Tempo de execução:
|Pos | Método                                                  | Tempo exec.               | Dif                        
|1   | Concorrência em CUDA Cores - Apenas processamento       | 3.545800                ms| +0.000000                ms
|2   | Concorrência em CUDA Cores - Com Alocação               | 7.247700                ms| +3.701900                ms
|3   | Concorrência em Threads de CPU                          | 378.142100              ms| +374.596300              ms
|4   | Linear em CPU                                           | 1139.133800             ms| +1135.588000             ms


Nenhum método apresentou erro!
```

### Diferença de valores

No final do processamento, caso não tenham ocorrido erros no processamento, os valores resultantes das operações são comparados.
Foi definida  uma margem de erro de 5% entre os valores de ponto flutuante resultados do processamento da CPU e da GPU, devido à divergência inerte de processamento e representação desse tipo de dado.

Não foram utilizadas variáveis do tipo double, de maior precisão, devido à limitação de algumas GPUs processarem esse tipo de dado.

En caso de erros, as diferenças são logadas no arquivo result.txt, juntamente com o quanto diferem da original. 

Vale ressaltar que essa margem foi adotada para acomodar ranges maiores de N, uma vez que a disparidade cresce conforme a quantidade de operações realizadas.
