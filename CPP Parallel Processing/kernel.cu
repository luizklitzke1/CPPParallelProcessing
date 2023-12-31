﻿//Alunos: Arthur B. Pinotti, Gustavo B. Bruder, Kaue Reblin, Luiz G. Klitzke, Rodrigo K. Franco.

#include <random>
#include <stdio.h>
#include <cassert>
#include <chrono>
#include <ppl.h>
#include <iostream>
#include <string>
#include <conio.h>

#include "windows.h"

#include "BenchUtils.h"

//Libs da NVidia para uso de CUDA Cores
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

FILE* fp;

#define BLOCK_SIZE 10 // Mesmo utilizado como tamanho de Tile - logo N deve ser um múltiplo de BLOCK_SIZE

struct Matrix
{
    UINT width  = 0;
    UINT height = 0;
    UINT stride = 0; //Manter tamanho de largura de uma linha em matrix representada em 1xN

    float* elements = nullptr;
};

__device__ float GetElement(const Matrix matrix, const UINT uiRow, const UINT uiCol)
{
    return matrix.elements[uiRow * matrix.stride + uiCol];
}

__device__ void SetElement(Matrix matrix, const UINT uiRow, const UINT uiCol, const float fValue)
{
    matrix.elements[uiRow * matrix.stride + uiCol] = fValue;
}

 __device__ Matrix GetSubMatrix(const Matrix matrix, const UINT uiRow, const UINT uiCol)
{
    Matrix subMatrix;
    subMatrix.width    = BLOCK_SIZE;
    subMatrix.height   = BLOCK_SIZE;
    subMatrix.stride   = matrix.stride;
    subMatrix.elements = &matrix.elements[matrix.stride * BLOCK_SIZE * uiRow
                                                        + BLOCK_SIZE * uiCol];

    return subMatrix;
}

__global__ void KernelMatrixProduct(const Matrix A, const Matrix B, Matrix C)
{
    const UINT uiBlockCol = blockIdx.x;
    const UINT uiBlockRow = blockIdx.y;

    //Submatriz computada pelo bloco
    Matrix subMatrixC = GetSubMatrix(C, uiBlockRow, uiBlockCol);

    float fSoma = 0;

    //Indices da submatriz computados pela thread
    const UINT uiThreadRow = threadIdx.y;
    const UINT uiThreadCol = threadIdx.x;

    for (size_t idxSubMatrix = 0; idxSubMatrix < (A.width / BLOCK_SIZE); ++idxSubMatrix)
    {
        //Sumatrizes de A e B a serem computadas para submatriz de C
        const Matrix subMatrixA = GetSubMatrix(A, uiBlockRow  , idxSubMatrix);
        const Matrix subMatrixB = GetSubMatrix(B, idxSubMatrix, uiBlockCol  );

        //Cache de memória do bloco a ser computado pela thread das submatrizes de A e B
        __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

        sharedA[uiThreadRow][uiThreadCol] = GetElement(subMatrixA, uiThreadRow, uiThreadCol);
        sharedB[uiThreadRow][uiThreadCol] = GetElement(subMatrixB, uiThreadRow, uiThreadCol);

        //Espera todas as threads do bloco carregarem suas sub-matrizes em memória compartilhada
        __syncthreads();

        for (size_t idxItemTile = 0; idxItemTile < BLOCK_SIZE; ++idxItemTile)
        {
            fSoma += sharedA[uiThreadRow][idxItemTile] * sharedB[idxItemTile][uiThreadCol];
        }

        //Espera todas as threads computarem um pedaço para seguir para a próxima
        __syncthreads();
    }

    //Escrever resultado computado localmente para a matrix C global
    SetElement(subMatrixC, uiThreadRow, uiThreadCol, fSoma);
}

cudaError_t CUDAMatrixProduct(const Matrix A, const Matrix B, Matrix C, UINT uiMatrixSize, msTime& processingTime, msTime& fullTime)
{
    fprintf(fp,"\n\n[CUDA CORES - INÍCIO]\n");

    Matrix A_GPU = { 0 };
    Matrix B_GPU = { 0 };
    Matrix C_GPU = { 0 };

    A_GPU.width = A_GPU.height = A_GPU.stride = A.width;
    B_GPU.width = B_GPU.height = B_GPU.stride = B.width;
    C_GPU.width = C_GPU.height = C_GPU.stride = C.width;

    dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid = dim3(std::ceil((float)uiMatrixSize / (float)dimBlock.x),
                        std::ceil((float)uiMatrixSize / (float)dimBlock.y));

    cudaError_t cudaStatus = cudaError_t::cudaSuccess;

    // Definir qual device vai ser utilizado
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(fp,"Erro ao buscar um cudaSetDevice. Verifique se sua GPU é compatível");
        goto FreeCuda;
    }

    // Log de specs do device
    {
        int deviceID;
        cudaDeviceProp devProps;

        cudaStatus = cudaGetDevice(&deviceID);
        if (cudaStatus != cudaSuccess) {
            fprintf(fp,"Erro ao pegar ID do device - cudaGetDevice() - Cod %d - %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
            goto FreeCuda;
        }

        cudaGetDeviceProperties(&devProps, deviceID);
        const int iCUDACores = CBenchUtils::GetCudaCores(devProps);

        fprintf(fp,"\nDevice \"%s\" selecionado.\n", devProps.name);
        fprintf(fp,"CUDA cores: %d\t| Multiprocessadores: %d\t| Warp size: %d\n", iCUDACores, devProps.multiProcessorCount, devProps.warpSize);
        fprintf(fp,"Max Blocks Per MultiProcessor: %d\t| Max Threads per block: %d\n", devProps.maxBlocksPerMultiProcessor, devProps.maxThreadsPerBlock);
        fprintf(fp,"Block Dim : %d - %d - %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
        fprintf(fp,"Grid  Dim: %d - %d - %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    }

    auto clockInicioCuda = std::chrono::high_resolution_clock::now();

    // Alocação de buffer de GPU para os vetores

    {
        cudaStatus = cudaMalloc(&A_GPU.elements, A_GPU.width * A_GPU.height * sizeof(float));
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erro ao alocar memória da matriz A - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc(&B_GPU.elements, B_GPU.width * B_GPU.height * sizeof(float));
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erro ao alocar memória do matriz B - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc(&C_GPU.elements, C_GPU.width * C_GPU.height * sizeof(float));
        if (cudaStatus != cudaSuccess)
        {
            fprintf(fp,"Erro ao alocar memória do matriz C - cudaMalloc()");
            goto FreeCuda;
        }
    }

    // Copiar memória dos vetores para o Buffer da GPU
    { 
        cudaStatus = cudaMemcpy(A_GPU.elements, A.elements, A_GPU.width * A_GPU.height * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erro ao copiar os valores da matriz A - cudaMemcpy()");
            goto FreeCuda;
        }

        cudaStatus = cudaMemcpy(B_GPU.elements, B.elements, B_GPU.width * B_GPU.height * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erro ao copiar os valores da matriz B - cudaMemcpy()");
            goto FreeCuda;
        }
    }

    auto clockInicioProcessamento = std::chrono::high_resolution_clock::now();
    KernelMatrixProduct << <dimGrid, dimBlock>> > (A_GPU, B_GPU, C_GPU);

    //Validar erros na chamada de Kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(fp,"Erro ao executar chamada do kernel - Cod %d - %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto FreeCuda;
    }

    //Validar sincronização dos devices após executar chamada de Kernel
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(fp,"Erro ao executar cudaDeviceSynchronize %d - Cod %d - %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto FreeCuda;
    }

    auto clockFinalProcessamento = std::chrono::high_resolution_clock::now();

    //Copiar dados do buffer de memória da GPU de volta para memória local do host
    cudaStatus = cudaMemcpy(C.elements, C_GPU.elements, uiMatrixSize * uiMatrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(fp,"Erro ao copiar memória do buffer da GPU  - cudaMemcpy()");
        goto FreeCuda;
    }

    auto clockFimCuda = std::chrono::high_resolution_clock::now();

    processingTime = clockFinalProcessamento - clockInicioProcessamento;
    fullTime       = clockFimCuda            - clockInicioCuda        ;

FreeCuda:
    cudaFree(A_GPU.elements);
    cudaFree(B_GPU.elements);
    cudaFree(C_GPU.elements);

    fprintf(fp,"\nTempo apenas de processamento com CUDA cores: %fms\n", processingTime.count());
    fprintf(fp,"Tempo total de processamento e alocação com CUDA cores : %fms\n" , fullTime.count());

    fprintf(fp,"\n[CUDA CORES - FIM]\n");

    return cudaStatus;
}

void LinearMatrixProduct(const Matrix A, const Matrix B, Matrix C, msTime& processingTime)
{
    fprintf(fp,"\n\n[PROCESSAMENTO LINEAR - INÍCIO]\n");

    auto clockInicioLinear = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < C.height; ++i)
    {
        for (int j = 0; j < C.width; ++j)
        {
            float fSum = 0.0f;

            for (int k = 0; k < A.width; ++k)
            {
                fSum += A.elements[i * A.width + k] * B.elements[k * B.width + j];
            }

            C.elements[i * C.width + j] = fSum;
        }
    }

    auto clockFimLinear = std::chrono::high_resolution_clock::now();

    processingTime = clockFimLinear - clockInicioLinear;

    fprintf(fp,"Tempo total de processamento linear: %fms\n", processingTime.count());
    fprintf(fp,"[PROCESSAMENTO LINEAR - FIM]\n");
}

void CPUConcurrencyMatrixProduct(const Matrix A, const Matrix B, Matrix C, msTime& processingTime)
{
    fprintf(fp, "\n\n[PROCESSAMENTO CONCORRENTE EM CPU - INÍCIO]\n");

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    const UINT uiSupportedThreads = std::thread::hardware_concurrency();

    fprintf(fp, "Threads suportadas: %hd\n", uiSupportedThreads);

    auto clockInicio = std::chrono::high_resolution_clock::now();

    Concurrency::parallel_for<int>(0, C.height, [&](int i)
    {
        for (int j = 0; j < C.width; ++j)
        {
            float fSum = 0.0f;

            for (int k = 0; k < A.width; ++k)
            {
                fSum += A.elements[i * A.width + k] * B.elements[k * B.width + j];
            }

            C.elements[i * C.width + j] = fSum;
        }
    });

    auto clockFim = std::chrono::high_resolution_clock::now();

    processingTime = clockFim - clockInicio;

    fprintf(fp,"Tempo total de processamento concorrente em CPU: %fms\n", processingTime.count());
    fprintf(fp,"[PROCESSAMENTO CONCORRENTE EM CPU - FIM]\n");
}

int main(int argc, char **argv)
{
    SetConsoleCP      (1252);
    SetConsoleOutputCP(1252);

    int   iMatrixSizeCFG     = 0;
    float fErroMarginPercent = 0;
    float fErrorMargin       = 0;

    {
        const std::string sTitulo   = "[Benchmark de processamento paralelo]";
        const std::string sOperacao = "Multiplicação de matriz NxN por outra matriz NxN - (N Sendo um número inteiro > 0 e MÚLTIPLO DE " + std::to_string(BLOCK_SIZE) + ")";

        printf("%s\n", sTitulo.c_str());
        printf("\n[Configurações]\nOperação: %s\n", sOperacao.c_str());

        while (std::cout << "Informe o valor de N: " && (!(std::cin >> iMatrixSizeCFG) || iMatrixSizeCFG <= 0 || iMatrixSizeCFG % BLOCK_SIZE))
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //discard input
            std::cout << "Valor inválido\n";
        }

        fErroMarginPercent = 0;
        while (std::cout << "Informe a porcentagem de margem de erro: " && (!(std::cin >> fErroMarginPercent) || fErroMarginPercent < 0))
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //discard input
            std::cout << "Valor inválido\n";
        }

        fErrorMargin = float(fErroMarginPercent) / 100.0f;

        fp = fopen("result.txt", "a");
        fprintf(fp, "%s\n", sTitulo.c_str());
        fprintf(fp, "Executado em: %s\n", CBenchUtils::GetTimeString().c_str());
        fprintf(fp, "\n[Configurações]\nOperação: %s\n\n", sOperacao.c_str());
    }

    fprintf(fp, "Valor de N: %d\n", iMatrixSizeCFG);
    fprintf(fp, "Margem de erro para cálculos de ponto flutuante: %.2f%%%\n\n", float(fErroMarginPercent));

    Matrix vA;
    vA.width  = iMatrixSizeCFG;
    vA.height = iMatrixSizeCFG;

    Matrix vB;
    vB.width  = iMatrixSizeCFG;
    vB.height = iMatrixSizeCFG;

    Matrix vCLinear;
    vCLinear.width  = iMatrixSizeCFG;
    vCLinear.height = iMatrixSizeCFG;

    Matrix vCParaleloCUDA;
    vCParaleloCUDA.width  = iMatrixSizeCFG;
    vCParaleloCUDA.height = iMatrixSizeCFG;

    vA.elements = (float*)malloc(vA.width * vA.height * sizeof(float));
    vB.elements = (float*)malloc(vB.width * vB.height * sizeof(float));

    vCLinear      .elements = (float*)malloc(vCLinear     .width  * vCLinear      .height * sizeof(float));
    vCParaleloCUDA.elements = (float*)malloc(vCParaleloCUDA.width * vCParaleloCUDA.height * sizeof(float));

    // Popular vetores com valores reais aleatórios
    {
        std::random_device device; //Gerar seed
        std::mt19937 rng(device());

        std::uniform_real_distribution<> getRandReal(0.1, 20.0);

        for (int i = 0; i < iMatrixSizeCFG; ++i)
        {
            //vA e vB . elements é uma matrix N * N, porém representada linearmente em memória para reduzir acesso - Coalesced memory
            for (int j = 0; j < iMatrixSizeCFG; ++j)
            {
                vA.elements[i * iMatrixSizeCFG + j] = getRandReal(rng);
                vB.elements[i * iMatrixSizeCFG + j] = getRandReal(rng);
            }
        }
    }

    std::vector<benchResults> aBenchResultsSuccess;
    aBenchResultsSuccess.reserve(3);

    std::vector<std::pair<benchResults, std::string>> aBenchResultsFailure;

    //Processamento paralelo com CUDA cores
    benchResults benchResultsCUDAFull   ;
    benchResults benchResultsCUDAProcess;

    benchResultsCUDAFull   .msTimeElapsed = std::chrono::nanoseconds::zero();
    benchResultsCUDAProcess.msTimeElapsed = std::chrono::nanoseconds::zero();

    try
    {
        benchResultsCUDAFull   .sMethod = "Concorrência em CUDA Cores - Com Alocação";
        benchResultsCUDAProcess.sMethod = "Concorrência em CUDA Cores - Apenas processamento";

        cudaError_t cudaStatus = CUDAMatrixProduct(vA, vB, vCParaleloCUDA, iMatrixSizeCFG, benchResultsCUDAProcess.msTimeElapsed, benchResultsCUDAFull.msTimeElapsed);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(fp, "Erro ao processar em CUDA");
            throw cudaStatus;
        }

        // Limpar devices para evitar erros de profiling
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(fp, "Erro ao executar cudaDeviceReset()");
            throw cudaStatus;
        }

        aBenchResultsSuccess.push_back(benchResultsCUDAFull   );
        aBenchResultsSuccess.push_back(benchResultsCUDAProcess);
    }
    catch (cudaError_t cudaStatus)
    {
        aBenchResultsFailure.push_back(std::make_pair(benchResultsCUDAFull   , cudaGetErrorString(cudaStatus)));
        aBenchResultsFailure.push_back(std::make_pair(benchResultsCUDAProcess, cudaGetErrorString(cudaStatus)));
    }

    //Processamento linear em CPU
    benchResults benchResultsLinear;
    benchResultsLinear.msTimeElapsed = std::chrono::nanoseconds::zero();

    try
    {
        benchResultsLinear.sMethod = "Linear em CPU";
        LinearMatrixProduct(vA, vB, vCLinear, benchResultsLinear.msTimeElapsed);
        aBenchResultsSuccess.push_back(benchResultsLinear);
    }
    catch (...)
    {
        aBenchResultsFailure.push_back(std::make_pair(benchResultsLinear, "Desconhecido"));
    }

    //Processamento com concorrencia em CPU
    benchResults benchResultsCPUThreads;
    benchResultsCPUThreads.msTimeElapsed = std::chrono::nanoseconds::zero();

    try
    {
        benchResultsCPUThreads.sMethod = "Concorrência em Threads de CPU";
        CPUConcurrencyMatrixProduct(vA, vB, vCLinear, benchResultsCPUThreads.msTimeElapsed);
        aBenchResultsSuccess.push_back(benchResultsCPUThreads);
    }
    catch (...)
    {
        aBenchResultsFailure.push_back(std::make_pair(benchResultsCPUThreads, "Desconhecido"));
    }

    if (aBenchResultsFailure.size() == 0)
    {
        bool bValoresDiferem = false;

        for (int i = 0; i < iMatrixSizeCFG * iMatrixSizeCFG; ++i)
        {
            const float& fLinear       = vCLinear      .elements[i];
            const float& fParaleloCUDA = vCParaleloCUDA.elements[i];

            const float fDif              = fabs(fParaleloCUDA - fLinear);
            const float fErrorMarginValue = fLinear * fErrorMargin;

            if (fDif > fErrorMargin)
            {
                fprintf(fp, "\n\nDIFERENÇA DE VALORES FORA DA MARGEM DE ERRO DE FLOAT - idx %d - [%d][%d]\nCPU: %f\nGPU: %f\n", i, i / iMatrixSizeCFG, i - (i / iMatrixSizeCFG * iMatrixSizeCFG), fLinear, fParaleloCUDA);
                fprintf(fp, "\Diferença %f  - Margem: %f", fDif, fErrorMargin);
                bValoresDiferem = true;
            }
        }

        if (bValoresDiferem == false)
        {
            fprintf(fp, "\nNenhum resultado dos métodos diferiu da margem de erro de ponto flutuante! (%.2f%%%)\n", float(fErroMarginPercent));
        }
    }

    //Liberar valores dos ponteiros de matrizes
    {
        free(vA.elements);
        free(vB.elements);

        free(vCLinear      .elements);
        free(vCParaleloCUDA.elements);
    }

    std::sort(aBenchResultsSuccess.begin(), aBenchResultsSuccess.end(), [](const benchResults& lhs, const benchResults& rhs)
        {
            return lhs.msTimeElapsed.count() < rhs.msTimeElapsed.count();
        });

    fprintf(fp, "\n\n[RESULTADOS]\n");
    fprintf(fp, "\nTempo de execução:\n");

    fprintf(fp, "|%s | %-55s | %-25s | %-27s\n", "Pos", "Método", "Tempo exec.", "Dif");
    
    for (int i = 0; i < aBenchResultsSuccess.size(); ++i)
    {
        const benchResults& benchResult = aBenchResultsSuccess[i];
        fprintf(fp, "|%d   | %-55s | %-24.6fms| +%-24.6fms\n", i + 1, benchResult.sMethod.c_str(), benchResult.msTimeElapsed.count(),
                                                                      benchResult.msTimeElapsed.count() - aBenchResultsSuccess.begin()->msTimeElapsed.count());
    }

    if (aBenchResultsFailure.size())
    {
        fprintf(fp, "\n\nMétodos com erro de execução:\n");

        for (const auto& faileure : aBenchResultsFailure)
            fprintf(fp, "%-55s - %s\n", faileure.first.sMethod.c_str(), faileure.second.c_str());
    }
    else
    {
        fprintf(fp, "\n\nNenhum método apresentou erro!\n");
    }

    fprintf(fp, "\n%s\n", std::string(120, '-').c_str());

    fclose(fp);

    printf("\nExecução finalizada, resultados salvos no arquivo results.txt, presente no mesmo path desse executável.\n\nInforme qualquer tecla para fechar o terminal.\n");
    _getch();

    return 0;
}
