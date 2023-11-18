﻿
#include <random>
#include <stdio.h>
#include <cassert>
#include <chrono>
#include <ppl.h>
#include <iostream>
#include <string>

#include "windows.h"

//Libs da NVidia para uso de CUDA Cores
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#include "BenchUtils.h"

FILE* fp;

struct Matrix
{
    int width = 0;
    int height = 0;
    int stride = 0;
    float* elements;
};

#define BLOCK_SIZE 32

__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

__global__ void KernelMatrixVectorProduct(const Matrix A, const Matrix B, Matrix C)
{
    const UINT uiBlockCol = blockIdx.x;
    const UINT uiBlockRow = blockIdx.y;

    //Submatriz computada pelo bloco
    Matrix subMatrixC = GetSubMatrix(C, uiBlockRow, uiBlockCol);

    float fSoma = 0;

    //Indices da da submatriz computados pela thread
    const UINT uiRowSub = threadIdx.y;
    const UINT uiColSub = threadIdx.x;

    for (size_t idxSubMatrix = 0; idxSubMatrix < (A.width / BLOCK_SIZE); ++idxSubMatrix)
    {
        //Sumatrizes de A e B a serem computados para submatriz de C
        const Matrix subMatrixA = GetSubMatrix(A, uiBlockRow  , idxSubMatrix);
        const Matrix subMatrixB = GetSubMatrix(B, idxSubMatrix, uiBlockCol  );

        //Cache de memória do bloco a ser computado pela thread das submatrizes de A e B
        //Sincronizado entre as threads do bloco para evitar acesso à memória global e overhead
        __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
        sharedA[uiRowSub][uiColSub] = GetElement(subMatrixA, uiRowSub, uiColSub);
        sharedB[uiRowSub][uiColSub] = GetElement(subMatrixB, uiRowSub, uiColSub);

        __syncthreads();

        for (size_t idxItemTile = 0; idxItemTile < BLOCK_SIZE; ++idxItemTile)
        {
            fSoma += sharedA[uiRowSub][idxItemTile] * sharedB[idxItemTile][uiColSub];
        }

        __syncthreads();
    }

    //Escrever resultado computado localmente para a matrix C global
    SetElement(subMatrixC, uiRowSub, uiColSub, fSoma);
}

cudaError_t CUDAMatrixVectorProduct(const Matrix A, const Matrix B, Matrix C, UINT uiMatrixSize, msTime& processingTime, msTime& fullTime)
{
    fprintf(fp,"\n\n[CUDA CORES - INÍCIO]\n");

    Matrix A_GPU;
    Matrix B_GPU;
    Matrix C_GPU;

    dim3 block_shape = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_shape  = dim3(std::ceil((float)uiMatrixSize / (float)block_shape.x),
                            std::ceil((float)uiMatrixSize / (float)block_shape.y));

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
        fprintf(fp,"Block Shape: %d - %d - %d\n", block_shape.x, block_shape.y, block_shape.z);
        fprintf(fp,"Grid  Shape: %d - %d - %d\n", grid_shape .x, grid_shape .y, grid_shape .z);
    }

    auto clockInicioCuda = std::chrono::high_resolution_clock::now();

    // Alocação de buffer de GPU para os vetores
    {
        cudaStatus = cudaMalloc((void**)&A_GPU, A_GPU.width * A_GPU.height * sizeof(float));
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erroi ao alocar memória da matriz A - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc((void**)&B_GPU, B_GPU.width * B_GPU.height * sizeof(float));
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erro ao alocar memória do matriz B - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc((void**)&C_GPU, C_GPU.width * C_GPU.height * sizeof(float));
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
    KernelMatrixVectorProduct << <grid_shape, block_shape >> > (A_GPU, B_GPU, C_GPU);

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
    cudaStatus = cudaMemcpy(C.elements, C_GPU.elements, uiMatrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(fp,"Erro ao copiar memória do buffer da GPU  - cudaMemcpy()");
        goto FreeCuda;
    }

FreeCuda:
    cudaFree(A_GPU.elements);
    cudaFree(B_GPU.elements);
    cudaFree(C_GPU.elements);

    auto clockFimCuda = std::chrono::high_resolution_clock::now();

    processingTime = clockFinalProcessamento - clockInicioProcessamento;
    fullTime       = clockFimCuda            - clockInicioCuda        ;

    fprintf(fp,"\nTempo apenas de processamento com CUDA cores: %fms\n", processingTime.count());
    fprintf(fp,"Tempo total de processamento e alocação com CUDA cores : %fms\n" , fullTime.count());

    fprintf(fp,"[CUDA CORES - FIM]\n");

    return cudaStatus;
}

void LinearMatrixVectorProduct(float *A, float* v1, float* v2, UINT uiMatrixSize, msTime& processingTime)
{
    fprintf(fp,"\n\n[PROCESSAMENTO LINEAR - INÍCIO]\n");

    auto clockInicioLinear = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < uiMatrixSize; ++i)
    {
        float fSum = 0.0f;

        for (int j = 0; j < uiMatrixSize; ++j)
        {
            fSum += A[i * uiMatrixSize + j] * v1[j];
        }

        v2[i] = fSum;
    }

    auto clockFimLinear = std::chrono::high_resolution_clock::now();

    processingTime = clockFimLinear - clockInicioLinear;

    fprintf(fp,"Tempo total de processamento linear: %fms\n", processingTime.count());
    fprintf(fp,"[PROCESSAMENTO LINEAR - FIM]\n");
}

void CPUConcurrencyMatrixVectorProduct(float* A, float* v1, float* v2, UINT uiMatrixSize, msTime& processingTime)
{
    fprintf(fp,"\n\n[PROCESSAMENTO CONCORRENTE EM CPU - INÍCIO]\n");

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    const UINT uiSupportedThreads = std::thread::hardware_concurrency();

    fprintf(fp, "Threads pro core: %hd\n", uiSupportedThreads);

    auto clockInicio = std::chrono::high_resolution_clock::now();

    Concurrency::parallel_for<int>(0, uiMatrixSize, [&](int i)
    {
        float fSum = 0.0f;

        for (int j = 0; j < uiMatrixSize; ++j)
        {
            fSum += A[i * uiMatrixSize + j] * v1[j];
        }

        v2[i] = fSum;
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

    UINT uiMatrixSizeCFG = 0;

    {
        const std::string sTitulo = "[Benchmark de processamento paralelo]";
        const std::string sOperacao = "Multiplicação de matriz NxN por vetor N - (N Sendo um número inteiro > 0)";

        printf("%s\n", sTitulo.c_str());
        printf("\n[Configurações]\nOperação: %s\n", sOperacao.c_str());

        while (std::cout << "Informe o valor de N: " && !(std::cin >> uiMatrixSizeCFG)) {
            std::cin.clear(); //clear bad input flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //discard input
            std::cout << "Valor inválido\n";
        }

        fp = fopen("result.txt", "a");
        fprintf(fp, "%s\n", sTitulo.c_str());
        fprintf(fp, "\n[Configurações]\nOperação: %s\n", sOperacao.c_str());
    }

    fprintf(fp, "Valor de N: %d\n", uiMatrixSizeCFG);

    Matrix vA;
    vA.width  = uiMatrixSizeCFG;
    vA.height = uiMatrixSizeCFG;

    Matrix vB;
    vB.width  = uiMatrixSizeCFG;
    vB.height = uiMatrixSizeCFG;

    Matrix vC;
    vC.width  = uiMatrixSizeCFG;
    vC.height = uiMatrixSizeCFG;

    vA.elements = (float*)malloc(vC.width * vC.height * sizeof(float));
    vB.elements = (float*)malloc(vC.width * vC.height * sizeof(float));
    vC.elements = (float*)malloc(vC.width * vC.height * sizeof(float));

    // Popular vetores com valores reais aleatórios
    {
        std::random_device device; //Gerar seed
        std::mt19937 rng(device());

        std::uniform_real_distribution<> getRandReal(0.1, 999.9);

        for (int i = 0; i < uiMatrixSizeCFG; ++i)
        {
            //Memoryu for coalesced access
            //vA e vB . elements é uma matrix N * N, porém representada linearmente para facilitar blocos de CUDA posteriormente
            for (int j = 0; j < uiMatrixSizeCFG; ++j)
            {
                vA.elements[i * uiMatrixSizeCFG + j] = getRandReal(rng);
                vB.elements[i * uiMatrixSizeCFG + j] = getRandReal(rng);
            }
        }
    }

    std::vector<benchResults> aBenchResultsSuccess;
    aBenchResultsSuccess.reserve(3);

    std::vector<std::pair<benchResults, std::string>> aBenchResultsFailure;

    //Processamento linear em CPU
    benchResults benchResultsLinear;

    try
    {
        benchResultsLinear.sMethod = "Linear em CPU";
        //LinearMatrixVectorProduct(vA, vB, vC, uiMatrixSizeCFG, benchResultsLinear.msTimeElapsed);
        aBenchResultsSuccess.push_back(benchResultsLinear);
    }
    catch (...)
    {
        aBenchResultsFailure.push_back(std::make_pair(benchResultsLinear, "Desconhecido"));
    }

    //Processamento com concorrencia em CPU
    benchResults benchResultsCPUThreads;
    try
    {
        benchResultsCPUThreads.sMethod = "Concorrência em Threads de CPU";
        //CPUConcurrencyMatrixVectorProduct(A, v1, v2, uiMatrixSizeCFG, benchResultsCPUThreads.msTimeElapsed);
        aBenchResultsSuccess.push_back(benchResultsCPUThreads);
    }
    catch (...)
    {
        aBenchResultsFailure.push_back(std::make_pair(benchResultsCPUThreads, "Desconhecido"));
    }

    //Processamento paralelo com CUDA cores
    benchResults benchResultsCUDAFull   ;
    benchResults benchResultsCUDAProcess;
    try
    {
        benchResultsCUDAFull   .sMethod = "Concorrência em CUDA Cores - Com Alocação";
        benchResultsCUDAProcess.sMethod = "Concorrência em CUDA Cores - Apenas processamento";

        cudaError_t cudaStatus = CUDAMatrixVectorProduct(vA, vB, vC, uiMatrixSizeCFG, benchResultsCUDAProcess.msTimeElapsed, benchResultsCUDAFull.msTimeElapsed);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(fp,"Erro ao processar soma em CUDA");
            throw cudaStatus;
        }

        // Limpar devices para evitar erros de profiling
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(fp,"Erro ao executar cudaDeviceReset()");
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

    //Liberar valores dos ponteiros de matrizes
    {
        free(vA.elements);
        free(vB.elements);
        free(vC.elements);
    }

    std::sort(aBenchResultsSuccess.begin(), aBenchResultsSuccess.end(), [](const benchResults& lhs, const benchResults& rhs)
        {
            return lhs.msTimeElapsed.count() < rhs.msTimeElapsed.count();
        });

    fprintf(fp, "\n\n[RESULTADOS]\n");
    fprintf(fp, "\nTempo de execução:\n");

    fprintf(fp, "|%s | %-55s | %-15s | %-17s\n", "Pos", "Método", "Tempo exec.", "Dif");
    
    for (int i = 0; i < aBenchResultsSuccess.size(); ++i)
    {
        const benchResults& benchResult = aBenchResultsSuccess[i];
        fprintf(fp, "|%d   | %-55s | %-14.6fms| +%-14.6fms\n", i + 1, benchResult.sMethod.c_str(), benchResult.msTimeElapsed.count(),
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

    fprintf(fp, "\n%s\n", std::string(110, '-').c_str());

    fclose(fp);

    return 0;
}
