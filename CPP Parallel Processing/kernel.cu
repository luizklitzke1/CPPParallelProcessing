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

__global__ void KernelMatrixVectorProduct(float* A, float* v1, float* v2, UINT uiMatrixSize)
{
    const int iMatrixRow = blockIdx.x * blockDim.x + threadIdx.x;
    const int iMatrixCol = blockIdx.y * blockDim.y + threadIdx.y;

    if (iMatrixCol == 0 && iMatrixRow < uiMatrixSize)
    {
        float fSum = 0.0f;

        for (int i = 0; i < uiMatrixSize; ++i)
        {
            fSum += A[iMatrixRow * uiMatrixSize + i] * v1[i];
        }

        v2[iMatrixRow] = fSum;
    }
}

cudaError_t CUDAMatrixVectorProduct(float* A, float* v1, float* v2, UINT uiMatrixSize, msTime& processingTime, msTime& fullTime)
{
    fprintf(fp,"\n\n[CUDA CORES - INÍCIO]\n");

    float* A_GPU ;
    float* v1_GPU;
    float* v2_GPU;

    dim3 block_shape = dim3(32, 32, 1);
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
        cudaStatus = cudaMalloc((void**)&A_GPU, uiMatrixSize * uiMatrixSize * sizeof(float));
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erroi ao alocar memória da matriz A - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc((void**)&v1_GPU, uiMatrixSize * sizeof(float));
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erro ao alocar memória do vetor 1 - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc((void**)&v2_GPU, uiMatrixSize * sizeof(float));
        if (cudaStatus != cudaSuccess)
        {
            fprintf(fp,"Erro ao alocar memória do vetor 2 - cudaMalloc()");
            goto FreeCuda;
        }
    }

    // Copiar memória dos vetores para o Buffer da GPU
    { 
        cudaStatus = cudaMemcpy(A_GPU, A, uiMatrixSize * uiMatrixSize * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erro ao copiar os valores da matriz A - cudaMemcpy()");
            goto FreeCuda;
        }

        cudaStatus = cudaMemcpy(v1_GPU, v1, uiMatrixSize * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) 
        {
            fprintf(fp,"Erro ao copiar os valores do vetor 1 - cudaMemcpy()");
            goto FreeCuda;
        }

    }

    auto clockInicioProcessamento = std::chrono::high_resolution_clock::now();
    KernelMatrixVectorProduct << <grid_shape, block_shape >> > (A_GPU, v1_GPU, v2_GPU, uiMatrixSize);

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

    //Copiar dados do buffer de memória da GPU - managed - de volta para memória local do host
    cudaStatus = cudaMemcpy(v2, v2_GPU, uiMatrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(fp,"Erro ao copiar memória do buffer da GPU  - cudaMemcpy()");
        goto FreeCuda;
    }

FreeCuda:
    cudaFree(A_GPU );
    cudaFree(v1_GPU);
    cudaFree(v2_GPU);

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

    float* A ; // Matriz N * N
    float* v1; // Vetor para mult
    float* v2; // Vetor resultado

    A  = (float*)malloc(uiMatrixSizeCFG * uiMatrixSizeCFG * sizeof(float));
    v1 = (float*)malloc(uiMatrixSizeCFG  * sizeof(float));
    v2 = (float*)malloc(uiMatrixSizeCFG  * sizeof(float));

    // Popular vetores com valores reais aleatórios
    {
        std::random_device device; //Gerar seed
        std::mt19937 rng(device());

        std::uniform_real_distribution<> getRandReal(0.1, 999.9);

        for (int i = 0; i < uiMatrixSizeCFG; ++i)
        {
            //A é uma matrix N * N, porém representada linearmente para facilitar blocos de CUDA posteriormente
            for (int j = 0; j < uiMatrixSizeCFG; ++j)
            {
                A[i * uiMatrixSizeCFG + j] = getRandReal(rng);
            }

            v1[i] = getRandReal(rng);
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
        LinearMatrixVectorProduct(A, v1, v2, uiMatrixSizeCFG, benchResultsLinear.msTimeElapsed);
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
        CPUConcurrencyMatrixVectorProduct(A, v1, v2, uiMatrixSizeCFG, benchResultsCPUThreads.msTimeElapsed);
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

        cudaError_t cudaStatus = CUDAMatrixVectorProduct(A, v1, v2, uiMatrixSizeCFG, benchResultsCUDAProcess.msTimeElapsed, benchResultsCUDAFull.msTimeElapsed);
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
        free(A );
        free(v1);
        free(v2);
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
