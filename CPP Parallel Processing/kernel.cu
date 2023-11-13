
#include <random>
#include <stdio.h>
#include <cassert>
#include <chrono>

//Libs da NVidia para uso de CUDA Cores
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "windows.h"

#define BLOCKS 1

//Limite de threads por bloco = 1024
#define THREADS_PER_BLOCK 1024

using msTime = std::chrono::duration<double, std::milli>;

int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;

    switch (devProp.major)
    {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
            else printf("Unknown device type\n");
            break;
        case 9: // Hopper
            if (devProp.minor == 0) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }

    return cores;
}

__global__ void addKernel(const int* vectorA, const int* vectorB, int* sumVector)
{
    const int idxThread = threadIdx.x;
    sumVector[idxThread] = vectorA[idxThread] + vectorB[idxThread];
}

cudaError_t addWithCuda(const int* vectorA, const int* vectorB, int* sumVector, msTime& processingTime)
{
    printf("\n\n[CUDA CORES - INÍCIO]\n");
    printf("BLOCOS: %d\nTHREADS POR BLOCO: %d\n", BLOCKS, THREADS_PER_BLOCK);

    auto clockInicioCuda = std::chrono::high_resolution_clock::now();

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    cudaError_t cudaStatus = cudaError_t::cudaSuccess;

    // Definir qual device vai ser utilizado
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("Erro ao buscar um cudaSetDevice. Verifique se sua GPU é compatível");
        goto FreeCuda;
    }

    // Log de specs do device
    {
        int deviceID;
        cudaDeviceProp devProps;

        cudaStatus = cudaGetDevice(&deviceID);
        if (cudaStatus != cudaSuccess) {
            printf("Erro ao pegar ID do device - cudaGetDevice() - Cod %d - %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
            goto FreeCuda;
        }

        cudaGetDeviceProperties(&devProps, deviceID);
        const int iCUDACores = getSPcores(devProps);

        printf("Device \"%s\" selecionado.\nO device possui %d CUDA cores.\n", devProps.name, iCUDACores);
    }
   
    // Alocação de buffer de GPU para os vetores
    {
        cudaStatus = cudaMalloc((void**)&dev_a, THREADS_PER_BLOCK * sizeof(int));
        if (cudaStatus != cudaSuccess) 
        {
            printf("Erroi ao alocar memória do vetor A - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc((void**)&dev_b, THREADS_PER_BLOCK * sizeof(int));
        if (cudaStatus != cudaSuccess) 
        {
            printf("Erro ao alocar memória do vetor B - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc((void**)&dev_c, THREADS_PER_BLOCK * sizeof(int));
        if (cudaStatus != cudaSuccess)
        {
            printf("Erro ao alocar memória do vetor de Soma - cudaMalloc()");
            goto FreeCuda;
        }
    }

    // Copiar memória dos vetores para o Buffer da GPU
    { 
        cudaStatus = cudaMemcpy(dev_a, vectorA, THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) 
        {
            printf("Erro ao copiar os valores do vetor A - cudaMemcpy()");
            goto FreeCuda;
        }

        cudaStatus = cudaMemcpy(dev_b, vectorB, THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) 
        {
            printf("Erro ao copiar os valores do vetor B - cudaMemcpy()");
            goto FreeCuda;
        }

    }
    
    //Cahmada do Kernel poara processamento paralelo, com um único bloco contendo uma threada para cada index do vetor
    addKernel << <BLOCKS, THREADS_PER_BLOCK >> > (dev_a, dev_b, dev_c);

    //Validar erros na chamada de Kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        printf("Erro ao executar addKernel() - Cod %d - %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto FreeCuda;
    }

    //Validar sincronização dos devices após executar chamada de Kernel
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        printf("Erro ao executar cudaDeviceSynchronize %d - Cod %d - %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto FreeCuda;
    }

    //Copiar dados do buffer de memória da GPU - managed - de volta para memória local do host
    cudaStatus = cudaMemcpy(sumVector, dev_c, THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        printf("Erro ao copiar memória do buffer da GPU  - cudaMemcpy()");
        goto FreeCuda;
    }

FreeCuda:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    auto clockFimCuda = std::chrono::high_resolution_clock::now();
    processingTime = clockFimCuda - clockInicioCuda;
    printf("Tempo total de processamento com CUDA cores: %fms\n", processingTime.count());

    printf("[CUDA CORES - FIM]\n");

    return cudaStatus;
}

void addLinear(const int* vectorA, const int* vectorB, int* sumVector, msTime& processingTime)
{
    printf("\n\n[PROCESSAMENTO LINEAR - INÍCIO]\n");

    auto clockInicioLinear = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < THREADS_PER_BLOCK; ++i)
    {
        sumVector[i] = vectorA[i] + vectorB[i];
    }

    auto clockFimLinear = std::chrono::high_resolution_clock::now();

    processingTime = clockFimLinear - clockInicioLinear;

    printf("Tempo total de processamento linear: %fms\n", processingTime.count());
    printf("[PROCESSAMENTO LINEAR - FIM]\n");
}

int main()
{
    SetConsoleCP(1252);
    SetConsoleOutputCP(1252);

    int vectorA  [THREADS_PER_BLOCK] = { 0 };
    int vectorB  [THREADS_PER_BLOCK] = { 0 };
    int sumvector[THREADS_PER_BLOCK] = { 0 };

    // Popular vetores com inteiros aleatórios - https://stackoverflow.com/questions/13445688/how-to-generate-a-random-number-in-c
    {
        std::random_device device;
        std::mt19937 rng(device());

        std::uniform_int_distribution<std::mt19937::result_type> getRandInt(0, (INT_MAX / 2) - 1);

        for (int i = 0; i < THREADS_PER_BLOCK; ++i)
        {
            vectorA[i] = getRandInt(rng);
            vectorB[i] = getRandInt(rng);
        }
    }

    msTime CUDAProcessingTime;
    //Processamento paraleo com CUDA cores
    {
        cudaError_t cudaStatus = addWithCuda(vectorA, vectorB, sumvector, CUDAProcessingTime);
        if (cudaStatus != cudaSuccess)
        {
            printf("Erro ao processar soma em CUDA");
            return 1;
        }

        // Validar somas
        {
            for (int i = 0; i < THREADS_PER_BLOCK; ++i)
            {
                const int valueA   = vectorA  [i];
                const int valueB   = vectorB  [i];
                const int sumValue = sumvector[i];

                assert(sumValue == valueA + valueB);

                if (sumValue != valueA + valueB)
                {
                    printf("[%d][ERRO DE SOMA][Diferença encontrada na soma! - %d + %d != %d\n", i, valueA, valueB, sumValue);
                    return 1;
                }

                //printf("[%d]%d + %d = %d\n", i, valueA, valueB, sumValue);
            }
        }

        // Limpar devices para evitar erros de profiling
        {
            cudaStatus = cudaDeviceReset();
            if (cudaStatus != cudaSuccess)
            {
                printf("Erro ao executar cudaDeviceReset()");
                return 1;
            }
        }
    }

    //Processamento Linear
    msTime linearProcessingTime;
    addLinear(vectorA, vectorB, sumvector, linearProcessingTime);

    printf("\n\n[DIF] Diferença entre processamento linear e paralelizado com CUDA cores = %f\n", linearProcessingTime.count() - CUDAProcessingTime.count());

    return 0;
}
