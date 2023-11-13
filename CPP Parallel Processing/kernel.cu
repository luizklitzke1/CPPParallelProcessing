
#include <random>
#include <stdio.h>
#include <cassert>

//Libs da NVidia para uso de CUDA Cores
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKS 1

//Limite de threas por bloco = 1024
#define THREADS_PER_BLOCK 1024

__global__ void addKernel(const int* vectorA, const int* vectorB, int* sumVector)
{
    const int idxThread = threadIdx.x;
    sumVector[idxThread] = vectorA[idxThread] + vectorB[idxThread];
}

cudaError_t addWithCuda(const int* vectorA, const int* vectorB, int* sumVector)
{
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
    
    { // Alocação de buffer de GPU para os vetores
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

    { // Copiar memória dos vetores para o Buffer da GPU
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
        printf("Erro ao executar cudaDeviceSynchronize %d  - Cod %d - %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
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

    return cudaStatus;
}

int main()
{
    int vectorA  [THREADS_PER_BLOCK] = { 0 };
    int vectorB  [THREADS_PER_BLOCK] = { 0 };
    int sumvector[THREADS_PER_BLOCK] = { 0 };

    {// Popular vetores com inteiros aleatórios - https://stackoverflow.com/questions/13445688/how-to-generate-a-random-number-in-c
        std::random_device device;
        std::mt19937 rng(device());

        std::uniform_int_distribution<std::mt19937::result_type> getRandInt(0, (INT_MAX / 2) - 1);

        for (int i = 0; i < THREADS_PER_BLOCK; ++i)
        {
            vectorA[i] = getRandInt(rng);
            vectorB[i] = getRandInt(rng);
        }
    }

    cudaError_t cudaStatus = addWithCuda(vectorA, vectorB, sumvector);
    if (cudaStatus != cudaSuccess)
    {
        printf("Erro ao processar soma em CUDA");
        return 1;
    }

    {//Validar somas
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

            printf("[%d]%d + %d = %d\n", i, valueA, valueB, sumValue);
        }
    }

    {//Limpar devices para evitar erros de profiling
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess)
        {
            printf("Erro ao executar cudaDeviceReset()");
            return 1;
        }
    }

    return 0;
}
