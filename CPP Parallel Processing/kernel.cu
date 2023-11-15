
#include <random>
#include <stdio.h>
#include <cassert>
#include <chrono>

#include "windows.h"

//Libs da NVidia para uso de CUDA Cores
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#define MATRIX_SIZE 40000

using msTime = std::chrono::duration<double, std::milli>;
using UINT = unsigned int;

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

cudaError_t CUDAMatrixVectorProduct(float* A, float* v1, float* v2, UINT uiMatrixSize, msTime& processingTime)
{
    printf("\n\n[CUDA CORES - INÍCIO]\n");

    float* A_GPU ;
    float* v1_GPU;
    float* v2_GPU;

    dim3 block_shape = dim3(32, 32);
    dim3 grid_shape  = dim3(max(1.0, std::ceil((float)uiMatrixSize / (float)block_shape.x)),
                            max(1.0, std::ceil((float)uiMatrixSize / (float)block_shape.y)));

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
        printf("Block Shape: %d - %d - %d\n", block_shape.x, block_shape.y, block_shape.z);
        printf("Grid  Shape: %d - %d - %d\n", grid_shape .x, grid_shape .y, grid_shape .z);
    }

    auto clockInicioCuda = std::chrono::high_resolution_clock::now();
   
    // Alocação de buffer de GPU para os vetores
    {
        cudaStatus = cudaMalloc((void**)&A_GPU, uiMatrixSize * uiMatrixSize * sizeof(float));
        if (cudaStatus != cudaSuccess) 
        {
            printf("Erroi ao alocar memória da matriz A - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc((void**)&v1_GPU, uiMatrixSize * sizeof(float));
        if (cudaStatus != cudaSuccess) 
        {
            printf("Erro ao alocar memória do vetor 1 - cudaMalloc()");
            goto FreeCuda;
        }

        cudaStatus = cudaMalloc((void**)&v2_GPU, uiMatrixSize * sizeof(float));
        if (cudaStatus != cudaSuccess)
        {
            printf("Erro ao alocar memória do vetor 2 - cudaMalloc()");
            goto FreeCuda;
        }
    }

    // Copiar memória dos vetores para o Buffer da GPU
    { 
        cudaStatus = cudaMemcpy(A_GPU, A, uiMatrixSize * uiMatrixSize * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) 
        {
            printf("Erro ao copiar os valores da matriz A - cudaMemcpy()");
            goto FreeCuda;
        }

        cudaStatus = cudaMemcpy(v1_GPU, v1, uiMatrixSize * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) 
        {
            printf("Erro ao copiar os valores do vetor 1 - cudaMemcpy()");
            goto FreeCuda;
        }

    }

    KernelMatrixVectorProduct << <grid_shape, block_shape >> > (A_GPU, v1_GPU, v2_GPU, uiMatrixSize);

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
    cudaStatus = cudaMemcpy(v2, v2_GPU, uiMatrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        printf("Erro ao copiar memória do buffer da GPU  - cudaMemcpy()");
        goto FreeCuda;
    }

FreeCuda:
    cudaFree(A_GPU );
    cudaFree(v1_GPU);
    cudaFree(v2_GPU);

    auto clockFimCuda = std::chrono::high_resolution_clock::now();
    processingTime = clockFimCuda - clockInicioCuda;
    printf("Tempo total de processamento com CUDA cores: %fms\n", processingTime.count());

    printf("[CUDA CORES - FIM]\n");

    return cudaStatus;
}

void linearMatrixVectorProduct(float *A, float* v1, float* v2, UINT uiMatrixSize, msTime& processingTime)
{
    printf("\n\n[PRODUTO LINEAR DE VETOR X MATRIZ - PROCESSAMENTO LINEAR - INÍCIO]\n");

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

    printf("Tempo total de processamento linear: %fms\n", processingTime.count());
    printf("[PRODUTO LINEAR DE VETOR X MATRIZ - PROCESSAMENTO LINEAR - FIM]\n");
}

int main(int argc, char **argv)
{
    SetConsoleCP      (1252);
    SetConsoleOutputCP(1252);

    float* A ; // Matriz N * N
    float* v1; // Vetor para mult
    float* v2; // Vetor resultado

    A  = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    v1 = (float*)malloc(MATRIX_SIZE  * sizeof(float));
    v2 = (float*)malloc(MATRIX_SIZE  * sizeof(float));

    // Popular vetores com valores reais aleatórios
    {
        std::random_device device; //Gerar seed
        std::mt19937 rng(device());

        std::uniform_real_distribution<> getRandReal(0.1, 999.9);

        for (int i = 0; i < MATRIX_SIZE; ++i)
        {
            //A é uma matrix N * N, porém representada linearmente para facilitar blocos de CUDA posteriormente
            for (int j = 0; j < MATRIX_SIZE; ++j)
            {
                A[i * MATRIX_SIZE + j] = getRandReal(rng);
            }

            v1[i] = getRandReal(rng);
        }
    }

    msTime CUDAProcessingTime;
    //Processamento paraleo com CUDA cores
    {
        cudaError_t cudaStatus = CUDAMatrixVectorProduct(A, v1, v2, MATRIX_SIZE, CUDAProcessingTime);
        if (cudaStatus != cudaSuccess)
        {
            printf("Erro ao processar soma em CUDA");
            return 1;
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
    linearMatrixVectorProduct(A, v1, v2, MATRIX_SIZE, linearProcessingTime);

    //Liberar valores dos ponteiros de matrizes
    {
        free(A );
        free(v1);
        free(v2);
    }

    printf("\n\n[DIF] Diferença entre processamento linear e paralelizado com CUDA cores = %fms\n", linearProcessingTime.count() - CUDAProcessingTime.count());

    return 0;
}
