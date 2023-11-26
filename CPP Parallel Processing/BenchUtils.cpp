#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include "windows.h"
#include <time.h>

#include "BenchUtils.h"
#include "sysinfoapi.h"

int CBenchUtils::GetCudaCores(cudaDeviceProp devProp)
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
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
            break;
        case 9: // Hopper
            if (devProp.minor == 0) cores = mp * 128;
            break;
        default:
            break;
    }

    return cores;
}

std::string CBenchUtils::GetTimeString()
{
    time_t now;
    struct tm* tm_now;
    char texto[40];

    time(&now);
    tm_now = localtime(&now);
    strftime(texto, sizeof(texto), "%d/%m/%Y - %H:%M:%S", tm_now);

    return texto;
}
