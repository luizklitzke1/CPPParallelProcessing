#pragma once

#include "cuda_runtime.h"
#include <string>

using msTime = std::chrono::duration<double, std::milli>;
using UINT = unsigned int;

struct benchResults
{
    std::string sMethod;
    msTime msTimeElapsed;
};

class CBenchUtils
{
public:
    static int GetCudaCores(cudaDeviceProp devProp);
};