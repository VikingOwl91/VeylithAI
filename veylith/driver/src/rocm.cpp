// tch exposes tensors on ROCm as Device::Cuda. Use the actual lowest HIP
// priority, which can be outside LibTorch's pool of stream priorities.
#include <ATen/Context.h>
#include <c10/hip/HIPStream.h>
#include <cstdio>
#include <exception>

extern "C" int veylith_start_rocm(int *priority, char *error, size_t length) noexcept {
    hipStream_t stream = nullptr;
    try {
        TORCH_CHECK(at::globalContext().hasROCM(), "LibTorch's ROCm backend is unavailable");
        C10_HIP_CHECK(hipSetDevice(0));
        int least, greatest;
        C10_HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
        C10_HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, least));
        C10_HIP_CHECK(hipStreamGetPriority(stream, priority));
        TORCH_CHECK(*priority == least, "HIP did not grant the requested stream priority");
        // This CLI uses one stream until process exit. All forward passes run on this thread.
        c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromExternal(stream, 0));
        return 0;
    } catch (const std::exception &err) {
        if (stream) (void)hipStreamDestroy(stream);
        std::snprintf(error, length, "%s", err.what());
        return 1;
    }
}
