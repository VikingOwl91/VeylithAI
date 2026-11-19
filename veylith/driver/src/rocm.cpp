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

// Dense training requires a separately attested runtime, device and CU mask.
#include <torch/csrc/api/include/torch/version.h>
#include <algorithm>
#include <vector>

static bool eight_bits(const std::vector<uint32_t> &mask) {
    return !mask.empty() && mask.front() == 0xff &&
           std::all_of(mask.begin() + 1, mask.end(), [](uint32_t w) { return w == 0; });
}
extern "C" int veylith_dense_rocm_identity(char *identity, size_t capacity,
                                           char *error, size_t length) noexcept {
    try {
        int device, count, runtime, driver;
        C10_HIP_CHECK(hipGetDevice(&device));
        C10_HIP_CHECK(hipGetDeviceCount(&count));
        TORCH_CHECK(device == 0 && count == 1, "expected one visible GPU at index zero");
        hipDeviceProp_t p{};
        C10_HIP_CHECK(hipGetDeviceProperties(&p, device));
        char pci[64]{};
        C10_HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), device));
        C10_HIP_CHECK(hipRuntimeGetVersion(&runtime));
        C10_HIP_CHECK(hipDriverGetVersion(&driver));
        size_t words = std::max(2, (p.multiProcessorCount + 31) / 32);
        // The implementation may copy only its populated words. Unwritten
        // trailing words represent no extra bits in that returned vector.
        std::vector<uint32_t> global(words, 0);
        C10_HIP_CHECK(hipExtStreamGetCUMask(nullptr, words, global.data()));
        TORCH_CHECK(eight_bits(global), "HIP global CU mask differs from 0xff");

        auto stream = c10::cuda::getCurrentCUDAStream(device).stream();
        TORCH_CHECK(stream != nullptr, "expected the selected low-priority nondefault stream");
        std::vector<uint32_t> active(words, 0);
        C10_HIP_CHECK(hipExtStreamGetCUMask(stream, words, active.data()));
        bool unwritten = false;
        if (!eight_bits(active)) {
            constexpr uint32_t sentinel = 0x5a5a5a5a;
            std::vector<uint32_t> probe(words, sentinel);
            C10_HIP_CHECK(hipExtStreamGetCUMask(stream, words, probe.data()));
            unwritten = std::all_of(active.begin(), active.end(), [](uint32_t w) { return w == 0; }) &&
                        std::all_of(probe.begin(), probe.end(), [=](uint32_t w) { return w == sentinel; });
            TORCH_CHECK(unwritten, "active HIP stream returned a CU mask different from 0xff");
        }
        const int written = std::snprintf(identity, capacity,
            "%s; arch=%s; pci=%s; hip_runtime=%d; hip_driver=%d; libtorch_headers=%s; "
            "hip_global_mask=0xff; stream=lowest_priority_no_custom_mask; active_stream_mask=%s",
            p.name, p.gcnArchName, pci, runtime, driver, TORCH_VERSION,
            unwritten ? "query_unwritten" : "0xff");
        TORCH_CHECK(written >= 0 && static_cast<size_t>(written) < capacity, "identity buffer too small");
        return 0;
    } catch (const std::exception &err) {
        std::snprintf(error, length, "%s", err.what());
        return 1;
    }
}

extern "C" int veylith_dense_rocm_memory(size_t *free_bytes, size_t *total_bytes,
                                         char *error, size_t length) noexcept {
    try {
        C10_HIP_CHECK(hipDeviceSynchronize());
        C10_HIP_CHECK(hipMemGetInfo(free_bytes, total_bytes));
        return 0;
    } catch (const std::exception &err) {
        std::snprintf(error, length, "%s", err.what());
        return 1;
    }
}
