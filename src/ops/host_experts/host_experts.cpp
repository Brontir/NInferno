// HostExperts (host_experts.h).
#include "ops/host_experts/host_experts.h"

#include "core/device.h"
#include "ops/host_experts/host_pool.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace ninferno::ops {

namespace {
constexpr int kGeneric = 0;
constexpr int kGemma4  = 1;

// The server polls the mailbox without pause for this long after a call (decode calls come a
// layer apart, and a token's layers follow each other within a millisecond), then sleeps between
// polls: kPollShort while the last call is recent, kPollLong after kIdle.
constexpr auto kSpin      = std::chrono::milliseconds(2);
constexpr auto kIdle      = std::chrono::milliseconds(200);
constexpr auto kPollShort = std::chrono::microseconds(20);
constexpr auto kPollLong  = std::chrono::microseconds(500);

template <class T>
T* mapped(std::size_t bytes, void** alias) {
    void* host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(alias, host, 0));
    return static_cast<T*>(host);
}

[[noreturn]] void fatal(const char* what) {
    std::fprintf(stderr, "FATAL host experts: %s\n", what);
    std::fflush(stderr);
    std::abort();
}
} // namespace

struct HostExperts::Call {
    int kind   = kGeneric;
    int tokens = 0;
    host::GenericLayer generic;
    host::Gemma4Layer gemma4;
};

HostExperts::HostExperts(std::int32_t hidden, std::int32_t top_k, int threads, host::Isa isa)
    : moe_(threads, isa), hidden_(hidden), top_k_(top_k), calls_(std::make_unique<std::deque<Call>>()) {
    if (hidden < 1 || top_k < 1 || top_k > 32) {
        throw std::invalid_argument("host experts: invalid staging geometry");
    }
    const std::size_t tokens      = host::kHostMaxTokens;
    const std::size_t assignments = tokens * static_cast<std::size_t>(top_k);
    void* alias                   = nullptr;
    x_host_  = mapped<std::uint16_t>(tokens * static_cast<std::size_t>(hidden) * 2, &x_alias_);
    ids_host_ = mapped<std::int32_t>(assignments * 4, &alias);
    ids_alias_ = static_cast<std::int32_t*>(alias);
    weights_host_  = mapped<float>(assignments * 4, &alias);
    weights_alias_ = static_cast<float*>(alias);
    out_host_ = mapped<std::uint16_t>(assignments * static_cast<std::size_t>(hidden) * 2, &out_alias_);
    mailbox_  = mapped<host::Mailbox>(sizeof(host::Mailbox), &alias);
    mailbox_alias_ = static_cast<host::Mailbox*>(alias);
    std::memset(static_cast<void*>(mailbox_), 0, sizeof(host::Mailbox));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sequence_), 2 * sizeof(std::uint64_t)));
    {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        host::launch_reset(sequence_, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    server_ = std::thread([this] { serve(); });
}

HostExperts::~HostExperts() {
    stop_.store(true);
    if (server_.joinable()) { server_.join(); }
    (void)cudaFree(sequence_);
    for (void* p : {static_cast<void*>(x_host_), static_cast<void*>(ids_host_),
                    static_cast<void*>(weights_host_), static_cast<void*>(out_host_),
                    static_cast<void*>(mailbox_)}) {
        if (p != nullptr) { (void)cudaFreeHost(p); }
    }
}

std::uint32_t HostExperts::call_for(int kind, const void* key, int tokens,
                                    const host::GenericLayer* generic,
                                    const host::Gemma4Layer* gemma4) {
    std::lock_guard<std::mutex> lock(calls_mutex_);
    const auto it = index_.find({kind, key, tokens});
    if (it != index_.end()) { return it->second; }
    Call& call  = calls_->emplace_back();
    call.kind   = kind;
    call.tokens = tokens;
    if (generic != nullptr) { call.generic = *generic; }
    if (gemma4 != nullptr) { call.gemma4 = *gemma4; }
    const auto index = static_cast<std::uint32_t>(by_index_.size());
    by_index_.push_back(&call);
    index_.emplace(std::make_tuple(kind, key, tokens), index);
    return index;
}

void HostExperts::serve() {
#if defined(__linux__)
    // The server is the pool's caller: it takes the core the workers leave to it.
    if (!moe_.cpus().empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(moe_.cpus().front(), &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
    std::uint64_t answered = 0;
    auto last              = std::chrono::steady_clock::now();
    while (!stop_.load(std::memory_order_relaxed)) {
        const std::uint64_t request = mailbox_->request;
        if (request != answered) {
            std::atomic_thread_fence(std::memory_order_acquire);
            const Call* call = nullptr;
            {
                std::lock_guard<std::mutex> lock(calls_mutex_);
                const std::uint32_t index = mailbox_->call;
                if (index >= by_index_.size()) { fatal("the mailbox names an unknown call"); }
                call = by_index_[index];
            }
            try {
                execute(*call);
            } catch (const std::exception& e) {
                fatal(e.what());
            }
            std::atomic_thread_fence(std::memory_order_release);
            mailbox_->done = request;
            answered       = request;
            last           = std::chrono::steady_clock::now();
            continue;
        }
        const auto idle = std::chrono::steady_clock::now() - last;
        if (idle < kSpin) {
            host::relax();
        } else {
            std::this_thread::sleep_for(idle < kIdle ? kPollShort : kPollLong);
        }
    }
}

void HostExperts::execute(const Call& call) {
    if (call.kind == kGeneric) {
        moe_.generic(call.generic, call.tokens, x_host_, ids_host_, out_host_);
    } else {
        moe_.gemma4(call.gemma4, call.tokens, x_host_, ids_host_, weights_host_, out_host_);
    }
}

namespace {

void require_tokens(int tokens, std::int32_t hidden, std::int32_t top_k, std::int32_t max_hidden,
                    std::int32_t max_top_k) {
    if (tokens < 1 || tokens > host::kHostMaxTokens) {
        throw std::invalid_argument("host experts: a CPU expert call takes 1 to 16 tokens");
    }
    if (hidden > max_hidden || top_k > max_top_k) {
        throw std::invalid_argument("host experts: layer wider than the staging buffers");
    }
}

} // namespace

void HostExperts::enqueue_generic(const host::GenericLayer& layer, int tokens, const void* x,
                                  const std::int32_t* ids, void* y, std::int32_t* slot_of,
                                  cudaStream_t stream) {
    require_tokens(tokens, layer.hidden, layer.top_k, hidden_, top_k_);
    const std::uint32_t call = call_for(kGeneric, layer.gate_up.codes, tokens, &layer, nullptr);
    const int assignments    = tokens * layer.top_k;
    host::PostArgs post;
    post.x         = x;
    post.x_bytes   = static_cast<std::size_t>(tokens) * layer.hidden * sizeof(std::uint16_t);
    post.ids       = ids;
    post.ids_count = assignments;
    post.x_host    = x_alias_;
    post.ids_host  = ids_alias_;
    post.mailbox   = mailbox_alias_;
    post.sequence  = sequence_;
    post.call      = call;
    host::launch_post(post, stream);
    host::AwaitArgs await;
    await.out_host  = out_alias_;
    await.out       = y;
    await.out_bytes = static_cast<std::size_t>(assignments) * layer.hidden * sizeof(std::uint16_t);
    await.slot_of   = slot_of;
    await.slots     = assignments;
    await.mailbox   = mailbox_alias_;
    await.sequence  = sequence_;
    host::launch_await(await, stream);
}

void HostExperts::enqueue_gemma4(const host::Gemma4Layer& layer, int tokens, const void* x,
                                 const std::int32_t* ids, const float* weights, void* out,
                                 cudaStream_t stream) {
    require_tokens(tokens, layer.hidden, layer.top_k, hidden_, top_k_);
    const std::uint32_t call = call_for(kGemma4, layer.gate_up.codes, tokens, nullptr, &layer);
    const std::size_t bytes  = static_cast<std::size_t>(tokens) * layer.hidden * sizeof(std::uint16_t);
    host::PostArgs post;
    post.x            = x;
    post.x_bytes      = bytes;
    post.ids          = ids;
    post.weights      = weights;
    post.ids_count    = tokens * layer.top_k;
    post.x_host       = x_alias_;
    post.ids_host     = ids_alias_;
    post.weights_host = weights_alias_;
    post.mailbox      = mailbox_alias_;
    post.sequence     = sequence_;
    post.call         = call;
    host::launch_post(post, stream);
    host::AwaitArgs await;
    await.out_host  = out_alias_;
    await.out       = out;
    await.out_bytes = bytes;
    await.mailbox   = mailbox_alias_;
    await.sequence  = sequence_;
    host::launch_await(await, stream);
}

} // namespace ninferno::ops
