#include "kld.h"
#include "core/fnv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ninferno::perplexity {
namespace {

// "NKLDREF2", then little-endian fields (the engine runs on little-endian hosts only).
constexpr char kMagic[8] = {'N', 'K', 'L', 'D', 'R', 'E', 'F', '2'};

template <class T>
void put(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void put_string(std::ofstream& out, const std::string& value) {
    put(out, static_cast<std::uint32_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

template <class T>
void put_vector(std::ofstream& out, const std::vector<T>& values) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
}

template <class T>
T get(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("KLD reference is truncated");
    return value;
}

std::string get_string(std::ifstream& in, std::uint32_t limit) {
    const auto size = get<std::uint32_t>(in);
    if (size > limit) throw std::runtime_error("KLD reference has an oversized string");
    std::string value(size, '\0');
    in.read(value.data(), size);
    if (!in) throw std::runtime_error("KLD reference is truncated");
    return value;
}

template <class T>
std::vector<T> get_vector(std::ifstream& in, std::uint64_t count) {
    std::vector<T> values(count);
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) throw std::runtime_error("KLD reference is truncated");
    return values;
}

} // namespace

std::uint64_t token_hash(std::span<const TokenId> tokens) {
    std::uint64_t hash = ninferno::fnv::kOffset; // FNV-1a over the ids' bytes
    for (const TokenId token : tokens) {
        auto bits = static_cast<std::uint32_t>(token);
        for (int i = 0; i < 4; ++i) {
            hash = ninferno::fnv::mix(hash, bits & 0xffU);
            bits >>= 8;
        }
    }
    return hash;
}

void write_kld_reference(const std::filesystem::path& path, const KldReference& reference) {
    const std::filesystem::path temporary = path.string() + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create KLD reference: " + temporary.string());
        out.write(kMagic, sizeof(kMagic));
        put(out, reference.top_k);
        put(out, reference.context);
        put(out, reference.stride);
        put(out, reference.window_start);
        put(out, static_cast<std::uint32_t>(reference.streams.size()));
        put_string(out, reference.model);
        put_string(out, reference.prefill_signature);
        for (const KldStream& stream : reference.streams) {
            const std::uint64_t slots = stream.targets * reference.top_k;
            if (stream.ids.size() != slots || stream.logprobs.size() != slots ||
                stream.target_logprobs.size() != stream.targets) {
                throw std::logic_error("KLD reference stream has an inconsistent shape");
            }
            put_string(out, stream.id);
            put(out, stream.token_hash);
            put(out, stream.targets);
            put_vector(out, stream.ids);
            put_vector(out, stream.logprobs);
            put_vector(out, stream.target_logprobs);
        }
        out.flush();
        if (!out) throw std::runtime_error("cannot write KLD reference: " + temporary.string());
    }
    std::filesystem::rename(temporary, path);
}

KldReference read_kld_reference(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open KLD reference: " + path.string());
    char magic[sizeof(kMagic)] = {};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not a KLD reference (ninferno-perplexity --kld-out): " +
                                 path.string());
    }
    KldReference reference;
    reference.top_k = get<std::uint32_t>(in);
    if (reference.top_k == 0 || reference.top_k > 64) {
        throw std::runtime_error("KLD reference has an invalid top-K");
    }
    reference.context = get<std::uint32_t>(in);
    reference.stride  = get<std::uint32_t>(in);
    if (reference.context < 2 || reference.stride == 0 || reference.stride >= reference.context) {
        throw std::runtime_error("KLD reference has an invalid context/stride");
    }
    reference.window_start      = get<std::int32_t>(in);
    const auto streams          = get<std::uint32_t>(in);
    reference.model             = get_string(in, 4096);
    reference.prefill_signature = get_string(in, 1U << 20);
    // The file's own size bounds every count read from it.
    const auto file_bytes = static_cast<std::uint64_t>(std::filesystem::file_size(path));
    for (std::uint32_t s = 0; s < streams; ++s) {
        KldStream stream;
        stream.id                 = get_string(in, 1U << 16);
        stream.token_hash         = get<std::uint64_t>(in);
        stream.targets            = get<std::uint64_t>(in);
        const std::uint64_t slots = stream.targets * reference.top_k;
        if (stream.targets > file_bytes / 4 || slots > file_bytes / 8) {
            throw std::runtime_error("KLD reference stream is larger than the file");
        }
        stream.ids             = get_vector<TokenId>(in, slots);
        stream.logprobs        = get_vector<float>(in, slots);
        stream.target_logprobs = get_vector<float>(in, stream.targets);
        // A listed slot holds an id and a finite log-probability; an empty one -1 and -inf.
        for (std::uint64_t i = 0; i < slots; ++i) {
            const TokenId id = stream.ids[i];
            const float lp   = stream.logprobs[i];
            const bool empty = id == -1 && std::isinf(lp) && lp < 0.0f;
            if (!empty && (id < 0 || !std::isfinite(lp) || lp > 0.0f)) {
                throw std::runtime_error("KLD reference stream " + stream.id +
                                         " has an invalid slot");
            }
        }
        for (const float lp : stream.target_logprobs) {
            if (!std::isfinite(lp) || lp > 0.0f) {
                throw std::runtime_error("KLD reference stream " + stream.id +
                                         " has an invalid target log-probability");
            }
        }
        reference.streams.push_back(std::move(stream));
    }
    if (in.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("KLD reference has trailing bytes");
    }
    return reference;
}

double position_kld(std::span<const float> p, std::span<const float> q) {
    if (p.size() != q.size()) throw std::invalid_argument("position_kld: slot counts differ");
    double kld = 0.0, p_listed = 0.0, q_listed = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (std::isinf(p[i]) && p[i] < 0) continue; // an empty slot (ranked past the window)
        const double lp = p[i], lq = q[i];
        const double pi = std::exp(lp);
        p_listed += pi;
        q_listed += std::exp(lq);
        kld += pi * (lp - lq);
    }
    // The unlisted rest as one outcome of each distribution. Q's listed mass carries the FP32
    // rounding of its log-probabilities (logit - log-sum-exp: about |lse| ulps, 2e-6 relative at
    // |lse| 40, common to every slot), so its rest is floored at that noise, not at zero: a peaked
    // Q would otherwise read as putting no mass where P puts some. The floor keeps the result a
    // lower bound.
    constexpr double kQRestFloor = 1e-5;
    const double p_rest          = std::max(0.0, 1.0 - p_listed);
    const double q_rest          = std::max(kQRestFloor, 1.0 - q_listed);
    if (p_rest > 0.0) kld += p_rest * (std::log(p_rest) - std::log(q_rest));
    return std::max(0.0, kld);
}

void KldAggregate::add(double position, bool same_top1, double reference_logprob, double logprob) {
    kld.push_back(position);
    top1_same += same_top1 ? 1 : 0;
    reference_nll -= reference_logprob;
    nll -= logprob;
}

void KldAggregate::add(const KldAggregate& other) {
    kld.insert(kld.end(), other.kld.begin(), other.kld.end());
    top1_same += other.top1_same;
    reference_nll += other.reference_nll;
    nll += other.nll;
}

double KldAggregate::mean() const {
    if (kld.empty()) throw std::logic_error("KLD aggregate is empty");
    double sum = 0.0;
    for (const double v : kld) sum += v;
    return sum / static_cast<double>(kld.size());
}

double KldAggregate::quantile(double q) const {
    if (kld.empty()) throw std::logic_error("KLD aggregate is empty");
    std::vector<double> sorted = kld;
    const auto index           = static_cast<std::size_t>(
        std::clamp(q, 0.0, 1.0) * static_cast<double>(sorted.size() - 1) + 0.5);
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(index),
                     sorted.end());
    return sorted[index];
}

double KldAggregate::top1_agreement() const {
    if (kld.empty()) throw std::logic_error("KLD aggregate is empty");
    return static_cast<double>(top1_same) / static_cast<double>(kld.size());
}

double KldAggregate::ppl_ratio() const {
    if (kld.empty()) throw std::logic_error("KLD aggregate is empty");
    return std::exp((nll - reference_nll) / static_cast<double>(kld.size()));
}

} // namespace ninferno::perplexity
