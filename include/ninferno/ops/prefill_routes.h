#pragma once

namespace ninferno::ops {

/**
 * Prefill route profile.
 *
 * A prefill chunk's size is decided by where a request was cut (the chunk size, a cache frontier,
 * a restore base), not by the model, so a token's prefill result must not depend on the size of
 * the call that carries it. Inside a ScopedPrefillRoutes the ops whose route tables change with T
 * take, for every call of at least 17 tokens, only routes that give a column the same bytes.
 * Outside it (decode, speculative verify, drafters, other models) they keep their speed-tuned
 * tables: a verify pass is compared against its own draft and needs no invariance.
 *
 * The profile is a host-side, per-thread choice read when an op resolves its route, so it applies
 * to eager launches and to captures alike. Workspace capacity queries cover both profiles.
 */
[[nodiscard]] bool prefill_routes_active() noexcept;

class ScopedPrefillRoutes {
public:
    // `active` false selects the speed profile for the scope, so a verify pass nested in a
    // prefill (none today) could not inherit the prefill routes.
    explicit ScopedPrefillRoutes(bool active = true) noexcept;
    ~ScopedPrefillRoutes();

    ScopedPrefillRoutes(const ScopedPrefillRoutes&)            = delete;
    ScopedPrefillRoutes& operator=(const ScopedPrefillRoutes&) = delete;

private:
    bool previous_;
};

/**
 * Invariant route profile (ActivationMode::Bits16, ninferno/types.h).
 *
 * Inside a ScopedInvariantRoutes a token's bytes do not depend on the call that carries it at any
 * width: not on how many batch rows share a decode step, nor on where a prompt was cut (a prefix
 * cache frontier may leave a prefill call of a single token). The ops that read it take, at every
 * T, one summation order per problem with BF16 activations:
 *   - the generic projections (ops/generic): the register MMA with the problem's K split, never a
 *     table entry, a tiled, W4A4 or W4A8 route, and never a registered shape's own kernels;
 *   - the generic MoE (ops::moe): the same schedules for the router and both grouped GEMMs;
 *   - gemma4_attention: every call takes the prefill kernel, whose row blocks and key tiles sit
 *     on the absolute-position grid: a prompt cut anywhere, a verify call and the decode steps of
 *     its columns give a token the same bytes, whatever the batch's envelope;
 *   - causal_softmax_attention (every KV storage): one grouped kernel instance at every width, a
 *     call wider than 64 columns in chunks of 64, each token's key splits aligned to its own
 *     position (at most 64 over the keys it sees, a power of two of at least 512 keys each,
 *     CausalKvPartition::aligned_to_positions) and merged in split order: a decode step, a verify
 *     pass, a batch neighbour, a prompt cut or the KV capacity a session sized from free memory
 *     leaves a token's bytes unchanged;
 *   - gated_delta_net: every call takes the token-serial recurrent kernel (the one decode and
 *     verify run) instead of the 16-token chunked scan, so a prompt cut or a cache restore at any
 *     position, a KV-lease re-prefill and the decode steps give a token and the state the same
 *     bytes.
 * It changes routes only; a family whose ops do not read it gives no such guarantee. One that runs
 * inside the profile and does not read it: the DFlash drafter's attention over its own context
 * (context_softmax_attention, dense/context/launch.cu) takes its split count and key block from
 * the session's capacity (envelope.max_context), so a session sized from free memory can draft
 * other tokens and accept a different number of them; under greedy decoding the target's verify
 * still decides every emitted token, so the emitted bytes do not change.
 */
[[nodiscard]] bool invariant_routes_active() noexcept;

class ScopedInvariantRoutes {
public:
    explicit ScopedInvariantRoutes(bool active = true) noexcept;
    ~ScopedInvariantRoutes();

    ScopedInvariantRoutes(const ScopedInvariantRoutes&)            = delete;
    ScopedInvariantRoutes& operator=(const ScopedInvariantRoutes&) = delete;

private:
    bool previous_;
};

/**
 * Lanes of the current round.
 *
 * A verify or MTP pass of T columns may carry one sequence (a C1 copy round: 16 columns of one
 * request) or several (a C4 verify: 4 requests of 4 columns); the width alone does not tell them
 * apart, while the routes that pay at one lose at the other (the ggml matrix kernel at verify
 * widths gains with several sequences and loses with one). Inside a
 * ScopedRoundLanes the ops read the number of sequences in the call; 0 (outside any scope: prefill,
 * tests, capacity queries) leaves them to decide by width alone. Read when an op resolves its
 * route, so it applies to eager launches and to graph captures, whose profiles already differ by
 * batch size.
 */
[[nodiscard]] int round_lanes() noexcept;

class ScopedRoundLanes {
public:
    explicit ScopedRoundLanes(int lanes) noexcept;
    ~ScopedRoundLanes();

    ScopedRoundLanes(const ScopedRoundLanes&)            = delete;
    ScopedRoundLanes& operator=(const ScopedRoundLanes&) = delete;

private:
    int previous_;
};

} // namespace ninferno::ops
