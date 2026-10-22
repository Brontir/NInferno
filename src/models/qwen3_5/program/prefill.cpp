#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/qwen3_5_execution.h"
#include "models/qwen3_5/program/context_clock.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/planning/prefill_unit.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/device.h"
#include "ninferno/ops/gdn_replay.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/scatter.h"
#include "ninferno/ops/speculative_round.h"
#include "runtime/contract/timing.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninferno::models::qwen3_5::execution {
namespace {

DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (!state.execution.io.dflash_prefill || state.dflash_prefill_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            auto& frame = *state.execution.io.dflash_prefill;
            // Target execution and draft append use the same chunk bindings. Decode may have
            // used another compact row, or a checkpoint may have forked the destination slot.
            // The controls are copied from this call's own pageable value, which the copy stages
            // before it returns: the host runs a chunk ahead of the device, and a reused pinned
            // buffer would be read only when the stream reaches the copy, after the next chunk
            // (or another lane's prefill) has rewritten it.
            const qwen3_5::DFlashPrefillIngress ingress{
                .append_count           = features.ne[1],
                .state_destination_slot = state.state_destination_slot,
                .full_kv_table_row      = state.dflash_kv_table_row,
            };
            CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &ingress, sizeof(ingress),
                                       cudaMemcpyHostToDevice, state.execution.device.stream));
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, frame.append_count,
                                  frame.state_destination_slot, frame.full_kv_table_row,
                                  {exact, exact});
            (void)rewrite_checkpoint;
        });
}

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_linear_state_slots(state_source_slot, state_destination_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

PrefillChunkResult prefill_text_chunk(PrefillContext& state, std::span<const TokenId> ids,
                                      std::uint32_t nominal_length,
                                      std::optional<std::uint32_t> split_frontier,
                                      bool finalize_at_end) {
    TextContext card(state.execution.device, qwen3_5::detail::qwen_parameters(state.execution),
                     state.execution.work, state.text_kv, state.execution.state_images.linear(),
                     state.execution.io, state.execution.prefill_hidden,
                     state.execution.prefill_chunk, state.text_kv_base, state.mtp_kv,
                     &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    const std::span<const int> prompt(ids.data(), ids.size());
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
}

PrefillChunkResult prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                                            VisionPrefillSession& vision,
                                            std::uint32_t nominal_length,
                                            std::optional<std::uint32_t> split_frontier,
                                            bool finalize_at_end) {
    TextContext card(state.execution.device, qwen3_5::detail::qwen_parameters(state.execution),
                     state.execution.work, state.text_kv, state.execution.state_images.linear(),
                     state.execution.io, state.execution.prefill_hidden,
                     state.execution.prefill_chunk, state.text_kv_base, state.mtp_kv,
                     &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision,
                                  finalize_at_end, sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

} // namespace ninferno::models::qwen3_5::execution

namespace ninferno::models::qwen3_5::detail {

namespace {

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token);

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

} // namespace

void ProgramImpl::start_sequence(std::uint32_t lane, SequenceState& sequence,
                                 MaterializationTransaction& transaction) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    RequestControl& request = requests[lane];
    if (!transaction.plan || transaction.plan->impl_ == nullptr || !transaction.prepared ||
        !request.prefill) {
        throw std::invalid_argument("materialization staging is incomplete");
    }
    AdmissionCandidateImpl& request_plan = *transaction.plan->impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }
    auto& staged                           = *request.prefill;
    const auto started                     = Clock::now();
    const std::uint32_t prompt_tokens      = staged.prompt_tokens;
    const std::uint32_t base               = staged.base;
    const std::uint32_t initial_mtp_extent = staged.initial_mtp_extent;
    request.lifecycle                      = Lifecycle::Empty;
    // The last frontier the sequence can reach: its KV lease never grows past it.
    request.lease_ceiling = std::min(
        capacity, request_plan.summary.prompt_tokens +
                      (request_plan.summary.effective_output_tokens == 0
                           ? 0U
                           : request_plan.summary.effective_output_tokens - 1U));
    request.lease_settled        = false;
    request.lease_space_limited  = false;
    request.lease_full_target    = {};
    request.lease_minimum_target = {};
    try {
        const std::uint32_t state_slots = request_plan.demand.active_entitlement.device.state_slots;
        const bool preserving_source =
            (transaction.has_source || transaction.has_shared_source) &&
            transaction.source_mode == runtime::PrivateSourceMode::Retain;
        const bool text_prefix_fork    = request_plan.text_prefix_fork_required;
        const bool backend_prefix_fork = request_plan.backend_prefix_fork_required;
        if (request_plan.reuse == ReusePath::Root) {
            if (transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_activation ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0) ||
                transaction.backend_activation.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("root materialization reservations are incomplete");
            }
            release_sequence_kv(sequence);
            release_sequence_state(sequence);
            sequence.state = ActiveStateBinding{.read  = transaction.reserved_states[0],
                                                .write = transaction.reserved_states[0]};
            transaction.reserved_states[0] = {};
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else if (preserving_source) {
            const bool private_source_ready = transaction.has_source &&
                                              transaction.source_index < continuation_capacity &&
                                              continuation_slots[transaction.source_index].role ==
                                                  ContinuationSlotRole::Catalogued;
            const bool shared_source_ready =
                transaction.has_shared_source &&
                transaction.shared_source_index < shared_prefix_capacity &&
                shared_prefix_slots[transaction.shared_source_index].role ==
                    SharedPrefixSlotRole::Catalogued;
            if (private_source_ready == shared_source_ready ||
                transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_prefix_fork ||
                !transaction.prefix_forks_ready ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("retained materialization is incomplete");
            }
            const StateImageHandle selected =
                private_source_ready
                    ? selected_state(continuation_states[transaction.source_index],
                                     request_plan.reuse, request_plan.selected_checkpoint)
                    : shared_prefix_states[transaction.shared_source_index].state;
            const StateImageHandle current = transaction.reserved_states[0];
            if (state_store->residency(selected) == StateReplicaResidency::HostOnly) {
                if (state_store->role(current) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("Host retained Fork destination was not published");
                }
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else if (transaction.split_state_identity) {
                if (!private_source_ready ||
                    state_store->residency(selected) != StateReplicaResidency::Both) {
                    throw std::logic_error("StateImage identity split source changed");
                }
                state_store->split_device_replica_identity(selected, current);
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else {
                const StateImageSelectors selectors = state_store->begin_fork(selected, current);
                if (is_masked_draft_backend(speculative_backend) &&
                    state_images->dflash_local() != nullptr) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read           = selected,
                    .write          = current,
                    .fork_pending   = true,
                    .read_ownership = StateReadOwnership::ExternalOwner,
                };
            }
            transaction.reserved_states[0]   = {};
            transaction.split_state_identity = false;
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;
            sequence.rewrite_state.reset();
            sequence.rewrite_checkpoint = {};

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else {
            if (request_plan.state_fork_required !=
                transaction.state_fork_destination.has_value()) {
                throw std::logic_error("private materialization StateImage Fork is incomplete");
            }
            if (transaction.reserved_state_count > 1 ||
                (transaction.reserved_state_count != 0 && sequence.reserved_state)) {
                throw std::logic_error("private materialization StateImage reservation is invalid");
            }
            if (transaction.reserved_state_count == 1) {
                sequence.reserved_state          = transaction.reserved_states[0];
                transaction.reserved_states[0]   = {};
                transaction.reserved_state_count = 0;
            }
        }

        if (!preserving_source) {
            std::array<HostKVPageReplicaRelease, 2> stale_tail_replicas{};
            std::size_t stale_tail_count           = 0;
            const auto preflight_inactive_truncate = [&](KVAddressSpaceStore& addresses,
                                                         LogicalKVPageStore& pages,
                                                         KVAddressSpaceHandle address,
                                                         std::optional<std::uint32_t> frontier) {
                if (!frontier || (addresses.committed_frontier(address) == *frontier &&
                                  addresses.mapped_pages(address) == paged_kv_pages(*frontier))) {
                    return;
                }
                bool releases_tail               = false;
                const std::uint32_t target_pages = paged_kv_pages(*frontier);
                if (target_pages != 0) {
                    const LogicalKVPageHandle tail =
                        addresses.logical_page(address, target_pages - 1U);
                    const std::uint32_t columns =
                        *frontier -
                        (target_pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
                    if (columns != pages.committed_columns(tail) && pages.host_resident(tail)) {
                        if (host_kv_extents == nullptr ||
                            stale_tail_count == stale_tail_replicas.size()) {
                            throw std::logic_error("stale Host KV tail replica is not releasable");
                        }
                        stale_tail_replicas[stale_tail_count++] =
                            HostKVPageReplicaRelease{.pages = &pages, .page = tail};
                        releases_tail = true;
                    }
                }
                if (!addresses.can_destructive_truncate_inactive(address, *frontier,
                                                                 releases_tail)) {
                    throw std::logic_error(
                        "selected private KV frontier is not destructively materializable");
                }
            };
            if (!sequence.kv) {
                throw std::logic_error("materialization destination has no KV address space");
            }
            if (!text_prefix_fork) {
                preflight_inactive_truncate(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                                            transaction.text_activation_frontier);
            }
            if (sequence.kv->backend && !backend_prefix_fork) {
                preflight_inactive_truncate(*backend_kv_addresses, *backend_kv_pages,
                                            *sequence.kv->backend,
                                            transaction.backend_activation_frontier);
            }
            if (stale_tail_count != 0) {
                const std::span<const HostKVPageReplicaRelease> releases(stale_tail_replicas.data(),
                                                                         stale_tail_count);
                if (!host_kv_extents->release_page_replicas(releases)) {
                    throw std::logic_error(
                        "stale Host KV tail replicas cannot be released atomically");
                }
            }
            if (!text_prefix_fork && transaction.text_activation_frontier &&
                (text_kv_addresses->committed_frontier(sequence.kv->text) !=
                     *transaction.text_activation_frontier ||
                 text_kv_addresses->mapped_pages(sequence.kv->text) !=
                     paged_kv_pages(*transaction.text_activation_frontier))) {
                text_kv_addresses->destructive_truncate_inactive(
                    sequence.kv->text, *transaction.text_activation_frontier);
            }
            if (!backend_prefix_fork && transaction.backend_activation_frontier &&
                sequence.kv->backend &&
                (backend_kv_addresses->committed_frontier(*sequence.kv->backend) !=
                     *transaction.backend_activation_frontier ||
                 backend_kv_addresses->mapped_pages(*sequence.kv->backend) !=
                     paged_kv_pages(*transaction.backend_activation_frontier))) {
                backend_kv_addresses->destructive_truncate_inactive(
                    *sequence.kv->backend, *transaction.backend_activation_frontier);
            }
            if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        }
        if ((text_prefix_fork || backend_prefix_fork) && !transaction.prefix_forks_ready) {
            throw std::logic_error("materialization prefix forks are incomplete");
        }
        if (text_prefix_fork) {
            text_kv_addresses->commit_prefix_fork(std::move(*transaction.text_prefix_fork),
                                                  device.stream);
            transaction.text_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = sequence.kv->text;
                sequence.kv->text                         = *transaction.root_text_address;
                transaction.root_text_address.reset();
                if (!text_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Text KV source remained pinned after COW");
                }
            }
        } else {
            text_kv_addresses->commit_activation(std::move(*transaction.text_activation),
                                                 device.stream);
            transaction.text_activation.reset();
        }
        if (backend_prefix_fork) {
            backend_kv_addresses->commit_prefix_fork(std::move(*transaction.backend_prefix_fork),
                                                     device.stream);
            transaction.backend_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = *sequence.kv->backend;
                sequence.kv->backend                      = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
                if (!backend_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Backend KV source remained pinned after COW");
                }
            }
        } else if (transaction.backend_activation) {
            backend_kv_addresses->commit_activation(std::move(*transaction.backend_activation),
                                                    device.stream);
            transaction.backend_activation.reset();
        }
        transaction.prefix_forks_ready = false;
        transaction.text_activation_frontier.reset();
        transaction.backend_activation_frontier.reset();
        transaction.prepared = false;

        const bool preserve_rewrite =
            request_plan.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting;
        const auto activate_consumed_state = [&](StateImageHandle selected) {
            if (!request_plan.state_fork_required) {
                if (transaction.state_fork_destination ||
                    state_store->checkpoint_references(selected) != 0) {
                    throw std::logic_error("planned StateImage Move is no longer valid");
                }
                state_store->move_checkpoint_to_active(selected);
                sequence.state = ActiveStateBinding{.read = selected, .write = selected};
                return;
            }
            if (!transaction.state_fork_destination ||
                state_store->checkpoint_references(selected) == 0) {
                throw std::logic_error("planned StateImage Fork is no longer valid");
            }
            const StateImageHandle destination = *transaction.state_fork_destination;
            if (transaction.state_restored) {
                if (state_store->role(destination) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("restored StateImage Fork destination is unavailable");
                }
                sequence.state = ActiveStateBinding{.read = destination, .write = destination};
            } else {
                const std::uint32_t references = state_store->checkpoint_references(selected);
                const std::uint32_t lineage_references =
                    owned_checkpoint_references(sequence, selected);
                if (lineage_references > references) {
                    throw std::logic_error("consumed StateImage Fork ownership is inconsistent");
                }
                const StateReadOwnership read_ownership =
                    lineage_references == references ? StateReadOwnership::LineageCheckpoint
                                                     : StateReadOwnership::ExternalOwner;
                const StateImageSelectors selectors =
                    state_store->begin_fork(selected, destination);
                if (is_masked_draft_backend(speculative_backend) &&
                    state_images->dflash_local() != nullptr) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read           = selected,
                    .write          = destination,
                    .fork_pending   = true,
                    .read_ownership = read_ownership,
                };
            }
            transaction.state_fork_destination.reset();
        };
        if (request_plan.reuse == ReusePath::Root) {
            sequence.rewrite_checkpoint = {};
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.prefix_digests.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
        } else if (preserving_source) {
            const SequenceState* private_source =
                transaction.has_source ? &continuation_states[transaction.source_index] : nullptr;
            SharedPrefixState* shared_source =
                transaction.has_shared_source
                    ? &shared_prefix_states[transaction.shared_source_index]
                    : nullptr;
            const std::uint32_t source_text_frontier =
                private_source != nullptr ? private_source->text_kv_valid : shared_source->frontier;
            if (!sequence.kv || source_text_frontier < base) {
                throw std::logic_error("retained prefix has incomplete Text KV");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base       = base == 0 ? 0 : base - 1U;
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->mtp_kv_valid
                                                         : shared_source->backend_frontier;
                // Without a backend store the drafter reads the restored target KV: nothing of
                // its own can be missing.
                if (!request_plan.prepare_mtp ||
                    (backend_kv_cache() != nullptr && source_backend < mtp_base)) {
                    throw std::logic_error("retained prefix has incomplete MTP KV");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend)) {
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->dflash_context_frontier
                                                         : shared_source->frontier;
                if (source_backend < base) {
                    throw std::logic_error("retained prefix has incomplete DFlash KV");
                }
                sequence.dflash_context_frontier = base;
            }
            // The tail is the selected checkpoint's own: its image holds the hidden of its
            // frontier when it was captured with one (a source's endpoint flag says nothing about
            // its anchors or its rewrite checkpoint, and is cleared when the endpoint is dropped).
            sequence.tail_hidden_valid =
                base == prompt_tokens &&
                (private_source != nullptr
                     ? checkpoint_tail_hidden_valid(*private_source,
                                                    request_plan.selected_checkpoint)
                     : shared_source->tail_hidden_valid);
            if (shared_source != nullptr) {
                if (shared_source->active_references == std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("shared-prefix active reference overflow");
                }
                ++shared_source->active_references;
                sequence.shared_prefix_references.push_back(transaction.shared_source_index);
            }
            refresh_state_views(sequence);
            bind_sequence_kv(sequence);
        } else if (request_plan.reuse == ReusePath::PrivateEndpoint) {
            if (!state_store->valid(sequence.state.read) ||
                sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable) {
                throw std::logic_error("resident endpoint StateImage is not movable");
            }
            if (!preserve_rewrite && sequence.rewrite_state) {
                const StateImageHandle dropped = *sequence.rewrite_state;
                state_store->release_checkpoint_reference(dropped);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
                if (dropped != sequence.state.read &&
                    state_store->checkpoint_references(dropped) == 0 &&
                    !state_store->release(dropped)) {
                    throw std::logic_error("dropped rewrite StateImage could not be released");
                }
            }
            activate_consumed_state(sequence.state.read);
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend) &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            if (!sequence.rewrite_state || !state_store->valid(*sequence.rewrite_state) ||
                state_store->role(*sequence.rewrite_state) != StateImageRole::CheckpointImmutable ||
                (sequence.endpoint_valid &&
                 (!state_store->valid(sequence.state.read) ||
                  sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                  state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable))) {
                throw std::logic_error("resident rewrite StateImage is not movable");
            }
            const StateImageHandle checkpoint = *sequence.rewrite_state;
            if (sequence.endpoint_valid && sequence.state.read == checkpoint) {
                throw std::logic_error("resident endpoint aliases its rewrite StateImage");
            }
            if (sequence.endpoint_valid && !state_store->release(sequence.state.read)) {
                throw std::logic_error("superseded endpoint StateImage could not be released");
            }
            if (!preserve_rewrite) {
                state_store->release_checkpoint_reference(checkpoint);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
            }
            activate_consumed_state(checkpoint);
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend)) {
                if (!dflash || (backend_kv_cache() && !sequence.kv->backend) ||
                    sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                sequence.dflash_context_frontier = base;
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.tail_hidden_valid = base == prompt_tokens;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        sequence.endpoint_valid = false;
        if (!preserving_source) { trim_sequence_kv(sequence, base, backend_kv_valid(sequence)); }
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            backend_kv_cache() == nullptr ? 0U
            : speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash ? prompt_tokens
                                                                : 0U;
        ensure_sequence_kv_mapped(sequence, prompt_tokens, backend_materialized);
        // A prompt past the context bound decodes plainly (SpeculativeOptions::context_limit).
        RequestSpeculationPlan speculation = request_plan.speculation;
        const bool past_context_limit =
            speculation_context_limit != 0 && prompt_tokens > speculation_context_limit;
        if (past_context_limit) {
            speculation.draft_depth = 0;
            speculation.step_down.clear();
        }
        // The prompt's context class bounds its depth (SpeculativeOptions::class_depth): only ever
        // lower, onto a step-down depth with that step's floor where it is one; only a request at
        // the default depth the bounds were decided for.
        const bool class_bounded =
            spec::class_bounds_apply(request_plan.speculation.draft_depth, default_draft_depth);
        bool class_plain               = false;
        std::uint32_t class_lowered_to = 0;
        if (class_bounded && speculation.draft_depth != 0) {
            std::vector<std::pair<std::uint32_t, double>> steps;
            for (const SpeculationStep& step : speculation.step_down) {
                steps.emplace_back(step.draft_tokens, step.acceptance_floor);
            }
            const spec::ClassClamp clamp = spec::clamp_to_class(
                speculation_class_depth, prompt_tokens, speculation.draft_depth, steps);
            if (clamp.lowered(speculation.draft_depth)) {
                speculation.draft_depth = clamp.depth;
                speculation.step_down.erase(speculation.step_down.begin(),
                                            speculation.step_down.begin() +
                                                static_cast<std::ptrdiff_t>(clamp.dropped));
                if (clamp.floor) { speculation.acceptance_floor = *clamp.floor; }
                if (clamp.depth == 0) { speculation.step_down.clear(); }
                class_plain      = clamp.depth == 0;
                class_lowered_to = clamp.depth;
            }
        }
        install_sampling(sequence, request, request_plan.sampling, speculation);
        request.class_bounded                   = class_bounded;
        request.speculative_stats.class_plain   = class_plain;
        request.speculative_stats.class_depth   = class_lowered_to;
        request.speculative_stats.context_plain = past_context_limit &&
                                                  speculative_backend != SpeculativeBackend::None &&
                                                  request_plan.speculation.draft_depth != 0;
        sequence.rope_delta = staged.prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);

        request.timings              = {};
        request.pending              = {};
        request.publish_continuation = request_plan.summary.publish_continuation;
        sequence.mtp_draft_count     = 0;
        sequence.tail_hidden_valid   = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.swap(materialization_ledger_);
        sequence.prefix_identity.swap(materialization_identity_);
        sequence.prefix_digests.swap(materialization_prefix_digests_);
        sequence.rebuild_work       = request_plan.root_rebuild_work;
        sequence.rebuild_tail_begin = request_plan.root_rebuild_tail_begin;

        if (is_masked_draft_backend(speculative_backend)) {
            if (!dflash || !io.dflash_prefill || !dflash_prefill_host_ingress ||
                (backend_kv_cache() && !sequence.kv->backend)) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
        }

        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle = Lifecycle::Prefilling;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane_best_effort(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult
ProgramImpl::advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    round_after_other_work = true;
    return advance_prefill(active_sequence(lane), requests[lane], failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_prefill_raw(std::uint32_t lane, bool terminal,
                                                          runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("prefill resolution requires a pending prefill token");
    }
    return resolve_non_speculative_pending(active_sequence(lane), requests[lane], 1, terminal,
                                           std::nullopt, failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_pending_raw(
    std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
    std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
    std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size() ||
        prefix_execution_splits.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (lanes.size() == 1 && lanes.front() < max_concurrency &&
        requests[lanes.front()].pending.kind == PendingKind::Begin) {
        const std::uint32_t lane = lanes.front();
        if (requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("prefill pending token no longer matches Program state");
        }
        if (cancelled.front()) {
            if (accepted_tokens.front() != 0 || !terminal.front()) {
                throw std::logic_error("cancelled prefill pending decision is invalid");
            }
            if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                throw std::logic_error("cancelled prefill lane is not strictly releasable");
            }
        } else {
            timing.pause();
            timing.include(resolve_non_speculative_pending(
                active_sequence(lane), requests[lane], accepted_tokens.front(),
                terminal.front() != 0, prefix_execution_splits.front(), failed_timing));
            timing.resume_post();
        }
        return timing.finish();
    }

    // Rows of the step's ordinary launch (every row of an ordinary engine; the plain rows of a
    // speculative engine) resolve one by one. The speculative launch's rows resolve together, in
    // launch order: their egress, frames and replay records are indexed by launch row.
    std::array<std::size_t, kMaximumConcurrency> launch_rows{};
    std::array<bool, kMaximumConcurrency> launch_row_seen{};
    std::size_t speculative_count = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            (requests[lane].pending.kind != PendingKind::Ordinary &&
             (requests[lane].pending.kind != PendingKind::Speculative ||
              speculative_backend == SpeculativeBackend::None))) {
            throw std::logic_error("decode pending batch no longer matches Program state");
        }
        if (requests[lane].pending.kind == PendingKind::Speculative) {
            const std::uint32_t launch_row = requests[lane].pending.launch_row;
            if (launch_row >= lanes.size() || launch_row_seen[launch_row]) {
                throw std::logic_error("speculative pending rows do not form one launch");
            }
            launch_row_seen[launch_row] = true;
            launch_rows[launch_row]     = row;
            ++speculative_count;
            continue;
        }
        if (cancelled[row]) {
            if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                throw std::logic_error("cancelled decode lane is not strictly releasable");
            }
        } else {
            timing.pause();
            timing.include(resolve_non_speculative_pending(
                active_sequence(lane), requests[lane], accepted_tokens[row], terminal[row] != 0,
                prefix_execution_splits[row], failed_timing));
            timing.resume_post();
        }
    }
    if (speculative_count == 0) { return timing.finish(); }
    std::array<std::uint32_t, kMaximumConcurrency> launch_lanes{};
    std::array<std::uint32_t, kMaximumConcurrency> launch_accepted{};
    std::array<std::uint8_t, kMaximumConcurrency> launch_terminal{};
    std::array<std::uint8_t, kMaximumConcurrency> launch_cancelled{};
    std::array<std::optional<std::uint32_t>, kMaximumConcurrency> launch_splits{};
    for (std::size_t i = 0; i < speculative_count; ++i) {
        if (!launch_row_seen[i]) {
            throw std::logic_error("speculative pending rows do not form one launch");
        }
        const std::size_t row = launch_rows[i];
        launch_lanes[i]       = lanes[row];
        launch_accepted[i]    = accepted_tokens[row];
        launch_terminal[i]    = terminal[row];
        launch_cancelled[i]   = cancelled[row];
        launch_splits[i]      = prefix_execution_splits[row];
    }
    timing.pause();
    timing.include(resolve_speculative_pending(
        std::span<const std::uint32_t>(launch_lanes.data(), speculative_count),
        std::span<const std::uint32_t>(launch_accepted.data(), speculative_count),
        std::span<const std::uint8_t>(launch_terminal.data(), speculative_count),
        std::span<const std::uint8_t>(launch_cancelled.data(), speculative_count),
        std::span<const std::optional<std::uint32_t>>(launch_splits.data(), speculative_count),
        failed_timing));
    timing.resume_post();
    return timing.finish();
}

// The rows of one speculative launch, in launch order.
runtime::ExecutionTiming ProgramImpl::resolve_speculative_pending(
    std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
    std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
    std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (lanes.empty() || lanes.front() >= max_concurrency) {
        throw std::logic_error("speculative pending batch is empty");
    }
    // The launch's window: its frames, egress stride, records and fold.
    const std::uint32_t launch_drafts = requests[lanes.front()].pending.launch_window;
    const auto window_it = std::find_if(launch_windows.begin(), launch_windows.end(),
                                        [&](const LaunchWindow& w) { return w.drafts == launch_drafts; });
    if (window_it == launch_windows.end()) {
        throw std::logic_error("speculative pending batch names no launch window");
    }
    const LaunchWindow& window = *window_it;

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    // A batch whose rows all continue decoding commits every licensed token: its tail is the
    // Fold (plus DFlash2 counter updates), all device-only and stream-ordered before the next
    // round. Only terminal or cancelled rows read tail results on the host or hand pinned
    // ingress to the tail, so only they need the host to wait for it.
    bool host_waits_for_tail = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative ||
            requests[lane].pending.launch_window != launch_drafts) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = active_sequence(lane);
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.prefix_digests.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (is_masked_draft_backend(speculative_backend) &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (!cancelled[row] && (committed == 0 || committed > pending.produced ||
                                 (!terminal[row] && committed != pending.produced)))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        const StateImageSelectors selectors = state_selectors(sequence);
        fold_rows[row] =
            ops::GdnReplayFoldRow{.source_state_slot      = selectors.source,
                                  .destination_state_slot = selectors.destination,
                                  .commit_columns         = static_cast<std::int32_t>(committed)};
        host_waits_for_tail = host_waits_for_tail || cancelled[row] != 0 || terminal[row] != 0;
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    const auto tail_started = Clock::now();
    try {
        timing.resume_submit();
        if (window.family_fold) {
            window.family_fold(
                std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                device.stream);
        } else if (window.fold) {
            window.fold->execute(
                std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                device.stream);
        }
        // Submit the fold now (WDDM batches submissions) so it runs while the host prepares the
        // commit below.
        (void)cudaStreamQuery(device.stream);

        // Sparse acceptance reads counts. Publish only the prefix licensed by the Frontend.
        if (speculative_backend == SpeculativeBackend::DFlash2) {
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (cancelled[row] || !requests[lanes[row]].sampling_host.token_counts) {
                    continue;
                }
                const auto count = static_cast<std::int32_t>(accepted_tokens[row]);
                Tensor ids =
                    window.dflash->licensed_tokens.slice(1, static_cast<std::int32_t>(row), 1)
                        .slice(0, 0, count)
                        .view({count});
                Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(lanes[row]), 1)
                                    .view({model.traits().public_tokens});
                ops::increment_token_counts(ids, counts, device.stream);
            }
        }

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && window.mtp) {
                const qwen3_5::MtpDecodeState& frame = *window.mtp;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else if (is_masked_draft_backend(speculative_backend) && window.dflash) {
                const qwen3_5::DFlashDecodeState& frame = *window.dflash;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, state_images->continuation_hidden_store(),
                         device.stream);
        }

        if (is_masked_draft_backend(speculative_backend)) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        if (host_waits_for_tail) {
            timing.begin_wait();
            device.synchronize();
            timing.end_wait();
        } else if (device.transfer_stream != nullptr) {
            // State-image transfers issued after this commit must observe the folded state.
            speculative_tail_ready_.record(device.stream);
            speculative_tail_ready_.wait(device.transfer_stream);
        }
        work.reset();
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        clear_execution_failure_lanes(lanes);
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = launch_drafts + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = active_sequence(lanes[row]);
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                if (!clear_lane_strict(sequence, request)) {
                    throw std::logic_error("cancelled speculative lane is not strictly releasable");
                }
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            settle_state_fork(sequence);
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            commit_generated_prefix_identity(sequence, pending.base_S,
                                             std::span<const TokenId>(token_base, committed),
                                             prefix_execution_splits[row]);
            advance_rebuild_work(sequence, pending.base_E + committed, prefill_chunk);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                request.lifecycle = Lifecycle::Finishable;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        clear_execution_failure_lanes(lanes);
        throw;
    }
    return timing.finish();
}

runtime::PrefillStepResult ProgramImpl::advance_prefill(SequenceState& sequence,
                                                        RequestControl& request,
                                                        runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    if (staged.pending_capture_offer != 0) {
        throw std::logic_error("prefill cannot advance while a capture offer is pending");
    }
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse};
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    // The unit's own prefill work, priced by the Engine against decode: its model calls, tokens,
    // attention pairs from its start position, and the images whose first span it reached.
    const std::uint32_t unit_begin = staged.cursor;
    std::uint64_t unit_calls       = 0;
    const auto unit_work           = [&]() {
        runtime::PrefillWork work =
            runtime::make_prefill_work(unit_begin, processed_prompt_tokens, 0, 0, prefill_chunk);
        work.chunks = unit_calls;
        for (const qwen3_5::VisionItem& item : staged.prompt.vision_items) {
            if (item.token_spans.empty()) { continue; }
            std::size_t first = item.token_spans.front().begin;
            for (const qwen3_5::TokenSpan& span : item.token_spans) {
                first = std::min(first, span.begin);
            }
            if (first >= unit_begin && first < staged.cursor) {
                ++work.vision_items;
                work.vision_patches += static_cast<std::uint64_t>(item.patch_count);
            }
        }
        return work;
    };
    // The first generated token is sampled by the prefill that reaches the prompt frontier. Its
    // mask depends only on the constraint's initial state, so it is computed and released before
    // any kernel of this unit is enqueued and installed in the lane's sampling configuration.
    const std::uint32_t mask_lane = sequence.lane;
    runtime::MaskedUnit masks(token_masks.get(), active_masks,
                              std::span<const std::uint32_t>(&mask_lane, 1));
    if (masks.active(0)) {
        masks.fill(0, {});
        masks.release();
        const ops::SamplingConfig masked = masks.config(0, request.sampling_host);
        Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(mask_lane), 1);
        CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &masked, sizeof(masked),
                                   cudaMemcpyHostToDevice, device.stream));
    }
    try {
        if (staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == staged.cursor) {
            if (staged.cursor != staged.base ||
                !staged.capture_groups[staged.next_capture].shared ||
                staged.capture_groups[staged.next_capture].rewrite ||
                staged.capture_groups[staged.next_capture].long_anchor) {
                throw std::logic_error("zero-prefill capture is not a shared base promotion");
            }
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            staged.pending_capture_offer = next_capture_offer_id_;
            return runtime::PrefillStepResult{
                .summary = summary,
                .timing  = timing.finish(),
            };
        }
        StateImageSelectors selectors = state_selectors(sequence);
        Tensor rewrite_capture_hidden;
        Tensor* rewrite_capture_hidden_ptr = nullptr;
        if (staged.next_capture < staged.capture_groups.size()) {
            rewrite_capture_hidden = state_images->continuation_hidden_slot(selectors.destination);
            rewrite_capture_hidden_ptr = &rewrite_capture_hidden;
        }
        execution::PrefillContext schedule_state{
            {device, model, work, *state_images, replay_records ? &*replay_records : nullptr, io,
             prefill_hidden, prefill_chunk, capacity, proposal_head, draft_width,
             &speculation_load},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            rewrite_capture_hidden_ptr,
            selectors.source,
            selectors.destination,
            staged.initial_mtp_extent,
            0,
            dflash_prefill_host_ingress};
        // Forced tokens of another lane, or another lane's bind, may have repointed the prefill KV
        // row scalars since this lane's bind; every prefill step publishes its own rows.
        publish_kv_rows(sequence);

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = sequence.tail_hidden;
            const execution::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                model.mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision, bridge);
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                model.mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                             bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            staged.mtp_bridge = MtpBridgeMode::None;
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::optional<std::uint32_t> unit_capture =
                staged.next_capture < staged.capture_groups.size()
                    ? std::optional<std::uint32_t>(
                          staged.capture_groups[staged.next_capture].frontier)
                    : std::nullopt;
            std::uint32_t nominal =
                model.prefill_call_tokens(prefill_chunk, staged.cursor,
                                          runtime_support::staged_prefill_unit_end(
                                              staged.cursor, staged.prompt_tokens, unit_capture));
            if (staged.vision) {
                nominal = staged.vision->unit_end(staged.cursor, staged.cursor + nominal) -
                          staged.cursor;
            }
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (is_masked_draft_backend(speculative_backend)) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            std::uint32_t remaining          = nominal;
            std::uint32_t final_chunk_tokens = 0;
            bool finalized                   = false;
            while (remaining != 0) {
                schedule_state.text_kv_base           = staged.cursor;
                selectors                             = state_selectors(sequence);
                schedule_state.state_source_slot      = selectors.source;
                schedule_state.state_destination_slot = selectors.destination;
                schedule_state.dflash_kv_table_row =
                    sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                         : 0;
                if (staged.next_capture < staged.capture_groups.size()) {
                    rewrite_capture_hidden =
                        state_images->continuation_hidden_slot(selectors.destination);
                    schedule_state.rewrite_checkpoint_hidden = &rewrite_capture_hidden;
                } else {
                    schedule_state.rewrite_checkpoint_hidden = nullptr;
                }

                const bool final_candidate = staged.cursor + remaining == staged.prompt_tokens;
                const std::optional<std::uint32_t> capture_frontier =
                    staged.next_capture < staged.capture_groups.size()
                        ? std::optional<std::uint32_t>(
                              staged.capture_groups[staged.next_capture].frontier)
                        : std::nullopt;
                std::optional<std::uint32_t> split_frontier = capture_frontier;
                const auto rewrite_split                    = std::upper_bound(
                    staged.prompt.identity.rewrite_execution_frontiers.begin(),
                    staged.prompt.identity.rewrite_execution_frontiers.end(), staged.cursor);
                if (rewrite_split != staged.prompt.identity.rewrite_execution_frontiers.end() &&
                    (!split_frontier || *rewrite_split < *split_frontier)) {
                    split_frontier = *rewrite_split;
                }
                execution::PrefillChunkResult result;
                timing.pause();
                if (staged.vision) {
                    if (!workspace_plan.vision) {
                        throw std::logic_error("active Vision prefill lost its workspace plan");
                    }
                    mark_workspace_usage(workspace_plan.vision->capacity_bytes);
                    result = model.prefill_multimodal_chunk(schedule_state, staged.prompt,
                                                            *staged.vision, remaining,
                                                            split_frontier, final_candidate);
                } else {
                    result = model.prefill_text_chunk(
                        schedule_state, std::span<const TokenId>(staged.prompt.token_ids),
                        remaining, split_frontier, final_candidate);
                }
                timing.include(result.timing);
                timing.resume_post();
                ++unit_calls;
                if (result.processed_tokens == 0 || result.processed_tokens > remaining) {
                    throw std::logic_error("ordinary prefill chunk made invalid progress");
                }
                if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
                staged.cursor += result.processed_tokens;
                processed_prompt_tokens += result.processed_tokens;
                remaining -= result.processed_tokens;
                final_chunk_tokens     = result.processed_tokens;
                sequence.text_kv_valid = staged.cursor;
                // A call that ends on its split frontier with the capture output bound writes the
                // hidden of its last token into the destination image's continuation slot (the
                // ModelExecution contract; the image the sequence commits below). The image then
                // carries the tail of its own frontier, and so does a checkpoint captured from it:
                // a later prompt that ends there samples from it without a prefill (zero suffix).
                // Any other call leaves that slot behind the frontier.
                sequence.tail_hidden_valid = schedule_state.rewrite_checkpoint_hidden != nullptr &&
                                             split_frontier && staged.cursor == *split_frontier;
                if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
                if (is_masked_draft_backend(speculative_backend)) {
                    sequence.dflash_context_frontier = staged.cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));

                // Prompt transitions are canonical immediately. If this was the first write after
                // an immutable source, close the Fork before potentially freezing a new rewrite.
                settle_state_fork(sequence);
                const bool reached_capture = capture_frontier && staged.cursor == *capture_frontier;
                if (reached_capture) {
                    if (result.finalized) {
                        // The prompt-frontier state becomes publishable only after the generated
                        // Begin token is committed. commit() emits the offer for this group.
                    } else {
                        staged.elapsed_seconds +=
                            std::chrono::duration<double>(Clock::now() - started).count();
                        if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
                        staged.pending_capture_offer = next_capture_offer_id_;
                        return runtime::PrefillStepResult{
                            .summary                 = summary,
                            .processed_prompt_tokens = processed_prompt_tokens,
                            .work                    = unit_work(),
                            .timing                  = timing.finish(),
                        };
                    }
                }

                finalized = result.finalized;
                if (finalized || remaining == 0) { break; }
            }

            if (!finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary                 = summary,
                    .processed_prompt_tokens = processed_prompt_tokens,
                    .work                    = unit_work(),
                    .timing                  = timing.finish(),
                };
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            timing.resume_submit();
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(final_chunk_tokens) - 1, 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                static constexpr const char* kPaths[] = {
                    "root", "private endpoint", "private turn closure", "private response replay",
                    "private long anchor", "shared stable prefix"};
                const auto path = static_cast<std::size_t>(staged.reuse);
                throw std::logic_error(
                    std::string("zero-suffix reuse has no target tail hidden (") +
                    (path < std::size(kPaths) ? kPaths[path] : "unknown path") + " at " +
                    std::to_string(staged.base) + " of " + std::to_string(staged.prompt_tokens) +
                    " prompt tokens)");
            }
            model.sample_from_hidden(schedule_state, sequence.tail_hidden,
                                     checked_i32(staged.prompt_tokens, "sample position"),
                                     ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                model.mtp_bridge_and_propose(
                    schedule_state, io.token, sequence.tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                sequence.mtp_kv_valid = staged.prompt_tokens;
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                staged.mtp_bridge = MtpBridgeMode::None;
            }
        }

        // The tail hidden is the prompt's final hidden on every path (copied, or reused whole).
        numeric_status.clear();
        numeric_status.check(NumericStatus::Launch::Prefill,
                             sequence.tail_hidden.view({sequence.tail_hidden.ne[0], 1, 1}), nullptr,
                             device.stream);
        copy_round_token();
        std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        masks.collect_faults();
        prefill_fault = numeric_status.faulted(NumericStatus::Launch::Prefill, 0) ? 1U : 0U;
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds       = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.prefix_digests.append_generated(std::span<const TokenId>(host_tokens, 1),
                                                 sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (is_masked_draft_backend(speculative_backend) &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        staged.prompt.release_all_media_payloads();
        if (staged.vision) { staged.vision->retire_handoff(); }

        const bool prompt_frontier_capture =
            staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == prompt_tokens;
        if (!prompt_frontier_capture) { request.prefill.reset(); }
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens     = std::span<const TokenId>(host_tokens, 1),
                                               .row_faults = prefill_fault != 0U
                                                                 ? std::span<const std::uint8_t>(
                                                                     &prefill_fault, 1)
                                                                 : std::span<const std::uint8_t>{}},
            .processed_prompt_tokens = processed_prompt_tokens,
            .work                    = unit_work(),
            .complete                = true,
            .timing                  = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        const std::uint32_t lane = sequence.lane;
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        throw;
    }
}


} // namespace ninferno::models::qwen3_5::detail
