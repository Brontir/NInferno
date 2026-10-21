# NInferno Resource Scheduling and Context Cache

This document defines the rules by which NInferno selects, validates, and commits context state under limited Device/Host resources. It is the maintainer authority for admission
materialization, prefix reuse, cache retention, and pressure planning.

The problem this document solves is:

> After the Scheduler has selected a waiting request, from which recoverable position should it start, and which inactive contexts should it retain or degrade,
> in order to reach, at the lowest predicted cost, a terminal state that is physically executable and does not break the completion guarantee of any active request?

The top-level request order and lifecycle are defined by [Engine Architecture](engine-architecture.md). The physical contracts of KV pages, address spaces,
replicas, and consumer views are defined by [Paged KV Context Store](paged-kv-cache.md).

---

## 1. Terminology and Boundaries

This document uses the following terms:

| Term | Meaning |
|---|---|
| owner | An active sequence, private continuation, or shared prefix |
| checkpoint | The immutable complete state of a recoverable frontier within an owner |
| candidate | A root or exact checkpoint source that the incoming request can adopt |
| planning target | The complete logical terminal state of a candidate together with all affected inactive owners |
| reservation | Capacity that has been deducted from the global available capacity and set aside for future materialization or transaction purposes |
| resource plan | An ordered physical transition sealed by the Program for one target |
| placement | The Device/Host replica state of one complete StateImage or one KV logical page |

In the rest of this document, the planning target is abbreviated as `target`, meaning the terminal state of one resource decision.

Prefix reuse is one kind of source selection in admission. It reuses an already computed prefix with complete identity and state coverage, reducing redundant
prefill while preserving model semantics and request order. Different prefill chunkings or numerical paths are not required to produce identical logits or generated tokens.

The current product setting is a single GPU, a single resident model, a fixed `max_concurrency=1..8`, and non-preemptible active requests.
This yields two basic rules:

1. The Scheduler first determines the request for the current attempt, and the resource layer optimizes only the materialization of that request;
2. Once a request is published as Active, its maximum legal execution range has a complete resource guarantee, and the inactive cache policy
   may no longer borrow that capacity.

---

## 2. Decision Ownership

### 2.1 Scheduler

The Scheduler selects:

- the current FIFO head or a backfill request that has been proven safe;
- the timing of admission attempts;
- the execution order of prefill, control, and decode.

The Scheduler does not read checkpoint placement, victim, page, or allocator facts.

### 2.2 ResourceManager

The ResourceManager owns the logical cache policy:

- the private/shared catalog and the SessionIndex;
- the candidate shortlist and logical claims;
- retention class, hit history, and publication order;
- target search and cost comparison across candidates;
- logical adoption of the Program's terminal results.

The ResourceManager may know the Program summary that "a checkpoint can be restored from Device, Host, or root",
but it keeps no copy of physical occupancy, refcounts, free pages, or Host arena geometry.

The ResourceManager's logical protection of an active request is stored as an owner edge, not as a catalog capability or a refcount mirror.
A capability generation is a transient structural snapshot for planning/transactions; an owner edge is a logical lease that lasts until the active request's terminal
settlement. When another reader changes the Device/Host residency of the same owner, the generation may advance, but this does
not make existing owner edges stale. All active logical reference counts are derived from these edges.

### 2.3 Program

The Program is the sole physical authority and owns:

- the real objects, references, and replicas of State and KV;
- Device pools, Host stores, allocators, and reservations;
- continuation completeness and exact prefix identity;
- Move, Fork, COW, transfer, and last-reference release;
- the complete post-state and the physical feasibility of every transition stage;
- the `ResourcePlan` bound to the current resource state.

The call relationship is:

```text
Scheduler selects one request
        │
        ▼
ResourceManager enumerates candidates and compares targets
        │
        ▼
Program projects each complete target from real physical state
        │
        ▼
Program seals and executes one ResourcePlan
        │
        ▼
ResourceManager adopts one stable ResourceResult
```

Logical policy and physical facts interact through the candidate assessment, the opaque target handle, the `ResourcePlan`, and the
`ResourceResult`, not through a physical ledger shared across layers.

---

## 3. Resource Model

### 3.1 Capacity Axes

At startup the Program establishes a fixed capacity vector:

```text
PhysicalCapacity = {
    Device StateImage slots,
    Host StateImage slots,
    Device page groups for each typed KV pool,
    Host KV arena bytes and allocator geometry,
}
```

Capacity is derived from the loaded model configuration, the actual weight bindings, the enabled components, and the startup options. The physical representation of the weights affects the execution
workspace and the available budget; the mathematical state, concrete storage, and capacity of KV/GDN are the responsibility of the model implementation and the Program respectively.

These axes are independent. Headroom on one resource axis cannot compensate for a shortfall on another. The total free bytes of the Host KV likewise cannot replace
the allocator's judgment of whether a concrete extent geometry can be allocated.

The Engine also has fixed logical capacities:

- active lanes;
- private continuation catalog slots;
- shared prefix catalog slots;
- the checkpoint/anchor limit per continuation.

Lanes and catalog slots do not count toward physical bytes, but a target can be published only if it satisfies both the logical and the physical capacity.

### 3.2 Unique Physical Occupancy

For any physical resource dimension \(r\), the occupancy of state \(S\) is:

\[
Used_r(S)=
\sum_{x\in UniqueAllocations(S)} size_r(x)
+
\sum_{y\in ConcreteReservations(S)} size_r(y)
\]

`UniqueAllocations` includes:

- Device/Host StateImage replicas;
- Device KV page replicas;
- Host KV extents.

`ConcreteReservations` includes:

- future growth of an active request that has not yet been materialized;
- partial-tail COW destinations;
- copy destinations and replacement capacity of resource transitions.

When one physical allocation is referenced by multiple checkpoints or address spaces, it is counted once. Capacity that has been materialized
counts as an allocation, and capacity that is not yet materialized but protected counts as a reservation; one unit of capacity cannot be counted in both.

Global physical occupancy and owner-exclusive resources are two different views. The former comes directly from the unique
allocations and concrete reservations of the State/KV stores; the latter represents only the resources that can be settled when a single owner is removed in isolation. An allocation referenced by both a
private checkpoint and a shared prefix is counted once in global occupancy, but in neither owner's exclusive
resources. Owner-exclusive resources can be used for eviction effect, ownership transfer, and active
entitlement validation, but not for judging whether a checkpoint has a resident replica; that can only be queried from the corresponding physical
store.

Every stable state and every intermediate transition state must satisfy:

\[
Used_r(S)\le Capacity_r
\]

### 3.3 Complete Post-state and Joint Reclaim

Resource reclamation is determined by the final remaining references. For a set of logical actions \(V\):

\[
Reclaim_r(S,V)=
Used_r(S)-Used_r(PostState(S,V))
\]

`PostState` must simultaneously apply:

- the retain or consume of the selected source;
- all victim owner/checkpoint outcomes;
- all surviving Device/Host placements;
- active destination references and reservations;
- the release of zero-reference objects.

Therefore in general:

\[
Reclaim(S,A\cup B)\ne Reclaim(S,A)+Reclaim(S,B)
\]

Deleting one alias may release no allocation; when two owners together delete the last batch of references, the same allocation
is released only once. The ResourceManager does not compute or sum per-owner reclaim credit.

Move, Fork, partial-tail COW, and source placement are likewise determined by the complete post-state. That a source has multiple references in the initial state
does not mean it will necessarily be Forked in the end; conversely, after retaining the references required by any checkpoint it also cannot be treated as a Move.

### 3.4 Ordered Stage Peaks

A physical plan is an ordered sequence of stages between stable states:

\[
S_0\rightarrow S_1\rightarrow\cdots\rightarrow S_n
\]

It is executable if and only if:

\[
\forall r,\qquad
\max_{0\le i\le n} Used_r(S_i)\le Capacity_r
\]

The stage order must express the real dependencies:

```text
reserve destination
  -> copy
  -> verify and publish replacement
  -> release old replica
```

State restore, KV H2D/D2H, D2D fork, partial-tail COW, and Host extent replacement all follow this rule.
The final net occupancy cannot hide the transient peak of copy-before-release.

### 3.5 Resource revision

The Program exposes a monotonic `resource_revision`, so that read-only assessments and sealed plans are bound to one stable physical state.
The following changes advance the revision:

- the owner/reference topology changes;
- inactive Device/Host placement changes;
- global free/reserved capacity changes;
- the geometry state that affects later allocator feasibility changes.

When an active request performs mapped/reserved conversion within its own reservation, advances its committed frontier, or updates state contents,
the global available capacity does not change, so the revision is not advanced.

The intermediate stages of an open resource transition do not publish a revision. Once it reaches a stable commit/abort state,
if global physical facts have changed, the revision advances exactly once. The handle's owner/generation validation independently prevents stale capabilities caused by slot reuse.

---

## 4. Continuation and Checkpoint

### 4.1 Complete Recovery Conditions

The current Qwen3.5 model contains both pageable Full Attention KV and recurrent state that cannot be rolled back losslessly from an arbitrary later state.
Therefore a frontier can be reused if and only if the Program can prove:

1. a complete StateImage exists for that frontier;
2. the Main KV satisfies the typed coverage defined by the target;
3. the selected backend's KV and fixed state satisfy the same continuation;
4. the token, position, Vision, and mode identity exactly match the incoming prompt.

With only a token match, KV bytes, or a page match, what is missing is a complete continuation, not a partial cache hit.

### 4.2 Continuation Organization

A private continuation represents a session history that can continue to evolve:

```text
PrivateContinuation
├── exact target/token history
├── Main KV address space
├── optional backend KV address space
└── one or more immutable checkpoints
```

A shared prefix is an immutable reuse source and may be Forked by multiple private branches. Multiple checkpoints within one continuation
may reference different prefixes of the same address space, but each checkpoint has its own complete StateImage
identity.

### 4.3 Checkpoint Kinds

| Kind | Semantics |
|---|---|
| `SessionEndpoint` | The latest continuable state of a completed request |
| `TurnClosure` | The stable state in the current turn before the replaceable assistant suffix |
| `ResponseReplay` | The stable state before the generation opener from which the response can be regenerated |
| `LongAnchor` | An earlier long-context recovery point selected by the retention policy |
| `SharedStablePrefix` | An immutable stable prefix shared by multiple histories |

The frontier of `TurnClosure` and `ResponseReplay` must lie before the assistant suffix that the next request may replace.
This way, when a conversation system appends a new user message or rewrites the generated tail, the stable history can still be reused, and the mutable suffix is not mistaken for part of
the prefix identity.

After a checkpoint is published:

- its identity, frontier, and required coverage do not change;
- the StateImage and the protected KV prefix are no longer modified in place;
- Device/Host placement may be changed by resource transitions.

### 4.4 Valid and Device-ready

Checkpoint \(c\) is valid if and only if:

- the StateImage has at least one complete Device or Host replica;
- every required KV logical page has at least one Device or Host replica with consistent content epoch and sufficient coverage.

\[
\forall p\in RequiredPages(c),\qquad ValidDevice(p,c)\lor ValidHost(p,c)
\]

A valid checkpoint may need an H2D restore. It is Device-ready only when all execution requirements are already on the Device and the active future
reservation can be obtained.

### 4.5 Exact Identity

The Program's exact verification may use:

- token IDs and token types;
- position/MRoPE axes and RoPE delta;
- Vision spans and media digest;
- template/runtime mode;
- the checkpoint frontier.

Session keys, markers, hashes, and the prefix index only narrow the candidate set and do not prove a hit.

`rewrite_execution_frontiers` is also part of the exact identity. It records the exact token frontiers at which replay/root prefill must be segmented,
so that the reconstruction path adopts the same execution decomposition as the original generation path; it cannot be ignored in order to adopt an endpoint.
When NInferno's own accepted output forms a boundary that the history renderer can reconstruct exactly, the ownership chain is fixed as:

```text
Frontend detects the model-output reconstruction boundary
  -> Engine transports relative accepted-prefix metadata
  -> Program atomically commits resident identity and shortlist digest
```

The Frontend publishes metadata only when the canonical boundary ends exactly at a token frontier, and makes preview rollback, stop
truncation, and speculative partial accept obey the same accepted-prefix transaction. The Program converts the relative position to the
resident absolute frontier and commits it together with the token ledger, State/KV, and backend state; it does not scan the output text.
The ResourceManager neither infers nor relaxes this fact. This way, NInferno output that the client has not rewritten can continue to adopt an exact endpoint; if the client
reorders tool JSON, fills in defaults, or rewrites history, that is a different rendered identity and can only match an earlier legal checkpoint.

### 4.6 Request Cache Participation

Before entering the Engine, each request is fixed to one of these semantics:

- `ReadWrite`: allows reading exact candidates, capturing checkpoints, and publishing a continuation at finish;
- `Disabled`: allows only root materialization and releases all active context resources at the end.

A `Disabled` request still receives the normal active completion guarantee, but leaves no private/shared catalog entry,
SessionIndex binding, or Device/Host checkpoint replica. Serve warmup uses this semantics.

---

## 5. Placement Semantics of State and KV

### 5.1 StateImage

A StateImage is the unit of complete migration. Let the maximum active concurrency be \(C\) and the extra Device checkpoint slots be \(H\);
then:

\[
Capacity_{DeviceState}=C+H
\]

The first \(C\) units of capacity form the active guarantee, and all slots are still managed by the same Program pool, not pinned to lanes one-to-one.
Host State is charged by complete image and by independent slot capacity.

The semantics of State transitions are only:

| Operation | Condition and result |
|---|---|
| InPlace | The sole active writer advances on its own destination |
| Move | The private source is consumed, and the original StateImage becomes the active destination |
| Fork | An immutable source, or one that must still be retained, is written to a new private destination |
| Freeze | A committed active StateImage becomes an immutable checkpoint |
| Snapshot/Restore | A complete StateImage is copied between Device and Host |

The Program chooses Move or Fork from the complete post-state. The Fork source stays pinned until the destination's first state-writing commit completes;
this settlement belongs to the active model transaction and may not overlap with a new global resource transition.

The active State binding also records the lifetime ownership of the read source. For Root, Move, Restore, and active-capture Fork,
the initial read is provided by the current sequence; when a Fork is established from a retained source or from a consumed
source that still has a surviving checkpoint reference, the read is retained by the surviving owner/reference graph, and the active primary binding owns only the
write destination. This borrowed read participates in global physical occupancy and is protected by the source pin, but cannot be charged
again through the primary binding, and cannot be released on abort/release by the primary binding alone; if the surviving checkpoint is itself an optional checkpoint exclusively owned by the active
lineage, it counts toward the entitlement only through that checkpoint's ownership, and after that ownership
settles it decides whether the source can be released. After Fork settlement, both read and write switch to the destination, and lifetime ownership returns to
the active sequence.

### 5.2 Typed KV

Each continuation holds a separate typed address space for the Main KV and the selected backend KV. The frontier,
capacity, and placement of each pool are independent.

Resource planning uses only the following semantics:

- required logical pages and committed coverage;
- the available Device/Host replicas of each page;
- immutable references and the sole writer;
- the partial-tail COW requirement;
- the effect of restore, demote, or release on capacity and machine work.

The concrete page geometry, Host packed extents, COW, and block-table publication are defined by
[Paged KV Context Store](paged-kv-cache.md).

After a prefix Fork, the active address can contain both shared and private pages, in this order:

```text
immutable full pages | unique mutable full pages | optional unique mutable tail
```

Active capture only adds a reference to immutable full pages and Freezes unique writable full pages in place; an unaligned tail page is kept for the checkpoint, and only
one page is copied as the new active writer. An aligned frontier incurs no KV copy, and a new writer page is materialized when later growth requires it.
The Program's capture assessment, reservation, and commit use the same snapshot-shape analysis, so the resource credit,
transfer cost, and the page-ownership conditions that can actually be committed are consistent.

State and KV can be placed independently, for example State on Device, part of the Main KV on Device and the rest of the Main KV on Host, and
the backend KV with a different coverage. A checkpoint continues to exist only while every required component retains at least one valid
replica.

---

## 6. Active Completion Guarantee

### 6.1 Admission Reservation

Before a candidate is published as Active, the Program guarantees for it:

```text
required Device checkpoint prefix
+ remaining prompt growth
+ maximum effective output growth
+ target-defined provisional/speculative growth
+ State and partial-tail COW destinations
+ selected backend requirements
```

These resources are bound in the returned `SequenceHandle`. Materialization progress converts capacity only between allocation and reserved-but-unmapped,
and never hands the capacity to another request.

Active truncate or speculative rollback may remove mappings, but the corresponding capacity still belongs to that active reservation.
Only terminal release, or a resource transition that explicitly shrinks the active entitlement, can return capacity to the global pool.

The active entitlement includes only the active owner's destination, private growth reservations, and the
exclusive optional resources already attributed to that lineage. An immutable StateImage/KV source borrowed during a Fork cannot be charged again
through the primary binding: if it is retained by another surviving owner it exists only in global physical occupancy, and if it is exclusively owned by an optional
checkpoint of the active lineage it is counted once through that checkpoint. Otherwise the same allocation would be charged twice, and a legal
Fork would be misjudged as exceeding the active guarantee.

### 6.2 Terminal and Capture

A TerminalPending request keeps holding its `SequenceHandle` and complete reservation until the Program completes either:

- **Finish**: publishes one complete immutable checkpoint; or
- **Discard**: releases the entire active continuation.

If Finish cannot form a legal retained terminal state, Discard must be adopted. Optional capture may change
inactive owners through the same pressure graph, but a complete terminal state is committed only if it is strictly better than the private-only baseline; otherwise shared publication is skipped,
the active request is not blocked, and the private capture that was originally executable at the same frontier is not changed.

### 6.3 Persistent Backfill Proof

Whether the Scheduler allows backfill is decided by [Engine Architecture](engine-architecture.md#52-admission-order). The physical proof
provided by the Program uses the same feasibility oracle:

```text
hypothetical state =
    current stable state
    - protected donor active reservations after their terminal release
    + borrower full active reservation
    + head root materialization
    with all unprotected inactive cache releasable
```

Only if this complete state is feasible does the borrower qualify for persistent-safe backfill. The proof is bound to `resource_revision`
and does not use the borrower's or donors' predicted completion times.

---

## 7. Materialization Planning Problem

### 7.1 Candidate

The ResourceManager builds a bounded candidate set for the selected request:

- root;
- matching private endpoint;
- typed rewrite checkpoint;
- retained long anchor;
- shared stable prefix.

When the same capability is hit through different indexes, it is deduplicated first. The Program performs exact verification on each shortlist entry
and produces an `AdmissionCandidate`. A candidate fixes the source, destination, prompt, and request work, but does not decide in advance
Move/Fork, victims, or final placements.

Root always exists. It does not reuse continuation state and represents normal prefill from the start of the prompt.

### 7.2 Shared Publication Candidate

The three states of a shared prefix must not be conflated:

- a **boundary** is a logical content boundary indicated by the protocol or the Engine;
- a **candidate** is an optional write opportunity that the Frontend has resolved to an exact token frontier;
- an **owner** is an immutable checkpoint that the Program has published, with complete State/KV identity and placement.

An external protocol or C++ `PromptInput` submits at most four explicit markers per request. The Frontend can also generate at most three Engine
candidates: after all tools, after the consecutive leading System/Developer messages, and the full prompt. Identical frontiers are merged,
so each request has at most seven prepared candidates. This fixed limit is not a startup configuration.

The shared catalog is an Engine-wide public capacity, not a per-lineage quota. When the context cache is enabled, the default logical
capacity covers both the active-concurrency lower bound and the at most seven prepared candidates of a single request, that is,
`max(max_concurrency, kMaximumPreparedPromptCacheCandidatesPerRequest)`; an explicit configuration fully overrides the default. This lower bound allows an earlier
stable layer and a later rolling marker to both become owners, but whether to capture, retain, or replace is still decided only by the generic portfolio/pressure
planning, with no special case for Claude, compact, or token position.

A candidate stores evidence flags. The policy meaning is:

| Evidence | Admission capability on first occurrence |
|---|---|
| `ExplicitBoundary` | May participate in pressure, but still requires a strictly positive net gain |
| `RequestedAutomatic` | May participate in pressure, but still requires a strictly positive net gain |
| `DefaultAutomatic` | May use only the spare-capacity terminal state that does not degrade existing owners |
| `EngineStructural` | May use only the spare-capacity terminal state that does not degrade existing owners |
| `EngineObserved` | May be created only after at least two independent reuse domains have observed the same key |

Markers, evidence, and shortlist keys do not prove a hit; the Program re-verifies the complete identity for read, dedup, and publication. The order of sources for candidate creation is: an exact shared owner is directly deduplicated; the selected exact private reuse base is
promoted with zero prefill before the first model mutation; a later frontier is captured by active prefill. A candidate that precedes the selected
reuse base and has no ready exact source is kept only as a demand observation, and no state is rolled back and no prefill is redone.

A shared candidate must decide before prefill whether to add a split, while the physical state can only be determined when the frontier arrives. The implementation is therefore
split into two steps:

1. After admission has selected the materialization source, the ResourceManager chooses the candidate subset using the optimistic gain upper bound of the complete logical portfolio and
   the exact split cost returned by the Program; this step reserves no resources and promises no publication;
2. After the capture offer arrives, the `CapturePublicationPlanner` (`capture_planner.h`) searches complete physical targets
   from the current `resource_revision`, and may jointly demote/evict multiple legal inactive private/shared owners. When the shared catalog
   is full, logical replacement and the other physical victims belong to the same target and the same transaction.

The baseline of the actual planner is the private-only policy at the same frontier. Shared publication is an optional investment: when no feasible
target with strictly positive gain exists, the private baseline or Skip is executed.

Private checkpoints (TurnClosure, ResponseReplay, long anchor) use the same planner and the same rule:
if private-only capture is feasible in spare capacity it is published directly; if not feasible (the Device/Host State slots are full of inactive owners),
it enters the planner as a scenario, where the candidate is the new checkpoint of the active continuation it joins, the weight is that
continuation's retention weight, and the victims are all legal inactive owners (the request's own lineage is the evidence for a private
checkpoint and does not need the pressure evidence that shared requires). Only with a strictly positive gain does it evict/demote; otherwise Skip.
Without this path, once the catalog fills all State slots, every new request would skip its own checkpoint, while its endpoint would still evict an old owner
at finish: the old owner would stay forever and the new checkpoint would never be saved (repeated requests and the next conversation turn
would all re-prefill from root for a recurrent-state model).

### 7.3 Target

The planner's search node is a complete target, not an isolated eviction action. A target determines at once:

```text
selected candidate
selected source retained or consumed-to-active
every eligible owner retained or evicted
every surviving checkpoint and required coverage
every surviving State/KV Device/Host placement
logical publication destination
```

Eligible victims include only the inactive owners not protected by an active reference, selected-source protection, or an open transaction pin.
Zero-reference objects produced after the target is applied are released automatically.

The Program re-derives from this complete terminal state:

- Move or Fork;
- State/KV replacement and partial-tail COW;
- last-reference releases;
- the active completion reservation;
- Host allocator geometry;
- ordered stage peaks;
- immediate work and future recovery work.

A candidate's `identity target` applies only the incoming activation and the necessary source disposition, leaving the checkpoint contents and placements of the other
eligible inactive owners unchanged. It is the root of that candidate's pressure degradation graph.

### 7.4 Monotonic Degradation Graph

A pressure target expands from the candidate's identity target along monotonic degradation directions:

- remove a redundant Device or Host replica;
- demote Device State/KV to Host-only;
- shorten checkpoint coverage that is no longer retained;
- delete a checkpoint;
- evict a complete inactive owner.

The Program generates semantic successors from the current residual deficit, checkpoint frontiers, page/extent boundaries, and shared references. The search does not enumerate permutations of physical operations; each target is mapped by the Program to a unique canonical stage order.

A single owner action temporarily yields no reclaim and cannot be deleted before the joint post-state, because it may form a last-reference release
together with another action.

---

## 8. Planning Objective and Algorithm

In this section, `incumbent` means the best feasible complete target found so far.

### 8.1 Separation of Feasibility and Value

The Program first determines whether a target is structurally legal, physically feasible, and still degradable. The ResourceManager compares value only among targets that are
physically feasible and whose logical publication can be adopted.

The cost model does not participate in capacity decisions. Any prediction error can only change "which feasible plan is chosen" and can never let an infeasible plan pass.

### 8.2 Machine Cost

The Program folds the physical work into:

```text
Immediate =
    ordered transfer phases
  + remaining Text/Vision prefill
  + required State/KV copy work
```

A transfer batch of the same phase and the same direction uses:

\[
Transfer =
\max(batch\_ns+operations\cdot operation\_ns,\ bytes\cdot ns\_per\_byte)
\]

The prefill work of the Text suffix uses:

\[
AttentionPairs = B\,S+\frac{S(S+1)}{2}
\]

where \(B\) is the number of reused prefix tokens and \(S\) is the number of remaining suffix tokens. Vision item/patch work uses independent components of the same
startup-resolved machine model.

Transfer coefficients are chosen by hardware; prefill coefficients are chosen by the `prefill_signature` generated from the hardware and the actual Text/Vision config, binding structure, and Uses. The signature does not use checkpoint/release names and does not contain weight values. The two sets of coefficients are resolved
independently; when there is no matching calibration, the corresponding generic coefficients are used. The cost only affects ordering among feasible plans and is not a load or execution gate.

### 8.3 Portfolio Value and Future Loss

The ResourceManager stores the most recent 32 requests that were successfully materialized as Active. Each demand record stores the reuse domain
and the prefix keys that request exact-matched, selected, or could create; it does not store the prompt, media content, raw session text, or API
cache keys. A request with an Engine session key uses its stable digest as the domain; each request without a session key forms an independent
domain.

For a checkpoint \(p\) and the complete portfolio state \(S\):

\[
Saving(p,S)=\max(0,\ Rebuild(p)-Recovery(p,S))
\]

`Rebuild` is the canonical prefill cost from root to that frontier. The Program enumerates all supported restore/copy/interval-prefill recovery recipes from the actual State/KV placement;
the common runner prices them with the same immutable machine cost model of this planning problem and takes the minimum as `Recovery`. The Program does not receive the cost model and does not pick
a recipe for the policy layer. Each demand takes only the maximum saving over all matching checkpoints:

\[
EmpiricalValue(S)=\sum_q\max_{p\ matches\ q}Saving(p,S)
\]

Thus nested prefixes such as tools, instructions, and the full prompt are not priced repeatedly for the same future request. Adding the not-yet-expired
explicit shared credit gives the public value:

\[
PublicValue(S)=EmpiricalValue(S)+SharedCredit(S)
\]

Private retention means each owner has one more unobserved future reuse. When comparing the baseline \(S_b\) with the target \(S_t\),
keeping checkpoint identity unchanged, first compute each checkpoint's saving loss, then take the maximum within the owner:

\[
PrivateLoss_o(S_b,S_t)=w_o\max_{p\in o}
\max(0,Saving(p,S_b)-Saving(p,S_t))
\]

`PrivateLoss` of different owners is summed. This way, nested checkpoints of the same continuation are not all treated as future requests,
nor does a still-usable end checkpoint mask the loss of an earlier TurnClosure or long anchor. The weights are:

| Private retention | prior weight |
|---|---:|
| Disposable | 1 |
| RecentPrivate | 4 |
| LiveSession | 16 |

A shared owner has no fixed retention multiplier. `ExplicitBoundary` or `RequestedAutomatic` brings an owner-scoped credit at publication time;
it is consumed on the first later exact match, or expires after 32 successful materializations.
Multiple pieces of evidence at the same frontier do not stack credits.

For the candidate identity state \(I_c\) of a mandatory materialization and a pressure target \(T\):

\[
FutureLoss_c(T)=\max(0,PublicValue(I_c)-PublicValue(T))
+\sum_o PrivateLoss_o(I_c,T),\qquad
J(c,T)=Now(c,T)+FutureLoss_c(T)
\]

For an optional capture, compare the baseline state \(S_b\) (private-only publication for shared capture, or Skip when the baseline itself
is infeasible; Skip for private capture) with the complete target \(S_t\):

\[
NetGain=PublicValue(S_t)-PublicValue(S_b)
+\sum_o PrivateGain_o(S_b,S_t)-\sum_o PrivateLoss_o(S_b,S_t)-ImmediateDelta
\]

\(PrivateGain_o\) is the same expression in the opposite direction:
\(w_o\max_{p\in o}\max(0,Saving(p,S_t)-Saving(p,S_b))\). Pressure can only lower the saving of existing owners,
so the gain comes only from the published candidate checkpoint: its owner weight is the retention weight of the continuation it joins;
when the baseline has already published that private checkpoint, the weight is 0 (not counted twice).

Before execution, the candidate subset additionally subtracts the exact `SplitCost`; after the capture offer arrives the split has already happened and is not charged again.
`ImmediateDelta` includes the transfer/copy cost of the shared target relative to the private baseline, multiplied by the number of the current capture owner
and the other requests blocked by this global transition. Publication happens only if `NetGain>0`; if equality, overflow, or saturation prevents proving a strictly
positive gain, the baseline is kept.

A legal `ConsumedToActive` of the selected source is an ownership transfer and is not counted as eviction. Other checkpoints that are deleted
or degraded along with the pressure enter the future loss through the public value and the private transition loss.

### 8.4 Ordering Hints Are Not Proofs

For a target that has not been assessed, the Program returns the physical residual, the estimated remaining steps, the transfer/recovery work, and a
stable ordinal. The ResourceManager merges these facts with the portfolio policy and the logical publication conditions into a priority.
The construction target contains both the physical capacity and the logical adoptability; a zero physical deficit cannot substitute for the publication-slot check.

These values only decide the exploration order and whether the optional quota boundary is worth continuing to search: they cannot mark a target feasible, cannot prove that the
incumbent is optimal, and cannot participate in physical readiness. A combined action may cancel a copy, so the predicted cost is not a mathematical lower bound usable for pruning. Only a complete exact assessment together with the logical-adoption check can produce an adoptable plan.

### 8.5 No-pressure Path

The planner must first read all exact candidate identities. Only when the Program states explicitly that none of the candidates needs or is able to
improve the current materialization work through pressure may the best adoptable identity be sealed directly. As long as any candidate
is still expandable, the same bounded pressure domain is built; it cannot be skipped because the estimated cost is higher than the incumbent.

The final shared-capture split is chosen by the ResourceManager after the selected candidate has been determined. The Program validates and
consumes the `FinalScheduleIntent` inside the seal; the returned `ResourcePlan` no longer has a post-seal mutation API, and the summary the Scheduler sees
is already final.

### 8.6 Correctness Incumbent

If no identity target satisfies both the physical and the logical publication conditions, the planner assesses, outside the optional budget:

```text
root candidate
+ release all unprotected inactive cache
```

The root maximal target is built from the candidate-specific eligible domain; the selected source has no victim cell. It retains no
sacrificable cache value, but guarantees that a heuristic, target budget, or planning budget failure does not misjudge a runnable request as blocked.
Active requirements, source required coverage, the writer, and transaction pins always stay in the Program's requirement
union.

If the isolated root exceeds the static contract, the result is permanently infeasible; if the isolated root is feasible but the active guarantee,
a lane, or an open transaction blocks it, the result is temporarily blocked.

### 8.7 Bounded Heuristic Search

Materialization and shared capture use two typed entrypoints. The incumbent of materialization is the verified
identity or the root maximal; the incumbent of shared capture is Skip, replaced only by an exact `NetGain>0`.

Within one planning problem:

1. The ResourceManager mints the candidate/owner IDs and keeps the unique mapping to catalog capabilities; the Program builds, for each
   candidate, an eligible victim domain structurally lacking the selected source.
2. The Program holds the opaque targets and the sharded construction state; the Runtime orders the legal options by the predicted complete J and the deficit improvement,
   balancing value against forming a feasible plan quickly, without constructing State/KV actions.
3. Complete candidates are evaluated first, and after a failure repair continues according to the exact physical and logical feedback. Conditions such as Host geometry that cannot be proven from total bytes
   require earlier exact feedback. Ordinary alternatives are kept outside the local construction.
4. The complete joint post-state settles unique objects, aliases, Move/Fork, COW, Host geometry, and stage peaks.
   Only results that are physically feasible and logically adoptable participate in the final J and the stable tie-break comparison.
5. When any optional planning, work, or capacity budget is exhausted, the best verified incumbent is returned.

Construction, repair, ordinary expansion, the maximal target, and seal share the candidate-specific domain and the exact evaluator.
Cursors, the target arena, and borrowed summaries are valid only within the same planning session/revision; stopping or discarding a batch cannot produce a
real resource mutation. After the search ends, old state is not resumed across decode/prefill.

The budgets are counted work, never time (`materialization_budget.h`,
[Deterministic decisions](deterministic-decisions.md)): the same catalog, prompt and boundary select the same cached states on
every machine, and a CPU test replays any decision. Every Program operation is charged work units from discrete facts of the
operation itself. One unit is the nominal host cost of one construction-option step (about 1 µs):

| Operation | Units |
|---|---|
| construction option | 1 |
| exact target assessment (identity, maximal, constructed, pending) | 200 + its projection work (one per owner decision or checkpoint-recovery alternative it projected) |
| target expansion | 200 + the targets it committed |

The planner predicts the next operation's cost from the units the previous ones were charged. Nothing bounds the search's
wall time but these units, so an assessment and an expansion are charged at the worst host cost expected of one (about
200 µs, against a typical 20 µs): at 200 µs an assessment, a busy boundary still ends within about 10 ms and an idle one
within 50 ms. Every unit cost and boundary size is a declared estimate awaiting calibration on lab hosts against the
request log's `search_elapsed_ns` against `search_units` and `planning_elapsed_ns` against `planning_units`; a wrong one changes
only how much optional planning a boundary does.

The Engine provides an optional quota shared across the admission boundary: one ledger that the head and every backfill
inspection charge, so none of them resets it. It is 50 000 units when no other request is runnable and 10 000 when others
wait (the former 50 ms and 10 ms at the nominal unit, and at most 250 and 50 assessments). The mandatory identity assessments are charged to it too. The planner
first searches with a window of at most 5 000 units, then decides whether to extend it from the expected improvement still
obtainable and the completion cost: a search may spend a twentieth of the predicted saving, at 1 000 predicted device
nanoseconds a unit, shared by the affected requests. An incomplete prediction gets only one exploration episode of at most
5 000 units, and realized gains cannot be used repeatedly to extend the quota. Targets, work, and session storage are still
independently bounded. A cancelled request keeps no optional headroom; its queue deadline is checked before inspection and
does not shorten the search.

The identity and root maximal correctness assessments are not cancelled when the optional quota is exhausted. The quota is
checked between Program operations against each operation's predicted cost; the final seal must still complete. The budget
therefore affects only cache quality, does not change mandatory readiness, and does not promise that the whole planning call
can be interrupted at an arbitrary moment.

Shared capture keeps an independent fixed target budget and a Skip incumbent. Its search may assess, once a target is
selectable, a twentieth of the capture's predicted rebuild saving shared by the waiting requests at a nominal 10 µs a target,
between 20 and 500 targets (four times that while no target is selectable). Uncapped it walked up to 4096 targets in 39-43 ms
(Qwen3.6-35B, concurrency 8, RTX 5090), about 10 µs a target, so the bounds equal the former 0.2-5 ms wall grant.

### 8.8 Objective Function and Determinism

For an exact feasible assessment:

```text
J = price(Program machine work)
  + ResourceManager portfolio loss
```

The Program forms only the canonical transfer/prefill work and enumerates the supported physical recipes for checkpoint recovery;
the ResourceManager provides demand, retention, and shared credit; the runner prices and compares uniformly using the immutable
machine cost model of the same planning problem. The cost model does not enter the Program API; it can only change the exploration and selection
order of verified targets, and cannot change physical status, readiness, source protection, or target generation.

Under equal `J`, the preferences in order are:

1. fewer impacts on checkpoints with observed hits, keeping the most recent hits;
2. fewer owner evictions, checkpoint deletions, copy operations, and transferred bytes;
3. less remaining Text/Vision prefill;
4. more reused prompt tokens;
5. the current session binding;
6. the stable ordinal of candidate and target.

The stop reason describes only the actual boundary: `no_pressure`, `queue_exhausted`, `target_budget`,
`expansion_capacity`, `planning_budget`, `work_budget`, or `insufficient_expected_gain`.
`planning_budget` also covers the case where the next indivisible operation is predicted not to fit in the remaining margin. None of them claims global optimality of the current target graph or of the real
TTFT.

### 8.9 Readiness

There are only four final outcomes:

| Outcome | Meaning |
|---|---|
| `PermanentlyInfeasible` | The isolated root completion requirement exceeds the current Engine contract |
| `TemporarilyBlocked` | A single request is feasible, but the current active/lane/transaction state blocks it |
| `NeedsTransfer` | The target is feasible and needs an asynchronous transfer or pressure stages |
| `Ready` | The target is already Device-ready and can be published synchronously at the current boundary |

`TemporarilyBlocked` is rechecked only after a lane, FIFO protection, open transaction, or resource revision changes.
Ordinary decode does not rerun the planner.

---

## 9. Resource Transition

### 9.1 Three Objects

A terminal state that needs allocation, pressure, or transfer goes through:

```text
ResourcePlan
  -> RunningTransaction
  -> ResourceResult
```

- `ResourcePlan`: an immutable, move-only, revision-bound transition sealed by the Program;
- `RunningTransaction`: the sole global resource transition the Program is currently executing;
- `ResourceResult`: the complete stable result, either Committed or Aborted.

A finish, discard, or inactive release that is already covered by the active reservation and needs no new allocation/transfer can complete as a
zero-stage atomic transition, but still obeys the same serialization, revision, and complete-result rules.

### 9.2 Start

Before the first mutation, the Program re-validates:

1. the plan's `resource_revision`;
2. the capability generations of the source, victims, and destinations;
3. the logical publication capacity already reserved by the ResourceManager;
4. the current allocator's capacity and geometry for each ordered stage;
5. the source/victim dependencies and active protections.

A stale or no longer feasible plan has no physical side effects before start. After a successful start, the source, victims, and destinations are
claimed/pinned, the plan is linearly consumed, and the candidate or target is no longer replaced.

Before calling the Program start, the ResourceManager first establishes the unique logical transaction record and marks all sources/victims and
publication destinations as claimed/reserved. The Program completes its own preflight before the first physical mutation.
When a synchronous start returns Aborted, the ResourceManager rolls back the complete logical claim; after a successful start, the logical record and the Program
transaction coexist until stable result adoption or Engine-wide cleanup.

### 9.3 Progress

A transaction may advance across synchronous copies or CUDA events. Each replacement follows:

```text
destination reserved
  -> copy complete
  -> epoch/coverage verified
  -> replacement published
  -> old replica released when unreferenced
```

Materialization and active capture share one monotonic pressure phase:

```text
HostReleases -> CopyPreparation -> CopiesInFlight? -> CopyPublication -> Committed
```

The phase is the sole authority for the global pressure order. Each owner work item may store execution facts about whether the destination/copy has actually been committed and whether the
mutation has been published, used to form the absolute result; these facts cannot change or bypass the global phase. With no copy, it goes from
`CopyPreparation` directly to `CopyPublication`. Cancellation is adopted only at a stable phase boundary, and a Host
release or victim eviction that has already been published is reported through the absolute `ResourceResult`.

There is at most one global resource transition at a time. Existing active execution may interleave with a transfer only if it uses its own existing reservation
and does not touch the transaction mappings.

### 9.4 Commit

Commit publishes:

- the new active sequence or checkpoint;
- the selected source disposition;
- the committed owner/checkpoint degradation;
- stable State/KV placements;
- transaction pin/reservation cleanup;
- the necessary `resource_revision` change.

The `ResourceResult` contains the absolute result the ResourceManager needs to adopt the logical terminal state, and contains no
delta from which the upper layer could rebuild physical occupancy. Each owner claim is bound to the exact `CheckpointRef` deletion set, not merely the deletion count;
the ResourceManager verifies item by item that the final summary is exactly the original checkpoint set minus that set. A replacement with the same count but different
checkpoints therefore fails before any catalog mutation. The ResourceManager has reserved the adoption storage before start, so adoption after commit
cannot make any new capacity decision.

### 9.5 Abort

Cancellation or an execution failure before publication can yield a request-local abort, as long as the Program can still form a stable physical state.
Abort guarantees:

- the selected source is still valid;
- the target has no half-Active or half-published checkpoint;
- unused destinations, reservations, and pins are released;
- victim demotions/deletions that have safely completed are explicitly recorded in the `ResourceResult`.

A legal pressure degradation completed after a replacement has been published does not need to be copied back to the original placement. The ResourceManager
adopts the actual terminal state in the abort result, rather than replaying intermediate receipts.

If the Program can no longer form an interpretable stable state, it enters Engine-wide failure.

---

## 10. Retention, Session, and Publication

### 10.1 Retention Policy

The private retention class affects portfolio value through the owner prior and provides no fixed eviction order. The value of a shared owner
comes from real demand and the not-yet-expired explicit credit. The planner may combine:

- deleting redundant replicas;
- demoting a Device replica to Host-only;
- deleting checkpoints or suffix coverage that are no longer retained;
- evicting a complete inactive owner.

If a placement change would delete a checkpoint's last complete State or required KV coverage, the same
target must first establish a replacement, or delete that checkpoint at the same time.

Placement changes only at the resource boundaries of admission, capture, finish, or explicit inactive release. Ordinary decode
runs no periodic promotion/demotion.

### 10.2 Session Ordering

The SessionKey belongs to the ResourceManager and provides only candidate lookup and binding. The Program does not read the SessionKey.

Each request that may update the SessionIndex receives a monotonic `publication_order`:

- a new result can replace the current binding only if its order is greater;
- an older request that completes later becomes an anonymous cache or is released per policy;
- a matching private source may consume-to-active when legal;
- a shared source always stays immutable;
- abort can restore only the former binding for which it still holds the exact claim.

The capability generation determines whether a handle is still valid, and the publication order determines which completed result updates the session; the two have different responsibilities.

### 10.3 Logical Publication Capacity

The ResourceManager must reserve the active/private/shared destination slot before the Program mutation. The logical
adoption after the Program commit no longer expands capacity or searches for slots.

A private Move can reuse the source descriptor; root or Fork needs a destination. An optional capture without a slot is skipped;
a terminal Finish without legal publication capacity adopts Discard.

---

## 11. Configuration Semantics and Boundedness

`ContextCacheOptions` is resolved into fixed capacities at Engine construction:

| Configuration axis | Architectural meaning |
|---|---|
| `device_state_slots` | Device checkpoint StateImages that may be retained beyond the active guarantee |
| `host_state_slots` | Complete Host StateImages |
| `host_kv_capacity_bytes` | Total capacity of the typed packed Host KV arena |
| `max_private_continuations` | Private owner/catalog capacity |
| `max_shared_prefixes` | Shared immutable owner/catalog capacity |
| `max_long_anchors_per_continuation` | Limit on retained long checkpoints per private history |

All stores, catalogs, the Program's unique-object scratch, and reusable planner frontiers are built to the resolved limits.
The target arena, hash, and assessment storage of each pressure planning session obtain their
capacity at the session start at fixed limits; target expansion must not exceed that capacity. The capacity required by logical claim manifests and adoption must be obtained before
the Program mutation. The external markers per request are fixed at most four, the Frontend candidates at most three, and the demand window is fixed at
32; these are product semantics and algorithmic boundedness, not tunable capacity axes.
The configuration must satisfy:

- the private continuation capacity covers at least all active requests;
- the total Device State capacity is `max_concurrency + device_state_slots`;
- Host State and Host KV are charged independently;
- logical counts, address-space counts, and storage sizes are representable;
- the Program can deliver the minimum active capacity and the selected backend requirements.

When the context cache is disabled, root-only semantics apply: inactive continuations are neither read nor published, the Device/Host checkpoint
capacity is zero, but the State/KV reservation required by active execution is unchanged.

Public defaults and startup commands are maintained by `EngineOptions`, the [CLI](../cli.md), and [Serving](../serving.md), and are not duplicated in this architecture
document.

---

## 12. Core Invariants

1. The Scheduler selects the request first; the resource layer cannot change request order by cache value.
2. The ResourceManager is the sole authority for logical cache policy, and the Program is the sole authority for physical state.
3. Physical occupancy is counted by unique allocations and concrete reservations, not counted repeatedly by logical references.
4. Reclaim, Move/Fork, and COW must be derived from the complete joint post-state.
5. Every ordered transition stage must satisfy capacity and allocator geometry.
6. A prefix hit must have exact identity, a complete StateImage, and all required typed KV coverage.
7. A published checkpoint is immutable; each active continuation has at most one mutable writer.
8. The active completion reservation cannot be borrowed or reclaimed before the terminal resource result is adopted.
9. There is at most one global resource transition at a time, and a sealed plan is bound to `resource_revision`.
10. A stale rejection before start has no physical side effects; after start the candidate or target cannot be replaced.
11. Program commit/abort both return a complete stable terminal state; the ResourceManager adoption is preallocated and cannot fail.
12. Root plus releasing all unprotected inactive cache is the correctness fallback outside the bounded search.
13. The planner cost model only orders feasible targets and does not participate in physical correctness.
14. A session binding accepts only a greater publication order.
15. Retention or capture failure must degrade to skip/release, so that the active lane has a finite terminal state.
16. Shared publication must be strictly better than the private-only baseline; a single demand can contribute only its maximum
    saving across nested prefixes.
17. Candidate selection is not a resource reservation; the actual capture target must re-prove the complete physical terminal state against the current revision after the frontier
    arrives.

---

## 13. Implementation Locations

| Responsibility | Primary location |
|---|---|
| logical catalog, claims, session policy | `src/runtime/engine/context_cache/resource_manager.h` |
| bounded target ledger and common search primitives | `src/runtime/engine/context_cache/resource_search.h` |
| cross-candidate bounded planner | `src/runtime/engine/context_cache/materialization_planner.h` |
| typed capture-publication bounded entrypoint (shared and private) | `src/runtime/engine/context_cache/capture_planner.h` |
| shared/private portfolio value | `src/runtime/engine/context_cache/context_portfolio_value.h` |
| machine cost model | `src/runtime/engine/context_cache/context_cost.cpp` |
| prefill measurement signature | `src/models/qwen3_5/measurement.cpp` |
| common resource summaries | `src/runtime/contract/resources.h` |
| unique-object projection contract | `src/models/qwen3_5/program/planning/resource_projection.h` |
| target domain, projection, and transaction | `src/models/qwen3_5/program/planning/`, `src/models/qwen3_5/program/transactions/` |
| State stores | `src/models/qwen3_5/program/storage/state_store.h` |
| logical KV/address spaces | `src/models/qwen3_5/program/storage/kv_store.h` |
| Host KV extents | `src/models/qwen3_5/program/storage/host_kv_store.h` |
| public capacity options | `include/ninferno/types.h` |

The paths serve to locate the current implementation and do not change the ownership boundaries defined in this document.
