# NInferno Paged KV Context Store

This document defines the physical storage and consumption contract of NInferno growing KV. It is the maintainer authority for typed KV pools, logical pages,
Device/Host replicas, address spaces, reservations, block tables, and GPU consumer views.

It answers only two questions:

1. Whether a selected logical context target can be fulfilled by the current KV stores;
2. How the fulfilled KV is consumed directly by a model execution unit.

Request ordering and lifecycle are defined by the [Engine architecture](engine-architecture.md); candidates, retention, pressure targets,
and resource costs are defined by [Resource scheduling and context cache](resource-scheduling-and-context-cache.md). The KV Store does not choose
a request, source, or victim.

---

## 1. Physical model

Growing KV uses a set of homogeneous pools fixed at startup. Each pool:

- stores all planes that share the same frontier and lifetime;
- uses a fixed token page size, plane order, and page-group count;
- owns an independent physical page-ID namespace and free capacity;
- provides one logical address space per active sequence;
- is addressed directly by consumers through block tables.

A request's KV need not be physically contiguous, and is not permanently bound to a control lane. All active and inactive address spaces
share pool capacity; an active request obtains, through a reservation, completion capacity that no other request will use.

Paged storage covers KV that grows with context. DFlash local cyclic state, Vision/query temporary K/V, and other fixed
state have different lifetimes and are managed by their own StateImage or workspace contracts.

---

## 2. Three independent granularities

The KV architecture distinguishes:

| Granularity | Meaning | Current contract |
|---|---|---|
| allocation granularity | How much token payload a pool acquires or releases at once | `P=64` for every growing pool |
| valid-frontier granularity | Up to which logical position a consumer may read | 1 token |
| reusable-state granularity | Which frontier has a complete model continuation | target-defined checkpoint |

A page boundary is neither an Attention mask boundary nor a prefix hit boundary. A valid frontier may lie at any offset
inside a page.

The KV Store can represent, truncate, or protect a prefix at any token frontier; this does not prove that the model can resume from that position.
A reusable frontier must have both a complete StateImage and target-defined backend state; for the specific rules see
[Continuation and checkpoint](resource-scheduling-and-context-cache.md#4-continuation-and-checkpoint).

---

## 3. Typed pool set and capacity

### 3.1 Pool set

The model configuration and the selected speculative backend determine the pool set at startup:

```text
ordinary:
    Main Text

MTP:
    Main Text
    MTP

DFlash / DFlash2:
    Main Text
    Draft Full, when the selected draft config contains full-attention layers
```

Speculative backends are mutually exclusive within one Engine, so there are currently at most two growing pools. All five layers of the official DFlash2
configuration are local attention, so only the Main growing pool is needed, and its draft context uses cyclic storage.

| Pool | Content | Logical frontier |
|---|---|---|
| Main Text | Target full-attention K/V and their code/scale planes | target materialized KV frontier |
| MTP | MTP persistent K/V and their code/scale planes | MTP KV frontier |
| Draft Full | Full-context K/V of the selected draft | draft context frontier |

Main Text and MTP use the BF16, INT8-G64, FP8-E4M3FN-row256, NVFP4-G16, or K8V4
KV profile selected by the Engine; Draft Full uses its own BF16 profile. The physical layout under the `BFloat16` name is BF16 K and FP16 V,
and the write side converts BF16 V to FP16 once. K8V4 is a closed asymmetric profile, not a runtime bit-width combination: K is fixed to
FP8-E4M3FN-row256, and V is fixed to NVFP4-G16.

`PagedKVStorageLayout` resolves the selected closed profile into the K/V data/scale plane schema; the target planner expands that schema per
layer and determines plane ordinals. The common pool implementation only receives the already expanded
`KVPageGeometry`, plane inventory, and capacity, and does not interpret the storage mode.

### 3.2 Main capacity

Let:

- \(S\): `max_context` of a single sequence;
- \(C\): `max_concurrency`;
- \(P=64\): Main page size;
- \(L=\lceil S/P\rceil\): logical page capacity of a single address space;
- \(M\): physical page-group count of the Main pool.

The valid range is:

\[
M_{min}=\max(L,C)
\]

\[
M_{max}=C\,L
\]

\(M_{min}\) guarantees both that one request can reach \(S\) and that every active lane can obtain at least one page.
\(M_{max}\) is the physical upper bound when all active requests reach the per-sequence ceiling simultaneously.

The explicit policy of `kv_capacity` resolves to:

\[
M=\left\lceil K_{main}/P\right\rceil
\]

and requires \(K_{main}\ge S\) and \(M\in[M_{min},M_{max}]\).

### 3.3 Automatic capacity

The automatic policy resolves \(M\) once after weights are loaded, from the currently available device memory \(F\) and the headroom \(R\) that must be kept.
The model planner provides an affine `SequenceCapacityCurve` based on the bound parameters and the selected execution domain:

\[
B(M)=B_{min}+(M-M_{min})B_{step}
\]

where \(B(M)\) is the complete runtime Device reservation for that Main capacity, including:

- Main and selected-backend typed pools;
- active/checkpoint State storage;
- block tables and fixed persistent state;
- unified workspace;
- CUDA Graph allowance.

Automatic selects:

\[
M=
\min\left(
M_{max},
M_{min}+
\left\lfloor
\frac{F-R-B_{min}}{B_{step}}
\right\rfloor
\right)
\]

and requires \(F\ge R+B_{min}\). When \(M_{min}=M_{max}\), that single point is taken directly. The target uses the same production layout builder
to produce \(B_{min}\), \(B_{step}\), and the final layout, and validates the capacity curve; the resolver neither duplicates model dimension formulas nor guesses
capacity by allocation probing.

The final public Main KV capacity is \(M\,P\) token-equivalents. Page rounding only adds physical padding and
does not enlarge the single-sequence logical ceiling \(S\).

### 3.4 Backend capacity

The logical page capacity of the selected backend is still \(L\). Physical capacity is derived from the Main \(M\):

```text
backend off:
    Main physical pages = M

MTP with draft window K:
    Main physical pages = M
    MTP physical pages  = M + C * ceil((K - 1) / P)

DFlash / DFlash2 with full-attention layers:
    Main physical pages        = M
    Draft Full physical pages = M
```

The extra MTP pages cover only the provisional lead of each active row relative to Main within one speculative round,
and do not enlarge the logical capacity of any address space. Draft Full has no such provisional lead; a draft configuration without a full layer
does not allocate this pool.

The pools are physically separate. A free page of one pool cannot become the payload of another pool. Program establishes the
complete typed capacity vector once at startup, and does not grow or re-split pool geometry at runtime.

### 3.5 Host capacity

The Host replicas of Main and the selected backend share one startup-fixed pinned `HostKVArena`, but each allocation
carries its own typed page layout. Host capacity is charged by actual packed bytes and allocator extent geometry;
it does not enlarge the Device active entitlement or the single-sequence context ceiling.

---

## 4. Page group and physical layout

### 4.1 Grouping invariant

Planes enter the same pool only if they satisfy all of the following conditions:

1. They use the same logical cache ordinal;
2. They share the same committed frontier;
3. They are reserved, materialized, truncated, retained, and released together;
4. They use the same page size;
5. They have no semantics of independent release or independent capacity reuse.

Therefore the K, V, code, and scale of the Main layers can share one page-group ID; Main, MTP, and DFlash Full
must belong to different pools.

### 4.2 Page group

A pool-local page-group ID \(g\) simultaneously selects that logical block's payload across all grouped planes:

```text
page group g
├── layer/plane 0 slice g
├── layer/plane 1 slice g
├── ...
└── layer/plane n slice g
```

Planes have Engine-lifetime-stable backing, but do not need to form one contiguous blob. A page group is the smallest Device unit of allocation,
reservation, reference, and transfer.

Physical IDs may be arranged arbitrarily. The allocator may prefer to return contiguous IDs to improve locality, but correctness, admission,
and kernel launch topology do not depend on contiguity. Fixed-size page groups produce no variable-size external
fragmentation and need no Device compaction.

### 4.3 Closed Device plane orders

All registered growing pools use \(P=64\) and select one of two closed orders.

Main Text and MTP use page-major:

\[
[X,P,H,N_{physical}]
\]

DFlash Full uses head-major page run:

\[
[X,P,N_{physical},H]
\]

where:

- \(X=D\) denotes a K/V or quantized code plane;
- the INT8-G64 scale plane uses \(X=D/64\);
- the FP8-E4M3FN-row256 scale plane uses \(X=1\);
- the NVFP4-G16 code plane uses packed U8 \(X=D/2\), and the scale plane uses U8 \(X=D/16\);
- \(H\) is the number of KV heads;
- \(N_{physical}\) is the physical page count of that pool.

For logical position \(p\):

\[
b=\lfloor p/P\rfloor,\qquad o=p\bmod P,\qquad g=block\_table[b]
\]

The page-major address is:

\[
address=base+d\,nb_0+o\,nb_1+h\,nb_2+g\,nb_3
\]

The head-major address is:

\[
address=base+d\,nb_0+o\,nb_1+g\,nb_2+h\,nb_3
\]

Code and scale planes use the same \(g\), but use their own Tensor's leading coordinate and strides.
The exact persistent codec is defined by [`kv_cache_append.h`](../../include/ninferno/ops/kv_cache_append.h),
and the consumer arithmetic by [`softmax_attention.h`](../../include/ninferno/ops/softmax_attention.h);
the allocator only interprets plane bytes, order, and page-group identity.

The physical payload per token/head of the D256 Main/MTP profiles is:

| profile | K code + scale | V code + scale | K+V |
|---|---:|---:|---:|
| BF16 | 512 B | 512 B | 1024 B |
| INT8-G64 | 256 B + 8 B | 256 B + 8 B | 528 B |
| FP8-E4M3FN-row256 | 256 B + 2 B | 256 B + 2 B | 516 B |
| NVFP4-G16 | 128 B + 16 B | 128 B + 16 B | 288 B |
| K8V4 | 256 B + 2 B | 128 B + 16 B | 402 B |

The code and scale planes of K/V have their own dtype, leading extent, and group size; they still share page-group
identity, frontier, and lifetime. The capacity curve, Device/Host replicas, continuation transfer, and memory
summary are all computed from this typed plane inventory, and `2 * vector_bytes` must not be used in place of the asymmetric byte count of K8V4.

### 4.4 Logical position domain

The block table uses the autoregressive cache ordinal:

```text
logical block b covers positions [b*P, (b+1)*P)
```

RoPE, MRoPE, Vision axes, and `rope_delta` are Attention input metadata and do not change KV slot ownership.

### 4.5 Page payload

The logical page-group payload of a pool is:

\[
PageBytes=\sum_{plane} PlaneBytesPerToken\cdot P
\]

Startup physical bytes are obtained from the full span and alignment of each plane slab. `PageBytes` may differ between pools,
but all page groups within one pool are equivalent.

`P=64` simultaneously satisfies the current 32/64-key Attention tiles, 128-token aligned prefill chunks, bounded block-table
metadata, and bounded tail slack. Prefix hit granularity does not take part in page-size selection. Changing the page size, grouping,
or closed plane order is an architecture change.

---

## 5. Logical page and replicas

### 5.1 Logical page identity

A Device page ID is only the physical location of the current Device replica, not a prefix identity. Program maintains generation-checked
logical pages for each pool:

```text
LogicalKVPage
├── object generation
├── content epoch
├── committed columns [0, P]
├── protected columns
├── address-space references
├── active references and writer state
├── optional Device page-group lease
├── optional Host extent membership
└── transaction pins
```

The canonical content of a logical page is identified by `content epoch + committed columns`. Speculative or not-yet-
committed bytes do not extend committed coverage.

A replica is valid for the first \(n\) columns required by a checkpoint if and only if:

\[
replica.epoch=page.epoch
\quad\land\quad
replica.coverage\ge n
\]

### 5.2 Device replica

A Device replica uses the consumer-native plane layout of `4`. A `DeviceKVPageLease` exclusively owns one pool-local
physical page group; the generation protects against stale handles after release/reuse.

The pool distinguishes, for the same capacity unit:

```text
allocated page lease
reserved but not materialized page
globally available page
```

\[
allocated+reserved+available=capacity
\]

Materialize converts a reservation into a lease; dematerialize returns a lease to the same reservation. Neither changes the global capacity available to other
requests.

### 5.3 Host replica

A Host replica uses the logical-order packed `HostKVPageLayout`:

- it stores neither Device page IDs nor block-table holes;
- each page contains the complete grouped plane payload of that typed pool;
- variable-size extents are charged only for the actual page count;
- Main/backend layouts may allocate extents of different strides in the same arena.

The Host arena is a bounded variable-size allocator. A plan must validate extent geometry against the complete release/allocation recipe;
`free_bytes` is only an occupancy summary and not sufficient proof of allocatability.

### 5.4 Replica transfer

D2H/H2D transfer copies the complete page payload, and may merge adjacent physical IDs into fewer transfer runs.
The publication order of a replacement replica is:

```text
reserve destination
  -> copy page payload
  -> verify epoch and committed coverage
  -> publish replica
  -> release source when no longer required
```

While the copy is in progress, both source and destination are pinned. Before the copy completes, the destination does not enter an address space or execution
table; the only valid source also cannot be released first.

### 5.5 Descriptor lifetime

A logical descriptor does not constitute a third copy of the payload. It can continue to exist under Device-only, Host-only, or Both placements.
A descriptor can be reclaimed and its generation advanced only when references, replicas, and transaction pins are all zero.

---

## 6. KV address space

### 6.1 Owning membership

Each continuation holds one `KVAddressSpace` in each enabled pool:

```text
KVAddressSpace
├── ordered logical-block -> LogicalKVPage membership
├── committed frontier
├── checkpoint-protected frontier
├── active/inactive state
├── active growth reservation
└── optional execution-row lease
```

The address space owns the logical order; the block table is only the execution mirror of the active Device mapping. An inactive checkpoint
does not occupy an execution row and is not bound to its original control lane.

### 6.2 Three extents

Each address space distinguishes:

| Extent | Meaning |
|---|---|
| entitlement | Maximum Device page count an active request is guaranteed to obtain |
| membership | Logical pages that already belong to this address space |
| committed frontier | Canonical token prefix a consumer may read |

Membership may exceed the committed frontier, for example when a prefill chunk or a speculative window is materialized in advance.
These bytes become canonical state only after the target commits the frontier.

Membership and frontier may differ between pools; the Program model schedule defines the semantic relationship between them,
and the KV Store does not derive it on its own.

### 6.3 Active bundle

An active sequence holds:

```text
SequenceKVBundle
├── Main Text KVAddressSpace
└── selected backend KVAddressSpace, when enabled
```

Activation must atomically acquire, for all enabled pools:

- address-space descriptors;
- missing Device replicas;
- future growth reservations;
- partial-tail COW destinations;
- execution rows.

Failure in any pool must not publish a partial Active bundle. An active sequence cannot disable the selected backend on a per-request basis.

### 6.4 Execution rows

Each pool builds a fixed-address Device block-table matrix at startup:

\[
block\_tables[N_{logical},C]
\]

where \(N_{logical}=L\) and \(C=max\_concurrency\). Each active address space leases one row; each entry is an I32
pool-local physical page ID.

Activation publishes the Device page IDs of the address-space membership in bulk to the leased row. An inactive address space
has no row; the same continuation may obtain a different row at its next activation. An execution row does not own logical pages,
reservations, or frontiers.

---

## 7. Lifecycle

### 7.1 Activation

When activating from a root, a private source, or a shared source:

```text
create or claim destination address spaces
  -> reserve every typed pool
  -> restore required Host-only pages
  -> Move or Fork memberships
  -> create private partial tails when required
  -> lease execution rows
  -> publish complete table mappings
  -> publish Active sequence
```

ResourceManager selects the logical target, and Program decides Move/Fork/COW based on real references. Before Active publication,
the source remains valid, and any failure returns to a complete inactive terminal state.

### 7.2 Prefill and decode

Before a GPU unit launch, the target schedule gives, for each pool, the maximum logical position that may be written this time.
The KV Store materializes all required pages from that active address space's reservation and publishes the table slice.

The argument of `ensure_mapped_to_tokens()` is a lower bound of the coverage required by this phase. When existing membership is sufficient it returns directly,
including when membership exceeds the required range; it fails only when the required page count exceeds the entitlement. This operation does not advance the committed
frontier, does not trim existing mappings, and does not change the entitlement. New pages only convert the current address's reservation into
allocations; shortening the range can only be done by an explicit truncate.

Each phase guarantees only the pools it will write: target prefill/verify is responsible for Main KV; draft context append guarantees only
DFlash Full backend KV. All draft context of DFlash2 is written into fixed cyclic state and needs no paged KV materialization.

Only after the unit succeeds does Program advance the corresponding committed frontiers. If some layers have been written but the unit did not form a legal
commit, the new frontier is not visible.

Ordinary decode usually materializes one new Main page only when crossing a page boundary. When the tail page still has room, no
allocator work is needed.

### 7.3 Truncate and rollback

Each pool computes the membership to retain from its own canonical/provisional frontier:

\[
NeededPages_s=
\left\lceil
\frac{\max(CommittedFrontier_s,ProvisionalFrontier_s)}{P}
\right\rceil
\]

Trailing mappings may be released; the last partial page is kept. For an active address space, released Device leases return to
the same active reservation, not to global available capacity. Stale bytes in a page after the frontier do not enter the
consumer valid domain.

Speculative termination first completes the recurrent state, hidden, and draft context according to the final committed count, waits for GPU work to complete,
then publishes the committed frontier, and then trims the uncommitted tail pages. The verify mappings must not be trimmed early to satisfy
a shorter coverage requirement of a later phase. Terminal settlement finally releases the active reservation and retains a checkpoint according to the retention decision.

### 7.4 Deactivation, retain, and release

After an active sequence finishes, its address space first leaves execution:

- the execution row is released;
- the active writer/reference is dropped;
- the not-yet-materialized growth entitlement is released;
- according to the terminal target, it is kept as an immutable inactive address space, or all memberships are released.

Retain can be published only when Main, the backend, and StateImage together form the same complete continuation. Release drops
logical references one by one, and frees physical replicas only when the last reference disappears.

### 7.5 Stable execution unit

From block-table publication to GPU unit completion:

- the table entries of the selected rows do not change;
- page payloads are neither relocated nor evicted;
- logical memberships and readable frontiers are frozen for that unit;
- the execution row is not recycled;
- the allocator and transfers do not enter kernels.

Mapping, frontier commit, truncate, and row recycling all happen at the GPU boundary.

---

## 8. Sharing, Move, and COW

### 8.1 Private continuation

A private source can Move when its final post-state has no surviving immutable reference:

```text
inactive address space -> active address space
```

Move neither copies the KV payload nor changes logical page identity. The sole writer of a private tail may continue to append to the unprotected
suffix, as long as it does not overwrite the committed prefix of any surviving checkpoint.

### 8.2 Immutable Fork

A shared source, or a private source that must still be retained, uses Fork:

- full pages before the frontier gain an immutable address-space reference;
- the destination obtains its own State writer and growth reservation;
- shared pages have no writer;
- the source remains valid until the whole activation commits or aborts.

### 8.3 Non-page-aligned frontier

Let \(P=64\) and the checkpoint frontier \(F=1000\):

```text
full immutable pages = 15
partial tail columns = 40
```

A private Move can continue in place from position 1000. An immutable Fork shares the first 15 full pages and copies the 40 committed
columns of the tail into a private destination page; subsequent appends write only the private tail.

The checkpoint frontier is not rounded down to 960. Page-size allocation and token-level validity remain independent.

### 8.4 Writer invariant

The reference count expresses sharing and does not by itself decide write permission. Program verifies all of:

- surviving checkpoint protected coverage;
- writer cardinality;
- active references;
- content epoch;
- source/destination pins.

Any logical page has at most one writer at a time. A full page with multiple address-space references is immutable;
writing to a shared partial tail requires COW first.

---

## 9. Speculative and non-growing KV

### 9.1 Independent pool frontiers

MTP and the DFlash backend with a full layer use the backend pool of the Program KV Store, and do not establish an independent allocator.

Within one speculative unit, Main and the backend:

- each materialize from their own active reservation;
- may have different mapped/provisional frontiers;
- each commit their accepted frontier;
- each trim their rejected trailing mappings.

During MTP drafting the backend mapped extent may temporarily lead Main; DFlash Full usually lags behind Main. The provisional lead
does not constitute committed checkpoint coverage. Rejected bytes may remain in a partial page, but must be overwritten by a new
canonical write before any subsequent read.

### 9.2 Fixed and transient K/V

The following storage does not enter growing pools:

| Resource | Owner |
|---|---|
| DFlash/DFlash2 local sliding-window K/V | fixed per-sequence StateImage |
| DFlash/DFlash2 boundary-local snapshot | fixed checkpoint StateImage |
| Vision/query temporary K/V | Program workspace |

The cyclic K/V of DFlash/DFlash2 uses its own `CyclicKVCacheLayerView` and modulo/window semantics; it holds no page ID,
block table, or growing reservation. The model configuration determines its layer count, heads, and window.

---

## 10. Consumer contract

### 10.1 Single-sequence view

Single-sequence growing-cache Ops use the non-owning `PagedKVLayerView`:

```text
PagedKVLayerView
├── k_pages / v_pages
├── optional k_scale_pages / v_scale_pages
├── block_table          I32 [Nlogical]
├── head_dim
├── num_kv_heads
└── storage
```

The view contains only one layer's plane tensors and one row of the block table. It contains no allocator handle, request identity,
reservation, ownership, or frontier.

### 10.2 Batched view

Batched Ops use `PagedKVBatchLayerView`:

```text
PagedKVBatchLayerView
├── shared layer planes
├── block_tables         I32 [Nlogical, C]
├── head_dim / num_kv_heads
├── storage
└── table_rows[B]        separate Op input
```

`table_rows[b]` selects, for compact row \(b\), the corresponding active address-space row. The compact batch order, Engine lane,
and table row may all differ from one another. Per-row context length, valid columns, and positions are provided by other typed inputs of the Op.

### 10.3 Address translation

With the current \(P=64\):

```text
logical_block = position >> 6
page_offset   = position & 63
physical_page = block_table[logical_block]
```

The element address is then computed using the closed plane order of `4.3`. A batched consumer first selects the table row with `table_rows[b]`,
then performs the same translation.

The wrapper validates Tensor dtype, geometry, closed strides, table shape, and the execution envelope. The caller guarantees that:

- the logical blocks that may be accessed this time are materialized;
- the read domain does not exceed that sequence's frozen valid frontier;
- the writable K/V code and scale are all completed before frontier publication;
- the selected table rows are stable within the GPU unit.

### 10.4 Direct paged execution

Growing-cache Ops consume paged views directly. Page translation is computed and reused at page/tile granularity, and the inner loop does not interpret
request, pool kind, or allocator state. Kernel correctness must not depend on adjacent logical pages mapping to adjacent physical IDs.

Paging introduces no gather-to-contiguous cache or staging copy proportional to context length. Route-specific tile,
split, warp, and shared-memory schemes can be optimized independently, as long as they preserve the same logical Attention, persistent codec, and the address
contract above. The numerical and performance admission rules of an Op are in [Op development](op-development.md).

---

## 11. CUDA Graph and table publication

The plane bases and the block-table matrix base are stable for the Engine lifetime. What changes across replays is:

- table content;
- `table_rows` selectors;
- positions, context lengths, and valid counts;
- model state selectors.

Page IDs, request identity, and physical contiguity do not enter the graph key.

Before an execution unit that needs new mappings, Program does:

```text
materialize all required pages
  -> update host-side membership
  -> publish one contiguous table slice
  -> launch/replay consumer on the ordered stream
```

Bulk publication is used when multiple pages are added at the same boundary or a full membership is reactivated. The mapping update must precede the
consumer, and the same row must not be rewritten while a replay is in flight.

---

## 12. Core invariants

1. Each Device page-group lease carries at most one logical page replica in its pool.
2. A pool combines only planes that share frontier, lifetime, page size, and allocation semantics.
3. The K/V/code/scale planes of a pool use the same page-group ID for the same logical block.
4. Device occupancy is counted by allocated leases and reservations; logical aliases are not charged twice.
5. Logical page identity, content epoch, and physical page ID are independent of one another.
6. The valid frontier is exact to the token; page boundaries do not change Attention or checkpoint semantics.
7. The coverage required by a published checkpoint cannot be overwritten by a writer; any logical page has at most one writer.
8. Shared full pages are immutable; a non-aligned writable tail first establishes a private COW page.
9. Host/Device replacement is published only after the copy and epoch/coverage verification complete.
10. Active entitlement does not enter global available capacity before terminal release.
11. Within one GPU execution unit, membership, block tables, replicas, and the read frontier are stable.
12. An inactive address space does not occupy an execution row; an execution row does not own logical pages.
13. Main and backend pools reserve, materialize, commit, and truncate separately.
14. A growing-cache consumer accesses KV only through the paged view and block table, and obtains no allocator or ownership authority.
15. Kernel correctness does not depend on physical page ID contiguity, nor does it build request-contiguous KV through gather.
16. Checkpoint reusability is proven by the complete target continuation; the mere existence of KV pages does not constitute a hit.
17. The H2D of block table publication reads the pinned shadow of that execution row only when the stream reaches it. A later
    publication to the same row (including the publication of the next owner after release) first waits for the copy to complete before overwriting
    shadow entries that overlap a copy still in the queue; non-overlapping entries do not wait.

---

## 13. Implementation locations

| Responsibility | Primary location |
|---|---|
| Device page pools, reservations, and execution tables | `src/core/paged_kv_cache.*` |
| closed K/V data/scale plane schema | `src/core/paged_kv_storage.h` |
| Host packed page layout and arena | `src/core/host_kv_arena.*` |
| logical pages, references, and address spaces | `src/models/qwen3_5/program/storage/kv_store.h` |
| Host extent membership | `src/models/qwen3_5/program/storage/host_kv_store.h` |
| Program-level KV transition | `src/models/qwen3_5/program/transactions/` |
| model pool layout and capacity curve | `src/models/qwen3_5/program/planning/startup.cpp` |
| public paged consumer views | `src/core/paged_kv_cache.h` |
| growing-cache Ops | `include/ninferno/ops/`, `src/ops/` |

For the exact model state and backend mathematics see
[Qwen3.5 model](qwen3_5-model.md) and [DFlash](dflash.md); the persistent KV codec and the causal consumer
numerical contract are defined by the growing-cache Ops in the table above. Paths are for locating the current implementation, and do not elevate files or class names themselves to
external interfaces.
