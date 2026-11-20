# GPU facts table

`src/core/data/gpu_facts.json` is the PCI-ID reference table of
[deterministic decisions](deterministic-decisions.md). It records what a board *is*, never what it
should run. The build compiles it into `ninferno_core` (`cmake/EmbedText.cmake`), so a decision
never reads a file at run time. `core/machine_facts.h` parses it and derives `MachineFacts`, which
the query stores in `DeviceClass::facts`.

## Fields

Each row is a published reference configuration, keyed by its PCI product id. A product id is
not always a unique SKU: NVIDIA also uses subsystem vendor/product ids to distinguish boards.
The runtime lookup currently has no subsystem key; `variants` is audit metadata, not an exact
variant selector. See [the shared-identity limitation](https://github.com/Brontir/NInferno/issues/903).

| Field | Meaning |
|---|---|
| `id` | PCI product id, `0x` and four hex digits, as in [pci-ids](https://pci-ids.ucw.cz/read/PC/10de) |
| `name`, `chip` | product name; chip as `gpu_chip` spells it (`gb202`, `ad104`, `gh100`) |
| `cc` | compute capability x 10 (`89`) |
| `segment` | `geforce`, `pro` or `datacenter` |
| `form` | `desktop`, `laptop` or `server` |
| `sms` | multiprocessors of the board |
| `boost_mhz` | the **reference clock** (below); on a laptop the top of the published range, which it is not priced at |
| `boost_min_mhz` | laptops only, and on every laptop: the bottom of the published boost range |
| `l2_mib` | the whole SKU's L2 capacity in MiB (2^20 bytes), including fractional values; absent means unknown, not zero capacity |
| `bus_bits`, `gbps` | memory bus width and data rate per pin; bandwidth = `gbps` x `bus_bits` / 8 |
| `tdp_w` | total board power; on a laptop, the top TGP without Dynamic Boost |
| `vram_gib` | the marketed memory size |
| `pcie` | the highest PCIe generation the link supports, never the current one |
| `mma_slow` | only when true: `mma.sync` without tensor cores (TU116, TU117 and the GTX 16 boards of TU106) |
| `target` | only when false: listed for identification, not a target (H100, H200, B200) |
| `sources` | the datasheet or product page, a database page, and the pci-ids entry of the id |
| `variants` | only where another board is sold, or reports itself, under the id: one object per board, with the values in which it differs from the row (`vram_gib`, `gbps`, `boost_mhz`, `bus_bits`; a bus width must be a whole number of 32-bit channels, the only widths `machine_facts` reads). The facts-vs-driver audit reads it, so a recorded board passes the audit as information; `machine_facts` never reads it and decides from the driver's values alone |
| `subsystems` | only where another board under the id is told apart by its PCI subsystem and differs in a priced value (the TITAN V CEO Edition, subsystem `0x10DE12EB`, under the Quadro GV100's `0x1DBA`): one object per board with `id` (`0x`, subsystem vendor and device, eight hex digits), `name`, `sources`, and the values in which it differs (`boost_mhz`, `gbps`, `bus_bits`, `tdp_w`, `vram_gib`). Not on a laptop row. Unlike `variants`, `machine_facts` reads it (below) |
| `note` | caveats: an id shared by two memory sizes, values from secondary sources |

The reference clock is the published boost clock of NVIDIA's reference board. It is the value the
CUDA driver reports as the peak SM clock of a stock board, so a lab card's cost model sees the same
clock as before the table. The memory clock the facts carry is half the data rate, as the driver
spells it.

A laptop has no reference board: one name spans the OEMs' power limits (an RTX 4090 Laptop GPU
runs 80 to 150 W, 1455 to 2040 MHz). Under principle 7 (the unknown is conservative) its facts take
the **bottom** of the published boost range, `boost_min_mhz`, whatever the driver reports, so a
low-power laptop is never priced like the fastest one of its name; the source is *extrapolated*. A
laptop that runs below its published range (a battery mode, an OEM limit under NVIDIA's floor,
thermal throttling) is still priced at the bottom of the range: the bottom bounds the SKU as
published, not every machine. A laptop row without `boost_min_mhz` takes the lower of `boost_mhz`
and the driver's clock on the 2^(1/16) grid (the row's top without a driver clock). Every laptop
row in the table has it.

## How facts are derived

`machine_facts(DriverReport)` is a pure function:

- **A known id** of the reported architecture takes the row, whatever the driver reports: an
  overclock, a driver's rounding (14001 MHz against 14000), or no clocks at all on MIG or WSL. The
  source is *verified*.
- **A board its PCI subsystem names** (a row's `subsystems`, #903) takes that board's name and
  values, and stays *verified*. A subsystem the row does not list is the row's board, and so is a
  board named without one (a plan's `--plan-device`, a test's class). Where the row lists such
  boards and the driver was asked but could not give the subsystem (neither sysfs nor NVML), which
  board this is stays unknown: it is priced as the slowest of them (the lowest clock, memory clock,
  bus width and size), *extrapolated*.
- **A laptop row, or a device whose SM count differs from the row's**, is *extrapolated*. A laptop
  takes the bottom of its boost range (above). A MIG slice (fewer SMs) also gets `share` = its SMs
  over the board's, and its bandwidth scales by it.
- **A board on another memory than its row's** (one id, two boards: an RTX 4070 with GDDR6 under
  the GDDR6X board's id, an 8 GB RTX 3060 on a 128-bit bus) is *extrapolated* and priced at the
  driver's memory: the bus width it reports when that differs from the row's, and its memory clock
  on the 2^(1/16) grid when that is more than half a grid step (2.2 %) below the row's. A faster
  memory clock (an overclock) is not read, so such a board keeps the row's; a board on a faster
  memory than its row's (a GTX 1650 with GDDR6) is priced as the row. A value no memory has (a bus
  that is not a whole number of 32-bit channels, a clock under 100 MHz) is as good as unreported.
  The engine decides this from the driver's values alone. The facts-vs-driver audit (`tools/audit`)
  checks the engine's facts by the same rule, and separately fails a board whose memory differs
  from its row unless the row records that board in `variants`: record every such board there.
  A test reads `machine_facts.cpp` and fails if the audit's thresholds (978/1000, 100 000 kHz,
  32-bit channels) differ from the engine's.
- **An unknown id** takes the driver's SM and memory clocks rounded to the nearest 2^(k/16), and
  keeps the bus width. The source is *extrapolated*. With no clocks reported it is *none*, there is
  no roofline, and the cost model stands aside.

`cost_route`, `a8_candidates`, `board_of` (the board distance for route tables and measured gates)
and the speculation load model (`GpuKey`) read `DeviceClass::facts`. Nothing reads
`DeviceClass::sm_clock_khz`, `memory_clock_khz` or `memory_bus_bits` for a decision. The decision
fingerprint records the facts (`facts.machine.*`: source, reference and memory clock, bus width,
share, `mma_slow`); the driver's values enter it (`facts.driver.*`) only when they fed a fact
(`MachineFacts::driver_read`: an unknown id, a laptop row without `boost_min_mhz`, a board on
another memory than its row's), and are observed
(the manifest's `observed.driver.*`) otherwise: a verified board, a MIG slice and a laptop priced at
its range's bottom fingerprint alike whatever their driver reports.

## Adding a SKU

### L2 evidence

Use the exact board variant's capacity, not the full chip's capacity or an SM-count ratio.
For example, AD102 has 72 MiB on RTX 4090 and 96 MiB on L40S. TechPowerUp's cache fields
use MB/KB for binary cache units: 1536 KB is 1.5 MiB (RTX 3050 6GB), and 512 KB is 0.5 MiB
(T400 4GB). Keep that precision when converting to bytes. L2 capacity is not L2 bandwidth.

Each capacity added in the 2026-11-13 audit names its exact-SKU source in `sources` and
its provenance in `note`. The L40S and Tesla T4 values are also backed by published
hardware-owner reports in bytes. These are published facts; no local timing or tuning
is needed. Identification-only rows remain `target: false` even after their L2 is sourced.

Four rows remain unknown after that audit and #958:

- `0x1F09`, GTX 1660 SUPER on **TU106** (target): the available
  [1660 SUPER specification](https://www.techpowerup.com/gpu-specs/geforce-gtx-1660-super.c3458)
  describes TU116. Its 1.5 MiB must not be copied to TU106 without exact-SKU evidence.
- `0x2321`, H100 NVL (identification only): the
  [TechPowerUp NVL page](https://www.techpowerup.com/gpu-specs/h100-nvl-94-gb.c4327) gives 60 MB,
  while a [published H100 NVL research hardware table](https://repository.lib.ncsu.edu/server/api/core/bitstreams/a94ab66d-6435-46d5-acc1-28fe54766bfb/content)
  gives 50 MB. Resolve this with an unambiguous NVIDIA SKU specification or a whole-device
  `deviceQuery` report in bytes, not the GH100 full-chip figure or a two-GPU aggregate.
- `0x1F0A`, GTX 1650 on **TU106** (target, #958): the aggregator's 1 MiB is TU117's GTX 1650.
  TU106 carries at most 512 KiB per 32-bit controller (at most 2 MiB at 128 bits; a board may
  disable some), and no SKU page or `deviceQuery` of this board gives its value.
- `0x27B2`, RTX 4000 Ada Generation (target, #958): the aggregator's 48 MiB exceeds what five
  32-bit controllers carry on Ada (8 MiB each, at most 40 MiB at 160 bits).
- Three #905 rows by the same bound (their source pages give the full chip's or a wider board's
  figure): `0x26B2` RTX 5000 Ada (72 MiB at 256 bits; AD102 holds at most 64 there), `0x27B0`
  RTX 4000 SFF Ada (48 MiB at 160 bits; at most 40, as `0x27B2`), `0x1F83` GTX 1630 (1 MiB at 64
  bits; TU117 holds 256 KiB a controller, at most 512 KiB).

`ninferno_machine_facts_test` rejects another target row without L2. Its explicit
exceptions are `0x1F09`, `0x1F0A`, `0x27B2`, `0x26B2`, `0x27B0` and `0x1F83`; remove each when its
capacity is sourced. Unknown values
stay absent, preserving the conservative missing-fact behavior.

### Procedure

1. Take the id from pci-ids and the figures from NVIDIA's datasheet or product page. Use a
   database page (TechPowerUp) only for a figure NVIDIA does not publish, and say so in `note`.
2. Add one row, in the chip's block, with every field and at least the spec page and the pci-ids
   entry in `sources`. Set `mma_slow` only for a board without tensor cores. The row decides it
   for its id (`board_mma_slow`); `chip_mma_slow` (TU116, TU117) answers only for an id the table
   lacks, so every board of those chips carries it here.
3. Run `ninferno_machine_facts_test`. It parses the embedded table and checks every row: sources,
   ranges, the chip against `gpu_chip`, and `mma_slow` against the chip. A new lab card also goes
   into the test's lab list, with the values its driver reports.
4. If the board is a selector golden board, regenerate the goldens
   (`ninferno_selector_golden_test --update`) and review the diff.
5. Regenerate the decision atlas of every build set: `ninferno_decision_properties_test
   --update-atlas` on a build of each set (base: sm_86 or sm_89; full: sm_120a; sm75; sm70) writes
   `tests/data/decision_atlas/<set>.txt`, a line per row (#614 F5). The new row appears there, and
   any other row that moved is a decision the change made.

## Auditing identity coverage

`tools/audit/gpu_catalog.py` compares the registry with saved
[pci.ids](https://pci-ids.ucw.cz/v2.2/pci.ids) and NVIDIA's `README/supportedchips.html` lists
([620.70.06](https://download.nvidia.com/XFree86/Linux-x86_64/620.70.06/README/supportedchips.html),
[580.196.07](https://download.nvidia.com/XFree86/Linux-x86_64/580.196.07/README/supportedchips.html)).
Use current and legacy driver lists: one current driver branch is not an inventory of every
older board. Save the retrieval date and URLs beside the inputs; the report records their SHA-256
hashes. Each driver identity refers to a source id made from the filename and its full hash, so two
different driver versions saved as `supportedchips.html` cannot lose their separate provenance.
It runs offline, without CUDA, a GPU or a driver:

```bash
python3 tools/audit/gpu_catalog.py --pci-ids /path/to/pci.ids \
  --supported-chips /path/to/nvidia-620.70.06.html \
  --supported-chips /path/to/nvidia-580.196.07.html \
  --output /path/to/coverage.json
```

The report retains all observed subsystem identities and names, lists missing product ids by
chip generation, excludes audio/USB/PCI-bridge functions, and keeps unclassified driver identities
visible. Architecture follows a published chip name, not the marketing series: RTX 2050 is
Ampere, whereas RTX 3050 A is Ada. Triage labels distinguish OEM clock-range work, engineering
ids, mining boards, virtual/shared identities and platform review. They are not support claims:
an id in pci.ids or a driver list does not prove a released, physically tested SKU. A row marked
`multiple_driver_names` needs inspection; names alone do not prove different decision facts.
The dated census and remaining qualification work are tracked in
[#904](https://github.com/Brontir/NInferno/issues/904).

Keep three questions separate: **identity listed**, **decision facts complete and applicable to
that variant**, and **hardware qualified**. Registered product-id counts answer only the first.
Do not fill a missing SKU's L2 with the full chip's cache, take a desktop clock for its mobile
version, or infer a laptop's lower boost limit from one OEM's clock. For shared product ids, retain
the subsystem evidence before adding a generic row. Capacity-only variants with identical pricing
facts can use the existing `variants` metadata; variants that change reference clocks, SMs or L2
need identity resolution before their facts can be called verified. A SoC or virtual identity
also needs the relevant platform or partition contract; adding a name cannot establish it.

Check conflicting sources field by field. For example, RTX 5880 Ada uses NVIDIA's published
960 GB/s (20 Gbps on 384 bits), rather than TechPowerUp's 864 GB/s; Quadro RTX 5000's total board
power is 265 W, while its 230 W graphics-power figure describes a different quantity. The row's
sources and note record these distinctions. The catalogue audit discovers gaps; it never writes
or guesses production specifications.
