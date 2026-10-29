"""The card's facts and its telemetry through nvidia-smi (Windows and Linux, any driver): the user
measurement kit's view of the GPU (docs/maintainer/user-measurement-kit.md).

Field names differ between drivers (clocks_event_reasons.* since 535, clocks_throttle_reasons.*
before; some fields are N/A on GeForce or under WDDM): each field is probed once through
`--help-query-gpu`, an unsupported one is left out and an N/A value is kept as None, never as a
pass."""
import os
import platform
import subprocess
import threading
import time

CARD_FIELDS = [
    "name", "pci.bus_id", "driver_version", "vbios_version", "memory.total", "compute_cap",
    "power.limit", "power.default_limit", "power.max_limit", "enforced.power.limit",
    "clocks.max.sm", "clocks.max.mem", "clocks.applications.graphics",
    "clocks.default_applications.graphics", "pcie.link.gen.max", "pcie.link.width.max",
    "driver_model.current", "pstate", "display_active",
]
# The event reasons that reject a measured window (thermal or hardware slowdown, power brake), the
# ones only recorded (a power cap under load is the card's stock behaviour), and an application
# clocks setting (a clock lock): the suffixes are the same under both names.
REASONS = ["active", "sw_power_cap", "hw_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown",
           "sw_thermal_slowdown", "applications_clocks_setting", "gpu_idle", "sync_boost"]
SAMPLE_FIELDS = ["timestamp", "clocks.sm", "clocks.mem", "temperature.gpu", "power.draw",
                 "utilization.gpu", "memory.used", "pstate"]

_known = None


def known_fields():
    """The query fields this driver takes (nvidia-smi --help-query-gpu)."""
    global _known
    if _known is None:
        try:
            text = subprocess.run(["nvidia-smi", "--help-query-gpu"], capture_output=True, text=True,
                                  timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            text = ""
        _known = {line.strip().strip('"') for line in text.splitlines() if line.strip().startswith('"')}
    return _known


def reason_prefix():
    """clocks_event_reasons (535+) or clocks_throttle_reasons (older), or None."""
    fields = known_fields()
    for prefix in ("clocks_event_reasons", "clocks_throttle_reasons"):
        if f"{prefix}.active" in fields:
            return prefix
    return None


def _value(text):
    text = text.strip()
    if not text or text.startswith("[N/A]") or text in ("N/A", "[Not Supported]", "Not Supported"):
        return None
    try:
        return float(text)
    except ValueError:
        return text


def query(fields, device):
    """One row of `fields` (those the driver knows) for GPU `device` (nvidia-smi index)."""
    usable = [f for f in fields if not known_fields() or f in known_fields()]
    if not usable:
        return {}
    out = subprocess.run(["nvidia-smi", f"--id={device}", f"--query-gpu={','.join(usable)}",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return {}
    values = [v for v in out.stdout.strip().splitlines()[0].split(",")]
    return {f: _value(v) for f, v in zip(usable, values)}


def compute_apps(device_bus):
    """The compute processes nvidia-smi lists on the card (empty or None under WDDM)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    apps = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and (not device_bus or parts[1].lower() == str(device_bus).lower()):
            apps.append({"pid": int(parts[0]) if parts[0].isdigit() else None,
                         "used_mib": _value(parts[2])})
    return apps


def card_facts(device):
    facts = query(CARD_FIELDS, device)
    prefix = reason_prefix()
    facts["reason_fields"] = prefix
    if prefix:
        facts["reasons_now"] = query([f"{prefix}.{r}" for r in REASONS], device)
    return facts


def host_facts():
    """The OS, the CPU and the RAM; no user, host name or path."""
    ram = None
    try:
        if os.name == "nt":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("pt", ctypes.c_ulonglong), ("pa", ctypes.c_ulonglong),
                            ("vt", ctypes.c_ulonglong), ("va", ctypes.c_ulonglong),
                            ("ext", ctypes.c_ulonglong)]
            s = Status()
            s.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            ram = int(s.total)
        else:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        pass
    out = {"os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
           "cpu": platform.processor() or None, "cpus": os.cpu_count(), "ram_bytes": ram}
    if os.name == "nt":
        out["on_battery"] = on_battery()
    return out


def on_battery():
    """Windows: True on battery, False on AC, None when unknown (GetSystemPowerStatus)."""
    try:
        import ctypes

        class Power(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                        ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
        p = Power()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p)):
            return None
        return {0: True, 1: False}.get(p.ac)
    except (OSError, AttributeError):
        return None


class Sampler:
    """nvidia-smi samples every `period_ms` in a thread, columnar, with marks for the windows the kit
    measures (idle, warmup, request i)."""

    def __init__(self, device, period_ms=500):
        prefix = reason_prefix()
        self.fields = [f for f in SAMPLE_FIELDS if not known_fields() or f in known_fields()]
        if prefix:
            self.fields += [f"{prefix}.{r}" for r in REASONS]
        self.device, self.period_ms = device, period_ms
        self.columns = {f: [] for f in self.fields}
        self.t = []
        self.marks = []
        self._proc = None
        self._thread = None
        self._t0 = None

    def start(self):
        self._t0 = time.monotonic()
        self._proc = subprocess.Popen(["nvidia-smi", f"--id={self.device}",
                                       f"--query-gpu={','.join(self.fields)}",
                                       "--format=csv,noheader,nounits", f"-lms={self.period_ms}"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            values = line.strip().split(",")
            if len(values) != len(self.fields):
                continue
            self.t.append(round(time.monotonic() - self._t0, 3))
            for f, v in zip(self.fields, values):
                self.columns[f].append(_value(v))

    def mark(self, name, start, end):
        """A window [start, end] (seconds since start()) named `name`."""
        self.marks.append({"name": name, "start": round(start, 3), "end": round(end, 3)})

    def now(self):
        return time.monotonic() - self._t0

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=10)
        cols = {f: v for f, v in self.columns.items() if f != "timestamp"}
        return {"period_ms": self.period_ms, "t": self.t, "columns": cols, "marks": self.marks,
                "reason_fields": reason_prefix()}
