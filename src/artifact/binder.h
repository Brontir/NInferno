#pragma once

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace ninferno::artifact {

// Device: in the device weight arena. Host: host bytes the loader reads itself. Mapped: pinned
// host memory the device reads in place (zero copy over the bus) - for a table of which a step
// gathers a few rows only, so its bytes need not occupy device memory. Values: owning host values.
enum class Residency { Device, Host, Mapped, Values };

// What finish() does with artifact bindings no parameter() call asked for. A binding's component
// is its name up to the first '/' ("text/layers/0/..." -> "text"). A component the model bound
// nothing of is one it does not load (MTP weights with speculation off, a vision tower of a
// text-only run): its bindings are listed in the plan, not refused. Inside a component the model
// did load, an unbound binding is a weight the artifact carries and the model ignores.
enum class UnusedBindings {
    Allow,                    // list them in the plan only
    RefuseInLoadedComponents, // throw ArtifactError naming them
};

struct ParameterReference {
    std::string name;
    Shape shape;
    Binding binding;
    Residency residency = Residency::Device;
};

struct HostValues {
    QType format           = QType::FP32;
    std::uint64_t elements = 0;
    std::vector<std::byte> data;

    [[nodiscard]] float scalar_f32() const;
    [[nodiscard]] std::vector<std::int32_t> integers() const;
};

// Collects selected logical demands; neither physical object IDs nor whole-artifact profiles
// determine parameter shapes or native operation support.
class Binder {
public:
    explicit Binder(const Reader& reader);

    [[nodiscard]] ParameterReference parameter(std::string_view name, Shape shape,
                                               Residency residency = Residency::Device,
                                               std::optional<QType> exact_format = {});
    [[nodiscard]] ParameterReference binding(std::string name, const Binding& binding, Shape shape,
                                             Residency residency,
                                             std::optional<QType> exact_format = {});
    [[nodiscard]] const Use& use(std::string_view parameter, std::string_view input) const;
    [[nodiscard]] bool contains(std::string_view parameter) const;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

    void require_device(ObjectHandle object, std::uint64_t alignment = 256);
    void require_mapped(ObjectHandle object);
    [[nodiscard]] std::span<const std::byte> host_object(ObjectHandle object);
    [[nodiscard]] ObjectHandle resource(std::string_view component, std::string_view role);
    [[nodiscard]] HostValues values(const Binding& binding, std::optional<QType> format = {});
    [[nodiscard]] MaterializationPlan finish(UnusedBindings unused = UnusedBindings::Allow) &&;

private:
    struct Demand {
        bool device             = false;
        bool host               = false;
        bool mapped             = false;
        std::uint64_t alignment = 256;
        std::vector<std::byte> host_data;
    };

    const Reader& reader_;
    std::vector<Demand> demands_;
    std::set<std::string, std::less<>> bound_; // binding names parameter() was asked for
    std::uint64_t read_bytes_        = 0;
    std::uint64_t owned_value_bytes_ = 0;
};

} // namespace ninferno::artifact
