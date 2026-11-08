// The artifact writer's hidden temporaries (#552): what an interrupted conversion leaves behind
// and who removes it. POSIX only (fork, kill); the Windows build shares the code, not this test.
//
// - A temporary of a writer that is gone (a SIGKILL, an OOM kill) is removed by the next
//   conversion into the same directory; a temporary a live writer holds is not, and that writer
//   still publishes its artifact; a file that merely looks like a temporary (not ours) stays.
// - SIGINT and SIGTERM remove the temporary of a writer that has not finished, and the process
//   still dies of the signal.
// - A writer destroyed without finishing (an exception) removes its temporary.

#include "convert/writer.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace cv = ninferno::convert;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAILED: " << what << '\n';
        ++g_failures;
    }
}

cv::ArtifactLayout small_layout(const std::string& name) {
    return cv::layout_artifact(name, R"({"objects":[]})", 8192, 1ULL << 30);
}

// Hidden temporaries ".<anything>.XXXXXX.tmp" in `dir`.
std::vector<fs::path> temporaries(const fs::path& dir) {
    std::vector<fs::path> out;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (name.size() > 4 && name[0] == '.' && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            out.push_back(entry.path());
        }
    }
    return out;
}

// Forks a child that opens a writer on `target` (default dispositions for SIGINT and SIGTERM,
// whatever the parent inherited) and waits. Returns its pid once the temporary exists.
pid_t spawn_writer(const fs::path& target) {
    int ready[2];
    if (pipe(ready) != 0) { std::exit(2); }
    const pid_t pid = fork();
    if (pid == 0) {
        close(ready[0]);
        std::signal(SIGINT, SIG_DFL);
        std::signal(SIGTERM, SIG_DFL);
        cv::ArtifactWriter writer(target, small_layout(target.filename().string()));
        const char byte = 'r';
        if (write(ready[1], &byte, 1) != 1) { _exit(3); }
        pause();
        _exit(4);
    }
    close(ready[1]);
    char byte = 0;
    if (read(ready[0], &byte, 1) != 1) {
        std::cerr << "child did not start\n";
        std::exit(2);
    }
    close(ready[0]);
    return pid;
}

int wait_signal(pid_t pid) {
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) ? WTERMSIG(status) : -1;
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
}

} // namespace

int main() {
    const fs::path root =
        fs::temp_directory_path() / ("ninferno_writer_tmp_" + std::to_string(getpid()));
    fs::remove_all(root);
    fs::create_directories(root);

    // SIGKILL leaves the temporary; the next conversion into the directory removes it.
    {
        const auto dir = root / "kill";
        fs::create_directories(dir);
        const auto pid = spawn_writer(dir / "a.ninferno");
        check(temporaries(dir).size() == 1, "a live writer has one temporary");
        kill(pid, SIGKILL);
        check(wait_signal(pid) == SIGKILL, "the child died of SIGKILL");
        check(temporaries(dir).size() == 1, "SIGKILL leaves the temporary");
        write_text(dir / ".notes.abcdef.tmp", "not an artifact, keep"); // looks like one, is not
        write_text(dir / ".x.tmp", "short name, keep");
        {
            cv::ArtifactWriter writer(dir / "b.ninferno", small_layout("b.ninferno"));
            // b's own temporary exists, the dead one is gone, the lookalikes stay
            check(temporaries(dir).size() == 3, "the stale temporary was removed, lookalikes kept");
            check(fs::exists(dir / ".notes.abcdef.tmp") && fs::exists(dir / ".x.tmp"),
                  "files that are not ours stay");
        }
        check(temporaries(dir).size() == 2, "an unfinished writer removed its own temporary");
    }

    // A live writer's temporary is never removed by another conversion in the directory.
    {
        const auto dir = root / "live";
        fs::create_directories(dir);
        cv::ArtifactWriter first(dir / "a.ninferno", small_layout("a.ninferno"));
        check(temporaries(dir).size() == 1, "the first writer has a temporary");
        check(cv::ArtifactWriter::remove_stale_temporaries(dir) == 0, "a direct sweep skips it");
        {
            cv::ArtifactWriter second(dir / "b.ninferno", small_layout("b.ninferno"));
            check(temporaries(dir).size() == 2, "a second writer leaves the live temporary");
            second.finish({});
        }
        check(temporaries(dir).size() == 1, "the live temporary is still there");
        first.finish({});
        check(fs::exists(dir / "a.ninferno") && fs::exists(dir / "b.ninferno"),
              "both artifacts were published");
        check(temporaries(dir).empty(), "no temporary remains after publishing");
    }

    // A sparse temporary whose writer died before its header (all zeros) goes too.
    {
        const auto dir = root / "zeros";
        fs::create_directories(dir);
        std::ofstream(dir / ".z.ninferno.AbC123.tmp", std::ios::binary).put('\0');
        check(cv::ArtifactWriter::remove_stale_temporaries(dir) == 1, "an empty temporary goes");
    }

    // SIGINT and SIGTERM: the temporary is removed and the process still dies of the signal.
    for (const int number : {SIGINT, SIGTERM}) {
        const auto dir = root / ("signal" + std::to_string(number));
        fs::create_directories(dir);
        const auto pid = spawn_writer(dir / "a.ninferno");
        check(temporaries(dir).size() == 1, "the writer has a temporary before the signal");
        kill(pid, number);
        check(wait_signal(pid) == number, "the child died of the signal it was sent");
        check(temporaries(dir).empty(), "the signal removed the temporary");
    }

    fs::remove_all(root);
    if (g_failures == 0) { std::cout << "writer temporaries: ok\n"; }
    return g_failures == 0 ? 0 : 1;
}
