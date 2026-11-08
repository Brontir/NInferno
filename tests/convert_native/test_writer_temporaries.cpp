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

#include <cerrno>
#include <chrono>
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

// Forks a child that opens a writer on `target` and waits (it has the parent's signal actions).
// Returns its pid once the temporary exists.
pid_t spawn_writer(const fs::path& target) {
    int ready[2];
    if (pipe(ready) != 0) { std::exit(2); }
    const pid_t pid = fork();
    if (pid == 0) {
        close(ready[0]);
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

// File systems that cannot lock (ENOLCK, EOPNOTSUPP) and one where the lock is held (EWOULDBLOCK).
int lock_unsupported(int, bool) {
    errno = ENOLCK;
    return -1;
}

int lock_busy(int, bool) {
    errno = EWOULDBLOCK;
    return -1;
}

// Two hours ago: older than the sweep's default age.
void make_old(const fs::path& path) {
    fs::last_write_time(path, fs::file_time_type::clock::now() - std::chrono::hours(2));
}

} // namespace

int main() {
    const fs::path root =
        fs::temp_directory_path() / ("ninferno_writer_tmp_" + std::to_string(getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
    // A test run in the background may inherit ignored signals: the handler goes only where the
    // default action is.
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);

    // SIGINT and SIGTERM (first: no writer of this process has installed the handler yet, so the
    // children start with the default actions): the temporary is removed and the process still dies
    // of the signal.
    for (const int number : {SIGINT, SIGTERM}) {
        const auto dir = root / ("signal" + std::to_string(number));
        fs::create_directories(dir);
        const auto pid = spawn_writer(dir / "a.ninferno");
        check(temporaries(dir).size() == 1, "the writer has a temporary before the signal");
        kill(pid, number);
        check(wait_signal(pid) == number, "the child died of the signal it was sent");
        check(temporaries(dir).empty(), "the signal removed the temporary");
    }

    // The handlers of the application stay: a handler with its flags (SA_SIGINFO, SA_RESTART) and
    // an ignored signal are not replaced when a writer registers.
    {
        const pid_t pid = fork();
        if (pid == 0) {
            struct sigaction custom{};
            custom.sa_sigaction = [](int, siginfo_t*, void*) {};
            sigemptyset(&custom.sa_mask);
            sigaddset(&custom.sa_mask, SIGUSR1);
            custom.sa_flags = SA_SIGINFO | SA_RESTART;
            sigaction(SIGINT, &custom, nullptr);
            std::signal(SIGTERM, SIG_IGN);
            const auto dir = root / "preserve";
            fs::create_directories(dir);
            cv::ArtifactWriter writer(dir / "a.ninferno", small_layout("a.ninferno"));
            struct sigaction now{};
            sigaction(SIGINT, nullptr, &now);
            struct sigaction term{};
            sigaction(SIGTERM, nullptr, &term);
            const bool kept =
                now.sa_sigaction == custom.sa_sigaction &&
                (now.sa_flags & (SA_SIGINFO | SA_RESTART)) == (SA_SIGINFO | SA_RESTART) &&
                sigismember(&now.sa_mask, SIGUSR1) == 1 && term.sa_handler == SIG_IGN;
            _exit(kept ? 0 : 1);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        check(
            WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "a writer leaves the application's SIGINT handler and an ignored SIGTERM as they are");
    }

    // A dead writer's temporary (SIGKILL) is removed by the next conversion of the same file, once
    // it is old enough; a young one is kept (a writer that holds no lock may still be writing),
    // and so is whatever is not a temporary of the files being written.
    {
        const auto dir = root / "kill";
        fs::create_directories(dir);
        const auto pid = spawn_writer(dir / "a.ninferno");
        check(temporaries(dir).size() == 1, "a live writer has one temporary");
        kill(pid, SIGKILL);
        check(wait_signal(pid) == SIGKILL, "the child died of SIGKILL");
        const auto dead = temporaries(dir).front();
        check(fs::exists(dead), "SIGKILL leaves the temporary");
        write_text(dir / ".notes.abcdef.tmp", "not an artifact, keep"); // looks like one, is not
        write_text(dir / ".x.tmp", "short name, keep");
        {
            cv::ArtifactWriter young(dir / "a.ninferno", small_layout("a.ninferno"));
            check(fs::exists(dead), "a temporary written a moment ago is not stale");
        }
        make_old(dead);
        make_old(dir / ".notes.abcdef.tmp");
        write_text(dir / ".other.ninferno.AbC123.tmp", std::string(8, '\0')); // another artifact's
        make_old(dir / ".other.ninferno.AbC123.tmp");
        {
            cv::ArtifactWriter writer(dir / "a.ninferno", small_layout("a.ninferno"));
            check(!fs::exists(dead), "the old temporary of the same file was removed");
            check(fs::exists(dir / ".other.ninferno.AbC123.tmp"),
                  "an old temporary of another file stays");
            check(fs::exists(dir / ".notes.abcdef.tmp") && fs::exists(dir / ".x.tmp"),
                  "files that are not ours stay");
        }
        check(temporaries(dir).size() == 3, "an unfinished writer removed its own temporary");
    }

    // A live writer's temporary is never removed by another conversion in the directory, however
    // old it looks (it holds the lock).
    {
        const auto dir = root / "live";
        fs::create_directories(dir);
        cv::ArtifactWriter first(dir / "a.ninferno", small_layout("a.ninferno"));
        check(temporaries(dir).size() == 1, "the first writer has a temporary");
        make_old(temporaries(dir).front());
        check(cv::ArtifactWriter::remove_stale_temporaries(dir, {"a.ninferno"},
                                                           std::chrono::seconds(0)) == 0,
              "a direct sweep skips a locked one");
        {
            cv::ArtifactWriter second(dir / "a.ninferno.part-0001",
                                      small_layout("a.ninferno.part-0001"));
            check(temporaries(dir).size() == 2, "a second writer leaves the live temporary");
            second.finish({});
        }
        check(temporaries(dir).size() == 1, "the live temporary is still there");
        first.finish({});
        check(fs::exists(dir / "a.ninferno") && fs::exists(dir / "a.ninferno.part-0001"),
              "both artifacts were published");
        check(temporaries(dir).empty(), "no temporary remains after publishing");
    }

    // The age and the name rule of the direct sweep; a sparse temporary whose writer died before
    // its header (all zeros) goes too, a part file's temporary with its part's name.
    {
        const auto dir = root / "rules";
        fs::create_directories(dir);
        const auto zeros = dir / ".z.ninferno.AbC123.tmp";
        const auto part  = dir / ".z.ninferno.part-0001.QwE456.tmp";
        const auto other = dir / ".y.ninferno.AbC123.tmp";
        std::ofstream(zeros, std::ios::binary).put('\0');
        std::ofstream(part, std::ios::binary).put('\0');
        std::ofstream(other, std::ios::binary).put('\0');
        const std::vector<std::string> names = {"z.ninferno", "z.ninferno.part-0001"};
        check(cv::ArtifactWriter::remove_stale_temporaries(dir, names) == 0,
              "young temporaries stay at the default age");
        make_old(zeros);
        make_old(part);
        make_old(other);
        check(cv::ArtifactWriter::remove_stale_temporaries(dir, names) == 2,
              "old temporaries of the entry and of a part go");
        check(fs::exists(other), "an old temporary of a name not written stays");
        check(cv::ArtifactWriter::remove_stale_temporaries(dir, {"y.ninferno"}) == 1,
              "and goes for its own name");
    }

    // Where the file system cannot lock, the age and name rules alone decide (a held lock, on the
    // other hand, protects an old file).
    {
        const auto dir   = root / "nolock";
        const auto stale = dir / ".n.ninferno.AbC123.tmp";
        const auto young = dir / ".n.ninferno.QwE456.tmp";
        fs::create_directories(dir);
        std::ofstream(stale, std::ios::binary).put('\0');
        std::ofstream(young, std::ios::binary).put('\0');
        make_old(stale);
        cv::ArtifactWriter::set_lock_for_test(lock_busy);
        check(cv::ArtifactWriter::remove_stale_temporaries(dir, {"n.ninferno"}) == 0 &&
                  fs::exists(stale),
              "a held lock protects an old temporary");
        cv::ArtifactWriter::set_lock_for_test(lock_unsupported);
        check(cv::ArtifactWriter::remove_stale_temporaries(dir, {"n.ninferno"}) == 1,
              "without locks an old temporary is swept");
        check(!fs::exists(stale) && fs::exists(young),
              "and a young one stays, whether or not the file system locks");
        cv::ArtifactWriter::set_lock_for_test(nullptr);
    }

    fs::remove_all(root);
    if (g_failures == 0) { std::cout << "writer temporaries: ok\n"; }
    return g_failures == 0 ? 0 : 1;
}
