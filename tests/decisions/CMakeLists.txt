# The tests of a decisions-only tree (NINFERNO_DECISIONS_ONLY, CMakeLists.txt): the decision core's
# host-only tests alone, the registrations every test tree includes (tests/cmake/DecisionTests.cmake).
# Added with the build tree's tests/ directory as its binary directory, so the executables and their
# working directory are where a full tree has them.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(${PROJECT_SOURCE_DIR}/tests/cmake/NInfernoTests.cmake)
include(${PROJECT_SOURCE_DIR}/tests/cmake/DecisionTests.cmake)
