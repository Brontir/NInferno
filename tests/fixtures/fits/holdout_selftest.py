"""The --holdout contract of a fit script (tests/test_fitted_holdouts.cpp, #715): run from the repository
root with --holdout, a script whose constant has a Checked record prints its held-out score on its last line
as `holdout: <the record's error>`. This one stands in for such a script in the test's self-check."""
import sys

if '--holdout' not in sys.argv:
    sys.exit('usage: holdout_selftest.py --holdout')
print('fitted on 3 boards, held out 2')
print('holdout: 2.5 % mean absolute error')
if '--trailing' in sys.argv:
    # A line after the score: the score is no longer the last line, so the runner must not read it.
    print('done')
