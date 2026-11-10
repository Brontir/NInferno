#!/bin/bash
# commit+push then rebuild/run the authorities test on the stand
cd C:/Users/MixaPC/wt_630fix
git clang-format --binary /c/Users/MixaPC/AppData/Local/Programs/Python/Python312/Scripts/clang-format $(git merge-base HEAD origin/master) -- ":(exclude)third_party/**" >/dev/null 2>&1
git add -A; git commit -qm "$1

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"; git push -q origin HEAD:refactor/614-route-authorities 2>&1 | tail -3
ssh -F $HOME/.ssh/config.ninfer l4060 'set +u; . /root/claude/lab.sh; cuda_env; W=/root/claude/wt_630; B=/root/claude/build_630_89; git -C $W fetch -q origin refactor/614-route-authorities; git -C $W checkout -q -f --detach FETCH_HEAD; git -C $W log --oneline -1; export CCACHE_DIR=/root/.ccache CCACHE_BASEDIR=$W CCACHE_NOHASHDIR=1; cd $B && nice -n 15 ninja -j8 ninferno_route_authorities_test 2>&1 | tail -3; timeout 900 ./tests/ninferno_route_authorities_test 2>&1 | tail -40; echo auth_rc=${PIPESTATUS[0]}' </dev/null
