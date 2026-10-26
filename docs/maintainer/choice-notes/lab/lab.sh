#!/bin/bash
# Helpers for a Claude session on a shared stand. The stand's GPU queue (/root/queue/runner.sh) waits
# while /root/GPU_BUSY exists; a session takes that lock the same way and releases only its own.
# Everything this session creates lives under /root/claude.
TAG_FILE=/root/claude/lock.tag
lock() {  # lock PURPOSE: wait until no queue job runs and the lock is free, then hold it
  local tag="claude-session:$1"
  while true; do
    if [ ! -e /root/GPU_BUSY ] && [ -z "$(ls /root/queue/running 2>/dev/null)" ]; then
      if ( set -o noclobber; echo "$tag" > /root/GPU_BUSY ) 2>/dev/null; then
        sleep 2
        if [ "$(cat /root/GPU_BUSY 2>/dev/null)" = "$tag" ] && [ -z "$(ls /root/queue/running 2>/dev/null)" ]; then
          echo "$tag" > $TAG_FILE; echo "locked: $tag"; return 0
        fi
      fi
    fi
    sleep 10
  done
}
unlock() {
  if [ -f $TAG_FILE ] && [ "$(cat /root/GPU_BUSY 2>/dev/null)" = "$(cat $TAG_FILE)" ]; then
    rm -f /root/GPU_BUSY; echo "unlocked: $(cat $TAG_FILE)"
  fi
  rm -f $TAG_FILE
}
held() { [ -f $TAG_FILE ] && [ "$(cat /root/GPU_BUSY 2>/dev/null)" = "$(cat $TAG_FILE)" ]; }
cuda_env() { local c=$(ls -d /usr/local/cuda-13* 2>/dev/null | sort -V | tail -1); export PATH=${c:-/usr/local/cuda}/bin:$PATH; }
ccache_env() { export CCACHE_DIR=/root/.ccache CCACHE_BASEDIR=/root/claude/wt_tst CCACHE_NOHASHDIR=1; }
