#!/usr/bin/env bash
# ------------------------------------------------------------------------
# 训练监控脚本（系统 cron 每小时执行一次）
#
# 职责：
#   1. 检查 tmux 窗口里的训练进程是否存活、metrics.csv 是否在更新；
#   2. 正常 → 什么都不做；
#   3. 异常（崩溃/挂死）→ 抓取现场证据（pane 尾部、nvidia-smi、OOM 日志），
#      若存在可恢复检查点（last.ckpt / checkpoint_*.ckpt）则用 RF_RESUME 从
#      最近检查点续训；若训练已正常完成（epoch 达上限）则不重启；
#      24h 内连续崩溃 ≥3 次则停止自动重启，留下证据等人工处理。
#
# 用法：
#   bash monitor_train.sh          # 正常检查
#   bash monitor_train.sh --dry    # 只检查打印状态，不执行重启动作
# ------------------------------------------------------------------------
set -u

TMUX_BIN=/usr/bin/tmux
UV_BIN=/home/liu/.local/bin/uv
PROJECT_ROOT=/home/liu/wzt/rf-detr
SCRIPT_PATH=src/scripts/experiments_tmp/train_qnorm_eumix.py
WINDOW=oversample
OUTPUT_DIR=/home/liu/wzt/rf-detr/output/0809-SHWX-rfdetr-medium-rare-oversample-SSCL-Proj-QNormEUMix
EPOCHS=100                      # 训练总轮数（metrics.csv 中 epoch 从 0 计，达到 99 视为完成）
STALE_THRESHOLD=1800            # metrics.csv 超过 1800 秒未更新视为挂死
MAX_RESTARTS_PER_DAY=3          # 24h 内最大自动重启次数
DRY="${1:-}"

MONITOR_DIR="$OUTPUT_DIR/monitor"
STATE_FILE="$MONITOR_DIR/restart_state"
LOG_FILE="$MONITOR_DIR/monitor.log"
mkdir -p "$MONITOR_DIR"

now_ts() { date +%s; }
log() { echo "[$(date '+%F %T')] $*" >> "$LOG_FILE"; }

# 检测训练进程（cmdline 含脚本路径，且进程本身是 python/uv 而非 bash 包装）
proc_alive() {
    local pid
    for pid in $(pgrep -f "$SCRIPT_PATH" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        if grep -q "$SCRIPT_PATH" "/proc/$pid/cmdline" 2>/dev/null && \
           grep -qiE "python|uv" "/proc/$pid/comm" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

# metrics.csv 最后修改时间（秒），文件不存在返回 0
metrics_mtime() {
    local f="$OUTPUT_DIR/metrics.csv"
    if [ -f "$f" ]; then stat -c %Y "$f"; else echo 0; fi
}

# 最近一次检查点的 epoch 号（从 checkpoint_{epoch}.ckpt / metrics 推断）
last_epoch_from_ckpt() {
    ls "$OUTPUT_DIR"/checkpoint_*.ckpt 2>/dev/null | sed -E 's/.*checkpoint_([0-9]+)\.ckpt/\1/' | sort -n | tail -1
}

# 选取最近的可恢复检查点：last.ckpt（每 epoch 更新，含优化器/EMA/调度器）优先
pick_resume_ckpt() {
    local last="$OUTPUT_DIR/last.ckpt"
    if [ -f "$last" ]; then echo "$last"; return; fi
    ls -t "$OUTPUT_DIR"/checkpoint_*.ckpt 2>/dev/null | head -1
}

# 抓取崩溃现场证据
capture_evidence() {
    local ts="$1"
    local f="$MONITOR_DIR/crash_$ts.log"
    {
        echo "===== 崩溃现场抓取时间: $(date '+%F %T') ====="
        echo "--- tmux pane 尾部 (200 行) ---"
        "$TMUX_BIN" capture-pane -t "$WINDOW" -p -S -200 2>&1 | tail -200
        echo
        echo "--- metrics.csv 尾部 ---"
        tail -5 "$OUTPUT_DIR/metrics.csv" 2>&1
        echo
        echo "--- nvidia-smi ---"
        nvidia-smi 2>&1 | head -25
        echo
        echo "--- 内存 ---"
        free -g
        echo
        echo "--- 输出目录 ---"
        ls -lt "$OUTPUT_DIR" 2>&1 | head -15
        echo
        echo "--- 内核 OOM 日志 (dmesg 尾部) ---"
        dmesg 2>/dev/null | tail -20 || echo "(dmesg 不可读)"
        echo
        echo "--- 训练相关进程 ---"
        ps aux | grep -E "train_qnorm|train_rare|python" | grep -v grep | head -8
    } > "$f" 2>&1
    log "已抓取崩溃现场: $f"
    echo "$f"
}

# 训练是否已正常完成（metrics 达到最后 epoch）
is_completed() {
    local max_epoch
    max_epoch=$(awk -F, 'NR>1 {if ($1+0 > m) m=$1+0} END {print m+0}' "$OUTPUT_DIR/metrics.csv" 2>/dev/null)
    [ "$max_epoch" -ge $((EPOCHS - 1)) ]
}

# 重启计数：24h 内的重启次数是否已达上限
too_many_restarts() {
    [ ! -f "$STATE_FILE" ] && return 1
    local now
    now=$(now_ts)
    local recent
    recent=$(awk -v now="$now" -v limit=86400 '{if (now - $1 < limit) c++} END {print c+0}' "$STATE_FILE")
    [ "$recent" -ge "$MAX_RESTARTS_PER_DAY" ]
}

record_restart() {
    echo "$(now_ts)" >> "$STATE_FILE"
}

restart_training() {
    local ckpt="$1"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)

    if too_many_restarts; then
        log "!!! 24h 内自动重启已达 ${MAX_RESTARTS_PER_DAY} 次，停止自动续训，请人工检查 $MONITOR_DIR/crash_*.log"
        return 1
    fi

    # 重建干净的 tmux 窗口（若用户正附着会被踢出，重启后窗口名不变）
    "$TMUX_BIN" kill-window -t "$WINDOW" 2>/dev/null || true
    sleep 1
    "$TMUX_BIN" new-session -d -s "$WINDOW"

    if [ -n "$ckpt" ]; then
        "$TMUX_BIN" send-keys -t "$WINDOW" \
            "RF_RESUME=$ckpt cd $PROJECT_ROOT && $UV_BIN run $SCRIPT_PATH" Enter
        record_restart
        log "已从检查点续训: $ckpt"
    else
        # 无检查点（epoch 0 内崩溃）：从头启动一次，依赖重启上限防死循环
        "$TMUX_BIN" send-keys -t "$WINDOW" \
            "cd $PROJECT_ROOT && $UV_BIN run $SCRIPT_PATH" Enter
        record_restart
        log "无检查点，从头启动（重启计数+1）"
    fi
    return 0
}

main() {
    # --- 1. 训练是否存活 ---
    if proc_alive; then
        local mtime
        mtime=$(metrics_mtime)
        local age=$(( $(now_ts) - mtime ))
        if [ "$age" -lt "$STALE_THRESHOLD" ]; then
            log "检查: 训练正常运行 (进程存活, metrics 更新于 ${age}s 前)"
            echo "OK: 训练正常运行"
            return 0
        fi
        # 进程在但指标长期不更新 → 挂死，走崩溃处理
        log "警告: 进程存活但 metrics.csv 已 ${age}s 未更新，判定挂死"
        echo "HUNG: 进程存活但 metrics 停滞 ${age}s，需要重启"
    else
        echo "DEAD: 未检测到训练进程"
        log "警告: 未检测到训练进程"
    fi

    # --- 2. 正常完成则不重启 ---
    if [ -f "$OUTPUT_DIR/metrics.csv" ] && is_completed; then
        log "训练已正常完成（epoch >= $((EPOCHS - 1))），不重启"
        echo "DONE: 训练已正常完成"
        return 0
    fi

    # --- 3. 崩溃处理：抓现场 → 选检查点 → 续训 ---
    local ts ckpt
    ts=$(date +%Y%m%d_%H%M%S)
    capture_evidence "$ts"
    ckpt=$(pick_resume_ckpt)
    if [ -n "$ckpt" ]; then
        echo "RESTART: 从检查点 $ckpt 续训"
    else
        echo "RESTART: 无检查点，从头启动"
    fi

    if [ "$DRY" = "--dry" ]; then
        log "DRY 模式: 跳过实际重启 (ckpt=$ckpt)"
        return 0
    fi
    restart_training "$ckpt"
}

main
