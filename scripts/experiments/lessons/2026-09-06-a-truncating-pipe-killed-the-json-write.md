# 2026-09-06 — a truncating pipe killed the script before it wrote its output (#3667)

**Cost:** ~10 min, one job re-run.

**What broke:** an analysis was run as `python analyse.py --json out.json | sed -n
"1,26p"` to keep the terminal readable. `sed` closed the pipe after 26 lines,
python took SIGPIPE, and the process died **before** the `--json` write at the
end of `main()`. The console output looked complete and correct — it was the
first 26 lines, which is what was asked for — and the JSON simply did not exist.

`| head -N` and `| sed -n "1,Np"` do this; `| tail -N` does not, because `tail`
reads to EOF.

**Prevented?** *Advice only.* Never truncate the stdout of a script that also
writes a file. Redirect to a log and `tail` the log
(`python analyse.py > run.log 2>&1; tail -20 run.log`), which is also what leaves
something to read when it fails.
