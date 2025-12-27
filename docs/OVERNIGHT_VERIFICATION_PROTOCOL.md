# Overnight Verification Protocol

**Date:** 2025-12-28
**Status:** Active
**Git Commit:** 1ed0ae5 [clean]

---

## Context

User is sleeping while verification runs (~2.25 hours total). Claude is authorized to handle verification and minor fixes autonomously.

**Verification Run Started:** 2025-12-28 03:04:27
**Run ID (Original):** 20251228_030427
**Expected Completion:** ~05:30 (2 hours from start)

---

## Authorized Actions

### ✅ Allowed (Autonomous)

**1. Run Full Verification:**
- Original Phase D version (~2 hours)
- Optimized Phase D version (~15-20 minutes)
- Automated comparison script
- Document results

**2. Minor Bug Fixes:**
- Import errors
- Tensor shape mismatches
- Seed control issues (determinism breaks)
- Batching order bugs
- Floating-point tolerance adjustments
- Progress logging fixes

**3. Performance Tweaks:**
- Batch size adjustments (if GPU OOM)
- Memory optimization
- Minor refactoring that preserves outputs

**4. Documentation:**
- Record actual speedup
- Document issues and fixes
- Update verification status

### ❌ Not Allowed (Needs User Approval)

- ❌ Change evaluation logic
- ❌ Modify original (slow) version
- ❌ Change metric calculations
- ❌ Make architectural decisions
- ❌ Commit changes that alter scientific results

---

## Decision Tree

```
┌─ Verification passes?
│   └─ ✅ Commit results, update docs, celebrate
│
┌─ Verification fails with fixable bug? (import, tensor shape, etc.)
│   └─ 🔧 Fix, re-run, document in OVERNIGHT_VERIFICATION_LOG.md
│
┌─ Verification fails with unfixable issue?
│   └─ ⚠️ Stop, document problem, wait for user
│
└─ Optimization needs architectural change?
    └─ ⚠️ Stop, document findings, wait for user
```

---

## Logging Requirement

**If ANY changes made while user sleeps:**

Create `docs/OVERNIGHT_VERIFICATION_LOG.md` with:
1. Timestamp of issue
2. What failed and why
3. Every fix applied (with code diffs)
4. Re-run results
5. Final status

**If no changes needed:**
- Document successful verification
- Record actual speedup
- Update main docs

---

## Commit Policy

**Automatic Commits Allowed:**
- Verification results (if clean pass)
- Documentation updates
- OVERNIGHT_VERIFICATION_LOG.md

**Leave Uncommitted (User Review Required):**
- Any code changes to `tier2/integrated_eval_fast.py`
- Any code changes to `experiments/phase_d_integrated_eval_fast.py`
- Any changes that fix bugs

**Rationale:** Code changes should be reviewed, even if they work.

---

## Expected Outcomes

### Best Case ✅
```
1. Original version completes (~2 hours)
2. Optimized version completes (~15-20 minutes)
3. Comparison passes (all metrics match)
4. Speedup documented (5-10×)
5. User wakes to: "Verification complete! 7× speedup confirmed!"
```

### Fixable Issue 🔧
```
1. Verification fails (e.g., import error, tensor shape)
2. Claude identifies root cause
3. Claude applies minimal fix
4. Claude re-runs verification
5. User wakes to: "Found issue X, applied fix Y, verification passed"
6. Code changes left uncommitted for review
```

### Blocker ⚠️
```
1. Verification fails with architectural issue
2. Claude documents problem clearly
3. Claude stops and waits
4. User wakes to: "Hit blocker Z, needs your decision"
5. No code changes attempted
```

---

## Monitoring Schedule

Claude will check progress:
- Every 10 minutes during original run
- Immediately start optimized version when original completes
- Run comparison immediately after optimized completes

---

## Communication

**When user returns, provide:**

1. **Summary:** One-line status (pass/fail/blocked)
2. **Metrics:** Actual runtime for both versions
3. **Speedup:** Calculated improvement factor
4. **Issues:** Any problems encountered and fixes applied
5. **Next Steps:** Recommendations based on outcome

---

## Rollback Plan

If optimization verification fails catastrophically:
- Original (slow) version remains untouched
- Can always fall back to proven baseline
- Optimized version can be debugged separately
- Zero risk to existing working code

---

**END OF PROTOCOL**

**Status:** Claude authorized to proceed autonomously
**User Status:** Sleeping - handle verification independently
**Guardrails:** Active (no architectural changes, document everything)
