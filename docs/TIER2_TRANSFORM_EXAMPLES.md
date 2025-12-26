# Tier-2 Transform Validation Log

**Purpose:** Document human validation of semantic transforms for Phase B evaluation.

**Requirement:** Minimum 5 spot-checks total (at least 1 per category recommended).

**Status:** 🔴 INCOMPLETE (0/5 validations completed)

---

## Validation Checklist

- [ ] At least 5 transforms validated
- [ ] All 3 categories represented (preserving, changing, gotcha)
- [ ] Validation notes documented
- [ ] Date and validator recorded

---

## Transform Suite Information

- **Suite ID:** `tier2_transforms_v1`
- **LLM Model:** (e.g., `gpt-4`, `claude-3.5-sonnet`, `manual`)
- **Generation Date:** YYYY-MM-DD
- **Suite Hash:** (from JSON output)
- **Total Transforms:** (from JSON output)

---

## Preserving Transforms (Meaning-Preserving)

**Expected behavior:** Δd̃ ≈ 0 (small boundary distance change, <0.5)

### Validation 1

- **Transform ID:** `preserving_00`
- **Original:** "..."
- **Transformed:** "..."
- **Description:** (from LLM output)
- **Validated:** ✅ YES / ❌ NO
- **Validator:** (your name/initials)
- **Date:** YYYY-MM-DD
- **Notes:**
  - Does the transform preserve sentiment? (YES/NO)
  - Is the wording sufficiently different? (YES/NO)
  - Any concerns or edge cases?

### Validation 2

- **Transform ID:** `preserving_01`
- **Original:** "..."
- **Transformed:** "..."
- **Description:** (from LLM output)
- **Validated:** ✅ YES / ❌ NO
- **Validator:** (your name/initials)
- **Date:** YYYY-MM-DD
- **Notes:**
  - (same questions as above)

---

## Changing Transforms (Intent-Flipping)

**Expected behavior:** Δd̃ >> 0 (large boundary distance change, >1.5)

### Validation 1

- **Transform ID:** `changing_00`
- **Original:** "..."
- **Transformed:** "..."
- **Description:** (from LLM output)
- **Validated:** ✅ YES / ❌ NO
- **Validator:** (your name/initials)
- **Date:** YYYY-MM-DD
- **Notes:**
  - Does the transform flip sentiment? (YES/NO)
  - Is the surface structure similar? (YES/NO)
  - Is the flip clear and unambiguous? (YES/NO)

### Validation 2

- **Transform ID:** `changing_01`
- **Original:** "..."
- **Transformed:** "..."
- **Description:** (from LLM output)
- **Validated:** ✅ YES / ❌ NO
- **Validator:** (your name/initials)
- **Date:** YYYY-MM-DD
- **Notes:**
  - (same questions as above)

---

## Gotcha Transforms (Surface-Preserving + Intent-Flip)

**Expected behavior:** Δd̃ intermediate (model struggles, 0.2-3.0)

### Validation 1

- **Transform ID:** `gotcha_00`
- **Original:** "..."
- **Transformed:** "..."
- **Description:** (from LLM output)
- **Validated:** ✅ YES / ❌ NO
- **Validator:** (your name/initials)
- **Date:** YYYY-MM-DD
- **Notes:**
  - Does the transform look similar on the surface? (YES/NO)
  - Does it subtly flip intent (sarcasm, irony, negation)? (YES/NO)
  - Is it genuinely tricky/adversarial? (YES/NO)

### Validation 2

- **Transform ID:** `gotcha_01`
- **Original:** "..."
- **Transformed:** "..."
- **Description:** (from LLM output)
- **Validated:** ✅ YES / ❌ NO
- **Validator:** (your name/initials)
- **Date:** YYYY-MM-DD
- **Notes:**
  - (same questions as above)

---

## Additional Validations

(Add more as needed to reach ≥5 total)

### Validation N

- **Transform ID:** `..._##`
- **Category:** preserving / changing / gotcha
- **Original:** "..."
- **Transformed:** "..."
- **Validated:** ✅ YES / ❌ NO
- **Validator:** (your name/initials)
- **Date:** YYYY-MM-DD
- **Notes:** ...

---

## Validation Summary

- **Total Validated:** 0 / 5 minimum
- **Preserving:** 0 / 1 minimum
- **Changing:** 0 / 1 minimum
- **Gotcha:** 0 / 1 minimum
- **Status:** 🔴 INCOMPLETE / 🟢 COMPLETE

---

## Sign-off

Once validation is complete:

- **Validator(s):** (names)
- **Date:** YYYY-MM-DD
- **Meets Phase B requirements:** YES / NO
- **Comments:** (any final notes about transform suite quality)

---

## Example Validation Entry (Reference)

### Example: Preserving Transform

- **Transform ID:** `preserving_05`
- **Original:** "I love this product"
- **Transformed:** "I adore this product"
- **Description:** Synonym substitution (love → adore)
- **Validated:** ✅ YES
- **Validator:** Alice (AI Researcher)
- **Date:** 2025-12-27
- **Notes:**
  - Preserves sentiment? YES - both clearly positive
  - Wording different? YES - synonym swap is sufficient
  - Concerns? None - clean meaning-preserving transform

### Example: Changing Transform

- **Transform ID:** `changing_03`
- **Original:** "This service is excellent"
- **Transformed:** "This service is terrible"
- **Description:** Sentiment word flip (excellent → terrible)
- **Validated:** ✅ YES
- **Validator:** Bob (ML Engineer)
- **Date:** 2025-12-27
- **Notes:**
  - Flips sentiment? YES - positive to negative
  - Surface structure similar? YES - only one word changed
  - Clear flip? YES - unambiguous sentiment reversal

### Example: Gotcha Transform

- **Transform ID:** `gotcha_07`
- **Original:** "Great experience, highly recommend"
- **Transformed:** "Great experience, if you enjoy wasting money"
- **Description:** Sarcasm via ironic conditional
- **Validated:** ✅ YES
- **Validator:** Charlie (Security Researcher)
- **Date:** 2025-12-27
- **Notes:**
  - Surface similar? YES - starts with same positive phrase
  - Flips intent? YES - sarcastic conditional negates the positive
  - Genuinely tricky? YES - a naive classifier might miss the sarcasm

---

## Notes

- This log complements the JSON transform suite artifact
- Validation can be done during `tier2_transform_generate.py` interactive workflow
- Or by manually editing the JSON file and updating this log
- Keep this file version-controlled alongside transform suite JSON
