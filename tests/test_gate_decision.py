"""Unit tests for the validated decision policy and the composed SEND/HOLD pipeline.

Pure logic — no model, no GPU. Locks the frozen operating point (WORK_MAP §4k) and
the composition rules against regression.  Run: ``pytest tests/test_gate_decision.py``
"""

from mirrorfield.mcp.uncertainty import decide, load_gate_thresholds, send_hold_decision


def test_thresholds_are_the_frozen_validated_values():
    th = load_gate_thresholds()
    assert th["tau_abstain"] == 0.7763
    assert th["tau_present"] == 0.8684
    assert th["tau_abstain"] < th["tau_present"]


def test_decide_boundaries():
    th = load_gate_thresholds()
    assert decide(th["tau_present"] + 0.01, False) == "PRESENT"
    assert decide(th["tau_present"], False) == "PRESENT"                       # >= present
    assert decide((th["tau_abstain"] + th["tau_present"]) / 2, False) == "VERIFY"
    assert decide(th["tau_abstain"] - 0.01, False) == "ABSTAIN"
    assert decide(th["tau_abstain"], False) == "VERIFY"                        # in the VERIFY band


def test_decide_safe_defaults():
    assert decide(None, False) == "VERIFY"      # no signal -> VERIFY, never a silent PRESENT
    assert decide(0.99, True) == "VERIFY"       # warming up overrides a high score


def test_send_hold_maps_wrongness_to_send_verify_hold():
    assert send_hold_decision(0.95, False)["decision"] == "SEND"      # PRESENT
    assert send_hold_decision(0.50, False)["decision"] == "HOLD"      # ABSTAIN (likely wrong)
    assert send_hold_decision(0.80, False)["decision"] == "VERIFY"    # in-between


def test_harm_overrides_to_hold():
    d = send_hold_decision(0.99, False, harm_score=0.9)              # confident but harmful
    assert d["decision"] == "HOLD"
    assert d["reason"] == "harmful_content"
    assert d["wrongness_decision"] == "PRESENT"                      # records wrongness verdict too


def test_harm_below_threshold_does_not_hold():
    assert send_hold_decision(0.99, False, harm_score=0.1)["decision"] == "SEND"


def test_harm_not_checked_is_flagged_not_assumed_safe():
    d = send_hold_decision(0.99, False, harm_score=None)
    assert d["decision"] == "SEND"
    assert d["harm_score"] is None
    assert "harm not checked" in d["reason"]    # honesty: 'not checked' != 'safe'
