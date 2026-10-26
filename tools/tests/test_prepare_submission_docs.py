"""Release gates must distinguish human validation from the author-declared D38 AI review."""
from copy import deepcopy

import pytest

from tools.prepare_submission_docs import validate_final_inputs


def base():
    return {"meta.preliminary": False, "meta.cohort": "confirmatory", "meta.analysis_dirty": False,
            "meta.analysis_commit": "abcdef0", "meta.account_spend_usd": 123.0,
            "meta.account_spend_source": "Opening/closing balances and credited top-ups for the named window",
            "coding.handcheck.n": 100, "coding.handcheck.raw_agreement": .96, "coding.handcheck.kappa": .92}


def d38():
    r = base()
    r.update({"coding.handcheck.n": "pending", "coding.handcheck.raw_agreement": "pending",
              "coding.handcheck.kappa": "pending", "meta.account_spend_usd": "pending",
              "meta.account_spend_source": "pending", "review.human.status": "not performed (D38)",
              "review.ai.status": "completed AI second review (D38)", "review.ai.reviewer_type": "AI",
              "review.ai.human_validation": False, "review.ai.blinded_to_classifier_labels": True,
              "review.ai.n": 100, "review.ai.n_agree": 96, "review.ai.raw_agreement": .96,
              "review.ai.kappa": .92, "review.ai.sheet_sha256": "a"*64,
              "review.ai.reviewer_metadata_sha256": "b"*64, "review.account.status": "deferred (D38)",
              "costs.key_usage.net_usd": 120.0, "costs.key_usage.scope": "current API key only",
              "costs.key_usage.source_sha256": "c"*64})
    text = ("Human validation was not completed (D38). The AI second review does not replace human validation. "
            "Account-wide balance and top-up reconciliation was deferred; no whole-account spending total is claimed.\n"
            "## LLM Usage Statement\nAI research assistance is disclosed.")
    return r, text


def test_original_human_and_account_path_remains_supported():
    validate_final_inputs(base(), "## LLM Usage Statement\nAI research assistance is disclosed.")


def test_d38_needs_explicit_deviations_and_identified_ai_evidence():
    r, text = d38()
    validate_final_inputs(r, text)
    assert r["review.ai.human_validation"] is False
    assert r["coding.handcheck.n"] == "pending"


@pytest.mark.parametrize("key", ["review.human.status", "review.ai.status", "review.ai.sheet_sha256",
                                 "review.ai.reviewer_metadata_sha256", "review.ai.n",
                                 "review.account.status", "costs.key_usage.scope", "costs.key_usage.source_sha256"])
def test_d38_does_not_silently_accept_missing_inputs(key):
    r, text = d38()
    del r[key]
    with pytest.raises(ValueError):
        validate_final_inputs(r, text)


@pytest.mark.parametrize("key,value", [("review.ai.human_validation", True),
                                       ("review.ai.blinded_to_classifier_labels", False),
                                       ("review.ai.raw_agreement", .99),
                                       ("meta.analysis_dirty", True),
                                       ("meta.analysis_commit", ""),
                                       ("costs.key_usage.scope", "whole account")])
def test_false_or_mismatched_proof_blocks_final_package(key, value):
    r, text = d38()
    r[key] = value
    with pytest.raises(ValueError):
        validate_final_inputs(r, text)


@pytest.mark.parametrize("phrase", ["Human validation was not completed (D38).",
                                    "does not replace human validation",
                                    "reconciliation was deferred",
                                    "no whole-account spending total is claimed",
                                    "## LLM Usage Statement"])
def test_disclosures_must_be_visible_in_report(phrase):
    r, text = d38()
    with pytest.raises(ValueError):
        validate_final_inputs(r, text.replace(phrase, ""))


def test_pending_editorial_source_is_not_final():
    r, text = d38()
    for leftover in (" {{missing}}", " <!-- author TODO -->"):
        with pytest.raises(ValueError):
            validate_final_inputs(deepcopy(r), text + leftover)


@pytest.mark.parametrize("scope", ["human", "account"])
def test_stale_completed_fields_cannot_contradict_d38(scope):
    r, text = d38()
    for key, value in base().items():
        if key.startswith("coding.handcheck." if scope == "human" else "meta.account_spend"):
            r[key] = value
    with pytest.raises(ValueError):
        validate_final_inputs(r, text)


def test_blank_llm_statement_is_rejected():
    r, text = d38()
    with pytest.raises(ValueError):
        validate_final_inputs(r, text.replace("AI research assistance is disclosed.", ""))
