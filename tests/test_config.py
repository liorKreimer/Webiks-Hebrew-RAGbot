import logging
from webiks_hebrew_ragbot import config


def test_warn_if_unknown_retrieval_mode_silent_for_known_values(caplog):
    with caplog.at_level(logging.WARNING):
        config.warn_if_unknown_retrieval_mode(config.RETRIEVAL_MODE_DENSE)
        config.warn_if_unknown_retrieval_mode(config.RETRIEVAL_MODE_HYBRID)

    assert caplog.records == []


def test_warn_if_unknown_retrieval_mode_warns_on_typo(caplog):
    with caplog.at_level(logging.WARNING):
        config.warn_if_unknown_retrieval_mode("Hybrid")  # wrong case, a realistic typo

    assert len(caplog.records) == 1
    assert "RETRIEVAL_MODE" in caplog.records[0].message
    assert "Hybrid" in caplog.records[0].message
