from kalshi_culture_analyzer.config import Config
from kalshi_culture_analyzer.rules import evaluate_rules


def test_aggressive_burst_rule():
    config = Config()
    features = {
        "aggressive_burst_count": config.aggressive.burst_trade_count,
        "aggressive_burst_contracts": config.aggressive.burst_contracts,
        "max_aggressive_yes_trade": 0,
        "trailing_volume": 0,
        "yes_ask_size": 0,
        "mid_change_window": 0,
        "aggressive_step_contracts": 0,
        "yes_mid": 0.5,
    }
    score, rules, explanation = evaluate_rules("MKT", features, {}, config)
    assert "aggressive_burst" in rules
    assert score > 0


def test_large_trade_rule():
    config = Config()
    features = {
        "aggressive_burst_count": 0,
        "aggressive_burst_contracts": 0,
        "max_aggressive_yes_trade": config.large_trade.absolute_contracts,
        "trailing_volume": config.large_trade.absolute_contracts,
        "yes_ask_size": config.large_trade.absolute_contracts,
        "mid_change_window": 0,
        "aggressive_step_contracts": 0,
        "yes_mid": 0.5,
    }
    score, rules, explanation = evaluate_rules("MKT", features, {}, config)
    assert "large_aggressive" in rules
    assert score > 0

