from stock_analysis.trading.risk import RiskConfig, check_kill_switch


def _config(**kwargs) -> RiskConfig:
    defaults = dict(max_daily_loss_pct=None, max_drawdown_pct=None, liquidate_on_trip=False)
    defaults.update(kwargs)
    return RiskConfig(**defaults)


def test_no_limits_configured_never_trips():
    result = check_kill_switch(
        config=_config(),
        session_start_value=100_000,
        current_value=1,
        equity_curve=[("2024-01-01", 500_000)],
    )
    assert result.tripped is False
    assert result.reason is None


def test_daily_loss_just_under_limit_does_not_trip():
    result = check_kill_switch(
        config=_config(max_daily_loss_pct=0.05),
        session_start_value=100_000,
        current_value=95_001,  # 4.999% loss
        equity_curve=[],
    )
    assert result.tripped is False


def test_daily_loss_at_limit_trips():
    result = check_kill_switch(
        config=_config(max_daily_loss_pct=0.05),
        session_start_value=100_000,
        current_value=95_000,  # exactly 5% loss
        equity_curve=[],
    )
    assert result.tripped is True
    assert "daily loss" in result.reason


def test_daily_loss_over_limit_trips():
    result = check_kill_switch(
        config=_config(max_daily_loss_pct=0.05),
        session_start_value=100_000,
        current_value=80_000,
        equity_curve=[],
    )
    assert result.tripped is True


def test_gain_never_trips_daily_loss():
    result = check_kill_switch(
        config=_config(max_daily_loss_pct=0.05),
        session_start_value=100_000,
        current_value=150_000,
        equity_curve=[],
    )
    assert result.tripped is False


def test_drawdown_just_under_limit_does_not_trip():
    result = check_kill_switch(
        config=_config(max_drawdown_pct=0.20),
        session_start_value=100_000,
        current_value=80_001,
        equity_curve=[("2024-01-01", 100_000)],
    )
    assert result.tripped is False


def test_drawdown_at_limit_trips():
    result = check_kill_switch(
        config=_config(max_drawdown_pct=0.20),
        session_start_value=100_000,
        current_value=80_000,
        equity_curve=[("2024-01-01", 100_000)],
    )
    assert result.tripped is True
    assert "drawdown" in result.reason


def test_drawdown_uses_historical_peak_not_just_session_start():
    # Peak was set on an earlier day, well above today's session-start value.
    result = check_kill_switch(
        config=_config(max_drawdown_pct=0.20),
        session_start_value=90_000,
        current_value=79_000,  # 21% below the historical peak of 100_000
        equity_curve=[("2024-01-01", 100_000), ("2024-01-02", 90_000)],
    )
    assert result.tripped is True


def test_daily_loss_checked_before_drawdown_when_both_configured():
    result = check_kill_switch(
        config=_config(max_daily_loss_pct=0.05, max_drawdown_pct=0.50),
        session_start_value=100_000,
        current_value=90_000,
        equity_curve=[("2024-01-01", 100_000)],
    )
    assert result.tripped is True
    assert "daily loss" in result.reason


def test_zero_session_start_value_does_not_divide_by_zero():
    result = check_kill_switch(
        config=_config(max_daily_loss_pct=0.05),
        session_start_value=0,
        current_value=0,
        equity_curve=[],
    )
    assert result.tripped is False
