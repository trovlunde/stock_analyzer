"""CLI for evaluating and comparing ML movement classifiers."""

import argparse

import pandas as pd

from stock_analysis.ai.helpers import get_index_data, classifiers as _classifiers_registry
from stock_analysis.ai.technical_analysis.movement_classification import (
    compare_variants,
    evaluate_all_classifiers,
    resolve_backtest_period,
    run_movement_backtests,
    temporal_holdout_split,
    train_classifier_single_stock,
    test_model_performance,
    test_classifiers,
)
from stock_analysis.ai.technical_analysis.prepare_classification_data import (
    prepare_classification_data,
    prepare_classification_data_enhanced,
)
from sklearn.ensemble import RandomForestClassifier


def _prepare_data(ticker, period, threshold, use_extra_features, enhanced):
    data = get_index_data(ticker, period)
    if data is None or data.empty:
        raise ValueError(f"No data for {ticker}")
    prepare_fn = (
        prepare_classification_data_enhanced
        if enhanced
        else prepare_classification_data
    )
    prepared = prepare_fn(
        data,
        predict_weekly=False,
        threshold=threshold,
        use_extra_features=use_extra_features,
    )
    return data, prepared.dropna()


def _resolve_classifier_names(names):
    """Resolve CLI --classifier values to registry keys; None means all."""
    if names is None:
        return None
    if len(names) == 1 and names[0].lower() == "all":
        return None
    lower_map = {k.lower(): k for k in _classifiers_registry}
    resolved = []
    for name in names:
        canonical = lower_map.get(name.lower())
        if canonical is None:
            available = list(_classifiers_registry.keys())
            raise ValueError(f"Unknown classifier {name!r}. Available: {available}")
        resolved.append(canonical)
    return resolved


def cmd_compare(args):
    """Compare multiple classifiers with temporal holdout."""
    _, prepared = _prepare_data(
        args.ticker, args.period, args.threshold, args.extra_features, enhanced=True
    )
    classifier_names = _resolve_classifier_names(getattr(args, "classifier", None))
    result = evaluate_all_classifiers(
        prepared,
        threshold=args.threshold,
        use_extra_features=args.extra_features,
        holdout_months=args.holdout_months,
        test_size=args.test_size,
        plot=not args.no_plot,
        classifier_names=classifier_names,
    )
    metrics = result["holdout_metrics"]
    print(
        f"\nHoldout summary: {result['best_classifier']} — "
        f"accuracy={metrics['accuracy']:.3f}, f1={metrics['f1_weighted']:.3f}"
    )


def cmd_train(args):
    """Train a single classifier with temporal holdout evaluation."""
    data, prepared_daily = _prepare_data(
        args.ticker, args.period, args.threshold, args.extra_features, enhanced=False
    )
    prepared_weekly = prepare_classification_data(
        data,
        predict_weekly=True,
        threshold=args.threshold,
        use_extra_features=args.extra_features,
    ).dropna()

    if args.holdout_months > 0:
        daily_train, daily_holdout = temporal_holdout_split(
            prepared_daily, holdout_months=args.holdout_months
        )
        weekly_train, weekly_holdout = temporal_holdout_split(
            prepared_weekly, holdout_months=args.holdout_months
        )
    else:
        daily_train, daily_holdout = prepared_daily, None
        weekly_train, weekly_holdout = prepared_weekly, None

    use_holdout = args.holdout_months > 0
    eval_split = 0 if use_holdout else 0.2
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    daily_model, daily_scaler, daily_meta = train_classifier_single_stock(
        daily_train,
        use_extra_features=args.extra_features,
        threshold=args.threshold,
        classifier=clf,
        overfit_check=args.overfit_check,
        eval_split=eval_split,
        verbose=not args.quiet,
    )
    weekly_model, weekly_scaler, weekly_meta = train_classifier_single_stock(
        weekly_train,
        predict_weekly=True,
        use_extra_features=args.extra_features,
        threshold=args.threshold,
        classifier=RandomForestClassifier(n_estimators=100, random_state=42),
        overfit_check=args.overfit_check,
        eval_split=eval_split,
        verbose=not args.quiet,
    )

    eval_daily, eval_weekly, eval_stock, backtest_label = resolve_backtest_period(
        prepared_daily,
        prepared_weekly,
        data,
        holdout_months=args.holdout_months if use_holdout else 0,
        cutoff_date=(
            prepared_daily.index.max() - pd.DateOffset(months=args.holdout_months)
            if use_holdout else None
        ),
        train_meta=daily_meta,
    )

    test_model_performance(
        eval_daily,
        daily_model,
        daily_scaler,
        eval_weekly,
        weekly_model,
        weekly_scaler,
        threshold=args.threshold,
        use_extra_features=args.extra_features,
    )
    run_movement_backtests(
        data,
        daily_model,
        daily_scaler,
        prepared_daily,
        weekly_model,
        weekly_scaler,
        prepared_weekly,
        args.threshold,
        args.extra_features,
        eval_daily,
        eval_weekly,
        eval_stock,
        backtest_label,
    )


def cmd_diagnose(args):
    """Print ADF stationarity and OLS regression diagnostics on prepared classification features."""
    from stock_analysis.ai.helpers import get_features
    from stock_analysis.diagnostics.stationarity import check_stationarity_frame
    from stock_analysis.diagnostics.regression import regression_diagnostics

    _, prepared = _prepare_data(
        args.ticker, args.period, args.threshold, args.extra_features, enhanced=False
    )

    feature_cols = [c for c in get_features(args.extra_features) if c in prepared.columns]
    X = prepared[feature_cols]
    y = prepared["return"]

    stat_report = check_stationarity_frame(X)
    print("\n=== Stationarity Report (ADF) ===")
    print(stat_report.to_string())

    reg = regression_diagnostics(X, y)
    print("\n=== Regression Diagnostics (OLS vs return) ===")
    if reg["insufficient_data"]:
        print("  insufficient_data: True")
    else:
        print(f"  r_squared:        {reg['r_squared']:.4f}")
        print(f"  durbin_watson:    {reg['durbin_watson']:.4f}")
        print(f"  condition_number: {reg['condition_number']:.4f}")
        print(f"  n_obs:            {reg['n_obs']}")
        print("  coefficients:")
        for feat, coef in reg["coefficients"].items():
            print(f"    {feat}: {coef:.6f}")


def cmd_strategies(args):
    """Compare ensemble, RSI, and MA strategies on temporal test fold."""
    test_classifiers()


def cmd_variants(args):
    """Compare movement-classifier pipeline variants on a shared holdout."""
    compare_variants(
        ticker=args.ticker,
        period=args.period,
        holdout_months=args.holdout_months,
        verbose=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate ML movement classifiers with temporal train/test splits"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser(
        "compare",
        help="Grid-search classifiers, evaluate best on holdout",
        description=(
            "Grid-search registered classifiers with TimeSeriesSplit CV, then evaluate the "
            "best model on a temporal holdout set. Includes both sklearn ensembles (Random "
            "Forest, SVM, KNN, Voting) and LightGBM boosters. Use --classifier to target a "
            "specific model family: 'lightgbm' for fast leaf-wise boosting (best on tabular "
            "data, ~15 MB extra dep); sklearn ensembles for no extra deps or voting combos. "
            "Omit --classifier to let CV pick the best across all registered classifiers."
        ),
    )
    compare.add_argument("--ticker", default="^GSPC")
    compare.add_argument("--period", default="20y")
    compare.add_argument("--holdout-months", type=int, default=12)
    compare.add_argument("--test-size", type=float, default=0.2)
    compare.add_argument("--threshold", type=float, default=0.005)
    compare.add_argument("--extra-features", action="store_true")
    compare.add_argument("--no-plot", action="store_true")
    compare.add_argument(
        "--classifier",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Classifier(s) to evaluate (case-insensitive). "
            "Use 'all' or omit to run every registered classifier. "
            "Prefer 'lightgbm' for fast tabular boosting (leaf-wise trees, often best CV F1). "
            "Prefer sklearn ensembles ('Random Forest', 'Voting Classifier') when you need "
            "no extra dependencies or interpretable feature importances. "
            "E.g. --classifier lightgbm  or  --classifier 'Random Forest' LightGBM"
        ),
    )
    compare.set_defaults(func=cmd_compare)

    train = sub.add_parser(
        "train", help="Train RandomForest daily+weekly models with holdout eval"
    )
    train.add_argument("--ticker", default="^GSPC")
    train.add_argument("--period", default="20y")
    train.add_argument("--holdout-months", type=int, default=12)
    train.add_argument("--threshold", type=float, default=0.005)
    train.add_argument("--extra-features", action="store_true")
    train.add_argument("--overfit-check", action="store_true")
    train.add_argument("--quiet", action="store_true")
    train.set_defaults(func=cmd_train)

    strategies = sub.add_parser(
        "strategies", help="Compare ensemble vs RSI vs MA on temporal test fold"
    )
    strategies.set_defaults(func=cmd_strategies)

    variants = sub.add_parser(
        "variants",
        help="Compare dual-RF pipeline variants on the same temporal holdout",
    )
    variants.add_argument("--ticker", default="^GSPC")
    variants.add_argument("--period", default="20y")
    variants.add_argument("--holdout-months", type=int, default=12)
    variants.set_defaults(func=cmd_variants)

    diagnose = sub.add_parser(
        "diagnose",
        help="ADF stationarity and OLS regression diagnostics on prepared classification features",
    )
    diagnose.add_argument("--ticker", default="^GSPC")
    diagnose.add_argument("--period", default="20y")
    diagnose.add_argument("--threshold", type=float, default=0.005)
    diagnose.add_argument("--extra-features", action="store_true")
    diagnose.set_defaults(func=cmd_diagnose)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
