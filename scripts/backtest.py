#!/usr/bin/env python3
"""
AlphaStrike Backtesting CLI

Run historical backtests on the AlphaStrike trading strategy.

Usage:
    python scripts/backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31
    python scripts/backtest.py --symbol ETHUSDT --start 2024-06-01 --end 2024-12-31 --output ./reports

Options:
    --symbol      Trading pair (required)
    --start       Start date YYYY-MM-DD (required)
    --end         End date YYYY-MM-DD (required)
    --interval    Candle interval (default: 1m)
    --balance     Initial balance (default: 10000)
    --leverage    Leverage (default: 5)
    --slippage    Slippage in bps (default: 5)
    --output      Output directory for reports (default: ./backtest_results)
    --download    Force re-download of data
    --verbose     Enable verbose logging
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AlphaStrike Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading pair (e.g., BTCUSDT)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interval",
        default="1m",
        help="Candle interval (default: 1m)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10000.0,
        help="Initial balance (default: 10000)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=5,
        help="Leverage (default: 5)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backtest_results"),
        help="Output directory for reports (default: ./backtest_results)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force re-download of data",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("backtest")

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 1

    if start_date >= end_date:
        logger.error("Start date must be before end date")
        return 1

    logger.info("=" * 60)
    logger.info("ALPHASTRIKE BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Balance: ${args.balance:,.2f}")
    logger.info(f"Leverage: {args.leverage}x")
    logger.info("=" * 60)

    try:
        # Import components
        from src.backtest.data_loader import DataLoader
        from src.backtest.engine import BacktestConfig, BacktestEngine
        from src.backtest.report import ReportGenerator
        from src.backtest.simulator import ExecutionSimulator
        from src.data.database import Database
        from src.exchange.adapters.weex.adapter import WEEXRESTClient
        from src.execution.signal_processor import SignalProcessor
        from src.features.pipeline import FeaturePipeline
        from src.ml.ensemble import MLEnsemble

        # Create config
        config = BacktestConfig(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=args.balance,
            leverage=args.leverage,
            slippage_bps=args.slippage,
            interval=args.interval,
        )

        # Initialize database
        logger.info("Initializing database...")
        database = Database()
        await database.initialize()

        # Initialize REST client
        logger.info("Initializing exchange client...")
        rest_client = WEEXRESTClient()
        await rest_client.initialize()

        # Load or download data
        logger.info("Loading historical data...")
        data_loader = DataLoader(database=database, rest_client=rest_client)

        candles = await data_loader.fetch_candles(
            symbol=config.symbol,
            start_time=config.start_date,
            end_time=config.end_date,
            interval=config.interval,
            force_download=args.download,
        )

        if not candles:
            logger.error("No candle data available for the specified period")
            return 1

        logger.info(f"Loaded {len(candles)} candles")

        # Initialize trading components
        logger.info("Initializing ML models...")
        feature_pipeline = FeaturePipeline()
        ml_ensemble = MLEnsemble(models_dir=Path("models"))
        ml_ensemble.check_and_reload_models()
        signal_processor = SignalProcessor()
        simulator = ExecutionSimulator(slippage_bps=config.slippage_bps)

        # Create and run engine
        engine = BacktestEngine(
            config=config,
            feature_pipeline=feature_pipeline,
            ml_ensemble=ml_ensemble,
            signal_processor=signal_processor,
            execution_simulator=simulator,
        )

        logger.info("Running backtest...")
        result = engine.run(candles)

        # Generate and save report
        report = ReportGenerator(
            metrics=result.metrics,
            trades=result.trades,
            config=config,
            equity_curve=result.equity_curve,
        )

        # Print summary to console
        print("\n" + report.generate_summary())

        # Save detailed reports
        args.output.mkdir(parents=True, exist_ok=True)
        report.save_report(args.output)
        logger.info(f"Reports saved to: {args.output}")

        # Cleanup
        await rest_client.close()
        await database.close()

        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed and models are available")
        return 1

    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
