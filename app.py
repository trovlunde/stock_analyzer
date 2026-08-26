"""Backward-compatible entry point; prefer `python -m stock_analysis.web.app`."""

from stock_analysis.web.app import app, create_app, Config
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    create_app()
    logger.info(f"Starting Flask server on {Config.HOST}:{Config.PORT}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
    )
