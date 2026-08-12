@echo off
cd /d "%~dp0\..\.."
uv run python scripts\finviz\finviz_scrape.py
