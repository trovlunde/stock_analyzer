@echo off
cd /d "%~dp0"
uv run python finviz_recs\finviz_scrape.py
