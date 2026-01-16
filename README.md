# Sentiment Production Pipeline

## Setup Guide

Clone the repository:

```bash
git clone https://github.com/Nouman64-cat/sentiment-production-pipeline.git
```

### Environment Setup

We recommend **uv** for instant setup, but standard **pip** is fully supported.

| Step                | Option A: uv (Recommended)                                  | Option B: pip (Standard)                                                      |
| :------------------ | :---------------------------------------------------------- | :---------------------------------------------------------------------------- |
| **1. Install Tool** | `pip install uv`                                            | (Pre-installed)                                                               |
| **2. Initialize**   | `uv sync`                                                   | `python -m venv .venv`                                                        |
| **3. Activate**     | _(Auto-handled by `uv run`)_ or `source .venv/bin/activate` | `source .venv/bin/activate` (Mac/Linux)<br>`.venv\Scripts\activate` (Windows) |
| **4. Install Deps** | **Done!** (Handled by step 2)                               | `pip install -r requirements.txt`                                             |
