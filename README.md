# AI Trading Bot — Dashboard

This repository contains a Streamlit dashboard for monitoring the AI trading system (predictions, logs, P&L, stats).

Quick start (local):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-deploy.txt
streamlit run dashboard.py
```

Docker (build & run):

```bash
docker build -t ai-dashboard .
docker run -p 8501:8501 ai-dashboard
```

Deploy options:

- Streamlit Cloud: push repo to GitHub and create app on share.streamlit.io (easy, free for public repos).
- Deploy container to Render / Railway / DigitalOcean App / AWS ECS using the provided `Dockerfile` or `Procfile`.

Keep-alive:

- To avoid sleeping, use an external ping service (UptimeRobot) or a scheduled GitHub Action that requests your app URL periodically.

Notes:

- Logs are written to `logs/predictions.csv`. Keep `logs/` persisted when deploying if you want to retain history.
- If you use Streamlit Cloud, add `requirements-deploy.txt` to GitHub and set the app to run `dashboard.py`.

If you want, I can:
- Create a GitHub Actions workflow to build and push a Docker image,
- Prepare a small `keep_alive` GitHub Action to ping the app URL periodically,
- Or set up a full backend API + static frontend architecture.
