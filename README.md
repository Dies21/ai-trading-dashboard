**AI Trading Dashboard — Развёртывание**

- **Кратко:** этот репозиторий содержит Streamlit-дэшборд `dashboard.py` и Dockerfile для контейнерного развёртывания. CI собирает и пушит образ в GitHub Container Registry (GHCR).

**Локальная разработка**

- Создайте виртуальное окружение и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-deploy.txt
```

- Запуск Streamlit локально:

```powershell
streamlit run dashboard.py --server.port 8502
```

**Сборка Docker-образа локально**

```powershell
docker build -t ai-dashboard .
docker run --rm -p 8501:8501 --name ai-dashboard-run ai-dashboard
```

Если Docker Desktop требует WSL2, установите WSL и перезагрузите систему. Проверьте `wsl -l -v`.

**Публикация в GitHub Container Registry (GHCR)**

1. Workflow `/.github/workflows/docker-build-push.yml` автоматически собирает и пушит образ при push в `main`.
2. Workflow помечает образ тегами `latest` и с commit SHA.
3. Альтернативно можно вручную залогиниться и запушить:

```powershell
docker login ghcr.io -u <GITHUB_USERNAME>
# введите PAT с правами packages:write
docker tag ai-dashboard ghcr.io/<OWNER>/ai-trading-dashboard:latest
docker push ghcr.io/<OWNER>/ai-trading-dashboard:latest
```

**Развёртывание контейнера как публичного сайта**

Варианты:
- Render: создайте Web Service и укажите Docker image из GHCR (`ghcr.io/<OWNER>/ai-trading-dashboard:latest`). Render автоматически запускает контейнер и пробрасывает порт.
- Fly.io / Railway: аналогично — используйте image из GHCR или настройте автоматический деплой из репозитория.

Пример для Render:
1. Войдите в Render, создайте New → Web Service.
2. Choose "Docker" и укажите image `ghcr.io/<OWNER>/ai-trading-dashboard:latest`.
3. Нажмите Create — сервис запустит контейнер и сделает его доступным по публичному URL.

**Советы по безопасности**
- PAT (если используете вручную) храните в GitHub Secrets или в менеджере секретов провайдера.
- Если приложение использует ключи API (например CCXT), храните их в переменных окружения в хостинге (Render Secrets, GitHub Secrets для Actions и т.д.).

Если хотите — могу автоматически добавить инструкцию по деплою на Render в workflow (CI → deploy). Скажите, куда именно вы хотите сразу задеплоить: `Render`, `Fly`, `Railway` или `Другой`.
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
 - Build a Docker image and push to GitHub Container Registry (GHCR). A workflow is provided: `.github/workflows/docker-build-push.yml`.
	 You can then deploy the image to Render/Cloud using the image `ghcr.io/<your-org>/ai-trading-dashboard:latest`.

Keep-alive:

- To avoid sleeping, use an external ping service (UptimeRobot) or a scheduled GitHub Action that requests your app URL periodically.

Notes:

- Logs are written to `logs/predictions.csv`. Keep `logs/` persisted when deploying if you want to retain history.
- If you use Streamlit Cloud, add `requirements-deploy.txt` to GitHub and set the app to run `dashboard.py`.

If you want, I can:
- Create a GitHub Actions workflow to build and push a Docker image,
- Prepare a small `keep_alive` GitHub Action to ping the app URL periodically,
- Or set up a full backend API + static frontend architecture.
