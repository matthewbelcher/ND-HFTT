# Frontend

Browser frontend for the prediction market listener. It serves a single-page UI for:

- account creation with password auth
- login with session cookie
- market discovery and live prices
- historical trade charting from persisted trade data
- order entry and cancel
- admin-only market creation, status changes, and resolution

## Environment

Default built-in admin credentials:

- username: `admin`
- password: `MarketAdmin!2026Forever`

When `frontend/server.py` starts with a valid `DB_DSN` or `FRONTEND_DB_DSN`, it will ensure this account exists, is marked admin, and has that password.

Required:

- `FRONTEND_LISTENER_AUTH_TOKEN` or `CLIENT_LISTENER_AUTH_TOKEN`
- `FRONTEND_LISTENER_HOST` or `CLIENT_LISTENER_HOST`
- `FRONTEND_LISTENER_PORT` or `CLIENT_LISTENER_PORT`

Optional:

- `FRONTEND_HOST` default `0.0.0.0`
- `FRONTEND_PORT` default `8080`
- `FRONTEND_SESSION_TTL_SECONDS` default `86400`
- `FRONTEND_SESSION_SECURE_COOKIE` default `false`

Optional admin bootstrap on startup:

- `FRONTEND_BOOTSTRAP_ADMIN_USERNAME`
- `FRONTEND_BOOTSTRAP_ADMIN_PASSWORD`
- `FRONTEND_DB_DSN` or `DB_DSN`

If bootstrap variables are set, the server will:

1. create the account if it does not exist
2. authenticate it through the listener
3. mark that account as `is_admin = TRUE` directly in the database

## Run

```bash
cd /home/ubuntu/Distributed-Prediction-Market-Server
export FRONTEND_LISTENER_HOST=127.0.0.1
export FRONTEND_LISTENER_PORT=9001
export FRONTEND_LISTENER_AUTH_TOKEN='<shared-token>'
python3 frontend/server.py
```

Open `http://<ec2-host>:8080/`.
