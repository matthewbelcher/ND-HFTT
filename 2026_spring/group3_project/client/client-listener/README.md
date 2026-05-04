# Client Listener (TCP)

Async TCP listener that accepts newline-delimited JSON messages and writes accepted orders to Aurora DSQL.

## Protocol

Each TCP message is one JSON object ending in `\n`.

### Ping

```json
{"action":"ping"}
```

### Submit order

```json
{
  "action":"submit_order",
  "auth_token":"replace-with-token",
  "request":{
    "request_id":"3de2e1e8-6bbd-4ca5-8fe2-b5372faf9d8c",
    "account_id":"f478e3e1-33dc-4fb8-a3ef-b4f426e8c13f",
    "market_id":"3f190438-9094-46a7-abd1-0b6cbcf813cc",
    "side":"YES",
    "qty":3,
    "price_cents":57,
    "time_in_force":"GTC",
    "ingress_ts_ns":1713133377123456789
  }
}
```

### Cancel order

```json
{
  "action":"cancel_order",
  "auth_token":"replace-with-token",
  "request":{
    "cancel_id":"9d87ec16-35de-40e4-b97f-85a5cb2faf89",
    "order_id":"3de2e1e8-6bbd-4ca5-8fe2-b5372faf9d8c",
    "account_id":"f478e3e1-33dc-4fb8-a3ef-b4f426e8c13f",
    "reason":"user requested cancel"
  }
}
```

## Auth

Default mode is a static shared token.

Environment:

- `CLIENT_LISTENER_AUTH_MODE=static-token`
- `CLIENT_LISTENER_AUTH_TOKEN=<shared-token>`

Optional AWS Secrets Manager mode:

- `CLIENT_LISTENER_AUTH_MODE=aws-secretsmanager`
- `CLIENT_LISTENER_AUTH_SECRET_ARN=<secret-arn>`
- `AWS_REGION=us-east-2`

Secret value can be raw token text or JSON with `token` or `auth_token`.

## Run

```bash
cd client/client-listener
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set env vars (example with your DSQL endpoint):

```bash
export CLIENT_LISTENER_HOST=0.0.0.0
export CLIENT_LISTENER_PORT=9001
export CLIENT_LISTENER_DB_DSN='postgresql://<user>:<password>@4btv7qq43k4ztw3tjubmbyf3su.dsql.us-east-2.on.aws:5432/postgres?sslmode=require'
export CLIENT_LISTENER_AUTH_MODE=static-token
export CLIENT_LISTENER_AUTH_TOKEN='dev-shared-token'
python3 server.py
```

## Test Client

Use existing `account_id` and `market_id` values already present in DB:

```bash
python3 test_client.py \
  --host 127.0.0.1 \
  --port 9001 \
  --token dev-shared-token \
  --account-id <account-uuid> \
  --market-id <market-uuid> \
  --count 10 \
  --concurrency 5
```
