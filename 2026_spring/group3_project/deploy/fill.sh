#!/usr/bin/env bash
# Render /tmp/pm.env from env vars, fetching what's missing from AWS.
#
# Required (in env or passed inline):
#   DSQL_HOST       - your DSQL endpoint hostname
#   AWS_REGION      - default: us-east-2
#   SECRET_NAME     - default: prod/prediction-market/listener-auth
#
# After running, scp /tmp/pm.env to each EC2 box.

set -euo pipefail

: "${DSQL_HOST:?DSQL_HOST must be set, e.g. export DSQL_HOST=...dsql.us-east-2.on.aws}"
AWS_REGION="${AWS_REGION:-us-east-2}"
SECRET_NAME="${SECRET_NAME:-prod/prediction-market/listener-auth}"

if [[ -z "${SECRET_ARN:-}" ]]; then
  echo "looking up SECRET_ARN for $SECRET_NAME ..." >&2
  SECRET_ARN="$(aws secretsmanager describe-secret \
    --secret-id "$SECRET_NAME" --region "$AWS_REGION" \
    --query ARN --output text)"
fi

if [[ -z "$SECRET_ARN" || "$SECRET_ARN" == "None" ]]; then
  echo "ERROR: could not resolve SECRET_ARN for $SECRET_NAME in $AWS_REGION" >&2
  exit 1
fi

cat > /tmp/pm.env <<EOF
DSQL_HOST=${DSQL_HOST}
AWS_REGION=${AWS_REGION}
CLIENT_LISTENER_HOST=0.0.0.0
CLIENT_LISTENER_PORT=9000
CLIENT_LISTENER_AUTH_MODE=aws-secretsmanager
CLIENT_LISTENER_AUTH_SECRET_ARN=${SECRET_ARN}
CLIENT_LISTENER_WORKERS=2
CLIENT_LISTENER_DB_MIN_POOL=8
CLIENT_LISTENER_DB_MAX_POOL=64
CLIENT_LISTENER_METRICS_PORT=9101
CLIENT_LISTENER_ACCOUNT_SESSION_TTL_SECONDS=43200
MATCHMAKER_INSTANCE_ID=matcher-pm-worker
MATCHMAKER_INSTANCE_INDEX=0
MATCHMAKER_TOTAL_INSTANCES=1
MATCHMAKER_POLL_INTERVAL_MS=200
MATCHMAKER_MARKET_SCAN_LIMIT=50
MATCHMAKER_REQUEUE_INTERVAL_MS=5000
MATCHMAKER_METRICS_PORT=9111
EXECUTOR_INSTANCE_ID=executor-pm-worker
EXECUTOR_POLL_INTERVAL_MS=1000
EXECUTOR_MARKET_SCAN_LIMIT=50
EOF

echo "wrote /tmp/pm.env:"
cat /tmp/pm.env
