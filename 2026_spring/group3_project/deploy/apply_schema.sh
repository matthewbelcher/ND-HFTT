#!/usr/bin/env bash
# Apply tables-create-dsql.sql against the configured DSQL cluster.
# Run this once on initial deploy and after every schema change.
#
# Required env:
#   DSQL_HOST   - cluster hostname (e.g. xxx.dsql.us-east-2.on.aws)
#   AWS_REGION  - region of the DSQL cluster (e.g. us-east-2)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DSQL_HOST:?DSQL_HOST is required}"
REGION="${AWS_REGION:?AWS_REGION is required}"

if [[ -x /usr/local/bin/aws ]]; then
  AWS_CLI=/usr/local/bin/aws
elif command -v aws >/dev/null 2>&1; then
  AWS_CLI="$(command -v aws)"
else
  echo "aws CLI not found; install aws cli and retry." >&2
  exit 1
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "psql not found; install postgresql-client and retry." >&2
  exit 1
fi

TOKEN="$($AWS_CLI dsql generate-db-connect-admin-auth-token \
  --hostname "$HOST" --region "$REGION" --output text)"
ENC_TOKEN="$(RAW_TOKEN="$TOKEN" python3 -c \
  'import os, urllib.parse; print(urllib.parse.quote(os.environ["RAW_TOKEN"], safe=""))')"
DB_DSN="postgresql://admin:${ENC_TOKEN}@${HOST}:5432/postgres?sslmode=require"

echo "applying schema from $ROOT_DIR/tables-create-dsql.sql"
psql "$DB_DSN" -v ON_ERROR_STOP=1 -f "$ROOT_DIR/tables-create-dsql.sql"
echo "schema applied"
