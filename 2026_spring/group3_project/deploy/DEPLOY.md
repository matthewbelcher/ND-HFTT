# Manual EC2 Deployment Runbook

This runbook deploys the prediction-market services across **2 listener
boxes + 1 worker box**, fronted by an AWS Network Load Balancer, talking
to an existing Aurora DSQL cluster. It mirrors the topology validated in
`test/results/two-server-lb-notes.txt` and accommodates the perf changes
landed in the same change set (CTE submit, bucketed cash, multi-process
listener with `SO_REUSEPORT`, work-queue matcher, `/metrics` endpoint).

## 1. Topology

```
                +-----------------+
                | Trading clients |
                +--------+--------+
                         | TCP 80
                  +------v------+
                  |  AWS NLB    |
                  +------+------+
                         | TCP 9000 (cross-zone)
        +----------------+----------------+
        |                                 |
+-------v-------+                 +-------v-------+
| listener-1    |                 | listener-2    |
| (workers=N)   |                 | (workers=N)   |
+-------+-------+                 +-------+-------+
        |                                 |
        +----------------+----------------+
                         | asyncpg / TLS 5432
                  +------v------+
                  |  Aurora DSQL |
                  +------^------+
                         | asyncpg / TLS 5432
                  +------+------+
                  |  worker     |
                  | matcher +   |
                  | executor    |
                  +-------------+
```

You can scale by adding listener boxes (NLB target group), running more
matcher processes (set `MATCHMAKER_TOTAL_INSTANCES`/`MATCHMAKER_INSTANCE_INDEX`
so each owns a stable slice of markets), or splitting matcher and
executor onto separate boxes.

## 2. AWS prerequisites (one-time)

1. **VPC** with at least 2 public subnets in different AZs (or 1 AZ for a
   minimal-cost dev deploy).
2. **Aurora DSQL cluster**. The repo currently points at
   `4btv7qq43k4ztw3tjubmbyf3su.dsql.us-east-2.on.aws`
   (`scripts/component_lib.sh`); set `DSQL_HOST` to your own.
3. **IAM role `prediction-market-ec2`** to attach to every EC2 instance,
   with policies:
   - `dsql:DbConnectAdmin` on your cluster ARN
   - `secretsmanager:GetSecretValue` on the listener auth secret ARN
   - `cloudwatch:PutLogEvents`, `logs:CreateLogStream`, `logs:CreateLogGroup`
4. **Secrets Manager** secret e.g.
   `prod/prediction-market/listener-auth` with body
   `{"token":"<random-256-bit-token>"}`.
5. **Security groups**:
   - `sg-listener` - inbound TCP 9000 from `sg-nlb`; inbound TCP 9101+
     (metrics) from `sg-monitoring` only; outbound 5432 to DSQL.
   - `sg-nlb` - inbound TCP 80 from your client CIDRs.
   - `sg-worker` - outbound 5432 to DSQL only.
6. **Apply the schema** from your laptop:
   ```bash
   DSQL_HOST=... AWS_REGION=us-east-2 ./deploy/apply_schema.sh
   ```

## 3. Launch the EC2 instances

| Tag             | AMI            | Size       | SG          | Notes                   |
|-----------------|----------------|------------|-------------|-------------------------|
| `role=listener` | Ubuntu 22.04   | c6i.large  | sg-listener | NLB target              |
| `role=listener` | Ubuntu 22.04   | c6i.large  | sg-listener | NLB target              |
| `role=worker`   | Ubuntu 22.04   | t3.small   | sg-worker   | matcher + executor only |

Attach the `prediction-market-ec2` instance profile to all three.

## 4. Per-instance bootstrap

SSH into the box and run:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip awscli git postgresql-client jq
sudo useradd -m -s /bin/bash market || true
sudo install -d -o market -g market /var/log/prediction-market

sudo -u market -i bash <<'EOF'
cd $HOME
git clone https://github.com/<your-org>/Distributed-Prediction-Market-Server.git
cd Distributed-Prediction-Market-Server
python3 -m venv .venv-pm
.venv-pm/bin/pip install --upgrade pip
.venv-pm/bin/pip install \
  -r client/client-listener/requirements.txt \
  -r matchmaker/requirements.txt \
  -r executor/requirements.txt \
  -r frontend/requirements.txt
EOF
```

## 5. Install the environment file

```bash
sudo install -m 0640 -o root -g market \
  /home/market/Distributed-Prediction-Market-Server/deploy/prediction-market.env.example \
  /etc/prediction-market.env
sudo $EDITOR /etc/prediction-market.env   # fill in DSQL host, secret ARN, etc.
```

## 6. Install systemd units

```bash
SERVICE_DIR=/home/market/Distributed-Prediction-Market-Server/deploy/systemd
sudo cp $SERVICE_DIR/listener.service     /etc/systemd/system/
sudo cp $SERVICE_DIR/matchmaker.service   /etc/systemd/system/
sudo cp $SERVICE_DIR/executor.service     /etc/systemd/system/
sudo systemctl daemon-reload
```

### On listener boxes
```bash
sudo systemctl enable --now listener.service
sudo systemctl status listener.service
```

### On the worker box
```bash
sudo systemctl enable --now matchmaker.service executor.service
sudo systemctl status matchmaker.service executor.service
```

## 7. NLB and target group

1. Create a target group `tg-listener` (target type: instance, protocol
   TCP, port 9000, health check TCP/9000).
2. Register both listener instance IDs in the target group.
3. Create a Network Load Balancer (`internet-facing` or `internal`) with
   listener `TCP:80` forwarding to `tg-listener`. Enable cross-zone load
   balancing.
4. Validate connectivity from your laptop:
   ```bash
   nc -vz <nlb-dns> 80
   echo '{"action":"ping"}' | nc -q1 <nlb-dns> 80
   ```

## 8. End-to-end smoke

From your laptop, against the NLB:

```bash
TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id prod/prediction-market/listener-auth \
  --query 'SecretString' --output text | jq -r .token)

# Apply schema if not done already
DSQL_HOST=... AWS_REGION=us-east-2 ./deploy/apply_schema.sh

# Run the RPC features test through the NLB
python test/rpc_features_test.py \
  --listener-host <nlb-dns> --listener-port 80 \
  --auth-token "$TOKEN"
```

## 9. Re-run the scaling benchmark

Launch a fourth EC2 instance in the same VPC as the load generator
(e.g. `c6i.large`, sg with outbound TCP 80 to the NLB) and run:

```bash
DSQL_HOST=... AWS_REGION=us-east-2 \
LISTENER_HOST=<nlb-dns> LISTENER_PORT=80 \
./test/run_submit_benchmark_ec2.sh
```

Compare the resulting `test/results/*.csv` with the pre-change baseline
(`test/results/two-server-lb.csv`). With the perf work in this change
set, p50 single-client latency should drop from ~25 ms toward 6-10 ms,
and the saturation plateau should move well past the previous
~200 ops/s point.

## 10. Observability

Every listener worker exposes Prometheus text format on
`CLIENT_LISTENER_METRICS_PORT + worker_index` (default base port `9101`).
The matchmaker exposes on `MATCHMAKER_METRICS_PORT` (default `9111`).
Useful counters and histograms:

- `listener_submit_total` / `listener_submit_errors_total` /
  `listener_submit_idempotent_total`
- `listener_submit_latency_ms_bucket{le=...}` / `_sum` / `_count`
- `listener_occ_retries_total`
- `listener_pool_size` / `listener_pool_idle`
- `matchmaker_fills_total` / `matchmaker_fill_skipped_total`
- `matchmaker_fill_latency_ms` (histogram)
- `matchmaker_claimed_markets_total` / `matchmaker_requeued_markets_total`
- `matchmaker_queue_max_lag_ms` / `matchmaker_queue_depth_sample`

Scrape these from a Prometheus or CloudWatch agent on a separate
monitoring host inside the same VPC.

## 11. Rotation of the DSQL auth token

`run_forever.sh` already handles token rotation by restarting the
component every `COMPONENT_MAX_LIFETIME_SECONDS` (default 600s) after
re-running `aws dsql generate-db-connect-admin-auth-token`. No extra
steps required - just keep `systemd` Restart=always and the supervisor
loop in place.
