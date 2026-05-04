#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu 22.04 EC2 box for the prediction market services.
#
# Idempotent: safe to re-run after pulling new code or changing the role.
#
# Usage (run on the EC2 box as a sudoer, e.g. ubuntu@...):
#   sudo ROLE=listener REPO_URL=https://github.com/USER/Distributed-Prediction-Market-Server.git \
#        BRANCH=andrew2 ./bootstrap_box.sh
#
#   ROLE: listener | worker | both | none   (default: none)
#         - listener installs+enables listener.service
#         - worker   installs+enables matchmaker.service and executor.service
#         - both     installs+enables all three
#         - none     just clones the code and sets up the venv
#
#   REPO_URL: HTTPS or SSH url to clone. If you need a PAT, embed it in the URL:
#             https://<user>:<pat>@github.com/<user>/<repo>.git
#
#   BRANCH:   git branch to check out (default: andrew2)

set -euo pipefail

ROLE="${ROLE:-none}"
REPO_URL="${REPO_URL:?REPO_URL must be set, e.g. https://github.com/USER/REPO.git}"
BRANCH="${BRANCH:-andrew2}"
APP_USER="${APP_USER:-market}"
APP_HOME="/home/${APP_USER}"
APP_DIR="${APP_HOME}/Distributed-Prediction-Market-Server"
LOG_DIR="/var/log/prediction-market"

if [[ $EUID -ne 0 ]]; then
  echo "this script needs sudo; re-run as: sudo ROLE=$ROLE ... $0" >&2
  exit 1
fi

case "$ROLE" in
  listener|worker|both|none) ;;
  *) echo "ROLE must be one of: listener | worker | both | none (got: $ROLE)" >&2; exit 1 ;;
esac

echo "==> installing apt packages"
apt-get update -y
apt-get install -y python3-venv python3-pip git postgresql-client jq unzip curl

# AWS CLI v2 is required for `aws dsql ...`. apt's awscli is v1 and lacks it,
# so we always install v2 from the official zip even if v1 is already there.
if ! /usr/local/bin/aws --version 2>/dev/null | grep -q '^aws-cli/2'; then
  echo "==> installing AWS CLI v2"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -qo /tmp/awscliv2.zip -d /tmp/
  /tmp/aws/install --update --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli
fi
/usr/local/bin/aws --version

echo "==> ensuring user '${APP_USER}' exists"
if ! id "${APP_USER}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "${APP_USER}"
fi
install -d -o "${APP_USER}" -g "${APP_USER}" "${LOG_DIR}"

echo "==> cloning / updating repo at ${APP_DIR} (branch=${BRANCH})"
sudo -u "${APP_USER}" -i bash <<EOF
set -euo pipefail
if [[ ! -d "${APP_DIR}/.git" ]]; then
  git clone -b "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
else
  cd "${APP_DIR}"
  git fetch origin
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
fi
EOF

echo "==> installing python deps into venv"
sudo -u "${APP_USER}" -i bash <<EOF
set -euo pipefail
cd "${APP_DIR}"
if [[ ! -x .venv-pm/bin/python ]]; then
  python3 -m venv .venv-pm
fi
.venv-pm/bin/pip install --upgrade pip
.venv-pm/bin/pip install \
  -r client/client-listener/requirements.txt \
  -r matchmaker/requirements.txt \
  -r executor/requirements.txt \
  -r frontend/requirements.txt
EOF

if [[ ! -f /etc/prediction-market.env ]]; then
  echo "==> seeding /etc/prediction-market.env from example (you MUST edit this)"
  install -m 0640 -o root -g "${APP_USER}" \
    "${APP_DIR}/deploy/prediction-market.env.example" \
    /etc/prediction-market.env
  echo "    edit it now: sudo nano /etc/prediction-market.env"
fi

install_unit() {
  local unit="$1"
  cp "${APP_DIR}/deploy/systemd/${unit}" "/etc/systemd/system/${unit}"
}

case "$ROLE" in
  listener)
    echo "==> installing listener.service"
    install_unit listener.service
    systemctl daemon-reload
    systemctl enable --now listener.service
    systemctl status --no-pager listener.service || true
    ;;
  worker)
    echo "==> installing matchmaker.service + executor.service"
    install_unit matchmaker.service
    install_unit executor.service
    systemctl daemon-reload
    systemctl enable --now matchmaker.service executor.service
    systemctl status --no-pager matchmaker.service || true
    systemctl status --no-pager executor.service   || true
    ;;
  both)
    echo "==> installing listener + matchmaker + executor"
    install_unit listener.service
    install_unit matchmaker.service
    install_unit executor.service
    systemctl daemon-reload
    systemctl enable --now listener.service matchmaker.service executor.service
    ;;
  none)
    echo "==> ROLE=none: code is in place, no systemd units enabled"
    ;;
esac

echo "==> done; remember to populate /etc/prediction-market.env if it still has placeholders"
