#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-2}"

if [[ -z "${SG_LISTENER:-}" ]]; then
  SG_LISTENER="$(aws ec2 describe-security-groups --region "$AWS_REGION" \
    --filters Name=group-name,Values=pm-listener \
    --query 'SecurityGroups[0].GroupId' --output text)"
fi

if [[ -z "${SG_WORKER:-}" ]]; then
  SG_WORKER="$(aws ec2 describe-security-groups --region "$AWS_REGION" \
    --filters Name=group-name,Values=pm-worker \
    --query 'SecurityGroups[0].GroupId' --output text)"
fi

ip=$(curl -s https://checkip.amazonaws.com)
echo "whitelisting $ip"
for sg in "$SG_LISTENER" "$SG_WORKER"; do
  aws ec2 authorize-security-group-ingress --group-id $sg \
    --protocol tcp --port 22 --cidr "${ip}/32" --region "$AWS_REGION" 2>&1 \
    | grep -v 'InvalidPermission.Duplicate' || true
done