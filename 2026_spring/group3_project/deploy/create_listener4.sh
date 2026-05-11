cat > /tmp/launch_listener4.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

aws ec2 run-instances \
  --image-id "$UBUNTU_AMI" \
  --instance-type c6i.large \
  --key-name pm-key \
  --security-group-ids "$SG_LISTENER" \
  --subnet-id "$SUBNET_B" \
  --associate-public-ip-address \
  --iam-instance-profile Name=prediction-market-ec2 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pm-listener-4},{Key=role,Value=listener}]' \
  --region "$AWS_REGION"
EOF

chmod +x /tmp/launch_listener4.sh
/tmp/launch_listener4.sh