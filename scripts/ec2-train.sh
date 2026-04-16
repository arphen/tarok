#!/usr/bin/env bash
# scripts/ec2-train.sh — Launch a g5.2xlarge spot instance, deploy tarok, and start training.
#
# Prerequisites (run once on your Mac):
#   brew install awscli
#   aws configure                  # set region, access key, secret
#   aws s3 mb s3://YOUR_BUCKET     # create checkpoint bucket
#
# Usage:
#   ./scripts/ec2-train.sh \
#       --key     my-keypair \
#       --sg      sg-0123456789abcdef0 \
#       --bucket  s3://my-tarok-checkpoints \
#       --model   checkpoints/Petra_Novak/iter_090.pt \
#       --config  ec2-g5-1h
#
# The script will:
#   1. Request a spot instance (g5.2xlarge, Deep Learning Base AMI)
#   2. Wait for SSH to be ready
#   3. rsync your code (excluding .venv, build artifacts, node_modules)
#   4. Bootstrap: install uv, Rust, build the Rust engine
#   5. Start training inside a tmux session with live logging to train.log
#   6. Set up a cron job to sync checkpoints to S3 every 5 minutes
#
# Monitor (from your Mac, after the script finishes):
#   ssh -t ubuntu@$IP "tmux attach -t train"          # live output
#   ssh ubuntu@$IP "tail -f ~/tarok/train.log"         # scrolling log
#   aws s3 sync $BUCKET/checkpoints/ checkpoints/      # pull latest checkpoints
#   ./scripts/ec2-train.sh --terminate $INSTANCE_ID    # terminate when done
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${TAROK_ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# ─── Defaults ────────────────────────────────────────────────────────────────
KEY_NAME="${EC2_KEY:-${TAROK_EC2_KEY:-}}"
SG_ID="${EC2_SG:-${TAROK_EC2_SG:-}}"
S3_BUCKET="${EC2_BUCKET:-${TAROK_EC2_BUCKET:-}}"
MODEL_PATH="${MODEL:-${TAROK_EC2_MODEL:-}}"
CONFIG="${CONFIG:-${TAROK_EC2_CONFIG:-ec2-g5-1h}}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_TYPE="${EC2_INSTANCE_TYPE:-${TAROK_EC2_INSTANCE_TYPE:-g5.2xlarge}}"
SPOT_PRICE="${EC2_SPOT_PRICE:-${TAROK_EC2_SPOT_PRICE:-1.20}}"        # max bid in USD/hr
INSTANCE_PROFILE_NAME="${EC2_INSTANCE_PROFILE:-${TAROK_EC2_INSTANCE_PROFILE:-ec2-s3-tarok}}"

AMI_ID="${EC2_AMI:-${TAROK_EC2_AMI:-auto}}"

SUBNET_ID="${EC2_SUBNET:-${TAROK_EC2_SUBNET:-}}"   # leave empty to use default VPC
KEY_PATH="${EC2_KEY_PATH:-${TAROK_EC2_KEY_PATH:-$HOME/.ssh/${KEY_NAME}.pem}}"

# ─── Arg parsing ─────────────────────────────────────────────────────────────
TERMINATE_ID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --key)      KEY_NAME="$2";      shift 2 ;;
        --sg)       SG_ID="$2";         shift 2 ;;
        --bucket)   S3_BUCKET="$2";     shift 2 ;;
        --model)    MODEL_PATH="$2";    shift 2 ;;
        --config)   CONFIG="$2";        shift 2 ;;
        --region)   REGION="$2";        shift 2 ;;
        --ami)      AMI_ID="$2";        shift 2 ;;
        --subnet)   SUBNET_ID="$2";     shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --spot-price) SPOT_PRICE="$2";  shift 2 ;;
        --instance-profile) INSTANCE_PROFILE_NAME="$2"; shift 2 ;;
        --terminate) TERMINATE_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Terminate shortcut ───────────────────────────────────────────────────────
if [[ -n "$TERMINATE_ID" ]]; then
    echo "==> Terminating instance $TERMINATE_ID …"
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$TERMINATE_ID"
    echo "Done."
    exit 0
fi

# ─── Validate required args ───────────────────────────────────────────────────
for var in KEY_NAME SG_ID S3_BUCKET MODEL_PATH; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --${var//_/-,,} is required"
        exit 1
    fi
done

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: model file not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$KEY_PATH" ]]; then
    echo "ERROR: SSH key file not found: $KEY_PATH"
    exit 1
fi

if [[ "$AMI_ID" == "auto" ]]; then
    echo "==> Resolving latest Deep Learning GPU AMI for region $REGION…"
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
        --query "sort_by(Images,&CreationDate)[-1].ImageId" \
        --output text)
    if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
        echo "ERROR: could not auto-resolve a Deep Learning GPU AMI in region $REGION"
        echo "Set TAROK_EC2_AMI or pass --ami explicitly."
        exit 1
    fi
fi

echo "==> Project root : $PROJECT_ROOT"
echo "==> Config       : $CONFIG"
echo "==> Model        : $MODEL_PATH"
echo "==> S3 bucket    : $S3_BUCKET"
echo "==> Region       : $REGION"
echo "==> AMI          : $AMI_ID"
echo "==> Instance type: $INSTANCE_TYPE (spot, max \$$SPOT_PRICE/hr)"
echo ""

# ─── 1. Upload starting checkpoint to S3 ────────────────────────────────────
echo "==> Uploading starting checkpoint to S3…"
MODEL_S3_KEY="tarok-run/start/$(basename "$MODEL_PATH")"
aws s3 cp "$MODEL_PATH" "$S3_BUCKET/$MODEL_S3_KEY" --region "$REGION"

# ─── 2. Request spot instance ────────────────────────────────────────────────
echo "==> Requesting spot instance…"

SUBNET_SPEC=""
if [[ -n "$SUBNET_ID" ]]; then
    SUBNET_SPEC="\"SubnetId\": \"$SUBNET_ID\","
fi

LAUNCH_SPEC=$(cat <<EOF
{
    "ImageId": "$AMI_ID",
    "InstanceType": "$INSTANCE_TYPE",
    "KeyName": "$KEY_NAME",
    "SecurityGroupIds": ["$SG_ID"],
    $SUBNET_SPEC
    "BlockDeviceMappings": [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": { "VolumeSize": 60, "VolumeType": "gp3", "DeleteOnTermination": true }
        }
    ],
    "IamInstanceProfile": {
        "Name": "$INSTANCE_PROFILE_NAME"
    }
}
EOF
)

# Note: the IAM instance profile "$INSTANCE_PROFILE_NAME" needs s3:GetObject/PutObject/ListBucket on your bucket.
# Create it once:
#   aws iam create-role --role-name $INSTANCE_PROFILE_NAME \
#       --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
#   aws iam attach-role-policy --role-name $INSTANCE_PROFILE_NAME \
#       --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
#   aws iam create-instance-profile --instance-profile-name $INSTANCE_PROFILE_NAME
#   aws iam add-role-to-instance-profile --instance-profile-name $INSTANCE_PROFILE_NAME --role-name $INSTANCE_PROFILE_NAME

REQUEST_ID=$(aws ec2 request-spot-instances \
    --region "$REGION" \
    --spot-price "$SPOT_PRICE" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "$LAUNCH_SPEC" \
    --query "SpotInstanceRequests[0].SpotInstanceRequestId" \
    --output text)

echo "    Spot request: $REQUEST_ID"

# ─── 3. Wait for instance ID ─────────────────────────────────────────────────
echo "==> Waiting for spot request to be fulfilled…"
INSTANCE_ID=""
for i in $(seq 1 30); do
    INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
        --region "$REGION" \
        --spot-instance-request-ids "$REQUEST_ID" \
        --query "SpotInstanceRequests[0].InstanceId" \
        --output text 2>/dev/null || true)
    if [[ -n "$INSTANCE_ID" && "$INSTANCE_ID" != "None" ]]; then
        break
    fi
    echo "    ($i/30) still waiting…"
    sleep 10
done

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
    echo "ERROR: spot request not fulfilled after 5 minutes. Check AWS console."
    exit 1
fi

echo "    Instance ID: $INSTANCE_ID"
echo ""
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  To terminate when done:                                    │"
echo "  │  ./scripts/ec2-train.sh --terminate $INSTANCE_ID           │"
echo "  └─────────────────────────────────────────────────────────────┘"
echo ""

# ─── 4. Wait for running + get IP ────────────────────────────────────────────
echo "==> Waiting for instance to enter 'running' state…"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo "    Public IP: $PUBLIC_IP"

# ─── 5. Wait for SSH ─────────────────────────────────────────────────────────
echo "==> Waiting for SSH to be ready…"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes"
for i in $(seq 1 24); do
    if ssh $SSH_OPTS -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" "true" 2>/dev/null; then
        break
    fi
    echo "    ($i/24) SSH not ready yet…"
    sleep 10
done

SSH="ssh $SSH_OPTS -i $KEY_PATH ubuntu@$PUBLIC_IP"

# ─── 6. rsync code ───────────────────────────────────────────────────────────
echo "==> Syncing code to instance…"
rsync -az --progress \
    -e "ssh $SSH_OPTS -i $KEY_PATH" \
    --exclude='.git' \
    --exclude='backend/.venv' \
    --exclude='backend/target' \
    --exclude='engine-rs/target' \
    --exclude='frontend/node_modules' \
    --exclude='frontend/dist' \
    --exclude='checkpoints/' \
    --exclude='**/__pycache__' \
    --exclude='*.pyc' \
    --exclude='test-results/' \
    "$PROJECT_ROOT/" \
    "ubuntu@$PUBLIC_IP:~/tarok/"

# ─── 7. Bootstrap + start training ───────────────────────────────────────────
echo "==> Running bootstrap + starting training…"

REMOTE_MODEL="~/tarok/checkpoints/start/$(basename "$MODEL_PATH")"

$SSH bash <<ENDSSH
set -euo pipefail

echo "── Installing uv ──────────────────────────────────────────────"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"

echo "── Installing Rust ────────────────────────────────────────────"
if ! command -v rustc &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
fi
source "\$HOME/.cargo/env"

echo "── Installing Python deps ─────────────────────────────────────"
cd ~/tarok/backend
uv sync --default-index https://pypi.org/simple --extra dev --quiet

echo "── Building Rust engine ───────────────────────────────────────"
LIBTORCH_USE_PYTORCH=1 uv run --default-index https://pypi.org/simple \
    maturin develop --release --manifest-path ../engine-rs/Cargo.toml

echo "── Downloading starting checkpoint ────────────────────────────"
mkdir -p ~/tarok/checkpoints/start
aws s3 cp $S3_BUCKET/$MODEL_S3_KEY $REMOTE_MODEL

echo "── Setting up S3 checkpoint sync (every 5 min) ─────────────────"
(crontab -l 2>/dev/null || true; echo "*/5 * * * * aws s3 sync ~/tarok/checkpoints/ $S3_BUCKET/checkpoints/ --region $REGION --quiet") | crontab -

echo "── Starting training in tmux ──────────────────────────────────"
tmux new-session -d -s train -x 220 -y 50
tmux send-keys -t train "
    cd ~/tarok && \
    export PATH=\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && \
    source \$HOME/.cargo/env && \
    cd backend && \
    PYTHONPATH=src:../model/src uv run --default-index https://pypi.org/simple \
        python ../training-lab/train_and_evaluate.py \
        --config ../training-lab/configs/${CONFIG}.yaml \
        --checkpoint $REMOTE_MODEL \
        2>&1 | tee ~/tarok/train.log
" Enter

echo ""
echo "✅  Training started. Instance: $INSTANCE_ID  IP: $PUBLIC_IP"
ENDSSH

# ─── 8. Print monitoring instructions ────────────────────────────────────────
cat <<EOF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Instance : $INSTANCE_ID
  IP       : $PUBLIC_IP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Watch live output (attach to tmux):
        ssh -i $KEY_PATH ubuntu@$PUBLIC_IP -t "tmux attach -t train"

  Tail the scrolling log:
        ssh -i $KEY_PATH ubuntu@$PUBLIC_IP "tail -f ~/tarok/train.log"

  Pull checkpoints to your Mac:
    aws s3 sync $S3_BUCKET/checkpoints/ checkpoints/ec2-run/

  Terminate instance when done:
    ./scripts/ec2-train.sh --terminate $INSTANCE_ID

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
