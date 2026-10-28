#!/usr/bin/env bash
# Start, stop or check the benchmark instance, or open a shell on it.
# usage: scripts/cloud/ec2.sh start|stop|status|ssh
set -euo pipefail

INSTANCE=${NANA_EC2_INSTANCE:?set NANA_EC2_INSTANCE to the instance id}
REGION=${NANA_EC2_REGION:?set NANA_EC2_REGION, e.g. us-east-2}
HOST=${NANA_EC2_HOST:?set NANA_EC2_HOST to user@address}
KEY=${NANA_EC2_KEY:?set NANA_EC2_KEY to the ssh key file}

aws_ec2() { aws ec2 --region "$REGION" "$@"; }

state() {
    aws_ec2 describe-instances --instance-ids "$INSTANCE" \
        --query 'Reservations[0].Instances[0].State.Name' --output text
}

case "${1:-status}" in
    start)
        aws_ec2 start-instances --instance-ids "$INSTANCE" > /dev/null
        echo "starting $INSTANCE..."
        aws_ec2 wait instance-running --instance-ids "$INSTANCE"
        until ssh -i "$KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$HOST" true 2> /dev/null; do
            sleep 3
        done
        echo "up: ssh -i $KEY $HOST"
        ;;
    stop)
        aws_ec2 stop-instances --instance-ids "$INSTANCE" > /dev/null
        echo "stopping $INSTANCE"
        ;;
    status)
        echo "$INSTANCE: $(state)"
        ;;
    ssh)
        exec ssh -i "$KEY" "$HOST"
        ;;
    *)
        echo "usage: $0 start|stop|status|ssh" >&2
        exit 2
        ;;
esac
