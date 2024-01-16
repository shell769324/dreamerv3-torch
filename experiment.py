import os
import sys

import boto3

module_path = ".."
sys.path.append(os.path.abspath(module_path))

client = boto3.client('bedrock-agent')
info = client.create_agent(agentName="tester",
                           agentResourceRoleArn="arn:aws:iam::605719699353:role/rookery",
                           foundationModel="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2:1:200k",
                           instruction="You will play a survival game with the goal to get diamond.",
                           idleSessionTTLinSeconds=1800)
print(info['agent'])
