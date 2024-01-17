import os
import sys

import boto3

module_path = ".."
sys.path.append(os.path.abspath(module_path))

sts_client = boto3.client("sts")
assume_role_result = sts_client.assume_role(
    RoleArn="arn:aws:iam::605719699353:role/AmazonBedrockExecutionRoleForAgents_rookery",
    RoleSessionName="bedrock_test",
    DurationSeconds=900)

client = boto3.client('bedrock-agent',
                      aws_access_key_id=assume_role_result['Credentials']['AccessKeyId'],
                      aws_secret_access_key=assume_role_result['Credentials']['SecretAccessKey'],
                      aws_session_token=assume_role_result['Credentials']['SessionToken'])
info = client.create_agent(agentName="tester",
                           agentResourceRoleArn="arn:aws:iam::605719699353:role/AmazonBedrockExecutionRoleForAgents_rookery",
                           foundationModel="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2:1:200k",
                           instruction="You will play a survival game with the goal to get diamond.",
                           idleSessionTTLInSeconds=1800)
print(info['agent'])
