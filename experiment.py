import os
import sys

import boto3

module_path = ".."
sys.path.append(os.path.abspath(module_path))


def create_bedrock_client():
    sts_client = boto3.client("sts")
    assume_role_result = sts_client.assume_role(
        RoleArn="arn:aws:iam::605719699353:role/AmazonBedrockExecutionRoleForAgents_rookery",
        RoleSessionName="bedrock_test",
        DurationSeconds=900)
    return boto3.client('bedrock-agent',
                        aws_access_key_id=assume_role_result['Credentials']['AccessKeyId'],
                        aws_secret_access_key=assume_role_result['Credentials']['SecretAccessKey'],
                        aws_session_token=assume_role_result['Credentials']['SessionToken'])


def create_agent(client):
    return client.create_agent(
        agentName="tester",
        agentResourceRoleArn="arn:aws:iam::605719699353:role/AmazonBedrockExecutionRoleForAgents_rookery",
        foundationModel="anthropic.claude-v2:1",
        instruction="You will play a survival game with the goal to get diamond.",
        idleSessionTTLInSeconds=1800)


def alias_agent(client, alias):
    info = client.create_agent_alias(
        agentId="8SQUSFV9XU",
        agentAliasName=alias
    )
    print(info)


def call_agent(alias="tester1"):
    client = boto3.client('bedrock-agent-runtime')
    client.invoke_agent(
        agentId="8SQUSFV9XU",

    )


alias_agent(create_bedrock_client(), "tester1")
