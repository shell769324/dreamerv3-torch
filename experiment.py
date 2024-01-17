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


def call_agent(session_id="test"):
    client = boto3.client('bedrock-agent-runtime')
    while True:
        input_text = input("Enter next prompt:")
        should_end = False
        if input_text.lower().strip(' ') in ["quit", "q"]:
            should_end = True
        response = client.invoke_agent(
            agentId="8SQUSFV9XU",
            agentAliasId="DWEBWA9AED",
            sessionId=session_id,
            inputText=input_text,
            endSession=should_end
        )
        event_stream = response['completion']
        output = ""
        for event in event_stream:
            if 'chunk' in event:
                output += event["chunk"]["bytes"].decode("utf8")
        print(output)
        if should_end:
            return


call_agent()
