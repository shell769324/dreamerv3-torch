import json
import os
import sys

import boto3
import botocore

module_path = ".."
sys.path.append(os.path.abspath(module_path))

client = boto3.client('bedrock')
print(client.list_foundation_models())
