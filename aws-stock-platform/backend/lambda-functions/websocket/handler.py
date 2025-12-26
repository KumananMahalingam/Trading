import json
import boto3

dynamodb = boto3.resource('dynamodb')

def connect_handler(event, context):
    """Handle WebSocket connections"""
    return {'statusCode': 200}

def disconnect_handler(event, context):
    """Handle WebSocket disconnections"""
    return {'statusCode': 200}