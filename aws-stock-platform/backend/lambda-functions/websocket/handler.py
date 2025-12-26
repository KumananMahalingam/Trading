import json
import boto3
import os
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
connections_table = dynamodb.Table(os.environ['CONNECTIONS_TABLE'])

def connect_handler(event, context):
    """Handle WebSocket $connect route"""
    connection_id = event['requestContext']['connectionId']

    print(f"New connection: {connection_id}")

    try:
        # Store connection
        connections_table.put_item(
            Item={
                'connectionId': connection_id,
                'timestamp': datetime.now().isoformat(),
                'ttl': int(datetime.now().timestamp()) + 86400  # 24 hours
            }
        )

        return {'statusCode': 200}

    except Exception as e:
        print(f"Error storing connection: {e}")
        return {'statusCode': 500}


def disconnect_handler(event, context):
    """Handle WebSocket $disconnect route"""
    connection_id = event['requestContext']['connectionId']

    print(f"Disconnection: {connection_id}")

    try:
        connections_table.delete_item(Key={'connectionId': connection_id})
        return {'statusCode': 200}

    except Exception as e:
        print(f"Error removing connection: {e}")
        return {'statusCode': 500}


def default_handler(event, context):
    """Handle WebSocket default route (messages)"""
    connection_id = event['requestContext']['connectionId']

    try:
        body = json.loads(event.get('body', '{}'))
        action = body.get('action', 'unknown')

        print(f"Message from {connection_id}: {action}")

        # Echo back for now
        api_gateway = boto3.client(
            'apigatewaymanagementapi',
            endpoint_url=f"https://{event['requestContext']['domainName']}/{event['requestContext']['stage']}"
        )

        response = {
            'action': 'message',
            'data': 'Message received',
            'timestamp': datetime.now().isoformat()
        }

        api_gateway.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(response)
        )

        return {'statusCode': 200}

    except Exception as e:
        print(f"Error handling message: {e}")
        return {'statusCode': 500}


def broadcast_prediction(ticker, prediction_data):
    """
    Broadcast new prediction to all connected clients
    Called by other Lambda functions
    """
    try:
        # Get all connections
        response = connections_table.scan()
        connections = response.get('Items', [])

        if not connections:
            print("No active connections")
            return

        message = {
            'action': 'prediction_update',
            'ticker': ticker,
            'data': prediction_data,
            'timestamp': datetime.now().isoformat()
        }

        api_gateway = boto3.client('apigatewaymanagementapi')

        for connection in connections:
            try:
                api_gateway.post_to_connection(
                    ConnectionId=connection['connectionId'],
                    Data=json.dumps(message)
                )
            except Exception as e:
                print(f"Failed to send to {connection['connectionId']}: {e}")
                # Connection is stale, remove it
                connections_table.delete_item(
                    Key={'connectionId': connection['connectionId']}
                )

        print(f"Broadcast to {len(connections)} connections")

    except Exception as e:
        print(f"Error broadcasting: {e}")