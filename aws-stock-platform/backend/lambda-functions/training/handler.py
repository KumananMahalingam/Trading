import json
import boto3
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/python')
sys.path.insert(0, '/var/task/utils')

# Import YOUR actual functions
from integrated_pipeline import (
    fetch_stock_data,
    calculate_technical_indicators,
    generate_alphas_with_groq,
    prepare_dataframe_for_alpha
)
from lstm_predictor import train_ensemble
import torch

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    """
    Trains a new LSTM model for a ticker
    Uses YOUR actual training pipeline
    """
    try:
        ticker = event.get('ticker', '').upper()

        if not ticker:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Ticker required'})
            }

        print(f"🏋️ Starting training for {ticker}")

        # ====================================================================
        # STEP 1: FETCH DATA (using YOUR function)
        # ====================================================================

        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years

        stock_df = fetch_stock_data(
            ticker,
            start_date.isoformat() + 'Z',
            end_date.isoformat() + 'Z'
        )

        if stock_df.empty or len(stock_df) < 100:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Insufficient data',
                    'ticker': ticker,
                    'data_points': len(stock_df)
                })
            }

        print(f"✓ Fetched {len(stock_df)} days of data")

        # ====================================================================
        # STEP 2: CALCULATE TECHNICAL INDICATORS (YOUR function)
        # ====================================================================

        stock_df = calculate_technical_indicators(stock_df)
        print(f"✓ Calculated technical indicators")

        # ====================================================================
        # STEP 3: GENERATE ALPHAS (YOUR function)
        # ====================================================================

        # For now, use simple alphas (Groq requires API key setup)
        # TODO: Uncomment this once Groq API key is in SSM
        # alpha_text = generate_alphas_with_groq(
        #     ticker,
        #     ticker,  # company name
        #     stock_df,
        #     []  # no related companies
        # )

        # Temporary: use simple alphas
        from integrated_pipeline import generate_simple_alphas
        alpha_text = generate_simple_alphas(ticker)

        print(f"✓ Generated alphas:\n{alpha_text[:200]}...")

        # ====================================================================
        # STEP 4: PREPARE COMPREHENSIVE DATAFRAME (YOUR function)
        # ====================================================================

        comprehensive_df = prepare_dataframe_for_alpha(
            ticker,
            stock_df,
            {},  # daily_sentiments (empty for basic training)
            [],  # related_companies
            None,  # alternative_data
            None   # economic_data
        )

        if comprehensive_df is None or comprehensive_df.empty:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Failed to prepare data',
                    'ticker': ticker
                })
            }

        print(f"✓ Prepared {len(comprehensive_df)} rows with {len(comprehensive_df.columns)} features")

        # ====================================================================
        # STEP 5: TRAIN ENSEMBLE MODEL (YOUR function)
        # ====================================================================

        print(f"🧠 Training ensemble model...")

        ensemble, metrics, scalers = train_ensemble(
            ticker=ticker,
            company_name=ticker,
            comprehensive_df=comprehensive_df,
            alpha_text=alpha_text,
            n_models=2,  # 2 models in ensemble
            window_size=30,
            device='cpu'  # Lambda uses CPU
        )

        if ensemble is None:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Training failed',
                    'ticker': ticker
                })
            }

        print(f"✓ Training complete!")
        print(f"  Directional Accuracy: {metrics['Directional Accuracy']:.1f}%")
        print(f"  RMSE: {metrics['RMSE']:.6f}")

        # ====================================================================
        # STEP 6: UPLOAD MODEL TO S3
        # ====================================================================

        model_path = f'/tmp/{ticker}_ensemble.pth'
        torch.save(ensemble.state_dict(), model_path)

        s3.upload_file(
            model_path,
            os.environ['MODELS_BUCKET'],
            f'models/{ticker}_ensemble.pth'
        )

        print(f"✓ Model uploaded to S3")

        # ====================================================================
        # STEP 7: SAVE METADATA
        # ====================================================================

        metadata = {
            'ticker': ticker,
            'trained_at': datetime.now().isoformat(),
            'metrics': {
                'directional_accuracy': float(metrics['Directional Accuracy']),
                'rmse': float(metrics['RMSE']),
                'sharpe_ratio': float(metrics.get('Sharpe Ratio', 0)),
                'up_precision': float(metrics.get('Up Precision', 0)),
                'down_precision': float(metrics.get('Down Precision', 0))
            },
            'num_alphas': 5,
            'data_points': len(comprehensive_df),
            'window_size': 30,
            'n_models': 2
        }

        # Save to S3
        s3.put_object(
            Bucket=os.environ['MODELS_BUCKET'],
            Key=f'metadata/{ticker}_metadata.json',
            Body=json.dumps(metadata),
            ContentType='application/json'
        )

        # Cache alphas in DynamoDB
        try:
            alphas_table = dynamodb.Table(os.environ.get('ALPHAS_TABLE', 'stock-alphas'))
            alphas_table.put_item(Item={
                'ticker': ticker,
                'alphas': alpha_text,
                'generated_at': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"Warning: Could not cache alphas: {e}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'ticker': ticker,
                'metrics': metadata['metrics'],
                'model_location': f's3://{os.environ["MODELS_BUCKET"]}/models/{ticker}_ensemble.pth',
                'message': f'Model trained successfully for {ticker}'
            })
        }

    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'ticker': ticker if 'ticker' in locals() else 'unknown'
            })
        }