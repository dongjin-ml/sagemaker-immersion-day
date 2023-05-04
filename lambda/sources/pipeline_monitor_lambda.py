import json
from pprint import pprint

def lambda_handler(event, context):
    # TODO implement
    
    pprint (event)
    print ("==")
    
    strPipelineArn = event["detail"]["pipelineArn"]
    strStepName = event["detail"]["stepName"]
    strCurrentStepStatus = event["detail"]["currentStepStatus"]
    strFailReasion = event["detail"]["failureReason"]
    strEndTime = event["detail"]["stepEndTime"]
    
    
    print (f'strPipelineArn: {strPipelineArn}')
    print (f'strStepName: {strStepName}')
    print (f'strCurrentStepStatus: {strCurrentStepStatus}')
    print (f'strFailReasion: {strFailReasion}')
    print (f'strEndTime: {strEndTime}')
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }