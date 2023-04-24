import os
import sys
import shutil
import argparse
import sagemaker
import subprocess
from distutils.dir_util import copy_tree
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer 
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession
    
class deploy():
    
    def __init__(self, args):
         
        self.args = args
        print (self.args)
        
        ## copy requirement.txt
        src = os.path.join(self.args.prefix_deploy, "requirements", "requirements.txt")
        dest = os.path.join(self.args.prefix_deploy, "inference")
        shutil.copy2(src, dest)
        
        print (os.listdir(dest))
        
    def _create_endpoint(self,):
        
#         print ("self.args.local_mode", self.args.local_mode)
#         if self.args.local_mode == "True":
#             from sagemaker.local import LocalSession
#             strInstanceType = "local"
#             sagemaker_session = LocalSession()
#             sagemaker_session.config = {'local': {'local_code': True}}

#         else:
#             strInstanceType = self.args.instance_type
#             sagemaker_session = sagemaker.Session()
            
            
        sagemaker_session = sagemaker.Session()    
        print (f"Endpoint-name: {self.args.endpoint_name}")
        print ("sagemaker_session", sagemaker_session) 
        print ("self.args.execution_role", self.args.execution_role)
                
        xgb_model = XGBoostModel(
            model_data=self.args.model_data,
            role=self.args.execution_role,
            source_dir=os.path.join(self.args.prefix_deploy, "inference"),
            entry_point="inference.py",
            framework_version="1.3-1",
            sagemaker_session=sagemaker_session
        )
        
        xgb_predictor = xgb_model.deploy(
            endpoint_name=self.args.endpoint_name,
            instance_type=self.args.instance_type, 
            initial_instance_count=1,
            serializer=CSVSerializer('text/csv'), ## 미적용 시 default: application/x-npy, boto3 기반 invocation시 무시
            deserializer=JSONDeserializer('application/json'), ## 미적용 시 default: application/x-npy, boto3 기반 invocation시 무시
            wait=True,
            log=True,
        )
        
    def execution(self, ):
        
        self._create_endpoint() 
        
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_deploy", type=str, default="/opt/ml/processing/")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    parser.add_argument("--instance_type", type=str, default="ml.g4dn.xlarge")    
    parser.add_argument("--model_data", type=str, default="model_data")
    parser.add_argument("--endpoint_name", type=str, default="endpoint_name")
    parser.add_argument("--execution_role", type=str, default="execution_role")
    #parser.add_argument("--local_mode", type=str, default="local_mode")
    
    args, _ = parser.parse_known_args()
           
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    dep = deploy(args)
    dep.execution()