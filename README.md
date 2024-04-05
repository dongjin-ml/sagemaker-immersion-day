# sagemaker-immersion-day

# 1. 개요
## 1.1 워크샵 목적

세이지 메이커의 기본적인 개요를 이해하고, 데이터 전처리, 모델 훈련, 및 모델 배포, 세이지 메이커 파이프라인을 이해합니다. 그리고 이를 기반으로 실제 핸즈온 코드 실습을 진행 합니다.


## 1.2. 사전 지식
* Python Coding 을 1년 이상 해본 분
* ML 백그라운드가 있는 분
* 데이터 전처리, 모델 훈련, 모델 배포 경험이 있는 분


# 2. 랩 가이드
## 2.1. 랩 환경
- [Amazon SageMaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) 에서, 아래의 2.2, 2.3 이 모두 테스트 되었습니다. "2.2 처음 시작시 추천 실행 노트북" 만을 위해서는 SageMaker Studio 에서 사용이 가능합니다.

## 2.2 처음 시작시 추천 실행 노트북  
* 0.setup.ipynb 
* 1.training.ipynb ( Cloud Mode Not Local Mode)
* 2.evaluation.ipynb ( Cloud Mode Not Local Mode)
* 3.deploy.ipynb  ( Cloud Mode Not Local Mode)
* 4-1.pipeline.ipynb 
* 6.clean-up.ipynb 


## 2.3 로컬 모드 학습을 위한 추천 실행 노트뷱
* 1.training.ipynb ( Local Mode)
* 2.evaluation.ipynb (  Local Mode)
* 4-2.pipeline-local-mode.ipynb
    
## 심화 과정  
* 7.advanced_byom_xgboost_regression_sagemaker_deploy.ipynb
  * Bring Your Own Model 기반 SageMaker Endpoint배포 실습
  * 이 프로젝트에서는 일반적인 ML 학습 방식인 Scikit-learn과 XGBoost를 사용하여 로컬에서 학습한 모델을 이용하여, SageMaker의 Endpoint API 추론 기능을 활용하여 모델을 배포합니다.

    
    
# A. 참고 정보
- [Local Mode in Python SDK for Classic Notebook](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode)
- [Local mode support in Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html)
    


