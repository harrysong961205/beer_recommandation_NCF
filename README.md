# NCF를 이용한 맥주 추천 시스템
맥주 데이터를 바탕으로 NCF를 이용하여 맥주 추천 시스템을 구현하였다.

데이터 출처 : https://www.kaggle.com/datasets/rdoume/beerreviews
개발 환경 : Colab
프레임 워크 : Pytorch

## 1. 전처리
사용 데이터 : "beer_id"(맥주번호),"review_profilename"(리뷰 작성자), "beer_style"(맥주 스타일)
1-1. 사용 데이터를 이용하여 beer_id 와 review_profilename 사이에 interaction이 존재하는 항목을 바탕으로 matrix 생성
1-2. beer_id와 beer_style를 dict형태로 저장

## 2. 모델링
MF 와 MLP로 구성 후
- MF에서는 Matrix 그대로 삽입
- MLP에서는 Matrix와 beer_id 별 beer_style를 추가하여 layer embedding 함

## 3. UIdataset 생성
- Neg ratio(interaction이 존재하지 않는 항목의 비율)을 100배로 설정하고 random으로 추출

## 4. 평가 metrics 
- recall k : model에서 추출된 상위 k개의 결과 중, 실제로 유저와 interaction 이 있는 비율.
- ndcg k : model에서 추춘된 상위 k개의 결과 중, 실제 유저와 intercation 이 있는 비율, 순서에 가중치 존재.
- score : recall * 0.75 + ndcg * 0.25

## 5.train
- batch size : 64
- embedding dim : 256
- layer dim : 256
- drop out : 0.10
- lr : 0.0025
- epoch : 25

