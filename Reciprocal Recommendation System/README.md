
## Reciprocal Recommendation System
Akiva Kleinerman, Ariel Rosenfeld(2021)의 Supporting users in fnding successful matches in reciprocal recommender systems 논문 구현

### Paper

| NO | Paper | Link |
| --- | --- | --- |
| 1 |  Providing Explanations for Recommendations in Reciprocal Environments | [Link](https://arxiv.org/pdf/1807.01227.pdf)

---------

### Description

- Reciprocal Recommendation System(상호추천시스템; RRS)은 'user간의 interaction'을 기반으로 'user-user'를 추천하고, 추천이 된 이유를 같이 제공하는 'Explainable recommender system'입니다.
- 해당 추천알고리즘은 추천 받는 유저와 추천 되는 유저간의 'successful interaction(성공적인 상호작용)' 을 기반으로 한 추천 알고리즘으로,
  추천 받을 유저와 추천 후보 집단간의 상호 관심도를 기반으로 한 협업필터링과 추천 받을 유저의 긍정 응답 예측모델을 통해 추천 유저가 제공되고
  추천된 유저와 추천받을 유저의 feature간 correlation을 기반으로 'explanation'이 가능한 상관관계 기반 추천 설명을 제공합니다.

<algorithm>

(1) Reciprocal Collaborative Filtering Recommendation)
 - 추천 받을 유저 x와 추천 후보 집단 유저 y 간의 상호 관심도를 연산함
(2) Predicting replies of Recommended users
 - 추천 받을 유저 x와 추천 후보 집단 유저 y에 대한 implicit data(로그인 수, 정보업데이트 수, 긍정 응답률) 과 profile data(나이, 키, 체중, 연봉 등)을 통해
   상호 유저간 긍정적 응답률을 예측함 (adaboost 모델 사용)
(3) Optimal Weighted Score Recommendations Schema
  - 유저 특성에 따라 선호와 응답률을 중시하는 비중이 다르므로, 추천 받을 유저 x의 과거 유저들과의 positive data를 기반으로 가중치를 찾아 (1)의 상호관심도와 (2)의 긍정응답률
    에 곱해 최종 점수를 도출 
(4) Corrleation-based Reciprocal Explanation
  - 추천 받을 유저 x와 추천된 후보 y 와의 특성값과의 상관관계를 측정하여 상관관계가 가장 큰 특성값 상위 k개로 추천의 이유를 설명함


<전통적 추천시스템과 차이>
- user에게 item을 추천하는 것은 user의 선호(preference) 만 고려하면 되지만 user-user의 경우에는 추천 '받는' user와 추천 '된' user의 선호(preference)에 따라 추천의 성공률이 달려있음
- 전통적인 people to people 추천은 추천을 받아들이는 유저에게만 결정되거나, 추천 받는 유저와 추천 되는 유저의 동일 가중치인 semi-personalized(반개인화) 추천에 가까움
- 해당 추천알고리즘은 추천 받는 유저와 추천 되는 유저간의 'successful interaction(성공적인 상호작용)' 을 기반으로 한 추천 알고리즘


<한계점>
- 실제 서비스에 적용해봤을 때 유의미한 상관관계 feature가 나오지 않아 설명이 어려운 부분이 있었음
