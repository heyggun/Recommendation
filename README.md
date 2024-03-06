## Recommendation System 
추천시스템을 공부하면서 코드로 구현해보고 있는 repository 입니다.

- 대표적인 추천 시스템 서적인 Recommender Systems: The Textbook 정리 : 
- 협업 필터링과 콘텐츠 기반 필터링의 차이에 관련해 정리 : https://kimgeonhee.notion.site/9b598c7db74249c38e70069912bce8a5?pvs=4


## [1] Cotent-based Filtering (콘텐츠 기반 필터링)
1-1. Content-based filtering
- 책 추천 알고리즘

## [2] Collaborative Filtering (협업 필터링) 

2-1. model_based_Collaborative Filtering 
- 남성 유저 - 여성 유저의 interaction history data로 유저를 추천하는 알고리즘
- Matrix Factorization을 ALS(Alternating Least Squares) 알고리즘을 통해 계산

2-2. Neural Collaborative Filtering (MF + MLP) user-item recommender 
- 딥러닝 + 협업 필터링을 사용한 user에게 적적한 음악을 추천하는 알고리즘
- recommendation system에서 쓰이던 Matrix Factorization 의 한계점( 기존의 MF가 linear하고 fixed 해서, user-item의 complexity한 관계를 표현하지 못함)을 보완해서 MLP를 앙상블함

2-3. Neural Graph Collaborative Filtering
- Neural + graph를 활용한 영화 추천 알고리즘

## [3] Reciprocal Recommendation System (상호 추천 시스템) user-user recommender
- 일명 상호 추천 시스템 논문 정리 : https://www.notion.so/kimgeonhee/reciprocal-recommendation-system-89bfa66c913e496083a5415a0c6924da
- 기존의 user-item 이 아니라 user-user 즉 human-human 추천에서 양방향 선호를 반영할 수 있는 알고리즘

## [4] Image-based Recommendation
 - feature로 이미지를 활용해서 추천해보면 어떨까 하면서 테스트 

## [5] review_based_recommendation/data
 - 의류 판매 상품 리뷰 분석을 통한 상품 추천 여부 예측 (자연어 리뷰를 가지고 추천해보기 위해서 E-Commerce 데이터를 활용)
