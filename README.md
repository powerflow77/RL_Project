# RL_Project

## 전체 구성
- **게임 목표**
  - Agent가 최단 시간 내에 Target에 도착하게 만들기

- **구성 요소**
  - Ag
 
## Environment
- **Map**
  - 9x16 txt 파일 생성 후 값들을 받아 맵을 생성<br/>

![poster](./916.png)

- **State**
  - 이 Environment에서 State와 Observation은 동일한 값이다.
  - State는 Map의 전체 값 그 자체를 받는다.
  - ANN에 입력 시에는 np.ravel로 Flattening 시켜서 입력으로 넣어준다.


![poster](./gg.jpg)
