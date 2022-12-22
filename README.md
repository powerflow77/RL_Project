# RL_Project

## 전체 구성
- **게임 목표**
  - Agent가 최단 시간 내에 Target에 도착하게 만들기

- **구성 요소**
  - Ag
 
## Environment
- **Map**
  - 9x16 txt 파일 생성 후 값들을 받아 맵을 생성.<br/>값의 의미는 다음과 같음.<br/>
    - 0: 그냥 땅(Agent가 자유롭게 이동할 수 있는 pixel)<br/>
    - 1: 벽(해당 pixel을 밟을 수도, 무시하고 지나갈 수도 없음.)<br/>
    - 2: 가시덤불(해당 pixel을 밟을 수는 있지만 즉시 음의 Reward를 받음.)<br/>
    - 3: Agent<br/>
    - 4: Cookie(해당 pixel에서는 양의 Reward를 받지만, 매 timestep에 -1의 Reward를 받고 있기 때문에<br/>Cookie를 다 챙기면 오히려 전체 Reward를 낮추게 된다.)
    - 5: 도착 지점
    - map.txt --> ![poster](./916.PNG)
    

- **State**
  - 이 Environment에서 State와 Observation은 동일한 값이다.<br/>
  - State는 Map의 전체 값 그 자체를 받는다.<br/>
  - ANN에 입력 시에는 np.ravel로 Flattening 시켜서 입력으로 넣어준다.<br/>


![poster](./gg.jpg)
