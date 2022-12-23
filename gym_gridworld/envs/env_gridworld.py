import os
import gym
import sys
import copy
import numpy as np

from gym import spaces

from PIL import Image as Image
import matplotlib.pyplot as plt

# 그리드 각 요소들의 색깔 지정해줌
COLORS = {0: [0.0, 0.0, 0.0], 1: [0.5, 0.5, 0.5],
          2: [0.0, 1.0, 0.0], 3: [1.0, 0.0, 0.0],
          4: [0.0, 0.0, 1.0], 5: [1.0, 0.5, 0.0] }


class GridworldEnv(gym.Env):
    num_env = 0 

    #🔥🔥🔥
    def __init__(self):
        # Action space
        #              [가만히, 위, 아래, 좌, 우]
        self.actions = [0, 1, 2, 3, 4]
        self.action_space = spaces.Discrete(5)

        # 행값은 상하
        # 열값은 좌우로 이동 시킴
        #                           가만히      위          아래       좌          우     
        # self.action_pos_dict = { 0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1] }
        self.action_pos_dict = { 0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}

        # self.action_pos_dict[1] ----> [-1, 0]
        # self.action_pos_dict[1][0] ----> -1

        # Observation space
        self.obs_shape = [9,16]#[128, 128, 3]  # Observation space shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.float32)

        # Construct grid map
        file_path = os.path.dirname(os.path.realpath(__file__))
        #os.path.realpath(__file__)   표준경로 + 이름 불러옴

        self.grid_map_path = os.path.join(file_path, 'maplemap.txt') 
        # 파일 경로에 map.txt를 붙여서 경로 만드는 거임
        # text 파일로 맵 정의했었음


        self.initial_map = self.read_grid_map(self.grid_map_path)   # 아래에 구현된 함수 있음.
        #                                                             맵 값을 np.ndarray로 불러옴.


        self.current_map = copy.deepcopy(self.initial_map)          # 복사해옴
        self.observation = self.gridmap_to_observation(self.initial_map)  # 환경 맵(np.ndarray) 그 자체를 넣어줌.
        #                                                                   함수 설명은 아래에 있음.
        self.grid_shape = self.initial_map.shape # (16, 16)

        # Agent actions
        # state는 좌표들임.
        self.start_state, self.target_state, self.Final_state = self.get_agent_states(self.initial_map)

        # 에이전트의 초기 state는 start_state로 줌.
        self.agent_state = copy.deepcopy(self.start_state)


        # Set other parameters
        GridworldEnv.num_env += 1
        self.fig_num = GridworldEnv.num_env
        self.fig = plt.figure(self.fig_num)
        plt.show(block=False)
        plt.axis('off')


    #🔥🔥🔥
    def step(self, action):

        #                [가만히, 위, 아래, 좌, 우]
        # self.actions = [0,      1,    2,  3,  4]


        # agent_state라는 값에는 맵에서의 좌표값이 들어감.

        action = int(action)

        # next_state = (x, x)
        # 행값은 상하, 열값은 좌우로 이동시킴
        #                             제자리      위          아래       좌          우
        # self.action_pos_dict = { 0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}

        # e.g. 에이전트는 처음에 [2, 2]에 있었음.
        # action이 2이라고 하면
        # next_state = (2 + self.action_pos_dict[2][0],
        #               2 + self.action_pos_dict[2][1])이 됨.
        #
        # next_state = (2 + 1,
        #               2 + 0)가 되는 거임. -----> 좌표 (3, 2)로 이동함. -----> 열은 그대로고, 한 행 아래로 이동.
        #               아래로 이동하는 action을 해주는 거임.
        # next_state = [3, 2] 출력해줌.
        next_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                      self.agent_state[1] + self.action_pos_dict[action][1])




        # Stay in place
        # 제자리 action을 하면....
        # (원래의 observation, reward=0, dond=False)를 에이전트에게 준다.
        if action == 0:
            return (self.observation, 0, False)

        # Out of bounds condition
        # 겜 맵을 벗어나려는 경우에 해당.
        # grid_shape는 [16, 16]에 해당함.
        # 에이전트가 경계를 벗어나는 동작을 하려고 하면 '제자리 action'으로 막아줌.
        if next_state[0] < 0 or next_state[0] >= self.grid_shape[0]:
            return (self.observation, 0, False)
        if next_state[1] < 0 or next_state[1] >= self.grid_shape[1]:
            return (self.observation, 0, False)



        # current_map은 위와 같이 생겼음.
        # 진행 상황에 따라 각 칸의 값은 다를 수 있음.
        # agent_state는 [2, 2]와 같은 좌표값을 말함. ----> [2, 2]라고 예를 들겠음.
        # 맵의 각 위치에는 컬러 값이 쓰여 있음.  # 0: Black; 1: Gray; 2: Green; 3: Red, 4: Blue 5: yellow
        # 0: 그냥 길, 1: 벽, 2: 가시덤불, 3: 에이전트
        # cur_color = self.current_map[2, 2] = 3이 나옴
        cur_color = self.current_map[self.agent_state[0], self.agent_state[1]]

        # 에이전트가 이동한 위치에서의 컬러값을 가져옴.
        new_color = self.current_map[next_state[0], next_state[1]]

        # 새로 이동하려는 위치의 컬러값이 1(벽)이면
        if new_color == 1:        # 지금 그대로의 값들을 내보내줌
            return (self.observation, 0, False)

        # 새로 이동하려는 위치의 컬러값이 0(그냥 길)이면... 
        elif new_color == 0:  # Black - empty

            # 그리고 현재 위치의 컬러값이 3(에이전트)이면...
            if cur_color == 3:
                # 인간이 보기 쉽도록 그리드의 색깔을 서로 바꿔줌.
                self.current_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_map[next_state[0], next_state[1]] = 3
            self.agent_state = copy.deepcopy(next_state)
            self.observation = self.gridmap_to_observation(self.current_map)
            return (self.observation, 0, False)
       
        # 새로 이동하려는 위치의 컬러값이 2(가시덤불)이면
        elif new_color == 2:

            #그리고 현재 위치의 컬러값이 3(에이전트)이면
            if cur_color ==3:
                #현재위치의 값은 길의 색으로 변경,  이동한 발판을 agent색으로 변경
                self.current_map[self.agent_state[0],self.agent_state[1]] = 0
                self.current_map[next_state[0],next_state[1]] = 3
            self.agent_state = copy.deepcopy(next_state)
            self.observation = self.gridmap_to_observation(self.current_map)
            return (self.observation, -0.1, False)

            # # 현재 위치의 색깔을 0(검정)으로 해줌.
            # self.current_map[self.agent_state[0], self.agent_state[1]] = 0
            # # 만약 위에서 왔다면



            # self.current_map[next_state[0], next_state[1]] = 2
            # '''
            # # 기본 발판의 색을 2(가시덤불)로 유지
            # # 발판을 2번 넘어감 새로 이동한 위치의 색깔을 3(agent)으로 바꿈
            # self.current_map[next_state[0], next_state[1]] = 2
            # self.current_map[next_state[0]+self.action_pos_dict[action][0], next_state[1]+self.action_pos_dict[action][1]] = 3
            # self.agent_state = copy.deepcopy(tuple(list(next_state).__add__(self.action_pos_dict[action])))
            # '''
        #새로 이동하려는 위치의 컬러값이 4(쿠키)이면    
        elif new_color == 4:
            # 현재 위치의 색깔을 0(검정)으로 해줌.
            self.current_map[self.agent_state[0], self.agent_state[1]] = 0

            self.current_map[next_state[0], next_state[1]] = 3
            self.agent_state = copy.deepcopy(next_state)

            # for i in range(len(self.target_state)):
            #     if next_state == self.target_state[i]:
            #         r+=1
            self.observation = self.gridmap_to_observation(self.current_map)
            return(self.observation,1,False)

        elif new_color ==5:
            self.current_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_map[next_state[0], next_state[1]] = 3
            self.agent_state = copy.deepcopy(next_state)            
            self.observation = self.gridmap_to_observation(self.current_map)
            


            return(self.observation, 100, True)  


                     
            

        # 위 과정을 거침에 따라 처음에 봤던 grid_map과 shape은 같지만 각 칸의 값이 달라질 거임.
        # 지금 현재의 map 상황을 에이전트에게 observation으로 제공해줄 거임.
        # self.observation = self.gridmap_to_observation(self.current_map)
        # if next_state in self.target_state:
        #     target_observation = copy.deepcopy(self.observation)
        #     return(target_observation,1,True)
        # else:
        #     return(self.observation,0,False)
        # for i in range(len(self.target_state)):
        #     if next_state == self.target_state[i]:
        #         r+=1
        #     target_observation = copy.deepcopy(self.observation)
        #     return(target_observation,r,True)
        
        
            

        # if next_state[0] == self.target_state[0] and next_state[1] == self.target_state[1]:
        #     target_observation = copy.deepcopy(self.observation)
        #     self.reset()


    #🔥🔥🔥
    # 모든 걸 initialization
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        self.current_map = copy.deepcopy(self.initial_map)
        self.observation = self.gridmap_to_observation(self.initial_map)
        return self.observation

    # 그림 보여주기
    def render(self):
        img = self.observation
        plt.clf()
        plt.imshow(img)
        self.fig.canvas.draw()
        plt.pause(0.00001)


    def read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as f: # 어떤 파일(map에 대한 txt 파일)을 여는데 f라고 지칭하겠음
            grid_map = f.readlines()  # readlines() 해주면, DataFrame한 것처럼 txt 파일 내용 전부 가져옴.




        
            # list ----> 일단 논외
            # map  ----> s를 x라는 변수로 바꿔주고 뒤에 있는   list(~~~~)에 맵핑 시켜줌.
            # list ----> 이것도 일단 논외
            # map  ----> x가 x.split(' ')로 분리됨.

            # x.split(' ') ----> 공백을 기준으로 다 분리
            # lambda y: int(y) ----> 여기에 맵핑해서 정수형으로 변환
            # 다시 괄호 바깥쪽으로 나가줌.

            # list(          map(lambda x:   list( map( lambda y: int(y),   x.split(' ') ) ),    s)        )


        grids = np.array(   list(map(lambda x:
                                  list(map(lambda y: int(y),
                                           x.split(' '))), grid_map))   )




        return grids
        # grids는 아래와 같이 생겼음.
        # 맵을 np.ndarray로 불러옴.
        # 위 map을 함수 정의 밖에 쓰면 이상한 밑줄 나와서 저렇게 썼음. 이유는 모름.




    def get_agent_states(self, initial_map):
        start_state = None
        target_state = None
        Final_state = None


        # 3은 시작점을 의미
        # np.where ----> (array([2],  dtype=int64), array([2], dtype=int64))
        # list(map( lambda x: x[0],   np.where(grids == 3)  ))  ---->  [2, 2] = start_state
        start_state = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(initial_map == 3)
        ))
        Final_state = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(Final_state == 5)
        ))

        k = np.where(initial_map == 4)
        target_state = np.append(k[0],k[1]).reshape(2,-1).T.tolist()


        if start_state == [None, None] or target_state == [None, None]:
            sys.exit('Start or target state not specified')

        # [2, 2], [11, 4]
        return start_state, target_state, Final_state



    def gridmap_to_observation(self, grid_map, obs_shape=None):

        observation = grid_map
        return observation
