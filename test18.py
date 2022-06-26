"""
my codes
note: -- 目前存在会在某两点之间来回跳步的问题，导致程序陷入死循环
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

positions = {0: (0, 0), 1: (3, 0), 2: (11, 0), 3: (0, 1), 4: (3, 1), 5: (9, 1), 6: (11, 1),
             7: (14, 1), 8: (0, 3), 9: (3, 3), 10: (8, 3), 11: (9, 3), 12: (10, 3), 13: (2, 4), 14: (3, 5),
             15: (9, 5), 16: (14, 5), 17: (17, 5), 18: (0, 6), 19: (2, 6), 20: (3, 6), 21: (3, 7), 22: (9, 7),
             23: (14, 7), 24: (17, 7), 25: (0, 8), 26: (3, 8), 27: (3, 9), 28: (14, 9), 29: (14, 10),
             30: (17, 10), 31: (2, 11), 32: (3, 11), 33: (10, 11), 34: (14, 11), 35: (14, 12), 36: (17, 12),
             37: (2, 12), 38: (3, 12), 39: (10, 12)}

node0 = {}
node1 = {ACTION_UP: 4}
node2 = {ACTION_UP: 6}
node3 = {ACTION_RIGHT: 4}
node4 = {ACTION_UP: 9, ACTION_DOWN: 1, ACTION_LEFT: 3, ACTION_RIGHT: 5}
node5 = {ACTION_UP: 11, ACTION_LEFT: 4, ACTION_RIGHT: 6}
node6 = {ACTION_DOWN: 2, ACTION_LEFT: 5, ACTION_RIGHT: 7}
node7 = {ACTION_UP: 16, ACTION_LEFT: 6}
node8 = {ACTION_RIGHT: 9}
node9 = {ACTION_UP: 14, ACTION_DOWN: 4, ACTION_LEFT: 8}
node10 = {ACTION_RIGHT: 11}
node11 = {ACTION_UP: 15, ACTION_DOWN: 5, ACTION_LEFT: 10, ACTION_RIGHT: 12}
node12 = {ACTION_LEFT: 11}
node13 = {ACTION_UP: 19}
node14 = {ACTION_UP: 20, ACTION_DOWN: 9, ACTION_RIGHT: 15}
node15 = {ACTION_UP: 22, ACTION_DOWN: 11, ACTION_LEFT: 14, ACTION_RIGHT: 16}
node16 = {ACTION_UP: 23, ACTION_DOWN: 7, ACTION_LEFT: 15, ACTION_RIGHT: 17}
node17 = {ACTION_LEFT: 16}
node18 = {ACTION_RIGHT: 19}
node19 = {ACTION_DOWN: 13, ACTION_LEFT: 18, ACTION_RIGHT: 20}
node20 = {ACTION_UP: 21, ACTION_DOWN: 14, ACTION_LEFT: 19}
node21 = {ACTION_UP: 26, ACTION_DOWN: 20, ACTION_RIGHT: 22}
node22 = {ACTION_DOWN: 15, ACTION_LEFT: 21, ACTION_RIGHT: 23}
node23 = {ACTION_UP: 28, ACTION_DOWN: 16, ACTION_LEFT: 22, ACTION_RIGHT: 24}
node24 = {ACTION_LEFT: 23}
node25 = {ACTION_RIGHT: 26}
node26 = {ACTION_UP: 27, ACTION_DOWN: 21, ACTION_LEFT: 25}
node27 = {ACTION_UP: 32, ACTION_DOWN: 26, ACTION_RIGHT: 28}
node28 = {ACTION_UP: 29, ACTION_DOWN: 23, ACTION_LEFT: 27}
node29 = {ACTION_UP: 34, ACTION_DOWN: 28, ACTION_RIGHT: 30}
node30 = {ACTION_LEFT: 29}
node31 = {ACTION_RIGHT: 32}
node32 = {ACTION_UP: 38, ACTION_DOWN: 27, ACTION_LEFT: 31, ACTION_RIGHT: 33}
node33 = {ACTION_UP: 39,  ACTION_LEFT: 32, ACTION_RIGHT: 34}
node34 = {ACTION_UP: 35, ACTION_DOWN: 29, ACTION_LEFT: 33}
node35 = {ACTION_DOWN: 34, ACTION_RIGHT: 36}
node36 = {ACTION_LEFT: 35}
node37 = {ACTION_RIGHT: 38}
node38 = {ACTION_DOWN: 32, ACTION_LEFT: 37, ACTION_RIGHT: 39}
node39 = {ACTION_DOWN: 33, ACTION_LEFT: 38}

node = [node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13,
        node14, node15, node16, node17, node18, node19, node20, node21, node22, node23, node24, node25, node26,
        node27, node28, node29, node30, node31, node32, node33, node34, node35, node36, node37, node38, node39]

# (node16[ACTION_LEFT])
# print(node)


# probability for exploration   探索
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

#
gamma = 0.8
# step size
ALPHA = 0.5

START = 1    # 1号节点 -- 起点
GOAL = 39  # 29号节点 -- 终点

# steps_done = 0


#  step函数 -- 返回 新状态 奖励
def step(state, action):
    x0, y0 = positions[state]
    next_state = node[state][action]
    x1, y1 = positions[next_state]
    x_goal, y_goal = positions[GOAL]

    # if next_state == GOAL:
    #     reward = 10
    #
    # else:
    #
    #     reward = -1

    reward1 = -(abs(x_goal - x1) + abs(y_goal - y1))   # -- 收敛要快点 --曼哈顿距离

    reward2 = -math.sqrt(sum([(x_goal - x1)**2, (y_goal - y1)**2]))  # -- 收敛要慢点 -- 欧式距离

    reward0 = -(abs(x0 - x1) + abs(y0 - y1))

    reward = reward2

    return next_state, reward

# 选取动作--暂时不用函数表达
# def select_action(state):


num_episodes = 50


# 主函数
def main():

    action_spaces = len(actions)
    state_spaces = len(node)

    q_value = np.zeros((state_spaces, action_spaces))

    episode_list = []
    train_rewards_list = []
    for episode in range(num_episodes):  # 总的训练次数
        episode_list.append(episode+1)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)

        state = START
        while state != GOAL:  # 训练一次，直到结束
            # 创建随机数
            sample_num = random.random()

            state_ac_available = []
            node_state = node[state]

            for i in node_state.keys():
                state_ac_available.append(i)

            if sample_num > eps_threshold:  # 利用
                state_dict = {}
                for j in state_ac_available:
                    state_dict[j] = q_value[state, j]
                    action = [key_ for key_, value_ in state_dict.items() if value_ == max(list(state_dict.values()))]

            else:  # 探索
                action = random.sample(state_ac_available, 1)

            next_state, reward = step(state, action[0])
            # loss = next_state - state

            ##
            next_state_ac_available = []
            node_next_state = node[next_state]

            for i in node_next_state.keys():
                next_state_ac_available.append(i)

            # next_state_dict = {}
            # for j in next_state_ac_available:

            next_state_q_value = [q_value[next_state, j] for j in next_state_ac_available]
            next_state_max_q_value = max(next_state_q_value)

            # next_state_dict[j] = q_value[next_state, j]

            # next_state_max_value = [value_ for value_ in next_state_dict.values() if value_ == max(list(next_state_dict.values()))]

            # Q-learning algorithm !!
            q_value[state, action[0]] += ALPHA * (reward + gamma * next_state_max_q_value - q_value[state, action[0]])

            # loss = next_state - state

            # Update state
            state = next_state

        ####
        state = START
        train_rewards = 0
        for s in range(50):
        # while state != GOAL:
            train_state_ac_available = []
            train_node_state = node[state]

            for u in train_node_state.keys():
                train_state_ac_available.append(u)

            train_state_dict = {}

            for j in train_state_ac_available:
                train_state_dict[j] = q_value[state, j]
                action = [key_ for key_, value_ in train_state_dict.items() if value_ == max(list(train_state_dict.values()))]

            new_state, reward = step(state, action[0])
            train_rewards += reward
            if new_state == GOAL:
                break
            state = new_state

        train_rewards_list.append(train_rewards)

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.plot(episode_list, train_rewards_list)

    plt.xlabel("Episodes")  # 横坐标意义
    plt.ylabel("Rewards")   # 纵坐标意义
    # plt.savefig("./test18.png")
    plt.show()

    print("Training completed over {} episodes".format(num_episodes))
    print("***开始部署***")

    # 部署算法
    state = START
    rewards = 0
    paths = []

    while state != GOAL:
    # for s in range(100):
        paths.append(state)
        deploy_state_ac_available = []
        deploy_node_state = node[state]

        for i in deploy_node_state.keys():
            deploy_state_ac_available.append(i)

        state_dict = {}

        for j in deploy_state_ac_available:
            state_dict[j] = q_value[state, j]
            action = [key_ for key_, value_ in state_dict.items() if value_ == max(list(state_dict.values()))]

        new_state, reward = step(state, action[0])
        rewards = rewards + reward

        # if new_state == GOAL:
        #     break
        state = new_state

    paths.append(GOAL)
    print("路径paths为：{}".format(paths))
    print("总的奖励为: {}".format(rewards))
    paths_length = 0
    for v in range(len(paths)-1):

        cal_x0, cal_y0 = positions[paths[v]]
        cal_x1, cal_y1 = positions[paths[v+1]]

        paths_length = paths_length + abs(cal_x0 - cal_x1) + abs(cal_y0 - cal_y1)

    print("路径长度为：{}".format(paths_length))


if __name__ == "__main__":
    main()