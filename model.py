from random import choice
from collections import deque
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from tqdm import tqdm
import random
import os
from tetris.tetris import Environment
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

PATH = ""
# Create models folder
if not os.path.isdir(f'{PATH}models'):
    os.makedirs(f'{PATH}models')
# Create results folder
if not os.path.isdir(f'{PATH}results'):
    os.makedirs(f'{PATH}results')

TstartTime = time.time()

######################################################################################
class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)
        self._train_dir = os.path.join(self._log_write_dir, 'train')

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


# tensorboard_callback = TensorBoard(log_dir='logs')
######################################################################################
# Agent class
class DQNAgent:
    def __init__(self, name, env, conv_list, dense_list, util_list):
        self.env = env
        self.conv_list = conv_list
        self.dense_list = dense_list
        self.name = [str(name) + " | " + "".join(str(c) + "C | " for c in conv_list) + "".join(
            str(d) + "D | " for d in dense_list) + "".join(u + " | " for u in util_list)][0]

        # Main model
        self.model = self.create_model(self.conv_list, self.dense_list)

        # Target network
        self.target_model = self.create_model(self.conv_list, self.dense_list)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(name, log_dir="{}logs/{}-{}".format(PATH, name, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Creates a convolutional block given (filters) number of filters, (dropout) dropout rate,
    # (bn) a boolean variable indecating the use of BatchNormalization,
    # (pool) a boolean variable indecating the use of MaxPooling2D 
    def conv_block(self, inp, filters=64, bn=True, pool=True, dropout=0.2):
        _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPooling2D(pool_size=(2, 2))(_)
        if dropout > 0:
            _ = Dropout(0.2)(_)
        return _

    # Creates the model with the given specifications:
    def create_model(self, conv_list, dense_list):
        # Defines the input layer with shape = ENVIRONMENT_SHAPE
        input_layer = Input(shape=self.env.ENVIRONMENT_SHAPE)
        # Defines the first convolutional block:
        _ = self.conv_block(input_layer, filters=conv_list[0], bn=False, pool=False)
        # If number of convolutional layers is 2 or more, use a loop to create them.
        if len(conv_list) > 1:
            for c in conv_list[1:]:
                _ = self.conv_block(_, filters=c)
        # Flatten the output of the last convolutional layer.
        _ = Flatten()(_)

        # Creating the dense layers:
        for d in dense_list:
            _ = Dense(units=d, activation='relu')(_)
        # The output layer has 5 nodes (one node per action)
        output = Dense(units=self.env.ACTION_SPACE_SIZE,
                       activation='linear', name='output')(_)

        # Put it all together:
        model = Model(inputs=input_layer, outputs=[output])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=0.99),  # 0.001
                      loss={'output': 'mse'},
                      metrics={'output': 'accuracy'})

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step, episode):
        # Start training only if certain number of samples is already saved

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states.reshape(-1, *env.ENVIRONMENT_SHAPE))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states.reshape(-1, *env.ENVIRONMENT_SHAPE))

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(x=np.array(X).reshape(-1, *env.ENVIRONMENT_SHAPE),
                       y=np.array(y),
                       batch_size=MINIBATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        if episode % 100 == 0:
            K.clear_session()

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *env.ENVIRONMENT_SHAPE))


######################################################################################
def get_lr():
    progress = CURRENT_EPISODE / EPISODES
    delta = INITIAL_LR - FINAL_LR
    lr = INITIAL_LR - (progress * delta)
    return lr


def save_model_and_weights(agent, model_name, episode, max_reward, average_reward, min_reward):
    checkpoint_name = f"{model_name}| Eps({episode}) | max({max_reward:_>7.2f}) | avg({average_reward:_>7.2f}) | min({min_reward:_>7.2f}).model"
    agent.model.save(f'{PATH}models/{checkpoint_name}')
    best_weights = agent.model.get_weights()
    return best_weights


def to_numpy_and_flatten(xss):
    return np.array([x for xs in xss for x in xs])
######################################################################################
# ## Constants:
# RL Constants:
DISCOUNT = 0.99
INITIAL_LR = 0.01
FINAL_LR = 0.0001
LEARNING_RATE = get_lr
REPLAY_MEMORY_SIZE = 3_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)
MIN_REWARD = 1000  # For model save
SAVE_MODEL_EVERY = 1000  # Episodes
SHOW_EVERY = 20  # Episodes
EPISODES = 5000  # Number of episodes
#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW = False
INITIAL_HUMAN_ACTIONS = 0  # must be >= MINIBATCH_SIZE or ==0
######################################################################################
CURRENT_EPISODE = 0
# Models Arch :
# [{[conv_list], [dense_list], [util_list], MINIBATCH_SIZE, {EF_Settings}, {ECC_Settings}} ]

models_arch = [
    # {"conv_list": [32], "dense_list": [32, 32], "util_list": ["ECC2", "1A-5Ac"],
    #  "MINIBATCH_SIZE": 128, "best_only": False,
    #  "EF_Settings": {"EF_Enabled": False},
    #  "ECC_Settings": {"ECC_Enabled": False}},
    # {"conv_list": [32], "dense_list": [32, 32, 32], "util_list": ["ECC2", "1A-5Ac"],
    #  "MINIBATCH_SIZE": 128, "best_only": False,
    #  "EF_Settings": {"EF_Enabled": False},
    #  "ECC_Settings": {"ECC_Enabled": False}},
    # {"conv_list": [32], "dense_list": [32, 32], "util_list": ["ECC2", "1A-5Ac"],
    #  "MINIBATCH_SIZE": 128, "best_only": False,
    #  "EF_Settings": {"EF_Enabled": True, "FLUCTUATIONS": 2},
    #  "ECC_Settings": {"ECC_Enabled": True, "MAX_EPS_NO_INC": int(EPISODES * 0.2)}}
    {"conv_list": [32], "dense_list": [32, 32, 32], "util_list": ["ECC2", "1A-5Ac"],
     "MINIBATCH_SIZE": 128, "best_only": False,
     "EF_Settings": {"EF_Enabled": True, "FLUCTUATIONS": 2},
     "ECC_Settings": {"ECC_Enabled": True, "MAX_EPS_NO_INC": int(EPISODES * 0.2)}}
]

# A dataframe used to store grid search results
res = pd.DataFrame(columns=["Model Name", "Convolution Layers", "Dense Layers", "Batch Size", "ECC", "EF",
                            "Best Only", "Average Reward", "Best Average", "Epsilon 4 Best Average",
                            "Best Average On", "Max Reward", "Epsilon 4 Max Reward", "Max Reward On",
                            "Total Training Time (min)", "Time Per Episode (sec)"])
######################################################################################
# Grid Search:
for i, m in enumerate(models_arch):
    startTime = time.time()  # Used to count episode training time
    MINIBATCH_SIZE = m["MINIBATCH_SIZE"]

    # Exploration settings :
    # Epsilon Fluctuation (EF):
    EF_Enabled = m["EF_Settings"]["EF_Enabled"]  # Enable Epsilon Fluctuation
    MAX_EPSILON = 1  # Maximum epsilon value
    MIN_EPSILON = 0.001  # Minimum epsilon value
    if EF_Enabled:
        FLUCTUATIONS = m["EF_Settings"]["FLUCTUATIONS"]  # How many times epsilon will fluctuate
        FLUCTUATE_EVERY = int(EPISODES / FLUCTUATIONS)  # Episodes
        EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / FLUCTUATE_EVERY)
        epsilon = 1  # not a constant, going to be decayed
    else:
        EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / (0.8 * EPISODES))
        epsilon = 1  # not a constant, going to be decayed

    # Initialize some variables: 
    best_average = -100
    best_score = -100

    # Epsilon Conditional Constantation (ECC):
    ECC_Enabled = m["ECC_Settings"]["ECC_Enabled"]
    avg_reward_info = [
        [1, best_average, epsilon]]  # [[episode1, reward1 , epsilon1] ... [episode_n, reward_n , epsilon_n]]
    max_reward_info = [[1, best_score, epsilon]]
    if ECC_Enabled: MAX_EPS_NO_INC = m["ECC_Settings"][
        "MAX_EPS_NO_INC"]  # Maximum number of episodes without any increment in reward average
    eps_no_inc_counter = 0  # Counts episodes with no increment in reward

    # For stats
    ep_rewards = [best_average]

    env = Environment(show_game=SHOW_PREVIEW)

    agent = DQNAgent(f"M{i}", env, m["conv_list"], m["dense_list"], m["util_list"])
    MODEL_NAME = agent.name

    best_weights = [agent.model.get_weights()]

    # Uncomment these two lines if you want to show preview on your screen
    # WINDOW          = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
    # clock           = pygame.time.Clock()

    # Iterate over episodes
    episode_count = 0
    for episode in range(1, EPISODES + 1):  # tqdm(, ascii=True, unit='episode'):
        CURRENT_EPISODE = episode
        if m["best_only"]: agent.model.set_weights(best_weights[0])
        # agent.target_model.set_weights(best_weights[0])

        score_increased = False
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1
        action = None
        # Reset environment and get initial state
        current_state = to_numpy_and_flatten(env.reset())
        game_over = env.game_over

        if INITIAL_HUMAN_ACTIONS > 0 and len(agent.replay_memory) <= INITIAL_HUMAN_ACTIONS:
            while not game_over:
                print(f'recorded plays: {len(agent.replay_memory)}')
                action = env.get_action()
                quit_, new_state, reward, game_over = env.step(action, draw=True)
                new_state = to_numpy_and_flatten(new_state)
                episode_reward += reward
                if action in [0, 1, 2, 3, 4]:
                    if INITIAL_HUMAN_ACTIONS > 0 and INITIAL_HUMAN_ACTIONS < MINIBATCH_SIZE:
                        raise Exception('INITIAL_HUMAN_ACTIONS must be bigger than MINIBATCH_SIZE or 0')
                    agent.update_replay_memory((current_state, action, reward, new_state, game_over))
            if len(agent.replay_memory) >= INITIAL_HUMAN_ACTIONS:
                for i in range(100):
                    agent.train(game_over, step)

        while not game_over:
            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))

            else:
                # Get random action 
                action = choice(env.ACTION_SPACE)

            draw = False
            # Uncomment the next block if you want to show preview on your screen
            if SHOW_PREVIEW and not episode % SHOW_EVERY:
                draw = True

            quit_, new_state, reward, game_over = env.step(action, draw)
            new_state = to_numpy_and_flatten(new_state)

            # Transform new continuous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, game_over))
            if len(agent.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
                agent.train(game_over, step, episode)

            current_state = new_state
            step += 1

        if ECC_Enabled: eps_no_inc_counter += 1
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        print(f'episode {episode_count}/{EPISODES} reward:\t', episode_reward)

        if not episode % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, learning_rate=get_lr())

            # Save models, but only when avg reward is greater or equal a set value
            if not episode % SAVE_MODEL_EVERY:
                # Save Agent :
                _ = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward, min_reward)

            if average_reward > best_average:
                best_average = average_reward
                # update ECC variables:
                avg_reward_info.append([episode, best_average, epsilon])
                eps_no_inc_counter = 0
                # Save Agent :
                best_weights[0] = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward,
                                                         min_reward)

            if ECC_Enabled and eps_no_inc_counter >= MAX_EPS_NO_INC:
                epsilon = avg_reward_info[-1][2]  # Get epsilon value of the last best reward
                eps_no_inc_counter = 0

        if episode_reward > best_score:
            try:
                best_score = episode_reward
                max_reward_info.append([episode, best_score, epsilon])

                # Save Agent :
                best_weights[0] = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward,
                                                         min_reward)

            except:
                pass

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        # Epsilon Fluctuation:
        if EF_Enabled:
            if not episode % FLUCTUATE_EVERY:
                epsilon = MAX_EPSILON
        episode_count += 1

    endTime = time.time()
    total_train_time_sec = round((endTime - startTime))
    total_train_time_min = round((endTime - startTime) / 60, 2)
    time_per_episode_sec = round((total_train_time_sec) / EPISODES, 3)

    # Get Average reward:
    average_reward = round(sum(ep_rewards) / len(ep_rewards), 2)

    # Update Results DataFrames:
    res = res.append({"Model Name": MODEL_NAME, "Convolution Layers": m["conv_list"], "Dense Layers": m["dense_list"],
                      "Batch Size": m["MINIBATCH_SIZE"], "ECC": m["ECC_Settings"], "EF": m["EF_Settings"],
                      "Best Only": m["best_only"], "Average Reward": average_reward,
                      "Best Average": avg_reward_info[-1][1], "Epsilon 4 Best Average": avg_reward_info[-1][2],
                      "Best Average On": avg_reward_info[-1][0], "Max Reward": max_reward_info[-1][1],
                      "Epsilon 4 Max Reward": max_reward_info[-1][2], "Max Reward On": max_reward_info[-1][0],
                      "Total Training Time (min)": total_train_time_min, "Time Per Episode (sec)": time_per_episode_sec}
                     , ignore_index=True)
    res = res.sort_values(by='Best Average')
    avg_df = pd.DataFrame(data=avg_reward_info, columns=["Episode", "Average Reward", "Epsilon"])
    max_df = pd.DataFrame(data=max_reward_info, columns=["Episode", "Max Reward", "Epsilon"])

    # Save dataFrames
    res.to_csv(f"{PATH}results/Results.csv")
    avg_df.to_csv(f"{PATH}results/{MODEL_NAME}-Results-Avg.csv")
    max_df.to_csv(f"{PATH}results/{MODEL_NAME}-Results-Max.csv")

TendTime = time.time()
######################################################################################
print(f"Training took {round((TendTime - TstartTime) / 60)} Minutes ")
print(f"Training took {round((TendTime - TstartTime) / 3600)} Hours ")
######################################################################################