import copy
import pprint
import pygame
import random

REWARD_NONE = 0
REWARD_1ROW_CLEARED = 150
REWARD_2ROW_CLEARED = 300
REWARD_3ROW_CLEARED = 600
REWARD_4ROW_CLEARED = 1200
REWARD_LOW_PLACEMENT = 20
PUNISHMENT_GAME_OVER = -1000
PUNISHMENT_LEAVING_GAP = -70

colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


class Figure:
    x = 0
    y = 0

    figures = [
        [[4, 5, 6, 7], [1, 5, 9, 13]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[4, 5, 6, 10], [1, 5, 9, 8], [0, 4, 5, 6], [1, 2, 5, 9]],
        [[3, 5, 6, 7], [2, 6, 10, 11], [5, 6, 7, 9], [1, 2, 6, 10], ],
        [[1, 5, 6, 9], [4, 5, 6, 9], [1, 4, 5, 9], [1, 4, 5, 6]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = self.type  # random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])


class Tetris:
    level = 1
    score = 0
    state = "start"
    field = []
    height = 0
    width = 0
    x = 100
    y = 60
    zoom = 20
    figure = None

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.field = []
        self.score = 0
        self.state = "start"
        self.stamina = 1
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(-1)
            self.field.append(new_line)

    def new_figure(self):
        self.figure = Figure(3, 0)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > -1:
                        intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            blanks = 0
            for j in range(self.width):
                if self.field[i][j] == -1:
                    blanks += 1
            if blanks == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += lines ** 2
        if lines >= 4:
            return REWARD_4ROW_CLEARED
        elif lines == 3:
            return REWARD_3ROW_CLEARED
        elif lines == 2:
            return REWARD_2ROW_CLEARED
        elif lines == 1:
            return REWARD_1ROW_CLEARED
        return REWARD_NONE

    def freeze(self):
        leftmost_point = 100
        rightmost_point = 0
        highest_point = 100
        lowest_point = 0
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
                    if i + self.figure.y < highest_point:
                        highest_point = i + self.figure.y
                    if i + self.figure.y > lowest_point:
                        lowest_point = i + self.figure.y
                    if j + self.figure.x < leftmost_point:
                        leftmost_point = j + self.figure.x
                    if j + self.figure.x > rightmost_point:
                        rightmost_point = j + self.figure.x
        reward = self.break_lines()
        reward += lowest_point
        gap = False
        for i in range(highest_point, lowest_point+1):
            for j in range(leftmost_point, rightmost_point+1):
                if i + 1 < self.height:
                    if self.field[i+1][j] == -1:
                        reward += PUNISHMENT_LEAVING_GAP
                        gap = True
                        break
            if gap: break

        self.new_figure()
        if self.intersects():
            self.state = "gameover"
            reward += PUNISHMENT_GAME_OVER
        return reward

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        reward = self.freeze()
        if reward > 0:
            reward = reward * self.stamina
            self.stamina = 1
        return reward

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            reward = self.freeze()
            if reward > 0:
                reward = reward * self.stamina
                self.stamina = 1
            return reward
        return REWARD_NONE

    def go_side(self, dx):
        self.stamina *= 0.9
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        self.stamina *= 0.9
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation


class Environment:
    HEIGHT = 20  # Height of the field
    WIDTH = 10  # Width of the field and the walls

    ENVIRONMENT_SHAPE = (HEIGHT, WIDTH, 1)
    ACTION_SPACE = [0, 1, 2, 3, 4]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)

    def __init__(self, show_game=True):
        self.show_game = show_game
        # Initialize the game engine

        # Define some colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)

        size = (400, 500)

        if self.show_game:
            pygame.init()
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Tetris")
            self.clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self.quit = False
        self.fps = 10
        self.game = Tetris(20, 10)
        self.counter = 0
        self.state = copy.deepcopy(self.game.field)
        self.game_over = False

    def reset(self):
        self.quit = False
        if self.show_game:
            self.clock = pygame.time.Clock()
        self.counter = 0
        self.state = copy.deepcopy(self.game.field)
        self.game_over = False
        self.game.__init__(20, 10)
        return self.state

    def draw(self):
        self.screen.fill(self.WHITE)

        for i in range(self.game.height):
            for j in range(self.game.width):
                pygame.draw.rect(self.screen, self.GRAY,
                                 [self.game.x + self.game.zoom * j, self.game.y + self.game.zoom * i, self.game.zoom,
                                  self.game.zoom], 1)
                if self.game.field[i][j] > -1:
                    pygame.draw.rect(self.screen, colors[self.game.field[i][j]],
                                     [self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1,
                                      self.game.zoom - 2, self.game.zoom - 1])

        if self.game.figure is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in self.game.figure.image():
                        pygame.draw.rect(self.screen, colors[self.game.figure.color],
                                         [self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                                          self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                                          self.game.zoom - 2, self.game.zoom - 2])

        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(self.game.score), True, self.BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press Enter", True, (255, 215, 0))

        self.screen.blit(text, [0, 0])
        if self.game.state == "gameover":
            self.screen.blit(text_game_over, [20, 200])
            self.screen.blit(text_game_over1, [15, 265])

        pygame.display.flip()

    def get_state(self):
        state = copy.deepcopy(self.game.field)
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.game.figure.image():
                    state[i + self.game.figure.y][j + self.game.figure.x] = self.game.figure.color
        return state

    def step(self, action, draw=True):
        # pygame.event.pump()
        if self.game.figure is None:
            self.game.new_figure()
        self.counter += 1
        if self.counter > 100000:
            self.counter = 0

        reward = REWARD_NONE
        if action == -1:
            self.quit = True
        elif action == 0:
            self.game.rotate()
        elif action == 1:
            self.game.go_side(-1)
        elif action == 2:
            self.game.go_side(1)
        elif action == 3:
            reward = self.game.go_space()
        elif action == 4:
            reward = self.game.go_down()
        elif action == 5:
            self.reset()

        if self.counter % (self.fps // self.game.level // 2) == 0:  # or self.pressing_down
            if self.game.state == "start":
                reward += self.game.go_down()

        if self.game.state == 'gameover':
            self.game_over = True

        if self.show_game and draw:
            self.draw()

        self.state = self.get_state()

        return self.quit, self.state, reward, self.game_over

    def get_action(self):
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                action = -1  # 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # 'rotate'
                # if event.key == pygame.K_DOWN:
                #     self.pressing_down = True
                if event.key == pygame.K_LEFT:
                    action = 1  # 'left'
                if event.key == pygame.K_RIGHT:
                    action = 2  # 'right'
                if event.key == pygame.K_SPACE:
                    action = 3  # 'space'
                if event.key == pygame.K_DOWN:
                    action = 4  # 'down'
                if event.key == pygame.K_RETURN:
                    action = 5  # 'reset'
        self.clock.tick(self.fps)
        return action


if __name__ == '__main__':
    env = Environment(True)
    mode = 'human'
    quit_ = False
    state = None
    while not quit_:
        action = env.get_action()
        quit_, state, reward, game_over = env.step(action)

        # if reward != 0:
        #     print(reward)
            # pprint.pprint(state)

    pygame.quit()
