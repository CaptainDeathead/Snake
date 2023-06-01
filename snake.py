import random
import numpy as np
import neat
import os
import matplotlib.pyplot as plt
import pygame
import time
from multiprocessing import Pool

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# use gpu for cupy
#np.cuda.Device(0).use()

class SnakeGame:
    def __init__(self, width=600, height=600, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.last_apple_time = time.time()  # Track the time when the last apple was picked up
        self.apple_timeout = 3.0  # Timeout duration in seconds
        self.apple_penalty = 0  # Penalty for not picking up apples

        # make the snake 2 segments long
        self.snake.append((self.snake[0][0], self.snake[0][1] + 1))

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

    def generate_food(self):
        while True:
            food = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if food not in self.snake:
                return food

    def get_state(self):
        return np.array([abs(self.snake[0][0] - self.food[0]), abs(self.snake[0][1] - self.food[1])]).flatten()

    def get_state_size(self):
        return self.grid_width * self.grid_height

    def get_action_size(self):
        return 4  # Up, Down, Left, Right

    def step(self, action):
        dx, dy = 0, 0
        if action == 0:  # Up
            dy = -1
        elif action == 1:  # Down
            dy = 1
        elif action == 2:  # Left
            dx = -1
        elif action == 3:  # Right
            dx = 1

        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_width
            or new_head[1] < 0
            or new_head[1] >= self.grid_height
            or new_head in self.snake
        ):
            self.game_over = True
            reward = -1
            self.score -= 1
            return self.get_state(), reward, self.game_over

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self.generate_food()
            reward = 1
            self.last_apple_time = time.time()
        else:
            self.snake.pop()
            reward = 0

        if time.time() - self.last_apple_time > self.apple_timeout:
            self.game_over = True
            reward = -1
            self.score -= 1
            self.apple_penalty += 1

        self.play_game()

        return self.get_state(), reward, self.game_over

    def reset(self):
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        return self.get_state()

    def draw_game(self):
        self.screen.fill((0, 0, 0))  # Clear the screen

        for segment in self.snake:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (segment[0] * self.grid_size, segment[1] * self.grid_size, self.grid_size, self.grid_size),
            )

        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (self.food[0] * self.grid_size, self.food[1] * self.grid_size, self.grid_size, self.grid_size),
        )

        pygame.display.flip()

    def play_game(self):
        self.clock.tick(1000)
        self.draw_game()

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = SnakeGame()
    state = game.get_state()
    prev_action = None

    while not game.game_over:
        action = net.activate(state.flatten())

        if prev_action is not None:
            if (action.index(max(action)) + prev_action) % 2 == 0:
                snake_direction = prev_action
            else:
                snake_direction = action.index(max(action))
        else:
            snake_direction = action.index(max(action))

        prev_action = snake_direction

        if snake_direction == 0:
            game.step(2)
        elif snake_direction == 1:
            game.step(3)
        elif snake_direction == 2:
            game.step(0)
        elif snake_direction == 3:
            game.step(1)

        state = game.get_state()

    score = game.score + game.apple_penalty
    return score

def eval_genomes(genomes, config):
    population_size = len(genomes)
    num_threads = os.cpu_count() + 100

    for batch_start in range(0, population_size, num_threads):
        batch_end = min(batch_start + num_threads, population_size)
        batch_genomes = genomes[batch_start:batch_end]

        pool = Pool(processes=len(batch_genomes))
        fitness_values = pool.starmap(eval_genome, [(genome, config) for genome_id, genome in batch_genomes])

        for (genome_id, genome), fitness in zip(batch_genomes, fitness_values):
            genome.fitness = fitness

    # wait for all threads to finish playing
    while any(genome.fitness is None for genome_id, genome in genomes):
        pass

    pool.close()
    pool.join()

def train_ai(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 300)
    print(f'\nBest genome:\n{winner}')

if __name__ == '__main__':
    config_path = 'config.txt'
    train_ai(config_path)
