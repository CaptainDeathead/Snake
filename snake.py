import random
import cupy as np
import neat
import os
import matplotlib.pyplot as plt
import pygame
import time
from multiprocessing import Pool

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

        # Initialize Pygame
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
        # return the head position, food position
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

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (segment[0] * self.grid_size, segment[1] * self.grid_size,
                                                       self.grid_size, self.grid_size))

        # Draw food
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0] * self.grid_size, self.food[1] * self.grid_size,
                                                    self.grid_size, self.grid_size))

        # Update the display
        pygame.display.flip()

    def play_game(self):
        # tick
        self.clock.tick(1000)
        self.draw_game()

def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = eval_genome(genome, config)

# train_ai function
def train_ai(config_file, num_runs):

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create a list to store the population objects
    populations = []

    # Create the populations and add them to the list
    for _ in range(num_runs):
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        populations.append(population)

    # Run the populations concurrently
    with Pool() as pool:
        pool.starmap(population.run, [(eval_genomes, 300) for population in populations])

    # Display the winning genomes for each population
    for i, population in enumerate(populations):
        winner = population.best_genome()
        print(f'\nBest genome for population {i + 1}:\n{winner}')

# eval_genome function
def eval_genome(genome, config):
    # create the network with the snake head and food as inputs
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # create the game
    game = SnakeGame()

    # get the initial state
    state = game.get_state()

    # initialize the previous action to None
    prev_action = None

    # run the game
    while not game.game_over:
        # get the action from the network
        action = net.activate(state.flatten())

        if prev_action is not None:
            # check if the AI picked the opposite direction of its current movement
            if (action.index(max(action)) + prev_action) % 2 == 0:
                snake_direction = prev_action

            else:
                # determine the snake's direction based on the highest output
                snake_direction = action.index(max(action))
        else:
            # determine the snake's direction based on the highest output
            snake_direction = action.index(max(action))
        
        # set the previous action
        prev_action = snake_direction

        # map the snake direction to actual movements
        if snake_direction == 0:  # Up
            game.step(2)  # move left
        elif snake_direction == 1:  # Down
            game.step(3)  # move right
        elif snake_direction == 2:  # Left
            game.step(0)  # move up
        elif snake_direction == 3:  # Right
            game.step(1)  # move down

        # get the next state
        state = game.get_state()

    # return the score
    return game.score

# run the train_ai function
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    num_runs = 50  # Number of AIs to run concurrently
    train_ai(config_path, num_runs)