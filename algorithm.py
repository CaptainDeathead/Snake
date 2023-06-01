import random
import pygame
import time

pygame.init()

screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Snake")

class Game:
    def __init__(self):
        self.grid = [[0 for i in range(10)] for j in range(10)]
        self.snake = [(5, 5)]
        self.direction = "right"
        self.food = (random.randint(0, 9), random.randint(0, 9))
        self.score = 0
        self.start_time = time.time()
        self.found_food_last = time.time()

    def input_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != "down":
                    self.direction = "up"
                elif event.key == pygame.K_LEFT and self.direction != "right":
                    self.direction = "left"
                elif event.key == pygame.K_DOWN and self.direction != "up":
                    self.direction = "down"
                elif event.key == pygame.K_RIGHT and self.direction != "left":
                    self.direction = "right"

    def move_snake(self):
        if time.time() - self.start_time > 0.04:
            head = self.snake[0]
            if self.direction == "up":
                head = (head[0], head[1] - 1)
            elif self.direction == "left":
                head = (head[0] - 1, head[1])
            elif self.direction == "down":
                head = (head[0], head[1] + 1)
            elif self.direction == "right":
                head = (head[0] + 1, head[1])
            self.snake.insert(0, head)
            if head == self.food:
                self.found_food_last = time.time()
                self.generate_food()
            else:
                self.snake.pop()
            if head[0] < 0 or head[0] > 9 or head[1] < 0 or head[1] > 9:
                self.end_game()
            if head in self.snake[1:]:
                self.end_game()
            self.start_time = time.time()

    def generate_food(self):
        self.score += 1
        self.food = (random.randint(0, 9), random.randint(0, 9))

    def end_game(self):
        score = 0
        snake = [(0, 0)]
        direction = "right"
        food = (random.randint(0, 9), random.randint(0, 9))

    def main(self):
        while True:
            screen.fill((0, 0, 0))
            snake_rects = []
            for i in range(len(self.snake)):
                snake_rects.append(pygame.Rect(self.snake[i][0] * 40, self.snake[i][1] * 40, 40, 40))
            food_rect = pygame.Rect(self.food[0] * 40, self.food[1] * 40, 40, 40)
            pygame.draw.rect(screen, (255, 0, 0), food_rect)
            for i in range(len(snake_rects)):
                pygame.draw.rect(screen, (0, 255, 0), snake_rects[i])
            pygame.display.update()
            self.input_handler()
            self.move_snake()

    def update_gui(self):
        size = 80
        screen.fill((0, 0, 0))
        snake_rects = []
        for i in range(len(self.snake)):
            snake_rects.append(pygame.Rect(self.snake[i][0] * size, self.snake[i][1] * size, size, size))
        food_rect = pygame.Rect(self.food[0] * size, self.food[1] * size, size, size)
        pygame.draw.rect(screen, (255, 0, 0), food_rect)
        for i in range(len(snake_rects)):
            pygame.draw.rect(screen, (0, 255, 0), snake_rects[i])
        pygame.display.update()

    def reset(self):
        self.grid = [[0 for i in range(10)] for j in range(10)]
        self.snake = [(5, 5)]
        self.direction = "right"
        self.food = (random.randint(0, 9), random.randint(0, 9))
        self.score = 0
        self.start_time = time.time()
        self.found_food_last = time.time()

class Ai:
    import neat
    import os
    import pickle

    def __init__(self):
        self.game = Game()
        self.config = self.neat.config.Config(self.neat.DefaultGenome, self.neat.DefaultReproduction,
                                              self.neat.DefaultSpeciesSet, self.neat.DefaultStagnation,
                                              self.os.path.join(self.os.path.dirname(__file__), 'config.txt'))
        self.p = self.neat.Population(self.config)
        self.p.add_reporter(self.neat.StdOutReporter(True))
        self.stats = self.neat.StatisticsReporter()
        self.p.add_reporter(self.stats)
        self.p.add_reporter(self.neat.Checkpointer(5, filename_prefix='checkpoints/neat-checkpoint-'))

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = self.neat.nn.FeedForwardNetwork.create(genome, config)
            while True:
                if time.time() - self.game.found_food_last > 5:
                    self.game.reset()
                    genome.fitness -= 1
                    break
                self.game.input_handler()
                self.game.move_snake()
                self.game.update_gui()
                inputs = (self.game.snake[0][0], self.game.snake[0][1], self.game.food[0], self.game.food[1])
                output = net.activate(inputs)
                output = output.index(max(output))
                if output == 0 and self.game.direction != "down":
                    self.game.direction = "up"
                elif output == 1 and self.game.direction != "right":
                    self.game.direction = "left"
                elif output == 2 and self.game.direction != "up":
                    self.game.direction = "down"
                elif output == 3 and self.game.direction != "left":
                    self.game.direction = "right"
                if self.game.snake[0] == self.game.food:
                    genome.fitness += 1
                    print("in food")
                    self.game.generate_food()
                if self.game.snake[0][0] < 0 or self.game.snake[0][0] > 9 or self.game.snake[0][1] < 0 or self.game.snake[0][1] > 9:
                    self.game.reset()
                    genome.fitness -= 1
                    break
                if self.game.snake[0] in self.game.snake[1:]:
                    self.game.reset()
                    genome.fitness -= 1
                    break
                #genome.fitness += 0.001

if __name__ == '__main__':
    ai = Ai()
    ai.p.run(ai.eval_genomes, 100000)