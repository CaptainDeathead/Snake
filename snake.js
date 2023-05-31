// Get the canvas element
const canvas = document.getElementById("gameCanvas");
const context = canvas.getContext("2d");

// Define the size of each cell in the grid
let gridSize = 20;
const canvasSize = 600;

// Set up the initial snake position and direction
let snake = [{ x: 0, y: 0 }];
let dx = gridSize;
let dy = 0;

// Set up the initial food position
let food = {
  x: Math.floor(Math.random() * (canvasSize / gridSize)) * gridSize,
  y: Math.floor(Math.random() * (canvasSize / gridSize)) * gridSize
};

// Set up the game loop
let lastTime = 0;
const snakeSpeed = 100; // Snake speed in milliseconds

function gameLoop(currentTime) {
  // Calculate the elapsed time since the last frame
  const deltaTime = currentTime - lastTime;
  lastTime = currentTime;

  // Clear the canvas
  context.clearRect(0, 0, canvas.width, canvas.height);

  // Move the snake
  const head = { x: snake[0].x + dx, y: snake[0].y + dy };
  snake.unshift(head);

  // Check if the snake has collided with the walls or its own body
  if (
    head.x < 0 ||
    head.x >= canvas.width ||
    head.y < 0 ||
    head.y >= canvas.height ||
    snake.some((segment, index) => index !== 0 && segment.x === head.x && segment.y === head.y)
  ) {
    // reload the page
    window.location.reload();
  }

  // Check if the snake has collided with the food
  if (head.x === food.x && head.y === food.y) {
    // Increase the length of the snake
    snake.push({});

    // Generate new food position
    food = {
      x: Math.floor(Math.random() * (canvasSize / gridSize)) * gridSize,
      y: Math.floor(Math.random() * (canvasSize / gridSize)) * gridSize
    };
  } else {
    snake.pop();
  }

  // Draw the snake
  snake.forEach(segment => {
    context.fillStyle = "green";
    context.fillRect(segment.x, segment.y, gridSize, gridSize);
  });

  // Draw the food
  context.fillStyle = "red";
  context.fillRect(food.x, food.y, gridSize, gridSize);

  // Call the game loop again
  setTimeout(() => {
    requestAnimationFrame(gameLoop);
  }, snakeSpeed);
}

// Listen for keydown events
document.addEventListener("keydown", changeDirection);

// Change the snake's direction based on the arrow keys
function changeDirection(event) {
  const LEFT_KEY = 37;
  const RIGHT_KEY = 39;
  const UP_KEY = 38;
  const DOWN_KEY = 40;

  const keyPressed = event.keyCode;

  if (keyPressed === LEFT_KEY && dx !== gridSize) {
    dx = -gridSize;
    dy = 0;
  }

  if (keyPressed === RIGHT_KEY && dx !== -gridSize) {
    dx = gridSize;
    dy = 0;
  }

  if (keyPressed === UP_KEY && dy !== gridSize) {
    dx = 0;
    dy = -gridSize;
  }

  if (keyPressed === DOWN_KEY && dy !== -gridSize) {
    dx = 0;
    dy = gridSize;
  }
}

// Resize the canvas to fit the window
function resizeCanvas() {
  canvas.width = 600;
  canvas.height = 600;
}

// Call the resizeCanvas function initially and whenever the window is resized
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// Start the game loop
requestAnimationFrame(gameLoop);
