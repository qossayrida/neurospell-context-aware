# import pygame
# import random
# import time
#
# # Initialize pygame
# pygame.init()
#
# # Constants
# WINDOW_WIDTH = 700
# WINDOW_HEIGHT = 700
# GRID_ROWS = 6
# GRID_COLS = 6
# CELL_WIDTH = WINDOW_WIDTH // GRID_COLS
# CELL_HEIGHT = WINDOW_HEIGHT // GRID_ROWS
# FONT_SIZE = 60
#
# # Colors
# BLACK = (0, 0, 0)
# GRAY = (100, 100, 100)  # Gray for non-flashing letters
# WHITE = (255, 255, 255)  # White for flashing letters
# FLASH_COLOR = (80, 80, 80)  # Light gray highlight for flash
#
# # P300 Speller Matrix (6x6)
# characters = [
#     ['A', 'B', 'C', 'D', 'E', 'F'],
#     ['G', 'H', 'I', 'J', 'K', 'L'],
#     ['M', 'N', 'O', 'P', 'Q', 'R'],
#     ['S', 'T', 'U', 'V', 'W', 'X'],
#     ['Y', 'Z', '1', '2', '3', '4'],
#     ['5', '6', '7', '8', '9', '0']
# ]
#
# # Set up the display
# window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
# pygame.display.set_caption('P300 Speller')
#
# # Font for displaying characters
# font = pygame.font.SysFont(None, FONT_SIZE)
#
#
# def draw_grid(flash_row=None, flash_col=None):
#     window.fill(BLACK)
#
#     for row in range(GRID_ROWS):
#         for col in range(GRID_COLS):
#             # Draw a background highlight if this row or column is flashing
#             if row == flash_row or col == flash_col:
#                 pygame.draw.rect(window, FLASH_COLOR, (col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
#                 char_color = WHITE  # Flashing letters should be white
#             else:
#                 char_color = GRAY  # Non-flashing letters should be gray
#
#             # Draw the character on top of the highlight
#             character = characters[row][col]
#             text = font.render(character, True, char_color)
#             window.blit(text, (col * CELL_WIDTH + CELL_WIDTH // 4, row * CELL_HEIGHT + CELL_HEIGHT // 4))
#
#     pygame.display.update()
#
#
# def main():
#     running = True
#     flash_time = 0.3  # Time to flash a row/column in seconds
#
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#
#         # Randomly choose to flash a row or a column
#         flash_type = random.choice(['row', 'col'])
#
#         if flash_type == 'row':
#             flash_row = random.randint(0, GRID_ROWS - 1)
#             flash_col = None
#         else:
#             flash_row = None
#             flash_col = random.randint(0, GRID_COLS - 1)
#
#         # Draw the grid with the subtle flashing background
#         draw_grid(flash_row, flash_col)
#
#         # Wait for the flash time
#         time.sleep(flash_time)
#
#         # Clear the flashing effect
#         draw_grid()
#
#         # Wait before the next flash
#         time.sleep(flash_time)
#
#     pygame.quit()
#
#
# if __name__ == "__main__":
#     main()


import pygame
import random
import time

# Initialize pygame
pygame.init()

# Constants
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
GRID_ROWS = 6
GRID_COLS = 6
CELL_WIDTH = WINDOW_WIDTH // GRID_COLS
CELL_HEIGHT = WINDOW_HEIGHT // GRID_ROWS
FONT_SIZE = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (40, 40, 40)
FLASH_COLOR = (255, 255, 255)

# P300 Speller Matrix (6x6)
characters = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '0']
]

# Set up the display
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('P300 Speller')

# Font for displaying characters
font = pygame.font.SysFont(None, FONT_SIZE)


def draw_grid(flash_row=None, flash_col=None):
    window.fill(BLACK)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            character = characters[row][col]
            if row == flash_row or col == flash_col:
                color = FLASH_COLOR
            else:
                color = WHITE
            text = font.render(character, True, color)
            window.blit(text, (col * CELL_WIDTH + CELL_WIDTH // 4, row * CELL_HEIGHT + CELL_HEIGHT // 4))

    pygame.display.update()


def main():
    running = True
    flash_time = 0.2  # Time to flash a row/column in seconds

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Randomly choose to flash a row or a column
        flash_type = random.choice(['row', 'col'])

        if flash_type == 'row':
            flash_row = random.randint(0, GRID_ROWS - 1)
            flash_col = None
        else:
            flash_row = None
            flash_col = random.randint(0, GRID_COLS - 1)

        # Draw the grid with the flashing row/column
        draw_grid(flash_row, flash_col)

        # Wait for the flash time
        time.sleep(flash_time)

        # Clear the flashing effect
        draw_grid()

        # Wait before the next flash
        time.sleep(flash_time)

    pygame.quit()


if __name__ == "__main__":
    main()
