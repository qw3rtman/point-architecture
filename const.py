HEIGHT, WIDTH = 320, 320
C = 7 # map classes

GAP, STEPS = 2, 5

BACKGROUND = [255, 255, 255]
COLORS = [
    (238, 123,   8),  # ego-vehicle
    (128,  64, 128),  # road
    (  0,   0, 142),  # lane
    (204, 6, 5),      # red light
    (250, 210, 1),    # yellow light
    (39, 232, 51),    # green light
    (0, 0, 142),      # vehicle
    (220, 20, 60)     # pedestrian
]

ACTIONS = ['S', 'F', 'L', 'R']
