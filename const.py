MAP_SIZE = 150 # cached MAP_SIZE, training MAP_SIZE is a CLI argument
GAP, STEPS = 1, 5

BACKGROUND = [255, 255, 255, 255]
COLORS = [
    (128,  64, 128, 100),    # road
    (238, 123,   8, 255),    # ego-vehicle
    (  0,   0, 142, 255),    # other vehicle
    (220,  20,  60, 255),    # pedestrian
    (204,   6,   5, 255),    # red light
    (250, 210,   1, 255),    # yellow light
    (39,  232,  51, 255),    # green light
    (181, 161, 201, 255),    # yield
    (204,   6,   5, 255),    # stop
    (158, 167, 239, 255)     # speed limit
]

LANDMARKS = {
    '205': 7,           # yield
    '206': 8,           # stop
    '274': 9            # speed limit
}

#ACTIONS = ['X', 'S', 'L', 'R']
ACTIONS = ['L', 'R', 'S', 'F', 'LL', 'LR']
