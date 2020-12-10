MAP_SIZE = 50
GAP, STEPS = 1, 5

BACKGROUND = [255, 255, 255]
COLORS = [
    (128,  64, 128),    # road
    (238, 123,   8),    # ego-vehicle
    (  0,   0, 142),    # other vehicle
    (220,  20,  60),    # pedestrian
    (204,   6,   5),    # red light
    (250, 210,   1),    # yellow light
    (39,  232,  51),    # green light
    (181, 161, 201),    # yield
    (204,   6,   5),    # stop
    (158, 167, 239)     # speed limit
]

LANDMARKS = {
    '205': 7,           # yield
    '206': 8,           # stop
    '274': 9            # speed limit
}

#ACTIONS = ['X', 'S', 'L', 'R']
ACTIONS = ['L', 'R', 'S', 'F', 'LL', 'LR']
