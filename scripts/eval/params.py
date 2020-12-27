from pathlib import Path

def get_jobs(model):
    jobs = []

    routes_dir = Path('/u/nimit/Documents/robomaster/2020_CARLA_challenge/leaderboard/data/routes_testing')
    for route_dir in routes_dir.iterdir():
        job = f"""ROUTES={route_dir} TEAM_CONFIG={model} ./run_model"""

        jobs.append(job)
        print(job)

    return jobs
