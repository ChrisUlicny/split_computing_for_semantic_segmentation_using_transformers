config = {
    "result_dir": "/results/culicny/simulation/split{}/channels{}/run_{}/",
    "result_name": "car{}/",
    "repeat": 3,
    "repeat_start_idx": 0,
    "episodes": 1,
    "comment": "Test Config",
    'requirements_search': None,
    'simulation': {
        'dt': 1,
        'simulation_type': 'ours',
        'steps': 1000,
        'training_steps': 0,
        'gui_enabled': False,
        'gui_timeout': 0,
    },
    'vehicles': [
        {
            'count': 10,
            'speed_ms': 11,
            'speed_variation': 2,
            'ips': 23.1e10,
            'model_location': "",
            'task': {
                'ttl': 0.1,
                'generation_period': 0.1,
                'generation_delay_range': 0
            },
            'requirements': {
                'latency': 0.01,
                'accuracy': 0.8
            },
            'weights': {
                'latency': 1,
                'accuracy': 1
            }
        },
    ],
    'base_stations': {
        "count": 1,
        "min_radius": 30,
        "ips": 41.29e12,
        "bandwidth": 100e6,
        # nemal by byt taky velky aby sa oplatilo stale off-loadovat, aby off-load bol rychlejsi
        # ako vypocet na zariadeni ale stale menise aby sa always offload neoplatil
        # split lepsi ako prvy split
        # okolo toho menit resource bloky a vykon zariadenia
        "resource_blocks": 2000,
        "tx_frequency": 2e9,
        "tx_power": 0.1,
        "coverage_radius": 500
    }
}