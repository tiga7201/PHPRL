from env.instance_generator import create_demo_instance

from baseline.hga.decoder import HgaDecoder
from baseline.hga.hga import HGA, HGAConfig
from baseline.hga.local_search import (
    CriticalPathLocalSearch,
    LocalSearchConfig,
)


instance = create_demo_instance()

decoder = HgaDecoder(
    instance=instance,
    phase1_checkpoint="checkpoints/pgnn_phase1.pt",
)

local_search = CriticalPathLocalSearch(
    instance=instance,
    decoder=decoder,
    config=LocalSearchConfig(
        max_iterations=5,
        max_neighbors=20,
        strategy="best",
    ),
    seed=42,
)

solver = HGA(
    instance=instance,
    decoder=decoder,
    local_search=local_search,
    config=HGAConfig(
        population_size=20,
        generations=30,
        elite_size=2,
        local_search_elite_size=2,
        local_search_interval=5,
    ),
    seed=42,
)

result = solver.solve()

print("best makespan:", result.best_makespan)
print("total evaluations:", result.evaluations)
print("GA evaluations:", result.ga_evaluations)
print(
    "local-search evaluations:",
    result.local_search_evaluations,
)
print("elapsed seconds:", result.elapsed_seconds)