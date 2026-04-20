import copy
import random
from typing import List, Tuple

from env.fjspwf_env import FJSPWFEnv
from env.instance_generator import generate_random_instance, InstanceData


Gene = Tuple[int, int]          # (job_id, op_id)
Action = Tuple[int, int, int, int]  # (job_id, op_id, machine_id, worker_id)


def init_population(instance: InstanceData, pop_size: int = 20) -> List[List[Gene]]:
    """
    染色体只编码工序访问顺序：
    [(job_id, op_id), ...]
    """
    all_ops: List[Gene] = []
    for job_id, ops in enumerate(instance.jobs):
        for op_id in range(len(ops)):
            all_ops.append((job_id, op_id))

    population = []
    for _ in range(pop_size):
        chrom = copy.deepcopy(all_ops)
        random.shuffle(chrom)
        population.append(chrom)
    return population


def choose_action_for_gene(valid_actions: List[Action], gene: Gene) -> Action | None:
    """
    从当前合法动作里，筛出与 gene=(job_id, op_id) 对应的动作。
    如果有多个 machine-worker 组合，采用一个简单启发式：
    选预计完成时间更早的动作（通过 env.step 的即时规则间接体现前瞻）
    这里先不访问内部复杂状态，只做稳定选择：machine_id, worker_id 最小者。
    """
    job_id, op_id = gene
    candidates = [a for a in valid_actions if a[0] == job_id and a[1] == op_id]
    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[2], x[3]))
    return candidates[0]


def decode_chromosome(chromosome: List[Gene], instance: InstanceData) -> FJSPWFEnv:
    """
    按染色体顺序解码。
    核心逻辑：
    - 依次扫描染色体
    - 只要某个 gene 对应的工序当前可执行，就在其合法 machine-worker 组合中选一个动作执行
    - 若当前扫描一轮没有任何 gene 可执行，则从 env 当前合法动作里选一个“兜底动作”
      以保证调度能继续推进
    """
    env = FJSPWFEnv(instance)
    env.reset()

    scheduled = set()

    while True:
        valid_actions = env.get_valid_actions()
        if len(valid_actions) == 0:
            break

        if not valid_actions:
            raise RuntimeError("No valid actions available before termination.")

        progressed = False

        for gene in chromosome:
            if gene in scheduled:
                continue

            action = choose_action_for_gene(valid_actions, gene)
            if action is None:
                continue

            env.step(action)
            scheduled.add(gene)
            progressed = True
            break

        if progressed:
            continue

        # 兜底：如果这一轮染色体里没有任何当前可执行工序，
        # 说明当前环境的可行动作与染色体扫描顺序卡住了。
        # 这时从合法动作里选一个稳定动作执行，保证解码完成。
        fallback_action = sorted(valid_actions, key=lambda x: (x[0], x[1], x[2], x[3]))[0]
        env.step(fallback_action)
        scheduled.add((fallback_action[0], fallback_action[1]))

    return env


def fitness(chromosome: List[Gene], instance: InstanceData) -> float:
    """
    适应度用 makespan。
    越小越好。
    """
    env = decode_chromosome(chromosome, instance)

    if len(env.schedule) == 0:
        return float("inf")

    makespan = max(op.end for op in env.schedule)
    return float(makespan)


def selection(population: List[List[Gene]], fitnesses: List[float]) -> List[Gene]:
    """
    轮盘赌选择。
    用 1 / fitness 作为概率，makespan 越小，被选中概率越高。
    """
    inv = [1.0 / max(f, 1e-8) for f in fitnesses]
    total = sum(inv)
    probs = [x / total for x in inv]

    r = random.random()
    cum = 0.0
    for chrom, p in zip(population, probs):
        cum += p
        if r <= cum:
            return copy.deepcopy(chrom)

    return copy.deepcopy(population[-1])


def crossover(parent1: List[Gene], parent2: List[Gene]) -> List[Gene]:
    """
    Order Crossover (OX)
    适合排列编码。
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[a:b] = parent1[a:b]

    ptr = b
    for gene in parent2:
        if gene in child:
            continue
        if ptr >= size:
            ptr = 0
        child[ptr] = gene
        ptr += 1

    return child


def mutate(chromosome: List[Gene], mut_rate: float = 0.1) -> None:
    """
    随机交换两个位置。
    """
    if random.random() < mut_rate:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]


def run_ga(
    instance: InstanceData,
    pop_size: int = 30,
    generations: int = 50,
    mut_rate: float = 0.1,
    elite_size: int = 2,
):
    """
    一个最小可用 GA：
    - 初始化
    - 评估
    - 精英保留
    - 轮盘赌选择
    - OX 交叉
    - swap 变异
    """
    if elite_size < 1:
        elite_size = 1
    elite_size = min(elite_size, pop_size)

    population = init_population(instance, pop_size)

    best_chrom = None
    best_fit = float("inf")

    history = []

    for gen in range(generations):
        fitnesses = [fitness(chrom, instance) for chrom in population]

        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1])
        gen_best_chrom, gen_best_fit = ranked[0]

        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_chrom = copy.deepcopy(gen_best_chrom)

        history.append({
            "generation": gen + 1,
            "best_makespan": float(gen_best_fit),
            "global_best_makespan": float(best_fit),
        })

        if gen % 10 == 0 or gen == generations - 1:
            print(
                f"Gen {gen:03d} | "
                f"gen_best={gen_best_fit:.4f} | "
                f"global_best={best_fit:.4f}"
            )

        # 精英保留
        new_population = [copy.deepcopy(chrom) for chrom, _ in ranked[:elite_size]]

        # 生成剩余个体
        while len(new_population) < pop_size:
            p1 = selection(population, fitnesses)
            p2 = selection(population, fitnesses)
            child = crossover(p1, p2)
            mutate(child, mut_rate)
            new_population.append(child)

        population = new_population

    return best_chrom, best_fit, history


def evaluate_ga_on_seeds(
    seeds: List[int],
    num_jobs: int = 5,
    num_machines: int = 3,
    num_workers: int = 3,
    min_ops_per_job: int = 2,
    max_ops_per_job: int = 4,
    pop_size: int = 30,
    generations: int = 50,
    mut_rate: float = 0.1,
    elite_size: int = 2,
):
    results = []

    for seed in seeds:
        instance = generate_random_instance(
            seed=seed,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )

        best_chrom, best_fit, history = run_ga(
            instance=instance,
            pop_size=pop_size,
            generations=generations,
            mut_rate=mut_rate,
            elite_size=elite_size,
        )

        results.append({
            "seed": seed,
            "best_makespan": float(best_fit),
            "history": history,
        })

        print(f"Seed {seed} | GA best makespan = {best_fit:.4f}")

    return results


if __name__ == "__main__":
    instance = generate_random_instance(
        seed=42,
        num_jobs=5,
        num_machines=3,
        num_workers=3,
        min_ops_per_job=2,
        max_ops_per_job=4,
    )

    best_chrom, best_fit, history = run_ga(
        instance,
        pop_size=20,
        generations=50,
        mut_rate=0.1,
        elite_size=2,
    )

    print("\nGA best makespan:", best_fit)
    print("Best chromosome length:", len(best_chrom))