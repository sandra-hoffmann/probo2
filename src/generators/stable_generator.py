import src.generators.generator_utils as gen_utils
import numpy as np
import networkx as nx
from click import progressbar


def generate_instances(configs):
    """Generate AFs with planted stable extensions.

    Each instance gets `e` pairwise incomparable argument sets, made stable by
    attacks: every argument outside a set is attacked by a member of that set,
    and no attack connects two arguments that share a set. Random noise attacks
    between arguments that share no set follow. Incoming attacks do not affect
    stability, so every planted set stays stable under the noise: each instance
    has at least `e` stable extensions, noise may add more.
    """
    general_configs = configs['general']
    stable_configs = configs['stable']
    num_instances = general_configs['num']
    num_args = gen_utils.get_num_args(general_configs, num_instances)
    e_params = _init_sampling_parameters(stable_configs, num_instances)
    size_range = tuple(stable_configs['size_extension'])
    noise_prob = stable_configs.get('noise_prob', 0.05)

    instances = []
    with progressbar(list(zip(num_args, e_params)), label='Generating instances') as parameters:
        for a, e in parameters:
            graph, _ = generate_instance(int(a), int(e), size_range, noise_prob)
            instances.append(graph)
    if general_configs['save']:
        return gen_utils.save_instances(instances, general_configs, 'stable')
    return instances


def generate_instance(a, e, size_range, noise_prob, max_tries=100):
    """One AF with `a` arguments (1..a) and `e` planted stable extensions.

    Redraws the sets when they cannot be planted (no valid witness attacker,
    or an argument that nothing may attack). Returns (graph, extensions).
    """
    for _ in range(max_tries):
        extensions = _draw_extensions(a, e, size_range)
        if extensions is None:
            continue
        graph = _plant(a, extensions, noise_prob)
        if graph is not None:
            return graph, extensions
    raise RuntimeError(f"could not plant {e} extensions in {a} arguments "
                       f"after {max_tries} tries; lower e or the set sizes")


def _draw_extensions(a, e, size_range):
    """Draw `e` pairwise incomparable subsets of 1..a.

    Incomparable because a stable extension cannot contain another one.
    The maximum size is capped at a-1, so every set leaves an argument
    outside and DC-ST keeps a negative class.
    """
    lo, hi = size_range
    hi = min(hi, a - 1)
    extensions = []
    for _ in range(e):
        for _ in range(100):
            size = np.random.randint(lo, hi + 1)
            ext = frozenset(int(x) for x in
                            np.random.choice(np.arange(1, a + 1), size, replace=False))
            if all(not (ext <= other or other <= ext) for other in extensions):
                extensions.append(ext)
                break
        else:
            return None
    return extensions


def _plant(a, extensions, noise_prob):
    """Build the attack relation that makes every set in `extensions` stable.

    Returns None if some argument outside a set has no valid attacker in it,
    or if an argument would stay unattacked and nothing may attack it.
    """
    args = range(1, a + 1)
    member_of = {x: set() for x in args}
    for i, ext in enumerate(extensions):
        for x in ext:
            member_of[x].add(i)

    def compatible(x, y):
        # sharing a planted set: an attack between x and y would break it
        return bool(member_of[x] & member_of[y])

    graph = nx.DiGraph()
    graph.add_nodes_from(args)

    # witness attacks: each set attacks every argument outside it
    for ext in extensions:
        members = sorted(ext)
        for x in args:
            if x in ext:
                continue
            candidates = [b for b in members if not compatible(b, x)]
            if not candidates:
                return None
            graph.add_edge(int(np.random.choice(candidates)), x)

    # noise: only between arguments that share no set,
    # self-attacks only for arguments in no set
    for x in args:
        for y in args:
            if x == y:
                if not member_of[x] and np.random.random() < noise_prob:
                    graph.add_edge(x, x)
            elif not compatible(x, y) and np.random.random() < noise_prob:
                graph.add_edge(x, y)

    # an unattacked argument would enter the grounded extension
    for x in args:
        if graph.in_degree(x) == 0:
            attackers = [y for y in args if y != x and not compatible(x, y)]
            if not attackers:
                return None
            graph.add_edge(int(np.random.choice(attackers)), x)
    return graph


def _init_sampling_parameters(stable_configs, num_instances):
    """Number of planted extensions per instance, uniform in num_extension."""
    lo, hi = stable_configs['num_extension']
    return np.random.randint(lo, hi + 1, size=num_instances)