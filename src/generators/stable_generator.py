import src.generators.generator_utils as gen_utils
import numpy as np
import networkx as nx
from click import progressbar


def generate_instances(configs):
    """Generate AFs with planted stable extensions sharing a common core.

    Each instance gets `e` stable extensions E_i = C ∪ R_i: a common core C,
    contained in every extension, plus pairwise incomparable random parts R_i.
    The sets are made stable by attacks: every argument outside a set is
    attacked by a member of that set, and no attack connects two arguments
    that share a set. Random noise attacks between arguments that share no set
    follow. Incoming attacks do not affect stability, so every planted set
    stays stable under the noise: each instance has at least `e` stable
    extensions, noise may add more.

    The core makes DS non-trivial: core arguments are in every planted
    extension, so they are skeptically accepted unless noise adds an extension
    without them. Without a core the intersection is almost always empty and
    every argument is DS-rejected.
    """
    general_configs = configs['general']
    stable_configs = configs['stable']
    num_instances = general_configs['num']
    num_args = gen_utils.get_num_args(general_configs, num_instances)
    e_params, core_params = _init_sampling_parameters(stable_configs, num_instances)
    size_range = tuple(stable_configs['size_extension'])
    noise_prob = stable_configs.get('noise_prob', 0.05)

    instances = []
    with progressbar(list(zip(num_args, e_params, core_params)),
                     label='Generating instances') as parameters:
        for a, e, c in parameters:
            graph, _ = generate_instance(int(a), int(e), size_range, noise_prob,
                                         core_size=int(c))
            instances.append(graph)
    if general_configs['save']:
        return gen_utils.save_instances(instances, general_configs, 'stable')
    return instances


def generate_instance(a, e, size_range, noise_prob, core_size=0, max_tries=100):
    """One AF with `a` arguments (1..a) and `e` planted stable extensions.

    Every extension contains the same core of `core_size` arguments; the
    random part of each extension has a size drawn from `size_range`.
    Redraws when the sets cannot be planted (no valid witness attacker, an
    argument that nothing may attack, or no argument left outside all sets
    while a core exists). Returns (graph, extensions).
    """
    for _ in range(max_tries):
        extensions = _draw_extensions(a, e, size_range, core_size)
        if extensions is None:
            continue
        graph = _plant(a, extensions, noise_prob)
        if graph is not None:
            return graph, extensions
    raise RuntimeError(f"could not plant {e} extensions with core {core_size} in "
                       f"{a} arguments after {max_tries} tries; lower e, the core "
                       f"or the set sizes")


def _draw_extensions(a, e, size_range, core_size):
    """Draw a core and `e` pairwise incomparable random parts.

    Incomparable because a stable extension cannot contain another one; with
    a common core, E_i and E_j are incomparable exactly when their random
    parts are. Random parts are drawn outside the core.

    Core arguments can only be attacked by arguments in no set, since they
    share a set with every other set member. With a core, at least one
    argument must therefore stay outside all sets, otherwise the core would
    be unattacked and end up in the grounded extension.
    """
    args = np.arange(1, a + 1)
    core = frozenset(int(x) for x in np.random.choice(args, core_size, replace=False))
    rest = np.array([x for x in args if x not in core])

    lo, hi = size_range
    lo = max(lo, 1)                      # empty random parts are never incomparable
    hi = min(hi, len(rest) - 1)          # leave an argument outside every set
    if hi < lo:
        return None

    parts = []
    for _ in range(e):
        for _ in range(100):
            size = np.random.randint(lo, hi + 1)
            part = frozenset(int(x) for x in np.random.choice(rest, size, replace=False))
            if all(not (part <= other or other <= part) for other in parts):
                parts.append(part)
                break
        else:
            return None

    extensions = [core | part for part in parts]
    if core and len(set().union(*extensions)) == a:
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
    """Number of planted extensions and core size per instance, each uniform
    in its configured range. size_core defaults to [0, 0] (no core)."""
    lo, hi = stable_configs['num_extension']
    c_lo, c_hi = stable_configs.get('size_core', [0, 0])
    e = np.random.randint(lo, hi + 1, size=num_instances)
    c = np.random.randint(c_lo, c_hi + 1, size=num_instances)
    return e, c
