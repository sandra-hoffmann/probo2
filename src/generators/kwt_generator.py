import random
import networkx as nx


class KwtDungTheoryGenerator():
    """Python port of TweetyProject's KwtDungTheoryGenerator.

    Builds an AF around a planted set of preferred extensions: the skeptical
    arguments (ideal extension plus other skeptical arguments) are in every
    extension, credulous arguments in some of them, unaccepted arguments in
    none. Attacks are added so that every extension argument outside the
    ideal extension is attacked from outside and defended.

    Args:
        num_arguments: number of arguments; the instance has num_arguments - 1
            regular arguments plus the self-attacking argument 'c'
        num_skept_arguments: number of skeptically accepted arguments
        size_ideal_extension: size of the ideal extension (first part of the
            skeptical arguments)
        num_cred_arguments: number of credulously accepted arguments
        num_pref_exts: number of planted preferred extensions (at least 2)
        p_ideal_attacked: probability an unaccepted argument attacks an ideal one
        p_ideal_attack_back: probability an ideal argument defends against it
        p_other_skept_args_attacked: probability an unaccepted argument attacks
            a skeptical argument outside the ideal extension
        p_other_skept_args_attack_back: probability a credulous extension
            member attacks back
        p_cred_args_attacked: probability an outside argument attacks a
            credulous extension member
        p_cred_args_attack_back: probability a credulous member attacks back
        p_other_attacks: probability of an attack between unaccepted arguments
    """

    def __init__(self, num_arguments, num_skept_arguments, size_ideal_extension,
                 num_cred_arguments, num_pref_exts, p_ideal_attacked,
                 p_ideal_attack_back, p_other_skept_args_attacked,
                 p_other_skept_args_attack_back, p_cred_args_attacked,
                 p_cred_args_attack_back, p_other_attacks):
        self.num_arguments = num_arguments
        self.num_skept_arguments = num_skept_arguments
        self.size_ideal_extension = size_ideal_extension
        self.num_cred_arguments = num_cred_arguments
        self.num_pref_exts = num_pref_exts
        self.p_ideal_attacked = p_ideal_attacked
        self.p_ideal_attack_back = p_ideal_attack_back
        self.p_other_skept_args_attacked = p_other_skept_args_attacked
        self.p_other_skept_args_attack_back = p_other_skept_args_attack_back
        self.p_cred_args_attacked = p_cred_args_attacked
        self.p_cred_args_attack_back = p_cred_args_attack_back
        self.p_other_attacks = p_other_attacks

    def generate_instance(self):
        """Generate one AF.

        Produces the same graph as the original list-based port for the same
        random state: every loop iterates in the original order and draws the
        same random numbers. The speedup comes only from set membership checks
        and from iterating over the credulous members of an extension directly
        instead of scanning the whole extension and skipping skeptical ones,
        which never drew a random number.
        """
        rand = random.random
        af = nx.DiGraph()
        add_edge = af.add_edge
        args = [f'a{i}' for i in range(0, self.num_arguments - 1)]
        af.add_nodes_from(args)
        skept_args = args[:self.num_skept_arguments]
        cred_args = args[self.num_skept_arguments:
                         self.num_skept_arguments + self.num_cred_arguments]
        pref_exts = [skept_args.copy() for _ in range(0, self.num_pref_exts)]
        ideal_ext = skept_args[:self.size_ideal_extension]
        other_skept_args = skept_args[self.size_ideal_extension:]
        unaccepted_arguments = args[self.num_skept_arguments + self.num_cred_arguments:]

        # ideal_ext and other_skept_args partition skept_args, so "not in
        # ideal_ext and not in other_skept_args" is "not skeptical"
        skept_set = set(skept_args)

        # distribute the credulous arguments over the extensions
        for cred_arg in cred_args:
            num_exts = random.randint(1, self.num_pref_exts - 1)
            for index in random.sample(range(0, self.num_pref_exts), num_exts):
                pref_exts[index].append(cred_arg)

        # credulous members of each extension, in extension order
        cred_in_ext = [[x for x in ext if x not in skept_set] for ext in pref_exts]

        # all arguments in the ideal extension should be attacked by some
        # unaccepted argument (empty grounded extension) and defended by the
        # ideal extension
        for arg_ideal in ideal_ext:
            for arg_unaccepted in unaccepted_arguments:
                if rand() < self.p_ideal_attacked:
                    add_edge(arg_unaccepted, arg_ideal)
                    for arg_ideal_back in ideal_ext:
                        if rand() < self.p_ideal_attack_back:
                            add_edge(arg_ideal_back, arg_unaccepted)

        # skeptical arguments outside the ideal extension are attacked by
        # unaccepted arguments and defended by credulous extension members
        for other_skept_arg in other_skept_args:
            for unaccepted_arg in unaccepted_arguments:
                if rand() < self.p_other_skept_args_attacked:
                    add_edge(unaccepted_arg, other_skept_arg)
                    for members in cred_in_ext:
                        for pref_arg in members:
                            if rand() < self.p_other_skept_args_attack_back:
                                add_edge(pref_arg, unaccepted_arg)

        # every credulous argument in an extension is attacked from outside
        # the extension and defended by the extension
        for pref_ext, members in zip(pref_exts, cred_in_ext):
            pref_set = set(pref_ext)
            outside = [a for a in args if a not in skept_set and a not in pref_set]
            for pref_arg in members:
                for arg in outside:
                    if rand() < self.p_cred_args_attacked:
                        add_edge(arg, pref_arg)
                    for o_pref_arg in members:
                        if rand() < self.p_cred_args_attack_back:
                            add_edge(o_pref_arg, arg)

        # some other attacks between unaccepted arguments
        for a in unaccepted_arguments:
            for b in unaccepted_arguments:
                if rand() < self.p_other_attacks:
                    add_edge(a, b)

        # self-attacking argument c makes stable extensions unlikely
        af.add_node('c')
        add_edge('c', 'c')
        for a in unaccepted_arguments:
            if bool(random.getrandbits(1)):
                if bool(random.getrandbits(1)):
                    add_edge(a, 'c')
                else:
                    add_edge('c', a)
        return af


if __name__ == '__main__':
    generator = KwtDungTheoryGenerator(1000, 50, 3, 3, 5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    for _ in range(0, 10):
        generator.generate_instance()