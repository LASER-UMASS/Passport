import os
import json
import pickle
import sys
sys.setrecursionlimit(100000)
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')))
from gallina import GallinaTermParser, traverse_postorder
from lark.exceptions import UnexpectedCharacters, ParseError
import argparse
from utils import iter_proofs, SexpCache
from lark.tree import Tree

# Count occurrences of names in the training data.
# Based heavily on extract_proof_steps for the boilerplate.

projs_split = json.load(open('../projs_split.json'))

# For now we are naive and lump everything into one bucket,
# whether it's the name of a module, datatype, or constant.
# The AST orders these meaningfully, so hopefully the model reflects the meaning in the end.
# If not, there are simple changes we can make later to reflect that.
idents = {}

get_locals = False
term_parser = GallinaTermParser(caching=True, include_locals=True, include_defs=True)
sexp_cache = SexpCache('../sexp_cache', readonly=True)

def parse_goal(g):
    return term_parser.parse(sexp_cache[g['sexp']])

def incr_ident(ident):
    if ident not in idents:
        idents[ident] = 1
    else:
        idents[ident] += 1

def is_name(node):
    if get_locals:
        return node.data == 'constructor_var' or node.data == 'constructor_name'
    else:
        return node.data == 'names__id__t'

def count(filename, proof_data):
    proj = filename.split(os.path.sep)[2]
    if not proj in projs_split['projs_train']:
        return

    proof_start = proof_data['steps'][0]
    goal_id = proof_start['goal_ids']['fg'][0]
    goal_ast = parse_goal(proof_data['goals'][str(goal_id)])

    # count occurrences within a goal
    def count_in_goal(node):
        if is_name(node):
            ident = node.children[0].data
            incr_ident(ident)
        else:
            children = []
            for c in node.children:
                if isinstance(c, Tree):
                    children.append(c)
            node.children = children

    traverse_postorder(goal_ast, count_in_goal)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Count occurrences of named datatypes and constants in the data')
    arg_parser.add_argument('--data_root', type=str, default='../data',
                                help='The folder for CoqGym')
    arg_parser.add_argument('--output', type=str, default='./names/',
                                help='The output file')
    arg_parser.add_argument('--locals', action='store_true', help='Get local names instead of global names')
    args = arg_parser.parse_args()
    print(args)
    
    get_locals = args.locals

    iter_proofs(args.data_root, count, include_synthetic=False, show_progress=True)

    dirname = args.output
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = 'names.pickle'
    if get_locals: # I apologize to the functional programming gods, but Python is so confusing
        filename = 'locals.pickle'

    names_file = open(os.path.join(dirname, filename), 'wb')
    pickle.dump(idents, names_file)
    names_file.close()

    print('output saved to ', args.output)