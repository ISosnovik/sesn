'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
from collections import OrderedDict


def parse_range_tokens(tokens):
    '''Convert list of tokens into a list of sorted integers according to special rules.
    # Input:
        tokens: list of str. [str1, str2, ..., strN]
    # Output:
        list: [int1, int2, ..., intM]
    # Rules:
        1) "start|end" -> range(start, end)
        2) "start|end|step" -> range(start, end, step)
        3) "int" -> [int]
    # Example:
        >>> tokens = ["4|7", "1", "10|20|3"]
        >>> parse_range_tokens(tokens)
        [1, 4, 5, 6, 10, 13, 16, 19]
    '''
    step_list = []
    for token in tokens:
        steps = [int(t) for t in token.split('|')]
        if len(steps) > 1:
            steps = list(range(*steps))
        step_list.extend(steps)
    step_list = sorted(step_list)
    return step_list


def repr1line(obj):
    '''custom convenience dumper to YAML compatible format
    the output is 1 line. keys a sorted by two rules: models (Y/n), abc
    '''
    if obj is None:
        return 'null'

    if isinstance(obj, bool):
        return str(obj).lower()

    if isinstance(obj, str):
        return "'{}'".format(obj)

    if isinstance(obj, (int, float)):
        return str(obj)

    if isinstance(obj, (tuple, set)):
        return repr1line(list(obj))

    if isinstance(obj, list):
        els = map(lambda x: repr1line(x), obj)
        return '[{}]'.format(', '.join(els))

    if isinstance(obj, dict):
        keys = list(obj.keys())
        keys.remove('model')
        keys = ['model'] + sorted(keys)
        items = [(k, obj[k]) for k in keys]
        obj = OrderedDict(items)
        s = '{'
        for i, (k, v) in enumerate(obj.items()):
            s += '{}: {}, '.format(k, repr1line(v))
        s = s[:-2] + '}'
        return s


def dump_list_element_1line(obj):
    return '- {}\n'.format(repr1line(obj))
