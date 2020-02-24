#!/usr/bin/python3
# -*- coding:utf-8 -*-
import collections
import logging
import re

import six
from colorama import Back, Fore, Style

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except ImportError:
    collectionsAbc = collections


def highlight(keyword: str, target: str, color=Fore.BLACK + Back.YELLOW) -> str:
    """
    use given color to highlight keyword in target string

    Args:
        keyword(str): highlight string
        target(str): target string
        color(str): string represent the color, use black foreground
        and yellow background as default

    Returns:
        (str) target string with keyword highlighted

    """
    return re.sub(keyword, color + r"\g<0>" + Style.RESET_ALL, target)


def find_key(param_dict: dict, key: str) -> dict:
    """
    find key in dict

    Args:
        param_dict(dict):
        key(str):

    Returns:
        (dict)

    Examples::
        >>> d = dict(abc=2, ab=4, c=4)
        >>> find_key(d, "ab")
        {'abc': 2, 'ab':4}

    """
    find_result = {}
    for k, v in param_dict.items():
        if re.search(key, k):
            find_result[k] = v
        if isinstance(v, dict):
            res = find_key(v, key)
            if res:
                find_result[k] = res
    return find_result


def diff_dict(dict1: dict, dict2: dict) -> dict:
    """
    find difference between src dict and dst dict

    Args:
        src(dict): src dict
        dst(dict): dst dict

    Returns:
        (dict) dict contains all the difference key

    """
    diff_result = {}
    for k, v in dict1.items():
        if k not in dict2:
            diff_result[k] = v
        elif dict2[k] != v:
            if isinstance(v, dict):
                diff_result[k] = diff_dict(v, dict2[k])
            else:
                diff_result[k] = v
    return diff_result


def _assert_with_logging(cond, msg):
    logger = logging.getLogger(__name__)
    if not cond:
        logger.debug(msg)
    assert cond, msg


def update(d, u):
    for k, v in six.iteritems(u):
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = update(dv, v)
        else:
            d[k] = v
    return d


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """
    Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )
