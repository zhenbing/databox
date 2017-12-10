# encoding: utf-8 
'''
Created on Sep 14, 2015

@author: root
'''
from collections import namedtuple
from functools import update_wrapper
from threading import RLock
import time

################################################################################
### LRU Cache function decorator
################################################################################

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the lru_cache() will hash
        the key multiple times on a cache miss.

    """

    __slots__ = 'hashvalue'

    def __init__(self, tup):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(args, kwds, typed,
              kwd_mark=(object(),),
              fasttypes={int, str, frozenset, type(None)}):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    """
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


def lru_cache(maxsize=100, timeout=600, typed=False, args_base=0):
    def _cache_controller(viewfunc):

        cache = dict()
        stats = [0, 0]  # make statistics updateable non-locally
        HITS, MISSES = 0, 1  # names for the stats fields
        cache_get = cache.get  # bound method to lookup key or return None
        _len = len  # localize the global len() function
        lock = RLock()  # because linkedlist updates aren't threadsafe
        root = []  # root of the circular doubly linked list
        root[:] = [root, root, None, None]  # initialize by pointing to self
        nonlocal_root = [root]  # make updateable non-locally
        PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

        if maxsize == 0:
            def wrapper(*args, **kwds):
                # no caching, just do a statistics update after a successful call
                result = viewfunc(*args, **kwds)
                stats[MISSES] += 1
                return result

        elif maxsize is None:
            def wrapper(*args, **kwds):
                t_args = args if args_base == 0 else args[1:]
                # simple caching without ordering or size limit
                key = _make_key(t_args, kwds, typed)

                result = cache_get(key, root)  # root used here as a unique not-found sentinel

                if result is not root:
                    old_time = result[1]

                    if timeout is not None:
                        if (int(time.time()) - old_time) <= timeout:
                            stats[HITS] += 1
                            return result[0]
                    else:
                        stats[HITS] += 1
                        return result[0]

                cache[key] = result = viewfunc(*args, **kwds), int(time.time())
                stats[MISSES] += 1
                return result[0]

        else:
            def wrapper(*args, **kwds):
                t_args = args if args_base == 0 else args[1:]
                # size limited caching that tracks accesses by recency
                key = _make_key(t_args, kwds, typed) if kwds or typed else t_args

                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # record recent use of the key by moving it to the front of the list
                        root, = nonlocal_root
                        link_prev, link_next, key, result = link

                        old_time = result[1]

                        if timeout is not None:
                            if (int(time.time()) - old_time) <= timeout:
                                link_prev[NEXT] = link_next
                                link_next[PREV] = link_prev
                                last = root[PREV]
                                last[NEXT] = root[PREV] = link
                                link[PREV] = last
                                link[NEXT] = root
                                stats[HITS] += 1

                                return result[0]
                        else:
                            link_prev[NEXT] = link_next
                            link_next[PREV] = link_prev
                            last = root[PREV]
                            last[NEXT] = root[PREV] = link
                            link[PREV] = last
                            link[NEXT] = root
                            stats[HITS] += 1

                            return result[0]

                result = viewfunc(*args, **kwds), int(time.time())

                with lock:
                    root, = nonlocal_root
                    stats[MISSES] += 1
                    #                     if key in cache:
                    #                         pass
                    if _len(cache) >= maxsize:
                        # use the old root to store the new key and result
                        oldroot = root
                        oldroot[KEY] = key

                        oldroot[RESULT] = result

                        # empty the oldest link and make it the new root
                        root = nonlocal_root[0] = oldroot[NEXT]
                        oldkey = root[KEY]
                        oldvalue = root[RESULT]
                        root[KEY] = root[RESULT] = None
                        # now update the cache dictionary for the new links
                        if oldkey in cache:
                            del cache[oldkey]
                        cache[key] = oldroot
                    else:
                        # put result in a new link at the front of the list
                        last = root[PREV]
                        link = [last, root, key, result]
                        last[NEXT] = root[PREV] = cache[key] = link

                return result[0]

        def cache_info():
            """Report cache statistics"""
            with lock:
                return _CacheInfo(stats[HITS], stats[MISSES], maxsize, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            with lock:
                cache.clear()
                root = nonlocal_root[0]
                root[:] = [root, root, None, None]
                stats[:] = [0, 0]

        wrapper.__wrapped__ = viewfunc
        #         wrapper.cache_info = cache_info
        #         wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, viewfunc)

    return _cache_controller
