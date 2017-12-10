import datetime
import json
import re

import numpy as np

DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
DATE_FMT = "%Y-%m-%d"

# raw_sql = '''    not (year < 10 or date == '2010-06-05 00:00:00') or (day>10) and not ( day == 10 and day ==23  )  and month in [24,    45, 654]   '''

# raw_sql = ''' (date == 0.5) and ( year not in [24, 654]  ) or not ( day == 1 and month > 3)  '''

# raw_sql = ''' x>1 and ((x>1) or ( not (    x   <   2) and x==34) or (z==1))'''

COND_TOKENS = ["date", "year", "month", 'day', "hour", "minute"]
COND_OPCODES = ["in", "notin", ">=", "<=", ">", "<", "==", "!="]
COND_LOGICALS = ["and", "or"]


class CondParser(object):
    def __init__(self, raw_sql):

        raw_sql = re.sub("==", " == ", raw_sql)
        raw_sql = re.sub(">", " > ", raw_sql)
        raw_sql = re.sub("<", " < ", raw_sql)
        raw_sql = re.sub(">=", " >= ", raw_sql)
        raw_sql = re.sub("<=", " <= ", raw_sql)

        raw_sql = re.sub("\s+", " ", raw_sql)
        raw_sql = re.sub(r",\s+", ",", raw_sql)
        raw_sql = re.sub(r":\s+", ":", raw_sql)
        raw_sql = re.sub(r"not\s+in", "notin", raw_sql)
        raw_sql = re.sub(r"\(\s+", "(", raw_sql)
        raw_sql = re.sub(r"\s+\)", ")", raw_sql)

        #         print(raw_sql)
        self.raw_sql = raw_sql

    def _get_brackets(self, s):
        left = 1
        for idx, c in enumerate(s):
            if c == "(": left += 1
            if c == ")": left -= 1
            if left == 0: return idx, s[:idx]
        raise ValueError("invalid expr: (" + s.replace("notin", "not in"))

    def parse(self):
        rst = []
        self._parse(self.raw_sql, rst)
        return rst

    def validate(self, s, cond):
        l = len(cond)
        if l == 1:
            raise ValueError("invalid expr: " + s)
        if l == 2:
            opcode, value = cond
            if opcode != "not":
                raise ValueError("invalid expr: " + " ".join(cond))
            if not isinstance(value, list):
                raise ValueError("invalid expr: " + " ".join(cond))
            return

        if l > 3:
            cond_r = cond[2:]
            token = cond[0];
            opcode = cond[1]
            #             cond.clear( )
            cond[:] = [cond[0], cond[1], " ".join(cond_r)]
            l = len(cond)

        if l == 3:
            token, opcode, value = cond
            if token not in COND_TOKENS or opcode not in COND_OPCODES:
                raise ValueError("invalid expr: " + " ".join(cond))
            done = False;
            value = cond[-1];
            if len(value) > 1:
                t0 = value[0];
                t1 = value[-1]
                if t0 in ['"', "'"] and t1 in ['"', "'"]:
                    value = value[1:-1]  # is string
                    done = True
                elif t0 == "[" and t1 == "]":
                    try:
                        if value.find("{") >= 0: raise
                        value = value.replace("'", '"')
                        value = json.loads(value)  # is array
                    except:
                        raise ValueError("invalid expr: " + value)
                    if opcode not in ["in", "notin"]:
                        raise ValueError("invalid expr: " + opcode + " " + json.dumps(value))
                    done = True
            if done == False:
                if value.find(".") >= 0:
                    try:
                        value = float(value)  # is float
                    except:
                        raise ValueError("invalid expr: " + value)
                else:
                    try:
                        value = int(value)  # is int
                    except:
                        raise ValueError("invalid expr: " + value)
            cond[-1] = value
            if opcode in ["in", "notin"]:
                if not isinstance(value, list):
                    raise ValueError("invalid expr: " + opcode + " " + json.dumps(value))
            if opcode == "notin": cond[1] = "not in"
            return
        raise ValueError("invalid expr: " + s)

    def _parse(self, s, rst):
        s = s.strip();
        l = len(s)
        beg, idx, tmp = 0, 0, []
        while idx < l:
            c = s[idx]
            if c == "(":
                pos, token = self._get_brackets(s[idx + 1:])
                sub_cond = []
                self._parse(token, sub_cond)
                tmp.append(sub_cond)
                idx = idx + pos + 3;
                beg = idx
            elif c == " ":
                token = s[beg:idx]
                idx += 1;
                beg = idx
                tmp.append(token)
            else:
                idx += 1

        if beg < idx:
            tmp.append(s[beg:idx])

        idx = 0;
        l = len(tmp)
        if l == 0: return
        if l == 1:
            token = tmp[0]
            if isinstance(token, list):
                rst.extend(token)
                return
            else:
                raise ValueError("invalid expr: " + s.replace("notin", "not in"))

        cond = [];
        token = None
        while idx < l:
            token = tmp[idx]
            if isinstance(token, list):
                rst.append(token)
                if idx + 1 < l:
                    opcode = tmp[idx + 1]
                    if opcode not in COND_LOGICALS:
                        raise ValueError("invalid expr: " + json.dumps(token) + " " + opcode)
                    rst.append(opcode)
                idx += 2
            elif token == "not":
                value = tmp[idx + 1]
                if not isinstance(value, list):
                    raise ValueError("invalid expr: not " + value)
                rst.append(["not", value])
                if idx + 2 < l:
                    opcode = tmp[idx + 2]
                    if opcode not in COND_LOGICALS:
                        raise ValueError("invalid expr: " + json.dumps(token) + " " + opcode)
                    rst.append(opcode)
                idx += 3
            elif token in COND_LOGICALS:
                self.validate(s, cond)
                rst.append(cond)
                rst.append(token)
                cond = []
                idx += 1
            else:
                cond.append(token)
                idx += 1

        if len(cond) > 0:
            self.validate(s, cond)
            if len(rst) == 0:
                rst.extend(cond)
            else:
                rst.append(cond)

        if len(rst) > 0:
            last = rst[-1]
            if last in COND_LOGICALS:
                raise ValueError("invalid expr: " + s.replace("notin", "not in"))


def _parse_dates(times):
    '''
    如果 times 为 datetime，返回 [ datetime ]
    如果 times 为 %Y-%m-%d %H:%M:%S，返回 [ datetime ]
    '''
    if times is None: return []

    if not isinstance(times, (tuple, list)):
        times = [times]

    rets = []
    for atime in times:
        if isinstance(atime, datetime.datetime):
            rets.append(atime)
        else:
            stime = atime.strip()
            try:
                atime = datetime.datetime.strptime(stime, DATE_FMT)
            except:
                atime = datetime.datetime.strptime(stime, DATETIME_FMT)
            rets.append(atime)
    return rets


# print(CondParser(raw_sql).parse())

class TimeSlice(object):
    def __init__(self, timeval=None, ):
        '''
        timeval 为 datetime, datestr, timerange 对象 或 列表    
        [ ["key", "opcode", "val"], "and/or", [ "not", ["key", "opcode", "val"]] ]
        if key == "in", val 可以为 ["value1", "value2"]
        else:  val 可以为 value 或 ["key", "opcode", "val"]
        
        key    : "date", "year", "month", 'day', "hour", "minute"
        opcode3: "in", "not in", ">=", "<=", ">", "<", "==", "!="
        opcode2: "not"
                
        '''
        self.timeval = timeval

    def cmp_oper(self, key, opcode, val, ncdataset, cdftime):
        ''' 
        key    : "date", "year", "month", 'day', "hour", "minute"
        opcode3: "in", "not in", ">=", "<=", ">", "<", "==", "!="
        opcode2: "not"
        '''

        grid_times = ncdataset.variables["times"]
        grid_years = ncdataset.variables["years"]
        grid_months = ncdataset.variables["months"]
        grid_days = ncdataset.variables["days"]
        grid_hours = ncdataset.variables["hours"]
        grid_minutes = ncdataset.variables["minutes"]

        def get_cubeval(key):
            if key == "date":
                return grid_times[:]
            elif key == "year":
                return grid_years[:]
            elif key == "month":
                return grid_months[:]
            elif key == "days":
                return grid_days[:]
            elif key == "hours":
                return grid_hours[:]
            elif key == "minutes":
                return grid_minutes[:]
            raise ValueError("invalid key: %s" % key)

        if opcode in ["in", "not in"]:  # 如果是 in/not in 操作符，val 为数值数组
            if not isinstance(val, (tuple, list)):
                raise ValueError("require list/tuple: %s %s %s" % (key, opcode, str(val)))

            cubeval = get_cubeval(key)
            if key == "date":
                timeval = np.array(_parse_dates(val))
                num_timeval = cdftime.date2num(timeval)
            else:
                num_timeval = np.array(list(map(lambda a: int(a), val)))

            ix = np.in1d(cubeval.ravel(), num_timeval).reshape(cubeval.shape)
            if opcode == "in":
                return ix
            else:
                return np.logical_not(ix)

        if isinstance(val, (tuple, list)):
            raise ValueError("require string: %s %s %s" % (key, opcode, str(val)))

        cubeval = get_cubeval(key)
        if key == "date":
            timeval = np.array(_parse_dates(val))
            num_timeval = cdftime.date2num(timeval)[0]
        else:
            num_timeval = int(val)

        if opcode == ">=":
            return cubeval >= num_timeval
        elif opcode == "<=":
            return cubeval <= num_timeval
        elif opcode == ">":
            return cubeval > num_timeval
        elif opcode == "<":
            return cubeval < num_timeval
        elif opcode == "==":
            return cubeval == num_timeval
        elif opcode == "!=":
            return cubeval != num_timeval

        raise ValueError("invalid expr: %s %s %s" % (key, opcode, str(val)))

    def log_oper(self, expr1, opcode, expr2, ncdataset, cdftime):
        '''
        opcode : "and", "or"
        '''
        if not isinstance(expr1, (tuple, list)):
            raise ValueError("require list/tuple: %s %s" % (str(expr1), opcode))
        if not isinstance(expr1, (tuple, list)):
            raise ValueError("require list/tuple: %s %s" % (opcode, str(expr2),))

        ix1 = self.one_oper(expr1, ncdataset, cdftime)
        ix2 = self.one_oper(expr2, ncdataset, cdftime)

        if opcode == "and":
            return np.logical_and(ix1, ix2)
        elif opcode == "or":
            return np.logical_or(ix1, ix2)
        raise ValueError("invalid expr: %s %s %s" % (str(expr1), opcode, str(expr2)))

    def not_oper(self, opcode, val, ncdataset, cdftime):
        if isinstance(val, (tuple, list)):  # not 操作符, val 为表达式，需要重新迭代计算
            ix = self.one_oper(val, ncdataset, cdftime)
            return np.logical_not(ix)

        raise ValueError("require list/tuple: %s %s" % (opcode, str(val)))

    def one_oper(self, expr, ncdataset, cdftime):
        '''        
        [ ["date", "in", ['2010-08-08', '2010-09-25']] , "or" , ["not", [ ["date", ">=", '2010-09-01 12:00:00' ] , "or" , ["month", "==", 9 ] ] ] ] 
        
        not: len(expr) == 2
        in, not in, >=, <=, >, <, ==, !=, and, or : len(expr) == 3
        array                        : others 
        '''
        l = len(expr)
        while l == 1:
            expr = expr[0]
            l = len(expr)

        if l == 2:  # 如果长度为 2，则为 not 数组
            opcode, val = expr
            if opcode == "not":
                rst2 = self.not_oper(opcode, val, ncdataset, cdftime)
            else:
                raise ValueError(str(expr))

        elif l == 3:
            key, opcode, val = expr
            if opcode in ["and", "or"]:
                rst2 = self.log_oper(key, opcode, val, ncdataset, cdftime)

            elif opcode in ["in", "not in", ">=", "<=", ">", "<", "==", "!="]:

                if key not in ["date", "year", "month", 'day', "hour", "minute"]:
                    raise ValueError("bad key: %s %s %s" % (key, opcode, str(val)))

                rst2 = self.cmp_oper(key, opcode, val, ncdataset, cdftime)
            else:
                raise ValueError(str(expr))

        else:
            if l % 2 == 0:
                raise ValueError(str(expr))

            for pos in range(1, l, 2):
                opcode = expr[pos]
                if opcode not in ["and", "or"]:
                    raise ValueError(str(expr))

            expr0 = expr[0]
            opcode = expr[1]
            expr1 = expr[2]

            rst2 = self.log_oper(expr0, opcode, expr1, ncdataset, cdftime)

            for pos in range(3, l, 2):
                opcode = expr[pos]
                expr1 = expr[pos + 1]

                if not isinstance(expr1, (tuple, list)):
                    raise ValueError("require list/tuple: %s %s" % (str(expr1), opcode))

                ix1 = self.one_oper(expr1, ncdataset, cdftime)

                if opcode == "and":
                    rst2 = np.logical_and(ix1, rst2)
                elif opcode == "or":
                    rst2 = np.logical_or(ix1, rst2)
        return rst2

    def get_slices(self, ncdataset, cdftime):
        if not self.timeval:
            return []

        #         if isinstance(self.timeval, [ bytes ]):
        #             self.timeval = self.timeval.encode("utf-8")

        if not isinstance(self.timeval, (tuple, list)):
            self.timeval = CondParser(self.timeval).parse()

        ix = self.one_oper(self.timeval, ncdataset, cdftime)

        return ix
