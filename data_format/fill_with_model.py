# -*- coding: utf-8 -*-
import re
import sys


def fill_model(model_str, params):
    tofill = get_paras(model_str)
    for i in tofill:
        if i in params:
            model_str = model_str.replace('${'+i+'}', params[i])
    return model_str


def get_paras(model_str):
    reobj = re.compile('\\$\\{(.*?)\\}')
    li = reobj.findall(model_str)
    return li


def fill_with_tsv(file_name, sql, sep='\t'):
    seg_num = 0
    is_first_line = True
    with open(file_name) as fp:
        for line in fp:
            segs = line.strip('\n').split(sep)
            if is_first_line:
                seg_num = len(segs)
                is_first_line = False
            else:
                if len(segs) != seg_num:
                    continue
            para = {}
            for i in range(seg_num):
                para[str(i)] = segs[i]
            print(fill_model(sql, para))


# 以下这些是例子
sql = "update label set is_deleted=1 where id=${0};"
sql = "update label set definition='${3}',value_type=${4},rule='${5}', produce_way='${6}',explanation='${7}' where id=${0};"
sql = "INSERT INTO `label`(`node_id`, `label_name`, `definition`, `value_type`, `rule`, `produce_way`, `explanation`, `update_unit`) VALUES (${1},'${2}','${3}',${4},'${5}','${6}','${7}',${8});"
sql = "update label_indicator set cron_expr='${6}' where label_id=${0};"
sql = "update label_indicator set indicator_description='${3}', indicator_standard='${4}',indicator_sql=\"${5}\" where label_id=${0};"
sql = "update label_indicator set indicator_description='${2}', indicator_standard='${3}',indicator_sql=\"${4}\" where label_id=${0};"
sql = "INSERT INTO `label_indicator`(`label_id`, `indicator_name`, `indicator_description`, `indicator_standard`, `indicator_sql`, `result_type`, `cron_expr`, `is_shown_in_label`) VALUES (${0}, '${2}', '${3}', '${4}', \"${5}\", ${6}, '${7}', ${8});"
sql = "INSERT INTO `label_indicator`(`label_id`, `indicator_name`, `indicator_description`, `indicator_standard`, `indicator_sql`, `result_type`, `cron_expr`, `is_shown_in_label`) VALUES (${0}, '${1}', '${2}', '${3}', \"${4}\", ${5}, '${6}', ${7});"
sql = "update label_indicator set is_deleted=1 where id=${0};"
sql = "update label_indicator set indicator_standard='${1}' where label_id=${0};"
sql = "update label_indicator set indicator_description='${3}', indicator_standard='${4}',indicator_sql=\"${5}\" where id=${0};"
sql = "update label_indicator set indicator_standard='${1}' where id=${0};"
sql = "update label_indicator set indicator_description='${3}' where id=${0};"
sql = "update label_indicator set indicator_description='${3}', indicator_sql=\"${5}\" where id=${0};"
sql = "update label_indicator set cron_expr='${2}' where id=${0};"
sql = "d.node(name='${1}', label='${1}', color='red')"
sql = "create_edge(6, ${0}, ${1}, True, '${2}')"
sql = "create_node(3, '${0}', '${1}')"
sql = "d.node(name='${0}', label='${0}', color='red', style='filled')"
sql = "d.node(name='${0}', label='${0}', width='${1}', height='${1}')"
sql = "d.edge('${0}', '${1}', label='${2}%')"
sql = "'${0}',"
fill_with_tsv(sys.argv[1], sql)
