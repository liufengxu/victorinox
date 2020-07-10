# -*- coding:utf-8 -*-
import flask
from flask import Flask
import traceback
import json
import datetime as dt
import logging

import easy64
from light_web.connector import query_ad

app = Flask(__name__)


def get_duration():
    return (dt.datetime.now() - flask.g.start).total_seconds() * 1000


def get_logging_msg(result, err=None):
    url = flask.request.url
    method = flask.request.method
    msg = result.copy()
    if err is not None:
        msg["err"] = str(err)
        msg["err_trace"] = traceback.format_exc()
    return '\t'.join((url, method, json.dumps(msg)))


@app.before_request
def before_request():
    flask.g.start = dt.datetime.now()


@app.route('/evaluate', methods=['POST'])
def evaluate():
    input_json = flask.request.get_json(force=True, silent=True)
    if not input_json or 'content' not in input_json:
        message = "参数错误"
        data = []
    else:
        content = input_json.get('content')
        length = len(content)
        content = easy64.trans(content)
        data = {
            "result": content,
            "length": length
        }
        message = "success"
    resp = {
        "code": 0,
        "message": message,
        "duration": get_duration(),
        "data": data
    }
    logging.error(resp)
    return flask.jsonify(resp)


@app.route('/score', methods=['POST'])
def score():
    input_json = flask.request.get_json(force=True, silent=True)
    if not input_json or 'content' not in input_json:
        message = "参数错误"
        data = []
    else:
        content = input_json.get('content')
        result_score = query_ad("select score from xiyou_player where name = '{0}'".format(content))
        rs = -1
        if not result_score.empty:
            rs = int(result_score['score'][0])
        data = {
            "score": rs,
            "name": content
        }
        message = "success"
    resp = {
        "code": 0,
        "message": message,
        "duration": get_duration(),
        "data": data
    }
    logging.error(resp)
    return flask.jsonify(resp)


@app.route('/alive', methods=['GET'])
def heartbeat():
    resp = {
        "code": 0,
        "message": 200,
        "duration": get_duration()
    }
    return flask.jsonify(resp)


if __name__ == '__main__':
    app.run(debug=True, port=8085)
