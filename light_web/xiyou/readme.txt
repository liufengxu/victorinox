部署时需要把easy64拷贝到项目目录下

执行结果：
(work) liufengxu@liufengxudeMacBook-Pro:~/PycharmProjects/test$	curl -d '{"content": "BailongMa"}' http://127.0.0.1:8085/score
{
  "code": 0, 
  "data": {
    "name": "BailongMa", 
    "score": 60
  }, 
  "duration": 2.741, 
  "message": "success"
}
(work) liufengxu@liufengxudeMacBook-Pro:~/PycharmProjects/test$     curl -d '{"content": "BailongMao"}' http://127.0.0.1:8085/score
{
  "code": 0, 
  "data": {
    "name": "BailongMao", 
    "score": -1
  }, 
  "duration": 2.938, 
  "message": "success"
}

(work) liufengxu@liufengxudeMacBook-Pro:~/PycharmProjects/test$     curl http://127.0.0.1:8085/alive
{
  "code": 0, 
  "duration": 0.054, 
  "message": 200
}

(work) liufengxu@liufengxudeMacBook-Pro:~/PycharmProjects/test$	curl -d '{"content": "aSd+|"}' http://127.0.0.1:8085/evaluate
{
  "code": 0, 
  "data": {
    "length": 5, 
    "result": "cc]w)m"
  }, 
  "duration": 0.44, 
  "message": "success"
}

