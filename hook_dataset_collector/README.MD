* 服务启动，注意 `--disable-overlap-schedule` 这个是必须的 
``` sh
HF_ENDPOINT=https://hf-mirror.com \
SGL_HOOK_REQ_INFO_DIR=`pwd`/data \
python3 ./sglang_launch_server.py \
    --model-path="Qwen/Qwen3-30B-A3B" \
    --disable-overlap-schedule \
    --tp 2 --ep 2 \
    --load-format=dummy
```

* 压测
```
HF_ENDPOINT=https://hf-mirror.com \
python3 -m sglang.bench_serving --backend sglang \
        --dataset-name sharegpt --num-prompts 2000 \
        --request-rate 100 \
        --host 127.0.0.1
```

* 导出数据（接口截获了`start_profile`，调用接口，通知服务端导出数据），数据会保存在 `SGL_HOOK_REQ_INFO_DIR` 设定的目录 
* 不触发接口不会有数据导出！
```
curl http://localhost:30000/start_profile
```