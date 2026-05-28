# GLM5 run_prefill nsys capture

This note records the working flow used for case 712 on the B200 pod, and the files synced back to `.8`.

## Paths

B200 pod:

```bash
NS=kimiz-glm5
POD=glm5-jsonl-b200-piecewise-rebench-20260526
RUN_DIR=/workspace/cache/refactor_test_aic/aic_infer_cmp_nsys_profile
DATA=/workspace/cache/results/case712_prefill_data
MODEL=/workspace/cache/huggingface/hub/models--nvidia--GLM-5-NVFP4/snapshots/local
OUT=/workspace/cache/results/case712_prefill_analysis_20260526_targetnsys
```

`.8` synced outputs:

```bash
/raid/kimi/ds4_new/b200_glm5_pccg_data/nsys/case712_prefill_target/
  case712_target_node.nsys-rep
  case712_target_node.sqlite
  run_prefill_case712_targetnsys.log
```

`.8` synced replay script:

```bash
/raid/kimi/ds4_new/refactor_test_aic/aic_infer_cmp_nsys_profile/run_prefill_glm5.py
```

## run_prefill_glm5.py requirements

The GLM5 replay script must use piecewise CUDA graph, while normal CUDA graph is disabled:

```python
disable_cuda_graph = True
disable_piecewise_cuda_graph = False
enforce_piecewise_cuda_graph = True
```

Piecewise token count must be the smallest multiple of 8 greater than or equal to the target extend token count:

```python
piecewise_capture_tokens = ((total_extend_tokens + 7) // 8) * 8
```

For case 712:

```text
actual extend tokens = 16331
piecewise tokens     = 16336
```

## Working nsys command

The command that successfully produced `.nsys-rep` and `.sqlite` was target-only capture with CUDA graph node tracing:

```bash
kubectl -n kimiz-glm5 exec glm5-jsonl-b200-piecewise-rebench-20260526 -- bash -lc '
set -euo pipefail
OUT=/workspace/cache/results/case712_prefill_analysis_20260526_targetnsys
DATA=/workspace/cache/results/case712_prefill_data
MODEL=/workspace/cache/huggingface/hub/models--nvidia--GLM-5-NVFP4/snapshots/local
mkdir -p "$OUT"
rm -f "$OUT"/case712_target_node.* "$OUT"/run_prefill_case712_targetnsys.log "$OUT"/run_prefill.exit
cd /workspace/cache/refactor_test_aic/aic_infer_cmp_nsys_profile
export PYTHONPATH=/sgl-workspace/sglang/python:/workspace/cache/refactor_test_aic:/workspace/cache/refactor_test_aic/aic_infer_cmp_nsys_profile:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --kill=sigkill \
  --flush-on-cudaprofilerstop=false \
  --export=sqlite \
  -o "$OUT/case712_target_node" \
  python3 run_prefill.py \
    --data-dir "$DATA" \
    --csv-case-id 712 \
    --csv-path "$DATA/signed_error/aic_vs_measured_signed_error_cases.csv" \
    --model "$MODEL" \
    --tp-size 8 \
    --ep-size 1 \
  > "$OUT/run_prefill_case712_targetnsys.log" 2>&1

code=$?
echo $code > "$OUT/run_prefill.exit"
echo OUT=$OUT
ls -lh "$OUT"
'
```

Important: this may stop at `RangeGeneration` after the target batch finishes. If files are not generated, force the session to stop:

```bash
kubectl -n kimiz-glm5 exec glm5-jsonl-b200-piecewise-rebench-20260526 -- bash -lc '
nsys sessions list
nsys stop --session=profile-30763
'
```

Use the actual session name from `nsys sessions list`; in the successful run it was `profile-30763`.

After `nsys stop`, the generated files were:

```text
/workspace/cache/results/case712_prefill_analysis_20260526_targetnsys/case712_target_node.nsys-rep
/workspace/cache/results/case712_prefill_analysis_20260526_targetnsys/case712_target_node.sqlite
```

## Sync outputs back to `.8`

```bash
TMP=/tmp/case712_target_nsys
rm -rf "$TMP"
mkdir -p "$TMP"

kubectl -n kimiz-glm5 cp \
  glm5-jsonl-b200-piecewise-rebench-20260526:/workspace/cache/results/case712_prefill_analysis_20260526_targetnsys/case712_target_node.nsys-rep \
  "$TMP/case712_target_node.nsys-rep"

kubectl -n kimiz-glm5 cp \
  glm5-jsonl-b200-piecewise-rebench-20260526:/workspace/cache/results/case712_prefill_analysis_20260526_targetnsys/case712_target_node.sqlite \
  "$TMP/case712_target_node.sqlite"

kubectl -n kimiz-glm5 cp \
  glm5-jsonl-b200-piecewise-rebench-20260526:/workspace/cache/results/case712_prefill_analysis_20260526_targetnsys/run_prefill_case712_targetnsys.log \
  "$TMP/run_prefill_case712_targetnsys.log"

ssh root@10.6.131.8 'mkdir -p /raid/kimi/ds4_new/b200_glm5_pccg_data/nsys/case712_prefill_target'

scp "$TMP"/* root@10.6.131.8:/raid/kimi/ds4_new/b200_glm5_pccg_data/nsys/case712_prefill_target/
```

## Sync replay script back to `.8`

```bash
kubectl -n kimiz-glm5 cp \
  glm5-jsonl-b200-piecewise-rebench-20260526:/workspace/cache/refactor_test_aic/aic_infer_cmp_nsys_profile/run_prefill.py \
  /tmp/run_prefill_glm5.py

scp /tmp/run_prefill_glm5.py \
  root@10.6.131.8:/raid/kimi/ds4_new/refactor_test_aic/aic_infer_cmp_nsys_profile/run_prefill_glm5.py
```

Verify the piecewise rounding on `.8`:

```bash
ssh root@10.6.131.8 \
  'grep -n "piecewise_capture_tokens" /raid/kimi/ds4_new/refactor_test_aic/aic_infer_cmp_nsys_profile/run_prefill_glm5.py'
```

Expected:

```text
piecewise_capture_tokens = ((total_extend_tokens + 7) // 8) * 8
```

## What did not work

These attempts ran the replay but did not produce a usable `.nsys-rep`:

1. `--trace=cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop`
   - The target batch ran, but the process hung at `cudaProfilerStop()` / `RangeGeneration`.

2. `--capture-range=nvtx --nvtx-capture "Scheduler.run_batch: {...}"`
   - The replay ran, but nsys stayed in `StartRange`; the PyTorch NVTX range did not trigger capture.

3. `--capture-range=none` full-process capture
   - Produced a very large qdstrm and got stuck in `Generation`.

The working path is target-only `cudaProfilerApi` capture plus explicit `nsys stop --session=...` if report generation does not finish automatically.

## Case 712 sanity checks

Expected log lines:

```text
[target] bs=2, latency=1310.249ms
req[0] prefix=110784 extend=2699
req[1] prefix=0 extend=13632
[piecewise] actual extend tokens=16331
[piecewise] cuda graph tokens=[16336]
[server_args] ... 'disable_cuda_graph': True ... 'disable_piecewise_cuda_graph': False ... 'enforce_piecewise_cuda_graph': True ...
[profile] cudaProfilerStart
[replay_run_batch] {'bs': 2, 'forward_mode': 'EXTEND', 'input_length': [16331], 'past_kv_length': [97152]} latency_ms=1461-1463
Prefill batch, #new-seq: 2, #new-token: 16384, #cached-token: 110784, ... cuda graph: True
```

## Case 712 AIC attention module piecewise graph capture

This is the valid module-level AIC attention collection path for GLM5 case 712.
Do not use a timeline unless sqlite proves graph replay.

Patched collector on `.8`:

```text
/raid/kimi/ds4_new/aiconfigurator/collector/sglang/collect_mla_module.py
```

Required behavior:

1. Set `AIC_ENABLE_MODULE_PIECEWISE_REPLAY=1`.
2. Set `AIC_ENABLE_PIECEWISE_CUDA_GRAPH=1`.
3. Set `AIC_PIECEWISE_CUDA_GRAPH_TOKENS` to the smallest multiple of 8 that is
   greater than or equal to the real extend token count.
4. Capture module piecewise graph with the real module `ForwardBatch`, not the
   default synthetic `bs=1,prefix=0` batch.
5. Run `model_runner.init_piecewise_cuda_graphs()` under `torch.no_grad()` so
   the Dynamo grad-mode guard matches replay.
6. Run nsys with `--cuda-graph-trace=node`, then export sqlite and verify
   non-zero `cudaGraphLaunch`, `graphId`, and `graphNodeId`.

Known-good B200 validation for case 712 attn module:

```text
bs=2
seq_length=8165
prefix/past_kv=55392
piecewise tokens=16336
SGLang commit=bc8d64bf36c687580ea9d4dc17fed8bcd8e62395
total_kernels 851
graph_kernels 96
graph_nodes 24
graph_ids 3
cudaGraphLaunch_v10000 12
```

Local `.nsys-rep` copied from B200:

```text
/home/scratch.kimiz_gpu_2/docker_v/agent_work_space/nsys/case712/repro/modules/default_mem_fraction_module_pcg_real_batch_nograd/case712_aic_attn_module_bs2_s8165_prefix55392_default_mfs_module_pcg_real_batch_nograd_piecewise16336_bc8d64bf_node.nsys-rep
```
