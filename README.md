# Final-project

## Quick start

<details>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>


<details>
<summary>Training & Inference & Evaluation</summary>


- Training on Multiple GPUs(default with 4 gpus):

```shell
# train on multi-gpu
sh Classifier3D/tools/run_swinunetr.sh
```

- Inference on Multiple GPUs(default with 4 gpus):

```shell
# inference on multi-gpu
sh Classifier3D/tools/run_test_swinunetr.sh
```

- Evaluation:

```shell

python Eval/eval.py
python Eval/WeightedLogloss.py
```

</details>


<details>
<summary>Deploy</summary>

```shell
python Deploy/deploy.py
```

</details>


## C++ API

<details>
<summary>Requirements</summary>

- Libtorch
- InsightToolkit-5.4rc01
- Cmake(>=3.8)

```shell
cd FinalProject_C && makedir build
cd build
cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch/ .. && make
./Test epoch_82.pt data.nii.gz
```

</details>