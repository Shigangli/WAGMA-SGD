# WAGMA-SGD
**WAGMA-SGD** is a **decentralized asynchronous** SGD for distributed deep learning training based on **model averaging**. The key idea of WAGMA-SGD is to use a novel wait-avoiding group allreduce to average the models among processes. The synchronization is relaxed by making the collectives externally-triggerable, namely, a collective can be initiated without requiring that all the processes enter it. Thus, it can better handle the deep learning training with load imbalance. Since WAGMA-SGD only reduces the data within non-overlapping groups of process, it significantly improves the parallel scalability. WAGMA-SGD may bring staleness to the weights. However, the **staleness is bounded**. WAGMA-SGD is based on model averaging, rather than gradient averaging. Therefore, after the periodic synchronization is conducted, it guarantees a consistent model view amoung processes.


Demo
---------
The wait-avoiding group allreduce operation is implemented in `./WAGMA-SGD-modules/fflib3/`. To use it, simply configure and compile fflib3 as to an .so library by conducting `cmake ..` and `make` in the directory `./WAGMA-SGD-modules/fflib3/lib/`. A script to run WAGMA-SGD on ResNet-50/ImageNet with SLURM job scheduler can be found [here](https://github.com/Shigangli/WAGMA-SGD/blob/main/test-models/tf-models-r1.11/official/resnet/test_imagenet_scripts/daint_imagenet_wagma_sgd.sh).
Generally, to evaluate other neural network models with the [customized optimizers](https://github.com/Shigangli/WAGMA-SGD/tree/main/test-models/tf-models-r1.11/official/utils) (e.g., wait-avoiding group allreduce), one can simply wrap the default optimizer using the customized optimizers. See the example for ResNet-50 [here](https://github.com/Shigangli/WAGMA-SGD/blob/main/test-models/tf-models-r1.11/official/resnet/resnet_run_loop_wagma_sgd.py#L386).

For the deep learning tasks implemented in TensorFlow, we implemented custom C++ operators, in which we may call the wait-avoiding group allreduce operation or other communication operations (according to the specific parallel SGD algorithm) to average the models. Next, we register the C++ operators to TensorFlow, which can then be used to build the TensorFlow computational graph to implement the SGD algorithms. Similarly, for the deep learning tasks implemented in PyTorch, one can utilize pybind11 to call C++ operators in Python.

Publication
-----------

The work of WAGMA-SGD is pulished in TPDS'21. See the [paper](https://shigangli.github.io/files/wagmaSGD.pdf) for details. To cite our work:
```bibtex
@ARTICLE{9271898,
  author={Li, Shigang and Ben-Nun, Tal and Nadiradze, Giorgi and Girolamo, Salvatore Di and Dryden, Nikoli and Alistarh, Dan and Hoefler, Torsten},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  title={Breaking (Global) Barriers in Parallel Stochastic Optimization With Wait-Avoiding Group Averaging},
  year={2021},
  volume={32},
  number={7},
  pages={1725-1739},
  doi={10.1109/TPDS.2020.3040606}}
```

License
-------
See [LICENSE](LICENSE).
