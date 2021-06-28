# WAGMA-SGD
WAGMA-SGD is a decentralized asynchronous SGD based on wait-avoiding group model averaging. The synchronization is relaxed by making the collectives externally-triggerable, namely, a collective can be initiated without requiring that all the processes enter it. It partially reduces the data within non-overlapping groups of process, improving the parallel scalability.


Demo
---------
The wait-avoiding group allreduce operation is implemented in `./wagmaSGD-modules/fflib3/`. To use it, simply configure and compile fflib3 as to an .so library by conducting `cmake ..` and `make` in the directory `./wagmaSGD-modules/fflib3/lib/`.




Publication
-----------

The work of WAGMA-SGD is pulished in TPDS'21. If you use WAGMA-SGD, cite us:
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
