# WAGMA-SGD
WAGMA-SGD is a decentralized asynchronous SGD based on wait-avoiding group model averaging. The synchronization is relaxed by making the collectives externally-triggerable, namely, namely, a collective can be initiated without requiring that all the processes enter it. It partially reduces the data within non-overlapping groups of process, improving the parallel scalability.


Demo
---------



Publication
-----------

The work of WAGMA-SGD is pulished in TPDS'21. Cite us:
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
