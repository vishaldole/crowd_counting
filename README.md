
## Datasets

The models are trained and evaluated on the following datasets:

1. **UCF50**: A crowd counting dataset.
2. **QNRF**: A large-scale crowd counting dataset.
3. **Shanghai_A**: A dataset part of the ShanghaiTech dataset.
4. **Shanghai_B**: Another dataset part of the ShanghaiTech dataset.

## Models

### Original FPANet Model

![approach_image](https://github.com/user-attachments/assets/ebc5e815-1e6e-441b-b527-6a7bf2f78ef9)

The original FPANet model is implemented in the following scripts:

- `ucf50.py`
- `qnrf.py`
- `shanghai_a.py`
- `shanghai_b.py`

### Enhanced FPANet Model

![noval_model drawio](https://github.com/user-attachments/assets/2dcd9004-6d3b-4fc8-9494-6d16705f3bc2)

The enhanced FPANet model is implemented in the following scripts:

- `noval_ucf50.py`
- `noval_qnrf.py`
- `noval_shanghai_a.py`
- `noval_shanghai_b.py`

## Graph Generation

Each model script has a corresponding script for generating and saving graphs of the results:

- `ucf50_show_graph.py`
- `qnrf_show_graph.py`
- `shanghai_a_show_graph.py`
- `shanghai_b_show_graph.py`

The generated graphs are saved in the `results` folder under directories named after the respective model scripts.
