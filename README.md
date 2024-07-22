
## Datasets

The models are trained and evaluated on the following datasets:

1. **UCF50**: A crowd counting dataset.
2. **QNRF**: A large-scale crowd counting dataset.
3. **Shanghai_A**: A dataset part of the ShanghaiTech dataset.
4. **Shanghai_B**: Another dataset part of the ShanghaiTech dataset.

## Models

### Original FPANet Model

The original FPANet model is implemented in the following scripts:

- `ucf50.py`
- `qnrf.py`
- `shanghai_a.py`
- `shanghai_b.py`

### Enhanced FPANet Model

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

## Installation

To run the scripts, you need to have Python installed along with the required dependencies. You can install the dependencies using the following command:

```sh
pip install -r requirements.txt
