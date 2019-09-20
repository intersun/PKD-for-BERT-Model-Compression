# Patient Knowledge Distillation for BERT Model Compression
Knowledge distillation for BERT model


## Installation
Run command below to install the environment
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```


## Training
#### Objective Function
```math
L = (1 - \alpha) L_CE + \alpha * L_DS + \beta * L_PT,
```
where L_CE is the CrossEntropy loss, DS is the usual Distillation loss, and PT is the proposed loss. Please see our paper below for more details.

#### Data Preprocess
Modify the HOME_DATA_FOLDER in *envs.py* and put all data under it (by default it is **./data**), RTE data is uploaded for your convenience.

* The folder name under HOME_DATA_FOLDER should be 
  * *data_raw*: store the raw datas of all tasks. So put downloaded raw data under here
    * MRPC
    * RTE 
    * ... (other tasks)
  * *data_feat*: store the *tokenized* data under this folder (optional)
    * MRPC
    * RTE 
    * ...
 * *models*
   * pretrained: put downloaded pretrained model (bert-base-uncased) under this folder
   
#### Predefinted Training 
Run *NLI_KD_training.py* to start training, you can set *DEBUG* = True to run some pre-defined arguments
  * set *argv = get_predefine_argv('glue', 'RTE', 'finetune_teacher')* or *argv = get_predefine_argv('glue', 'RTE', 'finetune_student')* to start the normal fine-tuning
  * run *run_glue_benchmark.py* to get teacher's prediction for KD or PKD. 
    * set *output_all_layers = True* for patient teacher
    * set *output_all_layers = False* for normal teacher
  * set *argv = get_predefine_argv('glue', 'RTE', 'kd')* to start the vanilla KD
  * set *argv = get_predefine_argv('glue', 'RTE', 'kd.cls')* to start the vanilla KD
  
## Contributing
This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.


## Citation
If you find this code useful for your research, please consider citing:

    @article{sun2019patient,
    title={Patient Knowledge Distillation for BERT Model Compression},
    author={Sun, Siqi and Cheng, Yu and Gan, Zhe and Liu, Jingjing},
    journal={arXiv preprint arXiv:1908.09355},
    year={2019}
    }

Paper is available at [here](https://arxiv.org/abs/1908.09355).  
