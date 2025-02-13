## HazardComp Dataset 
original datasets: 20220512-relation-2006 & train_sceneGraphs.json

Preprocessed datasets:Rule124_triplet_50_set1_answer.json & Rule124_triplet_50_set1_answer_program.json & Rule124_triplet_50_set1_answer_program_list.json & Rule124_triplet_50_set1_set2_answer_pairs.json & Rule124_triplet_50_set2_answer.json & Rule124_triplet_50_set2_answer_program.json & Rule124_triplet_50_set2_answer_program_list.json & Rule124_triplet_balanced_test_answer_program.json & Rule124_triplet_balanced_test_answer_program_list.json & Rule124_triplet_balanced_train_answer_program.json & Rule124_triplet_balanced_train_answer_program_list.json & Rule124_rule_triplet_pair.json & Rule56_triplet_balanced_test_answer.json & Rule56_triplet_balanced_train_answer.json

image roi_feats and bbox_feats: demo/datasets/hazard_detection 

## Visual feature generation[Optional]:
Please following the instruction of https://github.com/MILVLG/bottom-up-attention.pytorch to install the environment and extract features.

## Data Preprocessing [Optional]:
If you want to know how the programs and training data are generated, please follow the following steps:
1. Using balanced_answer_generation.ipynb to generate Rule124_triplet_balanced_train_answer_program_list.json
2.
  ```
    python preprocess.py hazard_detection
  ```
 please care about the arguments and out_dir


## Hazard Detection Training and Evaluation
- train the model using Rule124
  ```
   python run_experiments.py --do_balanced --model TreeSparsePostv2 --id TreeSparsePost2Full --stacking 2 --batch_size 256
  ```
- Test the model on the test split:
  ```
    python inference_testset.py --stacking 2 --batch_size 256 --load_from [model_Path]
  ```

 - Test the model using one image and one rule:
   ```
    python inference.py --stacking 2 --batch_size 256 --load_from [model_Path] --data [roi_feats_after_processing_path]
  ```
 - Run the demo:
   ```
    streamlit run app_simple.py
  ```
