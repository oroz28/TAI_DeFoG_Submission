from source.subgraphs_helper import run_subgraphs
import sys

# read dataset argument from command line
if __name__ == "__main__":
    dataset_arg = sys.argv[1]
    
    ## PLANAR DATASET ##
    # # if you want to run specific subgraphs, uncomment this section
    # subgraphs_to_run = [
    #     "/home/group-2/TAI_defog/tai_defog/validation_planar_pt/graph_002/subgraph_graph_002_len_15.pt",
    #     "/home/group-2/TAI_defog/tai_defog/validation_planar_pt/graph_003/subgraph_graph_003_len_15.pt",
    #     ]
    # run_subgraphs(dataset_arg, num_steps=[50], num_folds=5, subgraph_files=subgraphs_to_run)
    # # run all subgraphs
    # run_subgraphs(dataset_arg, num_steps=[50], num_folds=1)
    
    ## QM9 DATASET ##
    # run all subgraphs
    run_subgraphs(dataset_arg, num_steps=[50], num_folds=1, samples_per_fold=1024)
    
    # # if you want to run specific subgraphs, uncomment this section
    # subgraphs_to_run = [
    #     "/home/group-2/TAI_defog/tai_defog/fixed_subgraphs_selected_qm9_pt/graph_19858/graph_19858_chain_2.pt",
    #     "/home/group-2/TAI_defog/tai_defog/fixed_subgraphs_selected_qm9_pt/graph_4077/graph_4077_chain_2.pt",
    #     ]
    # run_subgraphs(dataset_arg, num_steps=[50], num_folds=5, subgraph_files=subgraphs_to_run,  samples_per_fold=1024)