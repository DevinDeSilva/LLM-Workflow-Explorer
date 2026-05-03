#source .venv/bin/activate

#python baseline_FullContextAnswer.py --evaluation chatbs-base
#python baseline_LLMbased.py --evaluation chatbs-base
#python explainer_experiment.py --evaluation chatbs-base
#python baseline_VectorSimilarityAnswer.py --evaluation chatbs-base
python baseline_HippoRAG.py --evaluation chatbs-base
python baseline_HyperGraphRAG.py --evaluation chatbs-base

# cd baselines/grasp
# source .venv/bin/activate
# export GRASP_INDEX_DIR=/home/desild/work/research/LLM-Workflow-Explorer/baselines/grasp/kg_index
# grasp file configs/chatbs-base.yaml   --input-file /home/desild/work/research/LLM-Workflow-Explorer/evaluations/chatbs-base/ground_truth/ground_truth_data.jsonl   --output-file /home/desild/work/research/LLM-Workflow-Explorer/evaluations/chatbs-base/explainer/grasp/exp_202604201325/RESULTS.jsonl   --progress

# cd /home/desild/work/research/LLM-Workflow-Explorer
# source .venv/bin/activate
python evaluation_results.py --evaluation chatbs-base
python answer_winrate_evaluation.py --evaluation chatbs-base