source .venv/bin/activate

# python baseline_FullContextAnswer.py --evaluation biomni-base
# python baseline_LLMbased.py --evaluation biomni-base
# python explainer_experiment.py --evaluation biomni-base
# python baseline_VectorSimilarityAnswer.py --evaluation biomni-base
# python baseline_HippoRAG.py --evaluation biomni-base
# python baseline_HyperGraphRAG.py --evaluation biomni-base
# python explainer_experiment.py --evaluation biomni-base

# cd baselines/grasp
# source .venv/bin/activate
# export GRASP_INDEX_DIR=/home/desild/work/research/LLM-Workflow-Explorer/baselines/grasp/kg_index
# mkdir /home/desild/work/research/LLM-Workflow-Explorer/evaluations/biomni-base/explainer/grasp/exp_202604201325
# grasp file configs/biomni-base.yaml   --input-file /home/desild/work/research/LLM-Workflow-Explorer/evaluations/biomni-base/ground_truth/ground_truth_data.jsonl   --output-file /home/desild/work/research/LLM-Workflow-Explorer/evaluations/biomni-base/explainer/grasp/exp_202604201325/RESULTS.jsonl   --progress

# cd /home/desild/work/research/LLM-Workflow-Explorer
# source .venv/bin/activate
python evaluation_results.py --evaluation biomni-base
python answer_winrate_evaluation.py --evaluation biomni-base