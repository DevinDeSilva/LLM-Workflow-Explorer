#source .venv/bin/activate

#python baseline_FullContextAnswer.py --evaluation chatbs-openai
#python baseline_LLMbased.py --evaluation chatbs-openai
#python explainer_experiment.py --evaluation chatbs-openai

cd baselines/grasp
source .venv/bin/activate
export GRASP_INDEX_DIR=/home/desild/work/research/LLM-Workflow-Explorer/baselines/grasp/kg_index
grasp file configs/chatbs-openai.yaml   --input-file /home/desild/work/research/LLM-Workflow-Explorer/evaluations/chatbs-openai/ground_truth/ground_truth_data.jsonl   --output-file /home/desild/work/research/LLM-Workflow-Explorer/evaluations/chatbs-openai/explainer/grasp/exp_202604201325/RESULTS.jsonl   --progress

cd /home/desild/work/research/LLM-Workflow-Explorer
source .venv/bin/activate
python evaluation_results.py --evaluation chatbs-openai