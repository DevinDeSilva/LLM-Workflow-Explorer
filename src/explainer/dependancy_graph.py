from typing import Any, Dict, List, TypeVar, Tuple, Optional

import dspy
from pydantic import BaseModel
from icecream import ic
import re
import logging
import copy

from src.config.object_search import ObjectSearchConfig
from src.embeddings.base import BaseEmbeddings
from src.llm.base import BaseLLM  
from src.vector_db.base import BaseVectorDB
from src.explainer.object_search import ObjectSearch
from src.utils.graph_manager import GraphManager
from src.utils.adjacency_matrix import build_adjacency_matrix, incoming_edges
from src.config.experiment import TTLConfig
from src.config.experiment import ApplicationInfo

from src.templates.dependancy_graph import (
    SubQuestionSignature, SubQuestionVerificationSignature,
    BuildTopologyGraphSignature
    )
from src.utils.utils import clean_string_list

logger = logging.getLogger(__name__)

class QuestionNode(BaseModel):
    id:str
    question:str
    node_type:Optional[str] = None
    
    def solve(self, 
              schema_info, predecessor_info, max_tries = 5
              ) -> Dict[str, Any]:
        # use the provided information to solve the question at this node and return the answer.
        tries = 0
        answered = False
        
        while tries < max_tries:
            # use predecessor_info to create a prompt for solving the node.
            # the prompt should include the original question, the schema information, and the information from the predecessor nodes.
            # the model should then output the answer to the question at the node, as well as any relevant schema information used, any objects retrieved from the graph, any synthetic questions created, and any intermediary results.
            
            # if the model fails to answer the question or if we determine that the answer is incorrect, we can try again by providing feedback to the model and asking it to revise its answer.
            
            break
        
        return {}
    

T = TypeVar("T")
Edge = Tuple[T, T]

class DependancyGraph:
    def __init__(  
        self, 
        graph_loc:str,
        app_info:ApplicationInfo,
        llm: BaseLLM,
        embedder:BaseEmbeddings,
        vector_db:BaseVectorDB,
        object_search_config:ObjectSearchConfig,
        ttl_config:TTLConfig
    ) -> None:
        self.vertices:Dict[str, QuestionNode] = {}
        
        self.app_info = app_info
        
        self.graph_manager = GraphManager(
            graph_file=graph_loc,
            config=ttl_config
        )
        
        self.llm = llm
        self.embedder = embedder
        self.vector_db = vector_db
        
        self.object_db = ObjectSearch(
            self.graph_manager,
            self.embedder,
            self.vector_db,
            object_search_config
        )
        
        # LLM Predictors
        ## Sub Question predictor 
        self.information_required_predictor = dspy.Predict(
            SubQuestionSignature
        )
        # self.information_required_predictor.demos = (
        #     self.build_information_required_fewshot_examples()
        # )
        
        ## Filter Sub Questions 
        self.filter_sub_question_predictor = dspy.Predict(
            SubQuestionVerificationSignature
        )
        
        ## Build Sub Question Graph
        self.build_topology_graph_predictor = dspy.Predict(
            BuildTopologyGraphSignature
        )
    
    def information_required(
        self,
        query: str,
        schema_context: str,
    ) -> List[str]:
        application_context = (self.app_info.description or "").strip()

        with dspy.context(lm=self.llm.llm):
            prediction = self.information_required_predictor(
                user_query=query.strip(),
                schema_context=schema_context.strip(),
                application_context=application_context,
            )
            
            clean_sub_questions = clean_string_list(
                getattr(prediction, "sub_questions", [])
            )
            
            if not clean_sub_questions:
                return []
            
            prediction = self.filter_sub_question_predictor(
                original_question = query,
                sub_questions = [f"{i}). {q}" for i,q in enumerate(clean_sub_questions)]
            )
            
        clean_sub_question_num = getattr(prediction, "filtered_sub_question", [])
        filtered_sub_questions = [q for i,q in enumerate(clean_sub_questions) if i not in clean_sub_question_num]

        return filtered_sub_questions
        
    def build_toplevel_dependancy_graph(self, user_query:str, info_req:List[QuestionNode]) -> None:
        self.vertices.update({v.id:v for v in info_req})
        self.vertices["0"] = QuestionNode(id="0", question=user_query)
        self.in_degree:List[int] = [0]*len(self.vertices)
        self.out_degree:List[int] = [0]*len(self.vertices)
          
        #add edges
        with dspy.context(lm=self.llm.llm):
            graph_content = self.build_topology_graph_predictor(
                original_question=user_query.strip(),
                sub_questions=["{}. {}".format(v.id,v.question) for v in info_req],
            )

            topo_graph_rep = getattr(graph_content, "topology_graph", None)
            
        if not topo_graph_rep:
            ValueError("Failed to build topology graph")
                
        (self.adjacency_matrix, self.edges) = build_adjacency_matrix(topo_graph_rep)

        logger.info("Graph content:")
        logger.info("--------------------------------")
        logger.info(graph_content)
        logger.info("Adjacency matrix:")
        logger.info("--------------------------------")
        for row in self.adjacency_matrix:
            logger.info(row)
            
        # use adjecency matrix to populate edges.
        for _,v1 in self.vertices.items():
            for _,v2 in self.vertices.items():
                if self.adjacency_matrix[int(v2.id)][int(v2.id)] == 1:
                    self.out_degree[int(v2.id)] += 1
                    self.in_degree[int(v1.id)] += 1
                    
        ic(self.in_degree)
        ic(self.out_degree)
        
        # set state of the nodes
        for _,v in self.vertices.items():
            if self.in_degree[int(v.id)] == 0:
                v.node_type = "leaf"
            elif self.out_degree[int(v.id)] == 0:
                v.node_type = "root"
            else:
                v.node_type = "inner"
        
        
    def user_query_to_requirements(
        self, query: str, schema_context: str = ""
    ) -> Dict[str, Any]:
        ic(query)
        info_req = self.information_required(query, schema_context)
        ic(info_req)
        self.build_toplevel_dependancy_graph(
            query,
            [QuestionNode(
                id = str(i+1),
                question= x
                ) 
            for i,x in enumerate(info_req)]
        )
        
        return {
            "user_query": query,
            "information_required": info_req
        }
    
    @staticmethod
    def solve_node(
        adj_matrix, 
        node_id, 
        schema_info,
        node_map:Dict[str, QuestionNode],
        max_tries = 5,
        ) -> Dict[str, Any]:
        in_nodes = incoming_edges(adj_matrix, node_id)
        
        predecessor_info = {}
        for in_node in in_nodes:
            predecessor_info[in_node] = DependancyGraph.solve_node(
                adj_matrix, 
                in_node, 
                schema_info,
                node_map,
                max_tries
                )
            
        # solve the node using predecessor_info
        node_info = {
            "id": node_id,
            "schema_info_used": None,
            "retrieved_objects": None,
            "synthetic_questions_plan": None,
            "intermediary_results": None,
            "answer": None,
        }
        
        node_data = node_map[node_id].solve(
            schema_info,
            predecessor_info
        )
        
        node_info['answer'] = node_data.get("answer")
        
        return node_info
        
    def _recursive_solve_nodes(self):
        # find leaf nodes and solve them first.
        # propagate the answers to the parent nodes and repeat the process until we reach the root node.
        root:QuestionNode = self.vertices["0"]
        
    
    def process_dependancy_graph(
        self,
        schema_context:str
    ):
        # process the graph in a bottom up manner starting from leaf nodes.
        # for each leaf node, we perform object search to retreive relevant information from the graph.
        # we then use the retreived information to answer the question at the node.
        # we then propagate the answer to the parent nodes and repeat the process until we reach the root node.
        pass
        
        
    
