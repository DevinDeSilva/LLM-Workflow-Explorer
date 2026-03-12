from typing import Dict, Optional, List
from src.utils.graph_manager import GraphManager
from src.explorer.executable_program import ExecutableProgram
from src.utils.utils import create_timestamp_id

class FunctionalOntology:
    def __init__(self, graph_manager:GraphManager) -> None:
        self.graph_manager = graph_manager

    def add_parameter(self, main_id:str, inp_n, inp, dtype="xsd:string", required="true") -> Dict:
        """ ex:intParameterA
            a             fno:Parameter ;
            fno:predicate ex:startValue ;
            fno:type      xsd:integer ;
            fno:required  "true"^^xsd:boolean .
            
        """   
        inp_id = create_timestamp_id(main_id + "_param")
        self.graph_manager.add_to_graph(
            inp_id,
            "a",
            "fno:Parameter"
        )
        self.graph_manager.add_to_graph(
            inp_id,
            "fno:predicate",
            inp_n,
            literal=True,
            dtype="xsd:string"
        )
        
        self.graph_manager.add_to_graph(
            inp_id,
            "rdfs:label",
            inp,
            literal=True,
            dtype="xsd:string"
        )
        self.graph_manager.add_to_graph(
            inp_id,
            "fno:type",
            dtype
        )
        self.graph_manager.add_to_graph(
            inp_id,
            "fno:required",
            required,
            literal=True,
            dtype="xsd:boolean"
        )
        
        return {"id": inp_id, "predicate": inp, "dtype": dtype, "required": required}

    def add_return(self, main_id:str, out_n, out, dtype="xsd:string", required="true") -> Dict:
        """ 
        
        ex:sumOutput
        a             fno:Output ;
        fno:predicate ex:sumResult ;
        fno:type      xsd:integer ;
        fno:required  "true"^^xsd:boolean .
            
        """   
        out_id = create_timestamp_id(main_id + "_output")
        self.graph_manager.add_to_graph(
            out_id,
            "a",
            "fno:Output"
        )
        self.graph_manager.add_to_graph(
            out_id,
            "fno:predicate",
            out_n,
            literal=False,
            dtype="xsd:string"
        )
        
        self.graph_manager.add_to_graph(
            out_id,
            "rdfs:label",
            out,
            literal=False,
            dtype="xsd:string"
        )
        self.graph_manager.add_to_graph(
            out_id,
            "fno:type",
            dtype
        )
        self.graph_manager.add_to_graph(
            out_id,
            "fno:required",
            required,
            literal=True,
            dtype="xsd:boolean"
        )

        return {"id": out_id, "predicate": out, "dtype": dtype, "required": required}

    def add_problem(self, main_id:str, question:str, broader:Optional[str]=None) -> Dict:
        """ 
        ex:sumProblem
        a                   fno:Problem ;
        fno:name            "The sum problem"^^xsd:string ;
        dcterms:description "This handles the problem of adding two integers to each other."^^xsd:string ;
        skos:broader        ex:mathProblem .
        """
        
        problem_id = create_timestamp_id(main_id + "_problem")
        self.graph_manager.add_to_graph(
            problem_id,
            "a",
            "fno:Problem"
        )
        self.graph_manager.add_to_graph(
            problem_id,
            "fno:name",
            question,
            literal=True,
            dtype="xsd:string"
        )
        
        if broader:
            self.graph_manager.add_to_graph(
                problem_id,
                "skos:broader",
                broader,
                literal=False
            )

        return {"id": problem_id, "name": question, "broader": broader}

    def add_function(self, main_id:str, name:str, question_id:str, parameters:List[Dict], returns:List[Dict]) -> Dict:
        """
        ex:sumFunction
        a                   fno:Function ;
        fno:name            "The sum function"^^xsd:string ;
        dcterms:description "This function can do the sum of two integers."^^xsd:string ;
        fno:solves          ex:sumProblem ;
        fno:implements      ex:sumAlgorithm ;
        fno:expects         ( ex: intParameterA ex:intParameterB ) ;
        fno:returns         ( ex:sumOutput ) .
        """
        
        function_id = create_timestamp_id(main_id + "_function")
        self.graph_manager.add_to_graph(
            function_id,
            "a",
            "fno:Function"
        )
        self.graph_manager.add_to_graph(
            function_id,
            "fno:name",
            name,
            literal=True,
            dtype="xsd:string"
        )
        
        self.graph_manager.add_to_graph(
            function_id,
            "fno:solves",
            question_id,
            literal=False
        )
        
        for param in parameters:
            self.graph_manager.add_to_graph(
                function_id,
                "fno:expects",
                param['id'],
                literal=False
            )
            
        for ret in returns:
            self.graph_manager.add_to_graph(
                function_id,
                "fno:returns",
                ret['id'],
                literal=False
            )

        return {"id": function_id, "name": name, "solves": question_id, "parameters": parameters, "returns": returns}
        
    def add_implementation(self, main_id:str, code:str) -> Dict:
        """
        ex:leftPadImplementation
        a         fnoi:NpmPackage ;
        doap:name "left-pad" .
        
        """
        
        function_id = create_timestamp_id(main_id + "_function")
        self.graph_manager.add_to_graph(
            function_id,
            "a",
            "fnoi:SQLImplementation"
        )
        
        self.graph_manager.add_to_graph(
            function_id,
            "rdfs:label",
            code,
            literal=True,
            dtype="xsd:string"
        )
        
        return {"id":function_id, "label":code}

    def add_mapping(self, main_id:str, function_id:str, implementation_id:str, parameters:List[Dict], returns:List[Dict]):
        """
        ex:leftPadMapping
        a                    fno:Mapping ;
        fno:function         ex:leftPad ;
        fno:implementation   ex:leftPadImplementation ;
        fno:methodMapping    [ a                fnom:StringMethodMapping ;
                            fnom:method-name "doLeftPadding" ] ;
        fno:parameterMapping [ a                                    fnom:PositionParameterMapping ;
                            fnom:functionParameter               ex:inputStringParameter ;
                            fnom:implementationParameterPosition "2"^^xsd:int ] ;
        fno:parameterMapping [ a                                    fnom:PositionParameterMapping ;
                            fnom:functionParameter               ex:paddingParameter ;
                            fnom:implementationParameterPosition "1"^^xsd:int ] ;
        fno:returnMapping    [ a                   fnom:DefaultReturnMapping ;
                            fnom:functionOutput ex:outputStringOutput ] .
        """
        
        mapping_id = create_timestamp_id(main_id + "_mapping")
        self.graph_manager.add_to_graph(
            mapping_id,
            "a",
            "fno:Mapping"
        )
        
        self.graph_manager.add_to_graph(
            mapping_id,
            "fno:function",
            function_id
        )

        self.graph_manager.add_to_graph(
            mapping_id,
            "fno:implementation",
            implementation_id
        )

        for i, param in enumerate(parameters):
            _id = create_timestamp_id(main_id + "_parameter_mapping_" + str(i))
            self.graph_manager.add_to_graph(
                mapping_id,
                "fno:parameterMapping",
                _id,
                literal=False
            )

            self.graph_manager.add_to_graph(
                _id,
                "a",
                "fnom:PositionParameterMapping"
            )
            self.graph_manager.add_to_graph(
                _id,
                "fnom:functionParameter",
                param['id'],
                literal=False
            )
            self.graph_manager.add_to_graph(
                _id,
                "fnom:implementationParameterPosition",
                str(i),
                literal=True,
                dtype="xsd:int"
            )

        for i, ret in enumerate(returns):
            _id = create_timestamp_id(main_id + "_return_mapping_" + str(i))
            self.graph_manager.add_to_graph(
                mapping_id,
                "fno:returnMapping",
                _id,
                literal=False
            )

            self.graph_manager.add_to_graph(
                _id,
                "a",
                "fnom:DefaultReturnMapping"
            )
            self.graph_manager.add_to_graph(
                _id,
                "fnom:functionOutput",
                ret['id'],
                literal=False
            )
            
    def add_example(self, main_id:str, function_id:str, example_usage:str, example_output:Dict):
        execution_id = create_timestamp_id(main_id + "_executes")
        self.graph_manager.add_to_graph(
            execution_id,
            "a",
            "fno:Execution"
        )
        
        self.graph_manager.add_to_graph(
            function_id,
            "fno:executes",
            execution_id
        )
        
        self.graph_manager.add_to_graph(
            execution_id,
            "rdfs:label",
            example_usage,
            literal=True,
            dtype="xsd:string"
        )
        
        return {"id": execution_id, "label": example_usage, "output": example_output}

        
        

    def add_fno_graph(self, index:int, exp:ExecutableProgram, question:str, categories:str):
        """
        Adds an FNO graph representation to the given ExecutableProgram.
        R: add_fno_graph
        """
        exp.program_id = "ques:eprog{}".format(index)
        

        param_input, param_output = [], []
        for k,v in exp.input_spec.items():
            param_input.append(self.add_parameter(exp.program_id+f"_{k}", k, v))
            
        for k,v in exp.output_spec.items():
            param_output.append(self.add_parameter(exp.program_id+f"_{k}", k, v))
            
        question_id = self.add_problem(
            exp.program_id, 
            question)
            
        function = self.add_function(
            exp.program_id, 
            categories,
            question_id['id'], 
            param_input, 
            param_output
            )
        
        implementation = self.add_implementation(
            exp.program_id,
            exp.code
        )

        example = self.add_example(
            exp.program_id,
            function['id'],
            exp.example_usage,
            exp.example_output.to_dict(orient='list')
        )
        
        

        #parameters
        self.add_mapping(
                exp.program_id, 
                function['id'], 
                implementation['id'],
                param_input, 
                param_output
        )
    
    