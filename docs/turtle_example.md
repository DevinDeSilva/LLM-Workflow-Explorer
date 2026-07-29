### Turtle File examples

The following example shows a Turtle serialization snippet for an LLM-based workflow step in ChatBS-NexGen. In this execution, an LLM uses a system prompt that encodes the patient's profile and health indicators, together with a user prompt requesting light, energy-boosting food recommendations before exercise, to generate a dietary recommendation for the patient. The LLM invocation is represented as a generative task that consumes the system and user prompts and produces the generated recommendation. The snippet illustrates how the workflow graph records the model used, input artifacts, generated output, and the provenance relation linking the output to the execution that produced it.

```bash
@prefix ex:       <http://testwebsite/testProgram#> .
@prefix prov:     <http://www.w3.org/ns/prov#> .
@prefix provone:  <http://purl.dataone.org/provone/2015/01/15/ontology#> .
@prefix sio:      <http://semanticscience.org/resource/> .
@prefix workflow: <http://www.semanticweb.org/acer/ontologies/2026/1/WorkFlow/> .
@prefix rdfs:     <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dc:       <http://purl.org/dc/elements/1.1/> .

ex:Generative_Task-id_20260420105659_302
    a workflow:Generative_Task ;
    rdfs:label "LLM text generation task" ;
    dc:description "Generates a dietary recommendation using an LLM." ;
    prov:used ex:LLM-id_20260420105659_302 ;
    sio:SIO_000313 ex:id_20260420105659_302 .

ex:LLM-id_20260420105659_302
    a workflow:Large_Language_Models ;
    workflow:llm_model "gpt-4o" ;
    sio:SIO_000230 ex:Data-id_20260420105652_456-system_prompt ;
    sio:SIO_000230 ex:Data-id_20260420105659_811-user_prompt ;
    sio:SIO_000229 ex:LLM_Output-id_20260420105659_302 .

ex:Data-id_20260420105652_456-system_prompt
    a provone:Data ;
    rdfs:label "system_prompt" ;
    prov:value "You are a health care assistant..." .

ex:Data-id_20260420105659_811-user_prompt
    a provone:Data ;
    rdfs:label "user_prompt" ;
    prov:value "I need suggestions for light and energy-boosting foods..." .

ex:Data-id_20260420105659_338-generated_answer
    a provone:Data ;
    rdfs:label "generated_answer" ;
    prov:value "To fuel your body effectively before a gym workout..." ;
    prov:wasGeneratedBy ex:id_20260420105659_302 .

ex:Generation-id_20260420105659_282-generated_answer
    a prov:Generation ;
    provone:hadEntity ex:Data-id_20260420105659_338-generated_answer ;
    provone:hadOutPort ex:llm_chat-generated_answer .
```