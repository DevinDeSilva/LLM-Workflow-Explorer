\name{RExplAnnotator}
\alias{RExplAnnotator}
\alias{add_metadata_to_object}
\alias{add_to_graph}
\alias{add_to_namespace}
\alias{create_prov_module}
\alias{curie}
\alias{decorator_to_triples}
\alias{default_prov_config_path}
\alias{entity_marking}
\alias{extract_decorators}
\alias{find_decorators}
\alias{get_files_to_check}
\alias{get_time_stamp}
\alias{get_unq_id}
\alias{load_prov_config}
\alias{make_ttl_namespace}
\alias{name_concat}
\alias{process_files}
\alias{provChannel}
\alias{provMakeList}
\alias{provProgram}
\alias{provProgramExecution}
\alias{provWasInformedBy}
\alias{provWasPartOf}
\alias{prov_module}
\alias{save_prov_graph}
\alias{validate_namespaces}
\title{RExplAnnotator provenance helpers}
\description{
Create PROV-One RDF provenance graphs for R workflow programs, channels,
executions, inputs, outputs, and LLM-assisted tasks.
}
\usage{
create_prov_module(config = NULL, config_path = NULL)
prov_module(config = NULL, config_path = NULL)
}
\arguments{
\item{config}{Optional configuration list.}
\item{config_path}{Optional path to a YAML configuration file.}
}
\details{
The main entry point is \code{create_prov_module()}, which returns an
environment containing a graph, configuration, and bound helper functions such
as \code{provProgram()}, \code{provChannel()}, \code{provProgramExecution()},
and \code{save_prov_graph()}.
}
\value{
For \code{create_prov_module()}, an environment containing the RDF graph and
bound provenance helper functions. Other helpers return provenance record lists,
character vectors, or invisibly return \code{TRUE}.
}
\examples{
cfg <- load_prov_config()
names(make_ttl_namespace(cfg))
}
